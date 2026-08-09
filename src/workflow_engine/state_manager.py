"""WorkflowStateManager — thread-safe state mutation for workflow execution.

Extracted from renderer.py to enforce the architectural rule that render stays
pure.  All state transitions go through this class under a single lock, and
the renderer receives immutable snapshots.
"""

from __future__ import annotations

import copy
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Dict, Optional

from .models import (
    AgentProgress,
    AgentStatus,
    PhaseProgress,
    ReviewerEvidence,
    SubagentProgress,
    SubagentStatus,
    WorkflowMetrics,
    WorkflowProject,
    WorkflowStatus,
)


class WorkflowStateManager:
    """Owns all mutations to a WorkflowProject during execution.

    Thread-safety: every public method acquires ``_lock`` before mutating.
    Callers that need a consistent view for rendering should call
    ``snapshot()`` which returns the project under the same lock.
    """

    def __init__(self, project: WorkflowProject) -> None:
        self._project = project
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._writer_wait_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._writers_waiting = 0
        # O(1) lookup: agent label -> AgentProgress reference.
        # All inserts, lookups and removals happen under ``_lock`` so that the
        # map stays consistent with the project.phases[*].agents lists even
        # under heavy parallel event delivery.
        self._label_to_agent: Dict[str, AgentProgress] = {}
        # AC4: 主 context 增量 token 计数。仅由 engine 在明确需要向主 chat
        # 注入文本的路径上累加（目前唯一合法路径是 workflow 完成时的
        # project.result）。中间结果不得通过其他路径增加此计数器。
        self._delta_context_tokens: int = 0
        self._label_counters: Dict[str, int] = {}
        self._rebuild_indexes()
        self._next_agent_call_index = (
            max(
                (
                    agent.call_index
                    for phase in self._project.phases
                    for agent in phase.agents
                ),
                default=-1,
            )
            + 1
        )

    # ------------------------------------------------------------------
    # Public: state-changing events
    # ------------------------------------------------------------------

    def on_phase_changed(self, title: str) -> None:
        """Record a new phase start."""
        with self._write_locked():
            phase = PhaseProgress(title=title, started_at=time.time())
            self._project.phases.append(phase)
            self._project.status = WorkflowStatus.RUNNING

    def on_agent_started(
        self,
        label: str,
        tool: str,
        phase: str,
        task_summary: str = "",
        model: str | None = None,
        role: str | None = None,
    ) -> str:
        """Add an agent entry to the current (or matching) phase.

        Returns the effective, UI-visible label. User-generated workflow
        scripts can accidentally reuse labels (for example several
        ``task-analysis`` calls in parallel). State transitions are keyed by
        label, so duplicate labels are disambiguated at the source.
        """
        with self._write_locked():
            if self._is_workflow_terminal():
                return label or "agent"
            target_phase = self._find_or_create_phase(phase)
            now = time.time()
            effective_label = self._make_unique_label(label or "agent")
            call_index = self._next_agent_call_index
            self._next_agent_call_index += 1
            agent = AgentProgress(
                label=effective_label,
                tool=tool,
                model=model,
                role=role,
                task_summary=task_summary,
                status=AgentStatus.RUNNING,
                started_at=now,
                activity_updated_at=now,
                call_index=call_index,
            )
            target_phase.agents.append(agent)
            # Map insert must happen atomically with the agent-list append so
            # that concurrent callers of on_agent_done/on_agent_failed always
            # see a consistent view: if it is in `phases[*].agents` it is also
            # in `_label_to_agent`.
            self._label_to_agent[effective_label] = agent
            self._project.metrics.total_agents += 1
            return effective_label

    def on_agent_done(self, label: str, result: dict) -> None:
        """Update agent status to DONE with metrics from result."""
        with self._write_locked():
            agent = self._label_to_agent.get(label)
            if agent is None:
                # O(n) fallback: keeps backwards-compatible behaviour when an
                # agent was registered through a path that bypassed
                # _label_to_agent (e.g. a misbehaved caller). This branch also
                # acts as a canary for consistency bugs in the map itself.
                agent = self._find_agent(label)
                if agent is None:
                    return
                # Repair the map on hit so subsequent lookups stay O(1).
                self._label_to_agent[label] = agent

            # Sticky terminal: any terminal state is final — DONE does not
            # overwrite CANCELLED, FAILED, or an already-DONE agent.
            if self._is_terminal_agent(agent):
                return

            token_usage = result.get("token_usage", 0)
            duration_s = result.get("duration_s", 0.0)
            cached = result.get("cached", False)
            was_terminal = self._is_terminal_agent(agent)
            was_cached = agent.status == AgentStatus.CACHED

            if cached:
                agent.status = AgentStatus.CACHED
                if not was_cached:
                    self._project.metrics.cached_agents += 1
            else:
                agent.status = AgentStatus.DONE

            agent.token_usage = token_usage
            agent.duration_s = duration_s
            agent.error = None
            raw_result = result.get("result")
            agent.result = str(raw_result) if raw_result is not None else None
            agent.current_activity = ""
            agent.finished_at = time.time()

            if not was_terminal:
                self._project.metrics.completed_agents += 1
            self._project.metrics.total_tokens += token_usage
            self._project.metrics.total_duration_s += duration_s

    def on_agent_failed(self, label: str, error: str, result: str | None = None) -> None:
        """Update agent status to FAILED."""
        with self._write_locked():
            agent = self._label_to_agent.get(label)
            if agent is None:
                # Same O(n) fallback as on_agent_done.
                agent = self._find_agent(label)
                if agent is None:
                    return
                self._label_to_agent[label] = agent

            # Sticky terminal: any terminal state is final — FAILED does not
            # overwrite DONE, CANCELLED, CACHED, or an already-FAILED agent.
            if self._is_terminal_agent(agent):
                return
            now = time.time()
            agent.status = AgentStatus.FAILED
            agent.error = error
            agent.result = str(result) if result is not None else None
            agent.current_activity = ""
            agent.finished_at = now
            if agent.duration_s <= 0 and agent.started_at:
                agent.duration_s = max(0.0, now - agent.started_at)
            self._project.metrics.failed_agents += 1
            self._project.metrics.completed_agents += 1

    def update_agent_activity(self, label: str, activity: str) -> None:
        """Update the live activity hint for a running agent (non-blocking)."""
        with self._write_locked():
            agent = self._label_to_agent.get(label)
            if agent is None:
                return
            if self._is_terminal_agent(agent):
                return
            agent.current_activity = activity
            agent.activity_updated_at = time.time()

    def update_agent_subagents(
        self,
        label: str,
        updates: Sequence[Mapping[str, object]],
    ) -> bool:
        """Merge child-agent snapshots by stable provider thread identity.

        Returns ``True`` only when renderable state changed, allowing callers
        to avoid redundant Feishu card refreshes for replayed ACP frames.
        """
        with self._write_locked():
            agent = self._label_to_agent.get(label)
            if agent is None or self._is_terminal_agent(agent):
                return False

            changed = False
            by_source = {item.source_id: item for item in agent.subagents}
            terminal = {
                SubagentStatus.COMPLETED,
                SubagentStatus.FAILED,
                SubagentStatus.CANCELLED,
            }
            latest_index: int | None = None

            for update in updates:
                source_id = str(update.get("source_id") or "").strip()
                if not source_id:
                    continue
                raw_status = str(update.get("status") or "running").strip().lower()
                try:
                    status = SubagentStatus(raw_status)
                except ValueError:
                    status = SubagentStatus.RUNNING
                progress = str(update.get("progress") or "").strip()
                raw_model = update.get("model")
                model = str(raw_model).strip() if raw_model else None

                child = by_source.get(source_id)
                if child is None:
                    child = SubagentProgress(
                        source_id=source_id,
                        status=status,
                        progress=progress,
                        model=model,
                    )
                    agent.subagents.append(child)
                    by_source[source_id] = child
                    latest_index = len(agent.subagents) - 1
                    changed = True
                    continue

                # Terminal child state is sticky. A replayed/late "running"
                # activity must not resurrect a child that already finished.
                if child.status in terminal and status != child.status:
                    continue
                child_changed = False
                if child.status != status:
                    child.status = status
                    child_changed = True
                if progress and child.progress != progress:
                    child.progress = progress
                    child_changed = True
                if model and child.model != model:
                    child.model = model
                    child_changed = True
                if child_changed:
                    latest_index = agent.subagents.index(child)
                    changed = True

            if changed and latest_index is not None:
                latest = agent.subagents[latest_index]
                status_text = {
                    SubagentStatus.RUNNING: "执行中",
                    SubagentStatus.COMPLETED: "已完成",
                    SubagentStatus.FAILED: "失败",
                    SubagentStatus.CANCELLED: "已取消",
                }[latest.status]
                agent.current_activity = (
                    f"子 Agent {latest_index + 1} · {latest.progress or status_text}"
                )
                agent.activity_updated_at = time.time()
            return changed

    def on_agent_aborted(self, label: str, reason: str = "Aborted by race loser") -> None:
        """Mark a running agent as CANCELLED (e.g. race() loser abort).

        Unlike on_agent_failed, this does not increment failed_agents —
        aborted agents are an expected outcome of race/tournament semantics
        and should not be counted as failures.
        """
        with self._write_locked():
            agent = self._label_to_agent.get(label)
            if agent is None:
                agent = self._find_agent(label)
                if agent is None:
                    return
                self._label_to_agent[label] = agent

            # Sticky terminal: any terminal state is final — CANCELLED does not
            # overwrite DONE, FAILED, CACHED, or an already-CANCELLED agent.
            if self._is_terminal_agent(agent):
                return
            now = time.time()
            agent.status = AgentStatus.CANCELLED
            agent.error = reason
            agent.current_activity = ""
            agent.finished_at = now
            if agent.duration_s <= 0 and agent.started_at:
                agent.duration_s = max(0.0, now - agent.started_at)
            self._project.metrics.completed_agents += 1

    def on_workflow_done(self, result: str) -> None:
        """Mark workflow as completed."""
        with self._write_locked():
            if self._project.status in (WorkflowStatus.CANCELLED, WorkflowStatus.FAILED):
                return
            self._project.status = WorkflowStatus.COMPLETED
            self._project.result = result
            self._project.finished_at = time.time()
            # Close last phase
            if self._project.phases:
                last_phase = self._project.phases[-1]
                if last_phase.finished_at is None:
                    last_phase.finished_at = time.time()
            self._project.metrics.phases_completed = len(self._project.phases)

    def on_workflow_failed(self, error: str) -> None:
        """Mark workflow as failed."""
        with self._write_locked():
            # Sticky terminal: if already CANCELLED or COMPLETED, do not overwrite.
            if self._project.status in (WorkflowStatus.CANCELLED, WorkflowStatus.COMPLETED):
                return
            now = time.time()
            self._project.status = WorkflowStatus.FAILED
            self._project.error = error
            self._project.finished_at = now
            self._close_open_agents(
                error=f"Workflow failed before agent completed: {error}",
                finished_at=now,
            )
            for phase in self._project.phases:
                if phase.finished_at is None:
                    phase.finished_at = now

    def on_workflow_cancelled(self, reason: str = "Workflow cancelled") -> None:
        """Mark workflow as cancelled without rewriting it as a failure."""
        with self._write_locked():
            # Sticky terminal: COMPLETED is final — nothing overwrites nothing.
            # FAILED can be overwritten by CANCELLED because user stop takes
            # precedence over a runtime failure.
            if self._project.status == WorkflowStatus.COMPLETED:
                return
            now = time.time()
            self._project.status = WorkflowStatus.CANCELLED
            self._project.error = reason
            self._project.finished_at = now
            self._close_open_agents(error=reason, finished_at=now)
            for phase in self._project.phases:
                if phase.finished_at is None:
                    phase.finished_at = now

    def add_context_tokens(self, tokens: int) -> None:
        """Increment the main-context token counter (AC4 isolation).

        This counter is *not* a budget gate — it is an audit counter that
        tracks how many tokens-worth of text the workflow intends to inject
        into the main agent chat context.  Only the engine's final-result
        path should call this method; any other path inflating it is a
        regression of the AC4 isolation guarantee.
        """
        with self._write_locked():
            self._delta_context_tokens += max(0, int(tokens))

    def record_reviewer_evidence(self, evidence: ReviewerEvidence) -> None:
        """Persist one immutable snapshot of an independent review call."""
        with self._write_locked():
            self._project.reviewer_evidence.append(evidence.model_copy(deep=True))

    @property
    def delta_context_tokens(self) -> int:
        """Observed main-context token delta for the current workflow run."""
        with self._lock:
            return self._delta_context_tokens

    # ------------------------------------------------------------------
    # Public: snapshot for read-only consumption
    # ------------------------------------------------------------------

    def snapshot(self) -> WorkflowProject:
        """Return a deep copy of the project state for read-only consumption.

        The copy is created under lock, guaranteeing a consistent point-in-time
        view. Callers (e.g. renderer) can safely read the returned object
        without risk of concurrent mutation.
        """
        while self._has_waiting_writer():
            time.sleep(0.001)
        with self._lock:
            source_metrics = self._project.metrics
            metrics = WorkflowMetrics.model_construct(
                total_agents=source_metrics.total_agents,
                completed_agents=source_metrics.completed_agents,
                failed_agents=source_metrics.failed_agents,
                cached_agents=source_metrics.cached_agents,
                total_tokens=source_metrics.total_tokens,
                total_duration_s=source_metrics.total_duration_s,
                phases_completed=source_metrics.phases_completed,
            )
            phases = [
                PhaseProgress.model_construct(
                    title=phase.title,
                    agents=[
                        AgentProgress.model_construct(
                            label=agent.label,
                            tool=agent.tool,
                            model=agent.model,
                            role=agent.role,
                            task_summary=agent.task_summary,
                            status=agent.status,
                            token_usage=agent.token_usage,
                            duration_s=agent.duration_s,
                            error=agent.error,
                            started_at=agent.started_at,
                            finished_at=agent.finished_at,
                            current_activity=agent.current_activity,
                            activity_updated_at=agent.activity_updated_at,
                            result=agent.result,
                            call_index=agent.call_index,
                            subagents=[
                                item.model_copy(deep=True)
                                for item in agent.subagents
                            ],
                        )
                        for agent in phase.agents
                    ],
                    started_at=phase.started_at,
                    finished_at=phase.finished_at,
                )
                for phase in self._project.phases
            ]
            return WorkflowProject.model_construct(
                workflow_id=self._project.workflow_id,
                name=self._project.name,
                description=self._project.description,
                status=self._project.status,
                requirement=self._project.requirement,
                script_path=self._project.script_path,
                meta=copy.deepcopy(self._project.meta) if self._project.meta else None,
                metrics=metrics,
                phases=phases,
                result=self._project.result,
                error=self._project.error,
                started_at=self._project.started_at,
                finished_at=self._project.finished_at,
                pending=(copy.deepcopy(self._project.pending) if self._project.pending else None),
                initiator_user_id=self._project.initiator_user_id,
                selected_tools=(
                    list(self._project.selected_tools) if self._project.selected_tools is not None else None
                ),
                tool_model_map=dict(self._project.tool_model_map),
                run_spec=copy.deepcopy(self._project.run_spec),
                reviewer_evidence=[
                    evidence.model_copy(deep=True)
                    for evidence in self._project.reviewer_evidence
                ],
                orchestrator_selection_state=copy.deepcopy(self._project.orchestrator_selection_state),
                review_selection_state=copy.deepcopy(self._project.review_selection_state),
            )

    @property
    def project(self) -> WorkflowProject:
        """Direct access (for backward compat); prefer snapshot()."""
        return self._project

    # ------------------------------------------------------------------
    # Private helpers (called under lock)
    # ------------------------------------------------------------------

    @contextmanager
    def _write_locked(self):
        """Acquire the state lock while making waiting writers visible."""
        with self._writer_wait_lock:
            self._writers_waiting += 1
        self._lock.acquire()
        with self._writer_wait_lock:
            self._writers_waiting -= 1
        try:
            yield
        finally:
            self._lock.release()

    def _has_waiting_writer(self) -> bool:
        with self._writer_wait_lock:
            return self._writers_waiting > 0

    def _find_or_create_phase(self, phase_title: str) -> PhaseProgress:
        """Find existing phase by title, or fall back to the last active phase.

        When an agent() call omits the phase field (resolved as "default"),
        assign it to the most recently created phase rather than creating a
        spurious "default" phase.
        """
        if phase_title and phase_title != "default":
            for phase in self._project.phases:
                if phase.title == phase_title:
                    return phase
        elif self._project.phases:
            return self._project.phases[-1]
        new_phase = PhaseProgress(title=phase_title or "default", started_at=time.time())
        self._project.phases.append(new_phase)
        return new_phase

    def _find_agent(self, label: str) -> Optional[AgentProgress]:
        """Find an agent by label across all phases (most recent first)."""
        for phase in reversed(self._project.phases):
            for agent in reversed(phase.agents):
                if agent.label == label:
                    return agent
        return None

    def _rebuild_indexes(self) -> None:
        """Build fast lookup indexes from an existing project snapshot."""
        for phase in self._project.phases:
            for agent in phase.agents:
                if not agent.label:
                    continue
                self._label_to_agent[agent.label] = agent

    def _make_unique_label(self, requested: str) -> str:
        """Return a label not already present in the current project."""
        base = requested.strip() or "agent"
        if base not in self._label_to_agent:
            self._label_counters.setdefault(base, 1)
            return base

        next_index = self._label_counters.get(base, 1) + 1
        while True:
            candidate = f"{base} #{next_index}"
            if candidate not in self._label_to_agent:
                self._label_counters[base] = next_index
                return candidate
            next_index += 1

    @staticmethod
    def _is_terminal_agent(agent: AgentProgress) -> bool:
        return agent.status in (
            AgentStatus.DONE,
            AgentStatus.FAILED,
            AgentStatus.CACHED,
            AgentStatus.CANCELLED,
        )

    def _is_workflow_terminal(self) -> bool:
        return self._project.status in (
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
            WorkflowStatus.CANCELLED,
        )

    def _close_open_agents(self, *, error: str, finished_at: float) -> None:
        """Move non-terminal agents out of RUNNING/PENDING for terminal cards."""
        for phase in self._project.phases:
            for agent in phase.agents:
                if self._is_terminal_agent(agent):
                    continue
                agent.status = AgentStatus.FAILED
                agent.error = error
                agent.finished_at = finished_at
                if agent.duration_s <= 0 and agent.started_at:
                    agent.duration_s = max(0.0, finished_at - agent.started_at)
                self._project.metrics.failed_agents += 1
                self._project.metrics.completed_agents += 1
