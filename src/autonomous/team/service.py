"""Journal-backed coordinator for visible employee collaboration runs."""

from __future__ import annotations

import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Callable, Protocol

from ..ingress.models import canonical_utc
from ..journal.blob_store import BlobStore
from ..journal.frame import JournalEvent
from ..journal.writer import CommitState, JournalWriter
from .coordinator import DecisionProvider, TeamCoordinatorActor
from .models import TeamRunPhase

MAX_HANDOFFS = 8
MAX_DEPTH = 4
MAX_FANOUT = 4
_MAX_TASK_CHARS = 4_000
_RESULT_CONTEXT_CHARS = 4_000
_RUN_EVENT_FIELDS = {
    "team.run.created": frozenset(
        {
            "tenant_key",
            "message_id",
            "chat_id",
            "requester_principal_id",
            "task_digest",
            "max_handoffs",
            "max_depth",
            "max_fanout",
        }
    ),
    "team.run.stopping": frozenset({"reason_code"}),
    "team.run.completed": frozenset({"result_digest", "history_record_id"}),
    "team.run.action_required": frozenset({"error_code"}),
}


class TeamServiceError(RuntimeError):
    """A team run could not safely progress."""


@dataclass(frozen=True, slots=True)
class TeamTarget:
    agent_id: str
    name: str
    role: str = ""
    capabilities: tuple[str, ...] = ()
    runtime_status: str = "ready"
    mailbox_load: int = 0


@dataclass(frozen=True, slots=True)
class TeamAttemptResult:
    status: str
    output: str = ""
    history_record_id: str = ""
    error_code: str = ""
    retry_allowed: bool = True


@dataclass(frozen=True, slots=True)
class TeamRunState:
    run_id: str
    tenant_key: str
    message_id: str
    chat_id: str
    requester_principal_id: str
    task_digest: str
    status: str = "running"
    result: str = ""


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TeamServiceError(f"invalid Team run {name}")
    return value


class TeamAdmissionError(TeamServiceError):
    """A team run was rejected before durable admission."""

    def __init__(self, error_code: str) -> None:
        super().__init__(_required_text(error_code, "admission error_code"))
        self.error_code = error_code


def _reduce_team_run_event(
    state: TeamRunState | None,
    event: JournalEvent,
) -> TeamRunState | None:
    """Apply the exact, monotonic Team run state machine."""

    fields = _RUN_EVENT_FIELDS.get(event.event_type)
    if fields is None:
        return state
    if frozenset(event.payload) != fields:
        raise TeamServiceError(f"invalid {event.event_type} payload")
    if event.event_type == "team.run.created":
        if state is not None:
            raise TeamServiceError("duplicate Team run creation")
        if (
            event.payload["max_handoffs"] != MAX_HANDOFFS
            or event.payload["max_depth"] != MAX_DEPTH
            or event.payload["max_fanout"] != MAX_FANOUT
        ):
            raise TeamServiceError("invalid Team run bounds")
        digest = _required_text(event.payload["task_digest"], "task_digest")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise TeamServiceError("invalid Team run task_digest")
        return TeamRunState(
            run_id=_required_text(event.aggregate_id, "run_id"),
            tenant_key=_required_text(event.payload["tenant_key"], "tenant_key"),
            message_id=_required_text(event.payload["message_id"], "message_id"),
            chat_id=_required_text(event.payload["chat_id"], "chat_id"),
            requester_principal_id=_required_text(
                event.payload["requester_principal_id"], "requester_principal_id"
            ),
            task_digest=digest,
        )
    if state is None:
        raise TeamServiceError("Team run transition precedes creation")
    if event.event_type == "team.run.stopping":
        if state.status != "running":
            raise TeamServiceError("illegal Team run stopping transition")
        _required_text(event.payload["reason_code"], "reason_code")
        return replace(state, status="stopping")
    if event.event_type == "team.run.completed":
        if state.status != "running":
            raise TeamServiceError("illegal Team run completed transition")
        digest = _required_text(event.payload["result_digest"], "result_digest")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise TeamServiceError("invalid Team result_digest")
        _required_text(event.payload["history_record_id"], "history_record_id")
        return replace(state, status="completed")
    if state.status not in {"running", "stopping"}:
        raise TeamServiceError("illegal Team run action-required transition")
    _required_text(event.payload["error_code"], "error_code")
    return replace(state, status="action_required")


class TeamBackend(Protocol):
    def list_active(self, tenant_key: str, chat_id: str) -> tuple[TeamTarget, ...]: ...

    def submit(
        self,
        *,
        run_id: str,
        step_id: str,
        target: TeamTarget,
        tenant_key: str,
        chat_id: str,
        message_id: str,
        requester_principal_id: str,
        instruction: str,
        deadline_at: str,
    ) -> str: ...

    def result(self, acceptance_id: str) -> TeamAttemptResult | None: ...

    def cancel(
        self,
        acceptance_id: str,
        *,
        run_id: str,
        step_id: str,
    ) -> TeamAttemptResult: ...

    def notify(
        self,
        message_id: str,
        chat_id: str,
        result: str,
        *,
        idempotency_key: str = "",
    ) -> None: ...

    def submit_direct(
        self,
        *,
        target: TeamTarget,
        tenant_key: str,
        chat_id: str,
        message_id: str,
        requester_principal_id: str,
        instruction: str,
    ) -> str: ...


class EmployeeTeamService:
    """Coordinate a bounded analyst -> reviewer -> synthesizer team run."""

    def __init__(
        self,
        *,
        writer: JournalWriter,
        backend: TeamBackend,
        attempt_timeout_seconds: float = 600.0,
        poll_seconds: float = 0.1,
        clock: Callable[[], datetime] | None = None,
        runtime_mode: str,
        blob_store: BlobStore | None = None,
        active_key_id: str = "",
        coordinator_tool: str = "coco",
        coordinator_model: str = "",
        coordinator_profile: str = "",
        coordinator_effort: str = "",
        coordinator_decision_provider: DecisionProvider | None = None,
    ) -> None:
        if runtime_mode not in {"legacy_pipeline", "coordinator"}:
            raise ValueError("invalid team runtime mode")
        if runtime_mode == "coordinator" and (blob_store is None or not active_key_id):
            raise ValueError("coordinator mode requires encrypted Blob storage")
        self._writer = writer
        self._backend = backend
        self._timeout = float(attempt_timeout_seconds)
        self._poll = float(poll_seconds)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="employee-team")
        self._lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        self._closed = False
        self._stop = threading.Event()
        self._runtime_mode = runtime_mode
        self._coordinator = (
            TeamCoordinatorActor(
                writer=writer,
                blob_store=blob_store,
                active_key_id=active_key_id,
                backend=backend,
                coordinator_tool=coordinator_tool,
                coordinator_model=coordinator_model,
                coordinator_profile=coordinator_profile,
                coordinator_effort=coordinator_effort,
                attempt_timeout_seconds=attempt_timeout_seconds,
                poll_seconds=poll_seconds,
                clock=clock,
                decision_provider=coordinator_decision_provider,
            )
            if runtime_mode == "coordinator"
            else None
        )

    @property
    def runtime_mode(self) -> str:
        """Expose the selected path without treating local tests as cutover proof."""

        return self._runtime_mode

    @property
    def persistent_coordinator_active(self) -> bool:
        return self._coordinator is not None

    def start_task(
        self,
        *,
        tenant_key: str,
        message_id: str,
        chat_id: str,
        requester_principal_id: str,
        task: str,
    ) -> TeamRunState:
        values = (tenant_key, message_id, chat_id, requester_principal_id, task)
        if not all(isinstance(value, str) and value.strip() == value and value for value in values):
            raise ValueError("team task coordinates are required")
        if len(task) > _MAX_TASK_CHARS:
            raise ValueError("team task exceeds maximum length")
        if self._coordinator is not None:
            run_id = self._coordinator.task_run_id(
                tenant_key=tenant_key,
                chat_id=chat_id,
                message_id=message_id,
            )
            with self._lock:
                if self._closed:
                    raise TeamServiceError("team service is closed")
            existing = self._coordinator.projection().runs.get(run_id)
            if existing is not None:
                state = self._adapt_v2(existing)
                if state.status != "running":
                    raise TeamAdmissionError(f"team_run_{state.status}")
                return state

            targets = tuple(self._backend.list_active(tenant_key, chat_id))
            if not targets:
                raise TeamAdmissionError("no_active_team_employee")
            with self._lock:
                if self._closed:
                    raise TeamServiceError("team service is closed")
            run, created = self._coordinator.admit_task(
                tenant_key=tenant_key,
                message_id=message_id,
                chat_id=chat_id,
                requester_principal_id=requester_principal_id,
                task=task,
            )
            state = self._adapt_v2(run)
            if not created and state.status != "running":
                raise TeamAdmissionError(f"team_run_{state.status}")
            return state
        run_id = "teamrun_" + hashlib.sha256(
            f"{tenant_key}\0{message_id}".encode()
        ).hexdigest()
        with self._lock:
            if self._closed:
                raise TeamServiceError("team service is closed")
            existing = self.get_run(run_id)
            if existing is not None:
                if existing.status != "running":
                    raise TeamAdmissionError(f"team_run_{existing.status}")
                return existing

        targets = tuple(self._backend.list_active(tenant_key, chat_id))
        with self._lock:
            if self._closed:
                raise TeamServiceError("team service is closed")
            existing = self.get_run(run_id)
            if existing is not None:
                if existing.status != "running":
                    raise TeamAdmissionError(f"team_run_{existing.status}")
                return existing
            if not targets:
                raise TeamAdmissionError("no_active_team_employee")
            state = TeamRunState(
                run_id=run_id,
                tenant_key=tenant_key,
                message_id=message_id,
                chat_id=chat_id,
                requester_principal_id=requester_principal_id,
                task_digest=hashlib.sha256(task.encode()).hexdigest(),
            )
            self._commit(
                JournalEvent(
                    event_type="team.run.created",
                    aggregate_id=run_id,
                    payload={
                        "tenant_key": tenant_key,
                        "message_id": message_id,
                        "chat_id": chat_id,
                        "requester_principal_id": requester_principal_id,
                        "task_digest": state.task_digest,
                        "max_handoffs": MAX_HANDOFFS,
                        "max_depth": MAX_DEPTH,
                        "max_fanout": MAX_FANOUT,
                    },
                )
            )
            self._executor.submit(self._execute, state, task, targets)
        return state

    def get_run(self, run_id: str) -> TeamRunState | None:
        if self._coordinator is not None:
            run = self._coordinator.projection().runs.get(run_id)
            return None if run is None else self._adapt_v2(run)
        state: TeamRunState | None = None
        for frame in self._writer.replay():
            for event in frame.events:
                if event.aggregate_id != run_id:
                    continue
                state = _reduce_team_run_event(state, event)
        return state

    def dispatch_direct(
        self,
        *,
        target: TeamTarget,
        tenant_key: str,
        chat_id: str,
        message_id: str,
        requester_principal_id: str,
        instruction: str,
    ) -> str:
        """Admit one explicit @ directly to exactly one employee Inbox."""

        submit = getattr(self._backend, "submit_direct", None)
        if not callable(submit):
            raise TeamServiceError("direct employee dispatch is unavailable")
        return submit(
            target=target,
            tenant_key=tenant_key,
            chat_id=chat_id,
            message_id=message_id,
            requester_principal_id=requester_principal_id,
            instruction=instruction,
        )

    def record_collaboration_event(self, **coordinates: str) -> bool:
        if self._coordinator is None:
            return False
        return self._coordinator.record_collaboration_event(**coordinates)

    def recover(self) -> int:
        """Terminalize runs whose in-memory instruction was lost on restart."""
        if self._coordinator is not None:
            return self._coordinator.recover()
        latest: dict[str, TeamRunState] = {}
        for frame in self._writer.replay():
            for event in frame.events:
                if event.event_type == "team.run.created":
                    state = self.get_run(event.aggregate_id)
                    if state is not None:
                        latest[event.aggregate_id] = state
        pending = [
            state for state in latest.values() if state.status in {"running", "stopping"}
        ]
        for state in pending:
            self._action_required(state, "restart_instruction_unavailable")
        return len(pending)

    def close(self) -> None:
        if self._coordinator is not None:
            self._coordinator.close()
        with self._lock:
            if self._closed:
                return
            self._closed = True
            run_ids = {
                event.aggregate_id
                for frame in self._writer.replay()
                for event in frame.events
                if event.event_type == "team.run.created"
            }
            for run_id in sorted(run_ids):
                current = self.get_run(run_id)
                if current is not None and current.status == "running":
                    self._commit(
                        JournalEvent(
                            event_type="team.run.stopping",
                            aggregate_id=run_id,
                            payload={"reason_code": "team_service_stopping"},
                        )
                    )
            self._stop.set()
        self._executor.shutdown(wait=True, cancel_futures=False)

    @staticmethod
    def _adapt_v2(run: object) -> TeamRunState:
        phase = run.phase
        status = (
            "completed"
            if phase is TeamRunPhase.COMPLETED
            else "action_required"
            if phase is TeamRunPhase.BLOCKED
            else "canceled"
            if phase is TeamRunPhase.CANCELED
            else "running"
        )
        return TeamRunState(
            run_id=run.run_id,
            tenant_key=run.tenant_key,
            message_id=run.message_id,
            chat_id=run.chat_id,
            requester_principal_id=run.requester_principal_id,
            task_digest=run.task_ref.payload_hash,
            status=status,
        )

    def _can_progress(self, state: TeamRunState) -> bool:
        if self._stop.is_set():
            return False
        current = self.get_run(state.run_id)
        return current is None or current.status == "running"

    def _execute(
        self,
        state: TeamRunState,
        task: str,
        targets: tuple[TeamTarget, ...],
    ) -> None:
        try:
            lead = targets[0]
            reviewer = targets[1] if len(targets) > 1 else lead
            alternate = targets[2] if len(targets) > 2 else None

            analysis = self._run_step(
                state,
                step_id="analysis",
                depth=1,
                target=lead,
                instruction=(
                    "你是团队主执行者。分析并完成下面任务，输出可交接的方案、证据和风险。\n\n"
                    + task
                ),
            )
            if (
                analysis.status != "completed"
                and analysis.retry_allowed
                and self._can_progress(state)
            ):
                analysis = self._retry_step(state, "analysis", 1, alternate or reviewer, task)
            if analysis.status != "completed":
                self._action_required(state, analysis.error_code or "analysis_failed")
                return

            review_instruction = (
                "你是独立评审员工。审查上一位员工的交付，指出缺陷并给出修订版。\n\n"
                f"原始任务：\n{task}\n\n上一位员工交付：\n"
                f"{analysis.output[:_RESULT_CONTEXT_CHARS]}"
            )
            if not self._can_progress(state):
                self._action_required(state, "team_service_stopping")
                return
            review = self._run_step(
                state,
                step_id="review",
                depth=2,
                target=reviewer,
                instruction=review_instruction,
            )
            if (
                review.status != "completed"
                and review.retry_allowed
                and alternate is not None
                and self._can_progress(state)
            ):
                review = self._retry_step(state, "review", 2, alternate, review_instruction)
            if review.status != "completed":
                self._action_required(state, review.error_code or "review_failed")
                return

            synthesis_instruction = (
                "你是团队负责人。综合执行稿和评审稿，直接输出最终可交付结果；不要描述协作过程。\n\n"
                f"原始任务：\n{task}\n\n执行稿：\n{analysis.output[:_RESULT_CONTEXT_CHARS]}"
                f"\n\n评审稿：\n{review.output[:_RESULT_CONTEXT_CHARS]}"
            )
            if not self._can_progress(state):
                self._action_required(state, "team_service_stopping")
                return
            synthesis = self._run_step(
                state,
                step_id="synthesis",
                depth=3,
                target=lead,
                instruction=synthesis_instruction,
            )
            if (
                synthesis.status != "completed"
                and synthesis.retry_allowed
                and alternate is not None
                and self._can_progress(state)
            ):
                synthesis = self._retry_step(state, "synthesis", 3, alternate, synthesis_instruction)
            if synthesis.status != "completed":
                self._action_required(state, synthesis.error_code or "synthesis_failed")
                return

            with self._lock:
                if not self._can_progress(state):
                    self._action_required(state, "team_service_stopping")
                    return
                self._commit_effect(state.run_id, "notify", "prepared")
                self._commit_effect(state.run_id, "notify", "executing")
                self._backend.notify(state.message_id, state.chat_id, synthesis.output)
                self._commit_effect(state.run_id, "notify", "committed")
                self._commit(
                    JournalEvent(
                        event_type="team.run.completed",
                        aggregate_id=state.run_id,
                        payload={
                            "result_digest": hashlib.sha256(synthesis.output.encode()).hexdigest(),
                            "history_record_id": synthesis.history_record_id,
                        },
                    )
                )
        except Exception:
            self._action_required(state, "team_coordinator_failed")

    def _retry_step(
        self,
        state: TeamRunState,
        base_step: str,
        depth: int,
        target: TeamTarget,
        instruction: str,
    ) -> TeamAttemptResult:
        return self._run_step(
            state,
            step_id=f"{base_step}-retry",
            depth=min(depth + 1, MAX_DEPTH),
            target=target,
            instruction=instruction,
        )

    def _run_step(
        self,
        state: TeamRunState,
        *,
        step_id: str,
        depth: int,
        target: TeamTarget,
        instruction: str,
    ) -> TeamAttemptResult:
        with self._lock:
            if depth > MAX_DEPTH:
                return TeamAttemptResult("action_required", error_code="team_depth_exceeded")
            if not self._can_progress(state):
                return TeamAttemptResult(
                    "action_required", error_code="team_service_stopping"
                )
            # Re-read after the progress hook so even a re-entrant close between
            # admission checks cannot place work beyond the stopping fence.
            current = self.get_run(state.run_id)
            if self._stop.is_set() or (current is not None and current.status != "running"):
                return TeamAttemptResult(
                    "action_required", error_code="team_service_stopping"
                )
            aggregate = f"{state.run_id}:{step_id}"
            digest = hashlib.sha256(instruction.encode()).hexdigest()
            deadline = self._clock() + timedelta(seconds=self._timeout)
            deadline_at = canonical_utc(deadline, "team_deadline_at")
            self._commit(
                JournalEvent(
                    event_type="team.step.prepared",
                    aggregate_id=aggregate,
                    payload={
                        "run_id": state.run_id,
                        "step_id": step_id,
                        "agent_id": target.agent_id,
                        "depth": depth,
                        "instruction_digest": digest,
                        "deadline_at": deadline_at,
                    },
                )
            )
            self._commit_effect(aggregate, "employee_dispatch", "prepared")
            self._commit_effect(aggregate, "employee_dispatch", "executing")
            try:
                acceptance_id = self._backend.submit(
                    run_id=state.run_id,
                    step_id=step_id,
                    target=target,
                    tenant_key=state.tenant_key,
                    chat_id=state.chat_id,
                    message_id=state.message_id,
                    requester_principal_id=state.requester_principal_id,
                    instruction=instruction,
                    deadline_at=deadline_at,
                )
            except Exception:
                self._commit_effect(aggregate, "employee_dispatch", "action_required")
                return TeamAttemptResult("action_required", error_code="team_dispatch_failed")
            self._commit(
                JournalEvent(
                    event_type="team.step.submitted",
                    aggregate_id=aggregate,
                    payload={"run_id": state.run_id, "step_id": step_id, "acceptance_id": acceptance_id},
                )
            )
        while self._clock() < deadline and not self._stop.is_set():
            result = self._backend.result(acceptance_id)
            if result is not None:
                return self._terminalize_step_result(state, step_id, aggregate, result)
            time.sleep(self._poll)
        if self._stop.is_set():
            canceled = self._backend.cancel(
                acceptance_id,
                run_id=state.run_id,
                step_id=step_id,
            )
            return self._terminalize_step_result(
                state, step_id, aggregate, canceled
            )
        result = self._backend.result(acceptance_id)
        if result is not None:
            return self._terminalize_step_result(state, step_id, aggregate, result)
        canceled = self._backend.cancel(
            acceptance_id,
            run_id=state.run_id,
            step_id=step_id,
        )
        return self._terminalize_step_result(
            state,
            step_id,
            aggregate,
            canceled,
        )

    def _terminalize_step_result(
        self,
        state: TeamRunState,
        step_id: str,
        aggregate: str,
        result: TeamAttemptResult,
    ) -> TeamAttemptResult:
        with self._lock:
            current = self.get_run(state.run_id)
            if self._stop.is_set() or (current is not None and current.status != "running"):
                return TeamAttemptResult(
                    "action_required",
                    error_code="team_service_stopping",
                    retry_allowed=False,
                )
            self._commit_effect(
                aggregate,
                "employee_dispatch",
                "committed" if result.status == "completed" else "action_required",
            )
            event_type = "team.step.completed" if result.status == "completed" else "team.step.failed"
            self._commit(
                JournalEvent(
                    event_type=event_type,
                    aggregate_id=aggregate,
                    payload={
                        "run_id": state.run_id,
                        "step_id": step_id,
                        "status": result.status,
                        "history_record_id": result.history_record_id or "none",
                        "result_digest": hashlib.sha256(result.output.encode()).hexdigest(),
                        "error_code": result.error_code or "none",
                    },
                )
            )
        return result

    def _action_required(self, state: TeamRunState, error_code: str) -> None:
        with self._lock:
            current = self.get_run(state.run_id)
            if current is not None and current.status not in {"running", "stopping"}:
                return
            effect_states: dict[tuple[str, str], str] = {}
            for frame in self._writer.replay():
                for event in frame.events:
                    if not event.event_type.startswith("team.effect."):
                        continue
                    if event.aggregate_id != state.run_id and not event.aggregate_id.startswith(
                        state.run_id + ":"
                    ):
                        continue
                    effect_type = event.payload.get("effect_type")
                    if isinstance(effect_type, str) and effect_type:
                        effect_states[(event.aggregate_id, effect_type)] = event.event_type.rsplit(".", 1)[-1]
            for (aggregate_id, effect_type), effect_state in effect_states.items():
                if effect_state in {"prepared", "executing"}:
                    self._commit_effect(aggregate_id, effect_type, "action_required")
            failure_notice = (
                "⚠️ 团队任务未能自动收敛，已安全停止并转为人工处理。"
                f"错误码：`{error_code}`"
            )
            if current is None or current.status != "stopping":
                notify_aggregate = f"{state.run_id}:failure-notify"
                try:
                    self._commit_effect(notify_aggregate, "notify", "prepared")
                    self._commit_effect(notify_aggregate, "notify", "executing")
                    self._backend.notify(state.message_id, state.chat_id, failure_notice)
                    self._commit_effect(notify_aggregate, "notify", "committed")
                except Exception:
                    self._commit_effect(notify_aggregate, "notify", "action_required")
            self._commit(
                JournalEvent(
                    event_type="team.run.action_required",
                    aggregate_id=state.run_id,
                    payload={"error_code": error_code},
                )
            )

    def _commit_effect(self, aggregate_id: str, effect_type: str, state: str) -> None:
        self._commit(
            JournalEvent(
                event_type=f"team.effect.{state}",
                aggregate_id=aggregate_id,
                payload={"effect_type": effect_type},
            )
        )

    def _commit(self, event: JournalEvent) -> None:
        with self._lock:
            if event.event_type in _RUN_EVENT_FIELDS:
                _reduce_team_run_event(self.get_run(event.aggregate_id), event)
            with self._writer.transaction_guard():
                last = self._writer.get_last_frame()
                result = self._writer.commit(
                    (event,),
                    self._writer.get_aggregate_versions((event.aggregate_id,)),
                    expected_head_sequence=0 if last is None else last.sequence,
                    expected_head_hash="" if last is None else last.frame_hash,
                )
        if result.state is not CommitState.ANCHORED:
            raise TeamServiceError("team event was not anchored")
