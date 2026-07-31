"""WorkflowEngine — orchestrates multi-step AI workflows via Node.js runtime."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..engine_base import BaseEngine, EngineRunState
from ..spec_engine.review_agents import ReviewAgentBinding
from .bridge import RuntimeBridge
from .constants import (
    DEFAULT_MAX_CONCURRENT,
    MAX_TOTAL_AGENTS,
    NODE_MIN_VERSION,
    PROGRESS_HEARTBEAT_S,
    STATE_FILENAME,
)
from .errors import _strip_internal_details
from .executor import AgentExecutor
from .history import WorkflowHistory
from .journal import WorkflowJournal
from .models import (
    AgentCallParams,
    AgentCallResult,
    AgentStatus,
    ReviewerEvidence,
    WorkflowMetrics,
    WorkflowProject,
    WorkflowStatus,
)
from .progress_coalescer import ProgressCoalescer
from .renderer import WorkflowProgressRenderer
from .run_spec import WorkflowRunSpec
from .state_manager import WorkflowStateManager

logger = logging.getLogger(__name__)


def _remove_owned_script_artifact(path: str | None, root_path: str) -> None:
    """Remove only Workflow's generated source or immutable temp copy."""
    if not path:
        return
    candidate = os.path.realpath(os.path.abspath(path))
    basename = os.path.basename(candidate)
    generated_dir = os.path.realpath(os.path.join(root_path, ".ghostap", "workflow_scripts"))
    temp_root = os.path.realpath(tempfile.gettempdir())
    generated_source = bool(
        os.path.dirname(candidate) == generated_dir
        and basename.endswith(".js")
        and (basename.startswith("generated-workflow-") or basename == "generated_workflow.js")
    )
    execution_copy = bool(
        basename.startswith("ghostap-confirmed-")
        and basename.endswith(".js")
        and os.path.commonpath((candidate, temp_root)) == temp_root
    )
    if not (generated_source or execution_copy):
        return
    try:
        os.remove(candidate)
    except OSError:
        pass


def _node_version_required_text() -> str:
    """Return the user-visible Node.js version-gate message.

    The minimum version is derived from :data:`NODE_MIN_VERSION` so that all
    user-facing strings stay in sync when the requirement is bumped.
    """
    return (
        f"Node.js >= {NODE_MIN_VERSION[0]} is required for workflow mode. "
        f"Please install Node.js and ensure it's in PATH."
    )


def _decode_result_payload(result_text: str) -> Any:
    """Decode JSON result wrappers, including JSON strings that contain JSON."""
    parsed: Any = result_text
    for _ in range(3):
        if not isinstance(parsed, str):
            return parsed
        text = parsed.strip()
        if not text or not text.startswith(("{", "[", '"')):
            return parsed
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            return parsed
    return parsed


def _terminal_failure_from_result(result_text: str) -> str | None:
    """Return a user-visible error when a normal JS result encodes failure.

    Only triggers when the *top-level* result is purely an error — not when
    the result contains partial agent errors alongside valid output (which is
    normal for scripts that try-catch individual agent failures).
    """
    parsed = _decode_result_payload(str(result_text or ""))
    if not isinstance(parsed, dict):
        return None

    error = parsed.get("error")
    error_text = str(error or "").strip()
    message_text = str(parsed.get("message") or "").strip()
    status = str(parsed.get("status") or "").strip().lower()

    # A result is only a terminal failure if it looks like a *pure* error
    # payload — i.e., it has no meaningful output alongside the error.
    # Scripts that catch sub-agent errors and include them in a report (e.g.,
    # {"final_report": "...", "error": "agent X failed"}) are NOT failures.
    has_meaningful_output = any(
        k not in ("error", "message", "status", "stage", "fallback") and bool(v) for k, v in parsed.items()
    )

    if has_meaningful_output:
        return None

    is_failure = (
        bool(parsed.get("fallback"))
        or status in {"error", "failed", "failure"}
        or ("error" in parsed and bool(error_text))
    )
    if not is_failure:
        return None

    reason = error_text or message_text
    if not reason:
        return None
    stage = str(parsed.get("stage") or "").strip()
    if stage:
        return f"{stage}: {reason}"
    return reason


def _terminal_failure_from_project(project: WorkflowProject) -> str | None:
    """Fail closed when the bridge returns while agent calls are unfinished."""
    for phase in project.phases:
        for agent in phase.agents:
            if agent.status in (AgentStatus.RUNNING, AgentStatus.PENDING):
                label = agent.label or "agent"
                status = agent.status.value if hasattr(agent.status, "value") else str(agent.status)
                return f"Workflow finished while agent {label} was still {status}"

    metrics = project.metrics
    if metrics.total_agents > metrics.completed_agents:
        return f"Workflow finished before all agent calls completed ({metrics.completed_agents}/{metrics.total_agents})"
    return None


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@dataclass
class WorkflowEngineCallbacks:
    """Event callbacks for the Workflow Engine handler layer."""

    on_progress: Optional[Callable[[dict[str, Any]], None]] = None
    on_done: Optional[Callable[[WorkflowProject], None]] = None
    on_error: Optional[Callable[[str], None]] = None
    on_log: Optional[Callable[[str], None]] = None
    on_phase: Optional[Callable[[str], None]] = None
    on_agent_start: Optional[Callable[[str, str], None]] = None  # (label, tool)
    # AC4: on_agent_done is a lightweight meta-info callback; the payload
    # deliberately excludes the agent output/parsed body so that handler-layer
    # subscribers cannot accidentally leak intermediate results into the main
    # agent chat context. Only final results are delivered via on_done.
    on_agent_done: Optional[Callable[[str, dict], None]] = None  # (label, meta_dict)


# ---------------------------------------------------------------------------
# WorkflowEngine
# ---------------------------------------------------------------------------


class WorkflowEngine(BaseEngine):
    """Orchestrates multi-step AI workflows using a Node.js runtime bridge.

    Lifecycle:
        1. Handler calls execute_workflow(requirement, script_path, callbacks)
        2. Engine creates journal, executor, bridge
        3. Bridge spawns Node.js subprocess running the workflow script
        4. Script issues agent() calls via JSON-RPC → bridge dispatches to executor
        5. Executor creates one-shot ACP/CLI sessions per agent call
        6. Results flow back through the bridge → script continues
        7. On done/error, engine updates project state and fires callbacks
    """

    _state_filename: str = STATE_FILENAME
    _gc_label: str = "Workflow"
    _gc_threshold_default: float = 85.0

    # Cache root for workflow state (mirrors project path under ~/.cache/ghostAp)
    _CACHE_ROOT: str = "~/.cache/ghostAp"

    def __init__(
        self,
        chat_id: str,
        root_path: str,
        agent_type: str = "coco",
        engine_name: str = "Coco",
        model_name: Optional[str] = None,
    ):
        super().__init__(chat_id, root_path, agent_type, engine_name, model_name)

        # Workflow-specific state — initialized to IDLE so handler code
        # can set pending state before execute_workflow() is called.
        self._project: Optional[WorkflowProject] = WorkflowProject()
        self._bridge: Optional[RuntimeBridge] = None
        self._journal: Optional[WorkflowJournal] = None
        self._executor: Optional[AgentExecutor] = None
        self._renderer_wf: Optional[WorkflowProgressRenderer] = None
        self._state_manager: Optional[WorkflowStateManager] = None
        self._progress_coalescer: Optional[ProgressCoalescer] = None
        self._cancel_event = threading.Event()
        self._callbacks: Optional[WorkflowEngineCallbacks] = None
        self._run_spec: Optional[WorkflowRunSpec] = None

        # Heartbeat: periodically re-renders the progress card while a run is
        # active so the live elapsed counters keep advancing even when no
        # agent start/done/phase event fires during a long blocking agent()
        # call. Plain Event (no lock needed — set/clear/wait are atomic).
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None

        # Handler-owned lifecycle tokens. They make script generation and the
        # scheduler hand-off cancellable before execute_workflow() claims the
        # engine, without weakening the runtime cancel_event reuse contract.
        self._workflow_selection_owner: Any = None
        self._script_generation_owner: Any = None
        self._workflow_start_owner: Any = None
        self._retired_lifecycle_owners: list[Any] = []
        self._closing = False
        self._run_done_event = threading.Event()
        self._run_done_event.set()
        self._run_thread_id: int | None = None

        # Counters for safety fuse
        self._agent_call_count: int = 0

        # Map JSON-RPC request_id → effective agent label. Used by
        # _handle_agent_aborted to look up agents by request_id instead of
        # raw label string, which avoids mismatches when state_manager
        # disambiguates duplicate labels (e.g. "agent-1" → "agent-1 #2").
        self._request_to_label: dict[Any, str] = {}
        self._request_to_label_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def workflow_project(self) -> Optional[WorkflowProject]:
        with self._lock:
            return self._project

    @property
    def cancel_event(self) -> threading.Event:
        return self._cancel_event

    def _state_dir(self) -> str:
        """Return the cache directory for workflow state (outside project tree).

        Mirrors the project absolute path under ``~/.cache/ghostAp`` so that
        state files do not pollute the project directory or git status.
        Example: project at ``/data00/home/user/work/proj``
        → ``~/.cache/ghostAp/data00/home/user/work/proj/``
        """
        import os
        from pathlib import Path

        cache_root = os.path.abspath(os.path.expanduser(self._CACHE_ROOT))
        abs_project = os.path.abspath(self.root_path)
        _, tail = os.path.splitdrive(abs_project)
        parts = [part for part in Path(tail).parts if part not in (os.sep, "")]
        return os.path.join(cache_root, *parts)

    def save_state(self, filepath: Optional[str] = None) -> str:
        """Persist workflow state to ~/.cache/ghostAp/<project_path>/ instead of project root."""
        import os

        if not filepath:
            state_dir = self._state_dir()
            os.makedirs(state_dir, exist_ok=True)
            filepath = os.path.join(state_dir, self._state_filename)
        return super().save_state(filepath)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _retire_lifecycle_owner_locked(self, owner: Any) -> None:
        """Keep a revoked owner visible until cleanup proves quiescence."""
        if owner is None:
            return
        if any(existing is owner for existing in self._retired_lifecycle_owners):
            return
        self._retired_lifecycle_owners.append(owner)

    def retire_lifecycle_owner(self, owner: Any) -> None:
        """Persist a revoked lifecycle owner across cleanup retries."""
        with self._lock:
            self._retire_lifecycle_owner_locked(owner)

    def cleanup(self):
        """Override to remove orphaned pending script files and release
        thread-pool resources (executor + bridge). Safe to call more than
        once; shutdown is idempotent. Returns whether workers quiesced.
        """
        deadline = time.monotonic() + 30.0
        quiesced = True
        with self._lock:
            self._closing = True
            run_was_active = not self._run_done_event.is_set()
            run_thread_id = self._run_thread_id
            live_lifecycle_owners = tuple(
                owner
                for owner in (
                    self._workflow_selection_owner,
                    self._script_generation_owner,
                    self._workflow_start_owner,
                )
                if owner is not None
            )
            for owner in live_lifecycle_owners:
                self._retire_lifecycle_owner_locked(owner)
            lifecycle_owners = tuple(self._retired_lifecycle_owners)
            for owner in lifecycle_owners:
                stop_event = getattr(owner, "stop_event", None)
                if stop_event is not None:
                    stop_event.set()
                heartbeat_stop_event = getattr(owner, "heartbeat_stop_event", None)
                if heartbeat_stop_event is not None:
                    heartbeat_stop_event.set()
            self._workflow_selection_owner = None
            self._script_generation_owner = None
            self._workflow_start_owner = None
            project = self._project
            artifact_paths = [
                path
                for owner in lifecycle_owners
                for path in (
                    getattr(owner, "source_script_path", None),
                    getattr(owner, "execution_script_path", None),
                )
                if path
            ]
            if project is not None:
                if project.script_path:
                    artifact_paths.append(project.script_path)
                if project.pending and project.pending.script_path:
                    artifact_paths.append(project.pending.script_path)
        for owner in lifecycle_owners:
            delivery_lock = getattr(owner, "delivery_lock", None)
            if delivery_lock is not None:
                remaining = max(0.0, deadline - time.monotonic())
                acquired = delivery_lock.acquire(timeout=remaining)
                if acquired:
                    delivery_lock.release()
                else:
                    quiesced = False
            done_event = getattr(owner, "done_event", None)
            claimed_event = getattr(owner, "claimed_event", None)
            worker_started_event = getattr(
                owner,
                "worker_started_event",
                None,
            )
            if (
                done_event is not None
                and (claimed_event is None or not claimed_event.is_set())
                and (worker_started_event is None or not worker_started_event.is_set())
            ):
                done_event.set()

        # Ensure any lingering heartbeat thread is stopped (best-effort).
        try:
            self._stop_heartbeat()
        except Exception as e:
            logger.debug("Heartbeat stop during cleanup failed: %s", str(e))

        # For an active run, only request cancellation here. Its execution
        # owner is the sole thread allowed to drain/null the bridge and publish
        # IDLE. This keeps manager cleanup bounded even if an ACP worker ignores
        # cancellation. An orphaned bridge without an execution owner can be
        # drained directly within the shared cleanup deadline.
        bridge = self._bridge
        if bridge is not None:
            try:
                bridge.stop()
                if not run_was_active:
                    wait_for_workers = getattr(
                        bridge,
                        "wait_for_workers",
                        None,
                    )
                    if callable(wait_for_workers):
                        remaining = max(
                            0.0,
                            deadline - time.monotonic(),
                        )
                        if wait_for_workers(timeout=remaining) is False:
                            quiesced = False
            except Exception:
                logger.debug(
                    "WorkflowEngine bridge stop failed",
                    exc_info=True,
                )
                quiesced = False
            if not run_was_active and quiesced and self._bridge is bridge:
                self._bridge = None

        # Release AgentExecutor thread pool (prevents thread leak across runs).
        if not run_was_active and quiesced and self._executor is not None:
            try:
                self._executor.shutdown(wait=False)
            except Exception:
                logger.debug("WorkflowEngine executor shutdown failed")
            self._executor = None

        for artifact_path in dict.fromkeys(artifact_paths):
            _remove_owned_script_artifact(artifact_path, self.root_path)

        # Clear request_id → label mapping to prevent cross-run leaks
        with self._request_to_label_lock:
            self._request_to_label.clear()

        super().cleanup()

        current_thread_id = threading.get_ident()
        for owner in lifecycle_owners:
            claimed_event = getattr(owner, "claimed_event", None)
            worker_started_event = getattr(
                owner,
                "worker_started_event",
                None,
            )
            done_event = getattr(owner, "done_event", None)
            worker_thread_id = getattr(owner, "worker_thread_id", None)
            has_worker = bool(
                (claimed_event is not None and claimed_event.is_set())
                or (worker_started_event is not None and worker_started_event.is_set())
            )
            if not has_worker or done_event is None:
                continue
            if worker_thread_id == current_thread_id:
                quiesced = False
                continue
            remaining = max(0.0, deadline - time.monotonic())
            if not done_event.wait(timeout=remaining):
                quiesced = False

        if run_was_active:
            if run_thread_id == current_thread_id:
                quiesced = False
            else:
                remaining = max(0.0, deadline - time.monotonic())
                if not self._run_done_event.wait(timeout=remaining):
                    quiesced = False
        if not quiesced:
            logger.warning("WorkflowEngine cleanup timed out before worker quiescence")
        else:
            with self._lock:
                self._retired_lifecycle_owners = [
                    owner
                    for owner in self._retired_lifecycle_owners
                    if not any(owner is cleaned_owner for cleaned_owner in lifecycle_owners)
                ]
        return quiesced

    # ------------------------------------------------------------------
    # Main execution entry point
    # ------------------------------------------------------------------

    def _build_legacy_run_spec(
        self,
        *,
        requirement: str | None,
        selected_tools: list[str] | None,
        initiator_user_id: str | None,
    ) -> WorkflowRunSpec:
        """Adapt direct/legacy callers to a complete explicit contract.

        Production confirmation never takes this path. Keeping the adapter at
        the boundary lets existing template and engine tests use the older
        signature without allowing a partially-bound mutable project inside
        the engine.
        """
        tools = tuple(
            dict.fromkeys(
                str(tool or "").strip()
                for tool in (selected_tools or [self._agent_type or "coco"])
                if str(tool or "").strip()
            )
        ) or ("coco",)
        primary_tool = tools[0]
        configured_model = (
            str(self._model_name).strip()
            if primary_tool == self._agent_type and self._model_name
            else None
        )
        orchestrator = ReviewAgentBinding(
            provider="legacy",
            tool_name=primary_tool,
            display_name=primary_tool,
            agent_type=primary_tool,
            model_name=configured_model,
            model_display_name=configured_model,
            selection_key=f"legacy:{primary_tool}:{configured_model or 'default'}",
            use_default_model=configured_model is None,
        )
        return WorkflowRunSpec(
            orchestrator=orchestrator,
            reviewers=(),
            tool_model_map={
                tool: configured_model if tool == primary_tool else None
                for tool in tools
            },
            task=str(requirement or "Workflow execution").strip() or "Workflow execution",
            chat_id=str(self.chat_id or "workflow"),
            topic_id=None,
            budget=MAX_TOTAL_AGENTS,
            deadline=None,
            auto_reviewer=True,
            initiator_user_id=initiator_user_id,
            allowed_tools=tools,
            enforce_tool_allowlist=selected_tools is not None,
        )

    def execute_workflow(
        self,
        requirement: Optional[str] = None,
        script_path: str = "",
        callbacks: Optional[WorkflowEngineCallbacks] = None,
        *,
        run_spec: Optional[WorkflowRunSpec] = None,
        selected_tools: Optional[list[str]] = None,
        initiator_user_id: Optional[str] = None,
        start_owner: Any = None,
        source_script_path: Optional[str] = None,
    ) -> WorkflowProject:
        """Execute a workflow script end-to-end.

        Args:
            requirement: Legacy form of the user's requirement. New handler
                paths pass it inside ``run_spec``.
            script_path: Absolute path to the .js workflow script.
            callbacks: Optional event callbacks for progress/completion.
            run_spec: Frozen confirmation-time execution contract.
            selected_tools: Optional tool whitelist; agents may only use these tools.
            start_owner: Optional handler lifecycle token for a queued start.
            source_script_path: Stable generated source retained for save/reuse.

        Returns:
            The final WorkflowProject with status, metrics, and result.

        Raises:
            RuntimeError: If Node.js is unavailable or the bridge fails fatally.
        """
        run_callbacks = callbacks or WorkflowEngineCallbacks()
        if run_spec is None:
            run_spec = self._build_legacy_run_spec(
                requirement=requirement,
                selected_tools=selected_tools,
                initiator_user_id=initiator_user_id,
            )
        else:
            if requirement is not None and requirement.strip() != run_spec.task:
                raise ValueError("Workflow requirement conflicts with frozen run spec")
            if selected_tools is not None and tuple(selected_tools) != run_spec.allowed_tools:
                raise ValueError("Workflow selected tools conflict with frozen run spec")
            if initiator_user_id is not None and initiator_user_id != run_spec.initiator_user_id:
                raise ValueError("Workflow initiator conflicts with frozen run spec")

        requirement = run_spec.task
        selected_tools = (
            list(run_spec.allowed_tools)
            if run_spec.enforce_tool_allowlist
            else None
        )
        initiator_user_id = run_spec.initiator_user_id

        # Parse meta from the generated script so we can honor meta.maxConcurrent
        # before the bridge / executor thread pools are created.
        script_meta = None
        max_concurrent = DEFAULT_MAX_CONCURRENT
        try:
            if script_path:
                from .templates import parse_template_meta

                script_content = None
                try:
                    with open(script_path, "r", encoding="utf-8") as f:
                        script_content = f.read()
                except OSError:
                    script_content = None
                if script_content:
                    script_meta = parse_template_meta(script_content)
            if script_meta is not None and script_meta.max_concurrent:
                max_concurrent = int(script_meta.max_concurrent)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to parse workflow script meta: %s", repr(exc))
            script_meta = None
            max_concurrent = DEFAULT_MAX_CONCURRENT

        # Initialize project state
        workflow_id = uuid.uuid4().hex[:12]
        project = WorkflowProject(
            workflow_id=workflow_id,
            status=WorkflowStatus.RUNNING,
            requirement=requirement,
            script_path=source_script_path or script_path,
            meta=script_meta,
            metrics=WorkflowMetrics(),
            started_at=time.time(),
            selected_tools=selected_tools,
            initiator_user_id=initiator_user_id,
            tool_model_map=dict(run_spec.tool_model_map),
            run_spec=run_spec.to_dict(),
        )

        # Claim a queued start and publish runtime state in one critical
        # section. If /stop_wf won the race, do not clear its cancellation or
        # replace the IDLE project with a new RUNNING one.
        start_rejected = False
        rejected_active_owner = False
        rejected_due_running = False
        rejected_project: Optional[WorkflowProject] = None
        previous_source_path: str | None = None
        with self._lock:
            if self._closing:
                logger.info("[WorkflowEngine] Dropping start while engine is closing")
                start_rejected = True
                rejected_project = self._project or WorkflowProject()
            elif self._run_state != EngineRunState.IDLE:
                logger.info("[WorkflowEngine] Dropping duplicate/concurrent start")
                start_rejected = True
                rejected_due_running = True
                rejected_active_owner = bool(
                    start_owner is not None
                    and self._workflow_start_owner is start_owner
                    and start_owner.claimed_event.is_set()
                )
                rejected_project = self._project or WorkflowProject()
            elif start_owner is not None:
                current_owner = self._workflow_start_owner
                if (
                    current_owner is not start_owner
                    or start_owner.stop_event.is_set()
                    or start_owner.claimed_event.is_set()
                ):
                    if current_owner is start_owner:
                        self._retire_lifecycle_owner_locked(
                            start_owner,
                        )
                        self._workflow_start_owner = None
                    logger.info("[WorkflowEngine] Dropping cancelled queued start")
                    start_rejected = True
                    rejected_project = self._project or WorkflowProject()
                else:
                    start_owner.claimed_event.set()
                    worker_started_event = getattr(
                        start_owner,
                        "worker_started_event",
                        None,
                    )
                    if worker_started_event is not None:
                        worker_started_event.set()
                    try:
                        object.__setattr__(
                            start_owner,
                            "worker_thread_id",
                            threading.get_ident(),
                        )
                    except Exception:
                        pass
            if not start_rejected:
                # WorkflowEngine instances are reused for the same chat/root
                # path. A prior completed/cancelled run leaves this event set;
                # only the lifecycle winner may establish the new boundary.
                previous_project = self._project
                if (
                    previous_project is not None
                    and previous_project.status
                    in {
                        WorkflowStatus.COMPLETED,
                        WorkflowStatus.FAILED,
                        WorkflowStatus.CANCELLED,
                    }
                    and previous_project.script_path != project.script_path
                ):
                    previous_source_path = previous_project.script_path
                self._cancel_event.clear()
                self._agent_call_count = 0
                # Per-run components must never leak into constructor-error
                # handling for the next run.
                self._bridge = None
                self._journal = None
                self._executor = None
                self._renderer_wf = None
                self._state_manager = None
                self._progress_coalescer = None
                self._callbacks = run_callbacks
                self._run_spec = run_spec
                self._project = project
                self._run_state = EngineRunState.RUNNING
                self._run_thread_id = threading.get_ident()
                self._run_done_event.clear()

        if start_rejected:
            if not rejected_active_owner and not (rejected_due_running and start_owner is None):
                _remove_owned_script_artifact(script_path, self.root_path)
                done_event = getattr(start_owner, "done_event", None)
                if done_event is not None:
                    done_event.set()
            return rejected_project or WorkflowProject()

        _remove_owned_script_artifact(
            previous_source_path,
            self.root_path,
        )
        with self._request_to_label_lock:
            self._request_to_label.clear()

        try:
            # Initialize components inside the lifecycle guard so constructor
            # failures cannot strand RUNNING state or a queued-start owner.
            self._journal = WorkflowJournal(self.root_path, workflow_id)
            self._executor = AgentExecutor(
                cwd=self.root_path,
                cancel_event=self._cancel_event,
                max_workers=max_concurrent,
                on_activity=self._handle_agent_activity,
            )
            self._state_manager = WorkflowStateManager(project)
            self._renderer_wf = WorkflowProgressRenderer(project)

            # Initialize progress coalescer (debounced card updates)
            if self._callbacks and self._callbacks.on_progress:
                self._progress_coalescer = ProgressCoalescer(
                    on_progress=self._callbacks.on_progress,
                )

            # Check Node.js availability
            if not RuntimeBridge.check_node_available():
                raise RuntimeError(_node_version_required_text())

            # Publish the bridge under the engine lock, then let its own
            # lifecycle gate protect the potentially slow Node readiness wait.
            # stop() can therefore cancel a hung start without waiting for
            # this engine lock, while RuntimeBridge guarantees no post-stop
            # subprocess or executor publication.
            with self._lock:
                if self._cancel_event.is_set() or (
                    start_owner is not None
                    and (self._workflow_start_owner is not start_owner or start_owner.stop_event.is_set())
                ):
                    raise RuntimeError("Workflow cancelled")
                self._bridge = RuntimeBridge(
                    script_path=script_path,
                    cwd=self.root_path,
                    max_concurrent=max_concurrent,
                    on_agent_call=self._handle_agent_call,
                    on_agent_aborted=self._handle_agent_aborted,
                    on_phase=self._handle_phase,
                    on_log=self._handle_log,
                    cancel_event=self._cancel_event,
                    allowed_tools=selected_tools,
                    initiator_user_id=project.initiator_user_id,
                    workflow_deadline_monotonic=run_spec.deadline,
                )
                bridge = self._bridge

            bridge.start()

            with self._lock:
                if (
                    self._bridge is not bridge
                    or self._cancel_event.is_set()
                    or (
                        start_owner is not None
                        and (self._workflow_start_owner is not start_owner or start_owner.stop_event.is_set())
                    )
                ):
                    raise RuntimeError("Workflow cancelled")
                # Start the heartbeat before releasing the same lifecycle
                # lock. stop() can therefore either cancel both resources or
                # win before either one is started.
                self._start_heartbeat()

            # Run the event loop (blocks until done/error/timeout)
            result_text = self._bridge.run()

            terminal_failure = _terminal_failure_from_result(result_text)
            if terminal_failure is None and self._state_manager is not None:
                terminal_failure = _terminal_failure_from_project(self._state_manager.snapshot())
            if terminal_failure is None:
                terminal_failure = self._run_committed_reviewers(result_text)

            # The terminal state commit shares the same engine lock as
            # stop(). Exactly one side wins: a stopped owner cannot publish a
            # later COMPLETED/FAILED state.
            with self._lock:
                owner_cancelled = bool(
                    self._cancel_event.is_set()
                    or (
                        start_owner is not None
                        and (self._workflow_start_owner is not start_owner or start_owner.stop_event.is_set())
                    )
                )
                if owner_cancelled:
                    project.status = WorkflowStatus.CANCELLED
                    project.error = "Workflow cancelled"
                    project.finished_at = time.time()
                    terminal_outcome = "cancelled"
                elif terminal_failure:
                    sanitized_error = _strip_internal_details(terminal_failure)
                    project.result = result_text
                    project.status = WorkflowStatus.FAILED
                    project.error = sanitized_error
                    project.finished_at = time.time()
                    terminal_outcome = "failed"
                else:
                    project.result = result_text
                    project.status = WorkflowStatus.COMPLETED
                    project.finished_at = time.time()
                    terminal_outcome = "completed"

            if terminal_outcome == "cancelled":
                self._state_manager.on_workflow_cancelled("Workflow cancelled")
                return project
            if terminal_outcome == "failed":
                sanitized_error = _strip_internal_details(terminal_failure)
                self._state_manager.on_workflow_failed(terminal_failure)

                logger.error("[WorkflowEngine:%s] Failed: %s", workflow_id, terminal_failure)

                self._fire_progress()
                if self._callbacks.on_error:
                    self._callbacks.on_error(sanitized_error)
                return project

            # Success path
            # AC4: 仅最终汇总结果计入主 context 增量（字符估算（以字符数作为 token 的近似）。
            # 中间 agent 输出不得通过其他路径进入主 context。
            if self._state_manager:
                self._state_manager.add_context_tokens(len(result_text or ""))
            self._state_manager.on_workflow_done(result_text)

            logger.info(
                "[WorkflowEngine:%s] Completed — agents=%d, duration=%.1fs",
                workflow_id,
                project.metrics.completed_agents,
                time.time() - (project.started_at or 0),
            )

            self._fire_progress()
            if self._callbacks.on_done:
                self._callbacks.on_done(project)

        except RuntimeError as e:
            error_msg = str(e)
            sanitized_error = _strip_internal_details(error_msg)
            with self._lock:
                runtime_cancelled = bool(
                    self._cancel_event.is_set()
                    or (
                        start_owner is not None
                        and (self._workflow_start_owner is not start_owner or start_owner.stop_event.is_set())
                    )
                )
                if runtime_cancelled:
                    project.status = WorkflowStatus.CANCELLED
                    project.error = "Workflow cancelled"
                    project.finished_at = time.time()
                else:
                    project.status = WorkflowStatus.FAILED
                    project.error = sanitized_error
                    project.finished_at = time.time()
            if runtime_cancelled:
                if self._state_manager:
                    self._state_manager.on_workflow_cancelled("Workflow cancelled")
            else:
                if self._state_manager:
                    self._state_manager.on_workflow_failed(error_msg)

            logger.error("[WorkflowEngine:%s] Failed: %s", workflow_id, error_msg)

            self._fire_progress()
            if self._callbacks.on_error:
                self._callbacks.on_error(sanitized_error)

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            sanitized_error = _strip_internal_details(error_msg)
            with self._lock:
                runtime_cancelled = bool(
                    self._cancel_event.is_set()
                    or (
                        start_owner is not None
                        and (self._workflow_start_owner is not start_owner or start_owner.stop_event.is_set())
                    )
                )
                if runtime_cancelled:
                    project.status = WorkflowStatus.CANCELLED
                    project.error = "Workflow cancelled"
                    project.finished_at = time.time()
                else:
                    project.status = WorkflowStatus.FAILED
                    project.error = sanitized_error
                    project.finished_at = time.time()
            if self._state_manager:
                if runtime_cancelled:
                    self._state_manager.on_workflow_cancelled("Workflow cancelled")
                else:
                    self._state_manager.on_workflow_failed(error_msg)

            logger.exception("[WorkflowEngine:%s] Unexpected error", workflow_id)

            self._fire_progress()
            if self._callbacks and self._callbacks.on_error:
                self._callbacks.on_error(sanitized_error)

        finally:
            bridge_quiesced = True
            # Stop the progress heartbeat before flushing the final card so no
            # stray re-render races the terminal render below.
            self._stop_heartbeat()

            # Flush any pending progress update (stop() forces final flush
            if self._progress_coalescer:
                self._progress_coalescer.stop()

            # Cleanup bridge
            bridge = self._bridge
            if bridge:
                try:
                    bridge.stop()
                    wait_for_workers = getattr(
                        bridge,
                        "wait_for_workers",
                        None,
                    )
                    if callable(wait_for_workers):
                        if wait_for_workers() is False:
                            bridge_quiesced = False
                except Exception:
                    bridge_quiesced = False
                    logger.debug(
                        "WorkflowEngine bridge quiescence failed",
                        exc_info=True,
                    )
                if bridge_quiesced and self._bridge is bridge:
                    self._bridge = None

            # Release AgentExecutor thread pool (prevents thread leak across runs)
            if bridge_quiesced and self._executor:
                try:
                    self._executor.shutdown(wait=True)
                except Exception:
                    bridge_quiesced = False
                    logger.debug("WorkflowEngine executor shutdown (finally) failed")
                if bridge_quiesced:
                    self._executor = None

            source_path = getattr(start_owner, "source_script_path", None)
            if source_path and project.status in {
                WorkflowStatus.CANCELLED,
                WorkflowStatus.FAILED,
            }:
                _remove_owned_script_artifact(
                    source_path,
                    self.root_path,
                )
                if project.script_path == source_path:
                    project.script_path = None

            # Persist state
            try:
                self.save_state()
            except Exception as save_err:
                logger.debug("Failed to save workflow state: %s", save_err)

            # Record in execution history
            try:
                history = WorkflowHistory(self.root_path)
                history.record(project)
            except Exception as hist_err:
                logger.debug("Failed to record workflow history: %s", hist_err)

            execution_path = getattr(
                start_owner,
                "execution_script_path",
                None,
            )
            if execution_path and execution_path == script_path:
                _remove_owned_script_artifact(
                    execution_path,
                    self.root_path,
                )
            done_event = getattr(start_owner, "done_event", None)
            if done_event is not None:
                done_event.set()
            if not bridge_quiesced and start_owner is not None:
                stop_event = getattr(start_owner, "stop_event", None)
                if stop_event is not None:
                    stop_event.set()
                heartbeat_stop_event = getattr(
                    start_owner,
                    "heartbeat_stop_event",
                    None,
                )
                if heartbeat_stop_event is not None:
                    heartbeat_stop_event.set()
            with self._lock:
                if bridge_quiesced and (start_owner is not None and self._workflow_start_owner is start_owner):
                    self._workflow_start_owner = None
                self._run_thread_id = None
                self._run_done_event.set()
                if bridge_quiesced:
                    # Publish IDLE last: a new run cannot claim shared
                    # component fields until prior cleanup is complete.
                    self._run_state = EngineRunState.IDLE
                else:
                    # Fail closed. cleanup() can retry the bounded bridge
                    # drain later, but this instance must never be reused.
                    self._closing = True
                    self._run_state = EngineRunState.STOPPING
            if not bridge_quiesced:
                logger.error(
                    "[WorkflowEngine:%s] Worker quiescence failed; engine retained as a closing tombstone",
                    workflow_id,
                )

        return project

    # ------------------------------------------------------------------
    # BaseEngine hooks
    # ------------------------------------------------------------------

    def _on_stop(self) -> None:
        """Cancel workflow execution when stop() is called.

        The owning execute thread drains the bridge and AgentExecutor before
        publishing IDLE; this method only signals cancellation so stop()
        remains responsive.
        """
        self._cancel_event.set()
        # Best-effort: ensure the heartbeat cannot outlive a stop() call.
        self._heartbeat_stop.set()
        if self._bridge:
            try:
                self._bridge.stop()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Bridge callbacks
    # ------------------------------------------------------------------

    def _run_committed_reviewers(self, result_text: str) -> str | None:
        """Invoke every explicitly confirmed reviewer and persist evidence.

        Reviewer calls intentionally bypass the workflow journal cache. A
        selected reviewer is a promise of an independent backend invocation
        for this run; cached text is not proof that the promise was kept.
        """
        run_spec = getattr(self, "_run_spec", None)
        if run_spec is None or run_spec.auto_reviewer:
            return None

        failures: list[str] = []
        for index, reviewer in enumerate(run_spec.reviewers, start=1):
            started_at = time.time()
            params = AgentCallParams(
                prompt=(
                    "Independently review the completed Workflow deliverable below. "
                    "Identify correctness gaps, unmet requirements, security or reliability "
                    "risks, and give a clear verdict supported by evidence.\n\n"
                    f"Original task:\n{run_spec.task}\n\n"
                    f"Deliverable:\n{result_text}"
                ),
                tool=reviewer.tool_name,
                model=None if reviewer.use_default_model else reviewer.model_name,
                role="workflow_reviewer",
                label=f"reviewer-{index}-{reviewer.tool_name}",
                phase="Independent Review",
            )
            result = self._handle_agent_call(
                params,
                deadline_monotonic=run_spec.deadline,
                forced_binding=reviewer,
                allow_cache=False,
            )
            evidence = ReviewerEvidence(
                reviewer_index=index,
                selection_key=reviewer.selection_key,
                display_name=reviewer.display_name,
                tool=reviewer.tool_name,
                model=None if reviewer.use_default_model else reviewer.model_name,
                status="failed" if result.error else "completed",
                output=result.output,
                stop_reason=result.stop_reason,
                error=result.error,
                cached=bool(result.cached),
                token_usage=result.token_usage,
                duration_s=result.duration_s,
                started_at=started_at,
                finished_at=time.time(),
            )
            if self._state_manager is not None:
                self._state_manager.record_reviewer_evidence(evidence)
            elif self._project is not None:
                self._project.reviewer_evidence.append(evidence)
            if result.error:
                failures.append(
                    f"{reviewer.tool_name}/{reviewer.model_name or 'default'}: {result.error}"
                )

        if failures:
            return "Independent review failed: " + "; ".join(failures)
        return None

    def _handle_agent_call(
        self,
        params: AgentCallParams,
        *,
        cancel_event=None,
        request_id=None,
        deadline_monotonic: float | None = None,
        forced_binding: ReviewAgentBinding | None = None,
        allow_cache: bool = True,
    ) -> AgentCallResult:
        """Handle an agent() call from the JS runtime.

        Flow:
            1. Safety fuse check (MAX_TOTAL_AGENTS)
            2. Resolve missing model from user's tool-model bindings
            3. Journal cache lookup
            4. Execute via AgentExecutor (creates one-shot session)
            5. Store result in journal
            6. Fire progress callbacks
        """
        # Work on a private copy: the bridge/test caller may retain its params,
        # but the confirmed run binding is authoritative for execution.
        params = params.model_copy(deep=True)
        run_spec = getattr(self, "_run_spec", None)
        if forced_binding is not None:
            params.tool = forced_binding.tool_name
            params.model = None if forced_binding.use_default_model else forced_binding.model_name
        elif run_spec is not None:
            params.tool = params.tool or run_spec.orchestrator.tool_name
            if params.tool in run_spec.tool_model_map:
                # Assign even when the confirmed value is None. None means the
                # user explicitly chose that tool's default model and an
                # invented model in generated JS must not override it.
                params.model = run_spec.tool_model_map[params.tool]

        effective_deadline = deadline_monotonic
        if run_spec is not None and run_spec.deadline is not None:
            effective_deadline = (
                min(effective_deadline, run_spec.deadline)
                if effective_deadline is not None
                else run_spec.deadline
            )

        with self._lock:
            self._agent_call_count += 1
            count = self._agent_call_count
        label = params.label or f"agent-{count}"

        # Legacy direct callers still resolve from the complete adapted project.
        if run_spec is None and not params.model and params.tool and self._project:
            params.model = self._resolve_model_for_tool(params.tool)

        # Safety fuse
        call_budget = run_spec.budget if run_spec is not None else MAX_TOTAL_AGENTS
        if count > call_budget:
            error_msg = f"Agent call limit exceeded ({call_budget})"
            logger.warning("[WorkflowEngine] %s", error_msg)
            return AgentCallResult(error=error_msg, tool=params.tool, model=params.model)
        if effective_deadline is not None and time.monotonic() >= effective_deadline:
            error_msg = "Workflow deadline exhausted before agent execution"
            logger.warning("[WorkflowEngine] %s", error_msg)
            return AgentCallResult(error=error_msg, tool=params.tool, model=params.model)

        # Extract a short task summary from the prompt (first meaningful line, max 60 chars)
        task_summary = ""
        if params.prompt:
            for line in params.prompt.strip().splitlines():
                line = line.strip()
                if line and not line.startswith(("Role:", "#", "---", "**Subagent")):
                    task_summary = line[:80]
                    if len(line) > 80:
                        task_summary += "..."
                    break

        # Register agent in state manager
        if self._state_manager:
            label = self._state_manager.on_agent_started(
                label,
                tool=params.tool,
                phase=params.phase or "default",
                task_summary=task_summary,
                model=params.model,
                role=params.role,
            )

        # Track request_id → effective label for abort-by-request-id lookup
        if request_id is not None:
            with self._request_to_label_lock:
                self._request_to_label[request_id] = label

        cache_key = WorkflowJournal.compute_key(
            params.prompt,
            params.tool,
            params.model,
            role=params.role,
            output_schema=params.output_schema,
        )

        # Tool whitelist enforcement
        project = self._project
        if project and project.selected_tools and params.tool:
            if params.tool not in project.selected_tools:
                error_msg = f"Tool '{params.tool}' not in allowed list: {project.selected_tools}"
                logger.warning("[WorkflowEngine] %s", error_msg)
                if self._state_manager:
                    self._state_manager.on_agent_failed(label, error_msg)
                return AgentCallResult(error=error_msg, tool=params.tool, model=params.model)

        # Fire agent start callbacks
        if self._callbacks and self._callbacks.on_agent_start:
            self._callbacks.on_agent_start(label, params.tool)
        self._fire_progress()

        # Journal cache lookup
        if allow_cache and self._journal:
            cached = self._journal.get_cached(cache_key)
            if cached is not None:
                logger.debug("[WorkflowEngine] Cache hit for %s", label)
                cached_result = AgentCallResult(
                    output=cached.output,
                    parsed=cached.parsed,
                    stop_reason=cached.stop_reason,
                    token_usage=0,  # No tokens consumed on cache hit
                    duration_s=0.0,
                    cached=True,
                    tool=params.tool,
                    model=params.model,
                )
                if self._state_manager:
                    self._state_manager.on_agent_done(
                        label,
                        {
                            "token_usage": 0,
                            "duration_s": 0.0,
                            "cached": True,
                        },
                    )
                self._fire_progress()
                return cached_result

        # Execute via AgentExecutor (pass per-call cancel event for race/tournament abort)
        result = self._executor.execute(
            params,
            cancel_event=cancel_event,
            deadline_monotonic=effective_deadline,
        )

        # A selected Reviewer is a completion promise, not merely a backend
        # round-trip. Only an explicit end_turn with non-empty output proves
        # that the independent review finished. Unknown/new stop reasons fail
        # closed and are retained in durable Reviewer evidence.
        if forced_binding is not None and result.error is None:
            stop_reason = (result.stop_reason or "").strip().casefold()
            if stop_reason != "end_turn":
                reason = stop_reason or "missing_stop_reason"
                result = result.model_copy(
                    update={
                        "error": (
                            "Reviewer did not complete normally "
                            f"(stop_reason={reason})"
                        )
                    }
                )
            elif not str(result.output or "").strip():
                result = result.model_copy(update={"error": "Reviewer returned empty output"})

        # Store in journal (only on success)
        if allow_cache and result.error is None and self._journal:
            self._journal.store(cache_key, result)

        # Update state
        if self._state_manager:
            if result.error:
                self._state_manager.on_agent_failed(label, result.error)
            else:
                self._state_manager.on_agent_done(
                    label,
                    {
                        "token_usage": result.token_usage,
                        "duration_s": result.duration_s,
                        "cached": False,
                    },
                )

        # Fire agent done callback — AC4: only meta info, no output body.
        if self._callbacks and self._callbacks.on_agent_done:
            # Hand-rolled payload: deliberately excludes output/parsed
            # so callers cannot leak intermediate results into the main
            # agent context.
            payload = {
                "label": label,
                "tool": params.tool,
                "model": result.model if result else None,
                "token_usage": result.token_usage if result else 0,
                "duration_s": result.duration_s if result else 0.0,
                "cached": bool(result.cached) if result else False,
                "error": result.error if result else None,
            }
            self._callbacks.on_agent_done(label, payload)

        self._fire_progress()
        return result

    def _handle_agent_activity(self, label: str, activity: str) -> None:
        """Update live activity hint for a running agent from ACP events.

        Fires a debounced progress update so the card reflects the new activity
        without waiting for the next heartbeat cycle.
        """
        if self._state_manager:
            self._state_manager.update_agent_activity(label, activity)
            self._fire_progress()

    def _handle_agent_aborted(self, label: str, reason: str, *, request_id=None) -> None:
        """Handle an agent_aborted notification from the JS runtime.

        Called when a race() loser (or tournament elimination) agent is
        aborted. Updates the progress card so the agent no longer shows as
        '执行中'. The ACP session is interrupted via the per-call cancel_event
        set by the bridge's _handle_abort_request.

        Uses request_id for authoritative lookup (avoids label mismatch when
        state_manager disambiguates duplicate labels), falling back to raw
        label for backward compatibility.
        """
        effective_label = label
        if request_id is not None:
            with self._request_to_label_lock:
                mapped = self._request_to_label.get(request_id)
            if mapped:
                effective_label = mapped
        logger.info(
            "[WorkflowEngine] Agent aborted: %s (reason=%s, request_id=%s)",
            effective_label,
            reason,
            request_id,
        )
        if self._state_manager:
            self._state_manager.on_agent_aborted(effective_label, reason)
        self._fire_progress(immediate=True)

    def _handle_phase(self, title: str) -> None:
        """Handle a phase() notification from the JS runtime."""
        logger.info("[WorkflowEngine] Phase: %s", title)

        if self._state_manager:
            self._state_manager.on_phase_changed(title)

        if self._callbacks and self._callbacks.on_phase:
            self._callbacks.on_phase(title)

        self._fire_progress()

    def _handle_log(self, message: str) -> None:
        """Handle a log() notification from the JS runtime."""
        logger.debug("[WorkflowEngine] Log: %s", message)

        if self._callbacks and self._callbacks.on_log:
            self._callbacks.on_log(message)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fire_progress(self, immediate: bool = False) -> None:
        """Fire progress callback via coalescer (non-blocking, debounced).

        Args:
            immediate: If True, bypass the coalescer debounce and flush
                immediately. Used for abort events where the user needs to
                see cancelled agents disappear from '执行中' quickly.
        """
        if not self._renderer_wf:
            return
        if not self._callbacks or not self._callbacks.on_progress:
            return
        try:
            if self._state_manager:
                snapshot = self._state_manager.snapshot()
                card_data = self._renderer_wf.render_progress_card(snapshot)
            else:
                card_data = self._renderer_wf.render_progress_card()
            if self._progress_coalescer:
                if immediate:
                    self._progress_coalescer.flush_immediate(card_data)
                else:
                    self._progress_coalescer.enqueue(card_data)
            else:
                self._callbacks.on_progress(card_data)
        except Exception:
            logger.debug("on_progress callback failed", exc_info=True)

    # ------------------------------------------------------------------
    # Progress heartbeat
    # ------------------------------------------------------------------

    def _start_heartbeat(self) -> None:
        """Start the daemon heartbeat thread that re-renders the progress card.

        Idempotent: only one heartbeat thread runs per workflow. The thread
        exits when ``_heartbeat_stop`` is set (see :meth:`_stop_heartbeat`).
        """
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return
        # Fresh cancellation boundary for this run (the event may be set from a
        # previous run when the engine instance is reused by the manager).
        self._heartbeat_stop.clear()
        thread = threading.Thread(
            target=self._heartbeat_loop,
            name="WorkflowHeartbeat",
            daemon=True,
        )
        self._heartbeat_thread = thread
        thread.start()

    def _stop_heartbeat(self) -> None:
        """Signal the heartbeat thread to stop and join it (best-effort)."""
        self._heartbeat_stop.set()
        thread = self._heartbeat_thread
        if thread is not None:
            try:
                thread.join(timeout=2.0)
            except Exception as e:
                logger.debug("Heartbeat join failed: %s", str(e))
            self._heartbeat_thread = None

    def _heartbeat_loop(self) -> None:
        """Re-render the progress card every ``PROGRESS_HEARTBEAT_S`` seconds.

        Goes through the coalescer (debounced) so it never spams Feishu. Any
        error is swallowed at debug level — a failed heartbeat must never
        terminate the run.
        """
        while not self._heartbeat_stop.wait(PROGRESS_HEARTBEAT_S):
            try:
                self._fire_progress()
            except Exception as e:
                logger.debug("Heartbeat progress fire failed: %s", str(e))

    def _resolve_model_for_tool(self, tool: str) -> str | None:
        """Resolve the model for a tool from user's selection bindings.

        Uses the tool_model_map populated by start_execution() from the
        user's orchestrator/review agent selections.
        """
        project = self._project
        if not project:
            return None
        return project.tool_model_map.get(tool) or None

    # ------------------------------------------------------------------
    # Status / snapshot
    # ------------------------------------------------------------------

    def get_status_text(self) -> str:
        """Return a compact one-line status string."""
        if self._renderer_wf:
            return self._renderer_wf.render_compact_status()
        return "Workflow: idle"

    def get_progress_card(self) -> Optional[dict[str, Any]]:
        """Return current Feishu card JSON for the workflow progress."""
        if self._renderer_wf:
            if self._state_manager:
                return self._renderer_wf.render_progress_card(self._state_manager.snapshot())
            return self._renderer_wf.render_progress_card()
        return None

    def get_journal_stats(self) -> dict:
        """Return journal cache statistics."""
        if self._journal:
            return self._journal.stats()
        return {"total": 0, "hits": 0, "misses": 0}
