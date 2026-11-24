"""WorkflowEngine — orchestrates multi-step AI workflows via Node.js runtime."""

from __future__ import annotations

import inspect
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional

from src.card.events import CardEvent, CardEventType

from ..engine_base import BaseEngine, EngineRunState
from .agent_pool import resolve_effective_model_name
from .bridge import RuntimeBridge
from .constants import (
    DEFAULT_MAX_CONCURRENT,
    MAX_TOTAL_AGENTS,
    NODE_MIN_VERSION,
    PROGRESS_DEBOUNCE_S,
    PROGRESS_HEARTBEAT_S,
    STATE_FILENAME,
)
from .errors import _strip_internal_details
from .executor import AgentExecutor
from .journal import WorkflowJournal
from .models import (
    AgentCallParams,
    AgentCallResult,
    AgentStatus,
    WorkflowMetrics,
    WorkflowProject,
    WorkflowStatus,
)
from .progress_coalescer import ProgressCoalescer
from .renderer import WorkflowProgressRenderer
from .run_spec import WorkflowRunSpec
from .state_manager import WorkflowStateManager
from .storage import (
    remove_owned_workflow_artifact,
    workflow_project_storage_root,
    workflow_scripts_dir,
)

logger = logging.getLogger(__name__)


def _remove_owned_script_artifact(path: str | None, root_path: str) -> None:
    """Remove only Workflow's generated source or immutable temp copy."""
    remove_owned_workflow_artifact(
        path,
        root_path=root_path,
        scripts_dir=workflow_scripts_dir(root_path),
    )


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
        except (TypeError, ValueError, RecursionError):
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


def _agent_card_result_text(result: AgentCallResult) -> str:
    """Return the sanitized, complete terminal text kept for card rendering."""
    raw: Any = result.output
    if (raw is None or raw == "") and result.parsed is not None:
        raw = json.dumps(result.parsed, ensure_ascii=False, default=str)
    text = _strip_internal_details(str(raw or ""))
    return text.replace("\x00", "").strip()


_REVIEWER_ROLES = frozenset(
    {"reviewer", "verifier", "adversarial_verifier", "auditor", "judge"}
)


def _is_reviewer_role(role: str | None) -> bool:
    normalized = str(role or "").strip().lower()
    return normalized in _REVIEWER_ROLES or normalized.startswith("verifier-")


def _binding_roles(binding: Any) -> set[str]:
    roles: set[str] = set()
    primary = str(getattr(binding, "role", None) or "").strip().lower()
    if primary:
        roles.add(primary)
    for item in getattr(binding, "roles", ()) or ():
        text = str(item or "").strip().lower()
        if text:
            roles.add(text)
    return roles


def _is_startup_or_breaker_error(error: str | None) -> bool:
    lowered = str(error or "").strip().lower()
    if not lowered:
        return False
    return any(
        marker in lowered
        for marker in (
            "circuit breaker:",
            "acpstartuperror",
            "session creation",
            "binary missing",
            "failed to start",
        )
    )


def _accepts_kwarg(fn: Callable[..., Any], name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return True
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == name and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@dataclass
class WorkflowEngineCallbacks:
    """Event callbacks for the Workflow Engine handler layer."""

    on_progress: Optional[Callable[[list[dict[str, Any]]], None]] = None
    on_done: Optional[Callable[[WorkflowProject], None]] = None
    on_cancelled: Optional[Callable[[WorkflowProject], None]] = None
    # Runtime failures include the authoritative terminal project so the
    # delivery layer can flush every coalesced execution page. ``None`` is
    # reserved for pre-project/setup errors raised by non-engine callers.
    on_error: Optional[
        Callable[[str, Optional[WorkflowProject]], None]
    ] = None
    on_log: Optional[Callable[[str], None]] = None
    on_phase: Optional[Callable[[str], None]] = None
    on_agent_start: Optional[Callable[[str, str], None]] = None  # (label, tool)
    on_agent_event: Optional[Callable[[str, CardEvent], None]] = None
    # AC4: on_agent_done is a lightweight meta-info callback; the payload
    # deliberately excludes the agent output/parsed body so that handler-layer
    # subscribers cannot accidentally leak intermediate results into the main
    # agent chat context. Only final results are delivered via on_done.
    on_agent_done: Optional[Callable[[str, dict], None]] = None  # (label, meta_dict)
    image_uploader: Optional[Callable[[Any], Optional[str]]] = None


# ---------------------------------------------------------------------------
# WorkflowEngine
# ---------------------------------------------------------------------------


class WorkflowEngine(BaseEngine):
    """Orchestrates multi-step AI workflows using a Node.js runtime bridge.

    Lifecycle:
        1. Handler calls execute_workflow(script_path=..., run_spec=..., callbacks=...)
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

    # Optional test/config override for the centralized Workflow storage root.
    _CACHE_ROOT: str | None = None

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
        self._stream_progress_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._last_stream_progress_at = 0.0

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
        """Return this project's state directory outside the project tree."""
        return workflow_project_storage_root(self.root_path, self._CACHE_ROOT)

    def save_state(self, filepath: Optional[str] = None) -> str:
        """Persist state below ``~/.ghostap/workflow/projects``."""
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

    def release_lifecycle_owner(self, owner: Any) -> bool:
        """Release a quiesced owner; uncertain sessions remain cleanup roots."""
        if owner is None:
            return False
        with self._lock:
            if getattr(owner, "active_generation_session", None) is not None:
                return False
            done_event = getattr(owner, "done_event", None)
            if done_event is not None and not done_event.is_set():
                return False
            tracked = any(
                candidate is owner
                for candidate in (
                    self._script_generation_owner,
                    self._workflow_start_owner,
                    *self._retired_lifecycle_owners,
                )
            )
            if self._script_generation_owner is owner:
                self._script_generation_owner = None
            if self._workflow_start_owner is owner:
                self._workflow_start_owner = None
            self._retired_lifecycle_owners = [
                candidate
                for candidate in self._retired_lifecycle_owners
                if candidate is not owner
            ]
            return tracked

    def has_uncertain_lifecycle_session(self) -> bool:
        """Return whether reuse could overlap an unconfirmed ACP session."""
        with self._lock:
            owners = (
                self._script_generation_owner,
                self._workflow_start_owner,
                *self._retired_lifecycle_owners,
            )
            return any(
                owner is not None
                and getattr(owner, "active_generation_session", None) is not None
                for owner in owners
            )

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
            active_session = getattr(owner, "active_generation_session", None)
            session_quiesced = active_session is None
            if active_session is not None:
                session_quiesced = True
                try:
                    if active_session.cancel(wait=True, timeout=2.0) is False:
                        session_quiesced = False
                except Exception:
                    session_quiesced = False
                    logger.debug("Workflow generation cancel retry failed", exc_info=True)
                try:
                    active_session.close()
                except Exception:
                    session_quiesced = False
                    logger.debug("Workflow generation close retry failed", exc_info=True)
                if session_quiesced:
                    try:
                        object.__setattr__(owner, "active_generation_session", None)
                    except Exception:
                        session_quiesced = False
                if not session_quiesced:
                    quiesced = False
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
            worker_exited_event = getattr(
                owner,
                "worker_exited_event",
                None,
            )
            if (
                done_event is not None
                and session_quiesced
                and (
                    (
                        (claimed_event is None or not claimed_event.is_set())
                        and (
                            worker_started_event is None
                            or not worker_started_event.is_set()
                        )
                    )
                    or (
                        worker_exited_event is not None
                        and worker_exited_event.is_set()
                    )
                )
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

    def execute_workflow(
        self,
        script_path: str,
        callbacks: Optional[WorkflowEngineCallbacks] = None,
        *,
        run_spec: WorkflowRunSpec,
        start_owner: Any = None,
        source_script_path: Optional[str] = None,
    ) -> WorkflowProject:
        """Execute a workflow script end-to-end.

        Args:
            script_path: Absolute path to the .js workflow script.
            callbacks: Optional event callbacks for progress/completion.
            run_spec: Frozen automatic-admission execution contract.
            start_owner: Optional handler lifecycle token for a queued start.
            source_script_path: Stable generated source retained for save/reuse.

        Returns:
            The final WorkflowProject with status, metrics, and result.

        Raises:
            RuntimeError: If Node.js is unavailable or the bridge fails fatally.
        """
        run_callbacks = callbacks or WorkflowEngineCallbacks()
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
                from .script_gen import extract_meta_from_script

                script_content = None
                try:
                    with open(script_path, "r", encoding="utf-8") as f:
                        script_content = f.read()
                except OSError:
                    script_content = None
                if script_content:
                    script_meta = extract_meta_from_script(script_content)
            if script_meta is not None:
                configured_concurrency = script_meta.get(
                    "maxConcurrent",
                    script_meta.get("max_concurrent"),
                )
                if configured_concurrency:
                    max_concurrent = int(configured_concurrency)
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
                self._last_stream_progress_at = 0.0
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
                on_attempt=self._handle_agent_attempt,
                on_subagent_update=self._handle_agent_subagent_update,
                on_card_event=self._handle_agent_card_event,
                image_uploader=run_callbacks.image_uploader,
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
                artifact_dir = os.path.join(self._state_dir(), "artifacts", workflow_id)
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
                    workflow_deadline_monotonic=run_spec.deadline,
                    artifact_dir=artifact_dir,
                    agent_pool=[b.to_dict() for b in run_spec.agent_pool],
                    orchestrator_agent_id=run_spec.orchestrator_agent_id,
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
                self._fire_progress()
                if self._callbacks.on_cancelled:
                    self._callbacks.on_cancelled(self._state_manager.snapshot())
                return project
            if terminal_outcome == "failed":
                sanitized_error = _strip_internal_details(terminal_failure)
                self._state_manager.on_workflow_failed(terminal_failure)

                logger.error("[WorkflowEngine:%s] Failed: %s", workflow_id, terminal_failure)

                self._fire_progress()
                if self._callbacks.on_error:
                    self._callbacks.on_error(
                        sanitized_error,
                        self._state_manager.snapshot(),
                    )
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
            if runtime_cancelled and self._callbacks.on_cancelled:
                terminal_project = (
                    self._state_manager.snapshot()
                    if self._state_manager is not None
                    else project.model_copy(deep=True)
                )
                self._callbacks.on_cancelled(terminal_project)
            elif not runtime_cancelled and self._callbacks.on_error:
                terminal_project = (
                    self._state_manager.snapshot()
                    if self._state_manager is not None
                    else project.model_copy(deep=True)
                )
                self._callbacks.on_error(sanitized_error, terminal_project)

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
            if runtime_cancelled and self._callbacks.on_cancelled:
                terminal_project = (
                    self._state_manager.snapshot()
                    if self._state_manager is not None
                    else project.model_copy(deep=True)
                )
                self._callbacks.on_cancelled(terminal_project)
            elif not runtime_cancelled and self._callbacks.on_error:
                terminal_project = (
                    self._state_manager.snapshot()
                    if self._state_manager is not None
                    else project.model_copy(deep=True)
                )
                self._callbacks.on_error(sanitized_error, terminal_project)

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
            if bridge_quiesced and start_owner is not None:
                self.release_lifecycle_owner(start_owner)
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

    def _build_auto_fallback_bindings(
        self,
        run_spec: WorkflowRunSpec,
        primary_binding: Any,
        *,
        requested_role: str | None,
        attempted_ids: set[str],
        explicit_timeout: bool,
    ) -> list[Any]:
        """Select safe pool-only fallback candidates on startup/circuit-breaker failures."""
        primary_roles = _binding_roles(primary_binding)
        normalized_requested = str(requested_role or "").strip().lower()
        is_reviewer = _is_reviewer_role(normalized_requested) or (
            not normalized_requested and any(_is_reviewer_role(r) for r in primary_roles)
        )
        target_roles = {normalized_requested} if normalized_requested else primary_roles
        candidates: list[tuple[int, int, Any]] = []
        for idx, binding in enumerate(run_spec.agent_pool):
            if binding.agent_id in attempted_ids:
                continue
            roles = _binding_roles(binding)
            if is_reviewer:
                # Never auto-fallback a reviewer/verifier onto a non-reviewer worker.
                if not any(_is_reviewer_role(r) for r in roles):
                    continue
            probe_params = self._params_for_binding(
                AgentCallParams(prompt="", agent_id=binding.agent_id),
                binding,
                run_spec,
                explicit_timeout=explicit_timeout,
            )
            if callable(getattr(self._executor, "is_blacklisted", None)) and self._executor.is_blacklisted(probe_params):
                continue
            role_match = 0 if (target_roles and bool(target_roles & roles)) else 1
            non_reviewer_bonus = 0 if (not is_reviewer and not any(_is_reviewer_role(r) for r in roles)) else 1
            candidates.append((role_match * 10 + non_reviewer_bonus, idx, binding))
        candidates.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in candidates]

    @staticmethod
    def _params_for_binding(
        base_params: AgentCallParams,
        binding: Any,
        run_spec: WorkflowRunSpec,
        *,
        explicit_timeout: bool = False,
    ) -> AgentCallParams:
        updated = base_params.model_copy(deep=True)
        updated.agent_id = binding.agent_id
        updated.tool = binding.tool_name
        updated.model = binding.model_name
        updated.profile = binding.profile
        updated.effort = binding.effort
        spec_effective = (
            run_spec.binding_specs.get(binding.agent_id) or {}
        ).get("effective_model")
        updated.effective_model = spec_effective or binding.effective_model_name()
        if not updated.role:
            binding_role = getattr(binding, "role", None)
            if binding_role:
                updated.role = binding_role
            else:
                binding_roles = getattr(binding, "roles", ()) or ()
                if len(binding_roles) == 1:
                    updated.role = binding_roles[0]
        del explicit_timeout
        return updated

    def _is_call_cancelled(
        self,
        result: AgentCallResult | None,
        per_call_cancel: threading.Event | None,
    ) -> bool:
        if self._cancel_event.is_set():
            return True
        if per_call_cancel is not None and per_call_cancel.is_set():
            return True
        if result is not None and str(getattr(result, "stop_reason", "") or "").lower() == "cancelled":
            return True
        return False

    def _execute_single_candidate(
        self,
        params: AgentCallParams,
        *,
        count: int,
        cancel_event: Any,
        request_id: Any,
        effective_deadline: float | None,
        allow_cache: bool,
        explicit_timeout: bool,
    ) -> AgentCallResult:
        label = params.label or f"agent-{count}"
        task_summary = ""
        if params.prompt:
            for line in params.prompt.strip().splitlines():
                line = line.strip()
                if line and not line.startswith(("Role:", "#", "---", "**Subagent")):
                    task_summary = line[:80]
                    if len(line) > 80:
                        task_summary += "..."
                    break

        if self._state_manager:
            label = self._state_manager.on_agent_started(
                label,
                tool=params.tool,
                phase=params.phase or "default",
                task_summary=task_summary,
                model=params.model,
                role=params.role,
                agent_id=params.agent_id,
                effort=params.effort,
                profile=params.profile,
                effective_model=params.effective_model,
            )
            params.label = label

        if request_id is not None:
            with self._request_to_label_lock:
                self._request_to_label[request_id] = label

        cache_key = WorkflowJournal.compute_key(
            params.prompt,
            params.tool,
            params.effective_model or params.model,
            role=params.role,
            output_schema=params.output_schema,
            effort=params.effort,
            profile=params.profile,
        )

        project = self._project
        if project and project.selected_tools and params.tool:
            if params.tool not in project.selected_tools:
                error_msg = f"Tool '{params.tool}' not in allowed list: {project.selected_tools}"
                logger.warning("[WorkflowEngine] %s", error_msg)
                if self._state_manager:
                    self._state_manager.on_agent_failed(label, error_msg)
                return AgentCallResult(
                    error=error_msg,
                    agent_id=params.agent_id,
                    tool=params.tool,
                    model=params.model,
                    role=params.role,
                    effort=params.effort,
                    profile=params.profile,
                    effective_model=params.effective_model,
                )

        if self._callbacks and self._callbacks.on_agent_start:
            if _accepts_kwarg(self._callbacks.on_agent_start, "meta"):
                self._callbacks.on_agent_start(
                    label,
                    params.tool,
                    meta={
                        "agent_id": params.agent_id,
                        "model": params.model,
                        "role": params.role,
                        "effort": params.effort,
                        "profile": params.profile,
                        "effective_model": params.effective_model,
                    },
                )
            else:
                self._callbacks.on_agent_start(label, params.tool)
        self._fire_progress()

        if allow_cache and self._journal:
            cached = self._journal.get_cached(cache_key)
            if cached is not None:
                logger.debug("[WorkflowEngine] Cache hit for %s", label)
                cached_result = AgentCallResult(
                    output=cached.output,
                    parsed=cached.parsed,
                    stop_reason=cached.stop_reason,
                    token_usage=0,
                    duration_s=0.0,
                    cached=True,
                    agent_id=params.agent_id,
                    tool=params.tool,
                    model=params.model,
                    role=params.role,
                    effort=params.effort,
                    profile=params.profile,
                    effective_model=params.effective_model,
                )
                if self._state_manager:
                    self._state_manager.on_agent_done(
                        label,
                        {
                            "token_usage": 0,
                            "duration_s": 0.0,
                            "cached": True,
                            "result": _agent_card_result_text(cached_result),
                        },
                    )
                if self._callbacks and self._callbacks.on_agent_done:
                    self._callbacks.on_agent_done(
                        label,
                        {
                            "label": label,
                            "agent_id": params.agent_id,
                            "tool": params.tool,
                            "model": cached_result.model,
                            "role": params.role,
                            "effort": params.effort,
                            "profile": params.profile,
                            "effective_model": params.effective_model,
                            "token_usage": 0,
                            "duration_s": 0.0,
                            "cached": True,
                            "error": None,
                        },
                    )
                self._fire_progress()
                return cached_result

        exec_kwargs: dict[str, Any] = {
            "cancel_event": cancel_event,
            "deadline_monotonic": effective_deadline,
        }
        if explicit_timeout and _accepts_kwarg(self._executor.execute, "explicit_timeout"):
            exec_kwargs["explicit_timeout"] = True

        result = self._executor.execute(params, **exec_kwargs)
        result = result.model_copy(
            update={
                "agent_id": params.agent_id,
                "tool": params.tool,
                "model": params.model,
                "role": params.role,
                "effort": params.effort,
                "profile": params.profile,
                "effective_model": params.effective_model,
            }
        )

        if allow_cache and result.error is None and self._journal:
            self._journal.store(cache_key, result)

        if self._state_manager:
            if result.error:
                self._state_manager.on_agent_failed(
                    label,
                    result.error,
                    result=_agent_card_result_text(result),
                    token_usage=result.token_usage,
                    duration_s=result.duration_s,
                )
            else:
                self._state_manager.on_agent_done(
                    label,
                    {
                        "token_usage": result.token_usage,
                        "duration_s": result.duration_s,
                        "cached": False,
                        "result": _agent_card_result_text(result),
                    },
                )

        if self._callbacks and self._callbacks.on_agent_done:
            payload = {
                "label": label,
                "agent_id": params.agent_id,
                "tool": params.tool,
                "model": result.model if result else None,
                "role": params.role,
                "effort": params.effort,
                "profile": params.profile,
                "effective_model": params.effective_model,
                "token_usage": result.token_usage if result else 0,
                "duration_s": result.duration_s if result else 0.0,
                "cached": bool(result.cached) if result else False,
                "error": result.error if result else None,
            }
            self._callbacks.on_agent_done(label, payload)

        self._fire_progress()
        return result

    def _handle_agent_call(
        self,
        params: AgentCallParams,
        *,
        cancel_event=None,
        request_id=None,
        deadline_monotonic: float | None = None,
        allow_cache: bool = True,
        fallback_agent_ids: list[str] | tuple[str, ...] | None = None,
        explicit_timeout: bool = False,
    ) -> AgentCallResult:
        """Handle an agent() call from the JS runtime with pool validation & fallback."""
        params = params.model_copy(deep=True)
        use_explicit_timeout = bool(
            explicit_timeout or getattr(params, "explicit_timeout", False)
        )

        raw_fallbacks: list[str] = []
        for source in (fallback_agent_ids, getattr(params, "fallback_agent_ids", None)):
            if isinstance(source, (list, tuple)):
                for item in source:
                    text = str(item or "").strip()
                    if text and text not in raw_fallbacks:
                        raw_fallbacks.append(text)

        run_spec = getattr(self, "_run_spec", None)
        if run_spec is None and raw_fallbacks:
            return AgentCallResult(
                error="Workflow agent_pool is required to resolve fallbackAgentIds",
                agent_id=params.agent_id,
                tool=params.tool,
                model=params.model,
                role=params.role,
                effort=params.effort,
                profile=params.profile,
                effective_model=params.effective_model,
            )

        requested_role = params.role
        candidate_bindings: list[Any] = []
        if run_spec is not None:
            try:
                primary_binding = run_spec.agent_binding(params.agent_id)
                candidate_bindings.append(primary_binding)
                seen_ids = {primary_binding.agent_id}
                primary_roles = _binding_roles(primary_binding)
                normalized_requested = str(requested_role or "").strip().lower()
                is_reviewer = _is_reviewer_role(normalized_requested) or (
                    not normalized_requested and any(_is_reviewer_role(r) for r in primary_roles)
                )
                for fb_id in raw_fallbacks:
                    fb_binding = run_spec.agent_binding(fb_id)
                    if is_reviewer and not any(
                        _is_reviewer_role(r) for r in _binding_roles(fb_binding)
                    ):
                        logger.warning(
                            "[WorkflowEngine] Ignoring non-reviewer explicit fallback agent_id=%s for reviewer step primary=%s",
                            fb_binding.agent_id,
                            primary_binding.agent_id,
                        )
                        continue
                    if fb_binding.agent_id not in seen_ids:
                        candidate_bindings.append(fb_binding)
                        seen_ids.add(fb_binding.agent_id)
            except ValueError as exc:
                return AgentCallResult(
                    error=str(exc),
                    agent_id=params.agent_id,
                    tool=params.tool,
                    model=params.model,
                    role=params.role,
                    effort=params.effort,
                    profile=params.profile,
                    effective_model=params.effective_model,
                )
            params = self._params_for_binding(
                params,
                candidate_bindings[0],
                run_spec,
                explicit_timeout=use_explicit_timeout,
            )

        effective_deadline = deadline_monotonic
        if run_spec is not None and run_spec.deadline is not None:
            effective_deadline = (
                min(effective_deadline, run_spec.deadline)
                if effective_deadline is not None
                else run_spec.deadline
            )

        # Legacy direct callers without run_spec
        if run_spec is None:
            with self._lock:
                self._agent_call_count += 1
                count = self._agent_call_count
            if not params.model and params.tool and self._project:
                params.model = self._resolve_model_for_tool(params.tool)
            if params.effective_model is None and params.tool and (
                params.effort or params.profile or params.model
            ):
                params.effective_model = resolve_effective_model_name(
                    params.tool,
                    params.model,
                    profile=params.profile,
                    effort=params.effort,
                )
            if count > MAX_TOTAL_AGENTS:
                error_msg = f"Agent call limit exceeded ({MAX_TOTAL_AGENTS})"
                logger.warning("[WorkflowEngine] %s", error_msg)
                return AgentCallResult(
                    error=error_msg,
                    agent_id=params.agent_id,
                    tool=params.tool,
                    model=params.model,
                    role=params.role,
                    effort=params.effort,
                    profile=params.profile,
                    effective_model=params.effective_model,
                )
            if effective_deadline is not None and time.monotonic() >= effective_deadline:
                error_msg = "Workflow deadline exhausted before agent execution"
                logger.warning("[WorkflowEngine] %s", error_msg)
                return AgentCallResult(
                    error=error_msg,
                    agent_id=params.agent_id,
                    tool=params.tool,
                    model=params.model,
                    role=params.role,
                    effort=params.effort,
                    profile=params.profile,
                    effective_model=params.effective_model,
                )
            return self._execute_single_candidate(
                params,
                count=count,
                cancel_event=cancel_event,
                request_id=request_id,
                effective_deadline=effective_deadline,
                allow_cache=allow_cache,
                explicit_timeout=use_explicit_timeout,
            )

        # Strict run_spec path with bounded cross-member fallback
        call_budget = run_spec.budget
        base_label = params.label
        attempted_ids: set[str] = set()
        auto_expanded = False
        idx = 0
        last_result: AgentCallResult | None = None

        while idx < len(candidate_bindings):
            binding = candidate_bindings[idx]
            idx += 1
            cand_params = self._params_for_binding(
                params,
                binding,
                run_spec,
                explicit_timeout=use_explicit_timeout,
            )
            if base_label:
                cand_params.label = base_label

            # Fast-skip already blacklisted binding ONLY if a healthy alternative exists
            is_blacklisted_fn = getattr(self._executor, "is_blacklisted", None)
            if callable(is_blacklisted_fn) and is_blacklisted_fn(cand_params):
                if not raw_fallbacks and not auto_expanded:
                    auto_expanded = True
                    extra = self._build_auto_fallback_bindings(
                        run_spec,
                        binding,
                        requested_role=requested_role,
                        attempted_ids=attempted_ids | {binding.agent_id},
                        explicit_timeout=use_explicit_timeout,
                    )
                    candidate_bindings.extend(extra)
                has_healthy_remaining = any(
                    not is_blacklisted_fn(
                        self._params_for_binding(
                            params,
                            other,
                            run_spec,
                            explicit_timeout=use_explicit_timeout,
                        )
                    )
                    for other in candidate_bindings[idx:]
                )
                if has_healthy_remaining:
                    logger.info(
                        "[WorkflowEngine] Skipping blacklisted agent_id=%s (%s) in favor of fallback",
                        binding.agent_id,
                        binding.tool_name,
                    )
                    attempted_ids.add(binding.agent_id)
                    continue

            if self._is_call_cancelled(last_result, cancel_event):
                return last_result or AgentCallResult(
                    error="Cancelled before execution",
                    stop_reason="cancelled",
                    agent_id=cand_params.agent_id,
                    tool=cand_params.tool,
                    model=cand_params.model,
                    role=cand_params.role,
                    effort=cand_params.effort,
                    profile=cand_params.profile,
                    effective_model=cand_params.effective_model,
                )

            if effective_deadline is not None and time.monotonic() >= effective_deadline:
                if last_result is not None:
                    return last_result
                error_msg = "Workflow deadline exhausted before agent execution"
                logger.warning("[WorkflowEngine] %s", error_msg)
                return AgentCallResult(
                    error=error_msg,
                    agent_id=cand_params.agent_id,
                    tool=cand_params.tool,
                    model=cand_params.model,
                    role=cand_params.role,
                    effort=cand_params.effort,
                    profile=cand_params.profile,
                    effective_model=cand_params.effective_model,
                )

            with self._lock:
                if self._agent_call_count >= call_budget:
                    if last_result is not None:
                        return last_result
                    error_msg = f"Agent call limit exceeded ({call_budget})"
                    logger.warning("[WorkflowEngine] %s", error_msg)
                    return AgentCallResult(
                        error=error_msg,
                        agent_id=cand_params.agent_id,
                        tool=cand_params.tool,
                        model=cand_params.model,
                        role=cand_params.role,
                        effort=cand_params.effort,
                        profile=cand_params.profile,
                        effective_model=cand_params.effective_model,
                    )
                self._agent_call_count += 1
                count = self._agent_call_count

            attempted_ids.add(binding.agent_id)
            result = self._execute_single_candidate(
                cand_params,
                count=count,
                cancel_event=cancel_event,
                request_id=request_id,
                effective_deadline=effective_deadline,
                allow_cache=allow_cache,
                explicit_timeout=use_explicit_timeout,
            )
            last_result = result
            if result.error is None:
                return result

            if self._is_call_cancelled(result, cancel_event):
                return result
            if effective_deadline is not None and time.monotonic() >= effective_deadline:
                return result

            if not raw_fallbacks and not auto_expanded and _is_startup_or_breaker_error(result.error):
                auto_expanded = True
                extra = self._build_auto_fallback_bindings(
                    run_spec,
                    binding,
                    requested_role=requested_role,
                    attempted_ids=attempted_ids,
                    explicit_timeout=use_explicit_timeout,
                )
                candidate_bindings.extend(extra)

        return last_result or AgentCallResult(
            error="No eligible agent bindings remained for fallback",
            agent_id=params.agent_id,
            tool=params.tool,
            model=params.model,
            role=params.role,
            effort=params.effort,
            profile=params.profile,
            effective_model=params.effective_model,
        )

    def _handle_agent_activity(self, label: str, activity: str) -> None:
        """Update live activity hint for a running agent from ACP events.

        Fires a debounced progress update so the card reflects the new activity
        without waiting for the next heartbeat cycle.
        """
        if self._state_manager:
            self._state_manager.update_agent_activity(label, activity)
            self._fire_progress()

    def _handle_agent_card_event(self, label: str, event: CardEvent) -> None:
        """Persist one direct-call execution event and refresh at a safe rate."""
        if not self._state_manager:
            return
        if not self._state_manager.record_agent_card_event(label, event):
            return
        if self._callbacks and self._callbacks.on_agent_event:
            try:
                self._callbacks.on_agent_event(label, event)
            except Exception:
                logger.debug(
                    "Workflow process event callback failed",
                    exc_info=True,
                )

        if event.type in {
            CardEventType.TEXT_DELTA,
            CardEventType.REASONING_DELTA,
            CardEventType.TOOL_DELTA,
        }:
            now = time.monotonic()
            with self._stream_progress_lock:
                if now - self._last_stream_progress_at < PROGRESS_DEBOUNCE_S:
                    return
                self._last_stream_progress_at = now
        self._fire_progress()

    def _handle_agent_attempt(self, label: str, attempt: int) -> None:
        """Expose bounded executor retries in the authoritative agent row."""
        if self._state_manager:
            self._state_manager.update_agent_attempt(label, attempt)
            self._fire_progress()

    def _handle_agent_subagent_update(
        self,
        label: str,
        updates: tuple[dict[str, object], ...],
    ) -> None:
        """Fold ACP child activity into the existing Workflow progress card."""
        if self._state_manager and self._state_manager.update_agent_subagents(
            label,
            updates,
        ):
            self._fire_progress()

    def _handle_agent_aborted(self, label: str, reason: str, *, request_id=None) -> None:
        """Handle an agent_aborted notification from the JS runtime.

        Called when a race() loser (or tournament elimination) agent is
        aborted. Updates the progress card so the agent no longer shows as
        '执行中'. The ACP session is interrupted via the per-call cancel_event
        set by the bridge's _handle_abort_request.

        Uses request_id for authoritative lookup (avoids label mismatch when
        state_manager disambiguates duplicate labels). Raw-label routing is
        only available to legacy notifications that omit request_id.
        """
        effective_label = label
        if request_id is not None:
            with self._request_to_label_lock:
                mapped = self._request_to_label.get(request_id)
            if mapped is None:
                logger.warning(
                    "Ignoring agent abort for unknown request_id=%s",
                    request_id,
                )
                return
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
            card_data = self.get_progress_cards()
            if not card_data:
                return
            if self._progress_coalescer:
                if immediate:
                    self._progress_coalescer.flush_immediate(card_data)
                else:
                    self._progress_coalescer.enqueue(card_data)
            else:
                self._callbacks.on_progress(card_data)
        except Exception:
            logger.debug("on_progress callback failed", exc_info=True)

    def clear_stale_renderer_for_generation(self) -> None:
        """Clear any prior run's renderer and state manager before GENERATING_SCRIPT."""
        with self._lock:
            self._renderer_wf = None
            self._state_manager = None

    def get_progress_cards(self) -> list[dict[str, Any]]:
        """Return status, direct-call execution streams, then result ledgers."""
        with self._lock:
            if self._project and self._project.status == WorkflowStatus.GENERATING_SCRIPT:
                return []
            renderer = self._renderer_wf
            state_manager = self._state_manager
        if not renderer:
            return []
        if state_manager:
            return renderer.render_progress_cards(state_manager.snapshot())
        return renderer.render_progress_cards()

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
        with self._lock:
            project = self._project
            renderer = self._renderer_wf
        if project and project.status == WorkflowStatus.GENERATING_SCRIPT:
            return "⏳ Workflow 正在生成编排脚本…"
        if renderer:
            return renderer.render_compact_status()
        if project and project.status == WorkflowStatus.FAILED:
            err = _strip_internal_details(project.error)
            if err:
                first_line = " ".join(err.splitlines()).strip()
                if len(first_line) > 200:
                    first_line = first_line[:199].rstrip() + "…"
                return f"❌ Workflow 失败：{first_line}"
            return "❌ Workflow 已失败"
        if project and project.status == WorkflowStatus.CANCELLED:
            return "🚫 Workflow 已取消"
        if project and project.status == WorkflowStatus.COMPLETED:
            return "✅ Workflow 已完成"
        return "Workflow: idle"
