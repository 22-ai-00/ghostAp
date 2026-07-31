"""SlockEngine — Multi-Agent collaboration engine (mouthpiece mode).

Inherits BaseEngine lifecycle and integrates AgentRegistry, MemoryManager,
TaskRouter, and Mouthpiece for orchestrating virtual agent teams.
"""

from __future__ import annotations

import asyncio as _asyncio
import contextlib
import logging
import os
import re
import secrets
import shlex
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from concurrent.futures import wait as futures_wait
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Optional

from ..agent_session import close_session_safely, create_engine_session
from ..config import get_settings
from ..engine_base import BaseEngine, EngineRunState
from .activation import activation_serialized
from .agent_registry import AgentRegistry
from .bounded_executor import BoundedExecutor, QueueFullError
from .card_templates import build_card_wrapper, build_status_panel_card
from .collaboration_orchestrator import CollaborationOrchestrator
from .escalation_manager import EscalationManager
from .exceptions import SecurityPolicyDegradedError, TaskQueueFullError
from .memory_manager import MemoryManager, default_slock_storage_base
from .models import (
    ABORT_OPTIONS,
    SKIP_OPTIONS,
    AgentIdentity,
    AgentStatus,
    DiscussionStatus,
    EscalationLevel,
    EscalationRequest,
    SlockChannel,
    SlockTask,
    TaskStatus,
    TaskTimelineEvent,
    TeamSnapshot,
)
from .mouthpiece import Mouthpiece
from .observer_queue import ObserverLearningQueue, TaskStatusNotifier
from .progress_tracker import ProgressTracker
from .task_board_manager import TaskBoardManager
from .task_chain_manager import TaskChainManager
from .task_queue import QueuedTask, TaskQueue
from .task_router import TaskRouter

logger = logging.getLogger(__name__)

_SHARED_LOOP: _asyncio.AbstractEventLoop | None = None
_SHARED_LOOP_LOCK = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
_LEGACY_EMPLOYEE_METRIC_INIT_LOCK = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock


class _EmployeeSessionLease:
    """Keep one isolated backend session registered until Actor recycle."""

    def __init__(self, engine: "SlockEngine", agent_id: str, session: object) -> None:
        self._engine = engine
        self._agent_id = agent_id
        self._session = session
        self._closed = False

    def send_prompt(self, prompt: str, *, timeout: float):
        return self._session.send_prompt(prompt, timeout=timeout)  # type: ignore[attr-defined]

    def is_server_healthy(self) -> bool:
        probe = getattr(self._session, "is_server_healthy", None)
        return bool(probe()) if callable(probe) else not self._closed

    def cancel(self) -> None:
        cancel = getattr(self._session, "cancel", None)
        if callable(cancel):
            cancel()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._engine._lock:
            if self._engine._agent_sessions.get(self._agent_id) is self._session:
                del self._engine._agent_sessions[self._agent_id]
        close_session_safely(self._session)


def _employee_acp_workspace(agent: AgentIdentity) -> str:
    raw_workspace = agent.workspace_path
    if (
        not isinstance(raw_workspace, str)
        or not raw_workspace
        or "\x00" in raw_workspace
        or not os.path.isabs(raw_workspace)
    ):
        raise SecurityPolicyDegradedError("employee ACP workspace is unavailable")
    workspace = os.path.realpath(raw_workspace)
    if not os.path.isdir(workspace):
        raise SecurityPolicyDegradedError("employee ACP workspace is unavailable")
    return workspace


def _positive_seconds(value: object, default: float) -> float:
    """Return a positive timeout value, falling back for mocks or invalid config."""
    if isinstance(value, bool):
        return default
    if isinstance(value, Real):
        seconds = float(value)
    elif isinstance(value, str):
        try:
            seconds = float(value.strip())
        except ValueError:
            return default
    else:
        return default
    return seconds if seconds > 0 else default


def _get_shared_loop() -> _asyncio.AbstractEventLoop:
    """Get the shared async bridge event loop.

    Delegates to the unified bridge in utils.async_helpers to avoid
    multiple competing event loops across the process.
    """
    from src.utils.async_helpers import _get_bridge_loop

    return _get_bridge_loop()


@dataclass
class SlockEngineCallbacks:
    """Callbacks for slock engine lifecycle events."""

    on_agent_wake: Optional[Callable[[AgentIdentity], None]] = None
    on_agent_thinking: Optional[Callable[[AgentIdentity], None]] = None
    on_agent_running: Optional[Callable[[AgentIdentity, str], None]] = None
    on_agent_done: Optional[Callable[[AgentIdentity, str], None]] = None
    on_agent_error: Optional[Callable[[AgentIdentity, str], None]] = None
    on_task_claimed: Optional[Callable[[SlockTask, AgentIdentity], None]] = None
    on_message_routed: Optional[Callable[[str, AgentIdentity], None]] = None
    on_escalation: Optional[Callable[[EscalationRequest], None]] = None
    on_error: Optional[Callable[[str], None]] = None
    # Discussion card lifecycle callbacks
    on_card_send: Optional[Callable[[dict], Optional[str]]] = None
    on_card_update: Optional[Callable[[str, dict], bool]] = None
    # Final result delivery callback for queued tasks
    on_final_result: Optional[Callable[[str, str, Optional[str]], None]] = None  # (task_id, result, card_message_id)


class SlockStreamProcessor:
    """Builds a streaming progress card that updates as agents work.

    Tracks agent state transitions and produces progressive card snapshots
    that a handler can send/update to show real-time team activity.

    Usage:
        processor = SlockStreamProcessor(engine)
        callbacks = processor.build_callbacks()
        engine.execute_parallel(assignments, callbacks)
        # Each callback triggers processor.get_progress_card() update
    """

    def __init__(self, engine: "SlockEngine", *, on_update: Optional[Callable[[dict], None]] = None):
        self._engine = engine
        self._on_update = on_update  # Called with card dict on each state change
        self._start_time = time.time()
        self._agent_activity: dict[str, str] = {}  # agent_id → last activity description
        self._completed_count = 0
        self._error_count = 0
        self._total_tasks = 0

    def build_callbacks(self) -> SlockEngineCallbacks:
        """Build callbacks that feed into this stream processor."""
        return SlockEngineCallbacks(
            on_agent_wake=self._on_wake,
            on_agent_thinking=self._on_thinking,
            on_agent_running=self._on_running,
            on_agent_done=self._on_done,
            on_agent_error=self._on_agent_error,
            on_error=self._on_error,
        )

    def set_total_tasks(self, count: int) -> None:
        """Set the total task count for progress tracking."""
        self._total_tasks = count

    def _on_wake(self, agent: AgentIdentity) -> None:
        self._agent_activity[agent.agent_id] = f"{agent.emoji} {agent.name}: waking..."
        self._emit_update()

    def _on_thinking(self, agent: AgentIdentity) -> None:
        self._agent_activity[agent.agent_id] = f"{agent.emoji} {agent.name}: 💭 thinking..."
        self._emit_update()

    def _on_running(self, agent: AgentIdentity, task: str) -> None:
        short_task = task[:60] + "..." if len(task) > 60 else task
        self._agent_activity[agent.agent_id] = f"{agent.emoji} {agent.name}: 🔄 {short_task}"
        self._emit_update()

    def _on_done(self, agent: AgentIdentity, result: str) -> None:
        self._agent_activity[agent.agent_id] = f"{agent.emoji} {agent.name}: ✅ done"
        self._completed_count += 1
        self._emit_update()

    def _on_agent_error(self, agent: AgentIdentity, error: str) -> None:
        short_err = error[:60] + "..." if len(error) > 60 else error
        self._agent_activity[agent.agent_id] = f"{agent.emoji} {agent.name}: ❌ {short_err}"
        self._error_count += 1
        self._emit_update()

    def _on_error(self, error_msg: str) -> None:
        self._error_count += 1
        self._emit_update()

    def _emit_update(self) -> None:
        """Emit a card update if a callback is registered."""
        if self._on_update:
            self._on_update(self.get_progress_card())

    def get_progress_card(self) -> dict:
        """Build the current progress card snapshot."""
        elapsed = time.time() - self._start_time
        channel = self._engine.channel
        team_name = channel.team_name if channel else "Slock"

        # Progress header
        if self._total_tasks > 0:
            progress_pct = int(self._completed_count / self._total_tasks * 100)
            header_title = f"⚡ {team_name} — {progress_pct}% ({self._completed_count}/{self._total_tasks})"
        else:
            header_title = f"⚡ {team_name} — Running"

        elements: list[dict] = []

        # Agent activity lines
        for activity in self._agent_activity.values():
            elements.append({"tag": "markdown", "content": activity})

        if not elements:
            elements.append({"tag": "markdown", "content": "*Waiting for agents...*"})

        # Footer with stats
        stats_parts = [f"⏱ {elapsed:.0f}s"]
        if self._completed_count:
            stats_parts.append(f"✅ {self._completed_count}")
        if self._error_count:
            stats_parts.append(f"❌ {self._error_count}")
        elements.append({
            "tag": "markdown",
            "content": " | ".join(stats_parts),
            "text_size": "notation",
        })

        return build_card_wrapper(
            header_title=header_title,
            header_template="blue",
            elements=elements,
            mobile_optimize=True,
        )


class AgentCancellationError(Exception):
    """Raised when an agent execution is cancelled via cancellation token."""


class SlockEngine(BaseEngine):
    """Multi-Agent collaboration engine using mouthpiece pattern.

    Manages a team of virtual agents within a single Feishu group,
    routing messages, managing tasks, and formatting output through
    the mouthpiece mechanism.

    Implements SlockEngineContext protocol to expose shared state to
    composed managers (TaskBoardManager, EscalationManager) via
    readonly properties/methods, eliminating lambda closure injection.

    Lock ordering (always acquire in this order to prevent deadlocks):
        1. self._lock (inherited RLock from BaseEngine)
        2. self._executor_lock (plain threading.Lock)
        3. BoundedExecutor._lock (leaf lock, never held while acquiring above)
    """

    _BUILTIN_DANGEROUS_PATTERNS: tuple[str, ...] = (
        r"\brm\s+-[^\n]*[rf][^\n]*/",
        r"\b(?:curl|wget)\b",
        r"\b(?:nc|ncat)\b",
        r"(?:^|[\s'\"])/(?:etc|root|proc|sys|dev)(?:/|\b)",
        r"(?:^|[\s'\"])(?:\.\./)+(?:etc|root|proc|sys|dev)(?:/|\b)",
    )

    # ------------------------------------------------------------------
    # SlockEngineContext protocol implementation
    # ------------------------------------------------------------------

    @property
    def channel(self) -> Optional[SlockChannel]:
        """当前激活的 SlockChannel（只读）。"""
        return self._channel

    @property
    def dirty(self) -> bool:
        """任务看板是否需要持久化（只读）。"""
        return self._dirty

    def set_dirty(self, value: bool) -> None:
        """设置 dirty 标志。"""
        self._set_dirty(value)

    def execute_agent(
        self,
        agent: AgentIdentity,
        content: str,
        callbacks: Any,
        *,
        freshness_check: bool = True,
    ) -> Optional[str]:
        """执行单个 agent 的响应周期。"""
        return self._execute_agent(agent, content, callbacks, freshness_check=freshness_check)

    def resolve_agent_for_role(self, role: str, channel_id: str) -> Optional[AgentIdentity]:
        """为指定角色在 channel 中解析最佳可用 agent。"""
        return self._resolve_agent_for_role(role, channel_id)

    _state_filename = ".slock_engine_state.json"
    _gc_label = "Slock"
    _gc_threshold_default = 85.0

    # Immutable state machine transition table — shared by transition_agent and
    # try_lock_for_move to ensure a single source of truth.  Values are tuples
    # (immutable) to prevent accidental runtime mutation.
    VALID_TRANSITIONS: dict[AgentStatus, tuple[AgentStatus, ...]] = {
        AgentStatus.IDLE: (AgentStatus.WAKING, AgentStatus.MOVING, AgentStatus.DISCUSSING),
        AgentStatus.WAKING: (AgentStatus.THINKING, AgentStatus.IDLE),
        AgentStatus.THINKING: (AgentStatus.RUNNING, AgentStatus.IDLE),
        AgentStatus.RUNNING: (AgentStatus.CHECKING, AgentStatus.IDLE),
        AgentStatus.CHECKING: (AgentStatus.SENDING, AgentStatus.RUNNING, AgentStatus.IDLE),
        AgentStatus.SENDING: (AgentStatus.IDLE,),
        AgentStatus.MOVING: (AgentStatus.IDLE,),
        AgentStatus.DISCUSSING: (AgentStatus.IDLE,),
        AgentStatus.PENDING_DISCUSSION: (AgentStatus.DISCUSSING, AgentStatus.IDLE),
    }

    def __init__(
        self,
        chat_id: str,
        root_path: str,
        agent_type: str = "coco",
        engine_name: str = "Slock",
        model_name: Optional[str] = None,
        *,
        memory_base_path: str = "",
    ):
        super().__init__(chat_id, root_path, agent_type, engine_name, model_name)

        # Core subsystems
        storage_base_path = memory_base_path or default_slock_storage_base()
        self._registry = AgentRegistry.legacy(base_path=storage_base_path)
        self._memory = MemoryManager(base_path=storage_base_path)
        self._channel_trust_rules: dict[str, dict[str, str]] = {}
        self._settings = get_settings()
        shell_patterns = list(self._BUILTIN_DANGEROUS_PATTERNS)
        shell_patterns.extend(getattr(self._settings, "slock_dangerous_shell_patterns", []) or [])
        self._dangerous_shell_patterns = re.compile(r"|".join(shell_patterns), re.IGNORECASE)
        claims_path = os.path.join(storage_base_path, "claims", f"{chat_id}.json")
        self._router = TaskRouter(
            persist_path=claims_path,
            memory_backend=self._memory,
            engine_status_getter=self.get_agent_status,
            session_affinity_window=float(getattr(get_settings(), "slock_session_affinity_window", 120)),
        )
        self._observer_queue = ObserverLearningQueue(memory=self._memory, router=self._router)
        self._mouthpiece = Mouthpiece()
        self._task_queue = TaskQueue(max_size=get_settings().slock_max_queue_size)

        # Dispatch loop: event-driven consumer that dequeues tasks on IDLE signals
        # Use prepare_bootstrap() / finish_bootstrap() public API to control this.
        self._bootstrap_ready = threading.Event()
        self._bootstrap_ready.set()  # Default: ready. Call prepare_bootstrap() to clear.
        self._dispatch_thread: Optional[threading.Thread] = None
        self._dispatch_stop = threading.Event()

        # Proactive Patrol loop: periodic check for SLA violations, idle agent auto-claim
        self._patrol_thread: Optional[threading.Thread] = None
        self._patrol_stop = threading.Event()
        self._patrol_interval = _positive_seconds(getattr(get_settings(), "slock_patrol_interval", 30), 30.0)

        # Thread pool for parallel agent execution
        self._executor: Optional[BoundedExecutor] = None
        self._executor_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

        # Independent executor for inter-agent discussions (decoupled from main agent execution)
        self._discussion_executor = BoundedExecutor(max_workers=2, max_queue_size=6)

        # Shared executor for parallel prompt context reads (reused across prompt builds)
        self._prompt_context_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="slock-ctx")

        # Sidebar channel for lightweight inter-agent communication
        from .sidebar_channel import SidebarChannel
        self._sidebar_channel = SidebarChannel()

        # Relationship graph for inter-agent collaboration history (lazy init)
        self._relationship_graph = None  # Initialized on first channel activation

        # Channel state
        self._channel: Optional[SlockChannel] = None
        self._tasks: list[SlockTask] = []
        self._dirty = False  # dirty-flag for debounced task board persistence
        self._agent_statuses: dict[str, AgentStatus] = {}
        self._agent_sessions: dict[str, object] = {}
        self._legacy_employee_session_calls = 0
        self._legacy_employee_session_calls_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._agent_execution_errors: dict[str, str] = {}
        self._escalations: list[EscalationRequest] = []
        self._escalation_retry_counts: dict[str, int] = {}

        # Persistent card delivery callbacks (set via set_card_callbacks)
        self._card_send_fn: Optional[Callable[[dict], Optional[str]]] = None
        self._card_update_fn: Optional[Callable[[str, dict], bool]] = None

        # Cancellation tokens for per-agent execution control
        self._cancel_events: dict[str, threading.Event] = {}
        self._cancel_events_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

        # Discussion state: tracks active inter-agent discussions per channel
        self._active_discussions: dict[str, list] = {}  # channel_id -> list[DiscussionThread]
        self._discussions_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._discussion_manager: Optional[object] = None  # Lazy-initialized

        # Autonomous resolver for ambiguity handling (AC-5)
        from .autonomous_resolver import AutonomousResolver
        self._autonomous_resolver: Optional[AutonomousResolver] = AutonomousResolver(
            llm_callback=self._summarize_via_llm,
        )

        # Status card auto-refresh: channel_id → last sent status card message_id
        self._status_card_msg_ids: dict[str, str] = {}
        self._status_refresh_timer: Optional[threading.Timer] = None

        # Proactive follow-up: track delivered results awaiting user feedback
        # Maps task_id → (delivery_time, agent_id, result_preview)
        self._pending_followups: dict[str, tuple[float, str, str]] = {}
        self._followup_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

        # Collaboration subsystem prerequisites (must precede TaskBoardManager)
        self._chain_manager = TaskChainManager()
        self._task_notifier = TaskStatusNotifier()

        # Composed managers (share lock and state references via SlockEngineContext)
        self._task_mgr = TaskBoardManager(
            lock=self._lock,
            tasks=self._tasks,
            context=self,
            router=self._router,
            memory=self._memory,
            registry_get=self._registry.get,
            chain_manager=self._chain_manager,
            notifier=self._task_notifier,
        )
        self._escalation_mgr = EscalationManager(
            lock=self._lock,
            escalations=self._escalations,
            retry_counts=self._escalation_retry_counts,
            context=self,
            router=self._router,
            transition_agent=self.transition_agent,
            flush_if_dirty=self._task_mgr._flush_if_dirty,
            get_executor_fn=self._get_executor,
            escalation_timeout_s=get_settings().slock_escalation_timeout,
        )

        # Collaboration orchestration subsystem
        self._collaboration_orchestrator = CollaborationOrchestrator(
            chain_manager=self._chain_manager,
            notifier=self._task_notifier,
            resolve_agent=self._resolve_agent_for_role,
            dispatch_task=self._dispatch_collaboration_task,
            register_task=self._register_collaboration_task,
            add_task_fn=self._task_mgr.add_task,
            claim_task_fn=self._task_mgr.claim_task,
            get_task_result_fn=self._get_task_execution_result,
            persist_fn=self._persist_plans,
        )
        self._task_notifier.subscribe(self._collaboration_orchestrator)

        # Progress tracker for rate-limited card updates
        from .card_channel_adapter import LazyCardChannel
        from .card_templates.progress import build_progress_overview_card

        _card_channel = LazyCardChannel(
            send_fn_getter=lambda: self._card_send_fn,
            update_fn_getter=lambda: self._card_update_fn,
        )

        self._progress_tracker = ProgressTracker(
            card_channel=_card_channel,
            card_builder=lambda state: build_progress_overview_card(
                plans=self._collaboration_orchestrator.list_active_plans(),
                agents=list(self._registry.list_agents()),
                team_name=getattr(self._channel, 'team_name', '') if self._channel else '',
                channel_id=getattr(self._channel, 'channel_id', '') if self._channel else '',
                highlight_plan_id=state.entity_id if state.entity_type == "plan" else "",
            ),
            min_interval=1.0,
            auto_flush=True,
            flush_period=3.0,
        )
        self._collaboration_orchestrator.set_progress_tracker(self._progress_tracker)

        # Register LLM summarization callback for memory compression
        self._memory.set_llm_callback(self._summarize_via_llm)

    def _summarize_via_llm(self, prompt: str) -> Optional[str]:
        """LLM callback for memory summarization via ACP session.

        Called synchronously from _summarize_text (which is itself invoked in a
        background thread with a join-timeout).  Returns the LLM response text
        directly so the memory compressor can use the result inline.

        Must NOT be called while holding self._memory._lock (caller ensures this).
        """
        try:
            session = create_engine_session(
                agent_type=self._agent_type,
                cwd=self.root_path,
                model_name=None,
                thread_id="slock_memory_summarize",
                auto_approve=False,
            )
            if session is None:
                logger.warning("Failed to create session for memory summarization")
                return None
            try:
                result = session.send_prompt(prompt, timeout=60)
                if result and result.text:
                    logger.debug(
                        "Memory summarization LLM returned %d chars",
                        len(result.text),
                    )
                    return result.text
                return None
            except Exception as exc:
                from src.utils.errors import redact_sensitive
                logger.warning("Memory summarization LLM call failed: %s", redact_sensitive(str(exc)), exc_info=True)
                return None
            finally:
                close_session_safely(session)
        except Exception as exc:
            from src.utils.errors import redact_sensitive
            logger.warning("Memory summarization setup failed: %s", redact_sensitive(str(exc)), exc_info=True)
            return None

    @property
    def registry(self) -> AgentRegistry:
        return self._registry

    def list_agents(self, channel_id=None):
        """List all registered agents. Delegates to registry."""
        return self._registry.list_agents(channel_id=channel_id)

    @staticmethod
    def _normalize_wake_policy(policy: str) -> str:
        normalized = (policy or "").strip().lower().replace("-", "_")
        return normalized

    def _effective_wake_policy(self, agent: AgentIdentity) -> str:
        """Resolve effective wake policy for an agent.

        Precedence (highest first): agent.wake_policy > channel.wake_policy >
        settings.slock_default_wake_policy. Empty/whitespace values fall
        through to the next layer; the function never returns empty.
        """
        agent_pol = self._normalize_wake_policy(getattr(agent, "wake_policy", "") or "")
        if agent_pol:
            return agent_pol
        if self._channel is not None:
            ch_pol = self._normalize_wake_policy(getattr(self._channel, "wake_policy", "") or "")
            if ch_pol:
                return ch_pol
        return self._normalize_wake_policy(
            getattr(get_settings(), "slock_default_wake_policy", "smart_judge") or "smart_judge"
        )

    def _apply_wake_policy(self, text: str, agents: list[AgentIdentity]) -> list[AgentIdentity]:
        """Drop agents whose effective wake policy disqualifies them for ``text``.

        Currently only ``on_mention`` filters: such agents stay candidates only
        when the text explicitly mentions them. ``smart_judge`` and unknown
        values pass through unchanged so behavior stays compatible with the
        legacy router.
        """
        if not agents:
            return agents
        filtered: list[AgentIdentity] = []
        for agent in agents:
            policy = self._effective_wake_policy(agent)
            if policy == "on_mention":
                if self._router._extract_mention(text or "", [agent]) is None:
                    continue
            filtered.append(agent)
        return filtered

    @property
    def memory(self) -> MemoryManager:
        return self._memory

    @property
    def router(self) -> TaskRouter:
        return self._router

    @property
    def mouthpiece(self) -> Mouthpiece:
        return self._mouthpiece

    def get_channel(self) -> Optional[SlockChannel]:
        """Public accessor for the active SlockChannel instance."""
        return self._channel

    def send_card(self, card_dict: dict) -> Optional[str]:
        """Send a card via the registered card delivery callback.

        Returns the message_id on success, or None if no callback is set.
        """
        if self._card_send_fn is None:
            return None
        return self._card_send_fn(card_dict)

    @property
    def tasks(self) -> list[SlockTask]:
        return list(self._tasks)

    @property
    def collaboration_orchestrator(self) -> CollaborationOrchestrator:
        """Access the collaboration orchestrator for plan management."""
        return self._collaboration_orchestrator

    @property
    def task_notifier(self) -> TaskStatusNotifier:
        """Access the task status notifier for event subscription."""
        return self._task_notifier

    @property
    def task_queue(self) -> TaskQueue:
        """Access the event-driven task queue for QUEUE_WAIT consumers."""
        return self._task_queue

    # ------------------------------------------------------------------
    # Dispatch loop — event-driven queue consumer
    # ------------------------------------------------------------------

    def enqueue_task(self, task: "QueuedTask") -> int:
        """Enqueue a task for dispatch-loop consumption.

        The handler calls this instead of blocking on wait_for_idle.
        Returns the 1-based queue position.
        """
        return self._task_queue.enqueue(task)

    def prepare_bootstrap(self) -> None:
        """Signal that bootstrap is about to start.

        Clears the bootstrap-ready event so the dispatch loop will wait
        until finish_bootstrap() is called. Should be called before
        activate_channel() or start_dispatch_loop() to ensure tasks are
        not dispatched before agents are registered.

        This is the public API; callers should use this instead of
        manipulating _bootstrap_ready directly.
        """
        self._bootstrap_ready.clear()
        logger.info("Bootstrap prepared: dispatch loop will wait for finish_bootstrap()")

    def signal_bootstrap_complete(self) -> None:
        """Signal that default role bootstrap has finished.

        The dispatch loop waits for this event before consuming queued tasks,
        ensuring agents are registered before routing.

        .. deprecated::
            Prefer finish_bootstrap() as the public API. This method is
            kept for backward compatibility with existing callers.
        """
        self._bootstrap_ready.set()
        logger.info("Bootstrap complete signal received, dispatch loop unblocked")

    def finish_bootstrap(self) -> None:
        """Signal that bootstrap has completed.

        Sets the bootstrap-ready event, unblocking the dispatch loop.
        This is the public API; handlers should call this instead of
        signal_bootstrap_complete() directly.

        Pair with prepare_bootstrap() to ensure the dispatch loop waits
        for agent registration before consuming tasks.
        """
        self.signal_bootstrap_complete()

    def start_dispatch_loop(self) -> None:
        """Start the background dispatch loop thread (idempotent)."""
        if self._dispatch_thread and self._dispatch_thread.is_alive():
            return
        self._dispatch_stop.clear()
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop,
            name=f"slock-dispatch-{self.chat_id[:8]}",
            daemon=True,
        )
        self._dispatch_thread.start()
        logger.info("Dispatch loop started for chat=%s", self.chat_id)

    def _dispatch_loop(self) -> None:
        """Background loop: wait for IDLE signal → dequeue → route → execute.

        Design constraints:
        - Does NOT hold Condition lock while calling _execute_agent
        - Uses single-consumer dequeue to prevent thundering herd
        - Respects _dispatch_stop for graceful shutdown
        - Bootstrap self-healing: retries up to 3 times with exponential backoff
          if registry is empty after bootstrap timeout, sends recovery card,
          and retains all queued tasks.
        """
        # Wait for bootstrap with self-healing retries
        bootstrap_timeout = _positive_seconds(getattr(get_settings(), "slock_bootstrap_timeout", 10), 10.0)
        max_retries = 3
        bootstrap_ok = False

        for attempt in range(max_retries + 1):
            self._bootstrap_ready.wait(timeout=bootstrap_timeout)
            if self._dispatch_stop.is_set():
                return

            if self._bootstrap_ready.is_set() and self.list_agents():
                bootstrap_ok = True
                break

            if self._task_queue.size() == 0:
                logger.info(
                    "Bootstrap registry empty with no queued tasks; "
                    "dispatch loop entering standby",
                )
                break

            if attempt < max_retries:
                backoff = min(2 ** (attempt + 1), 30)
                logger.warning(
                    "Bootstrap attempt %d/%d: registry empty after %ds, "
                    "retrying in %ds (tasks retained)",
                    attempt + 1, max_retries, bootstrap_timeout, backoff,
                )
                # Send recovery status card
                self._send_bootstrap_recovery_card(attempt + 1, max_retries)
                # Reset the event so next wait is fresh
                self._bootstrap_ready.clear()
                if self._dispatch_stop.wait(timeout=backoff):
                    return
            else:
                logger.warning(
                    "Bootstrap failed after %d retries, dispatch loop proceeding "
                    "with empty registry — tasks will be retained and retried on agent arrival",
                    max_retries,
                )

        if bootstrap_ok:
            logger.info("Dispatch loop active, waiting for tasks...")
        else:
            logger.info("Dispatch loop active (degraded: no agents yet), waiting for tasks...")

        while not self._dispatch_stop.is_set():
            # Block until IDLE notification or timeout (for periodic liveness check)
            self._task_queue.wait_for_idle(timeout=5.0)

            if self._dispatch_stop.is_set():
                break

            # Dequeue and dispatch tasks while agents are available
            while not self._dispatch_stop.is_set():
                task = self._task_queue.dequeue()
                if task is None:
                    break  # Queue empty, go back to waiting

                self._dispatch_single_task(task)

        logger.info("Dispatch loop stopped for chat=%s", self.chat_id)

    def _dispatch_single_task(self, task: "QueuedTask") -> None:
        """Route and submit a single dequeued task to the executor.

        Called from _dispatch_loop. Finds the best idle agent and submits
        execution to the bounded executor. If no agent is idle, re-enqueues
        with a next_retry_at timestamp (non-blocking backoff).
        Sends timeout notification if the task has waited too long.

        Thread-safety: self._channel is read under self._lock to avoid
        race with activate_channel/deactivate_channel. Snapshot is taken
        inside the lock and used outside to minimize critical section.
        """
        import time as _time

        # Non-blocking backoff: skip tasks not yet eligible for retry
        if task.next_retry_at > 0 and _time.time() < task.next_retry_at:
            # Put it back and let the loop continue to other tasks
            try:
                self._task_queue.enqueue(task)
            except TaskQueueFullError:
                logger.warning("Task queue is full, task %s will be discarded. Consider increasing slock_max_queue_size in settings.", task.task_id)
                # 如果有回调，通知用户
                if task.callbacks and task.callbacks.on_error:
                    task.callbacks.on_error(f"Task queue is full, task {task.task_id} has been discarded. Please try again later or increase queue size.")
            except Exception as e:
                logger.warning("Failed to re-enqueue task %s: %s", task.task_id, str(e))
            return

        # 锁内快照：避免与 activate_channel/deactivate_channel 的并发修改
        with self._lock:
            channel_snapshot = self._channel

        if not channel_snapshot:
            logger.warning("Dispatch loop: no channel active, dropping task %s", task.task_id)
            return

        # If the task was enqueued before bootstrap and the channel bootstrap failed,
        # mark it unschedulable and dequeue (do not leave waiting forever).
        if task.bootstrap_pending and getattr(channel_snapshot, 'bootstrap_failed', False):
            logger.warning(
                "Task %s marked unschedulable: channel %s bootstrap failed",
                task.task_id, task.chat_id,
            )
            task.status = 'unschedulable'
            self._send_bootstrap_failed_card(task)
            return

        # Check if task has exceeded queue wait timeout
        wait_timeout = get_settings().slock_queue_wait_timeout
        waited = _time.time() - task.enqueue_time
        if waited > wait_timeout:
            logger.warning(
                "Dispatch: task %s waited %.1fs (timeout=%ds), sending timeout card",
                task.task_id, waited, wait_timeout,
            )
            self._send_timeout_card(task, waited)
            return

        agents = self.list_agents(channel_id=channel_snapshot.channel_id)
        if not agents:
            # No agents registered yet — re-enqueue with non-blocking backoff
            task.retry_count += 1
            task.next_retry_at = _time.time() + min(2 ** task.retry_count, 30)
            logger.debug("Dispatch: no agents registered, re-enqueueing task %s (retry=%d, next_at=+%ds)", task.task_id, task.retry_count, min(2 ** task.retry_count, 30))
            try:
                self._task_queue.enqueue(task)
            except TaskQueueFullError:
                logger.warning("Task queue is full, task %s will be discarded. Consider increasing slock_max_queue_size in settings.", task.task_id)
                # 如果有回调，通知用户
                if task.callbacks and task.callbacks.on_error:
                    task.callbacks.on_error(f"Task queue is full, task {task.task_id} has been discarded. Please try again later or increase queue size.")
            except Exception as e:
                logger.warning("Failed to re-enqueue task %s: %s", task.task_id, str(e))
            return

        # Find idle agent via router (after wake-policy filtering)
        candidates = self._apply_wake_policy(task.text, agents)
        routing_result = self._router.route_message_with_fallback(task.text, candidates)

        from .task_router import RoutingStatus
        if routing_result.status == RoutingStatus.ASSIGNED and routing_result.agent:
            # Route through TaskBoard for unified lifecycle tracking
            agent = routing_result.agent
            callbacks = task.callbacks

            # Wrap callbacks to deliver final result via task's callback
            original_on_done = callbacks.on_agent_done if callbacks else None
            queued_task_id = task.task_id
            card_message_id = task.card_message_id
            final_callback = task.final_result_callback

            # Create a board-level SlockTask for lifecycle tracking
            board_task = self._task_mgr.add_task(task.text)
            if board_task is None:
                # Board full — re-enqueue with backoff
                task.retry_count += 1
                task.next_retry_at = _time.time() + min(2 ** task.retry_count, 30)
                with contextlib.suppress(Exception):
                    self._task_queue.enqueue(task)
                return

            board_task_id = board_task.task_id

            def wrapped_on_done(a: AgentIdentity, result: str) -> None:
                if original_on_done:
                    original_on_done(a, result)
                # Deliver final result via callback
                if final_callback:
                    try:
                        final_callback(queued_task_id, result, card_message_id)
                    except Exception as e:
                        logger.error("Failed to deliver final result for task %s: %s", queued_task_id, e, exc_info=True)
                # Notify queue that agent is idle
                self._task_queue.notify_idle()

            wrapped_callbacks = SlockEngineCallbacks(
                on_agent_wake=callbacks.on_agent_wake if callbacks else None,
                on_agent_thinking=callbacks.on_agent_thinking if callbacks else None,
                on_agent_running=callbacks.on_agent_running if callbacks else None,
                on_agent_done=wrapped_on_done,
                on_agent_error=callbacks.on_agent_error if callbacks else None,
                on_task_claimed=callbacks.on_task_claimed if callbacks else None,
                on_message_routed=callbacks.on_message_routed if callbacks else None,
                on_escalation=callbacks.on_escalation if callbacks else None,
                on_error=callbacks.on_error if callbacks else None,
                on_card_send=callbacks.on_card_send if callbacks else None,
                on_card_update=callbacks.on_card_update if callbacks else None,
            )

            logger.info(
                "Dispatch: assigning task %s (board=%s) to agent %s (waited %.1fs)",
                queued_task_id, board_task_id, agent.name, waited,
            )
            # Execute through TaskBoard lifecycle: claim → execute → review
            executor = self._get_executor()
            try:
                executor.submit(
                    self._task_mgr.execute_task,
                    board_task_id,
                    agent.agent_id,
                    wrapped_callbacks,
                )
            except Exception as exc:
                logger.error("Failed to submit board task to executor: %s", exc, exc_info=True)
        elif routing_result.status == RoutingStatus.QUEUE_WAIT:
            # All agents still busy — put task back with non-blocking backoff
            task.retry_count += 1
            task.next_retry_at = _time.time() + min(2 ** task.retry_count, 30)
            try:
                self._task_queue.enqueue(task)
            except Exception as e:
                logger.error("Failed to re-enqueue task %s: %s", task.task_id, e, exc_info=True)
        else:
            # NO_MATCH — likely chitchat that slipped through, just drop
            logger.debug("Dispatch: NO_MATCH for task %s, dropping", task.task_id)

    def _handle_agent_plan_output(
        self,
        agent: AgentIdentity,
        result: str,
        original_message: str,
        channel_id: str,
        callbacks: Optional[SlockEngineCallbacks],
    ) -> bool:
        """Parse a [PLAN]...[/PLAN] block from agent output and register with orchestrator.

        Returns True if a valid plan was created, False otherwise.
        """
        import re

        # Extract the plan block
        plan_match = re.search(r'\[PLAN\](.*?)\[/PLAN\]', result, re.DOTALL)
        if not plan_match:
            return False

        plan_body = plan_match.group(1).strip()
        if not plan_body:
            return False

        # Parse sub-tasks: [SUB:role] description [DEPENDS:n,m]
        sub_pattern = re.compile(
            r'\[SUB:([^\]]+)\]\s*(.+?)(?:\s*\[DEPENDS:([^\]]+)\])?\s*$',
            re.MULTILINE,
        )
        sub_tasks = []
        for match in sub_pattern.finditer(plan_body):
            role = match.group(1).strip()
            description = match.group(2).strip()
            depends_str = match.group(3)
            depends_on = []
            if depends_str:
                depends_on = [d.strip() for d in depends_str.split(",") if d.strip()]
            sub_tasks.append({
                "role": role,
                "description": description,
                "depends_on": depends_on,
            })

        if not sub_tasks:
            return False

        # Create a plan via the orchestrator
        try:
            plan = self._collaboration_orchestrator.create_plan_from_agent_output(
                planner_agent=agent,
                original_task_content=original_message,
                sub_tasks=sub_tasks,
                channel_id=channel_id,
            )
            if plan:
                logger.info(
                    "Agent %s created plan with %d steps for: %s",
                    agent.name, len(sub_tasks), original_message[:80],
                )
                return True
        except Exception as exc:
            logger.error("Failed to create plan from agent output: %s", exc, exc_info=True)

        return False

    @staticmethod
    def _format_plan_acknowledgment(result: str) -> str:
        """Format a plan output into a user-friendly acknowledgment message."""
        import re

        plan_match = re.search(r'\[PLAN\](.*?)\[/PLAN\]', result, re.DOTALL)
        if not plan_match:
            return result

        plan_body = plan_match.group(1).strip()
        # Count sub-tasks
        sub_count = plan_body.count("[SUB:")

        # Build acknowledgment
        acknowledgment = (
            f"📋 **已创建协作计划** ({sub_count} 个子任务)\n\n"
            f"{plan_body}\n\n"
            f"---\n"
            f"计划已注册，子任务将自动分配给对应角色执行。"
        )

        # If there's content after [/PLAN], append it
        after_plan = result.split("[/PLAN]")[-1].strip()
        if after_plan:
            acknowledgment += f"\n\n{after_plan}"

        return acknowledgment

    def _fan_out_to_others(
        self,
        source_agent: AgentIdentity,
        message: str,
        channel_id: str,
        callbacks: Optional[SlockEngineCallbacks],
    ) -> None:
        """Re-dispatch a message to all agents EXCEPT the source agent.

        Called when an agent decides (via [DELEGATE:ALL]) that a task should
        involve all team members. The source agent already produced its own
        response; this dispatches to the rest.
        """
        agents = self.list_agents(channel_id=channel_id)
        others = [a for a in agents if a.agent_id != source_agent.agent_id]
        if not others:
            return

        logger.info(
            "Agent-driven fan-out: %s delegated to %d other agents",
            source_agent.name, len(others),
        )
        executor = self._get_executor()
        for agent in others:
            try:
                executor.submit(self._execute_agent, agent, message, callbacks)
            except Exception as exc:
                logger.warning("Fan-out: failed to submit to %s: %s", agent.name, repr(exc))

    def _send_timeout_card(self, task: "QueuedTask", waited_seconds: float) -> None:
        """Send a timeout notification card for a task that waited too long."""
        try:
            from .card_templates.queue_feedback import build_timeout_notify_card

            card = build_timeout_notify_card(
                task_id=task.task_id,
                message_preview=task.text,
                waited_seconds=waited_seconds,
            )
            if self._card_send_fn:
                self._card_send_fn(card)
        except Exception as e:
            logger.warning("Failed to send timeout card for task %s: %s", task.task_id, e, exc_info=True)

    def _send_bootstrap_recovery_card(self, attempt: int, max_retries: int) -> None:
        """Send a status card informing users that bootstrap is being retried."""
        try:
            from .card_templates.queue_feedback import build_no_agent_available_card

            card = build_no_agent_available_card(
                team_name=self.chat_id or "Team",
                hint=f"正在第 {attempt}/{max_retries} 次尝试恢复角色注册...",
            )
            if self._card_send_fn:
                self._card_send_fn(card)
        except Exception as e:
            logger.warning("Failed to send bootstrap recovery card (attempt %d): %s", attempt, e, exc_info=True)

    def _send_bootstrap_failed_card(self, task: "QueuedTask") -> None:
        """Send a notification card when a task is unschedulable due to bootstrap failure."""
        try:
            from .card_templates.queue_feedback import build_no_agent_available_card

            card = build_no_agent_available_card(
                team_name=task.chat_id,
                hint="Bootstrap failed — no agents were created. Task marked unschedulable.",
            )
            if self._card_send_fn:
                self._card_send_fn(card)
        except Exception as e:
            logger.warning("Failed to send bootstrap-failed card for task %s: %s", task.task_id, e, exc_info=True)

    def stop_dispatch_loop(self) -> None:
        """Stop the dispatch loop gracefully."""
        self._dispatch_stop.set()
        self._bootstrap_ready.set()  # Wake bootstrap wait so shutdown is not delayed by bootstrap timeout.
        self._task_queue.notify_idle()  # Wake the loop so it can check the stop flag
        if self._dispatch_thread and self._dispatch_thread.is_alive():
            self._dispatch_thread.join(timeout=5.0)
            if self._dispatch_thread.is_alive():
                logger.warning("Dispatch loop did not stop within 5s for chat=%s", self.chat_id)

    # ------------------------------------------------------------------
    # Proactive Patrol loop — periodic SLA check + idle agent auto-claim
    # ------------------------------------------------------------------

    def start_patrol_loop(self) -> None:
        """Start the background patrol loop (idempotent)."""
        if self._patrol_thread and self._patrol_thread.is_alive():
            return
        self._patrol_stop.clear()
        self._patrol_thread = threading.Thread(
            target=self._patrol_loop,
            name=f"slock-patrol-{self.chat_id[:8]}",
            daemon=True,
        )
        self._patrol_thread.start()
        logger.info("Patrol loop started for chat=%s (interval=%ds)", self.chat_id, self._patrol_interval)

    def stop_patrol_loop(self) -> None:
        """Stop the patrol loop gracefully."""
        self._patrol_stop.set()
        if self._patrol_thread and self._patrol_thread.is_alive():
            self._patrol_thread.join(timeout=5.0)

    def _patrol_loop(self) -> None:
        """Background loop: runs every _patrol_interval seconds.

        Responsibilities:
        1. Check tasks that exceeded their SLA deadline → send reminder/escalate
        2. Check idle agents → trigger TaskQueue notify_idle to wake dispatch loop
        3. Purge stale session affinity entries from the router
        """
        while not self._patrol_stop.is_set():
            self._patrol_stop.wait(timeout=self._patrol_interval)
            if self._patrol_stop.is_set():
                break

            try:
                self._patrol_check_sla()
                self._patrol_idle_agents()
                self._patrol_purge_stale()
                self._patrol_detect_orphan_tasks()
                self._patrol_proactive_followup()
                self._patrol_renew_active_claims()
                self._sidebar_channel.expire_stale()
            except Exception as exc:
                logger.warning("Patrol loop error: %s", exc, exc_info=True)

        logger.info("Patrol loop stopped for chat=%s", self.chat_id)

    def _patrol_check_sla(self) -> None:
        """Check tasks exceeding their SLA deadline and send notifications."""
        now = time.time()
        with self._lock:
            tasks_snapshot = list(self._tasks)

        for task in tasks_snapshot:
            if task.status != TaskStatus.IN_PROGRESS:
                continue
            # Set deadline on claim if not already set
            if task.deadline_at is None and task.claimed_at:
                task.deadline_at = task.claimed_at + task.sla_seconds
            if task.deadline_at is None:
                continue
            if now > task.deadline_at:
                # SLA violated — send escalation notification
                overdue_s = now - task.deadline_at
                logger.warning(
                    "Patrol: task %s overdue by %.0fs (claimed_by=%s)",
                    task.task_id, overdue_s, task.claimed_by,
                )
                self._patrol_escalate_overdue(task, overdue_s)
                # Extend deadline to avoid repeated escalation every interval
                task.deadline_at = now + task.sla_seconds

    def _patrol_detect_orphan_tasks(self) -> None:
        """Detect tasks stuck IN_PROGRESS for more than 2x SLA and reset them."""
        now = time.time()
        with self._lock:
            tasks_snapshot = list(self._tasks)

        for task in tasks_snapshot:
            if task.status != TaskStatus.IN_PROGRESS:
                continue
            if not task.claimed_at:
                continue
            orphan_threshold = task.sla_seconds * 2
            if (now - task.claimed_at) > orphan_threshold:
                logger.warning(
                    "Patrol: orphan task %s — stuck %.0fs (claimed_by=%s), resetting to TODO",
                    task.task_id, now - task.claimed_at, task.claimed_by,
                )
                with self._lock:
                    task.status = TaskStatus.TODO
                    task.claimed_by = None
                    task.claimed_at = None
                    task.deadline_at = None
                    task.timeline.append(TaskTimelineEvent(
                        event_type="orphan_recovered",
                        agent_id="patrol",
                        timestamp=now,
                        detail=f"Reset after {orphan_threshold:.0f}s orphan threshold",
                    ))

    def _patrol_escalate_overdue(self, task: "SlockTask", overdue_seconds: float) -> None:
        """Send a notification card for an overdue task."""
        try:
            from .card_templates.common import build_card_wrapper
            agent = self._registry.get(task.claimed_by) if task.claimed_by else None
            agent_name = agent.name if agent else "未知 Agent"
            content = (
                f"⏰ 任务超时 **{overdue_seconds:.0f}s**\n\n"
                f"> {task.content[:100]}\n\n"
                f"负责人: **{agent_name}** · 如长时间无进展将自动重新分配"
            )
            card = build_card_wrapper(
                header_title="⏰ 任务 SLA 超时",
                header_template="orange",
                elements=[{"tag": "markdown", "content": content}],
            )
            if self._card_send_fn:
                self._card_send_fn(card)
        except Exception as exc:
            logger.warning("Failed to send SLA escalation card: %s", exc, exc_info=True)

    def _patrol_idle_agents(self) -> None:
        """If there are idle agents and queued tasks, trigger dispatch."""
        channel_id = self._channel.channel_id if self._channel else self.chat_id
        agents = self._registry.list_agents(channel_id=channel_id)
        has_idle = any(
            self.get_agent_status(a.agent_id) == AgentStatus.IDLE
            for a in agents
        )
        if has_idle and self._task_queue.size() > 0:
            self._task_queue.notify_idle()

    def _patrol_purge_stale(self) -> None:
        """Purge stale session affinity entries from the router."""
        try:
            self._router.purge_stale_affinity()
        except Exception:
            pass

    def _patrol_proactive_followup(self) -> None:
        """Check delivered results that haven't received user feedback — send proactive follow-up."""
        settings = get_settings()
        if not getattr(settings, "slock_proactive_followup_enabled", True):
            return

        delay = getattr(settings, "slock_proactive_followup_delay", 60)
        now = time.time()

        with self._followup_lock:
            expired = [
                (task_id, agent_id, preview)
                for task_id, (delivery_time, agent_id, preview) in self._pending_followups.items()
                if now - delivery_time > delay
            ]
            for task_id, _, _ in expired:
                del self._pending_followups[task_id]

        for task_id, agent_id, preview in expired:
            self._send_followup_card(task_id, agent_id, preview)

    def register_pending_followup(self, task_id: str, agent_id: str, result_preview: str) -> None:
        """Register a delivered result for proactive follow-up tracking."""
        with self._followup_lock:
            self._pending_followups[task_id] = (time.time(), agent_id, result_preview[:200])

    def cancel_pending_followup(self, task_id: str) -> None:
        """Cancel follow-up when user acknowledges or interacts with the result."""
        with self._followup_lock:
            self._pending_followups.pop(task_id, None)

    def _send_followup_card(self, task_id: str, agent_id: str, preview: str) -> None:
        """Send a proactive follow-up card asking if the result was helpful."""
        try:
            from .card_templates.common import build_card_wrapper

            agent = self._registry.get(agent_id)
            agent_name = agent.name if agent else "Agent"
            content = (
                f"💬 **{agent_name}** 已完成任务并交付结果：\n\n"
                f"> {preview}\n\n"
                f"如需调整或有后续需求，请直接回复或 @{agent_name}。"
            )
            card = build_card_wrapper(
                header_title="💬 需要进一步帮助吗？",
                header_template="blue",
                elements=[{"tag": "markdown", "content": content}],
            )
            if self._card_send_fn:
                self._card_send_fn(card)
        except Exception as exc:
            logger.debug("Failed to send follow-up card for task %s: %s", task_id, exc, exc_info=True)

    # ------------------------------------------------------------------
    # Collaboration helpers
    # ------------------------------------------------------------------

    def _patrol_renew_active_claims(self) -> None:
        """Renew heartbeats for all actively running tasks to prevent TTL expiry.

        Also purges expired claims (agent crashed without releasing).
        """
        # Renew claims for tasks currently in IN_PROGRESS
        with self._lock:
            active_tasks = [
                (t.task_id, t.claimed_by)
                for t in self._tasks
                if t.status == TaskStatus.IN_PROGRESS and t.claimed_by
            ]

        task_claim = self._task_mgr._router.task_claim
        for task_id, agent_id in active_tasks:
            task_claim.renew(task_id, agent_id)

        # Purge expired claims (catches crashed agents)
        purged = task_claim.purge_expired()
        if purged:
            logger.info("Patrol: purged %d expired task claims", purged)

    def _resolve_agent_for_role(self, role: str, channel_id: str) -> Optional[AgentIdentity]:
        """Find the best idle agent matching a role in this channel.

        Uses skill-based scoring when multiple agents match the same role:
        prefers idle agents with higher average skill success rate.
        Falls back to first idle agent when no skill data exists.
        """
        if not self._channel:
            return None

        candidates: list[tuple[AgentIdentity, float]] = []
        for agent_id in self._channel.agents:
            agent = self._registry.get(agent_id)
            if agent and agent.role == role:
                status = self._agent_statuses.get(agent_id, AgentStatus.IDLE)
                if status == AgentStatus.IDLE:
                    # Score by skill profiles (higher is better)
                    score = self._compute_skill_score(agent_id)
                    candidates.append((agent, score))

        if not candidates:
            return None

        # Return highest-scoring idle agent
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def _compute_skill_score(self, agent_id: str) -> float:
        """Compute a skill score for agent selection (0.0 - 100.0).

        Returns 50.0 (neutral) when no skill data is available,
        ensuring new agents don't get unfairly deprioritized.
        """
        try:
            profiles = self._router.get_skill_profiles(agent_id)
            if not profiles:
                return 50.0
            total_rate = sum(p.success_rate for p in profiles)
            return total_rate / len(profiles)
        except Exception:
            logger.warning("Failed to compute skill score for agent %s", agent_id, exc_info=True)
            return 50.0

    def _dispatch_collaboration_task(self, task: SlockTask, agent: AgentIdentity) -> None:
        """Dispatch a collaboration task to an agent.

        NOTE: Task must already be registered in the task board via
        _register_collaboration_task or TaskBoardManager.add_task before
        calling this method. This method only tracks progress and executes.
        """
        # Track progress for this task
        self._progress_tracker.update(
            task.task_id, entity_type="task", status="dispatched",
            detail=f"{agent.name} 执行中",
        )
        # Execute via the engine's execute_task method
        callbacks = self._build_collaboration_task_callbacks(task)
        self.execute_task(task.task_id, agent.agent_id, callbacks)

    def _build_collaboration_task_callbacks(self, task: SlockTask) -> SlockEngineCallbacks:
        """Build visible callbacks for orchestrated plan steps."""
        channel_id = task.created_in or (self._channel.channel_id if self._channel else self.chat_id)

        def on_card_send(card: dict) -> Optional[str]:
            if self._card_send_fn is None:
                return None
            return self._card_send_fn(card)

        def on_card_update(msg_id: str, card: dict) -> bool:
            if self._card_update_fn is None:
                return False
            return self._card_update_fn(msg_id, card)

        def on_agent_done(done_agent: AgentIdentity, result: str) -> None:
            if not result or self._card_send_fn is None:
                return
            try:
                card = self._mouthpiece.format_card(
                    done_agent,
                    result,
                    channel_id=channel_id,
                    task_id=task.task_id,
                )
                self._card_send_fn(card)
            except Exception:
                logger.warning(
                    "Failed to send collaboration task result card for task %s",
                    task.task_id,
                    exc_info=True,
                )

        return SlockEngineCallbacks(
            on_agent_done=on_agent_done,
            on_card_send=on_card_send,
            on_card_update=on_card_update,
        )

    def _register_collaboration_task(self, task: SlockTask) -> None:
        """Register a collaboration task in the task list for tracking/persistence.

        Called by CollaborationOrchestrator when creating step tasks.
        Since _dispatch_collaboration_task already appends to _tasks, this is a no-op
        if the task is already present (avoids duplication).
        """
        with self._lock:
            if not any(t.task_id == task.task_id for t in self._tasks):
                self._tasks.append(task)
                self._dirty = True

    def _get_task_execution_result(self, task_id: str) -> Optional[str]:
        """Get the execution result of a completed task for context passing.

        Called by CollaborationOrchestrator to provide predecessor context
        to successor steps, enabling collaboration continuity.
        """
        with self._lock:
            for task in self._tasks:
                if task.task_id == task_id and task.status == TaskStatus.DONE:
                    return task.execution_result or None
        return None

    def _persist_plans(self) -> None:
        """Persist all plans to disk (called by orchestrator after state changes)."""
        channel = self._channel
        if not channel:
            return
        channel_id = channel.channel_id if hasattr(channel, 'channel_id') else ""
        if not channel_id:
            return
        try:
            plans = self._collaboration_orchestrator.get_all_plans()
            self._memory.write_plans(channel_id, plans)
        except OSError:
            logger.warning("Failed to persist collaboration plans")

    def _restore_plans(self) -> None:
        """Restore collaboration plans from disk (called on engine activation)."""
        channel = self._channel
        if not channel:
            return
        channel_id = channel.channel_id if hasattr(channel, 'channel_id') else ""
        if not channel_id:
            return
        try:
            plans = self._memory.read_plans(channel_id)
            if plans:
                self._collaboration_orchestrator.restore_plans(plans, channel_id)
                logger.info("Restored %d plans from disk", len(plans))
        except Exception:
            logger.warning("Failed to restore collaboration plans", exc_info=True)

    # ------------------------------------------------------------------
    # Public API: exposed for DiscussionManager (avoids private access)
    # ------------------------------------------------------------------

    def set_card_callbacks(
        self,
        send_fn: Optional[Callable[[dict], Optional[str]]],
        update_fn: Optional[Callable[[str, dict], bool]],
    ) -> None:
        """Set persistent card delivery callbacks for progress tracking.

        Should be called by the handler once card delivery is available.
        """
        self._card_send_fn = send_fn
        self._card_update_fn = update_fn

    def get_agent(self, agent_id: str) -> Optional[AgentIdentity]:
        """Get agent identity by ID. Public API for discussion/external use."""
        return self._registry.get(agent_id)

    def assign_task_to_agent(self, task_id: str, agent_id: str) -> bool:
        """Assign (claim) an existing task to an agent."""
        return self._task_mgr.claim_task(task_id, agent_id)

    def create_and_assign_task(self, content: str, agent_id: str) -> Optional["SlockTask"]:
        """Create a new task and immediately assign it to an agent.

        Returns the created SlockTask or None on failure.
        """
        task = self._task_mgr.add_task(content)
        if not task:
            return None
        success = self._task_mgr.claim_task(task.task_id, agent_id)
        if not success:
            return task  # Created but not claimed — still return it
        return task

    def find_agent_by_name(self, name: str, channel_id: Optional[str] = None) -> Optional[AgentIdentity]:
        """Find agent by display name. Public API for NLI discussion routing."""
        return self._registry.find_by_name(name, channel_id=channel_id)

    def build_agent_prompt(self, agent: AgentIdentity, message: str, memory=None) -> str:
        """Build full prompt for an agent. Public wrapper around _build_agent_prompt.

        If memory is None, reads from memory manager automatically.
        """
        if memory is None:
            memory = self._memory.read_agent_memory(agent.agent_id)
        return self._build_agent_prompt(agent, message, memory)

    def run_agent_session(
        self,
        agent: AgentIdentity,
        prompt: str,
        *,
        timeout: Optional[float] = None,
        env: Optional[dict[str, str]] = None,
    ) -> Optional[str]:
        """Run an ACP session for the agent. Public wrapper around _run_acp_session."""
        if agent.security_profile != "employee_v1":
            return self._run_acp_session(agent, prompt, timeout=timeout)
        self._record_legacy_employee_session_call()
        if env is None:
            raise SecurityPolicyDegradedError(
                "employee session environment is not explicitly scoped"
            )
        from src.agent_session.factory import employee_session_environment
        from src.employee_session_scope import (
            EmployeeSessionOutcome,
            employee_session_outcome_capture,
        )

        with employee_session_outcome_capture() as capture:
            with employee_session_environment(env):
                result = self._run_acp_session(agent, prompt, timeout=timeout)
        if capture.outcome is EmployeeSessionOutcome.TIMEOUT:
            raise TimeoutError("employee ACP session timed out")
        if capture.outcome is EmployeeSessionOutcome.CANCELED:
            from concurrent.futures import CancelledError

            raise CancelledError("employee ACP session was canceled")
        return result

    def preview_employee_session_prompt(
        self,
        agent: AgentIdentity,
        prompt: str,
    ) -> str:
        """Build the exact employee prompt used by the legacy one-shot path.

        Shadow mode calls this pure preview to detect input drift without
        starting a second backend session or issuing another model call.
        """

        if agent.security_profile != "employee_v1":
            raise ValueError("employee prompt preview requires employee_v1")
        from src.autonomous.runtime.employee_session import EmployeeSessionBootstrap

        bootstrap = EmployeeSessionBootstrap.from_agent(
            tenant_key="employee",
            agent=agent,
            project_root=self.root_path,
            identity_version=0,
        )
        return bootstrap.wrap_prompt(prompt)

    @property
    def legacy_employee_session_calls(self) -> int:
        """Release metric: actor cutover requires this to remain zero."""

        lock = getattr(self, "_legacy_employee_session_calls_lock", None)
        if lock is None:
            return 0
        with lock:
            return self._legacy_employee_session_calls

    def _record_legacy_employee_session_call(self) -> None:
        lock = getattr(self, "_legacy_employee_session_calls_lock", None)
        if lock is None:
            with _LEGACY_EMPLOYEE_METRIC_INIT_LOCK:
                if not hasattr(self, "_legacy_employee_session_calls_lock"):
                    self._legacy_employee_session_calls = 0
                    self._legacy_employee_session_calls_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
                lock = self._legacy_employee_session_calls_lock
        with lock:
            self._legacy_employee_session_calls += 1

    def open_employee_session(
        self,
        agent: AgentIdentity,
        *,
        env: dict[str, str],
    ) -> _EmployeeSessionLease:
        """Open a reusable employee_v1 session for an EmployeeActor."""

        if agent.security_profile != "employee_v1":
            raise SecurityPolicyDegradedError("reusable lease requires employee_v1")
        if not isinstance(env, dict):
            raise SecurityPolicyDegradedError(
                "employee session environment is not explicitly scoped"
            )
        from src.agent_session.factory import employee_session_environment

        session_cwd = _employee_acp_workspace(agent)
        with employee_session_environment(env):
            session = create_engine_session(
                agent_type=agent.agent_type,
                cwd=session_cwd,
                model_name=agent.model_name or None,
                thread_id=f"employee_actor_{agent.agent_id}",
                auto_approve=True,
                require_tool_filter=True,
            )
        if session is None:
            raise RuntimeError("employee backend session creation failed")
        try:
            self._apply_tool_restrictions(session, agent)
        except Exception:
            close_session_safely(session)
            raise
        with self._lock:
            previous = self._agent_sessions.get(agent.agent_id)
            if previous is not None and previous is not session:
                close_session_safely(session)
                raise RuntimeError("employee already owns an active backend session")
            self._agent_sessions[agent.agent_id] = session
        return _EmployeeSessionLease(self, agent.agent_id, session)

    def run_agent_session_full(
        self,
        agent: AgentIdentity,
        prompt: str,
        *,
        timeout: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[object]:
        """Run an ACP session and return a result object with text and metadata.

        Returns a PromptResult-like object with .text, .stop_reason, and .output_tokens
        attributes for discussion token tracking. Falls back to run_agent_session
        for the actual execution.
        """
        if agent.security_profile == "employee_v1":
            raise SecurityPolicyDegradedError(
                "canonical employee discussion requires the durable Gateway"
            )
        from src.acp.models import PromptResult

        text = self._run_acp_session(agent, prompt, timeout=timeout)
        if text is None:
            return None

        # Estimate tokens from text length (4 chars per token as rough estimate)
        output_tokens = len(text) // 4 if text else 0

        return PromptResult(
            stop_reason="end_turn",
            text=text,
            output_tokens=output_tokens,  # type: ignore[call-arg]
        )

    # ------------------------------------------------------------------
    # Discussion Helpers: multi-thread parallel support
    # ------------------------------------------------------------------

    @staticmethod
    def build_discussion_config_from_settings(settings=None):
        """Build a DiscussionConfig from current settings. Shared by all discussion paths."""
        from .models import DiscussionConfig
        if settings is None:
            settings = get_settings()
        trigger_rules = [
            r.strip() for r in settings.slock_discussion_trigger_rules.split(",")
            if r.strip()
        ]
        return DiscussionConfig(
            max_rounds=settings.slock_max_discussion_rounds,
            token_budget=settings.slock_discussion_token_budget,
            trigger_rules=trigger_rules or ["coder->reviewer"],
            discussion_timeout=settings.slock_discussion_timeout,
        )

    def _enforce_discussion_budget(self, thread, callbacks: Optional[SlockEngineCallbacks]) -> bool:
        """Delegate discussion budget enforcement through DiscussionManager."""
        discussion_manager = getattr(self, "_discussion_manager", None)
        if discussion_manager is None:
            return True
        on_card_send = getattr(callbacks, "on_card_send", None) if callbacks else None
        settings = getattr(self, "_settings", None) or get_settings()
        return discussion_manager.check_budget_with_breaker(
            thread,
            settings,
            on_card_send=on_card_send,
        )

    def _trust_rules_path(self) -> str:
        return os.path.join(self._memory.base_path, "global", "TRUST_RULES.json")

    def _load_trust_rules(self) -> dict[str, dict[str, str]]:
        import json

        path = self._trust_rules_path()
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            logger.warning("Failed to load Slock trust rules from %s", path, exc_info=True)
            return {}
        if not isinstance(data, dict):
            return {}
        rules: dict[str, dict[str, str]] = {}
        for channel_id, channel_rules in data.items():
            if isinstance(channel_rules, dict):
                rules[str(channel_id)] = {
                    str(pair): str(value)
                    for pair, value in channel_rules.items()
                }
        return rules

    def _persist_trust_rules(self) -> None:
        import json

        path = self._trust_rules_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._channel_trust_rules, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)

    def _save_trust_rule(self, channel_id: str, role_pair: str, value: str) -> None:
        """Persist a channel-scoped trust rule for a role pair."""
        if not hasattr(self, "_channel_trust_rules"):
            self._channel_trust_rules = {}
        self._channel_trust_rules.setdefault(channel_id, {})[role_pair] = value
        self._persist_trust_rules()

    def _check_trust_bypass(self, channel_id: str, role_pair: str) -> bool:
        """Return True when a channel role-pair trust rule is currently valid."""
        if not hasattr(self, "_channel_trust_rules"):
            self._channel_trust_rules = {}
        if channel_id not in self._channel_trust_rules:
            loaded = self._load_trust_rules()
            if loaded:
                self._channel_trust_rules.update(loaded)

        value = self._channel_trust_rules.get(channel_id, {}).get(role_pair, "")
        if value == "permanent":
            return True
        if not value:
            return False
        try:
            expires_at = float(value)
        except (TypeError, ValueError):
            return False
        if time.time() < expires_at:
            return True

        channel_rules = self._channel_trust_rules.get(channel_id, {})
        channel_rules.pop(role_pair, None)
        with contextlib.suppress(Exception):
            self._persist_trust_rules()
        return False

    def _add_discussion(self, channel_id: str, thread) -> bool:
        """Add a discussion thread to the channel. Returns False if at capacity."""
        max_parallel = get_settings().slock_max_parallel_discussions
        with self._discussions_lock:
            discussions = self._active_discussions.setdefault(channel_id, [])
            if len(discussions) >= max_parallel:
                logger.warning(
                    "Channel %s at max parallel discussions (%d), rejecting new discussion",
                    channel_id, max_parallel,
                )
                return False
            discussions.append(thread)
        return True

    def _remove_discussion(self, channel_id: str, thread_id: str) -> None:
        """Remove a completed discussion thread from the channel."""
        with self._discussions_lock:
            discussions = self._active_discussions.get(channel_id, [])
            self._active_discussions[channel_id] = [
                t for t in discussions if t.thread_id != thread_id
            ]
            if not self._active_discussions[channel_id]:
                del self._active_discussions[channel_id]

    def find_active_discussion(self, channel_id: str, thread_id: str):
        """Find an active discussion thread by channel and thread id."""
        with self._discussions_lock:
            for thread in self._active_discussions.get(channel_id, []):
                if getattr(thread, "thread_id", "") == thread_id:
                    return thread
        return None

    def run_council(
        self,
        question: str,
        *,
        participants: Optional[list[AgentIdentity]] = None,
        chairman: Optional[AgentIdentity] = None,
        on_stage: Optional[Callable[[object], None]] = None,
        timeout: float = 300.0,
    ):
        """Run a same-question council flow across multiple Slock agents."""
        from .council_manager import CouncilManager

        channel_id = self._channel.channel_id if self._channel else self.chat_id
        selected = participants or list(self._registry.list_agents(channel_id=channel_id))
        return CouncilManager(engine=self).run(
            question,
            participants=list(selected),
            chairman=chairman,
            on_stage=on_stage,
            timeout=timeout,
        )

    def get_agent_status(self, agent_id: str) -> AgentStatus:
        """Get current status of an agent."""
        with self._lock:
            return self._agent_statuses.get(agent_id, AgentStatus.IDLE)

    def set_agent_status(self, agent_id: str, status: AgentStatus) -> None:
        """Update agent status (thread-safe, single source of truth).

        Lock hierarchy: engine._lock is the authoritative lock for agent status.
        Router reads status via engine.get_agent_status() — no dual-write.
        When transitioning to IDLE, notify the task queue so waiting consumers
        can dispatch queued work without polling.
        """
        with self._lock:
            self._agent_statuses[agent_id] = status
        # 设计意图：notify_idle 必须在锁外调用。
        # 原因：notify_idle 内部会获取 TaskQueue._cond（条件变量锁），
        # 若在 engine._lock 内调用，可能形成锁顺序反转导致死锁。
        # 同时保持临界区最小化，避免阻塞队列消费者线程。
        # 后续重构请勿将此调用移入锁内。
        if status == AgentStatus.IDLE:
            self._task_queue.notify_idle()

    def transition_agent(self, agent_id: str, to_status: AgentStatus) -> bool:
        """Transition agent through valid state machine paths.

        Valid transitions:
            IDLE → WAKING → THINKING → RUNNING → CHECKING → SENDING → IDLE

        Note: This method acquires self._lock (RLock) internally.  It is safe
        to call while self._lock is already held by the same thread (re-entrant),
        which is used by try_lock_for_move for atomic CAS semantics.
        """
        current = self.get_agent_status(agent_id)
        if to_status in self.VALID_TRANSITIONS.get(current, ()):
            self.set_agent_status(agent_id, to_status)
            return True

        logger.warning(
            "Invalid agent transition: %s -> %s (agent=%s)",
            current.value, to_status.value, agent_id,
        )
        return False

    def _transition_agent_or_abort(self, agent_id: str, to_status: AgentStatus) -> bool:
        """Transition during execution, forcing IDLE if the state machine rejects it."""
        if self.transition_agent(agent_id, to_status):
            return True
        logger.error(
            "Aborting Slock agent execution after invalid transition to %s (agent=%s)",
            to_status.value,
            agent_id,
        )
        self.set_agent_status(agent_id, AgentStatus.IDLE)
        return False

    def try_lock_for_move(self, agent_id: str) -> bool:
        """Atomically check IDLE and transition to MOVING.

        Returns True if the agent was IDLE and is now marked MOVING,
        preventing task assignment or other transitions during the move.
        Returns False if the agent is not IDLE (move should be rejected).

        Uses transition_agent internally (RLock is re-entrant) to ensure all
        state changes go through the unified state machine validation path.
        """
        with self._lock:
            current = self._agent_statuses.get(agent_id, AgentStatus.IDLE)
            if current != AgentStatus.IDLE:
                return False
            # Re-entrant call: self._lock is already held, transition_agent
            # acquires it again via get_agent_status/set_agent_status (RLock).
            return self.transition_agent(agent_id, AgentStatus.MOVING)

    def unlock_after_move(self, agent_id: str) -> None:
        """Release MOVING state back to IDLE after move completes or fails.

        Should be called in a finally block to prevent MOVING state leakage.
        """
        with self._lock:
            current = self._agent_statuses.get(agent_id)
            if current == AgentStatus.MOVING:
                self._agent_statuses[agent_id] = AgentStatus.IDLE
        # Notify queue consumers that an agent is now available
        self._task_queue.notify_idle()
        # Defensive L1 memory readability check after move
        self._verify_l1_memory_after_move(agent_id)

    def _verify_l1_memory_after_move(self, agent_id: str) -> str:
        """Verify L1 memory is readable after cross-group move (non-blocking).

        Logs ERROR if memory file cannot be read or if role is unexpectedly
        empty for an established agent — provides observability without
        disrupting the move flow.

        Returns a diagnostic string (empty string means all OK).
        """
        try:
            memory = self._memory.read_agent_memory(agent_id)
            path = self._memory.agent_memory_path(agent_id)
            if not memory.role and not memory.key_knowledge and not memory.active_context:
                logger.debug(
                    "L1 memory empty after move (may be expected for new agents) | agent=%s path=%s",
                    agent_id, path,
                )
                return ""
            if not memory.role and (memory.key_knowledge or memory.active_context):
                # Agent has history but role is missing — persona consistency at risk
                diag = (
                    f"L1 role section empty but agent has history | agent={agent_id} "
                    f"has_knowledge={bool(memory.key_knowledge)} has_context={bool(memory.active_context)} "
                    f"path={path}"
                )
                logger.error(diag)
                return diag
            logger.debug(
                "L1 memory verified after move | agent=%s has_role=%s has_knowledge=%s has_context=%s",
                agent_id, bool(memory.role), bool(memory.key_knowledge), bool(memory.active_context),
            )
            return ""
        except Exception as exc:
            from src.utils.errors import redact_sensitive
            diag = f"L1 memory read FAILED after move — persona consistency at risk | agent={agent_id} error={redact_sensitive(str(exc))}"
            logger.error(diag, exc_info=True)
            return diag

    # ------------------------------------------------------------------
    # Agent Cancellation
    # ------------------------------------------------------------------

    def _get_cancel_event(self, agent_id: str) -> threading.Event:
        """Get or create a cancellation event for an agent."""
        with self._cancel_events_lock:
            if agent_id not in self._cancel_events:
                self._cancel_events[agent_id] = threading.Event()
            return self._cancel_events[agent_id]

    def _clear_cancel_event(self, agent_id: str) -> None:
        """Remove the cancellation event after execution completes."""
        with self._cancel_events_lock:
            self._cancel_events.pop(agent_id, None)

    def cancel_agent(self, agent_id: str) -> bool:
        """Cancel a running agent by setting its cancellation event.

        Returns True if the agent had an active cancel event to set.
        """
        with self._cancel_events_lock:
            event = self._cancel_events.get(agent_id)
            if event is not None:
                event.set()
                logger.info("Cancellation requested for agent %s", agent_id)
                return True
        logger.debug("No active cancel event for agent %s", agent_id)
        return False

    # ------------------------------------------------------------------
    # Engine Lifecycle
    # ------------------------------------------------------------------

    @activation_serialized
    def activate_channel(self, channel: SlockChannel) -> None:
        """Activate slock mode for a channel.

        Creates memory directories and a workspace directory with a marker file.
        Starts the dispatch loop for event-driven task consumption.

        .. note::
            If default role bootstrap is needed, call prepare_bootstrap()
            BEFORE this method to ensure the dispatch loop waits for
            agent registration. Call finish_bootstrap() after roles
            are registered to unblock the dispatch loop.
        """
        self._channel = channel
        self._memory.ensure_directories(channel_id=channel.channel_id)
        self._memory.initialize_team_workspace(channel, project_path=self.root_path)

        # Lazily initialize relationship graph now that channel is known
        if self._relationship_graph is None:
            from .relationship_graph import RelationshipGraph
            graph_path = os.path.join(
                self._memory.base_path, "groups", channel.channel_id, "relationships.json"
            )
            self._relationship_graph = RelationshipGraph(graph_path)
        persisted_tasks = self._memory.read_task_board(channel.channel_id)
        if persisted_tasks:
            self._tasks.clear()
            self._tasks.extend(persisted_tasks)

        # Crash recovery: downgrade orphan IN_PROGRESS/IN_REVIEW tasks to TODO
        recovered = self._task_mgr.recover_orphan_tasks()
        if recovered:
            logger.info(
                "Channel %s: recovered %d orphan tasks on activation",
                channel.channel_id, len(recovered),
            )

        # Restore collaboration plans from disk
        self._restore_plans()

        # Start idle scan for auto-claiming orphan TODO tasks
        self._task_mgr.start_idle_scan()

        # Start dispatch loop for event-driven task consumption
        self.start_dispatch_loop()

        # Start proactive patrol loop for SLA checks and idle agent auto-claim
        self.start_patrol_loop()

        marker_data = {
            "channel_id": channel.channel_id,
            "team_name": channel.team_name,
            "name": channel.name,
            "owner_id": channel.owner_id,
            "root_path": self.root_path,
            "bootstrap_failed": channel.bootstrap_failed,
            "activated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        # Canonical app-level group marker under ~/.ghostap/slock/groups.
        canonical_dir = self._memory.get_group_base_path(channel.channel_id)
        os.makedirs(canonical_dir, exist_ok=True)
        canonical_marker = os.path.join(canonical_dir, ".slock_channel.json")
        self._write_channel_marker(canonical_marker, marker_data)

    @staticmethod
    def _write_channel_marker(marker_path: str, marker_data: dict) -> None:
        """Write or merge a channel marker atomically.

        If the marker file already exists, read the existing JSON and merge:
        - Only fill in fields that are missing or currently empty/None.
        - Never overwrite ``activated_at`` (preserve first activation time).
        Then write atomically via tmp + os.replace.
        """
        import json as _json

        existing: dict = {}
        if os.path.exists(marker_path):
            try:
                with open(marker_path, "r", encoding="utf-8") as f:
                    existing = _json.load(f)
            except (OSError, ValueError):
                existing = {}

        # Merge: for each key in marker_data, update only if the existing
        # value is missing/empty and the new value is non-empty.
        merged = dict(existing)
        for key, new_val in marker_data.items():
            # Never overwrite activated_at — preserve first activation time.
            if key == "activated_at" and key in merged and merged[key]:
                continue
            old_val = merged.get(key)
            if not old_val and new_val:
                merged[key] = new_val

        marker_parent = os.path.dirname(marker_path) or "."
        tmp_path = os.path.join(
            marker_parent,
            f".{os.path.basename(marker_path)}.{secrets.token_hex(12)}.tmp",
        )
        try:
            with open(tmp_path, "x", encoding="utf-8") as f:
                _json.dump(merged, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, marker_path)
            directory_fd = os.open(
                marker_parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass

    def execute(
        self,
        message: str,
        callbacks: Optional[SlockEngineCallbacks] = None,
        *,
        sender_id: str = "",
    ) -> Optional[str]:
        """Process an incoming message through the slock engine.

        Routes the message to the appropriate agent, wakes it,
        executes via ACP session, and returns formatted output.
        """
        with self._lock:
            if self._run_state == EngineRunState.STOPPING:
                return None
            self._run_state = EngineRunState.RUNNING

        try:
            # Get available agents for this channel
            channel_id = self._channel.channel_id if self._channel else self.chat_id
            agents = self._registry.list_agents(channel_id=channel_id)

            if not agents:
                with self._lock:
                    self._run_state = EngineRunState.IDLE
                return None

            # Route message to agent with fallback semantics
            from .task_router import RoutingStatus
            candidates = self._apply_wake_policy(message, agents)
            routing_result = self._router.route_message_with_fallback(message, candidates)

            if routing_result.status == RoutingStatus.ASSIGNED and routing_result.agent:
                target_agent = routing_result.agent
                if callbacks and callbacks.on_message_routed:
                    callbacks.on_message_routed(message, target_agent)

                # Execute agent lifecycle
                result = self._execute_agent(target_agent, message, callbacks)
                return result

            elif routing_result.status == RoutingStatus.QUEUE_WAIT:
                # All agents busy — enqueue to task queue for deferred execution
                logger.info(
                    "execute: all agents busy (busy_count=%d), enqueueing message",
                    routing_result.busy_count,
                )
                task = SlockTask(text=message, callbacks=callbacks)
                self._task_queue.enqueue(task)
                return None

            else:
                # NO_MATCH — no suitable agent found
                return None

        except Exception as e:
            from src.utils.errors import redact_sensitive
            error_msg = f"Slock engine error: {redact_sensitive(str(e))}"
            logger.error(error_msg, exc_info=True)
            if callbacks and callbacks.on_error:
                callbacks.on_error(error_msg)
            return None
        finally:
            with self._lock:
                if self._run_state == EngineRunState.RUNNING:
                    self._run_state = EngineRunState.IDLE

    def _execute_agent(
        self,
        agent: AgentIdentity,
        message: str,
        callbacks: Optional[SlockEngineCallbacks],
        *,
        freshness_check: bool = True,
    ) -> Optional[str]:
        """Execute a single agent's response cycle.

        IDLE → WAKING → THINKING → RUNNING → CHECKING → SENDING → IDLE
        """
        if agent.security_profile == "employee_v1":
            raise SecurityPolicyDegradedError(
                "canonical employee execution requires the durable Gateway"
            )
        agent_id = agent.agent_id
        channel_id = self._channel.channel_id if self._channel else self.chat_id
        progress_tracker = getattr(self, "_progress_tracker", None)

        # Set up cancellation event and watchdog timer
        cancel_event = self._get_cancel_event(agent_id)
        settings = get_settings()
        watchdog = threading.Timer(
            settings.slock_agent_execution_timeout,
            cancel_event.set,
        )
        watchdog.daemon = True
        watchdog.start()

        try:
            # IDLE → WAKING (early-return if agent is not IDLE)
            if not self.transition_agent(agent_id, AgentStatus.WAKING):
                logger.info("Agent %s busy, skipping execution", agent_id)
                return None

            self._memory.append_message_archive(
                channel_id,
                sender_type="user",
                content=message,
                agent_id=agent_id,
                agent_name=agent.name,
                metadata={"routed_to": agent_id},
            )
            freshness_baseline_ts = time.time()

            if callbacks and callbacks.on_agent_wake:
                callbacks.on_agent_wake(agent)

            # Check cancellation
            if cancel_event.is_set():
                raise AgentCancellationError(f"Agent {agent_id} cancelled")

            # Load agent memory context
            memory = self._memory.read_agent_memory(agent_id)

            # WAKING → THINKING
            if not self._transition_agent_or_abort(agent_id, AgentStatus.THINKING):
                return None
            if callbacks and callbacks.on_agent_thinking:
                callbacks.on_agent_thinking(agent)
            if progress_tracker:
                progress_tracker.update(
                    agent_id, entity_type="agent", progress_pct=20,
                    status="thinking", detail=f"{agent.name} 构思中",
                )

            # Build prompt with memory context and system prompt
            prompt = self._build_agent_prompt(agent, message, memory)

            # Check cancellation
            if cancel_event.is_set():
                raise AgentCancellationError(f"Agent {agent_id} cancelled")

            # THINKING → RUNNING
            if not self._transition_agent_or_abort(agent_id, AgentStatus.RUNNING):
                return None
            if callbacks and callbacks.on_agent_running:
                callbacks.on_agent_running(agent, message)
            if progress_tracker:
                progress_tracker.update(
                    agent_id, entity_type="agent", progress_pct=50,
                    status="running", detail=f"{agent.name} 执行中",
                )

            # Execute via ACP session
            try:
                result = self._run_acp_session(agent, prompt)
            except StopIteration:
                # This exception is raised when a mock side_effect is exhausted
                logger.warning("Mock side_effect exhausted for agent %s", agent.name)
                return None
            except Exception as exc:
                logger.exception("Unexpected error in ACP session for agent %s", agent.name)
                self.escalate(
                    agent,
                    f"Agent execution failed: {exc}",
                    level=EscalationLevel.BLOCKED,
                    context=message[:1000],
                    callbacks=callbacks,
                )
                return None
            if result is None:
                execution_errors = self._get_agent_execution_errors()
                error_detail = execution_errors.pop(agent_id, "")
                if error_detail:
                    self.escalate(
                        agent,
                        f"Agent execution failed: {error_detail}",
                        level=EscalationLevel.BLOCKED,
                        context=message[:1000],
                        callbacks=callbacks,
                    )
                    return None

            # Check cancellation after session
            if cancel_event.is_set():
                raise AgentCancellationError(f"Agent {agent_id} cancelled")

            # Autonomous resolution: if agent output shows uncertainty, attempt
            # autonomous resolution before delivering to user (AC-5)
            autonomous_resolver = getattr(self, "_autonomous_resolver", None)
            if result and autonomous_resolver and autonomous_resolver.has_ambiguity_markers(result):
                task_id = f"message:{channel_id}:{agent_id}"
                if autonomous_resolver.can_ask_question(task_id):
                    try:
                        _loop = _get_shared_loop()
                        _future = _asyncio.run_coroutine_threadsafe(
                            autonomous_resolver.attempt_resolve(
                                task_text=message,
                                context=result,
                                memory=self._memory,
                                task_id=task_id,
                                channel_id=channel_id,
                            ),
                            _loop,
                        )
                        resolve_result = _future.result(timeout=15)

                        from .autonomous_resolver import ResolveStatus
                        if resolve_result.status == ResolveStatus.RESOLVED:
                            # Use resolved text as augmented context, annotate assumptions
                            assumptions_note = ""
                            if resolve_result.assumptions:
                                assumptions_note = (
                                    "\n\n---\n⚠️ **基于以下假设完成:**\n"
                                    + "\n".join(f"- {a}" for a in resolve_result.assumptions)
                                )
                            result = result + assumptions_note
                        elif resolve_result.status in (
                            ResolveStatus.NEEDS_CLARIFICATION,
                            ResolveStatus.TIMEOUT,
                        ):
                            # Format structured question and include in output
                            question = autonomous_resolver.format_structured_question(
                                attempts_summary=resolve_result.reasoning_trace,
                                blocker="需要更多信息才能继续",
                                candidates=["请提供具体目标", "请描述期望结果", "请指定范围"],
                            )
                            autonomous_resolver.record_question_asked(task_id)
                            result = result + "\n\n---\n" + question
                    except Exception as e:
                        logger.warning("Autonomous resolution failed, continuing with original result: %s", e, exc_info=True)

            # RUNNING → CHECKING
            if not self._transition_agent_or_abort(agent_id, AgentStatus.CHECKING):
                return None
            if progress_tracker:
                progress_tracker.update(
                    agent_id, entity_type="agent", progress_pct=75,
                    status="checking", detail=f"{agent.name} 检查中",
                )

            # --- Freshness Gate (P0) ---
            # Before sending, check if new messages arrived since execution started.
            # If so, hold the draft and let agent re-evaluate (bounded retries).
            settings_obj = get_settings()
            if result and freshness_check and settings_obj.slock_freshness_gate_enabled:
                result = self._freshness_gate_check(
                    agent=agent,
                    original_message=message,
                    draft_result=result,
                    execution_start_ts=freshness_baseline_ts,
                    channel_id=channel_id,
                    max_retries=settings_obj.slock_freshness_max_retries,
                    reeval_timeout=settings_obj.slock_freshness_reeval_timeout,
                )

            # Sidebar marker parsing: extract lightweight inter-agent messages
            if result:
                from .sidebar_channel import SidebarChannel, SidebarMessage, SidebarMsgType
                cleaned_output, sidebar_markers = SidebarChannel.parse_output_markers(result)
                for msg_type_str, recipient_name, content in sidebar_markers:
                    recipient = self._registry.find_by_name(recipient_name)
                    if recipient:
                        msg_type = SidebarMsgType(msg_type_str.lower())
                        sidebar_msg = SidebarMessage(
                            sender_id=agent_id,
                            sender_name=agent.name,
                            recipient_id=recipient.agent_id,
                            msg_type=msg_type,
                            content=content,
                        )
                        self._sidebar_channel.post(sidebar_msg)
                if sidebar_markers:
                    result = cleaned_output

            # CHECKING → SENDING
            if not self._transition_agent_or_abort(agent_id, AgentStatus.SENDING):
                return None
            if progress_tracker:
                progress_tracker.update(
                    agent_id, entity_type="agent", progress_pct=90,
                    status="sending", detail=f"{agent.name} 发送中",
                )

            # Format output through mouthpiece
            if result:
                # Agent-driven planning: if the agent outputs a [PLAN] block,
                # parse it into sub-tasks and register with the collaboration orchestrator.
                if result.strip().startswith("[PLAN]"):
                    plan_created = self._handle_agent_plan_output(agent, result, message, channel_id, callbacks)
                    if plan_created:
                        # Plan was registered — the orchestrator will dispatch sub-tasks.
                        # Replace result with a plan acknowledgment for display.
                        result = self._format_plan_acknowledgment(result)

                # Agent-driven fan-out: if the agent determines this task should
                # involve all team members, it prefixes its response with [DELEGATE:ALL].
                # We strip the marker, execute the agent's own response, then re-dispatch
                # the original message to all other agents.
                if result.strip().startswith("[DELEGATE:ALL]"):
                    result = result.strip().removeprefix("[DELEGATE:ALL]").strip()
                    self._fan_out_to_others(agent, message, channel_id, callbacks)

                formatted = self._mouthpiece.format_text(agent, result)
                self._memory.append_message_archive(
                    channel_id,
                    sender_type="agent",
                    content=result,
                    agent_id=agent_id,
                    agent_name=agent.name,
                    metadata={"formatted": formatted},
                )
            else:
                formatted = None

            if callbacks and callbacks.on_agent_done:
                callbacks.on_agent_done(agent, result or "")

            # Persist execution record to agent's local history log
            try:
                self._memory.append_execution_record(
                    agent_id,
                    task_id=task_id or "",
                    chat_id=getattr(self._channel, "chat_id", ""),
                    role="assistant",
                    content=result or "",
                    tool_name=agent.agent_type,
                    model_name=agent.model_name,
                    success=True,
                )
            except Exception:
                logger.debug("failed to persist execution record for %s", agent_id, exc_info=True)

            # Mark progress complete and remove tracking
            if progress_tracker:
                progress_tracker.update(
                    agent_id, entity_type="agent", progress_pct=100,
                    status="done", detail=f"{agent.name} 完成",
                )
                progress_tracker.force_push(agent_id)
                progress_tracker.remove(agent_id)

            # SENDING → IDLE
            if not self._transition_agent_or_abort(agent_id, AgentStatus.IDLE):
                return None

            # Update agent memory with new context
            if result:
                context_entry = f"[{time.strftime('%Y-%m-%d %H:%M')}] Responded to: {message[:100]}"
                self._memory.update_agent_context(agent_id, context_entry)
                skill_tags = self._router.extract_skill_keywords(message)
                profiles = self._memory.record_skill_feedback(agent_id, skill_tags, quality_score=100.0)
                self._router.set_skill_profiles(agent_id, profiles)
                self._record_observer_learning(agent, message, skill_tags)

                # Behavior self-convergence tracking (Insight #5)
                self._check_behavior_convergence(agent, message, success=True)

                # Discussion hook: trigger inter-agent discussion if enabled
                self._maybe_trigger_discussion(agent, result, channel_id, callbacks)

                # @mention routing: notify mentioned agents
                self._route_at_mentions(result, agent_id)

                # Memory enhancement: trigger context summarization if threshold exceeded
                settings_obj = get_settings()
                self._memory.summarize_context(
                    agent_id, threshold=settings_obj.slock_memory_summarize_threshold
                )

                # Role evolution: periodically update memory.role from accumulated skills
                evolution_threshold = getattr(settings_obj, 'slock_role_evolution_threshold', 3)
                self._memory.evolve_agent_role(agent_id, agent, task_threshold=evolution_threshold)

            return formatted

        except AgentCancellationError:
            logger.warning("Agent %s execution cancelled", agent_id)
            self.set_agent_status(agent_id, AgentStatus.IDLE)
            # Close any active session
            with self._lock:
                session = self._agent_sessions.pop(agent_id, None)
            if session:
                close_session_safely(session)
            return None
        except Exception as exc:
            from src.utils.redact import redact_sensitive

            logger.exception(
                "Agent %s execution failed: %s",
                agent_id,
                redact_sensitive(str(exc)),
            )
            self.set_agent_status(agent_id, AgentStatus.IDLE)
            with self._lock:
                session = self._agent_sessions.pop(agent_id, None)
            if session:
                close_session_safely(session)
            if callbacks and callbacks.on_agent_error:
                callbacks.on_agent_error(agent, redact_sensitive(str(exc)))
            # Behavior self-convergence: record failure (Insight #5)
            self._check_behavior_convergence(agent, message, success=False)
            return None
        finally:
            watchdog.cancel()
            self._clear_cancel_event(agent_id)
            # Clean up autonomous resolver state for this task
            autonomous_resolver = getattr(self, "_autonomous_resolver", None)
            if autonomous_resolver:
                task_id = f"message:{channel_id}:{agent_id}"
                with contextlib.suppress(Exception):  # intentional: cleanup path
                    autonomous_resolver.cleanup_task(task_id)
                    autonomous_resolver.cleanup_stale()

    # ------------------------------------------------------------------
    # Freshness Gate + Draft Save (Slock Collaboration Insight #1 & #2)
    # ------------------------------------------------------------------

    def _freshness_gate_check(
        self,
        agent: AgentIdentity,
        original_message: str,
        draft_result: str,
        execution_start_ts: float,
        channel_id: str,
        max_retries: int = 2,
        reeval_timeout: float = 15.0,
    ) -> str:
        """Check freshness before sending and optionally re-evaluate the draft.

        If new messages arrived in the channel since the agent started executing,
        the draft is held and the agent is asked to re-evaluate with the new context.
        After max_retries, the draft is force-sent with a stale-context annotation.

        Returns the final result text to send.
        """
        for attempt in range(max_retries):
            new_msg_count = self._memory.count_messages_since(
                channel_id, execution_start_ts, exclude_agent_id=agent.agent_id
            )
            if new_msg_count == 0:
                return draft_result

            logger.info(
                "Freshness Gate: %d new message(s) since agent %s started (attempt %d/%d)",
                new_msg_count, agent.name, attempt + 1, max_retries,
            )

            new_messages = self._memory.get_messages_since(
                channel_id, execution_start_ts, exclude_agent_id=agent.agent_id, limit=3
            )
            new_context_lines = []
            for msg in new_messages:
                sender = msg.get("agent_name") or msg.get("sender_type", "user")
                content = msg.get("content", "")[:300]
                new_context_lines.append(f"[{sender}]: {content}")
            new_context_summary = "\n".join(new_context_lines)

            reeval_prompt = (
                f"你之前对以下消息生成了一个回复草稿，但在你生成回复期间，"
                f"群里又出现了新消息。请判断你的草稿是否仍然有效。\n\n"
                f"## 原始消息\n{original_message[:500]}\n\n"
                f"## 你的草稿回复\n{draft_result[:1000]}\n\n"
                f"## 新到达的消息\n{new_context_summary}\n\n"
                f"## 指令\n"
                f"如果你的草稿仍然有效且不需要修改，请原样输出草稿内容。\n"
                f"如果需要根据新上下文修改回复，请输出修改后的完整回复。\n"
                f"不要解释你的决策，直接输出最终回复。"
            )

            try:
                revised = self._run_acp_session(agent, reeval_prompt, timeout=reeval_timeout)
                if revised and revised.strip():
                    draft_result = revised.strip()
                    execution_start_ts = time.time()
            except Exception as exc:
                logger.warning(
                    "Freshness reeval failed for agent %s: %s, using original draft",
                    agent.name, exc,
                )
                break

        # Final check: if still stale after max retries, annotate
        final_new_count = self._memory.count_messages_since(
            channel_id, execution_start_ts, exclude_agent_id=agent.agent_id
        )
        if final_new_count > 0:
            draft_result += "\n\n---\n⚠️ *此回复基于较早上下文生成，发送期间群内有新消息到达。*"

        return draft_result

    def _record_observer_learning(
        self,
        actor: AgentIdentity,
        message: str,
        skill_tags: list[str],
    ) -> None:
        """Let idle team members learn potential skills from successful work."""
        channel_id = self._channel.channel_id if self._channel else self.chat_id
        for observer in self._registry.list_agents(channel_id=channel_id):
            if observer.agent_id == actor.agent_id:
                continue
            if self.get_agent_status(observer.agent_id) != AgentStatus.IDLE:
                continue
            self._observer_queue.enqueue(
                observer_id=observer.agent_id,
                actor_id=actor.agent_id,
                message=message,
                skill_tags=skill_tags,
            )

    def _check_behavior_convergence(
        self,
        agent: AgentIdentity,
        message: str,
        success: bool,
    ) -> None:
        """Track task outcomes and trigger avoidance strategy on repeated failures.

        Implements Slock Collaboration Insight #5: Behavior Self-Convergence.
        When an agent fails the same skill_tag N consecutive times, it writes an
        avoidance note to L1 memory and broadcasts that information to observers.
        """
        settings = get_settings()
        if not settings.slock_behavior_convergence_enabled:
            return

        skill_tags = self._router.extract_skill_keywords(message)
        if not skill_tags:
            return

        threshold = settings.slock_behavior_convergence_threshold
        channel_id = self._channel.channel_id if self._channel else self.chat_id

        for tag in skill_tags:
            self._memory.record_task_outcome(agent.agent_id, tag, success)

            if not success:
                consecutive = self._memory.get_consecutive_failures(agent.agent_id, tag)
                if consecutive >= threshold:
                    reason = (
                        f"Agent {agent.name} failed skill '{tag}' "
                        f"{consecutive} consecutive times"
                    )
                    self._memory.write_avoidance_strategy(agent.agent_id, tag, reason)
                    logger.info(
                        "Behavior convergence: %s avoids skill '%s' after %d failures",
                        agent.name, tag, consecutive,
                    )
                    # Notify idle observers so they learn about this avoidance
                    for observer in self._registry.list_agents(channel_id=channel_id):
                        if observer.agent_id == agent.agent_id:
                            continue
                        if self.get_agent_status(observer.agent_id) != AgentStatus.IDLE:
                            continue
                        self._observer_queue.enqueue(
                            observer_id=observer.agent_id,
                            actor_id=agent.agent_id,
                            message=f"[avoidance] {agent.name} avoids skill '{tag}': {reason}",
                            skill_tags=[tag],
                        )

    def _maybe_trigger_discussion(
        self,
        agent: AgentIdentity,
        result: str,
        channel_id: str,
        callbacks: Optional[SlockEngineCallbacks],
    ) -> None:
        """Post-execution hook: check if inter-agent discussion should be triggered.

        Only runs when slock_discussion_enabled is True. Uses DiscussionManager
        to evaluate trigger conditions and run the discussion loop if triggered.
        Sends discussion status cards for user visibility.
        """
        settings = get_settings()
        if not settings.slock_discussion_enabled:
            return

        # Skip discussion trigger for trivial/short responses
        if len(result.strip()) < 50:
            return

        # Skip if result is just an acknowledgment or status update
        _TRIVIAL_PATTERNS = (
            "已完成", "done", "ok", "好的", "收到", "✅", "已执行",
            "no issues", "没有问题", "通过", "passed",
        )
        result_lower = result.strip().lower()
        if any(result_lower.startswith(p.lower()) for p in _TRIVIAL_PATTERNS):
            return

        from .discussion_manager import DiscussionManager

        # Lazy-init discussion manager using shared config builder
        if self._discussion_manager is None:
            config = self.build_discussion_config_from_settings(settings)
            self._discussion_manager = DiscussionManager(
                engine=self, memory_manager=self._memory, config=config,
            )

        dm: DiscussionManager = self._discussion_manager  # type: ignore[assignment]
        thread = dm.should_trigger_discussion(agent, result, channel_id=channel_id)
        if thread is None:
            return

        # Check parallel discussion capacity
        if not self._add_discussion(channel_id, thread):
            logger.info("Discussion rejected: channel %s at max capacity", channel_id)
            return

        logger.info(
            "Discussion triggered | agent=%s channel=%s reason=%s participants=%s",
            agent.agent_id, channel_id, thread.trigger_reason, thread.participants,
        )

        # Set agent state to DISCUSSING (via state machine validation)
        if not self.transition_agent(agent.agent_id, AgentStatus.DISCUSSING):
            logger.info(
                "Discussion skipped: agent %s not in IDLE state, cannot transition to DISCUSSING",
                agent.agent_id,
            )
            self._remove_discussion(channel_id, thread.thread_id)
            return

        # Send "discussion in progress" card to user (only when live card enabled)
        from .card_templates import build_discussion_card_from_thread, build_discussion_summary_card_from_thread
        broadcast_rounds = getattr(settings, 'slock_discussion_broadcast_rounds', True)
        show_live_card = getattr(settings, 'slock_discussion_live_card', False)
        discussion_card_msg_id = None
        if show_live_card and callbacks and callbacks.on_card_send:
            try:
                card = build_discussion_card_from_thread(thread)
                discussion_card_msg_id = callbacks.on_card_send(card)
            except Exception as card_exc:
                logger.warning("Failed to send discussion start card: %s", card_exc, exc_info=True)

        # Define the discussion runner to be submitted to the executor
        def _run_discussion():
            watchdog_timeout = settings.slock_discussion_timeout
            watchdog_fired = threading.Event()

            def _watchdog_trigger():
                watchdog_fired.set()
                thread.status = DiscussionStatus.TIMEOUT
                logger.warning(
                    "Discussion watchdog fired: %s (timeout=%ds)",
                    thread.thread_id, watchdog_timeout,
                )

            watchdog = threading.Timer(watchdog_timeout, _watchdog_trigger)
            watchdog.daemon = True
            watchdog.start()

            try:
                # Set cancellation event on thread for cooperative cancellation
                thread.cancellation_event = watchdog_fired

                def on_round_complete(updated_thread):
                    """Broadcast per-round agent identity card and optionally update live card."""
                    if watchdog_fired.is_set():
                        return
                    # Per-round broadcast: send independent identity card for the respondent
                    if broadcast_rounds and updated_thread.messages and callbacks and callbacks.on_card_send:
                        last_msg = updated_thread.messages[-1]
                        respondent = self.registry.get(last_msg.sender_agent_id)
                        if respondent:
                            try:
                                round_card = self._mouthpiece.format_card(
                                    respondent,
                                    last_msg.content,
                                    channel_id=channel_id,
                                )
                                callbacks.on_card_send(round_card)
                            except Exception as bcast_exc:
                                logger.warning("Failed to broadcast round card: %s", bcast_exc, exc_info=True)
                    # Optional: update live card
                    if show_live_card and discussion_card_msg_id and callbacks and callbacks.on_card_update:
                        try:
                            card = build_discussion_card_from_thread(updated_thread)
                            callbacks.on_card_update(discussion_card_msg_id, card)
                        except Exception as upd_exc:
                            logger.warning("Failed to update discussion card: %s", upd_exc, exc_info=True)

                completed_thread = dm.run_discussion(
                    thread, result, on_round_complete=on_round_complete
                )

                # Persist conclusion to L2 shared memory
                if completed_thread.conclusion:
                    self._memory.append_discussion_conclusion(
                        channel_id, completed_thread.conclusion
                    )
                    # Sync conclusion to L1 active_context of all participants
                    for participant_id in completed_thread.participants:
                        conclusion_entry = (
                            f"[{time.strftime('%Y-%m-%d %H:%M')}] "
                            f"Discussion conclusion ({completed_thread.trigger_reason}): "
                            f"{completed_thread.conclusion[:500]}"
                        )
                        self._memory.update_agent_context(participant_id, conclusion_entry)

                # Send discussion summary card
                if callbacks and callbacks.on_card_send:
                    try:
                        summary_card = build_discussion_summary_card_from_thread(completed_thread)
                        callbacks.on_card_send(summary_card)
                    except Exception as card_exc:
                        logger.warning("Failed to send discussion summary card: %s", card_exc, exc_info=True)

                if callbacks and callbacks.on_agent_done:
                    callbacks.on_agent_done(
                        agent,
                        f"[Discussion {completed_thread.status.value}] {completed_thread.conclusion[:200]}",
                    )
            except Exception as exc:
                from src.utils.errors import redact_sensitive
                logger.error("Discussion failed: %s", redact_sensitive(str(exc)), exc_info=True)
            finally:
                watchdog.cancel()
                self.set_agent_status(agent.agent_id, AgentStatus.IDLE)
                self._remove_discussion(channel_id, thread.thread_id)

        # Submit to discussion executor (non-blocking)
        try:
            self._discussion_executor.submit(_run_discussion)
        except Exception as exc:
            from src.utils.errors import redact_sensitive
            logger.warning("Failed to submit discussion to executor: %s", redact_sensitive(str(exc)), exc_info=True)
            self.set_agent_status(agent.agent_id, AgentStatus.IDLE)
            self._remove_discussion(channel_id, thread.thread_id)

    def _route_at_mentions(self, content: str, source_agent_id: str) -> list[str]:
        """Extract @mentions from content and route to mentioned agents.

        Returns list of agent_ids that were successfully notified.
        """
        mention_pattern = re.compile(r"@([A-Za-z0-9_.:\-\u4e00-\u9fff]+)", re.UNICODE)
        mentions = mention_pattern.findall(content)
        if not mentions:
            return []

        routed: list[str] = []
        channel_id = self._channel.channel_id if self._channel else None
        for token in mentions:
            agent = self._registry.find_by_at_token(token, channel_id=channel_id)
            if agent and agent.agent_id != source_agent_id:
                # Update the mentioned agent's context with the mention
                mention_context = f"[@mention from {source_agent_id}] {content[:200]}"
                self._memory.update_agent_context(agent.agent_id, mention_context)
                routed.append(agent.agent_id)
                logger.info(
                    "@mention routed: %s mentioned %s",
                    source_agent_id, agent.agent_id,
                )
        return routed

    @staticmethod
    def _sanitize_roster_field(value: str, *, max_len: int = 80) -> str:
        """Keep user-authored roster metadata as a bounded single-line data field."""
        cleaned = re.sub(r"\s+", " ", value or "").strip()
        cleaned = cleaned.replace("`", "'")
        cleaned = re.sub(r"^[#>\-*+\s]+", "", cleaned)
        cleaned = re.sub(r"\s+[#>]+\s*", " ", cleaned)
        if len(cleaned) > max_len:
            cleaned = f"{cleaned[: max_len - 1].rstrip()}…"
        return cleaned

    def _render_team_roster(self, current: AgentIdentity) -> str:
        """Render a teammate roster section for inclusion in the agent prompt.

        The roster lists every other agent in the same channel using *only*
        fields the user supplied via the registry (name, role, personality
        traits). The function does not embed any built-in role taxonomy —
        an empty registry yields an empty string, and a renamed role is
        reflected verbatim. Returns "" when the feature is disabled, the
        cap is 0, or no other teammates exist.
        """
        settings = get_settings()
        if not getattr(settings, "slock_inject_team_roster", True):
            return ""
        cap = int(getattr(settings, "slock_team_roster_max_entries", 20) or 0)
        if cap <= 0:
            return ""
        channel_id = self._channel.channel_id if self._channel else self.chat_id
        try:
            peers = self._registry.list_agents(channel_id=channel_id)
        except Exception:
            logger.debug("Team roster: list_agents failed", exc_info=True)
            return ""
        # Drop self; preserve registry order so the user controls listing.
        peers = [p for p in peers if p.agent_id != current.agent_id]
        if not peers:
            return ""
        peers = peers[:cap]
        lines: list[str] = []
        for peer in peers:
            name = self._sanitize_roster_field(peer.name or peer.agent_id, max_len=48)
            if not name:
                name = peer.agent_id
            extras: list[str] = []
            role = self._sanitize_roster_field(peer.role or "", max_len=48)
            if role and role.lower() != "custom":
                extras.append(role)
            traits = [
                self._sanitize_roster_field(t, max_len=48)
                for t in (peer.personality_traits or [])
                if t and t.strip()
            ]
            traits = [t for t in traits if t]
            if traits:
                extras.append(", ".join(traits))
            suffix = f" — {' · '.join(extras)}" if extras else ""
            # Append relationship context hint if available
            rel_hint = ""
            if hasattr(self, "_relationship_graph") and self._relationship_graph:
                rel_hint = self._relationship_graph.get_interaction_context(
                    current.agent_id, peer.agent_id
                )
            if rel_hint:
                suffix += f" {rel_hint}"
            lines.append(f"- @{name}{suffix}")
        header = (
            "\n# Teammates in This Channel\n"
            "Other agents you can address with `@<name>` in this channel. "
            "Names, roles and traits below are user-defined via the agent registry.\n"
        )
        return header + "\n".join(lines)

    def _render_collaboration_context(self, agent: AgentIdentity, channel_id: str) -> str:
        """Render a dynamic collaboration context block for injection into agent prompts.

        Contains: task board summary, active collaboration plans, and behavioral rules.
        Returns "" if roster injection is disabled (isolation mode) or no meaningful content.
        """
        settings = get_settings()
        if not getattr(settings, "slock_inject_team_roster", True):
            return ""

        sections: list[str] = []

        # 1. Task Board Summary
        tasks = list(self._tasks)
        if tasks:
            from .models import TaskStatus
            todo_count = sum(1 for t in tasks if t.status == TaskStatus.TODO)
            in_progress_count = sum(1 for t in tasks if t.status == TaskStatus.IN_PROGRESS)
            done_count = sum(1 for t in tasks if t.status == TaskStatus.DONE)
            sections.append(f"任务看板: {todo_count} 待办 / {in_progress_count} 进行中 / {done_count} 已完成")

        # 2. Active Collaboration Plans
        try:
            active_plans = self._collaboration_orchestrator.list_active_plans(channel_id)
            if active_plans:
                plan_lines: list[str] = []
                for plan in active_plans:
                    template_label = plan.chain_template or "custom"
                    status_map = {
                        "executing": "执行中",
                        "pending_approval": "待确认",
                        "paused": "已暂停",
                    }
                    status_label = status_map.get(plan.status.value, plan.status.value)
                    plan_lines.append(f"协作计划: {template_label} ({status_label})")
                sections.extend(plan_lines)
        except Exception:
            pass  # Non-critical; skip if orchestrator unavailable

        # 3. Collaboration Rules
        rules = [
            "执行任务前先通过 Task Claim 声明所有权",
            "完成后在回复中 @ 下一位同事",
            "不确定是否该你处理时，先观察再行动",
            "遇到阻塞及时 escalate，不要无限重试",
        ]
        sections.append("协作规则:\n" + "\n".join(f"- {r}" for r in rules))

        # 4. Lightweight sidebar communication instructions
        sections.append(
            "# 轻量通讯\n"
            "你可以用以下前缀向队友发送非正式消息（不触发正式讨论）：\n"
            "- [FYI:@队友名] 信息通知（无需回复）\n"
            "- [QUESTION:@队友名] 快速提问（期望简短回复）\n"
            "- [OFFER:@队友名] 主动帮助提议\n"
            "每条消息限500字以内。仅在确实有价值时使用，不要每次都发。"
        )

        if not sections:
            return ""

        return "\n# Collaboration Context\n" + "\n".join(sections)

    def _build_agent_prompt(self, agent: AgentIdentity, message: str, memory) -> str:
        """Build the full prompt for an agent including system prompt and memory.

        Applies a total token budget (estimated as chars/3.5) to prevent context
        overflow.  Components are prioritized:
          1. system_prompt + permissions + user message (always included)
          2. role + key_knowledge (always included)
          3. team roster + collaboration context (trimmed if tight)
          4. conversation replay / active context (trimmed to budget)
          5. group memory / global wiki (trimmed last)
        """
        settings = get_settings()
        # Budget in characters (rough estimate: 1 token ≈ 3.5 chars for mixed CJK/EN)
        max_prompt_chars = getattr(settings, "slock_max_prompt_chars", 32000)

        logger.debug(
            "_build_agent_prompt: agent=%s memory_path=%s has_role=%s has_knowledge=%s has_context=%s",
            agent.agent_id,
            self._memory.agent_memory_path(agent.agent_id),
            bool(memory.role),
            bool(memory.key_knowledge),
            bool(memory.active_context),
        )
        parts: list[str] = []

        if agent.system_prompt:
            parts.append(agent.system_prompt)

        # Minimum authorization: inject permitted tools so ACP session scope is bounded.
        if agent.permissions:
            parts.append(
                f"\n# Authorized Tools\n"
                f"You are ONLY permitted to use the following tools: "
                f"{', '.join(agent.permissions)}."
            )

        # Teammate roster: dynamically derived from AgentRegistry.
        # Roster content is *entirely* user-defined — every line comes from the
        # agent record (name / role / personality_traits) that the user authored
        # via /role or registry edits. No role taxonomy is hardcoded here.
        roster_block = self._render_team_roster(agent)
        if roster_block:
            parts.append(roster_block)
            # Delegation instruction: allow agent to fan out tasks to all teammates
            parts.append(
                "\n# Task Delegation\n"
                "If the incoming task explicitly requires ALL teammates to participate "
                "(e.g., self-introductions, team-wide surveys, each-person responses), "
                "prefix your response with `[DELEGATE:ALL]` on the first line. "
                "This will automatically forward the same task to every other agent. "
                "You should still provide YOUR OWN response after the prefix.\n"
                "Only use this when the task semantically requires input from EVERY team member. "
                "Normal tasks that just need one expert should NOT use this prefix."
            )

            # Planning Protocol: agent can decompose complex tasks
            parts.append(
                "\n# Task Planning Protocol\n"
                "如果任务复杂（涉及多步骤、多角色协作、或需要拆解），你可以输出规划方案：\n"
                "1. 以 `[PLAN]` 作为回复的第一行前缀\n"
                "2. 每个子任务一行，格式: `[SUB:目标角色] 子任务描述`\n"
                "3. 如有依赖关系，在子任务后追加 `[DEPENDS:序号]`（序号从1开始）\n"
                "4. 规划结束后以 `[/PLAN]` 结尾\n\n"
                "示例:\n"
                "```\n"
                "[PLAN]\n"
                "[SUB:coder] 实现用户登录接口\n"
                "[SUB:coder] 实现权限校验中间件 [DEPENDS:1]\n"
                "[SUB:tester] 编写登录接口单元测试 [DEPENDS:1]\n"
                "[SUB:reviewer] 代码审查 [DEPENDS:2,3]\n"
                "[/PLAN]\n"
                "```\n\n"
                "仅当任务确实需要多步骤协作时才使用规划。简单的单步任务直接执行即可。"
            )

            # Sidebar communication protocol
            parts.append(
                "\n# 轻量通讯\n"
                "你可以用以下前缀向队友发送非正式消息（不触发正式讨论）：\n"
                "- [FYI:@队友名] 信息通知（无需回复）\n"
                "- [QUESTION:@队友名] 快速提问（期望简短回复）\n"
                "- [OFFER:@队友名] 主动帮助提议\n"
                "每条消息限500字以内。仅在确实有价值时使用，不要每次都发。"
            )

        if memory.role:
            parts.append(f"\n# Your Role\n{memory.role}")

        # Personality-Driven Style: inject behavioral directives from traits
        if hasattr(agent, 'personality_traits') and agent.personality_traits:
            from .personality_engine import PersonalityEngine
            if not hasattr(self, '_personality_engine'):
                self._personality_engine = PersonalityEngine()
            profile = self._personality_engine.get_profile(agent.agent_id, agent.personality_traits)
            style_block = profile.to_behavioral_prompt()
            if style_block:
                parts.append(f"\n# Your Communication Style\n{style_block}")

        if memory.key_knowledge:
            parts.append(f"\n# Key Knowledge\n{memory.key_knowledge}")

        # Reasoning snapshot injection: provide thinking continuity from prior turns
        if hasattr(memory, 'reasoning_snapshot') and memory.reasoning_snapshot:
            parts.append(f"\n# Prior Reasoning\n{memory.reasoning_snapshot[-1500:]}")

        # Sidebar messages: lightweight inter-agent communication
        sidebar_block = self._sidebar_channel.render_pending_for_prompt(agent.agent_id)
        if sidebar_block:
            parts.append(sidebar_block)

        # Memory Enhancement: use conversation replay instead of raw truncation
        channel_id = self._channel.channel_id if self._channel else self.chat_id
        settings = get_settings()
        replay_rounds = settings.slock_conversation_replay_rounds

        # Parallel memory reads across tiers (L1/L2/L3)
        def _read_replay():
            return self._memory.read_conversation_replay(channel_id, replay_rounds)

        def _read_group():
            return self._memory.read_group_memory(channel_id)

        def _read_global():
            return self._memory.read_global_wiki()

        executor = self._prompt_context_executor
        future_replay = executor.submit(_read_replay)
        future_group = executor.submit(_read_group)
        future_global = executor.submit(_read_global)

        replay = future_replay.result(timeout=5)
        group_memory = future_group.result(timeout=5)
        global_wiki = future_global.result(timeout=5)

        if replay:
            replay_lines = []
            for entry in replay:
                sender = entry.get("agent_name") or entry.get("sender_type", "user")
                content = entry.get("content", "")[:500]
                replay_lines.append(f"[{sender}]: {content}")
            parts.append("\n# Recent Conversation\n" + "\n".join(replay_lines))
        elif memory.active_context:
            # Fallback: use summarized/truncated active_context
            parts.append(f"\n# Recent Context\n{memory.active_context[-2000:]}")

        if group_memory:
            parts.append(f"\n# Team Shared Memory\n{group_memory[-2000:]}")

        if global_wiki:
            parts.append(f"\n# Global Knowledge\n{global_wiki[-2000:]}")

        # Discussion context: inject active discussion thread if present (copy-on-read for thread safety)
        if hasattr(self, '_active_discussions') and hasattr(self, '_discussions_lock'):
            from ..utils.redact import redact_sensitive
            with self._discussions_lock:
                active_threads = list(self._active_discussions.get(channel_id, []))
            for active_thread in active_threads:
                if active_thread.is_active and active_thread.messages:
                    disc_lines = []
                    for msg in active_thread.messages[-6:]:  # Last 6 messages for context
                        content = msg.content[:300]
                        if hasattr(active_thread, 'channel_id') and active_thread.channel_id != channel_id:
                            content = redact_sensitive(content)
                        disc_lines.append(f"[Round {msg.round_num}] {msg.sender_agent_id}: {content}")
                    parts.append("\n# Discussion Context\n" + "\n".join(disc_lines))
                    break  # Only inject the most recent active discussion

        # Dynamic collaboration context: task board + active plans + rules
        collab_context = self._render_collaboration_context(agent, channel_id)
        if collab_context:
            parts.append(collab_context)

        parts.append(f"\n# User Message\n{message}")

        # Budget enforcement: trim lower-priority sections if total exceeds limit
        combined = "\n".join(parts)
        if len(combined) > max_prompt_chars:
            # Strategy: rebuild with trimmed optional sections
            overshoot = len(combined) - max_prompt_chars
            # Trim from lowest priority first: global_wiki, group_memory, replay
            # Find and trim these sections
            trimmed_parts: list[str] = []
            trim_budget_remaining = overshoot
            for part in parts:
                if trim_budget_remaining <= 0:
                    trimmed_parts.append(part)
                    continue
                # Lower priority sections get trimmed
                if part.startswith("\n# Global Knowledge"):
                    if trim_budget_remaining >= len(part):
                        trim_budget_remaining -= len(part)
                        continue  # Drop entirely
                    else:
                        keep_chars = len(part) - trim_budget_remaining
                        trimmed_parts.append(part[:keep_chars])
                        trim_budget_remaining = 0
                elif part.startswith("\n# Team Shared Memory"):
                    if trim_budget_remaining >= len(part):
                        trim_budget_remaining -= len(part)
                        continue
                    else:
                        keep_chars = len(part) - trim_budget_remaining
                        trimmed_parts.append(part[:keep_chars])
                        trim_budget_remaining = 0
                elif part.startswith("\n# Recent Conversation") or part.startswith("\n# Recent Context"):
                    if trim_budget_remaining >= len(part) // 2:
                        # Keep at least half of conversation context
                        keep_chars = max(len(part) // 2, len(part) - trim_budget_remaining)
                        trimmed_parts.append(part[:keep_chars] + "\n[...truncated...]")
                        trim_budget_remaining -= (len(part) - keep_chars)
                    else:
                        keep_chars = len(part) - trim_budget_remaining
                        trimmed_parts.append(part[:keep_chars])
                        trim_budget_remaining = 0
                else:
                    trimmed_parts.append(part)
            combined = "\n".join(trimmed_parts)
            logger.info(
                "Prompt budget enforcement: trimmed %d chars (budget=%d, final=%d) agent=%s",
                overshoot, max_prompt_chars, len(combined), agent.agent_id,
            )

        return combined

    def _run_acp_session(
        self,
        agent: AgentIdentity,
        prompt: str,
        *,
        timeout: Optional[float] = None,
    ) -> Optional[str]:
        """Run an ACP session for the agent. Returns response text or None.

        Security: auto_approve=True suppresses interactive prompts (zero HI).
        Tool authorization is bounded by agent.permissions which is injected
        into the system prompt by _execute_agent, restricting which tools the
        agent may use.
        """
        execution_errors = self._get_agent_execution_errors()
        try:
            execution_errors.pop(agent.agent_id, None)
            thread_id = f"slock_agent_{agent.agent_id}"
            session_cwd = self.root_path
            if agent.security_profile == "employee_v1":
                session_cwd = _employee_acp_workspace(agent)
                prompt = self.preview_employee_session_prompt(agent, prompt)
            # NOTE: agent.permissions defines the least-privilege tool set for
            # this role (e.g. ["file_read"] for planner). Tool authorization should
            # be handled by the session layer once allowed_tools filtering is
            # supported natively in create_engine_session.
            session = create_engine_session(
                agent_type=agent.agent_type,
                cwd=session_cwd,
                model_name=agent.model_name or None,
                thread_id=thread_id,
                auto_approve=True,
                require_tool_filter=True,
            )
            if session is None:
                logger.warning("Failed to create ACP session for agent %s", agent.name)
                return None
            try:
                self._apply_tool_restrictions(session, agent)
            except Exception:
                close_session_safely(session)
                raise

            with self._lock:
                self._agent_sessions[agent.agent_id] = session
            try:
                result = session.send_prompt(
                    prompt,
                    timeout=timeout if timeout is not None else self.settings.coco_execution_timeout,
                )
                return result.text if result else None
            finally:
                with self._lock:
                    if self._agent_sessions.get(agent.agent_id) is session:
                        del self._agent_sessions[agent.agent_id]
                close_session_safely(session)

        except SecurityPolicyDegradedError:
            raise
        except Exception as e:
            from src.utils.errors import redact_sensitive
            execution_errors[agent.agent_id] = redact_sensitive(str(e))
            logger.error("ACP session error for agent %s: %s", agent.name, redact_sensitive(str(e)), exc_info=True)
            return None

    def _apply_tool_restrictions(self, session, agent: AgentIdentity) -> None:
        """Install per-agent ACP tool filter for Slock least-privilege execution."""
        set_filter = getattr(session, "set_tool_filter", None)
        if not callable(set_filter):
            raise SecurityPolicyDegradedError("ACP session does not support Slock tool filtering")

        settings = getattr(self, "_settings", get_settings())
        configured_roots = list(getattr(settings, "slock_tool_path_restrictions", []) or [])
        configured_roots = [os.path.realpath(path) for path in configured_roots if path]
        raw_project_root = str(getattr(self, "root_path", "") or "")
        project_root = os.path.realpath(raw_project_root) if raw_project_root else ""
        workspace_root = os.path.realpath(agent.workspace_path) if agent.workspace_path else ""
        read_roots = tuple(
            dict.fromkeys(
                root for root in (*configured_roots, project_root, workspace_root) if root
            )
        )
        write_roots = tuple(
            dict.fromkeys(root for root in (*configured_roots, project_root) if root)
        )

        dangerous = getattr(self, "_dangerous_shell_patterns", None)
        if dangerous is None:
            patterns = list(self._BUILTIN_DANGEROUS_PATTERNS)
            patterns.extend(getattr(settings, "slock_dangerous_shell_patterns", []) or [])
            dangerous = re.compile(r"|".join(patterns), re.IGNORECASE)
            self._dangerous_shell_patterns = dangerous

        permissions = set(agent.permissions or [])
        capabilities = set(agent.capabilities or [])
        employee_policy = agent.security_profile == "employee_v1"
        effective_write_roots = write_roots if employee_policy else read_roots
        configure_employee_sandbox = getattr(
            session,
            "configure_employee_sandbox",
            None,
        )
        if employee_policy and callable(configure_employee_sandbox):
            writable_cli_roots = (
                effective_write_roots
                if "file_write" in permissions and "file_write" in capabilities
                else ()
            )
            configure_employee_sandbox(
                read_only_roots=read_roots,
                writable_roots=writable_cli_roots,
            )

        def under_roots(path: str, roots: tuple[str, ...]) -> bool:
            if not roots:
                return False
            candidate = os.path.realpath(path)
            return any(candidate == root or candidate.startswith(root + os.sep) for root in roots)

        def is_sensitive_path(path: str) -> bool:
            candidate = os.path.realpath(path)
            parts = set(Path(candidate).parts)
            return (
                ".env" in parts
                or "vault" in {part.casefold() for part in parts}
                or "journal" in {part.casefold() for part in parts}
            )

        def path_from_tool_args(args: dict) -> str:
            path = str(args.get("path") or args.get("file_path") or "")
            if not path:
                return ""
            cwd = str(args.get("cwd") or raw_project_root or agent.workspace_path or "")
            return path if os.path.isabs(path) else os.path.join(cwd, path)

        def shell_path_tokens(command: str, cwd: str) -> list[str]:
            try:
                tokens = shlex.split(command)
            except ValueError:
                tokens = command.split()
            paths: list[str] = []
            for token in tokens:
                if token.startswith("/") or token.startswith("../") or token.startswith("./"):
                    paths.append(token if os.path.isabs(token) else os.path.join(cwd, token))
            return paths

        def invokes_lark_cli(command: str) -> bool:
            names = {"lark-cli", "lark_cli", "feishu-cli", "feishu_cli"}
            try:
                tokens = shlex.split(command)
            except ValueError:
                tokens = command.split()
            if any(os.path.basename(token).casefold() in names for token in tokens):
                return True
            return re.search(
                r"(?:^|[^A-Za-z0-9_-])(?:lark[-_]cli|feishu[-_]cli)"
                r"(?:$|[^A-Za-z0-9_-])",
                command,
                re.IGNORECASE,
            ) is not None

        def invokes_git_cli(command: str) -> bool:
            """Fail closed on git through shell; employee git uses its typed tool."""

            try:
                tokens = shlex.split(command)
            except ValueError:
                return True
            return any(os.path.basename(token).casefold() == "git" for token in tokens)

        git_write_commands = {
            "add",
            "am",
            "apply",
            "bisect",
            "branch",
            "checkout",
            "cherry-pick",
            "clean",
            "commit",
            "merge",
            "mv",
            "rebase",
            "reset",
            "restore",
            "revert",
            "rm",
            "stash",
            "switch",
            "tag",
        }
        git_read_commands = {
            "blame",
            "cat-file",
            "describe",
            "diff",
            "for-each-ref",
            "grep",
            "log",
            "ls-files",
            "ls-tree",
            "name-rev",
            "rev-parse",
            "shortlog",
            "show",
            "status",
            "whatchanged",
        }

        def parse_git_command(command: str) -> tuple[str, tuple[str, ...]] | None:
            try:
                tokens = shlex.split(command)
            except ValueError:
                return None
            if not tokens or os.path.basename(tokens[0]).casefold() != "git":
                return None
            tokens = tokens[1:]
            if not tokens or tokens[0].startswith("-"):
                return None
            return tokens[0].casefold(), tuple(tokens[1:])

        def git_authorized(command: str) -> bool:
            if employee_policy and not (
                {"shell", "git", "file_write"} <= permissions
                and {"shell", "git", "file_write"} <= capabilities
            ):
                return False
            if not employee_policy and "git" not in permissions:
                return False
            parsed = parse_git_command(command)
            if parsed is None:
                return False
            subcommand, git_tokens = parsed
            blocked_options = {
                "--unsafe-paths",
                "--ext-diff",
                "--textconv",
                "--exec",
                "--upload-pack",
                "--receive-pack",
            }
            if any(
                token in blocked_options
                or token.startswith("--output=")
                or token == "--output"
                or token.startswith("--open-files-in-pager")
                for token in git_tokens
            ):
                return False
            if subcommand in git_read_commands:
                return True
            if subcommand in git_write_commands:
                return "file_write" in permissions and (
                    not employee_policy or "file_write" in capabilities
                )
            return False

        def tool_filter(tool_name: str, args: dict | None = None) -> bool:
            args = args or {}
            normalized_tool = (tool_name or "").lower()
            if normalized_tool == "shell":
                if employee_policy and not (
                    {"shell", "git", "file_write"} <= permissions
                    and {"shell", "git", "file_write"} <= capabilities
                ):
                    return False
                if "shell" not in permissions or (
                    employee_policy and "shell" not in capabilities
                ):
                    return False
                if employee_policy and (
                    "file_write" not in permissions
                    or "file_write" not in capabilities
                ):
                    return False
                command = str(args.get("command") or "")
                if employee_policy and invokes_lark_cli(command):
                    return False
                if employee_policy and invokes_git_cli(command):
                    return False
                if dangerous.search(command):
                    return False
                cwd = os.path.realpath(
                    str(args.get("cwd") or raw_project_root or agent.workspace_path or "")
                )
                if not under_roots(cwd, effective_write_roots):
                    return False
                for path in shell_path_tokens(command, cwd):
                    if is_sensitive_path(path) or not under_roots(path, effective_write_roots):
                        return False
                return True

            path_tools = {"file_read", "file_write", "file_list", "grep", "search"}
            if normalized_tool in path_tools:
                if normalized_tool.startswith("file_write") and (
                    "file_write" not in permissions
                    or (employee_policy and "file_write" not in capabilities)
                ):
                    return False
                if normalized_tool in {"file_read", "file_list", "grep", "search"} and not (
                    {"file_read", "git", "shell"} & permissions
                ):
                    return False
                path = path_from_tool_args(args)
                if not path or is_sensitive_path(path):
                    return False
                roots = effective_write_roots if normalized_tool.startswith("file_write") else read_roots
                return under_roots(path, roots)
            if normalized_tool == "git" and employee_policy:
                if set(args) != {"command", "cwd"}:
                    return False
                command = str(
                    args.get("command")
                    or args.get("subcommand")
                    or args.get("operation")
                    or ""
                )
                if not git_authorized(command):
                    return False
                cwd = os.path.realpath(
                    str(args.get("cwd") or raw_project_root or agent.workspace_path or "")
                )
                return bool(cwd) and under_roots(cwd, effective_write_roots)
            if employee_policy:
                return False
            return True

        set_filter(tool_filter)

    def _get_agent_execution_errors(self) -> dict[str, str]:
        """Return the execution-error cache, initializing old test doubles lazily."""
        execution_errors = getattr(self, "_agent_execution_errors", None)
        if not isinstance(execution_errors, dict):
            execution_errors = {}
            self._agent_execution_errors = execution_errors
        return execution_errors

    # ------------------------------------------------------------------
    # Dirty flag helper
    # ------------------------------------------------------------------

    def _set_dirty(self, value: bool) -> None:
        """Set the dirty flag (used by composed managers)."""
        self._dirty = value
        # Task 19: Schedule debounced status card refresh on state change
        if value:
            self._schedule_status_refresh()

    def _schedule_status_refresh(self, delay: float = 3.0) -> None:
        """Schedule a debounced status card refresh.

        Cancels any pending timer and sets a new one. The actual refresh
        is performed by the on_status_refresh callback if registered.
        """
        if self._status_refresh_timer is not None:
            self._status_refresh_timer.cancel()

        if not self._status_card_msg_ids:
            return

        def _do_refresh():
            self._status_refresh_timer = None
            channel_id = self._channel.channel_id if self._channel else ""
            msg_id = self._status_card_msg_ids.get(channel_id)
            if not msg_id:
                return
            # Emit refresh via callback (handler registers this)
            cb = getattr(self, "_on_status_refresh_cb", None)
            if cb:
                try:
                    team_name = self._channel.team_name if self._channel else ""
                    status_card = self.get_status_card(team_name=team_name)
                    cb(msg_id, status_card)
                except Exception as exc:
                    logger.warning("Status refresh callback error: %s", exc, exc_info=True)

        self._status_refresh_timer = threading.Timer(delay, _do_refresh)
        self._status_refresh_timer.daemon = True
        self._status_refresh_timer.start()

    def register_status_refresh_callback(self, callback) -> None:
        """Register callback(msg_id, card_dict) for auto-refresh."""
        self._on_status_refresh_cb = callback

    # ------------------------------------------------------------------
    # Task Management (delegated to TaskBoardManager)
    # ------------------------------------------------------------------

    def add_task(self, content: str) -> Optional[SlockTask]:
        """Create a new task in the channel.

        Returns:
            SlockTask if successfully created.
            None if the channel has reached ``slock_max_open_tasks`` limit.

        Note:
            This is a breaking contract change — callers MUST handle the None
            return case (e.g. display a "team busy" card to the user) instead
            of unconditionally accessing task attributes.
        """
        return self._task_mgr.add_task(content)

    def claim_task(self, task_id: str, agent_id: str) -> bool:
        """Attempt to claim a task for an agent."""
        return self._task_mgr.claim_task(task_id, agent_id)

    def complete_task(self, task_id: str, agent_id: str) -> bool:
        """Mark a task as done."""
        return self._task_mgr.complete_task(task_id, agent_id)

    def execute_task(
        self,
        task_id: str,
        agent_id: str,
        callbacks: Optional[SlockEngineCallbacks] = None,
        *,
        request_review: bool = True,
        freshness_check: bool = True,
    ) -> Optional[str]:
        """Execute a task end-to-end: claim → execute → complete/rollback."""
        return self._task_mgr.execute_task(
            task_id,
            agent_id,
            callbacks,
            request_review=request_review,
            freshness_check=freshness_check,
        )

    def _rollback_task(self, task_id: str, agent_id: str) -> None:
        """Rollback a task to TODO state and release its claim."""
        self._task_mgr._rollback_task(task_id, agent_id)

    def _persist_task_board(self) -> None:
        """Persist task state for the active channel."""
        self._task_mgr._persist_task_board()

    def _trim_done_tasks(self) -> None:
        """Remove oldest DONE tasks when exceeding cap."""
        self._task_mgr._trim_done_tasks()

    def _flush_if_dirty(self, snapshot: list[SlockTask]) -> None:
        """Persist task board from a snapshot if dirty flag is set."""
        self._task_mgr._flush_if_dirty(snapshot)

    # ------------------------------------------------------------------
    # Escalation Protocol (delegated to EscalationManager)
    # ------------------------------------------------------------------

    _MAX_ESCALATION_RETRIES = 3

    def escalate(
        self,
        agent: AgentIdentity,
        reason: str,
        *,
        level: EscalationLevel = EscalationLevel.BLOCKED,
        task_id: Optional[str] = None,
        context: str = "",
        options: Optional[list[str]] = None,
        callbacks: Optional["SlockEngineCallbacks"] = None,
    ) -> EscalationRequest:
        """Raise an escalation request."""
        return self._escalation_mgr.escalate(
            agent, reason, level=level, task_id=task_id,
            context=context, options=options, callbacks=callbacks,
        )

    def resolve_escalation(
        self,
        escalation_id: str,
        resolution: str,
    ) -> Optional[EscalationRequest]:
        """Resolve a pending escalation with the admin's decision."""
        return self._escalation_mgr.resolve_escalation(escalation_id, resolution)

    def get_escalation(self, escalation_id: str) -> Optional[EscalationRequest]:
        """Get an escalation by ID."""
        return self._escalation_mgr.get_escalation(escalation_id)

    def get_pending_escalations(self) -> list[EscalationRequest]:
        """Return all unresolved escalations."""
        return self._escalation_mgr.get_pending_escalations()

    def get_escalation_card(self, escalation: EscalationRequest) -> dict:
        """Build the interactive card for an escalation request."""
        return self._escalation_mgr.get_escalation_card(escalation)

    def set_escalation_ui_callbacks(
        self,
        update_card_fn: Callable[[str, str], bool],
        send_text_fn: Callable[[str, str], None],
    ) -> None:
        """Wire UI callbacks for escalation timeout auto-abort notifications."""
        self._escalation_mgr.set_ui_callbacks(update_card_fn, send_text_fn)

    def resume_after_escalation(
        self,
        escalation: EscalationRequest,
        callbacks: Optional[SlockEngineCallbacks] = None,
    ) -> Optional[str]:
        """Resume agent work after an escalation has been resolved."""
        resolution = (escalation.resolution or "").strip()
        task_id = escalation.task_id
        if task_id and resolution in SKIP_OPTIONS:
            for task in self._tasks:
                if task.task_id == task_id:
                    task.status = TaskStatus.TODO
                    task.claimed_by = None
                    break
        elif task_id and resolution in ABORT_OPTIONS:
            for task in self._tasks:
                if task.task_id == task_id:
                    task.status = TaskStatus.DONE
                    break
        return self._escalation_mgr.resume_after_escalation(escalation, callbacks)

    def _force_complete_task(self, task_id: str, *, reason: str = "", actor_id: str = "system:engine") -> None:
        """Force-mark a task as DONE regardless of claimer."""
        self._task_mgr.force_complete_task(task_id, reason=reason, actor_id=actor_id)

    def _trim_escalations(self) -> None:
        """Delegate escalation trimming to the escalation manager."""
        self._escalation_mgr._trim_escalations()

    # ------------------------------------------------------------------
    # Action Card Side-Effect Gate (Slock Insight #6)
    # ------------------------------------------------------------------

    def propose_action(
        self,
        agent: AgentIdentity,
        action_type: str,
        title: str,
        *,
        description: str = "",
        command: str = "",
        impact_summary: str = "",
        reversible: bool = True,
        callbacks: Optional[SlockEngineCallbacks] = None,
    ) -> Optional[str]:
        """Propose a side-effect action that requires human confirmation.

        Returns the action_id if proposal card was sent, None otherwise.
        The action is held until the user approves or rejects via card callback.
        """
        from .card_templates.action_card import (
            ActionProposal,
            ActionType,
            build_action_proposal_card,
        )

        settings = get_settings()
        if not settings.slock_action_card_enabled:
            return None

        try:
            a_type = ActionType(action_type)
        except ValueError:
            a_type = ActionType.CUSTOM

        channel_id = self._channel.channel_id if self._channel else self.chat_id
        proposal = ActionProposal(
            action_type=a_type,
            agent_id=agent.agent_id,
            agent_name=agent.name,
            title=title,
            description=description,
            command=command,
            impact_summary=impact_summary,
            reversible=reversible,
            channel_id=channel_id,
        )

        card = build_action_proposal_card(proposal)
        msg_id = None
        if callbacks and callbacks.on_card_send:
            msg_id = callbacks.on_card_send(card)
        elif self._card_send_fn:
            msg_id = self._card_send_fn(card)

        if msg_id:
            proposal.card_message_id = msg_id

        # Store pending action for resolution
        if not hasattr(self, "_pending_actions"):
            self._pending_actions: dict[str, "ActionProposal"] = {}
        self._pending_actions[proposal.action_id] = proposal

        # Start timeout timer for auto-reject if no human response
        from ..config import get_settings as _get_settings
        action_timeout = getattr(_get_settings(), "slock_action_timeout_seconds", 300.0)  # 5 min default
        self._start_action_timeout(proposal.action_id, action_timeout)

        logger.info(
            "Action proposed: id=%s type=%s agent=%s title=%s",
            proposal.action_id, action_type, agent.name, title[:50],
        )
        return proposal.action_id

    def resolve_action(
        self,
        action_id: str,
        approved: bool,
    ) -> Any:
        """Resolve a pending action proposal (approve or reject).

        Returns the updated proposal, or None if not found.
        """
        from .card_templates.action_card import ActionStatus

        pending = getattr(self, "_pending_actions", {})
        proposal = pending.get(action_id)
        if not proposal:
            return None

        proposal.resolved_at = time.time()
        if approved:
            proposal.status = ActionStatus.APPROVED
        else:
            proposal.status = ActionStatus.REJECTED

        # Cancel timeout timer
        action_timers = getattr(self, "_action_timers", {})
        timer = action_timers.pop(action_id, None)
        if timer:
            timer.cancel()

        logger.info(
            "Action %s: id=%s agent=%s",
            "approved" if approved else "rejected",
            action_id, proposal.agent_name,
        )
        return proposal

    def complete_action(
        self,
        action_id: str,
        *,
        success: bool = True,
        result: str = "",
    ) -> Any:
        """Mark an approved action as completed or failed after execution."""
        from .card_templates.action_card import ActionStatus, build_action_result_card

        pending = getattr(self, "_pending_actions", {})
        proposal = pending.get(action_id)
        if not proposal:
            return None

        proposal.status = ActionStatus.COMPLETED if success else ActionStatus.FAILED
        proposal.result = result
        proposal.resolved_at = time.time()

        # Send result card
        result_card = build_action_result_card(proposal)
        if proposal.card_message_id and self._card_update_fn:
            self._card_update_fn(proposal.card_message_id, result_card)
        elif self._card_send_fn:
            self._card_send_fn(result_card)

        # Clean up
        pending.pop(action_id, None)
        return proposal

    def _start_action_timeout(self, action_id: str, timeout_seconds: float) -> None:
        """Start a daemon timer that auto-rejects an action if not resolved in time."""
        def _timeout_handler():
            pending = getattr(self, "_pending_actions", {})
            proposal = pending.get(action_id)
            if proposal is None:
                return  # Already resolved
            from .card_templates.action_card import ActionStatus
            if proposal.status != ActionStatus.PROPOSED:
                return  # Already resolved

            # Auto-reject due to timeout
            logger.info(
                "Action auto-rejected (timeout %.0fs): id=%s title=%s",
                timeout_seconds, action_id, proposal.title[:50],
            )
            proposal.status = ActionStatus.REJECTED
            proposal.resolved_at = time.time()
            proposal.result = f"⏰ 操作已超时自动拒绝（{int(timeout_seconds)}秒未响应）"

            # Update the card to show timeout status
            from .card_templates.action_card import build_action_result_card
            result_card = build_action_result_card(proposal)
            if proposal.card_message_id and self._card_update_fn:
                self._card_update_fn(proposal.card_message_id, result_card)

            # Clean up
            pending.pop(action_id, None)

        timer = threading.Timer(timeout_seconds, _timeout_handler)
        timer.daemon = True
        timer.name = f"action-timeout-{action_id[:8]}"
        timer.start()

        # Store timer ref for cancellation on early resolution
        if not hasattr(self, "_action_timers"):
            self._action_timers: dict[str, threading.Timer] = {}
        self._action_timers[action_id] = timer

    def capture_dissolve_snapshot(self) -> TeamSnapshot:
        """Capture the active channel, role bindings, and task board for undo."""
        channel = self._channel
        channel_id = channel.channel_id if channel else self.chat_id
        agents = self._registry.list_agents(channel_id=channel_id)
        return TeamSnapshot(
            channel_id=channel_id,
            team_name=channel.team_name if channel else "",
            owner_id=channel.owner_id if channel else "",
            channel=channel,
            agent_ids=[agent.agent_id for agent in agents],
            agent_bindings={agent.agent_id: agent.role for agent in agents},
            task_ids=[task.task_id for task in self._tasks],
            task_board_data=[task.to_dict() for task in self._tasks],
        )

    @activation_serialized
    def restore_from_snapshot(self, snapshot: TeamSnapshot | None) -> bool:
        """Restore a recently dissolved team snapshot."""
        if snapshot is None or snapshot.channel is None:
            return False
        self.activate_channel(snapshot.channel)
        self._tasks.clear()
        self._tasks.extend(SlockTask.from_dict(data) for data in snapshot.task_board_data)
        for agent_id in snapshot.agent_ids:
            self.set_agent_status(agent_id, AgentStatus.IDLE)
        self._persist_task_board()
        return True

    # ------------------------------------------------------------------
    # Parallel Execution
    # ------------------------------------------------------------------

    def _parallel_agent_limit(self, override: Optional[int] = None) -> int:
        """Return the validated parallel-agent limit from the engine settings snapshot."""
        value = self._settings.slock_max_parallel_agents if override is None else override
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("max_concurrent must be a positive integer")
        return value

    def _get_executor(self) -> BoundedExecutor:
        """Lazy-initialize the bounded thread pool executor."""
        with self._executor_lock:
            if self._executor is None:
                self._executor = BoundedExecutor(
                    max_workers=self._parallel_agent_limit(),
                    max_queue_size=self._settings.slock_max_queue_size,
                )
            return self._executor

    def execute_parallel(
        self,
        task_assignments: list[tuple[str, str]],
        callbacks: Optional[SlockEngineCallbacks] = None,
        *,
        timeout: float = 300.0,
    ) -> dict[str, Optional[str]]:
        """Execute multiple tasks in parallel using ThreadPoolExecutor.

        Args:
            task_assignments: List of (task_id, agent_id) tuples to execute concurrently.
            callbacks: Engine lifecycle callbacks.
            timeout: Maximum wall-clock time for the batch (seconds).

        Returns:
            Dict mapping task_id → formatted result (or None on failure).
        """
        with self._lock:
            if self._run_state == EngineRunState.STOPPING:
                return {tid: None for tid, _ in task_assignments}
            self._run_state = EngineRunState.RUNNING

        executor = self._get_executor()
        futures: dict[Future, str] = {}  # future → task_id
        task_to_agent: dict[str, str] = {}  # task_id → agent_id for cancellation
        results: dict[str, Optional[str]] = {}

        try:
            for task_id, agent_id in task_assignments:
                task_to_agent[task_id] = agent_id
                try:
                    future = executor.submit(
                        self.execute_task,
                        task_id,
                        agent_id,
                        callbacks,
                        request_review=False,
                        freshness_check=False,
                    )
                    futures[future] = task_id
                except (QueueFullError, RuntimeError) as e:
                    from src.utils.errors import redact_sensitive
                    logger.warning("Failed to submit task %s: %s", task_id, redact_sensitive(repr(e)))
                    results[task_id] = None
                    if callbacks and callbacks.on_error:
                        callbacks.on_error(f"Task {task_id} rejected: {redact_sensitive(str(e))}")

            for future in as_completed(futures, timeout=timeout):
                task_id = futures[future]
                try:
                    results[task_id] = future.result()
                except Exception as e:
                    from src.utils.errors import redact_sensitive
                    logger.error("Parallel task %s failed: %s", task_id, redact_sensitive(repr(e)), exc_info=True)
                    results[task_id] = None
                    if callbacks and callbacks.on_error:
                        callbacks.on_error(f"Task {task_id} failed: {redact_sensitive(str(e))}")

        except TimeoutError:
            logger.warning("Parallel execution timed out after %.1fs", timeout)
            # Cancel incomplete agents via cancellation events
            incomplete_futures = []
            for future, task_id in futures.items():
                if task_id not in results:
                    results[task_id] = None
                    incomplete_futures.append(future)
                    agent_id = task_to_agent.get(task_id)
                    if agent_id:
                        self.cancel_agent(agent_id)
            # Grace period: wait for cancelled agents to finish cleanup (event-driven)
            if incomplete_futures:
                futures_wait(incomplete_futures, timeout=5.0)
            if callbacks and callbacks.on_error:
                callbacks.on_error(f"Parallel execution timed out after {timeout}s")
        finally:
            with self._lock:
                if self._run_state == EngineRunState.RUNNING:
                    self._run_state = EngineRunState.IDLE

        return results

    def dispatch_pending_tasks(
        self,
        callbacks: Optional[SlockEngineCallbacks] = None,
        *,
        max_concurrent: Optional[int] = None,
    ) -> dict[str, Optional[str]]:
        """Auto-assign and execute all pending TODO tasks in parallel.

        For each TODO task, uses the TaskRouter to find the best available agent,
        then dispatches all assignments concurrently via execute_parallel.

        Args:
            callbacks: Engine lifecycle callbacks.
            max_concurrent: Positive integer overriding the configured parallel-agent limit.

        Returns:
            Dict mapping task_id → formatted result (or None on failure/skip).
        """
        limit = self._parallel_agent_limit(max_concurrent)
        channel_id = self._channel.channel_id if self._channel else self.chat_id
        agents = self._registry.list_agents(channel_id=channel_id)

        if not agents:
            return {}

        # Snapshot pending tasks under lock to avoid TOCTOU race
        with self._lock:
            pending = [t for t in self._tasks if t.status == TaskStatus.TODO]
        if not pending:
            return {}

        assignments: list[tuple[str, str]] = []
        assigned_agents: set[str] = set()

        for task in pending[:limit]:
            # Route task content to best available agent (not already assigned)
            available = [a for a in agents if a.agent_id not in assigned_agents]
            available = self._apply_wake_policy(task.content, available)
            target = self._router.route_message(task.content, available)
            if not target:
                continue  # no idle agent available for this task
            assignments.append((task.task_id, target.agent_id))
            assigned_agents.add(target.agent_id)

        if not assignments:
            return {}

        return self.execute_parallel(assignments, callbacks)

    # ------------------------------------------------------------------
    # Status & Cleanup
    # ------------------------------------------------------------------

    def get_status_card(self, team_name: str = "") -> dict:
        """Build the status panel card for all agents in this channel."""
        channel_id = self._channel.channel_id if self._channel else self.chat_id
        agents = self._registry.list_agents(channel_id=channel_id)
        agent_statuses = [(a, self.get_agent_status(a.agent_id)) for a in agents]
        current_tasks = {
            task.claimed_by: task
            for task in self._tasks
            if task.claimed_by and task.status in (TaskStatus.IN_PROGRESS, TaskStatus.IN_REVIEW)
        }
        return build_status_panel_card(
            agent_statuses,
            team_name=team_name,
            channel_id=channel_id,
            current_tasks=current_tasks,
        )

    def pause(self) -> None:
        """Pause the engine."""
        with self._lock:
            self._run_state = EngineRunState.STOPPING
            session = self._session  # snapshot under lock to avoid TOCTOU
            agent_sessions = list(self._agent_sessions.values())
            self._agent_sessions.clear()
        if session:
            with contextlib.suppress(Exception):  # intentional: cleanup path
                session.cancel()
        for agent_session in agent_sessions:
            with contextlib.suppress(Exception):  # intentional: cleanup path
                agent_session.cancel()

    def resume(self, callbacks: Optional[SlockEngineCallbacks] = None) -> None:
        """Resume the engine from paused state."""
        with self._lock:
            self._run_state = EngineRunState.IDLE

    def cleanup(self) -> None:
        """Clean up engine resources."""
        # Flush and stop observer learning queue
        self._observer_queue.shutdown()

        # Cancel escalation timeout timers
        self._escalation_mgr.shutdown_timers()

        # Shutdown progress tracker
        self._progress_tracker.shutdown()

        # Shutdown thread pool executor
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

        # Shutdown discussion executor
        self._discussion_executor.shutdown(wait=False)

        # Reset all agent statuses
        with self._lock:
            for agent_id in list(self._agent_statuses.keys()):
                self._agent_statuses[agent_id] = AgentStatus.IDLE
            agent_sessions = list(self._agent_sessions.values())
            self._agent_sessions.clear()
        for agent_session in agent_sessions:
            with contextlib.suppress(Exception):  # intentional: cleanup path
                agent_session.cancel()
        super().cleanup()

    @activation_serialized
    def deactivate(self) -> None:
        """Deactivate slock mode for this channel.

        Stops the engine, resets all agent statuses, and clears channel binding.
        After deactivation, the engine will refuse to execute new messages.
        """
        with self._lock:
            self._run_state = EngineRunState.STOPPING
            # Reset all agents to IDLE
            for agent_id in list(self._agent_statuses.keys()):
                self._agent_statuses[agent_id] = AgentStatus.IDLE
            self._channel = None
            session = self._session  # snapshot under lock to avoid TOCTOU
            agent_sessions = list(self._agent_sessions.values())
            self._agent_sessions.clear()

        # Cancel any running session using the snapshot
        if session:
            with contextlib.suppress(Exception):  # intentional: cleanup path
                session.cancel()
        for agent_session in agent_sessions:
            with contextlib.suppress(Exception):  # intentional: cleanup path
                agent_session.cancel()

        # Cancel escalation timeout timers
        self._escalation_mgr.shutdown_timers()

        # Stop idle scan thread
        self._task_mgr.stop_idle_scan()

        # Shutdown collaboration orchestrator (cancel plan timers)
        self._collaboration_orchestrator.shutdown()

        # Flush and stop observer learning queue
        self._observer_queue.shutdown()

        # Release thread pool resources
        with self._executor_lock:
            if self._executor is not None:
                self._executor.shutdown(wait=False)
                self._executor = None

        # Shutdown discussion executor
        self._discussion_executor.shutdown(wait=False)

        logger.info("SlockEngine deactivated for chat %s", self.chat_id)

    def stop_agent(self, agent_id: str) -> bool:
        """Stop a single agent: cancel its ACP session and reset status to IDLE.

        Returns True if the agent was found and stopped, False otherwise.
        Does not affect other agents or the engine's overall state.
        """
        with self._lock:
            if agent_id not in self._agent_statuses:
                return False
            self._agent_statuses[agent_id] = AgentStatus.IDLE
            agent_session = self._agent_sessions.pop(agent_id, None)

        # Wake queue consumers since agent is now available.
        task_queue = getattr(self, "_task_queue", None)
        if task_queue is not None:
            with contextlib.suppress(Exception):
                task_queue.notify_idle()

        # Signal cancellation to the executing thread
        cancel_agent = getattr(self, "cancel_agent", None)
        if callable(cancel_agent):
            with contextlib.suppress(Exception):
                cancel_agent(agent_id)

        if agent_session:
            with contextlib.suppress(Exception):  # intentional: cleanup path
                agent_session.cancel()

        logger.info("Stopped agent %s in chat %s", agent_id, self.chat_id)
        return True

    def cancel_employee_session(self, agent_id: str) -> bool:
        """Latch cancellation even if the employee ACP session is not registered yet."""

        cancel_event = self._get_cancel_event(agent_id)
        cancel_event.set()
        with self._lock:
            session = self._agent_sessions.pop(agent_id, None)
        if session is not None:
            with contextlib.suppress(Exception):
                session.cancel()
        logger.info("Employee session cancellation requested for %s", agent_id)
        return True

    @property
    def is_active(self) -> bool:
        """Check if the engine is active (has a channel and is not stopping)."""
        with self._lock:
            return self._channel is not None and self._run_state != EngineRunState.STOPPING
