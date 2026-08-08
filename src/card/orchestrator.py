"""TaskOrchestrator: manages task-level card sessions for multi-task execution.

Responsibilities:
- Parse plan info to identify tasks
- Create independent CardSession (wrapped in SessionRotator) per task
- Route ACP events to the correct task's session
- Broadcast task status changes to all active sessions
- Handle fallback to single-session mode when plan parsing fails
- Resolve which task_id an ACP event belongs to (TaskIdResolver)
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
import weakref
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

from src.card.events import CardEvent, CardEventType
from src.card.hooks import BackfillHook
from src.card.nav_link import format_task_continuation_link
from src.card.task_registry import TaskRegistry, TaskStatus
from src.card.tool_display import (
    collect_subagent_opaque_ids,
    extract_agent_tool_name,
    extract_tool_call_label,
    is_unhelpful_display_label,
    sanitize_subagent_display_text,
    summarize_tool_call_content,
)
from src.card.ui_text import UI_TEXT

if TYPE_CHECKING:
    from src.acp.models import ACPEvent
    from src.card.protocols import Dispatchable, StreamBridge
    from src.card.session.core import CardSession
    from src.card.session.rotator import SessionRotator

logger = logging.getLogger(__name__)

# Debounce window for broadcast (800ms) to coalesce rapid status changes.
# Larger window dramatically reduces structural events fan-out (N task cards × every plan_update),
# trading at-most ~0.8s task_list lag for far less Feishu API back-pressure and visible
# "all cards updating in lockstep with overlapping content" UX.
_BROADCAST_DEBOUNCE_MS = 800

# Minimum number of tasks for multi-card split
_MIN_TASKS_FOR_MULTI_CARD = 2
_AGENT_TOOL_TITLES = {"agent", "subagent", "task"}
_GENERIC_TASK_LABELS = {"", "agent", "subagent", "task", "子任务"}
_FINAL_SUMMARY_TASK_ID = "__final_summary__"
UNCONFIRMED_SUBAGENT_SUMMARY = "父任务已结束，子任务终态未确认"
_SUBAGENT_PROVIDER_STATUS: dict[str, TaskStatus] = {
    "pendinginit": "in_progress",
    "pending_init": "in_progress",
    "pending": "in_progress",
    "running": "in_progress",
    "completed": "completed",
    "cancelled": "cancelled",
    "shutdown": "completed",
    "errored": "failed",
    "failed": "failed",
    "notfound": "failed",
    "not_found": "failed",
    "interrupted": "cancelled",
}
_SUBAGENT_PROVIDER_PROGRESS = {
    "pendinginit": "准备中",
    "pending_init": "准备中",
    "pending": "准备中",
    "running": "执行中",
    "completed": "已完成",
    "cancelled": "已取消",
    "shutdown": "已完成并停止",
    "errored": "执行失败",
    "failed": "执行失败",
    "notfound": "状态不可用",
    "not_found": "状态不可用",
    "interrupted": "已中断",
}
_TERMINAL_TASK_STATUSES = frozenset({"completed", "failed", "cancelled"})
_TRANSIENT_SUBAGENT_PROGRESS = frozenset({
    "准备中",
    "执行中",
    "已启动",
    "正在启动",
    "启动未完成",
    "已与主 Agent 交互",
    "正在与主 Agent 交互",
    "交互未完成",
    "正在中断",
    "中断未完成",
    "动态已更新",
})


class TaskIdResolver:
    """Resolves which task_id an ACP event belongs to.

    Strategy:
    1. Track active plan step indices based on PLAN_UPDATE status transitions
    2. Support multiple concurrently active tasks (subagent scenarios)
    3. When resolving, prefer the most recently activated task
    4. Fall back to the most recently active task_id if inference fails

    Thread-safe: mutable state protected by a lock.
    """

    def __init__(self, task_ids: list[str]) -> None:
        self._task_ids = list(task_ids)
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._active_index: int = 0  # primary active index
        self._last_active_id: str = task_ids[0] if task_ids else ""
        self._active_task_ids: set[str] = set()  # tracks all currently in_progress tasks
        self._last_activated_time: dict[str, float] = {}  # task_id → monotonic timestamp

    @property
    def current_task_id(self) -> str:
        """The currently active task_id (most recently activated)."""
        with self._lock:
            return self._last_active_id

    @property
    def active_task_ids(self) -> set[str]:
        """Set of all currently in_progress task_ids."""
        with self._lock:
            return set(self._active_task_ids)

    def advance_to(self, index: int) -> None:
        """Advance active task to the given index (from plan step updates)."""
        with self._lock:
            self._advance_to_unlocked(index)

    def _advance_to_unlocked(self, index: int) -> None:
        """Internal advance without lock — caller must hold self._lock."""
        if 0 <= index < len(self._task_ids):
            task_id = self._task_ids[index]
            self._active_index = index
            self._last_active_id = task_id
            self._active_task_ids.add(task_id)
            self._last_activated_time[task_id] = time.monotonic()

    def resolve(self, acp_event: ACPEvent | None = None) -> str:
        """Resolve the task_id for a given ACP event.

        Uses plan step status transitions to infer which task is currently active.
        If the event contains a PLAN_UPDATE, uses the first in_progress entry index.
        When multiple tasks are active, returns the most recently activated one.

        Falls back to last known active task_id.
        """
        with self._lock:
            if acp_event is not None:
                from src.acp.models import ACPEventType
                if acp_event.event_type == ACPEventType.PLAN_UPDATE and acp_event.plan:
                    # Find all in_progress entries and advance to the latest one
                    for idx, entry in enumerate(acp_event.plan.entries):
                        if entry.status == "in_progress":
                            self._advance_to_unlocked(idx)
                            # Don't break — track all active entries

            return self._last_active_id

    def mark_active(self, task_id: str) -> None:
        """Explicitly mark a task_id as active."""
        with self._lock:
            if task_id in self._task_ids:
                self._last_active_id = task_id
                self._active_index = self._task_ids.index(task_id)
                self._active_task_ids.add(task_id)
                self._last_activated_time[task_id] = time.monotonic()

    def mark_inactive(self, task_id: str) -> None:
        """Mark a task_id as no longer active (completed/failed).

        If the deactivated task was the last_active_id, falls back to
        the most recently activated remaining task, or the first task_id.
        """
        with self._lock:
            self._active_task_ids.discard(task_id)
            self._last_activated_time.pop(task_id, None)
            if self._last_active_id == task_id:
                # Fall back to most recently activated remaining task
                if self._active_task_ids:
                    best = max(
                        self._active_task_ids,
                        key=lambda tid: self._last_activated_time.get(tid, 0),
                    )
                    self._last_active_id = best
                    if best in self._task_ids:
                        self._active_index = self._task_ids.index(best)
                elif self._task_ids:
                    # No active tasks — keep last known index
                    pass


class TaskOrchestrator:
    """Orchestrates task-level card sessions for a single engine execution.

    Not a singleton — one instance per engine execution (e.g., per Deep/Spec run).
    """

    def __init__(
        self,
        chat_id: str,
        session_creator: Callable[[str], CardSession],
        registry: TaskRegistry | None = None,
        *,
        bridge_factory: Callable[[Dispatchable], StreamBridge] | None = None,
        max_task_cards: int = 8,
    ) -> None:
        self._chat_id = chat_id
        self._registry = registry or TaskRegistry()
        self._session_creator = session_creator
        self._bridge_factory: Callable[[Dispatchable], StreamBridge] | None = bridge_factory
        self._max_task_cards = max_task_cards

        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._sessions: dict[str, SessionRotator | CardSession] = {}
        self._bridges: dict[str, StreamBridge] = {}  # per-task stream bridges
        self._thinking_session: CardSession | None = None
        self._plan_received = threading.Event()
        self._closed_event = threading.Event()
        self._fallback_mode = False
        self._fallback_session: CardSession | None = None
        self._resolver: TaskIdResolver | None = None
        self._subagent_task_ids: set[str] = set()
        self._subagent_summaries: dict[str, dict] = {}
        self._subagent_progress: dict[str, str] = {}
        self._terminal_task_ids: set[str] = set()
        self._terminal_task_summaries: dict[str, str] = {}
        # Physical card sessions that have received a lifecycle terminal event.
        # Logical overflow children are tracked separately so they cannot revive
        # or prematurely close a card shared with an active sibling.
        self._finalized_task_ids: set[str] = set()
        self._tool_task_bindings: dict[str, str] = {}

        # Debounce state for broadcast
        self._last_broadcast_time: float = 0
        self._pending_broadcast: bool = False
        self._pending_broadcast_task_ids: set[str] = set()
        self._broadcast_timer: threading.Timer | None = None

        # Flood-prevention: overflow task_ids map to the last session's task_id
        self._overflow_target: dict[str, str] = {}  # overflow_task_id → target_task_id
        self._overflow_separator_sent: set[str] = set()  # tracks first dispatch per overflow task

        # Registered plan tasks. Visible plan tasks get their own card as soon as
        # the plan is known; overflow tasks are folded into the final visible card.
        self._plan_tasks_pending: list[dict] = []  # ordered list of {task_id,name,status} not yet built
        self._plan_visible_task_ids: list[str] = []  # first N (max_task_cards) task_ids that may build cards
        self._thinking_finalized: bool = False  # whether thinking session was archived already

        # Task-level rotation counters (protected by self._lock)
        self._rotation_counts: dict[str, int] = {}

        # Subscribe to registry changes for auto-broadcast
        self._registry.subscribe(self._on_registry_status_change)

    @classmethod
    def from_settings(
        cls,
        chat_id: str,
        session_creator: Callable[[str], CardSession],
        thinking_session: SessionRotator | CardSession,
        *,
        bridge_class: type[StreamBridge] | None = None,
    ) -> TaskOrchestrator:
        """Factory: create an orchestrator from project settings.

        Reads `card.task_level_cards_enabled` and `card.max_task_cards` from settings,
        constructs the bridge_factory conditionally, and wires up the thinking session.

        Args:
            chat_id: The chat ID for this execution.
            session_creator: Callable that creates a CardSession given a task_id.
            thinking_session: The pre-plan session (or rotator) to use.
            bridge_class: Optional bridge class to use as factory. If None and
                         task_level_cards_enabled, no per-task bridges are created.
        """
        from src.config import get_settings

        settings = get_settings()
        multi_card_enabled = settings.card.task_level_cards_enabled

        bridge_factory: Callable[[Dispatchable], StreamBridge] | None = None
        if multi_card_enabled and bridge_class is not None:
            bridge_factory = bridge_class

        orchestrator = cls(
            chat_id=chat_id,
            session_creator=session_creator,
            bridge_factory=bridge_factory,
            max_task_cards=settings.card.max_task_cards,
        )
        orchestrator.set_thinking_session(thinking_session)
        return orchestrator

    @property
    def registry(self) -> TaskRegistry:
        """Access the task registry."""
        return self._registry

    @property
    def is_fallback_mode(self) -> bool:
        """Whether orchestrator is in single-session fallback mode."""
        return self._fallback_mode

    @property
    def has_plan(self) -> bool:
        """Whether a plan has been received (task sessions created)."""
        return self._plan_received.is_set()

    def reset(self) -> None:
        """Reset orchestrator state for a new cycle/iteration.

        Archives all active task sessions (sends ARCHIVED), then resets internal
        state to allow fresh plan detection. Used by Spec mode at cycle boundaries.
        """
        with self._lock:
            had_plan = self._plan_received.is_set() and not self._fallback_mode
            sessions_to_close = list(self._sessions.values()) if had_plan else []
            self._sessions.clear()
            self._bridges.clear()
            self._overflow_target.clear()
            self._overflow_separator_sent.clear()
            self._rotation_counts.clear()
            self._plan_tasks_pending.clear()
            self._plan_visible_task_ids.clear()
            self._thinking_finalized = False
            # Reset shared flags under lock to prevent TOCTOU with dispatch_to_task
            self._plan_received.clear()
            self._fallback_mode = False
            self._fallback_session = None
            self._resolver = None
            self._subagent_task_ids.clear()
            self._subagent_summaries.clear()
            self._subagent_progress.clear()
            self._terminal_task_ids.clear()
            self._terminal_task_summaries.clear()
            self._finalized_task_ids.clear()
            self._tool_task_bindings.clear()
            self._pending_broadcast_task_ids.clear()

        # Archive sessions OUTSIDE lock (I/O)
        for session in sessions_to_close:
            try:
                session.dispatch(CardEvent.archived("orchestrator_reset"))
            except Exception:
                logger.debug("TaskOrchestrator.reset: error archiving session")

    @property
    def resolver(self) -> TaskIdResolver | None:
        """Access the task_id resolver (available after on_plan_received)."""
        return self._resolver

    @property
    def active_session_count(self) -> int:
        """Number of active task sessions."""
        with self._lock:
            return len(self._sessions)

    def get_bridge(self, task_id: str) -> StreamBridge | None:
        """Get the stream bridge for a specific task (if bridge_factory was set)."""
        with self._lock:
            return self._bridges.get(task_id)

    def set_thinking_session(self, session: CardSession) -> None:
        """Set the thinking-phase session for pre-plan event routing.

        This session receives all events until on_plan_received() is called,
        at which point it is archived (completed) and per-task sessions take over.
        """
        self._thinking_session = session

    def dispatch_to_thinking(self, event: CardEvent) -> None:
        """Route an event to the thinking-phase session.

        Used before on_plan_received() is called. After plan reception,
        callers should use dispatch_to_task() instead.
        """
        if self._closed_event.is_set():
            return
        if self._thinking_session is not None:
            self._thinking_session.dispatch(event)

    def on_plan_received(self, plan_tasks: list[dict]) -> None:
        """Handle plan reception — register tasks and create per-task cards.

        Plan tasks are first-class execution cards: as soon as the agent exposes
        a multi-task plan, every visible plan item gets a Feishu message card.
        Later ``task`` tool calls are bound back to these plan cards when their
        label matches, instead of requiring a separate subagent identity.

        Falls back to single-session mode if plan is empty/invalid — in that case,
        the thinking session becomes the fallback session.

        Args:
            plan_tasks: List of dicts with at least 'task_id' and 'name' keys.
                       Falls back to single-session mode if empty or invalid.
        """
        if self._closed_event.is_set():
            return

        if not plan_tasks or not isinstance(plan_tasks, list):
            logger.info("TaskOrchestrator: no valid tasks in plan, entering fallback mode")
            self._enter_fallback_mode()
            return

        # Validate task format
        valid_tasks = []
        for t in plan_tasks:
            if isinstance(t, dict) and t.get("task_id") and t.get("name"):
                valid_tasks.append(t)

        if not valid_tasks:
            logger.info("TaskOrchestrator: no parseable tasks, entering fallback mode")
            self._enter_fallback_mode()
            return

        self._plan_received.set()

        # Register all tasks (registry SSOT — needed for task_list rendering once cards appear)
        for t in valid_tasks:
            self._registry.register(
                task_id=t["task_id"],
                name=t["name"],
                status=t.get("status", "pending"),
            )

        # Create resolver for task_id inference
        task_ids = [t["task_id"] for t in valid_tasks]
        self._resolver = TaskIdResolver(task_ids)

        # Compute overflow mapping up-front (the first max_cards tasks may build cards;
        # the rest will be folded into the last visible card whenever it gets created).
        max_cards = self._max_task_cards
        visible_ids = task_ids[:max_cards]
        overflow_target_id = visible_ids[-1] if visible_ids else None
        with self._lock:
            self._plan_tasks_pending = list(valid_tasks)
            self._plan_visible_task_ids = list(visible_ids)
            if overflow_target_id is not None:
                for t in valid_tasks[max_cards:]:
                    self._overflow_target[t["task_id"]] = overflow_target_id

        for t in valid_tasks[:max_cards]:
            try:
                self._ensure_task_session(t["task_id"])
            except Exception:
                logger.warning(
                    "TaskOrchestrator: eager session for plan task_id=%s failed",
                    t["task_id"], exc_info=True,
                )
                self._enter_fallback_mode()
                return

        logger.info(
            "TaskOrchestrator: plan registered with %d tasks (%d visible cards, %d overflow)",
            len(valid_tasks), len(visible_ids), max(0, len(valid_tasks) - len(visible_ids)),
        )

    def _ensure_task_session(self, task_id: str) -> bool:
        """Create a CardSession for ``task_id`` if it does not already exist.

        Returns True if a session exists (or was just created); False if the task is
        not in the registered plan, was an overflow target, or session creation failed.
        Idempotent: calling multiple times for the same task_id is a no-op.
        """
        if self._closed_event.is_set() or self._fallback_mode:
            return False

        with self._lock:
            if task_id in self._sessions:
                return True
            # Overflow tasks never get their own session — their target visible
            # session is created when the plan is received.
            if task_id in self._overflow_target:
                return False
            if task_id not in self._plan_visible_task_ids:
                # Not part of the registered plan (e.g. unknown id) — caller should fallback.
                return False
            should_finalize_thinking = not self._thinking_finalized

        # I/O outside the lock
        try:
            self._create_task_session(task_id)
        except Exception:
            logger.warning(
                "TaskOrchestrator: _create_task_session failed for task_id=%s, entering fallback",
                task_id, exc_info=True,
            )
            self._enter_fallback_mode()
            return False

        # If this newly-built session is the overflow visible-target, dispatch
        # the "flood-merged" notices for every overflow task it absorbs. We do
        # this once, at the moment the target session first appears.
        with self._lock:
            overflow_task_ids = [
                ot_id for ot_id, target in self._overflow_target.items()
                if target == task_id
            ]
            target_session = self._sessions.get(task_id)
        if target_session is not None and overflow_task_ids:
            for ot_id in overflow_task_ids:
                ot_item = self._registry.get(ot_id)
                ot_name = ot_item.name if ot_item else ot_id
                msg = UI_TEXT["orch_flood_merged"].format(task_name=ot_name)
                block_id = f"_flood_{ot_id}"
                try:
                    target_session.dispatch(CardEvent.text_started(block_id))
                    target_session.dispatch(CardEvent.text_delta(block_id, msg))
                    target_session.dispatch(CardEvent.text_done(block_id))
                except Exception:
                    logger.debug("Error dispatching flood notice for %s", ot_id)

        # First task session created → archive thinking session with plan summary.
        if should_finalize_thinking:
            with self._lock:
                if self._thinking_finalized:
                    return True  # someone beat us to it
                self._thinking_finalized = True
                pending = list(self._plan_tasks_pending)
                visible_count = len(self._plan_visible_task_ids)
                overflow_count = max(0, len(pending) - visible_count)
            task_names = [t["name"] for t in pending]
            self._finalize_thinking_session(task_names, overflow_count=overflow_count)

        return True

    def dispatch_to_task(self, task_id: str, event: CardEvent) -> None:
        """Route an event to the specific task's session.

        If task_id is unknown, falls back to the most recently active in_progress session
        (or the fallback/thinking session) rather than silently dropping the event.
        In fallback mode, dispatches to the single fallback session.
        When dispatching to an overflow target for the first time, inserts a visual separator.
        """
        if self._closed_event.is_set():
            return

        if self._fallback_mode:
            if self._fallback_session is not None:
                self._fallback_session.dispatch(event)
            return

        # Resolve overflow mapping (flood-prevention)
        resolved_id = self._overflow_target.get(task_id, task_id)
        is_overflow = task_id in self._overflow_target
        with self._lock:
            if task_id in self._terminal_task_ids:
                return

        # Idempotent safety net: visible plan tasks are normally created when
        # the plan arrives; late dynamic tasks may still need materialization.
        self._ensure_task_session(resolved_id)

        # Re-check fallback (session creation may have triggered _enter_fallback_mode on failure)
        if self._fallback_mode:
            if self._fallback_session is not None:
                self._fallback_session.dispatch(event)
            return

        with self._lock:
            session = self._sessions.get(resolved_id)
            if resolved_id in self._finalized_task_ids:
                return
            # Atomically check-then-add overflow separator flag under lock
            should_insert_separator = (
                is_overflow
                and task_id not in self._overflow_separator_sent
                and session is not None
            )
            is_first_overflow = should_insert_separator and len(self._overflow_separator_sent) == 0
            overflow_display_index = len(self._overflow_separator_sent)  # 0-based count before add
            if should_insert_separator:
                self._overflow_separator_sent.add(task_id)

        if session is None:
            # Fallback: route to most recently active in_progress task
            logger.warning(
                "TaskOrchestrator.dispatch_to_task: unknown task_id=%s, routing to active session",
                task_id,
            )
            fallback_session = self._find_active_session()
            if fallback_session is not None:
                fallback_session.dispatch(event)
            return

        # Insert overflow separator on first dispatch for this overflow task
        # Fold: only display full separator for the first 2 overflow tasks;
        # starting from the 3rd, dispatch a single collapsed count notice instead.
        if should_insert_separator:
            _MAX_VISIBLE_OVERFLOW = 2
            if overflow_display_index < _MAX_VISIBLE_OVERFLOW:
                task_item = self._registry.get(task_id)
                sep_task_name = task_item.name if task_item else task_id
                # Resolve status emoji for the overflow task
                status_key = f"orch_task_status_{task_item.status}" if task_item else "orch_task_status_pending"
                status_emoji = UI_TEXT.get(status_key, "⏳")
                sep_block_id = f"_sep_{task_id}"
                try:
                    session.dispatch(CardEvent(
                        type=CardEventType.SECTION_SEPARATOR,
                        payload={
                            "task_name": sep_task_name,
                            "block_id": sep_block_id,
                            "is_first_overflow": is_first_overflow,
                            "status_emoji": status_emoji,
                        },
                    ))
                except Exception:
                    logger.debug("TaskOrchestrator: error dispatching overflow separator for %s", task_id)
            elif overflow_display_index == _MAX_VISIBLE_OVERFLOW:
                # First folded item: emit collapsed notice with remaining count
                total_overflow = len(self._overflow_target)
                remaining = total_overflow - _MAX_VISIBLE_OVERFLOW
                if remaining > 0:
                    collapsed_msg = UI_TEXT["orch_overflow_collapsed"].format(count=remaining)
                    collapsed_block_id = "_sep_collapsed"
                    try:
                        session.dispatch(CardEvent.text_started(collapsed_block_id))
                        session.dispatch(CardEvent.text_delta(collapsed_block_id, collapsed_msg))
                        session.dispatch(CardEvent.text_done(collapsed_block_id))
                    except Exception:
                        logger.debug("TaskOrchestrator: error dispatching collapsed notice")

        session.dispatch(event)

    def _find_active_session(self) -> SessionRotator | CardSession | None:
        """Find the most recently active (in_progress) task session for fallback routing."""
        if self._resolver is not None:
            active_ids = self._resolver.active_task_ids
            with self._lock:
                for tid in active_ids:
                    if tid in self._sessions:
                        return self._sessions[tid]
        # Last resort: any session or thinking session
        with self._lock:
            if self._sessions:
                return next(iter(self._sessions.values()))
        return self._thinking_session

    def broadcast_status_change(self, task_id: str, new_status: TaskStatus) -> None:
        """Update a task's status and refresh the affected task card.

        Uses debounce to coalesce rapid consecutive status changes.
        """
        if self._closed_event.is_set():
            return

        self._registry.update_status(task_id, new_status)
        # The actual targeted refresh is triggered via the subscribe callback.

    def handle_plan_update(self, acp_event: ACPEvent, fallback_bridge: StreamBridge) -> None:
        """Unified plan detection + status broadcast entry point for renderers.

        Encapsulates the full plan-detection logic that was previously duplicated
        in Deep/Spec renderers:
        1. Check if PLAN_UPDATE event with sufficient entries
        2. Convert to task dicts and call on_plan_received() if threshold met
        3. Broadcast status changes for all entries

        Renderers only need to call this single method on every PLAN_UPDATE event.

        Args:
            acp_event: The ACP event (should be PLAN_UPDATE type).
            fallback_bridge: The bridge for fallback routing (unused here, kept for interface consistency).
        """
        if self._closed_event.is_set():
            return

        from src.acp.models import ACPEventType
        if acp_event.event_type != ACPEventType.PLAN_UPDATE:
            return

        if not acp_event.plan or not acp_event.plan.entries:
            return

        entries = acp_event.plan.entries

        # First PLAN_UPDATE with enough steps: create per-task sessions
        if not self._plan_received.is_set() and not self._fallback_mode:
            from src.card.task_registry import tasks_from_plan_entries
            if len(entries) >= _MIN_TASKS_FOR_MULTI_CARD:
                task_dicts = tasks_from_plan_entries(entries)
                if len(task_dicts) >= _MIN_TASKS_FOR_MULTI_CARD:
                    self.on_plan_received(task_dicts)

        # Broadcast task status changes from plan entries
        if self._plan_received.is_set() and not self._fallback_mode:
            for idx, entry in enumerate(entries):
                entry_task_id = f"step_{idx}"
                if entry.status == "in_progress":
                    # Ensure the visible card exists, then mark the task running.
                    # Overflow tasks route to their target; _ensure_task_session handles both.
                    resolved_id = self._overflow_target.get(entry_task_id, entry_task_id)
                    self._ensure_task_session(resolved_id)
                    self.broadcast_status_change(entry_task_id, "in_progress")
                elif entry.status == "completed":
                    self._finalize_task_session(entry_task_id, "completed")
                elif entry.status == "failed":
                    self._finalize_task_session(entry_task_id, "failed")

    def route_acp_event(self, acp_event: ACPEvent, fallback_bridge: StreamBridge) -> None:
        """Unified ACP event routing — resolve task_id and dispatch to the correct bridge.

        This is the single entry point for renderers to route ACP events in multi-card mode.
        Internally: resolve task_id → find per-task bridge → bridge.on_event(acp_event).

        If orchestrator has no plan yet or is in fallback mode, the event goes to fallback_bridge.
        If per-task bridges are not configured (no bridge_factory), dispatches the converted
        CardEvent directly to the task session.

        Args:
            acp_event: The raw ACP event to route.
            fallback_bridge: The bridge to use when routing cannot be resolved
                            (pre-plan phase or fallback mode).
        """
        if self._closed_event.is_set():
            return

        if self._route_bound_source_event(acp_event, fallback_bridge):
            return
        if self._route_collaboration_event(acp_event, fallback_bridge):
            return
        if self._route_bound_tool_task_event(acp_event):
            return
        if self._route_agent_task_event(acp_event):
            return

        # Before plan reception or in fallback mode → use fallback bridge
        if not self._plan_received.is_set() or self._fallback_mode:
            fallback_bridge.on_event(acp_event)
            return

        # Resolve which task this event belongs to
        if self._resolver is None:
            fallback_bridge.on_event(acp_event)
            return

        task_id = self._resolver.resolve(acp_event)
        if not task_id:
            fallback_bridge.on_event(acp_event)
            return

        # Ensure the per-task session (and its bridge) exists before routing.
        # Covers bridge-first routing paths and late dynamic task materialization.
        # NOTE: only ensure the *visible* (non-overflow) session here; overflow events
        # still flow through dispatch_to_task below to keep separator insertion correct.
        if task_id not in self._overflow_target:
            self._ensure_task_session(task_id)
        if self._fallback_mode:
            fallback_bridge.on_event(acp_event)
            return

        # Route to per-task bridge if available
        with self._lock:
            bridge = self._bridges.get(task_id)

        if bridge is not None:
            bridge.on_event(acp_event)
        else:
            # No per-task bridge — dispatch converted CardEvent directly to session
            from src.card.events import card_event_from_acp
            card_evt = card_event_from_acp(acp_event)
            self.dispatch_to_task(task_id, card_evt)

    def _route_bound_source_event(
        self,
        acp_event: ACPEvent,
        fallback_bridge: StreamBridge,
    ) -> bool:
        """Route source-tagged child text, thought, and media to its owner."""
        from src.acp.models import ACPEventType

        if acp_event.event_type not in {
            ACPEventType.TEXT_CHUNK,
            ACPEventType.THOUGHT_CHUNK,
            ACPEventType.IMAGE_CHUNK,
        }:
            return False
        source_id = str(getattr(acp_event, "source_id", "") or "").strip()
        if not source_id:
            return False

        with self._lock:
            if source_id in self._subagent_task_ids:
                task_id = source_id
            else:
                task_id = self._tool_task_bindings.get(source_id, "")
            resolved_id = self._overflow_target.get(task_id, task_id)
            finalized = (
                task_id in self._terminal_task_ids
                or resolved_id in self._finalized_task_ids
            )
            bridge = self._bridges.get(resolved_id)

        if not task_id:
            return False
        if not finalized:
            if bridge is not None:
                bridge.on_event(acp_event)
                return True
            if acp_event.event_type is not ACPEventType.IMAGE_CHUNK:
                from src.card.events import card_event_from_acp

                self.dispatch_to_task(task_id, card_event_from_acp(acp_event))
                return True

        # A closed task card rejects later mutations, and a task without its
        # own media bridge cannot upload the payload. Preserve the artifact on
        # the fallback card with an explicit owner instead of silently losing
        # it or attributing it to whichever task happens to be active.
        if acp_event.event_type is not ACPEventType.IMAGE_CHUNK:
            fallback_bridge.on_event(acp_event)
            return True

        task = self._registry.get(task_id)
        label = str(getattr(task, "name", "") or task_id).strip()
        image = getattr(acp_event, "image", None)
        if image is not None and label:
            name = image.name
            if label not in name:
                name = f"{label} · {name}"
            acp_event = replace(
                acp_event,
                image=replace(image, name=name[:120]),
            )
        fallback_bridge.on_event(acp_event)
        return True

    def _route_collaboration_event(
        self,
        acp_event: ACPEvent,
        fallback_bridge: StreamBridge,
    ) -> bool:
        """Project Codex collaboration snapshots onto stable child cards."""
        tool_call = getattr(acp_event, "tool_call", None)
        collaboration_tool = getattr(tool_call, "collaboration_tool", "")
        if (
            tool_call is None
            or not isinstance(collaboration_tool, str)
            or not collaboration_tool.strip()
        ):
            return False

        states_by_source = {
            str(item.get("source_id") or "").strip(): item
            for item in getattr(tool_call, "subagent_states", ())
            if isinstance(item, Mapping)
            and str(item.get("source_id") or "").strip()
        }
        source_ids = [
            str(source_id or "").strip()
            for source_id in getattr(tool_call, "collaboration_receivers", ())
            if str(source_id or "").strip()
        ]
        source_ids.extend(
            source_id
            for source_id in states_by_source
            if source_id not in source_ids
        )

        if not source_ids:
            if str(getattr(tool_call, "status", "") or "").lower() == "failed":
                fallback_bridge.on_event(acp_event)
            return True

        label = self._extract_agent_task_label(tool_call)
        opaque_ids = collect_subagent_opaque_ids(tool_call)
        # Materialize every receiver before applying terminal snapshots. This
        # is required when the card cap folds later receivers into the first
        # physical child card: an early completed receiver must not close that
        # shared card before its running sibling is registered.
        running_status_changed = False
        for source_id in source_ids:
            with self._lock:
                if source_id in self._terminal_task_ids:
                    continue
                known_subagent = source_id in self._subagent_task_ids
            if not known_subagent:
                self.create_subagent_session(source_id, label)
            with self._lock:
                known_subagent = source_id in self._subagent_task_ids
            if not known_subagent:
                fallback_bridge.on_event(acp_event)
                return True

        for source_id in source_ids:
            with self._lock:
                if source_id in self._terminal_task_ids:
                    continue
            self._rename_task_from_tool_label(source_id, tool_call)
            state = states_by_source.get(source_id, {})
            raw_status = str(state.get("status") or "running").strip().lower()
            status = _SUBAGENT_PROVIDER_STATUS.get(raw_status, "in_progress")
            message = sanitize_subagent_display_text(
                state.get("message"),
                fallback="",
                max_chars=180,
                opaque_ids=opaque_ids,
            )
            default_progress = _SUBAGENT_PROVIDER_PROGRESS.get(
                raw_status,
                "执行中",
            )
            progress = message or default_progress
            if (
                status in _TERMINAL_TASK_STATUSES
                and progress in _TRANSIENT_SUBAGENT_PROGRESS
            ):
                progress = default_progress
            self._publish_subagent_progress(source_id, progress)
            if status in _TERMINAL_TASK_STATUSES:
                self._finalize_task_session(
                    source_id,
                    status,
                    summary=progress,
                )
            else:
                item = self._registry.get(source_id)
                if item is not None and item.status != "in_progress":
                    self._registry.update_status(
                        source_id,
                        "in_progress",
                        notify=False,
                    )
                    running_status_changed = True
        if running_status_changed:
            self._broadcast_subagent_task_list()
        return True

    def _publish_subagent_progress(self, task_id: str, progress: str) -> bool:
        """Show one safe, de-duplicated collaboration status on its child card."""
        progress = str(progress or "").strip()
        if not progress:
            return False
        with self._lock:
            if task_id in self._terminal_task_ids:
                return False
            if self._subagent_progress.get(task_id) == progress:
                return False
            resolved_id = self._overflow_target.get(task_id, task_id)
            if resolved_id in self._finalized_task_ids:
                return False
            self._subagent_progress[task_id] = progress
            session = self._sessions.get(resolved_id)
        if session is not None:
            try:
                session.dispatch(CardEvent.progress_updated(0, 0, progress))
            except Exception:
                logger.debug(
                    "TaskOrchestrator: failed to publish child progress task_id=%s",
                    task_id,
                    exc_info=True,
                )
        return True

    def route_or_fallback(self, acp_event: ACPEvent, fallback_bridge: StreamBridge) -> bool:
        """Unified routing predicate + dispatch for renderers.

        Encapsulates the repeated condition:
            if has_plan and not is_fallback_mode: route_acp_event(...)
            else: fallback_bridge.on_event(...)

        Returns:
            True if event was routed through the orchestrator (multi-card path).
            False if event was sent to fallback_bridge (single-card path).
        """
        if self.is_agent_task_event(acp_event):
            self.route_acp_event(acp_event, fallback_bridge)
            return True
        if self._plan_received.is_set() and not self._fallback_mode:
            self.route_acp_event(acp_event, fallback_bridge)
            return True
        fallback_bridge.on_event(acp_event)
        return False

    def _on_registry_status_change(self, task_id: str, new_status: TaskStatus) -> None:
        """Callback from TaskRegistry when status changes — triggers targeted refresh."""
        self._schedule_broadcast({task_id})

    def _schedule_broadcast(self, task_ids: set[str]) -> None:
        """Schedule a debounced TASK_LIST_UPDATED refresh for affected task cards."""
        with self._lock:
            if self._closed_event.is_set():
                return
            self._pending_broadcast_task_ids.update(task_ids)
            now = time.monotonic()
            elapsed_ms = (now - self._last_broadcast_time) * 1000

            if elapsed_ms >= _BROADCAST_DEBOUNCE_MS:
                # Enough time has passed, broadcast immediately (release lock first)
                pass
            else:
                # Schedule delayed broadcast
                if self._broadcast_timer is not None:
                    self._broadcast_timer.cancel()
                remaining = (_BROADCAST_DEBOUNCE_MS - elapsed_ms) / 1000
                self._broadcast_timer = threading.Timer(remaining, self._do_broadcast)
                self._broadcast_timer.daemon = True
                self._broadcast_timer.start()
                return

        # Immediate broadcast (outside lock to avoid deadlock with session.dispatch)
        self._do_broadcast()

    def _do_broadcast(self) -> None:
        """Refresh TASK_LIST_UPDATED only on cards affected by status changes."""
        with self._lock:
            if self._closed_event.is_set():
                return
            self._last_broadcast_time = time.monotonic()
            pending_task_ids = set(self._pending_broadcast_task_ids)
            self._pending_broadcast_task_ids.clear()
            sessions = []
            seen_targets: set[str] = set()
            for pending_task_id in pending_task_ids:
                target_id = self._overflow_target.get(pending_task_id, pending_task_id)
                if target_id in seen_targets or target_id in self._finalized_task_ids:
                    continue
                session = self._sessions.get(target_id)
                if session is None:
                    continue
                seen_targets.add(target_id)
                sessions.append((target_id, session))
            if not sessions:
                return

        snapshot = self._registry.get_snapshot()
        tasks_payload = [
            {"task_id": s.task_id, "name": s.name, "status": s.status}
            for s in snapshot
        ]

        for task_id, session in sessions:
            with self._lock:
                if task_id in self._finalized_task_ids:
                    continue
            event: CardEvent = CardEvent(
                type=CardEventType.TASK_LIST_UPDATED,
                payload={"tasks": tasks_payload, "current_task_id": task_id},
            )
            try:
                session.dispatch(event)
            except Exception:
                logger.debug("Broadcast to task_id=%s failed", task_id, exc_info=True)

    def _create_task_session(self, task_id: str, *, is_subagent: bool = False) -> None:
        """Create a CardSession for a task and bind it."""
        session = self._session_creator(task_id)
        if is_subagent:
            self._apply_subagent_metadata(session, task_id)

        with self._lock:
            self._sessions[task_id] = session
            # Create per-task bridge if bridge_factory is configured
            if self._bridge_factory is not None:
                self._bridges[task_id] = self._bridge_factory(session)

        # Dispatch initial TASK_LIST_UPDATED so the card renders the task list header
        snapshot = self._registry.get_snapshot()
        tasks_payload = [
            {"task_id": s.task_id, "name": s.name, "status": s.status}
            for s in snapshot
        ]
        event: CardEvent = CardEvent(
            type=CardEventType.TASK_LIST_UPDATED,
            payload={"tasks": tasks_payload, "current_task_id": task_id},
        )
        session.dispatch(event)

    def rotate_task_session(self, task_id: str) -> bool:
        """Trigger continuation card for a task (when content exceeds byte_budget).

        Freezes the current task session (dispatches ARCHIVED), creates a new continuation
        session, and re-dispatches TASK_LIST_UPDATED to the new session.

        Returns True if rotation succeeded, False if task_id not found or already closed.
        """
        if self._closed_event.is_set():
            return False

        with self._lock:
            session = self._sessions.get(task_id)
            old_bridge = self._bridges.get(task_id)
        if session is None:
            return False

        # Freeze old bridge
        if old_bridge is not None:
            try:
                old_bridge.close_open_blocks()
            except Exception:
                logger.debug("Error closing bridge during rotation for task %s", task_id)

        task_item = self._registry.get(task_id)
        task_name = task_item.name if task_item else task_id

        # Phase 1: Create new session (I/O, outside lock)
        new_session = self._create_continuation_session(task_id, task_name, session)
        if new_session is None:
            return False

        # Register backfill callback BEFORE swap (eliminates TOCTOU window)
        rotation_count = self._rotation_counts.get(task_id, 0) + 1
        self._register_backfill_callback(session, new_session, task_name, rotation_count)

        # Phase 2: Atomic swap (inside lock, NO I/O)
        with self._lock:
            self._rotation_counts[task_id] = rotation_count
            self._sessions[task_id] = new_session
            if self._bridge_factory is not None:
                self._bridges[task_id] = self._bridge_factory(new_session)

        # Phase 3: Archive old + initialize new (I/O, outside lock)
        self._archive_old_session(session, task_name, rotation_count)
        self._initialize_continuation_card(new_session, task_id, session, rotation_count)

        logger.info("TaskOrchestrator: rotated task session for task_id=%s (续 %d)", task_id, rotation_count)
        return True

    def _create_continuation_session(
        self,
        task_id: str,
        task_name: str,
        old_session: SessionRotator | CardSession,
    ) -> CardSession | None:
        """Phase 1: Create a new continuation session. Returns None on failure."""
        try:
            new_session = self._session_creator(task_id)
            old_meta = getattr(old_session, "_metadata", None)
            if old_meta is not None:
                seq = self._rotation_counts.get(task_id, 0) + 2
                from dataclasses import replace
                new_session._metadata = replace(
                    new_session._metadata,
                    continuation_seq=seq - 1,
                    card_sequence=seq,
                    session_started_at=getattr(old_session, "session_started_at", new_session.session_started_at),
                    bridge_phrase="续接：",
                    is_subagent=old_meta.is_subagent,
                    parent_card_seq=old_meta.parent_card_seq,
                )
            return new_session
        except Exception:
            logger.warning(
                "TaskOrchestrator: session_creator failed for task_id=%s, rotation aborted",
                task_id, exc_info=True,
            )
            # Dispatch degradation notice on old session
            try:
                degrade_msg = UI_TEXT["orch_rotation_failed_notice"]
                old_session.dispatch(CardEvent.text_started("_rotation_failed"))
                old_session.dispatch(CardEvent.text_delta("_rotation_failed", degrade_msg))
                old_session.dispatch(CardEvent.text_done("_rotation_failed"))
            except Exception:
                logger.warning(
                    "TaskOrchestrator: failed to dispatch rotation degradation for task_id=%s",
                    task_id,
                )
            return None

    def _archive_old_session(
        self,
        old_session: SessionRotator | CardSession,
        task_name: str,
        rotation_count: int,
    ) -> None:
        """Phase 3: Archive old session with continuation navigation text."""
        msg = format_task_continuation_link(
            task_name=task_name,
            rotation_count=rotation_count,
            new_msg_id=None,
        )
        try:
            old_session.dispatch(CardEvent.text_started("_continuation"))
            old_session.dispatch(CardEvent.text_delta("_continuation", msg))
            old_session.dispatch(CardEvent.text_done("_continuation"))
            old_session.dispatch(CardEvent.archived(bridge_phrase=f"续接 #{rotation_count + 1} ↓"))
        except Exception:
            logger.debug("Error archiving task session for rotation, task_id=%s (name=%s)", task_name, task_name, exc_info=True)

    def _register_backfill_callback(
        self,
        old_session: SessionRotator | CardSession,
        new_session: CardSession,
        task_name: str,
        rotation_count: int,
    ) -> None:
        """Inject a BackfillHook on new_session to backfill deep-link on old card.

        Must be called BEFORE Phase 2 swap so the hook is registered while
        new_session is not yet exposed to concurrent dispatch (eliminates TOCTOU).
        """
        hook = BackfillHook(
            old_session_ref=weakref.ref(old_session),
            task_name=task_name,
            rotation_count=rotation_count,
        )
        new_session.add_hook(hook)

    def _initialize_continuation_card(
        self,
        new_session: CardSession,
        task_id: str,
        old_session: SessionRotator | CardSession,
        rotation_count: int,
    ) -> None:
        """Dispatch TASK_LIST_UPDATED and continuation hint to the new card."""
        # Task list header
        snapshot = self._registry.get_snapshot()
        tasks_payload = [
            {"task_id": s.task_id, "name": s.name, "status": s.status}
            for s in snapshot
        ]
        event: CardEvent = CardEvent(
            type=CardEventType.TASK_LIST_UPDATED,
            payload={"tasks": tasks_payload, "current_task_id": task_id},
        )
        new_session.dispatch(event)

        # Back-link hint
        page = rotation_count + 1
        old_msg_id = getattr(old_session, "delivered_message_id", "") or ""
        if old_msg_id:
            hint_msg = UI_TEXT["orch_back_link"].format(msg_id=old_msg_id)
        else:
            hint_msg = UI_TEXT["orch_continuation_hint"].format(page=page)
        try:
            new_session.dispatch(CardEvent.text_started("_continuation_hint"))
            new_session.dispatch(CardEvent.text_delta("_continuation_hint", hint_msg))
            new_session.dispatch(CardEvent.text_done("_continuation_hint"))
        except Exception:
            logger.debug("Error dispatching continuation hint for task_id=%s", task_id, exc_info=True)

    def _enter_fallback_mode(self) -> None:
        """Enter single-session fallback mode (no multi-card).

        If a thinking session exists, it becomes the fallback session.
        Dispatches a visible warning to inform the user.
        """
        self._fallback_mode = True
        self._plan_received.set()  # Prevent further plan processing
        if self._thinking_session is not None and self._fallback_session is None:
            self._fallback_session = self._thinking_session
        # Dispatch visible warning to fallback session
        if self._fallback_session is not None:
            try:
                warn_id = "_fallback_warn"
                self._fallback_session.dispatch(CardEvent.text_started(warn_id))
                self._fallback_session.dispatch(
                    CardEvent.text_delta(warn_id, UI_TEXT["orch_fallback_warning"])
                )
                self._fallback_session.dispatch(CardEvent.text_done(warn_id))
            except Exception:
                logger.debug("Error dispatching fallback warning", exc_info=True)
        logger.info("TaskOrchestrator: fallback mode — using single session")

    def _finalize_thinking_session(self, task_names: list[str], *, overflow_count: int = 0) -> None:
        """Archive the thinking session with a single concise plan summary.

        Merges the former _notify_thinking_of_tasks + _archive_thinking_session
        into one pass to avoid redundant task-list duplication on the card.
        """
        if self._thinking_session is None:
            return
        try:
            snapshot = self._registry.get_snapshot()
            tasks_payload = [
                {"task_id": s.task_id, "name": s.name, "status": s.status}
                for s in snapshot
            ]
            if tasks_payload:
                self._thinking_session.dispatch(CardEvent(
                    type=CardEventType.TASK_LIST_UPDATED,
                    payload={"tasks": tasks_payload, "current_task_id": ""},
                ))
            task_count = len(task_names)
            # Fold task list when >5 items to save card space
            if task_count > 5:
                visible_list = "\n".join(f"  {i+1}. {name}" for i, name in enumerate(task_names[:5]))
                task_list = visible_list + f"\n  …及 {task_count - 5} 项更多"
            else:
                task_list = "\n".join(f"  {i+1}. {name}" for i, name in enumerate(task_names))
            summary = UI_TEXT["orch_plan_archived"].format(
                task_count=task_count,
                task_list=task_list,
            )
            if overflow_count > 0:
                independent_count = task_count - overflow_count
                transition = "\n" + UI_TEXT["orch_plan_transition_hint_overflow"].format(
                    independent_count=independent_count,
                    merged_count=overflow_count,
                )
            else:
                transition = "\n" + UI_TEXT["orch_plan_transition_hint_no_link"]
            block_id = "_plan_summary"
            self._thinking_session.dispatch(CardEvent.text_started(block_id))
            self._thinking_session.dispatch(CardEvent.text_delta(block_id, summary + transition))
            self._thinking_session.dispatch(CardEvent.text_done(block_id))
            self._thinking_session.dispatch(CardEvent.archived())
        except Exception:
            logger.debug("Error finalizing thinking session", exc_info=True)
        self._thinking_session = None

    def set_fallback_session(self, session: CardSession) -> None:
        """Set the fallback session for single-session mode."""
        self._fallback_session = session

    def _route_agent_task_event(self, acp_event: ACPEvent) -> bool:
        """Route agent/subagent tool calls into an independent task card."""
        from src.acp.models import ACPEventType
        from src.card.events import card_event_from_acp

        if acp_event.event_type not in {
            ACPEventType.TOOL_CALL_START,
            ACPEventType.TOOL_CALL_UPDATE,
            ACPEventType.TOOL_CALL_DONE,
        }:
            return False

        tool_call = getattr(acp_event, "tool_call", None)
        if tool_call is None:
            return False

        activity_source_id = str(
            getattr(tool_call, "subagent_source_id", "") or ""
        ).strip()
        activity = str(
            getattr(tool_call, "subagent_activity", "") or ""
        ).strip().lower()
        tool_id = activity_source_id or str(
            getattr(tool_call, "id", "") or ""
        ).strip()
        if not tool_id:
            return False

        with self._lock:
            known_subagent = tool_id in self._subagent_task_ids
            finalized = tool_id in self._terminal_task_ids
        if not known_subagent and not self._is_agent_task(tool_call):
            return False
        if finalized:
            return True

        if (
            not known_subagent
            and self._plan_received.is_set()
            and self._is_task_tool(tool_call)
            and self._is_generic_task_label(self._extract_agent_task_label(tool_call))
        ):
            return False

        if not known_subagent:
            self.create_subagent_session(tool_id, self._extract_agent_task_label(tool_call))
            with self._lock:
                known_subagent = tool_id in self._subagent_task_ids
            if not known_subagent:
                return False

        if activity_source_id:
            self._route_subagent_activity(
                task_id=tool_id,
                activity=activity,
                acp_event=acp_event,
            )
            return True

        if acp_event.event_type != ACPEventType.TOOL_CALL_DONE:
            self._rename_task_from_tool_label(tool_id, tool_call)

        self.dispatch_to_task(tool_id, card_event_from_acp(acp_event))

        if acp_event.event_type == ACPEventType.TOOL_CALL_DONE:
            status = str(getattr(tool_call, "status", "") or "").strip().lower()
            content = str(getattr(tool_call, "content", "") or "").strip()
            if status == "failed":
                self._finalize_task_session(
                    tool_id,
                    "failed",
                    summary=summarize_tool_call_content(content, fallback=self._extract_agent_task_label(tool_call)),
                )
                self._publish_subagent_summary(tool_call, status="failed")
            else:
                self._finalize_task_session(
                    tool_id,
                    "completed",
                    summary=summarize_tool_call_content(content),
                )
                self._publish_subagent_summary(tool_call, status="completed")
        else:
            self._publish_subagent_summary(tool_call, status="running")
        return True

    def _route_subagent_activity(
        self,
        *,
        task_id: str,
        activity: str,
        acp_event: ACPEvent,
    ) -> None:
        """Render a child lifecycle operation without closing the child task."""
        tool_call = acp_event.tool_call
        if tool_call is None:
            return
        self._rename_task_from_tool_label(task_id, tool_call)

        provider_status = str(getattr(tool_call, "status", "") or "").lower()

        if activity == "started":
            progress = (
                "已启动"
                if provider_status == "completed"
                else ("启动未完成" if provider_status == "failed" else "正在启动")
            )
        elif activity == "interacted":
            progress = (
                "已与主 Agent 交互"
                if provider_status == "completed"
                else (
                    "交互未完成"
                    if provider_status == "failed"
                    else "正在与主 Agent 交互"
                )
            )
        elif activity == "interrupted":
            progress = (
                "已中断"
                if provider_status == "completed"
                else (
                    "中断未完成"
                    if provider_status == "failed"
                    else "正在中断"
                )
            )
        else:
            progress = "动态已更新"
        self._publish_subagent_progress(task_id, progress)

        if activity == "interrupted" and provider_status == "completed":
            self._finalize_task_session(
                task_id,
                "cancelled",
                summary="子代理已中断",
            )
            return

        item = self._registry.get(task_id)
        if item is not None and item.status != "in_progress":
            self._registry.update_status(task_id, "in_progress", notify=False)
            self._broadcast_subagent_task_list()

    def create_subagent_session(self, task_id: str, name: str) -> None:
        """Create a new session for a detected subagent task.

        Called when TOOL_STARTED with agent/subagent tool name is detected.
        """
        if self._closed_event.is_set() or self._fallback_mode:
            return

        overflow_target = ""
        with self._lock:
            if task_id in self._subagent_task_ids:
                return
            visible_subagents = [
                existing_id
                for existing_id in self._sessions
                if existing_id in self._subagent_task_ids
                and existing_id not in self._overflow_target
            ]
            if len(visible_subagents) >= self._max_task_cards:
                overflow_target = visible_subagents[-1] if visible_subagents else ""
                if not overflow_target:
                    return
                self._overflow_target[task_id] = overflow_target
            self._subagent_task_ids.add(task_id)

        # Register every subtask for the shared task list, but stop creating new
        # Feishu messages at the configured cap. Overflow details reuse the last
        # visible subagent card through the normal overflow routing path.
        self._registry.register(task_id=task_id, name=name, status="in_progress")
        if not overflow_target:
            self._create_task_session(task_id, is_subagent=True)
        self._broadcast_subagent_task_list()

    def _rename_task_from_tool_label(self, task_id: str, tool_call) -> bool:
        """Update a task card label when a later tool event reveals the real description."""
        label = self._extract_agent_task_label(tool_call)
        if self._is_generic_task_label(label):
            return False

        current = self._registry.get(task_id)
        if current is None or current.name == label:
            return False
        if not self._is_generic_task_label(current.name):
            return False

        updated = self._registry.update_name(task_id, label)
        if updated is None:
            return False

        with self._lock:
            session = self._sessions.get(task_id)
        if session is not None:
            try:
                session.dispatch(CardEvent.tool_model_changed(unit_label=label))
            except Exception:
                logger.debug("TaskOrchestrator: failed to update task card label for %s", task_id, exc_info=True)

        if task_id in self._subagent_task_ids:
            self._broadcast_subagent_task_list()
        else:
            self._schedule_broadcast({task_id})
        return True

    def _broadcast_subagent_task_list(self) -> None:
        """Refresh task-list blocks only on child task cards.

        New parallel task cards should see the growing task list, but parent plan
        task cards must remain frozen from subtask progress.
        """
        snapshot = self._registry.get_snapshot()
        tasks_payload = [
            {"task_id": s.task_id, "name": s.name, "status": s.status}
            for s in snapshot
        ]
        with self._lock:
            sessions = [
                (task_id, self._sessions[task_id])
                for task_id in self._subagent_task_ids
                if task_id in self._sessions and task_id not in self._finalized_task_ids
            ]

        for task_id, session in sessions:
            try:
                session.dispatch(CardEvent(
                    type=CardEventType.TASK_LIST_UPDATED,
                    payload={"tasks": tasks_payload, "current_task_id": task_id},
                ))
            except Exception:
                logger.debug("Broadcast to subagent task_id=%s failed", task_id, exc_info=True)

    def _route_bound_tool_task_event(self, acp_event: ACPEvent) -> bool:
        """Route a tool_call already bound to a plan task back to that task card."""
        from src.acp.models import ACPEventType
        from src.card.events import card_event_from_acp

        if acp_event.event_type not in {
            ACPEventType.TOOL_CALL_START,
            ACPEventType.TOOL_CALL_UPDATE,
            ACPEventType.TOOL_CALL_DONE,
        }:
            return False
        tool_call = getattr(acp_event, "tool_call", None)
        tool_id = str(getattr(tool_call, "id", "") or "").strip()
        if not tool_id:
            return False

        with self._lock:
            bound_task_id = self._tool_task_bindings.get(tool_id)

        if not bound_task_id and self._is_task_tool(tool_call):
            bound_task_id = self._match_plan_task_for_tool(tool_call)
            if bound_task_id:
                with self._lock:
                    self._tool_task_bindings[tool_id] = bound_task_id

        if not bound_task_id:
            return False

        if acp_event.event_type in {ACPEventType.TOOL_CALL_START, ACPEventType.TOOL_CALL_UPDATE}:
            self.broadcast_status_change(bound_task_id, "in_progress")
            if self._resolver is not None:
                self._resolver.mark_active(bound_task_id)

        self.dispatch_to_task(bound_task_id, card_event_from_acp(acp_event))

        if acp_event.event_type == ACPEventType.TOOL_CALL_DONE:
            status = str(getattr(tool_call, "status", "") or "").strip().lower()
            content = str(getattr(tool_call, "content", "") or "").strip()
            self._finalize_task_session(
                bound_task_id,
                "failed" if status == "failed" else "completed",
                summary=summarize_tool_call_content(content),
            )
            # Keep the stable media binding through orchestrator teardown.
            # Some providers emit source-tagged artifacts after the terminal
            # tool frame; those must degrade with explicit task attribution.
        return True

    def _match_plan_task_for_tool(self, tool_call) -> str:
        """Best-effort bind a ``task`` tool call to an existing plan item."""
        if not self._plan_received.is_set() or self._fallback_mode:
            return ""
        label = self._extract_agent_task_label(tool_call)
        normalized_label = self._normalize_task_label(label)
        if not normalized_label:
            return ""

        with self._lock:
            candidate_ids = list(self._plan_visible_task_ids)
        for snapshot in self._registry.get_snapshot():
            if snapshot.task_id not in candidate_ids:
                continue
            if snapshot.task_id in self._subagent_task_ids:
                continue
            candidate = self._normalize_task_label(snapshot.name)
            if not candidate:
                continue
            if candidate == normalized_label or candidate in normalized_label or normalized_label in candidate:
                return snapshot.task_id
        return ""

    @staticmethod
    def _normalize_task_label(value: str) -> str:
        return "".join(ch for ch in str(value).lower() if ch.isalnum())

    def _apply_subagent_metadata(self, session: CardSession, task_id: str) -> None:
        """Mark orchestrator-created subagent sessions with v2 parent/sequence metadata."""
        task_item = self._registry.get(task_id)
        if task_item is None:
            return
        parent_session = self._thinking_session or self._fallback_session or self._find_active_session()
        if parent_session is None:
            return
        metadata = getattr(session, "_metadata", None)
        if metadata is None:
            return
        from dataclasses import replace
        with self._lock:
            subagent_count = len([s for s in self._sessions.values() if getattr(s, "is_subagent", False)])
        branch_id = chr(ord("a") + subagent_count)
        parent_seq = str(getattr(parent_session, "sequence", 1))
        session._metadata = replace(
            metadata,
            unit_id=task_id,
            unit_kind="subagent",
            unit_label=task_item.name,
            card_sequence=f"{parent_seq}.{branch_id}",
            session_started_at=getattr(parent_session, "session_started_at", session.session_started_at),
            is_subagent=True,
            parent_card_seq=parent_seq,
            bridge_phrase=None,
        )

    def _publish_subagent_summary(self, tool_call, *, status: str) -> None:
        task_id = str(getattr(tool_call, "id", "") or "").strip()
        if not task_id:
            return

        with self._lock:
            session = self._sessions.get(task_id)
        metadata = getattr(session, "_metadata", None)
        existing = self._subagent_summaries.get(task_id, {})
        label = self._extract_agent_task_label(tool_call)
        tool = self._extract_agent_tool_name(tool_call)
        if existing:
            existing_label = str(existing.get("label") or "").strip()
            if (
                existing_label
                and not self._is_generic_task_label(existing_label)
                and not is_unhelpful_display_label(existing_label)
            ):
                label = existing_label
            if tool in {"subagent", "子代理"}:
                tool = str(existing.get("tool") or tool)
        summary = {
            **existing,
            "label": label,
            "tool": tool,
            "status": status,
        }
        if metadata is not None:
            summary["sequence"] = metadata.card_sequence
            if metadata.model_name:
                summary["model"] = metadata.model_name
        self._subagent_summaries[task_id] = summary

    @staticmethod
    def _is_agent_task(tool_call) -> bool:
        title = str(getattr(tool_call, "title", "") or "").strip().lower()
        kind = str(getattr(tool_call, "kind", "") or "").strip().lower()
        content = str(getattr(tool_call, "content", "") or "").strip()
        source_id = getattr(tool_call, "subagent_source_id", "")
        if isinstance(source_id, str) and source_id.strip():
            return True
        collaboration_tool = getattr(tool_call, "collaboration_tool", "")
        if isinstance(collaboration_tool, str) and collaboration_tool.strip():
            return True
        if kind == "agent" or title in _AGENT_TOOL_TITLES:
            return True
        return kind == "other" and "子代理：" in content

    @staticmethod
    def _is_task_tool(tool_call) -> bool:
        return str(getattr(tool_call, "title", "") or "").strip().lower() == "task"

    @staticmethod
    def _extract_agent_task_label(tool_call) -> str:
        opaque_ids = collect_subagent_opaque_ids(tool_call)
        path = str(getattr(tool_call, "subagent_path", "") or "").strip()
        if path:
            label = path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
            if label:
                return sanitize_subagent_display_text(
                    label,
                    opaque_ids=opaque_ids,
                    max_chars=60,
                    fallback="子任务",
                )
        return sanitize_subagent_display_text(
            extract_tool_call_label(
                tool_call,
                generic_labels=_GENERIC_TASK_LABELS,
                fallback="子任务",
                max_chars=240,
            ),
            fallback="子任务",
            max_chars=60,
            opaque_ids=opaque_ids,
        )

    @staticmethod
    def _is_generic_task_label(value: str) -> bool:
        return str(value or "").strip().lower() in _GENERIC_TASK_LABELS

    @staticmethod
    def _extract_agent_tool_name(tool_call) -> str:
        return extract_agent_tool_name(tool_call)

    @classmethod
    def is_agent_task_event(cls, acp_event: ACPEvent) -> bool:
        tool_call = getattr(acp_event, "tool_call", None)
        return tool_call is not None and cls._is_agent_task(tool_call)

    def _finalize_task_session(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        summary: str = "",
    ) -> None:
        """Mark one task card terminal so later task updates no longer patch it."""
        if status not in _TERMINAL_TASK_STATUSES:
            return

        with self._lock:
            resolved_id = self._overflow_target.get(task_id, task_id)
            if (
                task_id in self._terminal_task_ids
                or resolved_id in self._finalized_task_ids
            ):
                return
            self._terminal_task_ids.add(task_id)
            self._terminal_task_summaries[task_id] = summary
            is_subagent = task_id in self._subagent_task_ids
        if is_subagent:
            self._registry.update_status(task_id, status, notify=False)
            self._broadcast_subagent_task_list()
        else:
            self.broadcast_status_change(task_id, status)
        if self._resolver is not None:
            self._resolver.mark_inactive(task_id)

        with self._lock:
            grouped_task_ids = {
                resolved_id,
                *(
                    overflow_id
                    for overflow_id, target_id in self._overflow_target.items()
                    if target_id == resolved_id
                ),
            }
        grouped_items = [
            self._registry.get(grouped_id)
            for grouped_id in grouped_task_ids
        ]
        if any(
            item is not None and item.status not in _TERMINAL_TASK_STATUSES
            for item in grouped_items
        ):
            return

        statuses = {
            item.status
            for item in grouped_items
            if item is not None
        }
        if "failed" in statuses:
            terminal_status: TaskStatus = "failed"
        elif "cancelled" in statuses:
            terminal_status = "cancelled"
        else:
            terminal_status = "completed"
        terminal_summary = next(
            (
                self._terminal_task_summaries.get(grouped_id, "")
                for grouped_id in grouped_task_ids
                if (item := self._registry.get(grouped_id)) is not None
                and item.status == terminal_status
                and self._terminal_task_summaries.get(grouped_id, "")
            ),
            summary,
        )

        with self._lock:
            if resolved_id in self._finalized_task_ids:
                return
            self._finalized_task_ids.add(resolved_id)
            session = self._sessions.get(resolved_id)

        if session is not None:
            if terminal_status == "failed":
                event = CardEvent.failed(terminal_summary)
            elif terminal_status == "cancelled":
                event = CardEvent.cancelled(
                    reason=terminal_summary or "subagent_interrupted"
                )
            else:
                event = CardEvent.completed(summary=terminal_summary)
            try:
                session.dispatch(event)
            except Exception:
                logger.debug("TaskOrchestrator: failed to finalize task_id=%s", task_id, exc_info=True)

    def finalize_unfinished_subagents(
        self,
        *,
        status: TaskStatus = "cancelled",
        summary: str = UNCONFIRMED_SUBAGENT_SUMMARY,
    ) -> tuple[str, ...]:
        """Settle live child tasks without inventing a successful terminal.

        Parent completion proves only that the parent prompt ended. A child
        that never supplied an authoritative terminal ``agentsStates`` frame
        is therefore cancelled/unknown rather than presented as completed.
        """
        if status not in _TERMINAL_TASK_STATUSES:
            return ()
        with self._lock:
            task_ids = tuple(self._subagent_task_ids)
        unfinished = tuple(
            task_id
            for task_id in task_ids
            if (item := self._registry.get(task_id)) is not None
            and item.status not in _TERMINAL_TASK_STATUSES
        )
        for task_id in unfinished:
            self._finalize_task_session(task_id, status, summary=summary)
        return unfinished

    def finish_with_summary(self, summary: str, *, failed: bool = False) -> None:
        """Close task cards, then create one fresh final summary card."""
        if self._closed_event.is_set():
            return
        self._closed_event.set()

        self._cancel_broadcast_timer()
        self._registry.unsubscribe(self._on_registry_status_change)

        with self._lock:
            sessions = list(self._sessions.items())
            bridges = list(self._bridges.values())
            self._sessions.clear()
            self._bridges.clear()
            finalized = set(self._finalized_task_ids)

        self._close_bridges(bridges)

        terminal_event = CardEvent.failed(summary) if failed else CardEvent.completed()
        for task_id, session in sessions:
            if task_id in finalized:
                continue
            try:
                self._run_with_timeout(
                    lambda s=session, e=terminal_event: s.dispatch(e),  # type: ignore[misc]
                    timeout=5.0,
                )
            except Exception:
                logger.debug("Error closing task session", exc_info=True)

        self._create_final_summary_session(summary, failed=failed)
        self._fallback_session = None
        logger.info("TaskOrchestrator: finished with summary for chat_id=%s", self._chat_id)

    def _cancel_broadcast_timer(self) -> None:
        with self._lock:
            if self._broadcast_timer is not None:
                self._broadcast_timer.cancel()
                self._broadcast_timer = None
            self._pending_broadcast_task_ids.clear()

    def _close_bridges(self, bridges: list[StreamBridge]) -> None:
        for bridge in bridges:
            try:
                self._run_with_timeout(bridge.close_open_blocks, timeout=5.0)
            except Exception:
                logger.debug("Error closing bridge", exc_info=True)

    def _create_final_summary_session(self, summary: str, *, failed: bool) -> None:
        if not summary:
            return

        self._registry.register(
            task_id=_FINAL_SUMMARY_TASK_ID,
            name=UI_TEXT["orch_final_summary_task_name"],
            status="in_progress",
        )
        try:
            session = self._session_creator(_FINAL_SUMMARY_TASK_ID)
        except Exception:
            logger.debug("TaskOrchestrator: failed to create final summary session", exc_info=True)
            return

        snapshot = self._registry.get_snapshot()
        tasks_payload = [
            {"task_id": s.task_id, "name": s.name, "status": s.status}
            for s in snapshot
        ]
        session.dispatch(CardEvent.started())
        session.dispatch(CardEvent(
            type=CardEventType.TASK_LIST_UPDATED,
            payload={"tasks": tasks_payload, "current_task_id": _FINAL_SUMMARY_TASK_ID},
        ))
        block_id = "_final_summary"
        session.dispatch(CardEvent.text_started(block_id))
        session.dispatch(CardEvent.text_delta(block_id, summary))
        session.dispatch(CardEvent.text_done(block_id))
        session.dispatch(CardEvent.failed(summary) if failed else CardEvent.completed(summary=summary))

    def close(
        self,
        *,
        terminal_status: TaskStatus = "completed",
        summary: str = "",
    ) -> None:
        """Close all sessions and clean up.

        Includes timeout protection: bridge.close_open_blocks() and session.dispatch()
        are each given a 5s timeout. On timeout, the operation is skipped to prevent
        blocking the caller indefinitely.
        """
        if self._closed_event.is_set():
            return
        self._closed_event.set()

        self._cancel_broadcast_timer()

        # Unsubscribe from registry
        self._registry.unsubscribe(self._on_registry_status_change)

        # Close all task sessions and bridges
        with self._lock:
            sessions = list(self._sessions.items())
            bridges = list(self._bridges.values())
            self._sessions.clear()
            self._bridges.clear()
            finalized = set(self._finalized_task_ids)

        self._close_bridges(bridges)

        if terminal_status == "failed":
            terminal_event = CardEvent.failed(summary)
        elif terminal_status == "cancelled":
            terminal_event = CardEvent.cancelled(reason=summary or None)
        else:
            terminal_event = CardEvent.completed(summary=summary)
        for task_id, session in sessions:
            if task_id in finalized:
                continue
            try:
                self._run_with_timeout(
                    lambda s=session: s.dispatch(terminal_event),  # type: ignore[misc]
                    timeout=5.0,
                )
            except Exception:
                logger.debug("Error closing task session", exc_info=True)

        self._fallback_session = None

        logger.info("TaskOrchestrator: closed for chat_id=%s", self._chat_id)

    def _run_with_timeout(self, fn: Callable[[], None], *, timeout: float) -> None:
        """Run a close callable in an isolated daemon thread.

        Python cannot stop a running thread. A timed-out close worker therefore
        must be a daemon: abandoning a permanently blocked bridge then cannot
        keep the service process alive during interpreter shutdown.
        """
        future: concurrent.futures.Future[None] = concurrent.futures.Future()

        def _invoke() -> None:
            if not future.set_running_or_notify_cancel():
                return
            try:
                fn()
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(None)

        worker = threading.Thread(
            target=_invoke,
            name=f"orch-close-{self._chat_id}",
            daemon=True,
        )
        worker.start()
        try:
            future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            logger.warning("TaskOrchestrator: close operation timed out after %.1fs", timeout)
            raise TimeoutError(f"Operation timed out after {timeout}s")
