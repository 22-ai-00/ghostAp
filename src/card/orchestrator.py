"""Task-level card routing for programming sessions."""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

from src.card.events import CardEvent, CardEventType
from src.card.events.projector import (
    TERMINAL_AGENT_STATUSES,
    attribute_subagent_image,
    extract_agent_task_label,
    is_agent_task,
    is_generic_task_label,
    is_task_tool,
    project_activity,
    project_collaboration,
)
from src.card.task_registry import TaskRegistry, TaskStatus
from src.card.tool_display import summarize_tool_call_content
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
UNCONFIRMED_SUBAGENT_SUMMARY = "父任务已结束，子任务终态未确认"
_TERMINAL_TASK_STATUSES = frozenset({"completed", "failed", "cancelled"})
_SUBAGENT_GENERATION_WAIT_S = 40.0


class TaskIdResolver:
    """Thread-safe resolver for the most recently active plan task."""

    def __init__(self, task_ids: list[str]) -> None:
        self._task_ids = list(task_ids)
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
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
            self._last_active_id = task_id
            self._active_task_ids.add(task_id)
            self._last_activated_time[task_id] = time.monotonic()

    def resolve(self, acp_event: ACPEvent | None = None) -> str:
        """Resolve an event to the latest active task."""
        with self._lock:
            if acp_event is not None:
                from src.acp.models import ACPEventType
                if acp_event.event_type == ACPEventType.PLAN_UPDATE and acp_event.plan:
                    for idx, entry in enumerate(acp_event.plan.entries):
                        if entry.status == "in_progress":
                            self._advance_to_unlocked(idx)

            return self._last_active_id

    def mark_active(self, task_id: str) -> None:
        """Explicitly mark a task_id as active."""
        with self._lock:
            if task_id in self._task_ids:
                self._last_active_id = task_id
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


class TaskOrchestrator:
    """Own task cards and route one programming execution's events."""

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
        self._subagent_progress: dict[str, str] = {}
        self._terminal_task_ids: set[str] = set()
        self._terminal_task_summaries: dict[str, str] = {}
        self._subagent_generation_claims: dict[str, threading.Event] = {}
        self._lifecycle_epoch = 0
        # Physical card sessions that have received a lifecycle terminal event.
        # Logical overflow children are tracked separately so they cannot revive
        # or prematurely close a card shared with an active sibling.
        self._finalized_task_ids: set[str] = set()
        self._tool_task_bindings: dict[str, str] = {}

        # Debounce state for broadcast
        self._last_broadcast_time: float = 0
        self._pending_broadcast_task_ids: set[str] = set()
        self._broadcast_timer: threading.Timer | None = None

        # Flood-prevention: overflow task_ids map to the last session's task_id
        self._overflow_target: dict[str, str] = {}  # overflow_task_id → target_task_id
        self._overflow_separator_sent: set[str] = set()  # tracks first dispatch per overflow task

        # Registered plan tasks. Visible plan tasks get their own card as soon as
        # the plan is known; overflow tasks are folded into the final visible card.
        self._plan_visible_task_ids: list[str] = []  # first N (max_task_cards) task_ids that may build cards
        self._thinking_finalized: bool = False  # whether thinking session was archived already

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
        """Create an orchestrator from card settings."""
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
            self._lifecycle_epoch += 1
            generation_claims = list(
                self._subagent_generation_claims.values()
            )
            self._subagent_generation_claims.clear()
            had_plan = self._plan_received.is_set() and not self._fallback_mode
            sessions_to_close = list(self._sessions.values()) if had_plan else []
            self._sessions.clear()
            self._bridges.clear()
            self._overflow_target.clear()
            self._overflow_separator_sent.clear()
            self._plan_visible_task_ids.clear()
            self._thinking_finalized = False
            # Reset shared flags under lock to prevent TOCTOU with dispatch_to_task
            self._plan_received.clear()
            self._fallback_mode = False
            self._fallback_session = None
            self._resolver = None
            self._subagent_task_ids.clear()
            self._subagent_progress.clear()
            self._terminal_task_ids.clear()
            self._terminal_task_summaries.clear()
            self._finalized_task_ids.clear()
            self._tool_task_bindings.clear()
            self._pending_broadcast_task_ids.clear()

        for claim in generation_claims:
            claim.set()
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

    def set_thinking_session(self, session: CardSession) -> None:
        """Set the thinking-phase session for pre-plan event routing.

        This session receives all events until on_plan_received() is called,
        at which point it is archived (completed) and per-task sessions take over.
        """
        self._thinking_session = session

    def _task_list_event(
        self,
        current_task_id: str,
        *,
        force_running: str = "",
    ) -> CardEvent:
        tasks = [
            {
                "task_id": item.task_id,
                "name": item.name,
                "status": "in_progress" if item.task_id == force_running else item.status,
            }
            for item in self._registry.get_snapshot()
        ]
        return CardEvent(
            type=CardEventType.TASK_LIST_UPDATED,
            payload={"tasks": tasks, "current_task_id": current_task_id},
        )

    @staticmethod
    def _terminal_event(status: TaskStatus, summary: str = "") -> CardEvent:
        if status == "failed":
            return CardEvent.failed(summary)
        if status == "cancelled":
            return CardEvent.cancelled(reason=summary or None)
        return CardEvent.completed(summary=summary)

    @staticmethod
    def _dispatch_text(session: CardSession, block_id: str, text: str) -> None:
        session.dispatch(CardEvent.text_started(block_id))
        session.dispatch(CardEvent.text_delta(block_id, text))
        session.dispatch(CardEvent.text_done(block_id))

    @staticmethod
    def _close_bridge_quietly(bridge: StreamBridge | None, context: str) -> None:
        if bridge is None:
            return
        try:
            bridge.close_open_blocks()
        except Exception:
            logger.debug("TaskOrchestrator: failed to close %s bridge", context, exc_info=True)

    def _generation_is_current(self, task_id: str, lifecycle_epoch: int) -> bool:
        with self._lock:
            return (
                self._lifecycle_epoch == lifecycle_epoch
                and not self._closed_event.is_set()
                and task_id in self._subagent_task_ids
            )

    def _discard_generation(
        self,
        session: CardSession | None,
        bridge: StreamBridge | None,
        task_id: str,
        *,
        archive: bool,
    ) -> None:
        if archive and session is not None:
            try:
                session.dispatch(CardEvent.archived("stale_subagent_generation"))
            except Exception:
                logger.debug(
                    "TaskOrchestrator: failed to archive stale generation task_id=%s",
                    task_id,
                    exc_info=True,
                )
        self._close_bridge_quietly(bridge, f"generation {task_id}")

    def on_plan_received(self, plan_tasks: list[dict]) -> None:
        """Register a plan and eagerly create its visible task cards."""
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
                    self._dispatch_text(target_session, block_id, msg)
                except Exception:
                    logger.debug("Error dispatching flood notice for %s", ot_id)

        # First task session created → archive thinking session with plan summary.
        if should_finalize_thinking:
            snapshot = self._registry.get_snapshot()
            with self._lock:
                if self._thinking_finalized:
                    return True  # someone beat us to it
                self._thinking_finalized = True
                visible_count = len(self._plan_visible_task_ids)
                overflow_count = max(0, len(snapshot) - visible_count)
            task_names = [task.name for task in snapshot]
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
                        self._dispatch_text(session, collapsed_block_id, collapsed_msg)
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
                elif entry.status in {"completed", "failed"}:
                    self._finalize_task_session(entry_task_id, entry.status)

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
            acp_event = attribute_subagent_image(acp_event, label)
        fallback_bridge.on_event(acp_event)
        return True

    def _route_collaboration_event(
        self,
        acp_event: ACPEvent,
        fallback_bridge: StreamBridge,
    ) -> bool:
        """Project Codex collaboration snapshots onto stable child cards."""
        projection = project_collaboration(acp_event)
        if projection is None:
            return False
        if not projection.agents:
            if projection.failed_without_receiver:
                fallback_bridge.on_event(acp_event)
            return True
        # Materialize every receiver before applying terminal snapshots. This
        # is required when the card cap folds later receivers into the first
        # physical child card: an early completed receiver must not close that
        # shared card before its running sibling is registered.
        running_status_changed = False
        for agent in projection.agents:
            if projection.starts_new_generation and not self._reopen_subagent_generation(
                agent.source_id,
            ):
                fallback_bridge.on_event(acp_event)
                return True
            with self._lock:
                if agent.source_id in self._terminal_task_ids:
                    continue
                known_subagent = agent.source_id in self._subagent_task_ids
            if not known_subagent:
                self.create_subagent_session(agent.source_id, agent.label)
            with self._lock:
                known_subagent = agent.source_id in self._subagent_task_ids
            if not known_subagent:
                fallback_bridge.on_event(acp_event)
                return True

        for agent in projection.agents:
            source_id = agent.source_id
            with self._lock:
                if source_id in self._terminal_task_ids:
                    continue
            self._rename_task(source_id, agent.label)
            status: TaskStatus = (
                "in_progress" if agent.status == "running" else agent.status
            )  # type: ignore[assignment]
            self._publish_subagent_progress(source_id, agent.progress)
            if agent.status in TERMINAL_AGENT_STATUSES:
                self._finalize_task_session(
                    source_id,
                    status,
                    summary=agent.progress,
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

    def _reopen_subagent_generation(self, task_id: str) -> bool:
        """Open a fresh physical card when a stable child id starts new work."""
        while True:
            with self._lock:
                if task_id not in self._subagent_task_ids:
                    return True
                resolved_id = self._overflow_target.get(task_id, task_id)
                task_is_terminal = task_id in self._terminal_task_ids
                physical_is_finalized = (
                    resolved_id in self._finalized_task_ids
                )
                is_subagent_card = resolved_id in self._subagent_task_ids
                if not task_is_terminal and not physical_is_finalized:
                    return True
                claim = self._subagent_generation_claims.get(resolved_id)
                if claim is None:
                    claim = threading.Event()
                    self._subagent_generation_claims[resolved_id] = claim
                    lifecycle_epoch = self._lifecycle_epoch
                    owns_claim = True
                else:
                    lifecycle_epoch = self._lifecycle_epoch
                    owns_claim = False

            if not owns_claim:
                if not claim.wait(timeout=_SUBAGENT_GENERATION_WAIT_S):
                    logger.warning(
                        "TaskOrchestrator: timed out waiting for subagent "
                        "generation claim task_id=%s",
                        task_id,
                    )
                    return False
                with self._lock:
                    if self._lifecycle_epoch != lifecycle_epoch:
                        return False
                continue

            try:
                return self._commit_reopened_subagent_generation(
                    task_id=task_id,
                    resolved_id=resolved_id,
                    physical_is_finalized=physical_is_finalized,
                    is_subagent_card=is_subagent_card,
                    lifecycle_epoch=lifecycle_epoch,
                )
            finally:
                with self._lock:
                    current_claim = self._subagent_generation_claims.get(
                        resolved_id
                    )
                    if current_claim is claim:
                        self._subagent_generation_claims.pop(
                            resolved_id,
                            None,
                        )
                        claim.set()

    def _commit_reopened_subagent_generation(
        self,
        *,
        task_id: str,
        resolved_id: str,
        physical_is_finalized: bool,
        is_subagent_card: bool,
        lifecycle_epoch: int,
    ) -> bool:
        new_session = None
        new_bridge = None
        if physical_is_finalized:
            try:
                new_session = self._session_creator(resolved_id)
                if is_subagent_card:
                    self._apply_subagent_metadata(new_session, resolved_id)
                if self._bridge_factory is not None:
                    new_bridge = self._bridge_factory(new_session)
                if not self._generation_is_current(task_id, lifecycle_epoch):
                    self._discard_generation(
                        new_session, new_bridge, task_id, archive=False
                    )
                    return False
                new_session.dispatch(
                    self._task_list_event(task_id, force_running=task_id)
                )
                if not self._generation_is_current(task_id, lifecycle_epoch):
                    self._discard_generation(
                        new_session, new_bridge, task_id, archive=True
                    )
                    return False
            except Exception:
                self._close_bridge_quietly(new_bridge, f"rejected generation {task_id}")
                logger.warning(
                    "TaskOrchestrator: failed to open new subagent generation "
                    "task_id=%s",
                    task_id,
                    exc_info=True,
                )
                return False

        prior_item = self._registry.get(task_id)
        prior_status = prior_item.status if prior_item is not None else None
        self._registry.update_status(
            task_id,
            "in_progress",
            notify=False,
        )
        old_bridge = None
        stale_at_commit = False
        with self._lock:
            if (
                self._lifecycle_epoch != lifecycle_epoch
                or self._closed_event.is_set()
                or task_id not in self._subagent_task_ids
            ):
                stale_at_commit = True
            else:
                self._terminal_task_ids.discard(task_id)
                self._terminal_task_summaries.pop(task_id, None)
                self._subagent_progress.pop(task_id, None)
                if physical_is_finalized:
                    self._finalized_task_ids.discard(resolved_id)
                    old_bridge = self._bridges.get(resolved_id)
                    if new_session is not None:
                        self._sessions[resolved_id] = new_session
                    if (
                        self._bridge_factory is not None
                        and new_bridge is not None
                    ):
                        self._bridges[resolved_id] = new_bridge

        if stale_at_commit:
            if prior_status is not None:
                self._registry.update_status(
                    task_id,
                    prior_status,
                    notify=False,
                )
            self._discard_generation(
                new_session, new_bridge, task_id, archive=True
            )
            return False
        self._close_bridge_quietly(old_bridge, f"old generation {task_id}")
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

        for task_id, session in sessions:
            with self._lock:
                if task_id in self._finalized_task_ids:
                    continue
            try:
                session.dispatch(self._task_list_event(task_id))
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

        session.dispatch(self._task_list_event(task_id))


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
                self._dispatch_text(
                    self._fallback_session,
                    warn_id,
                    UI_TEXT["orch_fallback_warning"],
                )
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
            if self._registry.count:
                self._thinking_session.dispatch(self._task_list_event(""))
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
            self._dispatch_text(
                self._thinking_session, block_id, summary + transition
            )
            self._thinking_session.dispatch(CardEvent.archived())
        except Exception:
            logger.debug("Error finalizing thinking session", exc_info=True)
        self._thinking_session = None


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

        activity = project_activity(acp_event)
        tool_id = (activity.source_id if activity else "") or str(
            getattr(tool_call, "id", "") or ""
        ).strip()
        if not tool_id:
            return False

        with self._lock:
            known_subagent = tool_id in self._subagent_task_ids
            finalized = tool_id in self._terminal_task_ids
        if not known_subagent and not is_agent_task(tool_call):
            return False
        if finalized:
            return True

        if (
            not known_subagent
            and self._plan_received.is_set()
            and is_task_tool(tool_call)
            and is_generic_task_label(extract_agent_task_label(tool_call))
        ):
            return False

        if not known_subagent:
            self.create_subagent_session(tool_id, extract_agent_task_label(tool_call))
            with self._lock:
                known_subagent = tool_id in self._subagent_task_ids
            if not known_subagent:
                return False

        if activity is not None:
            self._route_subagent_activity(
                task_id=tool_id,
                projection=activity,
                tool_call=tool_call,
            )
            return True

        if acp_event.event_type != ACPEventType.TOOL_CALL_DONE:
            self._rename_task_from_tool_label(tool_id, tool_call)

        self.dispatch_to_task(tool_id, card_event_from_acp(acp_event))

        if acp_event.event_type == ACPEventType.TOOL_CALL_DONE:
            status = str(getattr(tool_call, "status", "") or "").strip().lower()
            content = str(getattr(tool_call, "content", "") or "").strip()
            terminal_status: TaskStatus = (
                "failed" if status == "failed" else "completed"
            )
            fallback = (
                extract_agent_task_label(tool_call)
                if terminal_status == "failed"
                else ""
            )
            self._finalize_task_session(
                tool_id,
                terminal_status,
                summary=summarize_tool_call_content(content, fallback=fallback),
            )
        return True

    def _route_subagent_activity(
        self,
        *,
        task_id: str,
        projection,
        tool_call,
    ) -> None:
        """Render a child lifecycle operation without closing the child task."""
        self._rename_task(task_id, projection.label)
        self._publish_subagent_progress(task_id, projection.progress)

        if projection.interrupted:
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
        return self._rename_task(task_id, extract_agent_task_label(tool_call))

    def _rename_task(self, task_id: str, label: str) -> bool:
        if is_generic_task_label(label):
            return False

        current = self._registry.get(task_id)
        if current is None or current.name == label:
            return False
        if not is_generic_task_label(current.name):
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
        with self._lock:
            sessions = [
                (task_id, self._sessions[task_id])
                for task_id in self._subagent_task_ids
                if task_id in self._sessions and task_id not in self._finalized_task_ids
            ]

        for task_id, session in sessions:
            try:
                session.dispatch(self._task_list_event(task_id))
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

        if not bound_task_id and is_task_tool(tool_call):
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
        label = extract_agent_task_label(tool_call)
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


    @classmethod
    def is_agent_task_event(cls, acp_event: ACPEvent) -> bool:
        tool_call = getattr(acp_event, "tool_call", None)
        return tool_call is not None and is_agent_task(tool_call)

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
            if terminal_status == "cancelled" and not terminal_summary:
                terminal_summary = "subagent_interrupted"
            try:
                session.dispatch(
                    self._terminal_event(terminal_status, terminal_summary)
                )
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
            self._lifecycle_epoch += 1
            generation_claims = list(
                self._subagent_generation_claims.values()
            )
            self._subagent_generation_claims.clear()
            sessions = list(self._sessions.items())
            bridges = list(self._bridges.values())
            self._sessions.clear()
            self._bridges.clear()
            finalized = set(self._finalized_task_ids)

        for claim in generation_claims:
            claim.set()
        self._close_bridges(bridges)

        terminal_event = self._terminal_event(terminal_status, summary)
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
