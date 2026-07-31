"""Programming mode adapter: bridges streaming card pattern to CardSession.

Bridges streaming card pattern to CardSession for
ProgrammingHandler.handle_response(). Supports all programming modes:
Coco/Claude/Aiden/Codex/Gemini/Traex/TTADK/Tui2ACP.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Callable

from src.card.events import CardEvent, CardEventType
from src.card.media_bridge import ACPImagePublisher
from src.card.render.live_ticker import LiveTicker
from src.card.session import CardSession
from src.card.session.rotator import SessionRotator
from src.card.state.models import CardMetadata, CardState, TaskListBlock, ToolBlock
from src.card.state.reducer import (
    MAX_COMPLETED_TOOL_BLOCKS,
    MAX_TOTAL_BLOCKS,
    card_state_requires_continuation,
)
from src.card.text_stream import append_stream_text
from src.card.tool_display import (
    collect_subagent_opaque_ids,
    extract_agent_tool_name,
    extract_tool_call_label,
    is_unhelpful_display_label,
    sanitize_subagent_display_text,
    sanitize_tool_failure_detail,
)

if TYPE_CHECKING:
    from src.acp.models import ACPEvent, ACPImageInfo, ToolCallInfo

logger = logging.getLogger(__name__)

# Mode name → (mode_emoji, display_name)
_MODE_DISPLAY: dict[str, tuple[str, str]] = {
    "coco": ("🤖", "Coco"),
    "claude": ("🧠", "Claude"),
    "aiden": ("⚡", "Aiden"),
    "codex": ("📝", "Codex"),
    "gemini": ("💎", "Gemini"),
    "traex": ("🚀", "Traex"),
    "ttadk": ("🛠️", "TTADK"),
    "tui2acp": ("🔌", "Tui2ACP"),
}

_AGENT_TOOL_TITLES = {"agent", "subagent", "task"}
_GENERIC_TASK_LABELS = {"", "agent", "subagent", "task", "子任务"}
_TERMINAL_AGENT_STATUSES = {"completed", "failed", "cancelled"}
_GENERIC_AGENT_FAILURE_DETAIL = "子任务执行失败"
_SUBAGENT_STATE_STATUS = {
    "pendinginit": "running",
    "pending_init": "running",
    "pending": "running",
    "running": "running",
    "completed": "completed",
    "errored": "failed",
    "failed": "failed",
    "interrupted": "cancelled",
    "shutdown": "completed",
    "notfound": "failed",
    "not_found": "failed",
}
_SUBAGENT_STATE_PROGRESS = {
    "pendinginit": "准备中",
    "pending_init": "准备中",
    "pending": "准备中",
    "running": "执行中",
    "completed": "已完成",
    "errored": "执行失败",
    "failed": "执行失败",
    "interrupted": "已中断",
    "shutdown": "已完成并停止",
    "notfound": "状态不可用",
    "not_found": "状态不可用",
}
_TOOL_CARD_EVENT_TYPES = frozenset({
    CardEventType.TOOL_STARTED,
    CardEventType.TOOL_DELTA,
    CardEventType.TOOL_DONE,
    CardEventType.TOOL_FAILED,
})


def build_programming_metadata(
    mode_name: str,
    *,
    tool_name: str | None = None,
    model_name: str | None = None,
    project_name: str | None = None,
    working_dir: str | None = None,
) -> CardMetadata:
    """Build CardMetadata for a programming mode session.

    Args:
        mode_name: One of coco/claude/aiden/codex/gemini/traex/ttadk.
        tool_name: Specific tool name (overrides mode default).
        model_name: Model name to display.
        project_name: Optional project name for header.
        working_dir: Current project/session working directory for v2 header.
    """
    mode_key = mode_name.lower()
    emoji, display = _MODE_DISPLAY.get(mode_key, ("🤖", mode_name))

    return CardMetadata(
        project_name=project_name,
        mode_name=display,
        mode_emoji=emoji,
        tool_name=tool_name or mode_key,
        model_name=model_name,
        engine_type=None,  # Programming mode is not an engine
        working_dir=working_dir,
        programming_text_sections=True,
    )


class ProgrammingCardSession:
    """Wraps CardSession for programming handler's specific needs.

    Includes text batching: TEXT_DELTA events are accumulated and flushed
    at regular intervals (default 0.3s) to avoid overwhelming the Feishu API.
    Structural events (tool start/done, etc.) trigger immediate flush.
    """

    _DEFAULT_FLUSH_INTERVAL = 0.3  # seconds

    def __init__(
        self,
        session: CardSession,
        *,
        flush_interval: float | None = None,
        base_metadata: CardMetadata | None = None,
        image_uploader: Callable[["ACPImageInfo"], str | None] | None = None,
        session_factory: Callable[[CardMetadata], CardSession] | None = None,
        continuation_visibility_timeout: float = 12.0,
    ) -> None:
        self._rotator = SessionRotator(session)
        self._base_metadata = (
            getattr(session, "_metadata", None)
            or base_metadata
            or CardMetadata()
        )
        self._session_factory = session_factory
        self._continuation_visibility_timeout = max(
            0.01,
            float(continuation_visibility_timeout),
        )
        self._dispatch_lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        self._capacity_delivery_failed = False
        self._capacity_failure_reason: str | None = None
        self._failed_retired_sessions: list[CardSession] = []
        self._active_tool_snapshots: dict[str, ToolBlock] = {}
        self._image_publisher = ACPImagePublisher(self, image_uploader)
        self._routed_image_ids: set[str] = set()
        self._text_active = False
        self._active_text_block_id = "_active_text"
        self._pending_text_block_id: str | None = None
        self._reasoning_active = False
        self._active_reasoning_block_id = "_active_reasoning"
        self._text_blocks_by_source: dict[str, str] = {}
        self._reasoning_blocks_by_source: dict[str, str] = {}
        self._active_text_sources: set[str] = set()
        self._active_reasoning_sources: set[str] = set()
        self._reasoning_sources_with_content: set[str] = set()
        self._pending_reasoning_item_breaks: set[str] = set()
        self._flush_interval = flush_interval or self._DEFAULT_FLUSH_INTERVAL
        # Text batching state
        self._pending_text = ""
        self._pending_text_by_block: dict[str, str] = {}
        self._main_text_transcript: list[tuple[str, str]] = []
        self._flush_lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        self._flush_lock_holder = threading.local()  # per-thread flag for lock ownership assertion
        self._flush_timer: threading.Timer | None = None
        self._agent_summaries: dict[str, dict] = {}
        self._text_turn_seq = 0
        self._reasoning_turn_seq = 0
        self._last_tool_boundary_seq = 0
        self._ticker_factory = LiveTicker
        self._ticker = None
        self._last_ticker_update_at: float | None = None
        self._ticker_update_min_interval = 5.0
        # TimerScheduler callbacks must stay lightweight. In production, ticker
        # metadata dispatch is submitted to the shared delivery pool so the
        # scheduler thread never runs reduce/render/delivery inline. Tests that
        # opt into sync delivery keep synchronous ticker dispatch by default for
        # deterministic assertions, but can force async via this private flag.
        self._ticker_dispatch_async = not getattr(session, "_sync_delivery", False)
        self._ticker_executor_factory = None

    @property
    def session(self) -> CardSession:
        return self._rotator.current

    @property
    def closed(self) -> bool:
        return self._rotator.closed or self._rotator.current.closed

    def start(self) -> None:
        """Start the card (creates initial card in Feishu)."""
        self._dispatch_card_event(CardEvent.started())
        self._dispatch_card_event(CardEvent.text_started("_active_text"))
        self._text_active = True
        self._text_turn_seq = max(self._text_turn_seq, 1)
        self._text_blocks_by_source["main"] = "_active_text"
        self._active_text_sources.add("main")
        self._start_ticker()

    def on_event(self, acp_event: "ACPEvent") -> None:
        """Process an ACP event (converts to CardEvent internally).

        Text deltas are batched for efficiency. Structural events flush immediately.
        """
        from src.acp.models import ACPEventType

        card_event = None
        if acp_event.event_type is ACPEventType.IMAGE_CHUNK:
            if self._handle_agent_image_event(acp_event):
                return
            self._flush_now()
            self._last_tool_boundary_seq += 1
            if self._text_active:
                self._close_text_blocks()
            if self._reasoning_blocks_by_source:
                self._close_reasoning_blocks(retire=True)
            self._image_publisher.handle(acp_event)
            if acp_event.image is not None:
                self._routed_image_ids.add(acp_event.image.image_id)
            return
        if getattr(acp_event, "event_type", None).name == "PLAN_UPDATE":
            self._handle_plan_update(acp_event)
            return

        if self._handle_collaboration_event(acp_event):
            return

        if self._handle_subagent_activity_event(acp_event):
            return

        if self._handle_agent_task_event(acp_event):
            return

        card_event = CardEvent.from_acp(acp_event)

        # Text delta: accumulate and schedule flush. ACP turns get stable,
        # per-turn block IDs so a later turn never appends to an earlier one.
        if card_event.type == CardEventType.TEXT_DELTA:
            text = card_event.payload.get("text", "")
            if text:
                source_key = self._source_key(acp_event)
                if self._reasoning_active:
                    self._close_reasoning_blocks()
                with self._flush_lock:
                    self._flush_lock_holder.held = True
                    try:
                        block_id = self._ensure_text_block(source_key)
                        if self._is_main_source(source_key):
                            self._record_main_text(block_id, text)
                        self._pending_text_block_id = block_id
                        self._pending_text_by_block[block_id] = self._pending_text_by_block.get(block_id, "") + text
                        self._schedule_flush()
                    finally:
                        self._flush_lock_holder.held = False
            return

        if card_event.type == CardEventType.REASONING_DELTA:
            self._flush_now()
            source_key = self._source_key(acp_event)
            block_id = self._ensure_reasoning_block(source_key)
            reasoning_text = card_event.payload.get("text", "")
            if source_key in self._pending_reasoning_item_breaks:
                if source_key in self._reasoning_sources_with_content and not reasoning_text.startswith("\n"):
                    reasoning_text = "\n" + reasoning_text
                self._pending_reasoning_item_breaks.discard(source_key)
            # Override the block_id in the delta to match the current reasoning block
            card_event = CardEvent(
                type=card_event.type,
                payload={**card_event.payload, "block_id": block_id, "text": reasoning_text},
            )
            self._dispatch_card_event(card_event)
            if reasoning_text.strip():
                self._reasoning_sources_with_content.add(source_key)
            return

        # Structural event: flush pending text first
        self._flush_now()

        # Tool events are hard execution boundaries. Retire both active stream
        # blocks so later analysis is appended after the tool in CardState,
        # preserving the provider's actual event order for timeline rendering.
        if card_event.type == CardEventType.TOOL_STARTED:
            self._last_tool_boundary_seq += 1
            if self._text_active:
                self._close_text_blocks()
            self._close_reasoning_blocks(retire=True)

        # Text resumed after tool
        if card_event.type == CardEventType.TEXT_STARTED:
            self._text_active = True
            block_id = card_event.payload.get("block_id") or self._active_text_block_id
            self._active_text_block_id = block_id
            self._text_blocks_by_source["main"] = block_id
            self._active_text_sources.add("main")

        if card_event.type in _TOOL_CARD_EVENT_TYPES:
            self._dispatch_tool_event(card_event)
        else:
            self._dispatch_card_event(card_event)

    def on_text(self, text: str) -> None:
        """Append text directly (for simple text-only streams)."""
        if text:
            with self._flush_lock:
                self._flush_lock_holder.held = True
                try:
                    block_id = self._ensure_text_block("main")
                    self._record_main_text(block_id, text)
                    self._pending_text_block_id = block_id
                    self._pending_text_by_block[block_id] = (
                        self._pending_text_by_block.get(block_id, "") + text
                    )
                    self._schedule_flush()
                finally:
                    self._flush_lock_holder.held = False

    def begin_continuation_turn(self) -> None:
        """Close active stream blocks before a same-card continuation turn."""
        self._flush_now()
        if self._reasoning_blocks_by_source:
            self._close_reasoning_blocks(retire=True)
        if self._text_active:
            self._close_text_blocks()
        self._last_tool_boundary_seq = max(
            self._last_tool_boundary_seq,
            self._text_turn_seq,
        )

    def finish(
        self,
        *,
        fallback_text: str = "",
        unfinished_subagent_status: str = "cancelled",
    ) -> None:
        """Complete the session normally.

        Args:
            fallback_text: If provided and the card contains no streamed text,
                this text is injected as a completion summary so the user sees
                the answer instead of a blank completed card.
            unfinished_subagent_status: Truthful fallback for children that did
                not emit an authoritative terminal snapshot. Defaults to
                ``cancelled``; parent completion does not prove child success.
        """
        self._flush_now()
        if self._reasoning_active:
            self._close_reasoning_blocks()
        if self._text_active:
            self._close_text_blocks()
        terminal_status = (
            unfinished_subagent_status
            if unfinished_subagent_status in {"completed", "failed", "cancelled"}
            else "cancelled"
        )
        self._finish_agent_summaries(terminal_status=terminal_status)
        # If the main Agent did not stream an answer, use fallback_text as its
        # completion summary. Subagent prose does not replace the parent answer.
        summary = ""
        if fallback_text:
            if not self._has_main_text():
                summary = fallback_text
                self._record_main_text("_summary", fallback_text)
        self._dispatch_card_event(CardEvent.completed(summary=summary))
        self._stop_ticker()

    def fail(
        self,
        error: str = "",
        *,
        unfinished_subagent_status: str = "failed",
    ) -> None:
        """Mark the session as failed."""
        self._cancel_timer()
        if self._text_active:
            self._flush_now()
            self._close_text_blocks()
        if self._reasoning_active:
            self._close_reasoning_blocks()
        terminal_status = (
            unfinished_subagent_status
            if unfinished_subagent_status in {"failed", "cancelled"}
            else "failed"
        )
        self._finish_agent_summaries(terminal_status=terminal_status)
        self._dispatch_card_event(CardEvent.failed(error))
        self._stop_ticker()

    def wait_for_user_confirmation(self, reason: str) -> None:
        """Close the card as blocked after bounded automatic continuation."""
        self._cancel_timer()
        self._flush_now()
        if self._reasoning_active:
            self._close_reasoning_blocks()
        if self._text_active:
            self._close_text_blocks()
        self._finish_agent_summaries(terminal_status="cancelled")
        self._dispatch_card_event(CardEvent.blocked(reason))
        self._stop_ticker()

    def cancel(self, *, reason: str = "cancelled") -> None:
        """Mark the parent and any live child cards as cancelled."""
        self._cancel_timer()
        self._flush_now()
        if self._text_active:
            self._close_text_blocks()
        if self._reasoning_active:
            self._close_reasoning_blocks()
        summary_changed = False
        terminal_statuses = {"completed", "failed", "cancelled"}
        for tool_id, existing in list(self._agent_summaries.items()):
            if existing.get("status") in terminal_statuses:
                continue
            self._agent_summaries[tool_id] = {
                **existing,
                "status": "cancelled",
            }
            summary_changed = True
        if summary_changed and not self._rotator.current.closed:
            try:
                self._dispatch_card_event(
                    CardEvent.tool_model_changed(
                        subagents=tuple(self._agent_summaries.values())
                    )
                )
            except Exception:
                logger.exception(
                    "Failed to publish cancelled subagent summary; "
                    "continuing parent terminal transition"
                )
        self._dispatch_card_event(CardEvent.cancelled(reason=reason))
        self._stop_ticker()

    def update_tool_model(self, tool_name: str | None = None, model_name: str | None = None) -> None:
        """Update the displayed tool/model in header subtitle."""
        self._flush_now()
        self._dispatch_card_event(CardEvent.tool_model_changed(tool_name, model_name))

    def get_message_id(self) -> str | None:
        """Get the message_id of the first card page (for message linking)."""
        current = self._rotator.current
        binding = current._delivery.get_binding(current.session_id)
        if binding and binding.pages:
            first_page = binding.pages.get(0)
            if first_page:
                return first_page.message_id
        return None

    def wait_until_visible(self, timeout: float) -> bool:
        """Wait for the initial async delivery and confirm a visible message."""
        current = self._rotator.current
        if not current.wait_delivery_idle(timeout=timeout):
            return False
        return self.get_message_id() is not None

    def wait_delivery_idle(self, timeout: float) -> bool:
        """Wait until every programming-card page has completed delivery."""
        deadline = time.monotonic() + timeout
        with self._dispatch_lock:
            sessions = (*self._failed_retired_sessions, self._rotator.current)
        for session in sessions:
            remaining = max(0.0, deadline - time.monotonic())
            if not session.wait_delivery_idle(timeout=remaining):
                return False
        return True

    def terminal_delivery_succeeded(self) -> bool:
        """Return whether the terminal programming card closed successfully."""
        with self._dispatch_lock:
            return (
                not self._capacity_delivery_failed
                and self._rotator.current.closed
                and all(
                    session.closed
                    for session in self._failed_retired_sessions
                )
            )

    def abort(self) -> None:
        """Stop local activity and close the card session without more delivery."""
        self._cancel_timer()
        self._stop_ticker()
        with self._dispatch_lock:
            retired = tuple(self._failed_retired_sessions)
            self._rotator.close()
            for session in retired:
                session.close()
            self._failed_retired_sessions.clear()
            self._active_tool_snapshots.clear()

    def get_final_text(self) -> str:
        """Return the complete ordered main-Agent transcript."""
        self._flush_now()
        with self._flush_lock:
            return "\n".join(
                content
                for _, content in self._main_text_transcript
                if content
            )

    def dispatch(self, event: CardEvent) -> None:
        """Route media-bridge events through the same capacity handoff gate."""
        self._dispatch_card_event(event)

    def _dispatch_card_event(self, event: CardEvent) -> bool:
        """Serialize projection, visible-first rotation, and event dispatch."""
        with self._dispatch_lock:
            if self._capacity_delivery_failed:
                return False
            current = self._rotator.current
            if current.closed:
                return False
            state = current.state
            if state is not None and self._requires_capacity_rotation(state, event):
                if not self._rotate_for_capacity(current, event):
                    self._capacity_delivery_failed = True
                    if self._capacity_failure_reason is None:
                        self._capacity_failure_reason = (
                            "continuation card could not be made visible and archived"
                        )
                    return False
            self._rotator.dispatch(event)
            return True

    def _dispatch_tool_event(self, event: CardEvent) -> bool:
        """Dispatch one tool event and retain active state across card pages."""
        with self._dispatch_lock:
            block_id = str((event.payload or {}).get("block_id") or "")
            if (
                event.type in {
                    CardEventType.TOOL_DELTA,
                    CardEventType.TOOL_DONE,
                    CardEventType.TOOL_FAILED,
                }
                and block_id
                and not self._current_card_has_tool(block_id)
            ):
                snapshot = self._active_tool_snapshots.get(block_id)
                tool_name = (
                    snapshot.tool_name
                    if snapshot is not None
                    else str((event.payload or {}).get("tool_summary") or "tool")
                )
                tool_input = snapshot.tool_input if snapshot is not None else ""
                if not self._dispatch_card_event(CardEvent.tool_started(
                    block_id,
                    tool_name or "tool",
                    tool_input or "",
                )):
                    return False
                if (
                    snapshot is not None
                    and snapshot.content
                    and not self._dispatch_card_event(CardEvent.tool_delta(
                        block_id,
                        snapshot.content,
                    ))
                ):
                    return False

            dispatched = self._dispatch_card_event(event)
            if dispatched:
                self._sync_active_tool_snapshot(event)
            return dispatched

    def _current_card_has_tool(self, block_id: str) -> bool:
        state = self._rotator.current.state
        if state is None:
            return False
        index = state.block_index.get(block_id)
        return (
            index is not None
            and index < len(state.blocks)
            and isinstance(state.blocks[index], ToolBlock)
        )

    def _sync_active_tool_snapshot(self, event: CardEvent) -> None:
        block_id = str((event.payload or {}).get("block_id") or "")
        if not block_id:
            return
        if event.type in {CardEventType.TOOL_DONE, CardEventType.TOOL_FAILED}:
            self._active_tool_snapshots.pop(block_id, None)
            return
        state = self._rotator.current.state
        if state is None:
            return
        index = state.block_index.get(block_id)
        if index is None or index >= len(state.blocks):
            return
        block = state.blocks[index]
        if isinstance(block, ToolBlock) and block.status == "active":
            self._active_tool_snapshots[block_id] = block

    @staticmethod
    def _requires_capacity_rotation(state: CardState, event: CardEvent) -> bool:
        if event.type is CardEventType.TOOL_STARTED:
            completed_tools = sum(
                block.kind == "tool_call" and block.status == "completed"
                for block in state.blocks
            )
            if completed_tools >= MAX_COMPLETED_TOOL_BLOCKS:
                return True
        return card_state_requires_continuation(state, event)

    def _rotate_for_capacity(
        self,
        old_session: CardSession,
        crossing_event: CardEvent,
    ) -> bool:
        factory = self._session_factory
        if factory is None:
            logger.error(
                "Programming card reached capacity without a continuation factory"
            )
            return False

        self._close_active_streams_for_capacity(old_session)
        old_state = old_session.state
        if old_state is None:
            logger.error("Programming card capacity rotation has no source state")
            return False

        task_list, active_tools = self._continuation_seed(
            old_state,
            crossing_event,
        )
        continuation_metadata = self._continuation_metadata(old_state)

        def create_visible_session() -> CardSession:
            new_session = factory(continuation_metadata)
            try:
                new_session.dispatch(CardEvent.started())
                if task_list is not None:
                    new_session.dispatch(CardEvent(
                        type=CardEventType.TASK_LIST_UPDATED,
                        payload={
                            "tasks": list(task_list.tasks),
                            "current_task_id": task_list.current_task_id,
                        },
                    ))
                for tool in active_tools:
                    new_session.dispatch(CardEvent.tool_started(
                        tool.block_id,
                        tool.tool_name or "tool",
                        tool.tool_input or "",
                    ))
                    if tool.content:
                        new_session.dispatch(CardEvent.tool_delta(
                            tool.block_id,
                            tool.content,
                        ))
                if not new_session.wait_delivery_idle(
                    timeout=self._continuation_visibility_timeout
                ):
                    raise RuntimeError("continuation delivery did not become idle")
                if not new_session.delivered_message_id:
                    raise RuntimeError("continuation card is not visible")
                return new_session
            except Exception:
                new_session.close()
                raise

        new_session = self._rotator.rotate(
            create_visible_session,
            enforce_max_rotations=False,
            archive_with_hint=False,
        )
        if new_session is None or new_session is old_session:
            logger.error(
                "Programming card capacity rotation failed; preserving full old card"
            )
            return False

        if not old_session.wait_delivery_idle(
            timeout=self._continuation_visibility_timeout
        ) or not old_session.closed:
            self._failed_retired_sessions.append(old_session)
            logger.error(
                "Programming card continuation is visible but old card archival failed"
            )
            return False

        self._reset_stream_state_after_capacity_rotation()
        return True

    def _close_active_streams_for_capacity(
        self,
        old_session: CardSession,
    ) -> None:
        """Close bounded stream blocks before freezing the old full card."""
        for source_key in list(self._active_text_sources):
            block_id = self._text_blocks_by_source.get(
                source_key,
                self._active_text_block_id,
            )
            old_session.dispatch(CardEvent.text_done(block_id))
        for source_key in list(self._active_reasoning_sources):
            block_id = self._reasoning_blocks_by_source.get(source_key)
            if block_id:
                old_session.dispatch(CardEvent.reasoning_done(block_id))
        self._reset_stream_state_after_capacity_rotation()

    def _reset_stream_state_after_capacity_rotation(self) -> None:
        self._text_active = False
        self._reasoning_active = False
        self._text_blocks_by_source.clear()
        self._reasoning_blocks_by_source.clear()
        self._active_text_sources.clear()
        self._active_reasoning_sources.clear()
        self._reasoning_sources_with_content.clear()
        self._pending_reasoning_item_breaks.clear()

    def _continuation_seed(
        self,
        state: CardState,
        crossing_event: CardEvent,
    ) -> tuple[TaskListBlock | None, tuple[ToolBlock, ...]]:
        task_list = next(
            (
                block
                for block in state.blocks
                if isinstance(block, TaskListBlock)
            ),
            None,
        )
        active_by_id = {
            block.block_id: block
            for block in state.blocks
            if isinstance(block, ToolBlock) and block.status == "active"
        }
        active_by_id.update(self._active_tool_snapshots)

        crossing_tool_id = str(
            (crossing_event.payload or {}).get("block_id") or ""
        )
        if crossing_event.type is CardEventType.TOOL_STARTED:
            active_by_id.pop(crossing_tool_id, None)

        # STARTED itself has no content block.  Keep one slot free for the
        # event that triggered this rollover, and one more for the task list
        # when present.  Older active tools remain visible on frozen cards and
        # stay in the adapter registry so a later update/completion can
        # materialize them on the then-current card without losing the event.
        active_slots = MAX_TOTAL_BLOCKS - 1 - int(task_list is not None)
        candidates = list(active_by_id.values())
        prioritized: list[ToolBlock] = []
        if crossing_tool_id and crossing_tool_id in active_by_id:
            prioritized.append(active_by_id[crossing_tool_id])
        prioritized.extend(
            block
            for block in reversed(candidates)
            if block.block_id != crossing_tool_id
        )
        selected_ids = {
            block.block_id
            for block in prioritized[:active_slots]
        }
        active_tools = tuple(
            block
            for block in candidates
            if block.block_id in selected_ids
        )
        return task_list, active_tools

    def _continuation_metadata(self, state: CardState) -> CardMetadata:
        continuation_seq = self._rotator.rotation_count + 1
        return replace(
            state.metadata,
            continuation_seq=continuation_seq,
            card_sequence=continuation_seq + 1,
            final_state_for_freeze=None,
            frozen=False,
            frozen_total_elapsed=None,
            bridge_phrase="续接：",
        )

    def _record_main_text(self, block_id: str, text: str) -> None:
        if not text:
            return
        with self._flush_lock:
            if (
                self._main_text_transcript
                and self._main_text_transcript[-1][0] == block_id
            ):
                previous_id, previous_text = self._main_text_transcript[-1]
                self._main_text_transcript[-1] = (
                    previous_id,
                    append_stream_text(previous_text, text),
                )
            else:
                self._main_text_transcript.append((block_id, text.lstrip("\n")))

    def _has_main_text(self) -> bool:
        with self._flush_lock:
            return any(
                bool(content.strip())
                for _, content in self._main_text_transcript
            )

    def _is_main_source(self, source_key: str) -> bool:
        return source_key == "main" or source_key not in self._agent_summaries

    # ---- Internal flush mechanism ----

    def _schedule_flush(self) -> None:
        """Schedule a flush timer if not already pending.

        IMPORTANT: Must only be called while holding ``_flush_lock``.
        """
        if not getattr(self._flush_lock_holder, "held", False):
            logger.error(
                "_schedule_flush called without holding _flush_lock — "
                "this is an internal state error, please report to maintainers"
            )
            raise RuntimeError("_schedule_flush must be called under _flush_lock")
        if self._flush_timer is None:
            self._flush_timer = threading.Timer(self._flush_interval, self._flush_now)
            self._flush_timer.daemon = True
            self._flush_timer.start()

    def _flush_now(self) -> None:
        """Flush pending text immediately."""
        self._cancel_timer()
        pending = ""
        block_id = self._active_text_block_id
        pending_by_block: dict[str, str] = {}
        with self._flush_lock:
            pending = self._pending_text
            block_id = self._pending_text_block_id or self._active_text_block_id
            pending_by_block = dict(self._pending_text_by_block)
            self._pending_text = ""
            self._pending_text_block_id = None
            self._pending_text_by_block.clear()
        if pending and block_id not in pending_by_block:
            pending_by_block[block_id] = pending
        if not self._rotator.current.closed:
            for pending_block_id, pending_text in pending_by_block.items():
                if pending_text:
                    self._dispatch_card_event(
                        CardEvent.text_delta(pending_block_id, pending_text)
                    )

    def _ensure_text_block(self, source_key: str) -> str:
        """Open or reuse the current logical text block for one source."""
        with self._dispatch_lock:
            if source_key in self._active_text_sources:
                return self._text_blocks_by_source.get(
                    source_key,
                    self._active_text_block_id,
                )
            block_id = self._current_text_block_id(source_key)
            source_kind = "main"
            source_sequence = None
            source_label = None
            source_ref = "main"
            if source_key != "main":
                source_ref = self._safe_source_suffix(source_key)
                summary = self._agent_summaries.get(source_key)
                if summary is not None:
                    source_kind = "subagent"
                    source_sequence = str(
                        summary.get("sequence") or ""
                    ).strip() or None
                    source_label = str(
                        summary.get("label") or ""
                    ).strip() or None
            dispatched = self._dispatch_card_event(
                CardEvent.text_started(
                    block_id,
                    source_kind=source_kind,
                    source_sequence=source_sequence,
                    source_label=source_label,
                    source_ref=source_ref,
                )
            )
            if dispatched:
                self._active_text_block_id = block_id
                self._text_blocks_by_source[source_key] = block_id
                self._active_text_sources.add(source_key)
                self._text_active = True
            return block_id

    def _ensure_reasoning_block(self, source_key: str) -> str:
        """Open or reuse the current logical reasoning block for one source."""
        with self._dispatch_lock:
            if source_key in self._active_reasoning_sources:
                return self._reasoning_blocks_by_source.get(
                    source_key,
                    self._active_reasoning_block_id,
                )
            block_id = (
                self._reasoning_blocks_by_source.get(source_key)
                or self._current_reasoning_block_id(source_key)
            )
            dispatched = self._dispatch_card_event(
                CardEvent.reasoning_started(block_id)
            )
            if dispatched:
                self._active_reasoning_block_id = block_id
                self._reasoning_blocks_by_source[source_key] = block_id
                self._active_reasoning_sources.add(source_key)
                self._reasoning_active = True
            return block_id

    def _close_text_blocks(self) -> None:
        with self._dispatch_lock:
            for source_key in list(self._active_text_sources):
                block_id = self._text_blocks_by_source.get(
                    source_key,
                    self._active_text_block_id,
                )
                self._dispatch_card_event(CardEvent.text_done(block_id))
            self._active_text_sources.clear()
            self._text_active = False

    def _close_reasoning_blocks(self, *, retire: bool = False) -> None:
        with self._dispatch_lock:
            source_keys = (
                self._reasoning_blocks_by_source
                if retire
                else self._active_reasoning_sources
            )
            for source_key in list(source_keys):
                self._close_reasoning_source(source_key, retire=retire)

    def _close_reasoning_source(self, source_key: str, *, retire: bool = False) -> None:
        """Close one source's reasoning block and optionally retire its ID."""
        with self._dispatch_lock:
            block_id = self._reasoning_blocks_by_source.get(source_key)
            if source_key in self._active_reasoning_sources and block_id:
                self._dispatch_card_event(CardEvent.reasoning_done(block_id))
                self._active_reasoning_sources.discard(source_key)
                if source_key in self._reasoning_sources_with_content and not retire:
                    self._pending_reasoning_item_breaks.add(source_key)
            if retire:
                self._reasoning_blocks_by_source.pop(source_key, None)
                self._pending_reasoning_item_breaks.discard(source_key)
                self._reasoning_sources_with_content.discard(source_key)
            self._reasoning_active = bool(self._active_reasoning_sources)

    def _current_text_block_id(self, source_key: str = "main") -> str:
        """Return the stable text block ID for the current ACP turn."""
        if self._text_turn_seq == 0:
            self._text_turn_seq = 1
        if self._last_tool_boundary_seq >= self._text_turn_seq:
            self._text_turn_seq = self._last_tool_boundary_seq + 1
        return self._block_id("text", self._text_turn_seq, source_key)

    def _current_reasoning_block_id(self, source_key: str = "main") -> str:
        """Return the next reasoning block ID after a hard execution boundary."""
        self._reasoning_turn_seq += 1
        return self._block_id("reasoning", self._reasoning_turn_seq, source_key)

    @staticmethod
    def _source_key(acp_event: "ACPEvent") -> str:
        source_id = getattr(acp_event, "source_id", None)
        if source_id and isinstance(source_id, str):
            return source_id.strip() or "main"
        return "main"

    @staticmethod
    def _safe_source_suffix(source_key: str) -> str:
        """Return a stable non-reversible source token for block identity."""
        if source_key == "main":
            return "main"
        digest = hashlib.sha256(
            source_key.encode("utf-8", errors="replace")
        ).hexdigest()
        return f"src_{digest[:12]}"

    def _block_id(self, kind: str, seq: int, source_key: str) -> str:
        if source_key == "main":
            return f"_turn_{seq}_{kind}" if seq > 1 else f"_active_{kind}"
        return f"_turn_{seq}_{kind}_{self._safe_source_suffix(source_key)}"

    def _cancel_timer(self) -> None:
        """Cancel any pending flush timer."""
        with self._flush_lock:
            if self._flush_timer is not None:
                self._flush_timer.cancel()
                self._flush_timer = None

    def _start_ticker(self) -> None:
        if self._ticker is not None:
            return
        from src.config import get_settings

        interval = get_settings().card.ticker_interval
        self._ticker = self._ticker_factory(
            session_id=self._rotator.current.session_id,
            on_frame=self._on_ticker_frame,
            interval=interval,
        )
        self._ticker.start()

    def _stop_ticker(self) -> None:
        ticker = self._ticker
        self._ticker = None
        if ticker is not None:
            ticker.stop()

    def _on_ticker_frame(self, frame: str) -> None:
        if not frame or self._rotator.current.closed:
            return
        current_frame = None
        current_state = self._rotator.current.state
        if current_state is not None:
            current_frame = current_state.metadata.live_ticker_frame
        if frame == current_frame:
            return
        now = time.monotonic()
        if (
            self._last_ticker_update_at is not None
            and now - self._last_ticker_update_at < self._ticker_update_min_interval
        ):
            return
        self._last_ticker_update_at = now
        if self._ticker_dispatch_async:
            try:
                executor = self._ticker_executor_factory() if self._ticker_executor_factory else None
                if executor is None:
                    from src.card.delivery.pool import get_delivery_pool

                    executor = get_delivery_pool()
                executor.submit(self._dispatch_ticker_frame, frame)
                return
            except RuntimeError:
                logger.debug("Ticker dispatch skipped because delivery pool is unavailable")
                return
            except Exception:
                logger.exception("Failed to submit ticker dispatch; dropping frame")
                return

        self._dispatch_ticker_frame(frame)

    def _dispatch_ticker_frame(self, frame: str) -> None:
        if not frame or self._rotator.current.closed:
            return
        self._dispatch_card_event(
            CardEvent.tool_model_changed(live_ticker_frame=frame)
        )

    def _handle_plan_update(self, acp_event: "ACPEvent") -> None:
        """Update the in-card task list in place.

        Plan/task changes never spawn a new Feishu card — the whole task list
        is updated in place until the current card reaches its state capacity;
        capacity rollover then replays only the latest task-list snapshot.
        """
        self._flush_now()
        card_event = CardEvent.from_acp(acp_event)
        self._dispatch_card_event(card_event)

    def _handle_agent_task_event(self, acp_event: "ACPEvent") -> bool:
        tool_call = getattr(acp_event, "tool_call", None)
        if tool_call is None:
            return False

        # Providers may change the title/kind/content shape between START,
        # UPDATE, and DONE. Once a call id is registered in the main-card
        # summary, keep routing by that stable identity instead of reclassifying
        # every frame. Programming mode intentionally does not create a separate
        # Feishu card here: the main card already owns the subtask summary and
        # execution stream.
        if (
            tool_call.id not in self._agent_summaries
            and not self._is_agent_task(tool_call)
        ):
            return False

        event_name = getattr(acp_event, "event_type", None).name if getattr(acp_event, "event_type", None) else ""
        existing = self._agent_summaries.get(tool_call.id)
        existing_status = str(
            (existing or {}).get("status") or ""
        ).strip().lower()
        incoming_status = (
            "failed"
            if event_name == "TOOL_CALL_DONE"
            and str(tool_call.status or "").strip().lower() == "failed"
            else "completed"
            if event_name == "TOOL_CALL_DONE"
            else "running"
        )
        if existing_status in _TERMINAL_AGENT_STATUSES:
            if existing_status == incoming_status == "failed":
                self._refresh_failed_agent_error(tool_call)
            return True

        if event_name == "TOOL_CALL_DONE":
            self._update_agent_summary(tool_call, status=incoming_status)
        else:
            self._update_agent_summary(tool_call, status="running")
        return True

    def _handle_collaboration_event(self, acp_event: "ACPEvent") -> bool:
        """Fold provider collaboration snapshots into stable child summaries."""
        tool_call = getattr(acp_event, "tool_call", None)
        if tool_call is None or not getattr(tool_call, "collaboration_tool", None):
            return False

        states_by_source = {
            str(item.get("source_id") or "").strip(): item
            for item in getattr(tool_call, "subagent_states", ())
            if isinstance(item, dict) and str(item.get("source_id") or "").strip()
        }
        source_ids = list(getattr(tool_call, "collaboration_receivers", ()))
        source_ids.extend(
            source_id
            for source_id in states_by_source
            if source_id not in source_ids
        )
        if not source_ids:
            # A failed spawn can terminate before the provider allocates a
            # child thread. Let the ordinary tool path render that failure;
            # successful structural no-op frames remain intentionally hidden.
            return str(getattr(tool_call, "status", "") or "").lower() != "failed"

        label_candidate = self._extract_agent_task_label(tool_call)
        opaque_ids = collect_subagent_opaque_ids(tool_call)
        changed = False
        for source_id in source_ids:
            source_id = str(source_id or "").strip()
            if not source_id:
                continue
            state = states_by_source.get(source_id, {})
            raw_status = str(state.get("status") or "running").strip().lower()
            incoming_status = _SUBAGENT_STATE_STATUS.get(raw_status, "running")
            existing = self._agent_summaries.get(source_id, {})
            existing_status = str(existing.get("status") or "").strip().lower()
            message = sanitize_subagent_display_text(
                state.get("message"),
                fallback="",
                max_chars=180,
                opaque_ids=opaque_ids,
            )
            if existing_status in _TERMINAL_AGENT_STATUSES:
                if incoming_status != existing_status:
                    continue
                if not message or message == str(existing.get("progress") or ""):
                    continue

            existing_label = str(existing.get("label") or "").strip()
            label = existing_label or label_candidate
            if self._is_generic_task_label(label):
                label = "子任务"

            progress = message or _SUBAGENT_STATE_PROGRESS.get(raw_status, "执行中")
            summary = {
                **existing,
                "label": label,
                "tool": str(existing.get("tool") or "子代理"),
                "status": incoming_status,
                "progress": progress,
            }
            summary.setdefault(
                "sequence",
                f"{self._rotator.current.sequence}.{chr(ord('a') + len(self._agent_summaries))}",
            )
            model = str(
                getattr(tool_call, "collaboration_model", None)
                or existing.get("model")
                or self._base_metadata.model_name
                or ""
            ).strip()
            if model:
                summary["model"] = model
            if summary != existing:
                self._agent_summaries[source_id] = summary
                changed = True

        if changed:
            self._publish_agent_summaries()
        # Collaboration calls are structural transport events. The stable child
        # summary is the user-facing representation; rendering every spawn/wait
        # invocation as a normal tool block creates duplicate, noisy history.
        return True

    def _handle_subagent_activity_event(self, acp_event: "ACPEvent") -> bool:
        """Update one child summary from a namespaced lifecycle activity."""
        tool_call = getattr(acp_event, "tool_call", None)
        source_id = str(
            getattr(tool_call, "subagent_source_id", None)
            or getattr(acp_event, "source_id", None)
            or ""
        ).strip()
        activity = str(
            getattr(tool_call, "subagent_activity", None) or ""
        ).strip().lower()
        if tool_call is None or not source_id or not activity:
            return False

        provider_status = str(getattr(tool_call, "status", "") or "").lower()
        activity_complete = provider_status == "completed"
        activity_failed = provider_status == "failed"
        if activity == "started":
            progress = (
                "已启动"
                if activity_complete
                else ("启动未完成" if activity_failed else "正在启动")
            )
            incoming_status = "running"
        elif activity == "interacted":
            progress = (
                "已与主 Agent 交互"
                if activity_complete
                else (
                    "交互未完成"
                    if activity_failed
                    else "正在与主 Agent 交互"
                )
            )
            incoming_status = "running"
        elif activity == "interrupted":
            progress = (
                "已中断"
                if activity_complete
                else ("中断未完成" if activity_failed else "正在中断")
            )
            incoming_status = "cancelled" if activity_complete else "running"
        else:
            progress = "动态已更新"
            incoming_status = "running"

        existing = self._agent_summaries.get(source_id, {})
        existing_status = str(existing.get("status") or "").strip().lower()
        if existing_status in _TERMINAL_AGENT_STATUSES:
            return True

        raw_path = str(getattr(tool_call, "subagent_path", None) or "").strip()
        path_label = raw_path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        label = str(existing.get("label") or "").strip()
        if not label or self._is_generic_task_label(label):
            label = sanitize_subagent_display_text(
                path_label,
                fallback="子任务",
                max_chars=60,
                opaque_ids=collect_subagent_opaque_ids(tool_call),
            )

        summary = {
            **existing,
            "label": label,
            "tool": str(existing.get("tool") or "子代理"),
            "status": incoming_status,
            "progress": progress,
        }
        summary.setdefault(
            "sequence",
            f"{self._rotator.current.sequence}.{chr(ord('a') + len(self._agent_summaries))}",
        )
        if self._base_metadata.model_name:
            summary.setdefault("model", self._base_metadata.model_name)
        if summary == existing:
            return True
        self._agent_summaries[source_id] = summary
        self._publish_agent_summaries()
        return True

    def _publish_agent_summaries(self) -> None:
        if self._rotator.current.closed:
            return
        self._dispatch_card_event(
            CardEvent.tool_model_changed(
                subagents=tuple(self._agent_summaries.values())
            )
        )

    def _handle_agent_image_event(self, acp_event: "ACPEvent") -> bool:
        """Route subtask media into the main card with task attribution."""
        source_id = str(getattr(acp_event, "source_id", "") or "").strip()
        if not source_id:
            return False
        summary = self._agent_summaries.get(source_id)
        if summary is None:
            return False
        image = getattr(acp_event, "image", None)
        if image is not None and image.image_id in self._routed_image_ids:
            return True

        label = str(summary.get("label") or "子任务").strip()
        if image is not None:
            attributed_name = f"{label} · {image.name}"
            acp_event = replace(
                acp_event,
                image=replace(image, name=attributed_name[:120]),
            )
        handled = self._image_publisher.handle(acp_event)

        if handled and image is not None:
            self._routed_image_ids.add(image.image_id)
        return handled

    def _update_agent_summary(
        self,
        tool_call: "ToolCallInfo",
        *,
        status: str,
    ) -> None:
        existing = self._agent_summaries.get(tool_call.id, {})
        existing_label = str(existing.get("label") or "").strip()
        is_terminal = status in _TERMINAL_AGENT_STATUSES
        if is_terminal:
            label = existing_label or self._terminal_agent_label(tool_call)
        else:
            label = self._extract_agent_task_label(tool_call)
            if (
                existing_label
                and not self._is_generic_task_label(existing_label)
                and not is_unhelpful_display_label(existing_label)
                and (
                    self._is_generic_task_label(label)
                    or is_unhelpful_display_label(label)
                )
            ):
                label = existing_label
        tool_name = self._extract_agent_tool_name(tool_call)
        if is_terminal and existing.get("tool"):
            tool_name = existing["tool"]
        summary = {
            **existing,
            "label": label,
            "tool": tool_name,
            "status": status,
        }
        if status == "failed":
            summary["error"] = self._agent_failure_detail(tool_call)
        else:
            summary.pop("error", None)
        summary.setdefault(
            "sequence",
            f"{self._rotator.current.sequence}.{chr(ord('a') + len(self._agent_summaries))}",
        )
        if self._base_metadata.model_name:
            summary.setdefault("model", self._base_metadata.model_name)
        self._agent_summaries[tool_call.id] = summary
        if not self._rotator.current.closed:
            self._dispatch_card_event(
                CardEvent.tool_model_changed(
                    subagents=tuple(self._agent_summaries.values())
                )
            )

    def _refresh_failed_agent_error(self, tool_call: "ToolCallInfo") -> None:
        existing = self._agent_summaries.get(tool_call.id)
        if existing is None:
            return
        existing_detail = str(existing.get("error") or "").strip()
        if (
            existing_detail
            and existing_detail != _GENERIC_AGENT_FAILURE_DETAIL
        ):
            return
        detail = self._agent_failure_detail(tool_call, fallback="")
        if not detail or detail == existing_detail:
            return
        self._agent_summaries[tool_call.id] = {
            **existing,
            "error": detail,
        }
        if not self._rotator.current.closed:
            self._dispatch_card_event(
                CardEvent.tool_model_changed(
                    subagents=tuple(self._agent_summaries.values())
                )
            )

    @staticmethod
    def _agent_failure_detail(
        tool_call: "ToolCallInfo",
        *,
        fallback: str = _GENERIC_AGENT_FAILURE_DETAIL,
    ) -> str:
        opaque_ids = (tool_call.id,)
        result = getattr(tool_call, "result", None)
        if result is not None:
            detail = sanitize_tool_failure_detail(
                result,
                fallback="",
                opaque_ids=opaque_ids,
                allow_unstructured=False,
            )
            if detail:
                return detail
        return sanitize_tool_failure_detail(
            tool_call.content,
            fallback=fallback,
            opaque_ids=opaque_ids,
            allow_unstructured=False,
        )

    @classmethod
    def _terminal_agent_label(cls, tool_call: "ToolCallInfo") -> str:
        title = str(tool_call.title or "").strip()
        if cls._is_generic_task_label(title) or is_unhelpful_display_label(title):
            return "子任务"
        return sanitize_tool_failure_detail(
            title,
            fallback="子任务",
            max_chars=60,
            opaque_ids=(tool_call.id,),
        )

    def _finish_agent_summaries(self, *, terminal_status: str) -> None:
        summary_changed = False
        for tool_id, existing in list(self._agent_summaries.items()):
            if existing.get("status") in {"completed", "failed", "cancelled"}:
                continue
            self._agent_summaries[tool_id] = {
                **existing,
                "status": terminal_status,
            }
            summary_changed = True
        if summary_changed and not self._rotator.current.closed:
            try:
                self._dispatch_card_event(
                    CardEvent.tool_model_changed(
                        subagents=tuple(self._agent_summaries.values())
                    )
                )
            except Exception:
                logger.exception("Failed to publish final subagent summary; continuing parent terminal transition")

    @staticmethod
    def _is_agent_task(tool_call: "ToolCallInfo") -> bool:
        title = (tool_call.title or "").strip().lower()
        kind = (tool_call.kind or "").strip().lower()
        content = (tool_call.content or "").strip()
        if kind == "agent" or title in _AGENT_TOOL_TITLES:
            return True
        # Some ACP backends expose agent tools as kind=other and put the
        # subagent identity in the formatted input. Never inspect the output of
        # concrete tools such as execute/read/edit: source code or command
        # output can legitimately contain this marker and must remain a normal
        # parent tool event.
        return kind == "other" and "子代理：" in content

    @staticmethod
    def _extract_agent_task_label(tool_call: "ToolCallInfo") -> str:
        raw_limit = ProgrammingCardSession._raw_agent_metadata_limit(
            tool_call,
            minimum=60,
        )
        return sanitize_subagent_display_text(
            extract_tool_call_label(
                tool_call,
                generic_labels=_GENERIC_TASK_LABELS,
                fallback="子任务",
                max_chars=raw_limit,
            ),
            fallback="子任务",
            max_chars=60,
            opaque_ids=collect_subagent_opaque_ids(tool_call),
        )

    @staticmethod
    def _is_generic_task_label(value: str) -> bool:
        return str(value or "").strip().lower() in _GENERIC_TASK_LABELS

    @staticmethod
    def _extract_agent_tool_name(tool_call: "ToolCallInfo") -> str:
        raw_limit = ProgrammingCardSession._raw_agent_metadata_limit(
            tool_call,
            minimum=24,
        )
        return sanitize_subagent_display_text(
            extract_agent_tool_name(
                tool_call,
                max_chars=raw_limit,
            ),
            fallback="子代理",
            max_chars=24,
            opaque_ids=collect_subagent_opaque_ids(tool_call),
        )

    @staticmethod
    def _raw_agent_metadata_limit(
        tool_call: "ToolCallInfo",
        *,
        minimum: int,
    ) -> int:
        """Keep raw candidates intact until ID/credential redaction runs."""
        return max(
            minimum,
            len(str(tool_call.id or "")),
            len(str(tool_call.title or "")),
            len(str(tool_call.content or "")),
        )
