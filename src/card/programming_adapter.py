"""Programming mode adapter: bridges streaming card pattern to CardSession.

Bridges streaming card pattern to CardSession for
ProgrammingHandler.handle_response(). Supports all programming modes.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Callable

from src.card.events import CardEvent, CardEventType
from src.card.events.projector import (
    attribute_subagent_image,
    finalize_summaries,
    merge_activity_summary,
    merge_agent_tool_summary,
    merge_collaboration_summaries,
    project_activity,
    project_collaboration,
)
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

if TYPE_CHECKING:
    from src.acp.models import ACPEvent, ACPImageInfo

logger = logging.getLogger(__name__)

# Mode name → (mode_emoji, display_name)
_MODE_DISPLAY: dict[str, tuple[str, str]] = {
    "coco": ("🤖", "Coco"),
    "claude": ("🧠", "Claude"),
    "aiden": ("⚡", "Aiden"),
    "codex": ("📝", "Codex"),
    "gemini": ("💎", "Gemini"),
    "traex": ("🚀", "Traex"),
    "grok": ("🌌", "Grok"),
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
        mode_name: Programming mode identifier.
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
        self._active_text_block_id = "_active_text"
        self._active_reasoning_block_id = "_active_reasoning"
        self._text_blocks_by_source: dict[str, str] = {}
        self._reasoning_blocks_by_source: dict[str, str] = {}
        self._active_text_sources: set[str] = set()
        self._active_reasoning_sources: set[str] = set()
        self._reasoning_sources_with_content: set[str] = set()
        self._pending_reasoning_item_breaks: set[str] = set()
        self._flush_interval = flush_interval or self._DEFAULT_FLUSH_INTERVAL
        # Text batching state
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

    def start(self) -> None:
        """Start the card (creates initial card in Feishu)."""
        self._dispatch_card_event(CardEvent.started())
        self._dispatch_card_event(CardEvent.text_started("_active_text"))
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
            if self._active_text_sources:
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
                if self._active_reasoning_sources:
                    self._close_reasoning_blocks()
                with self._flush_lock:
                    self._flush_lock_holder.held = True
                    try:
                        block_id = self._ensure_text_block(source_key)
                        if self._is_main_source(source_key):
                            self._record_main_text(block_id, text)
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
            if self._active_text_sources:
                self._close_text_blocks()
            self._close_reasoning_blocks(retire=True)

        # Text resumed after tool
        if card_event.type == CardEventType.TEXT_STARTED:
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
        if self._active_text_sources:
            self._close_text_blocks()
        self._last_tool_boundary_seq = max(
            self._last_tool_boundary_seq,
            self._text_turn_seq,
        )

    def finish(self) -> None:
        """Complete the session; result text must already be in the stream."""
        self._terminate(CardEvent.completed(), subagent_status="cancelled")

    def fail(
        self,
        error: str = "",
        *,
        unfinished_subagent_status: str = "failed",
    ) -> None:
        """Mark the session as failed."""
        terminal_status = (
            unfinished_subagent_status
            if unfinished_subagent_status in {"failed", "cancelled"}
            else "failed"
        )
        self._terminate(CardEvent.failed(error), subagent_status=terminal_status)

    def cancel(self, *, reason: str = "cancelled") -> None:
        """Mark the parent and unresolved children as cancelled."""
        self._terminate(
            CardEvent.cancelled(reason=reason),
            subagent_status="cancelled",
        )

    def _terminate(self, event: CardEvent, *, subagent_status: str) -> None:
        self._cancel_timer()
        self._flush_now()
        if self._active_text_sources:
            self._close_text_blocks()
        if self._active_reasoning_sources:
            self._close_reasoning_blocks()
        self._finish_agent_summaries(terminal_status=subagent_status)
        self._dispatch_card_event(event)
        self._stop_ticker()

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
        pending_by_block: dict[str, str] = {}
        with self._flush_lock:
            pending_by_block = dict(self._pending_text_by_block)
            self._pending_text_by_block.clear()
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
        handled, summaries = merge_agent_tool_summary(
            self._agent_summaries,
            acp_event,
            parent_sequence=self._rotator.current.sequence,
            base_model=self._base_metadata.model_name or "",
        )
        if handled and summaries != self._agent_summaries:
            self._agent_summaries = summaries
            self._publish_agent_summaries()
        return handled

    def _handle_collaboration_event(self, acp_event: "ACPEvent") -> bool:
        """Fold provider collaboration snapshots into stable child summaries."""
        projection = project_collaboration(acp_event)
        if projection is None:
            return False
        if not projection.agents:
            # A failed spawn can terminate before the provider allocates a
            # child thread. Let the ordinary tool path render that failure;
            # successful structural no-op frames remain intentionally hidden.
            return not projection.failed_without_receiver
        summaries = merge_collaboration_summaries(
            self._agent_summaries,
            projection,
            parent_sequence=self._rotator.current.sequence,
            base_model=self._base_metadata.model_name or "",
        )
        if summaries != self._agent_summaries:
            self._agent_summaries = summaries
            self._publish_agent_summaries()
        # Collaboration calls are structural transport events. The stable child
        # summary is the user-facing representation; rendering every spawn/wait
        # invocation as a normal tool block creates duplicate, noisy history.
        return True

    def _handle_subagent_activity_event(self, acp_event: "ACPEvent") -> bool:
        """Update one child summary from a namespaced lifecycle activity."""
        projection = project_activity(acp_event)
        if projection is None:
            return False
        summaries = merge_activity_summary(
            self._agent_summaries,
            projection,
            parent_sequence=self._rotator.current.sequence,
            base_model=self._base_metadata.model_name or "",
        )
        if summaries != self._agent_summaries:
            self._agent_summaries = summaries
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
        acp_event = attribute_subagent_image(acp_event, label)
        handled = self._image_publisher.handle(acp_event)

        if handled and image is not None:
            self._routed_image_ids.add(image.image_id)
        return handled

    def _finish_agent_summaries(self, *, terminal_status: str) -> None:
        summaries = finalize_summaries(self._agent_summaries, terminal_status)
        if summaries != self._agent_summaries and not self._rotator.current.closed:
            self._agent_summaries = summaries
            try:
                self._publish_agent_summaries()
            except Exception:
                logger.exception("Failed to publish final subagent summary; continuing parent terminal transition")
