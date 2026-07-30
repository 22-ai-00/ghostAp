"""Programming mode adapter: bridges streaming card pattern to CardSession.

Bridges streaming card pattern to CardSession for
ProgrammingHandler.handle_response(). Supports all programming modes:
Coco/Claude/Aiden/Codex/Gemini/Traex/TTADK.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Callable

from src.card.events import CardEvent, CardEventType
from src.card.media_bridge import ACPImagePublisher
from src.card.render.live_ticker import LiveTicker
from src.card.session import CardSession
from src.card.session.rotator import SessionRotator
from src.card.state.models import CardMetadata
from src.card.tool_display import (
    extract_agent_tool_name,
    extract_tool_call_label,
    is_unhelpful_display_label,
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
}

_AGENT_TOOL_TITLES = {"agent", "subagent", "task"}
_GENERIC_TASK_LABELS = {"", "agent", "subagent", "task", "子任务"}
_TERMINAL_AGENT_STATUSES = {"completed", "failed", "cancelled"}
_GENERIC_AGENT_FAILURE_DETAIL = "子任务执行失败"


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
    ) -> None:
        self._rotator = SessionRotator(session)
        self._base_metadata = (
            base_metadata
            or getattr(session, "_metadata", None)
            or CardMetadata()
        )
        self._image_publisher = ACPImagePublisher(self._rotator, image_uploader)
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
        self._rotator.dispatch(CardEvent.started())
        self._rotator.dispatch(CardEvent.text_started("_active_text"))
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
            self._rotator.dispatch(card_event)
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

        self._rotator.dispatch(card_event)

    def on_text(self, text: str) -> None:
        """Append text directly (for simple text-only streams)."""
        if text:
            with self._flush_lock:
                self._flush_lock_holder.held = True
                try:
                    block_id = self._ensure_text_block("main")
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
        unfinished_subagent_status: str = "completed",
    ) -> None:
        """Complete the session normally.

        Args:
            fallback_text: If provided and the card contains no streamed text,
                this text is injected as a completion summary so the user sees
                the answer instead of a blank completed card.
        """
        self._flush_now()
        if self._reasoning_active:
            self._close_reasoning_blocks()
        if self._text_active:
            self._close_text_blocks()
        terminal_status = (
            unfinished_subagent_status
            if unfinished_subagent_status in {"completed", "failed", "cancelled"}
            else "completed"
        )
        self._finish_agent_summaries(terminal_status=terminal_status)
        # If no text was streamed into the card, use fallback_text as completion
        # summary so the user sees the answer instead of a blank card.
        summary = ""
        if fallback_text:
            state = self._rotator.current.state
            has_text = any(b.kind == "text" and b.content for b in state.blocks) if state else False
            if not has_text:
                summary = fallback_text
        self._rotator.dispatch(CardEvent.completed(summary=summary))
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
        self._rotator.dispatch(CardEvent.failed(error))
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
        self._rotator.dispatch(CardEvent.blocked(reason))
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
                self._rotator.dispatch(
                    CardEvent.tool_model_changed(
                        subagents=tuple(self._agent_summaries.values())
                    )
                )
            except Exception:
                logger.exception(
                    "Failed to publish cancelled subagent summary; "
                    "continuing parent terminal transition"
                )
        self._rotator.dispatch(CardEvent.cancelled(reason=reason))
        self._stop_ticker()

    def update_tool_model(self, tool_name: str | None = None, model_name: str | None = None) -> None:
        """Update the displayed tool/model in header subtitle."""
        self._flush_now()
        self._rotator.dispatch(CardEvent.tool_model_changed(tool_name, model_name))

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
        """Wait until the single programming card has completed delivery."""
        return self._rotator.current.wait_delivery_idle(timeout=timeout)

    def terminal_delivery_succeeded(self) -> bool:
        """Return whether the terminal programming card closed successfully."""
        return self._rotator.current.closed

    def abort(self) -> None:
        """Stop local activity and close the card session without more delivery."""
        self._cancel_timer()
        self._stop_ticker()
        self._rotator.close()

    def get_final_text(self) -> str:
        """Extract accumulated text content from card state for context recording."""
        self._flush_now()
        state = self._rotator.current.state
        if not state:
            return ""
        parts = []
        for block in state.blocks:
            if block.kind == "text" and block.content:
                parts.append(block.content)
        return "\n".join(parts)

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
                    self._rotator.dispatch(CardEvent.text_delta(pending_block_id, pending_text))

    def _ensure_text_block(self, source_key: str) -> str:
        """Open or reuse the current logical text block for one source."""
        if source_key in self._active_text_sources:
            return self._text_blocks_by_source.get(source_key, self._active_text_block_id)
        block_id = self._current_text_block_id(source_key)
        self._active_text_block_id = block_id
        self._text_blocks_by_source[source_key] = block_id
        self._rotator.dispatch(CardEvent.text_started(block_id))
        self._active_text_sources.add(source_key)
        self._text_active = True
        return block_id

    def _ensure_reasoning_block(self, source_key: str) -> str:
        """Open or reuse the current logical reasoning block for one source."""
        if source_key in self._active_reasoning_sources:
            return self._reasoning_blocks_by_source.get(source_key, self._active_reasoning_block_id)
        block_id = self._reasoning_blocks_by_source.get(source_key) or self._current_reasoning_block_id(source_key)
        self._active_reasoning_block_id = block_id
        self._reasoning_blocks_by_source[source_key] = block_id
        self._rotator.dispatch(CardEvent.reasoning_started(block_id))
        self._active_reasoning_sources.add(source_key)
        self._reasoning_active = True
        return block_id

    def _close_text_blocks(self) -> None:
        for source_key in list(self._active_text_sources):
            block_id = self._text_blocks_by_source.get(source_key, self._active_text_block_id)
            self._rotator.dispatch(CardEvent.text_done(block_id))
        self._active_text_sources.clear()
        self._text_active = False

    def _close_reasoning_blocks(self, *, retire: bool = False) -> None:
        source_keys = (
            self._reasoning_blocks_by_source
            if retire
            else self._active_reasoning_sources
        )
        for source_key in list(source_keys):
            self._close_reasoning_source(source_key, retire=retire)

    def _close_reasoning_source(self, source_key: str, *, retire: bool = False) -> None:
        """Close one source's reasoning block and optionally retire its ID."""
        block_id = self._reasoning_blocks_by_source.get(source_key)
        if source_key in self._active_reasoning_sources and block_id:
            self._rotator.dispatch(CardEvent.reasoning_done(block_id))
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
        suffix = re.sub(r"[^a-zA-Z0-9_-]+", "_", source_key).strip("_")
        return suffix[:40] or "main"

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
        self._rotator.dispatch(CardEvent.tool_model_changed(live_ticker_frame=frame))

    def _handle_plan_update(self, acp_event: "ACPEvent") -> None:
        """Update the in-card task list in place.

        Plan/task changes never spawn a new Feishu card — the whole task list
        lives in one streaming card and is updated as the agent works through it.
        A new continuation card is only created when the current card nears the
        Feishu element/byte limit (handled by render-time pagination).
        """
        card_event = CardEvent.from_acp(acp_event)
        self._rotator.dispatch(card_event)

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
            self._rotator.dispatch(CardEvent.tool_model_changed(subagents=tuple(self._agent_summaries.values())))

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
            self._rotator.dispatch(
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
                self._rotator.dispatch(CardEvent.tool_model_changed(subagents=tuple(self._agent_summaries.values())))
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
        return extract_tool_call_label(
            tool_call,
            generic_labels=_GENERIC_TASK_LABELS,
            fallback="子任务",
            max_chars=60,
        )

    @staticmethod
    def _is_generic_task_label(value: str) -> bool:
        return str(value or "").strip().lower() in _GENERIC_TASK_LABELS

    @staticmethod
    def _extract_agent_tool_name(tool_call: "ToolCallInfo") -> str:
        return extract_agent_tool_name(tool_call)
