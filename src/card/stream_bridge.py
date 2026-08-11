"""ACPStreamBridge: normalizes ACP streaming into CardEvent sequences.

Public component in the session layer. Implements the StreamBridge protocol.
Used by Deep/Spec renderers for consistent ACP event → CardEvent conversion.
"""

from __future__ import annotations

import re
import threading
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from src.acp.models import ACPEvent, ACPImageInfo
    from src.card.protocols import Dispatchable


class ACPStreamBridge:
    """Normalize ACP streaming into programming-mode-like card blocks.

    Reused by Deep/Spec renderers so text, reasoning and tool panels follow
    the same event sequencing as direct programming mode.

    Implements the StreamBridge protocol defined in src.card.protocols.

    Thread-safety: a lock serializes on_event / bind / close_open_blocks so
    that concurrent calls (e.g. orchestrator broadcast timer vs. event loop)
    cannot interleave state mutations and dispatch calls.
    """

    def __init__(
        self,
        dispatchable: Dispatchable,
        *,
        image_uploader: Callable[["ACPImageInfo"], str | None] | None = None,
        preserve_tool_content: bool = False,
    ) -> None:
        from src.card.media_bridge import ACPImagePublisher

        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._dispatchable: Dispatchable = dispatchable
        self._image_publisher = ACPImagePublisher(dispatchable, image_uploader)
        self._preserve_tool_content = preserve_tool_content
        self._text_active: bool = False
        self._reasoning_active: bool = False
        # Per-turn counters generate unique block IDs at hard stream
        # boundaries. Soft alternation between text and reasoning keeps the
        # same logical blocks so Feishu cards do not fragment into one-token
        # markdown/panel rows.
        self._text_turn_seq: int = 0
        self._reasoning_turn_seq: int = 0
        self._active_text_block_id: str = "_active_text"
        self._active_reasoning_block_id: str = "_active_reasoning"
        self._text_blocks_by_source: dict[str, str] = {}
        self._reasoning_blocks_by_source: dict[str, str] = {}
        self._active_text_sources: set[str] = set()
        self._active_reasoning_sources: set[str] = set()
        self._reasoning_sources_with_content: set[str] = set()
        self._pending_reasoning_item_breaks: set[str] = set()

    def bind(self, dispatchable: Dispatchable) -> None:
        """Rebind to a new dispatchable target; closes open blocks first."""
        with self._lock:
            self._close_open_blocks_locked()
            self._dispatchable = dispatchable
            self._image_publisher.bind(dispatchable)
            self._text_active = False
            self._reasoning_active = False
            self._text_blocks_by_source.clear()
            self._reasoning_blocks_by_source.clear()
            self._active_text_sources.clear()
            self._active_reasoning_sources.clear()
            self._reasoning_sources_with_content.clear()
            self._pending_reasoning_item_breaks.clear()

    def on_event(self, acp_event: ACPEvent) -> None:
        """Process an ACP event and dispatch corresponding CardEvents."""
        from src.acp import ACPEventType
        from src.card.events import CardEvent, card_event_from_acp

        with self._lock:
            if acp_event.event_type == ACPEventType.IMAGE_CHUNK:
                self._close_open_blocks_locked(retire_reasoning=True)
                self._image_publisher.handle(acp_event)
                return
            source_key = self._source_key(acp_event)
            if acp_event.event_type == ACPEventType.THOUGHT_CHUNK:
                reasoning_block_id = self._ensure_reasoning_block_locked(source_key)
            elif acp_event.event_type == ACPEventType.TEXT_CHUNK:
                text_block_id = self._ensure_text_block_locked(source_key)
            elif acp_event.event_type == ACPEventType.TOOL_CALL_START:
                self._close_text_blocks_locked()
                self._retire_reasoning_blocks_locked()

            # Override block_id in the converted CardEvent to match our per-turn ID
            ce = card_event_from_acp(
                acp_event,
                preserve_tool_content=self._preserve_tool_content,
            )
            if acp_event.event_type == ACPEventType.THOUGHT_CHUNK and ce.payload.get("block_id"):
                reasoning_text = ce.payload.get("text", "")
                if source_key in self._pending_reasoning_item_breaks:
                    if (
                        source_key in self._reasoning_sources_with_content
                        and not reasoning_text.startswith("\n")
                    ):
                        reasoning_text = "\n" + reasoning_text
                    self._pending_reasoning_item_breaks.discard(source_key)
                ce = CardEvent(
                    type=ce.type,
                    payload={**ce.payload, "block_id": reasoning_block_id, "text": reasoning_text},
                )
                if reasoning_text.strip():
                    self._reasoning_sources_with_content.add(source_key)
            elif acp_event.event_type == ACPEventType.TEXT_CHUNK and ce.payload.get("block_id"):
                ce = CardEvent(type=ce.type, payload={**ce.payload, "block_id": text_block_id})
            self._dispatchable.dispatch(ce)

    def close_open_blocks(self) -> None:
        """Close any currently open reasoning or text blocks."""
        with self._lock:
            self._close_open_blocks_locked()

    def _close_open_blocks_locked(
        self,
        *,
        retire_reasoning: bool = False,
    ) -> None:
        """Internal: close open blocks while holding self._lock."""
        source_keys = (
            self._reasoning_blocks_by_source
            if retire_reasoning
            else self._active_reasoning_sources
        )
        for source_key in list(source_keys):
            self._close_reasoning_source_locked(
                source_key,
                retire=retire_reasoning,
            )
        self._close_text_blocks_locked()

    def _retire_reasoning_blocks_locked(self) -> None:
        """Retire every source mapping at a hard execution boundary."""
        for source_key in list(self._reasoning_blocks_by_source):
            self._close_reasoning_source_locked(source_key, retire=True)

    def _close_text_blocks_locked(self) -> None:
        """Close answer text at a hard execution boundary."""
        from src.card.events import CardEvent

        for source_key in list(self._active_text_sources):
            block_id = self._text_blocks_by_source.get(source_key, self._active_text_block_id)
            self._dispatchable.dispatch(CardEvent.text_done(block_id))
        self._active_text_sources.clear()
        self._text_active = False

    def _close_reasoning_source_locked(self, source_key: str, *, retire: bool = False) -> None:
        """Close one source's reasoning block while holding ``self._lock``."""
        from src.card.events import CardEvent

        block_id = self._reasoning_blocks_by_source.get(source_key)
        if source_key in self._active_reasoning_sources and block_id:
            self._dispatchable.dispatch(CardEvent.reasoning_done(block_id))
            self._active_reasoning_sources.discard(source_key)
            if source_key in self._reasoning_sources_with_content and not retire:
                self._pending_reasoning_item_breaks.add(source_key)
        if retire:
            self._reasoning_blocks_by_source.pop(source_key, None)
            self._pending_reasoning_item_breaks.discard(source_key)
            self._reasoning_sources_with_content.discard(source_key)
        self._reasoning_active = bool(self._active_reasoning_sources)

    def _ensure_text_block_locked(self, source_key: str) -> str:
        """Open the current logical text block if needed."""
        from src.card.events import CardEvent

        if source_key in self._active_text_sources:
            return self._text_blocks_by_source.get(source_key, self._active_text_block_id)
        self._text_turn_seq += 1
        bid = self._block_id("text", self._text_turn_seq, source_key)
        self._active_text_block_id = bid
        self._text_blocks_by_source[source_key] = bid
        self._dispatchable.dispatch(CardEvent.text_started(bid))
        self._active_text_sources.add(source_key)
        self._text_active = True
        return bid

    def _ensure_reasoning_block_locked(self, source_key: str) -> str:
        """Open the current logical reasoning block if needed."""
        from src.card.events import CardEvent

        if source_key in self._active_reasoning_sources:
            return self._reasoning_blocks_by_source.get(source_key, self._active_reasoning_block_id)
        bid = self._reasoning_blocks_by_source.get(source_key)
        if bid is None:
            self._reasoning_turn_seq += 1
            bid = self._block_id("reasoning", self._reasoning_turn_seq, source_key)
        self._active_reasoning_block_id = bid
        self._reasoning_blocks_by_source[source_key] = bid
        self._dispatchable.dispatch(CardEvent.reasoning_started(bid))
        self._active_reasoning_sources.add(source_key)
        self._reasoning_active = True
        return bid

    @staticmethod
    def _source_key(acp_event: ACPEvent) -> str:
        source_id = getattr(acp_event, "source_id", None)
        if source_id and isinstance(source_id, str):
            return source_id.strip() or "main"
        return "main"

    @staticmethod
    def _safe_suffix(source_key: str) -> str:
        suffix = re.sub(r"[^a-zA-Z0-9_-]+", "_", source_key).strip("_")
        return suffix[:40] or "main"

    def _block_id(self, kind: str, seq: int, source_key: str) -> str:
        if source_key == "main":
            return f"_turn_{seq}_{kind}" if seq > 1 else f"_active_{kind}"
        return f"_turn_{seq}_{kind}_{self._safe_suffix(source_key)}"
