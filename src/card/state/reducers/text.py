"""Text block sub-reducer."""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace

from ...events import CardEvent, CardEventType
from ...text_stream import append_stream_text
from ..models import CardState, TextBlock

_VALID_ELEMENT_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,19}$")


def _text_element_id(block_id: str) -> str:
    """Build a stable Card 2.0 element_id (letter-first, at most 20 chars)."""
    candidate = f"el_{block_id}"
    if _VALID_ELEMENT_ID.fullmatch(candidate):
        return candidate
    digest = hashlib.sha256(
        str(block_id).encode("utf-8", errors="replace")
    ).hexdigest()
    return f"el_{digest[:16]}"


def reduce_text(state: CardState, event: CardEvent) -> CardState:
    """Handle TEXT_STARTED / TEXT_DELTA / TEXT_DONE."""
    match event.type:
        case CardEventType.TEXT_STARTED:
            block_id = event.payload.get("block_id", "")
            source_kind = event.payload.get("source_kind", "main")
            if source_kind not in {"main", "subagent"}:
                source_kind = "main"
            source_sequence = str(
                event.payload.get("source_sequence") or ""
            ).strip() or None
            source_label = str(
                event.payload.get("source_label") or ""
            ).strip() or None
            source_ref = str(
                event.payload.get("source_ref") or "main"
            ).strip() or "main"
            new_block = TextBlock(
                block_id=block_id,
                status="active",
                element_id=_text_element_id(block_id),
                source_kind=source_kind,
                source_sequence=source_sequence,
                source_label=source_label,
                source_ref=source_ref,
            )
            return replace(state, blocks=state.blocks + (new_block,),
                           footer=replace(state.footer, status="thinking", status_text="💭 正在思考..."))

        case CardEventType.TEXT_DELTA:
            block_id = event.payload.get("block_id", "")
            text = event.payload.get("text", "")
            # O(1) lookup via block_index
            idx = state.block_index.get(block_id)
            if idx is not None and idx < len(state.blocks) and state.blocks[idx].kind == "text":
                b = state.blocks[idx]
                updated = replace(b, content=append_stream_text(b.content, text))
                blocks = state.blocks[:idx] + (updated,) + state.blocks[idx + 1:]
                return replace(state, blocks=blocks)
            # Auto-create block for convenience (from_acp uses "_active_text").
            text = text.lstrip("\n")
            new_block = TextBlock(block_id=block_id, status="active",
                                     element_id=_text_element_id(block_id), content=text)
            return replace(state, blocks=state.blocks + (new_block,),
                           footer=replace(state.footer, status="thinking", status_text="💭 正在思考..."))

        case CardEventType.TEXT_DONE:
            block_id = event.payload.get("block_id", "")
            idx = state.block_index.get(block_id)
            if idx is not None and idx < len(state.blocks) and state.blocks[idx].kind == "text":
                b = state.blocks[idx]
                updated = replace(b, status="completed", element_id=None)
                blocks = state.blocks[:idx] + (updated,) + state.blocks[idx + 1:]
                return replace(state, blocks=blocks,
                               footer=replace(state.footer, status=None, status_text=None))
            return replace(state, footer=replace(state.footer, status=None, status_text=None))

    return state
