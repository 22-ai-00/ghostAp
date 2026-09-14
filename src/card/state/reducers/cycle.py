"""Cycle lifecycle sub-reducer for Spec engine."""
from __future__ import annotations

import logging
from dataclasses import replace

from ...events import CardEvent, CardEventType
from ..models import CardState
from ._shared import build_header

logger = logging.getLogger(__name__)


def reduce_cycle(state: CardState, event: CardEvent) -> CardState:
    """Handle CYCLE_STARTED / CYCLE_DONE events."""
    if state.engine_ext is None:
        logger.warning("reduce_cycle called with engine_ext=None, event=%s", event.type)
        return state
    match event.type:
        case CardEventType.CYCLE_STARTED:
            cycle_num = event.payload.get("cycle_num", 1)
            max_cycles = event.payload.get("max_cycles", 1)
            ext = replace(state.engine_ext, cycle_num=cycle_num, max_cycles=max_cycles)
            metadata = replace(state.metadata, iteration_index=cycle_num, iteration_total=max_cycles)
            footer = replace(state.footer, status="tool_running",
                             status_text=f"⏳ 迭代 {cycle_num}/{max_cycles}")
            # Inject iteration/cycle info into header subtitle
            if max_cycles > 1:
                subtitle = f"Cycle {cycle_num}/{max_cycles}"
            else:
                subtitle = f"Iteration {cycle_num}"
            base_header = build_header(metadata, "running")
            header = replace(base_header, subtitle=subtitle) if state.header else None
            changes: dict = {"engine_ext": ext, "footer": footer, "terminal": "running", "metadata": metadata}
            if header:
                changes["header"] = header
            # Clear old content blocks on new cycle start to prevent unbounded
            # accumulation in truncation mode (when session rotation is capped).
            # For fresh sessions after successful rotation, blocks is already ()
            # so this is effectively a no-op.
            if state.blocks and not state.metadata.retain_full_history:
                changes["blocks"] = ()
            return replace(state, **changes)

        case CardEventType.CYCLE_DONE:
            cycle_num = event.payload.get("cycle_num", state.engine_ext.cycle_num)
            ext = replace(state.engine_ext, cycle_num=cycle_num)
            footer = replace(state.footer, status="idle", status_text=None)
            return replace(state, engine_ext=ext, footer=footer)

    return state
