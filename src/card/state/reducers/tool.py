"""Tool call sub-reducer."""
from __future__ import annotations

from dataclasses import replace

from ...events import CardEvent, CardEventType
from ...ui_text import UI_TEXT
from ..models import CardState, ToolBlock


def _is_task_tool_name(tool_name: str | None) -> bool:
    return str(tool_name or "").strip().lower() == "task"


def _is_helpful_task_summary(value: str | None) -> bool:
    text = str(value or "").strip()
    return bool(text) and text.lower() not in {"task", "任务", "{", "}", "[", "]"}


def _demote_latest_tool_blocks(blocks: tuple) -> tuple:
    """Clear latest-active flag from existing tool blocks."""
    return tuple(
        replace(block, is_latest_active=False)
        if block.kind == "tool_call" and getattr(block, "is_latest_active", False)
        else block
        for block in blocks
    )


def _promote_last_active_tool(blocks: tuple) -> tuple:
    """Mark the most recently started active tool as latest-active."""
    latest_idx = next(
        (idx for idx in range(len(blocks) - 1, -1, -1)
         if blocks[idx].kind == "tool_call" and blocks[idx].status == "active"),
        None,
    )
    if latest_idx is None:
        return blocks
    updated = tuple(
        replace(block, is_latest_active=(idx == latest_idx))
        if block.kind == "tool_call"
        else block
        for idx, block in enumerate(blocks)
    )
    return updated


def _replace_tool_blocks(
    blocks: tuple,
    *,
    block_id: str,
    replacement: ToolBlock,
) -> tuple:
    """Collapse every copy of one logical tool call into one block."""
    updated: list = []
    inserted = False
    for block in blocks:
        if (
            block.kind == "tool_call"
            and getattr(block, "block_id", "") == block_id
        ):
            if not inserted:
                updated.append(replacement)
                inserted = True
            continue
        updated.append(block)
    if not inserted:
        updated.append(replacement)
    return tuple(updated)


def _footer_for_active_tools(footer, blocks: tuple):
    active = next(
        (
            block
            for block in reversed(blocks)
            if block.kind == "tool_call" and block.status == "active"
        ),
        None,
    )
    if active is None:
        return replace(footer, status=None, status_text=None)
    tool_name = active.tool_name or "工具"
    return replace(
        footer,
        status="tool_running",
        status_text=UI_TEXT["card_tool_running"].format(tool_name=tool_name),
    )


def reduce_tool(state: CardState, event: CardEvent) -> CardState:
    """Handle TOOL_STARTED / TOOL_DELTA / TOOL_DONE / TOOL_FAILED."""
    match event.type:
        case CardEventType.TOOL_STARTED:
            block_id = event.payload.get("block_id", "")
            tool_name = event.payload.get("tool_name", "")
            tool_input = event.payload.get("tool_input", "")
            idx = state.block_index.get(block_id)
            existing = (
                state.blocks[idx]
                if (
                    idx is not None
                    and idx < len(state.blocks)
                    and state.blocks[idx].kind == "tool_call"
                )
                else None
            )
            if existing is not None and existing.status != "active":
                return state
            if existing is None:
                new_block = ToolBlock(
                    block_id=block_id,
                    status="active",
                    tool_name=tool_name,
                    tool_input=tool_input,
                    content="",
                    is_latest_active=True,
                )
            else:
                new_block = replace(
                    existing,
                    status="active",
                    tool_name=tool_name or existing.tool_name,
                    tool_input=tool_input or existing.tool_input,
                    is_latest_active=True,
                )
            blocks = _replace_tool_blocks(
                _demote_latest_tool_blocks(state.blocks),
                block_id=block_id,
                replacement=new_block,
            )
            footer = _footer_for_active_tools(state.footer, blocks)
            return replace(state, blocks=blocks, footer=footer)

        case CardEventType.TOOL_DELTA:
            block_id = event.payload.get("block_id", "")
            content = event.payload.get("content", "")
            # O(1) lookup via block_index
            idx = state.block_index.get(block_id)
            if idx is not None and idx < len(state.blocks) and state.blocks[idx].kind == "tool_call":
                b = state.blocks[idx]
                combined_content = b.content + content
                changes = {"content": combined_content}
                if _is_task_tool_name(b.tool_name) and _is_helpful_task_summary(combined_content):
                    changes["tool_summary"] = str(combined_content).strip()
                updated = replace(b, **changes)
                blocks = state.blocks[:idx] + (updated,) + state.blocks[idx + 1:]
                return replace(state, blocks=blocks)
            return state

        case CardEventType.TOOL_DONE:
            block_id = event.payload.get("block_id", "")
            tool_output = event.payload.get("tool_output", "")
            tool_summary = event.payload.get("tool_summary", "")
            # O(1) lookup via block_index
            idx = state.block_index.get(block_id)
            if idx is not None and idx < len(state.blocks) and state.blocks[idx].kind == "tool_call":
                b = state.blocks[idx]
                # Backward-compatible fallback: some tool adapters stream output via TOOL_DELTA
                # into block.content but don't populate tool_output on TOOL_DONE.
                if not tool_output and getattr(b, "content", ""):
                    tool_output = b.content
                if _is_task_tool_name(b.tool_name):
                    if not _is_helpful_task_summary(tool_summary):
                        if _is_helpful_task_summary(tool_output):
                            tool_summary = str(tool_output).strip()
                        elif _is_helpful_task_summary(b.tool_summary):
                            tool_summary = str(b.tool_summary).strip()
                updated = replace(b, status="completed", is_latest_active=False,
                                     tool_output=tool_output, tool_summary=tool_summary)
                blocks = _replace_tool_blocks(
                    state.blocks,
                    block_id=block_id,
                    replacement=updated,
                )
                blocks = _promote_last_active_tool(blocks)
                footer = _footer_for_active_tools(state.footer, blocks)
                return replace(state, blocks=blocks, footer=footer)
            footer = _footer_for_active_tools(state.footer, state.blocks)
            return replace(state, footer=footer)

        case CardEventType.TOOL_FAILED:
            block_id = event.payload.get("block_id", "")
            error = event.payload.get("error", "")
            # O(1) lookup via block_index
            idx = state.block_index.get(block_id)
            if idx is not None and idx < len(state.blocks) and state.blocks[idx].kind == "tool_call":
                b = state.blocks[idx]
                updated = replace(b, status="failed", tool_output=error, is_latest_active=False)
                blocks = _replace_tool_blocks(
                    state.blocks,
                    block_id=block_id,
                    replacement=updated,
                )
                blocks = _promote_last_active_tool(blocks)
                footer = _footer_for_active_tools(state.footer, blocks)
                return replace(state, blocks=blocks, footer=footer)
            footer = _footer_for_active_tools(state.footer, state.blocks)
            return replace(state, footer=footer)

    return state
