"""Regressions for ordinary programming-card continuation density."""

from __future__ import annotations

import json
from dataclasses import replace

from src.card.events import CardEvent, CardEventType
from src.card.programming_adapter import (
    PROGRAMMING_PROGRESS_BLOCKS_PER_CARD,
    ProgrammingCardSession,
    build_programming_metadata,
)
from src.card.render.budget import RenderBudget
from src.card.render.renderer import render_card
from src.card.state.models import CardState, TextBlock, ToolBlock
from src.card.state.reducer import MAX_TOTAL_BLOCKS, reduce_card_state


def _programming_state(*, progress_blocks: int, completed_tools: int) -> CardState:
    blocks = tuple(
        TextBlock(
            block_id=f"progress-{index}",
            content=f"进展 {index}",
            status="completed",
        )
        for index in range(progress_blocks)
    ) + tuple(
        ToolBlock(
            block_id=f"tool-{index}",
            tool_name="Read",
            tool_input=f"src/module_{index}.py",
            tool_summary=f"读取模块 {index}",
            status="completed",
        )
        for index in range(completed_tools)
    )
    return CardState(
        metadata=build_programming_metadata("codex"),
        blocks=blocks,
        terminal="running",
    )


def test_tool_volume_does_not_rotate_before_twelve_visible_progress_blocks() -> None:
    state = _programming_state(
        progress_blocks=PROGRAMMING_PROGRESS_BLOCKS_PER_CARD - 1,
        completed_tools=100,
    )

    assert ProgrammingCardSession._requires_capacity_rotation(
        state,
        CardEvent.text_started("progress-next"),
    ) is False


def test_thirteenth_visible_progress_block_starts_a_continuation_card() -> None:
    state = _programming_state(
        progress_blocks=PROGRAMMING_PROGRESS_BLOCKS_PER_CARD,
        completed_tools=0,
    )

    assert ProgrammingCardSession._requires_capacity_rotation(
        state,
        CardEvent.text_started("progress-next"),
    ) is True


def test_raw_history_keeps_a_hard_capacity_fallback() -> None:
    state = _programming_state(
        progress_blocks=0,
        completed_tools=(
            MAX_TOTAL_BLOCKS * PROGRAMMING_PROGRESS_BLOCKS_PER_CARD
        ),
    )

    assert ProgrammingCardSession._requires_capacity_rotation(
        state,
        CardEvent.tool_started("tool-overflow", "Read", "src/overflow.py"),
    ) is True


def test_twelve_short_progress_blocks_and_folded_tools_fit_one_wire_card() -> None:
    state = _programming_state(
        progress_blocks=PROGRAMMING_PROGRESS_BLOCKS_PER_CARD,
        completed_tools=100,
    )

    cards = render_card(state, RenderBudget())
    payload = json.dumps(cards[0].to_feishu_json(), ensure_ascii=False)

    assert len(cards) == 1
    assert payload.count("主 Agent · 进展") == PROGRAMMING_PROGRESS_BLOCKS_PER_CARD
    assert "📋 **执行记录** · 100 步" in payload
    assert len(payload.encode("utf-8")) <= 27 * 1024


def test_programming_projection_keeps_raw_history_until_semantic_rotation() -> None:
    metadata = build_programming_metadata("codex")
    state = reduce_card_state(None, CardEvent.started(), metadata)

    for index in range(110):
        block_id = f"tool-{index}"
        state = reduce_card_state(
            state,
            CardEvent.tool_started(block_id, "Read", f"src/{index}.py"),
            metadata,
        )
        state = reduce_card_state(
            state,
            CardEvent(
                type=CardEventType.TOOL_DONE,
                payload={
                    "block_id": block_id,
                    "tool_output": "done",
                    "tool_summary": "Read",
                },
            ),
            metadata,
        )

    assert len([block for block in state.blocks if block.kind == "tool_call"]) == 110


def test_archived_continuation_card_renders_one_canonical_next_card_hint() -> None:
    metadata = build_programming_metadata("codex")
    metadata = replace(
        metadata,
        continuation_seq=1,
        card_sequence=2,
    )
    running = reduce_card_state(None, CardEvent.started(), metadata)
    archived = reduce_card_state(
        running,
        CardEvent.archived(
            sequence=2,
            new_message_id="om_next",
            bridge_phrase="续接 #3 ↓",
            append_hint=False,
        ),
        metadata,
    )

    payload = json.dumps(
        render_card(archived, RenderBudget())[0].to_feishu_json(),
        ensure_ascii=False,
    )

    assert payload.count("本卡已停止更新") == 1
    assert payload.count("续接 #3 ↓") == 1
    assert "续接 #2 ↓" not in payload
