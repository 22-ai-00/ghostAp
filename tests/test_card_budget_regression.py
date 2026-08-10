"""Budget regression: extreme states must not exceed Feishu limits."""
from __future__ import annotations

import json

from src.card.render.budget import RenderBudget
from src.card.render.renderer import render_card
from src.card.shared.truncation import count_tagged_nodes
from src.card.state.models import CardMetadata, CardState, ContentBlock, HeaderState
from src.card.state.runtime_stats import RuntimeStats


def _build_extreme_state() -> CardState:
    blocks_list = []
    tasks = tuple(
        {
            "task_id": f"t{i}",
            "name": f"task {i}",
            "status": "completed" if i < 5 else ("in_progress" if i == 5 else "pending"),
        }
        for i in range(30)
    )
    blocks_list.append(ContentBlock(
        kind="task_list",
        block_id="tl",
        tasks=tasks,
        current_task_id="t5",
    ))
    for i in range(100):
        is_active = i >= 50
        blocks_list.append(ContentBlock(
            kind="tool_call",
            block_id=f"tool_{i}",
            tool_name="Edit" if i % 3 == 0 else ("Bash" if i % 3 == 1 else "Grep"),
            tool_summary=f"summary {i}",
            content="x" * 100,
            status="active" if is_active else "completed",
            is_latest_active=i == 99,
        ))
    blocks = tuple(blocks_list)
    state = CardState(
        metadata=CardMetadata(mode_name="Deep", mode_emoji="🧠", engine_type="deep"),
        header=HeaderState(title="Deep"),
        blocks=blocks,
        terminal="running",
    )
    object.__setattr__(state, "runtime_stats", RuntimeStats(elapsed_seconds=600.0, deep_phase="executing"))
    return state


def test_extreme_state_no_page_exceeds_node_limit():
    state = _build_extreme_state()

    pages = render_card(state, RenderBudget())

    for i, page in enumerate(pages):
        nodes = count_tagged_nodes(page._card_json)
        assert nodes <= 200, f"page {i} has {nodes} nodes > 200"


def test_extreme_state_no_page_exceeds_byte_limit():
    state = _build_extreme_state()

    pages = render_card(state, RenderBudget())

    for i, page in enumerate(pages):
        size = len(json.dumps(page._card_json).encode("utf-8"))
        assert size <= 30 * 1024, f"page {i} has {size} bytes > 30KB"


def test_extreme_state_sticky_head_node_cap():
    from src.card.render.sticky_head import STICKY_HEAD_MAX_NODES, build_sticky_head

    state = _build_extreme_state()

    sticky = build_sticky_head(state, state.metadata)

    total_nodes = sum(atom.node_count for atom in sticky)
    assert total_nodes <= STICKY_HEAD_MAX_NODES, f"sticky_head has {total_nodes} nodes > cap"


def test_extreme_state_continuation_pages_have_sticky():
    state = _build_extreme_state()

    pages = render_card(state, RenderBudget())

    for i, page in enumerate(pages):
        body = page._card_json["body"]["elements"]
        first = body[0]
        assert first.get("tag") == "markdown"
        assert "Deep" in first.get("content", ""), f"page {i} missing sticky banner"


def test_many_active_tool_lines_paginate_under_official_element_limit():
    blocks = tuple(
        ContentBlock(
            kind="tool_call",
            block_id=f"active-{idx}",
            tool_name="Read",
            tool_input=f'{{"path": "src/module_{idx}.py"}}',
            status="active",
            is_latest_active=idx == 79,
        )
        for idx in range(80)
    )
    state = CardState(
        metadata=CardMetadata(mode_name="Coco", mode_emoji="🛠️", engine_type="normal"),
        header=HeaderState(title="Many tools"),
        blocks=blocks,
        terminal="running",
    )

    pages = render_card(state, RenderBudget())

    assert len(pages) > 1
    for i, page in enumerate(pages):
        nodes = count_tagged_nodes(page._card_json)
        assert nodes <= 200, f"page {i} has {nodes} nodes > 200"
