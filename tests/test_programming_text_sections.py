"""Ordinary programming prose → source-aware Card 2.0 section regressions."""

from __future__ import annotations

import json
from dataclasses import replace

from src.card.events import CardEvent
from src.card.programming_adapter import build_programming_metadata
from src.card.render.atoms import flatten_to_atoms
from src.card.render.budget import RenderBudget
from src.card.render.payload_truncator import count_tagged_nodes
from src.card.render.renderer import render_card
from src.card.state.models import CardMetadata, CardState, ContentBlock
from src.card.state.reducers.lifecycle import reduce_lifecycle

_SECTION_BACKGROUNDS = {"blue-50", "green-50", "grey-50", "red-50"}


def _section_panels(state: CardState, budget: RenderBudget | None = None) -> list[dict]:
    cards = render_card(state, budget or RenderBudget())
    return [
        element
        for card in cards
        for element in card._card_json["body"]["elements"]
        if element.get("tag") == "collapsible_panel"
        and element.get("background_color") in _SECTION_BACKGROUNDS
    ]


def _panel_title(panel: dict) -> str:
    return str(panel.get("header", {}).get("title", {}).get("content", ""))


def _panel_body(panel: dict) -> dict:
    return next(
        element
        for element in panel.get("elements", [])
        if element.get("tag") == "markdown"
    )


def _programming_state(
    *blocks,
    terminal: str = "running",
    metadata=None,
) -> CardState:
    return CardState(
        blocks=tuple(blocks),
        terminal=terminal,
        metadata=metadata or build_programming_metadata("codex", model_name="gpt-5"),
    )


def test_each_programming_text_block_renders_as_one_bordered_section():
    state = _programming_state(
        ContentBlock(kind="text", block_id="t1", content="先核对计划。", status="completed"),
        ContentBlock(kind="text", block_id="t2", content="再检查代码。", status="active"),
    )

    panels = _section_panels(state)

    assert len(panels) == 2
    assert [_panel_title(panel) for panel in panels] == [
        "**<font color='blue'>主 Agent · 进展 1</font>**",
        "**<font color='blue'>主 Agent · 当前进展</font>**",
    ]
    assert [_panel_body(panel)["content"] for panel in panels] == [
        "先核对计划。",
        "再检查代码。",
    ]
    assert all(panel["expanded"] is True for panel in panels)
    assert all(panel["border"]["corner_radius"] == "8px" for panel in panels)


def test_one_text_block_with_blank_lines_stays_one_semantic_section():
    content = "先核对计划。\n\n再检查代码。\n\n```python\nprint('ok')\n```"
    state = _programming_state(
        ContentBlock(kind="text", block_id="t1", content=content, status="completed"),
    )

    panels = _section_panels(state)

    assert len(panels) == 1
    assert _panel_body(panels[0])["content"] == content


def test_empty_programming_text_block_does_not_render_a_section():
    state = _programming_state(
        ContentBlock(
            kind="text",
            block_id="_active_text",
            content="",
            element_id="el__active_text",
            status="active",
        ),
    )

    assert _section_panels(state) == []


def test_terminal_last_main_section_is_the_green_final_answer():
    state = _programming_state(
        ContentBlock(kind="text", block_id="t1", content="已完成预检。", status="completed"),
        ContentBlock(kind="text", block_id="t2", content="修复完成并通过测试。", status="completed"),
        terminal="completed",
    )

    panels = _section_panels(state)

    assert [panel["background_color"] for panel in panels] == [
        "blue-50",
        "green-50",
    ]
    assert "主 Agent · 进展 1" in _panel_title(panels[0])
    assert "最终答复" in _panel_title(panels[1])
    assert panels[1]["border"]["color"] == "green-100"


def test_programming_terminal_section_style_matrix():
    running = _programming_state(
        ContentBlock(
            kind="text",
            block_id="main-answer",
            content="主 Agent 已完成实现。",
            status="completed",
        ),
    )

    completed = reduce_lifecycle(running, CardEvent.completed())
    failed = reduce_lifecycle(running, CardEvent.failed("验证失败"))
    cancelled = reduce_lifecycle(running, CardEvent.cancelled())

    completed_panels = _section_panels(completed)
    failed_panels = _section_panels(failed)
    cancelled_panels = _section_panels(cancelled)

    assert [panel["background_color"] for panel in completed_panels] == [
        "green-50",
    ]
    assert "最终答复" in _panel_title(completed_panels[0])
    assert [panel["background_color"] for panel in failed_panels] == [
        "blue-50",
    ]
    assert [panel["background_color"] for panel in cancelled_panels] == [
        "blue-50",
    ]
    assert all(
        "最终答复" not in _panel_title(panel)
        for panel in [*failed_panels, *cancelled_panels]
    )

    failed_body = render_card(
        failed,
        RenderBudget(),
    )[0]._card_json["body"]["elements"]
    cancelled_body = render_card(
        cancelled,
        RenderBudget(),
    )[0]._card_json["body"]["elements"]
    assert any(
        element.get("tag") == "markdown"
        and "错误摘要" in str(element.get("content") or "")
        for element in failed_body
    )
    assert any(
        element.get("tag") == "markdown"
        and "已取消" in str(element.get("content") or "")
        for element in cancelled_body
    )


def test_system_text_is_a_hard_section_and_pagination_boundary():
    running = _programming_state(
        ContentBlock(
            kind="text",
            block_id="one-character-answer",
            content="甲\n",
            status="completed",
        ),
    )
    failed = reduce_lifecycle(running, CardEvent.failed("验证失败"))

    atoms = flatten_to_atoms(
        failed.blocks,
        RenderBudget(),
        unified_execution=True,
        terminal=True,
        segmented_text=True,
    )
    error_atom = next(atom for atom in atoms if atom.block_id == "_error")
    assert error_atom.node_count == 1
    assert error_atom.structural_overhead == 0

    body = render_card(
        failed,
        RenderBudget(),
    )[0]._card_json["body"]["elements"]
    panels = _section_panels(failed)
    assert len(panels) == 1
    assert _panel_body(panels[0])["content"] == "甲\n"
    assert any(
        element.get("tag") == "markdown"
        and str(element.get("content") or "").startswith("❌ **错误摘要**")
        for element in body
    )


def test_subagent_section_uses_latest_safe_task_brief_by_visible_sequence():
    metadata = replace(
        build_programming_metadata("codex"),
        subagents=(
            {
                "sequence": "1.a",
                "label": "核查后半计划",
                "tool": "Explore",
                "status": "running",
            },
        ),
    )
    state = _programming_state(
        ContentBlock(
            kind="text",
            block_id="subtask-text",
            content="发现两处生命周期矛盾。",
            status="active",
            source_kind="subagent",
            source_sequence="1.a",
            source_label="子任务",
            source_ref="src_a",
        ),
        metadata=metadata,
    )

    panels = _section_panels(state)

    assert len(panels) == 1
    assert panels[0]["background_color"] == "grey-50"
    assert panels[0]["border"]["color"] == "grey-200"
    assert "子代理 · 核查后半计划" in _panel_title(panels[0])
    assert _panel_body(panels[0])["content"] == "发现两处生命周期矛盾。"


def test_subagent_heading_sanitizes_markdown_controls_secrets_and_call_ids():
    secret = "sk-0123456789abcdef"
    unsafe = (
        "\x1b[31mcall_private_123 API_TOKEN="
        f"{secret} ![](/tmp/private.png)\u202e"
    )
    metadata = replace(
        build_programming_metadata("codex"),
        subagents=(
            {
                "sequence": "1.a",
                "label": unsafe,
                "tool": "agent",
                "status": "running",
            },
        ),
    )
    state = _programming_state(
        ContentBlock(
            kind="text",
            block_id="subtask-text",
            content="安全正文",
            source_kind="subagent",
            source_sequence="1.a",
            source_label=unsafe,
            source_ref="src_a",
        ),
        metadata=metadata,
    )

    rendered = json.dumps(
        render_card(state, RenderBudget())[0]._card_json,
        ensure_ascii=False,
    )

    assert "子代理 ·" in rendered
    assert secret not in rendered
    assert "call_private_123" not in rendered
    assert "\x1b" not in rendered
    assert "![](" not in rendered
    assert "\u202e" not in rendered


def test_different_text_sources_do_not_coalesce_one_character_fragment():
    state = _programming_state(
        ContentBlock(
            kind="text",
            block_id="a",
            content="甲\n",
            status="completed",
            source_kind="subagent",
            source_sequence="1.a",
            source_label="任务甲",
            source_ref="src_a",
        ),
        ContentBlock(
            kind="text",
            block_id="b",
            content="乙方输出",
            status="completed",
            source_kind="subagent",
            source_sequence="1.b",
            source_label="任务乙",
            source_ref="src_b",
        ),
    )

    panels = _section_panels(state)

    assert len(panels) == 2
    assert [_panel_body(panel)["content"] for panel in panels] == ["甲\n", "乙方输出"]


def test_same_source_one_character_fragment_keeps_existing_soft_join():
    state = _programming_state(
        ContentBlock(
            kind="text",
            block_id="a",
            content="数\n",
            status="completed",
            source_ref="main",
        ),
        ContentBlock(
            kind="text",
            block_id="empty",
            content="",
            status="completed",
            source_ref="main",
        ),
        ContentBlock(
            kind="text",
            block_id="b",
            content="字很大",
            status="completed",
            source_ref="main",
        ),
    )

    panels = _section_panels(state)

    assert len(panels) == 1
    assert _panel_body(panels[0])["content"] == "数字很大"


def test_coalesced_main_fragment_keeps_contiguous_section_ordinals():
    state = _programming_state(
        ContentBlock(
            kind="text",
            block_id="a",
            content="数\n",
            status="completed",
            source_ref="main",
        ),
        ContentBlock(
            kind="text",
            block_id="ordinal-empty",
            content="",
            status="completed",
            source_ref="main",
        ),
        ContentBlock(
            kind="text",
            block_id="b",
            content="字很大",
            status="completed",
            source_ref="main",
        ),
        ContentBlock(
            kind="text",
            block_id="c",
            content="第三段",
            status="active",
            source_ref="main",
        ),
    )

    panels = _section_panels(state)

    assert [_panel_title(panel) for panel in panels] == [
        "**<font color='blue'>主 Agent · 进展 1</font>**",
        "**<font color='blue'>主 Agent · 当前进展</font>**",
    ]


def test_nested_streaming_markdown_keeps_element_id_and_stable_signature():
    metadata = build_programming_metadata("codex")
    before = _programming_state(
        ContentBlock(
            kind="text",
            block_id="stream",
            content="正在检查",
            element_id="el_stream",
            status="active",
        ),
        metadata=metadata,
    )
    after = _programming_state(
        ContentBlock(
            kind="text",
            block_id="stream",
            content="正在检查渲染链路",
            element_id="el_stream",
            status="active",
        ),
        metadata=metadata,
    )

    first = render_card(before, RenderBudget())[0]
    second = render_card(after, RenderBudget())[0]
    body = _panel_body(_section_panels(before)[0])

    assert body["element_id"] == "el_stream"
    assert first.active_element is not None
    assert first.active_element.element_id == "el_stream"
    assert first.active_element.text == "正在检查"
    assert first.structure_signature == second.structure_signature


def test_nested_streaming_element_id_change_is_structural():
    metadata = build_programming_metadata("codex")
    first = render_card(
        _programming_state(
            ContentBlock(
                kind="text",
                block_id="stream",
                content="正在检查",
                element_id="el_stream_1",
                status="active",
            ),
            metadata=metadata,
        ),
        RenderBudget(),
    )[0]
    second = render_card(
        _programming_state(
            ContentBlock(
                kind="text",
                block_id="stream",
                content="正在检查",
                element_id="el_stream_2",
                status="active",
            ),
            metadata=metadata,
        ),
        RenderBudget(),
    )[0]

    assert first.structure_signature != second.structure_signature


def test_non_selected_active_source_content_changes_page_signature():
    metadata = build_programming_metadata("codex")
    main = ContentBlock(
        kind="text",
        block_id="main-stream",
        content="主 Agent 稳定输出",
        element_id="el_main",
        status="active",
    )
    subagent_before = ContentBlock(
        kind="text",
        block_id="subagent-stream",
        content="子代理第一帧",
        element_id="el_subagent",
        status="active",
        source_kind="subagent",
        source_sequence="1.a",
        source_label="核查流式更新",
        source_ref="src_a",
    )
    subagent_after = replace(
        subagent_before,
        content="子代理第二帧继续增长",
    )

    first = render_card(
        _programming_state(main, subagent_before, metadata=metadata),
        RenderBudget(),
    )[0]
    second = render_card(
        _programming_state(main, subagent_after, metadata=metadata),
        RenderBudget(),
    )[0]

    assert first.active_element is not None
    assert first.active_element.element_id == "el_main"
    assert second.active_element is not None
    assert second.active_element.element_id == "el_main"
    assert first.structure_signature != second.structure_signature


def test_bridge_phrase_prefixes_section_body_not_subagent_heading():
    metadata = replace(
        build_programming_metadata("codex"),
        bridge_phrase="承接上一张卡片：",
        subagents=(
            {
                "sequence": "1.a",
                "label": "核查后半计划",
                "status": "running",
            },
        ),
    )
    state = _programming_state(
        ContentBlock(
            kind="text",
            block_id="subtask-text",
            content="发现两处矛盾。",
            source_kind="subagent",
            source_sequence="1.a",
            source_label="核查后半计划",
            source_ref="src_a",
        ),
        metadata=metadata,
    )

    panel = _section_panels(state)[0]

    assert "承接上一张卡片" not in _panel_title(panel)
    assert _panel_body(panel)["content"].startswith(
        "承接上一张卡片：\n\n发现两处矛盾。"
    )


def test_engine_text_keeps_existing_plain_markdown_contract():
    state = CardState(
        blocks=(
            ContentBlock(
                kind="text",
                block_id="deep-text",
                content="Deep 正文",
                status="completed",
            ),
        ),
        metadata=CardMetadata(engine_type="deep", mode_name="Deep"),
    )

    body = render_card(state, RenderBudget())[0]._card_json["body"]["elements"]

    assert any(
        element.get("tag") == "markdown"
        and element.get("content") == "Deep 正文"
        for element in body
    )
    assert not any(
        element.get("tag") == "collapsible_panel"
        and element.get("background_color") in _SECTION_BACKGROUNDS
        for element in body
    )


def test_segmented_text_preserves_existing_summary_and_history_fold_contracts():
    metadata = replace(
        build_programming_metadata("codex"),
        subagents=(
            {
                "sequence": "1.a",
                "label": "核查分页边界",
                "tool": "Explore",
                "status": "completed",
            },
        ),
    )
    state = _programming_state(
        ContentBlock(
            kind="text",
            block_id="answer",
            content="修复完成。",
            status="completed",
        ),
        ContentBlock(
            kind="tool_call",
            block_id="command",
            status="completed",
            tool_name="bash",
            tool_input="uv run python -m pytest tests/test_example.py -q",
            tool_output="1 passed",
        ),
        terminal="completed",
        metadata=metadata,
    )

    body = render_card(state, RenderBudget())[0]._card_json["body"]["elements"]
    subtask_panel = next(
        element
        for element in body
        if "并行子任务" in _panel_title(element)
    )
    history_panel = next(
        element
        for element in body
        if "执行记录" in _panel_title(element)
    )

    assert subtask_panel["expanded"] is False
    assert subtask_panel["border"] == {
        "color": "orange",
        "corner_radius": "8px",
    }
    assert subtask_panel["padding"] == "8px 16px"
    assert subtask_panel["header"]["icon"] == {
        "tag": "standard_icon",
        "token": "down-small-ccm_outlined",
        "size": "16px 16px",
    }

    assert history_panel["expanded"] is False
    assert history_panel["border"] == {
        "color": "blue",
        "corner_radius": "8px",
    }
    assert history_panel["padding"] == "4px 12px"
    assert history_panel["header"]["icon"] == {
        "tag": "standard_icon",
        "token": "down-small-ccm_outlined",
        "size": "16px 16px",
    }


def test_many_short_sections_paginate_under_official_limits():
    blocks = tuple(
        ContentBlock(
            kind="text",
            block_id=f"text-{index}",
            content=f"阶段 {index} 已完成一项检查。",
            status="completed",
        )
        for index in range(80)
    )
    state = _programming_state(*blocks)

    cards = render_card(state, RenderBudget())

    assert len(cards) > 1
    for card in cards:
        assert count_tagged_nodes(card._card_json) <= 200
        assert len(
            json.dumps(card._card_json, ensure_ascii=False).encode("utf-8")
        ) <= 30 * 1024


def test_safe_chinese_section_stays_on_one_card_without_tail_page():
    content = "汉" * 3000
    state = _programming_state(
        ContentBlock(
            kind="text",
            block_id="safe-chinese",
            content=content,
            status="completed",
        ),
    )

    cards = render_card(state, RenderBudget())

    assert len(cards) == 1
    assert _panel_body(_section_panels(state)[0])["content"] == content
    assert len(
        json.dumps(cards[0]._card_json, ensure_ascii=False).encode("utf-8")
    ) < 27 * 1024


def test_huge_section_recursively_paginates_with_no_content_loss():
    content = "汉" * 30000
    state = _programming_state(
        ContentBlock(
            kind="text",
            block_id="huge-chinese",
            content=content,
            status="completed",
        ),
    )

    cards = render_card(state, RenderBudget())
    rendered_parts = [
        _panel_body(panel)["content"]
        for panel in _section_panels(state)
    ]

    assert len(cards) > 2
    assert "".join(rendered_parts) == content
    for card in cards:
        assert count_tagged_nodes(card._card_json) <= 200
        assert len(
            json.dumps(card._card_json, ensure_ascii=False).encode("utf-8")
        ) <= 30 * 1024


def test_long_subagent_titles_stay_within_page_byte_budget():
    blocks = tuple(
        ContentBlock(
            kind="text",
            block_id=f"subagent-{index}",
            content=f"子任务 {index} 完成核查。",
            status="completed",
            source_kind="subagent",
            source_sequence=f"1.{index}",
            source_label="核" * 60,
            source_ref=f"src_{index}",
        )
        for index in range(80)
    )
    state = _programming_state(*blocks)

    cards = render_card(state, RenderBudget())

    assert len(cards) > 1
    for card in cards:
        assert count_tagged_nodes(card._card_json) <= 200
        assert len(
            json.dumps(card._card_json, ensure_ascii=False).encode("utf-8")
        ) <= 30 * 1024
