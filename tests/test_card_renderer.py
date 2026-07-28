"""Tests for src/card/render/renderer.py — main render entry point."""


import json

import pytest

from src.card.events import CardEvent
from src.card.render.budget import RenderBudget
from src.card.render.renderer import (
    _assemble_card_json,
    compute_structure_signature,
    render_card,
)
from src.card.state.models import (
    ButtonSpec,
    CardMetadata,
    CardState,
    ContentBlock,
    FooterState,
    HeaderState,
    TaskListBlock,
)
from src.card.state.reducer import reduce_card_state


def _iter_dict_nodes(obj):
    """Yield all dict nodes in a nested (dict/list) structure."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _iter_dict_nodes(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from _iter_dict_nodes(it)


class TestRenderCardBasic:
    """Basic render_card() behavior."""

    def test_empty_state_returns_single_page(self):
        state = CardState()
        cards = render_card(state, RenderBudget())
        assert len(cards) == 1
        assert cards[0].page_index == 0
        assert cards[0].total_pages == 1

    def test_single_text_block(self):
        state = CardState(
            blocks=(ContentBlock(kind="text", block_id="t1", content="Hello world"),),
            header=HeaderState(title="Test", template="blue"),
        )
        cards = render_card(state, RenderBudget())
        assert len(cards) == 1
        card_json = cards[0]._card_json
        assert card_json["schema"] == "2.0"
        assert card_json["header"]["title"]["content"] == "Test"
        # body should have the text element
        body = card_json["body"]["elements"]
        assert any(el.get("content") == "Hello world" for el in body)

    def test_card_json_structure(self):
        state = CardState(
            blocks=(ContentBlock(kind="text", block_id="t1", content="content"),),
            header=HeaderState(title="Title", template="green"),
        )
        cards = render_card(state, RenderBudget())
        card_json = cards[0]._card_json
        assert "schema" in card_json
        assert "config" in card_json
        assert "header" in card_json
        assert "body" in card_json
        assert card_json["config"]["wide_screen_mode"] is True
        assert card_json["config"]["update_multi"] is True

    def test_image_block_renders_inline_feishu_image(self):
        state = CardState(
            blocks=(
                ContentBlock(
                    kind="image",
                    block_id="image:sha256-artifact",
                    image_key="img_generated",
                    alt="页面截图",
                    status="completed",
                ),
            ),
            header=HeaderState(title="Test", template="blue"),
        )

        body = render_card(state, RenderBudget())[0]._card_json["body"]["elements"]

        assert {
            "tag": "img",
            "img_key": "img_generated",
            "alt": {"tag": "plain_text", "content": "页面截图"},
        } in body

    def test_failed_image_block_renders_non_sensitive_fallback(self):
        state = CardState(
            blocks=(
                ContentBlock(
                    kind="image",
                    block_id="image:sha256-artifact",
                    alt="页面截图",
                    status="failed",
                ),
            ),
            header=HeaderState(title="Test", template="green"),
            terminal="completed",
        )

        body = render_card(state, RenderBudget())[0]._card_json["body"]["elements"]

        assert any(
            element.get("tag") == "markdown"
            and "图片产物暂时无法展示" in element.get("content", "")
            for element in body
        )

    def test_tainted_image_alt_cannot_break_render_or_signature_hashing(self):
        state = CardState(
            blocks=(
                ContentBlock(
                    kind="image",
                    block_id="image:tainted",
                    image_key="img_generated",
                    alt="screen\udcff\x00\nshot.png",
                    status="completed",
                ),
            ),
            header=HeaderState(title="Test", template="blue"),
        )

        signature = compute_structure_signature(state)
        rendered = render_card(state, RenderBudget())[0]
        image = next(
            element
            for element in rendered._card_json["body"]["elements"]
            if element.get("tag") == "img"
        )

        assert len(signature) == 32
        assert len(rendered.structure_signature) == 32
        assert image["alt"]["content"] == "screen� shot.png"
        json.dumps(rendered._card_json, ensure_ascii=False).encode("utf-8")


class TestSubtaskCardProjection:
    @pytest.mark.parametrize(
        ("terminal", "status_label", "template"),
        [
            pytest.param("running", "执行中", "orange", id="running"),
            pytest.param("completed", "已完成", "green", id="completed"),
            pytest.param("failed", "执行失败", "red", id="failed"),
        ],
    )
    def test_subtask_header_uses_human_label_and_terminal_style_only(
        self,
        terminal,
        status_label,
        template,
    ):
        state = CardState(
            terminal=terminal,
            header=HeaderState(title="legacy title", template="blue"),
            metadata=CardMetadata(
                project_name="ghostAp-internal",
                mode_name="Deep · Codex",
                tool_name="codex-internal",
                model_name="gpt-internal",
                engine_type="deep",
                unit_kind="subagent",
                unit_id="call_internal_123",
                unit_label="完成独立最终代码审查",
                card_sequence="1.e",
                is_subagent=True,
                parent_card_seq="1",
            ),
        )

        header = render_card(state, RenderBudget())[0]._card_json["header"]

        assert header["title"]["content"] == "🧬 子任务 · 完成独立最终代码审查"
        assert header["subtitle"]["content"] == (
            f"{status_label} · 子卡 #1.e · 来自主卡 #1"
        )
        assert header["template"] == template
        rendered = str(header)
        for internal_value in (
            "call_internal_123",
            "ghostAp-internal",
            "codex-internal",
            "gpt-internal",
        ):
            assert internal_value not in rendered

    @pytest.mark.parametrize(
        ("is_subagent", "expanded", "border_color"),
        [
            pytest.param(False, True, "indigo", id="parent"),
            pytest.param(True, False, "grey", id="child"),
        ],
    )
    def test_only_subtask_cards_use_compact_task_progress(
        self,
        is_subagent,
        expanded,
        border_color,
    ):
        task_list = TaskListBlock(
            block_id="tasks",
            current_task_id="t5",
            tasks=(
                {"task_id": "t1", "name": "探索代码", "status": "completed"},
                {"task_id": "t2", "name": "修复路由", "status": "completed"},
                {"task_id": "t3", "name": "单元测试", "status": "completed"},
                {"task_id": "t4", "name": "集成测试", "status": "completed"},
                {"task_id": "t5", "name": "最终代码审查", "status": "in_progress"},
            ),
        )
        state = CardState(
            blocks=(task_list,),
            metadata=CardMetadata(
                mode_name="Deep",
                engine_type="deep",
                unit_kind="subagent" if is_subagent else None,
                unit_label="完成独立最终代码审查" if is_subagent else None,
                card_sequence="1.e" if is_subagent else 1,
                is_subagent=is_subagent,
                parent_card_seq="1" if is_subagent else None,
            ),
        )

        body = render_card(state, RenderBudget())[0]._card_json["body"]["elements"]
        task_panels = [
            element
            for element in body
            if element.get("tag") == "collapsible_panel"
            and "最终代码审查" in str(element)
        ]

        assert len(task_panels) == 1
        panel = task_panels[0]
        assert panel["expanded"] is expanded
        assert panel["border"]["color"] == border_color
        if is_subagent:
            header_content = panel["header"]["title"]["content"]
            assert "整体 4/5" in header_content
            assert "当前 5/5" in header_content
            assert "✅" not in header_content

    def test_subtask_compact_task_names_stay_within_card_budget(self):
        long_name = "极长任务名称" * 5000
        state = CardState(
            blocks=(
                TaskListBlock(
                    block_id="tasks",
                    current_task_id="t1",
                    tasks=(
                        {
                            "task_id": "t1",
                            "name": long_name,
                            "status": "in_progress",
                        },
                    ),
                ),
            ),
            metadata=CardMetadata(
                engine_type="deep",
                unit_kind="subagent",
                unit_label="审查任务",
                card_sequence="1.a",
                is_subagent=True,
                parent_card_seq="1",
            ),
        )
        budget = RenderBudget()

        card_json = render_card(state, budget)[0]._card_json
        payload_size = len(
            json.dumps(card_json, ensure_ascii=False).encode("utf-8")
        )

        assert payload_size <= budget.byte_budget
        assert long_name not in str(card_json)
        assert "…" in str(card_json)

    @pytest.mark.parametrize("terminal", ["completed", "failed"])
    def test_terminal_subtask_omits_running_phase_banner(self, terminal):
        state = CardState(
            terminal=terminal,
            metadata=CardMetadata(
                engine_type="deep",
                unit_kind="subagent",
                unit_label="审查任务",
                card_sequence="1.a",
                is_subagent=True,
                parent_card_seq="1",
            ),
        )

        body = render_card(
            state,
            RenderBudget(),
        )[0]._card_json["body"]["elements"]

        assert not any(
            "Deep · 执行中" in str(element)
            for element in body
        )

    def test_subtask_failure_details_are_collapsed_by_default(self):
        raw_error = (
            '{"formatted_output":"FAILED tests/test_card_renderer.py::test_error_panel\\n'
            + "AssertionError: expected collapsible_panel\\n"
            + ("Traceback line\\n" * 400)
            + '"}'
        )
        state = reduce_card_state(
            CardState(
                metadata=CardMetadata(
                    engine_type="deep",
                    unit_kind="subagent",
                    unit_label="验证卡片渲染",
                    card_sequence="1.a",
                    is_subagent=True,
                    parent_card_seq="1",
                ),
            ),
            CardEvent.failed(raw_error),
        )

        cards = render_card(state, RenderBudget())
        error_panels = [
            element
            for card in cards
            for element in card._card_json["body"]["elements"]
            if element.get("tag") == "collapsible_panel"
            and "错误详情" in str(element.get("header", {}))
        ]

        assert error_panels
        assert all(panel["expanded"] is False for panel in error_panels)
        assert all(
            "测试失败：tests/test_card_renderer.py::test_error_panel"
            in panel["header"]["title"]["content"]
            for panel in error_panels
        )
        assert all(
            "formatted_output" not in panel["header"]["title"]["content"]
            for panel in error_panels
        )
        assert raw_error in "".join(
            str(child.get("content", ""))
            for panel in error_panels
            for child in panel.get("elements", [])
        )
        assert not any(
            raw_error in str(element)
            for card in cards
            for element in card._card_json["body"]["elements"]
            if element.get("tag") != "collapsible_panel"
        )


class TestUnifiedCardSections:
    def test_header_includes_execution_unit_label(self):
        from src.card.state.reducers._shared import build_header

        metadata = CardMetadata(
            engine_type="deep",
            mode_name="Deep · Coco",
            mode_emoji="🚀",
            unit_label="第 2 轮",
        )

        header = build_header(metadata, "running")

        assert "第 2 轮" in header.title

    def test_render_card_orders_status_body_and_appendix_sections(self):
        state = CardState(
            blocks=(
                ContentBlock(
                    kind="tool_call",
                    block_id="tool1",
                    tool_name="Read",
                    tool_input='{"path": "/src/main.py"}',
                    tool_output="read ok",
                    tool_summary="read ok",
                    status="completed",
                ),
                ContentBlock(kind="text", block_id="body1", content="正文内容"),
                ContentBlock(kind="phase", block_id="phase1", content="第 1 轮 · Build"),
            ),
            metadata=CardMetadata(engine_type="spec", mode_name="Spec · Coco", mode_emoji="📋"),
        )

        cards = render_card(state, RenderBudget())
        body = cards[0]._card_json["body"]["elements"]

        phase_idx = next(i for i, el in enumerate(body) if "第 1 轮 · Build" in str(el))
        text_idx = next(i for i, el in enumerate(body) if el.get("content") == "正文内容")
        # Completed tools now render as activity_digest (one-line summary) in body
        digest_idx = next(i for i, el in enumerate(body) if "已探索" in str(el))

        # Status (phase) comes first, then body atoms in original order (digest, text)
        assert phase_idx < digest_idx < text_idx

    def test_review_role_blocks_render_as_separate_collapsible_panels(self):
        state = CardState(
            blocks=(
                ContentBlock(
                    kind="review_role",
                    block_id="review_1_tester",
                    data={
                        "cycle_num": 1,
                        "title": "测试工程师",
                        "emoji": "🧪",
                        "status_text": "❌ 有建议",
                        "passed": False,
                        "suggestions": ["补充 schema 回归", "覆盖分页边界"],
                        "summary": "测试覆盖不足",
                        "agent_detail": "Codex / gpt-5.5",
                        "background_style": "wathet",
                        "border_color": "wathet",
                    },
                ),
                ContentBlock(
                    kind="review_role",
                    block_id="review_1_designer",
                    data={
                        "cycle_num": 1,
                        "title": "体验设计师",
                        "emoji": "🎨",
                        "status_text": "✅ PASS",
                        "passed": True,
                        "suggestions": [],
                        "summary": "",
                        "agent_detail": "",
                        "background_style": "green",
                        "border_color": "green",
                    },
                ),
            ),
            metadata=CardMetadata(engine_type="spec", mode_name="Spec", mode_emoji="📋"),
        )

        cards = render_card(state, RenderBudget())
        panels = [
            el for el in cards[0]._card_json["body"]["elements"]
            if el.get("tag") == "collapsible_panel" and "测试工程师" in str(el)
        ]

        assert len(panels) == 1
        assert panels[0]["expanded"] is True
        assert "补充 schema 回归" in str(panels[0])
        assert "Codex / gpt-5.5" in str(panels[0])
        assert any(
            el.get("tag") == "collapsible_panel" and "体验设计师" in str(el)
            for el in cards[0]._card_json["body"]["elements"]
        )

    def test_spec_plan_and_tasks_render_as_structured_panels_without_task_truncation(self):
        task_1 = "梳理 Spec phase_done 的结构化产物来源，并在同一卡片中展示方案规划，保留完整描述"
        task_2 = "新增 reducer，把任务分解转成一项任务一个 block，避免多个任务挤进同一段文本被截断"
        task_3 = "新增渲染器，把任务 3 的完整说明展示出来，确保后续模型提到任务 3 时用户能对应"
        state = CardState(
            blocks=(
                ContentBlock(
                    kind="spec_plan",
                    block_id="spec_plan_1",
                    data={
                        "cycle_num": 1,
                        "architecture": "复用 CardSession 结构化事件，只展示整理后的方案规划，不恢复 raw JSON。",
                        "tech_stack": ["CardEvent", "CardState", "Feishu Schema 2.0"],
                        "steps": ["解析 PLAN 产物", "派发结构化事件", "渲染方案规划面板"],
                        "file_changes": ["src/card/events/types.py", "src/card/render/spec_artifacts.py"],
                        "test_plan": ["事件、reducer、renderer、Spec callback 回归"],
                        "risks": [],
                    },
                ),
                ContentBlock(kind="spec_task", block_id="spec_task_1_1", data={"cycle_num": 1, "task_id": 1, "description": task_1, "dependencies": []}),
                ContentBlock(kind="spec_task", block_id="spec_task_1_2", data={"cycle_num": 1, "task_id": 2, "description": task_2, "dependencies": []}),
                ContentBlock(kind="spec_task", block_id="spec_task_1_3", data={"cycle_num": 1, "task_id": 3, "description": task_3, "dependencies": [1, 2]}),
            ),
            metadata=CardMetadata(engine_type="spec", mode_name="Spec", mode_emoji="📋"),
        )

        cards = render_card(state, RenderBudget())
        body = cards[0]._card_json["body"]["elements"]
        def _header_content(panel):
            return panel.get("header", {}).get("title", {}).get("content", "")

        plan_panels = [
            el for el in body
            if el.get("tag") == "collapsible_panel" and "🏗️ **方案规划**" in _header_content(el)
        ]
        task_panels = [
            el for el in body
            if el.get("tag") == "collapsible_panel" and _header_content(el).startswith("📝 **任务 ")
        ]
        task_list_panels = [
            el for el in body
            if el.get("tag") == "collapsible_panel" and _header_content(el).startswith("📝 **任务列表-")
        ]

        assert len(plan_panels) == 1
        assert "解析 PLAN 产物" in str(plan_panels[0])
        plan_body = plan_panels[0]["elements"][0]
        assert plan_body["background_style"] == "orange"
        assert plan_panels[0]["border"]["color"] == "orange"
        assert task_panels == []
        assert len(task_list_panels) == 1
        assert "任务列表-3" in _header_content(task_list_panels[0])
        task_list_body = task_list_panels[0]["elements"][0]["columns"][0]["elements"][0]["content"]
        assert f"1. {task_1} · 依赖：无 · 状态：待执行" in task_list_body
        assert f"3. {task_3} · 依赖：任务 1、任务 2 · 状态：待执行" in task_list_body
        assert "及其他" not in task_list_body
        assert "已截断" not in task_list_body

    def test_spec_task_list_panel_keeps_later_tasks_without_individual_cards(self):
        blocks = tuple(
            ContentBlock(
                kind="spec_task",
                block_id=f"spec_task_1_{idx}",
                data={
                    "cycle_num": 1,
                    "task_id": idx,
                    "description": f"任务 {idx} 的完整说明必须保留在卡片分页中，不能因为任务很多就被合并截断",
                    "dependencies": [idx - 1] if idx > 1 else [],
                },
            )
            for idx in range(1, 35)
        )
        state = CardState(
            blocks=blocks,
            metadata=CardMetadata(engine_type="spec", mode_name="Spec", mode_emoji="📋"),
        )

        cards = render_card(state, RenderBudget(byte_budget=6000, node_budget=42))
        all_elements = [el for card in cards for el in card._card_json["body"]["elements"]]
        individual_task_panels = [
            el for el in all_elements
            if el.get("tag") == "collapsible_panel"
            and el.get("header", {}).get("title", {}).get("content", "").startswith("📝 **任务 ")
        ]
        task_list_panels = [
            el for el in all_elements
            if el.get("tag") == "collapsible_panel"
            and el.get("header", {}).get("title", {}).get("content", "").startswith("📝 **任务列表-")
        ]

        assert individual_task_panels == []
        assert len(task_list_panels) == 1
        assert "任务列表-34" in str(task_list_panels[0])
        assert "34. 任务 34 的完整说明必须保留在卡片分页中" in str(task_list_panels[0])

    def test_bridge_phrase_is_prepended_to_first_text_body_atom(self):
        state = CardState(
            blocks=(ContentBlock(kind="text", block_id="t1", content="继续执行正文"),),
            metadata=CardMetadata(bridge_phrase="续接上一张卡片："),
        )

        cards = render_card(state, RenderBudget())
        body = cards[0]._card_json["body"]["elements"]

        assert any(
            el.get("content") == "续接上一张卡片：\n\n继续执行正文"
            for el in body
            if el.get("tag") == "markdown"
        )

    def test_bridge_phrase_prepended_to_execution_current_action(self):
        """Bridge phrase prepends into the live execution action."""
        state = CardState(
            blocks=(ContentBlock(kind="reasoning", block_id="r1", content="分析中...", status="active"),),
            metadata=CardMetadata(bridge_phrase="续接："),
        )

        cards = render_card(state, RenderBudget())
        body = cards[0]._card_json["body"]["elements"]

        current_rows = [
            el for el in body
            if el.get("tag") == "column_set" and "正在分析" in str(el)
        ]
        assert len(current_rows) == 1
        markdown = current_rows[0]["columns"][0]["elements"][0]
        assert markdown["text_align"] == "left"
        md_content = markdown["content"]
        assert md_content.startswith("续接：")

    def test_programming_card_renders_tool_in_execution_history(self):
        """Completed tools render in the chronological execution record."""
        state = CardState(
            blocks=(
                ContentBlock(
                    kind="tool_call",
                    block_id="tool1",
                    tool_name="read",
                    tool_input='{"path": "src/app.py"}',
                    tool_output="FULL_FILE_CONTENT_SHOULD_NOT_RENDER",
                    tool_summary="src/app.py",
                    status="completed",
                ),
                ContentBlock(kind="text", block_id="body1", content="正文内容"),
            ),
            metadata=CardMetadata(tool_name="Coco"),
        )

        cards = render_card(state, RenderBudget())
        body = cards[0]._card_json["body"]["elements"]

        assert "执行记录" in str(body)
        assert "read · src/app.py" in str(body)
        assert "正文内容" in str(body)
        assert "FULL_FILE_CONTENT_SHOULD_NOT_RENDER" not in str(body)


class TestStreamingMode:
    """streaming_mode in config."""

    def test_streaming_enabled_when_active_text_and_running(self):
        state = CardState(
            blocks=(
                ContentBlock(
                    kind="text",
                    block_id="t1",
                    content="typing...",
                    element_id="el_1",
                    status="active",
                ),
            ),
            terminal="running",
        )
        cards = render_card(state, RenderBudget())
        assert cards[0]._card_json["config"].get("streaming_mode") is True

    def test_rich_running_card_uses_official_cardkit_element_streaming(self):
        state = CardState(
            blocks=(
                ContentBlock(
                    kind="task_list",
                    tasks=(
                        {"task_id": "1", "name": "检查模块", "status": "in_progress"},
                    ),
                    current_task_id="1",
                ),
                ContentBlock(
                    kind="text",
                    block_id="t1",
                    content="分析这是一个代码质量审查任务，需要完整展示正文。",
                    element_id="el_1",
                    status="active",
                ),
            ),
            terminal="running",
        )

        cards = render_card(state, RenderBudget(engine_cmd="/deep"))
        card_json = cards[0]._card_json

        assert card_json["config"].get("streaming_mode") is True
        assert cards[0].active_element is not None
        assert cards[0].active_element.element_id == "el_1"
        assert cards[0].active_element.text == "分析这是一个代码质量审查任务，需要完整展示正文。"
        markdown_elements = [
            node
            for node in _iter_dict_nodes(card_json["body"]["elements"])
            if node.get("tag") == "markdown"
        ]
        assert any(node.get("element_id") == "el_1" for node in markdown_elements)

    def test_rich_running_streaming_text_changes_do_not_patch_full_card(self):
        prefix = "分析" * 40
        base_blocks = (
            ContentBlock(
                kind="task_list",
                tasks=(
                    {"task_id": "1", "name": "检查模块", "status": "in_progress"},
                ),
                current_task_id="1",
            ),
        )
        state_a = CardState(
            blocks=base_blocks
            + (
                ContentBlock(
                    kind="text",
                    block_id="t1",
                    content=prefix + "A",
                    element_id="el_1",
                    status="active",
                ),
            ),
            terminal="running",
        )
        state_b = CardState(
            blocks=base_blocks
            + (
                ContentBlock(
                    kind="text",
                    block_id="t1",
                    content=prefix + "B",
                    element_id="el_1",
                    status="active",
                ),
            ),
            terminal="running",
        )

        card_a = render_card(state_a, RenderBudget(engine_cmd="/deep"))[0]
        card_b = render_card(state_b, RenderBudget(engine_cmd="/deep"))[0]

        assert card_a.active_element is not None
        assert card_b.active_element is not None
        assert card_a.active_element.text.endswith("A")
        assert card_b.active_element.text.endswith("B")
        assert card_a.structure_signature == card_b.structure_signature

    def test_active_element_id_change_updates_structure_signature(self):
        """Switching active stream targets must PATCH the card before element updates."""
        state_a = CardState(
            blocks=(
                ContentBlock(
                    kind="text",
                    block_id="t1",
                    content="第一段",
                    element_id="el_1",
                    status="active",
                ),
            ),
            terminal="running",
        )
        state_b = CardState(
            blocks=(
                ContentBlock(
                    kind="text",
                    block_id="t1",
                    content="第一段",
                    element_id="el_2",
                    status="active",
                ),
            ),
            terminal="running",
        )

        card_a = render_card(state_a, RenderBudget(engine_cmd="/deep"))[0]
        card_b = render_card(state_b, RenderBudget(engine_cmd="/deep"))[0]

        assert card_a.active_element.element_id == "el_1"
        assert card_b.active_element.element_id == "el_2"
        assert card_a.structure_signature != card_b.structure_signature

    def test_streaming_disabled_when_completed(self):
        state = CardState(
            blocks=(
                ContentBlock(
                    kind="text",
                    block_id="t1",
                    content="done",
                    element_id="el_1",
                    status="active",
                ),
            ),
            terminal="completed",
        )
        cards = render_card(state, RenderBudget())
        assert "streaming_mode" not in cards[0]._card_json["config"]


class TestSchemaDivStyleSafety:
    """Regression tests: avoid illegal style fields on `div` (Feishu Schema 2.0)."""

    def test_warning_banner_does_not_style_div(self):
        state = CardState(
            blocks=(ContentBlock(kind="text", block_id="t1", content="hello"),),
            footer=FooterState(warning_banner="卡片解析应成功", warning_type="warning"),
        )
        cards = render_card(state, RenderBudget())
        card_json = cards[0]._card_json
        for node in _iter_dict_nodes(card_json):
            if node.get("tag") == "div":
                assert "padding" not in node
                assert "background_style" not in node

    def test_assemble_card_json_strips_collapsible_panel_background_style(self):
        body_elements = [
            {
                "tag": "column_set",
                "columns": [
                    {
                        "tag": "column",
                        "elements": [
                            {
                                "tag": "collapsible_panel",
                                "background_style": "default",
                                "header": {"title": {"tag": "markdown", "content": "执行单元"}},
                                "elements": [{"tag": "markdown", "content": "body"}],
                            }
                        ],
                    }
                ],
            }
        ]

        card_json = _assemble_card_json(
            CardState(),
            body_elements=body_elements,
            streaming=False,
            active_element=None,
        )
        panels = [node for node in _iter_dict_nodes(card_json) if node.get("tag") == "collapsible_panel"]

        assert panels
        assert all("background_style" not in panel for panel in panels)

    def test_streaming_disabled_when_no_active_element(self):
        state = CardState(
            blocks=(
                ContentBlock(
                    kind="text",
                    block_id="t1",
                    content="done text",
                    status="completed",
                ),
            ),
            terminal="running",
        )
        cards = render_card(state, RenderBudget())
        assert "streaming_mode" not in cards[0]._card_json["config"]


class TestActiveElement:
    """ActiveElement detection."""

    def test_active_text_detected(self):
        state = CardState(
            blocks=(
                ContentBlock(
                    kind="text",
                    block_id="t1",
                    content="streaming text",
                    element_id="el_stream",
                    status="active",
                ),
            ),
        )
        cards = render_card(state, RenderBudget())
        assert cards[0].active_element is not None
        assert cards[0].active_element.element_id == "el_stream"
        assert cards[0].active_element.text == "streaming text"

    def test_escaped_dirty_code_fence_is_normalized_for_markdown(self):
        content = (
            "Summary\n\n"
            "\\```ProgressCardAccumulator` class)python\n"
            "from __future__ import annotations\n"
            "# ---------------------------------------------------------------------------\n"
            "# Structured event type\n"
            "# ---------------------------------------------------------------------------\n"
            "@dataclass\n"
            "class ToolProgressEvent:\n"
            "    pass\n"
            "\\```"
        )
        state = CardState(
            blocks=(
                ContentBlock(
                    kind="text",
                    block_id="t1",
                    content=content,
                    element_id="el_stream",
                    status="active",
                ),
            ),
            terminal="running",
        )

        cards = render_card(state, RenderBudget())
        markdown = [
            el
            for el in cards[0]._card_json["body"]["elements"]
            if el.get("tag") == "markdown"
        ][0]

        assert "\\```" not in markdown["content"]
        assert "```ProgressCardAccumulator` class)python" not in markdown["content"]
        assert "```python\nfrom __future__ import annotations" in markdown["content"]
        assert markdown["content"].rstrip().endswith("```")
        assert cards[0].active_element is not None
        assert cards[0].active_element.text == markdown["content"]

    def test_active_markdown_closes_unfinished_inline_code_span(self):
        content = (
            "### 🔴 Critical\n\n"
            "**Issue:** `randH ex discards the screenshot error from crypto/rand.Read"
        )
        state = CardState(
            blocks=(
                ContentBlock(
                    kind="text",
                    block_id="t1",
                    content=content,
                    element_id="el_stream",
                    status="active",
                ),
            ),
            terminal="running",
        )

        cards = render_card(state, RenderBudget())
        markdown = [
            el
            for el in cards[0]._card_json["body"]["elements"]
            if el.get("tag") == "markdown"
        ][0]

        assert markdown["content"] == content + "`"
        assert cards[0].active_element is not None
        assert cards[0].active_element.text == content + "`"

    def test_completed_markdown_preserves_unfinished_inline_code_span(self):
        content = "**Issue:** `raw model output"
        state = CardState(
            blocks=(
                ContentBlock(
                    kind="text",
                    block_id="t1",
                    content=content,
                    element_id="el_stream",
                    status="completed",
                ),
            ),
            terminal="completed",
        )

        cards = render_card(state, RenderBudget())
        markdown = [
            el
            for el in cards[0]._card_json["body"]["elements"]
            if el.get("tag") == "markdown"
        ][0]

        assert markdown["content"] == content
        assert cards[0].active_element is None

    def test_active_markdown_closes_unfinished_code_fence(self):
        content = "Reviewing file\n\n```go\nfunc main() {\n"
        state = CardState(
            blocks=(
                ContentBlock(
                    kind="text",
                    block_id="t1",
                    content=content,
                    element_id="el_stream",
                    status="active",
                ),
            ),
            terminal="running",
        )

        cards = render_card(state, RenderBudget())
        markdown = [
            el
            for el in cards[0]._card_json["body"]["elements"]
            if el.get("tag") == "markdown"
        ][0]

        assert markdown["content"] == content + "```"
        assert cards[0].active_element is not None
        assert cards[0].active_element.text == content + "```"

    def test_one_character_active_text_waits_before_streaming(self):
        state = CardState(
            blocks=(
                ContentBlock(
                    kind="text",
                    block_id="t1",
                    content="数",
                    element_id="el_stream",
                    status="active",
                ),
            ),
            terminal="running",
        )

        cards = render_card(state, RenderBudget())

        assert cards[0].active_element is None
        assert "streaming_mode" not in cards[0]._card_json["config"]

    def test_no_active_element_when_completed(self):
        state = CardState(
            blocks=(
                ContentBlock(
                    kind="text",
                    block_id="t1",
                    content="final",
                    element_id="el_1",
                    status="completed",
                ),
            ),
        )
        cards = render_card(state, RenderBudget())
        assert cards[0].active_element is None



class TestColumnSetSignature:
    """Reasoning panel content changes should affect page signature."""

    def test_column_set_content_change_updates_signature(self):
        """Reasoning panel with different content should produce different signatures."""
        s1 = CardState(
            blocks=(ContentBlock(kind="reasoning", block_id="r1", content="thought A", status="active"),),
            terminal="running",
        )
        s2 = CardState(
            blocks=(ContentBlock(kind="reasoning", block_id="r1", content="thought B", status="active"),),
            terminal="running",
        )
        cards1 = render_card(s1, RenderBudget())
        cards2 = render_card(s2, RenderBudget())
        assert cards1[0].structure_signature != cards2[0].structure_signature


class TestStructureSignature:
    """compute_structure_signature tests."""

    def test_same_structure_same_signature(self):
        s1 = CardState(
            blocks=(ContentBlock(kind="text", block_id="t1", content="hello"),),
            terminal="running",
        )
        s2 = CardState(
            blocks=(ContentBlock(kind="text", block_id="t1", content="world"),),
            terminal="running",
        )
        # Content differs but structure is the same
        assert compute_structure_signature(s1) == compute_structure_signature(s2)

    def test_different_structure_different_signature(self):
        s1 = CardState(
            blocks=(ContentBlock(kind="text", block_id="t1", content="hello"),),
            terminal="running",
        )
        s2 = CardState(
            blocks=(
                ContentBlock(kind="text", block_id="t1", content="hello"),
                ContentBlock(kind="tool_call", block_id="tc1", tool_name="bash"),
            ),
            terminal="running",
        )
        assert compute_structure_signature(s1) != compute_structure_signature(s2)



class TestFooterAndButtons:
    """Footer and buttons only appear on last page."""

    def test_footer_on_single_page(self):
        state = CardState(
            blocks=(ContentBlock(kind="text", block_id="t1", content="text"),),
            footer=FooterState(status="thinking", status_text="🤔 思考中..."),
        )
        cards = render_card(state, RenderBudget())
        body = cards[0]._card_json["body"]["elements"]
        # Should have hr separator from footer
        assert any(el.get("tag") == "hr" for el in body)

    def test_buttons_on_single_page(self):
        state = CardState(
            blocks=(ContentBlock(kind="text", block_id="t1", content="text"),),
            buttons=(ButtonSpec(text="Stop", action_id="stop", type="danger"),),
        )
        cards = render_card(state, RenderBudget())
        body = cards[0]._card_json["body"]["elements"]
        # Single button renders as column_set with flex_mode 'none' (full width)
        assert any(el.get("tag") == "column_set" and el.get("flex_mode") == "none" for el in body)


class TestMultipleBlockTypes:
    """Rendering mixed block types."""

    def test_tool_block_renders_collapsible_panel(self):
        """Completed tool renders in a collapsed execution-history panel."""
        state = CardState(
            blocks=(
                ContentBlock(
                    kind="tool_call",
                    block_id="tc1",
                    tool_name="bash",
                    tool_input="ls -la",
                    status="completed",
                ),
            ),
        )
        cards = render_card(state, RenderBudget())
        body = cards[0]._card_json["body"]["elements"]
        assert any(el.get("tag") == "collapsible_panel" and "执行记录" in str(el) for el in body)

    def test_reasoning_block_renders(self):
        state = CardState(
            blocks=(
                ContentBlock(
                    kind="reasoning",
                    block_id="r1",
                    content="thinking about...",
                    status="active",
                ),
            ),
        )
        cards = render_card(state, RenderBudget())
        body = cards[0]._card_json["body"]["elements"]
        assert any(el.get("tag") == "column_set" and "正在分析" in str(el) for el in body)

    def test_spec_reasoning_full_mode_keeps_complete_text(self):
        long_content = "完整思考内容" * 120
        state = CardState(
            blocks=(
                ContentBlock(
                    kind="reasoning",
                    block_id="spec_reasoning",
                    content=long_content,
                    status="completed",
                    char_count=len(long_content),
                ),
            ),
            metadata=CardMetadata(engine_type="spec", mode_name="Spec", mode_emoji="📋", compact=False),
        )

        cards = render_card(state, RenderBudget(reasoning_tail_chars=200))
        body = str(cards[0]._card_json["body"]["elements"])

        assert long_content in body
        assert "…完整思考内容" not in body

    def test_spec_reasoning_compact_mode_truncates_to_222_chars(self):
        long_content = "a" * 260
        state = CardState(
            blocks=(
                ContentBlock(
                    kind="reasoning",
                    block_id="spec_reasoning",
                    content=long_content,
                    status="completed",
                    char_count=len(long_content),
                ),
            ),
            metadata=CardMetadata(engine_type="spec", mode_name="Spec", mode_emoji="📋", compact=True),
        )

        cards = render_card(state, RenderBudget())
        panels = [
            el for el in cards[0]._card_json["body"]["elements"]
            if el.get("tag") == "collapsible_panel" and "过程摘要" in str(el.get("header", {}))
        ]

        assert len(panels) == 1
        body = panels[0]["elements"][0]["content"]
        assert len(body) == 222
        assert body == "- " + ("a" * 219) + "…"

    def test_reasoning_blocks_paginate_under_feishu_node_limit(self):
        """Reasoning node estimates should keep rendered pages below the hard Feishu cap."""
        state = CardState(
            blocks=tuple(
                ContentBlock(
                    kind="reasoning",
                    block_id=f"r{idx}",
                    content=f"thinking about item {idx}",
                    status="completed",
                )
                for idx in range(80)
            ),
            metadata=CardMetadata(mode_name="Spec", engine_type="spec"),
        )

        cards = render_card(state, RenderBudget(node_budget=60))

        assert len(cards) > 1
        for card in cards:
            node_count = sum(1 for node in _iter_dict_nodes(card._card_json) if "tag" in node)
            assert node_count <= 200

    def test_plan_block_renders(self):
        state = CardState(
            blocks=(
                ContentBlock(
                    kind="plan",
                    block_id="p1",
                    content="✅ Step 1\n⏳ Step 2",
                    status="active",
                ),
            ),
        )
        cards = render_card(state, RenderBudget())
        body = cards[0]._card_json["body"]["elements"]
        panels = [el for el in body if el.get("tag") == "collapsible_panel"]
        assert len(panels) == 1

    def test_mixed_blocks_render_in_order(self):
        """Running programming cards put execution history before narration."""
        state = CardState(
            blocks=(
                ContentBlock(kind="text", block_id="t1", content="Intro"),
                ContentBlock(
                    kind="tool_call",
                    block_id="tc1",
                    tool_name="bash",
                    tool_input="echo hi",
                    status="completed",
                ),
                ContentBlock(kind="text", block_id="t2", content="Conclusion"),
            ),
        )
        cards = render_card(state, RenderBudget())
        body = cards[0]._card_json["body"]["elements"]
        intro_idx = next(i for i, el in enumerate(body) if "Intro" in str(el))
        conclusion_idx = next(i for i, el in enumerate(body) if "Conclusion" in str(el))
        history_idx = next(
            i for i, el in enumerate(body)
            if el.get("tag") == "collapsible_panel" and "执行记录" in str(el)
        )
        assert history_idx < intro_idx < conclusion_idx


class TestPagination:
    """Multi-page rendering."""

    def test_large_content_creates_multiple_pages(self):
        # Create content that exceeds budget
        big_text = "x" * 30000  # ~90KB estimated, way over 27KB budget
        state = CardState(
            blocks=(ContentBlock(kind="text", block_id="t1", content=big_text),),
        )
        budget = RenderBudget(byte_budget=5000)  # Very small budget
        cards = render_card(state, budget)
        assert len(cards) > 1
        # Check page indexing
        for i, card in enumerate(cards):
            assert card.page_index == i
            assert card.total_pages == len(cards)

    def test_all_pages_have_header(self):
        big_text = "line\n" * 5000
        state = CardState(
            blocks=(ContentBlock(kind="text", block_id="t1", content=big_text),),
            header=HeaderState(title="Multi-page", template="blue"),
        )
        budget = RenderBudget(byte_budget=5000)
        cards = render_card(state, budget)
        for card in cards:
            assert card._card_json["header"]["title"]["content"] == (
                f"Multi-page · 页 {card.page_index + 1}/{card.total_pages}"
            )

    def test_split_code_fence_pages_are_independently_parseable(self):
        big_code = "```go\n" + ("fmt.Println(\"hello\")\n" * 700) + "```\nDone"
        state = CardState(
            blocks=(
                ContentBlock(
                    kind="text",
                    block_id="t1",
                    content=big_code,
                    element_id="el_stream",
                    status="active",
                ),
            ),
            terminal="running",
        )

        cards = render_card(state, RenderBudget(byte_budget=5000))

        assert len(cards) > 1
        for card in cards:
            markdown = [
                el.get("content", "")
                for el in card._card_json["body"]["elements"]
                if el.get("tag") == "markdown" and "fmt.Println" in el.get("content", "")
            ]
            assert markdown
            assert all(text.count("```") % 2 == 0 for text in markdown)

    def test_section_layout_repeats_sticky_phase_banner_on_every_page(self):
        big_text = "line\n" * 5000
        state = CardState(
            blocks=(ContentBlock(kind="text", block_id="t1", content=big_text),),
            metadata=CardMetadata(mode_name="Deep", mode_emoji="🧠", engine_type="deep"),
        )
        budget = RenderBudget(byte_budget=5000)

        cards = render_card(state, budget)

        assert len(cards) > 1
        for card in cards:
            body = card._card_json["body"]["elements"]
            assert body[0]["tag"] == "markdown"
            assert "Deep" in body[0]["content"]

    def test_section_layout_keeps_appendix_on_last_page_only(self):
        """Completed tools render as activity_digest in body (not appendix).

        With the slim-flow redesign, completed tools are aggregated into a
        one-line activity_digest atom placed in body alongside text, rather
        than as collapsible panels in the appendix section.
        """
        big_text = "line\n" * 5000
        state = CardState(
            blocks=(
                ContentBlock(
                    kind="tool_call",
                    block_id="tool1",
                    tool_name="Bash",
                    tool_output="done",
                    status="completed",
                ),
                ContentBlock(kind="text", block_id="t1", content=big_text),
            ),
            metadata=CardMetadata(mode_name="Deep", mode_emoji="🧠", engine_type="deep"),
        )
        budget = RenderBudget(byte_budget=5000)

        cards = render_card(state, budget)

        assert len(cards) > 1
        # activity_digest is small and sits in body on the first page
        first_body = cards[0]._card_json["body"]["elements"]
        assert "已运行" in str(first_body), "activity_digest should appear in first page body"

    def test_section_layout_renders_sticky_task_list_once_per_page(self):
        big_text = "line\n" * 5000
        state = CardState(
            blocks=(
                ContentBlock(
                    kind="task_list",
                    block_id="tasks",
                    current_task_id="t2",
                    tasks=(
                        {"task_id": "t1", "name": "完成需求", "status": "completed"},
                        {"task_id": "t2", "name": "实现渲染", "status": "in_progress"},
                    ),
                ),
                ContentBlock(kind="text", block_id="t1", content=big_text),
            ),
            metadata=CardMetadata(mode_name="Deep", mode_emoji="🧠", engine_type="deep"),
        )
        budget = RenderBudget(byte_budget=5000)

        cards = render_card(state, budget)

        assert len(cards) > 1
        for card in cards:
            body = card._card_json["body"]["elements"]
            task_panels = [el for el in body if "任务列表" in str(el)]
            assert len(task_panels) == 1
            assert "实现渲染" in str(task_panels[0])
            assert "完成需求" in str(task_panels[0])
            assert "进行中 (1)" in str(task_panels[0])
            assert "已完成 (1)" in str(task_panels[0])


class TestApprovalRendering:
    """Approval state rendering: header color, buttons, footer text."""

    def _make_approval_state(self, tool_name: str = "bash") -> CardState:
        """Build a state with APPROVAL_REQUESTED applied via the reducer."""
        from src.card.events import CardEvent, CardEventType
        from src.card.state.reducer import reduce_card_state

        meta = CardMetadata(
            project_name="Ghost", mode_name="Deep Agent", mode_emoji="🧠",
            tool_name="coco", model_name="gpt-4o", engine_type="deep",
        )
        s = reduce_card_state(None, CardEvent.started(), metadata=meta)
        s = reduce_card_state(s, CardEvent(
            type=CardEventType.APPROVAL_REQUESTED,
            payload={"tool_name": tool_name, "description": "rm -rf /tmp/test"},
        ))
        return s

    def test_header_template_is_indigo(self):
        state = self._make_approval_state()
        cards = render_card(state, RenderBudget())
        assert len(cards) >= 1
        assert cards[0]._card_json["header"]["template"] == "indigo"

    def test_buttons_approve_reject_present(self):
        state = self._make_approval_state()
        cards = render_card(state, RenderBudget())
        body = cards[0]._card_json["body"]["elements"]
        # Buttons render as column_set (2 buttons)
        column_sets = [el for el in body if el.get("tag") == "column_set"]
        assert len(column_sets) >= 1, "Should have a column_set for approve/reject buttons"
        # Extract button texts from columns
        buttons = []
        for cs in column_sets:
            for col in cs.get("columns", []):
                for el in col.get("elements", []):
                    if el.get("tag") == "button":
                        buttons.append(el)
        assert len(buttons) == 2
        button_texts = [b["text"]["content"] for b in buttons]
        assert "✅ 批准" in button_texts
        assert "❌ 拒绝" in button_texts
        # Check button types
        button_types = {b["text"]["content"]: b["type"] for b in buttons}
        assert button_types["✅ 批准"] == "primary"
        assert button_types["❌ 拒绝"] == "danger"

    def test_footer_status_text_contains_tool_name(self):
        state = self._make_approval_state(tool_name="bash")
        cards = render_card(state, RenderBudget())
        body = cards[0]._card_json["body"]["elements"]
        # Footer is: hr + markdown(status_text) [+ optional progress]
        hr_elements = [el for el in body if el.get("tag") == "hr"]
        assert len(hr_elements) >= 1, "Footer should have hr separator"
        # Find footer text (markdown element after hr)
        footer_texts = [
            el["content"] for el in body
            if el.get("tag") == "markdown" and el.get("text_size") == "notation"
        ]
        assert any("等待审批" in t and "bash" in t for t in footer_texts), \
            f"Footer should mention '等待审批' and 'bash', got: {footer_texts}"

    def test_no_streaming_mode_during_approval(self):
        state = self._make_approval_state()
        cards = render_card(state, RenderBudget())
        assert "streaming_mode" not in cards[0]._card_json["config"]
