"""Programming-card execution flow rendering regressions."""

from __future__ import annotations

from src.card.render.budget import RenderBudget
from src.card.render.renderer import render_card
from src.card.state.models import CardMetadata, CardState, ContentBlock, FooterState


def _body(state: CardState) -> list[dict]:
    pages = render_card(state, RenderBudget())
    return pages[0]._card_json["body"]["elements"]


def _execution_elements(body: list[dict]) -> list[dict]:
    return [
        element
        for element in body
        if (
            "执行记录" in str(element)
            or "正在分析" in str(element)
            or "正在调用" in str(element)
            or "执行中" in str(element)
        )
    ]


def test_running_programming_card_shows_current_action_before_folded_history():
    state = CardState(
        blocks=(
            ContentBlock(
                kind="reasoning",
                block_id="r1",
                status="completed",
                content="先检查入口",
            ),
            ContentBlock(
                kind="tool_call",
                block_id="t1",
                status="completed",
                tool_name="Read",
                tool_input='{"path":"src/main.py"}',
                tool_output="sensitive file contents",
            ),
            ContentBlock(
                kind="reasoning",
                block_id="r2",
                status="active",
                content="检查真实事件",
            ),
        ),
        metadata=CardMetadata(mode_name="Codex", tool_name="codex", engine_type=None),
    )

    elements = _execution_elements(_body(state))

    assert [element["tag"] for element in elements] == ["column_set", "collapsible_panel"]
    assert "正在分析" in str(elements[0])
    assert "检查真实事件" in str(elements[0])
    assert "执行记录** · 2 步" in str(elements[1])
    assert elements[1]["expanded"] is False
    history = elements[1]["elements"][0]["content"]
    assert history.splitlines() == [
        "- 🧠 分析 · 先检查入口",
        "- ✅ Read · src/main.py",
    ]
    assert "sensitive file contents" not in str(elements)
    assert "自动总结" not in str(elements)


def test_running_programming_card_shows_active_command_as_current_action():
    state = CardState(
        blocks=(
            ContentBlock(
                kind="tool_call",
                block_id="t1",
                status="active",
                tool_name="Bash",
                tool_input="uv run python -m pytest tests/test_card_renderer.py -q",
                is_latest_active=True,
            ),
        ),
        metadata=CardMetadata(mode_name="Codex", tool_name="codex", engine_type=None),
    )

    elements = _execution_elements(_body(state))

    assert len(elements) == 1
    assert elements[0]["tag"] == "column_set"
    content = elements[0]["columns"][0]["elements"][0]["content"]
    assert content.splitlines() == [
        "⚙️ **运行测试**",
        "🔧 **执行中**：uv run python -m pytest tests/test_card_renderer.py -q",
    ]


def test_raw_command_title_is_shown_once_as_full_command_after_short_action():
    command = (
        "UV_CACHE_DIR=/tmp/ghostap-uv-cache uv run python -m pytest "
        "tests/test_card_execution_flow.py tests/test_footer_v2.py -q "
        "-k 'command or footer'"
    )
    state = CardState(
        blocks=(
            ContentBlock(
                kind="tool_call",
                block_id="t-command",
                status="active",
                tool_name=command,
                tool_input=command,
                is_latest_active=True,
            ),
        ),
        footer=FooterState(
            status="tool_running",
            status_text=f"🔧 执行中: {command}",
        ),
        metadata=CardMetadata(
            mode_name="Codex",
            tool_name="codex",
            engine_type=None,
        ),
    )

    body = _body(state)
    current = next(
        element
        for element in body
        if element.get("tag") == "column_set" and "执行中" in str(element)
    )
    content = current["columns"][0]["elements"][0]["content"]

    assert content.startswith("⚙️ **运行测试**\n🔧 **执行中**：")
    assert str(body).count(command) == 1
    assert "正在调用" not in content


def test_phase_like_tool_title_is_not_duplicated_or_left_in_footer():
    state = CardState(
        blocks=(
            ContentBlock(
                kind="tool_call",
                block_id="t-compact",
                status="active",
                tool_name="Context compacting",
                tool_input="Context compacting",
                is_latest_active=True,
            ),
        ),
        footer=FooterState(
            status="tool_running",
            status_text="🔧 执行中: Context compacting",
        ),
        metadata=CardMetadata(
            mode_name="Codex",
            tool_name="codex",
            engine_type=None,
        ),
    )

    body = _body(state)

    assert str(body).count("Context compacting") == 1
    assert str(body).count("执行中") == 0


def test_execution_history_does_not_mark_an_older_active_tool_completed():
    state = CardState(
        blocks=(
            ContentBlock(
                kind="tool_call",
                block_id="t1",
                status="active",
                tool_name="Read",
                tool_input='{"path":"src/first.py"}',
            ),
            ContentBlock(
                kind="reasoning",
                block_id="r2",
                status="active",
                content="协调并行结果",
            ),
        ),
        metadata=CardMetadata(mode_name="Codex", tool_name="codex", engine_type=None),
    )

    history = next(element for element in _body(state) if "执行记录" in str(element))

    assert "- ⏳ Read · src/first.py" in history["elements"][0]["content"]
    assert "- ✅ Read · src/first.py" not in history["elements"][0]["content"]


def test_completed_programming_card_puts_answer_before_execution_history():
    state = CardState(
        blocks=(
            ContentBlock(
                kind="reasoning",
                block_id="r1",
                status="completed",
                content="检查渲染顺序",
            ),
            ContentBlock(
                kind="tool_call",
                block_id="t1",
                status="completed",
                tool_name="Edit",
                tool_input='{"path":"src/card/render/renderer.py"}',
            ),
            ContentBlock(
                kind="text",
                block_id="answer",
                status="completed",
                content="已完成执行流优化。",
            ),
        ),
        terminal="completed",
        metadata=CardMetadata(mode_name="Codex", tool_name="codex", engine_type=None),
    )

    body = _body(state)
    answer_index = next(index for index, element in enumerate(body) if "已完成执行流优化" in str(element))
    history_index = next(index for index, element in enumerate(body) if "执行记录" in str(element))

    assert answer_index < history_index
    history = body[history_index]
    assert history["tag"] == "collapsible_panel"
    assert history["expanded"] is False
    assert "执行记录** · 2 步" in str(history)
    assert "正在分析" not in str(body)


def test_execution_history_keeps_latest_twelve_steps_in_chronological_order():
    blocks = tuple(
        ContentBlock(
            kind="reasoning",
            block_id=f"r{index}",
            status="completed",
            content=f"步骤 {index}",
        )
        for index in range(15)
    )
    state = CardState(
        blocks=blocks,
        terminal="completed",
        metadata=CardMetadata(mode_name="Codex", tool_name="codex", engine_type=None),
    )

    history = next(element for element in _body(state) if "执行记录" in str(element))
    content = history["elements"][0]["content"]

    assert "较早 3 步已折叠" in content
    assert "步骤 0" not in content
    assert content.index("步骤 3") < content.index("步骤 14")


def test_engine_cards_keep_existing_reasoning_panel_contract():
    state = CardState(
        blocks=(
            ContentBlock(
                kind="reasoning",
                block_id="r1",
                status="completed",
                content="Spec 分析",
            ),
        ),
        terminal="completed",
        metadata=CardMetadata(mode_name="Spec", engine_type="spec"),
    )

    body = _body(state)

    assert any("过程摘要" in str(element) for element in body)
    assert not any("执行记录" in str(element) for element in body)
