import json

import pytest

from src.card.builder import CardBuilder
from src.card.builders.core import CoreBuilder
from src.card.builders.deep import DeepBuilder
from src.card.builders.project import ProjectBuilder
from src.card.builders.system import SystemBuilder
from src.card.models import (
    EngineCardState,
    EngineStatusEntry,
    ModelOptionView,
    ToolOptionView,
)
from src.card.ui_text import UI_TEXT
from src.model_selection import DEFAULT_MODEL_OPTION_VALUE
from src.project.context import ProjectContext


def _collect_buttons(card: dict) -> list[dict]:
    buttons: list[dict] = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("tag") == "button":
                buttons.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(card)
    return buttons


def _collect_button_layout_blocks(card: dict) -> list[list[dict]]:
    """Return button groups by their immediate column_set layout block."""

    blocks: list[list[dict]] = []

    def collect_buttons(node) -> list[dict]:
        found: list[dict] = []
        if isinstance(node, dict):
            if node.get("tag") == "button":
                found.append(node)
            for value in node.values():
                found.extend(collect_buttons(value))
        elif isinstance(node, list):
            for item in node:
                found.extend(collect_buttons(item))
        return found

    def walk(node):
        if isinstance(node, dict):
            if node.get("tag") == "column_set":
                buttons = collect_buttons(node.get("columns", []))
                if buttons:
                    blocks.append(buttons)
                    return
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(card)
    return blocks


def _collect_selects(card: dict) -> list[dict]:
    selects: list[dict] = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("tag") == "select_static":
                selects.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(card)
    return selects


def test_core_builder_banner_element():
    """Verify that CoreBuilder._build_banner_element produces the correct column_set structure."""
    message = "Test Banner Message"

    # Test info banner (default)
    banner = CoreBuilder._build_banner_element(message, type="info")
    assert banner["tag"] == "column_set"
    assert banner["background_style"] == "wathet"  # info 使用浅蓝色
    assert "ℹ️" in banner["columns"][0]["elements"][0]["content"]
    assert message in banner["columns"][0]["elements"][0]["content"]

    # Test success banner
    banner = CoreBuilder._build_banner_element(message, type="success")
    assert banner["background_style"] == "green"  # success 使用绿色
    assert "✅" in banner["columns"][0]["elements"][0]["content"]

    # Test warning banner - Apple 风格优化：使用橙色代替黄色
    banner = CoreBuilder._build_banner_element(message, type="warning")
    assert banner["background_style"] == "orange"  # warning 使用橙色（更温和、更现代）
    assert "⚠️" in banner["columns"][0]["elements"][0]["content"]

    # Test error banner
    banner = CoreBuilder._build_banner_element(message, type="error")
    assert banner["background_style"] == "red"  # error 使用红色
    assert "❌" in banner["columns"][0]["elements"][0]["content"]


def test_project_builder_with_banner():
    """Verify that ProjectBuilder.build_project_response_card includes the banner."""
    project = ProjectContext(project_id="p1", project_name="P1", root_path="/tmp/p1")
    message = "Successfully added Claude"
    banner = CoreBuilder._build_banner_element(message, type="success")

    msg_type, card_json = ProjectBuilder.build_project_response_card(
        project, "Title", "Content", banner=banner
    )

    assert msg_type == "interactive"
    card = json.loads(card_json)

    # Banner should be the third element (Directory, HR, Banner, HR, Content, Buttons)
    elements = card["body"]["elements"]
    assert elements[2]["tag"] == "column_set"
    assert elements[2]["background_style"] == "green"  # success 使用绿色
    assert message in elements[2]["columns"][0]["elements"][0]["content"]


def test_help_card_mentions_new_chat_project_group():
    SystemBuilder._build_help_card_cached.cache_clear()

    msg_type, card_json = SystemBuilder.build_help_card(
        session_idle_timeout=600,
        session_idle_warn_at_remaining=120,
        lock_undo_window_seconds=300,
    )

    assert msg_type == "interactive"
    card_text = json.dumps(json.loads(card_json), ensure_ascii=False)
    assert "/new-chat" in card_text
    assert "项目群" in card_text


def test_help_card_omits_quick_action_buttons():
    """The documentation-only help card should leave quick actions to /menu."""
    SystemBuilder._build_help_card_cached.cache_clear()

    msg_type, card_json = SystemBuilder.build_help_card(
        session_idle_timeout=600,
        session_idle_warn_at_remaining=120,
        lock_undo_window_seconds=300,
    )

    assert msg_type == "interactive"
    card = json.loads(card_json)
    buttons = _collect_buttons(card)
    help_action_buttons = [
        button
        for button in buttons
        if button.get("value", {}).get("action")
        in {
            "enter_deep_prompt",
            "show_worktree_menu",
            "show_acp_menu",
            "show_ttadk_menu",
            "show_status",
            "switch_project",
        }
    ]

    assert help_action_buttons == []
    assert "常用操作" not in json.dumps(card, ensure_ascii=False)


def test_project_status_card_includes_switch_and_group_jump_without_duplicate_path():
    project = ProjectContext(project_id="p1", project_name="P1", root_path="/tmp/p1")
    project.bound_chat_id = "oc_group_123"
    project.bound_chat_name = "P1-dev"

    msg_type, card_json = ProjectBuilder.build_project_status_report_card(project, "/tmp/p1")

    assert msg_type == "interactive"
    card = json.loads(card_json)
    buttons = _collect_buttons(card)
    assert any(
        b["text"]["content"] == UI_TEXT["project_btn_open_group"]
        and b["multi_url"]["url"].endswith("openChatId=oc_group_123")
        for b in buttons
    )
    assert any(
        b["text"]["content"] == UI_TEXT["project_board_btn_switch"]
        and b.get("behaviors", [{}])[0].get("value", {}).get("action") == "switch_project"
        for b in buttons
    )

    markdown_contents = [
        e.get("content", "")
        for e in card["body"]["elements"]
        if isinstance(e, dict) and e.get("tag") == "markdown"
    ]
    assert sum("/tmp/p1" in content for content in markdown_contents) == 1


def test_project_board_includes_group_jump_for_bound_project():
    project = ProjectContext(project_id="p1", project_name="P1", root_path="/tmp/p1")
    project.bound_chat_id = "oc_group_123"
    project.bound_chat_name = "P1-dev"

    msg_type, card_json = ProjectBuilder.build_status_board_card([project], current_project_id=None)

    assert msg_type == "interactive"
    card = json.loads(card_json)
    buttons = _collect_buttons(card)
    assert any(
        b["text"]["content"] == UI_TEXT["project_btn_open_group"]
        and "openChatId=oc_group_123" in b["multi_url"]["url"]
        for b in buttons
    )


def test_system_builder_soft_failure_banner():
    """Verify that SystemBuilder.build_ttadk_soft_failure_card uses the banner."""
    message = "TTADK Timeout"
    msg_type, card_json = SystemBuilder.build_ttadk_soft_failure_card(message)

    card = json.loads(card_json)
    elements = card["body"]["elements"]
    assert elements[0]["tag"] == "column_set"
    assert elements[0]["background_style"] == "orange"  # soft failure 使用 warning 类型（Apple 风格优化：橙色）
    assert message in elements[0]["columns"][0]["elements"][0]["content"]


def test_system_error_card_action_buttons_have_callback_behaviors():
    """Error card detail/retry buttons must be clickable callback buttons in Feishu."""
    msg_type, card_json = SystemBuilder.build_error_card(
        RuntimeError("boom"),
        title="启动失败",
        summary="boom",
        detail_action={"action": "show_error_details", "session_id": "s1"},
        retry_action={"action": "deep_resume", "project_id": "p1"},
    )

    assert msg_type == "interactive"
    card = json.loads(card_json)
    buttons = _collect_buttons(card)
    assert len(buttons) == 2
    for button in buttons:
        assert button["value"]["action"]
        assert button.get("behaviors") == [{"type": "callback", "value": button["value"]}]


def test_ttadk_select_cards_keep_critical_fields_and_refresh_button():
    """TTADK tool/model/combined cards must keep select value and refresh fields stable."""
    tools = [ToolOptionView(name="coco", description="Coco agent")]
    models = [ModelOptionView(name="gpt-5", description="Flagship", display_name="GPT 5")]

    _, tool_json = SystemBuilder.build_ttadk_tool_select_card(
        tools, project_id="p1", current_tool="coco"
    )
    tool_card = json.loads(tool_json)
    tool_select = _collect_selects(tool_card)[0]
    assert tool_select["initial_option"] == "coco"
    assert tool_select["value"] == {"action": "select_ttadk_tool", "project_id": "p1"}
    assert tool_select["options"][0]["value"] == "coco"
    assert "Coco agent" in tool_select["options"][0]["text"]["content"]

    _, model_json = SystemBuilder.build_ttadk_model_select_card(
        models, tool_name="coco", project_id="p1", current_model="gpt-5"
    )
    model_card = json.loads(model_json)
    model_select = _collect_selects(model_card)[0]
    assert model_select["initial_option"] == "gpt-5"
    assert model_select["value"] == {
        "action": "select_ttadk_model",
        "tool_name": "coco",
        "project_id": "p1",
    }
    refresh_buttons = [b for b in _collect_buttons(model_card) if b["value"].get("action") == "refresh_ttadk_models"]
    assert refresh_buttons[0]["value"] == {
        "action": "refresh_ttadk_models",
        "tool_name": "coco",
        "project_id": "p1",
    }

    _, combined_json = SystemBuilder.build_ttadk_combined_select_card(
        tools,
        {"coco": models},
        project_id="p1",
        current_tool="coco",
        current_model="gpt-5",
    )
    combined_card = json.loads(combined_json)
    combined_selects = _collect_selects(combined_card)
    assert combined_selects[0]["value"] == {"action": "select_ttadk_combined_tool", "project_id": "p1"}
    assert combined_selects[1]["value"] == {
        "action": "select_ttadk_combined",
        "tool_name": "coco",
        "project_id": "p1",
    }


def test_acp_select_cards_keep_project_tool_thread_and_refresh_fields():
    """ACP selection cards must keep action payloads stable across helper extraction."""
    _, tool_json = SystemBuilder.build_acp_tool_select_card(
        [ToolOptionView(name="coco", description="Coco", emoji="🤖")],
        project_id="p1",
        current_tool="coco",
    )
    tool_buttons = _collect_buttons(json.loads(tool_json))
    assert tool_buttons[0]["type"] == "primary"
    assert tool_buttons[0]["value"] == {
        "action": "select_acp_tool",
        "tool_name": "coco",
        "project_id": "p1",
    }

    _, model_json = SystemBuilder.build_acp_model_select_card(
        [ModelOptionView(name="gpt-5", description="Flagship", display_name="GPT 5")],
        tool_name="coco",
        project_id="p1",
        current_model="gpt-5",
        thread_root_id="thread-1",
    )
    model_buttons = _collect_buttons(json.loads(model_json))
    default_model_button = model_buttons[0]
    assert default_model_button["type"] == "default"
    assert default_model_button["value"] == {
        "action": "select_acp_model",
        "tool_name": "coco",
        "model_name": DEFAULT_MODEL_OPTION_VALUE,
        "use_default_model": True,
        "project_id": "p1",
        "thread_root_id": "thread-1",
    }
    model_button = next(b for b in model_buttons if b["value"].get("model_name") == "gpt-5")
    assert model_button["type"] == "primary"
    assert model_button["value"] == {
        "action": "select_acp_model",
        "tool_name": "coco",
        "model_name": "gpt-5",
        "project_id": "p1",
        "thread_root_id": "thread-1",
    }
    assert model_buttons[-1]["value"] == {
        "action": "refresh_acp_models",
        "tool_name": "coco",
        "project_id": "p1",
        "thread_root_id": "thread-1",
    }


def test_acp_model_select_card_paginates_large_model_lists():
    """大模型列表（如 Traex ~90）应分页，避免飞书卡片元素超限 (ErrCode 11310)。"""
    models = [
        ModelOptionView(name=f"model_{i}", description=f"desc {i}")
        for i in range(90)
    ]

    _, page0_json = SystemBuilder.build_acp_model_select_card(
        models, tool_name="traex", project_id="p1", thread_root_id="t1", model_page=0
    )
    page0 = json.loads(page0_json)
    page0_buttons = _collect_buttons(page0)
    # First page should expose at most 20 concrete model buttons (+1 default +1 refresh + nav).
    model_only = [b for b in page0_buttons if str(b["value"].get("model_name", "")).startswith("model_")]
    assert len(model_only) == 20
    assert model_only[0]["value"]["model_name"] == "model_0"
    rendered = json.dumps(page0, ensure_ascii=False)
    assert "下一页" in rendered
    assert "上一页" not in rendered
    assert "第 1/5 页" in rendered
    # Page-nav button carries the page index via the refresh action.
    next_btn = next(b for b in page0_buttons if b["value"].get("model_page") == 1)
    assert next_btn["value"]["action"] == "refresh_acp_models"
    assert next_btn["value"]["tool_name"] == "traex"

    # Last page exposes the tail of the list with only a "previous" control.
    _, page4_json = SystemBuilder.build_acp_model_select_card(
        models, tool_name="traex", project_id="p1", thread_root_id="t1", model_page=4
    )
    page4 = json.loads(page4_json)
    page4_models = [
        b for b in _collect_buttons(page4)
        if str(b["value"].get("model_name", "")).startswith("model_")
    ]
    assert page4_models[-1]["value"]["model_name"] == "model_89"
    page4_rendered = json.dumps(page4, ensure_ascii=False)
    assert "上一页" in page4_rendered
    assert "下一页" not in page4_rendered


def test_acp_model_flow_status_cards_keep_refresh_and_ready_actions():
    """ACP 模型选择流的 loading/error/ready 三态应可原卡 PATCH。"""
    _, loading_json = SystemBuilder.build_acp_model_loading_card(
        "coco",
        project_id="p1",
        thread_root_id="thread-1",
    )
    loading_card = json.loads(loading_json)
    assert "coco" in loading_card["header"]["title"]["content"]
    assert "正在查询" in json.dumps(loading_card, ensure_ascii=False)
    assert _collect_buttons(loading_card) == []

    _, error_json = SystemBuilder.build_acp_model_error_card(
        "coco",
        project_id="p1",
        thread_root_id="thread-1",
    )
    error_buttons = _collect_buttons(json.loads(error_json))
    assert error_buttons[0]["value"] == {
        "action": "refresh_acp_models",
        "tool_name": "coco",
        "project_id": "p1",
        "thread_root_id": "thread-1",
    }

    _, ready_json = SystemBuilder.build_acp_programming_ready_card(
        "coco",
        "gpt-5.5",
        project_id="p1",
        thread_root_id="thread-1",
    )
    ready_card = json.loads(ready_json)
    assert "编程模式已就绪" in ready_card["header"]["title"]["content"]
    ready_buttons = _collect_buttons(ready_card)
    assert ready_buttons[0]["text"]["content"] == "切换模型"
    assert ready_buttons[0]["value"] == {
        "action": "refresh_acp_models",
        "tool_name": "coco",
        "project_id": "p1",
        "thread_root_id": "thread-1",
    }


def test_acp_programming_initializing_card_keeps_selected_tool_and_model():
    _, card_json = CardBuilder.build_acp_programming_initializing_card(
        "codex",
        "gpt-5.6-sol",
        project_id="p1",
        thread_root_id="thread-1",
    )

    card = json.loads(card_json)
    rendered = json.dumps(card, ensure_ascii=False)
    assert "正在初始化" in card["header"]["title"]["content"]
    assert "codex" in rendered
    assert "gpt-5.6-sol" in rendered
    assert _collect_buttons(card) == []


def test_acp_programming_failed_card_exposes_retry_and_model_selection_actions():
    _, card_json = CardBuilder.build_acp_programming_failed_card(
        "codex",
        "gpt-5.6-sol",
        "执行超时",
        project_id="p1",
        thread_root_id="thread-1",
    )

    card = json.loads(card_json)
    rendered = json.dumps(card, ensure_ascii=False)
    assert "初始化失败" in card["header"]["title"]["content"]
    assert "执行超时" in rendered
    buttons = _collect_buttons(card)
    assert [button["value"]["action"] for button in buttons] == [
        "select_acp_model",
        "refresh_acp_models",
    ]
    assert buttons[0]["value"] == {
        "action": "select_acp_model",
        "tool_name": "codex",
        "model_name": "gpt-5.6-sol",
        "project_id": "p1",
        "thread_root_id": "thread-1",
    }
    assert buttons[1]["value"] == {
        "action": "refresh_acp_models",
        "tool_name": "codex",
        "project_id": "p1",
        "thread_root_id": "thread-1",
    }


def test_system_error_card_uses_unified_error_visual_contract():
    """BaseHandler/SystemBuilder error card should share summary/details/retry visual contract."""
    msg_type, card_json = SystemBuilder.build_error_card(
        "boom",
        title="系统错误",
        details="stderr: boom\n/home/alice/project/.env SECRET_TOKEN=abc123",
        detail_action={"action": "show_error_details", "trace_id": "trace-1", "details": "stderr: boom"},
        retry_action={"action": "retry_original", "request_id": "req-1"},
    )
    card = json.loads(card_json)
    card_text = json.dumps(card, ensure_ascii=False)
    buttons = _collect_buttons(card)

    assert msg_type == "interactive"
    assert "错误摘要" in card_text
    assert "boom" in card_text
    assert "详情已收起" in card_text
    assert "stderr: boom" not in card_text
    assert "/home/alice/project" not in card_text
    assert "SECRET_TOKEN" not in card_text
    assert any(button["text"]["content"] == "查看详情" for button in buttons)
    assert any(button["text"]["content"].startswith("🔄") for button in buttons)
    detail_buttons = [button for button in buttons if button["value"].get("action") == "show_error_details"]
    assert detail_buttons
    detail_value = detail_buttons[0]["value"]
    assert detail_value["trace_id"] == "trace-1"
    assert "diagnostic_token" in detail_value
    assert "details" not in detail_value
    assert any(button["value"] == {"action": "retry_original", "request_id": "req-1"} for button in buttons)


def test_system_error_card_without_recoverable_action_hides_retry_button():
    """通用错误卡片没有原操作上下文时不得硬编码 show_status/retry_command。"""
    _, card_json = SystemBuilder.build_error_card(
        "boom",
        title="系统错误",
        details="trace 摘要: line 1",
        detail_action={"action": "show_error_details", "trace_id": "trace-2"},
    )
    card = json.loads(card_json)
    buttons = _collect_buttons(card)
    values = [button.get("value", {}) for button in buttons]
    card_text = json.dumps(card, ensure_ascii=False)

    assert "trace 摘要: line 1" not in card_text
    assert any(
        value.get("action") == "show_error_details"
        and value.get("trace_id") == "trace-2"
        and value.get("diagnostic_token")
        and "details" not in value
        for value in values
    )
    assert {"action": "show_status"} not in values
    assert {"action": "retry_command"} not in values
    assert not any(button["text"]["content"].startswith("🔄") for button in buttons)


def test_system_error_card_detail_action_payload_details_feed_diagnostic_store():
    from src.card.error_diagnostics import render_error_diagnostic

    _, card_json = SystemBuilder.build_error_card(
        "TTADK 启动失败",
        title="TTADK 暂不可用",
        severity="degraded",
        summary="系统已切换到备用路径",
        details="诊断详情已收起，点击“查看详情”可查看本次失败摘要。",
        detail_action={
            "action": "show_error_details",
            "trace_id": "trace-real-detail",
            "details": "real startup failure cwd=/data00/home/alice/project TOKEN=secret",
        },
    )
    buttons = _collect_buttons(json.loads(card_json))
    detail_value = next(button["value"] for button in buttons if button["value"].get("action") == "show_error_details")

    rendered = render_error_diagnostic(
        detail_value["diagnostic_token"],
        trace_id="trace-real-detail",
    )

    assert "real startup failure" in rendered
    assert "诊断详情已收起" not in rendered
    assert "/data00/home/alice" not in rendered
    assert "TOKEN=secret" not in rendered


def test_system_error_card_distinguishes_error_severity_visual_hierarchy():
    cases = {
        "recoverable": ("orange", "🟠 可恢复错误", "primary"),
        "degraded": ("yellow", "🟡 降级错误", "default"),
        "fatal": ("red", "🔴 致命错误", "default"),
    }

    for severity, (template, label, retry_type) in cases.items():
        _, card_json = SystemBuilder.build_error_card(
            "boom",
            title="系统错误",
            severity=severity,
            detail_action={"action": "show_error_details", "trace_id": severity},
            retry_action={
                "action": "retry_original",
                "request_id": severity,
                "original_mode": "Claude",
                "retry_mode": "Claude",
                "degraded_to": "Coco",
            },
        )
        card = json.loads(card_json)
        card_text = json.dumps(card, ensure_ascii=False)
        buttons = _collect_buttons(card)
        retry_buttons = [button for button in buttons if button["value"].get("action") == "retry_original"]

        assert card["header"]["template"] == template
        assert label in card_text
        assert retry_buttons[0]["type"] == retry_type


def test_degraded_error_card_without_complete_retry_payload_has_only_details_button():
    _, card_json = SystemBuilder.build_error_card(
        "部分能力不可用，已进入降级模式",
        title="系统降级",
        severity="degraded",
        detail_action={"action": "show_error_details", "trace_id": "degraded-no-quick-action"},
        continue_action={"action": "continue_degraded", "request_id": "req-degraded"},
        retry_action={"action": "retry_original", "request_id": "req-degraded"},
    )
    card = json.loads(card_json)
    buttons = _collect_buttons(card)
    retry_buttons = [button for button in buttons if button["value"].get("action") == "retry_original"]
    continue_buttons = [button for button in buttons if button["value"].get("action") == "continue_degraded"]
    button_labels = [button["text"]["content"] for button in buttons]
    button_blocks = _collect_button_layout_blocks(card)

    assert continue_buttons == []
    assert retry_buttons == []
    assert button_labels == [
        UI_TEXT["card_lifecycle_show_details"],
    ]
    assert [[button["text"]["content"] for button in block] for block in button_blocks[-1:]] == [
        [UI_TEXT["card_lifecycle_show_details"]],
    ]


@pytest.mark.parametrize(
    "retry_action",
    [
        None,
        {"action": "retry_original", "original_mode": "Claude", "retry_mode": "Claude"},
        {"action": "retry_original", "original_mode": "Claude", "retry_mode": "Claude", "degraded_to": ""},
    ],
)
def test_degraded_error_card_omits_retry_button_when_retry_payload_incomplete(retry_action):
    _, card_json = SystemBuilder.build_error_card(
        "部分能力不可用，已进入降级模式",
        title="系统降级",
        severity="degraded",
        detail_action={"action": "show_error_details", "trace_id": "trace-incomplete-retry"},
        continue_action={"action": "continue_degraded", "request_id": "req-degraded"},
        retry_action=retry_action,
    )

    buttons = _collect_buttons(json.loads(card_json))

    assert [button["value"].get("action") for button in buttons] == ["show_error_details"]


def test_degraded_error_card_with_quick_actions_keeps_single_decision_area():
    from src.utils.errors import GhostAPError

    _, card_json = SystemBuilder.build_error_card(
        GhostAPError(
            "部分能力不可用，已进入可用模式",
            quick_actions=["retry", "cancel"],
            context={"request_id": "req-quick"},
        ),
        title="系统降级",
        severity="degraded",
        detail_action={"action": "show_error_details", "trace_id": "trace-quick"},
        continue_action={"action": "continue_degraded", "request_id": "continue-quick", "degraded_to": "Aiden"},
        retry_action={
            "action": "retry_original",
            "request_id": "retry-quick",
            "original_mode": "TTADK",
            "retry_mode": "TTADK",
            "degraded_to": "Aiden",
        },
    )

    card = json.loads(card_json)
    buttons = _collect_buttons(card)
    button_blocks = _collect_button_layout_blocks(card)

    assert buttons[0]["text"]["content"] == UI_TEXT["card_lifecycle_degraded_primary"].format(mode="Aiden")
    assert buttons[0]["type"] == "primary"
    assert buttons[0]["value"] == {
        "action": "continue_degraded",
        "request_id": "continue-quick",
        "degraded_to": "Aiden",
    }
    assert [button["text"]["content"] for button in button_blocks[-3]] == [
        UI_TEXT["card_lifecycle_degraded_primary"].format(mode="Aiden")
    ]
    assert [button["value"]["action"] for button in button_blocks[-1]] == ["retry_original"]
    assert [button["value"]["action"] for button in button_blocks[-2]] == ["show_error_details"]
    assert all(button["type"] == "default" for button in button_blocks[-2] + button_blocks[-1])
    assert [button["value"]["action"] for button in buttons] == [
        "continue_degraded",
        "show_error_details",
        "retry_original",
    ]
    assert {button["value"].get("request_id") for button in buttons} >= {"continue-quick", "retry-quick"}


def test_degraded_error_card_sanitizes_summary_and_uses_action_allowlist():
    """降级错误卡 builder 边界必须拦截正文和按钮负载中的敏感字段。"""

    _, card_json = SystemBuilder.build_error_card(
        RuntimeError("boom cmd=rm -rf /tmp/secret cwd=/data00/home/user path=/data00/home/user/repo args=--token=abc"),
        title="Claude CLI 启动失败",
        severity="degraded",
        detail_action={
            "action": "show_error_details",
            "diagnostic_token": "diag-1",
            "trace_id": "trace-1",
            "cmd": "rm -rf /tmp/secret",
            "cwd": "/data00/home/user/repo",
            "path": "/data00/home/user/repo/file.py",
            "args": ["--token=abc"],
        },
        continue_action={
            "action": "continue_degraded",
            "degraded_to": "Coco",
            "request_id": "req-1",
            "cmd": "unsafe-command",
        },
        retry_action={
            "action": "retry_original",
            "original_mode": "Claude CLI",
            "retry_mode": "Claude CLI",
            "degraded_to": "Coco",
            "request_id": "req-1",
            "cwd": "/data00/home/user/repo",
        },
        details="stderr includes /data00/home/user/repo and TOKEN=abc",
    )

    card = json.loads(card_json)
    serialized = json.dumps(card, ensure_ascii=False)
    for leaked in ("rm -rf", "/data00/home/user", "--token=abc", "cmd=", "cwd=", "args="):
        assert leaked not in serialized

    buttons = _collect_buttons(card)
    allowed = {
        "action",
        "diagnostic_token",
        "trace_id",
        "request_id",
        "project_id",
        "degraded_to",
        "original_mode",
        "retry_mode",
    }
    for button in buttons:
        assert set(button["value"]) <= allowed
    assert buttons[0]["value"] == {"request_id": "req-1", "degraded_to": "Coco", "action": "continue_degraded"}
    assert buttons[1]["value"] == {
        "diagnostic_token": "diag-1",
        "trace_id": "trace-1",
        "action": "show_error_details",
    }
    assert buttons[2]["value"] == {
        "action": "retry_original",
        "request_id": "req-1",
        "degraded_to": "Coco",
        "original_mode": "Claude CLI",
        "retry_mode": "Claude CLI",
    }


def test_degraded_error_card_does_not_infer_degraded_to_from_legacy_next_mode():
    """Builder 只能消费上游显式 degraded_to，不能从 next_mode 修补降级语义。"""

    _, card_json = SystemBuilder.build_error_card(
        RuntimeError("ttadk failed"),
        title="TTADK 启动失败",
        severity="degraded",
        continue_action={"action": "continue_degraded", "next_mode": "Coco", "request_id": "req-legacy"},
        retry_action={
            "action": "retry_original",
            "original_mode": "TTADK",
            "retry_mode": "TTADK",
            "next_mode": "Coco",
            "request_id": "req-legacy",
        },
    )

    card = json.loads(card_json)
    serialized = json.dumps(card, ensure_ascii=False)
    buttons = _collect_buttons(card)

    assert "继续使用 Coco" not in serialized
    assert "当前暂未确定可继续模式" in serialized
    assert not any(button["value"].get("degraded_to") == "Coco" for button in buttons)
    assert all("next_mode" not in button["value"] for button in buttons)


def test_mobile_preview_covers_degraded_error_card_visual_contract():
    """ux/card_preview.html 必须可回归展示降级错误移动端卡片。"""
    from pathlib import Path

    html = (Path(__file__).resolve().parents[1] / "ux" / "card_preview.html").read_text(encoding="utf-8")

    assert ".header-yellow" in html
    assert "yellow — Degraded" in html
    assert ".button-row-primary .btn" in html
    assert ".button-row-secondary" in html
    assert "错误卡 · 降级错误" in html
    degraded_section = html.split("错误卡 · 降级错误", 1)[1].split("</div>\n</div>\n\n<!-- ==================== Footer Legend", 1)[0]
    assert degraded_section.index("button-row-primary") < degraded_section.index("button-row-secondary")
    assert degraded_section.index("继续使用 Coco") < degraded_section.index("查看详情")
    for expected in (
        "header-yellow",
        "🟡 降级错误",
        "错误摘要",
        "错误场景",
        "详情已收起",
        "查看详情",
        "重试原模式",
        "继续使用 Coco",
        "button-row-primary",
        "button-row-secondary",
        "btn-default",
        "btn-primary",
    ):
        assert expected in degraded_section
    assert "stderr" not in degraded_section
    assert "可用模式" not in degraded_section
    assert "available_mode" not in degraded_section


def test_select_option_long_name_and_description_are_mobile_safe():
    """长名称 + 描述应被压缩，避免移动端下拉项过宽。"""
    long_tool = ToolOptionView(name="tool-" + "x" * 80, description="desc-" + "y" * 120)

    _, card_json = SystemBuilder.build_ttadk_tool_select_card([long_tool], project_id="p1")
    select = _collect_selects(json.loads(card_json))[0]
    label = select["options"][0]["text"]["content"]

    assert len(label) <= 72
    assert label.endswith("…")
    assert select["options"][0]["value"] == long_tool.name


def test_acp_tool_button_long_name_and_description_are_mobile_safe():
    """ACP 工具按钮文案应压缩，但 callback payload 保留完整工具名。"""
    long_name = "tool-" + "x" * 80
    long_desc = "desc-" + "y" * 120

    _, card_json = SystemBuilder.build_acp_tool_select_card(
        [ToolOptionView(name=long_name, description=long_desc, emoji="🤖")],
        project_id="p1",
        current_tool=long_name,
    )
    button = _collect_buttons(json.loads(card_json))[0]

    assert len(button["text"]["content"]) <= 40
    assert button["text"]["content"].endswith("…")
    assert button["value"] == {"action": "select_acp_tool", "tool_name": long_name, "project_id": "p1"}


def test_acp_model_button_long_display_and_description_are_mobile_safe():
    """ACP 模型按钮展示名/描述应压缩，但 model_name 与 thread_root_id 不丢失。"""
    model_name = "model-" + "m" * 80
    display = "Display-" + "d" * 80
    desc = "Description-" + "x" * 120

    _, card_json = SystemBuilder.build_acp_model_select_card(
        [ModelOptionView(name=model_name, description=desc, display_name=display)],
        tool_name="coco",
        project_id="p1",
        current_model=model_name,
        thread_root_id="thread-1",
    )
    buttons = _collect_buttons(json.loads(card_json))
    button = next(b for b in buttons if b["value"].get("model_name") == model_name)

    assert len(button["text"]["content"]) <= 40
    assert button["text"]["content"].endswith("…")
    assert button["value"] == {
        "action": "select_acp_model",
        "tool_name": "coco",
        "model_name": model_name,
        "project_id": "p1",
        "thread_root_id": "thread-1",
    }


def test_ttadk_select_label_boundary_exact_limit_and_overflow():
    """TTADK 下拉项 72 字符边界：等长不截断，超长追加省略号。"""
    exact = "x" * 72
    overflow = "y" * 73

    _, exact_json = SystemBuilder.build_ttadk_tool_select_card([ToolOptionView(name=exact)], project_id="p1")
    exact_label = _collect_selects(json.loads(exact_json))[0]["options"][0]["text"]["content"]

    _, overflow_json = SystemBuilder.build_ttadk_tool_select_card([ToolOptionView(name=overflow)], project_id="p1")
    overflow_label = _collect_selects(json.loads(overflow_json))[0]["options"][0]["text"]["content"]

    assert exact_label == exact
    assert len(overflow_label) == 72
    assert overflow_label.endswith("…")


def test_selector_cards_empty_options_still_render_operable_shells():
    """空工具/模型列表不得生成坏卡片；可操作辅助入口仍保留。"""
    _, ttadk_tool_json = SystemBuilder.build_ttadk_tool_select_card([], project_id="p1")
    ttadk_tool_select = _collect_selects(json.loads(ttadk_tool_json))[0]
    assert ttadk_tool_select["options"] == []

    _, acp_tool_json = SystemBuilder.build_acp_tool_select_card([], project_id="p1")
    assert _collect_buttons(json.loads(acp_tool_json)) == []

    _, acp_model_json = SystemBuilder.build_acp_model_select_card([], tool_name="coco", project_id="p1")
    acp_model_buttons = _collect_buttons(json.loads(acp_model_json))
    assert [button["value"]["action"] for button in acp_model_buttons] == [
        "select_acp_model",
        "refresh_acp_models",
    ]
    assert acp_model_buttons[0]["value"] == {
        "action": "select_acp_model",
        "tool_name": "coco",
        "model_name": DEFAULT_MODEL_OPTION_VALUE,
        "use_default_model": True,
        "project_id": "p1",
        "thread_root_id": None,
    }


def test_deep_builder_warning_banner():
    """Verify that DeepBuilder.build_info_card uses the banner for warnings."""
    project = ProjectContext(project_id="p1", project_name="P1", root_path="/tmp/p1")
    state = EngineCardState(
        title="Running",
        content="Thinking...",
        engine_name="Coco",
        warning_banner="Low credits warning"
    )

    msg_type, card_json = DeepBuilder.build_info_card(project, state)
    card = json.loads(card_json)
    elements = card["body"]["elements"]

    # Find the banner in elements
    banner_found = False
    for el in elements:
        if el.get("tag") == "column_set" and el.get("background_style") == "orange":  # warning 使用橙色（Apple 风格优化）
            assert "Low credits warning" in el["columns"][0]["elements"][0]["content"]
            banner_found = True
            break

    assert banner_found is True


# ------------------------------------------------------------------
# CardBuilder facade diagnostics delegates
# ------------------------------------------------------------------


def test_card_builder_diagnostics_task_board_empty():
    """CardBuilder.build_task_board_content should show empty hint when no tasks exist."""

    content = CardBuilder.build_task_board_content(tasks=[])

    # DiagnosticsBuilder uses diag_no_active_tasks as the primary empty-state hint.
    assert UI_TEXT["diag_no_active_tasks"] in content


def test_card_builder_diagnostics_task_board_grouped_view():
    """Grouped task board view should include project name, status emoji and message."""

    class DummySpec:
        def __init__(self, name: str, task_type: str) -> None:
            self.name = name
            self.task_type = task_type

    class DummyState:
        def __init__(self) -> None:
            self.status = "running"
            self.progress_percent = 42
            self.progress_message = "syncing context"
            self.run_id = "run-12345"
            self.spec = DummySpec("Sync", "update")

    class DummyProject:
        def __init__(self, name: str) -> None:
            self.project_name = name

    class DummyProjectManager:
        def get_project_for_diagnostics(self, pid: str) -> DummyProject:
            return DummyProject(f"Project-{pid}")

    state = DummyState()
    groups = {"p1": [state]}

    content = CardBuilder.build_task_board_content(
        tasks=[state], groups=groups, project_manager=DummyProjectManager()
    )

    # Project name should be rendered from project manager
    assert "Project-p1" in content
    # Status emoji for "running" and progress message should be present
    assert "🔄" in content
    assert state.progress_message in content
    # Run id short form should contain the tail of run_id
    assert "run-12345" in content


def test_card_builder_unified_status_content_basic():
    """Unified status content should summarize engine entries with emojis."""

    entries = [
        EngineStatusEntry(
            mode="Deep",
            task_id="task-abcdef123456",
            name="DeepRun",
            status="running",
            info="executing",
        ),
        EngineStatusEntry(
            mode="Spec",
            task_id="task-ffffeeee1111",
            name="SpecRun",
            status="failed",
            info="error",
        ),
    ]

    content = CardBuilder.build_unified_status_content(entries, include_done=False, project_name="DemoProject")

    # Header should contain total count
    assert str(len(entries)) in content
    # Per-entry lines should include mode, name, info and mapped emojis
    assert "Deep" in content and "Spec" in content
    assert "DeepRun" in content and "SpecRun" in content
    assert "executing" in content and "error" in content
    assert "🔄" in content  # running
    assert "❌" in content  # failed
    # When include_done is False, the all-tasks hint should be appended
    assert UI_TEXT["diag_status_all_hint"] in content


def test_card_builder_format_engine_status_info_deep_uses_status_when_no_duration():
    """Deep mode with zero duration should fall back to raw status string."""

    class Dummy:
        def __init__(self) -> None:
            self.status = "RUNNING"

        def duration(self) -> float:
            return 0.0

    mode_label = UI_TEXT["diag_engine_deep"]
    info = CardBuilder.format_engine_status_info(mode_label, Dummy())
    assert info == "RUNNING"


def test_card_builder_format_engine_status_info_spec_includes_cycle_phase_and_criteria():
    """Spec mode should include cycle number, phase label and criteria ratio when available."""

    class Phase:
        def __init__(self, display_name: str) -> None:
            self.display_name = display_name

    class Cycle:
        def __init__(self) -> None:
            self.phase = Phase("Plan")

    class Dummy:
        def __init__(self) -> None:
            self.status = "SPEC"
            self.current_cycle_number = 2
            self.current_cycle = Cycle()
            self.satisfied_count = 2
            self.total_criteria = 5

        def duration(self) -> float:
            return 0.0

    mode_label = UI_TEXT["diag_engine_spec"]
    info = CardBuilder.format_engine_status_info(mode_label, Dummy())

    assert "2" in info
    assert "Plan" in info
    assert "2/5" in info


# ---------------------------------------------------------------------------
# Verify stop/stop_danger confirm dialog fields
# ---------------------------------------------------------------------------


class TestStopButtonConfirmDialogs:
    """Verify stop and stop_danger buttons include proper confirm dialogs."""

    def _make_executing_state(self):
        """Create an EngineCardState that is executing."""
        from src.card.builders.deep import EngineCardState
        return EngineCardState(
            is_executing=True,
            is_paused=False,
            content="x\n" * 60,
            compact=False,
            expanded=False,
            action_prefix="deep",
            engine_project_id="proj1",
            project_id="proj1",
        )

    def test_stop_danger_confirm_has_title_and_body(self):
        """stop_danger button JSON includes confirm.title.content and confirm.text.content."""
        from src.card.builders.deep import DeepBuilder
        state = self._make_executing_state()
        btn = DeepBuilder._create_button("stop_danger", state)
        assert "confirm" in btn
        assert "title" in btn["confirm"]
        assert "content" in btn["confirm"]["title"]
        assert "text" in btn["confirm"]
        assert "content" in btn["confirm"]["text"]
        # Verify it uses the danger body text
        from src.card.ui_text import UI_TEXT
        expected = UI_TEXT["card_btn_confirm_stop_danger_body"].format(engine_cmd="/deep")
        assert btn["confirm"]["text"]["content"] == expected


# ---------------------------------------------------------------------------
# timeout_display formatting (system help card)
# ---------------------------------------------------------------------------


class TestTimeoutDisplayFormatting:
    """Test timeout_display formatting logic in system help card."""

    @pytest.mark.parametrize("timeout_seconds,expected_substring", [
        (300, "5 分钟"),
        (3600, "60 分钟"),
        (7200, "2 小时"),
    ])
    def test_timeout_display_format(self, timeout_seconds, expected_substring):
        """Verify timeout display formatting for various second values."""
        import math

        timeout_minutes = max(1, math.ceil(timeout_seconds / 60))
        if timeout_minutes >= 120:
            hours = timeout_minutes // 60
            timeout_display = f"{hours} 小时" if timeout_seconds % 3600 == 0 else f"约 {hours} 小时"
        else:
            timeout_display = f"{timeout_minutes} 分钟" if timeout_seconds % 60 == 0 else f"约 {timeout_minutes} 分钟"

        assert timeout_display == expected_substring


# ---------------------------------------------------------------------------
# button_size parameter pass-through tests
# ---------------------------------------------------------------------------


class TestButtonSizePassthrough:
    """Verify button_size propagates from build_mode_buttons to final button dict."""

    @pytest.mark.parametrize("size", ["small", "medium", "large"])
    def test_button_size_in_mode_buttons(self, size):
        from src.card.shared import build_mode_buttons

        buttons = build_mode_buttons(mode=None, button_size=size)
        assert buttons, "Expected at least one button"
        for btn in buttons:
            assert btn["size"] == size, f"Expected size={size}, got {btn.get('size')}"


# ---------------------------------------------------------------------------
# stop button type config test
# ---------------------------------------------------------------------------


class TestStopButtonConfig:
    """Verify stop button uses danger type with confirm dialog."""

    def test_stop_button_config(self):
        from src.card.buttons_config import BUTTON_CONFIG

        assert BUTTON_CONFIG["stop"]["type"] == "danger"
        assert "confirm" in BUTTON_CONFIG["stop"]
        assert BUTTON_CONFIG["stop"]["confirm"]["title"] == "确认停止"
        assert BUTTON_CONFIG["stop_danger"]["type"] == "danger"


# ---------------------------------------------------------------------------
# Review hardening tests (from review cycle feedback)
# ---------------------------------------------------------------------------


class TestDeepActionPrefixStrip:
    """Verify action_prefix edge cases with empty/whitespace strings."""

    def test_empty_string_uses_fallback(self):
        """action_prefix='' should use deep_error_fallback_no_prefix."""
        from src.card.ui_text import UI_TEXT

        state = EngineCardState(
            title="Error occurred",
            content="",
            compact=True,
            terminal_state="error",
            action_prefix="",
        )
        _, card_json = CardBuilder.build_info_card(None, state)
        card = json.loads(card_json)
        body_text = json.dumps(card, ensure_ascii=False)
        assert UI_TEXT["deep_error_fallback_no_prefix"] in body_text
        assert "发送 / " not in body_text  # no broken "/ " command

class TestStopConfirmNoRollbackWording:
    """Verify stop confirm body does not contain alarming text and is decision-oriented."""

    def test_danger_body_content(self):
        from src.card.ui_text import UI_TEXT

        body = UI_TEXT["card_btn_confirm_stop_danger_body"]
        assert "回滚" not in body
        assert "强制停止" in body
        assert "普通停止" in body
        assert "{engine_cmd}" in body



class TestHelpCardTimeoutNotePosition:
    """Verify timeout_note is rendered in the bottom notation area, not between quick buttons and sections."""

    def test_timeout_note_after_sections(self):
        """timeout_note should appear after command sections, near tips (bottom)."""
        SystemBuilder._build_help_card_cached.cache_clear()
        _, card_json = SystemBuilder.build_help_card(
            session_idle_timeout=1800,
            session_idle_warn_at_remaining=180,
            lock_undo_window_seconds=300,
        )
        card = json.loads(card_json)
        body_elements = card["body"]["elements"]

        # Find positions of key elements
        timeout_note_idx = None
        quick_entry_idx = None
        for i, el in enumerate(body_elements):
            content = el.get("content", "")
            if "⏰" in content:
                timeout_note_idx = i
            if "system_help_tips" in content or "/help" in content:
                # tips is a notation-size markdown at the bottom
                if el.get("text_size") == "notation" and "发送" in content:
                    pass
            if "快速入口" in content or "system_help_quick_entry" in content:
                quick_entry_idx = i

        # timeout_note should exist and be near the end (after sections, near tips)
        assert timeout_note_idx is not None, "timeout_note not found in card elements"
        # It should NOT be directly after quick_entry (that was the old position)
        if quick_entry_idx is not None:
            # There should be at least several elements between quick_entry and timeout_note
            assert timeout_note_idx - quick_entry_idx > 3, (
                f"timeout_note (idx={timeout_note_idx}) is too close to quick_entry (idx={quick_entry_idx})"
            )


class TestBuildDeepCardAliasRemoved:
    """Guard: deprecated build_deep_card aliases stay removed."""

    def test_deep_card_aliases_removed(self):
        from src.card.builder import CardBuilder
        from src.card.builders.deep import DeepBuilder

        assert not hasattr(DeepBuilder, "build_deep_card")
        assert not hasattr(CardBuilder, "build_deep_card")
