"""Tests for Workflow orchestrator agent selection (AC2 related)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.feishu.handlers.workflow import WorkflowHandler
from src.workflow_engine.constants import (
    DEFAULT_ORCHESTRATOR_AGENT,
    ORCHESTRATOR_AGENT_OPTIONS,
)
from src.workflow_engine.models import (
    PendingConfirmation,
    WorkflowProject,
    WorkflowStatus,
)


def _create_mock_handler():
    """Create a mock WorkflowHandler with basic dependencies."""
    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.reply_card = MagicMock()
    handler.send_card_to_chat = MagicMock()
    handler.update_card = MagicMock()
    handler._reply_workflow_error = MagicMock()
    handler._show_tool_selection_card = MagicMock()
    handler._send_combined_selection_card = MagicMock()
    handler._resolve_tool_lists = MagicMock(return_value=({}, [], [], []))
    handler._get_root_path = MagicMock(return_value="/tmp")
    handler._get_project_for_chat = MagicMock(return_value=MagicMock(project_id="test_proj"))
    handler.get_engine_name = MagicMock(return_value="test_engine")
    return handler


def test_script_gen_uses_selected_agent_and_model():
    """脚本生成使用用户选择的 Agent 类型和模型。"""
    from src.workflow_engine.script_gen import build_script_gen_prompt

    # Test with different orchestrator agents
    for agent_type, _, _ in ORCHESTRATOR_AGENT_OPTIONS:
        # Test with default model
        prompt = build_script_gen_prompt(
            requirement="test",
            available_tools=["coco", "claude"],
            orchestrator_agent=agent_type,
        )

        # The agent type should appear in the prompt
        assert agent_type in prompt, f"Agent type {agent_type} not found in prompt"

        # Test with specific model
        orchestrator_binding = {
            "tool_name": agent_type,
            "model_name": "gpt-4",
            "use_default_model": False
        }

        prompt_with_model = build_script_gen_prompt(
            requirement="test",
            available_tools=["coco", "claude"],
            orchestrator_agent=agent_type,
            orchestrator_binding=orchestrator_binding,
        )

        # The agent type and model should appear in the prompt
        assert agent_type in prompt_with_model
        assert "gpt-4" in prompt_with_model
        assert "已选择的主 Agent" in prompt_with_model


def test_script_gen_default_agent():
    """未指定 orchestrator_agent 时使用默认值。"""
    from src.workflow_engine.script_gen import build_script_gen_prompt

    prompt = build_script_gen_prompt(
        requirement="test",
        available_tools=["coco"],
        # 不指定 orchestrator_agent
    )

    # Should use default agent
    assert DEFAULT_ORCHESTRATOR_AGENT in prompt


def _build_handler_for_regen():
    """Helper: build a WorkflowHandler that can call _generate_and_show_confirm_card
    or handle_workflow_regenerate_script without touching the real filesystem."""
    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.reply_text = MagicMock()
    handler.send_card_to_chat = MagicMock()
    handler.update_card = MagicMock()
    handler._reply_workflow_error = MagicMock()
    handler._build_confirm_card = MagicMock(return_value={"header": {"title": {"content": "confirm"}}})
    handler._read_pending_script = MagicMock(return_value="")
    handler._get_root_path = MagicMock(return_value="/tmp/test_proj")
    handler.get_engine_name = MagicMock(return_value="test_engine")
    return handler


def test_regenerate_script_preserves_orchestrator_agent():
    """点击"重新生成脚本"后，pending.orchestrator_agent 必须保持为 claude（不会被重置为 coco）。"""
    handler = _build_handler_for_regen()
    handler._schedule_generate_and_show_confirm_card = MagicMock()

    mock_project = MagicMock()
    mock_project.project_id = "test_proj"
    mock_project.root_path = "/tmp/test_proj"

    mock_engine = MagicMock()
    mock_engine.project = WorkflowProject(
        workflow_id="test_wf",
        status=WorkflowStatus.AWAITING_CONFIRM,
        pending=PendingConfirmation(
            script_path="/tmp/test_proj/.ghostap/workflow_scripts/generated_workflow.js",
            requirement="test requirement",
            meta={"tools": ["coco"]},
            is_fallback=False,
            initiator_user_id="test_user",
            engine_session_key="session_abc",
            selected_tools=["coco"],
            tools_mismatch=False,
            orchestrator_agent="claude",  # 用户之前选择的 agent
        ),
    )
    handler.ctx.workflow_engine_manager.get = MagicMock(return_value=mock_engine)
    handler.ctx.workflow_engine_manager.get_or_create = MagicMock(return_value=mock_engine)
    handler.ctx.project_manager.get_project = MagicMock(return_value=mock_project)
    handler._start_pending_workflow_execution = MagicMock(return_value=True)

    with (
        patch("src.feishu.handlers.workflow.os.path.exists", return_value=True),
        patch("src.feishu.handlers.workflow.os.remove"),
        patch("src.thread.get_current_sender_id", return_value="test_user"),
    ):
        handler.handle_workflow_regenerate_script(
            message_id="msg_regen",
            chat_id="test_chat",
            project_id="test_proj",
            value={
                "action": "workflow_regenerate_script",
                "engine_session_key": "session_abc",
                "project_id": "test_proj",
            },
        )

    assert mock_engine.project.pending.orchestrator_agent == "claude"
    assert mock_engine.project.status == WorkflowStatus.GENERATING_SCRIPT
    handler._schedule_generate_and_show_confirm_card.assert_called_once()
    handler._reply_workflow_error.assert_not_called()


def test_generate_and_show_confirm_card_preserves_orchestrator_agent():
    """_generate_and_show_confirm_card 重建 pending 时，应保留原有的 orchestrator_agent。"""
    handler = _build_handler_for_regen()

    mock_engine = MagicMock()
    mock_engine.project = WorkflowProject(
        workflow_id="test_wf",
        status=WorkflowStatus.GENERATING_SCRIPT,
        pending=PendingConfirmation(
            requirement="some requirement",
            initiator_user_id="test_user",
            engine_session_key="session_xyz",
            selected_tools=["coco"],
            orchestrator_agent="claude",
        ),
    )
    handler.ctx.workflow_engine_manager.get_or_create = MagicMock(return_value=mock_engine)

    # 让 discover_templates 返回空列表，确保走 AI 路径
    with patch("src.workflow_engine.templates.discover_templates", return_value=[]):
        with patch("src.thread.get_current_sender_id", return_value="test_user"):
            # Stub out _generate_script_via_ai to avoid real AI calls
            def fake_gen(requirement, root_path, selected_tools, engine, **kwargs):
                return ("/tmp/test_proj/.ghostap/workflow_scripts/generated_workflow.js", {"tools": ["coco"]}, False)
            handler._generate_script_via_ai = fake_gen

            handler._generate_and_show_confirm_card(
                message_id="msg",
                chat_id="test_chat",
                requirement="some requirement",
                project=MagicMock(project_id="test_proj"),
                root_path="/tmp/test_proj",
                selected_tools=["coco"],
                expected_session_key="session_xyz",
            )

    assert mock_engine.project.pending is not None
    assert mock_engine.project.pending.orchestrator_agent == "claude", (
        "_generate_and_show_confirm_card 未保留 orchestrator_agent"
    )


def test_generate_and_show_confirm_card_defaults_orchestrator_when_missing():
    """在没有现有 pending 或 orchestrator_agent 为空时，应使用 DEFAULT_ORCHESTRATOR_AGENT。"""
    handler = _build_handler_for_regen()

    mock_engine = MagicMock()
    mock_engine.project = WorkflowProject(
        workflow_id="test_wf",
        status=WorkflowStatus.GENERATING_SCRIPT,
        pending=PendingConfirmation(
            requirement="some requirement",
            initiator_user_id="test_user",
            engine_session_key="session_xyz",
            selected_tools=["coco"],
        ),
    )
    handler.ctx.workflow_engine_manager.get_or_create = MagicMock(return_value=mock_engine)

    with patch("src.workflow_engine.templates.discover_templates", return_value=[]):
        with patch("src.thread.get_current_sender_id", return_value="test_user"):
            def fake_gen(requirement, root_path, selected_tools, engine, **kwargs):
                return ("/tmp/test_proj/.ghostap/workflow_scripts/generated_workflow.js", {"tools": ["coco"]}, False)
            handler._generate_script_via_ai = fake_gen

            handler._generate_and_show_confirm_card(
                message_id="msg",
                chat_id="test_chat",
                requirement="some requirement",
                project=MagicMock(project_id="test_proj"),
                root_path="/tmp/test_proj",
                selected_tools=["coco"],
                expected_session_key="session_xyz",
            )

    assert mock_engine.project.pending.orchestrator_agent == DEFAULT_ORCHESTRATOR_AGENT


def test_generate_and_show_confirm_card_auto_starts_without_confirm_card():
    """脚本生成完成后应直接启动执行，而不是停在确认卡。"""
    handler = _build_handler_for_regen()
    handler.send_card_to_chat = MagicMock(return_value="generating_card")
    handler._resolve_origin = MagicMock(return_value="origin_msg")
    handler._replace_or_send_workflow_card = MagicMock()
    handler._start_pending_workflow_execution = MagicMock(return_value=True)

    mock_project = MagicMock()
    mock_project.project_id = "test_proj"
    mock_project.root_path = "/tmp/test_proj"

    mock_engine = MagicMock()
    mock_engine.project = WorkflowProject(
        workflow_id="test_wf",
        status=WorkflowStatus.GENERATING_SCRIPT,
        pending=PendingConfirmation(
            requirement="test requirement",
            initiator_user_id="test_user",
            engine_session_key="session_abc",
            selected_tools=["coco"],
            orchestrator_agent="coco",
        ),
    )
    handler.ctx.workflow_engine_manager.get_or_create = MagicMock(return_value=mock_engine)

    handler._generate_script_via_ai = MagicMock(
        return_value=(
            "/tmp/test_proj/.ghostap/workflow_scripts/generated_workflow.js",
            {"name": "test-wf", "description": "Test", "phases": [], "tools": ["coco"]},
            False,
        )
    )

    with patch("src.workflow_engine.templates.discover_templates", return_value=[]):
        handler._generate_and_show_confirm_card(
            message_id="loading_msg",
            chat_id="test_chat",
            requirement="test requirement",
            project=mock_project,
            root_path="/tmp/test_proj",
            selected_tools=["coco"],
            expected_session_key="session_abc",
        )

    handler._build_confirm_card.assert_not_called()
    handler._replace_or_send_workflow_card.assert_not_called()
    assert handler.send_card_to_chat.call_args.kwargs.get("origin_message_id") == "origin_msg"
    handler._start_pending_workflow_execution.assert_called_once_with(
        message_id="generating_card",
        chat_id="test_chat",
        project_id="test_proj",
        project=mock_project,
        root_path="/tmp/test_proj",
        engine=mock_engine,
        allow_server_side_start=True,
        generation_owner=handler._start_pending_workflow_execution.call_args.kwargs[
            "generation_owner"
        ],
    )
    assert mock_engine.project.pending.initiator_user_id == "test_user"


def test_generate_and_show_confirm_card_starts_from_loading_card_without_confirm_fallback():
    """生成完成后直接从 loading 卡进入执行，不再 fallback 到确认卡。"""
    handler = _build_handler_for_regen()
    handler.send_card_to_chat.return_value = "loading_msg"
    handler.update_card.return_value = False
    handler._replace_or_send_workflow_card = MagicMock()
    handler._start_pending_workflow_execution = MagicMock(return_value=True)

    mock_engine = MagicMock()
    mock_engine.project = WorkflowProject(
        workflow_id="test_wf",
        status=WorkflowStatus.GENERATING_SCRIPT,
        pending=PendingConfirmation(
            requirement="some requirement",
            initiator_user_id="test_user",
            engine_session_key="session_xyz",
            selected_tools=["coco"],
            orchestrator_agent="claude",
        ),
    )
    handler.ctx.workflow_engine_manager.get_or_create = MagicMock(return_value=mock_engine)

    with patch("src.workflow_engine.templates.discover_templates", return_value=[]):
        with patch("src.thread.get_current_sender_id", return_value="test_user"):
            handler._generate_script_via_ai = MagicMock(
                return_value=(
                    "/tmp/test_proj/.ghostap/workflow_scripts/generated_workflow.js",
                    {"tools": ["coco"]},
                    False,
                )
            )

            handler._generate_and_show_confirm_card(
                message_id="msg",
                chat_id="test_chat",
                requirement="some requirement",
                project=MagicMock(project_id="test_proj"),
                root_path="/tmp/test_proj",
                selected_tools=["coco"],
                expected_session_key="session_xyz",
            )

    handler._build_confirm_card.assert_not_called()
    handler.update_card.assert_not_called()
    handler._replace_or_send_workflow_card.assert_not_called()
    handler._start_pending_workflow_execution.assert_called_once()


def test_generate_and_show_confirm_card_replaces_loading_card_on_template_validation_failure():
    """Template validation errors must close the loading card instead of leaving it stuck."""
    handler = _build_handler_for_regen()
    handler.send_card_to_chat.return_value = "loading_msg"
    handler.update_card.return_value = True

    mock_engine = MagicMock()
    mock_engine.project = WorkflowProject(
        workflow_id="test_wf",
        status=WorkflowStatus.GENERATING_SCRIPT,
        pending=PendingConfirmation(
            requirement="bad-template",
            initiator_user_id="test_user",
            engine_session_key="session_xyz",
            selected_tools=["coco"],
            orchestrator_agent="claude",
        ),
    )
    handler.ctx.workflow_engine_manager.get_or_create = MagicMock(return_value=mock_engine)
    handler.reply_error = MagicMock()

    with patch(
        "src.workflow_engine.templates.discover_templates",
        return_value=[SimpleNamespace(name="bad-template")],
    ):
        with patch("src.workflow_engine.templates.load_template", return_value="not valid workflow js"):
            with patch("src.thread.get_current_sender_id", return_value="test_user"):
                handler._generate_and_show_confirm_card(
                    message_id="msg",
                    chat_id="test_chat",
                    requirement="bad-template",
                    project=MagicMock(project_id="test_proj"),
                    root_path="/tmp/test_proj",
                    selected_tools=["coco"],
                    expected_session_key="session_xyz",
                )

    handler.reply_error.assert_not_called()
    handler.update_card.assert_called_once()
    assert handler.update_card.call_args[0][0] == "loading_msg"
    updated_card = handler.update_card.call_args[0][1]
    assert "模板" in str(updated_card)
    assert mock_engine.project.status == WorkflowStatus.IDLE
    assert mock_engine.project.pending is None


def test_invalid_template_stops_generation_heartbeat_before_return():
    """An early validation return must not leave the 8-second updater alive."""
    handler = _build_handler_for_regen()
    handler.send_card_to_chat.return_value = "loading_msg"
    handler.update_card.return_value = True

    class FakeThread:
        def __init__(self, *, target, name, daemon):
            self.target = target
            self.name = name
            self.daemon = daemon
            self.started = False
            self.joined = False

        def start(self):
            self.started = True

        def join(self, timeout=None):
            self.joined = True

    created_threads = []

    def make_thread(**kwargs):
        thread = FakeThread(**kwargs)
        created_threads.append(thread)
        return thread

    mock_engine = MagicMock()
    mock_engine.project = WorkflowProject(
        workflow_id="test_wf",
        status=WorkflowStatus.GENERATING_SCRIPT,
        pending=PendingConfirmation(
            requirement="bad-template",
            initiator_user_id="test_user",
            engine_session_key="session_xyz",
            selected_tools=["coco"],
            orchestrator_agent="claude",
        ),
    )
    handler.ctx.workflow_engine_manager.get_or_create = MagicMock(return_value=mock_engine)

    with (
        patch("threading.Thread", side_effect=make_thread),
        patch(
            "src.workflow_engine.templates.discover_templates",
            return_value=[SimpleNamespace(name="bad-template")],
        ),
        patch("src.workflow_engine.templates.load_template", return_value="not valid workflow js"),
        patch("src.thread.get_current_sender_id", return_value="test_user"),
    ):
        handler._generate_and_show_confirm_card(
            message_id="msg",
            chat_id="test_chat",
            requirement="bad-template",
            project=MagicMock(project_id="test_proj"),
            root_path="/tmp/test_proj",
            selected_tools=["coco"],
            expected_session_key="session_xyz",
        )

    assert len(created_threads) == 1
    assert created_threads[0].started is True
    assert created_threads[0].joined is True


# ---------------------------------------------------------------------------
# 联合卡片 (combined card) 与 orchestrator 选择相关测试
# ---------------------------------------------------------------------------


def _flatten_actions_from_card(card: dict) -> list[dict]:
    """从卡片 dict 中递归提取所有 action 按钮（含 value）。"""
    elements = card.get("elements", [])
    if not elements and "body" in card and isinstance(card["body"], dict):
        elements = card["body"].get("elements", [])
    result: list[dict] = []

    def _walk(obj):
        if isinstance(obj, dict):
            if obj.get("tag") == "action":
                for act in obj.get("actions", []):
                    if isinstance(act, dict) and "value" in act:
                        result.append(act)
            elif obj.get("tag") == "button":
                if "value" in obj:
                    result.append(obj)
            elif obj.get("tag") == "select_static":
                if "value" in obj:
                    result.append(obj)
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(elements)
    _walk(card)
    return result


class _FakeProject:
    """一个简单的 project 对象，提供 selection snapshot 存储槽，供 SelectionFlowController 使用。"""

    def __init__(self):
        self.project_id = "proj_1"
        self.id = "proj_1"
        self.root_path = "/tmp/test_proj"
        self._wf_selection_snapshot: dict | None = None
        self._wf_selection_controller = None


def _build_orchestrator_handler_with_project():
    """构造一个带真实 project + mock engine 的 WorkflowHandler，适用于联合卡片相关测试。

    使用新的 SelectionFlowController 路径：controller 直接实例化或通过
    ``_get_selection_controller`` 获得，state 存储在 project 的
    ``_wf_selection_snapshot`` 上。
    """

    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.reply_text = MagicMock()
    handler.reply_card = MagicMock()
    handler.send_card_to_chat = MagicMock()
    handler.update_card = MagicMock()
    handler._reply_workflow_error = MagicMock()
    handler._get_root_path = MagicMock(return_value="/tmp/test_proj")
    handler.get_engine_name = MagicMock(return_value="test_engine")
    handler._resolve_tool_lists = MagicMock(return_value=({"coco": "AI编程助手"}, ["coco"], [], []))
    handler._get_workflow_models_for_tool = MagicMock(return_value=[
        {"name": "gpt-4", "display_name": "GPT-4"},
        {"name": "claude-3-opus", "display_name": "Claude 3 Opus"},
    ])
    handler._dispatch_workflow_review_tool_select = MagicMock()

    mock_project = _FakeProject()

    mock_engine = MagicMock()
    mock_engine.project = WorkflowProject(
        workflow_id="wf_1",
        status=WorkflowStatus.AWAITING_AGENT_SELECT,
        pending=PendingConfirmation(
            requirement="build a feature",
            initiator_user_id="user_1",
            engine_session_key="sess_abc",
            selected_tools=[],
        ),
    )
    handler.ctx.workflow_engine_manager.get = MagicMock(return_value=mock_engine)
    handler.ctx.workflow_engine_manager.get_or_create = MagicMock(return_value=mock_engine)

    handler._resolve_project_from_id = MagicMock(return_value=mock_project)
    handler.ctx.project_manager.get_project_for_chat = MagicMock(return_value=mock_project)

    return handler, mock_project, mock_engine


# ---------------------------------------------------------------------------
# controller 级别的卡片正确性测试（使用 SelectionFlowController 直接构建）
# ---------------------------------------------------------------------------


def test_orchestrator_combined_card_initial():
    """SelectionFlowController.build_orchestrator_combined_card 生成的卡片必须包含
    header、elements、工具选择按钮以及底部 finish 动作按钮。"""
    from src.card.actions.dispatch import (
        WORKFLOW_ORCHESTRATOR_FINISH,
        WORKFLOW_ORCHESTRATOR_SELECT_TOOL,
    )
    from src.workflow_engine.selection_flow import SelectionFlowController

    ctrl = SelectionFlowController(step=1)
    card = ctrl.build_orchestrator_combined_card(
        available_tools=[{"tool_name": "coco", "display_name": "Coco", "description": "AI编程助手"}],
        requirement="build a feature",
        session_key="sess_abc",
    )

    assert isinstance(card, dict)
    assert "header" in card
    # Card is wrapped by CardBuilder._wrap_card, so elements are under 'body'
    assert "body" in card
    assert "elements" in card["body"]

    actions = _flatten_actions_from_card(card)
    action_values = [a.get("value", {}).get("action", "") for a in actions]

    assert WORKFLOW_ORCHESTRATOR_SELECT_TOOL in action_values, (
        f"未在卡片中找到 orchestrator select tool 按钮；实际 actions: {action_values}"
    )
    # orchestrator 卡片在空选择时也应包含 finish 按钮（validate 失败显示错误信息，按钮仍存在）
    assert WORKFLOW_ORCHESTRATOR_FINISH in action_values, (
        f"卡片未包含 finish 按钮；实际 actions: {action_values}"
    )

    # 工具按钮 payload 中应含有 tool_name 字段
    tool_btn = next(
        (a for a in actions if a.get("value", {}).get("action") == WORKFLOW_ORCHESTRATOR_SELECT_TOOL),
        None,
    )
    assert tool_btn is not None
    assert tool_btn.get("value", {}).get("tool_name") == "coco"


def test_orchestrator_combined_card_has_inline_model_panel_when_expanded():
    """toggle_tool_expand('coco') 后，build_orchestrator_combined_card 必须包含
    WORKFLOW_ORCHESTRATOR_SELECT_MODEL 按钮（至少一个默认模型 + 一个具体模型）。"""
    from src.card.actions.dispatch import WORKFLOW_ORCHESTRATOR_SELECT_MODEL
    from src.workflow_engine.selection_flow import SelectionFlowController

    ctrl = SelectionFlowController(step=1)
    ctrl.toggle_tool_expand("coco", is_review=False)
    card = ctrl.build_orchestrator_combined_card(
        available_tools=[{"tool_name": "coco", "display_name": "Coco"}],
        available_models=[
            {"name": "gpt-4", "display_name": "GPT-4"},
            {"name": "claude-3-opus", "display_name": "Claude 3 Opus"},
        ],
        requirement="build a feature",
        session_key="sess_abc",
    )

    actions = _flatten_actions_from_card(card)
    model_actions = [
        a for a in actions
        if a.get("value", {}).get("action") == WORKFLOW_ORCHESTRATOR_SELECT_MODEL
    ]
    assert len(model_actions) >= 2, (
        f"展开后应当包含默认模型 + 至少一个具体模型按钮；实际共 {len(model_actions)} 个；"
        f"所有 actions: {[a.get('value', {}).get('action') for a in actions]}"
    )

    has_default = any(bool(a.get("value", {}).get("use_default_model")) for a in model_actions)
    has_specific = any(a.get("value", {}).get("model_name") == "gpt-4" for a in model_actions)
    assert has_default, "未找到 use_default_model=True 的按钮"
    assert has_specific, "未找到 model_name=gpt-4 的具体模型按钮"


def test_handle_orchestrator_select_tool_expands_model_panel():
    """调用 handle_workflow_orchestrator_select_tool 时应原地刷新卡片（update_card 被调用）。"""
    handler, mock_project, mock_engine = _build_orchestrator_handler_with_project()

    value = {
        "action": "workflow_orchestrator_select_tool",
        "tool_name": "coco",
        "provider": "workflow",
        "display_name": "Coco",
        "supports_model": True,
        "chat_id": "chat_1",
        "project_id": "proj_1",
        "engine_session_key": "sess_abc",
    }

    with patch("src.thread.get_current_sender_id", return_value="user_1"):
        handler.handle_workflow_orchestrator_select_tool(
            message_id="msg_1",
            chat_id="chat_1",
            project_id="proj_1",
            value=value,
        )

    # update_card 应该被调用，表示原地刷新卡片
    handler.update_card.assert_called()
    # 不应该报任何错误
    handler._reply_workflow_error.assert_not_called()

    # 二次检查：状态仍为 AWAITING_AGENT_SELECT（还没有进入下一步）
    assert mock_engine.project.status == WorkflowStatus.AWAITING_AGENT_SELECT


def test_handle_orchestrator_select_tool_accepts_dropdown_option():
    """select_static 回调通过 _option 返回选择值，handler 应使用它更新 pending tool。"""
    handler, mock_project, _mock_engine = _build_orchestrator_handler_with_project()

    value = {
        "action": "workflow_orchestrator_select_tool",
        "_option": "traex",
        "chat_id": "chat_1",
        "project_id": "proj_1",
        "engine_session_key": "sess_abc",
    }

    with patch("src.thread.get_current_sender_id", return_value="user_1"), \
            patch("src.workflow_engine.tool_registry.get_available_tools", return_value={"traex": "Traex"}):
        handler.handle_workflow_orchestrator_select_tool(
            message_id="msg_1",
            chat_id="chat_1",
            project_id="proj_1",
            value=value,
        )

    assert mock_project._wf_selection_snapshot["pending_tool_name"] == "traex"
    assert mock_project._wf_selection_snapshot["model_page"] == 0
    handler._reply_workflow_error.assert_not_called()


def test_handle_orchestrator_select_tool_model_page_keeps_panel_expanded():
    handler, mock_project, _mock_engine = _build_orchestrator_handler_with_project()

    value = {
        "action": "workflow_orchestrator_select_tool",
        "tool_name": "traex",
        "provider": "workflow",
        "display_name": "Traex",
        "supports_model": True,
        "model_page": 2,
        "chat_id": "chat_1",
        "project_id": "proj_1",
        "engine_session_key": "sess_abc",
    }

    with patch("src.thread.get_current_sender_id", return_value="user_1"), \
            patch("src.workflow_engine.tool_registry.get_available_tools", return_value={"traex": "Traex"}):
        handler.handle_workflow_orchestrator_select_tool(
            message_id="msg_1",
            chat_id="chat_1",
            project_id="proj_1",
            value=value,
        )

    assert mock_project._wf_selection_snapshot["pending_tool_name"] == "traex"
    assert mock_project._wf_selection_snapshot["model_page"] == 2
    handler.update_card.assert_called()
    handler._reply_workflow_error.assert_not_called()


def test_handle_orchestrator_remove_and_clear():
    """调用 handle_workflow_orchestrator_remove 以及 handle_workflow_orchestrator_clear 都应该触发 update_card。

    使用新的 SelectionFlowController：预先写入一个 selection item 并将其 snapshot
    持久化到 project._wf_selection_snapshot，随后走 handler 路径验证 update_card 被调用。
    """
    handler, mock_project, mock_engine = _build_orchestrator_handler_with_project()

    # 构造一个选中的 item（使用新的 SelectionFlowController）
    from src.workflow_engine.selection_flow import SelectionFlowController

    ctrl = SelectionFlowController(step=1)
    selection_key = ctrl.add_or_update_selection(
        {"tool_name": "coco", "display_name": "Coco", "model_name": "gpt-4"},
        is_review=False,
    )
    # 把 selection 状态持久化到 project，这样 handler._get_selection_controller 能读到
    mock_project._wf_selection_snapshot = ctrl.snapshot()
    assert len(ctrl.orchestrator_selections) == 1, "测试前应有一个选中项"

    # --- 1) 调用 remove：根据 selection_key 移除
    remove_value = {
        "action": "workflow_orchestrator_remove",
        "selection_key": selection_key,
        "chat_id": "chat_1",
        "project_id": "proj_1",
        "engine_session_key": "sess_abc",
    }
    with patch("src.thread.get_current_sender_id", return_value="user_1"):
        handler.handle_workflow_orchestrator_remove(
            message_id="msg_remove",
            chat_id="chat_1",
            project_id="proj_1",
            value=remove_value,
        )
    handler.update_card.assert_called()
    handler._reply_workflow_error.assert_not_called()
    handler.update_card.reset_mock()

    # remove 后，snapshot 中 orchestrator_selections 应该为空
    restored = SelectionFlowController()
    restored.restore(mock_project._wf_selection_snapshot)
    assert len(restored.orchestrator_selections) == 0

    # --- 2) 再添加一个 item，随后 clear
    ctrl2 = SelectionFlowController(step=1)
    ctrl2.add_or_update_selection(
        {"tool_name": "claude", "display_name": "Claude", "model_name": "sonnet"},
        is_review=False,
    )
    mock_project._wf_selection_snapshot = ctrl2.snapshot()

    clear_value = {
        "action": "workflow_orchestrator_clear",
        "chat_id": "chat_1",
        "project_id": "proj_1",
        "engine_session_key": "sess_abc",
    }
    with patch("src.thread.get_current_sender_id", return_value="user_1"):
        handler.handle_workflow_orchestrator_clear(
            message_id="msg_clear",
            chat_id="chat_1",
            project_id="proj_1",
            value=clear_value,
        )
    handler.update_card.assert_called()
    handler._reply_workflow_error.assert_not_called()

    # clear 后，snapshot 中 orchestrator_selections 应该为空
    restored2 = SelectionFlowController()
    restored2.restore(mock_project._wf_selection_snapshot)
    assert len(restored2.orchestrator_selections) == 0


def test_handle_orchestrator_finish_writes_binding():
    """handle_workflow_orchestrator_finish 应将 selected_items 写入 pending.orchestrator_binding
    并将状态转至 AWAITING_TOOL_SELECT。

    使用新的 SelectionFlowController：预先写入一个 tool + model 选择，再调用 handler。
    """
    handler, mock_project, mock_engine = _build_orchestrator_handler_with_project()

    from src.workflow_engine.selection_flow import SelectionFlowController

    ctrl = SelectionFlowController(step=1)
    ctrl.add_or_update_selection(
        {"tool_name": "coco", "display_name": "Coco", "model_name": "gpt-4"},
        is_review=False,
    )
    mock_project._wf_selection_snapshot = ctrl.snapshot()

    # 调用 finish
    finish_value = {
        "action": "workflow_orchestrator_finish",
        "chat_id": "chat_1",
        "project_id": "proj_1",
        "engine_session_key": "sess_abc",
    }
    with patch("src.thread.get_current_sender_id", return_value="user_1"):
        handler.handle_workflow_orchestrator_finish(
            message_id="msg_finish",
            chat_id="chat_1",
            project_id="proj_1",
            value=finish_value,
        )

    # orchestrator_binding 应包含 tool_name=coco
    binding = mock_engine.project.pending.orchestrator_binding
    assert binding is not None, "orchestrator_binding 未写入"
    assert binding.tool_name == "coco", (
        f"orchestrator_binding.tool_name 期望 coco，实际 {binding.tool_name}"
    )
    # model_name 也应被保留
    assert binding.model_name == "gpt-4"
    # 向后兼容：orchestrator_agent 应该被同步设置
    assert mock_engine.project.pending.orchestrator_agent == "coco"
    # 状态应转移到 AWAITING_TOOL_SELECT
    assert mock_engine.project.status == WorkflowStatus.AWAITING_TOOL_SELECT
    # 不应报错
    handler._reply_workflow_error.assert_not_called()


def test_review_finish_schedules_script_generation_without_blocking_callback():
    """review_finish should enqueue script generation instead of running it inline."""
    handler, mock_project, mock_engine = _build_orchestrator_handler_with_project()
    mock_engine.project.status = WorkflowStatus.AWAITING_TOOL_SELECT
    mock_engine.project.pending.orchestrator_agent = "coco"

    from src.workflow_engine.selection_flow import SelectionFlowController

    ctrl = SelectionFlowController(step=2)
    ctrl.add_or_update_selection(
        {"tool_name": "claude", "display_name": "Claude", "model_name": "sonnet"},
        is_review=True,
    )
    mock_project._wf_selection_snapshot = ctrl.snapshot()

    handler._generate_and_show_confirm_card = MagicMock()
    handler._submit_engine_task = MagicMock()

    finish_value = {
        "action": "workflow_review_finish",
        "chat_id": "chat_1",
        "project_id": "proj_1",
        "engine_session_key": "sess_abc",
    }
    with patch("src.thread.get_current_sender_id", return_value="user_1"):
        handler.handle_workflow_review_finish(
            message_id="msg_review_finish",
            chat_id="chat_1",
            project_id="proj_1",
            value=finish_value,
        )

    handler._generate_and_show_confirm_card.assert_not_called()
    handler._submit_engine_task.assert_called_once()
    assert mock_engine.project.status == WorkflowStatus.GENERATING_SCRIPT
    scheduled_fn = handler._submit_engine_task.call_args.args[0]
    assert callable(scheduled_fn)

    scheduled_fn()
    handler._generate_and_show_confirm_card.assert_called_once_with(
        message_id="msg_review_finish",
        chat_id="chat_1",
        requirement="build a feature",
        project=mock_project,
        root_path="/tmp/test_proj",
        selected_tools=["coco", "claude"],
        expected_session_key="sess_abc",
    )
    assert mock_engine.project.pending.review_agents
    assert mock_engine.project.pending.selected_tools == ["coco", "claude"]
    handler._reply_workflow_error.assert_not_called()


def test_payload_filter_preserves_new_fields():
    """filter_workflow_button_value 应保留 tool_name/provider/supports_model/model_name/name/use_default_model/_option/selection_key。"""
    from src.card.events.payloads import filter_workflow_button_value

    payload = {
        "action": "workflow_orchestrator_select_tool",
        "tool_name": "coco",
        "provider": "workflow",
        "supports_model": True,
        "model_name": "gpt-4",
        "name": "gpt-4",
        "use_default_model": False,
        "_option": "default",
        "selection_key": "sel_xyz",
        "model_page": 1,
        "chat_id": "chat_1",
        "project_id": "proj_1",
        "engine_session_key": "sess_abc",
        # 非法字段（模拟被注入的伪造 key）
        "admin_override": True,
        "confirmed": "yes",
    }

    filtered = filter_workflow_button_value(payload)

    # 新增字段必须被保留
    for key in ["tool_name", "provider", "supports_model", "model_name",
                "name", "use_default_model", "_option", "selection_key",
                "model_page"]:
        assert key in filtered, f"字段 {key} 未被 filter_workflow_button_value 保留"

    # 合法的基本字段也要被保留
    for key in ["action", "chat_id", "project_id", "engine_session_key"]:
        assert key in filtered, f"基础字段 {key} 未被保留"

    # 非法字段必须被剔除
    assert "admin_override" not in filtered
    assert "confirmed" not in filtered

    # 值也要被正确保留
    assert filtered["tool_name"] == "coco"
    assert filtered["model_name"] == "gpt-4"
    assert filtered["selection_key"] == "sel_xyz"
    assert filtered["supports_model"] is True


def test_invalid_tool_name_rejected_in_select_model():
    """测试在 handle_workflow_orchestrator_select_model 中无效的 tool_name 被拒绝。"""
    handler, mock_project, mock_engine = _build_orchestrator_handler_with_project()

    # 添加 MagicMock 用于 _send_combined_selection_card
    handler._send_combined_selection_card = MagicMock()

    # 设置验证返回拒绝结果
    handler._validate_tools_against_registry = MagicMock(return_value=([], ["invalid_tool"]))

    value = {
        "action": "workflow_orchestrator_select_model",
        "tool_name": "invalid_tool",
        "display_name": "Invalid Tool",
        "model_name": "gpt-4",
        "chat_id": "chat_1",
        "project_id": "proj_1",
        "engine_session_key": "sess_abc",
    }

    with patch("src.thread.get_current_sender_id", return_value="user_1"):
        handler.handle_workflow_orchestrator_select_model(
            message_id="msg_1",
            chat_id="chat_1",
            project_id="proj_1",
            value=value,
        )

    # 应该调用错误回复，而不是更新卡片
    handler._reply_workflow_error.assert_called_once()
    handler._send_combined_selection_card.assert_not_called()


def test_valid_tool_name_accepted_in_select_model():
    """测试有效的 tool_name 在 handle_workflow_orchestrator_select_model 中被接受。"""
    handler, mock_project, mock_engine = _build_orchestrator_handler_with_project()

    # 添加 MagicMock 用于 _send_combined_selection_card
    handler._send_combined_selection_card = MagicMock()

    # 设置验证返回接受结果
    handler._validate_tools_against_registry = MagicMock(return_value=(["coco"], []))
    handler._get_workflow_models_for_tool = MagicMock(return_value=[
        {"name": "gpt-4"},
        {"name": "Claude 3 Opus"},
    ])

    value = {
        "action": "workflow_orchestrator_select_model",
        "tool_name": "coco",
        "display_name": "Coco",
        "model_name": "gpt-4",
        "chat_id": "chat_1",
        "project_id": "proj_1",
        "engine_session_key": "sess_abc",
    }

    with patch("src.thread.get_current_sender_id", return_value="user_1"):
        handler.handle_workflow_orchestrator_select_model(
            message_id="msg_1",
            chat_id="chat_1",
            project_id="proj_1",
            value=value,
        )

    # 应该调用更新卡片，而不是错误回复
    handler._send_combined_selection_card.assert_called_once()
    handler._reply_workflow_error.assert_not_called()


@pytest.mark.parametrize(
    ("tool_name", "available_models", "model_name"),
    [
        (
            "traex",
            [
                {
                    "name": "gpt-5.6-sol",
                    "display_name": "GPT-5.6-Sol",
                    "description": "GPT-5.6-Sol",
                    "is_default": True,
                    "selection_variants": [
                        {
                            "name": "gpt-5.6-sol/max/xhigh",
                            "profile": "max",
                            "effort": "xhigh",
                            "display_name": "GPT-5.6-Sol · max · xhigh",
                            "is_variant_default": False,
                        }
                    ],
                }
            ],
            "gpt-5.6-sol/max/xhigh",
        ),
        (
            "codex",
            [
                {
                    "name": "gpt-5.6-sol",
                    "display_name": "GPT-5.6-Sol",
                    "description": "GPT-5.6-Sol",
                    "is_default": True,
                    "reasoning_efforts": ["high", "xhigh"],
                    "adapted_reasoning_effort": "high",
                }
            ],
            "gpt-5.6-sol/xhigh",
        ),
    ],
    ids=("traex-explicit-variant", "codex-reasoning-effort"),
)
def test_generated_composite_model_name_accepted(
    tool_name,
    available_models,
    model_name,
):
    handler, _mock_project, _mock_engine = _build_orchestrator_handler_with_project()
    handler._send_combined_selection_card = MagicMock()
    handler._validate_tools_against_registry = MagicMock(
        return_value=([tool_name], [])
    )
    handler._get_workflow_models_for_tool = MagicMock(
        return_value=available_models
    )

    value = {
        "action": "workflow_orchestrator_select_model",
        "tool_name": tool_name,
        "display_name": tool_name.title(),
        "model_name": model_name,
        "chat_id": "chat_1",
        "project_id": "proj_1",
        "engine_session_key": "sess_abc",
    }

    with patch("src.thread.get_current_sender_id", return_value="user_1"):
        handler.handle_workflow_orchestrator_select_model(
            message_id="msg_1",
            chat_id="chat_1",
            project_id="proj_1",
            value=value,
        )

    handler._send_combined_selection_card.assert_called_once()
    handler._reply_workflow_error.assert_not_called()


@pytest.mark.parametrize(
    "model_name",
    ["invalid_model", "gpt-4/max/evil"],
    ids=("unknown-family", "unadvertised-variant"),
)
def test_invalid_model_name_rejected(model_name):
    """测试无效的 model_name 被拒绝。"""
    handler, mock_project, mock_engine = _build_orchestrator_handler_with_project()

    # 添加 MagicMock 用于 _send_combined_selection_card
    handler._send_combined_selection_card = MagicMock()

    # 设置验证返回接受工具，但模型不在列表中
    handler._validate_tools_against_registry = MagicMock(return_value=(["coco"], []))
    handler._get_workflow_models_for_tool = MagicMock(return_value=[
        {"name": "gpt-4"},
        {"name": "Claude 3 Opus"},
    ])

    value = {
        "action": "workflow_orchestrator_select_model",
        "tool_name": "coco",
        "display_name": "Coco",
        "model_name": model_name,
        "chat_id": "chat_1",
        "project_id": "proj_1",
        "engine_session_key": "sess_abc",
    }

    with patch("src.thread.get_current_sender_id", return_value="user_1"):
        handler.handle_workflow_orchestrator_select_model(
            message_id="msg_1",
            chat_id="chat_1",
            project_id="proj_1",
            value=value,
        )

    # 应该调用错误回复
    handler._reply_workflow_error.assert_called_once()
    handler._send_combined_selection_card.assert_not_called()
