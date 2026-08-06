"""Tests for handler modules extracted from ws_client.py.

Each handler is tested with a fully-mocked HandlerContext so that no real
Feishu API calls or sessions are required.
"""

import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.feishu.handler_context import HandlerContext
from src.feishu.handlers.base import BaseHandler
from src.feishu.handlers.deep import DeepHandler
from src.feishu.handlers.diagnostics import DiagnosticsHandler
from src.feishu.handlers.engine_base import BaseEngineHandler
from src.feishu.handlers.programming import (
    ClaudeModeHandler,
    CocoModeHandler,
    ProgrammingModeHandler,
)
from src.feishu.handlers.project import ProjectHandler
from src.feishu.handlers.spec import SpecHandler
from src.feishu.handlers.system import SystemHandler
from src.feishu.handlers.workflow import WorkflowHandler
from src.feishu.slash_command_parser import SlashCommandParser
from src.mode.manager import InteractionMode
from src.project.context import ProjectContext, SessionSnapshot

# ======================================================================
# Shared fixture: mock HandlerContext
# ======================================================================


def _make_handler_context(**overrides) -> HandlerContext:
    """Build a HandlerContext with all dependencies mocked."""
    settings = MagicMock()
    settings.thread_programming_enabled = False
    settings.project_allowed_roots = []
    ctx = HandlerContext(
        settings=settings,
        api_client_factory=MagicMock(),
        message_callback=MagicMock(),
        coco_manager=MagicMock(),
        claude_manager=MagicMock(),
        aiden_manager=MagicMock(),
        codex_manager=MagicMock(),
        gemini_manager=MagicMock(),
        traex_manager=MagicMock(),
        tui2acp_manager=MagicMock(),
        intent_recognizer=MagicMock(),
        scheduler=MagicMock(),
        project_manager=MagicMock(),
        message_mapper=MagicMock(),
        message_linker=MagicMock(),
        mode_manager=MagicMock(),
        context_manager=MagicMock(),
        deep_engine_manager=MagicMock(),
        progress_reporter=MagicMock(),
        spec_engine_manager=MagicMock(),
        spec_reporter=MagicMock(),
        slock_engine_manager=MagicMock(),
        thread_manager=MagicMock(),
        image_handler_factory=MagicMock(),
        working_dirs={},
        working_dir_lock=threading.Lock(),
        pending_image_keys={},
        pending_image_lock=threading.Lock(),
        enable_streaming=False,
        managers={},
        handlers={},
    )
    for k, v in overrides.items():
        setattr(ctx, k, v)
    return ctx


def _set_all_programming_mode_flags(ctx, value: bool) -> None:
    ctx.mode_manager.is_coco_mode.return_value = value
    ctx.mode_manager.is_claude_mode.return_value = value
    ctx.mode_manager.is_aiden_mode.return_value = value
    ctx.mode_manager.is_codex_mode.return_value = value
    ctx.mode_manager.is_gemini_mode.return_value = value


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


# ======================================================================
# BaseHandler tests
# ======================================================================


class TestBaseHandler:
    def _make(self, **ctx_overrides):
        ctx = _make_handler_context(**ctx_overrides)
        return BaseHandler(ctx), ctx

    def test_get_working_dir_default(self):
        h, ctx = self._make()
        import os

        assert h.get_working_dir("chat1") == os.getcwd()

    def test_set_and_get_working_dir(self, tmp_path):
        h, ctx = self._make()
        success, result = h.set_working_dir("chat1", str(tmp_path))
        assert success is True
        assert result == str(tmp_path)
        assert h.get_working_dir("chat1") == str(tmp_path)

    def test_get_engine_name_claude(self):
        h, ctx = self._make()
        ctx.mode_manager.get_mode.return_value = InteractionMode.CLAUDE
        assert h.get_engine_name("chat1") == "Claude"

    def test_mode_to_context_source(self):
        from src.project import ContextSourceMode

        assert BaseHandler.mode_to_context_source(InteractionMode.SMART) == ContextSourceMode.SMART
        assert BaseHandler.mode_to_context_source(InteractionMode.COCO) == ContextSourceMode.COCO
        assert BaseHandler.mode_to_context_source(InteractionMode.CLAUDE) == ContextSourceMode.CLAUDE

    def test_normalize_interactive_card_content_removes_schema2_root_elements(self):
        card_json = json.dumps(
            {
                "schema": "2.0",
                "header": {"title": {"tag": "plain_text", "content": "title"}},
                "elements": [{"tag": "markdown", "content": "hello"}],
                "body": {"elements": [{"tag": "markdown", "content": "hello"}]},
            },
            ensure_ascii=False,
        )

        normalized = BaseHandler._normalize_interactive_card_content(card_json)
        card = json.loads(normalized)

        assert "elements" not in card
        assert card["body"]["elements"][0]["content"] == "hello"

    def test_inject_bridge_context_no_project(self):
        h, _ = self._make()
        assert h.inject_bridge_context("hello", None) == "hello"

    def test_inject_bridge_context_no_context(self):
        h, ctx = self._make()
        ctx.context_manager.store.get.return_value = None
        project = SimpleNamespace(project_id="p1")
        assert h.inject_bridge_context("hello", project) == "hello"

    def test_inject_bridge_context_with_bridge(self):
        h, ctx = self._make()
        mock_ctx = MagicMock()
        bridge = MagicMock()
        bridge.to_injection_prompt.return_value = "[bridge context]"
        mock_ctx.consume_bridge_summary.return_value = bridge
        ctx.context_manager.store.get.return_value = mock_ctx
        project = SimpleNamespace(project_id="p1", project_name="test")
        result = h.inject_bridge_context("hello", project)
        assert "[bridge context]" in result
        assert "hello" in result


class TestBaseEngineHandlerTemplateEntrypoints:
    def test_engine_base_exposes_lock_callback_and_status_template_facade(self):
        class DummyEngineHandler(BaseEngineHandler):
            def _get_engine_manager(self):
                return MagicMock()

            def _get_engine_name_prefix(self) -> str:
                return "Dummy"

            def _get_task_type(self) -> str:
                return "dummy_engine"

            def _show_status(self, message_id, chat_id, project=None):
                self.status_calls.append((message_id, chat_id, project))

            def _create_callbacks(self, message_id, chat_id, project, engine_name, root_path):
                return {"message_id": message_id, "chat_id": chat_id, "engine_name": engine_name, "root_path": root_path}

        ctx = _make_handler_context()
        h = DummyEngineHandler(ctx)
        h.status_calls = []
        h.lock_helper.handle_lock_conflict = MagicMock(side_effect=lambda body, *_args: body())
        project = SimpleNamespace(root_path="/repo", project_id="p1")
        ran = []

        h._run_engine_with_conflict_card(lambda: ran.append(True), project, "chat-1", "msg-1", "/dummy run")
        callbacks = h._build_engine_callbacks("msg-1", "chat-1", project, "Engine", "/repo")
        h._show_engine_status("msg-1", "chat-1", project)

        assert ran == [True]
        h.lock_helper.handle_lock_conflict.assert_called_once()
        assert h.lock_helper.handle_lock_conflict.call_args.args[1:] == ("/repo", "chat-1", "msg-1", "/dummy run")
        assert callbacks["engine_name"] == "Engine"
        assert h.status_calls == [("msg-1", "chat-1", project)]


class TestTopicEngineProjectResolution:
    @pytest.mark.parametrize("handler_type", [DeepHandler, SpecHandler, WorkflowHandler])
    def test_engine_handler_auto_creates_missing_project(self, handler_type):
        ctx = _make_handler_context()
        project = SimpleNamespace(project_id="p1", project_name="GhostAP", root_path="/repo")
        ctx.project_manager.get_or_create_project_for_path.return_value = (project, True)
        handler = handler_type(ctx)
        handler.get_working_dir = MagicMock(return_value="/repo")
        handler.reply_error = MagicMock()

        resolved = handler._ensure_project("m1", "c1", None)

        assert resolved is project
        ctx.project_manager.get_or_create_project_for_path.assert_called_once_with("/repo", "c1")
        handler.reply_error.assert_not_called()

    @pytest.mark.parametrize("handler_type", [DeepHandler, SpecHandler, WorkflowHandler])
    def test_engine_handler_reports_project_creation_failure(self, handler_type):
        ctx = _make_handler_context()
        ctx.project_manager.get_or_create_project_for_path.side_effect = RuntimeError(
            "workspace unavailable"
        )
        handler = handler_type(ctx)
        handler.get_working_dir = MagicMock(return_value="/repo")
        handler.reply_error = MagicMock()

        resolved = handler._ensure_project("m1", "c1", None)

        assert resolved is None
        handler.reply_error.assert_called_once()
        assert handler.reply_error.call_args.kwargs["title"] == "创建项目失败"

class TestHandlerContextDependencyView:
    def test_dependency_view_exposes_narrow_core_services_without_removing_fields(self):
        ctx = _make_handler_context()

        view = ctx.dependency_view()

        assert view.settings is ctx.settings
        assert view.scheduler is ctx.scheduler
        assert view.project_manager is ctx.project_manager
        assert view.message_linker is ctx.message_linker
        assert ctx.handlers == {}


# ======================================================================
# SystemHandler tests
# ======================================================================


class TestSystemHandlerPredicates:
    def test_exit_commands(self):
        assert SystemHandler.is_exit_command("/exit") is True
        assert SystemHandler.is_exit_command("/quit") is True
        assert SystemHandler.is_exit_command("/end_coco") is True
        assert SystemHandler.is_exit_command("/exit_claude") is True
        assert SystemHandler.is_exit_command("退出模式") is True
        assert SystemHandler.is_exit_command("退出编程模式") is True
        assert SystemHandler.is_exit_command("hello") is False
        assert SystemHandler.is_exit_command("/help") is False

    def test_deep_commands(self):
        assert SystemHandler.is_deep_command("/deep do stuff") is True
        assert SystemHandler.is_deep_command("/deep_status") is True
        assert SystemHandler.is_deep_command("/stop_deep") is True
        assert SystemHandler.is_deep_command("/help") is False
        assert SystemHandler.is_deep_command("deep") is False

    @pytest.mark.parametrize(
        ("text", "predicate", "expected"),
        [
            ("/deepfake goal", SystemHandler.is_deep_command, False),
            ("/deep\tgoal", SystemHandler.is_deep_command, True),
            ("/spec\tgoal", SystemHandler.is_spec_command, True),
            ("/wf\ngoal", SystemHandler.is_workflow_command, True),
        ],
    )
    def test_engine_command_predicates_follow_slash_parser_tokenization(
        self, text, predicate, expected
    ):
        assert predicate(text) is expected

    def test_interceptable_commands(self):
        m = SlashCommandParser.parse
        assert SystemHandler.is_interceptable_command_match(m("/help")) is True
        assert SystemHandler.is_interceptable_command_match(m("/帮助")) is True
        assert SystemHandler.is_interceptable_command_match(m("/codex")) is True
        assert SystemHandler.is_interceptable_command_match(m("/coco")) is True
        assert SystemHandler.is_interceptable_command_match(m("/coco_status")) is True
        assert SystemHandler.is_interceptable_command_match(m("/tools")) is True
        assert SystemHandler.is_interceptable_command_match(m("/btw remember this")) is True
        assert SystemHandler.is_interceptable_command_match(m("/coco_info")) is True
        assert SystemHandler.is_interceptable_command_match(m("/claude_info")) is True
        assert SystemHandler.is_interceptable_command_match(m("/gemini_info")) is True
        assert SystemHandler.is_interceptable_command_match(m("/projects")) is True
        assert SystemHandler.is_interceptable_command_match(m("/status")) is True
        assert SystemHandler.is_interceptable_command_match(m("/switch foo")) is True
        assert SystemHandler.is_interceptable_command_match(m("/new myproject /tmp")) is True
        assert SystemHandler.is_interceptable_command_match(m("/new-chat hermes")) is True
        assert SystemHandler.is_interceptable_command_match(m("/tasks")) is True
        assert SystemHandler.is_interceptable_command_match(m("/diff")) is True
        assert SystemHandler.is_interceptable_command_match(m("/trace")) is True
        assert SystemHandler.is_interceptable_command_match(m("/deep stuff")) is False
        assert SystemHandler.is_interceptable_command_match(m("hello")) is False


class TestSystemHandlerRouting:
    def _make(self):
        ctx = _make_handler_context()
        handler = SystemHandler(ctx)
        ctx.handlers.update({
            "coco": MagicMock(),
            "claude": MagicMock(),
            "project": MagicMock(),
            "deep": MagicMock(),
            "diagnostics": MagicMock(),
        })
        # Keep attributes for test assertions
        handler.coco_handler = ctx.handlers["coco"]
        handler.claude_handler = ctx.handlers["claude"]
        handler.project_handler = ctx.handlers["project"]
        handler.deep_handler = ctx.handlers["deep"]
        handler.diagnostics_handler = ctx.handlers["diagnostics"]
        return handler

    def test_route_help(self):
        h = self._make()
        h.show_full_help = MagicMock()
        h.handle_intercepted_command("m1", "c1", "/help", None, command_match=SlashCommandParser.parse("/help"))
        h.show_full_help.assert_called_once_with("m1", "c1", None)

    def test_subhandler_facade_exposes_minimal_responsibility_entries(self):
        h = self._make()

        assert h.help_commands.show_full_help.__func__ is h.show_full_help.__func__
        assert h.shell_commands.submit_shell_command.__func__ is h.submit_shell_command.__func__
        assert h.acp_commands.handle_acp_command.__func__ is h.handle_acp_command.__func__
        assert h.lock_commands.handle_force_release_repo_lock.__func__ is h.handle_force_release_repo_lock.__func__

    def test_route_coco_info(self):
        h = self._make()
        h.handle_intercepted_command("m1", "c1", "/coco_info", None, command_match=SlashCommandParser.parse("/coco_info"))
        h.coco_handler.show_info.assert_called_once_with("m1", "c1", None)

    def test_route_claude_info(self):
        h = self._make()
        h.handle_intercepted_command("m1", "c1", "/claude_info", None, command_match=SlashCommandParser.parse("/claude_info"))
        h.claude_handler.show_info.assert_called_once_with("m1", "c1", None)

    def test_route_projects(self):
        h = self._make()
        h.handle_intercepted_command("m1", "c1", "/projects", None, command_match=SlashCommandParser.parse("/projects"))
        h.project_handler.show_project_board.assert_called_once_with("m1", "c1")

    def test_route_new_chat_project(self):
        h = self._make()
        h.handle_intercepted_command(
            "m1",
            "c1",
            "/new-chat hermes",
            None,
            command_match=SlashCommandParser.parse("/new-chat hermes"),
        )
        h.project_handler.handle_new_chat_project.assert_called_once_with(
            "m1",
            "c1",
            {"name": "hermes"},
        )

    def test_route_new_chat_project_preserves_path_spaces(self):
        h = self._make()
        h.handle_intercepted_command(
            "m1",
            "c1",
            "/new-chat hermes dev /tmp/a b",
            None,
            command_match=SlashCommandParser.parse("/new-chat hermes dev /tmp/a b"),
        )
        h.project_handler.handle_new_chat_project.assert_called_once_with(
            "m1",
            "c1",
            {"name": "hermes", "suffix": "dev", "path": "/tmp/a b"},
        )

    def test_route_tasks(self):
        h = self._make()
        h.handle_intercepted_command("m1", "c1", "/tasks", None, command_match=SlashCommandParser.parse("/tasks"))
        h.diagnostics_handler.show_task_board.assert_called_once_with("m1", "c1", "/tasks", None)

    def test_route_diff(self):
        h = self._make()
        h.handle_intercepted_command("m1", "c1", "/diff", None, command_match=SlashCommandParser.parse("/diff"))
        h.diagnostics_handler.show_context_diff.assert_called_once_with("m1", "c1", "/diff", None)

    def test_route_trace(self):
        h = self._make()
        h.handle_intercepted_command("m1", "c1", "/trace msg123", None, command_match=SlashCommandParser.parse("/trace msg123"))
        h.diagnostics_handler.show_message_trace.assert_called_once_with("m1", "c1", "/trace msg123", None)

    def test_intercepted_unknown_slash_replies_without_help_card(self):
        h = self._make()
        h.reply_text = MagicMock()
        h.show_full_help = MagicMock()

        h.handle_intercepted_command(
            "m1",
            "c1",
            "/unknown_command",
            None,
            command_match=SlashCommandParser.parse("/unknown_command"),
        )

        h.reply_text.assert_called_once()
        assert "未知命令" in h.reply_text.call_args.args[1]
        h.show_full_help.assert_not_called()

    @pytest.mark.parametrize(
        "text",
        [
            "/goal refactor auth",
            "/goals",
            "/runs",
            "/approve ap_1",
            "/run goal_1",
            "/approvals",
            "/decisions",
        ],
    )
    def test_retired_autonomous_manager_commands_never_claim_success(self, text):
        h = self._make()
        h.reply_text = MagicMock()
        h.reply_card = MagicMock()

        h.handle_intercepted_command(
            "m1",
            "c1",
            text,
            None,
            command_match=SlashCommandParser.parse(text),
        )

        h.reply_text.assert_called_once()
        response = h.reply_text.call_args.args[1]
        assert "已退役" in response
        assert "未执行" in response
        assert "Slock" in response
        assert "/goal" not in response
        assert "/task" in response
        if text.startswith("/goal "):
            assert "/task refactor auth" in response
        h.reply_card.assert_not_called()

    @patch("src.thread.get_current_thread_id", return_value=None)
    def test_btw_forwards_to_active_programming_handler(self, _):
        h = self._make()
        project = MagicMock()
        project.project_id = "p1"
        codex = MagicMock()
        h.ctx.handlers["codex"] = codex
        h.mode_manager.get_mode.return_value = InteractionMode.CODEX
        h.mode_manager.is_programming_mode.return_value = True

        h.handle_intercepted_command(
            "m1",
            "c1",
            "/btw add a constraint",
            project,
            command_match=SlashCommandParser.parse("/btw add a constraint"),
        )

        codex.handle_message.assert_called_once_with("m1", "c1", "/btw add a constraint", project)

    @patch("src.thread.get_current_thread_id", return_value=None)
    def test_btw_without_active_programming_replies_usage(self, _):
        h = self._make()
        h.reply_text = MagicMock()
        h.mode_manager.get_mode.return_value = InteractionMode.SMART
        h.mode_manager.is_programming_mode.return_value = False

        h.handle_intercepted_command(
            "m1",
            "c1",
            "/btw add a constraint",
            None,
            command_match=SlashCommandParser.parse("/btw add a constraint"),
        )

        h.reply_text.assert_called_once()
        assert "没有活跃编程会话" in h.reply_text.call_args.args[1]

    def test_route_new_project(self):
        h = self._make()
        h.handle_intercepted_command(
            "m1",
            "c1",
            "/new myapp /tmp/myapp",
            None,
            command_match=SlashCommandParser.parse("/new myapp /tmp/myapp"),
        )
        h.project_handler.create_project.assert_called_once_with("m1", "c1", "myapp", "/tmp/myapp")

    def test_route_close_project(self):
        h = self._make()
        h.handle_intercepted_command("m1", "c1", "/close myapp", None, command_match=SlashCommandParser.parse("/close myapp"))
        h.project_handler.close_project.assert_called_once_with("m1", "c1", "myapp")

    def test_exit_current_mode_coco(self):
        ctx = _make_handler_context()
        ctx.mode_manager.get_mode.return_value = InteractionMode.COCO
        h = SystemHandler(ctx)
        ctx.handlers["coco"] = MagicMock()
        h.coco_handler = ctx.handlers["coco"]
        h.exit_current_mode("m1", "c1", None)
        h.coco_handler.exit_mode.assert_called_once_with("m1", "c1", None)









    def test_show_tools_list_uses_cached_availability_api(self):
        h = self._make()
        with patch("src.feishu.handlers.system.tool_registry") as mock_registry:
            mock_registry.get_availability.return_value = True
            h.reply_interactive_card = MagicMock()
            h.show_tools_list("m1", "c1", None)
            # 6 tools in metadata
            assert mock_registry.get_availability.call_count == 6
            h.reply_interactive_card.assert_called_once()


# ======================================================================
# ProgrammingModeHandler (CocoModeHandler / ClaudeModeHandler) tests
# ======================================================================


class TestCocoModeHandler:
    def _make(self, **ctx_overrides):
        ctx = _make_handler_context(**ctx_overrides)
        h = CocoModeHandler(ctx)
        ctx.handlers.update({
            "claude": MagicMock(),
            "aiden": MagicMock(),
            "codex": MagicMock(),
            "gemini": MagicMock(),
        })
        # Keep attributes for test assertions
        h._opposite_handler = ctx.handlers["claude"]
        h._claude_handler = h._opposite_handler
        h._aiden_handler = ctx.handlers["aiden"]
        h._codex_handler = ctx.handlers["codex"]
        h._gemini_handler = ctx.handlers["gemini"]
        return h, ctx

    def test_mode_attributes(self):
        h, _ = self._make()
        assert h.mode_name == "Coco"
        assert h.mode_emoji == "🤖"
        assert h.is_coco is True

    def test_session_manager(self):
        h, ctx = self._make()
        assert h._get_session_manager() is ctx.coco_manager

    def test_is_in_this_mode(self):
        h, ctx = self._make()
        ctx.mode_manager.get_mode.return_value = InteractionMode.COCO
        assert h._is_in_this_mode("c1") is True

    def test_is_in_opposite_mode(self):
        h, ctx = self._make()
        _set_all_programming_mode_flags(ctx, False)
        ctx.mode_manager.is_claude_mode.return_value = True
        assert h._is_in_opposite_mode("c1") is True

    def test_is_in_opposite_mode_checks_all_other_programming_modes(self):
        h, ctx = self._make()
        _set_all_programming_mode_flags(ctx, False)
        ctx.mode_manager.is_aiden_mode.return_value = True
        assert h._is_in_opposite_mode("c1") is True

    def test_enter_mode_on_manager(self):
        h, ctx = self._make()
        h._enter_mode_on_manager("c1")
        ctx.mode_manager.enter_programming_mode.assert_called_once_with("c1", InteractionMode.COCO, project_id=None)

    def test_enter_mode_on_manager_with_project(self):
        h, ctx = self._make()
        h._enter_mode_on_manager("c1", project_id="p1")
        ctx.mode_manager.enter_programming_mode.assert_called_once_with("c1", InteractionMode.COCO, project_id="p1")

    def test_get_interaction_mode(self):
        h, _ = self._make()
        assert h._get_interaction_mode() == InteractionMode.COCO

    def test_get_snapshot(self):
        h, _ = self._make()
        project = SimpleNamespace(coco_session_snapshot="snap", claude_session_snapshot=None)
        assert h._get_snapshot(project) == "snap"

    def test_set_mode_on_project_activate(self):
        h, _ = self._make()
        project = MagicMock()
        h._set_mode_on_project(project, True, "sid", 5)
        project.set_programming_mode.assert_called_once_with("coco", True, "sid", 5)
        assert project.acp_tool_name == "coco"

    def test_set_mode_on_project_deactivate(self):
        h, _ = self._make()
        project = MagicMock()
        h._set_mode_on_project(project, False)
        project.set_programming_mode.assert_called_once_with("coco", False)

    def test_update_snapshot_on_project(self):
        h, _ = self._make()
        project = MagicMock()
        h._update_snapshot_on_project(project, "hello", 3)
        project.update_programming_snapshot.assert_called_once_with("coco", "hello", 3, "")

    def test_clear_snapshot(self):
        h, _ = self._make()
        project = SimpleNamespace(coco_session_snapshot="snap")
        h._clear_snapshot_on_project(project)
        assert project.coco_session_snapshot is None

    def test_base_model_name_override_uses_project_model_when_tool_matches(self):
        h, _ = self._make()
        h.current_model = "handler-model"
        project = SimpleNamespace(acp_tool_name="coco", acp_model_name="project-model")

        assert h._get_model_name_override(project) == "project-model"

    def test_base_model_name_override_falls_back_to_current_model(self):
        h, _ = self._make()
        h.current_model = "handler-model"
        project = SimpleNamespace(acp_tool_name="aiden", acp_model_name="project-model")

        assert h._get_model_name_override(project) == "handler-model"

    def test_register_thread_context_preserves_selected_acp_tool_and_model(self):
        h, _ = self._make()
        project = SimpleNamespace(
            project_id="p1",
            acp_tool_name="coco",
            acp_model_name="selected-model",
        )
        session = SimpleNamespace(session_id="sid")
        thread_manager = MagicMock()

        with patch("src.thread.get_thread_manager", return_value=thread_manager):
            h._register_thread_context("root1", "chat1", project, session)

        thread_manager.register.assert_called_once()
        kwargs = thread_manager.register.call_args.kwargs
        assert kwargs["tool_name"] == "coco"
        assert kwargs["model_name"] == "selected-model"

    def test_exit_opposite_mode(self):
        h, _ = self._make()
        h.mode_manager.is_claude_mode.return_value = True
        h.mode_manager.is_aiden_mode.return_value = True
        h.mode_manager.is_codex_mode.return_value = False
        h.mode_manager.is_gemini_mode.return_value = False
        h._exit_opposite_mode("m1", "c1", project=None)
        h._opposite_handler.exit_mode.assert_called_once_with("m1", "c1", project=None, silent=False)
        h._aiden_handler.exit_mode.assert_called_once_with("m1", "c1", project=None, silent=False)
        h._codex_handler.exit_mode.assert_not_called()
        h._gemini_handler.exit_mode.assert_not_called()


class TestClaudeModeHandler:
    def _make(self, **ctx_overrides):
        ctx = _make_handler_context(**ctx_overrides)
        h = ClaudeModeHandler(ctx)
        ctx.handlers.update({
            "coco": MagicMock(),
            "aiden": MagicMock(),
            "codex": MagicMock(),
            "gemini": MagicMock(),
        })
        # Keep attributes for test assertions
        h._opposite_handler = ctx.handlers["coco"]
        h._coco_handler = h._opposite_handler
        h._aiden_handler = ctx.handlers["aiden"]
        h._codex_handler = ctx.handlers["codex"]
        h._gemini_handler = ctx.handlers["gemini"]
        return h, ctx

    def test_mode_attributes(self):
        h, _ = self._make()
        assert h.mode_name == "Claude"
        assert h.mode_emoji == "🔮"
        assert h.is_coco is False

    def test_session_manager(self):
        h, ctx = self._make()
        assert h._get_session_manager() is ctx.claude_manager

    def test_get_interaction_mode(self):
        h, _ = self._make()
        assert h._get_interaction_mode() == InteractionMode.CLAUDE

    def test_exit_opposite_mode(self):
        h, _ = self._make()
        h.mode_manager.is_coco_mode.return_value = True
        h.mode_manager.is_aiden_mode.return_value = True
        h.mode_manager.is_codex_mode.return_value = False
        h.mode_manager.is_gemini_mode.return_value = True
        h._exit_opposite_mode("m1", "c1", project=None)
        h._opposite_handler.exit_mode.assert_called_once_with("m1", "c1", project=None, silent=False)
        h._aiden_handler.exit_mode.assert_called_once_with("m1", "c1", project=None, silent=False)
        h._codex_handler.exit_mode.assert_not_called()
        h._gemini_handler.exit_mode.assert_called_once_with("m1", "c1", project=None, silent=False)





class TestProgrammingModeEnterExit:
    """Integration-level tests for enter_mode / exit_mode template methods."""

    def _make_coco(self):
        ctx = _make_handler_context()
        ctx.mode_manager.is_coco_mode.return_value = False
        ctx.mode_manager.is_claude_mode.return_value = False
        ctx.mode_manager.is_aiden_mode.return_value = False
        ctx.mode_manager.is_codex_mode.return_value = False
        ctx.mode_manager.is_gemini_mode.return_value = False
        ctx.mode_manager.get_mode.return_value = InteractionMode.SMART
        ctx.project_manager.validate_project_path.return_value = (True, "ok")
        ctx.project_manager.get_or_create_project_for_path.return_value = (None, False)

        mock_session = MagicMock()
        mock_session.session_id = "sid1"
        mock_session.is_resumed = False
        ctx.coco_manager.start_session.return_value = mock_session
        ctx.coco_manager.get_session.return_value = mock_session
        ctx.coco_manager.end_session.return_value = True

        h = CocoModeHandler(ctx)
        ctx.handlers.update({
            "claude": MagicMock(),
            "aiden": MagicMock(),
            "codex": MagicMock(),
            "gemini": MagicMock(),
        })
        # Keep attributes for test assertions
        h._opposite_handler = ctx.handlers["claude"]
        h._claude_handler = h._opposite_handler
        h._aiden_handler = ctx.handlers["aiden"]
        h._codex_handler = ctx.handlers["codex"]
        h._gemini_handler = ctx.handlers["gemini"]
        # Mock reply to avoid real API calls
        h.reply_text = MagicMock()
        h.reply_text = MagicMock()
        h.reply_card = MagicMock()
        h.reply_error = MagicMock()
        h.reply_card = MagicMock(return_value="reply_1")
        h.add_reaction = MagicMock()
        h.record_mode_transition = MagicMock()
        h.register_message_project = MagicMock()
        return h, ctx

    def test_enter_mode_already_in_mode(self):
        h, ctx = self._make_coco()
        ctx.mode_manager.get_mode.return_value = InteractionMode.COCO
        ctx.coco_manager.get_session_info.return_value = "session info"
        h.enter_mode("m1", "c1")
        h.reply_text.assert_called_once()
        assert "已经在" in str(h.reply_text.call_args)

    def test_retirement_failure_tombstones_matching_project_snapshot(self):
        handler = ProgrammingModeHandler.__new__(ProgrammingModeHandler)
        handler.mode_key = "coco"
        manager = MagicMock()
        manager.retire_session.side_effect = RuntimeError(
            "retirement lock unavailable"
        )
        handler._get_session_manager = MagicMock(return_value=manager)
        project = ProjectContext(
            project_id="project-1",
            project_name="project",
            root_path="/tmp",
        )
        project.set_programming_mode("coco", True, "retiring-session", 1)
        active = MagicMock()
        active.session_id = "retiring-session"

        with pytest.raises(RuntimeError, match="retirement lock unavailable"):
            handler._retire_finalization_session(
                chat_id="chat-1",
                project=project,
                thread_id=None,
                active_session=active,
            )

        assert project.coco_session_snapshot is None

    def test_late_retirement_preserves_concurrent_replacement_snapshot(self):
        handler = ProgrammingModeHandler.__new__(ProgrammingModeHandler)
        handler.mode_key = "coco"
        manager = MagicMock()
        manager.retire_session.return_value = None
        handler._get_session_manager = MagicMock(return_value=manager)
        project = ProjectContext(
            project_id="project-1",
            project_name="project",
            root_path="/tmp",
        )
        project.set_programming_mode("coco", True, "replacement-session", 1)
        old = MagicMock()
        old.session_id = "retiring-session"

        handler._retire_finalization_session(
            chat_id="chat-1",
            project=project,
            thread_id=None,
            active_session=old,
        )

        assert project.coco_session_snapshot is not None
        assert project.coco_session_snapshot.session_id == "replacement-session"

    def test_dead_session_replacement_never_reuses_concurrent_healthy_session(
        self,
    ):
        from src.acp.manager import SessionReplacementResult

        handler = ProgrammingModeHandler.__new__(ProgrammingModeHandler)
        handler.mode_key = "coco"
        handler._clear_snapshot_for_session = MagicMock()
        handler._get_agent_type_override = MagicMock(return_value=None)
        handler._get_model_name_override = MagicMock(return_value=None)
        handler._register_thread_context = MagicMock()
        concurrent = MagicMock()
        concurrent.session_id = "new-user-session"
        manager = MagicMock()
        manager.replace_session.return_value = SessionReplacementResult(
            session=concurrent,
            created=False,
        )
        handler._get_session_manager = MagicMock(return_value=manager)
        timed_out = MagicMock()
        timed_out.session_id = "old-task-session"

        with pytest.raises(RuntimeError, match="并发新会话"):
            handler._replace_timed_out_session(
                chat_id="chat-1",
                project=None,
                cwd="/tmp",
                thread_id=None,
                timed_out_session=timed_out,
                startup_budget_s=30,
            )

        handler._register_thread_context.assert_not_called()
        assert concurrent._force_dead is not True

    def test_dead_session_replacement_rejects_already_removed_session(self):
        from src.acp.manager import SessionReplacementResult

        handler = ProgrammingModeHandler.__new__(ProgrammingModeHandler)
        handler.mode_key = "coco"
        handler._clear_snapshot_for_session = MagicMock()
        handler._get_agent_type_override = MagicMock(return_value=None)
        handler._get_model_name_override = MagicMock(return_value=None)
        handler._register_thread_context = MagicMock()
        manager = MagicMock()
        manager.replace_session.return_value = SessionReplacementResult(
            session=None,
            created=False,
        )
        handler._get_session_manager = MagicMock(return_value=manager)
        timed_out = MagicMock()
        timed_out.session_id = "already-removed-session"

        with pytest.raises(RuntimeError, match="原会话已被移除"):
            handler._replace_timed_out_session(
                chat_id="chat-1",
                project=None,
                cwd="/tmp",
                thread_id=None,
                timed_out_session=timed_out,
                startup_budget_s=30,
            )

        handler._register_thread_context.assert_not_called()

    def test_retirement_persists_matching_snapshot_tombstone(self):
        handler = ProgrammingModeHandler.__new__(ProgrammingModeHandler)
        handler.mode_key = "coco"
        manager = MagicMock()
        manager.retire_session.return_value = None
        project_manager = MagicMock()
        handler.ctx = SimpleNamespace(project_manager=project_manager)
        handler._get_session_manager = MagicMock(return_value=manager)
        project = ProjectContext(
            project_id="project-1",
            project_name="project",
            root_path="/tmp",
        )
        project.set_programming_mode("coco", True, "retiring-session", 1)
        active = MagicMock()
        active.session_id = "retiring-session"

        handler._retire_finalization_session(
            chat_id="chat-1",
            project=project,
            thread_id=None,
            active_session=active,
        )

        assert project.coco_session_snapshot is None
        project_manager.persist_project_context.assert_called_once_with(project)

    def test_snapshot_persistence_cannot_consume_retirement_budget(self):
        handler = ProgrammingModeHandler.__new__(ProgrammingModeHandler)
        handler.mode_key = "coco"
        allow_persist = threading.Event()
        persist_started = threading.Event()
        project_manager = MagicMock()

        def blocking_persist(_project):
            persist_started.set()
            allow_persist.wait(timeout=2)
            return True

        project_manager.persist_project_context.side_effect = blocking_persist
        handler.ctx = SimpleNamespace(project_manager=project_manager)
        manager = MagicMock()
        handler._get_session_manager = MagicMock(return_value=manager)
        project = ProjectContext(
            project_id="project-1",
            project_name="project",
            root_path="/tmp",
        )
        project.set_programming_mode("coco", True, "retiring-session", 1)
        active = MagicMock()
        active.session_id = "retiring-session"
        started = time.monotonic()
        try:
            handler._retire_finalization_session(
                chat_id="chat-1",
                project=project,
                thread_id=None,
                active_session=active,
                retirement_budget_s=0.2,
            )
            elapsed = time.monotonic() - started
            assert persist_started.wait(timeout=0.1)
            assert elapsed < 0.2
            passed_timeout = manager.retire_session.call_args.kwargs["timeout"]
            assert 0 < passed_timeout < 0.2
        finally:
            allow_persist.set()

    @patch("src.thread.get_current_thread_id", return_value=None)
    def test_active_mode_without_live_session_starts_fresh_session(self, _mock_tid):
        h, ctx = self._make_coco()
        ctx.mode_manager.get_mode.return_value = InteractionMode.COCO
        ctx.coco_manager.get_session.return_value = None
        fresh_session = MagicMock()
        fresh_session.session_id = "fresh-after-retirement"
        fresh_session.is_resumed = False
        ctx.coco_manager.ensure_session.return_value = fresh_session
        project = MagicMock()
        project.project_id = "project-1"
        project.project_name = "project"
        project.root_path = "/tmp"
        project.coco_session_snapshot = None

        result = h.enter_mode(
            "m1",
            "c1",
            project=project,
            silent=True,
        )

        assert result is True
        ctx.coco_manager.ensure_session.assert_called_once()
        assert ctx.coco_manager.ensure_session.call_args.kwargs["session_id"] is None
        project.set_programming_mode.assert_called_once_with(
            "coco",
            True,
            "fresh-after-retirement",
            0,
        )

    def test_enter_mode_with_project_no_snapshot(self):
        h, ctx = self._make_coco()
        project = MagicMock()
        project.coco_session_snapshot = None
        project.root_path = "/tmp"
        project.project_name = "test"
        project.project_id = "test_id"
        h.enter_mode("m1", "c1", project=project)
        ctx.mode_manager.enter_programming_mode.assert_called_once_with("c1", InteractionMode.COCO, project_id="test_id")
        project.set_programming_mode.assert_called_once()
        h.record_mode_transition.assert_called_once()

    @patch("src.thread.get_current_thread_id", return_value=None)
    def test_enter_mode_with_selected_model_does_not_resume_stale_snapshot(self, mock_tid):
        h, ctx = self._make_coco()
        project = MagicMock()
        project.coco_session_snapshot = SessionSnapshot(
            session_id="stale-session",
            query_count=3,
            last_query="pwd",
            is_resumable=True,
        )
        project.root_path = "/tmp"
        project.project_name = "test"
        project.project_id = "test_id"
        project.acp_tool_name = "coco"
        project.acp_model_name = "fresh-model"
        fresh_session = MagicMock()
        fresh_session.session_id = "fresh-session"
        fresh_session.is_resumed = False
        ctx.coco_manager.ensure_session.return_value = fresh_session

        h.enter_mode("m1", "c1", project=project, silent=True)

        ctx.coco_manager.ensure_session.assert_called_once()
        assert ctx.coco_manager.ensure_session.call_args.kwargs["session_id"] is None
        assert ctx.coco_manager.ensure_session.call_args.kwargs["model_name"] == "fresh-model"
        project.set_programming_mode.assert_called_once_with("coco", True, "fresh-session", 0)

    def test_exit_mode_with_session(self):
        h, ctx = self._make_coco()
        project = MagicMock()
        project.project_id = "p1"
        project.project_name = "test"
        project.root_path = "/tmp"
        h.exit_mode("m1", "c1", project=project)
        ctx.mode_manager.exit_to_smart.assert_called_once_with("c1", project_id="p1")
        ctx.coco_manager.end_session.assert_called_once_with("c1", project_id="p1", thread_id=None)




    def test_show_info_with_project(self):
        h, ctx = self._make_coco()
        ctx.coco_manager.get_session_info.return_value = "some info"
        project = MagicMock()
        project.project_name = "test"
        project.project_id = "p1"
        project.root_path = "/tmp"
        h.show_info("m1", "c1", project=project)
        h.reply_card.assert_called_once()

    def test_show_info_no_session(self):
        h, ctx = self._make_coco()
        ctx.coco_manager.get_session_info.return_value = None
        h.show_info("m1", "c1")
        h.reply_text.assert_called_once()

    def test_card_enter_with_project_no_snapshot(self):
        h, ctx = self._make_coco()
        project = MagicMock()
        project.coco_session_snapshot = None
        project.root_path = "/tmp"
        project.project_name = "test"
        project.project_id = "p1"
        ctx.project_manager.get_project_for_chat.return_value = project
        h.enter_mode = MagicMock()
        h.handle_card_enter("m1", "c1", "p1")
        ctx.project_manager.set_active_project.assert_called_once_with("c1", "p1")
        h.enter_mode.assert_called_once_with("m1", "c1", project=project)

    def test_card_exit(self):
        h, ctx = self._make_coco()
        project = MagicMock()
        ctx.project_manager.get_project_for_chat.return_value = project
        h.exit_mode = MagicMock()
        h.handle_card_exit("m1", "c1", "p1")
        project.set_programming_mode.assert_called_once_with("coco", False)
        h.exit_mode.assert_called_once()


class TestTopLevelProgrammingState:

    def _make_coco_pending(self):
        ctx = _make_handler_context()
        ctx.settings.thread_programming_enabled = True
        ctx.mode_manager.is_coco_mode.return_value = False
        ctx.mode_manager.is_claude_mode.return_value = False
        ctx.mode_manager.is_aiden_mode.return_value = False
        ctx.mode_manager.is_codex_mode.return_value = False
        ctx.mode_manager.is_gemini_mode.return_value = False
        ctx.mode_manager.get_mode.return_value = InteractionMode.SMART
        ctx.project_manager.validate_project_path.return_value = (True, "ok")
        ctx.project_manager.get_or_create_project_for_path.return_value = (None, False)
        ctx.coco_manager.ensure_session = MagicMock()
        ctx.coco_manager.get_session.return_value = None
        ctx.coco_manager.end_session.return_value = False

        h = CocoModeHandler(ctx)
        ctx.handlers.update({
            "claude": MagicMock(),
            "aiden": MagicMock(),
            "codex": MagicMock(),
            "gemini": MagicMock(),
        })
        # Keep attributes for test assertions
        h._claude_handler = ctx.handlers["claude"]
        h._aiden_handler = ctx.handlers["aiden"]
        h._codex_handler = ctx.handlers["codex"]
        h._gemini_handler = ctx.handlers["gemini"]
        h.reply_text = MagicMock()
        h.reply_text = MagicMock()
        h.reply_card = MagicMock()
        h.reply_error = MagicMock()
        h.reply_card = MagicMock(return_value="reply_1")
        h.add_reaction = MagicMock()
        h.record_mode_transition = MagicMock()
        h.register_message_project = MagicMock()
        return h, ctx

    @patch("src.thread.get_current_thread_id", return_value=None)
    def test_enter_mode_thread_enabled_starts_top_level_session(self, mock_tid):
        h, ctx = self._make_coco_pending()
        project = MagicMock()
        project.coco_session_snapshot = None
        project.root_path = "/tmp"
        project.project_name = "test"
        project.project_id = "test_id"

        result = h.enter_mode("m1", "c1", project=project)

        assert result is True
        ctx.mode_manager.enter_programming_mode.assert_called_once_with("c1", InteractionMode.COCO, project_id="test_id")
        ctx.coco_manager.ensure_session.assert_called_once()
        h.add_reaction.assert_called_once()
        h.reply_card.assert_called_once()
        call_args = str(h.reply_card.call_args)
        assert "编程模式已开启" in call_args or "已开启" in call_args
        h.record_mode_transition.assert_called_once()
        assert "enter_coco_mode" in str(h.record_mode_transition.call_args)

    @patch("src.thread.get_current_thread_id", return_value=None)
    def test_enter_mode_activation_result_is_false_when_session_start_fails(self, mock_tid):
        h, ctx = self._make_coco_pending()
        ctx.coco_manager.ensure_session.side_effect = RuntimeError("startup failed")
        project = MagicMock()
        project.coco_session_snapshot = None
        project.root_path = "/tmp"
        project.project_name = "test"
        project.project_id = "test_id"

        result = h.enter_mode("m1", "c1", project=project, silent=True)

        assert result is False
        ctx.mode_manager.enter_programming_mode.assert_not_called()

    @patch("src.thread.get_current_thread_id", return_value=None)
    def test_enter_mode_thread_enabled_already_in_mode(self, mock_tid):
        h, ctx = self._make_coco_pending()
        ctx.mode_manager.get_mode.return_value = InteractionMode.COCO
        ctx.coco_manager.get_session.return_value = MagicMock()

        h.enter_mode("m1", "c1")

        h.reply_text.assert_called_once()
        assert "已经在Coco编程模式" in str(h.reply_text.call_args)

    @patch("src.thread.get_current_thread_id", return_value=None)
    def test_exit_mode_pending_slot_no_session(self, mock_tid):
        h, ctx = self._make_coco_pending()
        ctx.mode_manager.get_mode.return_value = InteractionMode.COCO
        project = MagicMock()
        project.project_id = "p1"
        project.project_name = "test"
        project.root_path = "/tmp"

        h.exit_mode("m1", "c1", project=project)

        ctx.mode_manager.exit_to_smart.assert_called_once_with("c1", project_id="p1")
        h.add_reaction.assert_called_once()
        h.reply_card.assert_called_once()
        call_args = str(h.reply_card.call_args)
        assert "已退出" in call_args

    @patch("src.thread.get_current_thread_id", return_value="thread_123")
    def test_enter_mode_in_thread_creates_session(self, mock_tid):
        h, ctx = self._make_coco_pending()
        mock_session = MagicMock()
        mock_session.session_id = "sid1"
        mock_session.is_resumed = False
        ctx.coco_manager.ensure_session.return_value = mock_session
        ctx.coco_manager.get_session.return_value = mock_session

        project = MagicMock()
        project.coco_session_snapshot = None
        project.root_path = "/tmp"
        project.project_name = "test"
        project.project_id = "test_id"

        h.enter_mode("m1", "c1", project=project, thread_id="thread_123")

        ctx.coco_manager.ensure_session.assert_called_once()
        assert "thread_123" in str(ctx.coco_manager.ensure_session.call_args)


    @patch("src.thread.get_current_thread_id", return_value="thread_789")
    def test_handle_message_session_not_found_gives_feedback(self, mock_tid):
        """handle_message 在线程内 session 未找到且 enter_mode 也失败时应给出明确反馈"""
        h, ctx = self._make_coco_pending()
        ctx.coco_manager.get_session.return_value = None
        ctx.coco_manager.ensure_session.side_effect = Exception("startup failed")

        project = MagicMock()
        project.project_id = "test_id"
        project.root_path = "/tmp"
        project.project_name = "test"
        project.coco_session_snapshot = None
        ctx.project_manager.validate_project_path.return_value = (True, "ok")
        ctx.project_manager.get_or_create_project_for_path.return_value = (project, False)

        h.reply_text = MagicMock()
        h.reply_text = MagicMock()
        h.add_reaction = MagicMock()

        h.handle_message("m1", "c1", "继续写", project)

        ctx.coco_manager.ensure_session.assert_called()
        h.reply_text.assert_called()
        call_str = str(h.reply_text.call_args)
        assert "启动失败" in call_str or "重新发送" in call_str




# ======================================================================
# ProjectHandler tests
# ======================================================================


class TestProjectHandler:
    def _make(self):
        ctx = _make_handler_context()
        h = ProjectHandler(ctx)
        h.reply_text = MagicMock()
        h.reply_text = MagicMock()
        h.reply_card = MagicMock()
        h.reply_error = MagicMock()
        h.reply_card = MagicMock(return_value="reply_1")
        h.add_reaction = MagicMock()
        h.register_message_project = MagicMock()
        return h, ctx

    def test_create_project_success(self):
        h, ctx = self._make()
        project = MagicMock()
        project.project_name = "test"
        project.project_id = "test"
        project.root_path = "/tmp"
        ctx.project_manager.create_project.return_value = (True, "ok", project)
        with patch("src.feishu.handlers.project.CardBuilder") as mock_cb:
            mock_cb.build_project_created_card.return_value = ("interactive", "{}")
            h.create_project("m1", "c1", "test", "/tmp")
            h.reply_card.assert_called_once()

    def test_create_project_failure(self):
        h, ctx = self._make()
        ctx.project_manager.create_project.return_value = (False, "already exists", None)
        h.create_project("m1", "c1", "test", "/tmp")
        h.reply_error.assert_called_once()

    def test_show_project_board(self):
        h, ctx = self._make()
        ctx.project_manager.get_all_projects.return_value = []
        ctx.project_manager.get_active_project.return_value = None
        h.show_project_board("m1", "c1")
        h.reply_card.assert_called_once()

    def test_show_project_status_no_project(self):
        h, ctx = self._make()
        h.show_project_board = MagicMock()
        h.show_project_status("m1", "c1", None)
        h.show_project_board.assert_called_once_with("m1", "c1")

    def test_close_project_success(self):
        h, ctx = self._make()
        project = MagicMock()
        ctx.project_manager.find_project_by_name_with_hint.return_value = (project, "")
        ctx.project_manager.close_project.return_value = (True, "closed")
        h.close_project("m1", "c1", "test")
        h.reply_text.assert_called_once()
        assert "✅" in str(h.reply_text.call_args)

    def test_close_project_not_found(self):
        h, ctx = self._make()
        ctx.project_manager.find_project_by_name_with_hint.return_value = (None, "")
        h.close_project("m1", "c1", "test")
        h.reply_error.assert_called_once()
        assert "❌" in str(h.reply_error.call_args) or "未找到" in str(h.reply_error.call_args)

class TestDeepHandler:
    def _make(self):
        ctx = _make_handler_context()
        h = DeepHandler(ctx)
        h.reply_text = MagicMock()
        h.reply_text = MagicMock()
        h.reply_card = MagicMock()
        h.reply_error = MagicMock()
        h.update_card = MagicMock()
        h.reply_card = MagicMock(return_value="reply_1")
        h.add_reaction = MagicMock()
        h.register_message_project = MagicMock()
        return h, ctx

    def test_handle_deep_command_status(self):
        h, ctx = self._make()
        h.show_deep_status = MagicMock()
        h.handle_deep_command("m1", "c1", "/deep_status", None)
        h.show_deep_status.assert_called_once()

    def test_handle_deep_command_stop(self):
        h, ctx = self._make()
        h.stop_deep_engine = MagicMock()
        h.handle_deep_command("m1", "c1", "/stop_deep", None)
        h.stop_deep_engine.assert_called_once()

    def test_handle_deep_command_status_all(self):
        h, ctx = self._make()
        h.show_deep_board = MagicMock()
        h.handle_deep_command("m1", "c1", "/deep_status all", None)
        h.show_deep_board.assert_called_once()

    def test_handle_deep_command_stop_all(self):
        h, ctx = self._make()
        h.stop_all_deep_engines = MagicMock()
        h.handle_deep_command("m1", "c1", "/stop_deep all", None)
        h.stop_all_deep_engines.assert_called_once()

    def test_handle_deep_command_empty(self):
        h, ctx = self._make()
        h.handle_deep_command("m1", "c1", "/deep", None)
        h.reply_error.assert_called_once()
        assert "请提供需求" in str(h.reply_error.call_args)

    def test_handle_deep_command_start(self):
        h, ctx = self._make()
        h.start_deep_engine = MagicMock()
        h.handle_deep_command("m1", "c1", "/deep implement feature X", None)
        h.start_deep_engine.assert_called_once_with("m1", "c1", "implement feature X", None)

    def test_start_deep_engine_passes_selected_coco_model_to_engine(self):
        h, ctx = self._make()
        project = SimpleNamespace(
            project_id="p1",
            project_name="Project",
            root_path="/repo/project",
            acp_tool_name="coco",
            acp_model_name="deepseek-v4pro",
        )
        engine = MagicMock()
        ctx.mode_manager.get_mode.return_value = InteractionMode.SMART
        ctx.deep_engine_manager.get.return_value = None
        ctx.deep_engine_manager.get_or_create.return_value = engine

        h._ensure_topic_engine_context = MagicMock(return_value="m1")
        h.ensure_request_id = MagicMock(return_value="req1")
        h.get_engine_name = MagicMock(return_value="Coco")
        h._submit_engine_task = MagicMock()

        h.start_deep_engine("m1", "c1", "implement feature X", project)

        ctx.deep_engine_manager.get_or_create.assert_called_once_with(
            "c1",
            "/repo/project",
            engine_name="Coco",
            model_name="deepseek-v4pro",
        )

    def test_start_deep_engine_passes_requirement_to_card_renderer(self):
        h, ctx = self._make()
        project = SimpleNamespace(
            project_id="p1",
            project_name="Project",
            root_path="/repo/project",
            acp_tool_name="coco",
            acp_model_name=None,
        )
        engine = MagicMock()
        callbacks = MagicMock()
        ctx.mode_manager.get_mode.return_value = InteractionMode.SMART
        ctx.deep_engine_manager.get.return_value = None
        ctx.deep_engine_manager.get_or_create.return_value = engine
        h._ensure_topic_engine_context = MagicMock(return_value="m1")
        h.ensure_request_id = MagicMock(return_value="req1")
        h.get_engine_name = MagicMock(return_value="Coco")
        h.renderer.create_deep_callbacks = MagicMock(return_value=callbacks)
        h._submit_engine_task = MagicMock(side_effect=lambda task, *_args: task())
        h._safe_execute_engine = MagicMock(
            side_effect=lambda *, executor_func, **_kwargs: executor_func()
        )

        h.start_deep_engine("m1", "c1", "优化 Deep 消息卡片", project)

        h.renderer.create_deep_callbacks.assert_called_once_with(
            "m1",
            "c1",
            project,
            "Coco",
            root_path="/repo/project",
            initial_message_id=None,
            requirement_text="优化 Deep 消息卡片",
        )

    def test_handle_deep_command_update(self):
        h, ctx = self._make()
        h.update_deep_context = MagicMock()
        h.handle_deep_command("m1", "c1", "/deep_update some new info", None)
        h.update_deep_context.assert_called_once()

    def test_stop_all_deep_engines(self):
        h, ctx = self._make()
        engine1 = MagicMock()
        engine2 = MagicMock()
        ctx.deep_engine_manager.get_active_engines.return_value = [engine1, engine2]
        h.stop_all_deep_engines("m1", "c1")
        engine1.stop.assert_called_once()
        engine2.stop.assert_called_once()
        h.reply_text.assert_called_once()

    def test_stop_all_deep_engines_none_running(self):
        h, ctx = self._make()
        ctx.deep_engine_manager.get_active_engines.return_value = []
        h.stop_all_deep_engines("m1", "c1")
        h.reply_error.assert_called_once()
        assert "没有" in str(h.reply_error.call_args)

    def test_show_deep_status_patch_success(self):
        h, ctx = self._make()
        # Setup mock project and engine
        project = MagicMock()
        project.root_path = "/path/to/project"
        project.project_id = "p1"

        engine = MagicMock()
        engine.project = MagicMock()
        engine.progress = MagicMock()
        engine.engine_name = "DeepEngine"

        ctx.deep_engine_manager.get.return_value = engine

        # Setup mock reporter
        ctx.progress_reporter.format_status.return_value = "Status Content"
        ctx.progress_reporter.get_status_title.return_value = "Status Title"
        ctx.progress_reporter.get_progress_info.return_value = {
            "progress_bar": "|||",
            "project_id": "p1",
            "is_executing": True,
            "is_paused": False,
            "completed": 3,
            "total": 10,
        }

        # Mock CardSession creation
        with patch("src.feishu.renderers.base.BaseRenderer.create_session") as mock_create:
            mock_session = MagicMock()
            mock_create.return_value = mock_session

            # Execute
            h.show_deep_status("msg1", "chat1", project=project, origin_message_id="origin1")

            # Verify CardSession was created and events dispatched
            mock_create.assert_called_once()
            calls = [c.args[0] for c in mock_session.dispatch.call_args_list]
            types = [c.type.value for c in calls]
            assert "started" in types
            assert "text_delta" in types

    def test_show_deep_status_patch_failure_fallback(self):
        h, ctx = self._make()
        # Setup mock project and engine
        project = MagicMock()
        project.root_path = "/path/to/project"
        project.project_id = "p1"

        engine = MagicMock()
        engine.project = MagicMock()
        engine.progress = MagicMock()
        engine.engine_name = "DeepEngine"

        ctx.deep_engine_manager.get.return_value = engine

        ctx.progress_reporter.format_status.return_value = "Status Content"
        ctx.progress_reporter.get_status_title.return_value = "Status Title"
        ctx.progress_reporter.get_progress_info.return_value = {
            "progress_bar": "|||",
            "project_id": "p1",
            "is_executing": False,
            "is_paused": False,
            "completed": 10,
            "total": 10,
        }

        # Mock CardSession creation
        with patch("src.feishu.renderers.base.BaseRenderer.create_session") as mock_create:
            mock_session = MagicMock()
            mock_create.return_value = mock_session

            # Execute
            h.show_deep_status("msg1", "chat1", project=project, origin_message_id="origin1")

            # Verify completed was dispatched (not executing)
            calls = [c.args[0] for c in mock_session.dispatch.call_args_list]
            types = [c.type.value for c in calls]
            assert "completed" in types


# ======================================================================
# DiagnosticsHandler tests
# ======================================================================


class TestDiagnosticsHandler:
    def _make(self):
        ctx = _make_handler_context()
        h = DiagnosticsHandler(ctx)
        h.reply_text = MagicMock()
        h.reply_text = MagicMock()
        h.reply_card = MagicMock()
        h.reply_card = MagicMock(return_value="reply_1")
        h.add_reaction = MagicMock()
        h.register_message_project = MagicMock()
        return h, ctx

    def test_show_task_board_no_project(self):
        """When no project is active and no project passed, should report no project."""
        h, ctx = self._make()
        ctx.project_manager.get_active_project.return_value = None
        h.show_task_board("m1", "c1", "/tasks", None)
        h.reply_text.assert_called()
        assert "没有" in str(h.reply_text.call_args)

    def test_show_task_board_all(self):
        """When /tasks all is used, shows all-project task board."""
        h, ctx = self._make()
        ctx.scheduler.get_all_tasks.return_value = []
        with patch("src.feishu.handlers.diagnostics.CardBuilder") as mock_cb:
            mock_cb.build_smart_response_card.return_value = ("interactive", "{}")
            h.show_task_board("m1", "c1", "/tasks all", None)
        h.reply_card.assert_called()

    def test_show_context_diff_no_project(self):
        """show_context_diff with no active project should report that."""
        h, ctx = self._make()
        ctx.project_manager.get_active_project.return_value = None
        h.show_context_diff("m1", "c1", "/diff", None)
        h.reply_text.assert_called_once()
        assert "没有" in str(h.reply_text.call_args)

    def test_show_message_trace_no_args(self):
        h, ctx = self._make()
        h.show_message_trace("m1", "c1", "/trace", None)
        h.reply_text.assert_called_once()


# ======================================================================
# Bug fix: Shell command fast-track heuristic
# ======================================================================


class TestShellCommandHeuristic:
    """Tests for SystemHandler.is_likely_shell_command()."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "ls",
            "pwd",
            "whoami",
            "date",
            "uptime",
            "ls -la",
            "git status",
            "cat foo.txt",
            "grep pattern file",
            "docker ps",
            "make build",
            "./restart.sh rr",
            "../scripts/restart rr",
            "bash ./restart.sh rr",
            "sh ./restart.sh rr",
            "uv run python -m pytest",
            "tree src/",
        ],
    )
    def test_detects_shell_commands(self, cmd):
        assert SystemHandler.is_likely_shell_command(cmd) is True

    @pytest.mark.parametrize(
        "cmd",
        [
            "/help",
            "/coco",
            "/deep implement X",
            "/exit",
            "帮我写一个函数",
            "请解释这段代码",
            "",
        ],
    )
    def test_ignores_non_shell(self, cmd):
        assert SystemHandler.is_likely_shell_command(cmd) is False

    def test_slash_commands_are_not_shell(self):
        assert SystemHandler.is_likely_shell_command("/projects") is False
        assert SystemHandler.is_likely_shell_command("/switch foo") is False


# ======================================================================
# Bug fix: ACP Session Manager project isolation
# ======================================================================


class TestACPSessionManagerProjectIsolation:
    """Tests for ACPSessionManager keyed by (chat_id, project_id)."""

    def test_session_key_with_project(self):
        from src.acp.manager import ACPSessionManager

        key = ACPSessionManager._session_key("chat1", "proj_a")
        assert key == "chat1:proj_a"

    def test_session_key_without_project(self):
        from src.acp.manager import ACPSessionManager

        key = ACPSessionManager._session_key("chat1")
        assert key == "chat1:_default_"
        key2 = ACPSessionManager._session_key("chat1", None)
        assert key2 == "chat1:_default_"

    def test_different_projects_get_different_keys(self):
        from src.acp.manager import ACPSessionManager

        k1 = ACPSessionManager._session_key("chat1", "proj_a")
        k2 = ACPSessionManager._session_key("chat1", "proj_b")
        k3 = ACPSessionManager._session_key("chat1")
        assert k1 != k2
        assert k1 != k3
        assert k2 != k3

    def test_get_session_isolated_by_project(self):
        from src.acp.manager import ACPSessionManager

        mgr = ACPSessionManager("coco", session_timeout=999999)

        # Manually insert two sessions for same chat, different projects
        import time

        s1 = MagicMock()
        s1.last_active = time.time()
        s1.session_id = "s1"
        s2 = MagicMock()
        s2.last_active = time.time()
        s2.session_id = "s2"

        k1 = mgr._session_key("chat1", "proj_a")
        k2 = mgr._session_key("chat1", "proj_b")
        mgr._sessions[k1] = s1
        mgr._sessions[k2] = s2

        assert mgr.get_session("chat1", project_id="proj_a") is s1
        assert mgr.get_session("chat1", project_id="proj_b") is s2
        # Default project should not find either
        assert mgr.get_session("chat1") is None

    def test_end_session_does_not_affect_other_projects(self):
        import time

        from src.acp.manager import ACPSessionManager

        mgr = ACPSessionManager("coco", session_timeout=999999)

        s1 = MagicMock()
        s1.last_active = time.time()
        s1.session_id = "s1"
        s1.message_count = 0
        s1.to_snapshot.return_value = {}
        s2 = MagicMock()
        s2.last_active = time.time()
        s2.session_id = "s2"

        mgr._sessions[mgr._session_key("chat1", "proj_a")] = s1
        mgr._sessions[mgr._session_key("chat1", "proj_b")] = s2

        mgr.end_session("chat1", project_id="proj_a")

        # proj_a session ended, proj_b untouched
        assert mgr.get_session("chat1", project_id="proj_a") is None
        assert mgr.get_session("chat1", project_id="proj_b") is s2

    def test_agent_session_manager_alias_from_manager(self):
        from src.acp.manager import ACPSessionManager, AgentSessionManager

        assert issubclass(AgentSessionManager, ACPSessionManager)
        assert AgentSessionManager._session_key("chat1", "proj_a") == ACPSessionManager._session_key("chat1", "proj_a")

    def test_agent_session_manager_alias_from_package_exports(self):
        from src.acp import ACPSessionManager, AgentSessionManager

        mgr = AgentSessionManager("coco", session_timeout=999999)
        assert isinstance(mgr, ACPSessionManager)
        assert mgr._session_key("chat1") == "chat1:_default_"

    def test_parse_session_key_roundtrip_basic_and_thread(self):
        from src.acp.manager import _DEFAULT_PROJECT, ACPSessionManager

        # 显式 project + 线程维度
        key = ACPSessionManager._session_key("chat-1", "proj-1", thread_id="thread-9")
        chat_id, project_id, thread_id = ACPSessionManager._parse_session_key(key)
        assert chat_id == "chat-1"
        assert project_id == "proj-1"
        assert thread_id == "thread-9"

        # 默认项目：应返回 project_id=None
        key_default = ACPSessionManager._session_key("chat-2")
        chat_id2, project_id2, thread_id2 = ACPSessionManager._parse_session_key(key_default)
        assert chat_id2 == "chat-2"
        assert project_id2 is None
        assert thread_id2 is None

        # 直接传入带占位符的历史 key，应与 helper 约定一致
        legacy_key = f"chat-3:{_DEFAULT_PROJECT}"
        chat_id3, project_id3, thread_id3 = ACPSessionManager._parse_session_key(legacy_key)
        assert chat_id3 == "chat-3"
        assert project_id3 is None
        assert thread_id3 is None

        # 非标准 key：至少应保留原始 key 作为 chat_id，避免日志上下文丢失
        weird_key = "just-a-single-token"
        chat_id4, project_id4, thread_id4 = ACPSessionManager._parse_session_key(weird_key)
        assert chat_id4 == weird_key
        assert project_id4 is None
        assert thread_id4 is None


# ======================================================================
# SystemHandler patch tests
# ======================================================================


class TestHelpCategoryPatch:
    def _make(self, ctx_overrides=None):
        if ctx_overrides is None:
            ctx_overrides = {}
        ctx = _make_handler_context(**ctx_overrides)
        h = SystemHandler(ctx)
        h.reply_text = MagicMock()
        h.reply_card = MagicMock()
        h.update_card = MagicMock()
        # Mock get_working_dir to return a valid path string for CardBuilder
        h.get_working_dir = MagicMock(return_value="/tmp")
        return h, ctx

    def test_handle_help_category_patch_success(self):
        h, ctx = self._make()

        ctx.mode_manager.get_mode.return_value = InteractionMode.SMART

        # Mock update_card
        h.update_card = MagicMock(return_value=True)

        with patch("src.card.builder.CardBuilder.build_help_card") as mock_build:
            mock_build.return_value = ("interactive", "{}")

            h.handle_help_category("msg1", "chat1", "main", origin_message_id="origin1")

            # Verify update_card called
            h.update_card.assert_called_once_with("origin1", "{}")
            # Verify reply_card NOT called
            h.reply_card.assert_not_called()

    def test_handle_help_category_patch_failure_fallback(self):
        h, ctx = self._make()

        ctx.mode_manager.get_mode.return_value = InteractionMode.SMART

        # Mock update_card failure
        h.update_card = MagicMock(return_value=False)

        with patch("src.card.builder.CardBuilder.build_help_card") as mock_build:
            mock_build.return_value = ("interactive", "{}")

            h.handle_help_category("msg1", "chat1", "main", origin_message_id="origin1")

            # Verify update_card called
            h.update_card.assert_called_once_with("origin1", "{}")
            # Verify fallback to reply_card
            h.reply_card.assert_called_once()

    def test_handle_help_category_no_origin_id(self):
        h, ctx = self._make()

        ctx.mode_manager.get_mode.return_value = InteractionMode.SMART

        mock_client = MagicMock()
        ctx.api_client_factory.return_value = mock_client

        with patch("src.card.builder.CardBuilder.build_help_card") as mock_build:
            mock_build.return_value = ("interactive", "{}")

            h.handle_help_category("msg1", "chat1", "main", origin_message_id=None)

            h.update_card.assert_not_called()
            h.reply_card.assert_called_once()


class TestSystemHandlerShellRepoLock:
    def test_execute_shell_sends_conflict_card_without_running_when_repo_locked(self):
        from src.repo_lock import LockConflictError

        ctx = _make_handler_context()
        h = SystemHandler(ctx)
        h.reply_card = MagicMock()
        h.add_reaction = MagicMock()
        h.send_lock_conflict_card = MagicMock()

        err = LockConflictError(
            "repo locked",
            holder_chat_id="other-chat",
            locked_since=1.0,
            root_path="/repo",
            last_active_time=1.0,
        )
        h.lock_helper._with_repo_lock_strict = MagicMock(side_effect=err)

        with patch("src.sandbox.executor.SandboxExecutor.execute") as mock_execute:
            result = h.execute_shell_and_reply("msg-1", "chat-1", "pwd", "/repo", None)

        assert result is None
        mock_execute.assert_not_called()
        h.send_lock_conflict_card.assert_called_once_with(
            err,
            "msg-1",
            "pwd",
            chat_id="chat-1",
        )
        h.reply_card.assert_not_called()


# ======================================================================
# AC-16: _with_repo_lock signature (no message_id/sender_id)
# ======================================================================


class TestWithRepoLockSignature:
    """AC-16: _with_repo_lock accepts only (root_path, chat_id, body_func)."""

    def test_invokes_body_func(self):
        ctx = _make_handler_context()
        h = BaseHandler(ctx)
        ctx.repo_lock_manager = None  # no lock manager → direct call

        called = []
        h._with_repo_lock("/tmp/repo", "chat_1", lambda: called.append(True))
        assert called == [True]


# ======================================================================
# AC-22: build_lock_help_body non-admin hides /lock /unlock
# ======================================================================


class TestLockHelpBodyVisibility:
    """AC-22: Non-admin help card shows '联系 Bot 管理员' instead of /lock /unlock."""

    def test_non_admin_sees_contact_hint(self):
        from src.card.builders.lock import build_lock_help_body
        body = build_lock_help_body(is_admin=False)
        assert "联系 Bot 管理员" in body
        assert "`/lock`" not in body
        assert "`/unlock`" not in body

    def test_admin_sees_lock_commands(self):
        from src.card.builders.lock import build_lock_help_body
        body = build_lock_help_body(is_admin=True)
        assert "`/lock`" in body
        assert "`/unlock`" in body


# ======================================================================
# AC-23: _build_lock_status_lines — no active locks → idle summary
# ======================================================================


class TestLockStatusNoiseReduction:
    """AC-23: /status shows lock section with placeholder when no active locks."""

    def test_no_active_locks_returns_placeholder(self):
        """When no locks are active, _build_lock_status_lines returns placeholder."""
        ctx = _make_handler_context()
        h = DiagnosticsHandler(ctx)

        # Both managers present but no active locks
        mock_chat_lock = MagicMock()
        mock_chat_lock.get_lock_info.return_value = None
        mock_repo_lock = MagicMock()

        ctx.chat_lock_manager = mock_chat_lock
        ctx.repo_lock_manager = mock_repo_lock

        result = h._build_lock_status_lines("chat_1")
        # F-11: when lock subsystem is enabled but no active locks, show "unlocked"
        assert "未锁定" in result

    def test_no_managers_returns_placeholder(self):
        """When neither lock manager is configured, returns empty string."""
        ctx = _make_handler_context()
        h = DiagnosticsHandler(ctx)
        ctx.chat_lock_manager = None
        ctx.repo_lock_manager = None

        result = h._build_lock_status_lines("chat_1")
        assert result == ""

    def test_active_chat_lock_shows_section(self):
        """When chat lock IS active, section is shown."""
        ctx = _make_handler_context()
        h = DiagnosticsHandler(ctx)

        mock_chat_lock = MagicMock()
        mock_entry = MagicMock()
        mock_entry.locked_by = "ou_admin123"
        mock_entry.locked_by_name = "Admin"
        mock_entry.locked_at_wall = 1700000000.0
        mock_entry.locked_at = 1000.0  # monotonic timestamp for format_lock_duration
        mock_chat_lock.get_lock_info.return_value = mock_entry

        ctx.chat_lock_manager = mock_chat_lock
        ctx.repo_lock_manager = None

        result = h._build_lock_status_lines("chat_1")
        assert "锁状态" in result
        assert "已锁定" in result


# ======================================================================
# AC-19: Spec handler lock integration
# ======================================================================


class TestSpecHandlerLockIntegration:
    """AC-19: start_spec_engine wraps execution in _with_repo_lock."""

    def _make_spec_handler(self):
        ctx = _make_handler_context()
        h = SpecHandler(ctx)
        h.reply_text = MagicMock()
        h.reply_card = MagicMock(return_value="reply_1")
        h.add_reaction = MagicMock()
        h.register_message_project = MagicMock()
        h.get_working_dir = MagicMock(return_value="/tmp/spec_repo")
        h.ensure_request_id = MagicMock(return_value="req-1")
        h.send_card_to_chat = MagicMock()

        mock_project = MagicMock()
        mock_project.project_id = "proj-1"
        mock_project.project_name = "myproj"
        mock_project.root_path = "/tmp/spec_repo"

        mock_engine = MagicMock()
        mock_engine.is_running = False
        ctx.spec_engine_manager.get.return_value = None
        ctx.spec_engine_manager.get_or_create.return_value = mock_engine

        return h, ctx, mock_project

    def test_scheduled_run_calls_with_repo_lock(self):
        """The task submitted to scheduler invokes _with_repo_lock."""
        h, ctx, mock_project = self._make_spec_handler()

        # Capture the lambda submitted to scheduler
        submitted_fn = None
        def capture_submit(spec, fn):
            nonlocal submitted_fn
            submitted_fn = fn
            handle = MagicMock()
            handle.run_id = "run-1"
            return handle
        ctx.scheduler.submit = capture_submit

        # Mock _with_repo_lock to track call
        lock_calls = []
        def mock_with_repo_lock(root_path, chat_id, body_func):
            lock_calls.append((root_path, chat_id))
            body_func()
        h.lock_helper._with_repo_lock = mock_with_repo_lock

        with patch("src.feishu.handlers.spec.CardBuilder") as mock_cb:
            mock_cb.build_info_card.return_value = ("interactive", "{}")
            # start_spec_engine(message_id, chat_id, requirement, project)
            h.start_spec_engine("msg-1", "chat-1", "fix the bug", mock_project)

        assert submitted_fn is not None
        submitted_fn(MagicMock())

        assert len(lock_calls) == 1
        assert lock_calls[0][0] == "/tmp/spec_repo"

    def test_scheduled_run_catches_lock_conflict(self):
        """LockConflictError from _with_repo_lock triggers conflict card."""
        import time

        from src.repo_lock import LockConflictError

        h, ctx, mock_project = self._make_spec_handler()

        submitted_fn = None
        def capture_submit(spec, fn):
            nonlocal submitted_fn
            submitted_fn = fn
            handle = MagicMock()
            handle.run_id = "run-1"
            return handle
        ctx.scheduler.submit = capture_submit

        def mock_with_repo_lock(root_path, chat_id, body_func):
            raise LockConflictError(
                "conflict", holder_chat_id="other_chat",
                locked_since=time.monotonic() - 60, root_path=root_path,
            )
        h.lock_helper._with_repo_lock = mock_with_repo_lock
        h.lock_helper.send_lock_conflict_card = MagicMock()

        with patch("src.feishu.handlers.spec.CardBuilder") as mock_cb:
            mock_cb.build_info_card.return_value = ("interactive", "{}")
            h.start_spec_engine("msg-1", "chat-1", "fix the bug", mock_project)

        assert submitted_fn is not None
        submitted_fn(MagicMock())

        h.lock_helper.send_lock_conflict_card.assert_called_once()
        call_args = h.lock_helper.send_lock_conflict_card.call_args[0]
        assert isinstance(call_args[0], LockConflictError)
        assert "/spec fix the bug" in call_args[2]


# ======================================================================
# ForceReleaseRepoLock handler tests
# ======================================================================


class TestForceReleaseRepoLockHandler:
    """Tests for SystemHandler.handle_force_release_repo_lock (two-step confirmation)."""

    def _make(self, *, admin_ids=None):
        ctx = _make_handler_context()
        ctx.chat_lock_manager = MagicMock()
        ctx.repo_lock_manager = MagicMock()
        if admin_ids is not None:
            ctx.chat_lock_manager.is_admin.side_effect = lambda uid: uid in admin_ids
        else:
            ctx.chat_lock_manager.is_admin.return_value = True
        h = SystemHandler(ctx)
        h.reply_text = MagicMock()
        h.reply_text = MagicMock()
        h.reply_card = MagicMock()
        h.reply_error = MagicMock()
        h.send_card_to_chat = MagicMock()
        h.send_text_to_chat = MagicMock()
        return h, ctx

    @patch("src.thread.get_current_sender_id", return_value="admin_1")
    def test_admin_sees_confirm_card(self, _mock_sender):
        """F-22: force release now shows a confirmation card instead of releasing immediately."""
        h, ctx = self._make(admin_ids={"admin_1"})
        ctx.repo_lock_manager.token_to_path.return_value = "/tmp/repo"
        ctx.repo_lock_manager.path_to_token.return_value = "tok123"

        h.handle_force_release_repo_lock(
            "msg-1", "chat-1", "proj-1",
            {"repo_token": "tok123"},
        )

        # Should NOT have force-released yet
        ctx.repo_lock_manager.force_release.assert_not_called()
        # Should have sent a confirmation card
        h.reply_card.assert_called_once()
        card_json = h.reply_card.call_args[0][1]
        assert "确认" in card_json or "confirm" in card_json.lower()

    @patch("src.thread.get_current_sender_id", return_value="admin_1")
    def test_confirm_force_release_executes(self, _mock_sender):
        """F-22: handle_confirm_force_release actually releases the lock."""
        import time as _t
        h, ctx = self._make(admin_ids={"admin_1"})
        lock_info = MagicMock()
        lock_info.chat_id = "chat_holder"
        ctx.repo_lock_manager.get_lock_info.return_value = lock_info
        ctx.repo_lock_manager.token_to_path.return_value = "/tmp/repo"

        h.handle_confirm_force_release(
            "msg-1", "chat-1", "proj-1",
            {"repo_token": "tok123", "timestamp": _t.time()},
        )

        ctx.repo_lock_manager.force_release.assert_called_once_with("/tmp/repo")
        h.reply_text.assert_called_once()
        assert "强制释放" in h.reply_text.call_args[0][1]

    @patch("src.thread.get_current_sender_id", return_value="admin_1")
    def test_confirm_force_release_expired(self, _mock_sender):
        """F-22: expired confirmation token is rejected."""
        import time as _t
        h, ctx = self._make(admin_ids={"admin_1"})

        h.handle_confirm_force_release(
            "msg-1", "chat-1", "proj-1",
            {"repo_token": "tok123", "timestamp": _t.time() - 9999},
        )

        ctx.repo_lock_manager.force_release.assert_not_called()
        h.reply_card.assert_called_once()
        assert "过期" in h.reply_card.call_args[0][1]

    @patch("src.thread.get_current_sender_id", return_value="admin_1")
    def test_cancel_force_release(self, _mock_sender):
        h, ctx = self._make(admin_ids={"admin_1"})
        h.handle_cancel_force_release("msg-1", "chat-1", "proj-1", {})
        h.reply_text.assert_called_once()
        assert "取消" in h.reply_text.call_args[0][1]

    @patch("src.thread.get_current_sender_id", return_value="user_2")
    def test_non_admin_rejected(self, _mock_sender):
        h, ctx = self._make(admin_ids={"admin_1"})

        h.handle_force_release_repo_lock("msg-1", "chat-1", "proj-1", {"repo_token": "tok"})

        ctx.repo_lock_manager.force_release.assert_not_called()
        h.reply_error.assert_called_once()
        assert "权限不足" in h.reply_error.call_args[0][1]

    @patch("src.thread.get_current_sender_id", return_value="admin_1")
    def test_missing_path_error(self, _mock_sender):
        h, ctx = self._make(admin_ids={"admin_1"})
        ctx.repo_lock_manager.token_to_path.return_value = None

        h.handle_force_release_repo_lock("msg-1", "chat-1", "proj-1", {"repo_token": "bad"})

        ctx.repo_lock_manager.force_release.assert_not_called()
        h.reply_error.assert_called_once()

    @patch("src.thread.get_current_sender_id", return_value="admin_1")
    def test_no_chat_lock_manager_rejected(self, _mock_sender):
        """When chat_lock_manager is None, fail-close rejects the request."""
        h, ctx = self._make()
        ctx.chat_lock_manager = None

        h.handle_force_release_repo_lock("msg-1", "chat-1", "proj-1", {"repo_token": "tok"})

        ctx.repo_lock_manager.force_release.assert_not_called()
        h.reply_error.assert_called_once()


# ======================================================================
# RetryCommand handler tests
# ======================================================================


class TestRetryCommandHandler:
    """Tests for the retry_command card action handler in action_registry."""

    def test_retry_command_empty_command_replies(self):
        """retry_command with empty command_text should reply with retry_empty_command."""
        from src.feishu.retry_handler import RetryCommandHandler
        dispatch = MagicMock()
        handler = RetryCommandHandler(dispatch)
        handler("mid_1", "chat_1", None, {"command_text": ""})
        dispatch.reply_text.assert_called_once()
        msg = dispatch.reply_text.call_args[0][1]
        assert "无法获取重试命令" in msg
        dispatch.process_with_intent.assert_not_called()

# ======================================================================
# Engine base LockConflictError tests
# ======================================================================


class TestEngineBaseLockConflict:
    """_safe_execute_engine catches LockConflictError and sends conflict card."""

    def test_lock_conflict_sends_card(self):
        ctx = _make_handler_context()
        from src.feishu.handlers.engine_base import BaseEngineHandler
        h = BaseEngineHandler(ctx)

        mock_project = MagicMock()
        mock_project.root_path = "/tmp/test"

        # Make _with_repo_lock raise LockConflictError
        from src.repo_lock import LockConflictError
        h.lock_helper._with_repo_lock = MagicMock(
            side_effect=LockConflictError(
                "conflict", holder_chat_id="chat_holder",
                locked_since=0.0, root_path="/tmp/test",
            )
        )
        h.lock_helper.send_lock_conflict_card = MagicMock()
        h.reply_text = MagicMock()

        h._safe_execute_engine(
            executor_func=MagicMock(),
            task_id="task-1",
            chat_id="chat-1",
            message_id="msg-1",
            project=mock_project,
            engine_name="TestEngine",
            reporter=MagicMock(),
            request_id="req-1",
        )

        h.lock_helper.send_lock_conflict_card.assert_called_once()
        err = h.lock_helper.send_lock_conflict_card.call_args[0][0]
        assert isinstance(err, LockConflictError)
        assert err.holder_chat_id == "chat_holder"

    def test_no_project_skips_lock(self):
        """When project is None, no lock is attempted."""
        ctx = _make_handler_context()
        from src.feishu.handlers.engine_base import BaseEngineHandler
        h = BaseEngineHandler(ctx)

        executor = MagicMock()
        h.reply_text = MagicMock()
        h.lock_helper.send_lock_conflict_card = MagicMock()

        h._safe_execute_engine(
            executor_func=executor,
            task_id="task-1",
            chat_id="chat-1",
            message_id="msg-1",
            project=None,
            engine_name="TestEngine",
            reporter=MagicMock(),
            request_id="req-1",
        )

        # Should execute body directly (no conflict)
        executor.assert_called_once()
        h.lock_helper.send_lock_conflict_card.assert_not_called()

    def test_lock_conflict_from_executor_passthrough(self):
        """AC-R04: LockConflictError thrown by executor_func must NOT be swallowed by _body's except Exception."""
        ctx = _make_handler_context()
        from src.feishu.handlers.engine_base import BaseEngineHandler
        from src.repo_lock import LockConflictError
        h = BaseEngineHandler(ctx)

        mock_project = MagicMock()
        mock_project.root_path = "/tmp/test"

        # executor_func itself raises LockConflictError (e.g. nested locking)
        executor = MagicMock(side_effect=LockConflictError(
            "nested conflict", holder_chat_id="chat_nested",
            locked_since=0.0, root_path="/tmp/nested",
        ))

        # _with_repo_lock should just call the body (no lock conflict at the guard level)
        h.lock_helper._with_repo_lock = MagicMock(side_effect=lambda rp, cid, body: body())
        h.lock_helper.send_lock_conflict_card = MagicMock()
        h.reply_text = MagicMock()

        h._safe_execute_engine(
            executor_func=executor,
            task_id="task-1",
            chat_id="chat-1",
            message_id="msg-1",
            project=mock_project,
            engine_name="TestEngine",
            reporter=MagicMock(),
            request_id="req-1",
        )

        # The LockConflictError should have been re-raised through _body,
        # caught by the outer except in _safe_execute_engine, and handled via conflict card.
        h.lock_helper.send_lock_conflict_card.assert_called_once()
        err = h.lock_helper.send_lock_conflict_card.call_args[0][0]
        assert isinstance(err, LockConflictError)
        assert err.holder_chat_id == "chat_nested"


# ======================================================================
# Programming handler LockConflictError tests
# ======================================================================


class TestProgrammingHandlerLockConflict:
    """CocoModeHandler.handle_message catches LockConflictError."""

    def test_lock_conflict_sends_card(self):
        ctx = _make_handler_context()
        ctx.settings.coco_execution_timeout = 30
        ctx.settings.card.collapsible_enabled = False

        h = CocoModeHandler(ctx)
        h.reply_text = MagicMock()
        h.reply_card = MagicMock(return_value="reply_1")
        h.add_reaction = MagicMock()
        h.register_message_project = MagicMock()
        h.record_mode_transition = MagicMock()
        h.inject_bridge_context = MagicMock(side_effect=lambda t, p, **kw: t)
        h.get_working_dir = MagicMock(return_value="/tmp/test")
        h.ensure_request_id = MagicMock(return_value="req-1")

        mock_project = MagicMock()
        mock_project.project_id = "proj-1"
        mock_project.root_path = "/tmp/test"

        # Make repo_lock_manager.acquire return failure (conflict)
        from src.repo_lock import AcquireResult
        mock_repo_lock = MagicMock()
        mock_repo_lock.acquire.return_value = AcquireResult(
            success=False, holder_chat_id="chat_other",
            locked_since=0.0, last_active_time=0.0,
        )
        ctx.repo_lock_manager = mock_repo_lock
        h.send_lock_conflict_card = MagicMock()

        mock_session = MagicMock()
        mock_session.session_id = "sess-1"
        mock_session.message_count = 1
        mock_session.last_query = "test"
        ctx.coco_manager.get_session.return_value = mock_session

        with patch("src.thread.get_current_is_p2p", return_value=False):
            h.handle_message("msg-1", "chat-1", "hello", project=mock_project)

        h.send_lock_conflict_card.assert_called_once()
        from src.repo_lock import LockConflictError
        err = h.send_lock_conflict_card.call_args[0][0]
        assert isinstance(err, LockConflictError)
        assert err.holder_chat_id == "chat_other"
        assert h.send_lock_conflict_card.call_args.kwargs["chat_id"] == "chat-1"


# ---------------------------------------------------------------------------
# AC-R05 / AC-R06: Non-streaming heartbeat for repo lock
# ---------------------------------------------------------------------------


class TestNonStreamingHeartbeat:
    """Verify that the non-streaming fallback path in ProgrammingModeHandler
    keeps the repo lock alive via periodic Event+Thread heartbeat and cleans up."""

    def _make_handler_and_mocks(self):
        """Create a minimal CocoModeHandler with mocked dependencies."""
        from unittest.mock import MagicMock, PropertyMock, patch

        from src.feishu.handlers.programming import CocoModeHandler

        ctx = MagicMock()
        ctx.settings = MagicMock()
        ctx.settings.coco_execution_timeout = 600
        ctx.settings.card.collapsible_enabled = False
        ctx.api_client_factory = MagicMock()

        with patch.object(CocoModeHandler, "settings", new_callable=PropertyMock, return_value=ctx.settings):
            handler = CocoModeHandler.__new__(CocoModeHandler)
            handler.ctx = ctx
            handler.im_client = MagicMock()
            handler._settings = ctx.settings

        handler.reply_text = MagicMock()
        handler.send_error_card = MagicMock()

        mock_session = MagicMock()
        mock_renderer = MagicMock()
        mock_renderer.get_final_content.return_value = "done"
        mock_streaming_mgr = MagicMock()

        return handler, mock_session, mock_renderer, mock_streaming_mgr, ctx

    def test_touch_called_during_blocking_prompt(self):
        """AC-R05: touch() is called at least once during a blocking send_prompt."""
        import threading as _threading
        from unittest.mock import MagicMock, patch

        handler, mock_session, mock_renderer, mock_streaming_mgr, ctx = self._make_handler_and_mocks()

        repo_lock_mgr = MagicMock()
        root_path = "/tmp/test_repo"

        # Make send_prompt block for ~1 second (enough for 0.2s interval)
        _done = _threading.Event()

        def blocking_prompt(*args, **kwargs):
            _done.wait(timeout=3)
            result = MagicMock()
            result.text = "ok"
            return result

        mock_session.send_prompt = blocking_prompt

        mock_streaming_mgr.create_streaming_card.return_value = None
        handler.get_streaming_manager = MagicMock(return_value=mock_streaming_mgr)
        handler.thinking_text = "🤔"
        handler._get_interaction_mode = MagicMock()
        handler.mode_name = "Coco"
        handler.is_coco = True
        handler.ensure_request_id = MagicMock(return_value="req-1")

        # Patch Event.wait to use a short interval so heartbeat fires quickly
        _orig_event_wait = _threading.Event.wait

        def _fast_wait(self_event, timeout=None):
            if timeout and timeout >= 10:  # only shorten the 30s heartbeat wait
                return _orig_event_wait(self_event, timeout=0.2)
            return _orig_event_wait(self_event, timeout=timeout)

        with patch.object(_threading.Event, "wait", _fast_wait):
            # Release after 0.4 second so touch fires at least once at 0.2s
            def release_later():
                import time
                time.sleep(0.4)
                _done.set()
            t = _threading.Thread(target=release_later, daemon=True)
            t.start()

            handler.handle_response(
                "msg-1", "chat-1", "hello", mock_session, None, "/tmp", "/tmp",
                _repo_lock_mgr=repo_lock_mgr, _root_path=root_path,
            )
            t.join(timeout=5)

        assert repo_lock_mgr.touch.call_count >= 1, (
            f"Expected touch() called at least once, got {repo_lock_mgr.touch.call_count}"
        )

    def test_heartbeat_thread_joined_on_success(self):
        """AC-R06: Heartbeat thread is stopped (Event.set + join) after send_prompt returns."""
        import threading as _threading
        from unittest.mock import MagicMock

        handler, mock_session, mock_renderer, mock_streaming_mgr, ctx = self._make_handler_and_mocks()

        repo_lock_mgr = MagicMock()
        root_path = "/tmp/test_repo"

        # send_prompt returns immediately
        mock_session.send_prompt.return_value = MagicMock(text="ok")

        mock_streaming_mgr.create_streaming_card.return_value = None
        handler.get_streaming_manager = MagicMock(return_value=mock_streaming_mgr)
        handler.thinking_text = "🤔"
        handler._get_interaction_mode = MagicMock()
        handler.mode_name = "Coco"
        handler.is_coco = True
        handler.ensure_request_id = MagicMock(return_value="req-1")

        # Track threads started during handle_response
        _threads_before = set(_threading.enumerate())

        handler.handle_response(
            "msg-1", "chat-1", "hello", mock_session, None, "/tmp", "/tmp",
            _repo_lock_mgr=repo_lock_mgr, _root_path=root_path,
        )

        # After handle_response, all heartbeat threads should have been joined
        import time as _time
        _deadline = _time.monotonic() + 2.0
        while _time.monotonic() < _deadline:
            _threads_after = set(_threading.enumerate())
            _new_threads = _threads_after - _threads_before
            hb_threads = [t for t in _new_threads if t.is_alive() and "heartbeat" in t.name.lower()]
            if not hb_threads:
                break
            _time.sleep(0.01)
        else:
            hb_threads = [t for t in (set(_threading.enumerate()) - _threads_before) if t.is_alive() and "heartbeat" in t.name.lower()]
        assert len(hb_threads) == 0, f"Heartbeat thread still alive: {hb_threads}"

    def test_heartbeat_thread_joined_on_exception(self):
        """AC-R06: Heartbeat thread is stopped even when send_prompt raises."""
        import threading as _threading
        from unittest.mock import MagicMock

        handler, mock_session, mock_renderer, mock_streaming_mgr, ctx = self._make_handler_and_mocks()

        repo_lock_mgr = MagicMock()
        root_path = "/tmp/test_repo"

        mock_session.send_prompt.side_effect = RuntimeError("boom")

        mock_streaming_mgr.create_streaming_card.return_value = None
        handler.get_streaming_manager = MagicMock(return_value=mock_streaming_mgr)
        handler.thinking_text = "🤔"
        handler._get_interaction_mode = MagicMock()
        handler.mode_name = "Coco"
        handler.is_coco = True
        handler.ensure_request_id = MagicMock(return_value="req-1")

        _threads_before = set(_threading.enumerate())

        handler.handle_response(
            "msg-1", "chat-1", "hello", mock_session, None, "/tmp", "/tmp",
            _repo_lock_mgr=repo_lock_mgr, _root_path=root_path,
        )

        import time as _time
        _deadline = _time.monotonic() + 2.0
        while _time.monotonic() < _deadline:
            _threads_after = set(_threading.enumerate())
            _new_threads = _threads_after - _threads_before
            hb_threads = [t for t in _new_threads if t.is_alive() and "heartbeat" in t.name.lower()]
            if not hb_threads:
                break
            _time.sleep(0.01)
        else:
            hb_threads = [t for t in (set(_threading.enumerate()) - _threads_before) if t.is_alive() and "heartbeat" in t.name.lower()]
        assert len(hb_threads) == 0, f"Heartbeat thread still alive after exception: {hb_threads}"

    def test_no_heartbeat_thread_when_no_lock_mgr(self):
        """No heartbeat thread is started when _repo_lock_mgr is None."""
        import threading as _threading
        from unittest.mock import MagicMock

        handler, mock_session, mock_renderer, mock_streaming_mgr, ctx = self._make_handler_and_mocks()

        mock_session.send_prompt.return_value = MagicMock(text="ok")

        mock_streaming_mgr.create_streaming_card.return_value = None
        handler.get_streaming_manager = MagicMock(return_value=mock_streaming_mgr)
        handler.thinking_text = "🤔"
        handler._get_interaction_mode = MagicMock()
        handler.mode_name = "Coco"
        handler.is_coco = True
        handler.ensure_request_id = MagicMock(return_value="req-1")

        _threads_before = set(_threading.enumerate())

        handler.handle_response(
            "msg-1", "chat-1", "hello", mock_session, None, "/tmp", "/tmp",
            _repo_lock_mgr=None, _root_path=None,
        )

        import time as _time
        _deadline = _time.monotonic() + 2.0
        while _time.monotonic() < _deadline:
            _threads_after = set(_threading.enumerate())
            _new_threads = _threads_after - _threads_before
            hb_threads = [t for t in _new_threads if t.is_alive() and "heartbeat" in t.name.lower()]
            if not hb_threads:
                break
            _time.sleep(0.01)
        else:
            hb_threads = [t for t in (set(_threading.enumerate()) - _threads_before) if t.is_alive() and "heartbeat" in t.name.lower()]
        assert len(hb_threads) == 0, "No heartbeat thread should be created when lock mgr is None"


# ======================================================================
# AC-18: /help always shows lock section even when admin_user_ids empty
# ======================================================================


class TestHelpCardLockAlwaysVisible:
    """F-20/AC-18: lock_enabled=True ensures /help always contains lock section."""

    def test_slock_section_present_in_main_help(self):
        """The main /help card should expose Slock mode and its core workflow."""
        from src.card.builders.system import SystemBuilder

        _msg_type, card_json = SystemBuilder.build_help_card(
            project=None,
            category="main",
            is_admin=False,
            lock_enabled=False,
            chat_id="",
            session_idle_timeout=600,
            session_idle_warn_at_remaining=120,
            lock_undo_window_seconds=300,
        )
        assert "Slock" in card_json or "slock" in card_json
        assert "/slock" in card_json
        assert "/hire" in card_json or "/role add" in card_json
        assert "/task status" in card_json

    def test_lock_section_present_when_no_admin_ids(self):
        """Even with admin_user_ids=frozenset(), the help card includes lock content."""
        from src.card.builders.system import SystemBuilder

        # lock_enabled=True is now hardcoded in system.py; verify the card builder
        # produces a lock section regardless of admin status.
        _msg_type, card_json = SystemBuilder.build_help_card(
            project=None,
            category="main",
            is_admin=False,
            lock_enabled=True,
            chat_id="",
            session_idle_timeout=600,
            session_idle_warn_at_remaining=120,
            lock_undo_window_seconds=300,
        )
        # The non-admin lock section title should be present
        assert "群锁定" in card_json

    def test_lock_section_absent_when_lock_disabled(self):
        """Baseline: lock_enabled=False should NOT include the lock section."""
        from src.card.builders.system import SystemBuilder

        _msg_type, card_json = SystemBuilder.build_help_card(
            project=None,
            category="main",
            is_admin=False,
            lock_enabled=False,
            chat_id="",
            session_idle_timeout=600,
            session_idle_warn_at_remaining=120,
            lock_undo_window_seconds=300,
        )
        assert "群锁定" not in card_json


# ======================================================================
# AC-16: _collect_lock_conflict_context auto-detects same-sender
# ======================================================================


class TestSameSenderAutoDetection:
    """F-18/AC-16: _collect_lock_conflict_context sets is_same_sender automatically."""

    def test_same_sender_detected(self):
        """When current sender matches lock holder's last_sender_id → is_same_sender=True."""
        from src.feishu.handlers.lock_helper import LockHelper
        from src.repo_lock import LockConflictError

        mock_handler = MagicMock()
        # Set up repo_lock_manager mock
        mock_repo_lock_mgr = MagicMock()
        mock_handler.ctx = MagicMock()
        mock_handler.ctx.repo_lock_manager = mock_repo_lock_mgr
        mock_handler.ctx.chat_lock_manager = MagicMock()
        mock_handler.ctx.chat_lock_manager.is_admin.return_value = False

        # Lock info where last_sender_id == current sender
        mock_lock_info = MagicMock()
        mock_lock_info.last_sender_id = "user_123"
        mock_repo_lock_mgr.get_lock_info.return_value = mock_lock_info
        mock_repo_lock_mgr.path_to_token.return_value = "tok_abc"

        helper = LockHelper(mock_handler)
        err = LockConflictError(
            "conflict", holder_chat_id="chat_other", locked_since=100.0,
            root_path="/tmp/repo", last_active_time=200.0,
        )

        with patch("src.thread.get_current_sender_id", return_value="user_123"), \
             patch("src.config.get_settings") as mock_settings:
            mock_settings.return_value.app_id = "app_test"
            ctx = helper._collect_lock_conflict_context(err)

        assert ctx.is_same_sender is True
        assert ctx.sender_id == "user_123"

    def test_different_sender_not_flagged(self):
        """When current sender differs from lock holder → is_same_sender=False."""
        from src.feishu.handlers.lock_helper import LockHelper
        from src.repo_lock import LockConflictError

        mock_handler = MagicMock()
        mock_repo_lock_mgr = MagicMock()
        mock_handler.ctx = MagicMock()
        mock_handler.ctx.repo_lock_manager = mock_repo_lock_mgr
        mock_handler.ctx.chat_lock_manager = MagicMock()
        mock_handler.ctx.chat_lock_manager.is_admin.return_value = False

        mock_lock_info = MagicMock()
        mock_lock_info.last_sender_id = "user_999"
        mock_repo_lock_mgr.get_lock_info.return_value = mock_lock_info
        mock_repo_lock_mgr.path_to_token.return_value = "tok_xyz"

        helper = LockHelper(mock_handler)
        err = LockConflictError(
            "conflict", holder_chat_id="chat_other", locked_since=100.0,
            root_path="/tmp/repo", last_active_time=200.0,
        )

        with patch("src.thread.get_current_sender_id", return_value="user_123"), \
             patch("src.config.get_settings") as mock_settings:
            mock_settings.return_value.app_id = "app_test"
            ctx = helper._collect_lock_conflict_context(err)

        assert ctx.is_same_sender is False

    def test_context_has_no_chat_hint_field(self):
        """chat_hint dead field has been removed from _LockConflictContext."""
        from src.feishu.handlers.lock_helper import LockHelper
        from src.repo_lock import LockConflictError

        mock_handler = MagicMock()
        mock_handler.ctx = MagicMock()
        mock_handler.ctx.repo_lock_manager = MagicMock()
        mock_handler.ctx.repo_lock_manager.get_lock_info.return_value = None
        mock_handler.ctx.repo_lock_manager.path_to_token.return_value = ""
        mock_handler.ctx.chat_lock_manager = MagicMock()
        mock_handler.ctx.chat_lock_manager.is_admin.return_value = False

        helper = LockHelper(mock_handler)
        err = LockConflictError(
            "conflict", holder_chat_id="chat_other", locked_since=100.0,
            root_path="/tmp/repo", last_active_time=200.0,
        )

        with patch("src.thread.get_current_sender_id", return_value="user_123"), \
             patch("src.config.get_settings") as mock_settings:
            mock_settings.return_value.app_id = "app_test"
            ctx = helper._collect_lock_conflict_context(err)

        assert not hasattr(ctx, "chat_hint")
