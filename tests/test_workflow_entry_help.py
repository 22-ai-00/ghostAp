"""Tests for workflow entry-card messaging, help card, and unknown-command handling.

Validates that:
- ``show_workflow_help`` replies with a Schema 2.0 card instead of raw Markdown text
- the help card mentions key commands and describes the current execution flow
- ``handle_workflow_command`` with an unknown subcommand produces an error message
  that points to ``/wf_help`` and lists major commands
- ``handle_show_workflow_menu`` (entry-card "开始" button) now launches the
  agent-selection flow directly instead of prompting the user to type
  ``/wf <需求>`` manually (the previous text-only response)
"""

import json
import unittest
from unittest.mock import MagicMock, patch


class TestWorkflowHelpText(unittest.TestCase):
    """Validate the structured Workflow help card."""

    def _make_handler(self):
        from src.feishu.handlers.workflow import WorkflowHandler

        handler = WorkflowHandler.__new__(WorkflowHandler)
        handler.ctx = MagicMock()
        handler.reply_text = MagicMock()
        handler.reply_error = MagicMock()
        handler.reply_card = MagicMock()
        handler.send_card_to_chat = MagicMock()
        handler.update_card = MagicMock()
        handler.get_working_dir = MagicMock(return_value="/tmp")
        handler.get_engine_name = MagicMock(return_value="coco")
        handler.ensure_request_id = MagicMock(return_value="req_1")
        return handler

    def test_help_mentions_core_commands(self):
        handler = self._make_handler()
        handler.show_workflow_help("msg")
        handler.reply_text.assert_not_called()
        handler.reply_card.assert_called_once()

        args, _kwargs = handler.reply_card.call_args
        card = args[1]
        self.assertEqual(card["schema"], "2.0")
        self.assertEqual(card["header"]["title"]["content"], "⚡ Workflow 使用帮助")
        self.assertEqual(card["header"]["template"], "turquoise")

        panels = [
            element
            for element in card["body"]["elements"]
            if element.get("tag") == "collapsible_panel"
        ]
        self.assertEqual(len(panels), 3)
        self.assertTrue(panels[0]["expanded"])
        self.assertTrue(all(not panel["expanded"] for panel in panels[1:]))

        text = json.dumps(card, ensure_ascii=False)
        # These are the commands we expect to appear in the help output.
        for snippet in ("/wf", "/wf_help", "/wf_status", "/stop_wf", "/wf_save", "/wf_list"):
            self.assertIn(snippet, text, f"help text missing command {snippet!r}")

    def test_help_flow_matches_new_execution_steps(self):
        """The execution panel must describe the current three-step orchestrator flow."""
        handler = self._make_handler()
        handler.show_workflow_help("msg")
        args, _kwargs = handler.reply_card.call_args
        text = json.dumps(args[1], ensure_ascii=False)
        self.assertIn("执行流程", text)
        # Assert current descriptions are present (Orchestrator Agent → Review Agent → Auto execute).
        for keyword in ("Agent", "工具", "自动生成", "执行"):
            self.assertIn(keyword, text, f"execution flow missing keyword {keyword!r}")


class TestWorkflowUnknownCommandMessaging(unittest.TestCase):
    """Validate unknown-command handler behavior."""

    def _make_handler(self):
        from src.feishu.handlers.workflow import WorkflowHandler

        handler = WorkflowHandler.__new__(WorkflowHandler)
        handler.ctx = MagicMock()
        handler.reply_text = MagicMock()
        handler.reply_error = MagicMock()
        handler.reply_card = MagicMock()
        handler.send_card_to_chat = MagicMock()
        handler.update_card = MagicMock()
        handler.get_working_dir = MagicMock(return_value="/tmp")
        handler.get_engine_name = MagicMock(return_value="coco")
        handler.ensure_request_id = MagicMock(return_value="req_1")
        return handler

    def test_unknown_command_uses_error_card(self):
        """Unknown commands should surface the unified error card (reply_card)."""
        handler = self._make_handler()
        handler.handle_workflow_command("msg", "chat", "/wf_does_not_exist", None)
        handler.reply_card.assert_called()

    def test_unknown_command_points_to_help(self):
        """Unknown-command error must mention /wf_help so the user knows where to look."""
        handler = self._make_handler()
        handler.handle_workflow_command("msg", "chat", "/wf_does_not_exist", None)
        args, kwargs = handler.reply_card.call_args
        # reply_card signature: (message_id, card_content)
        card = args[1]
        text = str(card)
        self.assertIn("/wf_help", text, "unknown-command error must reference /wf_help")

    def test_unknown_command_lists_commands(self):
        """Unknown-command error must list the main /wf subcommands."""
        handler = self._make_handler()
        handler.handle_workflow_command("msg", "chat", "/wf_does_not_exist", None)
        args, kwargs = handler.reply_card.call_args
        text = str(args[1])
        for cmd in ("/wf", "/wf_save", "/wf_list", "/wf_status", "/stop_wf"):
            self.assertIn(cmd, text, f"unknown-command error must list {cmd!r}")


class TestWorkflowEntryCardButton(unittest.TestCase):
    """Validate the entry-card start button launches default generation."""

    def _make_handler(self):
        from src.feishu.handlers.workflow import WorkflowHandler

        handler = WorkflowHandler.__new__(WorkflowHandler)
        handler.ctx = MagicMock()
        handler.reply_text = MagicMock()
        handler.reply_error = MagicMock()
        handler.reply_card = MagicMock()
        handler.send_card_to_chat = MagicMock()
        handler.update_card = MagicMock()
        handler.get_working_dir = MagicMock(return_value="/tmp")
        handler.get_engine_name = MagicMock(return_value="coco")
        handler.ensure_request_id = MagicMock(return_value="req_1")
        handler._schedule_generate_and_show_confirm_card = MagicMock()
        return handler

    @patch("src.feishu.handlers.workflow.WorkflowHandler._show_agent_selection_card")
    def test_start_button_launches_default_generation(self, show_agent_select):
        """Clicking "开始" skips manual selection and schedules generation."""
        handler = self._make_handler()

        from src.workflow_engine import bridge as bridge_module
        from src.workflow_engine.models import WorkflowProject, WorkflowStatus

        original = getattr(bridge_module.RuntimeBridge, "check_node_available", None)
        try:
            bridge_module.RuntimeBridge.check_node_available = staticmethod(lambda: True)

            fake_manager = MagicMock()
            fake_manager.get.return_value = None
            engine = MagicMock()
            engine.project = WorkflowProject()
            fake_manager.get_or_create.return_value = engine
            handler.ctx.workflow_engine_manager = fake_manager

            handler.handle_show_workflow_menu(
                message_id="msg",
                chat_id="chat",
                project_id="",
                value={
                    "action": "show_workflow_menu",
                    "chat_id": "chat",
                    "project_id": "",
                    "requirement": "编写登录接口测试",
                },
            )

            show_agent_select.assert_not_called()
            self.assertEqual(
                engine.project.status,
                WorkflowStatus.GENERATING_SCRIPT,
            )
            self.assertTrue(engine.project.pending.auto_reviewer)
            handler._schedule_generate_and_show_confirm_card.assert_called_once()
        finally:
            if original is not None:
                bridge_module.RuntimeBridge.check_node_available = original

    def test_start_button_blocks_without_requirement(self):
        """Without a task description the entry button must NOT proceed to agent selection."""
        from src.workflow_engine import bridge as bridge_module

        original = getattr(bridge_module.RuntimeBridge, "check_node_available", None)
        try:
            bridge_module.RuntimeBridge.check_node_available = staticmethod(lambda: True)

            handler = self._make_handler()
            fake_manager = MagicMock()
            fake_manager.get.return_value = None
            handler.ctx.workflow_engine_manager = fake_manager

            with patch.object(
                handler, "_show_agent_selection_card"
            ) as show_agent_select:
                handler.handle_show_workflow_menu(
                    message_id="msg",
                    chat_id="chat",
                    project_id="",
                    value={"action": "show_workflow_menu", "chat_id": "chat", "project_id": ""},
                )
                show_agent_select.assert_not_called()

                # Must surface an error card rather than silently ignoring.
                handler.reply_card.assert_called()
        finally:
            if original is not None:
                bridge_module.RuntimeBridge.check_node_available = original

    def test_start_button_requires_node(self):
        """If Node.js is missing the start button should surface an error card."""
        from src.workflow_engine import bridge as bridge_module

        original = getattr(bridge_module.RuntimeBridge, "check_node_available", None)
        try:
            bridge_module.RuntimeBridge.check_node_available = staticmethod(lambda: False)

            handler = self._make_handler()
            fake_manager = MagicMock()
            fake_manager.get.return_value = None
            handler.ctx.workflow_engine_manager = fake_manager

            handler.handle_show_workflow_menu("msg", "chat", "", {})
            handler.reply_card.assert_called()
        finally:
            if original is not None:
                bridge_module.RuntimeBridge.check_node_available = original

    @patch("src.feishu.handlers.workflow.WorkflowHandler._show_agent_selection_card")
    def test_start_button_blocks_when_running(self, show_agent_select):
        """If a workflow is already running the start button should reject the request."""
        from src.workflow_engine import bridge as bridge_module

        original = getattr(bridge_module.RuntimeBridge, "check_node_available", None)
        try:
            bridge_module.RuntimeBridge.check_node_available = staticmethod(lambda: True)

            handler = self._make_handler()
            fake_engine = MagicMock()
            fake_engine.is_running = True
            fake_engine.project = None
            fake_manager = MagicMock()
            fake_manager.get.return_value = fake_engine
            handler.ctx.workflow_engine_manager = fake_manager

            handler.handle_show_workflow_menu("msg", "chat", "", {})
            show_agent_select.assert_not_called()
            handler.reply_card.assert_called()
        finally:
            if original is not None:
                bridge_module.RuntimeBridge.check_node_available = original


class TestWorkflowEntryCardBodyKeywords(unittest.TestCase):
    """Lightweight assertions on the /wf entry-card body text."""

    def test_entry_card_body_mentions_orchestrator_and_flow(self):
        """The entry-card body must advertise /wf as the top-level orchestrator agent."""
        from src.card.ui_text import UI_TEXT

        body = UI_TEXT["workflow_entry_body"]
        self.assertIn("主编排 Agent", body, "entry card must mention '主编排 Agent'")
        self.assertIn("工具", body, "entry card must mention '工具'")
        self.assertIn("模型", body, "entry card must mention '模型'")
        self.assertIn("评审 Agent", body, "entry card must mention '评审 Agent'")
        self.assertIn("自动生成", body, "entry card must mention '自动生成'")
        self.assertIn("执行", body, "entry card must mention '执行'")
        self.assertIn("/wf", body, "entry card must reference the /wf command")

    def test_help_text_preserves_primitives(self):
        """The entry-card body must still list all four primitives (agent/parallel/pipeline/phase)."""
        from src.card.ui_text import UI_TEXT

        body = UI_TEXT["workflow_entry_body"]
        for primitive in ("agent()", "parallel()", "pipeline()", "phase()"):
            self.assertIn(
                primitive,
                body,
                f"entry card body lost the primitive {primitive!r}",
            )


if __name__ == "__main__":
    unittest.main()
