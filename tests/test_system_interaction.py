import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.feishu.handlers.system import SystemHandler


def _collect_buttons(card: dict) -> list[dict]:
    buttons = []

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


def _card_text(card: dict) -> str:
    return json.dumps(card, ensure_ascii=False)


class TestSystemInteraction(unittest.TestCase):
    def setUp(self):
        self.mock_ctx = MagicMock()
        self.mock_ctx.settings.app_id = "test_app"
        self.mock_ctx.settings.app_secret = "test_secret"

        # Mock mode manager
        self.mock_ctx.mode_manager.get_mode.return_value = "smart"
        self.mock_ctx.mode_manager.is_coco_mode.return_value = False

        # Mock project manager
        self.mock_ctx.project_manager.get_active_project.return_value = None

        # Mock working dir
        self.mock_ctx.working_dirs = {}

        self.handler = SystemHandler(self.mock_ctx)

        # Mock BaseHandler methods
        self.handler.reply_card = MagicMock()
        self.handler.reply_text = MagicMock()
        self.handler.update_card = MagicMock()
        self.handler.get_working_dir = MagicMock(return_value="/tmp")

    def test_show_full_help(self):
        """测试 /help 命令生成并发送交互式卡片"""
        # Execute
        self.handler.show_full_help("msg_1", "chat_1")

        # Verify reply_card called (production uses self.reply_card)
        self.handler.reply_card.assert_called_once()
        args = self.handler.reply_card.call_args
        msg_id = args[0][0]
        content = args[0][1]

        self.assertEqual(msg_id, "msg_1")

        # Verify content structure
        card = json.loads(content)
        self.assertIn("header", card)
        self.assertEqual(card["header"]["title"]["content"], "📖 GhostAP 使用帮助")

        # The full help is documentation-only: quick-action buttons belong in /menu.
        self.assertEqual(_collect_buttons(card), [])
        card_text = _card_text(card)
        self.assertNotIn("常用操作", card_text)
        self.assertNotIn("GhostAP 使用帮助 v", card_text)

    def test_handle_menu_command(self):
        """测试 /menu 命令生成并发送菜单卡片"""
        # Execute
        self.handler.handle_menu_command("msg_2", "chat_1")

        # Verify reply_card called
        self.handler.reply_card.assert_called_once()
        args = self.handler.reply_card.call_args
        content = args[0][1]

        # Verify content structure
        card = json.loads(content)
        self.assertEqual(card["header"]["title"]["content"], "📱 快捷菜单")

        # Verify buttons - search all elements and all column sets
        elements = card["body"]["elements"]
        found_actions = []
        for elem in elements:
            if elem.get("tag") == "column_set":
                for col in elem.get("columns", []):
                    for sub_elem in col.get("elements", []):
                        if sub_elem.get("tag") == "button":
                            val = sub_elem.get("value", {})
                            found_actions.append(val.get("action"))
            elif elem.get("tag") == "action":  # Also check action sets if used
                for action in elem.get("actions", []):
                    if action.get("tag") == "button":
                        val = action.get("value", {})
                        found_actions.append(val.get("action"))

        self.assertIn("new_project_prompt", found_actions)
        self.assertIn("switch_project", found_actions)

    def test_handle_help_category_patch(self):
        """测试帮助分类切换使用 Patch 更新卡片"""
        # Mock update_card to succeed
        self.handler.update_card.return_value = True

        # Execute
        self.handler.handle_help_category("msg_3", "chat_1", "deep", origin_message_id="origin_msg_id")

        # Verify update_card used
        self.handler.update_card.assert_called_once()
        self.handler.reply_card.assert_not_called()

        # Verify content contains Deep specific help
        args = self.handler.update_card.call_args
        content = args[0][1]
        self.assertIn("Deep Engine", content)

    def test_handle_help_category_reply_fallback(self):
        """测试 Patch 失败时回退到 Reply"""
        # Mock update_card to fail
        self.handler.update_card.return_value = False

        # Execute
        self.handler.handle_help_category("msg_4", "chat_1", "project", origin_message_id="origin_msg_id")

        # Verify update_card called but failed
        self.handler.update_card.assert_called_once()

        # Verify reply_card called as fallback
        self.handler.reply_card.assert_called_once()

        # Verify content contains Project specific help
        args = self.handler.reply_card.call_args
        content = args[0][1]
        self.assertIn("项目管理", content)

    def test_help_more_category_lists_full_spec_commands(self):
        self.handler.update_card.return_value = True

        self.handler.handle_help_category("msg_5", "chat_1", "more", origin_message_id="origin_msg_id")

        args = self.handler.update_card.call_args
        content = args[0][1]
        self.assertIn("/spec_pause", content)
        self.assertIn("/spec_resume", content)
        self.assertIn("/spec_metrics", content)
        self.assertIn("/spec_save", content)
        self.assertIn("/spec_export", content)
        self.assertIn("/spec_recover", content)


class TestUnifiedErrorCardPaths(unittest.TestCase):
    """Regression coverage for the three production error-card entry paths."""

    def _make_ctx(self):
        ctx = MagicMock()
        ctx.settings.app_id = "test_app"
        ctx.settings.app_secret = "test_secret"
        ctx.settings.ref_note_enabled = False
        ctx.settings.spec_execution_timeout = 3600
        return ctx

    def _assert_error_contract(self, card: dict, *, summary: str, detail_action: str, retry_action: str | None):
        text = _card_text(card)
        values = [button.get("value", {}) for button in _collect_buttons(card)]

        self.assertEqual(card["header"]["template"], "red")
        self.assertIn("错误", card["header"]["title"]["content"])
        self.assertIn(summary, text)
        self.assertIn("错误摘要", text)
        self.assertIn("详情已收起", text)
        self.assertNotIn("**详细信息**", text)
        self.assertTrue(any(value.get("action") == detail_action for value in values))
        detail_values = [value for value in values if value.get("action") == detail_action]
        self.assertTrue(detail_values)
        self.assertTrue(detail_values[0].get("diagnostic_token"))
        self.assertNotIn("details", detail_values[0])
        if retry_action is None:
            self.assertFalse(any(str(value.get("action", "")).endswith("_resume") for value in values))
        else:
            self.assertTrue(any(value.get("action") == retry_action for value in values))

    def test_base_handler_error_path_uses_detail_action_without_unrecoverable_retry(self):
        from src.feishu.handlers.base import BaseHandler

        handler = BaseHandler(self._make_ctx())
        handler.reply_card = MagicMock()

        handler.send_error_card(
            chat_id="chat-base",
            exc=RuntimeError("base boom"),
            title="系统错误",
            origin_message_id="msg-base",
        )

        handler.reply_card.assert_called_once()
        card = json.loads(handler.reply_card.call_args.args[1])
        self._assert_error_contract(
            card,
            summary="base boom",
            detail_action="show_error_details",
            retry_action=None,
        )

    def test_spec_handler_error_path_builds_contextual_detail_and_retry_actions(self):
        from src.feishu.handlers.spec import SpecHandler
        from src.feishu.renderers.spec_renderer import SpecRenderer

        handler = SpecHandler(self._make_ctx(), renderer=MagicMock())
        handler.get_card_delivery = MagicMock(return_value=MagicMock())
        handler.renderer = SpecRenderer(handler)
        handler.send_card_to_chat = MagicMock()
        project = SimpleNamespace(project_id="proj-1", project_name="Demo", root_path="/repo/demo")

        handler._on_engine_error(
            RuntimeError("spec boom"),
            task_id="task-spec",
            chat_id="chat-spec",
            message_id="msg-spec",
            project=project,
            engine_name="Coco",
            reporter=MagicMock(),
            request_id="req-spec",
        )

        handler.send_card_to_chat.assert_called_once()
        card = json.loads(handler.send_card_to_chat.call_args.args[1])
        self._assert_error_contract(
            card,
            summary="spec boom",
            detail_action="show_error_details",
            retry_action="spec_resume",
        )

    def test_engine_base_error_path_dispatches_contextual_failed_card(self):
        from src.card.session.factory import CardSessionFactory
        from src.card.state.models import CardMetadata
        from src.feishu.handlers.engine_base import BaseEngineHandler

        class DummyEngineHandler(BaseEngineHandler):
            def _get_engine_manager(self):
                raise NotImplementedError

            def _get_engine_name_prefix(self) -> str:
                return "Deep"

            def _get_task_type(self) -> str:
                return "deep_engine"

            def _show_status(self, message_id, chat_id, project=None):
                raise NotImplementedError

            def _create_callbacks(self, message_id, chat_id, project, engine_name, root_path):
                raise NotImplementedError

        session = CardSessionFactory(MagicMock()).create_snapshot(
            metadata=CardMetadata(engine_type="deep", mode_name="Deep", mode_emoji="🧠", tool_name="Coco")
        )
        handler = DummyEngineHandler(self._make_ctx())
        handler.renderer = SimpleNamespace(get_active_session=lambda: session)
        reporter = MagicMock()
        reporter.format_error.side_effect = lambda msg: f"formatted: {msg}"

        handler._on_engine_error(
            RuntimeError("deep boom"),
            task_id="task-deep",
            chat_id="chat-deep",
            message_id="msg-deep",
            project=None,
            engine_name="Coco",
            reporter=reporter,
            request_id="req-deep",
            action_prefix="deep",
        )

        _, card_json = session.snapshot()
        self._assert_error_contract(
            json.loads(card_json),
            summary="deep boom",
            detail_action="show_error_details",
            retry_action="deep_resume",
        )


if __name__ == "__main__":
    unittest.main()
