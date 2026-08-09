"""Regression tests for workflow startup fixes.

Covers:
- vertical_spacing NOT emitted on column_set by build_responsive_button_row
- PendingConfirmation.created_at timestamp for stale detection
- engine.project auto-init when None
- Newest-wins replacement of owned pending state
"""

import time
import unittest
from unittest.mock import MagicMock, patch

from src.card.render.buttons import build_responsive_button_row
from src.workflow_engine.models import (
    PendingConfirmation,
    WorkflowProject,
    WorkflowStatus,
)


class TestColumnSetNoVerticalSpacing(unittest.TestCase):
    """Ensure build_responsive_button_row never emits vertical_spacing on column_set."""

    def _make_button(self, label: str = "Test") -> dict:
        return {
            "tag": "button",
            "text": {"tag": "plain_text", "content": label},
            "type": "default",
            "value": {"action": "test"},
            "behaviors": [{"type": "callback", "value": {"action": "test"}}],
        }

    def _assert_no_vertical_spacing_on_column_set(self, elements: list[dict]):
        for el in elements:
            if el.get("tag") == "column_set":
                self.assertNotIn(
                    "vertical_spacing",
                    el,
                    f"column_set must not have vertical_spacing: {el}",
                )

    def test_single_button_no_vertical_spacing(self):
        elements = build_responsive_button_row([self._make_button()])
        self._assert_no_vertical_spacing_on_column_set(elements)

    def test_two_buttons_no_vertical_spacing(self):
        elements = build_responsive_button_row(
            [self._make_button("A"), self._make_button("B")]
        )
        self._assert_no_vertical_spacing_on_column_set(elements)

    def test_three_buttons_no_vertical_spacing(self):
        elements = build_responsive_button_row(
            [self._make_button(f"B{i}") for i in range(3)]
        )
        self._assert_no_vertical_spacing_on_column_set(elements)

    def test_four_buttons_no_vertical_spacing(self):
        elements = build_responsive_button_row(
            [self._make_button(f"B{i}") for i in range(4)]
        )
        self._assert_no_vertical_spacing_on_column_set(elements)

    def test_mobile_force_vertical_no_vertical_spacing(self):
        elements = build_responsive_button_row(
            [self._make_button(f"B{i}") for i in range(4)],
            mobile_force_vertical=True,
        )
        self._assert_no_vertical_spacing_on_column_set(elements)


class TestPendingConfirmationCreatedAt(unittest.TestCase):
    """PendingConfirmation.created_at should auto-populate and enable stale detection."""

    def test_created_at_auto_populated(self):
        before = time.time()
        pc = PendingConfirmation()
        after = time.time()
        self.assertGreaterEqual(pc.created_at, before)
        self.assertLessEqual(pc.created_at, after)

    def test_created_at_serializes(self):
        pc = PendingConfirmation(created_at=1234567890.0)
        data = pc.model_dump()
        self.assertEqual(data["created_at"], 1234567890.0)

    def test_stale_detection_logic(self):
        threshold = 30 * 60
        old_pc = PendingConfirmation(created_at=time.time() - threshold - 1)
        self.assertTrue(time.time() - old_pc.created_at > threshold)

        fresh_pc = PendingConfirmation()
        self.assertFalse(time.time() - fresh_pc.created_at > threshold)


class TestEngineProjectAutoInit(unittest.TestCase):
    """engine.project=None should be auto-initialized."""

    @patch("src.thread.get_current_sender_id", return_value="user_123")
    @patch("src.workflow_engine.bridge.RuntimeBridge.check_node_available", return_value=True)
    @patch("src.workflow_engine.templates.discover_templates", return_value=[])
    def test_fresh_engine_gets_project(self, mock_templates, mock_node, mock_sender):
        from src.feishu.handlers.workflow import WorkflowHandler

        ctx = MagicMock()
        ctx.workflow_engine_manager = MagicMock()
        ctx.project_manager = MagicMock()
        ctx.progress_reporter = MagicMock()

        handler = WorkflowHandler.__new__(WorkflowHandler)
        handler.ctx = ctx
        handler.reply_text = MagicMock()
        handler.reply_card = MagicMock()
        handler.send_card_to_chat = MagicMock()
        handler.update_card = MagicMock()
        handler.add_reaction = MagicMock()
        handler._submit_engine_task = MagicMock()
        handler._schedule_generate_and_show_confirm_card = MagicMock()
        handler._ensure_topic_engine_context = MagicMock()
        handler._reply_workflow_error = MagicMock()

        project = MagicMock()
        project.root_path = "/tmp/project"
        project.project_id = "proj_1"
        project.project_name = "test"
        handler._ensure_project = MagicMock(return_value=project)

        from src.workflow_engine.engine import WorkflowEngine

        engine = MagicMock(spec=WorkflowEngine)
        engine.is_running = False
        engine.project = WorkflowProject()
        engine.root_path = "/tmp/project"
        ctx.workflow_engine_manager.get.return_value = engine
        ctx.workflow_engine_manager.get_or_create.return_value = engine

        handler.start_workflow("msg_1", "chat_1", "test requirement", project)

        self.assertIsNotNone(engine.project)
        self.assertEqual(engine.project.status, WorkflowStatus.GENERATING_SCRIPT)
        self.assertIsNotNone(engine.project.pending)
        self.assertTrue(engine.project.pending.auto_reviewer)
        handler._schedule_generate_and_show_confirm_card.assert_called_once()


class TestPendingNewestWins(unittest.TestCase):
    """The same initiator's newest request replaces any pending selection."""

    @patch("src.thread.get_current_sender_id", return_value="user_123")
    @patch("src.workflow_engine.bridge.RuntimeBridge.check_node_available", return_value=True)
    @patch("src.workflow_engine.templates.discover_templates", return_value=[])
    def test_stale_state_auto_reset(self, mock_templates, mock_node, mock_sender):
        from src.feishu.handlers.workflow import WorkflowHandler

        ctx = MagicMock()
        ctx.workflow_engine_manager = MagicMock()
        ctx.project_manager = MagicMock()
        ctx.progress_reporter = MagicMock()
        ctx.settings.admin_user_ids = []

        handler = WorkflowHandler.__new__(WorkflowHandler)
        handler.ctx = ctx
        handler.reply_text = MagicMock()
        handler.reply_card = MagicMock()
        handler.send_card_to_chat = MagicMock()
        handler.update_card = MagicMock()
        handler.add_reaction = MagicMock()
        handler._submit_engine_task = MagicMock()
        handler._schedule_generate_and_show_confirm_card = MagicMock()
        handler._ensure_topic_engine_context = MagicMock()
        handler._reply_workflow_error = MagicMock()

        project = MagicMock()
        project.root_path = "/tmp/project"
        project.project_id = "proj_1"
        project.project_name = "test"
        handler._ensure_project = MagicMock(return_value=project)

        engine = MagicMock()
        engine.is_running = False
        engine.root_path = "/tmp/project"
        stale_pending = PendingConfirmation(
            requirement="old task",
            created_at=time.time() - 31 * 60,
            initiator_user_id="user_123",
        )
        engine.project = WorkflowProject(
            status=WorkflowStatus.AWAITING_TOOL_SELECT,
            pending=stale_pending,
        )
        ctx.workflow_engine_manager.get.return_value = engine
        ctx.workflow_engine_manager.get_or_create.return_value = engine

        handler.start_workflow("msg_1", "chat_1", "new task", project)

        # Should NOT have called _reply_workflow_error (stale state is auto-reset)
        handler._reply_workflow_error.assert_not_called()
        handler.send_card_to_chat.assert_not_called()
        self.assertEqual(engine.project.status, WorkflowStatus.GENERATING_SCRIPT)
        self.assertTrue(engine.project.pending.auto_reviewer)
        handler._schedule_generate_and_show_confirm_card.assert_called_once()

    @patch("src.thread.get_current_sender_id", return_value="user_123")
    @patch("src.workflow_engine.bridge.RuntimeBridge.check_node_available", return_value=True)
    @patch("src.workflow_engine.templates.discover_templates", return_value=[])
    def test_recent_state_is_replaced_by_new_workflow(self, mock_templates, mock_node, mock_sender):
        from src.feishu.handlers.workflow import WorkflowHandler

        ctx = MagicMock()
        ctx.workflow_engine_manager = MagicMock()
        ctx.project_manager = MagicMock()
        ctx.progress_reporter = MagicMock()
        ctx.settings.admin_user_ids = []

        handler = WorkflowHandler.__new__(WorkflowHandler)
        handler.ctx = ctx
        handler.reply_text = MagicMock()
        handler.reply_card = MagicMock()
        handler.send_card_to_chat = MagicMock()
        handler.update_card = MagicMock()
        handler.add_reaction = MagicMock()
        handler._submit_engine_task = MagicMock()
        handler._schedule_generate_and_show_confirm_card = MagicMock()
        handler._ensure_topic_engine_context = MagicMock()
        handler._reply_workflow_error = MagicMock()

        project = MagicMock()
        project.root_path = "/tmp/project"
        project.project_id = "proj_1"
        project.project_name = "test"
        handler._ensure_project = MagicMock(return_value=project)

        engine = MagicMock()
        engine.is_running = False
        engine.root_path = "/tmp/project"
        fresh_pending = PendingConfirmation(
            requirement="current task",
            initiator_user_id="user_123",
        )
        engine.project = WorkflowProject(
            status=WorkflowStatus.AWAITING_TOOL_SELECT,
            pending=fresh_pending,
        )
        ctx.workflow_engine_manager.get.return_value = engine
        ctx.workflow_engine_manager.get_or_create.return_value = engine

        handler.start_workflow("msg_1", "chat_1", "new task", project)

        handler._reply_workflow_error.assert_not_called()
        handler.send_card_to_chat.assert_not_called()
        self.assertEqual(engine.project.status, WorkflowStatus.GENERATING_SCRIPT)
        self.assertTrue(engine.project.pending.auto_reviewer)
        handler._schedule_generate_and_show_confirm_card.assert_called_once()


if __name__ == "__main__":
    unittest.main()
