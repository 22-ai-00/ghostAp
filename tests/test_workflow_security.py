"""Security tests for stopping a Workflow."""

import unittest
from unittest.mock import MagicMock, patch

from src.workflow_engine.models import WorkflowProject, WorkflowStatus


class TestWorkflowStopSecurity(unittest.TestCase):
    """Security tests for stop_workflow fail-closed behavior and admin source validation."""

    def _make_handler(self):
        from src.feishu.handlers.workflow import WorkflowHandler

        handler = WorkflowHandler.__new__(WorkflowHandler)
        handler.ctx = MagicMock()
        handler.reply_text = MagicMock()
        handler._reply_workflow_error = MagicMock()  # type: ignore[method-assign]
        handler.get_working_dir = MagicMock(return_value="/tmp/project")
        handler._get_root_path = MagicMock(return_value="/tmp/project")
        return handler

    def _make_running_engine(self, initiator="user_123"):
        """Create a mock engine in running state with a WorkflowProject."""
        engine = MagicMock()
        engine.is_running = True
        engine.project = WorkflowProject(
            status=WorkflowStatus.RUNNING,
            initiator_user_id=initiator,
            workflow_id="wf_1",
            name="test workflow",
        )
        engine.stop = MagicMock()
        return engine

    # ------------------------------------------------------------------
    # Stop Workflow Fail-Closed Tests
    # ------------------------------------------------------------------

    @patch("src.thread.get_current_sender_id", return_value="user_123")
    def test_stop_denied_when_stored_initiator_missing(self, mock_sender):
        """When engine.project.initiator_user_id is None, stop should be denied.

        Fail-closed: missing stored initiator -> cannot verify identity -> deny.
        """
        handler = self._make_handler()
        engine = self._make_running_engine(initiator=None)
        handler.ctx.workflow_engine_manager.get.return_value = engine
        handler.ctx.settings.admin_user_ids = []

        handler.stop_workflow("msg_1", "chat_1")

        engine.stop.assert_not_called()
        handler._reply_workflow_error.assert_called_once()

    @patch("src.thread.get_current_sender_id", return_value=None)
    def test_stop_denied_when_current_user_missing(self, mock_sender):
        """When get_current_sender_id() returns None, stop should be denied.

        Fail-closed: missing current user -> cannot verify identity -> deny.
        """
        handler = self._make_handler()
        engine = self._make_running_engine(initiator="user_123")
        handler.ctx.workflow_engine_manager.get.return_value = engine
        handler.ctx.settings.admin_user_ids = []

        handler.stop_workflow("msg_1", "chat_1")

        engine.stop.assert_not_called()
        handler._reply_workflow_error.assert_called_once()

    @patch("src.thread.get_current_sender_id", return_value=None)
    def test_stop_denied_when_both_missing(self, mock_sender):
        """When both initiator and current_user are None, stop should be denied.

        Fail-closed: both missing -> double denial -> still deny.
        """
        handler = self._make_handler()
        engine = self._make_running_engine(initiator=None)
        handler.ctx.workflow_engine_manager.get.return_value = engine
        handler.ctx.settings.admin_user_ids = []

        handler.stop_workflow("msg_1", "chat_1")

        engine.stop.assert_not_called()
        handler._reply_workflow_error.assert_called_once()

    @patch("src.thread.get_current_sender_id", return_value="attacker_456")
    def test_stop_denied_for_non_initiator_non_admin(self, mock_sender):
        """A user who is neither the initiator nor an admin cannot stop.

        Authorization check: must be initiator OR admin.
        """
        handler = self._make_handler()
        engine = self._make_running_engine(initiator="user_123")
        handler.ctx.workflow_engine_manager.get.return_value = engine
        handler.ctx.settings.admin_user_ids = ["admin_789"]

        handler.stop_workflow("msg_1", "chat_1")

        engine.stop.assert_not_called()
        handler._reply_workflow_error.assert_called_once()

    @patch("src.thread.get_current_sender_id", return_value="user_123")
    def test_stop_allowed_for_initiator(self, mock_sender):
        """The workflow initiator can stop successfully.

        Positive path: initiator matches -> allow stop.
        """
        handler = self._make_handler()
        engine = self._make_running_engine(initiator="user_123")
        handler.ctx.workflow_engine_manager.get.return_value = engine
        handler.ctx.settings.admin_user_ids = []

        handler.stop_workflow("msg_1", "chat_1")

        engine.stop.assert_called_once()
        handler.reply_text.assert_called_once()
        self.assertIn("Workflow 任务已停止", handler.reply_text.call_args[0][1])

    @patch("src.thread.get_current_sender_id", return_value="admin_789")
    def test_stop_allowed_for_admin(self, mock_sender):
        """An admin user (in admin_user_ids) can stop even if not the initiator.

        Admin bypass: admin in settings.admin_user_ids -> allow stop.
        """
        handler = self._make_handler()
        engine = self._make_running_engine(initiator="user_123")
        handler.ctx.workflow_engine_manager.get.return_value = engine
        handler.ctx.settings.admin_user_ids = ["admin_789"]

        handler.stop_workflow("msg_1", "chat_1")

        engine.stop.assert_called_once()
        handler.reply_text.assert_called_once()
        self.assertIn("Workflow 任务已停止", handler.reply_text.call_args[0][1])

    @patch("src.thread.get_current_sender_id", return_value="user_123")
    def test_stop_denied_when_engine_not_running(self, mock_sender):
        """If no engine is running, show appropriate message.

        Edge case: engine exists but is_running=False -> inform user.
        """
        handler = self._make_handler()
        engine = self._make_running_engine(initiator="user_123")
        engine.is_running = False
        handler.ctx.workflow_engine_manager.get.return_value = engine
        handler.ctx.settings.admin_user_ids = []

        handler.stop_workflow("msg_1", "chat_1")

        engine.stop.assert_not_called()
        handler._reply_workflow_error.assert_called_once()

    # ------------------------------------------------------------------
    # Admin Source Tests
    # ------------------------------------------------------------------

    @patch("src.thread.get_current_sender_id", return_value="admin_789")
    def test_admin_source_from_settings_not_config(self, mock_sender):
        """Verify admin_user_ids comes from self.ctx.settings.admin_user_ids, NOT config.

        Mock both settings and config with different admin lists. Only settings
        should be consulted for admin authorization.
        """
        handler = self._make_handler()
        engine = self._make_running_engine(initiator="user_123")
        handler.ctx.workflow_engine_manager.get.return_value = engine
        # settings has admin_789 (current user), config has different admin
        handler.ctx.settings.admin_user_ids = ["admin_789"]
        handler.ctx.config.admin_user_ids = ["other_admin_999"]

        handler.stop_workflow("msg_1", "chat_1")

        # Should be allowed because settings.admin_user_ids has admin_789
        engine.stop.assert_called_once()
        handler.reply_text.assert_called_once()
        self.assertIn("Workflow 任务已停止", handler.reply_text.call_args[0][1])

        # Now reverse: settings has wrong admin, config has correct one
        engine.stop.reset_mock()
        handler.reply_text.reset_mock()
        handler.ctx.settings.admin_user_ids = ["other_admin_999"]
        handler.ctx.config.admin_user_ids = ["admin_789"]

        handler.stop_workflow("msg_1", "chat_1")

        # Should be denied because settings.admin_user_ids does NOT have admin_789
        # (config is NOT consulted)
        engine.stop.assert_not_called()
        handler._reply_workflow_error.assert_called_once()

    @patch("src.thread.get_current_sender_id", return_value="admin_789")
    def test_admin_ids_empty_list_fallback(self, mock_sender):
        """When settings.admin_user_ids is None, it should fall back to empty list [].

        Uses getattr with default [] and `or []` for double safety.
        """
        handler = self._make_handler()
        engine = self._make_running_engine(initiator="user_123")
        handler.ctx.workflow_engine_manager.get.return_value = engine
        # Set admin_user_ids to None explicitly
        handler.ctx.settings.admin_user_ids = None

        handler.stop_workflow("msg_1", "chat_1")

        # Should be denied because None falls back to [] and admin_789 is not in []
        engine.stop.assert_not_called()
        handler._reply_workflow_error.assert_called_once()




if __name__ == "__main__":
    unittest.main()
