"""Regression tests for the RUNNING workflow progress-card "停止" button.

Validates:
- The ``WORKFLOW_STOP_RUNNING`` action constant exists and is stable.
- ``WorkflowHandler.handle_workflow_stop_running`` delegates to
  ``stop_workflow`` with the resolved project.
- The router mapping and (lightly) the action-registry wiring reference the
  new handler.
- ``_inject_workflow_stop_button`` appends a danger stop button while running.
"""

import threading
import unittest
from collections import defaultdict
from unittest.mock import MagicMock

from src.card.actions.dispatch import WORKFLOW_STOP_RUNNING
from src.workflow_engine.models import WorkflowProject, WorkflowStatus


class TestWorkflowStopButtonConstant(unittest.TestCase):
    def test_constant_value(self):
        """The stop-running action id is stable and matches the wiring."""
        self.assertEqual(WORKFLOW_STOP_RUNNING, "workflow_stop_running")


class TestHandleWorkflowStopRunning(unittest.TestCase):
    def setUp(self):
        from src.feishu.handlers.workflow import WorkflowHandler

        self.handler = WorkflowHandler.__new__(WorkflowHandler)
        self.handler.ctx = MagicMock()
        self.project = MagicMock(project_id="proj-1", root_path="/tmp/proj-1")
        self.handler._resolve_project_from_id = MagicMock(return_value=self.project)
        self.handler._get_root_path = MagicMock(return_value="/tmp/proj-1")
        self.handler.stop_workflow = MagicMock()

    def test_delegates_to_stop_workflow(self):
        """The card button handler resolves the project and delegates."""
        self.handler.handle_workflow_stop_running(
            "msg-1",
            "chat-1",
            "",
            {"action": WORKFLOW_STOP_RUNNING, "chat_id": "chat-1", "project_id": "proj-1"},
        )

        self.handler._resolve_project_from_id.assert_called_once_with("proj-1", "chat-1")
        self.handler.stop_workflow.assert_called_once_with(
            "msg-1",
            "chat-1",
            self.project,
            terminal_is_noop=True,
        )

    def test_prefers_explicit_project_id_argument(self):
        """An explicit project_id argument wins over the button value."""
        self.handler.handle_workflow_stop_running(
            "msg-2",
            "chat-2",
            "explicit-proj",
            {"action": WORKFLOW_STOP_RUNNING, "project_id": "value-proj"},
        )

        self.handler._resolve_project_from_id.assert_called_once_with("explicit-proj", "chat-2")
        self.handler.stop_workflow.assert_called_once_with(
            "msg-2",
            "chat-2",
            self.project,
            terminal_is_noop=True,
        )

    def test_terminal_stop_button_is_an_idempotent_notice(self):
        """A stale terminal card action must not surface an invalid-state error."""
        self.handler.stop_workflow = self.handler.__class__.stop_workflow.__get__(self.handler)
        engine = MagicMock()
        engine._lock = threading.RLock()
        engine.project = WorkflowProject(status=WorkflowStatus.FAILED)
        engine.is_running = False
        self.handler.ctx.workflow_engine_manager.get.return_value = engine
        self.handler.ctx.settings.admin_user_ids = []
        self.handler._reply_workflow_error = MagicMock()
        self.handler.reply_text = MagicMock()

        self.handler.stop_workflow(
            "msg-terminal",
            "chat-1",
            self.project,
            terminal_is_noop=True,
        )

        self.handler._reply_workflow_error.assert_not_called()
        self.handler.reply_text.assert_called_once_with(
            "msg-terminal",
            "Workflow 任务已结束，当前状态：失败。",
        )


class TestInjectWorkflowStopButton(unittest.TestCase):
    def setUp(self):
        from src.feishu.handlers.workflow import WorkflowHandler

        self.handler = WorkflowHandler.__new__(WorkflowHandler)

    def test_appends_stop_button_row(self):
        """A danger stop button carrying the correct value is appended."""
        card_data = {"header": {}, "elements": [{"tag": "markdown", "content": "running"}]}

        self.handler._inject_workflow_stop_button(
            card_data,
            "chat-1",
            "proj-1",
            is_running=True,
        )

        elements = card_data["elements"]
        # Original element preserved, plus an hr and at least one button row.
        self.assertGreater(len(elements), 1)
        self.assertTrue(any(e.get("tag") == "hr" for e in elements))

        # Find the button anywhere in the appended layout containers.
        found = self._find_stop_button(elements)
        self.assertIsNotNone(found, "stop button not found in card elements")
        self.assertEqual(found["type"], "danger")
        self.assertEqual(found["value"]["action"], WORKFLOW_STOP_RUNNING)
        self.assertEqual(found["value"]["chat_id"], "chat-1")
        self.assertEqual(found["value"]["project_id"], "proj-1")
        self.assertIn("confirm", found)

    def test_no_elements_is_noop(self):
        """Missing/invalid elements list must not raise."""
        card_data = {"header": {}}  # no elements key
        self.handler._inject_workflow_stop_button(
            card_data,
            "chat-1",
            "proj-1",
            is_running=True,
        )
        self.assertNotIn("elements", card_data)

    def test_terminal_progress_does_not_append_stop_button(self):
        card_data = {"header": {}, "elements": [{"tag": "markdown", "content": "failed"}]}

        self.handler._inject_workflow_stop_button(
            card_data,
            "chat-1",
            "proj-1",
            is_running=False,
        )

        self.assertIsNone(self._find_stop_button(card_data["elements"]))

    @staticmethod
    def _find_stop_button(elements):
        stack = list(elements)
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                if node.get("tag") == "button" and isinstance(node.get("value"), dict):
                    if node["value"].get("action") == WORKFLOW_STOP_RUNNING:
                        return node
                for v in node.values():
                    if isinstance(v, (list, dict)):
                        stack.append(v)
            elif isinstance(node, list):
                stack.extend(node)
        return None


class TestRouterAndRegistryWiring(unittest.TestCase):
    def test_action_registry_contains_new_handler(self):
        from src.feishu.action_registry import init_action_registry

        client = MagicMock()
        workflow = MagicMock()
        handlers = defaultdict(MagicMock)
        handlers["workflow"] = workflow
        client._handler_ctx.handlers = handlers

        actions = init_action_registry(client)

        self.assertIs(
            actions[WORKFLOW_STOP_RUNNING],
            workflow.handle_workflow_stop_running,
        )


if __name__ == "__main__":
    unittest.main()
