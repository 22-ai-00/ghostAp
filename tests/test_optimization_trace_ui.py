import json
import logging
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.card.builders.project import ProjectBuilder
from src.feishu.handlers.base import BaseHandler
from src.feishu.handlers.project import ProjectHandler
from src.project.context import ProjectContext, ProjectStatus
from src.tasking.scheduler import TaskContext, TaskScheduler, TaskSpec
from src.utils.trace import RequestIdFilter, TraceContext, get_trace_id


class TestOptimizationTraceUI(unittest.TestCase):
    def test_trace_context_propagation(self):
        """Test that request_id is propagated through contextvars."""
        req_id = "test-req-123"

        # 1. Basic ContextVar check
        with TraceContext(req_id):
            self.assertEqual(get_trace_id(), req_id)

            # 2. Logging Filter check
            record = logging.LogRecord("name", logging.INFO, "pathname", 1, "msg", (), None)
            f = RequestIdFilter()
            f.filter(record)
            self.assertEqual(record.request_id, req_id)

        # Context should be cleared
        self.assertIsNone(get_trace_id())

    def test_scheduler_context_propagation(self):
        """Test that TaskScheduler propagates contextvars."""
        scheduler = TaskScheduler(max_concurrent=1)
        req_id = "sched-req-456"

        result_container = {}
        completed = threading.Event()

        def task_fn(ctx: TaskContext):
            result_container["req_id"] = get_trace_id()
            completed.set()
            return "done"

        with TraceContext(req_id):
            # submit inside context
            spec = TaskSpec(chat_id="chat1", name="test_task")
            scheduler.submit(spec, task_fn)

        self.assertTrue(completed.wait(timeout=1))

        self.assertEqual(result_container.get("req_id"), req_id)
        scheduler.stop(shutdown_executor=True)

    def test_pagination_logic(self):
        """Test project board pagination logic in ProjectBuilder."""
        # Mock 12 projects
        projects = []
        for i in range(12):
            p = MagicMock(spec=ProjectContext)
            p.project_id = f"p{i}"
            p.project_name = f"Project {i}"
            p.root_path = f"/tmp/p{i}"
            p.last_active = 1000 + i
            p.coco_mode = False
            p.claude_mode = False
            p.status = ProjectStatus.IDLE
            p.get_status_emoji.return_value = "📁"
            projects.append(p)

        # Page 1 (size 5) -> 0..4
        # Sorted by last_active desc (so 11..0)
        # So page 1 should have p11..p7

        msg_type, content_json = ProjectBuilder.build_status_board_card(projects, page=1, page_size=5)
        card = json.loads(content_json)

        # Check title contains count
        found_count = False
        for elem in card["body"]["elements"]:
            if "content" in elem and "共 **12** 个项目" in elem["content"]:
                found_count = True
                break
        self.assertTrue(found_count)

        # Check content contains Project 11 but not Project 6
        json_str = json.dumps(card)
        self.assertIn("Project 11", json_str)
        self.assertIn("Project 7", json_str)
        self.assertNotIn("Project 6", json_str)

        # Check buttons: Should have Next, No Prev
        self.assertIn("switch_board_page", json_str)
        self.assertIn('"page": 2', json_str)
        self.assertNotIn('"page": 0', json_str)

        # Page 3 (size 5) -> 10..11 (items p1, p0)
        msg_type, content_json = ProjectBuilder.build_status_board_card(projects, page=3, page_size=5)
        json_str = json.dumps(json.loads(content_json))  # normalize

        self.assertIn("Project 1", json_str)
        self.assertIn("Project 0", json_str)
        self.assertNotIn("Project 2", json_str)

        # Check buttons: Should have Prev, No Next
        self.assertIn('"page": 2', json_str)  # Prev
        self.assertNotIn('"page": 4', json_str)  # Next

    def test_reply_error(self):
        """Test BaseHandler.reply_error."""
        ctx = MagicMock()
        ctx.message_linker.query.return_value = None
        handler = BaseHandler(ctx)

        # reply_error now delegates to send_error_card, which calls reply_card internally.
        # We mock send_error_card to verify reply_error passes the right args.
        handler.send_error_card = MagicMock()

        handler.reply_error("msg_1", "Something went wrong", title="Error Title")

        handler.send_error_card.assert_called_once_with(
            chat_id="",
            exc="Something went wrong",
            title="Error Title",
            origin_message_id="msg_1",
            severity="fatal",
        )

    def test_reply_error_binds_diagnostics_to_the_real_origin_chat(self):
        """A real card click must not be rejected by an ``unknown`` binding."""
        from src.card.error_diagnostics import render_error_diagnostic

        ctx = MagicMock()
        ctx.message_linker.query.return_value = {"chat_id": "chat_real"}
        handler = BaseHandler(ctx)
        handler.reply_card = MagicMock(return_value="card_reply")

        handler.reply_error("msg_1", "Something went wrong", title="Error Title")

        card = json.loads(handler.reply_card.call_args.args[1])
        nodes = [card]
        detail_action = None
        while nodes:
            node = nodes.pop()
            if isinstance(node, dict):
                value = node.get("value")
                if isinstance(value, dict) and value.get("action") == "show_error_details":
                    detail_action = value
                    break
                nodes.extend(node.values())
            elif isinstance(node, list):
                nodes.extend(node)

        self.assertIsNotNone(detail_action)
        assert detail_action is not None
        self.assertEqual(detail_action.get("chat_id"), "chat_real")
        self.assertNotIn("unknown", json.dumps(card))
        rendered = render_error_diagnostic(
            detail_action.get("diagnostic_token"),
            chat_id="chat_real",
            origin_message_id="msg_1",
        )
        self.assertIn("Something went wrong", rendered)
        self.assertNotIn("当前会话与原始错误卡不匹配", rendered)

        from src.feishu.action_registry import init_action_registry

        handlers = {
            name: MagicMock()
            for name in (
                "project",
                "system",
                "deep",
                "spec",
                "workflow",
                "claude",
                "aiden",
                "codex",
                "gemini",
                "traex",
                "grok",
                "dsh",
            )
        }
        handlers["coco"] = handler
        handler.handle_card_exit = MagicMock()
        client = SimpleNamespace(
            _handler_ctx=SimpleNamespace(handlers=handlers),
            _project_manager=MagicMock(),
        )
        handler.reply_text = MagicMock()
        actions = init_action_registry(client)
        actions["show_error_details"](
            "card_reply",
            "chat_real",
            None,
            detail_action,
        )
        callback_text = handler.reply_text.call_args.args[1]
        self.assertIn("Something went wrong", callback_text)
        self.assertNotIn("当前会话与原始错误卡不匹配", callback_text)

        handler.reply_text.reset_mock()
        actions["show_error_details"](
            "card_reply",
            "chat_other",
            None,
            detail_action,
        )
        rejected_text = handler.reply_text.call_args.args[1]
        self.assertIn("当前会话与原始错误卡不匹配", rejected_text)

    def test_handler_error_refactoring(self):
        """Verify that ProjectHandler uses reply_error."""
        ctx = MagicMock()
        handler = ProjectHandler(ctx)
        handler.reply_error = MagicMock()

        # Test case: project not found
        handler.project_manager.find_project_by_name_with_hint.return_value = (None, "")
        handler.project_manager.search_projects.return_value = []

        handler.switch_project("msg_1", "chat_1", "nonexistent")

        handler.reply_error.assert_called_once()
        call_args = handler.reply_error.call_args
        self.assertIn("未找到项目", call_args[0][1])


if __name__ == "__main__":
    unittest.main()
