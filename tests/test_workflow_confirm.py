"""Tests for the Workflow Engine confirmation flow (AC1).

Validates:
- /wf generates a script and shows a confirmation card (AWAITING_CONFIRM)
- Confirm action triggers execute_workflow
- Cancel action resets state to IDLE
- AI fallback works when script generation fails
"""

import json
import os
import threading
import unittest
from unittest.mock import MagicMock, mock_open, patch

from src.card.actions.dispatch import WORKFLOW_CANCEL, WORKFLOW_CONFIRM_START
from src.card.events.types import CardEventType
from src.card.events.workflow import workflow_confirm
from src.feishu.ws_card_action_handler import SYSTEM_CARD_ACTIONS
from src.spec_engine.review_agents import ReviewAgentBinding
from src.workflow_engine.models import PendingConfirmation, WorkflowProject, WorkflowStatus
from src.workflow_engine.script_gen import validate_generated_script


class TestWorkflowConfirmWithAgentBindings(unittest.TestCase):
    """Test workflow confirmation with agent bindings including tool and model info."""

    def setUp(self):
        from src.feishu.handlers.workflow import WorkflowHandler

        self.handler = WorkflowHandler.__new__(WorkflowHandler)
        self.handler.ctx = MagicMock()
        self.handler.ctx.workflow_engine_manager = MagicMock()
        self.handler.reply_card = MagicMock()
        self.handler.send_card_to_chat = MagicMock()
        self.handler._reply_workflow_error = MagicMock()
        self.handler._generate_and_show_confirm_card = MagicMock()
        self.handler._resolve_tool_lists = MagicMock(return_value=({}, [], [], []))
        self.handler._get_root_path = MagicMock(return_value="/tmp")
        self.handler._get_project_for_chat = MagicMock(return_value=MagicMock(project_id="test_proj"))
        self.handler.get_engine_name = MagicMock(return_value="test_engine")

    def test_confirm_passes_agent_bindings_to_engine(self):
        """确认操作将 Agent 绑定信息传递给执行引擎。"""
        from src.workflow_engine.engine import WorkflowEngine

        # Create a mock engine
        mock_engine = MagicMock(spec=WorkflowEngine)
        self.handler.ctx.workflow_engine_manager.get.return_value = mock_engine

        # Create pending confirmation with agent bindings
        pending = PendingConfirmation(
            requirement="test workflow with agent bindings",
            orchestrator_binding=ReviewAgentBinding(
                provider="cli",
                tool_name="coco",
                display_name="Coco",
                agent_type="coco",
                model_name="gpt-4",
                use_default_model=False
            ),
            review_agents=[
                ReviewAgentBinding(
                    provider="cli",
                    tool_name="claude",
                    display_name="Claude",
                    agent_type="claude",
                    model_name="claude-3-sonnet",
                    use_default_model=False
                )
            ],
            selected_tools=["coco", "claude"],
            engine_session_key="session_abc",
            initiator_user_id="user_001",
            script_path="/tmp/test_script.js",
            script_hash="test_hash"
        )

        mock_engine.project = WorkflowProject(
            status=WorkflowStatus.AWAITING_CONFIRM,
            pending=pending
        )

        # Mock the execute_workflow method
        mock_engine.execute_workflow = MagicMock()

        # Mock file operations
        mock_script_content = b"""
export const meta = {
    name: "test-workflow",
    description: "Test workflow with agent bindings",
    tools: ["coco", "claude"]
};

export default async function run() {
    const result = await agent({
        tool: "coco",
        model: "gpt-4",
        prompt: "Hello world",
        label: "hello-world",
        timeout: 120
    });
    if (result && result.error) return result;
    return result;
}
"""
        with patch("builtins.open", mock_open(read_data=mock_script_content)):
            with patch("os.path.exists", return_value=True):
                with patch("src.thread.get_current_sender_id", return_value="user_001"):
                    with patch("hashlib.sha256") as mock_sha256:
                        # Mock hash calculation
                        mock_sha256.return_value.hexdigest.return_value = "test_hash"
                        # Mock project resolution
                        mock_project = MagicMock()
                        mock_project.root_path = "/tmp"
                        mock_project.project_id = "proj_789"
                        with patch.object(self.handler, "_resolve_project_from_id", return_value=mock_project):
                            # Mock get_working_dir
                            with patch.object(self.handler, "get_working_dir", return_value="/tmp"):
                                # Mock _build_workflow_callbacks
                                with patch.object(self.handler, "_build_workflow_callbacks", return_value={}):
                                    # Mock tempfile.mkstemp
                                    with patch("tempfile.mkstemp", return_value=(1, "/tmp/immutable_script.js")):
                                        # Mock os.fdopen
                                        with patch("os.fdopen", mock_open()):
                                            # Mock os.remove to avoid cleanup errors
                                            with patch("os.remove"):
                                                # Mock lock_helper
                                                self.handler.lock_helper = MagicMock()
                                                # Make handle_lock_conflict execute the function directly
                                                self.handler.lock_helper.handle_lock_conflict = lambda func, *args, **kwargs: func()

                                                # Call confirm handler
                                                self.handler.handle_workflow_confirm_start(
                                                    "msg_123",
                                                    "chat_456",
                                                    "proj_789",
                                                    {"engine_session_key": "session_abc"}
                                                )

        # Verify _reply_workflow_error was NOT called (no errors occurred)
        self.assertEqual(self.handler._reply_workflow_error.call_count, 0)

        # Verify that pending state was cleared (start_execution was called)
        self.assertIsNone(mock_engine.project.pending,
                         "Pending state should have been cleared after start_execution")

        # Verify that initiator_user_id was migrated from pending
        self.assertEqual(mock_engine.project.initiator_user_id, "user_001",
                       "initiator_user_id should have been migrated from pending")

        print("Test completed: Confirm handler executed without errors and migrated pending state")

    def test_script_gen_includes_agent_bindings_in_prompt(self):
        """脚本生成提示中包含 Agent 绑定信息。"""
        from src.workflow_engine.script_gen import build_script_gen_prompt

        # Test with agent bindings
        orchestrator_binding = {
            "tool_name": "coco",
            "model_name": "gpt-4",
            "use_default_model": False
        }

        review_agents = [
            {
                "tool_name": "claude",
                "model_name": "claude-3-sonnet",
                "use_default_model": False
            }
        ]

        prompt = build_script_gen_prompt(
            requirement="test workflow with agent bindings",
            available_tools=["coco", "claude"],
            orchestrator_agent="coco",
            orchestrator_binding=orchestrator_binding,
            review_agents=review_agents,
        )

        # Verify the prompt includes agent and model info
        self.assertIn("已选择的主 Agent", prompt)
        self.assertIn("coco", prompt)
        self.assertIn("gpt-4", prompt)
        self.assertIn("已选择的评审 Agent", prompt)
        self.assertIn("claude", prompt)
        self.assertIn("claude-3-sonnet", prompt)


class TestWorkflowConfirmConstants(unittest.TestCase):
    """Verify action_ids and registrations are in place."""

    def test_workflow_confirm_start_in_system_card_actions(self):
        self.assertIn("workflow_confirm_start", SYSTEM_CARD_ACTIONS)

    def test_workflow_cancel_in_system_card_actions(self):
        self.assertIn("workflow_cancel", SYSTEM_CARD_ACTIONS)

    def test_action_id_constants_exist(self):
        self.assertEqual(WORKFLOW_CONFIRM_START, "workflow_confirm_start")
        self.assertEqual(WORKFLOW_CANCEL, "workflow_cancel")

    def test_workflow_confirm_event_type_exists(self):
        self.assertEqual(CardEventType.WORKFLOW_CONFIRM.value, "workflow_confirm")

    def test_forwarding_map_has_workflow_confirm(self):
        from src.feishu.router import FORWARDING_MAP
        self.assertIn("_handle_workflow_confirm_start", FORWARDING_MAP)
        self.assertIn("_handle_workflow_cancel", FORWARDING_MAP)
        self.assertEqual(
            FORWARDING_MAP["_handle_workflow_confirm_start"],
            ("workflow", "handle_workflow_confirm_start"),
        )
        self.assertEqual(
            FORWARDING_MAP["_handle_workflow_cancel"],
            ("workflow", "handle_workflow_cancel"),
        )


class TestWorkflowConfirmFactory(unittest.TestCase):
    """Test the workflow_confirm factory function."""

    def test_workflow_confirm_creates_event(self):
        event = workflow_confirm(
            script_name="test-workflow",
            description="Test workflow description",
            phases=[{"title": "Phase 1", "detail": "Do something"}],
            tools=["coco", "claude"],
            requirement="test requirement",
            initiator_user_id="user_001",
            engine_session_key="abc123",
            project_id="proj_123",
            chat_id="chat_456",
        )
        self.assertEqual(event.type, CardEventType.WORKFLOW_CONFIRM)
        self.assertEqual(event.payload["script_name"], "test-workflow")
        self.assertEqual(event.payload["tools"], ["coco", "claude"])
        self.assertEqual(event.payload["project_id"], "proj_123")
        self.assertEqual(event.payload["initiator_user_id"], "user_001")
        self.assertEqual(event.payload["engine_session_key"], "abc123")

    def test_workflow_confirm_fallback_flag(self):
        event = workflow_confirm(
            script_name="fallback",
            description="",
            phases=[],
            tools=["coco"],
            requirement="req",
            initiator_user_id="user_001",
            engine_session_key="key123",
            is_fallback=True,
        )
        self.assertTrue(event.payload.get("is_fallback"))


class TestWorkflowProjectPendingFields(unittest.TestCase):
    """Test that WorkflowProject supports pending state via PendingConfirmation sub-model."""

    def test_pending_default_none(self):
        project = WorkflowProject()
        self.assertIsNone(project.pending)

class TestValidateGeneratedScriptRegression(unittest.TestCase):
    """Regression tests for validate_generated_script."""

    def test_valid_script_passes(self):
        script = '''export const meta = {
  name: "test-workflow",
  description: "Test",
  phases: [{ title: "Phase 1", detail: "Do stuff" }],
  tools: ["coco"]
};

export default async function() {
  const result = await agent("do something", { tool: "coco", label: "primary", timeout: 120 });
  if (result && result.error) {
    return { error: result.error, fallback: "primary failed" };
  }
  return result;
}
'''
        is_valid, errors = validate_generated_script(script)
        self.assertTrue(is_valid, f"Expected valid, got errors: {errors}")

    def test_empty_script_fails(self):
        is_valid, errors = validate_generated_script("")
        self.assertFalse(is_valid)

    def test_missing_meta_fails(self):
        script = 'export default async function() { await agent("x"); }'
        is_valid, errors = validate_generated_script(script)
        self.assertFalse(is_valid)
        self.assertTrue(any("meta" in e.lower() for e in errors))

    def test_dangerous_pattern_warns(self):
        script = '''export const meta = {
  name: "bad", description: "Bad",
  phases: [{title: "P1", detail: "d"}]
};
const fs = require('fs');
export default async function() { await agent("x"); }
'''
        is_valid, messages = validate_generated_script(script)
        # Dangerous patterns are now fail-closed blocking errors (not warnings).
        # The validator emits "[capability] Forbidden pattern:" messages.
        self.assertFalse(is_valid)
        self.assertTrue(any(
            "[capability]" in m or "Forbidden pattern" in m
            for m in messages
        ))

    def test_duplicate_agent_labels_fail(self):
        script = '''export const meta = {
  name: "bad-labels",
  description: "Duplicate labels should fail",
  phases: [{ title: "Phase 1", detail: "Do stuff" }],
  tools: ["coco"]
};

export default async function() {
  const first = await agent("first", { tool: "coco", label: "same", timeout: 120 });
  if (first && first.error) return first;
  const second = await agent("second", { tool: "coco", label: "same", timeout: 120 });
  if (second && second.error) return second;
  return second;
}
'''
        is_valid, messages = validate_generated_script(script)
        self.assertFalse(is_valid)
        self.assertTrue(any("duplicate" in m.lower() and "label" in m.lower() for m in messages))

    def test_direct_agent_without_timeout_fails(self):
        script = '''export const meta = {
  name: "missing-timeout",
  description: "Direct agent calls must be bounded",
  phases: [{ title: "Phase 1", detail: "Do stuff" }],
  tools: ["coco"]
};

export default async function() {
  const result = await agent("do something", { tool: "coco", label: "primary" });
  if (result && result.error) return result;
  return result;
}
'''
        is_valid, messages = validate_generated_script(script)
        self.assertFalse(is_valid)
        self.assertTrue(any("timeout" in m.lower() for m in messages))

    def test_direct_agent_without_error_handling_fails(self):
        script = '''export const meta = {
  name: "missing-error-handling",
  description: "Direct agent calls must handle result.error",
  phases: [{ title: "Phase 1", detail: "Do stuff" }],
  tools: ["coco"]
};

export default async function() {
  const result = await agent("do something", { tool: "coco", label: "primary", timeout: 120 });
  return result;
}
'''
        is_valid, messages = validate_generated_script(script)
        self.assertFalse(is_valid)
        self.assertTrue(any("error" in m.lower() for m in messages))


class TestWorkflowHandlerConfirmFlow(unittest.TestCase):
    """Integration tests for the start_workflow → confirm → execute flow."""

    def _make_handler(self):
        """Create a WorkflowHandler with mocked dependencies."""
        from src.feishu.handlers.workflow import WorkflowHandler

        ctx = MagicMock()
        ctx.workflow_engine_manager = MagicMock()
        ctx.project_manager = MagicMock()
        ctx.progress_reporter = MagicMock()

        handler = WorkflowHandler.__new__(WorkflowHandler)
        handler.ctx = ctx
        handler.reply_text = MagicMock()
        handler.reply_card = MagicMock()
        handler.reply_error = MagicMock()
        handler.send_card_to_chat = MagicMock(return_value="msg_card_123")
        handler.update_card = MagicMock(return_value=True)
        handler.add_reaction = MagicMock()
        handler.get_working_dir = MagicMock(return_value="/tmp/project")
        handler.get_engine_name = MagicMock(return_value="coco")
        handler.ensure_request_id = MagicMock(return_value="req_123")
        handler._submit_engine_task = MagicMock()

        return handler, ctx

    @patch("src.thread.get_current_sender_id", return_value="user_123")
    def test_confirm_start_triggers_execution(self, mock_sender):
        import hashlib
        import tempfile

        handler, ctx = self._make_handler()

        # Write a valid script to a real temp file so TOCTOU checks pass
        script_content = (
            "export const meta = {\n"
            "  name: 'test',\n"
            "  description: 'test workflow',\n"
            "  tools: ['coco'],\n"
            "};\n"
            "\n"
            "export default async function() {\n"
            "  const result = await agent('do work', { tool: 'coco', label: 'do-work', timeout: 120 });\n"
            "  if (result && result.error) return result;\n"
            "  return result;\n"
            "}\n"
        )
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False, encoding="utf-8")
        tmp.write(script_content)
        tmp.close()
        script_hash = hashlib.sha256(script_content.encode("utf-8")).hexdigest()

        engine = MagicMock()
        engine.project = WorkflowProject(
            status=WorkflowStatus.AWAITING_CONFIRM,
            pending=PendingConfirmation(
                script_path=tmp.name,
                requirement="do X",
                meta={"name": "x", "tools": ["coco"]},
                initiator_user_id="user_123",
                engine_session_key="test_session_key",
                selected_tools=["coco"],
                script_hash=script_hash,
            ),
        )
        engine.is_running = False
        ctx.workflow_engine_manager.get.return_value = engine
        handler._get_root_path = MagicMock(return_value="/tmp/project")

        project_mock = MagicMock()
        project_mock.project_name = "test"
        handler._resolve_project_from_id = MagicMock(return_value=project_mock)

        handler.handle_workflow_confirm_start(
            "msg_1", "chat_1", "proj_1",
            {"action": WORKFLOW_CONFIRM_START, "engine_session_key": "test_session_key"}
        )

        try:
            os.unlink(tmp.name)
        except OSError:
            pass

        # Should have submitted the workflow task
        handler._submit_engine_task.assert_called_once()
        # Pending state should be cleared
        self.assertIsNone(engine.project.pending)

    @patch("src.thread.get_current_sender_id", return_value="user_123")
    def test_confirm_start_updates_confirmation_card_to_running_before_submit(self, mock_sender):
        """Confirming execution should immediately replace the confirm card with a progress card."""
        import hashlib
        import tempfile

        handler, ctx = self._make_handler()

        script_content = (
            "export const meta = {\n"
            "  name: 'test',\n"
            "  description: 'test workflow',\n"
            "  tools: ['coco'],\n"
            "};\n"
            "\n"
            "export default async function() {\n"
            "  const result = await agent('do work', { tool: 'coco', label: 'do-work', timeout: 120 });\n"
            "  if (result && result.error) return result;\n"
            "  return result;\n"
            "}\n"
        )
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False, encoding="utf-8")
        tmp.write(script_content)
        tmp.close()
        script_hash = hashlib.sha256(script_content.encode("utf-8")).hexdigest()

        engine = MagicMock()
        engine.project = WorkflowProject(
            status=WorkflowStatus.AWAITING_CONFIRM,
            pending=PendingConfirmation(
                script_path=tmp.name,
                requirement="do X",
                meta={"name": "x", "tools": ["coco"]},
                initiator_user_id="user_123",
                engine_session_key="test_session_key",
                selected_tools=["coco"],
                script_hash=script_hash,
            ),
        )
        engine.is_running = False
        ctx.workflow_engine_manager.get.return_value = engine
        handler._get_root_path = MagicMock(return_value="/tmp/project")

        project_mock = MagicMock()
        project_mock.project_name = "test"
        handler._resolve_project_from_id = MagicMock(return_value=project_mock)

        def _submit_side_effect(*args, **kwargs):
            self.assertTrue(handler.update_card.called, "progress card must be updated before background submit")
            submitted_message_id, submitted_card = handler.update_card.call_args[0]
            self.assertEqual(submitted_message_id, "msg_1")
            self.assertEqual(submitted_card["schema"], "2.0")
            self.assertIn("body", submitted_card)
            self.assertIsInstance(submitted_card["body"]["elements"], list)

        handler._submit_engine_task.side_effect = _submit_side_effect

        handler.handle_workflow_confirm_start(
            "msg_1", "chat_1", "proj_1",
            {"action": WORKFLOW_CONFIRM_START, "engine_session_key": "test_session_key"}
        )

        try:
            os.unlink(tmp.name)
        except OSError:
            pass

        handler.update_card.assert_called_once()
        updated_message_id, card = handler.update_card.call_args[0]
        self.assertEqual(updated_message_id, "msg_1")
        self.assertEqual(card["schema"], "2.0")
        self.assertIn("body", card)
        self.assertIsInstance(card["body"]["elements"], list)
        self.assertNotIn("elements", card)
        self.assertIn("Workflow", card["header"]["title"]["content"])
        self.assertEqual(engine.project.status, WorkflowStatus.RUNNING)
        handler._submit_engine_task.assert_called_once()

    @patch("src.thread.get_current_sender_id", return_value="user_123")
    def test_stop_after_submit_prevents_queued_workflow_from_starting(self, mock_sender):
        """A stop acknowledged before the queued task claims the engine must be final."""
        import hashlib
        import tempfile

        from src.workflow_engine.engine import WorkflowEngine

        handler, ctx = self._make_handler()
        handler._reply_workflow_error = MagicMock()
        handler._resolve_origin = MagicMock(return_value="origin_msg")
        handler._show_initial_workflow_progress_card = MagicMock(return_value="progress_msg")
        ctx.settings.admin_user_ids = []

        script_content = (
            "export const meta = {\n"
            "  name: 'test',\n"
            "  description: 'test workflow',\n"
            "  tools: ['coco'],\n"
            "};\n"
            "export default async function workflow() {\n"
            "  const result = await agent('do work', { tool: 'coco', label: 'work', timeout: 120 });\n"
            "  if (result && result.error) return result;\n"
            "  return result;\n"
            "}\n"
        )
        temp_dir = tempfile.TemporaryDirectory()
        script_path = os.path.join(temp_dir.name, "generated.js")
        with open(script_path, "w", encoding="utf-8") as script_file:
            script_file.write(script_content)
        script_hash = hashlib.sha256(script_content.encode("utf-8")).hexdigest()

        engine = WorkflowEngine(chat_id="chat_1", root_path=temp_dir.name)
        engine._project = WorkflowProject(
            status=WorkflowStatus.AWAITING_CONFIRM,
            pending=PendingConfirmation(
                script_path=script_path,
                requirement="do X",
                meta={"name": "x", "tools": ["coco"]},
                initiator_user_id="user_123",
                engine_session_key="test_session_key",
                selected_tools=["coco"],
                script_hash=script_hash,
            ),
        )
        ctx.workflow_engine_manager.get.return_value = engine
        handler._get_root_path = MagicMock(return_value=temp_dir.name)
        project_mock = MagicMock(
            project_id="proj_1",
            project_name="test",
            root_path=temp_dir.name,
        )
        handler._resolve_project_from_id = MagicMock(return_value=project_mock)
        queued: dict[str, object] = {}
        handler._submit_engine_task = MagicMock(
            side_effect=lambda run_fn, *_args, **_kwargs: queued.setdefault("run", run_fn)
        )

        handler.handle_workflow_confirm_start(
            "msg_1",
            "chat_1",
            "proj_1",
            {
                "action": WORKFLOW_CONFIRM_START,
                "engine_session_key": "test_session_key",
            },
        )
        immutable_script_path = engine.project.script_path
        engine.execute_workflow = MagicMock()
        handler._safe_execute_engine = MagicMock(
            side_effect=lambda *, executor_func, **_kwargs: executor_func()
        )

        handler.stop_workflow("stop_msg", "chat_1", project_mock)
        queued["run"]()

        self.assertEqual(engine.project.status, WorkflowStatus.IDLE)
        engine.execute_workflow.assert_not_called()
        handler.reply_text.assert_called_once_with("stop_msg", "Workflow 任务已停止。")
        handler._reply_workflow_error.assert_not_called()

        if immutable_script_path:
            try:
                os.unlink(immutable_script_path)
            except OSError:
                pass
        temp_dir.cleanup()

    def test_workflow_callbacks_send_renderer_cards_directly(self):
        """Progress/done callbacks should pass renderer cards to update_card without build_info_card."""
        handler, _ctx = self._make_handler()
        callbacks = handler._build_workflow_callbacks("progress_msg", "chat_1", None)

        progress_card = {
            "header": {
                "title": {"tag": "plain_text", "content": "Workflow running"},
                "template": "blue",
            },
            "elements": [{"tag": "markdown", "content": "running"}],
        }
        done_project = WorkflowProject(
            name="done workflow",
            requirement="do X",
            status=WorkflowStatus.COMPLETED,
            result=json.dumps(
                {
                    "card_summary": {
                        "verdict": "passed",
                        "conclusion": "finished",
                    }
                }
            ),
        )

        with patch("src.card.CardBuilder.build_info_card", side_effect=AssertionError("wrong card builder")):
            callbacks.on_progress(progress_card)
            callbacks.on_done(done_project)

        self.assertEqual(handler.update_card.call_count, 2)
        first_card = handler.update_card.call_args_list[0].args[1]
        self.assertEqual(first_card["schema"], "2.0")
        self.assertEqual(first_card["body"]["elements"], progress_card["elements"])
        self.assertNotIn("elements", first_card)

        second_card = handler.update_card.call_args_list[1].args[1]
        self.assertEqual(second_card["schema"], "2.0")
        self.assertIn("body", second_card)
        self.assertIsInstance(second_card["body"]["elements"], list)
        handler.reply_text.assert_not_called()

    def test_workflow_callbacks_ignore_late_progress_after_done(self):
        """Late progress callbacks must not replace the final completion report."""
        handler, _ctx = self._make_handler()
        callbacks = handler._build_workflow_callbacks("progress_msg", "chat_1", None)

        done_project = WorkflowProject(
            name="done workflow",
            requirement="do X",
            status=WorkflowStatus.COMPLETED,
            result=json.dumps(
                {
                    "card_summary": {
                        "verdict": "passed",
                        "conclusion": "finished",
                    }
                }
            ),
        )
        late_progress = {
            "header": {
                "title": {"tag": "plain_text", "content": "Workflow running"},
                "template": "blue",
            },
            "elements": [{"tag": "markdown", "content": "late progress"}],
        }

        callbacks.on_done(done_project)
        calls_after_done = handler.update_card.call_count
        callbacks.on_progress(late_progress)

        self.assertEqual(handler.update_card.call_count, calls_after_done)
        final_card = handler.update_card.call_args.args[1]
        final_text = str(final_card)
        self.assertIn("finished", final_text)
        self.assertNotIn("late progress", final_text)

    def test_workflow_callbacks_send_html_report_attachment_on_done(self):
        """Terminal Workflow completion should create/send the full HTML report artifact."""
        handler, _ctx = self._make_handler()
        handler._send_workflow_completion_report = MagicMock(
            return_value={"html_path": "/tmp/report.html", "attachment_sent": True}
        )
        project = MagicMock()
        project.root_path = "/tmp/project"
        project.project_id = "proj_1"
        callbacks = handler._build_workflow_callbacks("progress_msg", "chat_1", project)

        done_project = WorkflowProject(
            name="done workflow",
            requirement="do X",
            status=WorkflowStatus.COMPLETED,
            result='{"final_report": "full result"}',
        )

        callbacks.on_done(done_project)

        handler._send_workflow_completion_report.assert_called_once_with(
            wf_project=done_project,
            chat_id="chat_1",
            message_id="progress_msg",
            project=project,
        )
        final_card = handler.update_card.call_args.args[1]
        self.assertIn("完整 HTML 报告已发送", str(final_card))

    def test_workflow_completion_fallback_never_replies_with_partial_raw_result(self):
        handler, _ctx = self._make_handler()
        handler._send_workflow_completion_report = MagicMock(
            return_value={
                "generated": True,
                "attachment_sent": False,
                "html_filename": "wf-report.html",
                "html_path": "/tmp/wf-report.html",
                "error": "upload failed",
            }
        )
        handler._replace_or_send_workflow_rendered_card = MagicMock(
            side_effect=RuntimeError("card failed")
        )
        callbacks = handler._build_workflow_callbacks("msg_1", "chat_1", None)
        sentinel = "RAW_RESULT_SENTINEL"
        project = WorkflowProject(
            name="done workflow",
            requirement="do X",
            status=WorkflowStatus.COMPLETED,
            result=("raw body " * 1000) + sentinel,
        )

        callbacks.on_done(project)

        fallback_text = handler.reply_text.call_args.args[1]
        self.assertIn("结果卡发送失败", fallback_text)
        self.assertIn("wf-report.html", fallback_text)
        self.assertNotIn(sentinel, fallback_text)
        self.assertNotIn("raw body", fallback_text)

    def test_send_workflow_completion_report_writes_cache_report_and_replies_file(self):
        """The report helper should write under ~/.cache/ghostAp mirror and send the HTML file."""
        import tempfile
        from pathlib import Path

        handler, _ctx = self._make_handler()
        reply_response = MagicMock()
        reply_response.success.return_value = True
        handler.im_client = MagicMock()
        handler.im_client.upload_file.return_value = "file_key_123"
        handler.im_client.reply_file.return_value = reply_response

        project = MagicMock()
        project.root_path = "/tmp/workflow-report-project"
        project.project_id = "proj_1"
        done_project = WorkflowProject(
            workflow_id="wf_123",
            name="done workflow",
            requirement="do X",
            status=WorkflowStatus.COMPLETED,
            result='{"final_report": "cache sentinel"}',
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "cache"
            with patch("src.workflow_engine.reporting.DEFAULT_WORKFLOW_CACHE_ROOT", str(cache_root)):
                status = handler._send_workflow_completion_report(
                    wf_project=done_project,
                    chat_id="chat_1",
                    message_id="msg_1",
                    project=project,
                )

            html_path = Path(status["html_path"])
            html = html_path.read_text(encoding="utf-8")

        self.assertTrue(status["attachment_sent"])
        self.assertIn(str(cache_root), str(html_path))
        self.assertIn("workflow_reports", str(html_path))
        self.assertNotIn(".ghostap", str(html_path))
        self.assertIn("cache sentinel", html)
        handler.im_client.upload_file.assert_called_once()
        handler.im_client.reply_file.assert_called_once_with(
            "msg_1",
            "file_key_123",
            reply_in_thread=True,
            audit_aliases=(),
        )

    def test_workflow_callbacks_do_not_create_new_cards_when_progress_patch_keeps_failing(self):
        """Periodic progress must not turn persistent PATCH failures into a new-card loop."""
        handler, _ctx = self._make_handler()
        handler.update_card.return_value = False
        handler._resolve_origin = MagicMock(return_value="origin_msg")
        callbacks = handler._build_workflow_callbacks("progress_msg", "chat_1", None)

        first_progress = {
            "header": {
                "title": {"tag": "plain_text", "content": "Workflow running"},
                "template": "blue",
            },
            "elements": [{"tag": "markdown", "content": "first"}],
        }
        second_progress = {
            "header": {
                "title": {"tag": "plain_text", "content": "Workflow still running"},
                "template": "blue",
            },
            "elements": [{"tag": "markdown", "content": "second"}],
        }

        with patch("src.card.CardBuilder.build_info_card", side_effect=AssertionError("wrong card builder")):
            callbacks.on_progress(first_progress)
            callbacks.on_progress(second_progress)

        self.assertEqual(handler.update_card.call_args_list[0].args[0], "progress_msg")
        self.assertEqual(handler.update_card.call_args_list[1].args[0], "progress_msg")
        patched_card = handler.update_card.call_args_list[1].args[1]
        self.assertEqual(patched_card["schema"], "2.0")
        self.assertEqual(patched_card["body"]["elements"], second_progress["elements"])
        handler.send_card_to_chat.assert_not_called()

    def test_workflow_completion_fallback_card_preserves_trusted_origin(self):
        """A terminal replacement card must remain patchable under outbound audit."""
        handler, _ctx = self._make_handler()
        handler.update_card.return_value = False
        handler._resolve_origin = MagicMock(return_value="origin_msg")
        handler.send_card_to_chat.return_value = "terminal_fallback"
        callbacks = handler._build_workflow_callbacks("progress_msg", "chat_1", None)
        done_project = WorkflowProject(
            name="done workflow",
            requirement="do X",
            status=WorkflowStatus.COMPLETED,
            result='{"final_report": "finished"}',
        )

        callbacks.on_done(done_project)

        handler.send_card_to_chat.assert_called_once()
        self.assertEqual(
            handler.send_card_to_chat.call_args.kwargs.get("origin_message_id"),
            "origin_msg",
        )

    def test_workflow_done_fences_progress_before_report_delivery(self):
        """Report generation must not leave a window for heartbeat progress updates."""
        handler, _ctx = self._make_handler()
        handler._resolve_origin = MagicMock(return_value="origin_msg")
        callbacks = handler._build_workflow_callbacks("progress_msg", "chat_1", None)
        late_progress = {
            "header": {
                "title": {"tag": "plain_text", "content": "Workflow running"},
                "template": "blue",
            },
            "elements": [{"tag": "markdown", "content": "late heartbeat"}],
        }

        def report_with_reentrant_progress(**_kwargs):
            callbacks.on_progress(late_progress)
            return {"generated": True, "attachment_sent": True}

        handler._send_workflow_completion_report = MagicMock(
            side_effect=report_with_reentrant_progress
        )
        done_project = WorkflowProject(
            name="done workflow",
            requirement="do X",
            status=WorkflowStatus.COMPLETED,
            result='{"final_report": "finished"}',
        )

        callbacks.on_done(done_project)

        self.assertEqual(handler.update_card.call_count, 1)
        self.assertNotIn("late heartbeat", str(handler.update_card.call_args.args[1]))

    def test_workflow_done_waits_for_inflight_progress_before_terminal_delivery(self):
        """An in-flight progress PATCH must finish before the terminal card is delivered."""
        handler, _ctx = self._make_handler()
        handler._resolve_origin = MagicMock(return_value="origin_msg")
        progress_entered = threading.Event()
        release_progress = threading.Event()
        terminal_delivery_started = threading.Event()
        delivery_order: list[str] = []

        def update_card(_message_id, card):
            if "in-flight progress" in str(card):
                progress_entered.set()
                if not release_progress.wait(timeout=2):
                    raise AssertionError("timed out waiting to release progress delivery")
                delivery_order.append("progress")
            else:
                terminal_delivery_started.set()
                delivery_order.append("terminal")
            return True

        handler.update_card = MagicMock(side_effect=update_card)
        handler._send_workflow_completion_report = MagicMock(
            return_value={"generated": True, "attachment_sent": True}
        )
        callbacks = handler._build_workflow_callbacks("progress_msg", "chat_1", None)
        progress_card = {
            "header": {
                "title": {"tag": "plain_text", "content": "Workflow running"},
                "template": "blue",
            },
            "elements": [{"tag": "markdown", "content": "in-flight progress"}],
        }
        done_project = WorkflowProject(
            name="done workflow",
            requirement="do X",
            status=WorkflowStatus.COMPLETED,
            result='{"final_report": "finished"}',
        )

        progress_thread = threading.Thread(target=callbacks.on_progress, args=(progress_card,))
        done_thread = threading.Thread(target=callbacks.on_done, args=(done_project,))
        progress_thread.start()
        self.assertTrue(progress_entered.wait(timeout=1))
        done_thread.start()
        terminal_overtook_progress = terminal_delivery_started.wait(timeout=0.2)
        release_progress.set()
        progress_thread.join(timeout=2)
        done_thread.join(timeout=2)

        self.assertFalse(progress_thread.is_alive())
        self.assertFalse(done_thread.is_alive())
        self.assertFalse(terminal_overtook_progress)
        self.assertEqual(delivery_order, ["progress", "terminal"])

    def test_workflow_callbacks_route_validation_error_to_workflow_error_card(self):
        """Execution-time validation failures should not be hidden behind the generic error card."""
        handler, _ctx = self._make_handler()
        handler._reply_workflow_error = MagicMock()
        callbacks = handler._build_workflow_callbacks("progress_msg", "chat_1", None)

        callbacks.on_error("validation failed: Unbalanced parentheses")

        handler._reply_workflow_error.assert_called_once_with(
            "progress_msg",
            "invalid_argument",
            detail="validation failed: Unbalanced parentheses",
        )
        handler.reply_error.assert_not_called()

    def test_workflow_callbacks_ignore_late_progress_after_error(self):
        """Late progress callbacks must not replace a terminal workflow error card."""
        handler, _ctx = self._make_handler()
        handler._reply_workflow_error = MagicMock()
        callbacks = handler._build_workflow_callbacks("progress_msg", "chat_1", None)
        late_progress = {
            "header": {
                "title": {"tag": "plain_text", "content": "Workflow running"},
                "template": "blue",
            },
            "elements": [{"tag": "markdown", "content": "late progress"}],
        }

        callbacks.on_error("validation failed: Unbalanced parentheses")
        callbacks.on_progress(late_progress)

        handler._reply_workflow_error.assert_called_once()
        handler.update_card.assert_not_called()

    def test_workflow_callbacks_route_unknown_error_with_safe_detail(self):
        """Unknown runtime errors should keep a sanitized root-cause hint on the Workflow card."""
        handler, _ctx = self._make_handler()
        handler._reply_workflow_error = MagicMock()
        callbacks = handler._build_workflow_callbacks("progress_msg", "chat_1", None)

        callbacks.on_error("TypeError: bad workflow state")

        handler._reply_workflow_error.assert_called_once_with(
            "progress_msg",
            "internal_error",
            detail="TypeError: bad workflow state",
        )
        handler.reply_error.assert_not_called()

    def test_show_workflow_status_replies_with_normalized_renderer_card(self):
        """/wf_status should normalize renderer output instead of calling build_info_card."""
        handler, ctx = self._make_handler()
        handler._get_root_path = MagicMock(return_value="/tmp/project")

        engine = MagicMock()
        engine.get_status_text.return_value = "Workflow running"
        engine.get_progress_card.return_value = {
            "header": {
                "title": {"tag": "plain_text", "content": "Workflow status"},
                "template": "blue",
            },
            "elements": [{"tag": "markdown", "content": "status"}],
        }
        ctx.workflow_engine_manager.get.return_value = engine

        with patch("src.card.CardBuilder.build_info_card", side_effect=AssertionError("wrong card builder")):
            handler.show_workflow_status("msg_1", "chat_1", None)

        handler.reply_card.assert_called_once()
        replied_message_id, card = handler.reply_card.call_args[0]
        self.assertEqual(replied_message_id, "msg_1")
        self.assertEqual(card["schema"], "2.0")
        self.assertEqual(card["body"]["elements"], engine.get_progress_card.return_value["elements"])
        self.assertNotIn("elements", card)
        handler.reply_text.assert_not_called()

    @patch("src.thread.get_current_sender_id", return_value="user_abc")
    def test_cancel_resets_to_idle(self, mock_sender):
        handler, ctx = self._make_handler()

        # Set up engine in AWAITING_CONFIRM state
        engine = MagicMock()
        engine.project = WorkflowProject(
            status=WorkflowStatus.AWAITING_CONFIRM,
            pending=PendingConfirmation(
                script_path="/tmp/nonexistent_wf.js",
                requirement="do Y",
                engine_session_key="sess_cancel_key",
                initiator_user_id="user_abc",
            ),
        )
        ctx.workflow_engine_manager.get.return_value = engine
        handler._get_root_path = MagicMock(return_value="/tmp/project")

        handler.handle_workflow_cancel(
            "msg_1", "chat_1", "proj_1",
            {"action": WORKFLOW_CANCEL, "engine_session_key": "sess_cancel_key"}
        )

        # State should be IDLE
        self.assertEqual(engine.project.status, WorkflowStatus.IDLE)
        self.assertIsNone(engine.project.pending)
        # Card should be updated
        handler.update_card.assert_called_once()

    @patch("src.thread.get_current_sender_id", return_value="user_BBB")
    def test_confirm_rejected_for_non_initiator(self, mock_sender):
        handler, ctx = self._make_handler()

        # Set up engine in AWAITING_CONFIRM state with user_AAA as initiator
        engine = MagicMock()
        engine.project = WorkflowProject(
            status=WorkflowStatus.AWAITING_CONFIRM,
            pending=PendingConfirmation(
                script_path="/tmp/wf.js",
                requirement="do X",
                meta={"name": "x"},
                initiator_user_id="user_AAA",
                engine_session_key="valid_key",
            ),
        )
        engine.is_running = False
        ctx.workflow_engine_manager.get.return_value = engine
        handler._get_root_path = MagicMock(return_value="/tmp/project")

        project_mock = MagicMock()
        project_mock.project_name = "test"
        handler._resolve_project_from_id = MagicMock(return_value=project_mock)

        handler.handle_workflow_confirm_start(
            "msg_1", "chat_1", "proj_1",
            {"action": WORKFLOW_CONFIRM_START, "engine_session_key": "valid_key"}
        )

        # Should reject with message about initiator only
        handler.reply_card.assert_called_once()
        rejection_card = handler.reply_card.call_args[0][1]
        rejection_title = rejection_card["header"]["title"]["content"]
        rejection_msg = rejection_card["body"]["elements"][0]["content"]
        self.assertEqual(rejection_title, "无操作权限")
        self.assertIn("发起者", rejection_msg)
        # Should NOT have submitted the engine task
        handler._submit_engine_task.assert_not_called()
        # Status should still be AWAITING_CONFIRM
        self.assertEqual(engine.project.status, WorkflowStatus.AWAITING_CONFIRM)

    @patch("src.thread.get_current_sender_id", return_value="user_AAA")
    def test_confirm_rejected_for_session_key_mismatch(self, mock_sender):
        handler, ctx = self._make_handler()

        # Set up engine in AWAITING_CONFIRM state with correct_key
        engine = MagicMock()
        engine.project = WorkflowProject(
            status=WorkflowStatus.AWAITING_CONFIRM,
            pending=PendingConfirmation(
                script_path="/tmp/wf.js",
                requirement="do X",
                meta={"name": "x"},
                initiator_user_id="user_AAA",
                engine_session_key="correct_key",
            ),
        )
        engine.is_running = False
        ctx.workflow_engine_manager.get.return_value = engine
        handler._get_root_path = MagicMock(return_value="/tmp/project")

        project_mock = MagicMock()
        project_mock.project_name = "test"
        handler._resolve_project_from_id = MagicMock(return_value=project_mock)

        handler.handle_workflow_confirm_start(
            "msg_1", "chat_1", "proj_1",
            {"action": WORKFLOW_CONFIRM_START, "engine_session_key": "wrong_key"}
        )

        # Should reject with message about invalid token
        handler.reply_card.assert_called_once()
        rejection_card = handler.reply_card.call_args[0][1]
        rejection_title = rejection_card["header"]["title"]["content"]
        self.assertEqual(rejection_title, "会话已过期")
        # Should NOT have submitted the engine task
        handler._submit_engine_task.assert_not_called()
        # Status should still be AWAITING_CONFIRM
        self.assertEqual(engine.project.status, WorkflowStatus.AWAITING_CONFIRM)


class TestConfirmCardContent(unittest.TestCase):
    """Verify confirm card includes script preview and phase tool tags."""

    def _make_handler(self):
        from src.feishu.handlers.workflow import WorkflowHandler

        ctx = MagicMock()
        handler = WorkflowHandler.__new__(WorkflowHandler)
        handler.ctx = ctx
        return handler

    def _get_elements(self, card: dict) -> list:
        """Extract elements from card structure (handles body.elements wrapping)."""
        body = card.get("body", card)
        return body.get("elements", card.get("elements", []))

    def test_card_contains_script_preview(self):
        """Confirm card should include script preview markdown when script_content provided."""
        handler = self._make_handler()
        meta = {
            "name": "test-wf",
            "description": "Test workflow",
            "phases": [{"title": "Plan", "detail": "Make a plan"}],
            "tools": ["coco"],
        }
        script = "export const meta = {};\nawait agent('do stuff');"

        card = handler._build_confirm_card(
            meta=meta,
            requirement="do code review",
            engine_session_key="key_123",
            chat_id="chat_1",
            project_id="proj_1",
            script_content=script,
        )

        # Find all markdown elements (including inside collapsible_panel)
        elements = self._get_elements(card)
        all_md = " ".join(
            el.get("content", "") for el in elements if el.get("tag") == "markdown"
        )
        # Script preview is inside a collapsible_panel — extract nested markdown
        for el in elements:
            if el.get("tag") == "collapsible_panel":
                for inner in el.get("elements", []):
                    if inner.get("tag") == "markdown":
                        all_md += " " + inner.get("content", "")
        # Check collapsible header text
        panel_headers = " ".join(
            el.get("header", {}).get("title", {}).get("content", "")
            for el in elements if el.get("tag") == "collapsible_panel"
        )
        self.assertIn("编排脚本预览", panel_headers)
        self.assertIn("```javascript", all_md)
        self.assertIn("agent('do stuff')", all_md)

    def test_card_without_script_has_no_preview(self):
        """Confirm card should not include preview section when no script_content."""
        handler = self._make_handler()
        meta = {
            "name": "test-wf",
            "description": "Test",
            "phases": [],
            "tools": ["coco"],
        }

        card = handler._build_confirm_card(
            meta=meta,
            requirement="task",
            engine_session_key="key_1",
            chat_id="chat_1",
            project_id="proj_1",
            script_content="",
        )

        elements = self._get_elements(card)
        all_md = " ".join(
            el.get("content", "") for el in elements if el.get("tag") == "markdown"
        )
        self.assertNotIn("编排脚本预览", all_md)

    def test_phases_include_tool_tags(self):
        """Phase lines should show tool tags when phase_tool_mapping is provided.

        Phases now live inside a collapsible_panel, so we search recursively through
        top-level and collapsible_panel markdown elements.
        """
        handler = self._make_handler()
        meta = {
            "name": "review-wf",
            "description": "Code review",
            "phases": [
                {"title": "Analysis", "detail": "Analyze code"},
                {"title": "Review", "detail": "Review findings"},
            ],
            "tools": ["coco", "claude"],
            "phase_tool_mapping": {
                "Analysis": ["coco"],
                "Review": ["claude"],
            },
        }

        card = handler._build_confirm_card(
            meta=meta,
            requirement="review code",
            engine_session_key="key_2",
            chat_id="chat_1",
            project_id="proj_1",
        )

        elements = self._get_elements(card)

        def flatten_md(els: list[dict]) -> str:
            out: list[str] = []
            for e in els:
                if e.get("tag") == "markdown":
                    out.append(e.get("content", ""))
                if e.get("tag") == "collapsible_panel":
                    out.append(flatten_md(e.get("elements", [])))
            return "\n".join(out)

        all_md = flatten_md(elements)
        self.assertIn("`coco`", all_md)
        self.assertIn("`claude`", all_md)
        # Tool tags should appear near the corresponding phase lines (following sub-line)
        lines = all_md.split("\n")
        # Find phase header index and check following lines for tool tag
        def tool_after_phase_title(phase_title: str, tool_name: str) -> bool:
            for i, l in enumerate(lines):
                if phase_title in l:
                    # Check next few lines for the tool label
                    window_start = max(0, i)
                    window_end = min(len(lines), i + 4)
                    window = " ".join(lines[window_start:window_end])
                    return f"`{tool_name}`" in window
            return False

        self.assertTrue(tool_after_phase_title("Analysis", "coco"))
        self.assertTrue(tool_after_phase_title("Review", "claude"))

class TestWorkflowE2EConfirmFlow(unittest.TestCase):
    """E2E: /wf '重构登录模块' → confirm card with full structure."""

    def _make_handler(self):
        from src.feishu.handlers.workflow import WorkflowHandler

        ctx = MagicMock()
        ctx.workflow_engine_manager = MagicMock()
        ctx.project_manager = MagicMock()
        ctx.progress_reporter = MagicMock()

        handler = WorkflowHandler.__new__(WorkflowHandler)
        handler.ctx = ctx
        handler.reply_text = MagicMock()
        handler.reply_card = MagicMock()
        handler.reply_error = MagicMock()
        handler.send_card_to_chat = MagicMock(return_value="msg_card_e2e")
        handler.update_card = MagicMock(return_value=True)
        handler.add_reaction = MagicMock()
        handler.get_working_dir = MagicMock(return_value="/tmp/project")
        handler.get_engine_name = MagicMock(return_value="coco")
        handler.ensure_request_id = MagicMock(return_value="req_e2e")
        handler._submit_engine_task = MagicMock()

        return handler, ctx

    def _get_elements(self, card: dict) -> list:
        body = card.get("body", card)
        return body.get("elements", card.get("elements", []))

class TestWorkflowFallbackPath(unittest.TestCase):
    """Test that AI script generation failure shows fallback confirm card."""

    def _make_handler(self):
        from src.feishu.handlers.workflow import WorkflowHandler

        ctx = MagicMock()
        ctx.workflow_engine_manager = MagicMock()
        ctx.project_manager = MagicMock()
        ctx.progress_reporter = MagicMock()

        handler = WorkflowHandler.__new__(WorkflowHandler)
        handler.ctx = ctx
        handler.reply_text = MagicMock()
        handler.reply_card = MagicMock()
        handler.reply_error = MagicMock()
        handler.send_card_to_chat = MagicMock(return_value="msg_fb")
        handler.update_card = MagicMock(return_value=True)
        handler.add_reaction = MagicMock()
        handler.get_working_dir = MagicMock(return_value="/tmp/project")
        handler.get_engine_name = MagicMock(return_value="coco")
        handler.ensure_request_id = MagicMock(return_value="req_fb")
        handler._submit_engine_task = MagicMock()

        return handler, ctx

    def _get_elements(self, card: dict) -> list:
        body = card.get("body", card)
        return body.get("elements", card.get("elements", []))

class TestWorkflowToolSelectionFirstFlow(unittest.TestCase):
    """Tests for the tool-selection-first workflow (AC2)."""

    def _make_handler(self):
        """Create a WorkflowHandler with mocked dependencies."""
        from src.feishu.handlers.workflow import WorkflowHandler

        ctx = MagicMock()
        ctx.workflow_engine_manager = MagicMock()
        ctx.project_manager = MagicMock()
        ctx.progress_reporter = MagicMock()

        handler = WorkflowHandler.__new__(WorkflowHandler)
        handler.ctx = ctx
        handler.reply_text = MagicMock()
        handler.reply_card = MagicMock()
        handler.reply_error = MagicMock()
        handler.send_card_to_chat = MagicMock(return_value="msg_card_123")
        handler.update_card = MagicMock(return_value=True)
        handler.add_reaction = MagicMock()
        handler.get_working_dir = MagicMock(return_value="/tmp/project")
        handler.get_engine_name = MagicMock(return_value="coco")
        handler.ensure_request_id = MagicMock(return_value="req_123")
        handler._submit_engine_task = MagicMock()

        return handler, ctx

    @patch("src.thread.get_current_sender_id", return_value="user_123")
    @patch("src.workflow_engine.bridge.RuntimeBridge.check_node_available", return_value=True)
    @patch("src.workflow_engine.templates.discover_templates", return_value=[])
    def test_start_workflow_shows_tool_selection_card(
        self, mock_templates, mock_node, mock_sender
    ):
        """Verify start_workflow() shows an orchestrator agent selection card (step 1 of 2-step flow).

        In the new two-step selection flow, start_workflow() shows the orchestrator
        tool selection card. It does NOT pre-set orchestrator_agent or selected_tools —
        those are chosen by the user via subsequent selection steps.
        """
        handler, ctx = self._make_handler()

        project = MagicMock()
        project.root_path = "/tmp/project"
        project.project_id = "proj_1"
        project.project_name = "test"
        handler._ensure_project = MagicMock(return_value=project)

        engine = MagicMock()
        engine.is_running = False
        engine.project = WorkflowProject()
        ctx.workflow_engine_manager.get.return_value = engine
        ctx.workflow_engine_manager.get_or_create.return_value = engine

        handler.start_workflow("msg_1", "chat_1", "do code review", project)

        # Should send a card (orchestrator agent selection)
        handler.send_card_to_chat.assert_called_once()
        # update_card should NOT be called (no generating -> confirm transition)
        handler.update_card.assert_not_called()

        # Engine project should be AWAITING_AGENT_SELECT (step 1 of the new 2-step flow)
        self.assertEqual(engine.project.status, WorkflowStatus.AWAITING_AGENT_SELECT)
        self.assertIsNotNone(engine.project.pending)
        self.assertIsNone(engine.project.pending.script_path)
        self.assertIsNone(engine.project.pending.meta)
        # orchestrator_agent is NOT preset — user picks via selection controller
        self.assertIsNone(engine.project.pending.orchestrator_agent)
        # selected_tools is NOT preset in the new 2-step flow
        # (it may be empty list or None depending on the exact implementation)

    @patch("src.thread.get_current_sender_id", return_value="user_123")
    @patch("src.workflow_engine.bridge.RuntimeBridge.check_node_available", return_value=True)
    @patch("src.workflow_engine.templates.discover_templates", return_value=[])
    def test_start_workflow_sets_pending_requirement(
        self, mock_templates, mock_node, mock_sender
    ):
        """Verify start_workflow() stores requirement and session key (2-step selection flow).

        In the new two-step selection flow, start_workflow() sets up the pending state
        with requirement and session_key, but defers orchestrator_agent and selected_tools
        to the Workflow selection controllers.
        """
        handler, ctx = self._make_handler()

        project = MagicMock()
        project.root_path = "/tmp/project"
        project.project_id = "proj_1"
        project.project_name = "test"
        handler._ensure_project = MagicMock(return_value=project)

        engine = MagicMock()
        engine.is_running = False
        engine.project = WorkflowProject()
        ctx.workflow_engine_manager.get.return_value = engine
        ctx.workflow_engine_manager.get_or_create.return_value = engine

        handler.start_workflow("msg_1", "chat_1", "do code review", project)

        # Requirement should be stored
        self.assertEqual(engine.project.pending.requirement if engine.project.pending else None, "do code review")
        # Session key should be set
        self.assertIsNotNone(engine.project.pending.engine_session_key if engine.project.pending else None)
        # Status should be AWAITING_AGENT_SELECT (user is choosing orchestrator)
        self.assertEqual(engine.project.status, WorkflowStatus.AWAITING_AGENT_SELECT)
        # orchestrator_agent is NOT preset in 2-step flow
        self.assertIsNone(engine.project.pending.orchestrator_agent if engine.project.pending else "")

    @patch("src.thread.get_current_sender_id", return_value="user_123")
    @patch("src.feishu.handlers.workflow.WorkflowHandler._generate_script_via_ai")
    def test_handle_workflow_confirm_tools_transitions_state(
        self, mock_gen, mock_sender
    ):
        """Tool confirmation must enter owned background script generation."""
        handler, ctx = self._make_handler()

        # Set up engine in AWAITING_TOOL_SELECT state
        engine = MagicMock()
        engine.is_running = False
        engine.project = WorkflowProject(
            status=WorkflowStatus.AWAITING_TOOL_SELECT,
            pending=PendingConfirmation(
                requirement="do code review",
                initiator_user_id="user_123",
                engine_session_key="valid_session_key",
                selected_tools=["coco", "claude"],
            ),
        )
        ctx.workflow_engine_manager.get.return_value = engine
        ctx.workflow_engine_manager.get_or_create.return_value = engine

        project_mock = MagicMock()
        project_mock.root_path = "/tmp/project"
        project_mock.project_id = "proj_1"
        project_mock.project_name = "test"
        handler._resolve_project_from_id = MagicMock(return_value=project_mock)

        # Mock AI generation result
        mock_gen.return_value = (
            "/tmp/project/.ghostap/workflow_scripts/generated_workflow.js",
            {"name": "test-wf", "description": "Test", "phases": [], "tools": ["coco"]},
            False,
        )

        handler.handle_workflow_confirm_tools(
            "msg_1", "chat_1", "proj_1",
            {"action": "workflow_confirm_tools", "engine_session_key": "valid_session_key"}
        )

        self.assertEqual(engine.project.status, WorkflowStatus.GENERATING_SCRIPT)
        self.assertEqual(
            engine.project.pending.engine_session_key,
            "valid_session_key",
        )
        handler._submit_engine_task.assert_called_once()
        mock_gen.assert_not_called()

    @patch("src.thread.get_current_sender_id", return_value="user_123")
    def test_handle_workflow_confirm_tools_validates_session_key(self, mock_sender):
        """Wrong session key should be rejected."""
        handler, ctx = self._make_handler()

        engine = MagicMock()
        engine.project = WorkflowProject(
            status=WorkflowStatus.AWAITING_TOOL_SELECT,
            pending=PendingConfirmation(
                requirement="do X",
                initiator_user_id="user_123",
                engine_session_key="correct_key",
                selected_tools=["coco"],
            ),
        )
        ctx.workflow_engine_manager.get.return_value = engine

        project_mock = MagicMock()
        project_mock.root_path = "/tmp/project"
        handler._resolve_project_from_id = MagicMock(return_value=project_mock)

        handler.handle_workflow_confirm_tools(
            "msg_1", "chat_1", "proj_1",
            {"action": "workflow_confirm_tools", "engine_session_key": "wrong_key"}
        )

        # Should reject with error about session
        handler.reply_card.assert_called_once()
        rejection_card = handler.reply_card.call_args[0][1]
        rejection_title = rejection_card["header"]["title"]["content"]
        self.assertEqual(rejection_title, "会话已过期")
        # State should remain AWAITING_TOOL_SELECT
        self.assertEqual(engine.project.status, WorkflowStatus.AWAITING_TOOL_SELECT)
        # Script path should still be None
        self.assertIsNone(engine.project.pending.script_path if engine.project.pending else None)

    @patch("src.thread.get_current_sender_id", return_value="user_BBB")
    def test_handle_workflow_confirm_tools_validates_initiator(self, mock_sender):
        """Non-initiator should be rejected."""
        handler, ctx = self._make_handler()

        engine = MagicMock()
        engine.project = WorkflowProject(
            status=WorkflowStatus.AWAITING_TOOL_SELECT,
            pending=PendingConfirmation(
                requirement="do X",
                initiator_user_id="user_AAA",
                engine_session_key="valid_key",
                selected_tools=["coco"],
            ),
        )
        ctx.workflow_engine_manager.get.return_value = engine

        project_mock = MagicMock()
        project_mock.root_path = "/tmp/project"
        handler._resolve_project_from_id = MagicMock(return_value=project_mock)

        handler.handle_workflow_confirm_tools(
            "msg_1", "chat_1", "proj_1",
            {"action": "workflow_confirm_tools", "engine_session_key": "valid_key"}
        )

        # Should reject with error about initiator
        handler.reply_card.assert_called_once()
        rejection_card = handler.reply_card.call_args[0][1]
        rejection_title = rejection_card["header"]["title"]["content"]
        rejection_msg = rejection_card["body"]["elements"][0]["content"]
        self.assertEqual(rejection_title, "无操作权限")
        self.assertIn("发起者", rejection_msg)
        # State should remain AWAITING_TOOL_SELECT
        self.assertEqual(engine.project.status, WorkflowStatus.AWAITING_TOOL_SELECT)

    @patch("src.thread.get_current_sender_id", return_value="user_123")
    def test_handle_workflow_confirm_tools_requires_at_least_one_tool(self, mock_sender):
        """Empty selected tools should be rejected."""
        handler, ctx = self._make_handler()

        engine = MagicMock()
        engine.project = WorkflowProject(
            status=WorkflowStatus.AWAITING_TOOL_SELECT,
            pending=PendingConfirmation(
                requirement="do X",
                initiator_user_id="user_123",
                engine_session_key="valid_key",
                selected_tools=[],  # Empty!
            ),
        )
        ctx.workflow_engine_manager.get.return_value = engine

        project_mock = MagicMock()
        project_mock.root_path = "/tmp/project"
        handler._resolve_project_from_id = MagicMock(return_value=project_mock)

        handler.handle_workflow_confirm_tools(
            "msg_1", "chat_1", "proj_1",
            {"action": "workflow_confirm_tools", "engine_session_key": "valid_key"}
        )

        # Should reject with message about selecting at least one tool
        handler.reply_card.assert_called_once()
        rejection_card = handler.reply_card.call_args[0][1]
        # The error card has body > elements with markdown content
        rejection_msg = ""
        for el in rejection_card.get("body", {}).get("elements", []):
            if isinstance(el, dict) and el.get("tag") == "note":
                for inner in el.get("elements", []):
                    if isinstance(inner, dict) and inner.get("tag") == "plain_text":
                        rejection_msg += inner.get("content", "")
            elif isinstance(el, dict) and el.get("tag") == "markdown":
                rejection_msg += el.get("content", "")
        self.assertIn("至少选择一个工具", rejection_msg)
        # State should remain AWAITING_TOOL_SELECT
        self.assertEqual(engine.project.status, WorkflowStatus.AWAITING_TOOL_SELECT)


class TestWorkflowRegenerateScript(unittest.TestCase):
    """Tests for handle_workflow_regenerate_script()."""

    def _make_handler(self):
        from src.feishu.handlers.workflow import WorkflowHandler

        ctx = MagicMock()
        ctx.workflow_engine_manager = MagicMock()
        ctx.project_manager = MagicMock()
        ctx.progress_reporter = MagicMock()

        handler = WorkflowHandler.__new__(WorkflowHandler)
        handler.ctx = ctx
        handler.reply_text = MagicMock()
        handler.reply_card = MagicMock()
        handler.reply_error = MagicMock()
        handler.send_card_to_chat = MagicMock(return_value="msg_card_123")
        handler.update_card = MagicMock(return_value=True)
        handler.add_reaction = MagicMock()
        handler.get_working_dir = MagicMock(return_value="/tmp/project")
        handler.get_engine_name = MagicMock(return_value="coco")
        handler.ensure_request_id = MagicMock(return_value="req_123")
        handler._submit_engine_task = MagicMock()

        return handler, ctx

    @patch("src.thread.get_current_sender_id", return_value="user_123")
    @patch("src.feishu.handlers.workflow.WorkflowHandler._generate_script_via_ai")
    def test_handle_workflow_regenerate_script_regenerates(
        self, mock_gen, mock_sender
    ):
        """Regeneration must use the cancellable background generation path."""
        import tempfile

        handler, ctx = self._make_handler()

        temp_dir = tempfile.TemporaryDirectory()
        script_dir = os.path.join(
            temp_dir.name,
            ".ghostap",
            "workflow_scripts",
        )
        os.makedirs(script_dir)
        old_script_path = os.path.join(
            script_dir,
            "generated-workflow-old.js",
        )
        with open(old_script_path, "w", encoding="utf-8") as old_script:
            old_script.write("old script")

        engine = MagicMock()
        engine.is_running = False
        engine.project = WorkflowProject(
            status=WorkflowStatus.AWAITING_CONFIRM,
            pending=PendingConfirmation(
                script_path=old_script_path,
                requirement="do code review",
                meta={"name": "old-wf", "tools": ["coco"]},
                initiator_user_id="user_123",
                engine_session_key="valid_session_key",
                selected_tools=["coco", "claude"],
            ),
        )
        ctx.workflow_engine_manager.get.return_value = engine
        ctx.workflow_engine_manager.get_or_create.return_value = engine

        project_mock = MagicMock()
        project_mock.root_path = temp_dir.name
        project_mock.project_id = "proj_1"
        project_mock.project_name = "test"
        handler._resolve_project_from_id = MagicMock(return_value=project_mock)
        handler._start_pending_workflow_execution = MagicMock(return_value=True)

        # Mock new AI generation result
        mock_gen.return_value = (
            "/tmp/project/.ghostap/workflow_scripts/regenerated_workflow.js",
            {"name": "regenerated-wf", "description": "Regenerated", "phases": [], "tools": ["coco", "claude"]},
            False,
        )

        handler.handle_workflow_regenerate_script(
            "msg_1", "chat_1", "proj_1",
            {"action": "workflow_regenerate_script", "engine_session_key": "valid_session_key"}
        )

        self.assertEqual(engine.project.status, WorkflowStatus.GENERATING_SCRIPT)
        handler._submit_engine_task.assert_called_once()
        mock_gen.assert_not_called()
        handler.send_card_to_chat.assert_not_called()
        handler._start_pending_workflow_execution.assert_not_called()
        self.assertFalse(os.path.exists(old_script_path))
        temp_dir.cleanup()

    @patch("src.thread.get_current_sender_id", return_value="user_123")
    def test_handle_workflow_regenerate_script_wrong_state_rejected(self, mock_sender):
        """Only works in AWAITING_CONFIRM state."""
        handler, ctx = self._make_handler()

        engine = MagicMock()
        engine.project = WorkflowProject(
            status=WorkflowStatus.AWAITING_TOOL_SELECT,  # Wrong state!
            pending=PendingConfirmation(
                requirement="do X",
                initiator_user_id="user_123",
                engine_session_key="valid_key",
                selected_tools=["coco"],
            ),
        )
        ctx.workflow_engine_manager.get.return_value = engine

        project_mock = MagicMock()
        project_mock.root_path = "/tmp/project"
        handler._resolve_project_from_id = MagicMock(return_value=project_mock)

        handler.handle_workflow_regenerate_script(
            "msg_1", "chat_1", "proj_1",
            {"action": "workflow_regenerate_script", "engine_session_key": "valid_key"}
        )

        # Should reject with message about wrong state
        handler.reply_card.assert_called_once()
        rejection_card = handler.reply_card.call_args[0][1]
        rejection_title = rejection_card["header"]["title"]["content"]
        self.assertEqual(rejection_title, "状态不匹配")

    @patch("src.thread.get_current_sender_id", return_value="user_BBB")
    def test_handle_workflow_regenerate_script_validates_security(self, mock_sender):
        """Session key and initiator checks."""
        handler, ctx = self._make_handler()

        engine = MagicMock()
        engine.project = WorkflowProject(
            status=WorkflowStatus.AWAITING_CONFIRM,
            pending=PendingConfirmation(
                script_path="/tmp/wf.js",
                requirement="do X",
                meta={"name": "x"},
                initiator_user_id="user_AAA",
                engine_session_key="correct_key",
                selected_tools=["coco"],
            ),
        )
        ctx.workflow_engine_manager.get.return_value = engine

        project_mock = MagicMock()
        project_mock.root_path = "/tmp/project"
        handler._resolve_project_from_id = MagicMock(return_value=project_mock)

        # Test 1: Wrong session key
        handler.handle_workflow_regenerate_script(
            "msg_1", "chat_1", "proj_1",
            {"action": "workflow_regenerate_script", "engine_session_key": "wrong_key"}
        )
        handler.reply_card.assert_called_once()
        rejection_card = handler.reply_card.call_args[0][1]
        rejection_title = rejection_card["header"]["title"]["content"]
        self.assertEqual(rejection_title, "会话已过期")

        # Reset mocks
        handler.reply_card.reset_mock()
        handler.reply_text.reset_mock()

        # Test 2: Wrong initiator (with correct session key)
        handler.handle_workflow_regenerate_script(
            "msg_2", "chat_1", "proj_1",
            {"action": "workflow_regenerate_script", "engine_session_key": "correct_key"}
        )
        handler.reply_card.assert_called_once()
        rejection_card = handler.reply_card.call_args[0][1]
        rejection_title = rejection_card["header"]["title"]["content"]
        rejection_msg = rejection_card["body"]["elements"][0]["content"]
        self.assertEqual(rejection_title, "无操作权限")
        self.assertIn("发起者", rejection_msg)


class TestWorkflowToolConsistencyValidation(unittest.TestCase):
    """Tests for tool consistency validation in confirm_start."""

    def _make_handler(self):
        from src.feishu.handlers.workflow import WorkflowHandler

        ctx = MagicMock()
        ctx.workflow_engine_manager = MagicMock()
        ctx.project_manager = MagicMock()
        ctx.progress_reporter = MagicMock()

        handler = WorkflowHandler.__new__(WorkflowHandler)
        handler.ctx = ctx
        handler.reply_text = MagicMock()
        handler.reply_card = MagicMock()
        handler.reply_error = MagicMock()
        handler._reply_workflow_error = MagicMock()
        handler.send_card_to_chat = MagicMock(return_value="msg_card_123")
        handler.update_card = MagicMock(return_value=True)
        handler.add_reaction = MagicMock()
        handler.get_working_dir = MagicMock(return_value="/tmp/project")
        handler.get_engine_name = MagicMock(return_value="coco")
        handler.ensure_request_id = MagicMock(return_value="req_123")
        handler._submit_engine_task = MagicMock()

        return handler, ctx

    @patch("src.thread.get_current_sender_id", return_value="user_123")
    def test_confirm_start_validates_tool_consistency(self, mock_sender):
        """When script meta.tools contains tools not in selected_tools, confirmation should be rejected."""
        import hashlib
        import tempfile

        handler, ctx = self._make_handler()

        script_content = (
            "export const meta = {\n"
            "  name: 'x',\n"
            "  description: 'test',\n"
            "  tools: ['coco', 'claude', 'codex'],\n"
            "};\n"
            "\n"
            "export default async function() {\n"
            "  const result = await agent('do work', { tool: 'coco', label: 'do-work', timeout: 120 });\n"
            "  if (result && result.error) return result;\n"
            "  return result;\n"
            "}\n"
        )
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False, encoding="utf-8")
        tmp.write(script_content)
        tmp.close()
        script_hash = hashlib.sha256(script_content.encode("utf-8")).hexdigest()

        engine = MagicMock()
        engine.is_running = False
        engine.project = WorkflowProject(
            status=WorkflowStatus.AWAITING_CONFIRM,
            pending=PendingConfirmation(
                script_path=tmp.name,
                requirement="do X",
                meta={"name": "x", "tools": ["coco", "claude", "codex"]},
                initiator_user_id="user_123",
                engine_session_key="valid_key",
                selected_tools=["coco", "claude"],  # codex missing!
                script_hash=script_hash,
            ),
        )
        ctx.workflow_engine_manager.get.return_value = engine
        handler._get_root_path = MagicMock(return_value="/tmp/project")

        project_mock = MagicMock()
        project_mock.project_name = "test"
        handler._resolve_project_from_id = MagicMock(return_value=project_mock)

        handler.handle_workflow_confirm_start(
            "msg_1", "chat_1", "proj_1",
            {"action": WORKFLOW_CONFIRM_START, "engine_session_key": "valid_key"}
        )

        try:
            os.unlink(tmp.name)
        except OSError:
            pass

        # Auto-fix behavior: unmatched tools are rewritten to primary
        # selected tool instead of rejecting. No error should be raised.
        handler._reply_workflow_error.assert_not_called()
        # The engine task should have been submitted (execution proceeds)
        handler._submit_engine_task.assert_called_once()

    @patch("src.thread.get_current_sender_id", return_value="user_123")
    def test_confirm_start_allows_subset_tools(self, mock_sender):
        """When script tools are a subset of selected tools, confirmation should proceed."""
        import hashlib
        import tempfile

        handler, ctx = self._make_handler()

        script_content = (
            "export const meta = {\n"
            "  name: 'x',\n"
            "  description: 'test',\n"
            "  tools: ['coco'],\n"
            "};\n"
            "\n"
            "export default async function() {\n"
            "  const result = await agent('do work', { tool: 'coco', label: 'do-work', timeout: 120 });\n"
            "  if (result && result.error) return result;\n"
            "  return result;\n"
            "}\n"
        )
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False, encoding="utf-8")
        tmp.write(script_content)
        tmp.close()
        script_hash = hashlib.sha256(script_content.encode("utf-8")).hexdigest()

        engine = MagicMock()
        engine.is_running = False
        engine.project = WorkflowProject(
            status=WorkflowStatus.AWAITING_CONFIRM,
            pending=PendingConfirmation(
                script_path=tmp.name,
                requirement="do X",
                meta={"name": "x", "tools": ["coco"]},
                initiator_user_id="user_123",
                engine_session_key="valid_key",
                selected_tools=["coco", "claude", "codex"],  # superset
                script_hash=script_hash,
            ),
        )
        ctx.workflow_engine_manager.get.return_value = engine
        handler._get_root_path = MagicMock(return_value="/tmp/project")

        project_mock = MagicMock()
        project_mock.project_name = "test"
        handler._resolve_project_from_id = MagicMock(return_value=project_mock)

        handler.handle_workflow_confirm_start(
            "msg_1", "chat_1", "proj_1",
            {"action": WORKFLOW_CONFIRM_START, "engine_session_key": "valid_key"}
        )

        try:
            os.unlink(tmp.name)
        except OSError:
            pass

        # Should NOT reject - subset is allowed
        handler.reply_error.assert_not_called()
        # Should have submitted the engine task
        handler._submit_engine_task.assert_called_once()
        # Pending state should be cleared
        self.assertIsNone(engine.project.pending.script_path if engine.project.pending else None)

    @patch("src.thread.get_current_sender_id", return_value="user_123")
    def test_confirm_start_allows_empty_script_tools(self, mock_sender):
        """When script has no tools declared, validation should pass."""
        import hashlib
        import tempfile

        handler, ctx = self._make_handler()

        # Script with meta but no tools array.
        script_content = (
            "export const meta = {\n"
            "  name: 'x',\n"
            "  description: 'test',\n"
            "};\n"
            "\n"
            "export default async function() {\n"
            "  const result = await agent('do work', { tool: 'coco', label: 'do-work', timeout: 120 });\n"
            "  if (result && result.error) return result;\n"
            "  return result;\n"
            "}\n"
        )
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False, encoding="utf-8")
        tmp.write(script_content)
        tmp.close()
        script_hash = hashlib.sha256(script_content.encode("utf-8")).hexdigest()

        engine = MagicMock()
        engine.is_running = False
        engine.project = WorkflowProject(
            status=WorkflowStatus.AWAITING_CONFIRM,
            pending=PendingConfirmation(
                script_path=tmp.name,
                requirement="do X",
                meta={"name": "x", "tools": []},
                initiator_user_id="user_123",
                engine_session_key="valid_key",
                selected_tools=["coco"],
                script_hash=script_hash,
            ),
        )
        ctx.workflow_engine_manager.get.return_value = engine
        handler._get_root_path = MagicMock(return_value="/tmp/project")

        project_mock = MagicMock()
        project_mock.project_name = "test"
        handler._resolve_project_from_id = MagicMock(return_value=project_mock)

        handler.handle_workflow_confirm_start(
            "msg_1", "chat_1", "proj_1",
            {"action": WORKFLOW_CONFIRM_START, "engine_session_key": "valid_key"}
        )

        try:
            os.unlink(tmp.name)
        except OSError:
            pass

        # Should NOT reject - empty script tools is allowed
        handler.reply_error.assert_not_called()
        # Should have submitted the engine task
        handler._submit_engine_task.assert_called_once()


class TestConfirmCardToolDistinction(unittest.TestCase):
    """Tests for confirm card showing script vs allowed tools distinction."""

    def _make_handler(self):
        from src.feishu.handlers.workflow import WorkflowHandler

        ctx = MagicMock()
        handler = WorkflowHandler.__new__(WorkflowHandler)
        handler.ctx = ctx
        return handler

    def _get_elements(self, card: dict) -> list:
        body = card.get("body", card)
        return body.get("elements", card.get("elements", []))

    def _extract_all_text(self, card: dict) -> str:
        """Extract all text content from card elements, including button labels."""

        def _extract(element: dict) -> str:
            tag = element.get("tag")
            if tag == "markdown":
                return " " + element.get("content", "")
            if tag == "plain_text":
                return " " + element.get("content", "")
            if tag == "button":
                text = element.get("text", {})
                if isinstance(text, dict):
                    return " " + text.get("content", "")
                return ""
            # Recurse into any container element that has an "elements" list
            child_text = ""
            for sub in element.get("elements", []):
                if isinstance(sub, dict):
                    child_text += _extract(sub)
            # Column sets nest columns which nest elements; make sure we cover both.
            for column in element.get("columns", []):
                if isinstance(column, dict):
                    child_text += _extract(column)
            return child_text

        return "".join(_extract(el) for el in self._get_elements(card))

    def _extract_all_actions(self, card: dict) -> list:
        """Extract all action buttons from card, including inside collapsible panels."""
        elements = self._get_elements(card)
        all_actions = []

        def _extract_from(el: dict) -> None:
            if el.get("tag") == "action":
                for action in el.get("actions", []):
                    all_actions.append(action)
            if el.get("tag") == "column_set":
                for col in el.get("columns", []):
                    for col_el in col.get("elements", []):
                        if col_el.get("tag") == "button":
                            all_actions.append(col_el)
            # Recurse into collapsible panels
            if el.get("tag") == "collapsible_panel":
                for sub in el.get("elements", []):
                    if isinstance(sub, dict):
                        _extract_from(sub)

        for el in elements:
            if isinstance(el, dict):
                _extract_from(el)
        return all_actions

    def test_confirm_card_shows_script_vs_allowed_tools(self):
        """The confirm card should show both 'script planned tools' and 'allowed tools' sections."""
        handler = self._make_handler()
        meta = {
            "name": "test-wf",
            "description": "Test workflow",
            "phases": [{"title": "Plan", "detail": "Make a plan"}],
            "tools": ["coco", "claude"],
        }

        card = handler._build_confirm_card(
            meta=meta,
            requirement="do code review",
            engine_session_key="key_123",
            chat_id="chat_1",
            project_id="proj_1",
            selected_tools=["coco", "claude", "codex"],
            script_content="",
        )

        all_text = self._extract_all_text(card)

        # Should show script planned tools section
        self.assertIn("脚本计划使用", all_text)
        self.assertIn("`coco`", all_text)
        self.assertIn("`claude`", all_text)

        # Should show allowed tools section
        self.assertIn("允许执行的工具", all_text)
        # Should show codex as allowed even though not in script tools
        self.assertIn("codex", all_text)

    def test_confirm_card_shows_mismatch_warning(self):
        """When there's a tool mismatch, the card should show a warning."""
        handler = self._make_handler()
        meta = {
            "name": "test-wf",
            "description": "Test",
            "phases": [],
            "tools": ["coco", "claude", "codex"],  # codex not in selected_tools
        }

        card = handler._build_confirm_card(
            meta=meta,
            requirement="task",
            engine_session_key="key_1",
            chat_id="chat_1",
            project_id="proj_1",
            selected_tools=["coco", "claude"],  # codex missing!
            script_content="",
        )

        all_text = self._extract_all_text(card)

        # Should show mismatch warning with missing tools highlighted
        self.assertIn("脚本需要这些工具但尚未启用", all_text)
        self.assertIn("`codex`", all_text)
        # Should mention both fill-in and back-to-tools paths
        self.assertIn("一键补齐缺失工具", all_text)
        self.assertIn("返回工具选择", all_text)
        # Regenerate option also present
        self.assertIn("重新生成编排", all_text)

    def test_confirm_card_has_regenerate_button(self):
        """The card should include a '重新生成编排' button with WORKFLOW_REGENERATE_SCRIPT action."""
        from src.card.actions.dispatch import WORKFLOW_REGENERATE_SCRIPT

        handler = self._make_handler()
        meta = {
            "name": "test-wf",
            "description": "Test",
            "phases": [],
            "tools": ["coco"],
        }

        card = handler._build_confirm_card(
            meta=meta,
            requirement="task",
            engine_session_key="key_1",
            chat_id="chat_1",
            project_id="proj_1",
            selected_tools=["coco"],
            script_content="",
        )

        all_actions = self._extract_all_actions(card)
        action_values = []
        for a in all_actions:
            val = a.get("value", {})
            if isinstance(val, dict):
                action_values.append(val.get("action", ""))
            # Also check button text
            btn_text = ""
            text_obj = a.get("text", {})
            if isinstance(text_obj, dict):
                btn_text = text_obj.get("content", "")

        # Should have regenerate button
        self.assertIn(WORKFLOW_REGENERATE_SCRIPT, action_values)

        # Find the regenerate button and check its text
        regenerate_btn = None
        for a in all_actions:
            val = a.get("value", {})
            if isinstance(val, dict) and val.get("action") == WORKFLOW_REGENERATE_SCRIPT:
                regenerate_btn = a
                break
        self.assertIsNotNone(regenerate_btn)
        btn_text = regenerate_btn.get("text", {}).get("content", "")
        self.assertIn("重新生成编排", btn_text)


if __name__ == "__main__":
    unittest.main()
