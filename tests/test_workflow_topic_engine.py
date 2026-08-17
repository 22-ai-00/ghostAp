"""Tests for Workflow topic-engine scoping and state isolation.

Validates that Workflow, like Deep and Spec, is a topic-scoped engine
strategy that does not interfere with chat+project programming state.

Key contract (AGENTS.md lines 124-125):
  "Deep, Spec, and Workflow are engine strategies scoped to the Feishu
  topic/root thread; they must not replace chat+project programming state."

Test categories:
  1. TestAGENTSMDScopeStatement - verify AGENTS.md contains the scope statement
  2. TestWorkflowModeStateIsolation - verify workflow doesn't change chat mode
  3. TestAutoEnterWorkflowMode - verify auto-enter routing behavior
  4. TestDispatcherRouting - verify command routing to workflow handler
  5. TestModeIndependence - verify workflow and chat mode coexist
  6. TestWorkflowEngineManagerScoping - verify engine keying and cleanup
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from src.feishu.handler_context import HandlerContext
from src.feishu.handlers.workflow import WorkflowHandler
from src.mode import InteractionMode, ModeManager

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------



def _make_handler_context(**settings_overrides) -> HandlerContext:
    """Build the minimal dependency container used by topic-scope tests."""
    settings = MagicMock()
    for name, value in settings_overrides.items():
        setattr(settings, name, value)
    mock = MagicMock
    return HandlerContext(
        settings=settings,
        api_client_factory=mock(),
        message_callback=mock(),
        coco_manager=mock(),
        claude_manager=mock(),
        aiden_manager=mock(),
        codex_manager=mock(),
        gemini_manager=mock(),
        traex_manager=mock(),
        grok_manager=mock(),
        dsh_manager=mock(),
        intent_recognizer=mock(),
        scheduler=mock(),
        project_manager=mock(),
        message_mapper=mock(),
        message_linker=mock(),
        mode_manager=ModeManager(),
        context_manager=mock(),
        deep_engine_manager=mock(),
        progress_reporter=mock(),
        spec_engine_manager=mock(),
        spec_reporter=mock(),
        thread_manager=mock(),
        image_handler_factory=mock(),
        workflow_engine_manager=mock(),
    )


def _make_workflow_handler(**settings_overrides) -> WorkflowHandler:
    """Create a WorkflowHandler with mocked context and im_client."""
    ctx = _make_handler_context(**settings_overrides)
    handler = WorkflowHandler(ctx)
    handler.im_client = MagicMock()
    handler.reply_text = MagicMock()
    handler.reply_error = MagicMock()
    handler.reply_card = MagicMock()
    handler.send_card_to_chat = MagicMock(return_value="mock_card_msg_id")
    handler.update_card = MagicMock(return_value=True)
    handler.add_reaction = MagicMock()
    handler.get_working_dir = MagicMock(return_value="/tmp/test_project")
    handler.get_engine_name = MagicMock(return_value="Workflow")
    handler.ensure_request_id = MagicMock(return_value="req_123")
    return handler


def _make_mock_project(project_id: str = "proj_1", root_path: str = "/tmp/test_project"):
    """Create a mock ProjectContext."""
    project = MagicMock()
    project.project_id = project_id
    project.root_path = root_path
    project.project_name = "Test Project"
    return project


# ===========================================================================
# 0. Workflow command compatibility aliases
# ===========================================================================


class TestWorkflowCommandAliases(unittest.TestCase):
    """Public long-form aliases must reach the canonical Workflow actions."""

    def test_long_aliases_delegate_to_canonical_actions(self):
        project = _make_mock_project()
        cases = (
            (
                "/workflow\t实现登录",
                "start_workflow",
                ("msg_1", "chat_1", "实现登录", project),
            ),
            (
                "/workflow_status",
                "show_workflow_status",
                ("msg_1", "chat_1", project),
            ),
            (
                "/workflow_help",
                "show_workflow_help",
                ("msg_1",),
            ),
            (
                "/stop_workflow",
                "stop_workflow",
                ("msg_1", "chat_1", project),
            ),
        )

        for text, target_name, expected_args in cases:
            with self.subTest(text=text):
                handler = _make_workflow_handler()
                handler.start_workflow = MagicMock()
                handler.show_workflow_status = MagicMock()
                handler.show_workflow_help = MagicMock()
                handler.stop_workflow = MagicMock()
                handler._reply_workflow_error = MagicMock()

                handler.handle_workflow_command(
                    "msg_1",
                    "chat_1",
                    text,
                    project,
                )

                getattr(handler, target_name).assert_called_once_with(
                    *expected_args,
                )
                handler._reply_workflow_error.assert_not_called()


def test_workflow_help_card_describes_confirmed_agent_pool_contract():
    """完整帮助卡不得跳过 Workflow 唯一的用户确认门。"""
    handler = _make_workflow_handler()
    handler.reply_card = MagicMock()

    handler.show_workflow_help("msg_1")

    card = handler.reply_card.call_args.args[1]
    rendered = str(card)
    assert "/workflow_help" in rendered
    assert "1–8" in rendered
    assert "Agent Pool" in rendered
    assert "使用此池开始编排" in rendered
    assert "确认后自动生成、验证并执行" in rendered
    assert "自动编排** · 使用推荐可用工具和后端默认模型" not in rendered


# ===========================================================================
# 1. TestAGENTSMDScopeStatement
# ===========================================================================

class TestAGENTSMDScopeStatement(unittest.TestCase):
    """Verify AGENTS.md contains the correct scope statement for Workflow.

    These tests ensure the product contract is documented and visible to
    all developers. If these tests fail, the documentation has drifted
    from the implementation contract.
    """

    AGENTS_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "AGENTS.md"
    )

    def _read_agents_md(self) -> str:
        """Read AGENTS.md content."""
        with open(self.AGENTS_PATH, "r", encoding="utf-8") as f:
            return f.read()

    def test_agents_md_includes_workflow_in_scope(self):
        """Verify Workflow is documented as a topic/root-thread strategy."""
        content = self._read_agents_md()
        workflow_scope_lines = [
            line
            for line in content.splitlines()
            if "Workflow" in line and "作用于飞书话题/根线程的引擎策略" in line
        ]

        self.assertTrue(
            workflow_scope_lines,
            "AGENTS.md must scope Workflow execution to a Feishu topic/root thread",
        )

    def test_agents_md_states_no_state_replacement(self):
        """Verify AGENTS.md explicitly forbids replacing chat+project state.

        This is the core state isolation contract. Topic engines must
        never modify the chat's persistent programming mode (coco, claude, etc.).
        If this fails, the documentation no longer reflects the isolation guarantee.
        """
        content = self._read_agents_md()
        self.assertIn("不得替换聊天+项目编程状态", content,
            "AGENTS.md must explicitly state that topic engines must not replace chat+project programming state")


# ===========================================================================
# 2. TestWorkflowModeStateIsolation
# ===========================================================================

class TestWorkflowModeStateIsolation(unittest.TestCase):
    """Verify that Workflow mode is isolated from chat programming mode.

    The key invariant: entering workflow mode via /wf must never change
    the chat's persistent programming mode (e.g., coco, claude, smart).
    Workflow state is stored per-topic/root-thread, not per-chat.
    """

    def setUp(self):
        self.mode_manager = ModeManager()
        self.chat_id = "chat_123"
        self.project_id = "proj_1"
        # Start in COCO programming mode
        self.mode_manager.enter_coco_mode(self.chat_id)

    def test_workflow_does_not_change_chat_mode(self):
        """Verify /wf command does NOT change the chat's programming mode.

        This is the most critical isolation invariant. If workflow changed
        the chat mode, users would lose their programming context when
        running a workflow, violating the product contract.
        """
        # Verify we start in COCO mode
        self.assertEqual(
            self.mode_manager.get_mode(self.chat_id),
            InteractionMode.COCO,
            "Test setup: chat should be in COCO mode initially"
        )

        # Simulate entering workflow mode (via _ensure_topic_engine_context)
        # The workflow handler should NOT call mode_manager.set_mode()
        # for the chat-level mode
        handler = _make_workflow_handler()
        handler.ctx.mode_manager = self.mode_manager

        # Mock thread manager to simulate topic binding
        mock_thread_ctx = MagicMock()
        mock_thread_ctx.mode = "workflow"
        mock_thread_ctx.thread_root_id = "thread_456"
        mock_thread_ctx.chat_id = self.chat_id
        mock_thread_ctx.project_id = self.project_id
        handler.ctx.thread_manager.get.return_value = None
        handler.ctx.thread_manager.bind_engine.return_value = mock_thread_ctx

        project = _make_mock_project(self.project_id)

        # Call _ensure_topic_engine_context (this is what start_workflow calls)
        with patch("src.thread.get_current_thread_id", return_value=None):
            with patch("src.thread.set_current_thread_id"):
                handler._ensure_topic_engine_context(
                    mode="workflow",
                    message_id="msg_1",
                    chat_id=self.chat_id,
                    project=project,
                )

        # The chat mode must still be COCO - workflow must NOT change it
        self.assertEqual(
            self.mode_manager.get_mode(self.chat_id),
            InteractionMode.COCO,
            "Workflow must NOT change the chat's persistent programming mode"
        )
        # Verify thread_manager.bind_engine was called (topic-level binding)
        handler.ctx.thread_manager.bind_engine.assert_called_once()

    def test_workflow_uses_topic_context(self):
        """Verify workflow state is stored per-topic, not per-chat.

        Workflow uses thread_manager.bind_engine() to bind to the topic
        root, not mode_manager.set_mode() which would affect the whole chat.
        """
        handler = _make_workflow_handler()
        project = _make_mock_project()

        mock_thread_ctx = MagicMock()
        mock_thread_ctx.mode = "workflow"
        handler.ctx.thread_manager.get.return_value = None
        handler.ctx.thread_manager.bind_engine.return_value = mock_thread_ctx

        with patch("src.thread.get_current_thread_id", return_value=None):
            with patch("src.thread.set_current_thread_id"):
                thread_root = handler._ensure_topic_engine_context(
                    mode="workflow",
                    message_id="msg_1",
                    chat_id=self.chat_id,
                    project=project,
                )

        # Verify bind_engine was called with topic-level parameters
        handler.ctx.thread_manager.bind_engine.assert_called_once_with(
            thread_root_id="msg_1",  # message becomes root when no existing thread
            chat_id=self.chat_id,
            project_id=self.project_id,
            mode="workflow",
        )
        self.assertEqual(thread_root, "msg_1")

    def test_concurrent_workflows_in_different_topics(self):
        """Verify two different topics can have independent workflow states.

        Each topic (root thread) has its own workflow engine state.
        This is what allows multiple workflows to run simultaneously
        in different threads of the same chat.
        """
        handler = _make_workflow_handler()
        project = _make_mock_project()

        # Simulate two different thread roots in the same chat
        thread_root_1 = "thread_001"
        thread_root_2 = "thread_002"

        # Track bind_engine calls
        bind_calls = []
        def capture_bind(**kwargs):
            bind_calls.append(kwargs)
            ctx = MagicMock()
            ctx.thread_root_id = kwargs["thread_root_id"]
            ctx.mode = kwargs["mode"]
            return ctx

        handler.ctx.thread_manager.get.return_value = None
        handler.ctx.thread_manager.bind_engine.side_effect = capture_bind

        # First topic
        with patch("src.thread.get_current_thread_id", return_value=thread_root_1):
            with patch("src.thread.set_current_thread_id"):
                handler._ensure_topic_engine_context(
                    mode="workflow",
                    message_id="msg_1",
                    chat_id=self.chat_id,
                    project=project,
                )

        # Second topic
        with patch("src.thread.get_current_thread_id", return_value=thread_root_2):
            with patch("src.thread.set_current_thread_id"):
                handler._ensure_topic_engine_context(
                    mode="workflow",
                    message_id="msg_2",
                    chat_id=self.chat_id,
                    project=project,
                )

        # Verify two separate bindings with different thread roots
        self.assertEqual(len(bind_calls), 2)
        self.assertEqual(bind_calls[0]["thread_root_id"], thread_root_1)
        self.assertEqual(bind_calls[1]["thread_root_id"], thread_root_2)
        self.assertEqual(bind_calls[0]["chat_id"], bind_calls[1]["chat_id"])  # same chat
        self.assertEqual(bind_calls[0]["mode"], bind_calls[1]["mode"])  # both workflow

    def test_exit_returns_to_previous_chat_mode(self):
        """Verify /exit from workflow returns to previous chat mode, not default.

        When a user types /exit in a topic with an active workflow, the
        chat should return to its previous programming mode (e.g., COCO),
        not jump to some default like SMART. This preserves user context.
        """
        # Set up: chat is in COCO mode
        self.mode_manager.enter_coco_mode(self.chat_id)
        self.assertEqual(self.mode_manager.get_mode(self.chat_id), InteractionMode.COCO)

        # Simulate workflow being active in a topic
        handler = _make_workflow_handler()
        handler.ctx.mode_manager = self.mode_manager

        # The exit command should be handled by the system handler's
        # exit_current_mode, which for topic engines should clear the
        # topic binding but NOT change the chat mode
        from src.feishu.handlers.system import SystemHandler
        system_handler = SystemHandler(handler.ctx)
        system_handler.im_client = MagicMock()
        system_handler.reply_text = MagicMock()

        # Mock thread context for the topic
        mock_thread_ctx = MagicMock()
        mock_thread_ctx.mode = "workflow"
        mock_thread_ctx.thread_root_id = "thread_456"
        mock_thread_ctx.project_id = self.project_id

        thread_manager = MagicMock()
        thread_manager.get.return_value = mock_thread_ctx
        thread_manager.remove.return_value = mock_thread_ctx
        with (
            patch("src.thread.get_current_thread_id", return_value="thread_456"),
            patch("src.thread.get_thread_manager", return_value=thread_manager),
            patch("src.thread.set_current_thread_id") as clear_thread,
            patch.object(self.mode_manager, "exit_to_smart") as exit_to_smart,
        ):
            system_handler.exit_current_mode("msg_exit", self.chat_id, project=None)

        thread_manager.remove.assert_called_once_with("thread_456")
        clear_thread.assert_called_once_with(None)
        exit_to_smart.assert_not_called()
        system_handler.reply_text.assert_called_once()

        # The key assertion: chat mode remains COCO
        self.assertEqual(
            self.mode_manager.get_mode(self.chat_id),
            InteractionMode.COCO,
            "Exiting workflow must return to the previous chat mode, not change it"
        )


# ===========================================================================
# 3. TestAutoEnterWorkflowMode
# ===========================================================================




# ===========================================================================
# 5. TestModeIndependence
# ===========================================================================

class TestModeIndependence(unittest.TestCase):
    """Verify workflow and chat programming mode are fully independent.

    A chat can be in 'coco' programming mode while simultaneously having
    an active workflow in one of its topics. These states must not interfere.
    """

    def setUp(self):
        self.mode_manager = ModeManager()
        self.chat_id = "chat_123"
        self.project_id = "proj_1"

    def test_workflow_and_normal_mode_coexist(self):
        """Verify chat can be in 'coco' mode while topic has active workflow.

        This is the core independence test. The chat-level programming mode
        and topic-level workflow mode are separate dimensions that must
        not interfere with each other.
        """
        # Set chat to COCO programming mode
        self.mode_manager.enter_coco_mode(self.chat_id)
        self.assertEqual(self.mode_manager.get_mode(self.chat_id), InteractionMode.COCO)
        self.assertTrue(self.mode_manager.is_programming_mode(self.chat_id))

        # Simulate binding a workflow to a topic in the same chat
        handler = _make_workflow_handler()
        handler.ctx.mode_manager = self.mode_manager
        project = _make_mock_project(self.project_id)

        mock_thread_ctx = MagicMock()
        mock_thread_ctx.mode = "workflow"
        mock_thread_ctx.thread_root_id = "thread_456"
        handler.ctx.thread_manager.get.return_value = None
        handler.ctx.thread_manager.bind_engine.return_value = mock_thread_ctx

        with patch("src.thread.get_current_thread_id", return_value=None):
            with patch("src.thread.set_current_thread_id"):
                handler._ensure_topic_engine_context(
                    mode="workflow",
                    message_id="msg_1",
                    chat_id=self.chat_id,
                    project=project,
                )

        # Chat mode is still COCO - workflow binding didn't change it
        self.assertEqual(
            self.mode_manager.get_mode(self.chat_id),
            InteractionMode.COCO,
            "Chat programming mode must be independent from topic workflow mode"
        )
        self.assertTrue(
            self.mode_manager.is_programming_mode(self.chat_id),
            "Chat should still be considered in programming mode"
        )

    def test_workflow_completion_keeps_chat_mode(self):
        """Verify after workflow completes, chat programming mode is unchanged.

        When a workflow finishes (success or failure), the chat's programming
        mode must remain exactly as it was before the workflow started.
        """
        # Set chat to CLAUDE mode
        self.mode_manager.enter_claude_mode(self.chat_id)
        self.assertEqual(self.mode_manager.get_mode(self.chat_id), InteractionMode.CLAUDE)

        # Simulate workflow completion
        handler = _make_workflow_handler()
        handler.ctx.mode_manager = self.mode_manager

        # Mock engine completion
        mock_engine = MagicMock()
        mock_engine.is_running = False
        handler.ctx.workflow_engine_manager.get.return_value = mock_engine

        # Call stop_workflow (which would be called on completion or cancellation)
        with patch("src.thread.get_current_sender_id", return_value="user_1"):
            handler.stop_workflow(
                message_id="msg_1",
                chat_id=self.chat_id,
                project=None,
            )

        # Chat mode is still CLAUDE
        self.assertEqual(
            self.mode_manager.get_mode(self.chat_id),
            InteractionMode.CLAUDE,
            "Chat mode must survive workflow completion"
        )

    def test_workflow_cancellation_keeps_chat_mode(self):
        """Verify after cancelling a workflow, chat programming mode is unchanged.

        Same as completion test but for explicit cancellation via /stop_wf.
        """
        # Set chat to GEMINI mode
        self.mode_manager.enter_gemini_mode(self.chat_id)
        self.assertEqual(self.mode_manager.get_mode(self.chat_id), InteractionMode.GEMINI)

        # Simulate workflow cancellation
        handler = _make_workflow_handler()
        handler.ctx.mode_manager = self.mode_manager

        mock_engine = MagicMock()
        mock_engine.is_running = True
        mock_engine.project.initiator_user_id = "user_1"
        handler.ctx.workflow_engine_manager.get.return_value = mock_engine

        with patch("src.thread.get_current_sender_id", return_value="user_1"):
            handler.stop_workflow(
                message_id="msg_1",
                chat_id=self.chat_id,
                project=None,
            )

        # Verify engine.stop() was called
        mock_engine.stop.assert_called_once()

        # Chat mode is still GEMINI
        self.assertEqual(
            self.mode_manager.get_mode(self.chat_id),
            InteractionMode.GEMINI,
            "Chat mode must survive workflow cancellation"
        )


# ===========================================================================
# 6. TestWorkflowEngineManagerScoping
# ===========================================================================

class TestWorkflowEngineManagerScoping(unittest.TestCase):
    """Verify WorkflowEngineManager keys engines by chat_id AND root_path.

    This allows different projects in the same chat to have independent
    workflow engines, and ensures proper cleanup on completion.
    """

    def test_engine_manager_keys_by_chat_and_root_path(self):
        """Verify get_or_create uses both chat_id and root_path as the key.

        This is critical for multi-project chats - each project needs its
        own workflow engine so workflows don't interfere across projects.
        """
        from src.workflow_engine.manager import WorkflowEngineManager

        manager = WorkflowEngineManager()
        chat_id = "chat_123"
        root_path_1 = "/tmp/project_a"
        root_path_2 = "/tmp/project_b"

        # Create engine for project A
        engine_a = manager.get_or_create(
            chat_id=chat_id,
            root_path=root_path_1,
            engine_name="Workflow",
        )

        # Create engine for project B (same chat, different root)
        engine_b = manager.get_or_create(
            chat_id=chat_id,
            root_path=root_path_2,
            engine_name="Workflow",
        )

        # They should be different instances
        self.assertIsNot(engine_a, engine_b,
            "Different root_paths in same chat must get different engine instances")

        # Verify the internal key structure
        key_a = f"{chat_id}:{root_path_1}"
        key_b = f"{chat_id}:{root_path_2}"
        self.assertIn(key_a, manager._engines)
        self.assertIn(key_b, manager._engines)
        self.assertIs(manager._engines[key_a], engine_a)
        self.assertIs(manager._engines[key_b], engine_b)

        # Verify chat index contains both keys
        self.assertIn(key_a, manager._chat_keys[chat_id])
        self.assertIn(key_b, manager._chat_keys[chat_id])

        # Verify get() returns the right engine
        self.assertIs(manager.get(chat_id, root_path_1), engine_a)
        self.assertIs(manager.get(chat_id, root_path_2), engine_b)

    def test_engine_cleanup_on_completion(self):
        """Verify engine state is properly cleaned up after completion/cancellation.

        The manager's remove() method should properly clean up the engine
        instance and remove it from both _engines and _chat_keys.
        """
        from src.workflow_engine.manager import WorkflowEngineManager

        manager = WorkflowEngineManager()
        chat_id = "chat_123"
        root_path = "/tmp/project"

        # Create an engine
        engine = manager.get_or_create(chat_id, root_path, engine_name="Workflow")
        engine.cleanup = MagicMock()

        key = f"{chat_id}:{root_path}"
        self.assertIn(key, manager._engines)
        self.assertIn(chat_id, manager._chat_keys)

        # Remove it (simulating completion cleanup)
        manager.remove(chat_id, root_path)

        # Verify cleanup was called
        engine.cleanup.assert_called_once()

        # Verify it's removed from both data structures
        self.assertNotIn(key, manager._engines)
        self.assertNotIn(chat_id, manager._chat_keys)

        # Verify get() returns None after removal
        self.assertIsNone(manager.get(chat_id, root_path))


# ===========================================================================
# 7. TestWorkflowTopicRoutingAC22
# ===========================================================================

class TestWorkflowTopicRoutingAC22(unittest.TestCase):
    """AC22: Workflow topic 自由文本路由到 WorkflowHandler，不经过 intent recognition。

    Validates the dispatch logic in ``FeishuWSClient._dispatch_message_logic``:
    When auto_enter_mode='workflow', command_match=None, and project is present,
    free-text messages are routed directly to WorkflowHandler.handle_message()
    WITHOUT going through MessageDispatcher.process_with_intent().
    """

    def setUp(self):
        self.chat_id = "chat_123"
        self.message_id = "msg_123"
        self.text = "请继续分析刚才的结果"
        self.project = _make_mock_project("test_proj")

    def _make_client_with_mocks(self):
        """Create a FeishuWSClient instance with all necessary mocks."""
        from src.feishu.ws_client import FeishuWSClient

        client = FeishuWSClient.__new__(FeishuWSClient)
        client._handler_ctx = MagicMock()
        client._handler_ctx.handlers = {
            "coco": MagicMock(),
            "system": MagicMock(),
            "workflow": MagicMock(),
        }
        client._current_trust_can_dispatch = MagicMock(return_value=True)
        client._message_dispatcher = MagicMock()
        client._intent_recognizer = MagicMock()
        client._intent_recognizer.looks_like_shell = MagicMock(return_value=False)
        client._project_manager = MagicMock()
        client._project_manager.find_by_bound_chat_id = MagicMock(return_value=None)
        client.settings = MagicMock()
        client.settings.default_acp_tool = "coco"

        return client

    def test_dispatch_message_logic_workflow_topic_routing(self):
        """AC22: _dispatch_message_logic 对 Workflow topic 自由文本路由到 WorkflowHandler。

        验证：
        1. auto_enter_mode='workflow', command_match=None, project present
        2. workflow handler 的 handle_message 被调用
        3. MessageDispatcher.process_with_intent 不被调用
        """
        client = self._make_client_with_mocks()

        # Call the actual dispatch logic
        client._dispatch_message_logic(
            message_id=self.message_id,
            chat_id=self.chat_id,
            text=self.text,
            project=self.project,
            auto_enter_mode="workflow",
            command_match=None,  # No command match - free text
        )

        # Verify workflow handler was called with correct arguments
        handlers = client._handler_ctx.handlers
        handlers["workflow"].handle_message.assert_called_once_with(
            self.message_id, self.chat_id, self.text, self.project
        )

        # Verify intent recognition was NOT called
        client._message_dispatcher.process_with_intent.assert_not_called()

        # Verify processing reaction was added
        handlers["coco"].add_reaction.assert_called_once()

    def test_workflow_topic_routing_no_command_match(self):
        """AC22: Workflow topic 路由仅在无命令匹配时生效。

        When command_match is NOT None, the message should go through
        MessageDispatcher.process_with_intent instead of being routed directly.
        """
        client = self._make_client_with_mocks()
        mock_command_match = MagicMock()
        mock_command_match.command = "/wf_status"
        mock_command_match.args = ""

        client._dispatch_message_logic(
            message_id=self.message_id,
            chat_id=self.chat_id,
            text=self.text,
            project=self.project,
            auto_enter_mode="workflow",
            command_match=mock_command_match,  # Has command match
        )

        # Should go to intent recognition, NOT workflow handler
        client._handler_ctx.handlers["workflow"].handle_message.assert_not_called()
        client._message_dispatcher.process_with_intent.assert_called_once()

    def test_workflow_topic_routing_no_project(self):
        """Workflow topic fails closed when its bound project is unavailable."""
        client = self._make_client_with_mocks()

        client._dispatch_message_logic(
            message_id=self.message_id,
            chat_id=self.chat_id,
            text=self.text,
            project=None,  # No project
            auto_enter_mode="workflow",
            command_match=None,
        )

        handlers = client._handler_ctx.handlers
        handlers["workflow"].handle_message.assert_not_called()
        handlers["coco"].reply_text.assert_called_once()
        client._message_dispatcher.process_with_intent.assert_not_called()

    def test_workflow_topic_routing_wrong_mode(self):
        """AC22: Workflow topic 路由仅在 auto_enter_mode='workflow' 时生效。

        When auto_enter_mode is not 'workflow' (e.g., 'coco'), the message
        should NOT be routed to workflow handler.
        """
        client = self._make_client_with_mocks()

        client._dispatch_message_logic(
            message_id=self.message_id,
            chat_id=self.chat_id,
            text=self.text,
            project=self.project,
            auto_enter_mode="coco",  # Wrong mode
            command_match=None,
        )

        # Should NOT go to workflow handler
        client._handler_ctx.handlers["workflow"].handle_message.assert_not_called()
        client._message_dispatcher.process_with_intent.assert_called_once()


if __name__ == "__main__":
    unittest.main()
