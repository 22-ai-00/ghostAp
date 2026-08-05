"""Tests for slock dispatcher routing: command scoping + message routing.

Covers:
- AC2: unmanaged chat /role /task /team passthrough; /slock captured
- AC3: managed chat normal message routes through explicit smart route execution
- AC4: @AgentName precise routing
- AC10: unmanaged chat normal message passthrough
"""

from __future__ import annotations

from concurrent.futures import Future
from unittest.mock import ANY, MagicMock, patch

import pytest

from src.mode import InteractionMode
from src.slock_engine.slash_commands import NEEDS_ACTIVATION, is_slock_command


def _sync_submit(fn, *args, **kwargs):
    """Helper that executes executor.submit synchronously for deterministic tests."""
    future = Future()
    try:
        result = fn(*args, **kwargs)
        future.set_result(result)
    except Exception as exc:
        future.set_exception(exc)
    return future


# ============================================================
# AC2: Command Scoping
# ============================================================


class TestCommandScopingUnmanagedChat:
    """In unmanaged chats, only /slock and /new-team are captured."""

    def _make_manager(self, is_managed: bool):
        manager = MagicMock()
        manager.is_managed_chat.return_value = is_managed
        return manager

    @pytest.mark.parametrize("text", ["/slock", "/slock status", "/new-team MyTeam"])
    def test_global_commands_always_captured(self, text):
        """AC2: /slock and /new-team are captured even in unmanaged chats."""
        manager = self._make_manager(is_managed=False)
        result = is_slock_command(text, chat_id="unmanaged_chat", manager=manager)
        assert result  # SlockCommandResult.__bool__ returns is_command

    @pytest.mark.parametrize("text", [
        "/task list",
        "/task assign fix-bug Coder",
        "/new-role Writer",
    ])
    def test_chat_scoped_commands_need_activation_in_unmanaged(self, text):
        """AC2: Chat-scoped commands return NEEDS_ACTIVATION in unmanaged chats."""
        manager = self._make_manager(is_managed=False)
        assert is_slock_command(text, chat_id="unmanaged_chat", manager=manager) == NEEDS_ACTIVATION

    @pytest.mark.parametrize("text", [
        "/role list",
        "/role remove Coder",
        "/team list",
        "/team status Alpha",
        "/team dissolve Alpha",
    ])
    def test_team_role_commands_globally_captured(self, text):
        """AC2: /team and /role are global — captured even in unmanaged chats."""
        manager = self._make_manager(is_managed=False)
        result = is_slock_command(text, chat_id="unmanaged_chat", manager=manager)
        assert result  # Always captured when manager exists

    @pytest.mark.parametrize("text", [
        "/role list",
        "/task assign fix-bug Coder",
        "/team list",
        "/new-role Writer",
    ])
    def test_team_commands_captured_in_managed(self, text):
        """AC2: Team commands ARE captured in managed chats."""
        manager = self._make_manager(is_managed=True)
        result = is_slock_command(text, chat_id="managed_chat", manager=manager)
        assert result  # SlockCommandResult.__bool__ returns is_command


class TestSlockDoesNotStealProgrammingMode:
    """Slock passive routing must not preempt active programming sessions."""

    def _make_programming_client(self, *, managed: bool, should_activate: bool = False):
        from src.feishu.dispatcher import MessageDispatcher

        client = MagicMock()
        client.settings = MagicMock()
        client.settings.slock_passive_mode = True
        client._get_effective_mode.return_value = (InteractionMode.COCO, True)
        client._is_deep_command.return_value = False
        client._is_spec_command.return_value = False
        client._is_slock_command.return_value = False
        client._is_workflow_command.return_value = False
        client._is_slock_managed_chat.return_value = managed
        client._should_auto_activate_slock.return_value = should_activate
        client._is_exit_command.return_value = False
        client._is_interceptable_command_match.return_value = False
        handler = MagicMock()
        client._get_mode_handler.return_value = handler
        return client, handler, MessageDispatcher(client)

    def test_managed_slock_chat_in_programming_mode_routes_to_programming_handler(self):
        from src.feishu.dispatcher import FeishuRequestContext

        client, handler, dispatcher = self._make_programming_client(managed=True)

        dispatcher.process_request(
            FeishuRequestContext(
                message_id="msg_prog_managed",
                chat_id="project_group",
                text="继续修复这个问题",
            )
        )

        client._handle_slock_message.assert_not_called()
        client._auto_activate_slock.assert_not_called()
        handler.handle_message.assert_called_once_with(
            "msg_prog_managed",
            "project_group",
            "继续修复这个问题",
            None,
        )

    def test_task_like_text_in_programming_mode_does_not_auto_activate_slock(self):
        from src.feishu.dispatcher import FeishuRequestContext

        client, handler, dispatcher = self._make_programming_client(
            managed=False,
            should_activate=True,
        )

        dispatcher.process_request(
            FeishuRequestContext(
                message_id="msg_prog_task",
                chat_id="project_group",
                text="帮我实现单元测试",
            )
        )

        client._auto_activate_slock.assert_not_called()
        client._handle_slock_message.assert_not_called()
        handler.handle_message.assert_called_once_with(
            "msg_prog_task",
            "project_group",
            "帮我实现单元测试",
            None,
        )


class TestSlockDoesNotStealProjectContext:
    """Passive Slock auto-activation must not preempt project-scoped SMART routing."""

    def test_task_like_project_context_uses_intent_recognition_not_auto_activation(self):
        from src.agent.intent_recognizer import IntentResult, IntentType, TaskStep
        from src.feishu.dispatcher import FeishuRequestContext, MessageDispatcher
        from src.project import ProjectContext

        client = MagicMock()
        client.settings = MagicMock()
        client.settings.slock_passive_mode = True
        client._get_effective_mode.return_value = (InteractionMode.SMART, False)
        client._is_deep_command.return_value = False
        client._is_spec_command.return_value = False
        client._is_slock_command.return_value = False
        client._is_workflow_command.return_value = False
        client._is_slock_managed_chat.return_value = False
        client._should_auto_activate_slock.return_value = True
        client._is_exit_command.return_value = False
        client._is_interceptable_command_match.return_value = False
        client._intent_recognizer.recognize.return_value = IntentResult(
            confidence=0.9,
            tasks=[
                TaskStep(
                    intent=IntentType.ENTER_COCO,
                    data={},
                    description="Enter programming",
                )
            ],
        )
        dispatcher = MessageDispatcher(client)
        dispatcher.execute_single_task = MagicMock()

        project = ProjectContext("proj_1", "GhostAP", "/tmp")
        dispatcher.process_request(
            FeishuRequestContext(
                message_id="msg_project_task",
                chat_id="project_group",
                text="帮我实现单元测试",
                project=project,
            )
        )

        client._auto_activate_slock.assert_not_called()
        client._handle_slock_message.assert_not_called()
        client._intent_recognizer.recognize.assert_called_once()
        dispatcher.execute_single_task.assert_called_once()


# ============================================================
# AC3: Managed chat message routing to engine
# ============================================================


class TestManagedChatMessageRouting:
    """In slock-active chats, non-command messages route to engine."""

    def _make_handler(self):
        """Build a SlockHandler with mocked dependencies."""
        from src.feishu.handlers.slock import SlockHandler

        ctx = MagicMock()
        ctx.settings.slock_nli_confidence_threshold = 0.7
        ctx.settings.slock_nli_timeout = 5.0
        handler = SlockHandler(ctx)
        handler.reply_text = MagicMock()
        handler.send_card_to_chat = MagicMock()
        return handler

    @pytest.mark.slow
    def test_normal_message_calls_engine_execute(self):
        """AC3: Normal text in managed chat routes to the selected agent."""
        handler = self._make_handler()
        handler.send_card_to_chat = MagicMock(return_value="placeholder-msg-001")
        handler.update_card = MagicMock(return_value=True)

        engine = MagicMock()
        engine.channel = MagicMock()
        engine.channel.channel_id = "chat_123"
        engine._mouthpiece = MagicMock()
        engine._mouthpiece.format_card.return_value = {"header": {}, "elements": []}

        mock_agent = MagicMock()
        mock_agent.agent_id = "agent-default"
        mock_agent.agent_type = "coco"
        mock_agent.model_name = ""
        engine.registry.list_agents.return_value = [mock_agent]
        engine.router.route_message.return_value = mock_agent
        engine._execute_agent.return_value = "Agent response"

        executor = MagicMock()
        executor.submit.side_effect = _sync_submit
        engine._get_executor = MagicMock(return_value=executor)

        manager = MagicMock()
        manager.get_activated_engine.return_value = engine
        handler._get_engine_manager = MagicMock(return_value=manager)

        handler.handle_message("msg_1", "chat_123", "hello team", None)

        engine.router.route_message.assert_called_once_with("hello team", [mock_agent])
        engine._execute_agent.assert_called_once_with(mock_agent, "hello team", ANY)

    def test_no_engine_silently_returns(self):
        """If no engine active, handle_message does nothing."""
        handler = self._make_handler()
        manager = MagicMock()
        manager.get_activated_engine.return_value = None
        handler._get_engine_manager = MagicMock(return_value=manager)

        handler.handle_message("msg_1", "chat_123", "hello", None)
        # No crash, no reply


# ============================================================
# AC4: @AgentName precise routing
# ============================================================


class TestAtMentionRouting:
    """@AgentName routes precisely to that agent."""

    def test_at_mention_routes_to_named_agent(self):
        """AC4: @Coder routes to the Coder agent specifically."""
        from src.feishu.handlers.slock import SlockHandler

        ctx = MagicMock()
        ctx.settings.slock_nli_confidence_threshold = 0.7
        ctx.settings.slock_nli_timeout = 5.0
        handler = SlockHandler(ctx)
        handler.reply_text = MagicMock()
        handler.send_card_to_chat = MagicMock(return_value="placeholder-msg-001")
        handler.update_card = MagicMock(return_value=True)

        engine = MagicMock()
        engine.channel = MagicMock()
        engine.channel.channel_id = "chat_123"

        target_agent = MagicMock()
        target_agent.name = "Coder"
        target_agent.agent_type = "codex"
        engine.registry.find_by_name.return_value = target_agent
        engine._execute_agent.return_value = "Code fix applied"
        engine._mouthpiece = MagicMock()
        engine._mouthpiece.format_card.return_value = {"header": {}, "elements": []}

        executor = MagicMock()
        executor.submit.side_effect = _sync_submit
        engine._get_executor = MagicMock(return_value=executor)

        manager = MagicMock()
        manager.get_activated_engine.return_value = engine
        handler._get_engine_manager = MagicMock(return_value=manager)

        handler.handle_message("msg_1", "chat_123", "@Coder please fix the bug", None)

        engine.registry.find_by_name.assert_called_with("Coder", channel_id="chat_123")
        engine._execute_agent.assert_called_once_with(target_agent, "@Coder please fix the bug", ANY)

    @pytest.mark.slow
    def test_at_mention_unknown_agent_falls_to_smart_route(self):
        """If @UnknownAgent doesn't match, fall through to explicit smart routing."""
        from src.feishu.handlers.slock import SlockHandler

        ctx = MagicMock()
        ctx.settings.slock_nli_confidence_threshold = 0.7
        ctx.settings.slock_nli_timeout = 5.0
        handler = SlockHandler(ctx)
        handler.reply_text = MagicMock()
        handler.send_card_to_chat = MagicMock(return_value="placeholder-msg-001")
        handler.update_card = MagicMock(return_value=True)

        engine = MagicMock()
        engine.channel = MagicMock()
        engine.channel.channel_id = "chat_123"
        engine.registry.find_by_name.return_value = None  # Not found
        engine.execute.return_value = "Handled by default agent"
        engine._mouthpiece = MagicMock()
        engine._mouthpiece.format_card.return_value = {"header": {}, "elements": []}

        mock_agent = MagicMock()
        mock_agent.agent_id = "agent-default"
        mock_agent.agent_type = "coco"
        mock_agent.model_name = ""
        engine.registry.list_agents.return_value = [mock_agent]
        engine.router.route_message.return_value = mock_agent
        engine._execute_agent.return_value = "Handled by default agent"

        executor = MagicMock()
        executor.submit.side_effect = _sync_submit
        engine._get_executor = MagicMock(return_value=executor)

        manager = MagicMock()
        manager.get_activated_engine.return_value = engine
        handler._get_engine_manager = MagicMock(return_value=manager)

        handler.handle_message("msg_1", "chat_123", "@UnknownBot do something", None)

        engine.router.route_message.assert_called_once_with("@UnknownBot do something", [mock_agent])
        engine._execute_agent.assert_called_once_with(mock_agent, "@UnknownBot do something", ANY)


# ============================================================
# AC10: Unmanaged chat passthrough
# ============================================================


class TestUnmanagedChatPassthrough:
    """Normal messages in non-slock chats must not be intercepted."""

    def test_is_slock_command_false_for_normal_text(self):
        """AC10: Normal text is never a slock command."""
        assert not is_slock_command("hello world")
        assert not is_slock_command("let's fix this bug")
        assert not is_slock_command("")

    def test_no_manager_means_no_capture(self):
        """Without manager context, team commands are not captured."""
        assert not is_slock_command("/role list")
        assert not is_slock_command("/task list")
        assert not is_slock_command("/team status")


# ============================================================
# AC-14: Dispatcher routing chain priority
# ============================================================


class TestRoutingChainPriority:
    """AC-14: SlockModeHandler sits after SpecModeHandler and before ExitHandler."""

    def test_slock_checked_after_spec_before_exit(self):
        """AC-14: In dispatcher source, slock check follows spec and precedes exit.

        We verify by importing the dispatcher and inspecting process_with_intent
        source line order — spec check line < slock check line < exit check line.
        """
        import inspect

        from src.feishu.dispatcher import MessageDispatcher

        source = inspect.getsource(MessageDispatcher.process_with_intent)

        spec_pos = source.find("_is_spec_command")
        slock_pos = source.find("_is_slock_command")
        exit_pos = source.find("_is_exit_command")

        assert spec_pos != -1, "_is_spec_command not found in process_with_intent"
        assert slock_pos != -1, "_is_slock_command not found in process_with_intent"
        assert exit_pos != -1, "_is_exit_command not found in process_with_intent"

        assert spec_pos < slock_pos, (
            f"Spec ({spec_pos}) must appear before Slock ({slock_pos}) in routing chain"
        )
        assert slock_pos < exit_pos, (
            f"Slock ({slock_pos}) must appear before Exit ({exit_pos}) in routing chain"
        )

    def test_slock_command_does_not_fall_to_exit(self):
        """AC-14: When slock command matches, exit handler is never reached."""
        manager = MagicMock()
        manager.is_managed_chat.return_value = True

        # /slock status is a slock command, not an exit command
        assert is_slock_command("/slock status", chat_id="ch", manager=manager)

    def test_spec_takes_priority_over_slock_for_spec_command(self):
        """AC-14: /spec is not captured by slock — spec handler takes precedence."""
        manager = MagicMock()
        manager.is_managed_chat.return_value = True

        assert not is_slock_command("/spec", chat_id="ch", manager=manager)
        assert not is_slock_command("/spec start", chat_id="ch", manager=manager)


# ============================================================
# Auto-activate: no double-handle after auto-activate
# ============================================================



# ============================================================
# Fixture for dispatcher-level tests
# ============================================================


@pytest.fixture
def mock_client():
    """Create a mock client suitable for MessageDispatcher tests.

    Returns a MagicMock pre-configured so the dispatcher can run
    through process_with_intent without crashing on required calls.
    """
    from src.slock_engine.slash_commands import SlockCommandResult

    client = MagicMock()
    client.settings = MagicMock()
    client.settings.slock_passive_mode = True
    # Default: not in programming mode
    client._get_effective_mode.return_value = (MagicMock(), False)
    # Default: no deep/spec/slock commands
    client._is_deep_command.return_value = False
    client._is_spec_command.return_value = False
    client._is_slock_command.return_value = SlockCommandResult(is_command=False)
    # Default: not managed
    client._is_slock_managed_chat.return_value = False
    # Default: auto-activate disabled
    client._should_auto_activate_slock.return_value = False
    # Default: interceptable command — allows slash commands to terminate routing
    client._is_interceptable_command_match.return_value = True
    return client


# ============================================================
# AC-R3: Command penetration — slash commands bypass slock auto-activation
# ============================================================


class TestCommandPenetration:
    """AC-R3: Slash commands must bypass slock auto-activation."""

    def test_slash_coco_not_intercepted(self, mock_client):
        """Sending /coco in a passive-mode group should NOT trigger slock."""
        from src.feishu.dispatcher import FeishuRequestContext, MessageDispatcher

        mock_client.settings.slock_passive_mode = True
        mock_client._is_slock_managed_chat.return_value = False
        # Even if should_auto_activate would say yes, /coco is a command
        mock_client._should_auto_activate_slock.return_value = True

        dispatcher = MessageDispatcher(mock_client)
        ctx = FeishuRequestContext(
            message_id="msg1",
            chat_id="group1",
            text="/coco help me",
            chat_type="group",
        )
        dispatcher.process_request(ctx)
        # _auto_activate_slock should NOT have been called
        mock_client._auto_activate_slock.assert_not_called()
        # _should_auto_activate_slock should NOT have been called (short-circuited)
        mock_client._should_auto_activate_slock.assert_not_called()

    def test_slash_help_not_intercepted(self, mock_client):
        """Sending /help in a passive-mode group should NOT trigger slock."""
        from src.feishu.dispatcher import FeishuRequestContext, MessageDispatcher

        mock_client.settings.slock_passive_mode = True
        mock_client._is_slock_managed_chat.return_value = False
        mock_client._should_auto_activate_slock.return_value = True

        dispatcher = MessageDispatcher(mock_client)
        ctx = FeishuRequestContext(
            message_id="msg1",
            chat_id="group1",
            text="/help",
            chat_type="group",
        )
        dispatcher.process_request(ctx)
        mock_client._auto_activate_slock.assert_not_called()
        mock_client._should_auto_activate_slock.assert_not_called()


# ============================================================
# AC-R4: P2P isolation — P2P messages don't trigger auto-activation
# ============================================================


class TestP2PIsolation:
    """AC-R4: P2P single chat must not trigger slock auto-activation."""

    def test_p2p_task_message_no_auto_activate(self, mock_client):
        """Task-like message in P2P chat should NOT trigger slock."""
        from src.feishu.dispatcher import FeishuRequestContext, MessageDispatcher

        mock_client.settings.slock_passive_mode = True
        mock_client._is_slock_managed_chat.return_value = False
        # _should_auto_activate_slock returns False because chat_type != group
        mock_client._should_auto_activate_slock.return_value = False
        # Not in programming mode
        mock_client._get_effective_mode.return_value = (MagicMock(), False)
        # No command match for plain text — mock the pending image lock
        mock_client._pending_image_lock = MagicMock()
        mock_client._pending_image_only = set()

        dispatcher = MessageDispatcher(mock_client)
        ctx = FeishuRequestContext(
            message_id="msg1",
            chat_id="p2p_123",
            text="帮我写一个快速排序",
            chat_type="p2p",
        )
        dispatcher.process_request(ctx)
        mock_client._auto_activate_slock.assert_not_called()


# ============================================================
# AC-R7: SlockCommandResult __bool__ compatibility
# ============================================================


class TestSlockCommandResultType:
    """AC-R7: is_slock_command returns SlockCommandResult with proper __bool__."""

    def test_slock_command_returns_result_type(self):
        from src.slock_engine.slash_commands import SlockCommandResult, is_slock_command

        result = is_slock_command("/slock", chat_id="test_chat")
        assert isinstance(result, SlockCommandResult)
        assert bool(result) is True

    def test_non_slock_command_is_falsy(self):
        from src.slock_engine.slash_commands import SlockCommandResult, is_slock_command

        result = is_slock_command("hello world", chat_id="test_chat")
        assert isinstance(result, SlockCommandResult)
        assert bool(result) is False

    def test_needs_activation_constant(self):
        from src.slock_engine.slash_commands import (
            NEEDS_ACTIVATION,
            SlockCommandResult,
            is_slock_command,
        )

        # A chat-scoped subcommand in unmanaged chat should return NEEDS_ACTIVATION
        manager = MagicMock()
        manager.is_managed_chat.return_value = False
        result = is_slock_command("/task list", chat_id="unmanaged_chat", manager=manager)
        assert isinstance(result, SlockCommandResult)
        assert result == NEEDS_ACTIVATION
        assert bool(result) is False


# ============================================================
# AC-R1, AC-R8, AC-R10: Settings defaults and validation
# ============================================================


class TestSettingsValidation:
    """AC-R1, AC-R8, AC-R10: Settings defaults and validation."""

    def test_default_policy_is_allow_all(self):
        """AC-R1: Default policy should be allow_all (open-by-default for Slock groups)."""
        import os
        from unittest.mock import patch as _patch

        from src.config.settings import Settings

        env = os.environ.copy()
        env.pop("SLOCK_AUTO_ACTIVATE_DEFAULT_POLICY", None)
        with _patch.dict("os.environ", env, clear=True):
            s = Settings()
            assert s.slock_auto_activate_default_policy == "allow_all"

    def test_invalid_queue_timeout_rejected(self):
        """AC-R10: SLOCK_QUEUE_WAIT_TIMEOUT=-1 must raise ValidationError."""
        from unittest.mock import patch as _patch

        from pydantic import ValidationError

        from src.config.settings import Settings

        with _patch.dict("os.environ", {"SLOCK_QUEUE_WAIT_TIMEOUT": "-1"}):
            with pytest.raises(ValidationError):
                Settings()

    def test_queue_timeout_over_600_rejected(self):
        """slock_queue_wait_timeout > 600 must raise ValidationError."""
        from unittest.mock import patch as _patch

        from pydantic import ValidationError

        from src.config.settings import Settings

        with _patch.dict("os.environ", {"SLOCK_QUEUE_WAIT_TIMEOUT": "999"}):
            with pytest.raises(ValidationError):
                Settings()

    def test_deprecated_prefix_fallback(self):
        """AC-R8: SLOCK_TEAM_NAME_PREFIX is read with DeprecationWarning."""
        import warnings
        from unittest.mock import patch as _patch

        from src.config.settings import Settings

        with _patch.dict("os.environ", {
            "SLOCK_TEAM_NAME_PREFIX": "[OldPrefix]",
        }):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                Settings()
                # Should have emitted DeprecationWarning
                dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
                assert len(dep_warnings) >= 1
                assert "SLOCK_TEAM_NAME_PREFIX" in str(dep_warnings[0].message)


# ===========================================================================
# Task 17.1: Clarification card button click routing
# ===========================================================================


class TestClarificationCardButtonRouting:
    """Task 17.1: Test clarification card button click behavior.

    Simulates button clicks and verifies:
    - Click "是，这是任务": task enqueued or auto-activate triggered
    - Click "不是，只是聊天": card updated to "已忽略", no task created
    """

    def _make_handler(self):
        """Build a SlockHandler with mocked dependencies."""
        from src.feishu.handlers.slock import SlockHandler

        ctx = MagicMock()
        ctx.settings.slock_nli_confidence_threshold = 0.7
        ctx.settings.slock_nli_timeout = 5.0
        handler = SlockHandler(ctx)
        handler.reply_text = MagicMock()
        handler.send_card_to_chat = MagicMock(return_value="placeholder-msg-001")
        handler.update_card = MagicMock(return_value=True)
        handler.send_text_to_chat = MagicMock()
        return handler

    def test_clarify_confirm_with_existing_engine_calls_handle_message(self):
        """When engine exists, confirm button triggers handle_message to enqueue task."""
        from unittest.mock import PropertyMock, patch

        handler = self._make_handler()
        handler.handle_message = MagicMock()

        engine = MagicMock()
        engine.channel = MagicMock()
        engine.channel.channel_id = "chat_123"

        manager = MagicMock()
        manager.get_activated_engine.return_value = engine
        handler._get_engine_manager = MagicMock(return_value=manager)

        # Mock project_manager property
        mock_project_manager = MagicMock()
        mock_project_manager.get_project_for_chat.return_value = None

        value = {
            "action": "slock_clarify_confirm",
            "channel_id": "chat_123",
            "message_preview": "帮我写一个快速排序",
            "message_id": "original_msg_001",
        }

        with patch.object(
            type(handler), 'project_manager',
            new_callable=PropertyMock, return_value=mock_project_manager
        ):
            handler._handle_clarify_confirm("card_msg_001", "chat_123", value)

        # Should have called handle_message with the original message
        handler.handle_message.assert_called_once()
        call_args = handler.handle_message.call_args
        assert call_args[0][1] == "chat_123"  # chat_id
        assert call_args[0][2] == "帮我写一个快速排序"  # message text

    def test_clarify_confirm_without_engine_triggers_auto_activate(self):
        """When no engine exists, confirm button triggers activate_slock."""
        from unittest.mock import PropertyMock, patch

        handler = self._make_handler()
        handler.activate_slock = MagicMock(return_value=True)

        manager = MagicMock()
        manager.get_activated_engine.return_value = None  # No engine yet
        handler._get_engine_manager = MagicMock(return_value=manager)

        # Mock project_manager property
        mock_project_manager = MagicMock()
        mock_project_manager.get_project_for_chat.return_value = None

        value = {
            "action": "slock_clarify_confirm",
            "channel_id": "chat_new",
            "message_preview": "帮我创建一个新项目",
            "message_id": "original_msg_002",
        }

        with patch.object(
            type(handler), 'project_manager',
            new_callable=PropertyMock, return_value=mock_project_manager
        ):
            handler._handle_clarify_confirm("card_msg_002", "chat_new", value)

        # Should have called activate_slock with the requirement
        handler.activate_slock.assert_called_once()
        call_kwargs = handler.activate_slock.call_args[1]
        assert call_kwargs["chat_id"] == "chat_new"
        assert call_kwargs["requirement"] == "帮我创建一个新项目"

    def test_clarify_confirm_updates_card_to_confirmed(self):
        """Confirm button updates card to show '已确认这是任务'."""
        import json
        from unittest.mock import PropertyMock, patch

        handler = self._make_handler()
        handler.handle_message = MagicMock()

        engine = MagicMock()
        engine.channel = MagicMock()
        engine.channel.channel_id = "chat_123"

        manager = MagicMock()
        manager.get_activated_engine.return_value = engine
        handler._get_engine_manager = MagicMock(return_value=manager)

        # Mock project_manager property
        mock_project_manager = MagicMock()
        mock_project_manager.get_project_for_chat.return_value = None

        value = {
            "action": "slock_clarify_confirm",
            "channel_id": "chat_123",
            "message_preview": "我的任务内容",
            "message_id": "msg_003",
        }

        with patch.object(
            type(handler), 'project_manager',
            new_callable=PropertyMock, return_value=mock_project_manager
        ):
            handler._handle_clarify_confirm("card_msg_003", "chat_123", value)

        # Card should have been updated
        handler.update_card.assert_called_once()
        card_json = handler.update_card.call_args[0][1]
        card = json.loads(card_json)

        # Card should show confirmation
        assert "已确认这是任务" in card["header"]["title"]["content"] or \
               any("已确认这是任务" in e.get("content", "")
                   for e in card["body"]["elements"] if e.get("tag") == "markdown")

    def test_clarify_ignore_updates_card_to_ignored(self):
        """Ignore button updates card to show '已忽略' without creating task."""
        import json

        handler = self._make_handler()
        handler.handle_message = MagicMock()
        handler.activate_slock = MagicMock()

        value = {
            "action": "slock_clarify_ignore",
            "channel_id": "chat_123",
            "message_preview": "今天天气真好",
            "message_id": "msg_004",
        }

        handler._handle_clarify_ignore("card_msg_004", "chat_123", value)

        # Card should have been updated
        handler.update_card.assert_called_once()
        card_json = handler.update_card.call_args[0][1]
        card = json.loads(card_json)

        # Card should show ignored status
        assert "已忽略" in card["header"]["title"]["content"] or \
               any("已忽略" in e.get("content", "")
                   for e in card["body"]["elements"] if e.get("tag") == "markdown")

        # Should NOT have created any task
        handler.handle_message.assert_not_called()
        handler.activate_slock.assert_not_called()

    def test_clarify_ignore_does_not_call_engine_manager(self):
        """Ignore button should not try to get or activate engine."""
        handler = self._make_handler()
        handler._get_engine_manager = MagicMock()

        value = {
            "action": "slock_clarify_ignore",
            "channel_id": "chat_123",
            "message_preview": "只是闲聊",
            "message_id": "msg_005",
        }

        handler._handle_clarify_ignore("card_msg_005", "chat_123", value)

        # Should not have tried to get engine manager
        handler._get_engine_manager.assert_not_called()

    def test_clarify_ignore_card_shows_no_task_created(self):
        """Ignored card should explicitly state no task was created."""
        import json

        handler = self._make_handler()

        value = {
            "action": "slock_clarify_ignore",
            "channel_id": "chat_123",
            "message_preview": "随便聊聊",
            "message_id": "msg_006",
        }

        handler._handle_clarify_ignore("card_msg_006", "chat_123", value)

        card_json = handler.update_card.call_args[0][1]
        card = json.loads(card_json)

        # Card body should contain "不会创建任务" message
        body_text = "".join(
            e.get("content", "") for e in card["body"]["elements"]
            if e.get("tag") == "markdown"
        )
        assert "不会创建任务" in body_text


# ============================================================
# WP2: Auto-activate denied does not fallback to shell
# ============================================================


class TestAutoActivateDeniedDoesNotFallbackToShell:
    """WP2: When auto-activation is denied, dispatcher must:
    1. Send a denial card to the user
    2. NOT fall through to shell routing
    3. Return immediately after sending the card

    This ensures denied users don't accidentally get shell execution
    when slock auto-activation is blocked.
    """

    def test_denied_sends_activation_denied_card(self):
        """When auto-activate is denied, _reply_card must be called with denial card."""
        from unittest.mock import MagicMock, patch

        from src.feishu.dispatcher import FeishuRequestContext, MessageDispatcher

        client = MagicMock()
        client.settings = MagicMock()
        client.settings.slock_passive_mode = True
        client._get_effective_mode.return_value = (MagicMock(), False)
        client._is_deep_command.return_value = False
        client._is_spec_command.return_value = False
        client._is_slock_command.return_value = False
        client._is_workflow_command.return_value = False
        client._is_slock_managed_chat.return_value = False
        client._should_auto_activate_slock.return_value = True
        # Simulate activation denied
        client._auto_activate_slock.return_value = (False, "rate_limit")
        client._is_interceptable_command_match.return_value = False

        dispatcher = MessageDispatcher(client)
        ctx = FeishuRequestContext(
            message_id="msg_denied_001",
            chat_id="group_denied",
            text="请帮我实现一个快速排序算法",
            chat_type="group",
        )

        # Mock TaskClassifier to return "task" classification
        with patch(
            "src.slock_engine.task_classifier.TaskClassifier.classify_with_uncertainty",
            return_value=("task", 0.9),
        ):
            dispatcher.process_request(ctx)

        # Verify denial card was sent
        client._reply_card.assert_called_once()
        call_args = client._reply_card.call_args
        assert call_args[0][0] == "msg_denied_001"
        # Card content should contain JSON with denial info
        card_json = call_args[0][1]
        assert "自动协作模式激活受限" in card_json or "rate_limit" in card_json or "denied" in card_json.lower()

    def test_denied_does_not_fall_through_to_shell(self):
        """When auto-activate is denied, shell routing must NOT be reached."""
        from unittest.mock import MagicMock, patch

        from src.feishu.dispatcher import FeishuRequestContext, MessageDispatcher

        client = MagicMock()
        client.settings = MagicMock()
        client.settings.slock_passive_mode = True
        client._get_effective_mode.return_value = (MagicMock(), False)
        client._is_deep_command.return_value = False
        client._is_spec_command.return_value = False
        client._is_slock_command.return_value = False
        client._is_workflow_command.return_value = False
        client._is_slock_managed_chat.return_value = False
        client._should_auto_activate_slock.return_value = True
        # Simulate activation denied
        client._auto_activate_slock.return_value = (False, "admin_required")
        client._is_interceptable_command_match.return_value = False

        dispatcher = MessageDispatcher(client)
        ctx = FeishuRequestContext(
            message_id="msg_denied_002",
            chat_id="group_denied",
            text="帮我写一个网络爬虫",
            chat_type="group",
        )

        with patch(
            "src.slock_engine.task_classifier.TaskClassifier.classify_with_uncertainty",
            return_value=("task", 0.85),
        ):
            dispatcher.process_request(ctx)

        # Shell-related methods should NOT be called
        client._submit_shell_command.assert_not_called()
        client._intent_recognizer.recognize.assert_not_called()
        client._handle_coco_message.assert_not_called()

    def test_denied_returns_immediately_after_card(self):
        """Verify call chain returns after sending denial card (no downstream processing)."""
        from unittest.mock import MagicMock, patch

        from src.feishu.dispatcher import FeishuRequestContext, MessageDispatcher

        client = MagicMock()
        client.settings = MagicMock()
        client.settings.slock_passive_mode = True
        client._get_effective_mode.return_value = (MagicMock(), False)
        client._is_deep_command.return_value = False
        client._is_spec_command.return_value = False
        client._is_slock_command.return_value = False
        client._is_workflow_command.return_value = False
        client._is_slock_managed_chat.return_value = False
        client._should_auto_activate_slock.return_value = True
        # Simulate activation denied with not_whitelisted reason
        client._auto_activate_slock.return_value = (False, "not_whitelisted")
        client._is_interceptable_command_match.return_value = False

        dispatcher = MessageDispatcher(client)
        ctx = FeishuRequestContext(
            message_id="msg_denied_003",
            chat_id="group_denied",
            text="重构这个模块的代码",
            chat_type="group",
        )

        with patch(
            "src.slock_engine.task_classifier.TaskClassifier.classify_with_uncertainty",
            return_value=("task", 0.95),
        ):
            dispatcher.process_request(ctx)

        # Verify the exact sequence: _auto_activate_slock called, then _reply_card called
        # and nothing downstream executed
        assert client._auto_activate_slock.call_count == 1
        assert client._reply_card.call_count == 1

        # None of these should be called after denial
        client._handle_slock_message.assert_not_called()
        client._add_reaction.assert_not_called()  # No processing reaction for denied
        client._is_exit_command.assert_not_called()
        client._handle_intercepted_command.assert_not_called()

    def test_allowed_activation_still_works(self):
        """Sanity check: when activation is allowed, flow proceeds normally."""
        from unittest.mock import MagicMock, patch

        from src.feishu.dispatcher import FeishuRequestContext, MessageDispatcher

        client = MagicMock()
        client.settings = MagicMock()
        client.settings.slock_passive_mode = True
        client._get_effective_mode.return_value = (MagicMock(), False)
        client._is_deep_command.return_value = False
        client._is_spec_command.return_value = False
        client._is_slock_command.return_value = False
        client._is_workflow_command.return_value = False
        client._is_slock_managed_chat.return_value = False
        client._should_auto_activate_slock.return_value = True
        # Activation allowed
        client._auto_activate_slock.return_value = (True, "allowed")

        dispatcher = MessageDispatcher(client)
        ctx = FeishuRequestContext(
            message_id="msg_allowed_001",
            chat_id="group_new",
            text="实现用户登录功能",
            chat_type="group",
        )

        with patch(
            "src.slock_engine.task_classifier.TaskClassifier.classify_with_uncertainty",
            return_value=("task", 0.9),
        ):
            dispatcher.process_request(ctx)

        # When allowed: processing reaction added, no denial card
        client._add_reaction.assert_called()
        client._reply_card.assert_not_called()

    def test_all_three_denial_reasons_send_card(self):
        """All three denial reasons should trigger card send and stop routing."""
        from unittest.mock import MagicMock, patch

        from src.feishu.dispatcher import FeishuRequestContext, MessageDispatcher

        for reason in ["rate_limit", "admin_required", "not_whitelisted"]:
            client = MagicMock()
            client.settings = MagicMock()
            client.settings.slock_passive_mode = True
            client._get_effective_mode.return_value = (MagicMock(), False)
            client._is_deep_command.return_value = False
            client._is_spec_command.return_value = False
            client._is_slock_command.return_value = False
            client._is_workflow_command.return_value = False
            client._is_slock_managed_chat.return_value = False
            client._should_auto_activate_slock.return_value = True
            client._auto_activate_slock.return_value = (False, reason)
            client._is_interceptable_command_match.return_value = False

            dispatcher = MessageDispatcher(client)
            ctx = FeishuRequestContext(
                message_id=f"msg_{reason}",
                chat_id=f"group_{reason}",
                text="请实现任务分派功能",
                chat_type="group",
            )

            with patch(
                "src.slock_engine.task_classifier.TaskClassifier.classify_with_uncertainty",
                return_value=("task", 0.8),
            ):
                dispatcher.process_request(ctx)

            # Each reason should send card and NOT fall through
            client._reply_card.assert_called_once()
            client._submit_shell_command.assert_not_called()
            client._intent_recognizer.recognize.assert_not_called()


class TestSmartShellBypassesPassiveSlock:
    """SMART shell commands must never enter passive Slock classification."""

    @staticmethod
    def _make_client():
        from src.agent.intent_recognizer import IntentRecognizer

        client = MagicMock()
        client.settings = MagicMock()
        client.settings.slock_passive_mode = True
        client.settings.thread_programming_enabled = False
        client._get_effective_mode.return_value = (InteractionMode.SMART, False)
        client._is_topic_engine_context.return_value = False
        client._is_deep_command.return_value = False
        client._is_spec_command.return_value = False
        client._is_workflow_command.return_value = False
        client._is_slock_command.return_value = False
        client._is_slock_managed_chat.return_value = False
        client._should_auto_activate_slock.return_value = True
        client._is_interceptable_command_match.return_value = False
        client._pending_image_lock = MagicMock()
        client._pending_image_only = set()
        client._intent_recognizer = MagicMock(wraps=IntentRecognizer())
        client._get_working_dir.return_value = "/repo"
        return client

    @pytest.mark.parametrize(
        "text",
        ["pwd", "ls -la", "echo hi", "git status", "./restart.sh rr"],
    )
    def test_unmanaged_shell_only_uses_shell_route(self, text):
        from src.feishu.dispatcher import FeishuRequestContext, MessageDispatcher

        client = self._make_client()
        dispatcher = MessageDispatcher(client)

        with (
            patch("src.slock_engine.gateway.classify_message") as classify,
            patch("src.slock_engine.gateway.attempt_autonomous_resolve") as resolve,
        ):
            dispatcher.process_request(
                FeishuRequestContext(
                    message_id="msg_shell",
                    chat_id="group_unmanaged",
                    text=text,
                    shell_fast_tracked=True,
                    chat_type="group",
                )
            )

        classify.assert_not_called()
        resolve.assert_not_called()
        client._reply_card.assert_not_called()
        client._auto_activate_slock.assert_not_called()
        client._handle_slock_message.assert_not_called()
        client._system_handler.execute_shell_and_reply.assert_called_once_with(
            "msg_shell",
            "group_unmanaged",
            text,
            "/repo",
            None,
        )
        client._intent_recognizer.recognize.assert_called_once_with(text, "smart")

    def test_clarification_card_is_terminal(self):
        from src.agent.intent_recognizer import IntentResult, IntentType
        from src.feishu.dispatcher import FeishuRequestContext, MessageDispatcher
        from src.slock_engine.gateway import SlockClassification, SlockMessageClass

        client = self._make_client()
        client._intent_recognizer.recognize.return_value = IntentResult.single(
            IntentType.SHELL_COMMAND,
            data={"command": "这算任务吗"},
        )
        dispatcher = MessageDispatcher(client)
        unresolved = MagicMock(resolved=False)

        with (
            patch(
                "src.slock_engine.gateway.classify_message",
                return_value=SlockClassification(SlockMessageClass.UNCERTAIN),
            ),
            patch(
                "src.slock_engine.gateway.attempt_autonomous_resolve",
                return_value=unresolved,
            ),
        ):
            dispatcher.process_request(
                FeishuRequestContext(
                    message_id="msg_uncertain",
                    chat_id="group_unmanaged",
                    text="这算任务吗",
                    chat_type="group",
                )
            )

        client._reply_card.assert_called_once()
        client._intent_recognizer.recognize.assert_not_called()
        client._submit_shell_command.assert_not_called()
        client._system_handler.execute_shell_and_reply.assert_not_called()
