"""AC7: Zero Human Interaction end-to-end tests.

Validates: During Agent execution, NO confirmation/permission/interaction
messages are sent to users. The Agent operates fully autonomously.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.slock_engine.engine import SlockEngine, SlockEngineCallbacks
from src.slock_engine.models import (
    AgentIdentity,
    SlockChannel,
)
from src.slock_engine.mouthpiece import Mouthpiece

# Patterns that should NEVER appear in agent output messages
FORBIDDEN_INTERACTION_PATTERNS = [
    "confirm",
    "approve",
    "do you want",
    "would you like",
    "please verify",
    "are you sure",
    "permission",
    "authorize",
    "shall I",
    "may I",
    "请确认",
    "是否继续",
    "是否同意",
    "请授权",
    "需要您的确认",
    "是否允许",
]


@pytest.fixture
def engine_with_agent(tmp_path):
    """Create engine with a registered agent."""
    engine = SlockEngine(
        chat_id="chat_zero_hi",
        root_path=str(tmp_path / "project"),
        agent_type="coco",
        engine_name="Slock",
        memory_base_path=str(tmp_path / "slock_storage"),
    )
    channel = SlockChannel(
        channel_id="chat_zero_hi",
        name="Zero HI Team [Slock]",
        team_name="Zero HI Team",
        owner_id="admin",
    )
    engine.activate_channel(channel)

    agent = AgentIdentity(
        name="AutoAgent",
        emoji="🤖",
        agent_type="coco",
        model_name="test-model",
        role="coder",
        owner_group="chat_zero_hi",
    )
    engine.registry.register(agent)
    return engine, agent


class TestZeroHumanInteractionExecution:
    """Agent execution must never produce interaction requests."""

    def test_auto_approve_true_in_session_creation(self, engine_with_agent):
        """ACP session is created with auto_approve=True."""
        engine, agent = engine_with_agent
        session_kwargs: list[dict] = []

        def capture_create_session(**kwargs):
            session_kwargs.append(kwargs)
            mock_session = MagicMock()
            mock_session.send_prompt.return_value = MagicMock(text="Task completed")
            return mock_session

        with patch("src.slock_engine.engine.create_engine_session", side_effect=capture_create_session):
            engine.execute("Implement feature X")

        assert len(session_kwargs) >= 1
        assert session_kwargs[0].get("auto_approve") is True

    def test_no_forbidden_patterns_in_mouthpiece_output(self, engine_with_agent):
        """Mouthpiece formatted output contains no interaction patterns."""
        engine, agent = engine_with_agent

        # Simulate agent producing a clean response
        response = "I have implemented the feature. All tests pass."
        mouthpiece = Mouthpiece()
        text_output = mouthpiece.format_text(agent, response)
        card_output = mouthpiece.format_card(agent, response)

        for pattern in FORBIDDEN_INTERACTION_PATTERNS:
            assert pattern.lower() not in text_output.lower(), (
                f"Forbidden pattern '{pattern}' found in text output"
            )
            assert pattern.lower() not in str(card_output).lower(), (
                f"Forbidden pattern '{pattern}' found in card output"
            )

    def test_no_messages_sent_during_execution_except_result(self, engine_with_agent):
        """During _execute_agent, no intermediate messages are sent."""
        engine, agent = engine_with_agent
        messages_sent: list[str] = []

        # Track any callback messages
        callbacks = SlockEngineCallbacks(
            on_error=lambda msg: messages_sent.append(f"error:{msg}"),
        )

        mock_session = MagicMock()
        mock_session.send_prompt.return_value = MagicMock(text="Done")

        with patch("src.slock_engine.engine.create_engine_session", return_value=mock_session):
            result = engine.execute("Do task", callbacks=callbacks)

        # No error messages should have been sent
        assert len(messages_sent) == 0
        # Result should be a formatted string (not a question)
        assert result is not None
        for pattern in FORBIDDEN_INTERACTION_PATTERNS:
            assert pattern.lower() not in result.lower()

    def test_session_prompt_does_not_ask_questions(self, engine_with_agent):
        """The prompt sent to ACP session should not request user interaction."""
        engine, agent = engine_with_agent
        captured_prompts: list[str] = []

        mock_session = MagicMock()
        mock_session.send_prompt.return_value = MagicMock(text="OK")

        def capture_session(**kwargs):
            return mock_session

        with patch("src.slock_engine.engine.create_engine_session", side_effect=capture_session):
            engine.execute("Build the auth module")

        # Check the prompt sent to session
        if mock_session.send_prompt.called:
            prompt = mock_session.send_prompt.call_args[0][0]
            captured_prompts.append(prompt)

        for prompt in captured_prompts:
            # Prompt should not contain patterns that would elicit human interaction
            assert "ask the user" not in prompt.lower()
            assert "confirm with" not in prompt.lower()


class TestZeroHIToolInteractionSuppression:
    """Tool-level interaction suppression verification."""

    def test_codex_auto_approval_mode(self, engine_with_agent):
        """For codex agent_type, session must auto-approve file changes."""
        engine, _ = engine_with_agent

        # Register a codex-type agent
        codex_agent = AgentIdentity(
            name="CodexAgent",
            emoji="⚡",
            agent_type="codex",
            model_name="o3",
            role="coder",
            owner_group="chat_zero_hi",
        )
        engine.registry.register(codex_agent)

        session_kwargs: list[dict] = []

        def capture(**kwargs):
            session_kwargs.append(kwargs)
            mock = MagicMock()
            mock.send_prompt.return_value = MagicMock(text="Done")
            return mock

        with patch("src.slock_engine.engine.create_engine_session", side_effect=capture):
            # Route to codex agent specifically
            engine._execute_agent(codex_agent, "Fix bug", None)

        assert session_kwargs[0]["auto_approve"] is True

    def test_claude_auto_approval_mode(self, engine_with_agent):
        """For claude agent_type, session must auto-approve tool_use."""
        engine, _ = engine_with_agent

        claude_agent = AgentIdentity(
            name="ClaudeAgent",
            emoji="🎭",
            agent_type="claude",
            model_name="sonnet",
            role="reviewer",
            owner_group="chat_zero_hi",
        )
        engine.registry.register(claude_agent)

        session_kwargs: list[dict] = []

        def capture(**kwargs):
            session_kwargs.append(kwargs)
            mock = MagicMock()
            mock.send_prompt.return_value = MagicMock(text="Reviewed")
            return mock

        with patch("src.slock_engine.engine.create_engine_session", side_effect=capture):
            engine._execute_agent(claude_agent, "Review code", None)

        assert session_kwargs[0]["auto_approve"] is True

    def test_all_agent_types_get_auto_approve(self, engine_with_agent):
        """Every supported agent_type gets auto_approve=True."""
        engine, _ = engine_with_agent

        for agent_type in ["coco", "claude", "codex", "gemini"]:
            agent = AgentIdentity(
                name=f"{agent_type}-Agent",
                emoji="🔧",
                agent_type=agent_type,
                model_name="test",
                role="coder",
                owner_group="chat_zero_hi",
            )

            session_kwargs: list[dict] = []

            def capture(**kwargs):
                session_kwargs.append(kwargs)
                mock = MagicMock()
                mock.send_prompt.return_value = MagicMock(text="OK")
                return mock

            with patch("src.slock_engine.engine.create_engine_session", side_effect=capture):
                engine._execute_agent(agent, "task", None)

            assert session_kwargs[0]["auto_approve"] is True, (
                f"auto_approve not True for agent_type={agent_type}"
            )
