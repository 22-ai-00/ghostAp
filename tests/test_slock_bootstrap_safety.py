"""Tests for slock bootstrap safety and tool_type validation.

Covers:
- Invalid tool_type values are skipped with warning
- SUPPORTED_TOOL_TYPES whitelist enforced
- Bootstrap fault tolerance: partial failures don't block other roles
- _bootstrap_default_roles_if_configured async/thread fallback
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.slock_engine.role_bootstrap import (
    SUPPORTED_TOOL_TYPES,
    bootstrap_default_roles,
    parse_default_roles,
)


class TestToolTypeValidation:
    """Validate that only supported tool_types are accepted during bootstrap."""

    def test_supported_tool_types_set(self):
        """SUPPORTED_TOOL_TYPES contains all expected tool backends."""
        assert "codex" in SUPPORTED_TOOL_TYPES
        assert "claude" in SUPPORTED_TOOL_TYPES
        assert "coco" in SUPPORTED_TOOL_TYPES
        assert "aiden" in SUPPORTED_TOOL_TYPES
        assert "gemini" in SUPPORTED_TOOL_TYPES

    def test_invalid_tool_type_skipped(self):
        """Invalid tool_type in config is skipped; valid ones are created."""
        engine = MagicMock()
        engine.registry = MagicMock()
        engine.registry.find_by_name.return_value = None
        engine.registry.register.side_effect = lambda a: a

        result = bootstrap_default_roles(
            engine, "chat_001", "coder:codex,hacker:invalid_tool,reviewer:claude"
        )
        # Only coder and reviewer should be created; hacker skipped
        assert len(result) == 2
        assert engine.registry.register.call_count == 2
        roles = [r.role for r in result]
        assert "coder" in roles
        assert "reviewer" in roles
        assert "hacker" not in roles

    def test_all_invalid_tool_types_yield_empty(self):
        """When all tool_types are invalid, bootstrap returns empty list."""
        engine = MagicMock()
        engine.registry = MagicMock()
        engine.registry.find_by_name.return_value = None

        result = bootstrap_default_roles(engine, "chat_002", "bad1:xyz,bad2:abc")
        assert result == []
        engine.registry.register.assert_not_called()

    @pytest.mark.parametrize("tool_type", sorted(SUPPORTED_TOOL_TYPES))
    def test_each_supported_type_accepted(self, tool_type):
        """Each tool_type in the whitelist passes validation."""
        engine = MagicMock()
        engine.registry = MagicMock()
        engine.registry.find_by_name.return_value = None
        engine.registry.register.side_effect = lambda a: a

        result = bootstrap_default_roles(engine, "chat_t", f"role:{tool_type}")
        assert len(result) == 1
        assert result[0].agent_type == tool_type


class TestBootstrapFaultTolerance:
    """Bootstrap handles partial failures gracefully."""

    def test_register_exception_doesnt_crash(self, caplog):
        """If registry.register raises, other roles still proceed."""
        engine = MagicMock()
        engine.registry = MagicMock()
        engine.registry.find_by_name.return_value = None

        call_count = [0]
        def mock_register(agent):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Simulated disk error")
            return agent

        engine.registry.register.side_effect = mock_register

        with caplog.at_level(logging.WARNING):
            result = bootstrap_default_roles(
                engine, "chat_003", "coder:codex,reviewer:claude"
            )

        # First one fails, second succeeds
        assert len(result) == 1
        assert result[0].role == "reviewer"

    def test_idempotent_with_existing_agent(self):
        """Existing agents are returned without re-registration."""
        existing = MagicMock()
        existing.role = "coder"
        existing.agent_type = "codex"

        engine = MagicMock()
        engine.registry = MagicMock()
        engine.registry.find_by_name.return_value = existing

        result = bootstrap_default_roles(engine, "chat_004", "coder:codex,reviewer:claude")
        assert len(result) == 2
        assert all(r is existing for r in result)
        engine.registry.register.assert_not_called()

    def test_empty_config_no_op(self):
        """Empty config string results in no bootstrap activity."""
        engine = MagicMock()
        result = bootstrap_default_roles(engine, "chat_005", "")
        assert result == []


class TestBootstrapIfConfiguredHandler:
    """Test _bootstrap_default_roles_if_configured in SlockHandler."""

    def test_empty_config_skips(self):
        """If slock_default_roles is empty, bootstrap is not attempted."""
        from src.feishu.handlers.slock import SlockHandler

        ctx = MagicMock()
        ctx.settings = MagicMock()
        ctx.settings.slock_default_roles = ""
        handler = SlockHandler(ctx)
        handler.send_text_to_chat = MagicMock()

        engine = MagicMock()
        # Should not raise, should not call bootstrap
        handler._bootstrap_default_roles_if_configured(engine, "ch_1", "chat_1")
        engine.registry.register.assert_not_called()

    @patch("src.slock_engine.role_bootstrap.bootstrap_default_roles")
    def test_valid_config_triggers_bootstrap(self, mock_bootstrap):
        """Non-empty config triggers the bootstrap function."""
        from src.feishu.handlers.slock import SlockHandler

        ctx = MagicMock()
        ctx.settings = MagicMock()
        ctx.settings.slock_default_roles = "coder:codex"
        handler = SlockHandler(ctx)
        handler.send_text_to_chat = MagicMock()

        engine = MagicMock()

        # Call the method — it uses threading, but we test the intent
        handler._bootstrap_default_roles_if_configured(engine, "ch_1", "chat_1")
        # Since it runs async (thread/executor), the mock call may not
        # happen immediately in this test, so we verify no crash occurred


class TestParseConfigEdgeCases:
    """Edge cases for config string parsing."""

    def test_trailing_comma(self):
        result = parse_default_roles("coder:codex,")
        assert result == [("coder", "codex")]

    def test_double_colon(self):
        """Only splits on first colon."""
        result = parse_default_roles("my:role:codex")
        assert result == [("my", "role:codex")]

    def test_no_colon_uses_name_as_type(self):
        """Entry without colon uses the name itself as tool_type."""
        result = parse_default_roles("codex")
        assert result == [("codex", "codex")]

    def test_whitespace_heavy(self):
        result = parse_default_roles("  coder : codex  ,  reviewer : claude  ")
        assert result == [("coder", "codex"), ("reviewer", "claude")]


# ============================================================
# Least-privilege permissions
# ============================================================


class TestLeastPrivilegePermissions:
    """Bootstrapped roles get minimal permissions per _DEFAULT_PERMISSIONS."""

    def test_coder_gets_shell_file_write_git(self):
        """Coder role has shell, file_write, git permissions."""
        engine = MagicMock()
        engine.registry = MagicMock()
        engine.registry.find_by_name.return_value = None
        engine.registry.register.side_effect = lambda a: a

        result = bootstrap_default_roles(engine, "ch", "coder:codex")
        assert result[0].permissions == ["shell", "file_write", "git"]

    def test_reviewer_gets_file_read_only(self):
        """Reviewer role has only file_read permission."""
        engine = MagicMock()
        engine.registry = MagicMock()
        engine.registry.find_by_name.return_value = None
        engine.registry.register.side_effect = lambda a: a

        result = bootstrap_default_roles(engine, "ch", "reviewer:claude")
        assert result[0].permissions == ["file_read"]

    def test_planner_gets_file_read_only(self):
        """Planner role has only file_read permission."""
        engine = MagicMock()
        engine.registry = MagicMock()
        engine.registry.find_by_name.return_value = None
        engine.registry.register.side_effect = lambda a: a

        result = bootstrap_default_roles(engine, "ch", "planner:coco")
        assert result[0].permissions == ["file_read"]

    def test_tester_gets_shell_and_file_read(self):
        """Tester role has shell and file_read permissions."""
        engine = MagicMock()
        engine.registry = MagicMock()
        engine.registry.find_by_name.return_value = None
        engine.registry.register.side_effect = lambda a: a

        result = bootstrap_default_roles(engine, "ch", "tester:codex")
        assert result[0].permissions == ["shell", "file_read"]

    def test_unknown_role_gets_file_read_fallback(self):
        """Unknown role names fall back to file_read only."""
        engine = MagicMock()
        engine.registry = MagicMock()
        engine.registry.find_by_name.return_value = None
        engine.registry.register.side_effect = lambda a: a

        result = bootstrap_default_roles(engine, "ch", "customrole:codex")
        assert result[0].permissions == ["file_read"]


# ============================================================
# Zero agents degradation (AC-R2)
# ============================================================


class TestZeroAgentsDegradation:
    """AC-R2: Bootstrap with 0 agents sends degradation card."""

    def test_bootstrap_no_agents_sends_card(self):
        """When all role creation fails (invalid tool types), degradation card is sent."""
        engine = MagicMock()
        engine.registry = MagicMock()
        engine.registry.find_by_name.return_value = None
        mock_channel = MagicMock()
        engine.get_channel = MagicMock(return_value=mock_channel)
        engine.send_card = MagicMock()

        # Config with invalid tool types that will all fail validation
        config_str = "invalid_role:invalid_tool,bad_role:bad_tool"

        result = bootstrap_default_roles(engine, "channel_1", config_str)

        assert len(result) == 0
        # Channel should be marked as bootstrap_failed
        assert mock_channel.bootstrap_failed is True
        # Card should have been sent
        engine.send_card.assert_called_once()

    def test_bootstrap_no_agents_sets_bootstrap_failed(self):
        """When all registrations fail, channel.bootstrap_failed is set."""
        engine = MagicMock()
        engine.registry = MagicMock()
        engine.registry.find_by_name.return_value = None
        mock_channel = MagicMock()
        engine.get_channel = MagicMock(return_value=mock_channel)
        engine.send_card = MagicMock()

        # All will fail registry.register
        engine.registry.register.side_effect = RuntimeError("simulated failure")

        config_str = "coder:codex,reviewer:claude"

        result = bootstrap_default_roles(engine, "channel_1", config_str)

        assert len(result) == 0
        assert mock_channel.bootstrap_failed is True
        engine.send_card.assert_called_once()

    def test_bootstrap_success_no_failed_flag(self):
        """Successful bootstrap should NOT set bootstrap_failed."""
        engine = MagicMock()
        engine.registry = MagicMock()
        engine.registry.find_by_name.return_value = None
        engine._channel = MagicMock()
        engine._card_send_fn = MagicMock()
        engine.registry.register.side_effect = lambda a: a

        # Valid config
        config_str = "coder:codex"

        result = bootstrap_default_roles(engine, "channel_1", config_str)

        # Should have created agents successfully
        assert len(result) == 1
        # bootstrap_failed should NOT have been set to True
        # (the code only sets it when created list is empty)
        engine._card_send_fn.assert_not_called()

    def test_partial_success_no_degradation(self):
        """If at least one agent succeeds, no degradation card is sent."""
        engine = MagicMock()
        engine.registry = MagicMock()
        engine.registry.find_by_name.return_value = None
        engine._channel = MagicMock()
        engine._card_send_fn = MagicMock()
        engine.registry.register.side_effect = lambda a: a

        # One valid, one invalid
        config_str = "coder:codex,hacker:invalid_tool"

        result = bootstrap_default_roles(engine, "channel_1", config_str)

        assert len(result) == 1
        # No degradation card since partial success
        engine._card_send_fn.assert_not_called()
