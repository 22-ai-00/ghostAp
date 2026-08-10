"""Unit tests for ChatLockGate — ingress middleware for chat-lock interception.

Tests cover:
- No-op when chat_lock_manager is None
- check() delegates to should_block for message path
- check_card_action() delegates to should_block_card_action
- Fail-close: exception in _try_block blocks non-admin in locked chat
- Fail-close: exception in _try_block does NOT block admin
- Fail-close: card-action exception path silently blocks (no card sent)
- Dedup: _should_send_intercept returns True first time, False on repeat
- close() stops dedup cache cleanup thread
- Throttled path: handler.send_chat_lock_throttled_reply called on dedup hit
- Fallback: reply_text used when the intercept card cannot be delivered
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.feishu.chat_lock_gate import ChatLockGate
from src.feishu.message_cache import MessageCache
from src.feishu.slash_command_parser import CommandMatch, SlashCommandParser


def _make_gate(
    *,
    clm=None,
    handler=None,
    cache_ttl: int = 30,
):
    """Build a ChatLockGate with its required handler dependency."""
    if handler is None:
        handler = MagicMock()
    cache = MessageCache(ttl=cache_ttl, max_size=10_000, cleanup_interval=60)
    gate = ChatLockGate(chat_lock_manager=clm, dedup_cache=cache, handler=handler)
    return gate, handler


class TestNoClm(unittest.TestCase):
    """When chat_lock_manager is None, all checks pass through."""

    def test_check_returns_false(self):
        gate, _ = _make_gate(clm=None)
        assert gate.check("chat", "user", "msg") is False

    def test_check_card_action_returns_false(self):
        gate, _ = _make_gate(clm=None)
        assert gate.check_card_action("chat", "user", "msg") is False


class TestCheckMessage(unittest.TestCase):
    """Message-path blocking via check()."""

    def test_not_blocked_returns_false(self):
        clm = MagicMock()
        clm.should_block.return_value = False
        gate, _ = _make_gate(clm=clm, handler=MagicMock())

        assert gate.check("chat", "user", "msg") is False

    def test_blocked_returns_true_and_sends_card(self):
        clm = MagicMock()
        clm.should_block.return_value = True
        clm.is_admin.return_value = False
        clm.is_locked.return_value = True
        handler = MagicMock()
        gate, _ = _make_gate(clm=clm, handler=handler)

        assert gate.check("chat", "user", "msg") is True
        handler.send_chat_lock_intercept_card.assert_called_once_with("msg", "chat", clm)

    def test_blocked_falls_back_to_text_when_card_delivery_fails(self):
        clm = MagicMock()
        clm.should_block.return_value = True
        clm.is_admin.return_value = False
        clm.is_locked.return_value = True
        handler = MagicMock()
        handler.send_chat_lock_intercept_card.side_effect = RuntimeError("delivery failed")
        gate, _ = _make_gate(clm=clm, handler=handler)

        assert gate.check("chat", "user", "msg") is True
        handler.reply_text.assert_called_once()

    def test_blocked_spec_goal_with_tab_still_sends_visible_intercept_feedback(self):
        clm = MagicMock()
        clm.should_block.return_value = True
        handler = MagicMock()
        gate, _ = _make_gate(clm=clm, handler=handler)

        m = SlashCommandParser.parse("/spec\t实现登录功能")
        assert m is not None
        assert gate.check("chat", "user", "msg", command_match=m) is True
        handler.send_chat_lock_intercept_card.assert_called_once_with("msg", "chat", clm)

    def test_gate_check_never_reparses_text(self):
        """Gate must consume request-scoped CommandMatch (SSOT)."""
        clm = MagicMock()
        clm.should_block.return_value = True
        handler = MagicMock()
        gate, _ = _make_gate(clm=clm, handler=handler)

        m = CommandMatch(
            raw_text="/spec",
            normalized_text="/spec",
            raw_command="/spec",
            command="/spec",
            args="",
            has_args=False,
        )

        with patch("src.feishu.slash_command_parser.SlashCommandParser.parse") as p:
            assert gate.check("chat", "user", "msg", command_match=m) is True
            assert p.call_count == 0


class TestCheckCardAction(unittest.TestCase):
    """Card-action blocking via check_card_action()."""

    def test_not_blocked(self):
        clm = MagicMock()
        clm.should_block_card_action.return_value = False
        gate, _ = _make_gate(clm=clm)

        assert gate.check_card_action("chat", "user", "msg", action_type="btn") is False

    def test_blocked(self):
        clm = MagicMock()
        clm.should_block_card_action.return_value = True
        handler = MagicMock()
        gate, _ = _make_gate(clm=clm, handler=handler)

        assert gate.check_card_action("chat", "user", "msg", action_type="btn") is True


class TestFailClose(unittest.TestCase):
    """Fail-close semantics: exceptions → block non-admin in locked chat."""

    def test_exception_blocks_nonadmin_in_locked_chat(self):
        clm = MagicMock()
        clm.is_admin.return_value = False
        clm.is_locked.return_value = True
        clm.should_block.side_effect = RuntimeError("db down")
        handler = MagicMock()
        gate, _ = _make_gate(clm=clm, handler=handler)

        assert gate.check("chat", "user", "msg") is True
        handler.send_chat_lock_intercept_card.assert_called_once()

    def test_exception_does_not_block_admin(self):
        clm = MagicMock()
        clm.is_admin.return_value = True
        clm.should_block.side_effect = RuntimeError("db down")
        gate, _ = _make_gate(clm=clm)

        assert gate.check("chat", "admin", "msg") is False

    def test_exception_does_not_block_unlocked_chat(self):
        clm = MagicMock()
        clm.is_admin.return_value = False
        clm.is_locked.return_value = False
        clm.should_block.side_effect = RuntimeError("db down")
        gate, _ = _make_gate(clm=clm)

        assert gate.check("chat", "user", "msg") is False

    def test_card_action_exception_blocks_silently(self):
        """Card-action fail-close: blocks but does NOT send intercept card."""
        clm = MagicMock()
        clm.is_admin.return_value = False
        clm.is_locked.return_value = True
        clm.should_block_card_action.side_effect = RuntimeError("oops")
        handler = MagicMock()
        gate, _ = _make_gate(clm=clm, handler=handler)

        assert gate.check_card_action("chat", "user", "msg") is True
        # No intercept card for card-action path
        handler.send_chat_lock_intercept_card.assert_not_called()


class TestDedup(unittest.TestCase):
    """Dedup via _should_send_intercept."""

    def test_first_call_returns_true(self):
        gate, _ = _make_gate()
        assert gate._should_send_intercept("chat", "user") is True

    def test_repeat_call_returns_false(self):
        gate, _ = _make_gate()
        gate._should_send_intercept("chat", "user")
        assert gate._should_send_intercept("chat", "user") is False

    def test_different_pairs_independent(self):
        gate, _ = _make_gate()
        assert gate._should_send_intercept("chat_a", "user_1") is True
        assert gate._should_send_intercept("chat_b", "user_1") is True


class TestThrottledPath(unittest.TestCase):
    """When dedup suppresses the full card, throttled reply should fire."""

    def test_throttled_sends_throttled_reply(self):
        clm = MagicMock()
        clm.should_block.return_value = True
        handler = MagicMock()
        gate, _ = _make_gate(clm=clm, handler=handler)

        # Consume dedup slot
        gate._should_send_intercept("chat", "user")
        # Second block triggers throttled path
        assert gate.check("chat", "user", "msg") is True
        handler.send_chat_lock_throttled_reply.assert_called_once()
        handler.send_chat_lock_intercept_card.assert_not_called()

class TestClose(unittest.TestCase):
    """close() delegates to dedup cache."""

    def test_close_stops_cleanup(self):
        gate, _ = _make_gate()
        gate.close()
        # No exception — successful close

    def test_close_swallows_exception(self):
        gate, _ = _make_gate()
        gate._dedup.stop_cleanup_thread = MagicMock(side_effect=RuntimeError("oops"))
        gate.close()  # Should not raise
