"""Tests for EmojiHook behavior when message_id is empty.

Validates that EmojiHook gracefully skips add_reaction calls when
message_id is falsy (degraded delivery mode without reply_to).
"""

import unittest
from unittest.mock import MagicMock

from src.card.hooks import EmojiHook


class TestEmojiHookEmptyMessageId(unittest.TestCase):
    """EmojiHook must skip reaction when message_id is empty."""

    def _make_hook(self, message_id: str = "") -> tuple[EmojiHook, MagicMock]:
        add_reaction = MagicMock()
        hook = EmojiHook(add_reaction, message_id, chat_id="chat_123")
        return hook, add_reaction

    def test_empty_string_skips_reaction_completed(self):
        hook, add_reaction = self._make_hook("")
        state = MagicMock()
        hook.on_terminal(state, "completed")
        add_reaction.assert_not_called()

    def test_empty_string_skips_reaction_failed(self):
        hook, add_reaction = self._make_hook("")
        state = MagicMock()
        hook.on_terminal(state, "failed")
        add_reaction.assert_not_called()

    def test_empty_string_skips_reaction_cancelled(self):
        hook, add_reaction = self._make_hook("")
        state = MagicMock()
        hook.on_terminal(state, "cancelled")
        add_reaction.assert_not_called()

    def test_empty_string_skips_reaction_ttl_expired(self):
        hook, add_reaction = self._make_hook("")
        state = MagicMock()
        hook.on_terminal(state, "ttl_expired")
        add_reaction.assert_not_called()

    def test_none_like_empty_skips_reaction(self):
        """None coerced to empty string should also skip."""
        add_reaction = MagicMock()
        hook = EmojiHook(add_reaction, "", chat_id="chat_456")
        state = MagicMock()
        hook.on_terminal(state, "completed")
        add_reaction.assert_not_called()

    def test_nonempty_message_id_calls_reaction(self):
        """Sanity check: non-empty message_id triggers reaction."""
        hook, add_reaction = self._make_hook("msg_abc")
        state = MagicMock()
        hook.on_terminal(state, "completed")
        add_reaction.assert_called_once_with("msg_abc", "PARTY")

    def test_archived_reason_skips_even_with_message_id(self):
        """Archived reason should skip regardless of message_id."""
        hook, add_reaction = self._make_hook("msg_abc")
        state = MagicMock()
        hook.on_terminal(state, "archived")
        add_reaction.assert_not_called()

    def test_on_dispatched_is_noop(self):
        """on_dispatched should do nothing regardless of message_id."""
        hook, add_reaction = self._make_hook("")
        event = MagicMock()
        state = MagicMock()
        hook.on_dispatched(event, state)
        add_reaction.assert_not_called()


if __name__ == "__main__":
    unittest.main()
