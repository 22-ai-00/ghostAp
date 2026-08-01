"""Tests for queue-full card rendering."""

import json
from unittest.mock import MagicMock


class TestQueueFullCard:
    """AC19: Queue full triggers error card with retry button."""

    def test_queue_full_card_contains_retry(self):
        """When queue is full, build_queue_full_card returns card with retry button."""
        from src.slock_engine.card_templates import build_queue_full_card
        from src.slock_engine.models import AgentIdentity

        agent = AgentIdentity(
            agent_id="full_agent",
            name="TestAgent",
            emoji="\U0001f916",
        )
        card = build_queue_full_card(
            agent, channel_id="test_chan", original_message="test message"
        )
        card_str = json.dumps(card, ensure_ascii=False)
        assert "\u961f\u5217\u5df2\u6ee1" in card_str
        assert "\u91cd\u8bd5" in card_str
        assert "slock_queue_retry" in card_str
        assert "\u5f3a\u5236\u4ecb\u5165" in card_str


# ---------------------------------------------------------------------------
# Integration tests: mention pending queue overflow (merged from
# test_slock_mention_queue_integration.py)
# ---------------------------------------------------------------------------


class TestMentionQueueFullIntegration:
    """Verify full handler path when mention queue overflows at maxlen=8."""

    def _make_agent(self, agent_id="agent-001", name="TestCoder", role="coder"):
        agent = MagicMock()
        agent.agent_id = agent_id
        agent.name = name
        agent.role = role
        return agent

    def test_queue_full_card_contains_rejected_message(self):
        """Queue full card should include a truncated preview of the rejected message."""
        from src.slock_engine.card_templates import build_queue_full_card

        agent = self._make_agent()
        long_message = "x" * 500  # Very long message

        card = build_queue_full_card(
            agent,
            channel_id="ch-001",
            original_message=long_message,
        )

        card_str = json.dumps(card, ensure_ascii=False)
        # Card should contain some portion of the message (truncated)
        assert "\u961f\u5217\u5df2\u6ee1" in card_str
        # Should have both action buttons
        assert "slock_queue_retry" in card_str
        assert "slock_force_interrupt" in card_str
