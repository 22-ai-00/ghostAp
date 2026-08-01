"""Security tests Wave 7 — audit log and redaction.

Tests AC15 and AC16.
"""

import json
from unittest.mock import MagicMock

from src.slock_engine.models import (
    DiscussionConfig,
    DiscussionMessage,
    DiscussionThread,
)
from src.slock_engine.task_router import TaskClaim


class TestAC15AuditLog:
    """AC15: force_assign writes audit entry to SHARED_MEMORY.md."""

    def test_force_assign_writes_audit_log(self, tmp_path):
        """force_assign with operator_id triggers audit log write containing
        operator_id, target, action, and detail with override info."""
        mm = MagicMock()
        claim = TaskClaim(memory_manager=mm)

        # Pre-claim by another agent
        claim.claim("task-42", "agent-old")

        # Force assign with operator
        claim.force_assign(task_id="task-42", agent_id="agent-new", operator_id="admin-op-1")

        # Verify append_audit_log was called with correct signature
        mm.append_audit_log.assert_called_once_with(
            operator_id="admin-op-1",
            action="force_assign",
            target="task-42",
            detail="prev=agent-old new=agent-new",
        )


class TestAC16Redaction:
    """AC16: build_discussion_card_from_thread redacts sensitive content."""

    def test_sensitive_api_key_redacted_from_card(self):
        """API keys in message content are redacted before card rendering."""
        from src.slock_engine.card_templates import build_discussion_card_from_thread

        # Create a thread with a message containing a sensitive API key
        secret_content = "Here is the key: API_KEY=sk-abc123fakekey please use it"
        thread = DiscussionThread(
            thread_id="thread-redact-test",
            channel_id="ch-001",
            participants=["agent-a", "agent-b"],
            messages=[
                DiscussionMessage(
                    sender_agent_id="agent-a",
                    content=secret_content,
                    round_num=1,
                ),
            ],
            config=DiscussionConfig(max_rounds=3),
            trigger_reason="test redaction",
        )

        card = build_discussion_card_from_thread(thread, engine=None)
        card_json = json.dumps(card, ensure_ascii=False)

        # The raw secret value must NOT appear in the rendered card
        assert "sk-abc123fakekey" not in card_json, (
            "Sensitive API key value must be redacted from card output"
        )
        # The redaction marker should be present
        assert "<redacted>" in card_json or "redacted" in card_json.lower()
