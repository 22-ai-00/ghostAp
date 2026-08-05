"""Tests for discussion conclusion persistence — AC-R02, AC-R06, AC-R09.

Verifies:
- stop_discussion persists conclusion to L1 memory (AC-R02)
- Final Arbiter is invoked when max rounds exhausted (AC-R06)
- Conclusion notification card is sent after persistence (AC-R09)
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from src.slock_engine.discussion_manager import DiscussionManager
from src.slock_engine.models import (
    DiscussionConfig,
    DiscussionMessage,
    DiscussionStatus,
    DiscussionThread,
)


def _make_thread(
    status: DiscussionStatus = DiscussionStatus.ACTIVE,
    conclusion: str = "",
    participants: list[str] | None = None,
    messages: list[DiscussionMessage] | None = None,
    total_tokens_used: int = 0,
) -> DiscussionThread:
    return DiscussionThread(
        thread_id="thread-001",
        channel_id="ch-001",
        trigger_reason="Test topic",
        participants=participants or ["agent-1", "agent-2"],
        config=DiscussionConfig(),
        status=status,
        conclusion=conclusion,
        messages=messages or [],
        total_tokens_used=total_tokens_used,
    )


def _make_manager() -> DiscussionManager:
    engine = MagicMock()
    engine._memory = MagicMock()
    engine._memory.sync_discussion_conclusion_to_agents = MagicMock()
    engine.channel = MagicMock()
    engine.channel.channel_id = "ch-001"
    memory_mgr = MagicMock()
    memory_mgr.append_discussion_conclusion = MagicMock()
    memory_mgr.sync_discussion_conclusion_to_agents = MagicMock()
    mgr = DiscussionManager(engine=engine, memory_manager=memory_mgr)
    return mgr


class TestStopDiscussionPersistence:
    """AC-R02: stop_discussion 时结论必须被持久化到 L1 memory。"""

    def test_stop_persists_conclusion(self):
        """Stopping a discussion with conclusion persists to agent memory."""
        mgr = _make_manager()
        thread = _make_thread(
            status=DiscussionStatus.ACTIVE,
            conclusion="We agree on approach A.",
        )

        with patch.object(mgr, "_persist_conclusion") as mock_persist, \
             patch.object(mgr, "summarize_conclusion"):
            mgr.stop_discussion(thread)
            mock_persist.assert_called_once()

    def test_stop_sets_stopped_status(self):
        """Stopping sets status to MANUALLY_STOPPED."""
        mgr = _make_manager()
        thread = _make_thread(status=DiscussionStatus.ACTIVE)

        with patch.object(mgr, "_persist_conclusion"), \
             patch.object(mgr, "summarize_conclusion"):
            mgr.stop_discussion(thread)

        assert thread.status == DiscussionStatus.MANUALLY_STOPPED


class TestFinalArbiter:
    """AC-R06: max_rounds 耗尽时 Final Arbiter 强制收敛。"""

    def test_arbiter_fallback_on_low_budget(self):
        """When budget is insufficient, arbiter uses last message as conclusion."""
        mgr = _make_manager()
        thread = _make_thread(
            messages=[
                DiscussionMessage(
                    sender_agent_id="agent-1",
                    content="Final answer: use REST",
                    round_num=1,
                    timestamp=time.time(),
                ),
            ],
            total_tokens_used=9900,  # Nearly exhausted — only 100 remaining
        )
        # Config with low budget so remaining < arbiter_max_tokens
        thread.config = DiscussionConfig(token_budget=10000)

        with patch("src.config.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(slock_arbiter_max_tokens=500)
            mgr._run_final_arbiter(thread)

        # Should fallback to last message since budget (100) < max_tokens (500)
        assert "Final answer: use REST" in (thread.conclusion or "")


class TestConclusionNotification:
    """AC-R09: 结论持久化后发送通知卡片。"""

    def test_send_conclusion_notification_called(self):
        """_persist_conclusion triggers _send_conclusion_notification."""
        mgr = _make_manager()
        thread = _make_thread(
            conclusion="Final conclusion text",
            status=DiscussionStatus.CONVERGED,
        )

        with patch.object(mgr, "_send_conclusion_notification") as mock_notify, \
             patch.object(mgr, "_enrich_conclusion_with_speakers", return_value="Final conclusion text"), \
             patch.object(mgr, "_resolve_agent_display", side_effect=lambda x: x):
            mgr._persist_conclusion(thread)
            mock_notify.assert_called_once()


class TestDiscussionConclusionToKeyKnowledge:
    """AC-D2: Discussion conclusion persists to L2 and L1 key_knowledge."""

    def test_persist_conclusion_calls_l2_append(self):
        """After CONVERGED, conclusion is written to L2 SHARED_MEMORY."""
        mgr = _make_manager()
        thread = _make_thread(
            status=DiscussionStatus.CONVERGED,
            conclusion="Final decision: use approach B.",
            participants=["agent-1", "agent-2"],
        )
        thread.topic = "Architecture decision"

        mgr._persist_conclusion(thread)

        mgr._memory_manager.append_discussion_conclusion.assert_called_once()
        call_args = mgr._memory_manager.append_discussion_conclusion.call_args
        assert "ch-001" in call_args[0] or call_args[0][0] == "ch-001"

    def test_persist_conclusion_syncs_to_agents(self):
        """After CONVERGED, conclusion is synced to participating agents."""
        mgr = _make_manager()
        thread = _make_thread(
            status=DiscussionStatus.CONVERGED,
            conclusion="Agreed: refactor module X.",
            participants=["agent-a", "agent-b"],
        )

        mgr._persist_conclusion(thread)

        mgr._memory_manager.sync_discussion_conclusion_to_agents.assert_called_once()
        call_args = mgr._memory_manager.sync_discussion_conclusion_to_agents.call_args
        agent_ids = call_args[0][0]
        conclusion = call_args[0][1]
        assert "agent-a" in agent_ids
        assert "agent-b" in agent_ids
        assert "refactor module X" in conclusion

    def test_sync_conclusion_writes_to_key_knowledge(self):
        """sync_discussion_conclusion_to_agents calls append_to_agent_key_knowledge."""
        import tempfile

        from src.slock_engine.memory_manager import MemoryManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(base_path=tmpdir)
            # Initialize agent workspace so memory file exists
            mm.initialize_agent_workspace("agent-x")
            mm.initialize_agent_workspace("agent-y")

            mm.sync_discussion_conclusion_to_agents(
                ["agent-x", "agent-y"],
                "We decided to use pattern Z.",
                trigger_reason="code review",
            )

            # Verify key_knowledge was written
            mem_x = mm.read_agent_memory("agent-x")
            mem_y = mm.read_agent_memory("agent-y")
            assert "[DECISION]" in mem_x.key_knowledge
            assert "pattern Z" in mem_x.key_knowledge
            assert "[DECISION]" in mem_y.key_knowledge
            assert "pattern Z" in mem_y.key_knowledge

    def test_persist_conclusion_no_crash_on_empty(self):
        """_persist_conclusion with empty conclusion does not crash."""
        mgr = _make_manager()
        thread = _make_thread(
            status=DiscussionStatus.CONVERGED,
            conclusion="",
            participants=["agent-1"],
        )

        # Should not raise
        mgr._persist_conclusion(thread)
        mgr._memory_manager.append_discussion_conclusion.assert_not_called()
