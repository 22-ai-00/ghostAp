"""Chaos test: 100 replays create one logical event.

Verifies that the dedup mechanism ensures that even if the same logical
event is replayed 100 times (simulating at-least-once delivery), only one
inbox event is ever created in the projection state.
"""

from __future__ import annotations

from src.autonomous.journal import JournalWriter, MemoryAnchor
from src.autonomous.journal.projections import ProjectionRepository, ProjectionState
from src.autonomous.manager.admission import (
    Admission,
    AdmissionResult,
    DurableInbox,
    InboxEvent,
)

HMAC_KEY = b"test-chaos-ingress-key-at-least-32-bytes!"


def _make_writer(base_dir, anchor):
    return JournalWriter.open(
        base_dir,
        anchor=anchor,
        hmac_key=HMAC_KEY,
        writer_epoch=1,
    )


class TestIngressAckBoundaries:
    """100 replays of the same event must produce exactly one logical event."""

    def test_100_replays_one_logical_event(self, tmp_path):
        """Simulate at-least-once delivery: same event delivered 100 times."""
        anchor = MemoryAnchor()
        writer = _make_writer(tmp_path, anchor)
        state = ProjectionState()
        inbox = DurableInbox(writer, state)

        accepted_count = 0
        duplicate_count = 0

        for i in range(100):
            event = InboxEvent(
                tenant="t1",
                chat_id="chat_stress",
                message_id="msg_replay_target",
                source_type="user_message",
                payload={"attempt": i},
            )
            result = inbox.accept(event)
            if result is not None:
                accepted_count += 1
            else:
                duplicate_count += 1

        # Exactly one accepted, 99 duplicates
        assert accepted_count == 1
        assert duplicate_count == 99
        assert len(state.pending_inbox()) == 1

        writer.close()

    def test_100_replays_across_restarts(self, tmp_path):
        """Same event replayed across 100 restart cycles - still one logical event."""
        anchor = MemoryAnchor()
        accepted_count = 0

        for i in range(100):
            writer = _make_writer(tmp_path, anchor)
            repo = ProjectionRepository()
            state = repo.rebuild(writer.replay())
            inbox = DurableInbox(writer, state)

            event = InboxEvent(
                tenant="t1",
                chat_id="chat_restart",
                message_id="msg_persistent",
                source_type="user_message",
                payload={"cycle": i},
            )
            result = inbox.accept(event)
            if result is not None:
                accepted_count += 1
            writer.close()

        # Final verify
        writer = _make_writer(tmp_path, anchor)
        repo = ProjectionRepository()
        final_state = repo.rebuild(writer.replay())
        writer.close()

        assert accepted_count == 1
        assert len(final_state.pending_inbox()) == 1

    def test_100_distinct_events_all_accepted(self, tmp_path):
        """100 distinct events should all be accepted (not deduplicated)."""
        anchor = MemoryAnchor()
        writer = _make_writer(tmp_path, anchor)
        state = ProjectionState()
        inbox = DurableInbox(writer, state)

        for i in range(100):
            event = InboxEvent(
                tenant="t1",
                chat_id="chat_many",
                message_id=f"msg_{i:04d}",
                source_type="user_message",
                payload={"index": i},
            )
            result = inbox.accept(event)
            assert result is not None

        assert len(state.pending_inbox()) == 100
        writer.close()

    def test_100_replays_then_consume_once(self, tmp_path):
        """After 100 replay attempts, the single event can be consumed once."""
        anchor = MemoryAnchor()
        writer = _make_writer(tmp_path, anchor)
        state = ProjectionState()
        inbox = DurableInbox(writer, state)

        event_id = None
        for i in range(100):
            event = InboxEvent(
                tenant="t1",
                chat_id="chat_consume",
                message_id="msg_once",
                source_type="user_message",
                payload={"attempt": i},
            )
            result = inbox.accept(event)
            if result is not None:
                event_id = result

        assert event_id is not None

        # Consume the trigger
        inbox.consume_trigger(
            event_id,
            goal_id="goal_chaos",
            run_id="run_chaos",
            occurrence_key="occ_chaos",
            cursor_sequence=1,
        )

        # Verify consumed
        record = state.inbox[event_id]
        assert record.processed is True
        assert "run_chaos" in state.runs
        assert "occ_chaos" in state.occurrence_keys
        assert len(state.pending_inbox()) == 0

        writer.close()

    def test_100_replays_admission_level(self, tmp_path):
        """Admission-level dedup: 100 identical admit_event calls."""
        anchor = MemoryAnchor()
        writer = _make_writer(tmp_path, anchor)
        state = ProjectionState()
        admission = Admission(writer, state)

        accepted_count = 0
        for i in range(100):
            event = InboxEvent(
                tenant="t1",
                chat_id="chat_admission",
                message_id="msg_admission",
                source_type="user_message",
                payload={"round": i},
            )
            decision = admission.admit_event(event)
            if decision.result == AdmissionResult.ACCEPTED:
                accepted_count += 1

        assert accepted_count == 1
        assert len(state.pending_inbox()) == 1

        writer.close()

    def test_concurrent_style_interleaved_replays(self, tmp_path):
        """Interleave two different events, each replayed 50 times."""
        anchor = MemoryAnchor()
        writer = _make_writer(tmp_path, anchor)
        state = ProjectionState()
        inbox = DurableInbox(writer, state)

        accepted_a = 0
        accepted_b = 0

        for i in range(50):
            # Event A
            ea = InboxEvent(
                tenant="t1",
                chat_id="chat_x",
                message_id="msg_a",
                source_type="user_message",
            )
            if inbox.accept(ea) is not None:
                accepted_a += 1

            # Event B
            eb = InboxEvent(
                tenant="t1",
                chat_id="chat_x",
                message_id="msg_b",
                source_type="user_message",
            )
            if inbox.accept(eb) is not None:
                accepted_b += 1

        assert accepted_a == 1
        assert accepted_b == 1
        assert len(state.pending_inbox()) == 2

        writer.close()
