"""Integration test: inbox survives restart by replaying the journal."""

from __future__ import annotations

from src.autonomous.journal import JournalWriter, MemoryAnchor
from src.autonomous.journal.projections import ProjectionRepository, ProjectionState
from src.autonomous.manager.admission import (
    Admission,
    DurableInbox,
    InboxEvent,
)

HMAC_KEY = b"test-inbox-recovery-key-at-least-32-bytes!"


def _make_writer(base_dir, anchor):
    return JournalWriter.open(
        base_dir,
        anchor=anchor,
        hmac_key=HMAC_KEY,
        writer_epoch=1,
    )


class TestInboxSurvivesRestart:
    """The inbox state must survive a writer close + reopen cycle."""

    def test_inbox_events_survive_restart(self, tmp_path):
        anchor = MemoryAnchor()

        # Session 1: accept events
        writer = _make_writer(tmp_path, anchor)
        state = ProjectionState()
        inbox = DurableInbox(writer, state)

        e1 = InboxEvent(
            tenant="t1", chat_id="c1", message_id="m1", source_type="msg",
            payload={"text": "task one"},
        )
        e2 = InboxEvent(
            tenant="t1", chat_id="c1", message_id="m2", source_type="msg",
            payload={"text": "task two"},
        )
        eid1 = inbox.accept(e1)
        eid2 = inbox.accept(e2)
        assert eid1 is not None
        assert eid2 is not None

        writer.close()

        # Session 2: reopen and rebuild from journal
        writer2 = _make_writer(tmp_path, anchor)
        repo = ProjectionRepository()
        rebuilt_state = repo.rebuild(writer2.replay())

        # Inbox events must be present
        assert eid1 in rebuilt_state.inbox
        assert eid2 in rebuilt_state.inbox
        assert len(rebuilt_state.pending_inbox()) == 2

        # Dedup must still work
        inbox2 = DurableInbox(writer2, rebuilt_state)
        dup_event = InboxEvent(
            tenant="t1", chat_id="c1", message_id="m1", source_type="msg",
        )
        assert inbox2.accept(dup_event) is None

        writer2.close()

    def test_processed_events_survive_restart(self, tmp_path):
        anchor = MemoryAnchor()

        # Session 1: accept and process
        writer = _make_writer(tmp_path, anchor)
        state = ProjectionState()
        inbox = DurableInbox(writer, state)

        event = InboxEvent(
            tenant="t1", chat_id="c1", message_id="m1", source_type="msg",
        )
        eid = inbox.accept(event)
        inbox.consume_trigger(
            eid,
            goal_id="g1",
            run_id="r1",
            occurrence_key="occ_1",
            cursor_sequence=1,
        )
        writer.close()

        # Session 2: rebuild
        writer2 = _make_writer(tmp_path, anchor)
        repo = ProjectionRepository()
        rebuilt_state = repo.rebuild(writer2.replay())

        record = rebuilt_state.inbox[eid]
        assert record.processed is True
        assert record.goal_id == "g1"
        assert record.run_id == "r1"
        assert "occ_1" in rebuilt_state.occurrence_keys
        assert len(rebuilt_state.pending_inbox()) == 0

        writer2.close()

    def test_admission_one_shot_survives_restart(self, tmp_path):
        anchor = MemoryAnchor()

        # Session 1: create one-shot from event
        writer = _make_writer(tmp_path, anchor)
        state = ProjectionState()
        admission = Admission(writer, state)

        event = InboxEvent(
            tenant="t1", chat_id="c1", message_id="m1", source_type="msg",
            payload={"text": "build feature X"},
        )
        admission.admit_event(event)
        goal_id, run_id = admission.create_one_shot_from_event(
            event.event_id,
            tenant="t1",
            objective="Build feature X",
            owner_id="user1",
        )
        writer.close()

        # Session 2: rebuild and verify
        writer2 = _make_writer(tmp_path, anchor)
        repo = ProjectionRepository()
        rebuilt_state = repo.rebuild(writer2.replay())

        assert goal_id in rebuilt_state.goals
        assert run_id in rebuilt_state.runs
        assert rebuilt_state.goals[goal_id].state.value == "active"
        assert rebuilt_state.runs[run_id].goal_id == goal_id

        # Inbox event should be marked as processed
        record = rebuilt_state.inbox[event.event_id]
        assert record.processed is True

        writer2.close()

    def test_dedup_survives_restart(self, tmp_path):
        """Dedup keys persist across restarts."""
        anchor = MemoryAnchor()

        # Session 1: accept event
        writer = _make_writer(tmp_path, anchor)
        state = ProjectionState()
        inbox = DurableInbox(writer, state)

        event = InboxEvent(
            tenant="t1", chat_id="c1", message_id="m_dedup", source_type="msg",
        )
        inbox.accept(event)
        writer.close()

        # Session 2: same event should be deduplicated
        writer2 = _make_writer(tmp_path, anchor)
        repo = ProjectionRepository()
        rebuilt_state = repo.rebuild(writer2.replay())
        inbox2 = DurableInbox(writer2, rebuilt_state)

        same_event = InboxEvent(
            tenant="t1", chat_id="c1", message_id="m_dedup", source_type="msg",
        )
        assert inbox2.is_duplicate(same_event) is True
        assert inbox2.accept(same_event) is None

        writer2.close()

    def test_multiple_restart_cycles(self, tmp_path):
        """State accumulates correctly across multiple restart cycles."""
        anchor = MemoryAnchor()

        # Cycle 1: accept first event
        writer = _make_writer(tmp_path, anchor)
        state = ProjectionState()
        inbox = DurableInbox(writer, state)
        e1 = InboxEvent(tenant="t", chat_id="c", message_id="1", source_type="m")
        eid1 = inbox.accept(e1)
        writer.close()

        # Cycle 2: accept second event
        writer = _make_writer(tmp_path, anchor)
        repo = ProjectionRepository()
        state = repo.rebuild(writer.replay())
        inbox = DurableInbox(writer, state)
        e2 = InboxEvent(tenant="t", chat_id="c", message_id="2", source_type="m")
        eid2 = inbox.accept(e2)
        writer.close()

        # Cycle 3: accept third event
        writer = _make_writer(tmp_path, anchor)
        repo = ProjectionRepository()
        state = repo.rebuild(writer.replay())
        inbox = DurableInbox(writer, state)
        e3 = InboxEvent(tenant="t", chat_id="c", message_id="3", source_type="m")
        eid3 = inbox.accept(e3)
        writer.close()

        # Final verify: all three events present
        writer = _make_writer(tmp_path, anchor)
        repo = ProjectionRepository()
        final_state = repo.rebuild(writer.replay())
        assert eid1 in final_state.inbox
        assert eid2 in final_state.inbox
        assert eid3 in final_state.inbox
        assert len(final_state.pending_inbox()) == 3
        writer.close()
