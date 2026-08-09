"""Unit tests for the DurableInbox and Admission."""

from __future__ import annotations

import pytest

from src.autonomous.journal import JournalWriter, MemoryAnchor
from src.autonomous.journal.projections import ProjectionState
from src.autonomous.manager.admission import (
    Admission,
    AdmissionResult,
    DurableInbox,
    InboxEvent,
)

HMAC_KEY = b"test-inbox-key-at-least-32-bytes-long!!"


def _make_writer(tmp_path):
    anchor = MemoryAnchor()
    return JournalWriter.open(
        tmp_path,
        anchor=anchor,
        hmac_key=HMAC_KEY,
        writer_epoch=1,
    )


class TestDurableInbox:
    def test_accept_returns_event_id(self, tmp_path):
        writer = _make_writer(tmp_path)
        state = ProjectionState()
        inbox = DurableInbox(writer, state)

        event = InboxEvent(
            tenant="t1",
            chat_id="chat_1",
            message_id="msg_1",
            source_type="user_message",
            payload={"text": "hello"},
        )
        result = inbox.accept(event)
        assert result == event.event_id

    def test_accept_dedup_returns_none(self, tmp_path):
        writer = _make_writer(tmp_path)
        state = ProjectionState()
        inbox = DurableInbox(writer, state)

        event = InboxEvent(
            tenant="t1",
            chat_id="chat_1",
            message_id="msg_1",
            source_type="user_message",
        )
        # First accept
        result1 = inbox.accept(event)
        assert result1 is not None

        # Duplicate (same dedup key components)
        event2 = InboxEvent(
            tenant="t1",
            chat_id="chat_1",
            message_id="msg_1",
            source_type="user_message",
        )
        result2 = inbox.accept(event2)
        assert result2 is None

    def test_accept_different_source_type_not_dedup(self, tmp_path):
        writer = _make_writer(tmp_path)
        state = ProjectionState()
        inbox = DurableInbox(writer, state)

        event1 = InboxEvent(
            tenant="t1",
            chat_id="chat_1",
            message_id="msg_1",
            source_type="user_message",
        )
        event2 = InboxEvent(
            tenant="t1",
            chat_id="chat_1",
            message_id="msg_1",
            source_type="card_callback",
        )
        r1 = inbox.accept(event1)
        r2 = inbox.accept(event2)
        assert r1 is not None
        assert r2 is not None
        assert r1 != r2

    def test_is_duplicate(self, tmp_path):
        writer = _make_writer(tmp_path)
        state = ProjectionState()
        inbox = DurableInbox(writer, state)

        event = InboxEvent(
            tenant="t1",
            chat_id="c1",
            message_id="m1",
            source_type="msg",
        )
        assert inbox.is_duplicate(event) is False
        inbox.accept(event)
        assert inbox.is_duplicate(event) is True

    def test_pending_events(self, tmp_path):
        writer = _make_writer(tmp_path)
        state = ProjectionState()
        inbox = DurableInbox(writer, state)

        e1 = InboxEvent(tenant="t1", chat_id="c1", message_id="m1", source_type="msg")
        e2 = InboxEvent(tenant="t1", chat_id="c1", message_id="m2", source_type="msg")
        inbox.accept(e1)
        inbox.accept(e2)

        pending = inbox.pending_events()
        assert len(pending) == 2

    def test_consume_trigger_atomically(self, tmp_path):
        writer = _make_writer(tmp_path)
        state = ProjectionState()
        inbox = DurableInbox(writer, state)

        event = InboxEvent(
            tenant="t1",
            chat_id="c1",
            message_id="m1",
            source_type="user_message",
        )
        event_id = inbox.accept(event)

        # Consume trigger atomically
        inbox.consume_trigger(
            event_id,
            goal_id="goal_1",
            run_id="run_1",
            occurrence_key="occ_1",
            cursor_sequence=1,
            cursor_hash="abc",
        )

        # Verify all effects
        record = state.inbox[event_id]
        assert record.processed is True
        assert record.goal_id == "goal_1"
        assert record.run_id == "run_1"
        assert "run_1" in state.runs
        assert "occ_1" in state.occurrence_keys
        assert len(inbox.pending_events()) == 0

    def test_consume_trigger_unknown_event_raises(self, tmp_path):
        writer = _make_writer(tmp_path)
        state = ProjectionState()
        inbox = DurableInbox(writer, state)

        with pytest.raises(ValueError, match="not found"):
            inbox.consume_trigger(
                "nonexistent",
                goal_id="g1",
                run_id="r1",
                occurrence_key="o1",
                cursor_sequence=1,
            )

    def test_consume_trigger_already_processed_raises(self, tmp_path):
        writer = _make_writer(tmp_path)
        state = ProjectionState()
        inbox = DurableInbox(writer, state)

        event = InboxEvent(
            tenant="t1", chat_id="c1", message_id="m1", source_type="msg"
        )
        event_id = inbox.accept(event)
        inbox.consume_trigger(
            event_id,
            goal_id="g1",
            run_id="r1",
            occurrence_key="o1",
            cursor_sequence=1,
        )

        with pytest.raises(ValueError, match="already processed"):
            inbox.consume_trigger(
                event_id,
                goal_id="g2",
                run_id="r2",
                occurrence_key="o2",
                cursor_sequence=2,
            )


class TestAdmission:
    def test_admit_event_accepted(self, tmp_path):
        writer = _make_writer(tmp_path)
        state = ProjectionState()
        admission = Admission(writer, state)

        event = InboxEvent(
            tenant="t1",
            chat_id="c1",
            message_id="m1",
            source_type="user_message",
            payload={"text": "build a feature"},
        )
        decision = admission.admit_event(event)
        assert decision.result == AdmissionResult.ACCEPTED
        assert decision.event_id == event.event_id

    def test_admit_event_duplicate(self, tmp_path):
        writer = _make_writer(tmp_path)
        state = ProjectionState()
        admission = Admission(writer, state)

        event = InboxEvent(
            tenant="t1", chat_id="c1", message_id="m1", source_type="msg"
        )
        admission.admit_event(event)

        event2 = InboxEvent(
            tenant="t1", chat_id="c1", message_id="m1", source_type="msg"
        )
        decision = admission.admit_event(event2)
        assert decision.result == AdmissionResult.DUPLICATE

    def test_create_one_shot_from_event(self, tmp_path):
        writer = _make_writer(tmp_path)
        state = ProjectionState()
        admission = Admission(writer, state)

        event = InboxEvent(
            tenant="t1",
            chat_id="c1",
            message_id="m1",
            source_type="user_message",
            payload={"text": "do something"},
        )
        admission.admit_event(event)

        goal_id, run_id = admission.create_one_shot_from_event(
            event.event_id,
            tenant="t1",
            objective="Test objective",
            owner_id="user1",
        )

        assert goal_id is not None
        assert run_id is not None

        # Verify goal and run exist in state
        goal = admission.get_goal(goal_id)
        assert goal is not None
        assert goal.goal_type == "one_shot"
        assert goal.state == "active"

        run = admission.get_run(run_id)
        assert run is not None
        assert run.goal_id == goal_id
        assert run.state == "queued"

        # Verify inbox event is processed
        record = state.inbox[event.event_id]
        assert record.processed is True
        assert record.goal_id == goal_id
        assert record.run_id == run_id

    def test_create_one_shot_unknown_event_raises(self, tmp_path):
        writer = _make_writer(tmp_path)
        state = ProjectionState()
        admission = Admission(writer, state)

        with pytest.raises(ValueError, match="not found"):
            admission.create_one_shot_from_event("no_such_event")

    def test_create_one_shot_already_processed_raises(self, tmp_path):
        writer = _make_writer(tmp_path)
        state = ProjectionState()
        admission = Admission(writer, state)

        event = InboxEvent(
            tenant="t1", chat_id="c1", message_id="m1", source_type="msg"
        )
        admission.admit_event(event)
        admission.create_one_shot_from_event(event.event_id)

        with pytest.raises(ValueError, match="already processed"):
            admission.create_one_shot_from_event(event.event_id)

    def test_list_goals_and_runs(self, tmp_path):
        writer = _make_writer(tmp_path)
        state = ProjectionState()
        admission = Admission(writer, state)

        e1 = InboxEvent(tenant="t1", chat_id="c1", message_id="m1", source_type="msg")
        e2 = InboxEvent(tenant="t1", chat_id="c1", message_id="m2", source_type="msg")
        admission.admit_event(e1)
        admission.admit_event(e2)
        admission.create_one_shot_from_event(e1.event_id)
        admission.create_one_shot_from_event(e2.event_id)

        goals = admission.list_goals()
        assert len(goals) == 2

        runs = admission.list_runs()
        assert len(runs) == 2


class TestInboxEventDedup:
    """Verify dedup key composition: tenant + chat_id + message_id + source_type."""

    def test_dedup_key_deterministic(self):
        e1 = InboxEvent(
            tenant="t1", chat_id="c1", message_id="m1", source_type="msg"
        )
        e2 = InboxEvent(
            tenant="t1", chat_id="c1", message_id="m1", source_type="msg"
        )
        assert e1.dedup_key == e2.dedup_key

    def test_dedup_key_changes_with_tenant(self):
        e1 = InboxEvent(tenant="t1", chat_id="c1", message_id="m1", source_type="msg")
        e2 = InboxEvent(tenant="t2", chat_id="c1", message_id="m1", source_type="msg")
        assert e1.dedup_key != e2.dedup_key

    def test_dedup_key_changes_with_chat_id(self):
        e1 = InboxEvent(tenant="t1", chat_id="c1", message_id="m1", source_type="msg")
        e2 = InboxEvent(tenant="t1", chat_id="c2", message_id="m1", source_type="msg")
        assert e1.dedup_key != e2.dedup_key

    def test_dedup_key_changes_with_message_id(self):
        e1 = InboxEvent(tenant="t1", chat_id="c1", message_id="m1", source_type="msg")
        e2 = InboxEvent(tenant="t1", chat_id="c1", message_id="m2", source_type="msg")
        assert e1.dedup_key != e2.dedup_key

    def test_dedup_key_changes_with_source_type(self):
        e1 = InboxEvent(tenant="t1", chat_id="c1", message_id="m1", source_type="msg")
        e2 = InboxEvent(tenant="t1", chat_id="c1", message_id="m1", source_type="cb")
        assert e1.dedup_key != e2.dedup_key
