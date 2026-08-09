"""Unit tests for replay projections."""

from __future__ import annotations

import pytest

from src.autonomous.domain.enums import (
    EffectState,
    GoalState,
    RunState,
    StepState,
)
from src.autonomous.journal import JournalWriter, MemoryAnchor
from src.autonomous.journal.frame import JournalEvent
from src.autonomous.journal.projections import (
    InboxRecord,
    ProjectionError,
    ProjectionRepository,
    ProjectionState,
)

HMAC_KEY = b"test-projections-key-at-least-32-bytes!!"


def _make_writer(tmp_path):
    """Create a JournalWriter for testing."""
    anchor = MemoryAnchor()
    return JournalWriter.open(
        tmp_path,
        anchor=anchor,
        hmac_key=HMAC_KEY,
        writer_epoch=1,
    )


def _commit_events(writer, events):
    """Helper to commit events and return the frame."""
    aggregate_ids = {e.aggregate_id for e in events}
    expected = {
        agg_id: writer._aggregate_versions.get(agg_id, 0)
        for agg_id in aggregate_ids
    }
    result = writer.commit(tuple(events), expected)
    return result.frame


class TestProjectionState:
    def test_empty_state(self):
        state = ProjectionState()
        assert state.goals == {}
        assert state.runs == {}
        assert state.plans == {}
        assert state.steps == {}
        assert state.effects == {}
        assert state.inbox == {}
        assert state.dedup_keys == set()
        assert state.occurrence_keys == set()
        assert state.cursor_sequence == 0

    def test_pending_inbox_filters_processed_and_tombstoned(self):
        state = ProjectionState()
        state.inbox["e1"] = InboxRecord(
            event_id="e1", dedup_key="k1", source_type="msg",
            payload={}, received_at=1.0, processed=False,
        )
        state.inbox["e2"] = InboxRecord(
            event_id="e2", dedup_key="k2", source_type="msg",
            payload={}, received_at=2.0, processed=True,
        )
        state.inbox["e3"] = InboxRecord(
            event_id="e3", dedup_key="k3", source_type="msg",
            payload={}, received_at=3.0, tombstone=True,
        )
        pending = state.pending_inbox()
        assert len(pending) == 1
        assert pending[0].event_id == "e1"


class TestGoalProjection:
    def test_goal_created_and_state_changed(self, tmp_path):
        writer = _make_writer(tmp_path)
        repo = ProjectionRepository()

        # Create goal
        goal_event = JournalEvent(
            event_type="goal.created",
            aggregate_id="goal_1",
            payload={
                "goal_id": "goal_1",
                "tenant_key": "t1",
                "owner_id": "user1",
                "goal_type": "one_shot",
                "state": "draft",
                "spec": {"objective": "test goal"},
                "epochs": {"definition_version": 1, "admission_epoch": 1},
            },
        )
        frame = _commit_events(writer, [goal_event])
        repo.apply(frame)

        goal = repo.goal("goal_1")
        assert goal is not None
        assert goal.goal_id == "goal_1"
        assert goal.state == GoalState.DRAFT
        assert goal.tenant_key == "t1"

        # State change
        change_event = JournalEvent(
            event_type="goal.state_changed",
            aggregate_id="goal_1",
            payload={"state": "active"},
        )
        frame2 = _commit_events(writer, [change_event])
        repo.apply(frame2)

        goal = repo.goal("goal_1")
        assert goal.state == GoalState.ACTIVE

    def test_goal_activated(self, tmp_path):
        writer = _make_writer(tmp_path)
        repo = ProjectionRepository()

        goal_event = JournalEvent(
            event_type="goal.created",
            aggregate_id="goal_2",
            payload={
                "goal_id": "goal_2",
                "goal_type": "one_shot",
                "state": "draft",
            },
        )
        frame = _commit_events(writer, [goal_event])
        repo.apply(frame)

        activate_event = JournalEvent(
            event_type="goal.activated",
            aggregate_id="goal_2",
            payload={},
        )
        frame2 = _commit_events(writer, [activate_event])
        repo.apply(frame2)

        goal = repo.goal("goal_2")
        assert goal.state == GoalState.ACTIVE


class TestRunProjection:
    def test_run_created_and_state_changed(self, tmp_path):
        writer = _make_writer(tmp_path)
        repo = ProjectionRepository()

        run_event = JournalEvent(
            event_type="run.created",
            aggregate_id="run_1",
            payload={
                "run_id": "run_1",
                "goal_id": "goal_1",
                "goal_version": 1,
                "state": "queued",
                "occurrence_key": "occ_1",
            },
        )
        frame = _commit_events(writer, [run_event])
        repo.apply(frame)

        run = repo.run("run_1")
        assert run is not None
        assert run.run_id == "run_1"
        assert run.goal_id == "goal_1"
        assert run.state == RunState.QUEUED

        # State change
        change_event = JournalEvent(
            event_type="run.state_changed",
            aggregate_id="run_1",
            payload={"state": "executing"},
        )
        frame2 = _commit_events(writer, [change_event])
        repo.apply(frame2)

        run = repo.run("run_1")
        assert run.state == RunState.EXECUTING


class TestPlanProjection:
    def test_plan_created_indexes_steps(self, tmp_path):
        writer = _make_writer(tmp_path)
        repo = ProjectionRepository()

        plan_event = JournalEvent(
            event_type="plan.created",
            aggregate_id="plan_1",
            payload={
                "plan_id": "plan_1",
                "run_id": "run_1",
                "state": "draft",
                "epoch": 1,
                "steps": [
                    {"step_id": "step_1", "name": "first", "state": "pending"},
                    {"step_id": "step_2", "name": "second", "state": "pending",
                     "depends_on": ["step_1"]},
                ],
            },
        )
        frame = _commit_events(writer, [plan_event])
        repo.apply(frame)

        plan = repo.plan("plan_1")
        assert plan is not None
        assert len(plan.steps) == 2

        step1 = repo.step("step_1")
        assert step1 is not None
        assert step1.name == "first"

        step2 = repo.step("step_2")
        assert step2 is not None
        assert step2.depends_on == ("step_1",)

    def test_step_state_changed(self, tmp_path):
        writer = _make_writer(tmp_path)
        repo = ProjectionRepository()

        plan_event = JournalEvent(
            event_type="plan.created",
            aggregate_id="plan_2",
            payload={
                "plan_id": "plan_2",
                "run_id": "run_2",
                "state": "active",
                "epoch": 1,
                "steps": [
                    {"step_id": "step_a", "name": "alpha", "state": "pending"},
                ],
            },
        )
        frame = _commit_events(writer, [plan_event])
        repo.apply(frame)

        change_event = JournalEvent(
            event_type="step.state_changed",
            aggregate_id="step_a",
            payload={"state": "running"},
        )
        frame2 = _commit_events(writer, [change_event])
        repo.apply(frame2)

        step = repo.step("step_a")
        assert step.state == StepState.RUNNING


class TestEffectProjection:
    def test_effect_created_and_state_changed(self, tmp_path):
        writer = _make_writer(tmp_path)
        repo = ProjectionRepository()

        effect_event = JournalEvent(
            event_type="effect.created",
            aggregate_id="eff_1",
            payload={
                "effect_id": "eff_1",
                "state": "proposed",
                "risk_level": "r0",
                "capability": "shell.exec",
            },
        )
        frame = _commit_events(writer, [effect_event])
        repo.apply(frame)

        effect = repo.effect("eff_1")
        assert effect is not None
        assert effect.state == EffectState.PROPOSED

        change_event = JournalEvent(
            event_type="effect.state_changed",
            aggregate_id="eff_1",
            payload={"state": "committed"},
        )
        frame2 = _commit_events(writer, [change_event])
        repo.apply(frame2)

        effect = repo.effect("eff_1")
        assert effect.state == EffectState.COMMITTED


class TestInboxProjection:
    def test_inbox_event_received(self, tmp_path):
        writer = _make_writer(tmp_path)
        repo = ProjectionRepository()

        inbox_event = JournalEvent(
            event_type="inbox.event_received",
            aggregate_id="evt_1",
            payload={
                "event_id": "evt_1",
                "dedup_key": "abc123",
                "source_type": "user_message",
                "payload": {"text": "hello"},
            },
        )
        frame = _commit_events(writer, [inbox_event])
        repo.apply(frame)

        record = repo.inbox("evt_1")
        assert record is not None
        assert record.dedup_key == "abc123"
        assert not record.processed
        assert "abc123" in repo.state.dedup_keys

    def test_inbox_event_processed(self, tmp_path):
        writer = _make_writer(tmp_path)
        repo = ProjectionRepository()

        # First receive
        receive_event = JournalEvent(
            event_type="inbox.event_received",
            aggregate_id="evt_2",
            payload={
                "event_id": "evt_2",
                "dedup_key": "def456",
                "source_type": "card_callback",
                "payload": {},
            },
        )
        frame = _commit_events(writer, [receive_event])
        repo.apply(frame)

        # Then process
        process_event = JournalEvent(
            event_type="inbox.event_processed",
            aggregate_id="evt_2",
            payload={
                "event_id": "evt_2",
                "goal_id": "goal_x",
                "run_id": "run_x",
            },
        )
        frame2 = _commit_events(writer, [process_event])
        repo.apply(frame2)

        record = repo.inbox("evt_2")
        assert record.processed is True
        assert record.goal_id == "goal_x"
        assert record.run_id == "run_x"

    def test_inbox_tombstone(self, tmp_path):
        writer = _make_writer(tmp_path)
        repo = ProjectionRepository()

        receive_event = JournalEvent(
            event_type="inbox.event_received",
            aggregate_id="evt_3",
            payload={
                "event_id": "evt_3",
                "dedup_key": "ghi789",
                "source_type": "msg",
                "payload": {},
            },
        )
        frame = _commit_events(writer, [receive_event])
        repo.apply(frame)

        tombstone_event = JournalEvent(
            event_type="inbox.tombstone",
            aggregate_id="evt_3",
            payload={"event_id": "evt_3"},
        )
        frame2 = _commit_events(writer, [tombstone_event])
        repo.apply(frame2)

        record = repo.inbox("evt_3")
        assert record.tombstone is True
        assert len(repo.state.pending_inbox()) == 0


class TestOccurrenceAndCursorProjection:
    def test_occurrence_consumed(self, tmp_path):
        writer = _make_writer(tmp_path)
        repo = ProjectionRepository()

        event = JournalEvent(
            event_type="occurrence.consumed",
            aggregate_id="goal_1",
            payload={"occurrence_key": "occ_abc"},
        )
        frame = _commit_events(writer, [event])
        repo.apply(frame)

        assert "occ_abc" in repo.state.occurrence_keys

    def test_cursor_advanced(self, tmp_path):
        writer = _make_writer(tmp_path)
        repo = ProjectionRepository()

        event = JournalEvent(
            event_type="cursor.advanced",
            aggregate_id="cursor_1",
            payload={"sequence": 42, "hash": "abc"},
        )
        frame = _commit_events(writer, [event])
        repo.apply(frame)

        # cursor_sequence is updated by apply_frame to the frame sequence
        # The explicit cursor.advanced event also sets it
        assert repo.state.cursor_sequence == frame.sequence


class TestProjectionRepository:
    def test_rebuild_replays_all_frames(self, tmp_path):
        writer = _make_writer(tmp_path)

        # Commit multiple frames
        e1 = JournalEvent(
            event_type="goal.created",
            aggregate_id="g1",
            payload={"goal_id": "g1", "state": "draft", "goal_type": "one_shot"},
        )
        _commit_events(writer, [e1])

        e2 = JournalEvent(
            event_type="goal.state_changed",
            aggregate_id="g1",
            payload={"state": "active"},
        )
        _commit_events(writer, [e2])

        e3 = JournalEvent(
            event_type="run.created",
            aggregate_id="r1",
            payload={"run_id": "r1", "goal_id": "g1", "state": "queued"},
        )
        _commit_events(writer, [e3])

        # Rebuild from replay
        repo = ProjectionRepository()
        state = repo.rebuild(writer.replay())

        assert "g1" in state.goals
        assert state.goals["g1"].state == GoalState.ACTIVE
        assert "r1" in state.runs
        assert state.runs["r1"].goal_id == "g1"

    def test_unknown_event_types_are_skipped(self, tmp_path):
        writer = _make_writer(tmp_path)
        repo = ProjectionRepository()

        unknown_event = JournalEvent(
            event_type="future.unknown_event",
            aggregate_id="x1",
            payload={"data": "value"},
        )
        frame = _commit_events(writer, [unknown_event])
        # Should not raise
        repo.apply(frame)
        assert repo.state.cursor_sequence == frame.sequence

    def test_state_changed_for_unknown_aggregate_raises(self, tmp_path):
        writer = _make_writer(tmp_path)
        repo = ProjectionRepository()

        bad_event = JournalEvent(
            event_type="goal.state_changed",
            aggregate_id="nonexistent",
            payload={"state": "active"},
        )
        frame = _commit_events(writer, [bad_event])
        with pytest.raises(ProjectionError, match="unknown goal"):
            repo.apply(frame)
