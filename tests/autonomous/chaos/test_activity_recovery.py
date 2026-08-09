"""Chaos test: activity recovery after crash with checkpoints."""

from __future__ import annotations

import pytest

from src.autonomous.scheduler.activities import (
    ActivityExecutor,
    ActivityState,
    ActivityType,
    StaleLease,
)


class FakeJournal:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def write_event(self, event_type: str, payload: dict) -> None:
        self.events.append((event_type, payload))


@pytest.mark.asyncio
async def test_checkpointed_activity_resumable_after_restart() -> None:
    """Activity with checkpoint can be found and resumed after restart."""
    journal = FakeJournal()
    executor = ActivityExecutor(journal=journal)

    activity = await executor.create(
        activity_type=ActivityType.EXECUTION,
        run_id="run_1",
        step_id="step_1",
        fencing_token=5,
    )
    await executor.start(activity.activity_id, fencing_token=5)
    await executor.checkpoint(
        activity.activity_id,
        fencing_token=5,
        data={"files_processed": 42, "cursor": "abc123"},
        blob_ref="blob_snapshot_001",
    )

    # Simulate restart: executor discovers checkpointed activities
    resumable = executor.list_resumable()
    assert len(resumable) == 1
    assert resumable[0].activity_id == activity.activity_id
    assert resumable[0].last_checkpoint is not None
    assert resumable[0].last_checkpoint.data == {"files_processed": 42, "cursor": "abc123"}
    assert resumable[0].last_checkpoint.blob_ref == "blob_snapshot_001"

    # Resume by starting again from checkpointed state
    resumed = await executor.start(activity.activity_id, fencing_token=5)
    assert resumed.state is ActivityState.RUNNING


@pytest.mark.asyncio
async def test_multiple_checkpoints_keep_latest() -> None:
    """Only the last checkpoint is retained."""
    journal = FakeJournal()
    executor = ActivityExecutor(journal=journal)

    activity = await executor.create(
        activity_type=ActivityType.PLANNING,
        run_id="run_1",
        fencing_token=1,
    )
    await executor.start(activity.activity_id, fencing_token=1)

    await executor.checkpoint(
        activity.activity_id, fencing_token=1, data={"step": 1}
    )
    # Must restart to checkpoint again
    await executor.start(activity.activity_id, fencing_token=1)
    await executor.checkpoint(
        activity.activity_id, fencing_token=1, data={"step": 2}
    )

    assert activity.last_checkpoint is not None
    assert activity.last_checkpoint.data == {"step": 2}


@pytest.mark.asyncio
async def test_crash_during_activity_leaves_stale_for_reconciler() -> None:
    """If activity was running and system crashes, it remains in RUNNING
    state for the reconciler to pick up."""
    journal = FakeJournal()
    executor = ActivityExecutor(journal=journal)

    activity = await executor.create(
        activity_type=ActivityType.VERIFICATION,
        run_id="run_1",
        fencing_token=1,
    )
    await executor.start(activity.activity_id, fencing_token=1)

    # Simulate crash: activity is still RUNNING
    active = executor.list_active("run_1")
    assert len(active) == 1
    assert active[0].state is ActivityState.RUNNING

    # Reconciler can timeout stale activities
    timed_out = await executor.timeout(activity.activity_id)
    assert timed_out.state is ActivityState.TIMED_OUT


@pytest.mark.asyncio
async def test_heartbeat_validates_fencing_token() -> None:
    """Heartbeat from a stale worker is rejected."""
    journal = FakeJournal()
    executor = ActivityExecutor(journal=journal)

    activity = await executor.create(
        activity_type=ActivityType.EXECUTION,
        run_id="run_1",
        fencing_token=10,
    )
    await executor.start(activity.activity_id, fencing_token=10)

    # Valid heartbeat
    await executor.heartbeat(activity.activity_id, fencing_token=10)

    # Stale heartbeat rejected
    with pytest.raises(StaleLease):
        await executor.heartbeat(activity.activity_id, fencing_token=5)
