"""Unit tests for ActivityExecutor."""

from __future__ import annotations

import pytest

from src.autonomous.scheduler.activities import (
    ActivityExecutor,
    ActivityNotFound,
    ActivityState,
    ActivityType,
    StaleLease,
)


class FakeJournal:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def write_event(self, event_type: str, payload: dict) -> None:
        self.events.append((event_type, payload))


@pytest.fixture
def executor() -> ActivityExecutor:
    return ActivityExecutor(journal=FakeJournal())


@pytest.mark.asyncio
async def test_create_and_start(executor: ActivityExecutor) -> None:
    activity = await executor.create(
        activity_type=ActivityType.EXECUTION,
        run_id="run_1",
        step_id="step_1",
        lease_id="lease_1",
        fencing_token=10,
    )
    assert activity.state is ActivityState.PENDING
    assert activity.fencing_token == 10

    started = await executor.start(activity.activity_id, fencing_token=10)
    assert started.state is ActivityState.RUNNING
    assert started.started_at > 0


@pytest.mark.asyncio
async def test_stale_fencing_token_rejected(executor: ActivityExecutor) -> None:
    activity = await executor.create(
        activity_type=ActivityType.EXECUTION,
        run_id="run_1",
        step_id="step_1",
        fencing_token=10,
    )

    with pytest.raises(StaleLease):
        await executor.start(activity.activity_id, fencing_token=5)


@pytest.mark.asyncio
async def test_checkpoint_and_resume(executor: ActivityExecutor) -> None:
    activity = await executor.create(
        activity_type=ActivityType.PLANNING,
        run_id="run_1",
        fencing_token=1,
    )
    await executor.start(activity.activity_id, fencing_token=1)

    cp = await executor.checkpoint(
        activity.activity_id,
        fencing_token=1,
        data={"progress": 50},
        blob_ref="blob_abc",
    )
    assert cp.blob_ref == "blob_abc"
    assert activity.state is ActivityState.CHECKPOINTED

    # Resumable
    resumable = executor.list_resumable()
    assert len(resumable) == 1
    assert resumable[0].activity_id == activity.activity_id


@pytest.mark.asyncio
async def test_complete_success(executor: ActivityExecutor) -> None:
    activity = await executor.create(
        activity_type=ActivityType.VERIFICATION,
        run_id="run_1",
        fencing_token=1,
    )
    await executor.start(activity.activity_id, fencing_token=1)
    completed = await executor.complete(
        activity.activity_id, fencing_token=1, success=True
    )
    assert completed.state is ActivityState.SUCCEEDED


@pytest.mark.asyncio
async def test_complete_failure(executor: ActivityExecutor) -> None:
    activity = await executor.create(
        activity_type=ActivityType.RECONCILIATION,
        run_id="run_1",
        fencing_token=1,
    )
    await executor.start(activity.activity_id, fencing_token=1)
    completed = await executor.complete(
        activity.activity_id,
        fencing_token=1,
        success=False,
        error="verification criteria not met",
    )
    assert completed.state is ActivityState.FAILED
    assert completed.error == "verification criteria not met"


@pytest.mark.asyncio
async def test_cancel_running_activity(executor: ActivityExecutor) -> None:
    activity = await executor.create(
        activity_type=ActivityType.EXECUTION,
        run_id="run_1",
        fencing_token=1,
    )
    await executor.start(activity.activity_id, fencing_token=1)
    canceled = await executor.cancel(activity.activity_id)
    assert canceled.state is ActivityState.CANCELED


@pytest.mark.asyncio
async def test_timeout_activity(executor: ActivityExecutor) -> None:
    activity = await executor.create(
        activity_type=ActivityType.EXECUTION,
        run_id="run_1",
        fencing_token=1,
    )
    await executor.start(activity.activity_id, fencing_token=1)
    timed_out = await executor.timeout(activity.activity_id)
    assert timed_out.state is ActivityState.TIMED_OUT


@pytest.mark.asyncio
async def test_cannot_complete_terminal_activity(executor: ActivityExecutor) -> None:
    activity = await executor.create(
        activity_type=ActivityType.EXECUTION,
        run_id="run_1",
        fencing_token=1,
    )
    await executor.start(activity.activity_id, fencing_token=1)
    await executor.complete(activity.activity_id, fencing_token=1, success=True)

    with pytest.raises(ValueError):
        await executor.complete(activity.activity_id, fencing_token=1, success=False)


@pytest.mark.asyncio
async def test_activity_not_found(executor: ActivityExecutor) -> None:
    with pytest.raises(ActivityNotFound):
        await executor.start("nonexistent", fencing_token=1)


@pytest.mark.asyncio
async def test_list_active_filters_by_run(executor: ActivityExecutor) -> None:
    a1 = await executor.create(
        activity_type=ActivityType.EXECUTION,
        run_id="run_1",
        fencing_token=1,
    )
    await executor.create(
        activity_type=ActivityType.EXECUTION,
        run_id="run_2",
        fencing_token=1,
    )

    active = executor.list_active("run_1")
    assert len(active) == 1
    assert active[0].activity_id == a1.activity_id


@pytest.mark.asyncio
async def test_get_by_lease(executor: ActivityExecutor) -> None:
    activity = await executor.create(
        activity_type=ActivityType.EXECUTION,
        run_id="run_1",
        lease_id="lease_42",
        fencing_token=1,
    )

    found = executor.get_by_lease("lease_42")
    assert found is not None
    assert found.activity_id == activity.activity_id

    assert executor.get_by_lease("nonexistent") is None
