"""Chaos test: stale fencing tokens cannot dispatch after restart."""

from __future__ import annotations

import pytest

from src.autonomous.domain.plans import PlanStep
from src.autonomous.scheduler.activities import (
    ActivityExecutor,
    ActivityType,
    StaleLease,
)
from src.autonomous.scheduler.scheduler import DurableScheduler


class FakeJournal:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def write_event(self, event_type: str, payload: dict) -> None:
        self.events.append((event_type, payload))


@pytest.mark.asyncio
async def test_stale_fencing_token_cannot_dispatch_after_restart() -> None:
    """Simulates: worker acquires lease, system restarts with higher fencing
    counter, old token rejected on any operation."""
    journal = FakeJournal()
    scheduler = DurableScheduler(journal=journal)

    step = PlanStep(step_id="step_1", max_attempts=3)
    await scheduler.enqueue_step(step, "run_1", plan_epoch=1)

    old_lease = await scheduler.acquire_lease("step_1", "worker_1")
    assert old_lease is not None
    old_token = old_lease.fencing_token

    # Simulate restart: release the lease and re-acquire with new token
    await scheduler.release_lease(old_lease.lease_id, success=False)
    await scheduler.enqueue_step(step, "run_1", plan_epoch=1)
    new_lease = await scheduler.acquire_lease("step_1", "worker_2")
    assert new_lease is not None
    assert new_lease.fencing_token > old_token

    # Activity executor rejects old fencing token
    executor = ActivityExecutor(journal=journal)
    activity = await executor.create(
        activity_type=ActivityType.EXECUTION,
        run_id="run_1",
        step_id="step_1",
        lease_id=new_lease.lease_id,
        fencing_token=new_lease.fencing_token,
    )

    # Old token cannot start
    with pytest.raises(StaleLease):
        await executor.start(activity.activity_id, fencing_token=old_token)

    # New token works
    started = await executor.start(activity.activity_id, fencing_token=new_lease.fencing_token)
    assert started.state.value == "running"


@pytest.mark.asyncio
async def test_expired_lease_triggers_retry() -> None:
    """When a lease expires, the step is re-queued with retry backoff."""
    journal = FakeJournal()
    scheduler = DurableScheduler(journal=journal, default_lease_seconds=0.001)

    step = PlanStep(step_id="step_1", max_attempts=3)
    await scheduler.enqueue_step(step, "run_1", plan_epoch=1)

    lease = await scheduler.acquire_lease("step_1", "worker_1")
    assert lease is not None

    import time
    time.sleep(0.01)  # wait for expiry

    expired = await scheduler.check_expired_leases()
    assert len(expired) == 1
    assert expired[0].step_id == "step_1"

    # Lease is cleared, step can be re-acquired (after backoff)
    stats = scheduler.get_stats()
    assert stats.active_leases == 0


@pytest.mark.asyncio
async def test_fencing_token_monotonic_across_multiple_leases() -> None:
    """Fencing tokens always increase even after releases."""
    journal = FakeJournal()
    scheduler = DurableScheduler(journal=journal)

    tokens: list[int] = []
    for i in range(5):
        step = PlanStep(step_id=f"step_{i}", max_attempts=3)
        await scheduler.enqueue_step(step, "run_1", plan_epoch=1)
        lease = await scheduler.acquire_lease(f"step_{i}", f"w_{i}")
        if lease:
            tokens.append(lease.fencing_token)
            await scheduler.release_lease(lease.lease_id, success=True)

    # Tokens must be strictly increasing
    assert tokens == sorted(tokens)
    assert len(set(tokens)) == len(tokens)
