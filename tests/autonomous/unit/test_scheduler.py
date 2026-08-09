"""Unit tests for DurableScheduler."""

from __future__ import annotations

import pytest

from src.autonomous.domain.plans import PlanStep
from src.autonomous.scheduler.scheduler import (
    DurableScheduler,
)


class FakeJournal:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def write_event(self, event_type: str, payload: dict) -> None:
        self.events.append((event_type, payload))


def _make_step(step_id: str = "step_1", max_attempts: int = 3) -> PlanStep:
    return PlanStep(step_id=step_id, max_attempts=max_attempts)


@pytest.fixture
def scheduler() -> DurableScheduler:
    return DurableScheduler(journal=FakeJournal(), max_concurrent=2)


@pytest.mark.asyncio
async def test_enqueue_and_acquire(scheduler: DurableScheduler) -> None:
    step = _make_step()
    await scheduler.enqueue_step(step, "run_1", plan_epoch=1)

    ready = await scheduler.get_ready_steps()
    assert len(ready) == 1
    assert ready[0].step_id == "step_1"

    lease = await scheduler.acquire_lease("step_1", "worker_a")
    assert lease is not None
    assert lease.step_id == "step_1"
    assert lease.worker_id == "worker_a"
    assert lease.fencing_token == 1


@pytest.mark.asyncio
async def test_same_step_never_receives_two_leases(scheduler: DurableScheduler) -> None:
    step = _make_step()
    await scheduler.enqueue_step(step, "run_1", plan_epoch=1)

    lease1 = await scheduler.acquire_lease("step_1", "worker_1")
    lease2 = await scheduler.acquire_lease("step_1", "worker_2")

    assert lease1 is not None
    assert lease2 is None


@pytest.mark.asyncio
async def test_concurrent_acquire_only_one_succeeds(scheduler: DurableScheduler) -> None:
    step = _make_step()
    await scheduler.enqueue_step(step, "run_1", plan_epoch=1)

    # Simulate concurrent acquire (sequential in asyncio but tests the guard)
    results = []
    for i in range(5):
        lease = await scheduler.acquire_lease("step_1", f"worker_{i}")
        results.append(lease)

    granted = [r for r in results if r is not None]
    assert len(granted) == 1


@pytest.mark.asyncio
async def test_max_concurrent_enforced(scheduler: DurableScheduler) -> None:
    for i in range(3):
        await scheduler.enqueue_step(_make_step(f"step_{i}"), "run_1", plan_epoch=1)

    lease1 = await scheduler.acquire_lease("step_0", "w1")
    lease2 = await scheduler.acquire_lease("step_1", "w2")
    lease3 = await scheduler.acquire_lease("step_2", "w3")

    assert lease1 is not None
    assert lease2 is not None
    assert lease3 is None  # max_concurrent=2


@pytest.mark.asyncio
async def test_release_success_removes_from_queue(scheduler: DurableScheduler) -> None:
    await scheduler.enqueue_step(_make_step(), "run_1", plan_epoch=1)
    lease = await scheduler.acquire_lease("step_1", "w1")
    await scheduler.release_lease(lease.lease_id, success=True)

    ready = await scheduler.get_ready_steps()
    assert len(ready) == 0

    stats = scheduler.get_stats()
    assert stats.total_completed == 1


@pytest.mark.asyncio
async def test_release_failure_retries_with_backoff(scheduler: DurableScheduler) -> None:
    await scheduler.enqueue_step(_make_step(), "run_1", plan_epoch=1)
    lease = await scheduler.acquire_lease("step_1", "w1")
    await scheduler.release_lease(lease.lease_id, success=False)

    # Step should be back in queue but with future next_retry_after
    ready = await scheduler.get_ready_steps()
    assert len(ready) == 0  # not ready yet (backoff)

    stats = scheduler.get_stats()
    assert stats.total_failed == 1


@pytest.mark.asyncio
async def test_max_retries_moves_to_dead_letter(scheduler: DurableScheduler) -> None:
    await scheduler.enqueue_step(_make_step(max_attempts=1), "run_1", plan_epoch=1)
    lease = await scheduler.acquire_lease("step_1", "w1")
    await scheduler.release_lease(lease.lease_id, success=False)

    stats = scheduler.get_stats()
    assert stats.dead_letters == 1
    assert stats.queued == 0


@pytest.mark.asyncio
async def test_renew_extends_expiry(scheduler: DurableScheduler) -> None:
    await scheduler.enqueue_step(_make_step(), "run_1", plan_epoch=1)
    lease = await scheduler.acquire_lease("step_1", "w1")
    old_expires = lease.expires_at

    result = await scheduler.renew_lease(lease.lease_id)
    assert result is True
    # expiry was extended
    assert lease.expires_at > old_expires


@pytest.mark.asyncio
async def test_fencing_tokens_monotonically_increase(scheduler: DurableScheduler) -> None:
    await scheduler.enqueue_step(_make_step("s1"), "run_1", plan_epoch=1)
    await scheduler.enqueue_step(_make_step("s2"), "run_1", plan_epoch=1)

    l1 = await scheduler.acquire_lease("s1", "w1")
    await scheduler.release_lease(l1.lease_id, success=True)

    # Re-enqueue and acquire again
    await scheduler.enqueue_step(_make_step("s1"), "run_1", plan_epoch=1)
    l2 = await scheduler.acquire_lease("s1", "w1")

    assert l2.fencing_token > l1.fencing_token
