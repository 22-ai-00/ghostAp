"""Unit tests for TriggerService."""

from __future__ import annotations

import pytest

from src.autonomous.scheduler.triggers import (
    TriggerService,
    TriggerState,
    TriggerType,
)


class FakeJournal:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def write_event(self, event_type: str, payload: dict) -> None:
        self.events.append((event_type, payload))


@pytest.fixture
def service() -> TriggerService:
    return TriggerService(journal=FakeJournal())


@pytest.mark.asyncio
async def test_register_and_fire(service: TriggerService) -> None:
    trigger = await service.register(
        trigger_type=TriggerType.SCHEDULE,
        goal_template_id="template_1",
        tenant_key="tenant_1",
    )
    assert trigger.state is TriggerState.ACTIVE

    occ = await service.fire(trigger.trigger_id, goal_id="goal_1")
    assert occ is not None
    assert occ.trigger_id == trigger.trigger_id
    assert trigger.occurrence_count == 1


@pytest.mark.asyncio
async def test_max_occurrences_expires_trigger(service: TriggerService) -> None:
    trigger = await service.register(
        trigger_type=TriggerType.SCHEDULE,
        max_occurrences=2,
    )

    await service.fire(trigger.trigger_id)
    await service.fire(trigger.trigger_id)
    occ = await service.fire(trigger.trigger_id)

    assert occ is None
    assert trigger.state is TriggerState.EXPIRED


@pytest.mark.asyncio
async def test_pause_prevents_firing(service: TriggerService) -> None:
    trigger = await service.register(trigger_type=TriggerType.EVENT)
    await service.pause(trigger.trigger_id)

    occ = await service.fire(trigger.trigger_id)
    assert occ is None


@pytest.mark.asyncio
async def test_resume_after_pause(service: TriggerService) -> None:
    trigger = await service.register(trigger_type=TriggerType.EVENT)
    await service.pause(trigger.trigger_id)
    await service.resume(trigger.trigger_id)

    occ = await service.fire(trigger.trigger_id)
    assert occ is not None


@pytest.mark.asyncio
async def test_cancel_trigger(service: TriggerService) -> None:
    trigger = await service.register(trigger_type=TriggerType.STANDING)
    await service.cancel(trigger.trigger_id)
    assert trigger.state is TriggerState.CANCELED


@pytest.mark.asyncio
async def test_get_due_triggers(service: TriggerService) -> None:
    t1 = await service.register(
        trigger_type=TriggerType.SCHEDULE,
        schedule_cron="*/5 * * * *",
    )
    t1.next_fire_at = 0  # past due

    t2 = await service.register(
        trigger_type=TriggerType.SCHEDULE,
        schedule_cron="0 0 * * *",
    )
    t2.next_fire_at = 9999999999  # far future

    due = service.get_due_triggers()
    assert len(due) == 1
    assert due[0].trigger_id == t1.trigger_id


@pytest.mark.asyncio
async def test_list_active_by_tenant(service: TriggerService) -> None:
    await service.register(tenant_key="t1")
    await service.register(tenant_key="t2")
    await service.register(tenant_key="t1")

    t1_triggers = service.list_active("t1")
    assert len(t1_triggers) == 2
