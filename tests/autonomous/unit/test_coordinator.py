"""Unit tests for Coordinator and Planner."""

from __future__ import annotations

import pytest

from src.autonomous.coordinator import Coordinator
from src.autonomous.planner import Planner


class FakeJournal:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def write_event(self, event_type: str, payload: dict) -> None:
        self.events.append((event_type, payload))


@pytest.fixture
def coordinator() -> Coordinator:
    return Coordinator(journal=FakeJournal())


@pytest.fixture
def planner() -> Planner:
    return Planner(journal=FakeJournal())


@pytest.mark.asyncio
async def test_create_goal_returns_ids(coordinator: Coordinator) -> None:
    result = await coordinator.create_goal(
        description="implement feature X",
        owner_principal_id="user_1",
        tenant_key="tenant_1",
    )
    assert result.success
    assert result.goal_id.startswith("goal_")
    assert result.run_id.startswith("run_")


@pytest.mark.asyncio
async def test_pause_and_resume_goal(coordinator: Coordinator) -> None:
    result = await coordinator.create_goal(
        description="test",
        owner_principal_id="user_1",
        tenant_key="t1",
    )
    assert await coordinator.pause_goal(result.goal_id, "user_1")
    goal = await coordinator.get_goal(result.goal_id)
    assert goal["state"] == "paused"

    assert await coordinator.resume_goal(result.goal_id, "user_1")
    goal = await coordinator.get_goal(result.goal_id)
    assert goal["state"] == "active"


@pytest.mark.asyncio
async def test_cancel_goal(coordinator: Coordinator) -> None:
    result = await coordinator.create_goal(
        description="test",
        owner_principal_id="user_1",
        tenant_key="t1",
    )
    assert await coordinator.cancel_goal(result.goal_id, "user_1")
    goal = await coordinator.get_goal(result.goal_id)
    assert goal["state"] == "canceled"


@pytest.mark.asyncio
async def test_list_goals_by_tenant(coordinator: Coordinator) -> None:
    await coordinator.create_goal(description="a", owner_principal_id="u1", tenant_key="t1")
    await coordinator.create_goal(description="b", owner_principal_id="u1", tenant_key="t2")
    await coordinator.create_goal(description="c", owner_principal_id="u1", tenant_key="t1")

    goals = await coordinator.list_goals("t1")
    assert len(goals) == 2


@pytest.mark.asyncio
async def test_compile_plan_default_single_step(planner: Planner) -> None:
    result = await planner.compile_plan(
        goal_id="goal_1",
        run_id="run_1",
        description="implement login page",
    )
    assert result.success
    assert result.plan is not None
    assert len(result.plan.steps) == 1
    assert result.plan.steps[0].description == "implement login page"


@pytest.mark.asyncio
async def test_policy_denial_blocks_creation() -> None:
    class DenyAll:
        def check_activation(self, principal_id: str, goal_spec: dict) -> bool:
            return False

    coord = Coordinator(journal=FakeJournal(), policy_gate=DenyAll())
    result = await coord.create_goal(
        description="blocked",
        owner_principal_id="user_1",
        tenant_key="t1",
    )
    assert not result.success
    assert "denied" in result.error
