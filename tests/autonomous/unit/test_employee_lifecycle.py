"""Unit tests for Employee lifecycle management."""

from __future__ import annotations

import pytest

from src.autonomous.employees import (
    CollaborationPlanner,
    EmployeeManager,
    EmployeeState,
    WorkerType,
)


class FakeJournal:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def write_event(self, event_type: str, payload: dict) -> None:
        self.events.append((event_type, payload))


@pytest.fixture
def manager() -> EmployeeManager:
    return EmployeeManager(journal=FakeJournal())


@pytest.mark.asyncio
async def test_logical_employee_is_default_and_uses_no_bot(manager: EmployeeManager) -> None:
    employee = await manager.hire(template="coder")
    assert employee.worker_type is WorkerType.LOGICAL
    assert employee.bot_principal_id is None
    assert employee.state is EmployeeState.ACTIVE


@pytest.mark.asyncio
async def test_hire_with_explicit_type(manager: EmployeeManager) -> None:
    employee = await manager.hire(
        name="CodeBot",
        role="coder",
        tool="codex",
        model="gpt-4o",
        worker_type=WorkerType.VISIBLE,
    )
    assert employee.worker_type is WorkerType.VISIBLE
    assert employee.tool == "codex"


@pytest.mark.asyncio
async def test_dismiss_employee(manager: EmployeeManager) -> None:
    emp = await manager.hire(template="reviewer")
    dismissed = await manager.dismiss(emp.employee_id)
    assert dismissed.state is EmployeeState.DISMISSED
    assert dismissed.dismissed_at is not None


@pytest.mark.asyncio
async def test_cannot_dismiss_twice(manager: EmployeeManager) -> None:
    emp = await manager.hire(template="coder")
    await manager.dismiss(emp.employee_id)
    with pytest.raises(ValueError, match="already dismissed"):
        await manager.dismiss(emp.employee_id)


@pytest.mark.asyncio
async def test_suspend_and_reactivate(manager: EmployeeManager) -> None:
    emp = await manager.hire(template="tester")
    suspended = await manager.suspend(emp.employee_id)
    assert suspended.state is EmployeeState.SUSPENDED

    reactivated = await manager.reactivate(emp.employee_id)
    assert reactivated.state is EmployeeState.ACTIVE


@pytest.mark.asyncio
async def test_list_active_excludes_dismissed(manager: EmployeeManager) -> None:
    await manager.hire(template="coder")
    emp2 = await manager.hire(template="reviewer")
    await manager.dismiss(emp2.employee_id)

    active = manager.list_active()
    assert len(active) == 1
    assert active[0].role == "coder"


@pytest.mark.asyncio
async def test_list_by_role(manager: EmployeeManager) -> None:
    await manager.hire(name="A", role="coder")
    await manager.hire(name="B", role="coder")
    await manager.hire(name="C", role="reviewer")

    coders = manager.list_by_role("coder")
    assert len(coders) == 2


@pytest.mark.asyncio
async def test_collaboration_planner_assigns_steps(manager: EmployeeManager) -> None:
    await manager.hire(name="Alice", role="coder")
    await manager.hire(name="Bob", role="reviewer")

    planner = CollaborationPlanner(manager)
    steps = [
        {"step_id": "s1", "required_role": "coder"},
        {"step_id": "s2", "required_role": "reviewer"},
    ]
    assigned = planner.assign_steps(steps)
    assert assigned[0]["assigned_employee_id"] is not None
    assert assigned[1]["assigned_employee_id"] is not None
    assert assigned[0]["assigned_employee_id"] != assigned[1]["assigned_employee_id"]


@pytest.mark.asyncio
async def test_verification_separation_ensures_different_reviewer(
    manager: EmployeeManager,
) -> None:
    coder = await manager.hire(name="Alice", role="coder")
    await manager.hire(name="Bob", role="reviewer")

    planner = CollaborationPlanner(manager)
    result = planner.ensure_verification_separation(
        coder.employee_id,
        {"step_id": "verify_1", "required_role": "reviewer"},
    )
    assert result["assigned_employee_id"] != coder.employee_id
    assert result["verification_separation"] is True
