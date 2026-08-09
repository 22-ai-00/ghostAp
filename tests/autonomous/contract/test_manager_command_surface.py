"""Contract tests for Manager command surface completeness."""

from __future__ import annotations

import pytest

from src.autonomous.manager.handler import REQUIRED_COMMANDS, ManagerHandler


class FakeDep:
    """Minimal fake dependencies for ManagerHandler."""

    def list_pending_approvals(self) -> list:
        return []


@pytest.fixture
def handler() -> ManagerHandler:
    fake = FakeDep()
    return ManagerHandler(
        admission=fake,
        plan_compiler=fake,
        scheduler=fake,
        policy_engine=fake,
        reporter=fake,
        kill_switch=fake,
    )


def test_manager_surface_has_all_required_commands(handler: ManagerHandler) -> None:
    assert REQUIRED_COMMANDS <= handler.command_names


def test_no_placeholder_commands(handler: ManagerHandler) -> None:
    assert not handler.has_placeholder_commands()


def test_required_commands_cover_expected_operations() -> None:
    assert "goal.create" in REQUIRED_COMMANDS
    assert "goal.list" in REQUIRED_COMMANDS
    assert "goal.cancel" in REQUIRED_COMMANDS
    assert "run.start" in REQUIRED_COMMANDS
    assert "run.list" in REQUIRED_COMMANDS
    assert "approval.approve" in REQUIRED_COMMANDS
    assert "decision.respond" in REQUIRED_COMMANDS
    assert "employee.list" in REQUIRED_COMMANDS
    assert "health" in REQUIRED_COMMANDS
