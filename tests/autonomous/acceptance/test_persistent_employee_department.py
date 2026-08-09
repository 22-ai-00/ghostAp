"""Automated pre-cutover gates for the persistent employee department."""

from __future__ import annotations

import inspect
from pathlib import Path

from src.autonomous.acceptance.employee_release import EmployeeReleaseManifest
from src.autonomous.gateway import EmployeeTeamGateway
from src.autonomous.gateway.coordinator import EmployeeDispatchCoordinator
from src.autonomous.team.service import EmployeeTeamService
from src.config.settings import Settings

_PERSISTENT_REAL_TENANT_GATES = {
    "EMP-PERSISTENT-ACTOR",
    "EMP-DIRECT-MENTION-ACTOR",
    "EMP-TEAM-COORDINATOR",
    "EMP-PARTIAL-CONTEXT",
    "EMP-SELECTIVE-WAKE",
    "EMP-FIRE-PERSISTENT",
    "EMP-RUNTIME-AUTHORITY-ISOLATION",
    "EMP-SOAK-1",
    "EMP-SOAK-10",
    "EMP-SOAK-50",
}


def test_release_defaults_start_the_persistent_employee_department() -> None:
    fields = Settings.model_fields
    assert fields["autonomous_employee_runtime_mode"].default == "actor"
    assert fields["autonomous_team_runtime_mode"].default == "coordinator"


def test_shadow_actor_and_coordinator_modes_are_explicit_and_non_fallback() -> None:
    shadow = Settings(_env_file=None, autonomous_employee_runtime_mode="shadow")
    actor = Settings(
        _env_file=None,
        autonomous_employee_runtime_mode="actor",
        autonomous_team_runtime_mode="coordinator",
    )
    assert shadow.autonomous_employee_runtime_mode == "shadow"
    assert actor.autonomous_employee_runtime_mode == "actor"
    assert actor.autonomous_team_runtime_mode == "coordinator"


def test_legacy_runtime_modes_can_only_be_selected_explicitly() -> None:
    assert (
        inspect.signature(EmployeeDispatchCoordinator).parameters[
            "employee_runtime_mode"
        ].default
        is inspect.Parameter.empty
    )
    assert (
        inspect.signature(EmployeeTeamGateway).parameters["runtime_mode"].default
        is inspect.Parameter.empty
    )
    assert (
        inspect.signature(EmployeeTeamService).parameters["runtime_mode"].default
        is inspect.Parameter.empty
    )


def test_real_tenant_manifest_covers_persistent_runtime_cutover() -> None:
    manifest = EmployeeReleaseManifest.load(
        Path("src/autonomous/acceptance/employee_release_manifest.json")
    )
    gate_ids = {gate.gate_id for gate in manifest.gates}
    assert _PERSISTENT_REAL_TENANT_GATES <= gate_ids
    for gate in manifest.gates:
        if gate.gate_id in _PERSISTENT_REAL_TENANT_GATES:
            assert "observed_on_real_tenant" in gate.required_assertions
