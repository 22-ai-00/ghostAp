from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from src.autonomous.journal.blob_store import BlobRef
from src.autonomous.journal.frame import JournalEvent
from src.autonomous.team.models import TeamAssignmentV2, TeamRunPhase, TeamRunV2
from src.autonomous.team.projection import (
    TeamProjectionError,
    _apply_event,
    _assert_no_open_effects,
)


def _ref() -> BlobRef:
    return BlobRef(
        blob_hash="a" * 64,
        payload_hash="b" * 64,
        labels_hash=hashlib.sha256(b"{}").hexdigest(),
        key_ref="key",
        size=1,
    )


def test_team_run_contract_requires_done_criteria_and_enforces_turn_bounds() -> None:
    with pytest.raises(ValueError, match="done criteria"):
        TeamRunV2(
            "teamrun2_x",
            "tenant_1",
            "oc_team",
            "",
            "om_1",
            "ou_1",
            _ref(),
            "goal",
            (),
            "session",
        )
    run = TeamRunV2(
        "teamrun2_x",
        "tenant_1",
        "oc_team",
        "",
        "om_1",
        "ou_1",
        _ref(),
        "goal",
        ("verified",),
        "session",
    )
    with pytest.raises(ValueError, match="turn bound"):
        replace(run, turn_count=13)
    with pytest.raises(ValueError, match="handoff bound"):
        replace(run, handoff_count=9)
    with pytest.raises(ValueError, match="assignment bound"):
        replace(run, assignment_ids=tuple(f"assignment_{index}" for index in range(33)))
    with pytest.raises(ValueError, match="cyclic"):
        TeamAssignmentV2(
            "assignment_1",
            run.run_id,
            "agt_alpha",
            "execute",
            _ref(),
            depends_on=("assignment_1",),
        )


def test_terminal_transition_rejects_unresolved_effect() -> None:
    with pytest.raises(TeamProjectionError, match="unresolved effects"):
        _assert_no_open_effects(
            {("teamrun2_x:assignment:1", "employee_dispatch"): "executing"},
            "teamrun2_x",
        )


def test_team_run_phase_contract_exposes_required_states() -> None:
    assert {item.value for item in TeamRunPhase} == {
        "created",
        "planning",
        "dispatching",
        "reviewing",
        "revising",
        "finalizing",
        "completed",
        "blocking",
        "blocked",
        "canceled",
    }


def test_completed_run_requires_durable_evidence_for_every_done_criterion() -> None:
    run = TeamRunV2(
        "teamrun2_evidence",
        "tenant_1",
        "oc_team",
        "",
        "om_1",
        "ou_1",
        _ref(),
        "goal",
        ("deliverable_non_empty", "review_completed"),
        "session",
        phase=TeamRunPhase.REVIEWING,
    )
    event = JournalEvent(
        "team.v2.run.completed",
        run.run_id,
        {
            "run_id": run.run_id,
            "result_ref": _ref().to_dict(),
            "done_checks": {"deliverable_non_empty": True},
        },
    )

    with pytest.raises(TeamProjectionError, match="done criteria"):
        _apply_event({run.run_id: run}, {}, {}, {}, event)


def test_terminal_run_rejects_unrelated_new_open_effect() -> None:
    run = TeamRunV2(
        "teamrun2_terminal",
        "tenant_1",
        "oc_team",
        "",
        "om_1",
        "ou_1",
        _ref(),
        "goal",
        ("deliverable_non_empty",),
        "session",
        phase=TeamRunPhase.COMPLETED,
        final_result_ref=_ref(),
        final_done_checks={"deliverable_non_empty": True},
    )
    event = JournalEvent(
        "team.v2.effect.prepared",
        f"{run.run_id}:assignment:late",
        {"effect_type": "employee_dispatch"},
    )

    with pytest.raises(TeamProjectionError, match="terminal"):
        _apply_event({run.run_id: run}, {}, {}, {}, event)


def test_dispatch_authorization_consumes_exact_team_v2_assignment_effect() -> None:
    from types import SimpleNamespace

    from src.autonomous.gateway import EmployeeDispatchCoordinator

    run_id = "teamrun2_authority"
    part = {"team_run_id": run_id, "team_step_id": "3"}

    def authorized(event: JournalEvent) -> bool:
        coordinator = object.__new__(EmployeeDispatchCoordinator)
        coordinator._writer = SimpleNamespace(  # noqa: SLF001
            replay=lambda: (SimpleNamespace(events=(event,)),)
        )
        return coordinator._team_assignment_effect_is_active(part)  # noqa: SLF001

    assert authorized(
        JournalEvent(
            "team.v2.effect.executing",
            f"{run_id}:assignment:3",
            {"effect_type": "employee_dispatch"},
        )
    )
    assert not authorized(
        JournalEvent(
            "team.effect.executing",
            f"{run_id}:assignment:3",
            {"effect_type": "employee_dispatch"},
        )
    )
    assert not authorized(
        JournalEvent(
            "team.v2.effect.executing",
            f"{run_id}:3",
            {"effect_type": "employee_dispatch"},
        )
    )


def test_gateway_package_is_the_canonical_dispatch_import_surface() -> None:
    from src.autonomous.gateway import (
        EmployeeDispatchCoordinator,
        EmployeeTeamGateway,
    )
    from src.autonomous.gateway.coordinator import (
        EmployeeDispatchCoordinator as CoordinatorImplementation,
    )
    from src.autonomous.gateway.team import EmployeeTeamGateway as GatewayImplementation

    assert EmployeeDispatchCoordinator is CoordinatorImplementation
    assert EmployeeTeamGateway is GatewayImplementation


@pytest.mark.parametrize(
    ("terminal_phase", "aggregate_suffix", "repair_phase"),
    [
        (
            TeamRunPhase.COMPLETED,
            "final-notify:1",
            TeamRunPhase.FINALIZING,
        ),
        (
            TeamRunPhase.BLOCKED,
            "blocked-notify:1",
            TeamRunPhase.BLOCKING,
        ),
    ],
)
def test_legacy_terminal_notification_effect_replays_into_repair_phase(
    terminal_phase,
    aggregate_suffix,
    repair_phase,
) -> None:
    run = TeamRunV2(
        f"teamrun2_legacy_{terminal_phase.value}",
        "tenant_1",
        "oc_team",
        "",
        "om_1",
        "ou_1",
        _ref(),
        "goal",
        ("deliverable_non_empty",),
        "session",
        phase=terminal_phase,
        final_result_ref=(
            _ref()
            if terminal_phase is TeamRunPhase.COMPLETED
            else None
        ),
        final_done_checks=(
            {"deliverable_non_empty": True}
            if terminal_phase is TeamRunPhase.COMPLETED
            else {}
        ),
        error_code=(
            "team_task_failed"
            if terminal_phase is TeamRunPhase.BLOCKED
            else ""
        ),
    )
    runs = {run.run_id: run}
    effects = {}
    aggregate = f"{run.run_id}:{aggregate_suffix}"

    _apply_event(
        runs,
        {},
        effects,
        {},
        JournalEvent(
            "team.v2.effect.prepared",
            aggregate,
            {"effect_type": "notify"},
        ),
    )
    assert runs[run.run_id].phase is repair_phase
    assert effects[(aggregate, "notify")] == "prepared"

    for state in ("executing", "committed"):
        _apply_event(
            runs,
            {},
            effects,
            {},
            JournalEvent(
                f"team.v2.effect.{state}",
                aggregate,
                {"effect_type": "notify"},
            ),
        )
    assert runs[run.run_id].phase is repair_phase
    assert effects[(aggregate, "notify")] == "committed"
