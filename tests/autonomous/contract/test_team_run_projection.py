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


def test_terminal_run_rejects_new_open_effect() -> None:
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
        f"{run.run_id}:final-notify:1",
        {"effect_type": "notify"},
    )

    with pytest.raises(TeamProjectionError, match="terminal"):
        _apply_event({run.run_id: run}, {}, {}, {}, event)
