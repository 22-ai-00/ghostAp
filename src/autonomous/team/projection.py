"""Pure replay projection and transition guards for TeamRun V2."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from ..journal.blob_store import BlobRef
from ..journal.frame import JournalEvent
from .models import (
    MAX_TEAM_ASSIGNMENTS,
    MAX_TEAM_HANDOFFS,
    MAX_TEAM_TURNS,
    TeamAssignmentStatus,
    TeamAssignmentV2,
    TeamProjection,
    TeamRunPhase,
    TeamRunV2,
)


class TeamProjectionError(RuntimeError):
    pass


_PHASE_EDGES = {
    TeamRunPhase.CREATED: {
        TeamRunPhase.PLANNING,
        TeamRunPhase.BLOCKED,
        TeamRunPhase.CANCELED,
    },
    TeamRunPhase.PLANNING: {TeamRunPhase.DISPATCHING, TeamRunPhase.BLOCKED, TeamRunPhase.CANCELED},
    TeamRunPhase.DISPATCHING: {TeamRunPhase.REVIEWING, TeamRunPhase.BLOCKED, TeamRunPhase.CANCELED},
    TeamRunPhase.REVIEWING: {
        TeamRunPhase.REVISING,
        TeamRunPhase.COMPLETED,
        TeamRunPhase.BLOCKED,
        TeamRunPhase.CANCELED,
    },
    TeamRunPhase.REVISING: {
        TeamRunPhase.REVIEWING,
        TeamRunPhase.COMPLETED,
        TeamRunPhase.BLOCKED,
        TeamRunPhase.CANCELED,
    },
    TeamRunPhase.FINALIZING: {
        TeamRunPhase.BLOCKED,
        TeamRunPhase.CANCELED,
    },
    TeamRunPhase.BLOCKING: {
        TeamRunPhase.BLOCKED,
        TeamRunPhase.CANCELED,
    },
}


def _terminal(phase: TeamRunPhase) -> bool:
    return phase in {TeamRunPhase.COMPLETED, TeamRunPhase.BLOCKED, TeamRunPhase.CANCELED}


def _assert_no_open_effects(
    effects: dict[tuple[str, str], str], run_id: str
) -> None:
    if any(
        aggregate == run_id or aggregate.startswith(run_id + ":")
        for aggregate, _effect_type in (
            key for key, state in effects.items() if state in {"prepared", "executing"}
        )
    ):
        raise TeamProjectionError("team run has unresolved effects")


def _validated_done_checks(run: TeamRunV2, value: object) -> dict[str, bool]:
    if (
        not isinstance(value, dict)
        or set(value) != set(run.done_criteria)
        or any(
            type(item) is not bool or item is not True
            for item in value.values()
        )
    ):
        raise TeamProjectionError("team run done criteria are not satisfied")
    return dict(value)


def _has_pending_final_notification(
    effects: dict[tuple[str, str], str],
    run_id: str,
) -> bool:
    legacy = f"{run_id}:notify"
    prefix = f"{run_id}:final-notify:"
    states = [
        state
        for (aggregate, effect_type), state in effects.items()
        if effect_type == "notify"
        and (aggregate == legacy or aggregate.startswith(prefix))
    ]
    return bool(states) and "committed" not in states


def _has_pending_block_notification(
    effects: dict[tuple[str, str], str],
    run_id: str,
) -> bool:
    prefix = f"{run_id}:blocked-notify:"
    return not any(
        state == "committed"
        for (aggregate, effect_type), state in effects.items()
        if effect_type == "notify" and aggregate.startswith(prefix)
    )


def rebuild_team_projection(frames: Iterable[object]) -> TeamProjection:
    runs: dict[str, TeamRunV2] = {}
    assignments: dict[str, TeamAssignmentV2] = {}
    effects: dict[tuple[str, str], str] = {}
    collaboration_events: dict[str, str] = {}
    for frame in frames:
        for event in frame.events:
            if event.event_type.startswith("team.v2."):
                _apply_event(runs, assignments, effects, collaboration_events, event)
    return TeamProjection(runs, assignments, effects, collaboration_events)


def _apply_event(
    runs: dict[str, TeamRunV2],
    assignments: dict[str, TeamAssignmentV2],
    effects: dict[tuple[str, str], str],
    collaboration_events: dict[str, str],
    event: JournalEvent,
) -> None:
    payload = event.payload
    if event.event_type == "team.v2.run.created":
        if event.aggregate_id in runs:
            raise TeamProjectionError("duplicate TeamRun V2")
        runs[event.aggregate_id] = TeamRunV2(
            run_id=event.aggregate_id,
            tenant_key=str(payload["tenant_key"]),
            chat_id=str(payload["chat_id"]),
            project_id=str(payload.get("project_id", "")),
            message_id=str(payload["message_id"]),
            requester_principal_id=str(payload["requester_principal_id"]),
            task_ref=BlobRef.from_dict(payload["task_ref"]),
            goal=str(payload["goal"]),
            done_criteria=tuple(payload["done_criteria"]),
            coordinator_session_key=str(payload["coordinator_session_key"]),
            coordinator_tool=str(payload.get("coordinator_tool", "coco")),
            coordinator_model=str(payload.get("coordinator_model", "")),
            coordinator_profile=str(payload.get("coordinator_profile", "")),
            coordinator_effort=str(payload.get("coordinator_effort", "")),
        )
        return
    if event.event_type.startswith("team.v2.effect."):
        effect_type = str(payload["effect_type"])
        state = event.event_type.rsplit(".", 1)[-1]
        owner = next(
            (
                run
                for run_id, run in runs.items()
                if event.aggregate_id == run_id
                or event.aggregate_id.startswith(run_id + ":")
            ),
            None,
        )
        if (
            owner is not None
            and _terminal(owner.phase)
            and state in {"prepared", "executing"}
        ):
            raise TeamProjectionError("terminal TeamRun cannot open an effect")
        previous = effects.get((event.aggregate_id, effect_type))
        allowed = {
            None: {"prepared"},
            "prepared": {"executing", "action_required"},
            "executing": {"committed", "action_required"},
        }
        if state not in allowed.get(previous, set()):
            raise TeamProjectionError("invalid team effect transition")
        effects[(event.aggregate_id, effect_type)] = state
        return
    run_id = str(payload.get("run_id", event.aggregate_id))
    run = runs.get(run_id)
    if run is None:
        raise TeamProjectionError("TeamRun V2 event precedes creation")
    if event.event_type == "team.v2.run.finalization_started":
        if run.phase not in {
            TeamRunPhase.REVIEWING,
            TeamRunPhase.REVISING,
        }:
            raise TeamProjectionError("TeamRun V2 finalization is premature")
        done_checks = _validated_done_checks(run, payload.get("done_checks"))
        runs[run_id] = replace(
            run,
            phase=TeamRunPhase.FINALIZING,
            final_result_ref=BlobRef.from_dict(payload["result_ref"]),
            final_done_checks=done_checks,
        )
        return
    if event.event_type == "team.v2.run.finalization_reopened":
        if (
            run.phase is not TeamRunPhase.COMPLETED
            or run.final_result_ref is None
            or not run.final_done_checks
            or not _has_pending_final_notification(effects, run_id)
        ):
            raise TeamProjectionError(
                "TeamRun V2 finalization repair is not allowed"
            )
        runs[run_id] = replace(run, phase=TeamRunPhase.FINALIZING)
        return
    if event.event_type == "team.v2.run.block_notification_reopened":
        if (
            run.phase is not TeamRunPhase.BLOCKED
            or not _has_pending_block_notification(effects, run_id)
        ):
            raise TeamProjectionError(
                "TeamRun V2 block notification repair is not allowed"
            )
        runs[run_id] = replace(run, phase=TeamRunPhase.BLOCKING)
        return
    if event.event_type == "team.v2.run.phase_changed":
        destination = TeamRunPhase(str(payload["phase"]))
        if destination not in _PHASE_EDGES.get(run.phase, set()):
            raise TeamProjectionError("invalid TeamRun V2 phase transition")
        turn_count = int(payload.get("turn_count", run.turn_count))
        handoff_count = int(payload.get("handoff_count", run.handoff_count))
        if turn_count > MAX_TEAM_TURNS or handoff_count > MAX_TEAM_HANDOFFS:
            raise TeamProjectionError("TeamRun V2 bound exceeded")
        if _terminal(destination):
            _assert_no_open_effects(effects, run_id)
        runs[run_id] = replace(
            run,
            phase=destination,
            turn_count=turn_count,
            handoff_count=handoff_count,
            error_code=str(payload.get("error_code", "")),
        )
        return
    if event.event_type == "team.v2.run.completed":
        if run.phase not in {
            TeamRunPhase.REVIEWING,
            TeamRunPhase.REVISING,
            TeamRunPhase.FINALIZING,
        }:
            raise TeamProjectionError("TeamRun V2 completion is premature")
        _assert_no_open_effects(effects, run_id)
        done_checks = _validated_done_checks(run, payload.get("done_checks"))
        result_ref = BlobRef.from_dict(payload["result_ref"])
        if run.phase is TeamRunPhase.FINALIZING and (
            run.final_result_ref != result_ref
            or dict(run.final_done_checks) != done_checks
        ):
            raise TeamProjectionError("TeamRun V2 finalization evidence changed")
        runs[run_id] = replace(
            run,
            phase=TeamRunPhase.COMPLETED,
            final_result_ref=result_ref,
            final_done_checks=done_checks,
        )
        return
    if event.event_type == "team.v2.collaboration.observed":
        assignment_id = str(payload["assignment_id"])
        causal_event_id = str(payload["causal_event_id"])
        assignment = assignments.get(assignment_id)
        if (
            assignment is None
            or assignment.run_id != run_id
            or assignment.agent_id != str(payload["agent_id"])
            or assignment.status is not TeamAssignmentStatus.COMPLETED
        ):
            raise TeamProjectionError("invalid collaboration assignment authority")
        if run.phase in {
            TeamRunPhase.FINALIZING,
            TeamRunPhase.BLOCKING,
            TeamRunPhase.COMPLETED,
            TeamRunPhase.BLOCKED,
            TeamRunPhase.CANCELED,
        }:
            raise TeamProjectionError("collaboration run is terminal")
        if not causal_event_id or causal_event_id in collaboration_events:
            raise TeamProjectionError("duplicate collaboration causal event")
        collaboration_events[causal_event_id] = assignment_id
        return
    if event.event_type == "team.v2.assignment.created":
        assignment_id = event.aggregate_id
        if assignment_id in assignments or len(run.assignment_ids) >= MAX_TEAM_ASSIGNMENTS:
            raise TeamProjectionError("Team assignment bound or duplicate violated")
        dependencies = tuple(payload.get("depends_on", ()))
        if assignment_id in dependencies or any(item not in run.assignment_ids for item in dependencies):
            raise TeamProjectionError("invalid team assignment dependency")
        assignments[assignment_id] = TeamAssignmentV2(
            assignment_id=assignment_id,
            run_id=run_id,
            agent_id=str(payload["agent_id"]),
            role=str(payload["role"]),
            instruction_ref=BlobRef.from_dict(payload["instruction_ref"]),
            depends_on=dependencies,
        )
        runs[run_id] = replace(run, assignment_ids=run.assignment_ids + (assignment_id,))
        return
    assignment = assignments.get(event.aggregate_id)
    if assignment is None:
        raise TeamProjectionError("team assignment event precedes creation")
    if event.event_type == "team.v2.assignment.claimed":
        if assignment.status is not TeamAssignmentStatus.CREATED:
            raise TeamProjectionError("team assignment claim lost CAS")
        assignments[event.aggregate_id] = replace(
            assignment, status=TeamAssignmentStatus.CLAIMED
        )
    elif event.event_type == "team.v2.assignment.submitted":
        if assignment.status is not TeamAssignmentStatus.CLAIMED:
            raise TeamProjectionError("team assignment submit is invalid")
        assignments[event.aggregate_id] = replace(
            assignment,
            status=TeamAssignmentStatus.RUNNING,
            acceptance_id=str(payload["acceptance_id"]),
        )
    elif event.event_type == "team.v2.assignment.completed":
        if assignment.status is not TeamAssignmentStatus.RUNNING:
            raise TeamProjectionError("team assignment completion is invalid")
        assignments[event.aggregate_id] = replace(
            assignment,
            status=TeamAssignmentStatus.COMPLETED,
            contribution_ref=BlobRef.from_dict(payload["contribution_ref"]),
            history_record_id=str(payload.get("history_record_id", "")),
        )
    elif event.event_type == "team.v2.assignment.failed":
        if assignment.status not in {TeamAssignmentStatus.CLAIMED, TeamAssignmentStatus.RUNNING}:
            raise TeamProjectionError("team assignment failure is invalid")
        assignments[event.aggregate_id] = replace(
            assignment,
            status=TeamAssignmentStatus.FAILED,
            error_code=str(payload["error_code"]),
        )


__all__ = ["TeamProjectionError", "rebuild_team_projection"]
