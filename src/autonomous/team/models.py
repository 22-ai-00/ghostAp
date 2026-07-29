"""Frozen contracts for durable model-led team coordination."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

from ..journal.blob_store import BlobRef

MAX_TEAM_TURNS = 12
MAX_TEAM_ASSIGNMENTS = 32
MAX_TEAM_FANOUT = 4
MAX_TEAM_HANDOFFS = 8


class TeamRunPhase(StrEnum):
    CREATED = "created"
    PLANNING = "planning"
    DISPATCHING = "dispatching"
    REVIEWING = "reviewing"
    REVISING = "revising"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    BLOCKING = "blocking"
    BLOCKED = "blocked"
    CANCELED = "canceled"


class TeamAssignmentStatus(StrEnum):
    CREATED = "created"
    CLAIMED = "claimed"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class CoordinatorAction(StrEnum):
    ASSIGN = "assign"
    REVIEW = "review"
    REVISE = "revise"
    COMPLETE = "complete"
    BLOCK = "block"


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}", value) is None:
        raise ValueError(f"invalid {name}")
    return value


@dataclass(frozen=True, slots=True)
class TeamRunV2:
    run_id: str
    tenant_key: str
    chat_id: str
    project_id: str
    message_id: str
    requester_principal_id: str
    task_ref: BlobRef
    goal: str
    done_criteria: tuple[str, ...]
    coordinator_session_key: str
    coordinator_tool: str = "coco"
    coordinator_model: str = ""
    coordinator_profile: str = ""
    coordinator_effort: str = ""
    phase: TeamRunPhase = TeamRunPhase.CREATED
    turn_count: int = 0
    assignment_ids: tuple[str, ...] = ()
    handoff_count: int = 0
    final_result_ref: BlobRef | None = None
    final_done_checks: Mapping[str, bool] = field(default_factory=dict)
    error_code: str = ""

    def __post_init__(self) -> None:
        for value, name in (
            (self.run_id, "run_id"),
            (self.tenant_key, "tenant_key"),
            (self.chat_id, "chat_id"),
            (self.message_id, "message_id"),
            (self.requester_principal_id, "requester_principal_id"),
        ):
            _identifier(value, name)
        if not self.goal or not self.goal.strip() or len(self.goal) > 4_000:
            raise ValueError("invalid team goal")
        object.__setattr__(self, "done_criteria", tuple(self.done_criteria))
        object.__setattr__(self, "assignment_ids", tuple(self.assignment_ids))
        object.__setattr__(
            self,
            "final_done_checks",
            MappingProxyType(dict(self.final_done_checks)),
        )
        if not self.done_criteria or any(not item.strip() for item in self.done_criteria):
            raise ValueError("team run requires done criteria")
        if len(self.assignment_ids) > MAX_TEAM_ASSIGNMENTS:
            raise ValueError("team assignment bound exceeded")
        if not 0 <= self.turn_count <= MAX_TEAM_TURNS:
            raise ValueError("team turn bound exceeded")
        if not 0 <= self.handoff_count <= MAX_TEAM_HANDOFFS:
            raise ValueError("team handoff bound exceeded")
        if self.final_done_checks and (
            set(self.final_done_checks) != set(self.done_criteria)
            or any(
                type(value) is not bool or value is not True
                for value in self.final_done_checks.values()
            )
        ):
            raise ValueError("invalid final done checks")
        if self.phase in {
            TeamRunPhase.FINALIZING,
            TeamRunPhase.COMPLETED,
        } and (
            self.final_result_ref is None or not self.final_done_checks
        ):
            raise ValueError("finalizing team run lacks durable result")


@dataclass(frozen=True, slots=True)
class TeamAssignmentV2:
    assignment_id: str
    run_id: str
    agent_id: str
    role: str
    instruction_ref: BlobRef
    depends_on: tuple[str, ...] = ()
    status: TeamAssignmentStatus = TeamAssignmentStatus.CREATED
    acceptance_id: str = ""
    contribution_ref: BlobRef | None = None
    history_record_id: str = ""
    error_code: str = ""

    def __post_init__(self) -> None:
        _identifier(self.assignment_id, "assignment_id")
        _identifier(self.run_id, "run_id")
        _identifier(self.agent_id, "agent_id")
        object.__setattr__(self, "depends_on", tuple(self.depends_on))
        if self.assignment_id in self.depends_on or len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError("cyclic or duplicate assignment dependency")


@dataclass(frozen=True, slots=True)
class CoordinatorDecision:
    action: CoordinatorAction
    agent_ids: tuple[str, ...] = ()
    role: str = ""
    instruction: str = ""
    depends_on: tuple[str, ...] = ()
    done_checks: Mapping[str, bool] = field(default_factory=dict)
    reason_code: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_ids", tuple(self.agent_ids))
        object.__setattr__(self, "depends_on", tuple(self.depends_on))
        object.__setattr__(self, "done_checks", MappingProxyType(dict(self.done_checks)))
        if len(self.agent_ids) > MAX_TEAM_FANOUT or len(set(self.agent_ids)) != len(self.agent_ids):
            raise ValueError("coordinator fanout bound exceeded")
        if any(re.fullmatch(r"agt_[A-Za-z0-9_.:-]+", item) is None for item in self.agent_ids):
            raise ValueError("invalid coordinator agent ID")
        if self.action in {CoordinatorAction.ASSIGN, CoordinatorAction.REVIEW, CoordinatorAction.REVISE}:
            if not self.agent_ids or not self.instruction.strip():
                raise ValueError("coordinator assignment is incomplete")
        if self.action is CoordinatorAction.COMPLETE:
            if not self.done_checks or not all(self.done_checks.values()):
                raise ValueError("coordinator cannot forge completion")
        if self.action is CoordinatorAction.BLOCK and not self.reason_code:
            raise ValueError("blocked coordinator decision requires a reason")


@dataclass(frozen=True, slots=True)
class TeamProjection:
    runs: Mapping[str, TeamRunV2] = field(default_factory=dict)
    assignments: Mapping[str, TeamAssignmentV2] = field(default_factory=dict)
    effects: Mapping[tuple[str, str], str] = field(default_factory=dict)
    collaboration_events: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "runs", MappingProxyType(dict(self.runs)))
        object.__setattr__(self, "assignments", MappingProxyType(dict(self.assignments)))
        object.__setattr__(self, "effects", MappingProxyType(dict(self.effects)))
        object.__setattr__(
            self,
            "collaboration_events",
            MappingProxyType(dict(self.collaboration_events)),
        )


__all__ = [
    "CoordinatorAction",
    "CoordinatorDecision",
    "MAX_TEAM_ASSIGNMENTS",
    "MAX_TEAM_FANOUT",
    "MAX_TEAM_HANDOFFS",
    "MAX_TEAM_TURNS",
    "TeamAssignmentStatus",
    "TeamAssignmentV2",
    "TeamProjection",
    "TeamRunPhase",
    "TeamRunV2",
]
