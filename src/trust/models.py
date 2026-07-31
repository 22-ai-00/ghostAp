"""Immutable trust and authorization facts for Agent Platform actions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class TrustZone(StrEnum):
    OWNER_P2P = "owner_p2p"
    MANAGED_AGENT_GROUP = "managed_agent_group"
    EXTERNAL_OR_UNKNOWN_GROUP = "external_or_unknown_group"


class ActorKind(StrEnum):
    OWNER = "owner"
    EMPLOYEE = "employee"
    UNKNOWN = "unknown"


class ActionKind(StrEnum):
    PROJECT_READ = "project_read"
    PROJECT_WRITE = "project_write"
    PROJECT_TOOL = "project_tool"
    LOCAL_GIT = "local_git"
    TRIGGER_CONTROL = "trigger_control"
    EXTERNAL_MUTATION = "external_mutation"
    GRANT_ADMIN = "grant_admin"
    BACKEND_ADMIN = "backend_admin"
    CREDENTIAL_ADMIN = "credential_admin"
    HOST_SHELL = "host_shell"
    SYSTEM_ADMIN = "system_admin"


class ActionTargetKind(StrEnum):
    CURRENT_PROJECT = "current_project"
    CONNECTED_TARGET = "connected_target"
    EXTERNAL = "external"
    HOST_GLOBAL = "host_global"


class ActionDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"


class TriggerControlScope(StrEnum):
    DRAFT = "draft"
    CURRENT_RUN = "current_run"
    PERMANENT = "permanent"


class ManagedGroupStatus(StrEnum):
    ACTIVE = "active"
    TOMBSTONED = "tombstoned"


class ManagedGroupOrigin(StrEnum):
    GHOSTAP_CREATED = "ghostap_created"
    EMPLOYEE_CREATED = "employee_created"
    OWNER_ADOPTED = "owner_adopted"


@dataclass(frozen=True, slots=True)
class ProjectGrant:
    grant_id: str
    revision: int
    owner_id: str
    managed_group_id: str
    project_id: str
    canonical_root_ref: str
    backend_binding_ids: tuple[str, ...]
    connected_target_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ManagedGroupRecord:
    chat_id: str
    revision: int
    status: ManagedGroupStatus
    owner_id: str
    project_id: str
    canonical_root_ref: str
    origin: ManagedGroupOrigin
    receiving_bot_ref: str
    project_grant_id: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class EffectiveTrust:
    zone: TrustZone
    actor: ActorKind
    managed_group: ManagedGroupRecord | None
    group_revision: int | None
    grant_revision: int | None


@dataclass(frozen=True, slots=True)
class EmployeeAssignment:
    """Existing assignment that authorizes one Employee continuation."""

    run_id: str
    assignment_id: str


@dataclass(frozen=True, slots=True)
class EmployeeCausalContext:
    """Causal identity carried by an Employee collaboration event."""

    run_id: str
    assignment_id: str
    causal_event_id: str


@dataclass(frozen=True, slots=True)
class ActionRequest:
    """Runtime-confirmed action facts consumed by the pure action matrix."""

    trust: EffectiveTrust
    action: ActionKind
    target: ActionTargetKind
    grant: ProjectGrant | None = None
    canonical_root_ref: str | None = None
    backend_binding_id: str | None = None
    connected_target_ref: str | None = None
    owner_explicit: bool = False
    trigger_control_scope: TriggerControlScope | None = None
    employee_assignment: EmployeeAssignment | None = None
    employee_causal_context: EmployeeCausalContext | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.trust, EffectiveTrust):
            raise TypeError("trust must be EffectiveTrust")
        if not isinstance(self.action, ActionKind):
            raise TypeError("action must be a runtime-confirmed ActionKind")
        if not isinstance(self.target, ActionTargetKind):
            raise TypeError("target must be a runtime-confirmed ActionTargetKind")
        if not isinstance(self.owner_explicit, bool):
            raise TypeError("owner_explicit must be bool")
        if (
            self.trigger_control_scope is not None
            and not isinstance(
                self.trigger_control_scope,
                TriggerControlScope,
            )
        ):
            raise TypeError(
                "trigger_control_scope must be TriggerControlScope"
            )
