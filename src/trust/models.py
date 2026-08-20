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
