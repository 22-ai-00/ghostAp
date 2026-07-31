"""Deterministic trust-zone resolution from immutable registry snapshots."""

from __future__ import annotations

from dataclasses import dataclass

from .models import (
    ActorKind,
    EffectiveTrust,
    ManagedGroupRecord,
    ManagedGroupStatus,
    ProjectGrant,
    TrustZone,
)


def _unknown_trust() -> EffectiveTrust:
    return EffectiveTrust(
        zone=TrustZone.EXTERNAL_OR_UNKNOWN_GROUP,
        actor=ActorKind.UNKNOWN,
        managed_group=None,
        group_revision=None,
        grant_revision=None,
    )


@dataclass(frozen=True, slots=True)
class TrustZoneResolver:
    """Resolve one ingress/callback against frozen group and grant facts."""

    owner_id: str
    managed_groups: tuple[ManagedGroupRecord, ...]
    project_grants: tuple[ProjectGrant, ...]
    employee_bot_ids: frozenset[str]

    def __post_init__(self) -> None:
        if not isinstance(self.managed_groups, tuple):
            raise TypeError("managed_groups must be a tuple")
        if not isinstance(self.project_grants, tuple):
            raise TypeError("project_grants must be a tuple")
        if not isinstance(self.employee_bot_ids, frozenset):
            raise TypeError("employee_bot_ids must be a frozenset")

    def resolve(
        self,
        *,
        sender_id: str,
        chat_id: str,
        chat_type: str,
    ) -> EffectiveTrust:
        """Return the same effective trust for the same snapshot and facts."""

        if chat_type == "p2p" and sender_id == self.owner_id:
            return EffectiveTrust(
                zone=TrustZone.OWNER_P2P,
                actor=ActorKind.OWNER,
                managed_group=None,
                group_revision=None,
                grant_revision=None,
            )
        if chat_type != "group":
            return _unknown_trust()

        groups = tuple(
            record
            for record in self.managed_groups
            if record.chat_id == chat_id
        )
        if len(groups) != 1:
            return _unknown_trust()
        group = groups[0]
        if (
            group.status is not ManagedGroupStatus.ACTIVE
            or group.revision <= 0
            or group.owner_id != self.owner_id
        ):
            return _unknown_trust()

        grants = tuple(
            grant
            for grant in self.project_grants
            if grant.grant_id == group.project_grant_id
        )
        if len(grants) != 1:
            return _unknown_trust()
        grant = grants[0]
        if not _grant_matches_group(grant, group):
            return _unknown_trust()

        if sender_id == group.owner_id:
            actor = ActorKind.OWNER
        elif sender_id in self.employee_bot_ids:
            actor = ActorKind.EMPLOYEE
        else:
            return _unknown_trust()
        return EffectiveTrust(
            zone=TrustZone.MANAGED_AGENT_GROUP,
            actor=actor,
            managed_group=group,
            group_revision=group.revision,
            grant_revision=grant.revision,
        )


def _grant_matches_group(
    grant: ProjectGrant,
    group: ManagedGroupRecord,
) -> bool:
    return (
        grant.revision > 0
        and grant.owner_id == group.owner_id
        and grant.managed_group_id == group.chat_id
        and grant.project_id == group.project_id
        and grant.canonical_root_ref == group.canonical_root_ref
    )
