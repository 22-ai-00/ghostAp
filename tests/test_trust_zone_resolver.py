from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from src.access_control import AccessDecision, AccessOperation
from src.feishu.route_decision import RouteDecision, RouteTarget
from src.trust.models import (
    ActorKind,
    EffectiveTrust,
    ManagedGroupOrigin,
    ManagedGroupRecord,
    ManagedGroupStatus,
    ProjectGrant,
    TrustZone,
)
from src.trust.resolver import TrustZoneResolver

OWNER_ID = "ou_owner"
EMPLOYEE_ID = "ou_employee_bot"
GROUP_ID = "oc_managed"


def _grant(*, revision: int = 11) -> ProjectGrant:
    return ProjectGrant(
        grant_id="grant-1",
        revision=revision,
        owner_id=OWNER_ID,
        managed_group_id=GROUP_ID,
        project_id="project-1",
        canonical_root_ref="/work/project-1",
        backend_binding_ids=("codex-main",),
        connected_target_refs=("github-origin",),
    )


def _group(
    *,
    revision: int = 7,
    status: ManagedGroupStatus = ManagedGroupStatus.ACTIVE,
) -> ManagedGroupRecord:
    return ManagedGroupRecord(
        chat_id=GROUP_ID,
        revision=revision,
        status=status,
        owner_id=OWNER_ID,
        project_id="project-1",
        canonical_root_ref="/work/project-1",
        origin=ManagedGroupOrigin.GHOSTAP_CREATED,
        receiving_bot_ref="bot-main",
        project_grant_id="grant-1",
        created_at=datetime(2026, 7, 31, tzinfo=UTC),
    )


def _resolver(
    *,
    group: ManagedGroupRecord | None = None,
    grant: ProjectGrant | None = None,
) -> TrustZoneResolver:
    return TrustZoneResolver(
        owner_id=OWNER_ID,
        managed_groups=((group or _group()),),
        project_grants=((grant or _grant()),),
        employee_bot_ids=frozenset({EMPLOYEE_ID}),
    )


def test_owner_p2p_and_managed_group_owner_resolve_to_distinct_zones() -> None:
    resolver = _resolver()

    owner_p2p = resolver.resolve(
        sender_id=OWNER_ID,
        chat_id="oc_owner_p2p",
        chat_type="p2p",
    )
    managed_owner = resolver.resolve(
        sender_id=OWNER_ID,
        chat_id=GROUP_ID,
        chat_type="group",
    )

    assert owner_p2p == EffectiveTrust(
        zone=TrustZone.OWNER_P2P,
        actor=ActorKind.OWNER,
        managed_group=None,
        group_revision=None,
        grant_revision=None,
    )
    assert managed_owner == EffectiveTrust(
        zone=TrustZone.MANAGED_AGENT_GROUP,
        actor=ActorKind.OWNER,
        managed_group=_group(),
        group_revision=7,
        grant_revision=11,
    )
    assert resolver.resolve(
        sender_id=OWNER_ID,
        chat_id=GROUP_ID,
        chat_type="group",
    ) == managed_owner


def test_managed_employee_and_unknown_sources_resolve_fail_closed() -> None:
    resolver = _resolver()

    employee = resolver.resolve(
        sender_id=EMPLOYEE_ID,
        chat_id=GROUP_ID,
        chat_type="group",
    )
    unknown_member = resolver.resolve(
        sender_id="ou_stranger",
        chat_id=GROUP_ID,
        chat_type="group",
    )
    employee_p2p = resolver.resolve(
        sender_id=EMPLOYEE_ID,
        chat_id="oc_employee_p2p",
        chat_type="p2p",
    )

    assert employee.zone is TrustZone.MANAGED_AGENT_GROUP
    assert employee.actor is ActorKind.EMPLOYEE
    assert unknown_member.zone is TrustZone.EXTERNAL_OR_UNKNOWN_GROUP
    assert unknown_member.actor is ActorKind.UNKNOWN
    assert unknown_member.managed_group is None
    assert employee_p2p.zone is TrustZone.EXTERNAL_OR_UNKNOWN_GROUP
    assert employee_p2p.actor is ActorKind.UNKNOWN


@pytest.mark.parametrize(
    ("group", "grant"),
    [
        (_group(status=ManagedGroupStatus.TOMBSTONED), _grant()),
        (_group(), _grant(revision=0)),
        (_group(), None),
    ],
)
def test_inactive_or_invalid_managed_group_provenance_is_unknown(
    group: ManagedGroupRecord,
    grant: ProjectGrant | None,
) -> None:
    resolver = TrustZoneResolver(
        owner_id=OWNER_ID,
        managed_groups=(group,),
        project_grants=(() if grant is None else (grant,)),
        employee_bot_ids=frozenset({EMPLOYEE_ID}),
    )

    trust = resolver.resolve(
        sender_id=OWNER_ID,
        chat_id=GROUP_ID,
        chat_type="group",
    )

    assert trust.zone is TrustZone.EXTERNAL_OR_UNKNOWN_GROUP
    assert trust.actor is ActorKind.UNKNOWN


def test_effective_trust_is_immutable_in_access_and_route_contexts() -> None:
    trust = _resolver().resolve(
        sender_id=OWNER_ID,
        chat_id=GROUP_ID,
        chat_type="group",
    )
    access = AccessDecision(
        allowed=True,
        operation=AccessOperation.NORMAL_MESSAGE,
        reason_code="trusted_zone",
        prospective_allowed=True,
        effective_trust=trust,
    )
    route = RouteDecision(
        target=RouteTarget.PROGRAMMING_MODE,
        effective_trust=trust,
    )

    assert access.effective_trust is trust
    assert route.effective_trust is trust
    with pytest.raises(FrozenInstanceError):
        trust.group_revision = 8  # type: ignore[misc]
