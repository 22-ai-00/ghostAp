from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.trust.action_matrix import ActionMatrix, can_dispatch
from src.trust.models import (
    ActionDecision,
    ActionKind,
    ActionRequest,
    ActionTargetKind,
    ActorKind,
    EffectiveTrust,
    EmployeeAssignment,
    EmployeeCausalContext,
    ManagedGroupOrigin,
    ManagedGroupRecord,
    ManagedGroupStatus,
    ProjectGrant,
    TrustZone,
)

OWNER_ID = "ou_owner"
GROUP_ID = "oc_managed"


def _grant() -> ProjectGrant:
    return ProjectGrant(
        grant_id="grant-1",
        revision=11,
        owner_id=OWNER_ID,
        managed_group_id=GROUP_ID,
        project_id="project-1",
        canonical_root_ref="/work/project-1",
        backend_binding_ids=("codex-main", "claude-main"),
        connected_target_refs=("github-origin",),
    )


def _group() -> ManagedGroupRecord:
    return ManagedGroupRecord(
        chat_id=GROUP_ID,
        revision=7,
        status=ManagedGroupStatus.ACTIVE,
        owner_id=OWNER_ID,
        project_id="project-1",
        canonical_root_ref="/work/project-1",
        origin=ManagedGroupOrigin.GHOSTAP_CREATED,
        receiving_bot_ref="bot-main",
        project_grant_id="grant-1",
        created_at=datetime(2026, 7, 31, tzinfo=UTC),
    )


def _trust(actor: ActorKind = ActorKind.OWNER) -> EffectiveTrust:
    return EffectiveTrust(
        zone=TrustZone.MANAGED_AGENT_GROUP,
        actor=actor,
        managed_group=_group(),
        group_revision=7,
        grant_revision=11,
    )


def _owner_p2p() -> EffectiveTrust:
    return EffectiveTrust(
        zone=TrustZone.OWNER_P2P,
        actor=ActorKind.OWNER,
        managed_group=None,
        group_revision=None,
        grant_revision=None,
    )


def _request(
    action: ActionKind,
    target: ActionTargetKind,
    *,
    trust: EffectiveTrust | None = None,
    **kwargs: object,
) -> ActionRequest:
    return ActionRequest(
        trust=trust or _trust(),
        action=action,
        target=target,
        grant=_grant(),
        **kwargs,
    )


def test_action_matrix_has_only_allow_or_deny_never_ask() -> None:
    matrix = ActionMatrix()
    decisions = {
        matrix.decide(
            ActionRequest(
                trust=EffectiveTrust(
                    zone=zone,
                    actor=actor,
                    managed_group=None,
                    group_revision=None,
                    grant_revision=None,
                ),
                action=action,
                target=target,
            )
        )
        for zone in TrustZone
        for actor in ActorKind
        for action in ActionKind
        for target in ActionTargetKind
    }

    assert set(ActionDecision) == {
        ActionDecision.ALLOW,
        ActionDecision.DENY,
    }
    assert decisions <= set(ActionDecision)
    assert "ask" not in {decision.value for decision in ActionDecision}


@pytest.mark.parametrize(
    "action",
    [
        ActionKind.PROJECT_READ,
        ActionKind.PROJECT_WRITE,
        ActionKind.LOCAL_GIT,
        ActionKind.TRIGGER_CONTROL,
    ],
)
def test_managed_project_actions_need_no_approval(action: ActionKind) -> None:
    decision = ActionMatrix().decide(
        _request(
            action,
            ActionTargetKind.CURRENT_PROJECT,
            canonical_root_ref="/work/project-1",
        )
    )

    assert decision is ActionDecision.ALLOW


def test_configured_managed_project_tool_is_allowed() -> None:
    decision = ActionMatrix().decide(
        _request(
            ActionKind.PROJECT_TOOL,
            ActionTargetKind.CURRENT_PROJECT,
            canonical_root_ref="/work/project-1",
            backend_binding_id="codex-main",
        )
    )

    assert decision is ActionDecision.ALLOW


def test_owner_explicit_external_action_in_managed_group_is_not_reconfirmed() -> None:
    matrix = ActionMatrix()

    explicit = matrix.decide(
        _request(
            ActionKind.EXTERNAL_MUTATION,
            ActionTargetKind.EXTERNAL,
            owner_explicit=True,
        )
    )
    automatic_registered = matrix.decide(
        _request(
            ActionKind.EXTERNAL_MUTATION,
            ActionTargetKind.CONNECTED_TARGET,
            connected_target_ref="github-origin",
        )
    )
    automatic_unregistered = matrix.decide(
        _request(
            ActionKind.EXTERNAL_MUTATION,
            ActionTargetKind.EXTERNAL,
        )
    )

    assert explicit is ActionDecision.ALLOW
    assert automatic_registered is ActionDecision.ALLOW
    assert automatic_unregistered is ActionDecision.DENY


@pytest.mark.parametrize(
    "action_request",
    [
        _request(
            ActionKind.PROJECT_WRITE,
            ActionTargetKind.CURRENT_PROJECT,
            canonical_root_ref="/work/other",
        ),
        _request(
            ActionKind.PROJECT_TOOL,
            ActionTargetKind.CURRENT_PROJECT,
            canonical_root_ref="/work/project-1",
            backend_binding_id="unregistered-backend",
        ),
        _request(
            ActionKind.EXTERNAL_MUTATION,
            ActionTargetKind.CONNECTED_TARGET,
            connected_target_ref="unregistered-target",
        ),
        _request(ActionKind.GRANT_ADMIN, ActionTargetKind.HOST_GLOBAL),
        _request(ActionKind.BACKEND_ADMIN, ActionTargetKind.HOST_GLOBAL),
        _request(ActionKind.CREDENTIAL_ADMIN, ActionTargetKind.HOST_GLOBAL),
        _request(ActionKind.SYSTEM_ADMIN, ActionTargetKind.HOST_GLOBAL),
    ],
)
def test_managed_group_cannot_expand_root_backend_or_connected_targets(
    action_request: ActionRequest,
) -> None:
    assert ActionMatrix().decide(action_request) is ActionDecision.DENY


@pytest.mark.parametrize(
    "action",
    [
        ActionKind.HOST_SHELL,
        ActionKind.GRANT_ADMIN,
        ActionKind.BACKEND_ADMIN,
        ActionKind.CREDENTIAL_ADMIN,
        ActionKind.SYSTEM_ADMIN,
    ],
)
def test_original_host_shell_and_grant_admin_are_owner_p2p_only(
    action: ActionKind,
) -> None:
    matrix = ActionMatrix()

    owner_p2p = matrix.decide(
        ActionRequest(
            trust=_owner_p2p(),
            action=action,
            target=ActionTargetKind.HOST_GLOBAL,
        )
    )
    managed_owner = matrix.decide(
        _request(action, ActionTargetKind.HOST_GLOBAL)
    )

    assert owner_p2p is ActionDecision.ALLOW
    assert managed_owner is ActionDecision.DENY


def test_employee_bot_requires_existing_causal_assignment() -> None:
    matrix = ActionMatrix()
    existing = EmployeeAssignment(run_id="run-1", assignment_id="assignment-1")
    causal = EmployeeCausalContext(
        run_id="run-1",
        assignment_id="assignment-1",
        causal_event_id="event-1",
    )
    base = dict(
        trust=_trust(ActorKind.EMPLOYEE),
        action=ActionKind.PROJECT_WRITE,
        target=ActionTargetKind.CURRENT_PROJECT,
        grant=_grant(),
        canonical_root_ref="/work/project-1",
    )

    assert matrix.decide(ActionRequest(**base)) is ActionDecision.DENY
    assert matrix.decide(
        ActionRequest(
            **base,
            employee_assignment=EmployeeAssignment(
                run_id="run-other",
                assignment_id="assignment-1",
            ),
            employee_causal_context=causal,
        )
    ) is ActionDecision.DENY
    assert matrix.decide(
        ActionRequest(
            **base,
            employee_assignment=existing,
            employee_causal_context=causal,
        )
    ) is ActionDecision.ALLOW
    assert matrix.decide(
        ActionRequest(
            **base,
            employee_assignment=EmployeeAssignment(
                run_id="",
                assignment_id="",
            ),
            employee_causal_context=EmployeeCausalContext(
                run_id="",
                assignment_id="",
                causal_event_id="event-1",
            ),
        )
    ) is ActionDecision.DENY
    assert matrix.decide(
        ActionRequest(
            trust=_trust(ActorKind.EMPLOYEE),
            action=ActionKind.GRANT_ADMIN,
            target=ActionTargetKind.HOST_GLOBAL,
            grant=_grant(),
            employee_assignment=existing,
            employee_causal_context=causal,
        )
    ) is ActionDecision.DENY


@pytest.mark.parametrize(
    ("group_revision", "grant_revision", "killed", "paused"),
    [
        (8, 11, False, False),
        (7, 12, False, False),
        (7, 11, True, False),
        (7, 11, False, True),
    ],
)
def test_stale_group_or_grant_revision_cannot_dispatch(
    group_revision: int,
    grant_revision: int,
    killed: bool,
    paused: bool,
) -> None:
    assert can_dispatch(
        _trust(),
        current_group_revision=group_revision,
        current_grant_revision=grant_revision,
        killed=killed,
        paused=paused,
    ) is ActionDecision.DENY


def test_matching_live_revisions_can_dispatch_without_user_interaction() -> None:
    assert can_dispatch(
        _trust(),
        current_group_revision=7,
        current_grant_revision=11,
        killed=False,
        paused=False,
    ) is ActionDecision.ALLOW
