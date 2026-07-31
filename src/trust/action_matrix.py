"""Zero-prompt ALLOW/DENY action policy and stale-dispatch fence."""

from __future__ import annotations

from .models import (
    ActionDecision,
    ActionKind,
    ActionRequest,
    ActionTargetKind,
    ActorKind,
    EffectiveTrust,
    EmployeeAssignment,
    EmployeeCausalContext,
    ProjectGrant,
    TriggerControlScope,
    TrustZone,
)

_PROJECT_ACTIONS = frozenset(
    {
        ActionKind.PROJECT_READ,
        ActionKind.PROJECT_WRITE,
        ActionKind.LOCAL_GIT,
    }
)
_OWNER_P2P_ACTIONS = frozenset(
    {
        ActionKind.GRANT_ADMIN,
        ActionKind.BACKEND_ADMIN,
        ActionKind.CREDENTIAL_ADMIN,
        ActionKind.HOST_SHELL,
        ActionKind.SYSTEM_ADMIN,
    }
)


class ActionMatrix:
    """Decide runtime-confirmed action facts without an approval state."""

    def decide(self, request: ActionRequest) -> ActionDecision:
        trust = request.trust
        if _is_owner_p2p(trust):
            return self._decide_owner_p2p(request)
        if not _is_managed_actor(trust):
            return ActionDecision.DENY
        if not _grant_matches_trust(request.grant, trust):
            return ActionDecision.DENY
        if trust.actor is ActorKind.EMPLOYEE and not _continues_assignment(
            request.employee_assignment,
            request.employee_causal_context,
        ):
            return ActionDecision.DENY
        return self._decide_managed(request)

    @staticmethod
    def _decide_owner_p2p(request: ActionRequest) -> ActionDecision:
        if (
            request.action in _PROJECT_ACTIONS
            and request.target is ActionTargetKind.CURRENT_PROJECT
        ):
            return ActionDecision.ALLOW
        if (
            request.action is ActionKind.PROJECT_TOOL
            and request.target is ActionTargetKind.CURRENT_PROJECT
            and bool(request.canonical_root_ref)
            and bool(request.backend_binding_id)
        ):
            return ActionDecision.ALLOW
        if (
            request.action is ActionKind.TRIGGER_CONTROL
            and request.target is ActionTargetKind.CURRENT_PROJECT
            and request.trigger_control_scope is not None
        ):
            return ActionDecision.ALLOW
        if (
            request.action in _OWNER_P2P_ACTIONS
            and request.target is ActionTargetKind.HOST_GLOBAL
        ):
            return ActionDecision.ALLOW
        if (
            request.action is ActionKind.EXTERNAL_MUTATION
            and request.target is ActionTargetKind.EXTERNAL
            and request.owner_explicit
        ):
            return ActionDecision.ALLOW
        return ActionDecision.DENY

    @staticmethod
    def _decide_managed(request: ActionRequest) -> ActionDecision:
        grant = request.grant
        if grant is None:
            return ActionDecision.DENY
        if request.action in _PROJECT_ACTIONS:
            if (
                request.target is ActionTargetKind.CURRENT_PROJECT
                and request.canonical_root_ref == grant.canonical_root_ref
            ):
                return ActionDecision.ALLOW
            return ActionDecision.DENY
        if request.action is ActionKind.TRIGGER_CONTROL:
            if (
                request.target is not ActionTargetKind.CURRENT_PROJECT
                or request.canonical_root_ref != grant.canonical_root_ref
                or request.trigger_control_scope is None
            ):
                return ActionDecision.DENY
            if request.trust.actor is ActorKind.OWNER:
                return ActionDecision.ALLOW
            if request.trigger_control_scope in {
                TriggerControlScope.DRAFT,
                TriggerControlScope.CURRENT_RUN,
            }:
                return ActionDecision.ALLOW
            return ActionDecision.DENY
        if request.action is ActionKind.PROJECT_TOOL:
            if (
                request.target is ActionTargetKind.CURRENT_PROJECT
                and request.canonical_root_ref == grant.canonical_root_ref
                and request.backend_binding_id in grant.backend_binding_ids
            ):
                return ActionDecision.ALLOW
            return ActionDecision.DENY
        if request.action is ActionKind.EXTERNAL_MUTATION:
            if (
                request.trust.actor is ActorKind.OWNER
                and request.owner_explicit
                and request.target is ActionTargetKind.EXTERNAL
            ):
                return ActionDecision.ALLOW
            if (
                request.target is ActionTargetKind.CONNECTED_TARGET
                and request.connected_target_ref
                in grant.connected_target_refs
            ):
                return ActionDecision.ALLOW
        return ActionDecision.DENY


def can_dispatch(
    trust: EffectiveTrust,
    *,
    current_group_revision: int | None,
    current_grant_revision: int | None,
    killed: bool,
    paused: bool,
) -> ActionDecision:
    """Silently fence stale, killed, paused, or unknown dispatches."""

    if killed or paused:
        return ActionDecision.DENY
    if _is_owner_p2p(trust):
        revisions_match = (
            trust.group_revision is None
            and trust.grant_revision is None
            and current_group_revision is None
            and current_grant_revision is None
        )
        return (
            ActionDecision.ALLOW
            if revisions_match
            else ActionDecision.DENY
        )
    if not _is_managed_actor(trust):
        return ActionDecision.DENY
    if (
        current_group_revision != trust.group_revision
        or current_grant_revision != trust.grant_revision
    ):
        return ActionDecision.DENY
    return ActionDecision.ALLOW


def _is_owner_p2p(trust: EffectiveTrust) -> bool:
    return (
        trust.zone is TrustZone.OWNER_P2P
        and trust.actor is ActorKind.OWNER
        and trust.managed_group is None
    )


def _is_managed_actor(trust: EffectiveTrust) -> bool:
    return (
        trust.zone is TrustZone.MANAGED_AGENT_GROUP
        and trust.actor in {ActorKind.OWNER, ActorKind.EMPLOYEE}
        and trust.managed_group is not None
        and trust.group_revision == trust.managed_group.revision
        and trust.grant_revision is not None
    )


def _grant_matches_trust(
    grant: ProjectGrant | None,
    trust: EffectiveTrust,
) -> bool:
    group = trust.managed_group
    return (
        grant is not None
        and group is not None
        and grant.grant_id == group.project_grant_id
        and grant.revision == trust.grant_revision
        and grant.owner_id == group.owner_id
        and grant.managed_group_id == group.chat_id
        and grant.project_id == group.project_id
        and grant.canonical_root_ref == group.canonical_root_ref
    )


def _continues_assignment(
    assignment: EmployeeAssignment | None,
    causal_context: EmployeeCausalContext | None,
) -> bool:
    return (
        assignment is not None
        and causal_context is not None
        and bool(assignment.run_id)
        and bool(assignment.assignment_id)
        and bool(causal_context.causal_event_id)
        and causal_context.run_id == assignment.run_id
        and causal_context.assignment_id == assignment.assignment_id
    )
