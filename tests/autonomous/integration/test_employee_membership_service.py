from __future__ import annotations

import threading
from dataclasses import dataclass

import pytest

from src.autonomous.domain import EmployeeState
from src.autonomous.journal.frame import JournalEvent
from src.autonomous.journal.projections import ProjectionState
from src.autonomous.membership.lark import MembershipRemoteUnknown
from src.autonomous.membership.models import MembershipOperation, MembershipState
from src.autonomous.membership.service import (
    EmployeeMembershipService,
    MembershipAuthorizationError,
    MembershipBindingError,
    MembershipMutationRequest,
)
from src.autonomous.provisioning.hire_service import ProductionEmployeeHireService
from src.autonomous.workforce.projection import commit_workforce_events
from tests.autonomous.workforce_helpers import bot_binding_events, employee_created, make_writer


class _Remote:
    def __init__(self, *, observed: bool = False) -> None:
        self.observed = observed
        self.mutations: list[tuple[MembershipOperation, str, str]] = []
        self.observations: list[tuple[str, str, str, str]] = []
        self.mutation_error: Exception | None = None
        self.observation_error: Exception | None = None
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.inflight = 0
        self.max_inflight = 0
        self._lock = threading.Lock()

    def mutate(self, operation, *, chat_id, app_id):
        with self._lock:
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            self.mutations.append((MembershipOperation(operation), chat_id, app_id))
            self.entered.set()
            if self.block:
                assert self.release.wait(2)
            if self.mutation_error:
                raise self.mutation_error
            self.observed = operation is MembershipOperation.ADD
            return True
        finally:
            with self._lock:
                self.inflight -= 1

    def is_member(self, *, chat_id, agent_id, app_id, credential_ref):
        self.observations.append((chat_id, agent_id, app_id, credential_ref))
        if self.observation_error:
            raise self.observation_error
        return self.observed


@dataclass
class _Fixture:
    service: EmployeeMembershipService
    remote: _Remote
    hire: ProductionEmployeeHireService
    writer: object


def _fixture(
    tmp_path,
    *,
    member_groups=(),
    admin_ids=frozenset({"ou_admin"}),
    team_active=True,
    employee_state=EmployeeState.ACTIVE,
) -> _Fixture:
    writer = make_writer(tmp_path)
    state = ProjectionState()
    created = employee_created()
    created = JournalEvent(
        event_type=created.event_type,
        aggregate_id=created.aggregate_id,
        payload={
            **created.payload,
            "state": employee_state.value,
            "member_groups": list(member_groups),
        },
    )
    commit_workforce_events(writer, state, (created,))
    commit_workforce_events(writer, state, bot_binding_events())
    hire = ProductionEmployeeHireService(writer, state)
    remote = _Remote(observed="oc_team" in member_groups)
    service = EmployeeMembershipService(
        writer=writer,
        hire_service=hire,
        remote=remote,
        admin_principal_ids=admin_ids,
        team_owner_resolver=lambda chat_id: "ou_owner" if chat_id == "oc_team" else "",
        team_active_resolver=lambda _chat_id: team_active,
    )
    return _Fixture(service, remote, hire, writer)


def _request(*, requester="ou_admin", operation=MembershipOperation.ADD):
    return MembershipMutationRequest(
        tenant_key="tenant_1",
        chat_id="oc_team",
        agent_id="agt_1",
        requester_principal_id=requester,
        operation=operation,
    )


@pytest.mark.parametrize("requester", ["ou_admin", "ou_owner"])
def test_add_commits_only_after_employee_bot_confirms_membership(tmp_path, requester) -> None:
    fx = _fixture(tmp_path)

    outcome = fx.service.mutate(_request(requester=requester))

    assert outcome.state is MembershipState.ACTIVE
    assert outcome.confirmed is True
    assert fx.remote.mutations == [(MembershipOperation.ADD, "oc_team", "cli_1")]
    assert fx.remote.observations[-1] == ("oc_team", "agt_1", "cli_1", "cred_1")
    employee = fx.hire.synchronize_projection().employees["agt_1"]
    assert employee.member_groups == ("oc_team",)
    assert fx.service.is_degraded("agt_1", "oc_team") is False


def test_batch_membership_health_rebuilds_projection_once(
    tmp_path,
    monkeypatch,
) -> None:
    fx = _fixture(tmp_path)
    fx.service.mutate(_request())
    rebuild_calls = 0
    rebuild_projection = fx.service.rebuild_projection

    def counted_rebuild():
        nonlocal rebuild_calls
        rebuild_calls += 1
        return rebuild_projection()

    monkeypatch.setattr(fx.service, "rebuild_projection", counted_rebuild)

    health = fx.service.degraded_for(
        ("agt_1", "agt_missing", "agt_1"),
        "oc_team",
    )

    assert health == {
        "agt_1": False,
        "agt_missing": True,
    }
    assert rebuild_calls == 1


def test_remove_only_removes_chat_membership_not_global_employee(tmp_path) -> None:
    fx = _fixture(tmp_path, member_groups=("oc_team", "oc_other"))

    outcome = fx.service.mutate(_request(operation=MembershipOperation.REMOVE))

    assert outcome.state is MembershipState.ABSENT
    employee = fx.hire.synchronize_projection().employees["agt_1"]
    assert employee.member_groups == ("oc_other",)
    assert employee.state is EmployeeState.ACTIVE
    assert fx.hire.projection_state.bot_principals["bot_1"].credential_ref == "cred_1"


def test_retirement_cleanup_is_admin_remove_only_and_does_not_require_live_slock(tmp_path) -> None:
    fx = _fixture(tmp_path, member_groups=("oc_team",), team_active=False)
    commit_workforce_events(
        fx.writer,
        fx.hire.synchronize_projection(),
        (
            JournalEvent(
                event_type="employee.state_changed",
                aggregate_id="agt_1",
                payload={"state": EmployeeState.RETIRING.value},
            ),
        ),
    )

    outcomes = fx.service.retire_all(
        tenant_key="tenant_1",
        agent_id="agt_1",
        requester_principal_id="ou_admin",
    )

    assert len(outcomes) == 1
    assert outcomes[0].confirmed is True
    employee = fx.hire.synchronize_projection().employees["agt_1"]
    assert employee.state is EmployeeState.RETIRING
    assert employee.member_groups == ()
    assert fx.remote.mutations == [(MembershipOperation.REMOVE, "oc_team", "cli_1")]


def test_unauthorized_mutation_is_rejected_before_remote_or_journal(tmp_path) -> None:
    fx = _fixture(tmp_path)
    before = fx.writer.get_last_frame().sequence

    with pytest.raises(MembershipAuthorizationError):
        fx.service.mutate(_request(requester="ou_intruder"))

    assert fx.remote.mutations == []
    assert fx.writer.get_last_frame().sequence == before


def test_inactive_team_is_rejected_even_for_admin(tmp_path) -> None:
    fx = _fixture(tmp_path, team_active=False)
    before = fx.writer.get_last_frame().sequence

    with pytest.raises(MembershipBindingError, match="team is not active"):
        fx.service.mutate(_request())

    assert fx.remote.mutations == []
    assert fx.writer.get_last_frame().sequence == before


def test_unknown_mutation_and_observation_fail_closed_as_degraded(tmp_path) -> None:
    fx = _fixture(tmp_path)
    fx.remote.mutation_error = TimeoutError("unknown")
    fx.remote.observation_error = MembershipRemoteUnknown(
        "membership_observation_unknown"
    )

    outcome = fx.service.mutate(_request())

    assert outcome.state is MembershipState.DEGRADED
    assert outcome.confirmed is False
    assert fx.service.is_degraded("agt_1", "oc_team") is True
    assert fx.hire.synchronize_projection().employees["agt_1"].member_groups == ()


def test_confirmed_mutation_survives_unavailable_followup_observation(tmp_path) -> None:
    fx = _fixture(tmp_path)
    fx.remote.observation_error = MembershipRemoteUnknown(
        "membership_observation_permission_denied"
    )

    outcome = fx.service.mutate(_request())

    assert outcome.state is MembershipState.ACTIVE
    assert outcome.confirmed is True
    assert outcome.changed is True
    assert fx.hire.synchronize_projection().employees["agt_1"].member_groups == (
        "oc_team",
    )

    event_outcome = fx.service.reconcile_event(
        tenant_key="tenant_1",
        chat_id="oc_team",
        agent_id="agt_1",
        app_id="cli_1",
        observed_is_member=True,
    )

    assert event_outcome.state is MembershipState.ACTIVE
    assert event_outcome.confirmed is True
    assert event_outcome.changed is False
    assert fx.service.is_degraded("agt_1", "oc_team") is False


def test_idempotent_projection_uses_durable_fact_without_remote_observation(tmp_path) -> None:
    fx = _fixture(tmp_path)
    first = fx.service.mutate(_request())
    assert first.state is MembershipState.ACTIVE
    fx.remote.observations.clear()
    fx.remote.mutations.clear()
    fx.remote.observation_error = MembershipRemoteUnknown(
        "membership_observation_unknown"
    )
    before = fx.writer.get_last_frame().sequence

    outcome = fx.service.mutate(_request())

    assert outcome.state is MembershipState.ACTIVE
    assert outcome.confirmed is True
    assert outcome.changed is False
    assert fx.remote.mutations == []
    assert fx.remote.observations == []
    assert fx.writer.get_last_frame().sequence == before


def test_replay_preserves_last_committed_membership_after_idempotent_unknown(
    tmp_path,
) -> None:
    fx = _fixture(tmp_path)
    committed = fx.service.mutate(_request())
    assert committed.state is MembershipState.ACTIVE

    authority = fx.service._resolve_authority(_request())
    effect = fx.service._prepare(_request(), authority)
    fx.service._mark_executing(effect.effect_id)
    fx.service._mark_action_required(
        effect.effect_id,
        "idempotency_observation_unknown",
    )

    restarted = EmployeeMembershipService(
        writer=fx.writer,
        hire_service=fx.hire,
        remote=fx.remote,
        admin_principal_ids=frozenset({"ou_admin"}),
        team_owner_resolver=lambda _chat: "ou_owner",
        team_active_resolver=lambda _chat: True,
    )

    record = restarted.get("tenant_1", "oc_team", "agt_1")
    assert record is not None
    assert record.state is MembershipState.ACTIVE
    assert restarted.is_degraded("agt_1", "oc_team") is False


def test_failed_remove_cannot_make_repeat_add_report_false_success(tmp_path) -> None:
    fx = _fixture(tmp_path)
    assert fx.service.mutate(_request()).state is MembershipState.ACTIVE
    fx.remote.mutation_error = TimeoutError("remove unknown")
    fx.remote.observation_error = MembershipRemoteUnknown(
        "membership_observation_unknown"
    )

    removed = fx.service.mutate(
        _request(operation=MembershipOperation.REMOVE)
    )
    assert removed.state is MembershipState.DEGRADED
    assert fx.hire.synchronize_projection().employees["agt_1"].member_groups == (
        "oc_team",
    )

    repeated_add = fx.service.mutate(_request())

    assert repeated_add.confirmed is False
    assert repeated_add.state is MembershipState.DEGRADED
    assert fx.service.is_degraded("agt_1", "oc_team") is True


def test_replay_does_not_heal_old_add_unknown_after_degraded_remove(
    tmp_path,
) -> None:
    fx = _fixture(tmp_path)
    assert fx.service.mutate(_request()).state is MembershipState.ACTIVE
    fx.remote.mutation_error = TimeoutError("remove unknown")
    fx.remote.observation_error = MembershipRemoteUnknown(
        "membership_observation_unknown"
    )
    assert fx.service.mutate(
        _request(operation=MembershipOperation.REMOVE)
    ).state is MembershipState.DEGRADED

    authority = fx.service._resolve_authority(_request())
    legacy_add = fx.service._prepare(_request(), authority)
    fx.service._mark_executing(legacy_add.effect_id)
    fx.service._mark_action_required(
        legacy_add.effect_id,
        "idempotency_observation_unknown",
    )

    restarted = EmployeeMembershipService(
        writer=fx.writer,
        hire_service=fx.hire,
        remote=fx.remote,
        admin_principal_ids=frozenset({"ou_admin"}),
        team_owner_resolver=lambda _chat: "ou_owner",
        team_active_resolver=lambda _chat: True,
    )

    record = restarted.get("tenant_1", "oc_team", "agt_1")
    assert record is not None
    assert record.state is MembershipState.DEGRADED
    assert restarted.is_degraded("agt_1", "oc_team") is True


def test_same_chat_mutations_are_serialized(tmp_path) -> None:
    fx = _fixture(tmp_path)
    fx.remote.block = True
    errors: list[BaseException] = []

    def run() -> None:
        try:
            fx.service.mutate(_request())
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=run)
    second = threading.Thread(target=run)
    first.start()
    assert fx.remote.entered.wait(1)
    second.start()
    fx.remote.release.set()
    first.join(2)
    second.join(2)

    assert errors == []
    assert fx.remote.max_inflight == 1
    assert fx.hire.synchronize_projection().employees["agt_1"].member_groups == (
        "oc_team",
    )


def test_restart_rebuilds_membership_projection(tmp_path) -> None:
    fx = _fixture(tmp_path)
    outcome = fx.service.mutate(_request())

    restarted = EmployeeMembershipService(
        writer=fx.writer,
        hire_service=fx.hire,
        remote=fx.remote,
        admin_principal_ids=frozenset({"ou_admin"}),
        team_owner_resolver=lambda _chat: "ou_owner",
        team_active_resolver=lambda _chat: True,
    )

    assert restarted.get("tenant_1", "oc_team", "agt_1").state is outcome.state


def test_startup_audit_removes_stale_projected_membership_without_remote_mutation(
    tmp_path,
) -> None:
    fx = _fixture(tmp_path, member_groups=("oc_team", "oc_other"))
    fx.remote.observed = False

    summary = fx.service.reconcile_projected_memberships()

    assert summary.checked == 2
    assert summary.removed == 2
    assert summary.degraded == 0
    assert fx.remote.mutations == []
    assert fx.hire.synchronize_projection().employees["agt_1"].member_groups == ()


@pytest.mark.parametrize(
    "employee_state",
    [EmployeeState.VALIDATING, EmployeeState.READY_PENDING_VERIFICATION],
)
def test_startup_audit_includes_channel_recovery_employee_states(
    tmp_path,
    employee_state,
) -> None:
    fx = _fixture(
        tmp_path,
        member_groups=("oc_team",),
        employee_state=employee_state,
    )
    fx.remote.observed = False

    summary = fx.service.reconcile_projected_memberships()

    assert summary.checked == 1
    assert summary.removed == 1
    assert fx.remote.mutations == []
    assert fx.hire.synchronize_projection().employees["agt_1"].member_groups == ()


def test_startup_audit_keeps_confirmed_remote_membership_without_writes(
    tmp_path,
) -> None:
    fx = _fixture(tmp_path, member_groups=("oc_team",))
    fx.remote.observed = True
    assert fx.service.mutate(_request()).confirmed is True
    before = fx.writer.get_last_frame().sequence

    summary = fx.service.reconcile_projected_memberships()

    assert summary.checked == 1
    assert summary.confirmed == 1
    assert summary.removed == 0
    assert summary.degraded == 0
    assert fx.remote.mutations == []
    assert fx.writer.get_last_frame().sequence == before


def test_startup_audit_degrades_unknown_membership_and_blocks_dispatch(
    tmp_path,
) -> None:
    fx = _fixture(tmp_path, member_groups=("oc_team",))
    fx.remote.observation_error = MembershipRemoteUnknown(
        "membership_observation_unknown"
    )

    summary = fx.service.reconcile_projected_memberships()

    assert summary.checked == 1
    assert summary.confirmed == 0
    assert summary.removed == 0
    assert summary.degraded == 1
    assert fx.remote.mutations == []
    assert fx.service.is_degraded("agt_1", "oc_team") is True
    assert fx.hire.synchronize_projection().employees["agt_1"].member_groups == (
        "oc_team",
    )


def test_recovery_observes_executing_effect_without_replaying_mutation(tmp_path) -> None:
    fx = _fixture(tmp_path)
    request = _request()
    authority = fx.service._resolve_authority(request)
    effect = fx.service._prepare(request, authority)
    fx.service._mark_executing(effect.effect_id)
    fx.remote.observed = True

    recovered = fx.service.recover_pending()

    assert recovered == 1
    assert fx.remote.mutations == []
    assert fx.service.get("tenant_1", "oc_team", "agt_1").state is MembershipState.ACTIVE


def test_recovery_marks_prepared_effect_degraded_without_dispatch(tmp_path) -> None:
    fx = _fixture(tmp_path)
    request = _request()
    authority = fx.service._resolve_authority(request)
    fx.service._prepare(request, authority)

    recovered = fx.service.recover_pending()

    assert recovered == 1
    assert fx.remote.mutations == []
    record = fx.service.get("tenant_1", "oc_team", "agt_1")
    assert record.state is MembershipState.DEGRADED
    assert record.error_code == "prepared_recovery_unknown"


def test_membership_event_reconciles_from_employee_observation_without_mutation(tmp_path) -> None:
    fx = _fixture(tmp_path)
    fx.remote.observed = True

    outcome = fx.service.reconcile_event(
        tenant_key="tenant_1",
        chat_id="oc_team",
        agent_id="agt_1",
        app_id="cli_1",
        observed_is_member=True,
    )

    assert outcome.state is MembershipState.ACTIVE
    assert outcome.confirmed is True
    assert fx.remote.mutations == []
    assert fx.hire.synchronize_projection().employees["agt_1"].member_groups == (
        "oc_team",
    )


def test_bot_deleted_event_removes_only_observed_chat(tmp_path) -> None:
    fx = _fixture(tmp_path, member_groups=("oc_team", "oc_other"))
    fx.remote.observed = False

    outcome = fx.service.reconcile_event(
        tenant_key="tenant_1",
        chat_id="oc_team",
        agent_id="agt_1",
        app_id="cli_1",
        observed_is_member=False,
    )

    assert outcome.state is MembershipState.ABSENT
    assert fx.remote.mutations == []
    assert fx.hire.synchronize_projection().employees["agt_1"].member_groups == (
        "oc_other",
    )


def test_correlated_employee_event_cannot_recover_degraded_membership_without_query(
    tmp_path,
) -> None:
    fx = _fixture(tmp_path)
    fx.remote.mutation_error = TimeoutError("unknown")
    fx.remote.observation_error = MembershipRemoteUnknown(
        "membership_observation_permission_denied"
    )
    degraded = fx.service.mutate(_request())
    assert degraded.state is MembershipState.DEGRADED

    outcome = fx.service.reconcile_event(
        tenant_key="tenant_1",
        chat_id="oc_team",
        agent_id="agt_1",
        app_id="cli_1",
        observed_is_member=True,
    )

    assert outcome.state is MembershipState.DEGRADED
    assert outcome.confirmed is False
    assert fx.hire.synchronize_projection().employees["agt_1"].member_groups == ()


def test_uncorrelated_employee_event_cannot_override_membership(tmp_path) -> None:
    fx = _fixture(tmp_path)
    fx.remote.observation_error = MembershipRemoteUnknown(
        "membership_observation_permission_denied"
    )

    with pytest.raises(MembershipBindingError, match="event evidence"):
        fx.service.reconcile_event(
            tenant_key="tenant_1",
            chat_id="oc_team",
            agent_id="agt_1",
            app_id="cli_other",
            observed_is_member=True,
        )

    assert fx.hire.synchronize_projection().employees["agt_1"].member_groups == ()
