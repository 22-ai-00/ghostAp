from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.autonomous.ingress.projection import IngressProjectionState
from src.autonomous.ingress.service import EmployeeIngressService
from src.autonomous.journal.blob_store import AesGcmEncryptionProvider, BlobStore
from src.autonomous.journal.frame import JournalEvent
from src.autonomous.journal.projections import ProjectionState, apply_frame
from src.autonomous.provisioning.external_mutation_gate import (
    EmployeeExternalMutationGate,
    ExternalMutationFenced,
    ExternalMutationKind,
)
from src.autonomous.provisioning.fire_authority import JournalFireAuthority
from src.autonomous.provisioning.fire_effects import (
    ExecutionQuiesceEffect,
    MembershipCleanupEffect,
)
from src.autonomous.provisioning.fire_service import (
    EmployeeFireRequest,
    EmployeeFireService,
    FireServiceError,
)
from src.autonomous.provisioning.fire_state import (
    FIRE_EFFECT_ORDER,
    FireCleanupMode,
    FireEffectState,
    FirePhase,
    rebuild_fire_projection,
)
from src.autonomous.provisioning.hire_state import (
    DurableHireState,
    HireEffectState,
)
from tests.autonomous.workforce_helpers import (
    bot_binding_events,
    commit_events,
    employee_created,
    make_writer,
)


class _HireProjectionOwner:
    def __init__(
        self,
        state: ProjectionState,
        *,
        hire_states: tuple[DurableHireState, ...] = (),
        before_locked_sync=None,
    ) -> None:
        self.projection_state = state
        self.hire_states = hire_states
        self.before_locked_sync = before_locked_sync

    @contextmanager
    def employee_dispatch_guard(self):
        yield

    def synchronize_projection(self):
        return self.projection_state

    def synchronize_projection_unlocked(self):
        if self.before_locked_sync is not None:
            callback, self.before_locked_sync = self.before_locked_sync, None
            callback()
        return self.projection_state

    def apply_committed_frame_unlocked(self, frame):
        apply_frame(self.projection_state, frame)

    def list_states(self):
        return self.hire_states


class _Effect:
    def __init__(self, name: str, calls: list[str]) -> None:
        self.name = name
        self.calls = calls

    def execute(self, _state):
        self.calls.append(self.name)

    def observe(self, _state):
        return True


def _active_bound_fire_authority(tmp_path):
    writer = make_writer(tmp_path)
    state = ProjectionState()
    commit_events(writer, state, employee_created())
    commit_events(writer, state, *bot_binding_events())
    commit_events(
        writer,
        state,
        JournalEvent(
            event_type="employee.state_changed",
            aggregate_id="agt_1",
            payload={"state": "active"},
        ),
    )
    ingress = EmployeeIngressService(
        writer=writer,
        blob_store=BlobStore(
            tmp_path / "blobs",
            AesGcmEncryptionProvider(
                lambda _key: b"fire-ingress-data-key-32-bytes!!"
            ),
        ),
        ingress_state=IngressProjectionState(),
        active_key_id="k1",
    )
    authority = JournalFireAuthority(
        writer=writer,
        hire_service=_HireProjectionOwner(state),
        ingress_service=ingress,
        admin_principal_ids=frozenset({"ou_admin"}),
    )
    return writer, state, ingress, authority


def test_reconcile_draining_is_constant_time_without_pending_drains() -> None:
    class _Writer:
        def committed_tail(self, _from_sequence):
            return SimpleNamespace(sequence=0, frame_hash=""), ()

    service = EmployeeFireService(
        writer=_Writer(),  # type: ignore[arg-type]
        authority=object(),  # type: ignore[arg-type]
        effects={name: _Effect(name, []) for name in FIRE_EFFECT_ORDER},
    )

    assert service.reconcile_draining() == ()
    assert service.reconcile_draining() == ()


def test_fire_fences_then_waits_for_inflight_external_mutation_before_admission(
    tmp_path,
) -> None:
    writer, state, ingress, authority = _active_bound_fire_authority(tmp_path)
    gate = EmployeeExternalMutationGate()
    lease = gate.acquire(
        "tenant_1",
        "agt_1",
        ExternalMutationKind.APP_REGISTRATION,
    )
    calls: list[str] = []
    service = EmployeeFireService(
        writer=writer,
        authority=authority,
        effects={name: _Effect(name, calls) for name in FIRE_EFFECT_ORDER},
        external_mutation_gate=gate,
        external_mutation_wait_seconds=1.0,
    )
    done = threading.Event()
    errors: list[BaseException] = []

    def run_fire() -> None:
        try:
            service.start_fire(
                EmployeeFireRequest(
                    employee="Atlas",
                    tenant_key="tenant_1",
                    message_id="om_fire_external_lease",
                    chat_id="oc_dm",
                    requester_principal_id="ou_admin",
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=run_fire)
    thread.start()
    for _attempt in range(100):
        if gate.is_fenced("tenant_1", "agt_1"):
            break
        threading.Event().wait(0.01)
    assert gate.is_fenced("tenant_1", "agt_1") is True
    assert done.is_set() is False
    assert state.employees["agt_1"].state.value == "active"
    assert all(
        event.event_type != "fire.requested"
        for frame in writer.replay()
        for event in frame.events
    )

    lease.release()
    assert done.wait(1.0)
    thread.join(timeout=1.0)
    assert errors == []
    assert state.employees["agt_1"].state.value == "archived"
    assert calls == list(FIRE_EFFECT_ORDER)
    ingress.close()
    writer.close()


def test_fire_external_mutation_wait_timeout_does_not_admit_retirement(
    tmp_path,
) -> None:
    writer, state, ingress, authority = _active_bound_fire_authority(tmp_path)
    gate = EmployeeExternalMutationGate()
    lease = gate.acquire(
        "tenant_1",
        "agt_1",
        ExternalMutationKind.CREDENTIAL_PUT,
    )
    service = EmployeeFireService(
        writer=writer,
        authority=authority,
        effects={name: _Effect(name, []) for name in FIRE_EFFECT_ORDER},
        external_mutation_gate=gate,
        external_mutation_wait_seconds=0.01,
    )
    request = EmployeeFireRequest(
        employee="Atlas",
        tenant_key="tenant_1",
        message_id="om_fire_external_timeout",
        chat_id="oc_dm",
        requester_principal_id="ou_admin",
    )

    with pytest.raises(FireServiceError, match="external mutation is still active"):
        service.start_fire(request)

    assert state.employees["agt_1"].state.value == "active"
    assert all(
        event.event_type != "fire.requested"
        for frame in writer.replay()
        for event in frame.events
    )
    assert any(
        event.event_type == "employee.external_mutation_fenced"
        and event.payload.get("tenant_key") == "tenant_1"
        and event.payload.get("agent_id") == "agt_1"
        and event.payload.get("message_id") == request.message_id
        for frame in writer.replay()
        for event in frame.events
    )
    restarted_gate = EmployeeExternalMutationGate()
    EmployeeFireService(
        writer=writer,
        authority=authority,
        effects={name: _Effect(name, []) for name in FIRE_EFFECT_ORDER},
        external_mutation_gate=restarted_gate,
        external_mutation_wait_seconds=0.01,
    )
    assert restarted_gate.is_fenced("tenant_1", "agt_1") is True
    with pytest.raises(ExternalMutationFenced):
        restarted_gate.acquire(
            "tenant_1",
            "agt_1",
            ExternalMutationKind.CHANNEL_START,
        )
    lease.release()
    assert service.start_fire(request).phase is FirePhase.ARCHIVED
    ingress.close()
    writer.close()


@pytest.mark.parametrize(
    ("effect_id", "effect_type"),
    (
        ("slash-reconcile:2:1", "slash_reconciliation"),
        ("channel-start:2", "employee_channel_start"),
    ),
)
def test_restart_marker_does_not_bypass_unresolved_external_hire_effect(
    tmp_path,
    effect_id,
    effect_type,
) -> None:
    writer, state, ingress, _authority = _active_bound_fire_authority(tmp_path)
    hire = _HireProjectionOwner(state)
    authority = JournalFireAuthority(
        writer=writer,
        hire_service=hire,
        ingress_service=ingress,
        admin_principal_ids=frozenset({"ou_admin"}),
    )
    gate = EmployeeExternalMutationGate()
    lease = gate.acquire(
        "tenant_1",
        "agt_1",
        ExternalMutationKind.CREDENTIAL_PUT,
    )
    request = EmployeeFireRequest(
        employee="Atlas",
        tenant_key="tenant_1",
        message_id="om_fire_marker_unresolved_hire",
        chat_id="oc_dm",
        requester_principal_id="ou_admin",
    )
    service = EmployeeFireService(
        writer=writer,
        authority=authority,
        effects={name: _Effect(name, []) for name in FIRE_EFFECT_ORDER},
        external_mutation_gate=gate,
        external_mutation_wait_seconds=0.01,
    )

    with pytest.raises(FireServiceError, match="external mutation is still active"):
        service.start_fire(request)

    base = _pre_binding_hire_state(
        register_state=HireEffectState.COMMITTED,
        credential_committed=True,
    )
    hire.hire_states = (
        replace(
            base,
            effects=(
                *base.effects,
                (effect_id, HireEffectState.EXECUTING),
            ),
            effect_types=(
                *base.effect_types,
                (effect_id, effect_type),
            ),
            effect_metadata=(
                *base.effect_metadata,
                (effect_id, ()),
            ),
        ),
    )
    lease.release()

    with pytest.raises(FireServiceError, match="external mutation outcome"):
        service.start_fire(request)

    assert state.employees["agt_1"].state.value == "action_required"
    assert all(
        event.event_type != "fire.requested"
        for frame in writer.replay()
        for event in frame.events
    )
    ingress.close()
    writer.close()


def test_restart_marks_orphan_fence_action_required_when_hire_call_is_unresolved(
    tmp_path,
) -> None:
    writer, state, ingress, _authority = _active_bound_fire_authority(tmp_path)
    hire = _HireProjectionOwner(state)
    authority = JournalFireAuthority(
        writer=writer,
        hire_service=hire,
        ingress_service=ingress,
        admin_principal_ids=frozenset({"ou_admin"}),
    )
    initial_gate = EmployeeExternalMutationGate()
    lease = initial_gate.acquire(
        "tenant_1",
        "agt_1",
        ExternalMutationKind.CREDENTIAL_PUT,
    )
    request = EmployeeFireRequest(
        employee="Atlas",
        tenant_key="tenant_1",
        message_id="om_fire_marker_action_required",
        chat_id="oc_dm",
        requester_principal_id="ou_admin",
    )
    initial = EmployeeFireService(
        writer=writer,
        authority=authority,
        effects={name: _Effect(name, []) for name in FIRE_EFFECT_ORDER},
        external_mutation_gate=initial_gate,
        external_mutation_wait_seconds=0.01,
    )
    with pytest.raises(FireServiceError, match="external mutation is still active"):
        initial.start_fire(request)

    base = _pre_binding_hire_state(
        register_state=HireEffectState.COMMITTED,
        credential_committed=True,
    )
    hire.hire_states = (
        replace(
            base,
            effects=(
                *base.effects,
                ("slash-reconcile:2:1", HireEffectState.EXECUTING),
            ),
            effect_types=(
                *base.effect_types,
                ("slash-reconcile:2:1", "slash_reconciliation"),
            ),
            effect_metadata=(
                *base.effect_metadata,
                ("slash-reconcile:2:1", ()),
            ),
        ),
    )
    lease.release()
    restarted = EmployeeFireService(
        writer=writer,
        authority=authority,
        effects={name: _Effect(name, []) for name in FIRE_EFFECT_ORDER},
        external_mutation_gate=EmployeeExternalMutationGate(),
        external_mutation_wait_seconds=0.01,
    )

    assert restarted.recover() == ()
    assert state.employees["agt_1"].state.value == "action_required"
    assert all(
        event.event_type != "fire.requested"
        for frame in writer.replay()
        for event in frame.events
    )
    ingress.close()
    writer.close()


def test_restart_recovers_marker_anchored_before_fire_admission(tmp_path) -> None:
    writer, state, ingress, authority = _active_bound_fire_authority(tmp_path)
    initial_gate = EmployeeExternalMutationGate()
    lease = initial_gate.acquire(
        "tenant_1",
        "agt_1",
        ExternalMutationKind.CREDENTIAL_PUT,
    )
    request = EmployeeFireRequest(
        employee="Atlas",
        tenant_key="tenant_1",
        message_id="om_fire_marker_crash_recovery",
        chat_id="oc_dm",
        requester_principal_id="ou_admin",
    )
    initial = EmployeeFireService(
        writer=writer,
        authority=authority,
        effects={name: _Effect(name, []) for name in FIRE_EFFECT_ORDER},
        external_mutation_gate=initial_gate,
        external_mutation_wait_seconds=0.01,
    )

    with pytest.raises(FireServiceError, match="external mutation is still active"):
        initial.start_fire(request)
    lease.release()

    calls: list[str] = []
    restarted = EmployeeFireService(
        writer=writer,
        authority=authority,
        effects={name: _Effect(name, calls) for name in FIRE_EFFECT_ORDER},
        external_mutation_gate=EmployeeExternalMutationGate(),
        external_mutation_wait_seconds=0.01,
    )

    recovered = restarted.recover()

    assert len(recovered) == 1
    assert recovered[0].phase is FirePhase.ARCHIVED
    assert recovered[0].message_id == request.message_id
    assert calls == list(FIRE_EFFECT_ORDER)
    assert state.employees["agt_1"].state.value == "archived"
    ingress.close()
    writer.close()


def test_marker_recovery_timeout_keeps_live_hire_workforce_state_active(
    tmp_path,
) -> None:
    writer, state, ingress, authority = _active_bound_fire_authority(tmp_path)
    gate = EmployeeExternalMutationGate()
    lease = gate.acquire(
        "tenant_1",
        "agt_1",
        ExternalMutationKind.CREDENTIAL_PUT,
    )
    request = EmployeeFireRequest(
        employee="Atlas",
        tenant_key="tenant_1",
        message_id="om_fire_recover_live_lease",
        chat_id="oc_dm",
        requester_principal_id="ou_admin",
    )
    service = EmployeeFireService(
        writer=writer,
        authority=authority,
        effects={name: _Effect(name, []) for name in FIRE_EFFECT_ORDER},
        external_mutation_gate=gate,
        external_mutation_wait_seconds=0.01,
    )
    with pytest.raises(FireServiceError, match="external mutation is still active"):
        service.start_fire(request)

    assert service.recover() == ()
    assert state.employees["agt_1"].state.value == "active"
    lease.release()
    ingress.close()
    writer.close()


def test_legacy_short_fence_is_upgraded_to_recoverable_request(tmp_path) -> None:
    writer, state, ingress, authority = _active_bound_fire_authority(tmp_path)
    aggregate_id = EmployeeFireService._external_mutation_fence_aggregate_id(  # noqa: SLF001
        "tenant_1",
        "agt_1",
    )
    marker = JournalEvent(
        event_type="employee.external_mutation_fenced",
        aggregate_id=aggregate_id,
        payload={"tenant_key": "tenant_1", "agent_id": "agt_1"},
    )
    with writer.transaction_guard():
        last = writer.get_last_frame()
        result = writer.commit(
            (marker,),
            writer.get_aggregate_versions((aggregate_id,)),
            expected_head_sequence=0 if last is None else last.sequence,
            expected_head_hash="" if last is None else last.frame_hash,
        )
    assert result.state.value == "anchored"
    gate = EmployeeExternalMutationGate()
    lease = gate.acquire(
        "tenant_1",
        "agt_1",
        ExternalMutationKind.CREDENTIAL_PUT,
    )
    request = EmployeeFireRequest(
        employee="Atlas",
        tenant_key="tenant_1",
        message_id="om_fire_upgrade_short_marker",
        chat_id="oc_dm",
        requester_principal_id="ou_admin",
    )
    service = EmployeeFireService(
        writer=writer,
        authority=authority,
        effects={name: _Effect(name, []) for name in FIRE_EFFECT_ORDER},
        external_mutation_gate=gate,
        external_mutation_wait_seconds=0.01,
    )

    with pytest.raises(FireServiceError, match="external mutation is still active"):
        service.start_fire(request)

    full_markers = [
        event
        for frame in writer.replay()
        for event in frame.events
        if event.event_type == "employee.external_mutation_fenced"
        and event.payload.get("message_id") == request.message_id
    ]
    assert len(full_markers) == 1
    lease.release()
    restarted = EmployeeFireService(
        writer=writer,
        authority=authority,
        effects={name: _Effect(name, []) for name in FIRE_EFFECT_ORDER},
        external_mutation_gate=EmployeeExternalMutationGate(),
        external_mutation_wait_seconds=0.01,
    )

    recovered = restarted.recover()

    assert len(recovered) == 1
    assert recovered[0].phase is FirePhase.ARCHIVED
    assert state.employees["agt_1"].state.value == "archived"
    ingress.close()
    writer.close()


def test_concurrent_fire_requests_share_one_recoverable_fence_intent(tmp_path) -> None:
    writer, state, ingress, authority = _active_bound_fire_authority(tmp_path)
    gate = EmployeeExternalMutationGate()
    lease = gate.acquire(
        "tenant_1",
        "agt_1",
        ExternalMutationKind.CREDENTIAL_PUT,
    )
    services = tuple(
        EmployeeFireService(
            writer=writer,
            authority=authority,
            effects={name: _Effect(name, []) for name in FIRE_EFFECT_ORDER},
            external_mutation_gate=gate,
            external_mutation_wait_seconds=0.02,
        )
        for _index in range(2)
    )
    requests = tuple(
        EmployeeFireRequest(
            employee="Atlas",
            tenant_key="tenant_1",
            message_id=f"om_fire_concurrent_marker_{index}",
            chat_id="oc_dm",
            requester_principal_id="ou_admin",
        )
        for index in range(2)
    )
    barrier = threading.Barrier(2)

    def start(item):
        service, request = item
        barrier.wait(timeout=2)
        with pytest.raises(
            FireServiceError,
            match="external mutation is still active",
        ):
            service.start_fire(request)

    with ThreadPoolExecutor(max_workers=2) as pool:
        tuple(pool.map(start, zip(services, requests, strict=True)))

    full_markers = [
        event
        for frame in writer.replay()
        for event in frame.events
        if event.event_type == "employee.external_mutation_fenced"
        and "message_id" in event.payload
    ]
    assert len(full_markers) == 1
    lease.release()
    restarted = EmployeeFireService(
        writer=writer,
        authority=authority,
        effects={name: _Effect(name, []) for name in FIRE_EFFECT_ORDER},
        external_mutation_gate=EmployeeExternalMutationGate(),
        external_mutation_wait_seconds=0.01,
    )

    recovered = restarted.recover()

    assert len(recovered) == 1
    assert recovered[0].phase is FirePhase.ARCHIVED
    assert state.employees["agt_1"].state.value == "archived"
    ingress.close()
    writer.close()


def test_marker_recovery_resumes_new_fire_stream_once(tmp_path) -> None:
    writer, _state, ingress, authority = _active_bound_fire_authority(tmp_path)
    initial_gate = EmployeeExternalMutationGate()
    lease = initial_gate.acquire(
        "tenant_1",
        "agt_1",
        ExternalMutationKind.CREDENTIAL_PUT,
    )
    request = EmployeeFireRequest(
        employee="Atlas",
        tenant_key="tenant_1",
        message_id="om_fire_marker_single_resume",
        chat_id="oc_dm",
        requester_principal_id="ou_admin",
    )
    initial = EmployeeFireService(
        writer=writer,
        authority=authority,
        effects={name: _Effect(name, []) for name in FIRE_EFFECT_ORDER},
        external_mutation_gate=initial_gate,
        external_mutation_wait_seconds=0.01,
    )
    with pytest.raises(FireServiceError, match="external mutation is still active"):
        initial.start_fire(request)
    lease.release()
    executions = 0

    class _WaitingQuiesce:
        def execute(self, _state):
            nonlocal executions
            executions += 1

        def observe(self, _state):
            return False

    effects = {name: _Effect(name, []) for name in FIRE_EFFECT_ORDER}
    effects["execution_quiesce"] = _WaitingQuiesce()
    restarted = EmployeeFireService(
        writer=writer,
        authority=authority,
        effects=effects,
        external_mutation_gate=EmployeeExternalMutationGate(),
        external_mutation_wait_seconds=0.01,
    )

    recovered = restarted.recover()

    assert len(recovered) == 1
    assert recovered[0].phase is FirePhase.RETIRING
    assert recovered[0].effect_state("execution_quiesce") is FireEffectState.EXECUTING
    assert executions == 1
    ingress.close()
    writer.close()


def test_concurrent_fire_resume_does_not_append_duplicate_effect_transition(
    tmp_path,
) -> None:
    writer, _state, ingress, authority = _active_bound_fire_authority(tmp_path)

    class _WaitingQuiesce:
        def execute(self, _state):
            return None

        def observe(self, _state):
            return False

    initial_effects = {name: _Effect(name, []) for name in FIRE_EFFECT_ORDER}
    initial_effects["execution_quiesce"] = _WaitingQuiesce()
    initial = EmployeeFireService(
        writer=writer,
        authority=authority,
        effects=initial_effects,
    )
    waiting = initial.start_fire(
        EmployeeFireRequest(
            employee="Atlas",
            tenant_key="tenant_1",
            message_id="om_fire_concurrent_resume",
            chat_id="oc_dm",
            requester_principal_id="ou_admin",
        )
    )
    barrier = threading.Barrier(2)

    class _RacingQuiesce:
        def execute(self, _state):
            barrier.wait(timeout=2)

        def observe(self, _state):
            return True

    racing_effects = {name: _Effect(name, []) for name in FIRE_EFFECT_ORDER}
    racing_effects["execution_quiesce"] = _RacingQuiesce()
    services = (
        EmployeeFireService(
            writer=writer,
            authority=authority,
            effects=racing_effects,
        ),
        EmployeeFireService(
            writer=writer,
            authority=authority,
            effects=racing_effects,
        ),
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(
            pool.map(lambda service: service.resume(waiting.intent_id), services)
        )

    projection = rebuild_fire_projection(tuple(writer.replay()))
    assert projection[waiting.intent_id].phase is FirePhase.ARCHIVED
    assert all(result.phase is FirePhase.ARCHIVED for result in results)
    committed = [
        event
        for frame in writer.replay()
        for event in frame.events
        if event.event_type == "fire.effect.committed"
        and event.payload.get("effect_type") == "execution_quiesce"
    ]
    assert len(committed) == 1
    ingress.close()
    writer.close()


def _pre_binding_hire_state(
    *,
    register_state: HireEffectState = HireEffectState.PREPARED,
    credential_committed: bool = False,
) -> DurableHireState:
    effects = [("register-app", register_state)]
    metadata = [("register-app", (("app_id", "cli_registered"),))]
    if credential_committed:
        effects.append(("store-credential", HireEffectState.COMMITTED))
        metadata.append(
            (
                "store-credential",
                (
                    ("app_id", "cli_registered"),
                    ("credential_ref", "cred_live_secret"),
                ),
            )
        )
    return DurableHireState(
        intent_id="hire_intent",
        agent_id="agt_1",
        bot_principal_id="bot_planned",
        app_id=("cli_registered" if register_state is HireEffectState.COMMITTED else ""),
        effects=tuple(effects),
        effect_types=tuple((name, name) for name, _state in effects),
        effect_metadata=tuple(metadata),
    )


@pytest.mark.parametrize(
    ("effect_id", "effect_type"),
    (
        ("slash-reconcile:2:1", "slash_reconciliation"),
        ("channel-start:2", "employee_channel_start"),
    ),
)
def test_restart_pending_external_hire_effect_blocks_fire_admission(
    tmp_path,
    effect_id,
    effect_type,
) -> None:
    writer, state, ingress, _authority = _active_bound_fire_authority(tmp_path)
    base = _pre_binding_hire_state(
        register_state=HireEffectState.COMMITTED,
        credential_committed=True,
    )
    hire_state = replace(
        base,
        effects=(*base.effects, (effect_id, HireEffectState.EXECUTING)),
        effect_types=(*base.effect_types, (effect_id, effect_type)),
        effect_metadata=(*base.effect_metadata, (effect_id, ())),
    )
    authority = JournalFireAuthority(
        writer=writer,
        hire_service=_HireProjectionOwner(state, hire_states=(hire_state,)),
        ingress_service=ingress,
        admin_principal_ids=frozenset({"ou_admin"}),
    )
    calls: list[str] = []
    service = EmployeeFireService(
        writer=writer,
        authority=authority,
        effects={name: _Effect(name, calls) for name in FIRE_EFFECT_ORDER},
        external_mutation_gate=EmployeeExternalMutationGate(),
    )

    with pytest.raises(FireServiceError, match="external mutation outcome"):
        service.start_fire(
            EmployeeFireRequest(
                employee="Atlas",
                tenant_key="tenant_1",
                message_id=f"om_fire_pending_{effect_type}",
                chat_id="oc_dm",
                requester_principal_id="ou_admin",
            )
        )

    assert calls == []
    assert state.employees["agt_1"].state.value == "action_required"
    assert all(
        event.event_type != "fire.requested"
        for frame in writer.replay()
        for event in frame.events
    )
    assert any(
        event.event_type == "employee.external_mutation_fenced"
        for frame in writer.replay()
        for event in frame.events
    )
    ingress.close()
    writer.close()


def test_admission_atomically_commits_retiring_and_employee_ingress_closure(tmp_path):
    writer = make_writer(tmp_path)
    state = ProjectionState()
    commit_events(writer, state, employee_created())
    commit_events(writer, state, *bot_binding_events())
    commit_events(
        writer,
        state,
        JournalEvent(
            event_type="employee.state_changed",
            aggregate_id="agt_1",
            payload={"state": "active"},
        ),
    )
    ingress = EmployeeIngressService(
        writer=writer,
        blob_store=BlobStore(
            tmp_path / "blobs",
            AesGcmEncryptionProvider(lambda _key: b"fire-ingress-data-key-32-bytes!!"),
        ),
        ingress_state=IngressProjectionState(),
        active_key_id="k1",
    )
    hire = _HireProjectionOwner(state)
    authority = JournalFireAuthority(
        writer=writer,
        hire_service=hire,
        ingress_service=ingress,
        admin_principal_ids=frozenset({"ou_admin"}),
    )
    request = EmployeeFireRequest(
        employee="Atlas",
        tenant_key="tenant_1",
        message_id="om_fire",
        chat_id="oc_dm",
        requester_principal_id="ou_admin",
    )
    target = authority.resolve(request)

    authority.admit(request, target, "fire_intent")

    assert state.employees["agt_1"].state.value == "retiring"
    assert ("tenant_1", "agt_1") in ingress.state.closed_employees
    final = writer.get_last_frame()
    assert final is not None
    assert [event.event_type for event in final.events] == [
        "fire.requested",
        "employee.state_changed",
        "employee.ingress.closed",
    ]
    ingress.close()
    writer.close()


def test_configuring_employee_with_credentials_can_be_retired(tmp_path):
    writer = make_writer(tmp_path)
    state = ProjectionState()
    commit_events(writer, state, employee_created())
    commit_events(writer, state, *bot_binding_events())
    commit_events(
        writer,
        state,
        JournalEvent(
            event_type="employee.state_changed",
            aggregate_id="agt_1",
            payload={"state": "configuring"},
        ),
    )
    ingress = EmployeeIngressService(
        writer=writer,
        blob_store=BlobStore(
            tmp_path / "blobs",
            AesGcmEncryptionProvider(lambda _key: b"fire-ingress-data-key-32-bytes!!"),
        ),
        ingress_state=IngressProjectionState(),
        active_key_id="k1",
    )
    hire = _HireProjectionOwner(state)
    authority = JournalFireAuthority(
        writer=writer,
        hire_service=hire,
        ingress_service=ingress,
        admin_principal_ids=frozenset({"ou_admin"}),
    )
    request = EmployeeFireRequest(
        employee="Atlas",
        tenant_key="tenant_1",
        message_id="om_fire_configuring",
        chat_id="oc_dm",
        requester_principal_id="ou_admin",
    )

    authority.admit(request, authority.resolve(request), "fire_configuring")

    assert state.employees["agt_1"].state.value == "retiring"
    assert ("tenant_1", "agt_1") in ingress.state.closed_employees
    ingress.close()
    writer.close()


def test_pre_binding_employee_can_be_fired_and_archived(tmp_path):
    writer = make_writer(tmp_path)
    state = ProjectionState()
    created = employee_created()
    commit_events(
        writer,
        state,
        JournalEvent(
            event_type=created.event_type,
            aggregate_id=created.aggregate_id,
            payload={**created.payload, "state": "provisioning_app"},
        ),
    )
    ingress = EmployeeIngressService(
        writer=writer,
        blob_store=BlobStore(
            tmp_path / "blobs",
            AesGcmEncryptionProvider(
                lambda _key: b"fire-ingress-data-key-32-bytes!!"
            ),
        ),
        ingress_state=IngressProjectionState(),
        active_key_id="k1",
    )
    authority = JournalFireAuthority(
        writer=writer,
        hire_service=_HireProjectionOwner(
            state,
            hire_states=(_pre_binding_hire_state(),),
        ),
        ingress_service=ingress,
        admin_principal_ids=frozenset({"ou_admin"}),
    )

    class _NoopEffect:
        def execute(self, _state):
            return None

        def observe(self, _state):
            return True

    service = EmployeeFireService(
        writer=writer,
        authority=authority,
        effects={name: _NoopEffect() for name in FIRE_EFFECT_ORDER},
    )
    request = EmployeeFireRequest(
        employee="Atlas",
        tenant_key="tenant_1",
        message_id="om_fire_prebinding",
        chat_id="oc_dm",
        requester_principal_id="ou_admin",
    )

    target = authority.resolve(request)
    result = service.start_fire(request)

    assert target.pre_binding is True
    assert result.phase is FirePhase.ARCHIVED
    assert state.employees["agt_1"].state.value == "archived"
    assert ("tenant_1", "agt_1") in ingress.state.closed_employees
    ingress.close()
    writer.close()


def test_drain_waits_without_action_required_then_auto_reconciles(tmp_path):
    writer = make_writer(tmp_path)
    state = ProjectionState()
    commit_events(writer, state, employee_created())
    commit_events(writer, state, *bot_binding_events())
    commit_events(
        writer,
        state,
        JournalEvent(
            event_type="employee.state_changed",
            aggregate_id="agt_1",
            payload={"state": "active"},
        ),
    )
    ingress = EmployeeIngressService(
        writer=writer,
        blob_store=BlobStore(
            tmp_path / "blobs",
            AesGcmEncryptionProvider(
                lambda _key: b"fire-ingress-data-key-32-bytes!!"
            ),
        ),
        ingress_state=IngressProjectionState(),
        active_key_id="k1",
    )
    authority = JournalFireAuthority(
        writer=writer,
        hire_service=_HireProjectionOwner(state),
        ingress_service=ingress,
        admin_principal_ids=frozenset({"ou_admin"}),
    )
    active = True
    retired: list[str] = []
    calls: list[str] = []

    class _DrainEffect:
        def execute(self, fire_state):
            if not active:
                retired.append(fire_state.agent_id)

        def observe(self, _state):
            return not active

    effects = {name: _Effect(name, calls) for name in FIRE_EFFECT_ORDER}
    effects["execution_quiesce"] = _DrainEffect()
    service = EmployeeFireService(
        writer=writer,
        authority=authority,
        effects=effects,
    )
    request = EmployeeFireRequest(
        employee="Atlas",
        tenant_key="tenant_1",
        message_id="om_fire_drain",
        chat_id="oc_dm",
        requester_principal_id="ou_admin",
        drain=True,
    )

    waiting = service.start_fire(request)

    assert waiting.phase is FirePhase.RETIRING
    assert dict(waiting.effects)["execution_quiesce"].value == "executing"
    assert state.employees["agt_1"].state.value == "retiring"
    assert retired == []
    assert calls == []

    active = False
    reconciled = service.reconcile_draining()

    assert len(reconciled) == 1
    assert reconciled[0].phase is FirePhase.ARCHIVED
    assert retired == ["agt_1"]
    assert calls == list(FIRE_EFFECT_ORDER[1:])
    assert state.employees["agt_1"].state.value == "archived"
    ingress.close()
    writer.close()


def test_action_required_membership_cleanup_is_reexecuted_on_retry(tmp_path) -> None:
    writer, state, ingress, authority = _active_bound_fire_authority(tmp_path)
    calls = 0
    cleaned = False

    class _RetryableMembershipCleanup:
        def execute(self, _state):
            nonlocal calls, cleaned
            calls += 1
            if calls == 1:
                raise RuntimeError("transient remove failure")
            cleaned = True

        def observe(self, _state):
            return cleaned

    effects = {name: _Effect(name, []) for name in FIRE_EFFECT_ORDER}
    effects["membership_cleanup"] = _RetryableMembershipCleanup()
    service = EmployeeFireService(
        writer=writer,
        authority=authority,
        effects=effects,
    )
    request = EmployeeFireRequest(
        employee="Atlas",
        tenant_key="tenant_1",
        message_id="om_fire_retry_membership_cleanup",
        chat_id="oc_dm",
        requester_principal_id="ou_admin",
    )

    waiting = service.start_fire(request)
    assert waiting.phase is FirePhase.ACTION_REQUIRED
    assert waiting.effect_state("membership_cleanup") is FireEffectState.ACTION_REQUIRED
    assert calls == 1

    completed = service.start_fire(
        EmployeeFireRequest(
            employee="Atlas",
            tenant_key="tenant_1",
            message_id="om_fire_retry_membership_cleanup_again",
            chat_id="oc_dm",
            requester_principal_id="ou_admin",
        )
    )

    assert completed.phase is FirePhase.ARCHIVED
    assert calls == 2
    assert state.employees["agt_1"].state.value == "archived"
    ingress.close()
    writer.close()


def test_membership_cleanup_observation_uses_durable_cleanup_obligations() -> None:
    calls: list[tuple[str, str]] = []

    class _Membership:
        def retirement_confirmed_empty(self, *, tenant_key, agent_id):
            calls.append((tenant_key, agent_id))
            return False

    effect = MembershipCleanupEffect(_Membership(), object())
    state = SimpleNamespace(tenant_key="tenant_1", agent_id="agt_1")

    assert effect.observe(state) is False
    assert calls == [("tenant_1", "agt_1")]


def test_drain_action_required_after_quiesce_is_not_polled_again(tmp_path):
    writer, _state, ingress, authority = _active_bound_fire_authority(tmp_path)

    class _CountingWriter:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.replay_count = 0
            self.committed_tail_count = 0

        def replay(self, *args, **kwargs):
            self.replay_count += 1
            return self.wrapped.replay(*args, **kwargs)

        def committed_tail(self, *args, **kwargs):
            self.committed_tail_count += 1
            return self.wrapped.committed_tail(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

    class _UnresolvedCleanup:
        def execute(self, _state):
            raise RuntimeError("cleanup unavailable")

        def observe(self, _state):
            return False

    counting_writer = _CountingWriter(writer)
    effects = {name: _Effect(name, []) for name in FIRE_EFFECT_ORDER}
    effects["slash_cleanup"] = _UnresolvedCleanup()
    service = EmployeeFireService(
        writer=counting_writer,
        authority=authority,
        effects=effects,
    )
    waiting = service.start_fire(
        EmployeeFireRequest(
            employee="Atlas",
            tenant_key="tenant_1",
            message_id="om_fire_drain_cleanup_failure",
            chat_id="oc_dm",
            requester_principal_id="ou_admin",
            drain=True,
        )
    )
    assert counting_writer.replay_count == 0
    assert counting_writer.committed_tail_count > 0
    counting_writer.committed_tail_count = 0

    assert waiting.phase is FirePhase.ACTION_REQUIRED
    assert waiting.effect_state("execution_quiesce") is FireEffectState.COMMITTED
    assert waiting.effect_state("slash_cleanup") is FireEffectState.ACTION_REQUIRED
    assert service.reconcile_draining() == ()
    assert service.reconcile_draining() == ()
    assert counting_writer.replay_count == 0
    assert counting_writer.committed_tail_count == 0
    ingress.close()
    writer.close()


def test_reconcile_draining_evicts_cursor_advanced_by_another_service(tmp_path):
    writer, _state, ingress, authority = _active_bound_fire_authority(tmp_path)
    now = 10.0
    active = True

    class _CountingWriter:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.replay_count = 0
            self.committed_tail_count = 0

        def replay(self, *args, **kwargs):
            self.replay_count += 1
            return self.wrapped.replay(*args, **kwargs)

        def committed_tail(self, *args, **kwargs):
            self.committed_tail_count += 1
            return self.wrapped.committed_tail(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

    class _DrainEffect:
        def execute(self, _state):
            return None

        def observe(self, _state):
            return not active

    class _UnresolvedCleanup:
        def execute(self, _state):
            raise RuntimeError("cleanup unavailable")

        def observe(self, _state):
            return False

    counting_writer = _CountingWriter(writer)
    initial_effects = {name: _Effect(name, []) for name in FIRE_EFFECT_ORDER}
    initial_effects["execution_quiesce"] = _DrainEffect()
    initial = EmployeeFireService(
        writer=counting_writer,
        authority=authority,
        effects=initial_effects,
        monotonic=lambda: now,
    )
    waiting = initial.start_fire(
        EmployeeFireRequest(
            employee="Atlas",
            tenant_key="tenant_1",
            message_id="om_fire_cross_instance_drain",
            chat_id="oc_dm",
            requester_principal_id="ou_admin",
            drain=True,
        )
    )
    assert waiting.effect_state("execution_quiesce") is FireEffectState.EXECUTING

    active = False
    recovery_effects = {name: _Effect(name, []) for name in FIRE_EFFECT_ORDER}
    recovery_effects["execution_quiesce"] = _DrainEffect()
    recovery_effects["slash_cleanup"] = _UnresolvedCleanup()
    recovered = EmployeeFireService(
        writer=writer,
        authority=authority,
        effects=recovery_effects,
    ).recover()

    assert len(recovered) == 1
    advanced = recovered[0]
    assert advanced.phase is FirePhase.ACTION_REQUIRED
    assert advanced.effect_state("execution_quiesce") is FireEffectState.COMMITTED
    assert advanced.effect_state("slash_cleanup") is FireEffectState.ACTION_REQUIRED

    assert counting_writer.replay_count == 0
    assert counting_writer.committed_tail_count > 0
    counting_writer.committed_tail_count = 0
    assert initial.reconcile_draining() == ()
    assert counting_writer.replay_count == 0
    assert counting_writer.committed_tail_count == 1

    now += 0.5
    assert initial.reconcile_draining() == ()
    assert counting_writer.replay_count == 0
    assert counting_writer.committed_tail_count == 1
    ingress.close()
    writer.close()


def test_pending_drain_reconciliation_is_throttled_but_converges_on_due_poll(
    tmp_path,
):
    writer, state, ingress, authority = _active_bound_fire_authority(tmp_path)
    now = 10.0
    active = True
    quiesce_calls: list[str] = []

    def monotonic() -> float:
        return now

    class _DrainEffect:
        def execute(self, _state):
            quiesce_calls.append("execute")

        def observe(self, _state):
            return not active

    effects = {name: _Effect(name, []) for name in FIRE_EFFECT_ORDER}
    effects["execution_quiesce"] = _DrainEffect()
    service = EmployeeFireService(
        writer=writer,
        authority=authority,
        effects=effects,
        drain_reconcile_interval_seconds=0.5,
        monotonic=monotonic,
    )
    waiting = service.start_fire(
        EmployeeFireRequest(
            employee="Atlas",
            tenant_key="tenant_1",
            message_id="om_fire_throttled_drain",
            chat_id="oc_dm",
            requester_principal_id="ou_admin",
            drain=True,
        )
    )
    quiesce_calls.clear()

    assert waiting.phase is FirePhase.RETIRING
    assert service.reconcile_draining() == ()
    assert quiesce_calls == ["execute"]

    assert service.reconcile_draining() == ()
    now += 0.49
    assert service.reconcile_draining() == ()
    assert quiesce_calls == ["execute"]

    active = False
    now += 0.01
    reconciled = service.reconcile_draining()
    assert len(reconciled) == 1
    assert reconciled[0].phase is FirePhase.ARCHIVED
    assert quiesce_calls == ["execute", "execute"]
    assert state.employees["agt_1"].state.value == "archived"
    ingress.close()
    writer.close()


def test_recovery_reinstalls_committed_drain_actor_fence_before_cleanup(tmp_path):
    writer, state, ingress, authority = _active_bound_fire_authority(tmp_path)

    class _SimulatedCrash(BaseException):
        pass

    class _CrashDuringCleanup:
        def execute(self, _state):
            raise _SimulatedCrash

        def observe(self, _state):
            return False

    initial_effects = {name: _Effect(name, []) for name in FIRE_EFFECT_ORDER}
    initial_effects["slash_cleanup"] = _CrashDuringCleanup()
    initial = EmployeeFireService(
        writer=writer,
        authority=authority,
        effects=initial_effects,
    )
    with pytest.raises(_SimulatedCrash):
        initial.start_fire(
            EmployeeFireRequest(
                employee="Atlas",
                tenant_key="tenant_1",
                message_id="om_fire_restart_after_quiesce",
                chat_id="oc_dm",
                requester_principal_id="ou_admin",
                drain=True,
            )
        )

    order: list[str] = []
    retired: set[str] = set()

    class _Runtime:
        def retire_employee(self, agent_id):
            order.append("retire_actor")
            retired.add(agent_id)

        def is_retired(self, agent_id):
            return agent_id in retired

    runtime = _Runtime()

    class _FenceAwareCleanup:
        def execute(self, _state):
            pytest.fail("an already executing cleanup must be observed, not repeated")

        def observe(self, fire_state):
            order.append("observe_cleanup")
            return runtime.is_retired(fire_state.agent_id)

    restarted_effects = {name: _Effect(name, order) for name in FIRE_EFFECT_ORDER}
    restarted_effects["execution_quiesce"] = ExecutionQuiesceEffect(
        SimpleNamespace(
            state=SimpleNamespace(attempts={}),
            employee_runtime=runtime,
        ),
        grace_seconds=0,
    )
    restarted_effects["slash_cleanup"] = _FenceAwareCleanup()
    restarted = EmployeeFireService(
        writer=writer,
        authority=authority,
        effects=restarted_effects,
    )

    recovered = restarted.recover()

    assert len(recovered) == 1
    assert recovered[0].phase is FirePhase.ARCHIVED
    assert order[:2] == ["retire_actor", "observe_cleanup"]
    assert retired == {"agt_1"}
    assert state.employees["agt_1"].state.value == "archived"
    ingress.close()
    writer.close()


def test_registered_app_without_credential_fails_closed_and_stays_unarchived(
    tmp_path,
):
    writer = make_writer(tmp_path)
    state = ProjectionState()
    created = employee_created()
    commit_events(
        writer,
        state,
        JournalEvent(
            event_type=created.event_type,
            aggregate_id=created.aggregate_id,
            payload={**created.payload, "state": "provisioning_app"},
        ),
    )
    ingress = EmployeeIngressService(
        writer=writer,
        blob_store=BlobStore(
            tmp_path / "blobs",
            AesGcmEncryptionProvider(
                lambda _key: b"fire-ingress-data-key-32-bytes!!"
            ),
        ),
        ingress_state=IngressProjectionState(),
        active_key_id="k1",
    )
    hire = _HireProjectionOwner(
        state,
        hire_states=(
            _pre_binding_hire_state(register_state=HireEffectState.COMMITTED),
        ),
    )
    authority = JournalFireAuthority(
        writer=writer,
        hire_service=hire,
        ingress_service=ingress,
        admin_principal_ids=frozenset({"ou_admin"}),
    )
    calls: list[str] = []
    service = EmployeeFireService(
        writer=writer,
        authority=authority,
        effects={name: _Effect(name, calls) for name in FIRE_EFFECT_ORDER},
    )

    result = service.start_fire(
        EmployeeFireRequest(
            employee="Atlas",
            tenant_key="tenant_1",
            message_id="om_fire_unknown_app",
            chat_id="oc_dm",
            requester_principal_id="ou_admin",
        )
    )

    assert result.cleanup_mode is FireCleanupMode.EXTERNAL_UNKNOWN
    assert result.phase is FirePhase.ACTION_REQUIRED
    assert state.employees["agt_1"].state.value == "action_required"
    assert calls == []
    confirmation = EmployeeFireRequest(
        employee="Atlas",
        tenant_key="tenant_1",
        message_id="om_confirm_external_app",
        chat_id="oc_dm",
        requester_principal_id="ou_admin",
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        confirmations = tuple(
            executor.map(
                lambda _index: service.confirm_external_disposition(
                    confirmation,
                    "cli_registered",
                ),
                range(2),
            )
        )
    completed, repeated = confirmations
    assert completed.phase is FirePhase.ARCHIVED
    assert repeated == completed
    assert state.employees["agt_1"].state.value == "archived"
    assert calls == ["archive_move"]
    confirmation_event = next(
        event
        for frame in writer.replay()
        for event in frame.events
        if event.event_type == "fire.external_disposition_confirmed"
    )
    assert confirmation_event.payload["disposed_by"] == "ou_admin"
    assert confirmation_event.payload["disposition_ref"] == "cli_registered"
    assert sum(
        event.event_type == "fire.external_disposition_confirmed"
        for frame in writer.replay()
        for event in frame.events
    ) == 1
    ingress.close()
    writer.close()


def test_concurrent_fire_requests_share_one_live_external_cleanup_saga(tmp_path):
    writer = make_writer(tmp_path)
    state = ProjectionState()
    created = employee_created()
    commit_events(
        writer,
        state,
        JournalEvent(
            event_type=created.event_type,
            aggregate_id=created.aggregate_id,
            payload={**created.payload, "state": "provisioning_app"},
        ),
    )
    ingress = EmployeeIngressService(
        writer=writer,
        blob_store=BlobStore(
            tmp_path / "blobs",
            AesGcmEncryptionProvider(
                lambda _key: b"fire-ingress-data-key-32-bytes!!"
            ),
        ),
        ingress_state=IngressProjectionState(),
        active_key_id="k1",
    )
    hire = _HireProjectionOwner(
        state,
        hire_states=(
            _pre_binding_hire_state(register_state=HireEffectState.COMMITTED),
        ),
    )
    authority = JournalFireAuthority(
        writer=writer,
        hire_service=hire,
        ingress_service=ingress,
        admin_principal_ids=frozenset({"ou_admin"}),
    )
    service = EmployeeFireService(
        writer=writer,
        authority=authority,
        effects={name: _Effect(name, []) for name in FIRE_EFFECT_ORDER},
    )

    def submit(suffix: str):
        return service.start_fire(
            EmployeeFireRequest(
                employee="Atlas",
                tenant_key="tenant_1",
                message_id=f"om_fire_concurrent_{suffix}",
                chat_id="oc_dm",
                requester_principal_id="ou_admin",
            )
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(submit, ("one", "two")))

    assert results[0].intent_id == results[1].intent_id
    assert results[0].phase is FirePhase.ACTION_REQUIRED
    assert sum(
        event.event_type == "fire.requested"
        for frame in writer.replay()
        for event in frame.events
    ) == 1
    ingress.close()
    writer.close()


def test_admission_re_resolves_principal_bound_after_optimistic_resolve(tmp_path):
    writer = make_writer(tmp_path)
    state = ProjectionState()
    created = employee_created()
    commit_events(
        writer,
        state,
        JournalEvent(
            event_type=created.event_type,
            aggregate_id=created.aggregate_id,
            payload={**created.payload, "state": "provisioning_app"},
        ),
    )
    ingress = EmployeeIngressService(
        writer=writer,
        blob_store=BlobStore(
            tmp_path / "blobs",
            AesGcmEncryptionProvider(
                lambda _key: b"fire-ingress-data-key-32-bytes!!"
            ),
        ),
        ingress_state=IngressProjectionState(),
        active_key_id="k1",
    )

    hire = _HireProjectionOwner(
        state,
        hire_states=(_pre_binding_hire_state(),),
    )
    authority = JournalFireAuthority(
        writer=writer,
        hire_service=hire,
        ingress_service=ingress,
        admin_principal_ids=frozenset({"ou_admin"}),
    )
    request = EmployeeFireRequest(
        employee="Atlas",
        tenant_key="tenant_1",
        message_id="om_fire_binding_race",
        chat_id="oc_dm",
        requester_principal_id="ou_admin",
    )
    stale = authority.resolve(request)
    commit_events(writer, state, *bot_binding_events())

    admitted = authority.admit(request, stale, "fire_binding_race")

    assert stale.cleanup_mode is FireCleanupMode.SAFE_ABORT
    assert admitted.cleanup_mode is FireCleanupMode.BOUND
    fire_event = next(
        event
        for frame in writer.replay()
        for event in frame.events
        if event.event_type == "fire.requested"
    )
    assert fire_event.payload["bot_principal_id"] == "bot_1"
    assert fire_event.payload["credential_ref"] == "cred_1"
    ingress.close()
    writer.close()


def test_admission_rejects_name_rebound_to_different_agent(tmp_path) -> None:
    writer, state, ingress, _authority = _active_bound_fire_authority(tmp_path)

    def replace_employee() -> None:
        commit_events(
            writer,
            state,
            JournalEvent(
                event_type="employee.state_changed",
                aggregate_id="agt_1",
                payload={"state": "archived"},
            ),
        )
        commit_events(
            writer,
            state,
            JournalEvent(
                event_type="employee.name_released",
                aggregate_id="agt_1",
                payload={"name": "Atlas"},
            ),
        )
        replacement = employee_created("agt_2", "Atlas")
        commit_events(
            writer,
            state,
            JournalEvent(
                event_type=replacement.event_type,
                aggregate_id=replacement.aggregate_id,
                payload={**replacement.payload, "state": "active"},
            ),
        )
        commit_events(
            writer,
            state,
            *bot_binding_events(
                agent_id="agt_2",
                bot_principal_id="bot_2",
                app_id="cli_2",
                credential_ref="cred_2",
            ),
        )

    hire = _HireProjectionOwner(state, before_locked_sync=replace_employee)
    authority = JournalFireAuthority(
        writer=writer,
        hire_service=hire,
        ingress_service=ingress,
        admin_principal_ids=frozenset({"ou_admin"}),
    )
    request = EmployeeFireRequest(
        employee="Atlas",
        tenant_key="tenant_1",
        message_id="om_fire_name_rebound",
        chat_id="oc_dm",
        requester_principal_id="ou_admin",
    )
    stale = authority.resolve(request)

    with pytest.raises(FireServiceError, match="authority changed"):
        authority.admit(request, stale, "fire_name_rebound")

    assert all(
        event.event_type != "fire.requested"
        for frame in writer.replay()
        for event in frame.events
    )
    ingress.close()
    writer.close()


def test_archived_employee_is_reported_as_already_archived(tmp_path):
    writer = make_writer(tmp_path)
    state = ProjectionState()
    commit_events(writer, state, employee_created())
    commit_events(writer, state, *bot_binding_events())
    commit_events(
        writer,
        state,
        JournalEvent(
            event_type="employee.state_changed",
            aggregate_id="agt_1",
            payload={"state": "archived"},
        ),
    )
    ingress = EmployeeIngressService(
        writer=writer,
        blob_store=BlobStore(
            tmp_path / "blobs",
            AesGcmEncryptionProvider(lambda _key: b"fire-ingress-data-key-32-bytes!!"),
        ),
        ingress_state=IngressProjectionState(),
        active_key_id="k1",
    )
    authority = JournalFireAuthority(
        writer=writer,
        hire_service=_HireProjectionOwner(state),
        ingress_service=ingress,
        admin_principal_ids=frozenset({"ou_admin"}),
    )

    with pytest.raises(FireServiceError, match="already archived"):
        authority.resolve(
            EmployeeFireRequest(
                employee="Atlas",
                tenant_key="tenant_1",
                message_id="om_fire_again",
                chat_id="oc_dm",
                requester_principal_id="ou_admin",
            )
        )

    ingress.close()
    writer.close()


def test_action_required_recovery_is_noop_for_already_archived_employee(tmp_path):
    writer = make_writer(tmp_path)
    state = ProjectionState()
    commit_events(writer, state, employee_created())
    commit_events(
        writer,
        state,
        JournalEvent(
            event_type="employee.state_changed",
            aggregate_id="agt_1",
            payload={"state": "archived"},
        ),
    )
    ingress = EmployeeIngressService(
        writer=writer,
        blob_store=BlobStore(
            tmp_path / "blobs",
            AesGcmEncryptionProvider(
                lambda _key: b"fire-ingress-data-key-32-bytes!!"
            ),
        ),
        ingress_state=IngressProjectionState(),
        active_key_id="k1",
    )
    authority = JournalFireAuthority(
        writer=writer,
        hire_service=_HireProjectionOwner(state),
        ingress_service=ingress,
        admin_principal_ids=frozenset({"ou_admin"}),
    )
    sequence = writer.get_last_frame().sequence

    authority.mark_action_required("agt_1")

    assert state.employees["agt_1"].state.value == "archived"
    assert writer.get_last_frame().sequence == sequence
    ingress.close()
    writer.close()
