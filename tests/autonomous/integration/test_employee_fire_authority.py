from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from src.autonomous.ingress.projection import IngressProjectionState
from src.autonomous.ingress.service import EmployeeIngressService
from src.autonomous.journal.blob_store import AesGcmEncryptionProvider, BlobStore
from src.autonomous.journal.frame import JournalEvent
from src.autonomous.journal.projections import ProjectionState, apply_frame
from src.autonomous.provisioning.fire_authority import JournalFireAuthority
from src.autonomous.provisioning.fire_effects import ExecutionQuiesceEffect
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
        def replay(self):
            pytest.fail("idle drain reconciliation replayed the Journal")

    service = EmployeeFireService(
        writer=_Writer(),  # type: ignore[arg-type]
        authority=object(),  # type: ignore[arg-type]
        effects={name: _Effect(name, []) for name in FIRE_EFFECT_ORDER},
    )

    assert service.reconcile_draining() == ()
    assert service.reconcile_draining() == ()


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


def test_drain_action_required_after_quiesce_is_not_polled_again(tmp_path):
    writer, _state, ingress, authority = _active_bound_fire_authority(tmp_path)

    class _CountingWriter:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.replay_count = 0

        def replay(self, *args, **kwargs):
            self.replay_count += 1
            return self.wrapped.replay(*args, **kwargs)

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
    counting_writer.replay_count = 0

    assert waiting.phase is FirePhase.ACTION_REQUIRED
    assert waiting.effect_state("execution_quiesce") is FireEffectState.COMMITTED
    assert waiting.effect_state("slash_cleanup") is FireEffectState.ACTION_REQUIRED
    assert service.reconcile_draining() == ()
    assert service.reconcile_draining() == ()
    assert counting_writer.replay_count == 0
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
