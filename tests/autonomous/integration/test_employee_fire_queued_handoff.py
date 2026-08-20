from __future__ import annotations

import hashlib
import threading
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.autonomous.context.runtime import RuntimeRequesterChatAcl
from src.autonomous.domain import EmployeeState
from src.autonomous.ingress.models import (
    EmployeeIngressMetadata,
    EmployeeIngressPayload,
)
from src.autonomous.ingress.projection import IngressProjectionState
from src.autonomous.ingress.router import (
    DurableEmployeeIngressRouter,
    RouterQueueLimits,
)
from src.autonomous.ingress.service import EmployeeIngressService
from src.autonomous.journal.blob_store import AesGcmEncryptionProvider, BlobStore
from src.autonomous.journal.projections import apply_frame
from src.autonomous.outbox.delivery import (
    EmployeeOutboxDeliveryCoordinator,
    EmployeeOutboxItemDeliveryError,
)
from src.autonomous.outbox.lifecycle import EmployeeOutboxLifecycle
from src.autonomous.outbox.models import employee_outbox_id
from src.autonomous.outbox.projection import OutboxProjectionState
from src.autonomous.outbox.service import EmployeeOutboxService
from src.autonomous.provisioning import hire_service
from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime
from src.autonomous.provisioning.fire_authority import JournalFireAuthority
from src.autonomous.provisioning.fire_effects import (
    ChannelStopEffect,
    ExecutionQuiesceEffect,
)
from src.autonomous.provisioning.fire_service import (
    EmployeeFireRequest,
    EmployeeFireService,
)
from src.autonomous.provisioning.fire_state import (
    FIRE_EFFECT_ORDER,
    FireEffectState,
    FirePhase,
)
from src.autonomous.provisioning.hire_state import DurableHireState, HirePhase
from src.autonomous.supervisor.channel_models import ChannelProcessState
from src.autonomous.supervisor.employee_channels import (
    ChannelProcessStatus,
    ChannelSendReceipt,
    EmployeeChannelSupervisor,
)
from src.autonomous.workforce.registry import ProjectedAgentRegistry
from tests.autonomous.integration.test_employee_channel_process import (
    _worker,
)
from tests.autonomous.integration.test_employee_fire_authority import (
    _active_bound_fire_authority,
)
from tests.autonomous.integration.test_employee_runtime_recovery import (
    _seed_active_employee,
)
from tests.autonomous.integration.test_employee_runtime_recovery import (
    _service as _recovered_hire_service,
)
from tests.autonomous.integration.test_employee_runtime_recovery import (
    _writer as _recovery_writer,
)
from tests.autonomous.integration.test_employee_team_gateway import (
    _real_coordinator_harness,
)


class _NoopEffect:
    def execute(self, _state: object) -> None:
        return None

    def observe(self, _state: object) -> bool:
        return True


class _FireChannels:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self._stopped = False

    def stop(self, _agent_id: str) -> None:
        self._events.append("channel_stop")
        self._stopped = True

    def status(self, _agent_id: str) -> object:
        return SimpleNamespace(
            state=(
                ChannelProcessState.STOPPED
                if self._stopped
                else ChannelProcessState.READY
            )
        )


class _DeliveryChannels:
    def __init__(self, events: list[str], *, failures: int = 0) -> None:
        self._events = events
        self._failures = failures

    def send(self, _agent_id: str, **_kwargs: object) -> object:
        if self._failures:
            self._failures -= 1
            self._events.append("failure_response_delivery_failed")
            raise ConnectionError("injected delivery failure")
        self._events.append("failure_response_delivered")
        return SimpleNamespace(
            success=True,
            app_id="cli_alpha",
            generation=3,
            connection_id="conn_alpha",
            message_id="om_failure_response",
        )

    def update_card(self, _agent_id: str, **_kwargs: object) -> object:
        raise AssertionError("the failure response must be created exactly once")


class _ProductionLeaseChannels:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self._stopped = False

    def status(self, agent_id: str) -> ChannelProcessStatus:
        return ChannelProcessStatus(
            agent_id=agent_id,
            app_id="cli_employee",
            generation=1,
            pid=101,
            state=(
                ChannelProcessState.STOPPED
                if self._stopped
                else ChannelProcessState.READY
            ),
            tenant_key="tenant-a",
            bot_principal_id="bot_recover",
            identity={"app_id": "cli_employee"},
            ready_metadata={"connection_id": "conn_generation_1"},
        )

    def send(self, agent_id: str, **kwargs: object) -> ChannelSendReceipt:
        assert agent_id == "agt_recover"
        assert kwargs["generation"] == 1
        self._events.append("failure_response_delivered")
        return ChannelSendReceipt(
            request_id="send_retirement_failure",
            success=True,
            app_id="cli_employee",
            generation=1,
            connection_id="conn_generation_1",
            message_id="om_retirement_failure",
        )

    def update_card(self, _agent_id: str, **_kwargs: object) -> object:
        raise AssertionError("retirement failure response must be created once")

    def stop(self, agent_id: str) -> None:
        assert agent_id == "agt_recover"
        self._events.append("channel_stop")
        self._stopped = True


def _fire_request(*, message_id: str) -> EmployeeFireRequest:
    return EmployeeFireRequest(
        employee="alpha",
        tenant_key="tenant_1",
        message_id=message_id,
        chat_id="oc_admin",
        requester_principal_id="ou_admin",
    )


def _install_real_fire_authority(harness: object) -> JournalFireAuthority:
    hire = harness.hire

    def durable_state() -> DurableHireState:
        employee = hire.projection_state.employees["agt_alpha"]
        return DurableHireState(
            intent_id="hire_alpha",
            tenant_key="tenant_1",
            employee_name="alpha",
            agent_id="agt_alpha",
            bot_principal_id="bot_alpha",
            app_id="cli_alpha",
            credential_ref="cred_alpha",
            channel_generation=3,
            channel_identity_app_id="cli_alpha",
            channel_connection_id="conn_alpha",
            phase=HirePhase(employee.state.value),
        )

    def synchronize_projection() -> object:
        with hire.employee_dispatch_guard():
            return hire.synchronize_projection_unlocked()

    hire.synchronize_projection = synchronize_projection
    hire.apply_committed_frame_unlocked = lambda frame: apply_frame(
        hire.projection_state,
        frame,
    )
    hire.list_states = lambda: (durable_state(),)

    def current_employee_transport_snapshot() -> object:
        with hire.employee_dispatch_guard():
            projection = hire.synchronize_projection_unlocked()
            return (
                tuple(projection.employees.values()),
                tuple(projection.bot_principals.values()),
                (durable_state(),),
            )

    hire.current_employee_transport_snapshot = current_employee_transport_snapshot
    return JournalFireAuthority(
        writer=harness.writer,
        hire_service=hire,
        ingress_service=harness.ingress,
        admin_principal_ids=frozenset({"ou_admin"}),
    )


@contextmanager
def _queued_fire_harness(tmp_path, *, delivery_failures: int = 0):
    harness = _real_coordinator_harness(
        tmp_path / "dispatch",
        targeted_group_task=True,
    )
    outbox = EmployeeOutboxService(
        writer=harness.writer,
        blob_store=BlobStore(
            tmp_path / "outbox-blobs",
            AesGcmEncryptionProvider(lambda _key_ref: b"o" * 32),
        ),
        outbox_state=OutboxProjectionState(),
        active_key_id="outbox-key",
    )
    lifecycle = EmployeeOutboxLifecycle(outbox)
    harness.coordinator._attempt_lifecycle = lifecycle  # noqa: SLF001
    events: list[str] = []
    fire_channels = _FireChannels(events)
    delivery_channels = _DeliveryChannels(
        events,
        failures=delivery_failures,
    )
    runtime = object.__new__(EmployeeDepartmentRuntime)
    runtime._service = harness.hire  # noqa: SLF001
    runtime._channels = harness.channels  # noqa: SLF001
    runtime._fire = None  # noqa: SLF001
    runtime._ingress = harness.ingress  # noqa: SLF001
    runtime._router = harness.router  # noqa: SLF001
    runtime._dispatch = harness.coordinator  # noqa: SLF001
    runtime._outbox = outbox  # noqa: SLF001
    runtime._outbox_lifecycle = lifecycle  # noqa: SLF001
    delivery = EmployeeOutboxDeliveryCoordinator(
        outbox=outbox,
        channels=delivery_channels,
        authority_resolver=runtime._resolve_outbox_delivery_authority,
    )
    runtime._outbox_delivery = delivery  # noqa: SLF001
    runtime._employee_outbox_delivery_lock = threading.RLock()  # noqa: SLF001
    effects = {name: _NoopEffect() for name in FIRE_EFFECT_ORDER}
    effects["execution_quiesce"] = ExecutionQuiesceEffect(
        harness.coordinator,
        grace_seconds=0,
    )
    effects["channel_stop"] = ChannelStopEffect(fire_channels)
    authority = _install_real_fire_authority(harness)
    service = EmployeeFireService(
        writer=harness.writer,
        authority=authority,
        effects=effects,
    )
    runtime._fire = service  # noqa: SLF001

    try:
        yield SimpleNamespace(
            harness=harness,
            outbox=outbox,
            lifecycle=lifecycle,
            runtime=runtime,
            effects=effects,
            authority=authority,
            service=service,
            fire_channels=fire_channels,
            events=events,
            acceptance_id=harness.acceptance_ids[0],
        )
    finally:
        outbox.close()
        harness.close()


def _fire_service(
    subject: object,
    *,
    monotonic=None,
) -> EmployeeFireService:
    if monotonic is None:
        return subject.service
    return EmployeeFireService(
        writer=subject.harness.writer,
        authority=subject.authority,
        effects=subject.effects,
        monotonic=monotonic,
    )


def _reconcile_failure_response(subject: object) -> None:
    assert subject.runtime._reconcile_terminal_ingress() == 1  # noqa: SLF001
    assert subject.runtime._drain_employee_outbox_once() is True  # noqa: SLF001
    assert subject.runtime._drain_employee_outbox_once() is False  # noqa: SLF001


def test_fire_waits_for_queued_target_failure_delivery_before_channel_stop(
    tmp_path,
) -> None:
    with _queued_fire_harness(tmp_path) as subject:
        service = _fire_service(subject)

        waiting = service.start_fire(_fire_request(message_id="om_fire_queued"))

        routed = subject.harness.router.state.by_acceptance_id[
            subject.acceptance_id
        ]
        assert routed.state == "terminal"
        assert routed.reason_code == "context_unavailable"
        assert waiting.phase is FirePhase.RETIRING
        assert (
            waiting.effect_state("execution_quiesce")
            is FireEffectState.EXECUTING
        )
        assert subject.events == []

        _reconcile_failure_response(subject)

        control_id = employee_outbox_id(
            "tenant_1",
            "agt_alpha",
            f"control_{subject.acceptance_id}",
        )
        record = subject.outbox.get_record(control_id)
        assert record.binding is not None
        assert record.binding.bound_snapshot_version == record.latest_version
        assert subject.events == ["failure_response_delivered"]

        progressed = service.reconcile_draining()

        assert len(progressed) == 1
        assert progressed[0].phase is FirePhase.ARCHIVED
        assert subject.events == [
            "failure_response_delivered",
            "channel_stop",
        ]


def test_fire_restart_retries_failure_delivery_without_stopping_channel(
    tmp_path,
) -> None:
    with _queued_fire_harness(tmp_path, delivery_failures=1) as subject:
        now = [10.0]
        initial = _fire_service(subject, monotonic=lambda: now[0])
        waiting = initial.start_fire(
            _fire_request(message_id="om_fire_queued_restart")
        )
        assert waiting.phase is FirePhase.RETIRING

        restarted = _fire_service(subject, monotonic=lambda: now[0])
        recovered = restarted.recover()

        assert len(recovered) == 1
        assert recovered[0].phase is FirePhase.RETIRING
        assert subject.events == []
        assert subject.runtime._reconcile_terminal_ingress() == 1  # noqa: SLF001
        with pytest.raises(
            EmployeeOutboxItemDeliveryError,
            match="batch delivery deferred",
        ):
            subject.runtime._drain_employee_outbox_once()  # noqa: SLF001
        assert subject.events == ["failure_response_delivery_failed"]
        assert restarted.reconcile_draining() == ()
        assert subject.events == ["failure_response_delivery_failed"]

        assert subject.runtime._drain_employee_outbox_once() is True  # noqa: SLF001
        now[0] += 0.5
        progressed = restarted.reconcile_draining()

        assert len(progressed) == 1
        assert progressed[0].phase is FirePhase.ARCHIVED
        assert subject.events == [
            "failure_response_delivery_failed",
            "failure_response_delivered",
            "channel_stop",
        ]


def test_process_restart_recovers_delivery_only_channel_before_fire_stops_it(
    tmp_path,
) -> None:
    with _queued_fire_harness(tmp_path) as subject:
        waiting = subject.service.start_fire(
            _fire_request(message_id="om_fire_process_restart")
        )
        assert waiting.phase is FirePhase.RETIRING
        assert subject.runtime._reconcile_terminal_ingress() == 1  # noqa: SLF001
        subject.harness.ingress.gc_terminal_payloads()

        restarted_channels = EmployeeChannelSupervisor(
            secret_resolver=lambda *_: "employee-secret",
            worker_path=_worker(tmp_path),
            ready_timeout=1.0,
            stop_timeout=1.0,
            ingress_service=subject.harness.ingress,
            ingress_binding_resolver=lambda *_: ("tenant_1", "bot_alpha"),
        )
        restart_effects = dict(subject.effects)
        restart_effects["channel_stop"] = ChannelStopEffect(restarted_channels)
        restarted_fire = EmployeeFireService(
            writer=subject.harness.writer,
            authority=subject.authority,
            effects=restart_effects,
        )
        restarted_runtime = EmployeeDepartmentRuntime(runtime_enabled=True)
        restarted_runtime._service = subject.harness.hire  # noqa: SLF001
        restarted_runtime._channels = restarted_channels  # noqa: SLF001
        restarted_runtime._fire = restarted_fire  # noqa: SLF001
        restarted_runtime._ingress = subject.harness.ingress  # noqa: SLF001
        restarted_runtime._router = subject.harness.router  # noqa: SLF001
        restarted_runtime._dispatch = subject.harness.coordinator  # noqa: SLF001
        restarted_runtime._outbox = subject.outbox  # noqa: SLF001
        restarted_runtime._outbox_lifecycle = subject.lifecycle  # noqa: SLF001
        restarted_runtime._outbox_delivery = EmployeeOutboxDeliveryCoordinator(  # noqa: SLF001
            outbox=subject.outbox,
            channels=restarted_channels,
            authority_resolver=restarted_runtime._resolve_outbox_delivery_authority,
        )
        try:
            assert restarted_channels.status("agt_alpha") is None
            recovered = restarted_fire.recover()
            assert len(recovered) == 1
            assert recovered[0].phase is FirePhase.RETIRING

            recovered_agents = (
                restarted_runtime._recover_retirement_delivery_channels()  # noqa: SLF001
            )

            status = restarted_channels.status("agt_alpha")
            assert recovered_agents == ("agt_alpha",)
            assert status is not None
            assert status.state is ChannelProcessState.READY
            assert status.delivery_only is True
            assert status.app_id == "cli_alpha"
            assert status.generation == 3
            assert status.identity["app_id"] == "cli_alpha"
            assert status.ready_metadata["connection_id"] == "conn_alpha"

            assert restarted_runtime._drain_employee_outbox_once() is True  # noqa: SLF001
            progressed = restarted_fire.reconcile_draining()

            assert len(progressed) == 1
            assert progressed[0].phase is FirePhase.ARCHIVED
            stopped = restarted_channels.status("agt_alpha")
            assert stopped is not None
            assert stopped.state is ChannelProcessState.STOPPED
        finally:
            restarted_channels.close()


def test_retirement_delivery_recovery_ignores_unrelated_hire_action_required() -> None:
    hire_state = SimpleNamespace(
        tenant_key="tenant_1",
        agent_id="agt_unrelated",
        phase=HirePhase.ACTION_REQUIRED,
    )
    runtime = EmployeeDepartmentRuntime(runtime_enabled=True)
    runtime._service = SimpleNamespace(  # noqa: SLF001
        current_employee_transport_snapshot=lambda: ((), (), (hire_state,)),
    )
    runtime._channels = SimpleNamespace(  # noqa: SLF001
        start_delivery_only=lambda *_args, **_kwargs: pytest.fail(
            "ordinary hire recovery must not start a retirement Channel"
        ),
    )
    runtime._fire = SimpleNamespace(list_states=lambda: ())  # noqa: SLF001
    runtime._outbox = SimpleNamespace(  # noqa: SLF001
        list_pending_delivery_records=lambda: pytest.fail(
            "ordinary hire recovery must not scan retirement obligations"
        ),
    )

    assert runtime._recover_retirement_delivery_channels() == ()  # noqa: SLF001


def test_retirement_delivery_recovery_rejects_duplicate_hire_authority() -> None:
    employee = SimpleNamespace(
        tenant_key="tenant_1",
        agent_id="agt_alpha",
        bot_principal_id="bot_alpha",
        state=EmployeeState.RETIRING,
    )
    principal = SimpleNamespace(
        tenant_key="tenant_1",
        agent_id="agt_alpha",
        bot_principal_id="bot_alpha",
        app_id="cli_alpha",
        credential_ref="cred_alpha",
    )
    hire = DurableHireState(
        intent_id="hire_alpha",
        tenant_key="tenant_1",
        employee_name="alpha",
        agent_id="agt_alpha",
        bot_principal_id="bot_alpha",
        app_id="cli_alpha",
        credential_ref="cred_alpha",
        channel_generation=3,
        channel_identity_app_id="cli_alpha",
        channel_connection_id="conn_alpha",
        phase=HirePhase.RETIRING,
    )
    fire_state = SimpleNamespace(
        tenant_key="tenant_1",
        agent_id="agt_alpha",
        phase=FirePhase.RETIRING,
        effect_state=lambda _effect: FireEffectState.EXECUTING,
    )
    runtime = EmployeeDepartmentRuntime(runtime_enabled=True)
    runtime._service = SimpleNamespace(  # noqa: SLF001
        current_employee_transport_snapshot=lambda: (
            (employee,),
            (principal,),
            (hire, hire),
        ),
    )
    runtime._channels = SimpleNamespace(  # noqa: SLF001
        start_delivery_only=lambda *_args, **_kwargs: pytest.fail(
            "ambiguous authority must not launch a Channel"
        ),
    )
    runtime._fire = SimpleNamespace(list_states=lambda: (fire_state,))  # noqa: SLF001
    runtime._outbox = SimpleNamespace(  # noqa: SLF001
        list_pending_delivery_records=lambda: (),
    )

    with pytest.raises(RuntimeError, match="ambiguous"):
        runtime._recover_retirement_delivery_channels()  # noqa: SLF001


def test_runtime_recovery_retries_transient_retirement_channel_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"fire": 0, "delivery": 0, "opened": 0}

    class Service:
        projection_state = SimpleNamespace(employees={})

        def recover(self):
            return None

        def recover_replay_safe_action_required(self):
            return hire_service.ActionRequiredRecoveryResult(0, (), 0, 0)

        def list_states(self):
            return ()

        def mark_runtime_recovered(self):
            calls["opened"] += 1

    class Fire:
        def recover(self):
            calls["fire"] += 1
            return ()

    runtime = EmployeeDepartmentRuntime(runtime_enabled=True)
    runtime._service = Service()  # type: ignore[assignment]  # noqa: SLF001
    runtime._fire = Fire()  # type: ignore[assignment]  # noqa: SLF001
    runtime._execution_blockers = ("test-isolation",)  # noqa: SLF001
    monkeypatch.setattr(runtime, "_refresh_context_bindings", lambda _state: True)

    def recover_channel() -> tuple[str, ...]:
        calls["delivery"] += 1
        if calls["delivery"] == 1:
            raise RuntimeError("transient retirement Channel start")
        return ("agt_alpha",)

    monkeypatch.setattr(
        runtime,
        "_recover_retirement_delivery_channels",
        recover_channel,
    )

    with pytest.raises(RuntimeError, match="transient retirement Channel start"):
        runtime.recover()

    assert runtime.recover().failed == 0
    assert calls == {"fire": 2, "delivery": 2, "opened": 1}


def test_reporting_recovers_retirement_channel_after_delayed_outbox_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = EmployeeDepartmentRuntime(runtime_enabled=True)
    runtime._dispatch_recovery_pending = False  # noqa: SLF001
    runtime._employee_dispatch_next_gc_at = float("inf")  # noqa: SLF001
    calls: list[str] = []

    monkeypatch.setattr(
        runtime,
        "_reconcile_terminal_ingress",
        lambda: calls.append("terminal_response_anchored") or 1,
    )
    monkeypatch.setattr(
        runtime,
        "_recover_retirement_delivery_channels",
        lambda: calls.append("delivery_channel_ready") or ("agt_alpha",),
    )

    def drain_outbox() -> bool:
        assert calls == [
            "terminal_response_anchored",
            "delivery_channel_ready",
        ]
        calls.append("terminal_response_delivered")
        return True

    monkeypatch.setattr(runtime, "_drain_employee_outbox_once", drain_outbox)

    assert runtime._drain_employee_reporting_once() is True  # noqa: SLF001
    assert calls == [
        "terminal_response_anchored",
        "delivery_channel_ready",
        "terminal_response_delivered",
    ]


def test_real_hire_retirement_lease_delivers_before_quiesce_commits(
    tmp_path,
) -> None:
    hire_root = tmp_path / "real-hire"
    _seed_active_employee(hire_root, begin_revalidation=False)
    hire = _recovered_hire_service(_recovery_writer(hire_root, 2))
    writer = hire._writer  # noqa: SLF001
    ingress = EmployeeIngressService(
        writer=writer,
        blob_store=BlobStore(
            tmp_path / "real-ingress-blobs",
            AesGcmEncryptionProvider(lambda _key_ref: b"i" * 32),
        ),
        ingress_state=IngressProjectionState(),
        active_key_id="real-ingress-key",
    )
    outbox = EmployeeOutboxService(
        writer=writer,
        blob_store=BlobStore(
            tmp_path / "real-outbox-blobs",
            AesGcmEncryptionProvider(lambda _key_ref: b"o" * 32),
        ),
        outbox_state=OutboxProjectionState(),
        active_key_id="real-outbox-key",
    )
    lifecycle = EmployeeOutboxLifecycle(outbox)
    raw_chat_id = "oc_retirement_control"
    raw_message_id = "om_retirement_control"
    raw_root_id = "om_retirement_root"
    payload = EmployeeIngressPayload(
        schema_version=1,
        envelope_id="ing_" + "a" * 64,
        normalized_parts=(
            {
                "type": "message",
                "message_type": "text",
                "chat_type": "p2p",
                "content": {"text": "/status details"},
                "sender_id": "ou_admin",
                "sender_union_id": "on_admin",
                "sender_id_type": "open_id",
                "sender_type": "user",
                "sender_tenant_key": "tenant-a",
                "feishu_thread_id": "omt_retirement_control",
                "remote_chat_id": raw_chat_id,
                "remote_message_id": raw_message_id,
                "remote_root_id": raw_root_id,
            },
        ),
        attachment_descriptors=(),
    )
    metadata = EmployeeIngressMetadata(
        schema_version=1,
        envelope_id=payload.envelope_id,
        tenant_key="tenant-a",
        agent_id="agt_recover",
        bot_principal_id="bot_recover",
        app_id="cli_employee",
        channel_generation=1,
        connection_id="conn_generation_1",
        event_id="evt_retirement_control",
        message_id="om_" + hashlib.sha256(raw_message_id.encode()).hexdigest(),
        event_type="im.message.receive_v1",
        action_identity="",
        chat_id="oc_" + hashlib.sha256(raw_chat_id.encode()).hexdigest(),
        thread_root_message_id=(
            "om_" + hashlib.sha256(raw_root_id.encode()).hexdigest()
        ),
        sender_principal_id="ou_admin",
        received_at="2026-08-12T00:00:00Z",
        semantic_digest=payload.payload_sha256,
        payload_sha256=payload.payload_sha256,
        payload_size_bytes=payload.canonical_size_bytes,
        attachment_count=0,
        attachment_total_bytes=0,
    )
    acceptance_id = ingress.accept(
        metadata,
        payload,
        request_id="req_retirement_control",
    ).acceptance.acceptance_id
    ingress_record = ingress.record_snapshot(acceptance_id)
    assert ingress_record is not None
    events: list[str] = []
    channels = _ProductionLeaseChannels(events)
    requester_acl = RuntimeRequesterChatAcl(
        allowed_requesters=("ou_admin",),
    )

    class _Membership:
        def is_degraded(self, _agent_id: str, _team_id: str) -> bool:
            return False

    router = DurableEmployeeIngressRouter(
        writer=writer,
        ingress_service=ingress,
        registry_provider=lambda: ProjectedAgentRegistry(
            hire.projection_state,
            storage_base_path=str(tmp_path / "real-router-registry"),
        ),
        channel_status_provider=channels,
        requester_acl=requester_acl,
        queue_limits=RouterQueueLimits(4, 8, 16),
        membership_health=_Membership(),
    )
    retired: set[str] = set()
    actor_runtime = SimpleNamespace(
        retire_employee=lambda agent_id: retired.add(agent_id),
        is_retired=lambda agent_id: agent_id in retired,
    )
    coordinator = SimpleNamespace(
        state=SimpleNamespace(attempts={}),
        employee_runtime=actor_runtime,
        begin_employee_retirement=lambda **_kwargs: 0,
        employee_retirement_ready=lambda **_kwargs: (
            lifecycle.terminal_response_delivered(
                tenant_key="tenant-a",
                agent_id="agt_recover",
                attempt_id=f"control_{acceptance_id}",
            )
        ),
    )
    effects = {name: _NoopEffect() for name in FIRE_EFFECT_ORDER}
    effects["execution_quiesce"] = ExecutionQuiesceEffect(
        coordinator,
        grace_seconds=0,
    )
    effects["channel_stop"] = ChannelStopEffect(channels)
    authority = JournalFireAuthority(
        writer=writer,
        hire_service=hire,
        ingress_service=ingress,
        admin_principal_ids=frozenset({"ou_admin"}),
    )
    fire = EmployeeFireService(
        writer=writer,
        authority=authority,
        effects=effects,
    )
    runtime = object.__new__(EmployeeDepartmentRuntime)
    runtime._service = hire  # noqa: SLF001
    runtime._channels = channels  # noqa: SLF001
    runtime._fire = fire  # noqa: SLF001
    runtime._ingress = ingress  # noqa: SLF001
    runtime._router = router  # noqa: SLF001
    runtime._dispatch = None  # noqa: SLF001
    runtime._outbox = outbox  # noqa: SLF001
    runtime._outbox_lifecycle = lifecycle  # noqa: SLF001
    runtime._context_acl = requester_acl  # noqa: SLF001
    runtime._managed_group_registry = None  # noqa: SLF001
    runtime._employee_outbox_delivery_lock = threading.RLock()  # noqa: SLF001
    runtime._outbox_delivery = EmployeeOutboxDeliveryCoordinator(  # noqa: SLF001
        outbox=outbox,
        channels=channels,
        authority_resolver=runtime._resolve_outbox_delivery_authority,
    )
    # Exercise the production control consumer while the employee is active,
    # but leave its durable response pending so `/fire` must preserve exactly
    # the cutoff-owned outbound obligation.
    runtime._request_employee_outbox_delivery = lambda: None  # type: ignore[method-assign]  # noqa: SLF001
    try:
        assert runtime._handle_control_ingress(acceptance_id) is True  # noqa: SLF001
        router.rebuild_projection()
        routed = router.record_snapshot(acceptance_id)
        assert routed is not None
        assert routed.state == "accepted"
        assert routed.queued_sequence == 0
        assert ingress.record_snapshot(acceptance_id).disposition.reason_code == (
            "status_invalid_arguments"
        )

        waiting = fire.start_fire(
            EmployeeFireRequest(
                employee="Atlas",
                tenant_key="tenant-a",
                message_id="om_real_hire_fire",
                chat_id="oc_admin",
                requester_principal_id="ou_admin",
            )
        )
        assert waiting.phase is FirePhase.RETIRING
        assert (
            waiting.effect_state("execution_quiesce")
            is FireEffectState.EXECUTING
        )
        assert hire.list_states()[0].phase is HirePhase.RETIRING
        control_id = employee_outbox_id(
            "tenant-a",
            "agt_recover",
            f"control_{acceptance_id}",
        )
        pending = outbox.get_record(control_id)
        assert runtime._retirement_outbox_is_cutoff_owned(  # noqa: SLF001
            pending,
            requested_sequence=waiting.requested_sequence,
        )
        assert not runtime._retirement_outbox_is_cutoff_owned(  # noqa: SLF001
            replace(pending, chat_id="oc_wrong_retirement_target"),
            requested_sequence=waiting.requested_sequence,
        )
        assert not runtime._retirement_outbox_is_cutoff_owned(  # noqa: SLF001
            replace(pending, thread_root_message_id="om_wrong_retirement_root"),
            requested_sequence=waiting.requested_sequence,
        )
        assert runtime._drain_employee_outbox_once() is True  # noqa: SLF001
        bound = outbox.get_record(control_id)
        assert bound.binding is not None
        assert bound.binding.bound_snapshot_version == bound.latest_version
        assert fire.list_states()[0].effect_state("execution_quiesce") is (
            FireEffectState.EXECUTING
        )

        progressed = fire.reconcile_draining()

        assert len(progressed) == 1
        assert progressed[0].phase is FirePhase.ARCHIVED
        assert progressed[0].effect_state("execution_quiesce") is (
            FireEffectState.COMMITTED
        )
        assert events == ["failure_response_delivered", "channel_stop"]
    finally:
        ingress.close()
        outbox.close()
        hire.close()


def test_fire_without_tasks_remains_synchronous(tmp_path) -> None:
    writer, state, ingress, authority = _active_bound_fire_authority(tmp_path)
    events: list[str] = []
    retired: set[str] = set()
    runtime = SimpleNamespace(
        retire_employee=lambda agent_id: retired.add(agent_id),
        is_retired=lambda agent_id: agent_id in retired,
    )
    coordinator = SimpleNamespace(
        state=SimpleNamespace(attempts={}),
        employee_runtime=runtime,
        begin_employee_retirement=lambda **_kwargs: 0,
        employee_retirement_ready=lambda **_kwargs: True,
    )
    effects = {name: _NoopEffect() for name in FIRE_EFFECT_ORDER}
    effects["execution_quiesce"] = ExecutionQuiesceEffect(
        coordinator,
        grace_seconds=0,
    )
    effects["channel_stop"] = ChannelStopEffect(_FireChannels(events))
    service = EmployeeFireService(
        writer=writer,
        authority=authority,
        effects=effects,
    )
    try:
        result = service.start_fire(
            EmployeeFireRequest(
                employee="Atlas",
                tenant_key="tenant_1",
                message_id="om_fire_without_tasks",
                chat_id="oc_admin",
                requester_principal_id="ou_admin",
            )
        )

        assert result.phase is FirePhase.ARCHIVED
        assert retired == {"agt_1"}
        assert events == ["channel_stop"]
        assert state.employees["agt_1"].state.value == "archived"
    finally:
        ingress.close()
        writer.close()
