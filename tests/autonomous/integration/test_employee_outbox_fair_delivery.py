from __future__ import annotations

import threading
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from src.autonomous.domain.employees import BotPrincipal, EmployeeDefinition
from src.autonomous.domain.enums import EmployeeState, WorkerType
from src.autonomous.journal.anchor import MemoryAnchor
from src.autonomous.journal.blob_store import AesGcmEncryptionProvider, BlobStore
from src.autonomous.journal.writer import JournalWriter
from src.autonomous.outbox.delivery import (
    EmployeeDeliveryAuthority,
    EmployeeOutboxDeliveryCoordinator,
    EmployeeOutboxItemDeliveryError,
)
from src.autonomous.outbox.models import (
    EmployeeCardState,
    EmployeeOutboxSnapshot,
    employee_outbox_id,
)
from src.autonomous.outbox.projection import (
    OutboxProjectionError,
    OutboxProjectionState,
)
from src.autonomous.outbox.service import (
    EmployeeOutboxService,
    OutboxWriteDisabledError,
)
from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime
from src.autonomous.provisioning.hire_state import DurableHireState, HirePhase
from src.autonomous.supervisor.channel_models import EmployeeChannelOutboundError
from src.autonomous.supervisor.employee_channels import (
    ChannelProcessState,
    ChannelProcessStatus,
    ChannelSendReceipt,
)


def _snapshot(
    *,
    agent_id: str,
    attempt_id: str,
    created_at: str,
) -> EmployeeOutboxSnapshot:
    return EmployeeOutboxSnapshot(
        schema_version=1,
        outbox_id=employee_outbox_id("tenant-a", agent_id, attempt_id),
        tenant_key="tenant-a",
        agent_id=agent_id,
        attempt_id=attempt_id,
        chat_id="oc_team",
        thread_root_message_id="om_root",
        version=1,
        state=EmployeeCardState.QUEUED,
        title="排队任务",
        summary="任务已进入员工队列",
        progress_percent=0,
        card_json={"schema": "2.0", "body": {"elements": []}},
        created_at=created_at,
        terminal_version=0,
    )


def _runtime(tmp_path):
    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=MemoryAnchor(),
        hmac_key=b"j" * 32,
        writer_epoch=1,
    )
    service = EmployeeOutboxService(
        writer=writer,
        blob_store=BlobStore(
            tmp_path / "outbox-blobs",
            AesGcmEncryptionProvider(lambda _ref: b"b" * 32),
        ),
        outbox_state=OutboxProjectionState(),
        active_key_id="k1",
    )
    return service, writer


@dataclass
class _Channel:
    calls: list[str]

    def send(self, agent_id, *, generation, target, message, options=None):
        del target, message, options
        self.calls.append(agent_id)
        return ChannelSendReceipt(
            request_id=f"send_{agent_id}",
            success=True,
            app_id=f"cli_{agent_id}",
            generation=generation,
            connection_id=f"conn_{agent_id}",
            message_id=f"om_{agent_id}",
        )

    def update_card(self, agent_id, *, generation, message_id, card):
        del card
        self.calls.append(agent_id)
        return ChannelSendReceipt(
            request_id=f"update_{agent_id}",
            success=True,
            app_id=f"cli_{agent_id}",
            generation=generation,
            connection_id=f"conn_{agent_id}",
            message_id=message_id,
        )


def _coordinator(service, channel, available):
    def resolve(record):
        if record.agent_id not in available:
            raise EmployeeOutboxItemDeliveryError(
                "employee delivery authority is unavailable"
            )
        return EmployeeDeliveryAuthority(
            app_id=f"cli_{record.agent_id}",
            generation=1,
            connection_id=f"conn_{record.agent_id}",
        )

    return EmployeeOutboxDeliveryCoordinator(
        outbox=service,
        channels=channel,
        authority_resolver=resolve,
    )


def test_bounded_fair_drain_skips_unavailable_oldest_and_delivers_next(
    tmp_path,
) -> None:
    service, writer = _runtime(tmp_path)
    oldest = _snapshot(
        agent_id="agt_oldest",
        attempt_id="attempt-oldest",
        created_at="2026-08-12T00:00:00Z",
    )
    healthy = _snapshot(
        agent_id="agt_healthy",
        attempt_id="attempt-healthy",
        created_at="2026-08-12T00:00:01Z",
    )
    service.append_snapshot(oldest)
    service.append_snapshot(healthy)
    channel = _Channel(calls=[])
    coordinator = _coordinator(service, channel, {"agt_healthy"})
    try:
        result = coordinator.deliver_pending(max_items=2)

        assert result.pending_count == 2
        assert result.attempted_outbox_ids == (
            oldest.outbox_id,
            healthy.outbox_id,
        )
        assert result.failed_outbox_ids == (oldest.outbox_id,)
        assert result.delivered_outbox_ids == (healthy.outbox_id,)
        assert result.made_progress is True
        assert channel.calls == ["agt_healthy"]
        assert service.get_record(oldest.outbox_id).binding is None
        assert service.get_record(healthy.outbox_id).binding is not None
    finally:
        service.close()
        writer.close()
def test_production_resolver_isolates_oldest_missing_identity_and_delivers_next(
    tmp_path,
) -> None:
    service, writer = _runtime(tmp_path)
    oldest = _snapshot(
        agent_id="agt_oldest",
        attempt_id="attempt-oldest-production-resolver",
        created_at="2026-08-12T00:00:00Z",
    )
    healthy = _snapshot(
        agent_id="agt_healthy",
        attempt_id="attempt-healthy-production-resolver",
        created_at="2026-08-12T00:00:01Z",
    )
    service.append_snapshot(oldest)
    service.append_snapshot(healthy)

    app_id = "cli_agt_healthy"
    bot_principal_id = "bot_healthy"
    connection_id = "conn_agt_healthy"
    employee = EmployeeDefinition(
        agent_id=healthy.agent_id,
        tenant_key=healthy.tenant_key,
        owner_principal_id="ou_owner",
        name="Healthy",
        tool="codex",
        worker_type=WorkerType.VISIBLE,
        state=EmployeeState.ACTIVE,
        bot_principal_id=bot_principal_id,
    )
    principal = BotPrincipal(
        bot_principal_id=bot_principal_id,
        tenant_key=healthy.tenant_key,
        agent_id=healthy.agent_id,
        app_id=app_id,
        credential_ref="vault://healthy",
    )
    hire = DurableHireState(
        intent_id="hire_healthy",
        tenant_key=healthy.tenant_key,
        agent_id=healthy.agent_id,
        bot_principal_id=bot_principal_id,
        app_id=app_id,
        credential_ref=principal.credential_ref,
        channel_generation=1,
        channel_identity_app_id=app_id,
        channel_connection_id=connection_id,
        phase=HirePhase.ACTIVE,
    )

    class _ProductionResolverChannel(_Channel):
        def status(self, agent_id: str) -> ChannelProcessStatus | None:
            if agent_id != healthy.agent_id:
                return None
            return ChannelProcessStatus(
                agent_id=healthy.agent_id,
                app_id=app_id,
                generation=1,
                pid=101,
                state=ChannelProcessState.READY,
                tenant_key=healthy.tenant_key,
                bot_principal_id=bot_principal_id,
                identity={"app_id": app_id},
                ready_metadata={"connection_id": connection_id},
            )

    channel = _ProductionResolverChannel(calls=[])
    runtime = EmployeeDepartmentRuntime()
    runtime._service = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        current_employee_transport_snapshot=lambda: (
            (employee,),
            (principal,),
            (hire,),
        )
    )
    runtime._channels = channel  # type: ignore[assignment]  # noqa: SLF001
    runtime._outbox = service  # type: ignore[assignment]  # noqa: SLF001
    runtime._outbox_delivery = EmployeeOutboxDeliveryCoordinator(  # noqa: SLF001
        outbox=service,
        channels=channel,
        authority_resolver=runtime._resolve_outbox_delivery_authority,  # noqa: SLF001
    )
    try:
        assert runtime._drain_employee_outbox_once() is True  # noqa: SLF001
        assert channel.calls == [healthy.agent_id]
        assert service.get_record(oldest.outbox_id).binding is None
        assert service.get_record(healthy.outbox_id).binding is not None
    finally:
        service.close()
        writer.close()


def test_production_resolver_configuration_failure_propagates_without_isolation(
    tmp_path,
) -> None:
    service, writer = _runtime(tmp_path)
    oldest = _snapshot(
        agent_id="agt_oldest",
        attempt_id="attempt-programming-error",
        created_at="2026-08-12T00:00:00Z",
    )
    service.append_snapshot(oldest)
    channel = _Channel(calls=[])
    runtime = EmployeeDepartmentRuntime()
    runtime._service = object()  # type: ignore[assignment]  # noqa: SLF001
    runtime._channels = channel  # type: ignore[assignment]  # noqa: SLF001
    coordinator = EmployeeOutboxDeliveryCoordinator(
        outbox=service,
        channels=channel,
        authority_resolver=runtime._resolve_outbox_delivery_authority,  # noqa: SLF001
    )
    try:
        with pytest.raises(RuntimeError, match="snapshot is unavailable"):
            coordinator.deliver_pending(max_items=1)
        assert channel.calls == []
    finally:
        service.close()
        writer.close()


def test_production_resolver_duplicate_hire_authority_is_not_isolated(
    tmp_path,
) -> None:
    service, writer = _runtime(tmp_path)
    oldest = _snapshot(
        agent_id="agt_ambiguous",
        attempt_id="attempt-ambiguous-hire",
        created_at="2026-08-12T00:00:00Z",
    )
    service.append_snapshot(oldest)
    employee = EmployeeDefinition(
        agent_id=oldest.agent_id,
        tenant_key=oldest.tenant_key,
        owner_principal_id="ou_owner",
        name="Ambiguous",
        tool="codex",
        worker_type=WorkerType.VISIBLE,
        state=EmployeeState.ACTIVE,
        bot_principal_id="bot_ambiguous",
    )
    principal = BotPrincipal(
        bot_principal_id=employee.bot_principal_id,
        tenant_key=oldest.tenant_key,
        agent_id=oldest.agent_id,
        app_id="cli_ambiguous",
        credential_ref="vault://ambiguous",
    )
    hire = DurableHireState(
        intent_id="hire_ambiguous",
        tenant_key=oldest.tenant_key,
        agent_id=oldest.agent_id,
        bot_principal_id=employee.bot_principal_id,
        app_id=principal.app_id,
        credential_ref=principal.credential_ref,
        channel_generation=1,
        channel_identity_app_id=principal.app_id,
        channel_connection_id="conn_ambiguous",
        phase=HirePhase.ACTIVE,
    )
    channel = _Channel(calls=[])
    runtime = EmployeeDepartmentRuntime()
    runtime._service = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        current_employee_transport_snapshot=lambda: (
            (employee,),
            (principal,),
            (hire, hire),
        )
    )
    runtime._channels = channel  # type: ignore[assignment]  # noqa: SLF001
    coordinator = EmployeeOutboxDeliveryCoordinator(
        outbox=service,
        channels=channel,
        authority_resolver=runtime._resolve_outbox_delivery_authority,  # noqa: SLF001
    )
    try:
        with pytest.raises(RuntimeError, match="ambiguous") as raised:
            coordinator.deliver_pending(max_items=1)
        assert not isinstance(raised.value, EmployeeOutboxItemDeliveryError)
        assert channel.calls == []
    finally:
        service.close()
        writer.close()


def test_explicit_channel_failure_isolated_without_starving_healthy_record(
    tmp_path,
) -> None:
    service, writer = _runtime(tmp_path)
    oldest = _snapshot(
        agent_id="agt_oldest",
        attempt_id="attempt-oldest",
        created_at="2026-08-12T00:00:00Z",
    )
    healthy = _snapshot(
        agent_id="agt_healthy",
        attempt_id="attempt-healthy",
        created_at="2026-08-12T00:00:01Z",
    )
    service.append_snapshot(oldest)
    service.append_snapshot(healthy)

    class _UnavailableOldestChannel(_Channel):
        def send(self, agent_id, *, generation, target, message, options=None):
            if agent_id == oldest.agent_id:
                raise EmployeeChannelOutboundError("employee Channel is not ready")
            return super().send(
                agent_id,
                generation=generation,
                target=target,
                message=message,
                options=options,
            )

    channel = _UnavailableOldestChannel(calls=[])
    coordinator = _coordinator(
        service,
        channel,
        {oldest.agent_id, healthy.agent_id},
    )
    try:
        result = coordinator.deliver_pending(max_items=2)

        assert result.failed_outbox_ids == (oldest.outbox_id,)
        assert result.delivered_outbox_ids == (healthy.outbox_id,)
        assert channel.calls == [healthy.agent_id]
    finally:
        service.close()
        writer.close()


def test_fair_drain_rotates_one_attempt_per_call_and_retries_recovered_record(
    tmp_path,
) -> None:
    service, writer = _runtime(tmp_path)
    oldest = _snapshot(
        agent_id="agt_oldest",
        attempt_id="attempt-oldest",
        created_at="2026-08-12T00:00:00Z",
    )
    healthy = _snapshot(
        agent_id="agt_healthy",
        attempt_id="attempt-healthy",
        created_at="2026-08-12T00:00:01Z",
    )
    service.append_snapshot(oldest)
    service.append_snapshot(healthy)
    available = {"agt_healthy"}
    channel = _Channel(calls=[])
    coordinator = _coordinator(service, channel, available)
    try:
        failed = coordinator.deliver_pending(max_items=1)
        advanced = coordinator.deliver_pending(max_items=1)
        available.add("agt_oldest")
        recovered = coordinator.deliver_pending(max_items=1)

        assert failed.attempted_outbox_ids == (oldest.outbox_id,)
        assert failed.failed_outbox_ids == (oldest.outbox_id,)
        assert failed.made_progress is False
        assert advanced.attempted_outbox_ids == (healthy.outbox_id,)
        assert advanced.delivered_outbox_ids == (healthy.outbox_id,)
        assert recovered.attempted_outbox_ids == (oldest.outbox_id,)
        assert recovered.delivered_outbox_ids == (oldest.outbox_id,)
        assert channel.calls == ["agt_healthy", "agt_oldest"]
    finally:
        service.close()
        writer.close()


def test_fair_drain_enforces_the_requested_batch_bound(tmp_path) -> None:
    service, writer = _runtime(tmp_path)
    snapshots = tuple(
        _snapshot(
            agent_id=f"agt_{index}",
            attempt_id=f"attempt-{index}",
            created_at=f"2026-08-12T00:00:0{index}Z",
        )
        for index in range(3)
    )
    for snapshot in snapshots:
        service.append_snapshot(snapshot)
    channel = _Channel(calls=[])
    coordinator = _coordinator(
        service,
        channel,
        {snapshot.agent_id for snapshot in snapshots},
    )
    try:
        result = coordinator.deliver_pending(max_items=2)

        assert len(result.attempted_outbox_ids) == 2
        assert len(result.delivered_outbox_ids) == 2
        assert len(channel.calls) == 2
        assert service.get_record(snapshots[2].outbox_id).binding is None
    finally:
        service.close()
        writer.close()


def test_continuous_new_arrivals_do_not_starve_a_failed_record(tmp_path) -> None:
    service, writer = _runtime(tmp_path)
    oldest = _snapshot(
        agent_id="agt_oldest",
        attempt_id="attempt-oldest",
        created_at="2026-08-12T00:00:00Z",
    )
    service.append_snapshot(oldest)
    available: set[str] = set()
    authority_attempts: list[str] = []
    channel = _Channel(calls=[])

    def resolve(record):
        authority_attempts.append(record.agent_id)
        if record.agent_id not in available:
            raise EmployeeOutboxItemDeliveryError(
                "employee delivery authority is unavailable"
            )
        return EmployeeDeliveryAuthority(
            app_id=f"cli_{record.agent_id}",
            generation=1,
            connection_id=f"conn_{record.agent_id}",
        )

    coordinator = EmployeeOutboxDeliveryCoordinator(
        outbox=service,
        channels=channel,
        authority_resolver=resolve,
    )
    try:
        coordinator.deliver_pending(max_items=1)
        for index in range(1, 6):
            newcomer = _snapshot(
                agent_id=f"agt_new_{index}",
                attempt_id=f"attempt-new-{index}",
                created_at=f"2026-08-12T00:00:0{index}Z",
            )
            service.append_snapshot(newcomer)
            available.add(newcomer.agent_id)
            coordinator.deliver_pending(max_items=1)

        assert authority_attempts.count("agt_oldest") >= 2
    finally:
        service.close()
        writer.close()


def test_concurrent_fair_drains_never_send_the_same_effect_twice(tmp_path) -> None:
    service, writer = _runtime(tmp_path)
    snapshot = _snapshot(
        agent_id="agt_one",
        attempt_id="attempt-one",
        created_at="2026-08-12T00:00:00Z",
    )
    service.append_snapshot(snapshot)

    class _RacingChannel(_Channel):
        rendezvous = threading.Barrier(2)

        def send(self, agent_id, *, generation, target, message, options=None):
            try:
                self.rendezvous.wait(timeout=0.2)
            except threading.BrokenBarrierError:
                pass
            return super().send(
                agent_id,
                generation=generation,
                target=target,
                message=message,
                options=options,
            )

    channel = _RacingChannel(calls=[])
    coordinator = _coordinator(service, channel, {snapshot.agent_id})
    start = threading.Barrier(3)
    results = []
    errors: list[BaseException] = []

    def drain() -> None:
        try:
            start.wait(timeout=2.0)
            results.append(coordinator.deliver_pending(max_items=1))
        except BaseException as exc:  # pragma: no cover - assertion reports worker faults
            errors.append(exc)

    workers = [threading.Thread(target=drain) for _ in range(2)]
    for worker in workers:
        worker.start()
    start.wait(timeout=2.0)
    for worker in workers:
        worker.join(timeout=3.0)
    try:
        assert not errors
        assert all(not worker.is_alive() for worker in workers)
        assert len(results) == 2
        assert channel.calls == ["agt_one"]
    finally:
        service.close()
        writer.close()


@pytest.mark.parametrize(
    "error",
    [
        OutboxProjectionError("projection corrupt"),
        OutboxWriteDisabledError("anchor unavailable"),
        AssertionError("programming defect"),
        RuntimeError("untyped callback defect"),
    ],
)
def test_fair_drain_never_isolates_integrity_or_programming_errors(
    tmp_path,
    monkeypatch,
    error,
) -> None:
    service, writer = _runtime(tmp_path)
    snapshot = _snapshot(
        agent_id="agt_one",
        attempt_id="attempt-one",
        created_at="2026-08-12T00:00:00Z",
    )
    service.append_snapshot(snapshot)
    coordinator = _coordinator(service, _Channel(calls=[]), {snapshot.agent_id})
    monkeypatch.setattr(service, "prepare_delivery", lambda *_args: (_ for _ in ()).throw(error))
    try:
        with pytest.raises(type(error), match=str(error)):
            coordinator.deliver_pending(max_items=1)
    finally:
        service.close()
        writer.close()
