from __future__ import annotations

from dataclasses import dataclass

from src.autonomous.journal.anchor import MemoryAnchor
from src.autonomous.journal.blob_store import AesGcmEncryptionProvider, BlobStore
from src.autonomous.journal.writer import JournalWriter
from src.autonomous.outbox.delivery import (
    EmployeeDeliveryAuthority,
    EmployeeOutboxDeliveryCoordinator,
)
from src.autonomous.outbox.models import (
    EmployeeCardState,
    EmployeeOutboxSnapshot,
    employee_outbox_id,
)
from src.autonomous.outbox.projection import OutboxProjectionState
from src.autonomous.outbox.service import EmployeeOutboxService
from src.autonomous.supervisor.employee_channels import ChannelSendReceipt


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
            raise RuntimeError("employee delivery authority is unavailable")
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
