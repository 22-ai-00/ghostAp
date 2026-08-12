from __future__ import annotations

from dataclasses import replace

import pytest

from src.autonomous.outbox.delivery import (
    EmployeeDeliveryAuthority,
    EmployeeOutboxDeliveryCoordinator,
    EmployeeOutboxItemDeliveryError,
    EmployeeOutboxReceiptIntegrityError,
)
from src.autonomous.outbox.models import (
    DeliveryEffectState,
    EmployeeCardState,
    employee_outbox_id,
)
from src.autonomous.supervisor.employee_channels import ChannelSendReceipt
from tests.autonomous.integration.test_employee_outbox_delivery import (
    _Channel,
    _runtime,
    _snapshot,
)


def _coordinator(service, channel) -> EmployeeOutboxDeliveryCoordinator:
    return EmployeeOutboxDeliveryCoordinator(
        outbox=service,
        channels=channel,
        authority_resolver=lambda _record: EmployeeDeliveryAuthority(
            app_id="cli_employee",
            generation=3,
            connection_id="conn_employee",
        ),
    )


def test_mismatched_create_receipt_is_non_retryable_integrity_failure(
    tmp_path,
) -> None:
    service, writer, _anchor = _runtime(tmp_path)
    channel = _Channel(writer)
    coordinator = _coordinator(service, channel)
    snapshot = _snapshot()
    service.append_snapshot(snapshot)
    channel.send = lambda *args, **kwargs: ChannelSendReceipt(
        request_id="send_bad",
        success=True,
        app_id="cli_other",
        generation=3,
        connection_id="conn_employee",
        message_id="om_employee_card",
    )
    try:
        with pytest.raises(
            EmployeeOutboxReceiptIntegrityError,
            match="receipt",
        ) as raised:
            coordinator.deliver(snapshot.outbox_id)

        assert not isinstance(raised.value, EmployeeOutboxItemDeliveryError)
        record = service.get_record(snapshot.outbox_id)
        assert record.binding is None
        assert next(iter(record.effects.values())).state is DeliveryEffectState.EXECUTING
    finally:
        service.close()
        writer.close()


def test_mismatched_patch_receipt_is_non_retryable_integrity_failure(
    tmp_path,
) -> None:
    service, writer, _anchor = _runtime(tmp_path)
    channel = _Channel(writer)
    coordinator = _coordinator(service, channel)
    queued = _snapshot()
    service.append_snapshot(queued)
    try:
        assert coordinator.deliver(queued.outbox_id) is not None
        running = replace(
            queued,
            version=2,
            state=EmployeeCardState.RUNNING,
            progress_percent=50,
        )
        service.append_snapshot(running)
        channel.update_card = lambda *args, **kwargs: ChannelSendReceipt(
            request_id="update_bad",
            success=True,
            app_id="cli_employee",
            generation=3,
            connection_id="conn_employee",
            message_id="om_other_card",
        )

        with pytest.raises(
            EmployeeOutboxReceiptIntegrityError,
            match="receipt",
        ) as raised:
            coordinator.deliver(running.outbox_id)

        assert not isinstance(raised.value, EmployeeOutboxItemDeliveryError)
        record = service.get_record(running.outbox_id)
        assert record.binding is not None
        assert record.binding.bound_snapshot_version == 1
        patch = max(record.effects.values(), key=lambda effect: effect.snapshot_version)
        assert patch.snapshot_version == 2
        assert patch.state is DeliveryEffectState.EXECUTING
    finally:
        service.close()
        writer.close()


def test_fair_batch_never_isolates_mismatched_receipt(tmp_path) -> None:
    service, writer, _anchor = _runtime(tmp_path)
    oldest = _snapshot(
        outbox_id=employee_outbox_id(
            "tenant-a",
            "agt_oldest",
            "attempt-oldest-bad-receipt",
        ),
        agent_id="agt_oldest",
        attempt_id="attempt-oldest-bad-receipt",
        created_at="2026-08-12T00:00:00Z",
    )
    healthy = _snapshot(
        outbox_id=employee_outbox_id(
            "tenant-a",
            "agt_healthy",
            "attempt-healthy-after-bad-receipt",
        ),
        agent_id="agt_healthy",
        attempt_id="attempt-healthy-after-bad-receipt",
        created_at="2026-08-12T00:00:01Z",
    )
    service.append_snapshot(oldest)
    service.append_snapshot(healthy)

    class _MismatchedReceiptChannel(_Channel):
        def send(self, agent_id, *, generation, target, message, options=None):
            receipt = super().send(
                agent_id,
                generation=generation,
                target=target,
                message=message,
                options=options,
            )
            if agent_id != oldest.agent_id:
                return receipt
            return replace(receipt, app_id="cli_wrong")

    channel = _MismatchedReceiptChannel(writer)
    coordinator = _coordinator(service, channel)
    try:
        with pytest.raises(EmployeeOutboxReceiptIntegrityError, match="receipt"):
            coordinator.deliver_pending(max_items=2)

        assert [call[1] for call in channel.calls] == [oldest.agent_id]
        assert service.get_record(healthy.outbox_id).binding is None
    finally:
        service.close()
        writer.close()
