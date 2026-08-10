from __future__ import annotations

import inspect
from dataclasses import dataclass, replace

import pytest

from src.autonomous.journal.anchor import MemoryAnchor
from src.autonomous.journal.blob_store import AesGcmEncryptionProvider, BlobStore
from src.autonomous.journal.writer import JournalWriter
from src.autonomous.outbox.delivery import (
    EmployeeDeliveryAuthority,
    EmployeeOutboxDeliveryCoordinator,
)
from src.autonomous.outbox.lifecycle import EmployeeOutboxLifecycle
from src.autonomous.outbox.models import (
    DeliveryEffectState,
    EmployeeCardState,
    EmployeeOutboxSnapshot,
    employee_outbox_id,
    employee_outbox_uuid,
)
from src.autonomous.outbox.projection import OutboxProjectionState
from src.autonomous.outbox.service import EmployeeOutboxService
from src.autonomous.supervisor.employee_channels import ChannelSendReceipt


def _snapshot(**overrides: object) -> EmployeeOutboxSnapshot:
    values: dict[str, object] = {
        "schema_version": 1,
        "outbox_id": employee_outbox_id("tenant-a", "agt_alpha", "attempt-a"),
        "tenant_key": "tenant-a",
        "agent_id": "agt_alpha",
        "attempt_id": "attempt-a",
        "chat_id": "oc_team",
        "thread_root_message_id": "om_root",
        "version": 1,
        "state": EmployeeCardState.QUEUED,
        "title": "修复登录回调",
        "summary": "任务已进入员工队列",
        "progress_percent": 0,
        "card_json": {"schema": "2.0", "body": {"elements": []}},
        "created_at": "2026-07-14T00:00:00Z",
        "terminal_version": 0,
    }
    values.update(overrides)
    return EmployeeOutboxSnapshot(**values)


def _runtime(tmp_path, *, anchor=None, epoch=1):
    anchor = anchor or MemoryAnchor()
    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=anchor,
        hmac_key=b"j" * 32,
        writer_epoch=epoch,
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
    return service, writer, anchor


@dataclass
class _Status:
    app_id: str = "cli_employee"
    generation: int = 3
    connection_id: str = "conn_employee"


class _Channel:
    def __init__(self, writer: JournalWriter, *, wrong_message: bool = False) -> None:
        self.writer = writer
        self.wrong_message = wrong_message
        self.calls: list[tuple] = []

    def send(self, agent_id, *, generation, target, message, options=None):
        event_types = [event.event_type for frame in self.writer.replay() for event in frame.events]
        prepared = max(
            index for index, event_type in enumerate(event_types) if event_type == "employee.outbox.effect_prepared"
        )
        executing = max(
            index for index, event_type in enumerate(event_types) if event_type == "employee.outbox.effect_executing"
        )
        assert prepared < executing
        assert "employee.outbox.delivery_committed" not in event_types[executing + 1 :]
        self.calls.append(("send", agent_id, generation, target, message, options))
        return ChannelSendReceipt(
            request_id="send_1",
            success=True,
            app_id="cli_employee",
            generation=generation,
            connection_id="conn_employee",
            message_id="om_wrong" if self.wrong_message else "om_employee_card",
        )

    def update_card(self, agent_id, *, generation, message_id, card):
        event_types = [event.event_type for frame in self.writer.replay() for event in frame.events]
        assert event_types[-2:] == [
            "employee.outbox.effect_prepared",
            "employee.outbox.effect_executing",
        ]
        self.calls.append(("update", agent_id, generation, message_id, card))
        return ChannelSendReceipt(
            request_id="update_1",
            success=True,
            app_id="cli_employee",
            generation=generation,
            connection_id="conn_reconnected",
            message_id=message_id,
        )


def test_create_is_stable_uuid_employee_owned_and_effect_anchored_first(tmp_path) -> None:
    service, writer, _anchor = _runtime(tmp_path)
    channel = _Channel(writer)
    coordinator = EmployeeOutboxDeliveryCoordinator(
        outbox=service,
        channels=channel,
        authority_resolver=lambda _record: EmployeeDeliveryAuthority(
            app_id="cli_employee",
            generation=3,
            connection_id="conn_employee",
        ),
    )
    try:
        snapshot = _snapshot()
        service.append_snapshot(snapshot)

        binding = coordinator.deliver(snapshot.outbox_id)

        assert binding is not None
        assert binding.stable_uuid == employee_outbox_uuid(snapshot.outbox_id)
        assert binding.message_id == "om_employee_card"
        assert channel.calls == [
            (
                "send",
                "agt_alpha",
                3,
                "oc_team",
                {"card": snapshot.to_dict()["card_json"]},
                {
                    "uuid": binding.stable_uuid,
                    "reply_to": "om_root",
                    "reply_in_thread": True,
                },
            )
        ]
        effect = next(iter(service.state.by_outbox_id[snapshot.outbox_id].effects.values()))
        assert effect.state is DeliveryEffectState.COMMITTED
    finally:
        service.close()
        writer.close()


def test_delivery_coordinator_has_no_main_bot_fallback_port() -> None:
    parameters = inspect.signature(EmployeeOutboxDeliveryCoordinator.__init__).parameters
    source = inspect.getsource(EmployeeOutboxDeliveryCoordinator)

    assert set(parameters) == {
        "self",
        "outbox",
        "channels",
        "authority_resolver",
    }
    assert "main_bot" not in source.casefold()
    assert "reply_card" not in source
    assert "send_card_to_chat" not in source


def test_patch_keeps_message_and_rebinds_only_to_current_employee_authority(
    tmp_path,
) -> None:
    service, writer, _anchor = _runtime(tmp_path)
    channel = _Channel(writer)
    authority = EmployeeDeliveryAuthority(
        app_id="cli_employee",
        generation=3,
        connection_id="conn_employee",
    )
    coordinator = EmployeeOutboxDeliveryCoordinator(
        outbox=service,
        channels=channel,
        authority_resolver=lambda _record: authority,
    )
    try:
        queued = _snapshot()
        service.append_snapshot(queued)
        initial = coordinator.deliver(queued.outbox_id)
        running = replace(
            queued,
            version=2,
            state=EmployeeCardState.RUNNING,
            progress_percent=60,
        )
        service.append_snapshot(running)
        authority = EmployeeDeliveryAuthority(
            app_id="cli_employee",
            generation=4,
            connection_id="conn_reconnected",
        )

        rebound = coordinator.deliver(running.outbox_id)

        assert initial is not None and rebound is not None
        assert rebound.message_id == initial.message_id
        assert rebound.app_id == initial.app_id
        assert rebound.generation == 4
        assert rebound.connection_id == "conn_reconnected"
        assert rebound.bound_snapshot_version == 2
        assert channel.calls[-1][0:4] == (
            "update",
            "agt_alpha",
            4,
            "om_employee_card",
        )
    finally:
        service.close()
        writer.close()


def test_mismatched_create_receipt_is_not_committed(tmp_path) -> None:
    service, writer, _anchor = _runtime(tmp_path)
    channel = _Channel(writer, wrong_message=True)
    coordinator = EmployeeOutboxDeliveryCoordinator(
        outbox=service,
        channels=channel,
        authority_resolver=lambda _record: EmployeeDeliveryAuthority(
            app_id="cli_employee",
            generation=3,
            connection_id="conn_employee",
        ),
    )
    try:
        snapshot = _snapshot()
        service.append_snapshot(snapshot)
        # A create can choose a new message ID; mismatch here is induced through
        # the authority evidence instead of a pre-bound ID.
        channel.wrong_message = False
        channel.send = lambda *args, **kwargs: ChannelSendReceipt(
            request_id="send_bad",
            success=True,
            app_id="cli_other",
            generation=3,
            connection_id="conn_employee",
            message_id="om_employee_card",
        )

        with pytest.raises(RuntimeError, match="receipt"):
            coordinator.deliver(snapshot.outbox_id)

        record = service.state.by_outbox_id[snapshot.outbox_id]
        assert record.binding is None
        assert next(iter(record.effects.values())).state is DeliveryEffectState.EXECUTING
    finally:
        service.close()
        writer.close()


def test_restart_retries_unknown_create_with_same_uuid_without_new_effect(tmp_path) -> None:
    service, writer, anchor = _runtime(tmp_path)
    snapshot = _snapshot()
    service.append_snapshot(snapshot)
    effect = service.prepare_delivery(snapshot.outbox_id, snapshot.version)
    service.mark_effect_executing(effect.effect_id)
    service.close()
    writer.close()

    service2, writer2, _ = _runtime(tmp_path, anchor=anchor, epoch=2)
    channel = _Channel(writer2)
    coordinator = EmployeeOutboxDeliveryCoordinator(
        outbox=service2,
        channels=channel,
        authority_resolver=lambda _record: EmployeeDeliveryAuthority(
            app_id="cli_employee",
            generation=3,
            connection_id="conn_employee",
        ),
    )
    try:
        binding = coordinator.deliver(snapshot.outbox_id)

        assert binding is not None
        assert channel.calls[0][-1]["uuid"] == employee_outbox_uuid(snapshot.outbox_id)
        record = service2.state.by_outbox_id[snapshot.outbox_id]
        assert len(record.effects) == 1
        assert next(iter(record.effects.values())).effect_id == effect.effect_id
    finally:
        service2.close()
        writer2.close()


def test_newer_snapshot_cannot_overtake_unknown_create_effect(tmp_path) -> None:
    service, writer, _anchor = _runtime(tmp_path)
    try:
        queued = _snapshot()
        service.append_snapshot(queued)
        effect = service.prepare_delivery(queued.outbox_id, 1)
        service.mark_effect_executing(effect.effect_id)
        running = replace(
            queued,
            version=2,
            state=EmployeeCardState.RUNNING,
            progress_percent=60,
            summary="newer",
            card_json={"schema": "2.0", "version": 2},
        )
        service.append_snapshot(running)
        channel = _Channel(writer)
        coordinator = EmployeeOutboxDeliveryCoordinator(
            outbox=service,
            channels=channel,
            authority_resolver=lambda _record: EmployeeDeliveryAuthority(
                app_id="cli_employee",
                generation=3,
                connection_id="conn_employee",
            ),
        )

        first = coordinator.deliver(queued.outbox_id)
        assert first is not None and first.bound_snapshot_version == 1
        assert channel.calls[0][4] == {"card": queued.to_dict()["card_json"]}
        assert len(service.state.by_outbox_id[queued.outbox_id].effects) == 1

        channel.update_card = lambda agent_id, *, generation, message_id, card: (
            ChannelSendReceipt(
                request_id="update_2",
                success=True,
                app_id="cli_employee",
                generation=generation,
                connection_id="conn_employee",
                message_id=message_id,
            )
        )
        second = coordinator.deliver(queued.outbox_id)
        assert second is not None and second.bound_snapshot_version == 2
        assert len(service.state.by_outbox_id[queued.outbox_id].effects) == 2
    finally:
        service.close()
        writer.close()


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("completed", EmployeeCardState.COMPLETED),
        ("failed", EmployeeCardState.FAILED),
        ("canceled", EmployeeCardState.CANCELED),
        ("timeout", EmployeeCardState.FAILED),
        ("action_required", EmployeeCardState.ACTION_REQUIRED),
    ],
)
def test_gateway_lifecycle_renders_one_monotonic_card_for_all_terminal_states(
    tmp_path,
    status,
    expected,
) -> None:
    from src.autonomous.gateway.models import (
        GatewayExecutionResult,
        GatewayExecutionStatus,
    )
    from tests.autonomous.integration.test_employee_team_gateway import _binding

    service, writer, _anchor = _runtime(tmp_path)
    lifecycle = EmployeeOutboxLifecycle(service)
    binding = _binding()
    try:
        queued = lifecycle.queued(binding)
        running = lifecycle.running(binding)
        result = GatewayExecutionResult(
            status=GatewayExecutionStatus(status),
            output="done" if status == "completed" else "",
            safe_error_code="safe_failure" if status != "completed" else "",
        )
        terminal = lifecycle.terminal(binding, result)

        assert (queued.version, running.version, terminal.version) == (1, 2, 3)
        assert terminal.state is expected
        assert terminal.terminal_version == 3
        assert terminal.card_json["header"]["template"]
        assert lifecycle.terminal(binding, result) == terminal
    finally:
        service.close()
        writer.close()
