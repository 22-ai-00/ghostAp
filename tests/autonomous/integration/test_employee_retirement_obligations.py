from __future__ import annotations

import hashlib
import threading
from dataclasses import replace
from types import SimpleNamespace

from src.autonomous.gateway.models import (
    GatewayExecutionResult,
    GatewayExecutionStatus,
)
from src.autonomous.ingress.models import EmployeeIngressPayload
from src.autonomous.journal.anchor import MemoryAnchor
from src.autonomous.journal.blob_store import AesGcmEncryptionProvider, BlobStore
from src.autonomous.journal.writer import JournalWriter
from src.autonomous.outbox.delivery import (
    EmployeeDeliveryAuthority,
    EmployeeOutboxDeliveryCoordinator,
)
from src.autonomous.outbox.lifecycle import EmployeeOutboxLifecycle
from src.autonomous.outbox.projection import OutboxProjectionState
from src.autonomous.outbox.service import EmployeeOutboxService
from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime
from src.autonomous.provisioning.fire_effects import ChannelStopEffect
from src.autonomous.provisioning.fire_service import EmployeeFireService
from src.autonomous.provisioning.fire_state import FireEffectState, FirePhase
from src.autonomous.supervisor.channel_models import ChannelProcessState
from src.autonomous.supervisor.employee_channels import (
    ChannelSendReceipt,
    EmployeeChannelSupervisor,
)
from tests.autonomous.integration.test_employee_channel_process import (
    _worker,
)
from tests.autonomous.integration.test_employee_fire_queued_handoff import (
    _fire_request,
    _queued_fire_harness,
)
from tests.autonomous.integration.test_employee_team_gateway import (
    _binding,
    _real_coordinator_harness,
)


def _accept_unrouted_targeted_message(harness: object) -> str:
    first_id = harness.acceptance_ids[0]
    first_ingress = harness.ingress.record_snapshot(first_id)
    assert first_ingress is not None
    part = dict(harness.payload.normalized_parts[0])
    raw_message_id = "om_retirement_accepted"
    part["remote_message_id"] = raw_message_id
    payload = EmployeeIngressPayload(
        schema_version=1,
        envelope_id="ing_" + "9" * 64,
        normalized_parts=(part,),
        attachment_descriptors=(),
    )
    metadata = replace(
        first_ingress.metadata,
        envelope_id=payload.envelope_id,
        event_id="evt_retirement_accepted",
        message_id="om_" + hashlib.sha256(raw_message_id.encode()).hexdigest(),
        semantic_digest=payload.payload_sha256,
        payload_sha256=payload.payload_sha256,
        payload_size_bytes=payload.canonical_size_bytes,
    )
    return harness.ingress.accept(
        metadata,
        payload,
        request_id="req_retirement_accepted",
    ).acceptance.acceptance_id


def _accept_owner_status_message(subject: object) -> str:
    first_id = subject.harness.acceptance_ids[0]
    first_ingress = subject.harness.ingress.record_snapshot(first_id)
    assert first_ingress is not None
    raw_chat_id = "oc_retirement_status"
    raw_message_id = "om_retirement_status"
    raw_root_id = "om_retirement_status_root"
    payload = EmployeeIngressPayload(
        schema_version=1,
        envelope_id="ing_" + "8" * 64,
        normalized_parts=(
            {
                "type": "message",
                "message_type": "text",
                "chat_type": "p2p",
                "content": {"text": "/status"},
                "sender_id": "ou_employee_app_owner",
                "sender_union_id": "on_owner",
                "sender_id_type": "open_id",
                "sender_type": "user",
                "sender_tenant_key": "tenant_1",
                "feishu_thread_id": "omt_retirement_status",
                "remote_chat_id": raw_chat_id,
                "remote_message_id": raw_message_id,
                "remote_root_id": raw_root_id,
            },
        ),
        attachment_descriptors=(),
    )
    metadata = replace(
        first_ingress.metadata,
        envelope_id=payload.envelope_id,
        event_id="evt_retirement_status",
        chat_id="oc_" + hashlib.sha256(raw_chat_id.encode()).hexdigest(),
        message_id="om_" + hashlib.sha256(raw_message_id.encode()).hexdigest(),
        thread_root_message_id=(
            "om_" + hashlib.sha256(raw_root_id.encode()).hexdigest()
        ),
        semantic_digest=payload.payload_sha256,
        payload_sha256=payload.payload_sha256,
        payload_size_bytes=payload.canonical_size_bytes,
    )
    return subject.harness.ingress.accept(
        metadata,
        payload,
        request_id="req_retirement_status",
    ).acceptance.acceptance_id


def _accept_owner_status_control(harness: object) -> str:
    first_id = harness.acceptance_ids[0]
    first_ingress = harness.ingress.record_snapshot(first_id)
    assert first_ingress is not None
    raw_chat_id = "oc_retirement_status_race"
    raw_message_id = "om_retirement_status_race"
    raw_root_id = "om_retirement_status_root"
    payload = EmployeeIngressPayload(
        schema_version=1,
        envelope_id="ing_" + "8" * 64,
        normalized_parts=(
            {
                "type": "message",
                "message_type": "text",
                "chat_type": "p2p",
                "content": {"text": "/status"},
                "sender_id": "ou_employee_app_owner",
                "sender_union_id": "on_owner",
                "sender_id_type": "open_id",
                "sender_type": "user",
                "sender_tenant_key": "tenant_1",
                "feishu_thread_id": "omt_retirement_status_race",
                "remote_chat_id": raw_chat_id,
                "remote_message_id": raw_message_id,
                "remote_root_id": raw_root_id,
            },
        ),
        attachment_descriptors=(),
    )
    metadata = replace(
        first_ingress.metadata,
        envelope_id=payload.envelope_id,
        event_id="evt_retirement_status_race",
        chat_id="oc_" + hashlib.sha256(raw_chat_id.encode()).hexdigest(),
        message_id="om_" + hashlib.sha256(raw_message_id.encode()).hexdigest(),
        thread_root_message_id=(
            "om_" + hashlib.sha256(raw_root_id.encode()).hexdigest()
        ),
        sender_principal_id="ou_employee_app_owner",
        semantic_digest=payload.payload_sha256,
        payload_sha256=payload.payload_sha256,
        payload_size_bytes=payload.canonical_size_bytes,
    )
    return harness.ingress.accept(
        metadata,
        payload,
        request_id="req_retirement_status_race",
    ).acceptance.acceptance_id


def test_retirement_ready_waits_for_cutoff_accepted_ingress(tmp_path) -> None:
    harness = _real_coordinator_harness(
        tmp_path,
        targeted_group_task=True,
    )
    try:
        queued_id = harness.acceptance_ids[0]
        accepted_id = _accept_unrouted_targeted_message(harness)
        harness.router.rebuild_projection()
        accepted = harness.router.record_snapshot(accepted_id)
        assert accepted is not None and accepted.state == "accepted"

        harness.coordinator._attempt_lifecycle = SimpleNamespace(  # noqa: SLF001
            terminal_response_delivered=lambda **_kwargs: True,
        )
        assert (
            harness.coordinator.begin_employee_retirement(
                tenant_key="tenant_1",
                agent_id="agt_alpha",
            )
            == 1
        )
        harness.ingress.record_disposition(
            queued_id,
            state="terminal",
            reason_code="context_unavailable",
        )

        assert harness.ingress.record_snapshot(accepted_id).disposition is None
        assert (
            harness.coordinator.employee_retirement_ready(
                tenant_key="tenant_1",
                agent_id="agt_alpha",
            )
            is False
        )
    finally:
        harness.close()


def test_real_status_control_retires_only_after_delivery_only_send(
    tmp_path,
    monkeypatch,
) -> None:
    """Exercise accepted `/status` through the real control and fire paths."""

    with _queued_fire_harness(tmp_path) as subject:
        subject.harness.coordinator.begin_employee_retirement(
            tenant_key="tenant_1",
            agent_id="agt_alpha",
        )
        assert subject.runtime._reconcile_terminal_ingress() == 1  # noqa: SLF001
        assert subject.runtime._drain_employee_outbox_once() is True  # noqa: SLF001

        status_id = _accept_owner_status_message(subject)
        subject.harness.router.rebuild_projection()
        routed = subject.harness.router.record_snapshot(status_id)
        assert routed is not None
        assert routed.state == "accepted"
        assert routed.queued_sequence == 0

        hire = next(
            state
            for state in subject.harness.hire.list_states()
            if state.agent_id == "agt_alpha"
        )
        subject.harness.hire.list_states = lambda: (  # type: ignore[method-assign]
            replace(
                hire,
                requester_principal_id="ou_owner",
                requester_union_id="on_owner",
                channel_identity_app_id="cli_alpha",
            ),
        )
        subject.runtime._context_acl = SimpleNamespace(  # noqa: SLF001
            is_authorized=lambda _request: True,
        )
        subject.runtime._employee_status_summary = lambda **_kwargs: (  # type: ignore[method-assign]  # noqa: SLF001
            "员工状态：空闲"
        )
        subject.runtime._request_employee_outbox_delivery = lambda: None  # type: ignore[method-assign]  # noqa: SLF001

        assert subject.runtime._handle_control_ingress(status_id) is True  # noqa: SLF001
        ingress_record = subject.harness.ingress.record_snapshot(status_id)
        assert ingress_record is not None
        assert ingress_record.disposition is not None
        assert ingress_record.disposition.reason_code == "status_completed"
        pending = subject.outbox.list_pending_delivery_records()
        assert len(pending) == 1
        assert pending[0].attempt_id == f"control_{status_id}"

        waiting = subject.service.start_fire(
            _fire_request(message_id="om_fire_pending_status")
        )
        assert waiting.phase is FirePhase.RETIRING
        assert waiting.effect_state("execution_quiesce") is FireEffectState.EXECUTING
        assert subject.events == ["failure_response_delivered"]

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
        runtime = EmployeeDepartmentRuntime(runtime_enabled=True)
        runtime._service = subject.harness.hire  # noqa: SLF001
        runtime._channels = restarted_channels  # noqa: SLF001
        runtime._fire = restarted_fire  # noqa: SLF001
        runtime._ingress = subject.harness.ingress  # noqa: SLF001
        runtime._router = subject.harness.router  # noqa: SLF001
        runtime._dispatch = subject.harness.coordinator  # noqa: SLF001
        runtime._outbox = subject.outbox  # noqa: SLF001
        runtime._outbox_lifecycle = subject.lifecycle  # noqa: SLF001
        runtime._employee_outbox_delivery_lock = threading.RLock()  # noqa: SLF001
        sent: list[str] = []
        original_send = restarted_channels.send

        def send(agent_id: str, **kwargs: object) -> ChannelSendReceipt:
            receipt = original_send(agent_id, **kwargs)
            options = kwargs["options"]
            assert isinstance(options, dict)
            sent.append(str(options["uuid"]))
            return receipt

        monkeypatch.setattr(restarted_channels, "send", send)
        runtime._outbox_delivery = EmployeeOutboxDeliveryCoordinator(  # noqa: SLF001
            outbox=subject.outbox,
            channels=restarted_channels,
            authority_resolver=runtime._resolve_outbox_delivery_authority,
        )
        try:
            assert restarted_fire.recover()[0].phase is FirePhase.RETIRING
            assert runtime._recover_retirement_delivery_channels() == (  # noqa: SLF001
                "agt_alpha",
            )
            ready = restarted_channels.status("agt_alpha")
            assert ready is not None
            assert ready.state is ChannelProcessState.READY
            assert ready.delivery_only is True

            assert runtime._drain_employee_outbox_once() is True  # noqa: SLF001
            assert len(sent) == 1
            delivered = subject.outbox.list_pending_delivery_records()
            assert delivered == ()

            progressed = restarted_fire.reconcile_draining()
            assert len(progressed) == 1
            assert progressed[0].phase is FirePhase.ARCHIVED
            stopped = restarted_channels.status("agt_alpha")
            assert stopped is not None
            assert stopped.state is ChannelProcessState.STOPPED
        finally:
            restarted_channels.close()


def test_retirement_ready_rechecks_head_after_response_fence(tmp_path) -> None:
    harness = _real_coordinator_harness(
        tmp_path,
        targeted_group_task=True,
    )
    try:
        queued_id = harness.acceptance_ids[0]
        assert (
            harness.coordinator.begin_employee_retirement(
                tenant_key="tenant_1",
                agent_id="agt_alpha",
            )
            == 1
        )
        harness.ingress.record_disposition(
            queued_id,
            state="terminal",
            reason_code="context_unavailable",
        )
        injected: list[str] = []

        class _RacingLifecycle:
            def terminal_response_delivered(self, **_kwargs: object) -> bool:
                return True

            def employee_responses_delivered(self, **_kwargs: object) -> bool:
                if not injected:
                    injected.append(_accept_owner_status_control(harness))
                return True

        harness.coordinator._attempt_lifecycle = _RacingLifecycle()  # noqa: SLF001

        assert (
            harness.coordinator.employee_retirement_ready(
                tenant_key="tenant_1",
                agent_id="agt_alpha",
            )
            is False
        )
        assert len(injected) == 1
        raced = harness.ingress.record_snapshot(injected[0])
        assert raced is not None
        assert raced.disposition is None
    finally:
        harness.close()


def test_employee_response_fence_waits_for_latest_control_delivery(
    tmp_path,
) -> None:
    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=MemoryAnchor(),
        hmac_key=b"retirement-response-fence-key!!!",
    )
    outbox = EmployeeOutboxService(
        writer=writer,
        blob_store=BlobStore(
            tmp_path / "outbox-blobs",
            AesGcmEncryptionProvider(lambda _key_ref: b"o" * 32),
        ),
        outbox_state=OutboxProjectionState(),
        active_key_id="outbox-key",
    )
    lifecycle = EmployeeOutboxLifecycle(outbox)
    attempt_id = "control_acc_status_cutoff"

    class _Channels:
        def send(self, _agent_id: str, **_kwargs: object) -> ChannelSendReceipt:
            return ChannelSendReceipt(
                request_id="send_retirement_control",
                success=True,
                app_id="cli_alpha",
                generation=3,
                connection_id="conn_alpha",
                message_id="om_retirement_control",
            )

        def update_card(self, _agent_id: str, **_kwargs: object) -> object:
            raise AssertionError("terminal control must be created once")

    delivery = EmployeeOutboxDeliveryCoordinator(
        outbox=outbox,
        channels=_Channels(),
        authority_resolver=lambda _record: EmployeeDeliveryAuthority(
            app_id="cli_alpha",
            generation=3,
            connection_id="conn_alpha",
        ),
    )
    try:
        lifecycle.status_response(
            tenant_key="tenant_1",
            agent_id="agt_alpha",
            chat_id="oc_owner",
            thread_root_message_id="om_status_cutoff",
            command_acceptance_id="acc_status_cutoff",
            summary="员工状态：空闲",
            succeeded=True,
        )
        assert (
            lifecycle.employee_responses_delivered(
                tenant_key="tenant_1",
                agent_id="agt_alpha",
                attempt_ids=frozenset({attempt_id}),
            )
            is False
        )

        delivery.deliver_pending()

        assert (
            lifecycle.employee_responses_delivered(
                tenant_key="tenant_1",
                agent_id="agt_alpha",
                attempt_ids=frozenset({attempt_id}),
            )
            is True
        )
    finally:
        outbox.close()
        writer.close()


def test_employee_response_fence_rejects_binding_behind_latest_snapshot(
    tmp_path,
) -> None:
    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=MemoryAnchor(),
        hmac_key=b"retirement-latest-snapshot-key!!!",
    )
    outbox = EmployeeOutboxService(
        writer=writer,
        blob_store=BlobStore(
            tmp_path / "outbox-blobs",
            AesGcmEncryptionProvider(lambda _key_ref: b"o" * 32),
        ),
        outbox_state=OutboxProjectionState(),
        active_key_id="outbox-key",
    )
    lifecycle = EmployeeOutboxLifecycle(outbox)
    binding = _binding()

    class _Channels:
        def send(self, _agent_id: str, **_kwargs: object) -> ChannelSendReceipt:
            return ChannelSendReceipt(
                request_id="send_retirement_attempt",
                success=True,
                app_id="cli_alpha",
                generation=3,
                connection_id="conn_alpha",
                message_id="om_retirement_attempt",
            )

        def update_card(
            self,
            _agent_id: str,
            **_kwargs: object,
        ) -> ChannelSendReceipt:
            return ChannelSendReceipt(
                request_id="update_retirement_attempt",
                success=True,
                app_id="cli_alpha",
                generation=3,
                connection_id="conn_alpha",
                message_id="om_retirement_attempt",
            )

    delivery = EmployeeOutboxDeliveryCoordinator(
        outbox=outbox,
        channels=_Channels(),
        authority_resolver=lambda _record: EmployeeDeliveryAuthority(
            app_id="cli_alpha",
            generation=3,
            connection_id="conn_alpha",
        ),
    )
    attempt_ids = frozenset({binding.attempt_id})
    try:
        lifecycle.queued(binding)
        delivery.deliver_pending()
        lifecycle.running(binding)
        lifecycle.terminal(
            binding,
            GatewayExecutionResult(
                status=GatewayExecutionStatus.COMPLETED,
                output="done",
            ),
        )

        assert (
            lifecycle.employee_responses_delivered(
                tenant_key=binding.tenant_key,
                agent_id=binding.agent_id,
                attempt_ids=attempt_ids,
            )
            is False
        )

        delivery.deliver_pending()

        assert (
            lifecycle.employee_responses_delivered(
                tenant_key=binding.tenant_key,
                agent_id=binding.agent_id,
                attempt_ids=attempt_ids,
            )
            is True
        )
    finally:
        outbox.close()
        writer.close()
