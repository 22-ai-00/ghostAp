"""Terminal ownership contract for explicitly targeted Employee tasks."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest

from src.autonomous.authorization import EmployeeAuthorizationScope
from src.autonomous.domain import EmployeeState, WorkerType
from src.autonomous.ingress.models import (
    EmployeeIngressMetadata,
    EmployeeIngressPayload,
)
from src.autonomous.ingress.projection import IngressProjectionState
from src.autonomous.ingress.router import (
    RouterAuthoritySnapshot,
    RouterLifecycleRecord,
    RouterResponseObligation,
)
from src.autonomous.ingress.service import EmployeeIngressService
from src.autonomous.ingress.targeted_task import (
    TARGETED_TASK_INPUT_KIND,
    targeted_group_task_digest,
)
from src.autonomous.journal.anchor import MemoryAnchor
from src.autonomous.journal.blob_store import (
    AesGcmEncryptionProvider,
    BlobStore,
)
from src.autonomous.journal.writer import JournalWriter
from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime
from src.autonomous.supervisor.employee_channels import ChannelProcessState

_TENANT_KEY = "tenant_1"
_AGENT_ID = "agt_alpha"
_BOT_PRINCIPAL_ID = "bot_alpha"
_APP_ID = "cli_alpha"
_CHANNEL_GENERATION = 3
_CONNECTION_ID = "conn_alpha"
_TARGET_OPEN_ID = "ou_bot_alpha"
_RAW_CHAT_ID = "oc_targeted_group"
_RAW_ROOT_ID = "om_targeted_root"
_DESCRIPTION = "finish the targeted audit"


@dataclass(frozen=True, slots=True)
class _AcceptedIngress:
    writer: JournalWriter
    ingress: EmployeeIngressService
    metadata: EmployeeIngressMetadata
    payload: EmployeeIngressPayload
    acceptance_id: str
    blob_root: Path


@contextmanager
def _accepted_targeted_ingress(
    tmp_path: Path,
    *,
    marker: str,
) -> Iterator[_AcceptedIngress]:
    raw_message_id = f"om_targeted_{marker}"
    envelope_id = "ing_" + hashlib.sha256(marker.encode("utf-8")).hexdigest()
    part = {
        "type": "message",
        "message_type": "text",
        "chat_type": "group",
        "content": {"text": f"@_user_1 /task {_DESCRIPTION}"},
        "sender_id": "ou_requester",
        "sender_union_id": "on_requester",
        "sender_id_type": "open_id",
        "sender_type": "user",
        "sender_tenant_key": _TENANT_KEY,
        "feishu_thread_id": f"omt_{marker}",
        "mentions": (
            {
                "key": "@_user_1",
                "open_id": _TARGET_OPEN_ID,
                "tenant_key": _TENANT_KEY,
            },
        ),
        "remote_chat_id": _RAW_CHAT_ID,
        "remote_message_id": raw_message_id,
        "remote_root_id": _RAW_ROOT_ID,
    }
    payload = EmployeeIngressPayload(
        schema_version=1,
        envelope_id=envelope_id,
        normalized_parts=(part,),
        attachment_descriptors=(),
    )
    metadata = EmployeeIngressMetadata(
        schema_version=1,
        envelope_id=envelope_id,
        tenant_key=_TENANT_KEY,
        agent_id=_AGENT_ID,
        bot_principal_id=_BOT_PRINCIPAL_ID,
        app_id=_APP_ID,
        channel_generation=_CHANNEL_GENERATION,
        connection_id=_CONNECTION_ID,
        event_id=f"evt_{marker}",
        message_id="om_"
        + hashlib.sha256(raw_message_id.encode("utf-8")).hexdigest(),
        event_type="im.message.receive_v1",
        action_identity="",
        chat_id="oc_" + hashlib.sha256(_RAW_CHAT_ID.encode("utf-8")).hexdigest(),
        thread_root_message_id="om_"
        + hashlib.sha256(_RAW_ROOT_ID.encode("utf-8")).hexdigest(),
        sender_principal_id="ou_requester",
        received_at="2026-08-12T00:00:00.000000Z",
        semantic_digest=payload.payload_sha256,
        payload_sha256=payload.payload_sha256,
        payload_size_bytes=payload.canonical_size_bytes,
        attachment_count=0,
        attachment_total_bytes=0,
    )
    writer = JournalWriter.open(
        tmp_path / f"journal-{marker}",
        anchor=MemoryAnchor(),
        hmac_key=b"t" * 32,
    )
    blob_root = tmp_path / f"blobs-{marker}"
    ingress = EmployeeIngressService(
        writer=writer,
        blob_store=BlobStore(
            blob_root,
            AesGcmEncryptionProvider(lambda _key_ref: b"i" * 32),
        ),
        ingress_state=IngressProjectionState(),
        active_key_id="key-active",
    )
    acceptance_id = ingress.accept(
        metadata,
        payload,
        request_id=f"req_{marker}",
    ).acceptance.acceptance_id
    try:
        yield _AcceptedIngress(
            writer=writer,
            ingress=ingress,
            metadata=metadata,
            payload=payload,
            acceptance_id=acceptance_id,
            blob_root=blob_root,
        )
    finally:
        ingress.close()
        writer.close()


def _targeted_authority() -> RouterAuthoritySnapshot:
    return RouterAuthoritySnapshot(
        tenant_key=_TENANT_KEY,
        agent_id=_AGENT_ID,
        bot_principal_id=_BOT_PRINCIPAL_ID,
        app_id=_APP_ID,
        channel_generation=_CHANNEL_GENERATION,
        connection_id=_CONNECTION_ID,
        authorization_scope=EmployeeAuthorizationScope.MANAGED_GROUP,
        team_id=_RAW_CHAT_ID,
        requester_principal_id="ou_owner",
        projection_sequence=7,
        projection_hash="a" * 64,
        employee_version=1,
        tool="codex",
        model="",
        effort="high",
        effective_input_kind=TARGETED_TASK_INPUT_KIND,
        effective_input_digest=targeted_group_task_digest(_DESCRIPTION),
        target_bot_open_id_digest=hashlib.sha256(
            _TARGET_OPEN_ID.encode("utf-8")
        ).hexdigest(),
    )


def _terminal_record(
    accepted: _AcceptedIngress,
    *,
    reason_code: str,
    queued_sequence: int,
) -> RouterLifecycleRecord:
    ingress_record = accepted.ingress.record_snapshot(accepted.acceptance_id)
    assert ingress_record is not None
    acceptance = ingress_record.acceptance
    authority = _targeted_authority()
    return RouterLifecycleRecord(
        aggregate_id=ingress_record.aggregate_id,
        acceptance_id=accepted.acceptance_id,
        envelope_id=accepted.metadata.envelope_id,
        tenant_key=_TENANT_KEY,
        agent_id=_AGENT_ID,
        bot_principal_id=_BOT_PRINCIPAL_ID,
        app_id=_APP_ID,
        channel_generation=_CHANNEL_GENERATION,
        connection_id=_CONNECTION_ID,
        team_id=_RAW_CHAT_ID,
        message_id=accepted.metadata.message_id,
        event_type=accepted.metadata.event_type,
        requester_principal_id="ou_owner",
        state="terminal",
        accepted_sequence=acceptance.journal_sequence,
        authority=authority,
        indexed_chat_id=accepted.metadata.chat_id,
        indexed_thread_root_message_id=accepted.metadata.thread_root_message_id,
        response_obligation=(
            RouterResponseObligation.create(
                authority=authority,
                acceptance_id=accepted.acceptance_id,
                ingress_aggregate_id=ingress_record.aggregate_id,
                envelope_id=accepted.metadata.envelope_id,
                event_type=accepted.metadata.event_type,
                chat_id=_RAW_CHAT_ID,
                message_id=str(
                    accepted.payload.normalized_parts[0]["remote_message_id"]
                ),
                thread_root_message_id=_RAW_ROOT_ID,
            )
            if queued_sequence > 0
            else None
        ),
        queued_sequence=queued_sequence,
        reason_code=reason_code,
    )


def _runtime_for_handoff(
    record: RouterLifecycleRecord,
    ingress: EmployeeIngressService | None = None,
) -> EmployeeDepartmentRuntime:
    runtime = object.__new__(EmployeeDepartmentRuntime)
    runtime._closing = False
    runtime._ingress = (
        ingress
        if ingress is not None
        else SimpleNamespace(record_snapshot=lambda _acceptance_id: None)
    )
    runtime._router = SimpleNamespace(
        record_snapshot=lambda acceptance_id: (
            record if acceptance_id == record.acceptance_id else None
        )
    )
    return runtime


def _configure_transport(
    runtime: EmployeeDepartmentRuntime,
    *,
    generation: int,
    connection_id: str,
    bot_open_id: str,
) -> None:
    employee = SimpleNamespace(
        agent_id=_AGENT_ID,
        tenant_key=_TENANT_KEY,
        state=EmployeeState.ACTIVE,
        worker_type=WorkerType.VISIBLE,
        bot_principal_id=_BOT_PRINCIPAL_ID,
    )
    principal = SimpleNamespace(
        tenant_key=_TENANT_KEY,
        agent_id=_AGENT_ID,
        app_id=_APP_ID,
        credential_ref="vault://employee-alpha",
    )
    runtime._service = SimpleNamespace(
        synchronize_projection=lambda: SimpleNamespace(
            employees={_AGENT_ID: employee},
            bot_principals={_BOT_PRINCIPAL_ID: principal},
        )
    )
    runtime._channels = SimpleNamespace(
        status=lambda _agent_id: SimpleNamespace(
            state=ChannelProcessState.READY,
            tenant_key=_TENANT_KEY,
            agent_id=_AGENT_ID,
            bot_principal_id=_BOT_PRINCIPAL_ID,
            app_id=_APP_ID,
            generation=generation,
            identity={"app_id": _APP_ID, "open_id": bot_open_id},
            ready_metadata={"connection_id": connection_id},
        )
    )


class _ObservedIngress:
    def __init__(self, delegate: EmployeeIngressService, events: list[object]) -> None:
        self._delegate = delegate
        self._events = events

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)

    def record_disposition(
        self,
        acceptance_id: str,
        *,
        state: str,
        reason_code: str,
    ) -> object:
        self._events.append(("disposition", state, reason_code))
        return self._delegate.record_disposition(
            acceptance_id,
            state=state,
            reason_code=reason_code,
        )


class _RecordingLifecycle:
    def __init__(
        self,
        ingress: EmployeeIngressService,
        acceptance_id: str,
        events: list[object],
    ) -> None:
        self._ingress = ingress
        self._acceptance_id = acceptance_id
        self._events = events

    def task_failure_response(self, **kwargs: object) -> object:
        current = self._ingress.record_snapshot(self._acceptance_id)
        assert current is not None and current.disposition is None
        self._events.append(("failure_response", kwargs))
        return SimpleNamespace()


class _FailOnceLifecycle(_RecordingLifecycle):
    def __init__(
        self,
        ingress: EmployeeIngressService,
        acceptance_id: str,
        events: list[object],
    ) -> None:
        super().__init__(ingress, acceptance_id, events)
        self._failed = False

    def task_failure_response(self, **kwargs: object) -> object:
        current = self._ingress.record_snapshot(self._acceptance_id)
        assert current is not None and current.disposition is None
        if not self._failed:
            self._failed = True
            self._events.append(("failure_response_failed", kwargs))
            raise RuntimeError("injected task failure response write failure")
        return super().task_failure_response(**kwargs)


def _runtime_for_reconciliation(
    accepted: _AcceptedIngress,
    record: RouterLifecycleRecord,
    events: list[object],
) -> EmployeeDepartmentRuntime:
    runtime = object.__new__(EmployeeDepartmentRuntime)
    runtime._ingress = _ObservedIngress(accepted.ingress, events)
    runtime._router = SimpleNamespace(
        rebuild_projection=lambda: None,
        state=SimpleNamespace(by_acceptance_id={accepted.acceptance_id: record}),
    )
    runtime._outbox_lifecycle = _RecordingLifecycle(
        accepted.ingress,
        accepted.acceptance_id,
        events,
    )
    return runtime


def test_post_queue_targeted_terminal_retains_employee_handoff_ownership(
    tmp_path: Path,
) -> None:
    with _accepted_targeted_ingress(tmp_path, marker="postqueue_handoff") as accepted:
        record = _terminal_record(
            accepted,
            reason_code="authority_stale",
            queued_sequence=23,
        )
        runtime = _runtime_for_handoff(record, accepted.ingress)

        result = runtime._employee_handoff_projection_result(  # noqa: SLF001
            acceptance_id=accepted.acceptance_id,
            ingress=runtime._ingress,
            router=runtime._router,
            channel_generation=_CHANNEL_GENERATION,
            connection_id=_CONNECTION_ID,
        )
        assert result.value == "owned"


def test_prequeue_targeted_terminal_does_not_claim_employee_handoff(
    tmp_path: Path,
) -> None:
    with _accepted_targeted_ingress(tmp_path, marker="prequeue_handoff") as accepted:
        record = _terminal_record(
            accepted,
            reason_code="authority_denied",
            queued_sequence=0,
        )
        runtime = _runtime_for_handoff(record, accepted.ingress)

        result = runtime._employee_handoff_projection_result(  # noqa: SLF001
            acceptance_id=accepted.acceptance_id,
            ingress=runtime._ingress,
            router=runtime._router,
            channel_generation=_CHANNEL_GENERATION,
            connection_id=_CONNECTION_ID,
        )
        assert result.value == "durably_denied"


def test_post_queue_terminal_anchors_task_failure_before_ingress_disposition_despite_channel_drift(
    tmp_path: Path,
) -> None:
    with _accepted_targeted_ingress(tmp_path, marker="drifted_terminal") as accepted:
        record = _terminal_record(
            accepted,
            reason_code="authority_stale",
            queued_sequence=29,
        )
        events: list[object] = []
        runtime = _runtime_for_reconciliation(accepted, record, events)
        _configure_transport(
            runtime,
            generation=_CHANNEL_GENERATION + 1,
            connection_id="conn_replacement",
            bot_open_id="ou_bot_replacement",
        )
        stored = tuple(
            path.read_bytes()
            for path in accepted.blob_root.rglob("*")
            if path.is_file()
        )
        assert stored and all(_DESCRIPTION.encode("utf-8") not in raw for raw in stored)
        assert runtime._employee_ingress_transport_is_current(accepted.metadata) is False  # noqa: SLF001

        assert runtime._reconcile_terminal_ingress() == 1  # noqa: SLF001

        assert [event[0] for event in events] == [
            "failure_response",
            "disposition",
        ]
        response = events[0][1]
        assert response == {
            "tenant_key": _TENANT_KEY,
            "agent_id": _AGENT_ID,
            "chat_id": _RAW_CHAT_ID,
            "thread_root_message_id": _RAW_ROOT_ID,
            "command_acceptance_id": accepted.acceptance_id,
        }
        disposition = accepted.ingress.record_snapshot(accepted.acceptance_id)
        assert disposition is not None
        assert disposition.disposition is not None
        assert disposition.disposition.reason_code == "authority_stale"


def test_post_queue_permanent_blob_loss_uses_frozen_response_obligation(
    tmp_path: Path,
) -> None:
    with _accepted_targeted_ingress(tmp_path, marker="lost_blob_terminal") as accepted:
        record = _terminal_record(
            accepted,
            reason_code="context_unavailable",
            queued_sequence=34,
        )
        events: list[object] = []
        runtime = _runtime_for_reconciliation(accepted, record, events)
        for path in accepted.blob_root.rglob("*"):
            if path.is_file():
                path.unlink()

        assert runtime._reconcile_terminal_ingress() == 1  # noqa: SLF001

        assert [event[0] for event in events] == [
            "failure_response",
            "disposition",
        ]
        assert events[0][1] == {
            "tenant_key": _TENANT_KEY,
            "agent_id": _AGENT_ID,
            "chat_id": _RAW_CHAT_ID,
            "thread_root_message_id": _RAW_ROOT_ID,
            "command_acceptance_id": accepted.acceptance_id,
        }
        disposition = accepted.ingress.record_snapshot(accepted.acceptance_id)
        assert disposition is not None and disposition.disposition is not None
        assert disposition.disposition.reason_code == "context_unavailable"


def test_task_failure_response_failure_withholds_disposition_until_retry_succeeds(
    tmp_path: Path,
) -> None:
    with _accepted_targeted_ingress(tmp_path, marker="failure_retry") as accepted:
        record = _terminal_record(
            accepted,
            reason_code="authority_stale",
            queued_sequence=30,
        )
        events: list[object] = []
        runtime = _runtime_for_reconciliation(accepted, record, events)
        runtime._outbox_lifecycle = _FailOnceLifecycle(  # noqa: SLF001
            accepted.ingress,
            accepted.acceptance_id,
            events,
        )

        with pytest.raises(
            RuntimeError,
            match="injected task failure response write failure",
        ):
            runtime._reconcile_terminal_ingress()  # noqa: SLF001

        after_failure = accepted.ingress.record_snapshot(accepted.acceptance_id)
        assert after_failure is not None and after_failure.disposition is None
        assert [event[0] for event in events] == ["failure_response_failed"]

        assert runtime._reconcile_terminal_ingress() == 1  # noqa: SLF001

        after_retry = accepted.ingress.record_snapshot(accepted.acceptance_id)
        assert after_retry is not None and after_retry.disposition is not None
        assert after_retry.disposition.reason_code == "authority_stale"
        assert [event[0] for event in events] == [
            "failure_response_failed",
            "failure_response",
            "disposition",
        ]
        assert sum(event[0] == "disposition" for event in events) == 1


@pytest.mark.parametrize(
    "tampered_field",
    ("target_bot_open_id_digest", "effective_input_digest"),
)
def test_tampered_frozen_target_binding_neither_responds_nor_dispositions(
    tmp_path: Path,
    tampered_field: str,
) -> None:
    with _accepted_targeted_ingress(
        tmp_path,
        marker=f"tampered_{tampered_field}",
    ) as accepted:
        record = _terminal_record(
            accepted,
            reason_code="authority_stale",
            queued_sequence=32,
        )
        assert record.authority is not None
        record = replace(
            record,
            authority=replace(record.authority, **{tampered_field: "f" * 64}),
        )
        events: list[object] = []
        runtime = _runtime_for_reconciliation(accepted, record, events)

        assert runtime._reconcile_terminal_ingress() == 0  # noqa: SLF001

        ingress_record = accepted.ingress.record_snapshot(accepted.acceptance_id)
        assert ingress_record is not None and ingress_record.disposition is None
        assert events == []


@pytest.mark.parametrize(
    "reason_code",
    ("completed", "failed", "canceled", "timeout", "action_required"),
)
def test_execution_terminal_does_not_create_a_control_failure_response(
    tmp_path: Path,
    reason_code: str,
) -> None:
    with _accepted_targeted_ingress(
        tmp_path,
        marker=f"{reason_code}_terminal",
    ) as accepted:
        record = _terminal_record(
            accepted,
            reason_code=reason_code,
            queued_sequence=31,
        )
        events: list[object] = []
        runtime = _runtime_for_reconciliation(accepted, record, events)
        _configure_transport(
            runtime,
            generation=_CHANNEL_GENERATION,
            connection_id=_CONNECTION_ID,
            bot_open_id=_TARGET_OPEN_ID,
        )
        assert runtime._employee_ingress_transport_is_current(accepted.metadata) is True  # noqa: SLF001

        assert runtime._reconcile_terminal_ingress() == 1  # noqa: SLF001

        assert events == [("disposition", "terminal", reason_code)]


def test_mutated_targeted_authority_execution_terminal_withholds_disposition(
    tmp_path: Path,
) -> None:
    with _accepted_targeted_ingress(
        tmp_path,
        marker="mutated_execution_terminal",
    ) as accepted:
        record = _terminal_record(
            accepted,
            reason_code="completed",
            queued_sequence=33,
        )
        assert record.authority is not None
        record = replace(
            record,
            authority=replace(
                record.authority,
                authorization_scope=EmployeeAuthorizationScope.OWNER_P2P,
                effective_input_kind="",
                effective_input_digest="",
                target_bot_open_id_digest="",
            ),
        )
        events: list[object] = []
        runtime = _runtime_for_reconciliation(accepted, record, events)

        assert runtime._reconcile_terminal_ingress() == 0  # noqa: SLF001

        persisted = accepted.ingress.record_snapshot(accepted.acceptance_id)
        assert persisted is not None and persisted.disposition is None
        assert events == []
