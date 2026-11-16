"""Targeted tests for B075 typed Outbox errors and B068 incremental tail sync."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.autonomous.context.runtime import RuntimeRequesterChatAcl
from src.autonomous.domain import BotPrincipal, EmployeeDefinition, EmployeeState, WorkerType
from src.autonomous.ingress.models import EmployeeIngressMetadata, EmployeeIngressPayload
from src.autonomous.ingress.projection import IngressProjectionState
from src.autonomous.ingress.router import (
    DurableEmployeeIngressRouter,
    RouterQueueLimits,
    RouterWriteDisabledError,
)
from src.autonomous.ingress.service import EmployeeIngressService, IngressWriteDisabledError
from src.autonomous.journal.anchor import MemoryAnchor
from src.autonomous.journal.blob_store import AesGcmEncryptionProvider, BlobStore
from src.autonomous.journal.frame import GENESIS_HASH, JournalEvent
from src.autonomous.journal.projections import ProjectionState
from src.autonomous.journal.writer import JournalWriter
from src.autonomous.outbox import (
    EmployeeCardState,
    EmployeeOutboxLifecycle,
    EmployeeOutboxService,
    EmployeeOutboxSnapshot,
    OutboxNotFoundError,
    OutboxProjectionState,
    OutboxWriteDisabledError,
    employee_outbox_id,
)
from src.autonomous.outbox.service import OutboxServiceError
from src.autonomous.supervisor.employee_channels import ChannelProcessState, ChannelProcessStatus
from src.autonomous.workforce.registry import ProjectedAgentRegistry


class _Channels:
    def __init__(self) -> None:
        self.status_obj = ChannelProcessStatus(
            agent_id="agt_alpha",
            app_id="cli_alpha",
            generation=3,
            pid=101,
            state=ChannelProcessState.READY,
            tenant_key="tenant_1",
            bot_principal_id="bot_alpha",
            identity={"app_id": "cli_alpha", "open_id": "ou_bot_alpha"},
            ready_metadata={"connection_id": "conn_alpha"},
        )

    def status(self, agent_id: str):
        return self.status_obj if agent_id == "agt_alpha" else None


class _HealthyMembership:
    def is_degraded(self, _agent_id: str, _team_id: str) -> bool:
        return False


def _outbox_snapshot(attempt_id: str = "attempt-1") -> EmployeeOutboxSnapshot:
    return EmployeeOutboxSnapshot(
        schema_version=1,
        outbox_id=employee_outbox_id("tenant_1", "agt_alpha", attempt_id),
        tenant_key="tenant_1",
        agent_id="agt_alpha",
        attempt_id=attempt_id,
        chat_id="oc_team",
        thread_root_message_id="om_root",
        version=1,
        state=EmployeeCardState.QUEUED,
        title="Task status",
        summary="Queued",
        progress_percent=0,
        card_json={"schema": "2.0", "body": {"elements": []}},
        created_at="2026-07-14T00:00:00Z",
        terminal_version=0,
    )


def _remote_index(prefix: str, raw_value: str) -> str:
    return prefix + hashlib.sha256(raw_value.encode("utf-8")).hexdigest()


def _ingress_item(index: int) -> tuple[EmployeeIngressMetadata, EmployeeIngressPayload]:
    raw_message_id = f"om_router_alpha_{index}"
    part = {
        "type": "message",
        "message_type": "text",
        "chat_type": "group",
        "content": {"text": f"task {index}"},
        "sender_id": "ou_requester",
        "sender_union_id": "",
        "sender_id_type": "open_id",
        "sender_type": "user",
        "sender_tenant_key": "tenant_1",
        "feishu_thread_id": f"omt_{index}",
        "remote_chat_id": "oc_team",
        "remote_message_id": raw_message_id,
        "remote_root_id": "om_root",
    }
    digest = hashlib.sha256(f"inc-{index}".encode()).hexdigest()
    payload = EmployeeIngressPayload(
        schema_version=1,
        envelope_id="ing_" + digest,
        normalized_parts=(part,),
        attachment_descriptors=(),
    )
    suffix = hashlib.sha256(f"agt_alpha:{index}".encode()).hexdigest()[:24]
    metadata = EmployeeIngressMetadata(
        schema_version=1,
        envelope_id=payload.envelope_id,
        tenant_key="tenant_1",
        agent_id="agt_alpha",
        bot_principal_id="bot_alpha",
        app_id="cli_alpha",
        channel_generation=3,
        connection_id="conn_alpha",
        event_id=f"evt_{suffix}",
        message_id=_remote_index("om_", raw_message_id),
        event_type="im.message.receive_v1",
        action_identity="",
        chat_id=_remote_index("oc_", "oc_team"),
        thread_root_message_id=_remote_index("om_", "om_root"),
        sender_principal_id="ou_requester",
        received_at="2026-07-13T00:00:00Z",
        semantic_digest=payload.payload_sha256,
        payload_sha256=payload.payload_sha256,
        payload_size_bytes=payload.canonical_size_bytes,
        attachment_count=0,
        attachment_total_bytes=0,
    )
    return metadata, payload


def _build_router(
    writer: JournalWriter,
    ingress: EmployeeIngressService,
) -> DurableEmployeeIngressRouter:
    workforce = ProjectionState()
    workforce.cursor_sequence = 0
    workforce.cursor_hash = GENESIS_HASH
    workforce.employees["agt_alpha"] = EmployeeDefinition(
        agent_id="agt_alpha",
        tenant_key="tenant_1",
        owner_principal_id="ou_owner",
        name="alpha",
        tool="codex",
        model="gpt-5.6-sol",
        effort="xhigh",
        worker_type=WorkerType.VISIBLE,
        state=EmployeeState.ACTIVE,
        bot_principal_id="bot_alpha",
        member_groups=("oc_team",),
        aggregate_version=1,
    )
    workforce.bot_principals["bot_alpha"] = BotPrincipal(
        bot_principal_id="bot_alpha",
        tenant_key="tenant_1",
        agent_id="agt_alpha",
        app_id="cli_alpha",
        credential_ref="cred_alpha",
    )
    return DurableEmployeeIngressRouter(
        writer=writer,
        ingress_service=ingress,
        registry_provider=lambda: ProjectedAgentRegistry(workforce),
        channel_status_provider=_Channels(),
        requester_acl=RuntimeRequesterChatAcl(
            allowed_requesters=("ou_requester",),
            allowed_chats=("oc_team",),
        ),
        queue_limits=RouterQueueLimits(per_employee=4, per_team=8, global_limit=16),
        membership_health=_HealthyMembership(),
    )


def test_outbox_not_found_error_is_typed_and_does_not_mask_unexpected_key_errors(
    tmp_path: Path,
) -> None:
    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=MemoryAnchor(),
        hmac_key=b"k" * 32,
        writer_epoch=1,
    )
    store = BlobStore(
        tmp_path / "outbox-blobs",
        AesGcmEncryptionProvider(lambda _ref: b"b" * 32),
    )
    outbox = EmployeeOutboxService(
        writer=writer,
        blob_store=store,
        outbox_state=OutboxProjectionState(),
        active_key_id="k1",
    )
    try:
        assert issubclass(OutboxNotFoundError, OutboxServiceError)
        assert issubclass(OutboxNotFoundError, KeyError)

        missing_id = employee_outbox_id("tenant_1", "agt_alpha", "missing")
        with pytest.raises(OutboxNotFoundError):
            outbox.get_record(missing_id)
        with pytest.raises(OutboxNotFoundError):
            outbox.get_snapshot(missing_id)

        lifecycle = EmployeeOutboxLifecycle(outbox)
        assert (
            lifecycle.terminal_response_delivered(
                tenant_key="tenant_1",
                agent_id="agt_alpha",
                attempt_id="missing",
            )
            is False
        )

        # An unexpected internal KeyError must propagate rather than being coerced to False
        outbox.get_record = MagicMock(side_effect=KeyError("internal bug"))  # type: ignore[method-assign]
        with pytest.raises(KeyError, match="internal bug"):
            lifecycle.terminal_response_delivered(
                tenant_key="tenant_1",
                agent_id="agt_alpha",
                attempt_id="missing",
            )
    finally:
        outbox.close()
        writer.close()


def test_outbox_ingress_router_sync_incrementally_via_committed_tail_and_fail_closed_on_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor = MemoryAnchor()
    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=anchor,
        hmac_key=b"j" * 32,
        writer_epoch=1,
    )
    outbox_store = BlobStore(
        tmp_path / "outbox-blobs",
        AesGcmEncryptionProvider(lambda _ref: b"o" * 32),
    )
    ingress_store = BlobStore(
        tmp_path / "ingress-blobs",
        AesGcmEncryptionProvider(lambda _ref: b"i" * 32),
    )
    outbox = EmployeeOutboxService(
        writer=writer,
        blob_store=outbox_store,
        outbox_state=OutboxProjectionState(),
        active_key_id="k1",
    )
    ingress = EmployeeIngressService(
        writer=writer,
        blob_store=ingress_store,
        ingress_state=IngressProjectionState(),
        active_key_id="k1",
    )
    router = _build_router(writer, ingress)
    try:
        # Commit initial outbox + ingress records
        snap1 = _outbox_snapshot("att-1")
        outbox.append_snapshot(snap1)
        meta1, payload1 = _ingress_item(1)
        ingress.accept(meta1, payload1, request_id="req_1")
        router.synchronize_projection_unlocked()
        assert len(router.state.by_acceptance_id) == 1

        replay_calls: list[int] = []
        original_replay = writer.replay

        def counted_replay(from_sequence: int = 1, **kwargs):
            replay_calls.append(from_sequence)
            return original_replay(from_sequence, **kwargs)

        outbox_reads: list[str] = []
        orig_outbox_read = outbox._read_snapshot

        def counted_outbox_read(record, metadata):
            outbox_reads.append(record.outbox_id)
            return orig_outbox_read(record, metadata)

        ingress_reads: list[str] = []
        orig_ingress_read = ingress._read_record_payload

        def counted_ingress_read(record):
            ingress_reads.append(record.acceptance.acceptance_id)
            return orig_ingress_read(record)

        monkeypatch.setattr(writer, "replay", counted_replay)
        monkeypatch.setattr(outbox, "_read_snapshot", counted_outbox_read)
        monkeypatch.setattr(ingress, "_read_record_payload", counted_ingress_read)

        # Unrelated domain commit must advance all 3 projections without replay or historical Blob reads
        writer.commit(
            (JournalEvent(event_type="unrelated.tick", aggregate_id="tick:1", payload={}),),
            writer.get_aggregate_versions(("tick:1",)),
        )
        assert outbox.get_record(snap1.outbox_id).outbox_id == snap1.outbox_id
        ingress.synchronize_projection_unlocked()
        router.synchronize_projection_unlocked()
        assert replay_calls == []
        assert outbox_reads == []
        assert ingress_reads == []
        expected_seq = anchor.read().sequence
        assert outbox.state.cursor_sequence == expected_seq
        assert ingress.state.cursor_sequence == expected_seq
        assert router.state.cursor_sequence == expected_seq

        # Simulate another committed frame on Journal via writer.commit directly (new ingress event),
        # verifying Outbox skips historical blobs & Ingress only reads the newly appended Blob.
        meta2, payload2 = _ingress_item(2)
        blob_ref2 = ingress.blob_store.stage_and_publish(
            payload2.canonical_bytes,
            {
                "schema": "employee-ingress-v1",
                "tenant": meta2.tenant_key,
                "employee": meta2.agent_id,
                "envelope_id": meta2.envelope_id,
                "message_id": meta2.message_id,
                "payload_sha256": payload2.payload_sha256,
            },
            "k1",
        )
        writer.commit(
            (
                JournalEvent(
                    event_type="employee.ingress.accepted",
                    aggregate_id=meta2.dedup_key,
                    payload={
                        "metadata": meta2.to_dict(),
                        "acceptance_id": "acc_incremental_2",
                        "accepted_at": "2026-07-13T00:01:00Z",
                        "blob_ref": blob_ref2.to_dict(),
                        "transport_message_proof": True,
                    },
                ),
            ),
            writer.get_aggregate_versions((meta2.dedup_key,)),
        )
        assert outbox.get_record(snap1.outbox_id).outbox_id == snap1.outbox_id
        ingress.synchronize_projection_unlocked()
        router.synchronize_projection_unlocked()

        assert replay_calls == []
        assert outbox_reads == []
        assert ingress_reads == ["acc_incremental_2"]
        assert "acc_incremental_2" in router.state.by_acceptance_id

        # Fail closed when Journal anchor rolls back behind projection cursors
        anchor._state = anchor._state.__class__(
            sequence=1,
            frame_hash=writer.replay().__next__().frame_hash,
        )
        with pytest.raises(OutboxWriteDisabledError):
            outbox.get_record(snap1.outbox_id)
        with pytest.raises(IngressWriteDisabledError):
            ingress.synchronize_projection_unlocked()
        with pytest.raises(RouterWriteDisabledError):
            router.synchronize_projection_unlocked()
    finally:
        outbox.close()
        ingress.close()
        writer.close()
