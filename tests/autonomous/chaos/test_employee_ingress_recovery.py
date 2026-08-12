from __future__ import annotations

import hashlib
import multiprocessing
import secrets
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from src.autonomous.ingress.models import EmployeeIngressMetadata, EmployeeIngressPayload
from src.autonomous.ingress.projection import (
    IngressProjectionState,
    reduce_ingress_event,
)
from src.autonomous.ingress.service import (
    EmployeeIngressService,
    IngressBlobError,
    IngressClosedError,
    IngressWriteDisabledError,
)
from src.autonomous.journal.anchor import FileAnchor, MemoryAnchor
from src.autonomous.journal.blob_store import (
    AesGcmEncryptionProvider,
    BlobPublishError,
    BlobStore,
)
from src.autonomous.journal.frame import JournalEvent
from src.autonomous.journal.writer import JournalWriter

HMAC_KEY = b"employee-ingress-chaos-hmac-key-32"
DATA_KEY = b"employee-ingress-data-key-32byte"
IPC_ACK_BOUND_SECONDS = 1.5


def _transport_index(prefix: str, raw: str) -> str:
    return prefix + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _payload() -> EmployeeIngressPayload:
    return EmployeeIngressPayload(
        schema_version=1,
        envelope_id="ing_" + "1" * 64,
        normalized_parts=({"type": "text", "text": "durable work"},),
        attachment_descriptors=(),
    )


def _message_payload(
    *,
    raw_chat_id: str = "oc_bound_chat",
    raw_message_id: str = "om_bound_message",
) -> EmployeeIngressPayload:
    return EmployeeIngressPayload(
        schema_version=1,
        envelope_id="ing_" + "2" * 64,
        normalized_parts=(
            {
                "type": "message",
                "remote_chat_id": raw_chat_id,
                "remote_message_id": raw_message_id,
                "remote_root_id": "",
            },
        ),
        attachment_descriptors=(),
    )


def _bound_message(
    *,
    raw_chat_id: str,
    raw_message_id: str,
    **metadata_overrides: object,
) -> tuple[EmployeeIngressPayload, EmployeeIngressMetadata]:
    payload = _message_payload(
        raw_chat_id=raw_chat_id,
        raw_message_id=raw_message_id,
    )
    metadata = _metadata(
        payload,
        chat_id=_transport_index("oc_", raw_chat_id),
        message_id=_transport_index("om_", raw_message_id),
        **metadata_overrides,
    )
    return payload, metadata


def _metadata(payload: EmployeeIngressPayload, **overrides: object) -> EmployeeIngressMetadata:
    values: dict[str, object] = {
        "schema_version": 1,
        "envelope_id": payload.envelope_id,
        "tenant_key": "tenant_1",
        "agent_id": "agt_alpha",
        "bot_principal_id": "bot_alpha",
        "app_id": "cli_alpha",
        "channel_generation": 3,
        "connection_id": "conn_1",
        "event_id": "evt_1",
        "message_id": "om_1",
        "event_type": "im.message.receive_v1",
        "action_identity": "",
        "chat_id": "oc_1",
        "thread_root_message_id": "om_root",
        "sender_principal_id": "ou_requester",
        "received_at": "2026-07-13T00:00:00Z",
        "semantic_digest": payload.payload_sha256,
        "payload_sha256": payload.payload_sha256,
        "payload_size_bytes": payload.canonical_size_bytes,
        "attachment_count": 0,
        "attachment_total_bytes": 0,
    }
    values.update(overrides)
    return EmployeeIngressMetadata(**values)


def _store(root: Path) -> BlobStore:
    return BlobStore(root, AesGcmEncryptionProvider(lambda _ref: DATA_KEY))


def _service(
    tmp_path: Path,
    *,
    anchor=None,
    fs_ops=None,
) -> tuple[EmployeeIngressService, JournalWriter, BlobStore]:
    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=anchor or MemoryAnchor(),
        hmac_key=HMAC_KEY,
        writer_epoch=1,
        fs_ops=fs_ops,
    )
    store = _store(tmp_path / "ingress-blobs")
    return (
        EmployeeIngressService(
            writer=writer,
            blob_store=store,
            ingress_state=IngressProjectionState(),
            active_key_id="k1",
        ),
        writer,
        store,
    )


def _ipc_worker(base_dir: str, connection) -> None:
    root = Path(base_dir)
    writer = JournalWriter.open(
        root / "journal",
        anchor=FileAnchor(root / "anchor.json"),
        hmac_key=HMAC_KEY,
        writer_epoch=1,
    )
    service = EmployeeIngressService(
        writer=writer,
        blob_store=_store(root / "ingress-blobs"),
        ingress_state=IngressProjectionState(),
        active_key_id="k1",
    )
    payload = _payload()
    started = time.monotonic()
    ack = service.accept(_metadata(payload), payload, request_id="req_ipc")
    elapsed = time.monotonic() - started
    connection.send(
        {
            "ack": ack.to_dict(),
            "elapsed_seconds": elapsed,
            "bound_seconds": IPC_ACK_BOUND_SECONDS,
        }
    )
    connection.close()
    service.close()
    writer.close()


@pytest.mark.slow
def test_ipc_ack_only_after_anchored_acceptance(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_ipc_worker, args=(str(tmp_path), child))

    process.start()
    result = parent.recv()
    process.join(timeout=5)

    assert process.exitcode == 0
    acceptance = result["ack"]["acceptance"]
    anchor = FileAnchor(tmp_path / "anchor.json").read()
    assert result["ack"]["duplicate"] is False
    assert anchor.sequence == acceptance["journal_sequence"] == 1
    assert anchor.frame_hash == acceptance["journal_frame_hash"]
    assert result["elapsed_seconds"] <= result["bound_seconds"] == 1.5
    print(
        "EI-IPC-01 "
        f"elapsed_seconds={result['elapsed_seconds']:.6f} "
        f"bound_seconds={result['bound_seconds']:.1f} "
        f"anchored_sequence={anchor.sequence}"
    )


def test_restart_replay_returns_stable_acceptance_for_new_generation(
    tmp_path: Path,
) -> None:
    anchor = MemoryAnchor()
    service, writer, _blob_store = _service(tmp_path, anchor=anchor)
    payload = _payload()
    first = service.accept(_metadata(payload), payload, request_id="req_1")
    service.close()
    writer.close()

    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=anchor,
        hmac_key=HMAC_KEY,
        writer_epoch=2,
    )
    service = EmployeeIngressService(
        writer=writer,
        blob_store=_store(tmp_path / "ingress-blobs"),
        ingress_state=IngressProjectionState(),
        active_key_id="k1",
    )
    replay = service.accept(
        _metadata(payload, channel_generation=9, connection_id="conn_restart"),
        payload,
        request_id="req_2",
    )

    assert replay.duplicate is True
    assert replay.acceptance == first.acceptance
    assert replay.channel_generation == 9
    assert replay.connection_id == "conn_restart"
    assert len(tuple(writer.replay())) == 1
    service.close()
    writer.close()


def test_anchored_message_acceptance_is_queryable_before_waiter_registration(
    tmp_path: Path,
) -> None:
    service, writer, _blob_store = _service(tmp_path)
    payload, metadata = _bound_message(
        raw_message_id="om_remote_message",
        raw_chat_id="oc_remote_chat",
    )
    ack = service.accept(metadata, payload, request_id="req_anchored_lookup")

    observed = service.wait_for_anchored_message_acceptance(
        tenant_key=metadata.tenant_key,
        agent_id=metadata.agent_id,
        bot_principal_id=metadata.bot_principal_id,
        app_id=metadata.app_id,
        event_type="im.message.receive_v1",
        chat_id=metadata.chat_id,
        message_id=metadata.message_id,
        timeout=0,
    )

    assert observed == ack.acceptance
    assert observed.journal_sequence == writer.anchor.read().sequence
    assert observed.journal_frame_hash == writer.anchor.read().frame_hash
    service.close()
    writer.close()


def test_message_acceptance_waiter_has_no_accept_before_register_lost_wakeup(
    tmp_path: Path,
) -> None:
    service, writer, _blob_store = _service(tmp_path)
    payload, metadata = _bound_message(
        raw_message_id="om_waited_message",
        raw_chat_id="oc_waited_chat",
    )
    started = threading.Event()
    completed = threading.Event()
    observed = []

    def wait_for_acceptance() -> None:
        started.set()
        observed.append(
            service.wait_for_anchored_message_acceptance(
                tenant_key=metadata.tenant_key,
                agent_id=metadata.agent_id,
                bot_principal_id=metadata.bot_principal_id,
                app_id=metadata.app_id,
                event_type="im.message.receive_v1",
                chat_id=metadata.chat_id,
                message_id=metadata.message_id,
                timeout=1.0,
            )
        )
        completed.set()

    waiter = threading.Thread(target=wait_for_acceptance)
    waiter.start()
    assert started.wait(0.5)
    ack = service.accept(metadata, payload, request_id="req_waited")
    assert completed.wait(0.5)
    waiter.join(timeout=0.5)

    assert observed == [ack.acceptance]
    service.close()
    writer.close()


def test_message_acceptance_index_replays_across_restart_and_reconnect(
    tmp_path: Path,
) -> None:
    anchor = MemoryAnchor()
    service, writer, _blob_store = _service(tmp_path, anchor=anchor)
    payload, metadata = _bound_message(
        raw_message_id="om_reconnected_message",
        raw_chat_id="oc_reconnected_chat",
    )
    first = service.accept(metadata, payload, request_id="req_generation_3")
    service.close()
    writer.close()

    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=anchor,
        hmac_key=HMAC_KEY,
        writer_epoch=2,
    )
    service = EmployeeIngressService(
        writer=writer,
        blob_store=_store(tmp_path / "ingress-blobs"),
        ingress_state=IngressProjectionState(),
        active_key_id="k1",
    )
    replay = service.accept(
        _metadata(
            payload,
            message_id=metadata.message_id,
            chat_id=metadata.chat_id,
            channel_generation=9,
            connection_id="conn_reconnected",
        ),
        payload,
        request_id="req_generation_9",
    )

    observed = service.wait_for_anchored_message_acceptance(
        tenant_key=metadata.tenant_key,
        agent_id=metadata.agent_id,
        bot_principal_id=metadata.bot_principal_id,
        app_id=metadata.app_id,
        event_type=metadata.event_type,
        chat_id=metadata.chat_id,
        message_id=metadata.message_id,
        timeout=0,
    )

    assert replay.duplicate is True
    assert replay.channel_generation == 9
    assert replay.connection_id == "conn_reconnected"
    assert observed == first.acceptance == replay.acceptance
    service.close()
    writer.close()


def test_transport_message_index_keeps_first_acceptance_when_event_ids_differ(
    tmp_path: Path,
) -> None:
    service, writer, _blob_store = _service(tmp_path)
    payload, base_metadata = _bound_message(
        raw_message_id="om_same_transport_message",
        raw_chat_id="oc_same_transport_chat",
    )
    coordinates = {
        "message_id": base_metadata.message_id,
        "chat_id": base_metadata.chat_id,
    }
    first = service.accept(
        _metadata(payload, event_id="evt_first", **coordinates),
        payload,
        request_id="req_first_event",
    )
    service.accept(
        _metadata(payload, event_id="evt_second", **coordinates),
        payload,
        request_id="req_second_event",
    )

    observed = service.wait_for_anchored_message_acceptance(
        tenant_key="tenant_1",
        agent_id="agt_alpha",
        bot_principal_id="bot_alpha",
        app_id="cli_alpha",
        event_type="im.message.receive_v1",
        chat_id=coordinates["chat_id"],
        message_id=coordinates["message_id"],
        timeout=0,
    )

    assert observed == first.acceptance
    service.close()
    writer.close()


def test_message_acceptance_proof_requires_authenticated_raw_coordinates(
    tmp_path: Path,
) -> None:
    service, writer, _blob_store = _service(tmp_path)
    payload = _message_payload(
        raw_chat_id="oc_payload_chat",
        raw_message_id="om_payload_message",
    )
    metadata = _metadata(
        payload,
        chat_id=_transport_index("oc_", "oc_different_chat"),
        message_id=_transport_index("om_", "om_different_message"),
    )

    service.accept(metadata, payload, request_id="req_mismatched_coordinates")

    assert (
        service.wait_for_anchored_message_acceptance(
            tenant_key=metadata.tenant_key,
            agent_id=metadata.agent_id,
            bot_principal_id=metadata.bot_principal_id,
            app_id=metadata.app_id,
            event_type=metadata.event_type,
            chat_id=metadata.chat_id,
            message_id=metadata.message_id,
            timeout=0,
        )
        is None
    )
    service.close()
    writer.close()


def test_message_acceptance_proof_requires_a_normalized_message_part_and_survives_restart(
    tmp_path: Path,
) -> None:
    anchor = MemoryAnchor()
    service, writer, _blob_store = _service(tmp_path, anchor=anchor)
    payload = EmployeeIngressPayload(
        schema_version=1,
        envelope_id="ing_" + "3" * 64,
        normalized_parts=(
            {
                "type": "text",
                "remote_chat_id": "oc_wrong_part_type",
                "remote_message_id": "om_wrong_part_type",
            },
        ),
        attachment_descriptors=(),
    )
    metadata = _metadata(
        payload,
        chat_id=_transport_index("oc_", "oc_wrong_part_type"),
        message_id=_transport_index("om_", "om_wrong_part_type"),
    )
    service.accept(metadata, payload, request_id="req_wrong_part_type")
    service.close()
    writer.close()

    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=anchor,
        hmac_key=HMAC_KEY,
        writer_epoch=2,
    )
    service = EmployeeIngressService(
        writer=writer,
        blob_store=_store(tmp_path / "ingress-blobs"),
        ingress_state=IngressProjectionState(),
        active_key_id="k1",
    )

    assert (
        service.wait_for_anchored_message_acceptance(
            tenant_key=metadata.tenant_key,
            agent_id=metadata.agent_id,
            bot_principal_id=metadata.bot_principal_id,
            app_id=metadata.app_id,
            event_type=metadata.event_type,
            chat_id=metadata.chat_id,
            message_id=metadata.message_id,
            timeout=0,
        )
        is None
    )
    service.close()
    writer.close()


def test_legacy_acceptance_schema_replays_without_granting_transport_proof(
    tmp_path: Path,
) -> None:
    service, writer, _blob_store = _service(tmp_path)
    payload, metadata = _bound_message(
        raw_chat_id="oc_legacy_chat",
        raw_message_id="om_legacy_message",
    )
    service.accept(metadata, payload, request_id="req_current_schema")
    frame = next(writer.replay())
    accepted = next(
        event
        for event in frame.events
        if event.event_type == "employee.ingress.accepted"
    )
    legacy_payload = dict(accepted.payload)
    legacy_payload.pop("transport_message_proof")
    state = IngressProjectionState()

    reduce_ingress_event(
        state,
        JournalEvent(
            event_type=accepted.event_type,
            aggregate_id=accepted.aggregate_id,
            payload=legacy_payload,
        ),
        frame_sequence=frame.sequence,
        frame_hash=frame.frame_hash,
    )

    projected = next(iter(state.by_acceptance_id.values()))
    assert projected.transport_message_proof is False
    service.close()
    writer.close()


def test_semantic_digest_mismatch_is_rejected_before_blob_publication(
    tmp_path: Path,
) -> None:
    service, writer, blob_store = _service(tmp_path)
    payload = _payload()
    metadata = _metadata(payload, semantic_digest="b" * 64)

    with (
        patch.object(blob_store, "stage_and_publish", wraps=blob_store.stage_and_publish) as publish,
        pytest.raises(ValueError, match="semantic digest"),
    ):
        service.accept(metadata, payload, request_id="req_bad_semantic_digest")

    publish.assert_not_called()
    service.close()
    writer.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tenant_key", "tenant_other"),
        ("agent_id", "agt_other"),
        ("bot_principal_id", "bot_other"),
        ("app_id", "cli_other"),
        ("event_type", "im.chat.member.user.added_v1"),
        ("chat_id", _transport_index("oc_", "oc_other")),
        ("message_id", _transport_index("om_", "om_other")),
    ],
)
def test_message_acceptance_proof_rejects_every_mismatched_transport_binding(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    service, writer, _blob_store = _service(tmp_path)
    payload, metadata = _bound_message(
        raw_message_id="om_bound_message",
        raw_chat_id="oc_bound_chat",
    )
    service.accept(metadata, payload, request_id="req_bound")
    query = {
        "tenant_key": metadata.tenant_key,
        "agent_id": metadata.agent_id,
        "bot_principal_id": metadata.bot_principal_id,
        "app_id": metadata.app_id,
        "event_type": metadata.event_type,
        "chat_id": metadata.chat_id,
        "message_id": metadata.message_id,
        "timeout": 0,
    }
    query[field] = value

    assert service.wait_for_anchored_message_acceptance(**query) is None
    service.close()
    writer.close()


def test_message_acceptance_waiter_is_released_when_admission_stops(
    tmp_path: Path,
) -> None:
    service, writer, _blob_store = _service(tmp_path)
    result = []
    started = threading.Event()

    def wait_for_acceptance() -> None:
        started.set()
        result.append(
            service.wait_for_anchored_message_acceptance(
                tenant_key="tenant_1",
                agent_id="agt_alpha",
                bot_principal_id="bot_alpha",
                app_id="cli_alpha",
                event_type="im.message.receive_v1",
                chat_id=_transport_index("oc_", "oc_never"),
                message_id=_transport_index("om_", "om_never"),
                timeout=5.0,
            )
        )

    waiter = threading.Thread(target=wait_for_acceptance)
    waiter.start()
    assert started.wait(0.5)
    service.stop_admission()
    waiter.join(timeout=0.5)

    assert not waiter.is_alive()
    assert result == [None]
    service.close()
    writer.close()


def test_message_acceptance_waiters_are_bounded_fail_closed(tmp_path: Path) -> None:
    service, writer, _blob_store = _service(tmp_path)
    parked = {threading.Event() for _ in range(service._MAX_ACCEPTANCE_WAITERS)}
    service._acceptance_waiters = {
        ("occupied",): parked,
    }

    started = time.monotonic()
    observed = service.wait_for_anchored_message_acceptance(
        tenant_key="tenant_1",
        agent_id="agt_alpha",
        bot_principal_id="bot_alpha",
        app_id="cli_alpha",
        event_type="im.message.receive_v1",
        chat_id=_transport_index("oc_", "oc_overloaded"),
        message_id=_transport_index("om_", "om_overloaded"),
        timeout=5.0,
    )

    assert observed is None
    assert time.monotonic() - started < 0.5
    assert service._acceptance_waiters == {("occupied",): parked}
    service.close()
    writer.close()


@pytest.mark.parametrize("damage", ("missing", "corrupt"))
def test_restart_closes_employee_when_nonterminal_blob_is_missing_or_corrupt(
    tmp_path: Path,
    damage: str,
) -> None:
    anchor = MemoryAnchor()
    service, writer, _blob_store = _service(tmp_path, anchor=anchor)
    payload = _payload()
    ack = service.accept(_metadata(payload), payload, request_id="req_1")
    record = service.state.by_acceptance_id[ack.acceptance.acceptance_id]
    blob_path = tmp_path / "ingress-blobs" / f"{record.blob_ref.blob_id}.blob"
    service.close()
    writer.close()
    if damage == "missing":
        blob_path.unlink()
    else:
        blob_path.write_bytes(b"corrupt")

    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=anchor,
        hmac_key=HMAC_KEY,
        writer_epoch=2,
    )
    service = EmployeeIngressService(
        writer=writer,
        blob_store=_store(tmp_path / "ingress-blobs"),
        ingress_state=IngressProjectionState(),
        active_key_id="k1",
    )

    assert ("tenant_1", "agt_alpha") in service.state.closed_employees
    with pytest.raises(IngressClosedError):
        service.accept(_metadata(payload), payload, request_id="req_2")
    service.close()
    writer.close()


def test_corrupt_original_blob_turns_duplicate_into_closed_ingress(
    tmp_path: Path,
) -> None:
    service, writer, _store_value = _service(tmp_path)
    payload = _payload()
    ack = service.accept(_metadata(payload), payload, request_id="req_1")
    record = service.state.by_acceptance_id[ack.acceptance.acceptance_id]
    path = tmp_path / "ingress-blobs" / f"{record.blob_ref.blob_id}.blob"
    path.write_bytes(b"corrupt")

    with pytest.raises(IngressClosedError):
        service.accept(_metadata(payload), payload, request_id="req_2")

    assert ("tenant_1", "agt_alpha") in service.state.closed_employees
    service.close()
    writer.close()


@pytest.mark.parametrize("damage", ("missing", "corrupt"))
def test_live_unreadable_blob_uses_the_ingress_error_contract(
    tmp_path: Path,
    damage: str,
) -> None:
    service, writer, _store_value = _service(tmp_path)
    payload = _payload()
    ack = service.accept(_metadata(payload), payload, request_id="req_1")
    record = service.state.by_acceptance_id[ack.acceptance.acceptance_id]
    path = tmp_path / "ingress-blobs" / f"{record.blob_ref.blob_id}.blob"
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"corrupt")

    with pytest.raises(IngressBlobError):
        service.get_payload(ack.acceptance.acceptance_id)

    service.close()
    writer.close()


def test_blob_publish_failure_has_no_journal_acceptance(tmp_path: Path) -> None:
    service, writer, store = _service(tmp_path)
    payload = _payload()
    with patch.object(
        store,
        "stage_and_publish",
        side_effect=BlobPublishError("disk full"),
    ):
        with pytest.raises(IngressBlobError):
            service.accept(_metadata(payload), payload, request_id="req_1")

    assert len(tuple(writer.replay())) == 0
    assert service.state.by_dedup_key == {}
    service.close()
    writer.close()


def test_blob_readback_mismatch_quarantines_publish_before_journal(
    tmp_path: Path,
) -> None:
    service, writer, store = _service(tmp_path)
    payload = _payload()

    with patch.object(store, "read", return_value=b"wrong authenticated payload"):
        with pytest.raises(IngressBlobError):
            service.accept(_metadata(payload), payload, request_id="req_1")

    assert len(tuple(writer.replay())) == 0
    assert not tuple((tmp_path / "ingress-blobs").glob("*.blob"))
    assert len(tuple((tmp_path / "ingress-blobs" / "quarantine").glob("*.blob"))) == 1
    service.close()
    writer.close()


class _FailingFsync:
    def fsync_file(self, _file_or_fd) -> None:
        raise OSError("injected journal fsync failure")

    def fsync_directory(self, _directory) -> None:
        return None


class _RejectingAnchor(MemoryAnchor):
    def compare_and_swap(self, *args) -> bool:
        return False


@pytest.mark.parametrize("failure", ("fsync", "anchor"))
def test_journal_failure_quarantines_published_blob_without_acceptance(
    tmp_path: Path,
    failure: str,
) -> None:
    service, writer, _store_value = _service(
        tmp_path,
        anchor=_RejectingAnchor() if failure == "anchor" else MemoryAnchor(),
        fs_ops=_FailingFsync() if failure == "fsync" else None,
    )
    payload = _payload()

    with pytest.raises((OSError, IngressWriteDisabledError)):
        service.accept(_metadata(payload), payload, request_id="req_1")

    assert service.state.by_dedup_key == {}
    assert not tuple((tmp_path / "ingress-blobs").glob("*.blob"))
    assert len(tuple((tmp_path / "ingress-blobs" / "quarantine").glob("*.blob"))) == 1
    service.close()
    writer.close()


def test_orphan_quarantine_ignores_live_acceptance_and_collects_unanchored_blob(
    tmp_path: Path,
) -> None:
    service, writer, store = _service(tmp_path)
    payload = _payload()
    service.accept(_metadata(payload), payload, request_id="req_1")
    orphan = store.stage_and_publish(
        b"orphan",
        {
            "schema": "employee-ingress-v1",
            "tenant": "tenant_1",
            "employee": "agt_alpha",
            "envelope_id": "ing_" + "9" * 64,
            "dedup_key": "dedup_" + "8" * 64,
            "semantic_digest": secrets.token_hex(32),
        },
        "k1",
    )

    assert service.quarantine_unreferenced_blobs() == 1
    assert not (tmp_path / "ingress-blobs" / f"{orphan.blob_id}.blob").exists()
    assert (
        tmp_path / "ingress-blobs" / "quarantine" / f"{orphan.blob_id}.blob"
    ).exists()
    service.close()
    writer.close()


def test_explicitly_retained_shared_blob_survives_ingress_rebuild(
    tmp_path: Path,
) -> None:
    service, writer, store = _service(tmp_path)
    shared = store.stage_and_publish(
        b"shared group context",
        {"kind": "group_event", "tenant_key": "tenant_1"},
        "k1",
    )
    store.quarantine_blob(shared.blob_id)

    service.retain_shared_blob(shared.blob_id)
    service.rebuild_projection()

    assert store.read(shared) == b"shared group context"
    assert service.quarantine_unreferenced_blobs() == 0
    service.close()
    writer.close()


def test_restart_quarantines_blob_left_by_crash_before_journal_commit(
    tmp_path: Path,
) -> None:
    anchor = MemoryAnchor()
    service, writer, _store_value = _service(tmp_path, anchor=anchor)
    payload = _payload()
    with patch.object(writer, "commit", side_effect=SystemExit("crash")):
        with pytest.raises(SystemExit, match="crash"):
            service.accept(_metadata(payload), payload, request_id="req_1")

    assert len(tuple(writer.replay())) == 0
    assert len(tuple((tmp_path / "ingress-blobs").glob("*.blob"))) == 1
    service.close()
    writer.close()

    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=anchor,
        hmac_key=HMAC_KEY,
        writer_epoch=2,
    )
    service = EmployeeIngressService(
        writer=writer,
        blob_store=_store(tmp_path / "ingress-blobs"),
        ingress_state=IngressProjectionState(),
        active_key_id="k1",
    )

    assert service.state.by_acceptance_id == {}
    assert not tuple((tmp_path / "ingress-blobs").glob("*.blob"))
    assert len(tuple((tmp_path / "ingress-blobs" / "quarantine").glob("*.blob"))) == 1
    service.close()
    writer.close()


@pytest.mark.parametrize(
    ("state", "reason_code", "error"),
    (
        ("completed", "completed", "state"),
        ("terminal", "Bad Reason", "reason_code"),
    ),
)
def test_invalid_disposition_never_enters_journal_and_restart_replay_is_healthy(
    tmp_path: Path,
    state: str,
    reason_code: str,
    error: str,
) -> None:
    anchor = MemoryAnchor()
    service, writer, _store_value = _service(tmp_path, anchor=anchor)
    payload = _payload()
    ack = service.accept(_metadata(payload), payload, request_id="req_1")
    journal_before = writer.journal_path.read_bytes()

    with pytest.raises(ValueError, match=error):
        service.record_disposition(
            ack.acceptance.acceptance_id,
            state=state,
            reason_code=reason_code,
        )

    assert writer.journal_path.read_bytes() == journal_before
    assert len(tuple(writer.replay())) == 1
    service.close()
    writer.close()

    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=anchor,
        hmac_key=HMAC_KEY,
        writer_epoch=2,
    )
    service = EmployeeIngressService(
        writer=writer,
        blob_store=_store(tmp_path / "ingress-blobs"),
        ingress_state=IngressProjectionState(),
        active_key_id="k1",
    )
    replay = service.accept(
        _metadata(payload, channel_generation=4, connection_id="conn_restart"),
        payload,
        request_id="req_2",
    )

    assert replay.duplicate is True
    assert replay.acceptance == ack.acceptance
    assert len(tuple(writer.replay())) == 1
    service.close()
    writer.close()
