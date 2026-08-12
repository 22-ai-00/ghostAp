"""Fault injection for group-ledger anchor outcome ambiguity."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from src.autonomous.context.group_ledger import (
    GroupContextLedger,
    GroupEventPayload,
    GroupLedgerError,
)
from src.autonomous.ingress.projection import IngressProjectionState
from src.autonomous.ingress.service import EmployeeIngressService
from src.autonomous.journal.anchor import FileAnchor
from src.autonomous.journal.blob_store import AesGcmEncryptionProvider, BlobStore
from src.autonomous.journal.writer import JournalWriter

_HMAC_KEY = b"group-ledger-anchor-ambiguity-key"
_DATA_KEY = b"d" * 32


def _blob_store(root: Path) -> BlobStore:
    return BlobStore(root, AesGcmEncryptionProvider(lambda _ref: _DATA_KEY))


def test_post_replace_anchor_error_preserves_shared_blob_for_verified_reopen(
    tmp_path: Path,
) -> None:
    durable_anchor = FileAnchor(tmp_path / "anchor.json")

    class RaiseAfterAnchor:
        production_safe = True

        @staticmethod
        def read():
            return durable_anchor.read()

        @staticmethod
        def compare_and_swap(*args):
            assert durable_anchor.compare_and_swap(*args) is True
            raise OSError("anchor directory fsync outcome unknown")

    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=RaiseAfterAnchor(),
        hmac_key=_HMAC_KEY,
        writer_epoch=1,
    )
    store = _blob_store(tmp_path / "ingress-blobs")
    ingress = EmployeeIngressService(
        writer=writer,
        blob_store=store,
        ingress_state=IngressProjectionState(),
        active_key_id="k1",
    )
    ledger = GroupContextLedger(
        writer=writer,
        blob_store=store,
        active_key_id="k1",
        blob_retainer=ingress.retain_shared_blob,
        blob_releaser=ingress.release_shared_blob,
    )
    payload = GroupEventPayload(
        sender_id="ou_sender",
        sender_id_type="open_id",
        sender_type="user",
        sender_tenant_key="tenant_1",
        text="anchored group context",
        timestamp=1.0,
    )

    try:
        with pytest.raises(GroupLedgerError, match="was not anchored"):
            ledger.publish(
                tenant_key="tenant_1",
                chat_id="oc_anchor",
                thread_id="omt_anchor",
                message_id="om_anchor",
                transport_principal_id="bot_anchor",
                transport_event_id="evt_anchor",
                payload=payload,
            )

        assert durable_anchor.read().sequence == 1
        assert len(store.iter_blob_ids()) == 1
        assert ingress.quarantine_unreferenced_blobs() == 0
        assert len(store.iter_blob_ids()) == 1
    finally:
        ingress.close()
        writer.close()

    recovered_writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=durable_anchor,
        hmac_key=_HMAC_KEY,
        writer_epoch=2,
    )
    recovered_store = _blob_store(tmp_path / "ingress-blobs")
    recovered_ingress = EmployeeIngressService(
        writer=recovered_writer,
        blob_store=recovered_store,
        ingress_state=IngressProjectionState(),
        active_key_id="k1",
    )
    recovered_ledger = GroupContextLedger(
        writer=recovered_writer,
        blob_store=recovered_store,
        active_key_id="k1",
        blob_retainer=recovered_ingress.retain_shared_blob,
        blob_releaser=recovered_ingress.release_shared_blob,
    )
    try:
        context = recovered_ledger.window(
            tenant_key="tenant_1",
            chat_id="oc_anchor",
            current_message_id="om_anchor",
        )
        assert len(context.records) == 1
        assert recovered_store.read(context.records[0].payload_ref) == payload.to_bytes()
        assert recovered_ingress.quarantine_unreferenced_blobs() == 0
    finally:
        recovered_ingress.close()
        recovered_writer.close()


def test_rejected_anchor_releases_reservation_only_after_verified_ledger_rebuild(
    tmp_path: Path,
) -> None:
    durable_anchor = FileAnchor(tmp_path / "anchor.json")

    class RejectingAnchor:
        production_safe = True

        @staticmethod
        def read():
            return durable_anchor.read()

        @staticmethod
        def compare_and_swap(*_args):
            return False

    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=RejectingAnchor(),
        hmac_key=_HMAC_KEY,
        writer_epoch=1,
    )
    store = _blob_store(tmp_path / "ingress-blobs")
    ingress = EmployeeIngressService(
        writer=writer,
        blob_store=store,
        ingress_state=IngressProjectionState(),
        active_key_id="k1",
    )
    ledger = GroupContextLedger(
        writer=writer,
        blob_store=store,
        active_key_id="k1",
        blob_retainer=ingress.retain_shared_blob,
        blob_releaser=ingress.release_shared_blob,
    )

    try:
        with pytest.raises(GroupLedgerError, match="was not anchored"):
            ledger.publish(
                tenant_key="tenant_1",
                chat_id="oc_rejected",
                thread_id="",
                message_id="om_rejected",
                transport_principal_id="bot_rejected",
                transport_event_id="evt_rejected",
                payload=GroupEventPayload(
                    sender_id="ou_sender",
                    sender_id_type="open_id",
                    sender_type="user",
                    sender_tenant_key="tenant_1",
                    text="unanchored group context",
                    timestamp=1.0,
                ),
            )

        assert durable_anchor.read().sequence == 0
        ingress.quarantine_unreferenced_blobs()
        assert len(store.iter_blob_ids()) == 1

        assert ledger.rebuild_projection() == 0
        ingress.quarantine_unreferenced_blobs()
        assert store.iter_blob_ids() == ()
        assert len(tuple((store.root / "quarantine").glob("*.blob"))) == 1
    finally:
        ingress.close()
        writer.close()


def test_verified_rebuild_serializes_before_new_blob_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=FileAnchor(tmp_path / "anchor.json"),
        hmac_key=_HMAC_KEY,
        writer_epoch=1,
    )
    store = _blob_store(tmp_path / "ingress-blobs")
    ingress = EmployeeIngressService(
        writer=writer,
        blob_store=store,
        ingress_state=IngressProjectionState(),
        active_key_id="k1",
    )
    ledger = GroupContextLedger(
        writer=writer,
        blob_store=store,
        active_key_id="k1",
        blob_retainer=ingress.retain_shared_blob,
        blob_releaser=ingress.release_shared_blob,
    )
    rebuild_scanning = threading.Event()
    release_rebuild = threading.Event()
    publish_done = threading.Event()
    errors: list[BaseException] = []
    rebuild_thread: threading.Thread
    original_replay = writer.replay

    def blocked_replay(*args, **kwargs):
        if threading.current_thread() is rebuild_thread:
            rebuild_scanning.set()
            assert release_rebuild.wait(2)
        return original_replay(*args, **kwargs)

    monkeypatch.setattr(writer, "replay", blocked_replay)

    def rebuild() -> None:
        try:
            ledger.rebuild_projection()
        except BaseException as exc:
            errors.append(exc)

    def publish() -> None:
        try:
            ledger.publish(
                tenant_key="tenant_1",
                chat_id="oc_concurrent",
                thread_id="",
                message_id="om_concurrent",
                transport_principal_id="bot_concurrent",
                transport_event_id="evt_concurrent",
                payload=GroupEventPayload(
                    sender_id="ou_sender",
                    sender_id_type="open_id",
                    sender_type="user",
                    sender_tenant_key="tenant_1",
                    text="concurrent group context",
                    timestamp=1.0,
                ),
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            publish_done.set()

    rebuild_thread = threading.Thread(target=rebuild)
    publish_thread = threading.Thread(target=publish)
    try:
        rebuild_thread.start()
        assert rebuild_scanning.wait(2)
        publish_thread.start()
        published_before_verified_rebuild = publish_done.wait(0.1)
        release_rebuild.set()
        rebuild_thread.join(2)
        publish_thread.join(2)

        assert published_before_verified_rebuild is False
        assert not rebuild_thread.is_alive()
        assert not publish_thread.is_alive()
        assert errors == []
        assert ingress.quarantine_unreferenced_blobs() == 0
        assert len(store.iter_blob_ids()) == 1
    finally:
        release_rebuild.set()
        rebuild_thread.join(2)
        publish_thread.join(2)
        ingress.close()
        writer.close()
