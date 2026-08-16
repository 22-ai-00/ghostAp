from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.autonomous.a2a.journal import (
    RemoteDispatchLedger,
    RemoteDispatchLedgerError,
)
from src.autonomous.journal.anchor import FileAnchor
from src.autonomous.journal.blob_store import AesGcmEncryptionProvider, BlobStore
from src.autonomous.journal.writer import JournalWriter
from src.autonomous.remote.models import (
    RemoteAgentDescriptor,
    RemoteAttemptPhase,
    RemoteDispatchRequest,
    RemoteObservation,
    RemoteTaskState,
)

_HMAC_KEY = b"a2a-ledger-recovery-hmac-key-at-least-32"
_DATA_KEY = b"r" * 32


def _request() -> RemoteDispatchRequest:
    return RemoteDispatchRequest(
        acceptance_id="acceptance-recovery-1",
        run_id="run-recovery-1",
        assignment_id="assignment-recovery-1",
        attempt_id="attempt-recovery-1",
        message_id="message-recovery-1",
        context_id="context-recovery-1",
        instruction="recover this dispatch without resending it",
        descriptor=RemoteAgentDescriptor(
            tenant_key="tenant-recovery",
            agent_id="agt_remote-recovery",
            card_url="https://agent.example/.well-known/agent-card.json",
            endpoint_url="https://agent.example/a2a",
            card_digest="c" * 64,
            credential_ref="credential-recovery",
        ),
    )


def _store(root: Path) -> BlobStore:
    return BlobStore(root, AesGcmEncryptionProvider(lambda _key_ref: _DATA_KEY))


def _writer(
    root: Path,
    anchor: object,
    store: BlobStore,
    *,
    epoch: int,
) -> JournalWriter:
    return JournalWriter.open(
        root,
        anchor=anchor,  # type: ignore[arg-type]
        hmac_key=_HMAC_KEY,
        writer_epoch=epoch,
        blob_ref_validator=lambda ref: _published(store, ref),
    )


def _published(store: BlobStore, ref: object) -> bool:
    try:
        store.read(ref)  # type: ignore[arg-type]
    except Exception:
        return False
    return True


def test_bound_task_and_observation_rebuild_without_duplicate_events(
    tmp_path: Path,
) -> None:
    journal_root = tmp_path / "journal"
    blob_root = tmp_path / "blobs"
    anchor = FileAnchor(tmp_path / "anchor.json")
    request = _request()
    payload = b'{"state":"working"}'

    first_store = _store(blob_root)
    first_writer = _writer(journal_root, anchor, first_store, epoch=1)
    first_ledger = RemoteDispatchLedger(first_writer, first_store, "a2a-key-v1")
    prepared = first_ledger.prepare(request)
    first_ledger.mark_executing(prepared)
    bound = first_ledger.bind_task(prepared, "task-recovery-1")
    observation = RemoteObservation(
        observation_id="obs-recovery-working",
        state=RemoteTaskState.WORKING,
        context_id=bound.context_id,
        task_id=bound.task_id,
        payload_digest=hashlib.sha256(payload).hexdigest(),
    )
    recorded = first_ledger.record_observation(bound, observation, payload)
    first_sequence = anchor.read().sequence
    first_writer.close()
    first_store.close()

    second_store = _store(blob_root)
    second_writer = _writer(journal_root, anchor, second_store, epoch=2)
    second_ledger = RemoteDispatchLedger(second_writer, second_store, "a2a-key-v1")
    try:
        recovered = second_ledger.snapshot(request.acceptance_id)
        assert recovered is not None
        assert recovered.phase is RemoteAttemptPhase.TRACKING
        assert recovered.handle == bound
        assert recovered.observations == (recorded,)
        assert second_store.read(recovered.handle.instruction_ref) == request.instruction.encode()
        assert second_store.read(recorded.payload_ref) == payload  # type: ignore[arg-type]

        assert second_ledger.prepare(request) == bound
        assert second_ledger.bind_task(prepared, "task-recovery-1") == bound
        assert second_ledger.record_observation(bound, observation, payload) == recorded
        assert anchor.read().sequence == first_sequence
    finally:
        second_writer.close()
        second_store.close()


def test_unknown_send_outcome_survives_restart_as_non_resendable_phase(
    tmp_path: Path,
) -> None:
    journal_root = tmp_path / "journal"
    blob_root = tmp_path / "blobs"
    anchor = FileAnchor(tmp_path / "anchor.json")
    request = _request()

    first_store = _store(blob_root)
    first_writer = _writer(journal_root, anchor, first_store, epoch=1)
    first_ledger = RemoteDispatchLedger(first_writer, first_store, "a2a-key-v1")
    handle = first_ledger.prepare(request)
    first_ledger.mark_executing(handle)
    first_ledger.mark_send_uncertain(handle)
    first_writer.close()
    first_store.close()

    second_store = _store(blob_root)
    second_writer = _writer(journal_root, anchor, second_store, epoch=2)
    second_ledger = RemoteDispatchLedger(second_writer, second_store, "a2a-key-v1")
    try:
        recovered_handle = second_ledger.prepare(request)
        recovered = second_ledger.mark_executing(recovered_handle)

        assert recovered.phase is RemoteAttemptPhase.SEND_UNCERTAIN
        assert recovered.state is RemoteTaskState.OUTCOME_UNCERTAIN
        assert recovered.handle.task_id == ""
        assert anchor.read().sequence == 3
    finally:
        second_writer.close()
        second_store.close()


def test_post_anchor_exception_is_recoverable_and_never_republishes_instruction(
    tmp_path: Path,
) -> None:
    journal_root = tmp_path / "journal"
    blob_root = tmp_path / "blobs"
    durable_anchor = FileAnchor(tmp_path / "anchor.json")

    class RaiseAfterAnchor:
        production_safe = True

        @staticmethod
        def read():
            return durable_anchor.read()

        @staticmethod
        def compare_and_swap(*args):
            assert durable_anchor.compare_and_swap(*args) is True
            raise OSError("anchor fsync outcome intentionally ambiguous")

    first_store = _store(blob_root)
    first_writer = _writer(journal_root, RaiseAfterAnchor(), first_store, epoch=1)
    first_ledger = RemoteDispatchLedger(first_writer, first_store, "a2a-key-v1")
    with pytest.raises(RemoteDispatchLedgerError, match="was not anchored"):
        first_ledger.prepare(_request())
    assert durable_anchor.read().sequence == 1
    assert len(first_store.iter_blob_ids()) == 1
    first_writer.close()
    first_store.close()

    second_store = _store(blob_root)
    second_writer = _writer(journal_root, durable_anchor, second_store, epoch=2)
    second_ledger = RemoteDispatchLedger(second_writer, second_store, "a2a-key-v1")
    try:
        recovered = second_ledger.prepare(_request())
        assert second_ledger.snapshot(recovered.acceptance_id) is not None
        assert durable_anchor.read().sequence == 1
        assert len(second_store.iter_blob_ids()) == 1
    finally:
        second_writer.close()
        second_store.close()
