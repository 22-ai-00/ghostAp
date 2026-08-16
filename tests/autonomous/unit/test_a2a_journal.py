from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from src.autonomous.a2a.codec import (
    A2ANormalizedStatus,
    A2AObservationKind,
    NormalizedA2AObservation,
)
from src.autonomous.a2a.journal import (
    RemoteDispatchLedger,
    RemoteDispatchLedgerError,
)
from src.autonomous.journal.anchor import MemoryAnchor
from src.autonomous.journal.blob_store import AesGcmEncryptionProvider, BlobStore
from src.autonomous.journal.writer import JournalWriter
from src.autonomous.remote.models import (
    RemoteAgentDescriptor,
    RemoteAttemptPhase,
    RemoteDispatchRequest,
    RemoteObservation,
    RemoteTaskState,
)
from src.autonomous.remote.projection import (
    EVENT_CANCEL_REQUESTED,
    EVENT_EXECUTING,
    EVENT_OBSERVATION,
    EVENT_PREPARED,
    EVENT_SEND_UNCERTAIN,
    EVENT_TASK_BOUND,
)

_HMAC_KEY = b"a2a-ledger-test-hmac-key-at-least-32-bytes"
_DATA_KEY = b"a" * 32


def _descriptor() -> RemoteAgentDescriptor:
    return RemoteAgentDescriptor(
        tenant_key="tenant-1",
        agent_id="agt_remote-reviewer",
        card_url="https://agent.example/.well-known/agent-card.json",
        endpoint_url="https://agent.example/a2a",
        card_digest="b" * 64,
        credential_ref="credential-a2a-reviewer",
    )


def _request(**changes: object) -> RemoteDispatchRequest:
    values: dict[str, object] = {
        "acceptance_id": "acceptance-a2a-1",
        "run_id": "run-1",
        "assignment_id": "assignment-1",
        "attempt_id": "attempt-1",
        "message_id": "message-1",
        "context_id": "context-1",
        "instruction": "review the patch; private-marker-82",
        "descriptor": _descriptor(),
    }
    values.update(changes)
    return RemoteDispatchRequest(**values)  # type: ignore[arg-type]


def _open_stack(tmp_path: Path) -> tuple[RemoteDispatchLedger, JournalWriter, BlobStore]:
    store = BlobStore(
        tmp_path / "blobs",
        AesGcmEncryptionProvider(lambda _key_ref: _DATA_KEY),
    )
    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=MemoryAnchor(),
        hmac_key=_HMAC_KEY,
        writer_epoch=1,
        blob_ref_validator=lambda ref: _is_published(store, ref),
    )
    return RemoteDispatchLedger(writer, store, "remote-data-key-v1"), writer, store


def _is_published(store: BlobStore, ref: object) -> bool:
    try:
        store.read(ref)  # type: ignore[arg-type]
    except Exception:
        return False
    return True


def _event_types(writer: JournalWriter) -> list[str]:
    return [event.event_type for frame in writer.replay() for event in frame.events]


def _observation(
    handle,
    state: RemoteTaskState,
    payload: bytes,
    *,
    observation_id: str,
) -> RemoteObservation:
    return RemoteObservation(
        observation_id=observation_id,
        state=state,
        context_id=handle.context_id,
        task_id=handle.task_id,
        payload_digest=hashlib.sha256(payload).hexdigest(),
    )


def test_prepare_encrypts_instruction_before_anchoring_and_is_idempotent(
    tmp_path: Path,
) -> None:
    ledger, writer, store = _open_stack(tmp_path)
    request = _request()
    try:
        handle = ledger.prepare(request)
        replayed = ledger.prepare(request)

        assert replayed == handle
        assert store.read(handle.instruction_ref) == request.instruction.encode()
        assert _event_types(writer) == [EVENT_PREPARED]
        journal_bytes = writer.journal_path.read_bytes()
        assert request.instruction.encode() not in journal_bytes
        assert b"private-marker-82" not in journal_bytes
        assert handle.instruction_ref.labels == {
            "acceptance_id": request.acceptance_id,
            "kind": "a2a_remote_instruction",
            "tenant_key": request.descriptor.tenant_key,
        }
    finally:
        writer.close()
        store.close()


def test_prepare_rejects_acceptance_and_authority_idempotency_drift(
    tmp_path: Path,
) -> None:
    ledger, writer, store = _open_stack(tmp_path)
    request = _request()
    try:
        ledger.prepare(request)

        with pytest.raises(RemoteDispatchLedgerError, match="idempotency conflict"):
            ledger.prepare(_request(context_id="context-other"))
        with pytest.raises(RemoteDispatchLedgerError, match="authority collision"):
            ledger.prepare(
                _request(
                    acceptance_id="acceptance-a2a-2",
                    message_id="message-2",
                    context_id="context-2",
                )
            )

        assert _event_types(writer) == [EVENT_PREPARED]
    finally:
        writer.close()
        store.close()


def test_execute_bind_record_and_cancel_are_anchored_and_idempotent(
    tmp_path: Path,
) -> None:
    ledger, writer, store = _open_stack(tmp_path)
    try:
        prepared = ledger.prepare(_request())
        executing = ledger.mark_executing(prepared)
        assert ledger.mark_executing(prepared) == executing
        handle = ledger.bind_task(prepared, "task-1")
        assert ledger.bind_task(prepared, "task-1") == handle

        working_payload = b'{"kind":"status","state":"working"}'
        working = _observation(
            handle,
            RemoteTaskState.WORKING,
            working_payload,
            observation_id="obs-working-1",
        )
        recorded = ledger.record_observation(handle, working, working_payload)
        repeated = ledger.record_observation(handle, recorded, working_payload)
        assert repeated == recorded
        assert recorded.payload_ref is not None
        assert store.read(recorded.payload_ref) == working_payload

        canceled = ledger.request_cancel(handle)
        assert canceled.phase is RemoteAttemptPhase.CANCEL_REQUESTED
        assert canceled.cancel_requested is True
        assert ledger.request_cancel(handle) == canceled
        assert _event_types(writer) == [
            EVENT_PREPARED,
            EVENT_EXECUTING,
            EVENT_TASK_BOUND,
            EVENT_OBSERVATION,
            EVENT_CANCEL_REQUESTED,
        ]
    finally:
        writer.close()
        store.close()


def test_binding_and_observation_collisions_fail_closed_without_extra_frame(
    tmp_path: Path,
) -> None:
    ledger, writer, store = _open_stack(tmp_path)
    try:
        prepared = ledger.prepare(_request())
        ledger.mark_executing(prepared)
        handle = ledger.bind_task(prepared, "task-1")

        with pytest.raises(RemoteDispatchLedgerError, match="task ID drift"):
            ledger.bind_task(prepared, "task-other")

        payload = b'{"state":"working"}'
        first = _observation(
            handle,
            RemoteTaskState.WORKING,
            payload,
            observation_id="obs-collision-1",
        )
        ledger.record_observation(handle, first, payload)
        collision = replace(first, state=RemoteTaskState.FAILED)
        with pytest.raises(RemoteDispatchLedgerError, match="idempotency conflict"):
            ledger.record_observation(handle, collision, payload)

        assert _event_types(writer) == [
            EVENT_PREPARED,
            EVENT_EXECUTING,
            EVENT_TASK_BOUND,
            EVENT_OBSERVATION,
        ]
    finally:
        writer.close()
        store.close()


def test_normalized_payload_is_published_and_remote_completion_stays_a_claim(
    tmp_path: Path,
) -> None:
    ledger, writer, store = _open_stack(tmp_path)
    try:
        prepared = ledger.prepare(_request())
        ledger.mark_executing(prepared)
        handle = ledger.bind_task(prepared, "task-1")
        payload = b'{"kind":"task_snapshot","status":"claimed_completed"}'
        normalized = NormalizedA2AObservation(
            kind=A2AObservationKind.TASK_SNAPSHOT,
            context_id=handle.context_id,
            task_id=handle.task_id,
            status=A2ANormalizedStatus.CLAIMED_COMPLETED,
            canonical_payload=payload,
            payload_digest=hashlib.sha256(payload).hexdigest(),
            observation_id=f"obs_{hashlib.sha256(payload).hexdigest()}",
        )

        result = ledger.record_observation(handle, normalized)
        snapshot = ledger.snapshot(handle.acceptance_id)

        assert snapshot is not None
        assert snapshot.phase is RemoteAttemptPhase.TERMINAL
        assert snapshot.state is RemoteTaskState.CLAIMED_COMPLETED
        assert snapshot.claimed_completed is True
        assert result.payload_ref is not None
        assert store.read(result.payload_ref) == payload
        assert result.payload_ref.labels == {
            "acceptance_id": handle.acceptance_id,
            "kind": "a2a_remote_observation",
            "tenant_key": handle.descriptor.tenant_key,
        }
        assert ledger.request_cancel(handle) == snapshot
        assert EVENT_CANCEL_REQUESTED not in _event_types(writer)
    finally:
        writer.close()
        store.close()


def test_identical_artifact_chunks_keep_distinct_durable_stream_positions(
    tmp_path: Path,
) -> None:
    ledger, writer, store = _open_stack(tmp_path)
    try:
        prepared = ledger.prepare(_request())
        ledger.mark_executing(prepared)
        handle = ledger.bind_task(prepared, "task-1")
        status_payload = b'{"state":"working"}'
        ledger.record_observation(
            handle,
            _observation(
                handle,
                RemoteTaskState.WORKING,
                status_payload,
                observation_id="obs-status-before-chunks",
            ),
            status_payload,
        )
        chunk_payload = b'{"append":true,"artifactId":"artifact-1","text":"same"}'
        normalized_chunk = NormalizedA2AObservation(
            kind=A2AObservationKind.ARTIFACT,
            context_id=handle.context_id,
            task_id=handle.task_id,
            status=None,
            canonical_payload=chunk_payload,
            payload_digest=hashlib.sha256(chunk_payload).hexdigest(),
            observation_id=f"obs_{hashlib.sha256(chunk_payload).hexdigest()}",
            artifact_id="artifact-1",
            append=True,
        )

        first = ledger.record_observation(handle, normalized_chunk)
        second = ledger.record_observation(handle, normalized_chunk)
        retried_first = ledger.record_observation(
            handle,
            normalized_chunk,
            sequence=1,
        )

        assert first.observation_id.endswith("_1")
        assert second.observation_id.endswith("_2")
        assert first != second
        assert retried_first == first
        assert ledger.next_observation_sequence(handle) == 3
        snapshot = ledger.snapshot(handle.acceptance_id)
        assert snapshot is not None
        assert snapshot.observations[-2:] == (first, second)
    finally:
        writer.close()
        store.close()


def test_artifact_cannot_continue_after_anchored_last_chunk(tmp_path: Path) -> None:
    ledger, writer, store = _open_stack(tmp_path)
    try:
        prepared = ledger.prepare(_request())
        ledger.mark_executing(prepared)
        handle = ledger.bind_task(prepared, "task-1")
        status_payload = b'{"state":"working"}'
        ledger.record_observation(
            handle,
            _observation(
                handle,
                RemoteTaskState.WORKING,
                status_payload,
                observation_id="obs-before-final-chunk",
            ),
            status_payload,
        )
        final_payload = b'{"artifactId":"artifact-1","lastChunk":true}'
        final_chunk = NormalizedA2AObservation(
            kind=A2AObservationKind.ARTIFACT,
            context_id=handle.context_id,
            task_id=handle.task_id,
            status=None,
            canonical_payload=final_payload,
            payload_digest=hashlib.sha256(final_payload).hexdigest(),
            observation_id=f"obs_{hashlib.sha256(final_payload).hexdigest()}",
            artifact_id="artifact-1",
            append=True,
            last_chunk=True,
        )
        ledger.record_observation(handle, final_chunk)
        blobs_before = store.iter_blob_ids()
        continuation_payload = b'{"artifactId":"artifact-1","text":"late"}'
        continuation = NormalizedA2AObservation(
            kind=A2AObservationKind.ARTIFACT,
            context_id=handle.context_id,
            task_id=handle.task_id,
            status=None,
            canonical_payload=continuation_payload,
            payload_digest=hashlib.sha256(continuation_payload).hexdigest(),
            observation_id=f"obs_{hashlib.sha256(continuation_payload).hexdigest()}",
            artifact_id="artifact-1",
            append=True,
        )

        with pytest.raises(RemoteDispatchLedgerError, match="transition rejected"):
            ledger.record_observation(handle, continuation)

        assert store.iter_blob_ids() == blobs_before
        snapshot = ledger.snapshot(handle.acceptance_id)
        assert snapshot is not None
        assert snapshot.observations[-1].last_chunk is True
    finally:
        writer.close()
        store.close()


def test_invalid_remote_content_is_not_reflected_or_published(tmp_path: Path) -> None:
    ledger, writer, store = _open_stack(tmp_path)
    secret = "REMOTE_SECRET_SHOULD_NOT_REFLECT"
    try:
        prepared = ledger.prepare(_request(instruction="safe instruction"))
        ledger.mark_executing(prepared)
        handle = ledger.bind_task(prepared, "task-1")
        payload = secret.encode()
        untrusted = RemoteObservation(
            observation_id="obs-untrusted-1",
            state=RemoteTaskState.WORKING,
            context_id=secret,
            task_id=handle.task_id,
            payload_digest=hashlib.sha256(payload).hexdigest(),
        )

        with pytest.raises(RemoteDispatchLedgerError) as captured:
            ledger.record_observation(handle, untrusted, payload)

        assert secret not in str(captured.value)
        assert secret.encode() not in writer.journal_path.read_bytes()
        assert len(store.iter_blob_ids()) == 1
    finally:
        writer.close()
        store.close()


def test_send_uncertain_is_replay_safe_and_contains_only_stable_error_code(
    tmp_path: Path,
) -> None:
    ledger, writer, store = _open_stack(tmp_path)
    try:
        handle = ledger.prepare(_request())
        ledger.mark_executing(handle)
        uncertain = ledger.mark_send_uncertain(handle)

        assert uncertain.phase is RemoteAttemptPhase.SEND_UNCERTAIN
        assert uncertain.state is RemoteTaskState.OUTCOME_UNCERTAIN
        assert uncertain.error_code == "remote_send_outcome_unknown"
        assert ledger.mark_send_uncertain(handle) == uncertain
        assert _event_types(writer) == [
            EVENT_PREPARED,
            EVENT_EXECUTING,
            EVENT_SEND_UNCERTAIN,
        ]
    finally:
        writer.close()
        store.close()


def test_blob_publish_failure_never_creates_observation_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, writer, store = _open_stack(tmp_path)
    try:
        prepared = ledger.prepare(_request())
        ledger.mark_executing(prepared)
        handle = ledger.bind_task(prepared, "task-1")
        before = tuple(writer.replay())
        payload = b'{"state":"working"}'
        observation = _observation(
            handle,
            RemoteTaskState.WORKING,
            payload,
            observation_id="obs-stage-failure",
        )

        def fail_publish(*_args: object, **_kwargs: object) -> object:
            raise OSError("injected blob failure")

        monkeypatch.setattr(store, "stage_and_publish", fail_publish)
        with pytest.raises(
            RemoteDispatchLedgerError,
            match="observation publication failed",
        ):
            ledger.record_observation(handle, observation, payload)

        assert tuple(writer.replay()) == before
        assert EVENT_OBSERVATION not in _event_types(writer)
    finally:
        writer.close()
        store.close()


def test_snapshot_unknown_returns_none_and_stale_handle_is_rejected(tmp_path: Path) -> None:
    ledger, writer, store = _open_stack(tmp_path)
    try:
        assert ledger.snapshot("acceptance-unknown") is None
        prepared = ledger.prepare(_request())
        ledger.mark_executing(prepared)
        bound = ledger.bind_task(prepared, "task-1")

        with pytest.raises(RemoteDispatchLedgerError, match="task authority drift"):
            ledger.request_cancel(prepared)
        assert ledger.snapshot(bound.acceptance_id) is not None
    finally:
        writer.close()
        store.close()
