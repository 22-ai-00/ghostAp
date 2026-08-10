"""Employee data publish service backed by Journal + encrypted BlobStore."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from ..journal.blob_store import BlobError, BlobRef, BlobStore
from ..journal.frame import GENESIS_HASH, JournalEvent, TransactionFrame
from ..journal.writer import CommitResult, CommitState, JournalWriter
from .models import (
    EmployeeDataDocumentV1,
    ExecutionAttemptContext,
    ExecutionHistoryPayloadV1,
    ExecutionHistoryRecordV1,
)
from .policy import build_document_labels, build_history_labels, validate_blob_ref_labels
from .projection import (
    DataAuthority,
    DataProjectionState,
    JournalHead,
    is_data_event,
    reduce_data_event,
)


class DataServiceError(RuntimeError):
    """Base error for employee data service failures."""


class DataConflictError(DataServiceError):
    """An occurrence with different content already exists."""


class DataHeadRaceError(DataServiceError):
    """The journal head advanced between read and commit."""


class DataBlobError(DataServiceError):
    """Blob publish or validation failed."""


class DataWriteDisabledError(DataServiceError):
    """Data writes are disabled due to authority or anchor failure."""


@dataclass(frozen=True)
class RecordResult:
    """Outcome of a successful history record publish."""

    record: ExecutionHistoryRecordV1
    commit_result: CommitResult


@dataclass(frozen=True)
class DocumentResult:
    """Outcome of a successful document publish."""

    document: EmployeeDataDocumentV1
    commit_result: CommitResult


@dataclass(frozen=True)
class AttemptResult:
    """Outcome of starting an execution attempt binding."""

    context: ExecutionAttemptContext
    commit_result: CommitResult


@dataclass(frozen=True)
class StagedHistoryPayload:
    """Verified encrypted payload prepared without any domain or Journal lock."""

    record: ExecutionHistoryRecordV1
    blob_ref: BlobRef
    content_hash: str


class EmployeeDataService:
    """Transactional publish of employee data records into Journal+BlobStore."""

    def __init__(
        self,
        *,
        writer: JournalWriter,
        blob_store: BlobStore,
        data_state: DataProjectionState,
        active_key_id: str,
        shard_timezone: str = "UTC",
        authority_required: bool = False,
    ) -> None:
        self._writer = writer
        self._blob_store = blob_store
        self._state = data_state
        self._active_key_id = active_key_id
        self._shard_timezone = shard_timezone
        self._authority_required = authority_required
        self._mutex = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        self._known_cursor = (
            data_state.cursor_sequence,
            data_state.cursor_hash,
        )

    @property
    def state(self) -> DataProjectionState:
        return self._state

    @property
    def shard_timezone(self) -> str:
        return self._shard_timezone

    def get_head(self) -> JournalHead:
        with self._mutex:
            return JournalHead(
                sequence=self._state.cursor_sequence,
                logical_hash=self._state.cursor_hash,
            )

    @contextmanager
    def employee_dispatch_guard(self) -> Iterator[None]:
        """Hold data projection state without taking the Journal guard."""

        with self._mutex:
            yield

    def close(self) -> None:
        self._blob_store.close()

    @contextmanager
    def read_guard(self) -> Iterator[None]:
        """Hold a consistent data projection view during an authorized read."""
        with self._mutex:
            yield

    def cutover_to_canonical(self) -> DataAuthority:
        """Publish the one-way independent data-writer authority fence."""

        with self._mutex, self._writer.transaction_guard():
            self._synchronize_projection_unlocked()
            current = self._state.data_authority
            if current.mode == "canonical":
                return current
            if current.mode != "legacy":
                raise DataWriteDisabledError("unknown data authority mode")
            next_sequence = self._state.cursor_sequence + 1
            event = JournalEvent(
                event_type="employee.data.authority_cutover",
                aggregate_id="employee-data-authority",
                payload={
                    "epoch": current.epoch + 1,
                    "mode": "canonical",
                    "cutover_sequence": next_sequence,
                },
            )
            versions = self._writer.get_aggregate_versions([event.aggregate_id])
            result = self._writer.commit(
                [event],
                versions,
                expected_head_sequence=self._state.cursor_sequence,
                expected_head_hash=self._state.cursor_hash or None,
            )
            if result.state is not CommitState.ANCHORED:
                raise DataWriteDisabledError("data authority cutover not anchored")
            self._apply_frame(result)
            return self._state.data_authority

    def _require_write_authority_unlocked(self) -> None:
        if not self._authority_required:
            return
        authority = self._state.data_authority
        if authority.mode != "canonical" or authority.epoch < 1:
            raise DataWriteDisabledError("canonical data authority is unavailable")

    def rebuild_projection(self) -> DataProjectionState:
        """Replay and atomically replace the live projection under one owner."""
        with self._mutex:
            fresh = self._replay_into_unlocked(DataProjectionState())
            self._replace_state_unlocked(fresh)
            return fresh

    def get_history_payload(self, record_id: str) -> ExecutionHistoryPayloadV1:
        """Read and authenticate one canonical terminal history payload."""

        with self._mutex:
            self._synchronize_projection_unlocked()
            metadata = self._state.history_records.get(record_id)
            if metadata is None or metadata.tombstoned:
                raise KeyError(record_id)
            ref = BlobRef.from_dict(metadata.blob_ref)
            validate_blob_ref_labels(
                ref,
                build_history_labels(
                    metadata.tenant_key,
                    metadata.owner_principal_id,
                    metadata.record_id,
                ),
            )
            raw = self._blob_store.read(ref)
            payload = ExecutionHistoryPayloadV1.from_dict(json.loads(raw))
            if (
                payload.record_id != metadata.record_id
                or payload.occurrence_key != metadata.occurrence_key
                or _canonical_payload(payload) != raw
            ):
                raise DataBlobError("history payload verification failed")
            return payload

    def read_blob(self, ref: BlobRef) -> bytes:
        """Internal recovery read for a projected encrypted Blob reference."""

        if not isinstance(ref, BlobRef):
            raise TypeError("ref must be BlobRef")
        return self._blob_store.read(ref)

    def verify_live_blobs(self) -> None:
        """Authenticate every live history and document Blob before readiness."""

        with self._mutex:
            self._synchronize_projection_unlocked()
            history_ids = tuple(
                record.record_id
                for record in self._state.history_records.values()
                if not record.tombstoned
            )
            documents = tuple(
                document
                for document in self._state.employee_documents.values()
                if not document.tombstoned
            )
        for record_id in history_ids:
            self.get_history_payload(record_id)
        for document in documents:
            ref = BlobRef.from_dict(document.blob_ref)
            validate_blob_ref_labels(
                ref,
                build_document_labels(
                    tenant_key=document.tenant_key,
                    owner_principal_id=document.owner_principal_id,
                    document_id=document.document_id,
                    kind=document.kind.value,
                ),
            )
            content = self._blob_store.read(ref)
            if hashlib.sha256(content).hexdigest() != document.content_hash:
                raise DataBlobError("document Blob verification failed")

    def emit(
        self,
        *,
        principal_id: str,
        operation: str,
        resource_id: str,
        outcome: str,
        reason: str,
    ) -> bool:
        """Anchor one non-secret read audit; return false on any failure."""

        try:
            nonce = str(time.time_ns())
            digest = hashlib.sha256(
                "\x00".join(
                    (principal_id, operation, resource_id, outcome, reason, nonce)
                ).encode()
            ).hexdigest()
            event = JournalEvent(
                event_type="employee.data.read_audited",
                aggregate_id=f"data-audit:{digest}",
                payload={
                    "principal_id": principal_id,
                    "operation": operation,
                    "resource_id": resource_id,
                    "outcome": outcome,
                    "reason": reason,
                },
            )
            with self._mutex, self._writer.transaction_guard():
                self._synchronize_projection_unlocked()
                self._require_write_authority_unlocked()
                versions = self._writer.get_aggregate_versions([event.aggregate_id])
                result = self._writer.commit(
                    [event],
                    versions,
                    expected_head_sequence=self._state.cursor_sequence,
                    expected_head_hash=self._state.cursor_hash or None,
                )
                if result.state is not CommitState.ANCHORED:
                    return False
                self._apply_frame(result)
            return True
        except Exception:
            return False

    def quarantine_unreferenced_blobs(self) -> int:
        """Quarantine orphan blobs against one locked projection snapshot."""
        with self._mutex:
            live_ids = {
                blob_id
                for record in self._state.history_records.values()
                if (blob_id := record.blob_ref.get("blob_id", ""))
            }
            live_ids.update(
                blob_id
                for document in self._state.employee_documents.values()
                if (blob_id := document.blob_ref.get("blob_id", ""))
            )
        orphan_ids = set(self._blob_store.iter_blob_ids()) - live_ids
        quarantined = 0
        for blob_id in orphan_ids:
            try:
                self._blob_store.quarantine_blob(blob_id)
                quarantined += 1
            except Exception:
                continue
        return quarantined

    def start_attempt(self, context: ExecutionAttemptContext) -> AttemptResult:
        """Anchor an immutable attempt binding before ACP dispatch."""
        if not isinstance(context, ExecutionAttemptContext):
            raise TypeError("context must be ExecutionAttemptContext")
        with self._mutex, self._writer.transaction_guard():
            self._synchronize_projection_unlocked()
            self._require_write_authority_unlocked()
            if context.attempt_id in self._state.execution_attempts:
                raise DataConflictError(f"attempt already started: {context.attempt_id}")
            event = JournalEvent(
                event_type="employee.execution_attempt.started",
                aggregate_id=context.attempt_id,
                payload=context.to_dict(),
            )
            versions = self._writer.get_aggregate_versions([context.attempt_id])
            result = self._writer.commit(
                [event],
                versions,
                expected_head_sequence=self._state.cursor_sequence,
                expected_head_hash=self._state.cursor_hash or None,
            )
            if result.state != CommitState.ANCHORED:
                raise DataWriteDisabledError("attempt commit not anchored")
            self._apply_frame(result)
            return AttemptResult(context=context, commit_result=result)

    def record_history(
        self,
        record: ExecutionHistoryRecordV1,
        sensitive_payload: ExecutionHistoryPayloadV1,
    ) -> RecordResult:
        """Publish one terminal history record with encrypted sensitive payload."""
        if not isinstance(record, ExecutionHistoryRecordV1):
            raise TypeError("record must be ExecutionHistoryRecordV1")
        if not isinstance(sensitive_payload, ExecutionHistoryPayloadV1):
            raise TypeError("sensitive_payload must be ExecutionHistoryPayloadV1")
        if record.record_id != sensitive_payload.record_id:
            raise ValueError("record/payload ID mismatch")
        if record.occurrence_key != sensitive_payload.occurrence_key:
            raise ValueError("record/payload occurrence mismatch")
        staged = self.stage_history_payload(record, sensitive_payload)
        try:
            duplicate = False
            with self._mutex, self._writer.transaction_guard():
                self._synchronize_projection_unlocked()
                self._require_write_authority_unlocked()
                occ_key = (record.tenant_key, record.agent_id, record.occurrence_key)
                existing_id = self._state.history_by_occurrence.get(occ_key)
                if existing_id is not None:
                    existing = self._state.history_records[existing_id]
                    if existing.blob_ref.get("content_hash", "") == staged.content_hash:
                        duplicate = True
                    else:
                        raise DataConflictError(
                            f"occurrence conflict: {record.occurrence_key}"
                        )
                if not duplicate:
                    event = self.preflight_history_event_unlocked(staged)
                    versions = self._writer.get_aggregate_versions([record.record_id])
                    result = self._writer.commit(
                        [event],
                        versions,
                        expected_head_sequence=self._state.cursor_sequence,
                        expected_head_hash=self._state.cursor_hash or None,
                    )
                    if result.state != CommitState.ANCHORED:
                        raise DataWriteDisabledError("history commit not anchored")
                    self._apply_frame(result)
            if duplicate:
                self.quarantine_staged_history(staged)
                return RecordResult(
                    record=record,
                    commit_result=CommitResult(
                        frame=None,  # type: ignore[arg-type]
                        state=CommitState.ANCHORED,
                    ),
                )
            return RecordResult(record=record, commit_result=result)
        except Exception:
            self.quarantine_staged_history(staged)
            raise

    def stage_history_payload(
        self,
        record: ExecutionHistoryRecordV1,
        sensitive_payload: ExecutionHistoryPayloadV1,
    ) -> StagedHistoryPayload:
        """Publish and read back sensitive history before acquiring any lock."""

        if not isinstance(record, ExecutionHistoryRecordV1):
            raise TypeError("record must be ExecutionHistoryRecordV1")
        if not isinstance(sensitive_payload, ExecutionHistoryPayloadV1):
            raise TypeError("sensitive_payload must be ExecutionHistoryPayloadV1")
        if record.record_id != sensitive_payload.record_id:
            raise ValueError("record/payload ID mismatch")
        if record.occurrence_key != sensitive_payload.occurrence_key:
            raise ValueError("record/payload occurrence mismatch")
        with self._mutex:
            self._synchronize_projection_unlocked()
            self._require_write_authority_unlocked()
        labels = build_history_labels(
            record.tenant_key,
            record.owner_principal_id,
            record.record_id,
        )
        payload_bytes = _canonical_payload(sensitive_payload)
        blob_ref: BlobRef | None = None
        try:
            blob_ref = self._blob_store.stage_and_publish(
                payload_bytes,
                labels,
                self._active_key_id,
            )
            validate_blob_ref_labels(blob_ref, labels)
            if self._blob_store.read(blob_ref) != payload_bytes:
                raise DataBlobError("blob readback verification failed")
        except Exception as exc:
            if blob_ref is not None:
                self._quarantine_blob_ids({blob_ref.blob_id})
            if isinstance(exc, (BlobError, DataBlobError)):
                raise
            raise DataBlobError("history payload publication failed") from exc
        return StagedHistoryPayload(
            record=record,
            blob_ref=blob_ref,
            content_hash=hashlib.sha256(payload_bytes).hexdigest(),
        )

    def quarantine_staged_history(self, staged: StagedHistoryPayload) -> None:
        if not isinstance(staged, StagedHistoryPayload):
            raise TypeError("staged must be StagedHistoryPayload")
        self._quarantine_blob_ids({staged.blob_ref.blob_id})

    def preflight_history_event_unlocked(
        self,
        staged: StagedHistoryPayload,
    ) -> JournalEvent:
        """Build and reduce one metadata event under the caller-held data guard."""

        if not isinstance(staged, StagedHistoryPayload):
            raise TypeError("staged must be StagedHistoryPayload")
        self._require_write_authority_unlocked()
        record = staged.record
        occurrence = (record.tenant_key, record.agent_id, record.occurrence_key)
        if record.record_id in self._state.history_records or (
            occurrence in self._state.history_by_occurrence
        ):
            raise DataConflictError(f"occurrence conflict: {record.occurrence_key}")
        event_payload: dict[str, Any] = {
            **record.to_dict(),
            "safe_summary": record.safe_summary.text,
            "blob_ref": staged.blob_ref.to_dict(),
        }
        del event_payload["tool_usage"]
        event = JournalEvent(
            event_type="employee.history.recorded",
            aggregate_id=record.record_id,
            payload=event_payload,
        )
        probe = self._state.clone()
        reduce_data_event(
            probe,
            event,
            frame_sequence=self._state.cursor_sequence + 1,
            frame_hash="0" * 64,
        )
        return event

    def synchronize_projection_unlocked(self) -> None:
        self._synchronize_projection_unlocked()

    def preflight_frame_unlocked(self, frame: TransactionFrame) -> None:
        if not frame.committed:
            raise DataHeadRaceError("data frame must be committed")
        if frame.sequence != self._state.cursor_sequence + 1:
            raise DataHeadRaceError("data frame sequence is not continuous")
        if frame.previous_hash != (self._state.cursor_hash or GENESIS_HASH):
            raise DataHeadRaceError("data frame previous hash mismatch")
        probe = self._state.clone()
        for event in frame.events:
            if is_data_event(event.event_type):
                reduce_data_event(
                    probe,
                    event,
                    frame_sequence=frame.sequence,
                    frame_hash=frame.frame_hash,
                )

    def apply_committed_frame_unlocked(self, frame: TransactionFrame) -> None:
        self.preflight_frame_unlocked(frame)
        for event in frame.events:
            if is_data_event(event.event_type):
                reduce_data_event(
                    self._state,
                    event,
                    frame_sequence=frame.sequence,
                    frame_hash=frame.frame_hash,
                )
        self._state.cursor_sequence = frame.sequence
        self._state.cursor_hash = frame.frame_hash
        self._known_cursor = (frame.sequence, frame.frame_hash)

    def _quarantine_blob_ids(self, blob_ids: set[str]) -> None:
        for blob_id in blob_ids:
            try:
                self._blob_store.quarantine_blob(blob_id)
            except Exception:
                continue

    def publish_document(
        self,
        document: EmployeeDataDocumentV1,
        content: bytes,
    ) -> DocumentResult:
        """Publish one employee data document with encrypted content."""
        if not isinstance(document, EmployeeDataDocumentV1):
            raise TypeError("document must be EmployeeDataDocumentV1")
        if not isinstance(content, bytes):
            raise TypeError("content must be bytes")
        content_hash = hashlib.sha256(content).hexdigest()
        if content_hash != document.content_hash:
            raise ValueError("content hash does not match document")
        with self._mutex, self._writer.transaction_guard():
            self._synchronize_projection_unlocked()
            self._require_write_authority_unlocked()
            if document.document_id in self._state.employee_documents:
                raise DataConflictError(f"document already exists: {document.document_id}")
            labels = build_document_labels(
                tenant_key=document.tenant_key,
                owner_principal_id=document.owner_principal_id,
                document_id=document.document_id,
                kind=document.kind.value,
            )
            blob_ref = self._blob_store.stage_and_publish(
                content,
                labels,
                self._active_key_id,
            )
            validate_blob_ref_labels(blob_ref, labels)
            readback = self._blob_store.read(blob_ref)
            if readback != content:
                raise DataBlobError("blob readback verification failed")
            aggregate_id = (
                f"data-chain:{_hash(document.tenant_key)}:"
                f"{document.agent_id}:{document.kind.value}:"
                f"{_hash(document.source_id)}"
            )
            event_payload: dict[str, Any] = {
                **document.to_dict(),
                "kind": document.kind.value,
                "blob_ref": blob_ref.to_dict(),
            }
            event = JournalEvent(
                event_type="employee.data.published",
                aggregate_id=aggregate_id,
                payload=event_payload,
            )
            versions = self._writer.get_aggregate_versions([aggregate_id])
            result = self._writer.commit(
                [event],
                versions,
                expected_head_sequence=self._state.cursor_sequence,
                expected_head_hash=self._state.cursor_hash or None,
            )
            if result.state != CommitState.ANCHORED:
                raise DataWriteDisabledError("document commit not anchored")
            self._apply_frame(result)
            return DocumentResult(document=document, commit_result=result)

    def replay_into(self, state: DataProjectionState) -> DataProjectionState:
        """Full Journal replay populating a fresh state."""
        with self._mutex:
            replayed = self._replay_into_unlocked(state)
            if state is self._state:
                self._known_cursor = (
                    state.cursor_sequence,
                    state.cursor_hash,
                )
            return replayed

    def _replay_into_unlocked(
        self,
        state: DataProjectionState,
    ) -> DataProjectionState:
        for frame in self._writer.replay():
            for event in frame.events:
                if is_data_event(event.event_type):
                    reduce_data_event(
                        state,
                        event,
                        frame_sequence=frame.sequence,
                        frame_hash=frame.frame_hash,
                    )
            state.cursor_sequence = frame.sequence
            state.cursor_hash = frame.frame_hash
        return state

    def _synchronize_projection_unlocked(self) -> None:
        if (
            self._state.cursor_sequence,
            self._state.cursor_hash,
        ) != self._known_cursor:
            return
        last = self._writer.get_last_frame()
        sequence = 0 if last is None else last.sequence
        logical_hash = "" if last is None else last.frame_hash
        if self._state.cursor_sequence != sequence or self._state.cursor_hash != logical_hash:
            self._replace_state_unlocked(self._replay_into_unlocked(DataProjectionState()))

    def _replace_state_unlocked(self, fresh: DataProjectionState) -> None:
        self._state.history_records = fresh.history_records
        self._state.history_by_employee_day = fresh.history_by_employee_day
        self._state.history_by_task = fresh.history_by_task
        self._state.history_by_occurrence = fresh.history_by_occurrence
        self._state.execution_attempts = fresh.execution_attempts
        self._state.employee_documents = fresh.employee_documents
        self._state.latest_employee_document = fresh.latest_employee_document
        self._state.legacy_data_sources = fresh.legacy_data_sources
        self._state.data_authority = fresh.data_authority
        self._state.data_read_audits = fresh.data_read_audits
        self._state.cursor_sequence = fresh.cursor_sequence
        self._state.cursor_hash = fresh.cursor_hash
        self._known_cursor = (fresh.cursor_sequence, fresh.cursor_hash)

    def _apply_frame(self, result: CommitResult) -> None:
        frame = result.frame
        for event in frame.events:
            if is_data_event(event.event_type):
                reduce_data_event(
                    self._state,
                    event,
                    frame_sequence=frame.sequence,
                    frame_hash=frame.frame_hash,
                )
        self._state.cursor_sequence = frame.sequence
        self._state.cursor_hash = frame.frame_hash
        self._known_cursor = (frame.sequence, frame.frame_hash)


def _canonical_payload(payload: ExecutionHistoryPayloadV1) -> bytes:
    return json.dumps(
        payload.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]
