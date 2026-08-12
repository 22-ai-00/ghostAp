"""Durable encrypted employee ingress admission service."""

from __future__ import annotations

import json
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator, Protocol

from src.utils.path import canonicalize_user_home_path

from ..journal.blob_store import (
    AesGcmEncryptionProvider,
    BlobError,
    BlobReadError,
    BlobRef,
    BlobStore,
    KeyResolutionError,
)
from ..journal.frame import GENESIS_HASH, JournalEvent, TransactionFrame
from ..journal.writer import CommitResult, CommitState, JournalWriter
from .models import (
    EmployeeIngressAck,
    EmployeeIngressMetadata,
    EmployeeIngressPayload,
    IngressDisposition,
)
from .projection import (
    IngressProjectionState,
    IngressRecord,
    is_ingress_event,
    reduce_ingress_event,
)


class IngressServiceError(RuntimeError):
    """Base class for employee ingress failures."""


class IngressConflictError(IngressServiceError):
    """One durable dedup identity was replayed with different semantics."""


class IngressCorrelationError(IngressServiceError):
    """A fallback action did not carry trusted server correlation."""


class IngressBlobError(IngressServiceError):
    """Encrypted ingress payload publication or verification failed."""


class IngressBlobRetryableError(IngressBlobError):
    """An authenticated ingress payload dependency is temporarily unavailable."""


class IngressWriteDisabledError(IngressServiceError):
    """The ingress write did not reach an anchored Journal state."""


class IngressClosedError(IngressServiceError):
    """Admission is closed for the service or employee after recovery failure."""


class EmployeeKeyring(Protocol):
    """The employee data-key provider reused by the isolated ingress store."""

    active_key_id: str

    def resolve(self, key_ref: str) -> bytes: ...


class EmployeeIngressService:
    """Own one encrypted BlobStore and anchor ingress before returning an ACK."""

    def __init__(
        self,
        *,
        writer: JournalWriter,
        blob_store: BlobStore,
        ingress_state: IngressProjectionState,
        active_key_id: str,
    ) -> None:
        if not isinstance(writer, JournalWriter):
            raise TypeError("writer must be a JournalWriter")
        if not isinstance(blob_store, BlobStore):
            raise TypeError("blob_store must be a BlobStore")
        if not isinstance(ingress_state, IngressProjectionState):
            raise TypeError("ingress_state must be IngressProjectionState")
        if not isinstance(active_key_id, str) or not active_key_id:
            raise ValueError("active_key_id must be non-empty")
        self._writer = writer
        self._blob_store = blob_store
        self._state = ingress_state
        self._active_key_id = active_key_id
        self._mutex = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        self._shared_blob_mutex = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        self._retained_shared_blob_ids: set[str] = set()
        self._admission_closed = False
        self._closed = False
        self.rebuild_projection()

    @classmethod
    def from_keyring(
        cls,
        *,
        writer: JournalWriter,
        ingress_state: IngressProjectionState,
        keyring: EmployeeKeyring,
        blob_root: str | Path,
    ) -> EmployeeIngressService:
        """Create the dedicated ingress store using the employee data keyring."""

        if not isinstance(blob_root, (str, Path)) or not str(blob_root):
            raise ValueError("blob_root must be non-empty")
        blob_store = BlobStore(
            canonicalize_user_home_path(blob_root),
            AesGcmEncryptionProvider(keyring.resolve),
        )
        try:
            return cls(
                writer=writer,
                blob_store=blob_store,
                ingress_state=ingress_state,
                active_key_id=keyring.active_key_id,
            )
        except BaseException:
            blob_store.close()
            raise

    @property
    def state(self) -> IngressProjectionState:
        return self._state

    @property
    def blob_store(self) -> BlobStore:
        return self._blob_store

    def retain_shared_blob(self, blob_id: str) -> None:
        """Protect and restore a Journal-anchored blob co-owned by another projection."""

        with self._shared_blob_mutex:
            self._blob_store.restore_quarantined_blob(blob_id)
            self._retained_shared_blob_ids.add(blob_id)

    def release_shared_blob(self, blob_id: str) -> None:
        """Release a failed pre-commit shared-blob reservation."""

        with self._shared_blob_mutex:
            self._retained_shared_blob_ids.discard(blob_id)

    @contextmanager
    def employee_dispatch_guard(self, *, router: object | None = None) -> Iterator[None]:
        """Hold the complete Ingress tier, optionally including its Router."""

        with self._mutex:
            if router is None:
                yield
                return
            router_guard = getattr(router, "_ingress_dispatch_guard", None)
            if not callable(router_guard):
                raise TypeError("router does not expose the Ingress tier guard")
            with router_guard():
                yield

    def synchronize_projection_unlocked(self) -> None:
        self._synchronize_projection_unlocked()

    def dispatch_identity_unlocked(self, acceptance_id: str) -> tuple[object, ...]:
        record = self._state.by_acceptance_id.get(acceptance_id)
        if record is None or record.disposition is not None or record.payload_tombstoned:
            raise IngressBlobError("ingress acceptance is not dispatchable")
        return (
            record.aggregate_id,
            record.acceptance.acceptance_id,
            record.metadata,
            record.blob_ref.content_hash,
        )

    def apply_committed_frame_unlocked(self, frame: TransactionFrame) -> None:
        if not isinstance(frame, TransactionFrame):
            raise TypeError("frame must be TransactionFrame")
        if not frame.committed:
            raise IngressWriteDisabledError("ingress frame must be committed")
        if frame.sequence != self._state.cursor_sequence + 1:
            raise IngressWriteDisabledError("ingress frame sequence is not continuous")
        expected_previous = self._state.cursor_hash or GENESIS_HASH
        if frame.previous_hash != expected_previous:
            raise IngressWriteDisabledError("ingress frame previous hash mismatch")
        for event in frame.events:
            if is_ingress_event(event.event_type):
                reduce_ingress_event(
                    self._state,
                    event,
                    frame_sequence=frame.sequence,
                    frame_hash=frame.frame_hash,
                )
        self._state.cursor_sequence = frame.sequence
        self._state.cursor_hash = frame.frame_hash

    def close(self) -> None:
        """Close only the ingress-owned BlobStore; the writer has another owner."""

        with self._mutex:
            if self._closed:
                return
            self._closed = True
            self._blob_store.close()

    def stop_admission(self) -> None:
        """Reject new transport ACK work while retaining dispatch reads."""

        with self._mutex:
            self._ensure_open_unlocked()
            self._admission_closed = True


    def accept(
        self,
        metadata: EmployeeIngressMetadata,
        payload: EmployeeIngressPayload,
        *,
        request_id: str,
        action_correlation: str | None = None,
    ) -> EmployeeIngressAck:
        """Persist, fsync, and anchor one acceptance before returning its ACK."""

        if not isinstance(metadata, EmployeeIngressMetadata):
            raise TypeError("metadata must be EmployeeIngressMetadata")
        if not isinstance(payload, EmployeeIngressPayload):
            raise TypeError("payload must be EmployeeIngressPayload")
        self._validate_incoming_payload(metadata, payload)
        self._validate_action_correlation(metadata, action_correlation)
        with self._mutex, self._writer.transaction_guard():
            self._ensure_open_unlocked()
            if self._admission_closed:
                raise IngressClosedError("employee ingress admission is closed")
            self._synchronize_projection_unlocked()
            employee_key = (metadata.tenant_key, metadata.agent_id)
            if employee_key in self._state.closed_employees:
                raise IngressClosedError("employee ingress is closed")
            existing = self._state.by_dedup_key.get(metadata.dedup_key)
            if existing is not None:
                self._verify_duplicate_unlocked(existing, metadata, payload)
                return self._ack(
                    existing,
                    metadata,
                    request_id=request_id,
                    duplicate=True,
                )

            labels = _blob_labels(metadata)
            before_ids = set(self._blob_store.iter_blob_ids())
            try:
                blob_ref = self._blob_store.stage_and_publish(
                    payload.canonical_bytes,
                    labels,
                    self._active_key_id,
                )
                self._verify_ref_and_payload(blob_ref, metadata, payload)
            except (BlobError, IngressBlobError) as exc:
                self._quarantine_new_blobs_unlocked(before_ids)
                raise IngressBlobError("ingress payload publication failed") from exc

            accepted_at = _utc_now()
            event = JournalEvent(
                event_type="employee.ingress.accepted",
                aggregate_id=metadata.dedup_key,
                payload={
                    "metadata": metadata.to_dict(),
                    "acceptance_id": f"acc_{uuid.uuid4().hex}",
                    "accepted_at": accepted_at,
                    "blob_ref": blob_ref.to_dict(),
                },
            )
            versions = self._writer.get_aggregate_versions([metadata.dedup_key])
            try:
                result = self._writer.commit(
                    [event],
                    versions,
                    expected_head_sequence=self._state.cursor_sequence,
                    expected_head_hash=self._state.cursor_hash or None,
                )
            except Exception:
                self._quarantine_blob_unlocked(blob_ref)
                raise
            if result.state != CommitState.ANCHORED:
                self._quarantine_blob_unlocked(blob_ref)
                raise IngressWriteDisabledError("ingress acceptance was not anchored")
            self._apply_frame_unlocked(result)
            record = self._state.by_dedup_key[metadata.dedup_key]
            return self._ack(record, metadata, request_id=request_id, duplicate=False)

    def get_payload(self, acceptance_id: str) -> EmployeeIngressPayload:
        """Read and authenticate one accepted payload for a later trusted stage."""

        with self._mutex:
            self._ensure_open_unlocked()
            record = self._state.by_acceptance_id.get(acceptance_id)
            if record is None:
                raise KeyError(acceptance_id)
            if record.payload_tombstoned:
                raise IngressBlobError("ingress payload is tombstoned")
            return self._read_record_payload(record)

    @contextmanager
    def dispatch_snapshot_guard(
        self,
        acceptance_id: str,
    ) -> Iterator[tuple[IngressRecord, EmployeeIngressPayload]]:
        """Freeze one dispatchable Inbox record through a Router commit.

        The ingress mutex is the outer domain lock.  The caller may next take
        the Router mutex and finally the Journal guard; this method must not
        pre-acquire the Journal guard or it would invert that shared order.
        """

        with self._mutex:
            self._ensure_open_unlocked()
            self._synchronize_projection_unlocked()
            record = self._state.by_acceptance_id.get(acceptance_id)
            if record is None:
                raise KeyError(acceptance_id)
            if record.disposition is not None or record.payload_tombstoned:
                raise IngressBlobError("ingress acceptance is not dispatchable")
            yield record, self._read_record_payload(record)

    def record_disposition(
        self,
        acceptance_id: str,
        *,
        state: str,
        reason_code: str,
    ) -> IngressDisposition:
        """Anchor safe lifecycle metadata; this does not enqueue Router work."""

        with self._mutex, self._writer.transaction_guard():
            self._ensure_open_unlocked()
            self._synchronize_projection_unlocked()
            record = self._state.by_acceptance_id.get(acceptance_id)
            if record is None:
                raise KeyError(acceptance_id)
            if record.disposition is not None:
                raise IngressConflictError("ingress disposition already recorded")
            draft = IngressDisposition(
                schema_version=1,
                disposition_id=f"dsp_{uuid.uuid4().hex}",
                acceptance_id=acceptance_id,
                state=state,
                reason_code=reason_code,
                journal_sequence=self._state.cursor_sequence + 1,
                journal_frame_hash="0" * 64,
                recorded_at=_utc_now(),
            )
            event = JournalEvent(
                event_type="employee.ingress.dispositioned",
                aggregate_id=record.aggregate_id,
                payload={
                    "acceptance_id": draft.acceptance_id,
                    "disposition_id": draft.disposition_id,
                    "state": draft.state,
                    "reason_code": draft.reason_code,
                    "recorded_at": draft.recorded_at,
                },
            )
            self._commit_unlocked(record.aggregate_id, event)
            updated = self._state.by_acceptance_id[acceptance_id]
            if updated.disposition is None:
                raise IngressWriteDisabledError("disposition projection was not applied")
            return updated.disposition

    def gc_terminal_payloads(self) -> int:
        """Durably tombstone terminal payloads before moving their blobs aside."""

        with self._mutex, self._writer.transaction_guard():
            self._ensure_open_unlocked()
            self._synchronize_projection_unlocked()
            present_blob_ids = set(self._blob_store.iter_blob_ids())
            candidates = tuple(
                record
                for record in self._state.by_acceptance_id.values()
                if record.terminal and record.blob_ref.blob_id in present_blob_ids
            )
            collected = 0
            for candidate in candidates:
                if not candidate.payload_tombstoned:
                    event = JournalEvent(
                        event_type="employee.ingress.payload_tombstoned",
                        aggregate_id=candidate.aggregate_id,
                        payload={
                            "acceptance_id": candidate.acceptance.acceptance_id,
                            "tombstoned_at": _utc_now(),
                        },
                    )
                    self._commit_unlocked(candidate.aggregate_id, event)
                try:
                    self._blob_store.quarantine_blob(candidate.blob_ref.blob_id)
                except BlobError:
                    continue
                collected += 1
            return collected

    def quarantine_unreferenced_blobs(self) -> int:
        """Quarantine only blobs outside the ingress projection live set."""

        with self._mutex, self._writer.transaction_guard():
            self._ensure_open_unlocked()
            self._synchronize_projection_unlocked()
            return self._quarantine_unreferenced_blobs_unlocked()

    def rebuild_projection(self) -> IngressProjectionState:
        """Replay Journal state and verify nonterminal blobs before admission."""

        with self._mutex:
            self._ensure_open_unlocked()
            fresh = IngressProjectionState()
            verify_all = not getattr(self, "_projection_verified", False)
            known_acceptance_ids = frozenset(self._state.by_acceptance_id)
            inbox_retry_identities: set[tuple[str, str]] = set()
            anchor = self._writer.anchor.read()
            cursor_hash = "" if anchor.sequence == 0 else anchor.frame_hash
            if getattr(self, "_projection_verified", False) and (
                self._state.cursor_sequence,
                self._state.cursor_hash,
            ) == (anchor.sequence, cursor_hash):
                return self._state
            anchored_frame_hash = GENESIS_HASH
            for frame in self._writer.replay():
                if frame.sequence > anchor.sequence:
                    break
                for event in frame.events:
                    if is_ingress_event(event.event_type):
                        reduce_ingress_event(
                            fresh,
                            event,
                            frame_sequence=frame.sequence,
                            frame_hash=frame.frame_hash,
                        )
                    elif event.event_type == "employee.ingress.router_inbox_retry":
                        acceptance_id = (
                            event.payload.get("acceptance_id")
                            if isinstance(event.payload, dict)
                            else None
                        )
                        if isinstance(acceptance_id, str):
                            inbox_retry_identities.add(
                                (event.aggregate_id, acceptance_id)
                            )
                fresh.cursor_sequence = frame.sequence
                fresh.cursor_hash = frame.frame_hash
                anchored_frame_hash = frame.frame_hash
            if anchored_frame_hash != anchor.frame_hash:
                raise IngressWriteDisabledError(
                    "ingress projection cannot verify the Journal anchor"
                )
            for record in fresh.by_acceptance_id.values():
                if record.terminal or record.payload_tombstoned:
                    continue
                if (
                    record.aggregate_id,
                    record.acceptance.acceptance_id,
                ) in inbox_retry_identities:
                    # The Router has already anchored the availability budget;
                    # it alone decides when the next authenticated read is due.
                    continue
                if not verify_all and record.acceptance.acceptance_id in known_acceptance_ids:
                    continue
                try:
                    self._read_record_payload(record)
                except IngressBlobRetryableError:
                    # Availability failures do not revoke employee admission.
                    # The Router owns the durable retry budget once dispatch runs.
                    continue
                except (
                    BlobError,
                    IngressBlobError,
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                ):
                    fresh.closed_employees.add(record.employee_key)
            self._replace_state_unlocked(fresh)
            self._quarantine_unreferenced_blobs_unlocked()
            self._projection_verified = True
            return self._state

    def _ensure_open_unlocked(self) -> None:
        if self._closed or self._blob_store.closed:
            raise IngressClosedError("employee ingress service is closed")

    def _synchronize_projection_unlocked(self) -> None:
        anchor = self._writer.anchor.read()
        sequence = anchor.sequence
        frame_hash = "" if anchor.sequence == 0 else anchor.frame_hash
        if (self._state.cursor_sequence, self._state.cursor_hash) != (
            sequence,
            frame_hash,
        ):
            self.rebuild_projection()

    def _replace_state_unlocked(self, fresh: IngressProjectionState) -> None:
        self._state.by_dedup_key = fresh.by_dedup_key
        self._state.by_acceptance_id = fresh.by_acceptance_id
        self._state.closed_employees = fresh.closed_employees
        self._state.cursor_sequence = fresh.cursor_sequence
        self._state.cursor_hash = fresh.cursor_hash

    def _apply_frame_unlocked(self, result: CommitResult) -> None:
        frame = result.frame
        for event in frame.events:
            if is_ingress_event(event.event_type):
                reduce_ingress_event(
                    self._state,
                    event,
                    frame_sequence=frame.sequence,
                    frame_hash=frame.frame_hash,
                )
        self._state.cursor_sequence = frame.sequence
        self._state.cursor_hash = frame.frame_hash

    def _commit_unlocked(self, aggregate_id: str, event: JournalEvent) -> CommitResult:
        result = self._writer.commit(
            [event],
            self._writer.get_aggregate_versions([aggregate_id]),
            expected_head_sequence=self._state.cursor_sequence,
            expected_head_hash=self._state.cursor_hash or None,
        )
        if result.state != CommitState.ANCHORED:
            raise IngressWriteDisabledError("ingress lifecycle event was not anchored")
        self._apply_frame_unlocked(result)
        return result

    def _verify_duplicate_unlocked(
        self,
        record: IngressRecord,
        metadata: EmployeeIngressMetadata,
        payload: EmployeeIngressPayload,
    ) -> None:
        existing = record.metadata
        comparable_fields = (
            "tenant_key",
            "agent_id",
            "bot_principal_id",
            "app_id",
            "envelope_id",
            "event_id",
            "message_id",
            "event_type",
            "action_identity",
            "chat_id",
            "thread_root_message_id",
            "sender_principal_id",
            "semantic_digest",
            "payload_sha256",
            "payload_size_bytes",
            "attachment_count",
            "attachment_total_bytes",
        )
        if any(getattr(existing, field) != getattr(metadata, field) for field in comparable_fields):
            raise IngressConflictError("durable employee ingress conflict")
        try:
            self._verify_ref_and_payload(record.blob_ref, existing, payload)
        except (BlobReadError, KeyResolutionError, OSError) as exc:
            raise IngressBlobRetryableError(
                "authenticated ingress payload is temporarily unavailable"
            ) from exc
        except (
            BlobError,
            IngressBlobError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            self._state.closed_employees.add(record.employee_key)
            raise IngressClosedError("authenticated ingress payload is unavailable") from exc

    def _verify_ref_and_payload(
        self,
        blob_ref: BlobRef,
        metadata: EmployeeIngressMetadata,
        payload: EmployeeIngressPayload,
    ) -> None:
        if dict(blob_ref.labels or {}) != _blob_labels(metadata):
            raise IngressBlobError("ingress blob labels do not match authority")
        raw = self._blob_store.read(blob_ref)
        if raw != payload.canonical_bytes or blob_ref.payload_hash != metadata.payload_sha256:
            raise IngressBlobError("ingress blob payload verification failed")

    def _read_record_payload(self, record: IngressRecord) -> EmployeeIngressPayload:
        try:
            if dict(record.blob_ref.labels or {}) != _blob_labels(record.metadata):
                raise IngressBlobError("ingress blob labels do not match authority")
            raw = self._blob_store.read(record.blob_ref)
            decoded = json.loads(raw)
            payload = EmployeeIngressPayload.from_dict(decoded)
            self._validate_incoming_payload(record.metadata, payload)
            return payload
        except (BlobReadError, KeyResolutionError, OSError) as exc:
            raise IngressBlobRetryableError(
                "authenticated ingress payload is temporarily unavailable"
            ) from exc
        except IngressBlobError:
            raise
        except (BlobError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise IngressBlobError(
                "authenticated ingress payload is unavailable"
            ) from exc

    def _quarantine_new_blobs_unlocked(self, before_ids: set[str]) -> None:
        try:
            new_ids = set(self._blob_store.iter_blob_ids()) - before_ids
        except BlobError:
            return
        for blob_id in new_ids:
            try:
                self._blob_store.quarantine_blob(blob_id)
            except BlobError:
                continue

    def _quarantine_unreferenced_blobs_unlocked(self) -> int:
        with self._shared_blob_mutex:
            live_ids = {
                record.blob_ref.blob_id
                for record in self._state.by_acceptance_id.values()
                if not record.payload_tombstoned
            }
            live_ids.update(self._retained_shared_blob_ids)
            orphan_ids = set(self._blob_store.iter_blob_ids()) - live_ids
            for blob_id in orphan_ids:
                self._blob_store.quarantine_blob(blob_id)
            return len(orphan_ids)

    def _quarantine_blob_unlocked(self, blob_ref: BlobRef) -> None:
        try:
            self._blob_store.quarantine_blob(blob_ref.blob_id)
        except BlobError:
            pass

    @staticmethod
    def _validate_incoming_payload(
        metadata: EmployeeIngressMetadata,
        payload: EmployeeIngressPayload,
    ) -> None:
        if metadata.envelope_id != payload.envelope_id:
            raise ValueError("payload envelope does not match metadata")
        if metadata.payload_sha256 != payload.payload_sha256:
            raise ValueError("payload hash does not match metadata")
        if metadata.payload_size_bytes != payload.canonical_size_bytes:
            raise ValueError("payload size does not match metadata")
        if metadata.attachment_count != len(payload.attachment_descriptors):
            raise ValueError("payload attachment count does not match metadata")
        if metadata.attachment_total_bytes != payload.attachment_total_bytes:
            raise ValueError("payload attachment size does not match metadata")

    @staticmethod
    def _validate_action_correlation(
        metadata: EmployeeIngressMetadata,
        action_correlation: str | None,
    ) -> None:
        if metadata.event_id:
            return
        if (
            not isinstance(action_correlation, str)
            or not action_correlation
            or action_correlation != metadata.action_identity
        ):
            raise IngressCorrelationError(
                "fallback ingress requires trusted action correlation"
            )

    @staticmethod
    def _ack(
        record: IngressRecord,
        metadata: EmployeeIngressMetadata,
        *,
        request_id: str,
        duplicate: bool,
    ) -> EmployeeIngressAck:
        return EmployeeIngressAck(
            schema_version=1,
            request_id=request_id,
            acceptance=record.acceptance,
            agent_id=metadata.agent_id,
            app_id=metadata.app_id,
            channel_generation=metadata.channel_generation,
            connection_id=metadata.connection_id,
            semantic_digest=metadata.semantic_digest,
            duplicate=duplicate,
            acknowledged_at=_utc_now(),
        )


def _blob_labels(metadata: EmployeeIngressMetadata) -> dict[str, str]:
    return {
        "schema": "employee-ingress-v1",
        "tenant": metadata.tenant_key,
        "employee": metadata.agent_id,
        "envelope_id": metadata.envelope_id,
        "dedup_key": metadata.dedup_key,
        "semantic_digest": metadata.semantic_digest,
    }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
