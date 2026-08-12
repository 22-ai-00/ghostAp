"""Durable encrypted employee ingress admission service."""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
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
from ..journal.writer import (
    CommitResult,
    CommitState,
    JournalDeadlineExceededError,
    JournalWriter,
)
from .models import (
    EmployeeIngressAck,
    EmployeeIngressMetadata,
    EmployeeIngressPayload,
    IngressAcceptance,
    IngressDisposition,
)
from .projection import (
    IngressProjectionState,
    IngressRecord,
    MessageLogicalKey,
    is_ingress_event,
    message_deny_aggregate_id,
    message_logical_key,
    message_logical_key_from_metadata,
    message_transport_key,
    message_transport_key_from_metadata,
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


@dataclass(frozen=True, slots=True)
class MessageAcceptanceOutcome:
    """Durable winner plus the transport proven for the current operation.

    A wait reports its exact witnessed transport while retaining the canonical
    acceptance.  A deny that loses before its transport is witnessed reports
    the canonical winner's original transport instead.
    """

    status: str
    acceptance: IngressAcceptance | None
    channel_generation: int
    connection_id: str

    def __post_init__(self) -> None:
        if self.status not in {"accepted", "denied"}:
            raise ValueError("message acceptance outcome status is invalid")
        if (self.status == "accepted") != (self.acceptance is not None):
            raise ValueError("message acceptance outcome payload is inconsistent")
        if type(self.channel_generation) is not int or self.channel_generation <= 0:
            raise ValueError("message acceptance outcome generation is invalid")
        if not isinstance(self.connection_id, str) or not self.connection_id:
            raise ValueError("message acceptance outcome connection is invalid")


class EmployeeKeyring(Protocol):
    """The employee data-key provider reused by the isolated ingress store."""

    active_key_id: str

    def resolve(self, key_ref: str) -> bytes: ...


class EmployeeIngressService:
    """Own one encrypted BlobStore and anchor ingress before returning an ACK."""

    _MAX_ACCEPTANCE_WAITERS = 256

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
        self._transport_message_index: dict[tuple[str, ...], str] = {}
        self._acceptance_waiter_slots = threading.BoundedSemaphore(
            self._MAX_ACCEPTANCE_WAITERS
        )
        self._acceptance_progress = threading.Event()
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

    def record_snapshot(
        self,
        acceptance_id: str,
        *,
        deadline: float | None = None,
        allow_immediate: bool = False,
    ) -> IngressRecord | None:
        """Return one immutable in-memory record without replaying the Journal."""

        if not isinstance(acceptance_id, str) or not acceptance_id:
            return None
        hard_deadline = self._validated_deadline(deadline)
        if not self._acquire_mutex_before_deadline(
            hard_deadline,
            allow_immediate=allow_immediate,
        ):
            raise IngressWriteDisabledError(
                "ingress snapshot deadline expired before ingress lock"
            )
        try:
            return self._state.by_acceptance_id.get(acceptance_id)
        finally:
            self._mutex.release()

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
        self._index_accepted_frame_unlocked(frame)
        self._wake_resolved_acceptance_waiters_unlocked()

    def close(self) -> None:
        """Close only the ingress-owned BlobStore; the writer has another owner."""

        with self._mutex:
            if self._closed:
                return
            self._closed = True
            self._wake_all_acceptance_waiters_unlocked()
            self._blob_store.close()

    def stop_admission(self) -> None:
        """Reject new transport ACK work while retaining dispatch reads."""

        with self._mutex:
            self._ensure_open_unlocked()
            self._admission_closed = True
            self._wake_all_acceptance_waiters_unlocked()

    def wait_for_anchored_message_acceptance(
        self,
        *,
        tenant_key: str,
        agent_id: str,
        bot_principal_id: str,
        app_id: str,
        event_type: str,
        chat_id: str,
        message_id: str,
        channel_generation: int,
        connection_id: str,
        timeout: float,
    ) -> MessageAcceptanceOutcome | None:
        """Wait until this exact transport witnesses the logical winner.

        The lookup uses only secret-free metadata indexes.  Registration and
        the initial projection check share the ingress mutex with ``accept``;
        therefore accept-before-register and register-before-accept cannot
        lose a wakeup.  A stopped/closed service releases waiters fail-closed.
        """

        transport_key = message_transport_key(
            tenant_key=tenant_key,
            agent_id=agent_id,
            bot_principal_id=bot_principal_id,
            app_id=app_id,
            event_type=event_type,
            chat_id=chat_id,
            message_id=message_id,
            channel_generation=channel_generation,
            connection_id=connection_id,
        )
        logical_key = transport_key[:7]
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or float(timeout) < 0
            or float(timeout) > 30
        ):
            raise ValueError("acceptance wait timeout is invalid")
        deadline = time.monotonic() + float(timeout)
        if not self._acceptance_waiter_slots.acquire(blocking=False):
            return None
        try:
            while True:
                if not self._acquire_mutex_before_deadline(
                    deadline,
                    allow_immediate=True,
                ):
                    return None
                try:
                    if self._closed:
                        return None
                    self._ensure_open_unlocked()
                    observed = self._message_acceptance_outcome_unlocked(
                        logical_key,
                        transport_key=transport_key,
                    )
                    if observed is not None or self._admission_closed:
                        return observed
                    # Clear while holding the Ingress mutex.  Every state
                    # transition sets progress under that same mutex, so an
                    # accept between this clear and wait cannot be lost.
                    self._acceptance_progress.clear()
                finally:
                    self._mutex.release()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._acceptance_progress.wait(remaining)
        finally:
            # Slot release is lock-free from the Ingress perspective, so a
            # timed-out waiter cannot remain registered behind a contended
            # domain or registry mutex.
            self._acceptance_waiter_slots.release()

    def observe_anchored_message_acceptance(
        self,
        *,
        tenant_key: str,
        agent_id: str,
        bot_principal_id: str,
        app_id: str,
        event_type: str,
        chat_id: str,
        message_id: str,
        channel_generation: int,
        connection_id: str,
        deadline: float | None = None,
    ) -> MessageAcceptanceOutcome | None:
        """Observe an existing exact transport witness without waiting for one."""

        transport_key = message_transport_key(
            tenant_key=tenant_key,
            agent_id=agent_id,
            bot_principal_id=bot_principal_id,
            app_id=app_id,
            event_type=event_type,
            chat_id=chat_id,
            message_id=message_id,
            channel_generation=channel_generation,
            connection_id=connection_id,
        )
        hard_deadline = self._validated_deadline(deadline)
        if not self._acquire_mutex_before_deadline(hard_deadline):
            raise IngressWriteDisabledError(
                "message acceptance observation deadline expired before ingress lock"
            )
        try:
            if self._closed:
                return None
            self._ensure_open_unlocked()
            self._ensure_before_deadline(
                hard_deadline,
                operation="message acceptance observation",
            )
            return self._message_acceptance_outcome_unlocked(
                transport_key[:7],
                transport_key=transport_key,
            )
        finally:
            self._mutex.release()

    def deny_message_acceptance(
        self,
        *,
        tenant_key: str,
        agent_id: str,
        bot_principal_id: str,
        app_id: str,
        event_type: str,
        chat_id: str,
        message_id: str,
        channel_generation: int,
        connection_id: str,
        deadline: float | None = None,
    ) -> MessageAcceptanceOutcome:
        """Durably fence a logical message unless an acceptance already won.

        If acceptance won without an exact witness for these coordinates, the
        returned outcome identifies the canonical winner's original transport.
        """

        if event_type != "im.message.receive_v1":
            raise ValueError("message acceptance denial event type is invalid")
        transport_key = message_transport_key(
            tenant_key=tenant_key,
            agent_id=agent_id,
            bot_principal_id=bot_principal_id,
            app_id=app_id,
            event_type=event_type,
            chat_id=chat_id,
            message_id=message_id,
            channel_generation=channel_generation,
            connection_id=connection_id,
        )
        logical_key = transport_key[:7]
        hard_deadline = self._validated_deadline(deadline)
        if not self._acquire_mutex_before_deadline(hard_deadline):
            raise IngressWriteDisabledError(
                "message acceptance denial deadline expired before ingress lock"
            )
        try:
            try:
                guard = (
                    self._writer.transaction_guard()
                    if hard_deadline is None
                    else self._writer.transaction_guard(deadline=hard_deadline)
                )
                with guard:
                    self._ensure_before_deadline(
                        hard_deadline,
                        operation="message acceptance denial",
                    )
                    self._ensure_open_unlocked()
                    self._synchronize_projection_unlocked(deadline=hard_deadline)
                    observed = self._message_acceptance_outcome_unlocked(logical_key)
                    if observed is not None:
                        return observed
                    denied_at = _utc_now()
                    event = JournalEvent(
                        event_type="employee.ingress.message_acceptance_denied",
                        aggregate_id=message_deny_aggregate_id(transport_key),
                        payload={
                            "tenant_key": tenant_key,
                            "agent_id": agent_id,
                            "bot_principal_id": bot_principal_id,
                            "app_id": app_id,
                            "event_type": event_type,
                            "chat_id": chat_id,
                            "message_id": message_id,
                            "channel_generation": channel_generation,
                            "connection_id": connection_id,
                            "reason_code": "handoff_unconfirmed",
                            "denied_at": denied_at,
                        },
                    )
                    self._commit_unlocked(
                        event.aggregate_id,
                        event,
                        deadline=hard_deadline,
                    )
                    outcome = self._message_acceptance_outcome_unlocked(logical_key)
                    if outcome is None or outcome.status != "denied":
                        raise IngressWriteDisabledError(
                            "message acceptance denial projection was not applied"
                        )
                    return outcome
            except JournalDeadlineExceededError as exc:
                raise IngressWriteDisabledError(
                    "message acceptance denial deadline expired"
                ) from exc
        finally:
            self._mutex.release()


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
        transport_message_proof = self._payload_matches_transport_metadata(
            metadata,
            payload,
        )
        logical_key = (
            message_logical_key_from_metadata(metadata)
            if transport_message_proof
            else None
        )
        with self._mutex, self._writer.transaction_guard():
            self._ensure_open_unlocked()
            if self._admission_closed:
                raise IngressClosedError("employee ingress admission is closed")
            self._synchronize_projection_unlocked()
            employee_key = (metadata.tenant_key, metadata.agent_id)
            if employee_key in self._state.closed_employees:
                raise IngressClosedError("employee ingress is closed")
            if logical_key is not None:
                denied = self._state.message_acceptance_denials.get(logical_key)
                if denied is not None:
                    denied_acceptance_id = self._state.message_denied_acceptances.get(
                        logical_key
                    )
                    denied_record = self._state.by_acceptance_id.get(
                        denied_acceptance_id or ""
                    )
                    if denied_record is not None:
                        self._verify_logical_duplicate_unlocked(
                            denied_record,
                            metadata,
                            payload,
                        )
                        return self._ack(
                            denied_record,
                            metadata,
                            request_id=request_id,
                            duplicate=True,
                        )
                    return self._accept_denied_message_unlocked(
                        metadata,
                        payload,
                        request_id=request_id,
                    )

                winner_id = self._state.message_acceptance_winners.get(logical_key)
                winner = self._state.by_acceptance_id.get(winner_id or "")
                if winner is not None:
                    self._verify_logical_duplicate_unlocked(
                        winner,
                        metadata,
                        payload,
                    )
                    self._anchor_redelivery_witness_unlocked(winner, metadata)
                    return self._ack(
                        winner,
                        metadata,
                        request_id=request_id,
                        duplicate=True,
                    )

            existing = self._state.by_dedup_key.get(metadata.dedup_key)
            if existing is not None:
                self._verify_duplicate_unlocked(existing, metadata, payload)
                self._index_acceptance_unlocked(existing, payload)
                return self._ack(
                    existing,
                    metadata,
                    request_id=request_id,
                    duplicate=True,
                )

            if (
                metadata.event_type == "im.message.receive_v1"
                and not transport_message_proof
            ):
                return self._accept_invalid_transport_unlocked(
                    metadata,
                    payload,
                    request_id=request_id,
                )

            blob_ref = self._publish_payload_unlocked(metadata, payload)

            accepted_at = _utc_now()
            event = JournalEvent(
                event_type="employee.ingress.accepted",
                aggregate_id=metadata.dedup_key,
                payload={
                    "metadata": metadata.to_dict(),
                    "acceptance_id": f"acc_{uuid.uuid4().hex}",
                    "accepted_at": accepted_at,
                    "blob_ref": blob_ref.to_dict(),
                    "transport_message_proof": transport_message_proof,
                },
            )
            versions = self._writer.get_aggregate_versions([metadata.dedup_key])
            # A failed commit result is not proof that the monotonic anchor
            # stayed unchanged: FileAnchor may have replaced its file before
            # its directory fsync raised.  Keep the published blob until a
            # later verified projection rebuild can prove it is unreferenced.
            result = self._writer.commit(
                [event],
                versions,
                expected_head_sequence=self._state.cursor_sequence,
                expected_head_hash=self._state.cursor_hash or None,
            )
            if result.state != CommitState.ANCHORED:
                raise IngressWriteDisabledError("ingress acceptance was not anchored")
            self._apply_frame_unlocked(result)
            record = self._state.by_dedup_key[metadata.dedup_key]
            self._index_acceptance_unlocked(record, payload)
            return self._ack(record, metadata, request_id=request_id, duplicate=False)

    def get_payload(
        self,
        acceptance_id: str,
        *,
        deadline: float | None = None,
    ) -> EmployeeIngressPayload:
        """Read and authenticate one accepted payload for a later trusted stage."""

        hard_deadline = self._validated_deadline(deadline)
        if not self._acquire_mutex_before_deadline(hard_deadline):
            raise IngressBlobRetryableError(
                "ingress payload deadline expired before ingress lock"
            )
        try:
            self._ensure_before_deadline(
                hard_deadline,
                operation="ingress payload read",
            )
            self._ensure_open_unlocked()
            record = self._state.by_acceptance_id.get(acceptance_id)
            if record is None:
                raise KeyError(acceptance_id)
            if record.payload_tombstoned:
                raise IngressBlobError("ingress payload is tombstoned")
            payload = self._read_record_payload(record)
            self._ensure_before_deadline(
                hard_deadline,
                operation="ingress payload read",
            )
            return payload
        finally:
            self._mutex.release()

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

    def rebuild_projection(
        self,
        *,
        deadline: float | None = None,
    ) -> IngressProjectionState:
        """Replay Journal state and verify nonterminal blobs before admission."""

        hard_deadline = self._validated_deadline(deadline)
        if not self._acquire_mutex_before_deadline(hard_deadline):
            raise IngressWriteDisabledError(
                "ingress projection deadline expired before ingress lock"
            )
        try:
            self._ensure_before_deadline(
                hard_deadline,
                operation="ingress projection rebuild",
            )
            self._ensure_open_unlocked()
            fresh = IngressProjectionState()
            verify_all = not getattr(self, "_projection_verified", False)
            known_acceptance_ids = frozenset(self._state.by_acceptance_id)
            inbox_retry_identities: set[tuple[str, str]] = set()
            anchor = self._writer.anchor.read()
            self._ensure_before_deadline(
                hard_deadline,
                operation="ingress projection rebuild",
            )
            cursor_hash = "" if anchor.sequence == 0 else anchor.frame_hash
            if getattr(self, "_projection_verified", False) and (
                self._state.cursor_sequence,
                self._state.cursor_hash,
            ) == (anchor.sequence, cursor_hash):
                return self._state
            anchored_frame_hash = GENESIS_HASH
            replay = (
                self._writer.replay()
                if hard_deadline is None
                else self._writer.replay(deadline=hard_deadline)
            )
            for frame in replay:
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
                self._ensure_before_deadline(
                    hard_deadline,
                    operation="ingress projection rebuild",
                )
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
            self._ensure_before_deadline(
                hard_deadline,
                operation="ingress projection rebuild",
            )
            self._replace_state_unlocked(fresh)
            # Handoff callers need a deadline-bounded metadata refresh.  Blob
            # hygiene is an independent recovery concern and may scan storage.
            if hard_deadline is None:
                self._quarantine_unreferenced_blobs_unlocked()
            self._projection_verified = True
            return self._state
        finally:
            self._mutex.release()

    def _ensure_open_unlocked(self) -> None:
        if self._closed or self._blob_store.closed:
            raise IngressClosedError("employee ingress service is closed")

    def _synchronize_projection_unlocked(
        self,
        *,
        deadline: float | None = None,
    ) -> None:
        anchor = self._writer.anchor.read()
        self._ensure_before_deadline(
            deadline,
            operation="ingress projection synchronization",
        )
        sequence = anchor.sequence
        frame_hash = "" if anchor.sequence == 0 else anchor.frame_hash
        if (self._state.cursor_sequence, self._state.cursor_hash) != (
            sequence,
            frame_hash,
        ):
            self.rebuild_projection(deadline=deadline)

    @staticmethod
    def _validated_deadline(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        if (
            isinstance(deadline, bool)
            or not isinstance(deadline, (int, float))
            or not math.isfinite(float(deadline))
        ):
            raise ValueError("deadline must be a finite monotonic timestamp")
        return float(deadline)

    @staticmethod
    def _ensure_before_deadline(
        deadline: float | None,
        *,
        operation: str,
    ) -> None:
        if deadline is not None and time.monotonic() >= deadline:
            raise IngressWriteDisabledError(f"{operation} deadline expired")

    def _acquire_mutex_before_deadline(
        self,
        deadline: float | None,
        *,
        allow_immediate: bool = False,
    ) -> bool:
        if deadline is None:
            self._mutex.acquire()
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return bool(allow_immediate and self._mutex.acquire(blocking=False))
        return self._mutex.acquire(
            timeout=min(remaining, threading.TIMEOUT_MAX),
        )

    def _replace_state_unlocked(self, fresh: IngressProjectionState) -> None:
        self._state.by_dedup_key = fresh.by_dedup_key
        self._state.by_acceptance_id = fresh.by_acceptance_id
        self._state.message_acceptance_winners = fresh.message_acceptance_winners
        self._state.message_transport_witnesses = fresh.message_transport_witnesses
        self._state.message_acceptance_denials = fresh.message_acceptance_denials
        self._state.message_denied_acceptances = fresh.message_denied_acceptances
        self._state.closed_employees = fresh.closed_employees
        self._state.cursor_sequence = fresh.cursor_sequence
        self._state.cursor_hash = fresh.cursor_hash
        self._rebuild_transport_message_index_unlocked()

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
        self._index_accepted_frame_unlocked(frame)
        self._wake_resolved_acceptance_waiters_unlocked()

    @staticmethod
    def _transport_message_key(
        *,
        tenant_key: str,
        agent_id: str,
        bot_principal_id: str,
        app_id: str,
        event_type: str,
        chat_id: str,
        message_id: str,
    ) -> MessageLogicalKey:
        return message_logical_key(
            tenant_key=tenant_key,
            agent_id=agent_id,
            bot_principal_id=bot_principal_id,
            app_id=app_id,
            event_type=event_type,
            chat_id=chat_id,
            message_id=message_id,
        )

    @classmethod
    def _transport_message_key_for_record(
        cls,
        record: IngressRecord,
    ) -> MessageLogicalKey | None:
        metadata = record.metadata
        try:
            return cls._transport_message_key(
                tenant_key=metadata.tenant_key,
                agent_id=metadata.agent_id,
                bot_principal_id=metadata.bot_principal_id,
                app_id=metadata.app_id,
                event_type=metadata.event_type,
                chat_id=metadata.chat_id,
                message_id=metadata.message_id,
            )
        except ValueError:
            # Internal/team envelopes and legacy records can carry raw-looking
            # coordinates.  They are deliberately outside this proof index.
            return None

    @staticmethod
    def _payload_matches_transport_metadata(
        metadata: EmployeeIngressMetadata,
        payload: EmployeeIngressPayload,
    ) -> bool:
        """Bind public message indexes to authenticated raw coordinates."""

        if metadata.event_type != "im.message.receive_v1":
            return False
        if len(payload.normalized_parts) != 1:
            return False
        part = payload.normalized_parts[0]
        if not isinstance(part, Mapping) or part.get("type") != "message":
            return False
        raw_chat_id = part.get("remote_chat_id")
        raw_message_id = part.get("remote_message_id")
        if (
            not isinstance(raw_chat_id, str)
            or not raw_chat_id.startswith("oc_")
            or not isinstance(raw_message_id, str)
            or not raw_message_id.startswith("om_")
        ):
            return False
        return (
            metadata.chat_id == "oc_" + hashlib.sha256(raw_chat_id.encode("utf-8")).hexdigest()
            and metadata.message_id == "om_" + hashlib.sha256(raw_message_id.encode("utf-8")).hexdigest()
        )

    def _acceptance_for_key_unlocked(
        self,
        key: tuple[str, ...],
    ) -> IngressAcceptance | None:
        acceptance_id = self._transport_message_index.get(key)
        record = self._state.by_acceptance_id.get(acceptance_id or "")
        if (
            record is None
            or self._transport_message_key_for_record(record) != key
            or record.transport_message_proof is not True
        ):
            return None
        return record.acceptance

    def _message_acceptance_outcome_unlocked(
        self,
        key: MessageLogicalKey,
        *,
        transport_key: tuple[str, ...] | None = None,
    ) -> MessageAcceptanceOutcome | None:
        denial = self._state.message_acceptance_denials.get(key)
        if denial is not None:
            return MessageAcceptanceOutcome(
                status="denied",
                acceptance=None,
                channel_generation=denial.channel_generation,
                connection_id=denial.connection_id,
            )
        acceptance_id = self._state.message_acceptance_winners.get(key)
        if (
            transport_key is not None
            and self._state.message_transport_witnesses.get(transport_key)
            != acceptance_id
        ):
            return None
        record = self._state.by_acceptance_id.get(acceptance_id or "")
        if record is None or record.transport_message_proof is not True:
            return None
        return MessageAcceptanceOutcome(
            status="accepted",
            acceptance=record.acceptance,
            channel_generation=(
                record.metadata.channel_generation
                if transport_key is None
                else int(transport_key[7])
            ),
            connection_id=(
                record.metadata.connection_id
                if transport_key is None
                else str(transport_key[8])
            ),
        )

    def _index_acceptance_unlocked(
        self,
        record: IngressRecord,
        payload: EmployeeIngressPayload | None = None,
    ) -> None:
        key = self._transport_message_key_for_record(record)
        if (
            key is None
            or record.transport_message_proof is not True
            or record.disposition is not None
            or (
                payload is not None
                and not self._payload_matches_transport_metadata(
                    record.metadata,
                    payload,
                )
            )
        ):
            return
        existing_id = self._transport_message_index.get(key)
        existing = self._state.by_acceptance_id.get(existing_id or "")
        if (
            existing is None
            or record.acceptance.journal_sequence
            < existing.acceptance.journal_sequence
        ):
            self._transport_message_index[key] = record.acceptance.acceptance_id
        self._acceptance_progress.set()

    def _wake_resolved_acceptance_waiters_unlocked(self) -> None:
        self._acceptance_progress.set()

    def _index_accepted_frame_unlocked(self, frame: TransactionFrame) -> None:
        for event in frame.events:
            if event.event_type != "employee.ingress.accepted":
                continue
            acceptance_id = event.payload.get("acceptance_id")
            record = self._state.by_acceptance_id.get(
                acceptance_id if isinstance(acceptance_id, str) else ""
            )
            if record is not None:
                self._index_acceptance_unlocked(record)

    def _rebuild_transport_message_index_unlocked(self) -> None:
        self._transport_message_index.clear()
        records = sorted(
            self._state.by_acceptance_id.values(),
            key=lambda record: (
                record.acceptance.journal_sequence,
                record.acceptance.acceptance_id,
            ),
        )
        for record in records:
            self._index_acceptance_unlocked(record)

    def _wake_all_acceptance_waiters_unlocked(self) -> None:
        self._acceptance_progress.set()

    def _publish_payload_unlocked(
        self,
        metadata: EmployeeIngressMetadata,
        payload: EmployeeIngressPayload,
    ) -> BlobRef:
        before_ids = set(self._blob_store.iter_blob_ids())
        try:
            blob_ref = self._blob_store.stage_and_publish(
                payload.canonical_bytes,
                _blob_labels(metadata),
                self._active_key_id,
            )
            self._verify_ref_and_payload(blob_ref, metadata, payload)
            return blob_ref
        except (BlobError, IngressBlobError) as exc:
            self._quarantine_new_blobs_unlocked(before_ids)
            raise IngressBlobError("ingress payload publication failed") from exc

    def _accept_denied_message_unlocked(
        self,
        metadata: EmployeeIngressMetadata,
        payload: EmployeeIngressPayload,
        *,
        request_id: str,
    ) -> EmployeeIngressAck:
        blob_ref = self._publish_payload_unlocked(metadata, payload)
        recorded_at = _utc_now()
        event = JournalEvent(
            event_type="employee.ingress.denied_acceptance",
            aggregate_id=metadata.dedup_key,
            payload={
                "metadata": metadata.to_dict(),
                "acceptance_id": f"acc_{uuid.uuid4().hex}",
                "accepted_at": recorded_at,
                "blob_ref": blob_ref.to_dict(),
                "transport_message_proof": True,
                "disposition_id": f"dsp_{uuid.uuid4().hex}",
                "reason_code": "handoff_unconfirmed",
                "recorded_at": recorded_at,
            },
        )
        self._commit_unlocked(metadata.dedup_key, event)
        record = self._state.by_dedup_key.get(metadata.dedup_key)
        if record is None or not record.terminal:
            raise IngressWriteDisabledError(
                "denied ingress acceptance projection was not applied"
            )
        return self._ack(record, metadata, request_id=request_id, duplicate=False)

    def _accept_invalid_transport_unlocked(
        self,
        metadata: EmployeeIngressMetadata,
        payload: EmployeeIngressPayload,
        *,
        request_id: str,
    ) -> EmployeeIngressAck:
        blob_ref = self._publish_payload_unlocked(metadata, payload)
        recorded_at = _utc_now()
        event = JournalEvent(
            event_type="employee.ingress.invalid_transport_acceptance",
            aggregate_id=metadata.dedup_key,
            payload={
                "metadata": metadata.to_dict(),
                "acceptance_id": f"acc_{uuid.uuid4().hex}",
                "accepted_at": recorded_at,
                "blob_ref": blob_ref.to_dict(),
                "transport_message_proof": False,
                "disposition_id": f"dsp_{uuid.uuid4().hex}",
                "reason_code": "invalid_transport_proof",
                "recorded_at": recorded_at,
            },
        )
        self._commit_unlocked(metadata.dedup_key, event)
        record = self._state.by_dedup_key.get(metadata.dedup_key)
        if record is None or not record.terminal:
            raise IngressWriteDisabledError(
                "invalid transport acceptance projection was not applied"
            )
        return self._ack(record, metadata, request_id=request_id, duplicate=False)

    def _anchor_redelivery_witness_unlocked(
        self,
        winner: IngressRecord,
        metadata: EmployeeIngressMetadata,
    ) -> None:
        transport_key = message_transport_key_from_metadata(metadata)
        existing = self._state.message_transport_witnesses.get(transport_key)
        if existing is not None:
            if existing != winner.acceptance.acceptance_id:
                raise IngressConflictError("message transport witness owner conflict")
            return
        event = JournalEvent(
            event_type="employee.ingress.message_redelivery_accepted",
            aggregate_id=winner.aggregate_id,
            payload={
                "acceptance_id": winner.acceptance.acceptance_id,
                "tenant_key": metadata.tenant_key,
                "agent_id": metadata.agent_id,
                "bot_principal_id": metadata.bot_principal_id,
                "app_id": metadata.app_id,
                "event_type": metadata.event_type,
                "chat_id": metadata.chat_id,
                "message_id": metadata.message_id,
                "channel_generation": metadata.channel_generation,
                "connection_id": metadata.connection_id,
                "witnessed_at": _utc_now(),
            },
        )
        self._commit_unlocked(winner.aggregate_id, event)

    def _commit_unlocked(
        self,
        aggregate_id: str,
        event: JournalEvent,
        *,
        deadline: float | None = None,
    ) -> CommitResult:
        self._ensure_before_deadline(deadline, operation="ingress commit")
        versions = (
            self._writer.get_aggregate_versions([aggregate_id])
            if deadline is None
            else self._writer.get_aggregate_versions(
                [aggregate_id],
                deadline=deadline,
            )
        )
        result = self._writer.commit(
            [event],
            versions,
            expected_head_sequence=self._state.cursor_sequence,
            expected_head_hash=self._state.cursor_hash or None,
            **({} if deadline is None else {"deadline": deadline}),
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
        if record.payload_tombstoned:
            return
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

    def _verify_logical_duplicate_unlocked(
        self,
        record: IngressRecord,
        metadata: EmployeeIngressMetadata,
        payload: EmployeeIngressPayload,
    ) -> None:
        """Prove identical message content across event-id/envelope aliases."""

        existing = record.metadata
        comparable_fields = (
            "tenant_key",
            "agent_id",
            "bot_principal_id",
            "app_id",
            "message_id",
            "event_type",
            "action_identity",
            "chat_id",
            "thread_root_message_id",
            "sender_principal_id",
            "attachment_count",
            "attachment_total_bytes",
        )
        if any(
            getattr(existing, field) != getattr(metadata, field)
            for field in comparable_fields
        ):
            raise IngressConflictError("durable employee ingress conflict")
        aliased_payload = EmployeeIngressPayload(
            schema_version=payload.schema_version,
            envelope_id=existing.envelope_id,
            normalized_parts=payload.normalized_parts,
            attachment_descriptors=payload.attachment_descriptors,
        )
        if (
            aliased_payload.payload_sha256 != existing.payload_sha256
            or aliased_payload.payload_sha256 != existing.semantic_digest
            or aliased_payload.canonical_size_bytes != existing.payload_size_bytes
        ):
            raise IngressConflictError("durable employee ingress content conflict")
        if record.payload_tombstoned:
            return
        try:
            self._verify_ref_and_payload(
                record.blob_ref,
                existing,
                aliased_payload,
            )
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
            raise IngressClosedError(
                "authenticated ingress payload is unavailable"
            ) from exc

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
        with self._shared_blob_mutex:
            try:
                new_ids = set(self._blob_store.iter_blob_ids()) - before_ids
            except BlobError:
                return
            new_ids.difference_update(self._retained_shared_blob_ids)
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

    @staticmethod
    def _validate_incoming_payload(
        metadata: EmployeeIngressMetadata,
        payload: EmployeeIngressPayload,
    ) -> None:
        if metadata.envelope_id != payload.envelope_id:
            raise ValueError("payload envelope does not match metadata")
        if metadata.payload_sha256 != payload.payload_sha256:
            raise ValueError("payload hash does not match metadata")
        if metadata.semantic_digest != payload.payload_sha256:
            raise ValueError("payload semantic digest does not match payload")
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
            request_envelope_id=metadata.envelope_id,
            request_dedup_key=metadata.dedup_key,
            request_semantic_digest=metadata.semantic_digest,
            acceptance=record.acceptance,
            agent_id=metadata.agent_id,
            app_id=metadata.app_id,
            channel_generation=metadata.channel_generation,
            connection_id=metadata.connection_id,
            semantic_digest=record.acceptance.semantic_digest,
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
