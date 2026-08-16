"""Durable Journal ledger for outbound A2A dispatch attempts.

The ledger is the only module in the A2A adapter allowed to turn plaintext
instructions or normalized remote payloads into Journal metadata.  Sensitive
bytes are durably encrypted in ``BlobStore`` before their references are
anchored in the Journal.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import TYPE_CHECKING

from ..journal.blob_store import BlobStore
from ..journal.writer import CommitState, JournalWriter
from ..remote.models import (
    RemoteAttemptPhase,
    RemoteDispatchRequest,
    RemoteObservation,
    RemoteProjection,
    RemoteSnapshot,
    RemoteTaskHandle,
    RemoteTaskState,
)
from ..remote.projection import (
    RemoteProjectionError,
    cancel_requested_event,
    executing_event,
    observation_event,
    prepared_event,
    project_event,
    rebuild_remote_projection,
    send_uncertain_event,
    task_bound_event,
)
from .codec import MAX_OBSERVATION_BYTES, NormalizedA2AObservation

if TYPE_CHECKING:
    from ..journal.frame import JournalEvent


class RemoteDispatchLedgerError(RuntimeError):
    """A durable A2A mapping or lifecycle invariant was rejected."""


class RemoteDispatchLedger:
    """Serialize A2A projection refresh, validation, and anchored commits."""

    def __init__(
        self,
        writer: JournalWriter,
        blobs: BlobStore,
        key_ref: str,
    ) -> None:
        if not isinstance(writer, JournalWriter):
            raise TypeError("writer must be JournalWriter")
        if not isinstance(blobs, BlobStore):
            raise TypeError("blobs must be BlobStore")
        if not isinstance(key_ref, str) or not key_ref:
            raise ValueError("key_ref must be a non-empty string")
        self._writer = writer
        self._blobs = blobs
        self._key_ref = key_ref

    def projection(self) -> RemoteProjection:
        """Return a fresh projection containing anchored events only."""

        with self._writer.transaction_guard():
            projection, _head_sequence, _head_hash = self._refresh_unlocked()
            return projection

    def snapshot(self, acceptance_id: str) -> RemoteSnapshot | None:
        """Look up one durable attempt, returning ``None`` when it is unknown."""

        if not isinstance(acceptance_id, str) or not acceptance_id:
            raise ValueError("acceptance_id must be a non-empty string")
        projection = self.projection()
        key = projection.by_acceptance_id.get(acceptance_id)
        return None if key is None else projection.by_key[key]

    def next_observation_sequence(self, handle: RemoteTaskHandle) -> int:
        """Return the next stable zero-based event position for one attempt."""

        with self._writer.transaction_guard():
            projection, _head_sequence, _head_hash = self._refresh_unlocked()
            snapshot = self._require_snapshot(projection, handle)
            return len(snapshot.observations)

    def prepare(self, request: RemoteDispatchRequest) -> RemoteTaskHandle:
        """Publish the instruction and anchor stable authority before any send."""

        if not isinstance(request, RemoteDispatchRequest):
            raise TypeError("request must be RemoteDispatchRequest")
        instruction = request.instruction.encode("utf-8")
        with self._writer.transaction_guard():
            projection, head_sequence, head_hash = self._refresh_unlocked()
            existing = self._snapshot_for_acceptance(projection, request.acceptance_id)
            if existing is not None:
                self._validate_idempotent_prepare(existing, request, instruction)
                return existing.handle
            if request.descriptor.tenant_key == "":  # defensive for future models
                raise RemoteDispatchLedgerError("remote dispatch authority is invalid")
            if (request.run_id, request.assignment_id, request.attempt_id) in projection.by_key:
                raise RemoteDispatchLedgerError("remote dispatch authority collision")

            try:
                instruction_ref = self._blobs.stage_and_publish(
                    instruction,
                    {
                        "acceptance_id": request.acceptance_id,
                        "kind": "a2a_remote_instruction",
                        "tenant_key": request.descriptor.tenant_key,
                    },
                    self._key_ref,
                )
            except Exception:
                raise RemoteDispatchLedgerError("remote instruction publication failed") from None
            event = prepared_event(request, instruction_ref)
            projected = self._commit_unlocked(
                projection,
                event,
                head_sequence=head_sequence,
                head_hash=head_hash,
            )
            snapshot = self._snapshot_for_acceptance(projected, request.acceptance_id)
            if snapshot is None:  # pragma: no cover - guarded by projection validation
                raise RemoteDispatchLedgerError("remote dispatch preparation was lost")
            return snapshot.handle

    def mark_executing(self, handle: RemoteTaskHandle) -> RemoteSnapshot:
        """Anchor EXECUTING immediately before the caller performs the send."""

        with self._writer.transaction_guard():
            projection, head_sequence, head_hash = self._refresh_unlocked()
            snapshot = self._require_snapshot(projection, handle)
            if snapshot.phase is not RemoteAttemptPhase.PREPARED:
                return snapshot
            projected = self._commit_unlocked(
                projection,
                executing_event(snapshot.handle),
                head_sequence=head_sequence,
                head_hash=head_hash,
            )
            return self._required_projected_snapshot(projected, handle.acceptance_id)

    def bind_task(self, handle: RemoteTaskHandle, task_id: str) -> RemoteTaskHandle:
        """Freeze the first server task ID; exact replays are idempotent."""

        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task_id must be a non-empty string")
        with self._writer.transaction_guard():
            projection, head_sequence, head_hash = self._refresh_unlocked()
            snapshot = self._require_snapshot(
                projection,
                handle,
                allow_caller_unbound_task=True,
            )
            if snapshot.handle.task_id:
                if snapshot.handle.task_id != task_id:
                    raise RemoteDispatchLedgerError("remote task ID drift")
                return snapshot.handle
            projected = self._commit_unlocked(
                projection,
                task_bound_event(snapshot.handle, task_id),
                head_sequence=head_sequence,
                head_hash=head_hash,
            )
            return self._required_projected_snapshot(
                projected,
                handle.acceptance_id,
            ).handle

    def record_observation(
        self,
        handle: RemoteTaskHandle,
        observation: NormalizedA2AObservation | RemoteObservation,
        payload: bytes | None = None,
        *,
        observed_state: RemoteTaskState | None = None,
        sequence: int | None = None,
    ) -> RemoteObservation:
        """Publish normalized bytes and anchor one remote observation.

        ``NormalizedA2AObservation`` supplies its own canonical payload.  A
        caller passing a domain ``RemoteObservation`` must also pass the exact
        payload bytes; any caller-provided Blob reference is ignored because
        this ledger owns publication.
        """

        with self._writer.transaction_guard():
            projection, head_sequence, head_hash = self._refresh_unlocked()
            snapshot = self._require_snapshot(projection, handle)
            candidate, payload_bytes = self._observation_without_ref(
                observation,
                payload,
                snapshot,
                observed_state=observed_state,
                sequence=sequence,
            )
            next_sequence = len(snapshot.observations)
            if sequence is not None and sequence > next_sequence:
                raise RemoteDispatchLedgerError("remote observation sequence gap")
            duplicate = next(
                (item for item in snapshot.observations if item.observation_id == candidate.observation_id),
                None,
            )
            if duplicate is not None:
                if (
                    isinstance(observation, NormalizedA2AObservation)
                    and observation.kind.value == "artifact"
                    and sequence is None
                ):
                    raise RemoteDispatchLedgerError("remote observation sequence collision")
                self._validate_idempotent_observation(
                    duplicate,
                    candidate,
                    payload_bytes,
                )
                if sequence is not None and (sequence >= next_sequence or snapshot.observations[sequence] != duplicate):
                    raise RemoteDispatchLedgerError("remote observation sequence collision")
                return duplicate
            if sequence is not None and sequence < next_sequence:
                raise RemoteDispatchLedgerError("remote observation sequence collision")

            # Validate authority and state before publishing a potentially
            # orphaned encrypted blob.  Revalidate the final BlobRef event too.
            self._validate_event(projection, observation_event(snapshot.handle, candidate))
            try:
                payload_ref = self._blobs.stage_and_publish(
                    payload_bytes,
                    {
                        "acceptance_id": handle.acceptance_id,
                        "kind": "a2a_remote_observation",
                        "tenant_key": handle.descriptor.tenant_key,
                    },
                    self._key_ref,
                )
            except Exception:
                raise RemoteDispatchLedgerError("remote observation publication failed") from None
            anchored_observation = replace(candidate, payload_ref=payload_ref)
            projected = self._commit_unlocked(
                projection,
                observation_event(snapshot.handle, anchored_observation),
                head_sequence=head_sequence,
                head_hash=head_hash,
            )
            result = self._required_projected_snapshot(
                projected,
                handle.acceptance_id,
            )
            return next(
                item for item in result.observations if item.observation_id == anchored_observation.observation_id
            )

    def mark_send_uncertain(self, handle: RemoteTaskHandle) -> RemoteSnapshot:
        """Persist an ambiguous send outcome without exception or remote detail."""

        with self._writer.transaction_guard():
            projection, head_sequence, head_hash = self._refresh_unlocked()
            snapshot = self._require_snapshot(projection, handle)
            if snapshot.phase is RemoteAttemptPhase.SEND_UNCERTAIN:
                return snapshot
            projected = self._commit_unlocked(
                projection,
                send_uncertain_event(snapshot.handle),
                head_sequence=head_sequence,
                head_hash=head_hash,
            )
            return self._required_projected_snapshot(projected, handle.acceptance_id)

    def request_cancel(self, handle: RemoteTaskHandle) -> RemoteSnapshot:
        """Anchor cancel intent before the caller invokes remote cancellation."""

        with self._writer.transaction_guard():
            projection, head_sequence, head_hash = self._refresh_unlocked()
            snapshot = self._require_snapshot(projection, handle)
            if snapshot.cancel_requested or snapshot.phase is RemoteAttemptPhase.TERMINAL:
                return snapshot
            projected = self._commit_unlocked(
                projection,
                cancel_requested_event(snapshot.handle),
                head_sequence=head_sequence,
                head_hash=head_hash,
            )
            return self._required_projected_snapshot(projected, handle.acceptance_id)

    def _refresh_unlocked(self) -> tuple[RemoteProjection, int, str]:
        try:
            anchor, frames = self._writer.committed_tail(1)
            last = self._writer.get_last_frame()
            if (last is None and anchor.sequence != 0) or (
                last is not None and (last.sequence != anchor.sequence or last.frame_hash != anchor.frame_hash)
            ):
                raise RemoteDispatchLedgerError("Journal is not fully anchored")
            projection = rebuild_remote_projection(frames)
        except RemoteDispatchLedgerError:
            raise
        except Exception:
            raise RemoteDispatchLedgerError("remote Journal replay failed") from None
        logical_hash = "" if anchor.sequence == 0 else anchor.frame_hash
        return projection, anchor.sequence, logical_hash

    def _commit_unlocked(
        self,
        projection: RemoteProjection,
        event: JournalEvent,
        *,
        head_sequence: int,
        head_hash: str,
    ) -> RemoteProjection:
        projected = self._validate_event(projection, event)
        try:
            result = self._writer.commit(
                (event,),
                self._writer.get_aggregate_versions((event.aggregate_id,)),
                expected_head_sequence=head_sequence,
                expected_head_hash=head_hash,
            )
        except Exception:
            raise RemoteDispatchLedgerError("remote Journal commit failed") from None
        if result.state is not CommitState.ANCHORED:
            raise RemoteDispatchLedgerError("remote Journal commit was not anchored")
        return projected

    @staticmethod
    def _validate_event(
        projection: RemoteProjection,
        event: JournalEvent,
    ) -> RemoteProjection:
        try:
            return project_event(projection, event)
        except (RemoteProjectionError, TypeError, ValueError):
            raise RemoteDispatchLedgerError("remote Journal transition rejected") from None

    @staticmethod
    def _snapshot_for_acceptance(
        projection: RemoteProjection,
        acceptance_id: str,
    ) -> RemoteSnapshot | None:
        key = projection.by_acceptance_id.get(acceptance_id)
        return None if key is None else projection.by_key[key]

    @classmethod
    def _required_projected_snapshot(
        cls,
        projection: RemoteProjection,
        acceptance_id: str,
    ) -> RemoteSnapshot:
        snapshot = cls._snapshot_for_acceptance(projection, acceptance_id)
        if snapshot is None:  # pragma: no cover - guarded by projection validation
            raise RemoteDispatchLedgerError("remote Journal transition was lost")
        return snapshot

    @classmethod
    def _require_snapshot(
        cls,
        projection: RemoteProjection,
        supplied: RemoteTaskHandle,
        *,
        allow_caller_unbound_task: bool = False,
    ) -> RemoteSnapshot:
        if not isinstance(supplied, RemoteTaskHandle):
            raise TypeError("handle must be RemoteTaskHandle")
        snapshot = cls._snapshot_for_acceptance(projection, supplied.acceptance_id)
        if snapshot is None:
            raise RemoteDispatchLedgerError("remote dispatch is unknown")
        expected = snapshot.handle
        stable_supplied = replace(supplied, task_id="")
        stable_expected = replace(expected, task_id="")
        if stable_supplied != stable_expected:
            raise RemoteDispatchLedgerError("remote dispatch authority drift")
        if supplied.task_id != expected.task_id and not (allow_caller_unbound_task and not supplied.task_id):
            raise RemoteDispatchLedgerError("remote task authority drift")
        return snapshot

    def _validate_idempotent_prepare(
        self,
        snapshot: RemoteSnapshot,
        request: RemoteDispatchRequest,
        instruction: bytes,
    ) -> None:
        handle = snapshot.handle
        if (
            handle.acceptance_id != request.acceptance_id
            or handle.run_id != request.run_id
            or handle.assignment_id != request.assignment_id
            or handle.attempt_id != request.attempt_id
            or handle.message_id != request.message_id
            or handle.context_id != request.context_id
            or handle.descriptor != request.descriptor
        ):
            raise RemoteDispatchLedgerError("remote dispatch idempotency conflict")
        try:
            matches = self._blobs.read(handle.instruction_ref) == instruction
        except Exception:
            raise RemoteDispatchLedgerError("remote instruction recovery failed") from None
        if not matches:
            raise RemoteDispatchLedgerError("remote dispatch idempotency conflict")

    @staticmethod
    def _observation_without_ref(
        observation: NormalizedA2AObservation | RemoteObservation,
        payload: bytes | None,
        snapshot: RemoteSnapshot,
        *,
        observed_state: RemoteTaskState | None,
        sequence: int | None,
    ) -> tuple[RemoteObservation, bytes]:
        if sequence is not None and (isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0):
            raise ValueError("sequence must be a non-negative integer")
        if isinstance(observation, NormalizedA2AObservation):
            canonical = observation.canonical_payload
            if payload is not None and payload != canonical:
                raise RemoteDispatchLedgerError("remote observation payload mismatch")
            payload_bytes = canonical
            state = snapshot.state if observed_state is None else observed_state
            try:
                candidate = observation.to_remote_observation(
                    snapshot.handle,
                    observed_state=state,
                    sequence=(len(snapshot.observations) if sequence is None else sequence),
                )
            except Exception:
                raise RemoteDispatchLedgerError("remote observation binding rejected") from None
        elif isinstance(observation, RemoteObservation):
            if payload is None:
                raise ValueError("payload is required for RemoteObservation")
            payload_bytes = payload
            candidate = replace(observation, payload_ref=None)
        else:
            raise TypeError("observation must be a normalized or domain observation")
        if not isinstance(payload_bytes, bytes):
            raise TypeError("payload must be bytes")
        if len(payload_bytes) > MAX_OBSERVATION_BYTES:
            raise RemoteDispatchLedgerError("remote observation payload is too large")
        if hashlib.sha256(payload_bytes).hexdigest() != candidate.payload_digest:
            raise RemoteDispatchLedgerError("remote observation payload digest mismatch")
        return candidate, payload_bytes

    def _validate_idempotent_observation(
        self,
        existing: RemoteObservation,
        candidate: RemoteObservation,
        payload: bytes,
    ) -> None:
        if replace(existing, payload_ref=None) != candidate or existing.payload_ref is None:
            raise RemoteDispatchLedgerError("remote observation idempotency conflict")
        try:
            matches = self._blobs.read(existing.payload_ref) == payload
        except Exception:
            raise RemoteDispatchLedgerError("remote observation recovery failed") from None
        if not matches:
            raise RemoteDispatchLedgerError("remote observation idempotency conflict")


__all__ = ["RemoteDispatchLedger", "RemoteDispatchLedgerError"]
