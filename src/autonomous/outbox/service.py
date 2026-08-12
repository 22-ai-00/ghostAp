"""Encrypted Journal-backed employee Durable Outbox snapshot service."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Protocol

from src.utils.path import canonicalize_user_home_path

from ..journal.blob_store import AesGcmEncryptionProvider, BlobError, BlobRef, BlobStore
from ..journal.frame import GENESIS_HASH, JournalEvent
from ..journal.writer import CommitResult, CommitState, JournalWriter
from .models import (
    DeliveryEffectKind,
    DeliveryEffectState,
    EmployeeCardState,
    EmployeeOutboxBinding,
    EmployeeOutboxSnapshot,
    OutboxDeliveryEffect,
    advance_snapshot,
    employee_outbox_effect_id,
    employee_outbox_uuid,
)
from .projection import (
    OutboxProjectionState,
    OutboxRecord,
    OutboxSnapshotRecord,
    is_outbox_event,
    reduce_outbox_event,
)


class OutboxServiceError(RuntimeError):
    pass


class OutboxConflictError(OutboxServiceError):
    pass


class OutboxBlobError(OutboxServiceError):
    pass


class OutboxWriteDisabledError(OutboxServiceError):
    pass


class OutboxClosedError(OutboxServiceError):
    pass


class EmployeeKeyring(Protocol):
    active_key_id: str

    def resolve(self, key_ref: str) -> bytes: ...


class EmployeeOutboxService:
    """Publish immutable encrypted card snapshots before Journal visibility."""

    def __init__(
        self,
        *,
        writer: JournalWriter,
        blob_store: BlobStore,
        outbox_state: OutboxProjectionState,
        active_key_id: str,
    ) -> None:
        if not isinstance(writer, JournalWriter):
            raise TypeError("writer must be a JournalWriter")
        if not isinstance(blob_store, BlobStore):
            raise TypeError("blob_store must be a BlobStore")
        if not isinstance(outbox_state, OutboxProjectionState):
            raise TypeError("outbox_state must be OutboxProjectionState")
        if not isinstance(active_key_id, str) or not active_key_id:
            raise ValueError("active_key_id must be non-empty")
        self._writer = writer
        self._blob_store = blob_store
        self._state = outbox_state
        self._active_key_id = active_key_id
        self._mutex = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        self._admission_closed = False
        self._closed = False
        self.rebuild_projection()

    @classmethod
    def from_keyring(
        cls,
        *,
        writer: JournalWriter,
        outbox_state: OutboxProjectionState,
        keyring: EmployeeKeyring,
        blob_root: str | Path,
    ) -> EmployeeOutboxService:
        store = BlobStore(
            canonicalize_user_home_path(blob_root),
            AesGcmEncryptionProvider(keyring.resolve),
        )
        try:
            return cls(
                writer=writer,
                blob_store=store,
                outbox_state=outbox_state,
                active_key_id=keyring.active_key_id,
            )
        except BaseException:
            store.close()
            raise

    @property
    def state(self) -> OutboxProjectionState:
        return self._state

    @property
    def blob_store(self) -> BlobStore:
        return self._blob_store

    def stop_admission(self) -> None:
        with self._mutex:
            self._ensure_open_unlocked()
            self._admission_closed = True

    def close(self) -> None:
        with self._mutex:
            if self._closed:
                return
            self._closed = True
            self._blob_store.close()

    def append_snapshot(
        self,
        snapshot: EmployeeOutboxSnapshot,
    ) -> OutboxSnapshotRecord:
        if not isinstance(snapshot, EmployeeOutboxSnapshot):
            raise TypeError("snapshot must be EmployeeOutboxSnapshot")
        with self._mutex, self._writer.transaction_guard():
            self._ensure_open_unlocked()
            if self._admission_closed:
                raise OutboxClosedError("employee Outbox admission is closed")
            self._synchronize_projection_unlocked()
            employee_key = (snapshot.tenant_key, snapshot.agent_id)
            if employee_key in self._state.closed_employees:
                raise OutboxClosedError("employee Outbox is closed")
            current = self._state.by_outbox_id.get(snapshot.outbox_id)
            if current is not None and snapshot.version in current.snapshots:
                existing = current.snapshots[snapshot.version]
                try:
                    persisted = self._read_snapshot(current, existing)
                except (BlobError, OutboxBlobError, ValueError, json.JSONDecodeError) as exc:
                    self._state.closed_employees.add(employee_key)
                    raise OutboxClosedError("authenticated Outbox snapshot is unavailable") from exc
                if persisted != snapshot:
                    raise OutboxConflictError("durable employee Outbox conflict")
                return existing
            if current is None:
                if snapshot.version != 1 or snapshot.state is not EmployeeCardState.QUEUED:
                    raise OutboxConflictError("Outbox must start with queued version 1")
            else:
                try:
                    previous = self._read_snapshot(current, current.latest)
                    advance_snapshot(previous, snapshot)
                except ValueError as exc:
                    raise OutboxConflictError(str(exc)) from exc
                except (BlobError, OutboxBlobError, json.JSONDecodeError) as exc:
                    self._state.closed_employees.add(employee_key)
                    raise OutboxClosedError("authenticated Outbox snapshot is unavailable") from exc

            labels = _blob_labels(snapshot)
            before_ids = set(self._blob_store.iter_blob_ids())
            try:
                blob_ref = self._blob_store.stage_and_publish(
                    snapshot.canonical_bytes,
                    labels,
                    self._active_key_id,
                )
                self._verify_snapshot_blob(snapshot, blob_ref)
            except (BlobError, OutboxBlobError) as exc:
                self._quarantine_new_blobs_unlocked(before_ids)
                raise OutboxBlobError("Outbox snapshot publication failed") from exc

            event = JournalEvent(
                event_type="employee.outbox.snapshot_appended",
                aggregate_id=snapshot.outbox_id,
                payload={
                    "tenant_key": snapshot.tenant_key,
                    "agent_id": snapshot.agent_id,
                    "attempt_id": snapshot.attempt_id,
                    "chat_id": snapshot.chat_id,
                    "thread_root_message_id": snapshot.thread_root_message_id,
                    "version": snapshot.version,
                    "state": snapshot.state.value,
                    "progress_percent": snapshot.progress_percent,
                    "created_at": snapshot.created_at,
                    "terminal_version": snapshot.terminal_version,
                    "payload_sha256": snapshot.payload_sha256,
                    "blob_ref": blob_ref.to_dict(),
                },
            )
            try:
                self._commit_unlocked(snapshot.outbox_id, event)
            except Exception:
                self._quarantine_blob_unlocked(blob_ref)
                raise
            return self._state.by_outbox_id[snapshot.outbox_id].snapshots[snapshot.version]

    def get_snapshot(
        self,
        outbox_id: str,
        version: int | None = None,
    ) -> EmployeeOutboxSnapshot:
        with self._mutex:
            self._ensure_open_unlocked()
            self._synchronize_projection_unlocked()
            record = self._state.by_outbox_id.get(outbox_id)
            if record is None:
                raise KeyError(outbox_id)
            selected = record.latest_version if version is None else version
            metadata = record.snapshots.get(selected)
            if metadata is None or selected in record.tombstoned_versions:
                raise OutboxBlobError("Outbox snapshot is unavailable")
            return self._read_snapshot(record, metadata)

    def get_record(self, outbox_id: str) -> OutboxRecord:
        with self._mutex:
            self._ensure_open_unlocked()
            self._synchronize_projection_unlocked()
            record = self._state.by_outbox_id.get(outbox_id)
            if record is None:
                raise KeyError(outbox_id)
            return record

    def list_pending_delivery_records(self) -> tuple[OutboxRecord, ...]:
        """Return one immutable, stable-order view of cards needing delivery."""

        with self._mutex:
            self._ensure_open_unlocked()
            self._synchronize_projection_unlocked()
            return tuple(
                sorted(
                    (
                        record
                        for record in self._state.by_outbox_id.values()
                        if record.binding is None
                        or record.binding.bound_snapshot_version
                        < record.latest_version
                    ),
                    key=lambda record: (
                        record.latest.created_at,
                        record.outbox_id,
                    ),
                )
            )

    def record_collaboration_publication(
        self,
        *,
        outbox_id: str,
        team_run_id: str,
        assignment_id: str,
        causal_event_id: str,
    ) -> bool:
        """Anchor employee-Bot provenance for one visible team contribution."""

        if not all((outbox_id, team_run_id, assignment_id, causal_event_id)):
            raise ValueError("collaboration publication coordinates are required")
        with self._mutex, self._writer.transaction_guard():
            self._ensure_open_unlocked()
            self._synchronize_projection_unlocked()
            record = self._state.by_outbox_id.get(outbox_id)
            if record is None or record.binding is None:
                raise OutboxConflictError("collaboration publication is not delivered")
            snapshot = self._read_snapshot(record, record.latest)
            if not snapshot.state.terminal:
                raise OutboxConflictError("collaboration publication is not terminal")
            for frame in self._writer.replay():
                for existing in frame.events:
                    if existing.event_type != "employee.outbox.collaboration_published":
                        continue
                    if existing.payload.get("causal_event_id") != causal_event_id:
                        continue
                    return (
                        existing.aggregate_id == outbox_id
                        and existing.payload.get("team_run_id") == team_run_id
                        and existing.payload.get("assignment_id") == assignment_id
                    )
            event = JournalEvent(
                event_type="employee.outbox.collaboration_published",
                aggregate_id=outbox_id,
                payload={
                    "tenant_key": record.tenant_key,
                    "chat_id": record.chat_id,
                    "agent_id": record.agent_id,
                    "app_id": record.binding.app_id,
                    "generation": record.binding.generation,
                    "team_run_id": team_run_id,
                    "assignment_id": assignment_id,
                    "causal_event_id": causal_event_id,
                },
            )
            last = self._writer.get_last_frame()
            result = self._writer.commit(
                (event,),
                self._writer.get_aggregate_versions((outbox_id,)),
                expected_head_sequence=0 if last is None else last.sequence,
                expected_head_hash="" if last is None else last.frame_hash,
            )
            if result.state is not CommitState.ANCHORED:
                raise OutboxWriteDisabledError(
                    "collaboration publication was not anchored"
                )
            self._synchronize_projection_unlocked()
            return True

    def prepare_delivery(
        self,
        outbox_id: str,
        snapshot_version: int,
    ) -> OutboxDeliveryEffect:
        with self._mutex, self._writer.transaction_guard():
            self._ensure_open_unlocked()
            self._synchronize_projection_unlocked()
            record = self._state.by_outbox_id.get(outbox_id)
            if record is None:
                raise KeyError(outbox_id)
            if snapshot_version not in record.snapshots:
                raise KeyError(snapshot_version)
            if snapshot_version in record.tombstoned_versions:
                raise OutboxBlobError("Outbox delivery snapshot is tombstoned")
            active = sorted(
                (
                    effect
                    for effect in record.effects.values()
                    if effect.state
                    in {
                        DeliveryEffectState.PREPARED,
                        DeliveryEffectState.EXECUTING,
                    }
                ),
                key=lambda effect: (effect.snapshot_version, effect.attempt),
            )
            if active:
                return active[0]
            for effect in record.effects.values():
                if effect.snapshot_version == snapshot_version and effect.state in {
                    DeliveryEffectState.PREPARED,
                    DeliveryEffectState.EXECUTING,
                    DeliveryEffectState.COMMITTED,
                }:
                    return effect
            kind = DeliveryEffectKind.CREATE if record.binding is None else DeliveryEffectKind.PATCH
            attempts = [
                effect.attempt
                for effect in record.effects.values()
                if effect.snapshot_version == snapshot_version and effect.kind is kind
            ]
            attempt = max(attempts, default=0) + 1
            metadata = record.snapshots[snapshot_version]
            effect = OutboxDeliveryEffect(
                schema_version=1,
                effect_id=employee_outbox_effect_id(
                    outbox_id,
                    kind,
                    snapshot_version,
                    attempt,
                ),
                outbox_id=outbox_id,
                kind=kind,
                state=DeliveryEffectState.PREPARED,
                snapshot_version=snapshot_version,
                snapshot_sha256=metadata.payload_sha256,
                attempt=attempt,
                error_code="",
            )
            self._commit_unlocked(
                outbox_id,
                JournalEvent(
                    event_type="employee.outbox.effect_prepared",
                    aggregate_id=outbox_id,
                    payload={"effect": effect.to_dict()},
                ),
            )
            return self._state.by_outbox_id[outbox_id].effects[effect.effect_id]

    def mark_effect_executing(self, effect_id: str) -> OutboxDeliveryEffect:
        with self._mutex, self._writer.transaction_guard():
            self._ensure_open_unlocked()
            self._synchronize_projection_unlocked()
            record, effect = self._find_effect_unlocked(effect_id)
            if effect.state in {
                DeliveryEffectState.EXECUTING,
                DeliveryEffectState.COMMITTED,
            }:
                return effect
            if effect.state is not DeliveryEffectState.PREPARED:
                raise OutboxConflictError("Outbox effect cannot execute")
            self._commit_unlocked(
                record.outbox_id,
                JournalEvent(
                    event_type="employee.outbox.effect_executing",
                    aggregate_id=record.outbox_id,
                    payload={"effect_id": effect.effect_id},
                ),
            )
            return self._state.by_outbox_id[record.outbox_id].effects[effect.effect_id]

    def commit_delivery(
        self,
        effect_id: str,
        *,
        app_id: str,
        generation: int,
        connection_id: str,
        message_id: str,
    ) -> EmployeeOutboxBinding:
        with self._mutex, self._writer.transaction_guard():
            self._ensure_open_unlocked()
            self._synchronize_projection_unlocked()
            record, effect = self._find_effect_unlocked(effect_id)
            if effect.state is DeliveryEffectState.COMMITTED:
                if record.binding is None:
                    raise OutboxWriteDisabledError("committed delivery lacks binding")
                return record.binding
            if effect.state is not DeliveryEffectState.EXECUTING:
                raise OutboxConflictError("Outbox effect is not EXECUTING")
            if effect.kind is DeliveryEffectKind.PATCH:
                current = record.binding
                if current is None or current.message_id != message_id:
                    raise OutboxConflictError("Outbox patch message binding mismatch")
                if current.app_id != app_id:
                    raise OutboxConflictError("Outbox patch app binding mismatch")
            binding = EmployeeOutboxBinding(
                schema_version=1,
                outbox_id=record.outbox_id,
                stable_uuid=employee_outbox_uuid(record.outbox_id),
                app_id=app_id,
                generation=generation,
                connection_id=connection_id,
                message_id=message_id,
                bound_snapshot_version=effect.snapshot_version,
            )
            self._commit_unlocked(
                record.outbox_id,
                JournalEvent(
                    event_type="employee.outbox.delivery_committed",
                    aggregate_id=record.outbox_id,
                    payload={
                        "effect_id": effect.effect_id,
                        "binding": binding.to_dict(),
                    },
                ),
            )
            committed = self._state.by_outbox_id[record.outbox_id].binding
            if committed is None:
                raise OutboxWriteDisabledError("Outbox delivery binding was not applied")
            return committed

    def rebuild_projection(self) -> OutboxProjectionState:
        with self._mutex:
            self._ensure_open_unlocked()
            fresh = OutboxProjectionState()
            anchor = self._writer.anchor.read()
            anchored_hash = GENESIS_HASH
            for frame in self._writer.replay():
                if frame.sequence > anchor.sequence:
                    break
                for event in frame.events:
                    if is_outbox_event(event.event_type):
                        reduce_outbox_event(fresh, event)
                fresh.cursor_sequence = frame.sequence
                fresh.cursor_hash = frame.frame_hash
                anchored_hash = frame.frame_hash
            if anchored_hash != anchor.frame_hash:
                raise OutboxWriteDisabledError("Outbox projection cannot verify the Journal anchor")
            for record in fresh.by_outbox_id.values():
                for version, metadata in record.snapshots.items():
                    if version in record.tombstoned_versions:
                        continue
                    try:
                        self._read_snapshot(record, metadata)
                    except (BlobError, OutboxBlobError, ValueError, json.JSONDecodeError):
                        fresh.closed_employees.add(record.employee_key)
                        break
            self._replace_state_unlocked(fresh)
            self._quarantine_unreferenced_blobs_unlocked()
            return self._state

    def quarantine_unreferenced_blobs(self) -> int:
        with self._mutex, self._writer.transaction_guard():
            self._ensure_open_unlocked()
            self._synchronize_projection_unlocked()
            return self._quarantine_unreferenced_blobs_unlocked()

    def gc_superseded_snapshots(self) -> int:
        with self._mutex, self._writer.transaction_guard():
            self._ensure_open_unlocked()
            self._synchronize_projection_unlocked()
            present_blob_ids = set(self._blob_store.iter_blob_ids())
            collected = 0
            for outbox_id, record in tuple(self._state.by_outbox_id.items()):
                if (
                    not record.latest.state.terminal
                    or record.binding is None
                    or record.binding.bound_snapshot_version < record.latest_version
                    or any(
                        effect.state
                        in {
                            DeliveryEffectState.PREPARED,
                            DeliveryEffectState.EXECUTING,
                        }
                        for effect in record.effects.values()
                    )
                ):
                    continue
                for version, metadata in tuple(record.snapshots.items()):
                    current = self._state.by_outbox_id[outbox_id]
                    if version == current.latest_version:
                        continue
                    if version not in current.tombstoned_versions:
                        self._commit_unlocked(
                            outbox_id,
                            JournalEvent(
                                event_type="employee.outbox.snapshot_tombstoned",
                                aggregate_id=outbox_id,
                                payload={
                                    "version": version,
                                    "tombstoned_at": _utc_now(),
                                },
                            ),
                        )
                    blob_id = metadata.blob_ref.blob_id
                    if blob_id not in present_blob_ids:
                        continue
                    try:
                        self._blob_store.quarantine_blob(blob_id)
                    except BlobError:
                        continue
                    present_blob_ids.discard(blob_id)
                    collected += 1
            return collected

    def _read_snapshot(
        self,
        record: OutboxRecord,
        metadata: OutboxSnapshotRecord,
    ) -> EmployeeOutboxSnapshot:
        expected_labels = _record_blob_labels(record, metadata)
        if dict(metadata.blob_ref.labels or {}) != expected_labels:
            raise OutboxBlobError("Outbox blob labels do not match authority")
        raw = self._blob_store.read(metadata.blob_ref)
        snapshot = EmployeeOutboxSnapshot.from_dict(json.loads(raw))
        if (
            snapshot.outbox_id != record.outbox_id
            or snapshot.version != metadata.version
            or snapshot.payload_sha256 != metadata.payload_sha256
            or snapshot.canonical_bytes != raw
        ):
            raise OutboxBlobError("Outbox blob payload verification failed")
        return snapshot

    def _find_effect_unlocked(
        self,
        effect_id: str,
    ) -> tuple[OutboxRecord, OutboxDeliveryEffect]:
        for record in self._state.by_outbox_id.values():
            effect = record.effects.get(effect_id)
            if effect is not None:
                return record, effect
        raise KeyError(effect_id)

    def _verify_snapshot_blob(
        self,
        snapshot: EmployeeOutboxSnapshot,
        blob_ref: BlobRef,
    ) -> None:
        if dict(blob_ref.labels or {}) != _blob_labels(snapshot):
            raise OutboxBlobError("Outbox blob labels do not match snapshot")
        raw = self._blob_store.read(blob_ref)
        if raw != snapshot.canonical_bytes or blob_ref.payload_hash != snapshot.payload_sha256:
            raise OutboxBlobError("Outbox blob payload verification failed")

    def _ensure_open_unlocked(self) -> None:
        if self._closed or self._blob_store.closed:
            raise OutboxClosedError("employee Outbox service is closed")

    def _synchronize_projection_unlocked(self) -> None:
        anchor = self._writer.anchor.read()
        cursor_hash = "" if anchor.sequence == 0 else anchor.frame_hash
        if (self._state.cursor_sequence, self._state.cursor_hash) != (
            anchor.sequence,
            cursor_hash,
        ):
            self.rebuild_projection()

    def _replace_state_unlocked(self, fresh: OutboxProjectionState) -> None:
        self._state.by_outbox_id = fresh.by_outbox_id
        self._state.closed_employees = fresh.closed_employees
        self._state.cursor_sequence = fresh.cursor_sequence
        self._state.cursor_hash = fresh.cursor_hash

    def _commit_unlocked(self, aggregate_id: str, event: JournalEvent) -> CommitResult:
        result = self._writer.commit(
            [event],
            self._writer.get_aggregate_versions([aggregate_id]),
            expected_head_sequence=self._state.cursor_sequence,
            expected_head_hash=self._state.cursor_hash or None,
        )
        if result.state != CommitState.ANCHORED:
            raise OutboxWriteDisabledError("Outbox lifecycle event was not anchored")
        for committed_event in result.frame.events:
            if is_outbox_event(committed_event.event_type):
                reduce_outbox_event(self._state, committed_event)
        self._state.cursor_sequence = result.frame.sequence
        self._state.cursor_hash = result.frame.frame_hash
        return result

    def _quarantine_unreferenced_blobs_unlocked(self) -> int:
        live = {
            metadata.blob_ref.blob_id
            for record in self._state.by_outbox_id.values()
            for version, metadata in record.snapshots.items()
            if version not in record.tombstoned_versions
        }
        orphans = set(self._blob_store.iter_blob_ids()) - live
        for blob_id in orphans:
            self._blob_store.quarantine_blob(blob_id)
        return len(orphans)

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

    def _quarantine_blob_unlocked(self, blob_ref: BlobRef) -> None:
        try:
            self._blob_store.quarantine_blob(blob_ref.blob_id)
        except BlobError:
            pass


def _blob_labels(snapshot: EmployeeOutboxSnapshot) -> dict[str, str]:
    return {
        "schema": "employee-outbox-v1",
        "tenant": snapshot.tenant_key,
        "employee": snapshot.agent_id,
        "attempt_id": snapshot.attempt_id,
        "outbox_id": snapshot.outbox_id,
        "version": str(snapshot.version),
        "payload_sha256": snapshot.payload_sha256,
    }


def _record_blob_labels(
    record: OutboxRecord,
    metadata: OutboxSnapshotRecord,
) -> dict[str, str]:
    return {
        "schema": "employee-outbox-v1",
        "tenant": record.tenant_key,
        "employee": record.agent_id,
        "attempt_id": record.attempt_id,
        "outbox_id": record.outbox_id,
        "version": str(metadata.version),
        "payload_sha256": metadata.payload_sha256,
    }


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


__all__ = [
    "EmployeeOutboxService",
    "OutboxBlobError",
    "OutboxClosedError",
    "OutboxConflictError",
    "OutboxServiceError",
    "OutboxWriteDisabledError",
]
