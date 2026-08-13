"""Durable encrypted outbox for main-Bot terminal warnings.

The outbox owns warning persistence only.  Callers inject the main-Bot
transport; employee Channel authority is deliberately outside this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections import deque
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
)
from concurrent.futures import (
    TimeoutError as FutureTimeoutError,
)
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol

from src.utils.path import canonicalize_user_home_path

from ..journal.anchor import AnchorCorruptionError
from ..journal.blob_store import (
    AesGcmEncryptionProvider,
    BlobError,
    BlobRef,
    BlobStore,
)
from ..journal.frame import GENESIS_HASH, JournalEvent, JournalIntegrityError
from ..journal.writer import (
    CommitState,
    JournalClosedError,
    JournalDeadlineExceededError,
    JournalWriter,
)

_PREPARED_EVENT = "main_bot.warning.prepared"
_EXECUTING_EVENT = "main_bot.warning.executing"
_COMMITTED_EVENT = "main_bot.warning.committed"
_ACTION_REQUIRED_EVENT = "main_bot.warning.action_required"
_WARNING_EVENT_PREFIX = "main_bot.warning."
_BLOB_LABELS = {"kind": "main_bot_warning", "schema": "1"}
_SCHEMA_VERSION = 1
_MAX_COORDINATE_LENGTH = 2048
_MAX_TEXT_LENGTH = 100_000


class MainBotWarningOutboxError(RuntimeError):
    """Base error for the main-Bot warning outbox."""


class MainBotWarningConflictError(MainBotWarningOutboxError):
    """One stable idempotency key was reused for different content."""


class MainBotWarningCorruptionError(MainBotWarningOutboxError):
    """Durable warning state cannot be authenticated or replayed."""


class MainBotWarningWriteDisabledError(MainBotWarningOutboxError):
    """A warning transition did not cross the durable anchor boundary."""


class MainBotWarningClosedError(MainBotWarningOutboxError):
    """The warning outbox is closed."""


class MainBotWarningRetryableDeliveryError(MainBotWarningOutboxError):
    """The transport outcome is absent or ambiguous and must be replayed."""


class MainBotWarningPermanentDeliveryError(MainBotWarningOutboxError):
    """An explicitly classified permanent transport rejection."""

    def __init__(self, error_code: str) -> None:
        self.error_code = _validate_error_code(error_code)
        super().__init__(self.error_code)


class MainBotWarningState(str, Enum):
    PREPARED = "prepared"
    EXECUTING = "executing"
    COMMITTED = "committed"
    ACTION_REQUIRED = "action_required"


class MainBotWarningTransport(Protocol):
    """Main-Bot-only delivery boundary supplied by the WS composition root."""

    main_app_id: str

    def send_warning(
        self,
        *,
        message_id: str,
        tenant_key: str,
        chat_id: str,
        text: str,
        idempotency_key: str,
    ) -> str: ...


class MainBotWarningKeyring(Protocol):
    """Minimal composition boundary for the existing encrypted data keys."""

    active_key_id: str

    def resolve(self, key_ref: str) -> bytes: ...


@dataclass(frozen=True, slots=True)
class MainBotWarningRecord:
    warning_id: str
    main_app_id: str
    message_id: str
    tenant_key: str
    chat_id: str
    text: str
    idempotency_key: str
    reply_in_thread: bool
    state: MainBotWarningState
    attempt: int
    created_at: float
    error_code: str = ""


@dataclass(frozen=True, slots=True)
class _WarningMetadata:
    warning_id: str
    blob_ref: BlobRef
    payload_sha256: str
    state: MainBotWarningState
    attempt: int
    created_at: float
    error_code: str = ""


@dataclass(frozen=True, slots=True)
class MainBotWarningDrainResult:
    pending_count: int
    attempted_warning_ids: tuple[str, ...]
    committed_warning_ids: tuple[str, ...]
    failed_warning_ids: tuple[str, ...]
    action_required_warning_ids: tuple[str, ...]


def _origin_digest(tenant_key: str, chat_id: str, message_id: str) -> str:
    tenant = _validate_string(
        tenant_key,
        "tenant_key",
        maximum=_MAX_COORDINATE_LENGTH,
    )
    chat = _validate_string(chat_id, "chat_id", maximum=_MAX_COORDINATE_LENGTH)
    message = _validate_string(
        message_id,
        "message_id",
        maximum=_MAX_COORDINATE_LENGTH,
    )
    return hashlib.sha256(f"{tenant}\0{chat}\0{message}".encode("utf-8")).hexdigest()


def main_bot_warning_id(
    tenant_key: str,
    chat_id: str,
    message_id: str,
) -> str:
    """Return one opaque aggregate ID for a canonical ingress origin."""

    digest = _origin_digest(tenant_key, chat_id, message_id)
    return f"mbw_{digest}"


def main_bot_warning_idempotency_key(
    tenant_key: str,
    chat_id: str,
    message_id: str,
) -> str:
    """Return the sole Feishu UUID allowed for an origin warning."""

    digest = _origin_digest(tenant_key, chat_id, message_id)
    return f"employee-warning-{digest[:32]}"


class MainBotWarningOutbox:
    """Persist warning intent before any main-Bot transport attempt."""

    def __init__(
        self,
        *,
        writer: JournalWriter,
        blob_store: BlobStore,
        active_key_id: str,
        main_app_id: str,
    ) -> None:
        if not isinstance(writer, JournalWriter):
            raise TypeError("writer must be JournalWriter")
        if not isinstance(blob_store, BlobStore):
            raise TypeError("blob_store must be BlobStore")
        _validate_string(active_key_id, "active_key_id", maximum=512)
        _validate_string(main_app_id, "main_app_id", maximum=512)
        self._writer = writer
        self._blob_store = blob_store
        self._active_key_id = active_key_id
        self._main_app_id = main_app_id
        self._lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        self._delivery_lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        self._delivery_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="main-bot-warning-delivery",
        )
        self._delivery_futures: dict[str, Future[str]] = {}
        self._pending_warning_ids: deque[str] = deque()
        self._records: dict[str, _WarningMetadata] = {}
        self._cursor_sequence = 0
        self._cursor_hash = ""
        self._closed = False
        self.rebuild_projection()

    @classmethod
    def from_keyring(
        cls,
        *,
        writer: JournalWriter,
        keyring: MainBotWarningKeyring,
        blob_root: str | Path,
        main_app_id: str,
    ) -> MainBotWarningOutbox:
        """Build the owned BlobStore from composition-provided durable material."""

        active_key_id = getattr(keyring, "active_key_id", None)
        resolve = getattr(keyring, "resolve", None)
        _validate_string(active_key_id, "active_key_id", maximum=512)
        if not callable(resolve):
            raise TypeError("keyring must provide resolve")
        store = BlobStore(
            canonicalize_user_home_path(blob_root),
            AesGcmEncryptionProvider(resolve),
        )
        try:
            return cls(
                writer=writer,
                blob_store=store,
                active_key_id=active_key_id,
                main_app_id=main_app_id,
            )
        except BaseException:
            store.close()
            raise

    @property
    def blob_store(self) -> BlobStore:
        return self._blob_store

    def close(self) -> None:
        """Close the owned payload store; the injected Journal writer stays owned by composition."""

        with self._delivery_lock:
            with self._lock:
                if self._closed:
                    return
                if self._delivery_futures:
                    self._synchronize_projection_unlocked()
                    for warning_id in tuple(self._delivery_futures):
                        metadata = self._records.get(warning_id)
                        if metadata is not None:
                            self._release_terminal_delivery_future_unlocked(
                                warning_id,
                                metadata.state,
                            )
                if self._delivery_futures:
                    raise MainBotWarningRetryableDeliveryError("main-Bot warning delivery is still in flight")
                self._closed = True
                self._blob_store.close()
            self._delivery_executor.shutdown(wait=True, cancel_futures=False)

    def prepare(
        self,
        *,
        message_id: str,
        tenant_key: str,
        chat_id: str,
        text: str,
        idempotency_key: str,
    ) -> MainBotWarningRecord:
        return self.enqueue(
            message_id=message_id,
            tenant_key=tenant_key,
            chat_id=chat_id,
            text=text,
            idempotency_key=idempotency_key,
        )

    def enqueue(
        self,
        *,
        message_id: str,
        tenant_key: str,
        chat_id: str,
        text: str,
        idempotency_key: str,
    ) -> MainBotWarningRecord:
        values = _validated_payload(
            main_app_id=self._main_app_id,
            message_id=message_id,
            tenant_key=tenant_key,
            chat_id=chat_id,
            text=text,
            idempotency_key=idempotency_key,
        )
        expected_idempotency_key = main_bot_warning_idempotency_key(
            tenant_key,
            chat_id,
            message_id,
        )
        if idempotency_key != expected_idempotency_key:
            raise ValueError("idempotency_key must match the canonical warning origin")
        warning_id = main_bot_warning_id(tenant_key, chat_id, message_id)
        encoded = _canonical_json(values)
        payload_sha256 = hashlib.sha256(encoded).hexdigest()
        with self._lock, self._writer.transaction_guard():
            self._ensure_open_unlocked()
            self._synchronize_projection_unlocked()
            existing = self._records.get(warning_id)
            if existing is not None:
                record = self._hydrate_unlocked(existing)
                if _record_payload(record) != values:
                    raise MainBotWarningConflictError("warning idempotency key conflicts with durable content")
                return record

            try:
                blob_ref = self._blob_store.stage_and_publish(
                    encoded,
                    _BLOB_LABELS,
                    self._active_key_id,
                )
                if self._blob_store.read(blob_ref) != encoded:
                    raise MainBotWarningCorruptionError("warning payload verification failed")
            except BlobError as exc:
                raise MainBotWarningCorruptionError("warning payload publication failed") from exc

            event = JournalEvent(
                event_type=_PREPARED_EVENT,
                aggregate_id=warning_id,
                payload={
                    "schema_version": _SCHEMA_VERSION,
                    "payload_sha256": payload_sha256,
                    "blob_ref": blob_ref.to_dict(),
                },
            )
            # Do not quarantine the blob synchronously when commit/anchor
            # outcome is unknown.  FileAnchor may have replaced its state just
            # before a directory-fsync exception; deleting the blob then would
            # corrupt an actually anchored PREPARED event.  A later projection
            # rebuild safely quarantines only blobs proven unreferenced by the
            # monotonic anchor.
            self._commit_event_unlocked(event, "warning PREPARED event")
            return self._hydrate_unlocked(self._records[warning_id])

    def mark_executing(
        self,
        warning_id: str,
        *,
        deadline: float | None = None,
    ) -> MainBotWarningRecord:
        """Anchor the delivery attempt before invoking the main-Bot transport."""

        _validate_warning_id(warning_id)
        with self._deadline_lock(self._lock, deadline, "state lock"):
            with self._writer.transaction_guard(deadline=deadline):
                self._ensure_open_unlocked()
                self._synchronize_projection_unlocked(deadline=deadline)
                metadata = self._required_metadata_unlocked(warning_id)
                if metadata.state in {
                    MainBotWarningState.EXECUTING,
                    MainBotWarningState.COMMITTED,
                }:
                    return self._hydrate_unlocked(metadata)
                if metadata.state is not MainBotWarningState.PREPARED:
                    raise MainBotWarningConflictError("warning is not ready for delivery")
                event = JournalEvent(
                    event_type=_EXECUTING_EVENT,
                    aggregate_id=warning_id,
                    payload={
                        "schema_version": _SCHEMA_VERSION,
                        "attempt": metadata.attempt + 1,
                    },
                )
                self._commit_event_unlocked(
                    event,
                    "warning EXECUTING event",
                    deadline=deadline,
                )
                return self._hydrate_unlocked(self._records[warning_id])

    def mark_committed(
        self,
        warning_id: str,
        *,
        receipt_id: str,
        deadline: float | None = None,
    ) -> MainBotWarningRecord:
        """Record authoritative transport success without persisting its receipt."""

        _validate_warning_id(warning_id)
        _validate_string(receipt_id, "receipt_id", maximum=_MAX_COORDINATE_LENGTH)
        with self._deadline_lock(self._lock, deadline, "state lock"):
            with self._writer.transaction_guard(deadline=deadline):
                self._ensure_open_unlocked()
                self._synchronize_projection_unlocked(deadline=deadline)
                metadata = self._required_metadata_unlocked(warning_id)
                if metadata.state is MainBotWarningState.COMMITTED:
                    return self._hydrate_unlocked(metadata)
                if metadata.state is not MainBotWarningState.EXECUTING:
                    raise MainBotWarningConflictError("warning delivery is not executing")
                event = JournalEvent(
                    event_type=_COMMITTED_EVENT,
                    aggregate_id=warning_id,
                    payload={
                        "schema_version": _SCHEMA_VERSION,
                        "attempt": metadata.attempt,
                        "receipt_sha256": hashlib.sha256(receipt_id.encode("utf-8")).hexdigest(),
                    },
                )
                self._commit_event_unlocked(
                    event,
                    "warning COMMITTED event",
                    deadline=deadline,
                )
                return self._hydrate_unlocked(self._records[warning_id])

    def mark_action_required(
        self,
        warning_id: str,
        *,
        error_code: str,
        deadline: float | None = None,
    ) -> MainBotWarningRecord:
        """Stop automatic retry only for an explicit permanent rejection."""

        _validate_warning_id(warning_id)
        code = _validate_error_code(error_code)
        with self._deadline_lock(self._lock, deadline, "state lock"):
            with self._writer.transaction_guard(deadline=deadline):
                self._ensure_open_unlocked()
                self._synchronize_projection_unlocked(deadline=deadline)
                metadata = self._required_metadata_unlocked(warning_id)
                if metadata.state is MainBotWarningState.ACTION_REQUIRED:
                    if metadata.error_code != code:
                        raise MainBotWarningConflictError("warning permanent error classification changed")
                    return self._hydrate_unlocked(metadata)
                if metadata.state is not MainBotWarningState.EXECUTING:
                    raise MainBotWarningConflictError("warning delivery is not executing")
                event = JournalEvent(
                    event_type=_ACTION_REQUIRED_EVENT,
                    aggregate_id=warning_id,
                    payload={
                        "schema_version": _SCHEMA_VERSION,
                        "attempt": metadata.attempt,
                        "error_code": code,
                    },
                )
                self._commit_event_unlocked(
                    event,
                    "warning ACTION_REQUIRED event",
                    deadline=deadline,
                )
                return self._hydrate_unlocked(self._records[warning_id])

    def attempt_delivery(
        self,
        warning_id: str,
        transport: MainBotWarningTransport,
        *,
        deadline: float | None = None,
    ) -> MainBotWarningRecord:
        """Attempt one warning delivery after its EXECUTING frame is anchored."""

        send_warning = getattr(transport, "send_warning", None)
        if not callable(send_warning):
            raise TypeError("transport must provide send_warning")
        if getattr(transport, "main_app_id", None) != self._main_app_id:
            raise MainBotWarningConflictError("main-Bot warning transport app authority changed")
        deadline = self._validate_deadline(deadline)
        with self._deadline_lock(self._delivery_lock, deadline, "delivery lock"):
            record = self.mark_executing(warning_id, deadline=deadline)
            if record.state in {
                MainBotWarningState.COMMITTED,
                MainBotWarningState.ACTION_REQUIRED,
            }:
                self._release_terminal_delivery_future_unlocked(
                    warning_id,
                    record.state,
                )
                return record
            future = self._delivery_futures.get(warning_id)
            if future is None:
                if self._delivery_futures:
                    raise MainBotWarningRetryableDeliveryError("another main-Bot warning delivery is still in flight")
                self._remaining(deadline)
                future = self._delivery_executor.submit(
                    send_warning,
                    message_id=record.message_id,
                    tenant_key=record.tenant_key,
                    chat_id=record.chat_id,
                    text=record.text,
                    idempotency_key=record.idempotency_key,
                )
                self._delivery_futures[warning_id] = future
            try:
                remaining = self._remaining(deadline)
                receipt_id = future.result(timeout=remaining)
            except FutureTimeoutError as exc:
                if future.done():
                    try:
                        receipt_id = future.result()
                    except MainBotWarningPermanentDeliveryError as permanent:
                        result = self.mark_action_required(
                            warning_id,
                            error_code=permanent.error_code,
                            deadline=deadline,
                        )
                        self._delivery_futures.pop(warning_id, None)
                        return result
                    except BaseException:
                        self._delivery_futures.pop(warning_id, None)
                        raise
                else:
                    raise MainBotWarningRetryableDeliveryError("main-Bot warning delivery deadline exceeded") from exc
            except MainBotWarningPermanentDeliveryError as exc:
                result = self.mark_action_required(
                    warning_id,
                    error_code=exc.error_code,
                    deadline=deadline,
                )
                self._delivery_futures.pop(warning_id, None)
                return result
            except BaseException:
                self._delivery_futures.pop(warning_id, None)
                raise
            if not isinstance(receipt_id, str) or not receipt_id:
                self._delivery_futures.pop(warning_id, None)
                raise MainBotWarningRetryableDeliveryError("main-Bot warning transport returned an invalid receipt")
            result = self.mark_committed(
                warning_id,
                receipt_id=receipt_id,
                deadline=deadline,
            )
            self._delivery_futures.pop(warning_id, None)
            return result

    def _release_terminal_delivery_future_unlocked(
        self,
        warning_id: str,
        state: MainBotWarningState,
    ) -> None:
        """Forget only a completed future whose terminal state is anchored."""

        if state not in {
            MainBotWarningState.COMMITTED,
            MainBotWarningState.ACTION_REQUIRED,
        }:
            return
        future = self._delivery_futures.get(warning_id)
        if future is not None and future.done():
            self._delivery_futures.pop(warning_id, None)

    def pending_records(
        self,
        *,
        deadline: float | None = None,
    ) -> tuple[MainBotWarningRecord, ...]:
        """Return durable PREPARED/EXECUTING warnings in stable FIFO order."""

        deadline = self._validate_deadline(deadline)
        with self._deadline_lock(self._lock, deadline, "state lock"):
            self._ensure_open_unlocked()
            self._synchronize_projection_unlocked(deadline=deadline)
            records: list[MainBotWarningRecord] = []
            for metadata in sorted(
                (
                    value
                    for value in self._records.values()
                    if value.state
                    in {
                        MainBotWarningState.PREPARED,
                        MainBotWarningState.EXECUTING,
                    }
                ),
                key=lambda value: (value.created_at, value.warning_id),
            ):
                self._require_deadline(deadline, "payload hydration")
                records.append(self._hydrate_unlocked(metadata))
                self._require_deadline(deadline, "payload hydration")
            return tuple(records)

    def recover_pending(
        self,
        transport: MainBotWarningTransport,
        *,
        max_items: int = 100,
        deadline: float | None = None,
    ) -> MainBotWarningDrainResult:
        """Replay a bounded rotating batch; ambiguous outcomes remain recoverable."""

        if isinstance(max_items, bool) or not isinstance(max_items, int) or not 1 <= max_items <= 1000:
            raise ValueError("max_items must be between 1 and 1000")
        deadline = self._validate_deadline(deadline)
        with self._deadline_lock(self._delivery_lock, deadline, "delivery lock"):
            pending = self.pending_records(deadline=deadline)
            live_by_id = {record.warning_id: record for record in pending}
            retained: deque[str] = deque()
            queued_ids: set[str] = set()
            for warning_id in self._pending_warning_ids:
                if warning_id in live_by_id and warning_id not in queued_ids:
                    retained.append(warning_id)
                    queued_ids.add(warning_id)
            for record in pending:
                if record.warning_id not in queued_ids:
                    retained.append(record.warning_id)
                    queued_ids.add(record.warning_id)
            self._pending_warning_ids = retained

            attempted: list[str] = []
            committed: list[str] = []
            failed: list[str] = []
            action_required: list[str] = []
            attempt_limit = min(max_items, len(self._pending_warning_ids))
            for _attempt in range(attempt_limit):
                warning_id = self._pending_warning_ids.popleft()
                attempted.append(warning_id)
                try:
                    result = self.attempt_delivery(
                        warning_id,
                        transport,
                        deadline=deadline,
                    )
                except (
                    ConnectionError,
                    OSError,
                    TimeoutError,
                    MainBotWarningRetryableDeliveryError,
                ):
                    failed.append(warning_id)
                    self._pending_warning_ids.append(warning_id)
                    continue
                except BaseException:
                    self._pending_warning_ids.appendleft(warning_id)
                    raise
                if result.state is MainBotWarningState.COMMITTED:
                    committed.append(warning_id)
                elif result.state is MainBotWarningState.ACTION_REQUIRED:
                    action_required.append(warning_id)
                else:  # pragma: no cover - defensive state-machine fence
                    self._pending_warning_ids.appendleft(warning_id)
                    raise MainBotWarningCorruptionError("warning delivery returned a non-terminal state")
            return MainBotWarningDrainResult(
                pending_count=len(pending),
                attempted_warning_ids=tuple(attempted),
                committed_warning_ids=tuple(committed),
                failed_warning_ids=tuple(failed),
                action_required_warning_ids=tuple(action_required),
            )

    def deliver_pending(
        self,
        transport: MainBotWarningTransport,
        *,
        max_items: int = 100,
        deadline: float | None = None,
    ) -> MainBotWarningDrainResult:
        """Alias for lifecycle code that drains without a restart."""

        return self.recover_pending(
            transport,
            max_items=max_items,
            deadline=deadline,
        )

    def rebuild_projection(
        self,
        *,
        deadline: float | None = None,
    ) -> tuple[MainBotWarningRecord, ...]:
        deadline = self._validate_deadline(deadline)
        with self._deadline_lock(self._lock, deadline, "state lock"):
            with self._writer.transaction_guard(deadline=deadline):
                return self._rebuild_projection_unlocked(deadline=deadline)

    def _rebuild_projection_unlocked(
        self,
        *,
        deadline: float | None,
    ) -> tuple[MainBotWarningRecord, ...]:
        self._ensure_open_unlocked()
        self._require_deadline(deadline, "projection rebuild")
        try:
            anchor = self._writer.anchor.read()
        except (AnchorCorruptionError, OSError) as exc:
            raise MainBotWarningCorruptionError("warning projection cannot read Journal anchor") from exc
        records: dict[str, _WarningMetadata] = {}
        last_sequence = 0
        last_hash = GENESIS_HASH
        try:
            for frame in self._writer.replay(deadline=deadline):
                self._require_deadline(deadline, "projection replay")
                if frame.sequence > anchor.sequence:
                    break
                for event in frame.events:
                    self._require_deadline(deadline, "projection replay")
                    if event.event_type.startswith(_WARNING_EVENT_PREFIX):
                        self._apply_event_to(event, records)
                last_sequence = frame.sequence
                last_hash = frame.frame_hash
        except MainBotWarningCorruptionError:
            raise
        except JournalDeadlineExceededError as exc:
            raise MainBotWarningRetryableDeliveryError("main-Bot warning projection deadline exceeded") from exc
        except (
            JournalClosedError,
            JournalIntegrityError,
            OSError,
            TimeoutError,
        ) as exc:
            raise MainBotWarningCorruptionError("warning projection cannot replay Journal") from exc
        if last_sequence != anchor.sequence or last_hash != anchor.frame_hash:
            raise MainBotWarningCorruptionError("warning projection cannot verify Journal anchor")
        self._records = records
        self._cursor_sequence = last_sequence
        self._cursor_hash = "" if last_sequence == 0 else last_hash
        hydrated_records: list[MainBotWarningRecord] = []
        for metadata in sorted(
            records.values(),
            key=lambda item: (item.created_at, item.warning_id),
        ):
            self._require_deadline(deadline, "projection hydration")
            hydrated_records.append(self._hydrate_unlocked(metadata))
        # Blob hygiene is maintenance, not part of the close-time safety
        # proof. A finite caller deadline must not be consumed scanning
        # unrelated historical blobs.
        if deadline is None:
            self._quarantine_unreferenced_blobs_unlocked()
        return tuple(hydrated_records)

    def _synchronize_projection_unlocked(
        self,
        *,
        deadline: float | None = None,
    ) -> None:
        self._require_deadline(deadline, "projection synchronization")
        try:
            anchor, frames = self._writer.committed_tail(
                self._cursor_sequence + 1,
                deadline=deadline,
            )
        except JournalDeadlineExceededError as exc:
            raise MainBotWarningRetryableDeliveryError(
                "main-Bot warning projection deadline exceeded"
            ) from exc
        except (AnchorCorruptionError, JournalIntegrityError, OSError) as exc:
            raise MainBotWarningCorruptionError("warning projection cannot read Journal tail") from exc
        cursor_hash = "" if anchor.sequence == 0 else anchor.frame_hash
        if (self._cursor_sequence, self._cursor_hash) == (
            anchor.sequence,
            cursor_hash,
        ):
            return
        if self._cursor_sequence > anchor.sequence or not frames:
            raise MainBotWarningCorruptionError("warning projection cursor is ahead of Journal anchor")
        expected_sequence = self._cursor_sequence + 1
        expected_previous_hash = self._cursor_hash or GENESIS_HASH
        for frame in frames:
            self._require_deadline(deadline, "projection synchronization")
            if frame.sequence != expected_sequence or frame.previous_hash != expected_previous_hash:
                raise MainBotWarningCorruptionError("warning projection Journal tail is discontinuous")
            for event in frame.events:
                if event.event_type.startswith(_WARNING_EVENT_PREFIX):
                    self._apply_event_unlocked(event)
            expected_sequence += 1
            expected_previous_hash = frame.frame_hash
        if frames[-1].sequence != anchor.sequence or frames[-1].frame_hash != anchor.frame_hash:
            raise MainBotWarningCorruptionError("warning projection cannot verify Journal anchor")
        self._cursor_sequence = anchor.sequence
        self._cursor_hash = cursor_hash

    def _apply_event_unlocked(self, event: JournalEvent) -> None:
        self._apply_event_to(event, self._records)

    @staticmethod
    def _apply_event_to(
        event: JournalEvent,
        records: dict[str, _WarningMetadata],
    ) -> None:
        payload = event.payload
        if event.event_type == _PREPARED_EVENT:
            if event.aggregate_id in records:
                raise MainBotWarningCorruptionError("duplicate warning PREPARED event")
            if set(payload) != {"schema_version", "payload_sha256", "blob_ref"}:
                raise MainBotWarningCorruptionError("invalid warning PREPARED payload")
            if payload.get("schema_version") != _SCHEMA_VERSION:
                raise MainBotWarningCorruptionError("unsupported warning schema")
            payload_sha256 = payload.get("payload_sha256")
            if not _is_sha256(payload_sha256):
                raise MainBotWarningCorruptionError("invalid warning payload hash")
            try:
                blob_ref = BlobRef.from_dict(payload["blob_ref"])
            except (KeyError, TypeError, ValueError) as exc:
                raise MainBotWarningCorruptionError("invalid warning payload reference") from exc
            if dict(blob_ref.labels or {}) != _BLOB_LABELS:
                raise MainBotWarningCorruptionError("warning blob labels mismatch")
            if blob_ref.payload_hash != payload_sha256:
                raise MainBotWarningCorruptionError("warning payload hash mismatch")
            records[event.aggregate_id] = _WarningMetadata(
                warning_id=event.aggregate_id,
                blob_ref=blob_ref,
                payload_sha256=payload_sha256,
                state=MainBotWarningState.PREPARED,
                attempt=0,
                created_at=event.timestamp,
            )
            return

        current = records.get(event.aggregate_id)
        if current is None:
            raise MainBotWarningCorruptionError("warning transition has no PREPARED event")
        if event.event_type == _EXECUTING_EVENT:
            if set(payload) != {"schema_version", "attempt"}:
                raise MainBotWarningCorruptionError("invalid warning EXECUTING payload")
            attempt = _validated_attempt(payload, current.attempt + 1)
            if current.state is not MainBotWarningState.PREPARED:
                raise MainBotWarningCorruptionError("invalid warning EXECUTING transition")
            records[event.aggregate_id] = replace(
                current,
                state=MainBotWarningState.EXECUTING,
                attempt=attempt,
                error_code="",
            )
            return
        if event.event_type == _COMMITTED_EVENT:
            if set(payload) != {
                "schema_version",
                "attempt",
                "receipt_sha256",
            }:
                raise MainBotWarningCorruptionError("invalid warning COMMITTED payload")
            _validated_attempt(payload, current.attempt)
            if current.state is not MainBotWarningState.EXECUTING:
                raise MainBotWarningCorruptionError("invalid warning COMMITTED transition")
            if not _is_sha256(payload.get("receipt_sha256")):
                raise MainBotWarningCorruptionError("invalid warning receipt hash")
            records[event.aggregate_id] = replace(
                current,
                state=MainBotWarningState.COMMITTED,
            )
            return
        if event.event_type == _ACTION_REQUIRED_EVENT:
            if set(payload) != {
                "schema_version",
                "attempt",
                "error_code",
            }:
                raise MainBotWarningCorruptionError("invalid warning ACTION_REQUIRED payload")
            _validated_attempt(payload, current.attempt)
            if current.state is not MainBotWarningState.EXECUTING:
                raise MainBotWarningCorruptionError("invalid warning ACTION_REQUIRED transition")
            try:
                error_code = _validate_error_code(payload.get("error_code"))
            except ValueError as exc:
                raise MainBotWarningCorruptionError("invalid warning permanent error code") from exc
            records[event.aggregate_id] = replace(
                current,
                state=MainBotWarningState.ACTION_REQUIRED,
                error_code=error_code,
            )
            return
        raise MainBotWarningCorruptionError("unknown warning outbox event")

    def _commit_event_unlocked(
        self,
        event: JournalEvent,
        label: str,
        *,
        deadline: float | None = None,
    ) -> None:
        try:
            result = self._writer.commit(
                (event,),
                self._writer.get_aggregate_versions(
                    (event.aggregate_id,),
                    deadline=deadline,
                ),
                expected_head_sequence=self._cursor_sequence,
                expected_head_hash=self._cursor_hash,
                deadline=deadline,
            )
        except JournalDeadlineExceededError as exc:
            raise MainBotWarningRetryableDeliveryError(f"{label} deadline exceeded") from exc
        except (
            JournalClosedError,
            JournalIntegrityError,
            OSError,
            TimeoutError,
        ) as exc:
            raise MainBotWarningWriteDisabledError(f"{label} could not be durably anchored") from exc
        if result.state is not CommitState.ANCHORED:
            raise MainBotWarningWriteDisabledError(f"{label} was not anchored")
        self._apply_event_unlocked(event)
        self._cursor_sequence = result.frame.sequence
        self._cursor_hash = result.frame.frame_hash

    @staticmethod
    def _validate_deadline(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(float(deadline)):
            raise ValueError("deadline must be a finite monotonic timestamp")
        return float(deadline)

    @classmethod
    def _remaining(cls, deadline: float | None) -> float | None:
        value = cls._validate_deadline(deadline)
        if value is None:
            return None
        remaining = value - time.monotonic()
        if remaining <= 0:
            raise MainBotWarningRetryableDeliveryError("main-Bot warning delivery deadline exceeded")
        return min(remaining, threading.TIMEOUT_MAX)

    @classmethod
    def _require_deadline(cls, deadline: float | None, label: str) -> None:
        value = cls._validate_deadline(deadline)
        if value is not None and time.monotonic() >= value:
            raise MainBotWarningRetryableDeliveryError(f"main-Bot warning {label} deadline exceeded")

    @classmethod
    @contextmanager
    def _deadline_lock(
        cls,
        lock: threading.RLock,
        deadline: float | None,
        label: str,
    ) -> Iterator[None]:
        value = cls._validate_deadline(deadline)
        if value is None:
            with lock:
                yield
            return
        remaining = value - time.monotonic()
        if remaining <= 0 or not lock.acquire(timeout=min(remaining, threading.TIMEOUT_MAX)):
            raise MainBotWarningRetryableDeliveryError(f"main-Bot warning {label} deadline exceeded")
        try:
            yield
        finally:
            lock.release()

    def _required_metadata_unlocked(self, warning_id: str) -> _WarningMetadata:
        try:
            return self._records[warning_id]
        except KeyError as exc:
            raise KeyError(warning_id) from exc

    def _hydrate_unlocked(self, metadata: _WarningMetadata) -> MainBotWarningRecord:
        try:
            raw = self._blob_store.read(metadata.blob_ref)
            if hashlib.sha256(raw).hexdigest() != metadata.payload_sha256:
                raise MainBotWarningCorruptionError("warning payload hash mismatch")
            decoded = json.loads(raw)
            values = _validated_payload_mapping(decoded)
        except MainBotWarningCorruptionError:
            raise
        except (BlobError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            raise MainBotWarningCorruptionError("authenticated warning payload is unavailable") from exc
        expected_warning_id = main_bot_warning_id(
            values["tenant_key"],
            values["chat_id"],
            values["message_id"],
        )
        if metadata.warning_id != expected_warning_id:
            raise MainBotWarningCorruptionError("warning aggregate does not match encrypted origin")
        if values["main_app_id"] != self._main_app_id:
            raise MainBotWarningCorruptionError("warning main app authority does not match this runtime")
        return MainBotWarningRecord(
            warning_id=metadata.warning_id,
            **values,
            state=metadata.state,
            attempt=metadata.attempt,
            created_at=metadata.created_at,
            error_code=metadata.error_code,
        )

    def _ensure_open_unlocked(self) -> None:
        if self._closed or self._blob_store.closed:
            raise MainBotWarningClosedError("main-Bot warning outbox is closed")

    def _quarantine_unreferenced_blobs_unlocked(self) -> None:
        live_ids = {metadata.blob_ref.blob_id for metadata in self._records.values()}
        try:
            orphan_ids = set(self._blob_store.iter_blob_ids()) - live_ids
        except BlobError:
            return
        for blob_id in orphan_ids:
            try:
                self._blob_store.quarantine_blob(blob_id)
            except BlobError:
                continue


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _validate_string(value: object, name: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"{name} must be a non-empty bounded string")
    return value


def _validated_payload(
    *,
    main_app_id: object,
    message_id: object,
    tenant_key: object,
    chat_id: object,
    text: object,
    idempotency_key: object,
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "main_app_id": _validate_string(
            main_app_id,
            "main_app_id",
            maximum=512,
        ),
        "message_id": _validate_string(
            message_id,
            "message_id",
            maximum=_MAX_COORDINATE_LENGTH,
        ),
        "tenant_key": _validate_string(
            tenant_key,
            "tenant_key",
            maximum=_MAX_COORDINATE_LENGTH,
        ),
        "chat_id": _validate_string(
            chat_id,
            "chat_id",
            maximum=_MAX_COORDINATE_LENGTH,
        ),
        "text": _validate_string(text, "text", maximum=_MAX_TEXT_LENGTH),
        "idempotency_key": _validate_string(
            idempotency_key,
            "idempotency_key",
            maximum=_MAX_COORDINATE_LENGTH,
        ),
        # Warning replies deliberately retain legacy non-threaded semantics.
        # Freezing this option keeps a crash retry byte-for-byte equivalent
        # even when deployment defaults change between attempts.
        "reply_in_thread": False,
    }
    expected_key = main_bot_warning_idempotency_key(
        values["tenant_key"],
        values["chat_id"],
        values["message_id"],
    )
    if values["idempotency_key"] != expected_key:
        raise ValueError("idempotency_key does not match warning origin")
    return values


def _validated_payload_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "main_app_id",
        "message_id",
        "tenant_key",
        "chat_id",
        "text",
        "idempotency_key",
        "reply_in_thread",
    }:
        raise MainBotWarningCorruptionError("invalid warning payload fields")
    if value["reply_in_thread"] is not False:
        raise MainBotWarningCorruptionError("warning reply_in_thread contract changed")
    return _validated_payload(
        main_app_id=value["main_app_id"],
        message_id=value["message_id"],
        tenant_key=value["tenant_key"],
        chat_id=value["chat_id"],
        text=value["text"],
        idempotency_key=value["idempotency_key"],
    )


def _record_payload(record: MainBotWarningRecord) -> dict[str, Any]:
    return {
        "main_app_id": record.main_app_id,
        "message_id": record.message_id,
        "tenant_key": record.tenant_key,
        "chat_id": record.chat_id,
        "text": record.text,
        "idempotency_key": record.idempotency_key,
        "reply_in_thread": record.reply_in_thread,
    }


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_error_code(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 64
        or any(not (character.isascii() and (character.isalnum() or character in "_-.:")) for character in value)
    ):
        raise ValueError("invalid permanent delivery error code")
    return value


def _validate_warning_id(value: object) -> str:
    if not isinstance(value, str) or not value.startswith("mbw_") or not _is_sha256(value.removeprefix("mbw_")):
        raise ValueError("invalid warning_id")
    return value


def _validated_attempt(payload: Mapping[str, Any], expected: int) -> int:
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise MainBotWarningCorruptionError("unsupported warning schema")
    attempt = payload.get("attempt")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt != expected:
        raise MainBotWarningCorruptionError("invalid warning attempt")
    return attempt


__all__ = [
    "MainBotWarningClosedError",
    "MainBotWarningConflictError",
    "MainBotWarningCorruptionError",
    "MainBotWarningDrainResult",
    "MainBotWarningKeyring",
    "MainBotWarningOutbox",
    "MainBotWarningOutboxError",
    "MainBotWarningPermanentDeliveryError",
    "MainBotWarningRecord",
    "MainBotWarningRetryableDeliveryError",
    "MainBotWarningState",
    "MainBotWarningTransport",
    "MainBotWarningWriteDisabledError",
    "main_bot_warning_id",
    "main_bot_warning_idempotency_key",
]
