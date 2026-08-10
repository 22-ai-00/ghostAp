"""Fenced single-writer journal with synchronous durability and anchoring."""

from __future__ import annotations

import fcntl
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol, Sequence

from .anchor import AnchorProvider, AnchorState
from .blob_store import BlobRef
from .frame import (
    GENESIS_HASH,
    IncompleteFrameError,
    JournalEvent,
    JournalIntegrityError,
    TransactionFrame,
    decode_frame,
)

JOURNAL_FILENAME = "journal.jsonl"
LOCK_FILENAME = "writer.lock"


class WriterLockError(RuntimeError):
    """Another process already owns the journal writer lock."""


class AnchorMismatchError(JournalIntegrityError):
    """The local journal and monotonic anchor disagree."""


class JournalClosedError(RuntimeError):
    """The writer is closed or write-disabled after an anchor failure."""


class CommitState(str, Enum):
    ANCHORED = "anchored"
    DURABLE_NOT_ANCHORED = "durable_not_anchored"


@dataclass(frozen=True)
class CommitResult:
    frame: TransactionFrame
    state: CommitState


class FileSystemOperations(Protocol):
    def fsync_file(self, file_or_fd: Any) -> None: ...

    def fsync_directory(self, directory: str | Path) -> None: ...


class DefaultFileSystemOperations:
    def fsync_file(self, file_or_fd: Any) -> None:
        fd = file_or_fd if isinstance(file_or_fd, int) else file_or_fd.fileno()
        os.fsync(fd)

    def fsync_directory(self, directory: str | Path) -> None:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _append_record(
    path: Path,
    record: bytes,
    fs_ops: FileSystemOperations,
) -> None:
    with open(path, "ab", buffering=0) as file:
        written = file.write(record)
        if written != len(record):
            raise OSError(
                f"short journal write: expected {len(record)} bytes, wrote {written}"
            )
        fs_ops.fsync_file(file)


class JournalWriter:
    """The sole append authority for one local autonomous journal."""

    def __init__(
        self,
        base_dir: str | Path,
        *,
        anchor: AnchorProvider,
        hmac_key: bytes,
        writer_epoch: int,
        fs_ops: FileSystemOperations | None = None,
        blob_ref_validator: Any = None,
    ) -> None:
        if not isinstance(hmac_key, bytes) or len(hmac_key) < 32:
            raise ValueError("journal hmac key must be at least 32 bytes")
        if (
            isinstance(writer_epoch, bool)
            or not isinstance(writer_epoch, int)
            or writer_epoch < 0
        ):
            raise ValueError("writer_epoch must be a non-negative integer")
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.base_dir.chmod(0o700)
        self.journal_path = self.base_dir / JOURNAL_FILENAME
        self.lock_path = self.base_dir / LOCK_FILENAME
        self.anchor = anchor
        self._hmac_key = hmac_key
        self._writer_epoch = writer_epoch
        self._fs_ops = fs_ops or DefaultFileSystemOperations()
        self._blob_ref_validator = blob_ref_validator
        self._mutex = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        # Serialize cross-domain projection refresh + commit without reusing the
        # writer's non-reentrant leaf lock. Public writer operations called in a
        # transaction continue to acquire ``_mutex`` independently.
        self._transaction_mutex = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        self._closed = False
        self._write_disabled = False
        self._lock_file = open(self.lock_path, "a+b")
        os.chmod(self.lock_path, 0o600)
        try:
            fcntl.flock(
                self._lock_file.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            self._lock_file.close()
            raise WriterLockError(f"journal already has a writer: {self.base_dir}") from exc
        try:
            self.journal_path.touch(mode=0o600, exist_ok=True)
            self.journal_path.chmod(0o600)
            self._frames = self._load_and_recover()
            self._sequence = self._frames[-1].sequence if self._frames else 0
            self._previous_hash = (
                self._frames[-1].frame_hash if self._frames else GENESIS_HASH
            )
            self._aggregate_versions = self._rebuild_aggregate_versions(self._frames)
            self._verify_anchor()
        except BaseException:
            self.close()
            raise

    @classmethod
    def open(
        cls,
        base_dir: str | Path,
        *,
        anchor: AnchorProvider,
        hmac_key: bytes,
        writer_epoch: int = 0,
        fs_ops: FileSystemOperations | None = None,
        blob_ref_validator: Any = None,
    ) -> JournalWriter:
        return cls(
            base_dir,
            anchor=anchor,
            hmac_key=hmac_key,
            writer_epoch=writer_epoch,
            fs_ops=fs_ops,
            blob_ref_validator=blob_ref_validator,
        )

    def __enter__(self) -> JournalWriter:
        return self

    @property
    def writer_epoch(self) -> int:
        """Return the immutable epoch stamped on frames committed by this writer."""

        return self._writer_epoch

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        mutex = getattr(self, "_mutex", None)
        if mutex is None:
            return
        with mutex:
            if getattr(self, "_closed", True):
                return
            self._closed = True
            lock_file = getattr(self, "_lock_file", None)
            if lock_file is not None and not lock_file.closed:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                finally:
                    lock_file.close()

    def _ensure_writable(self) -> None:
        if self._closed or self._write_disabled:
            raise JournalClosedError("journal writer is closed for writes")

    def _load_and_recover(self) -> list[TransactionFrame]:
        raw = self.journal_path.read_bytes()
        if not raw:
            return []
        anchored_before_recovery = self.anchor.read()
        physical_lines = raw.splitlines(keepends=True)
        nonempty_indexes = [
            index
            for index, physical in enumerate(physical_lines)
            if physical.strip()
        ]
        if not nonempty_indexes:
            with open(self.journal_path, "r+b") as file:
                file.truncate(0)
                file.flush()
                self._fs_ops.fsync_file(file)
            self._fs_ops.fsync_directory(self.base_dir)
            return []
        last_nonempty_index = nonempty_indexes[-1]
        frames: list[TransactionFrame] = []
        valid_length = 0
        previous_hash = GENESIS_HASH
        expected_sequence = 1
        tail_incomplete = False
        for index, physical in enumerate(physical_lines):
            if not physical.strip():
                if index < last_nonempty_index:
                    raise JournalIntegrityError("blank record before journal tail")
                tail_incomplete = True
                break
            is_tail = index == last_nonempty_index
            try:
                frame = decode_frame(physical, self._hmac_key)
            except IncompleteFrameError:
                if not is_tail:
                    raise JournalIntegrityError("incomplete frame before journal tail")
                tail_incomplete = True
                break
            if frame.sequence != expected_sequence:
                raise JournalIntegrityError(
                    f"journal sequence mismatch at {frame.sequence}"
                )
            if frame.previous_hash != previous_hash:
                raise JournalIntegrityError(
                    f"journal previous hash mismatch at {frame.sequence}"
                )
            self._validate_blob_refs(frame.events)
            frames.append(frame)
            valid_length += len(physical)
            previous_hash = frame.frame_hash
            expected_sequence += 1
        if tail_incomplete:
            last_complete_sequence = frames[-1].sequence if frames else 0
            if anchored_before_recovery.sequence > last_complete_sequence:
                raise AnchorMismatchError(
                    "anchor confirms a journal tail that is incomplete locally"
                )
            with open(self.journal_path, "r+b") as file:
                file.truncate(valid_length)
                file.flush()
                self._fs_ops.fsync_file(file)
            self._fs_ops.fsync_directory(self.base_dir)
        return frames

    @staticmethod
    def _rebuild_aggregate_versions(
        frames: Sequence[TransactionFrame],
    ) -> dict[str, int]:
        versions: dict[str, int] = {}
        for frame in frames:
            event_aggregate_ids = {event.aggregate_id for event in frame.events}
            if (
                event_aggregate_ids != set(frame.expected_versions)
                or event_aggregate_ids != set(frame.aggregate_versions)
            ):
                raise JournalIntegrityError(
                    "event aggregate ids and version maps must match"
                )
            for aggregate_id, expected in frame.expected_versions.items():
                if versions.get(aggregate_id, 0) != expected:
                    raise JournalIntegrityError(
                        f"aggregate version mismatch for {aggregate_id}"
                    )
            for aggregate_id, version in frame.aggregate_versions.items():
                expected = frame.expected_versions.get(
                    aggregate_id,
                    versions.get(aggregate_id, 0),
                )
                if version != expected + 1:
                    raise JournalIntegrityError(
                        f"invalid aggregate version for {aggregate_id}"
                    )
                versions[aggregate_id] = version
        return versions

    def _verify_anchor(self) -> None:
        anchored = self.anchor.read()
        local = AnchorState(self._sequence, self._previous_hash)
        if anchored != local:
            raise AnchorMismatchError(
                "journal and monotonic anchor high-water mark differ"
            )

    def _validate_blob_refs(self, events: Sequence[JournalEvent]) -> None:
        if self._blob_ref_validator is None:
            return
        for event in events:
            reference = event.payload.get("blob_ref")
            if reference is None:
                continue
            if not isinstance(reference, dict):
                raise JournalIntegrityError("invalid blob reference")
            try:
                blob_ref = BlobRef.from_dict(reference)
            except (TypeError, ValueError) as exc:
                raise JournalIntegrityError("invalid blob reference") from exc
            if self._blob_ref_validator(blob_ref) is not True:
                raise JournalIntegrityError("blob reference is not published")

    def commit(
        self,
        events: Sequence[JournalEvent],
        expected_versions: dict[str, int],
        *,
        expected_head_sequence: int | None = None,
        expected_head_hash: str | None = None,
    ) -> CommitResult:
        self._ensure_writable()
        event_values = tuple(events)
        if not event_values:
            raise ValueError("cannot commit an empty transaction")
        if not all(isinstance(event, JournalEvent) for event in event_values):
            raise TypeError("events must contain JournalEvent values")
        self._validate_blob_refs(event_values)
        with self._mutex:
            self._ensure_writable()
            logical_head_hash = "" if self._sequence == 0 else self._previous_hash
            if (
                expected_head_sequence is not None
                and expected_head_sequence != self._sequence
            ) or (
                expected_head_hash is not None
                and expected_head_hash != logical_head_hash
            ):
                raise JournalIntegrityError("journal head mismatch")
            aggregate_ids = {event.aggregate_id for event in event_values}
            if set(expected_versions) != aggregate_ids:
                raise JournalIntegrityError(
                    "expected aggregate versions must cover transaction events"
                )
            for aggregate_id, expected in expected_versions.items():
                current = self._aggregate_versions.get(aggregate_id, 0)
                if (
                    isinstance(expected, bool)
                    or not isinstance(expected, int)
                    or expected != current
                ):
                    raise JournalIntegrityError(
                        f"aggregate version mismatch for {aggregate_id}"
                    )
            aggregate_versions = {
                aggregate_id: expected + 1
                for aggregate_id, expected in expected_versions.items()
            }
            frame = TransactionFrame.seal(
                tx_id=f"tx_{uuid.uuid4().hex}",
                sequence=self._sequence + 1,
                writer_epoch=self._writer_epoch,
                timestamp=time.time(),
                expected_versions=expected_versions,
                aggregate_versions=aggregate_versions,
                previous_hash=self._previous_hash,
                events=event_values,
                hmac_key=self._hmac_key,
            )
            try:
                _append_record(
                    self.journal_path,
                    frame.to_bytes(),
                    self._fs_ops,
                )
                self._fs_ops.fsync_directory(self.base_dir)
            except BaseException:
                self._write_disabled = True
                raise
            self._frames.append(frame)
            self._sequence = frame.sequence
            self._previous_hash = frame.frame_hash
            self._aggregate_versions.update(aggregate_versions)
            try:
                anchored = self.anchor.compare_and_swap(
                    frame.sequence - 1,
                    frame.previous_hash,
                    frame.sequence,
                    frame.frame_hash,
                )
            except Exception:
                anchored = False
            if anchored:
                return CommitResult(frame=frame, state=CommitState.ANCHORED)
            self._write_disabled = True
            return CommitResult(
                frame=frame,
                state=CommitState.DURABLE_NOT_ANCHORED,
            )

    def replay(self, from_sequence: int = 1) -> Iterator[TransactionFrame]:
        if isinstance(from_sequence, bool) or from_sequence < 1:
            raise ValueError("from_sequence must be >= 1")
        with self._mutex:
            frames = tuple(self._load_and_recover())
            self._rebuild_aggregate_versions(frames)
        for frame in frames:
            if frame.sequence >= from_sequence:
                yield frame

    def get_last_frame(self) -> TransactionFrame | None:
        with self._mutex:
            return self._frames[-1] if self._frames else None

    def get_aggregate_versions(
        self,
        aggregate_ids: Iterable[str],
    ) -> dict[str, int]:
        """Return an atomic snapshot of selected aggregate versions."""

        ids = tuple(aggregate_ids)
        with self._mutex:
            return {
                aggregate_id: self._aggregate_versions.get(aggregate_id, 0)
                for aggregate_id in ids
            }

    def verify_chain(self) -> tuple[bool, list[str]]:
        try:
            list(self.replay())
        except JournalIntegrityError as exc:
            return False, [str(exc)]
        return True, []

    @contextmanager
    def transaction_guard(self) -> Iterator[None]:
        """Serialize a domain projection refresh with its following commit."""
        with self._transaction_mutex:
            yield
