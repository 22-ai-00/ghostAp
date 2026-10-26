"""Monotonic journal anchor contracts."""

from __future__ import annotations

import fcntl
import json
import os
import stat
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

from .frame import GENESIS_HASH


@dataclass(frozen=True)
class AnchorState:
    """The externally anchored journal high-water mark."""

    sequence: int = 0
    frame_hash: str = GENESIS_HASH


class AnchorProvider(Protocol):
    """Monotonic compare-and-swap boundary for journal commits."""

    production_safe: bool

    def read(self) -> AnchorState: ...

    def compare_and_swap(
        self,
        expected_sequence: int,
        expected_hash: str,
        new_sequence: int,
        new_hash: str,
    ) -> bool: ...


class MemoryAnchor:
    """Thread-safe in-memory anchor for tests and offline probes only."""

    production_safe = False

    def __init__(self) -> None:
        self._state = AnchorState()
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

    def read(self) -> AnchorState:
        with self._lock:
            return self._state

    def compare_and_swap(
        self,
        expected_sequence: int,
        expected_hash: str,
        new_sequence: int,
        new_hash: str,
    ) -> bool:
        if (
            isinstance(expected_sequence, bool)
            or isinstance(new_sequence, bool)
            or expected_sequence < 0
            or new_sequence != expected_sequence + 1
            or len(expected_hash) != 64
            or len(new_hash) != 64
        ):
            return False
        with self._lock:
            if self._state != AnchorState(expected_sequence, expected_hash):
                return False
            self._state = AnchorState(new_sequence, new_hash)
            return True


class AnchorCorruptionError(RuntimeError):
    """A persistent anchor cannot be decoded without losing monotonicity."""


class FileAnchor:
    """Cross-process CAS anchor persisted as an atomically replaced file."""

    production_safe = True
    _VERSION = 1
    _HEX = frozenset("0123456789abcdef")

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock_path = self.path.with_name(f".{self.path.name}.lock")

    @contextmanager
    def _locked(self, operation: int) -> Iterator[None]:
        fd = os.open(
            self._lock_path,
            os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        try:
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, operation)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    @classmethod
    def _valid_hash(cls, value: object) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(char in cls._HEX for char in value)
        )

    @classmethod
    def _validate_state(cls, value: object) -> AnchorState:
        if not isinstance(value, dict) or set(value) != {
            "version",
            "sequence",
            "frame_hash",
        }:
            raise AnchorCorruptionError("invalid anchor envelope")
        version = value["version"]
        sequence = value["sequence"]
        frame_hash = value["frame_hash"]
        if version != cls._VERSION or isinstance(version, bool):
            raise AnchorCorruptionError("unsupported anchor version")
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 0
            or not cls._valid_hash(frame_hash)
            or (sequence == 0 and frame_hash != GENESIS_HASH)
        ):
            raise AnchorCorruptionError("invalid anchor state")
        return AnchorState(sequence=sequence, frame_hash=frame_hash)

    def _read_unlocked(self) -> AnchorState:
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        except FileNotFoundError:
            return AnchorState()
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise AnchorCorruptionError("anchor is not a regular file")
            with os.fdopen(fd, "rb") as handle:
                fd = -1
                raw = handle.read()
        finally:
            if fd >= 0:
                os.close(fd)
        def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
            decoded_object: dict[str, object] = {}
            for key, value in pairs:
                if key in decoded_object:
                    raise AnchorCorruptionError("duplicate anchor envelope key")
                decoded_object[key] = value
            return decoded_object

        try:
            decoded = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise AnchorCorruptionError("malformed anchor envelope") from exc
        return self._validate_state(decoded)

    def read(self) -> AnchorState:
        with self._locked(fcntl.LOCK_SH):
            return self._read_unlocked()

    def compare_and_swap(
        self,
        expected_sequence: int,
        expected_hash: str,
        new_sequence: int,
        new_hash: str,
    ) -> bool:
        if (
            isinstance(expected_sequence, bool)
            or not isinstance(expected_sequence, int)
            or isinstance(new_sequence, bool)
            or not isinstance(new_sequence, int)
            or expected_sequence < 0
            or new_sequence != expected_sequence + 1
            or not self._valid_hash(expected_hash)
            or not self._valid_hash(new_hash)
        ):
            return False
        with self._locked(fcntl.LOCK_EX):
            if self._read_unlocked() != AnchorState(expected_sequence, expected_hash):
                return False
            self._write_unlocked(AnchorState(new_sequence, new_hash))
            return True

    def confirm_durable(self, state: AnchorState) -> bool:
        """Confirm an ambiguous CAS replacement and durably sync its directory.

        A CAS can replace the file before its directory fsync raises. Keep the
        anchor lock through comparison and the repeat fsync so a competing CAS
        cannot change the file being confirmed.
        """

        with self._locked(fcntl.LOCK_EX):
            if self._read_unlocked() != state:
                return False
            fd = os.open(self.path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise AnchorCorruptionError("anchor is not a regular file")
                os.fsync(fd)
            finally:
                os.close(fd)
            directory_fd = os.open(
                self.path.parent,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            return True

    def _write_unlocked(self, state: AnchorState) -> None:
        payload = json.dumps(
            {
                "version": self._VERSION,
                "sequence": state.sequence,
                "frame_hash": state.frame_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        temp_path = self.path.with_name(
            f".{self.path.name}.{uuid.uuid4().hex}.tmp"
        )
        fd = os.open(
            temp_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        try:
            os.fchmod(fd, 0o600)
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("short anchor write")
                view = view[written:]
            os.fsync(fd)
            os.close(fd)
            fd = -1
            os.replace(temp_path, self.path)
            directory_fd = os.open(
                self.path.parent,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            if fd >= 0:
                os.close(fd)
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            raise
