"""Crash-safe, process-safe updates for the local dotenv configuration."""

from __future__ import annotations

import fcntl
import os
import re
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from types import MappingProxyType

from dotenv.parser import parse_stream

_ENV_KEY_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_ACTIVE_ASSIGNMENT_RE = re.compile(
    r"^(?:export\s+)?(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*="
)


@dataclass(frozen=True, slots=True)
class EnvFileSnapshot:
    """Complete active dotenv state observed while holding the file lock."""

    values: Mapping[str, str]
    rendered: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            MappingProxyType(dict(self.values)),
        )


class EnvFileStoreError(OSError):
    """Base class for phase-aware dotenv transaction failures."""


class EnvPreReplaceError(EnvFileStoreError):
    """The atomic rename did not complete, so the original file is intact."""


class EnvCommitUncertainError(EnvFileStoreError):
    """The rename completed but directory durability could not be confirmed."""

    def __init__(
        self,
        message: str,
        *,
        snapshot: EnvFileSnapshot | None,
    ) -> None:
        super().__init__(message)
        self.snapshot = snapshot


class EnvPostCommitCleanupError(EnvFileStoreError):
    """The file is durable, but transaction resource cleanup failed."""

    def __init__(self, message: str, *, snapshot: EnvFileSnapshot) -> None:
        super().__init__(message)
        self.snapshot = snapshot


class AtomicEnvFileStore:
    """Update dotenv state as one durable, process-safe transaction."""

    def __init__(self, path: str | os.PathLike[str] = ".env") -> None:
        self.path = Path(path)

    def update_many(self, updates: Mapping[str, str]) -> EnvFileSnapshot:
        """Apply fixed updates and return the complete lock-observed state."""

        normalized = self._validate_updates(updates)
        return self.update_with(lambda _current: normalized)

    def update_with(
        self,
        mutator: Callable[[Mapping[str, str]], Mapping[str, str]],
    ) -> EnvFileSnapshot:
        """Read, merge and replace while one inter-process lock is held.

        ``mutator`` receives the active values read from disk inside the lock
        and returns only the assignments that should be changed. This is the
        required API for set-valued read/modify/write operations.
        """

        path = self.path
        parent = path.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
            lock_path = path.with_name(f".{path.name}.lock")
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        except OSError as exc:
            raise EnvPreReplaceError(str(exc)) from exc

        lock_acquired = False
        replaced_snapshot: EnvFileSnapshot | None = None
        try:
            try:
                os.fchmod(lock_fd, 0o600)
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                lock_acquired = True
                existing = path.read_text(encoding="utf-8") if path.exists() else ""
            except OSError as exc:
                raise EnvPreReplaceError(str(exc)) from exc

            current = self._snapshot(existing)
            updates = self._validate_updates(mutator(current.values))
            if not updates:
                return current
            rendered = self._render(existing, updates)
            replaced_snapshot = self._durable_replace(rendered)
            return replaced_snapshot
        finally:
            active_exception = sys.exc_info()[0] is not None
            if lock_acquired:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except OSError:
                    # Closing the descriptor still releases the advisory lock;
                    # a release error cannot change an already resolved commit.
                    pass
            try:
                os.close(lock_fd)
            except OSError as exc:
                if active_exception:
                    # Never mask the transaction's phase-aware root failure.
                    pass
                elif replaced_snapshot is not None:
                    raise EnvPostCommitCleanupError(
                        str(exc),
                        snapshot=replaced_snapshot,
                    ) from exc
                else:
                    raise EnvPreReplaceError(str(exc)) from exc

    def read_snapshot(self) -> EnvFileSnapshot:
        """Return a complete snapshot serialized against concurrent writers."""

        return self.update_with(lambda _current: {})

    @staticmethod
    def _validate_updates(updates: Mapping[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for key, value in updates.items():
            if not isinstance(key, str) or not _ENV_KEY_RE.fullmatch(key):
                raise ValueError(f"invalid dotenv key: {key!r}")
            if not isinstance(value, str):
                raise TypeError(f"dotenv value for {key} must be text")
            if any(character in value for character in ("\n", "\r", "\x00")):
                raise ValueError(
                    f"dotenv value for {key} contains a forbidden character"
                )
            normalized[key] = value
        return normalized

    @staticmethod
    def _render(existing: str, updates: Mapping[str, str]) -> str:
        rendered: list[str] = []
        replaced: set[str] = set()
        for line in existing.splitlines(keepends=True):
            stripped = line.lstrip()
            match = (
                None
                if stripped.startswith("#")
                else _ACTIVE_ASSIGNMENT_RE.match(stripped)
            )
            key = match.group("key") if match is not None else ""
            if key not in updates:
                rendered.append(line)
                continue
            if key not in replaced:
                rendered.append(f"{key}={updates[key]}\n")
                replaced.add(key)

        for key, value in updates.items():
            if key in replaced:
                continue
            if rendered and not rendered[-1].endswith(("\n", "\r")):
                rendered[-1] = f"{rendered[-1]}\n"
            rendered.append(f"{key}={value}\n")
        return "".join(rendered)

    @staticmethod
    def _snapshot(rendered: str) -> EnvFileSnapshot:
        values: dict[str, str] = {}
        for binding in parse_stream(StringIO(rendered)):
            if binding.error or binding.key is None or binding.value is None:
                continue
            values[binding.key] = binding.value
        return EnvFileSnapshot(values=values, rendered=rendered)

    def _durable_replace(self, rendered: str) -> EnvFileSnapshot:
        path = self.path
        parent = path.parent
        temp_fd = -1
        temp_path: Path | None = None
        replaced = False
        try:
            temp_fd, temp_name = tempfile.mkstemp(
                prefix=f".{path.name}.tmp-",
                dir=parent,
            )
            temp_path = Path(temp_name)
            os.fchmod(temp_fd, 0o600)
            with os.fdopen(
                temp_fd,
                "w",
                encoding="utf-8",
                newline="",
            ) as stream:
                temp_fd = -1
                stream.write(rendered)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, path)
            replaced = True
            self._fsync_parent(parent)
            return self._snapshot(rendered)
        except _ParentDirectoryCleanupError as exc:
            raise EnvPostCommitCleanupError(
                str(exc),
                snapshot=self._snapshot(rendered),
            ) from exc
        except OSError as exc:
            if replaced:
                snapshot = self._reconcile_actual_file()
                raise EnvCommitUncertainError(
                    str(exc),
                    snapshot=snapshot,
                ) from exc
            raise EnvPreReplaceError(str(exc)) from exc
        finally:
            if temp_fd >= 0:
                active_exception = sys.exc_info()[0] is not None
                try:
                    os.close(temp_fd)
                except OSError as exc:
                    if not active_exception:
                        raise EnvPreReplaceError(str(exc)) from exc
            if temp_path is not None:
                active_exception = sys.exc_info()[0] is not None
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    if not active_exception:
                        if replaced:
                            raise EnvPostCommitCleanupError(
                                str(exc),
                                snapshot=self._snapshot(rendered),
                            ) from exc
                        raise EnvPreReplaceError(str(exc)) from exc

    @staticmethod
    def _fsync_parent(parent: Path) -> None:
        directory_fd = os.open(
            parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        except OSError:
            try:
                os.close(directory_fd)
            except OSError:
                pass
            raise
        try:
            os.close(directory_fd)
        except OSError as exc:
            raise _ParentDirectoryCleanupError(str(exc)) from exc

    def _reconcile_actual_file(self) -> EnvFileSnapshot | None:
        try:
            rendered = self.path.read_text(encoding="utf-8")
        except OSError:
            return None
        return self._snapshot(rendered)


class _ParentDirectoryCleanupError(OSError):
    """Parent fsync succeeded, but closing its descriptor failed."""
