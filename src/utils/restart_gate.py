"""Cross-process admission and drain gate for safe service restarts.

The gate uses two persistent ``flock`` files:

* tasks briefly take ``admission`` shared, then take ``drain`` shared and
  release ``admission``;
* a restart takes ``admission`` exclusive before waiting for ``drain``
  exclusive.

This ordering fences new work before waiting for already-running work.  Lock
files are intentionally never removed: replacing a lock file while another
process still has its old inode open would split the lock domain.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import logging
import math
import os
import re
import secrets
import signal
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

EX_TEMPFAIL = 75
_SUPPORTED_PLATFORMS = frozenset({"linux", "darwin"})
_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{20,64}$")
_LAUNCHD_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,200}$")
_LOCK_POLL_SECONDS = 0.025
_PROCESS_GROUP_CLEANUP_BUDGET_SECONDS = 2.0
_MIN_PROCESS_GROUP_PHASE_SECONDS = 0.01
_IDENTITY_VERSION = 1
_MAX_IDENTITY_FILE_BYTES = 8192
_GENERATION_PARTICIPATING = "participating"
_GENERATION_READY = "ready"
_GENERATION_FAILED = "failed"
_BROAD_GATE_DIRECTORIES = frozenset(
    {
        Path("/"),
        Path("/tmp"),
        Path("/var/tmp"),
        Path("/private/tmp"),
        Path("/dev/shm"),
    }
)


class RestartGateError(RuntimeError):
    """Base exception for restart-gate safety failures."""


class RestartGateTimeout(RestartGateError):
    """The shared restart deadline expired."""


class RestartRunStatus(str, Enum):
    """Result of a generation-guarded restart request."""

    RESTARTED = "restarted"
    COALESCED = "coalesced"
    TIMED_OUT = "timed_out"
    FAILED = "failed"


@dataclass(frozen=True)
class RestartRunResult:
    """Generation-guarded operation result."""

    status: RestartRunStatus
    exit_code: int


def _require_supported_platform() -> None:
    if sys.platform not in _SUPPORTED_PLATFORMS:
        raise RestartGateError(
            f"restart gate requires POSIX flock; unsupported platform: {sys.platform}"
        )


def _validate_token(token: str) -> str:
    if not _TOKEN_PATTERN.fullmatch(token):
        raise RestartGateError("invalid restart generation token")
    return token


def _new_token() -> str:
    # Prefix with an alphanumeric character so the token is safe as a separate
    # argparse value (a urlsafe token may otherwise begin with ``-`` and be
    # misclassified as another CLI option).
    return _validate_token(f"g{secrets.token_urlsafe(23)}")


def _absolute_path(path: str | os.PathLike[str], *, label: str) -> Path:
    value = Path(path)
    if not value.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return Path(os.path.abspath(value))


def _dedicated_gate_path(
    path: str | os.PathLike[str],
    *,
    label: str,
) -> Path:
    value = _absolute_path(path, label=label)
    if (
        value in _BROAD_GATE_DIRECTORIES
        or len(value.parts) <= 2
        or value == Path.home()
    ):
        raise RestartGateError(
            f"{label} must be a dedicated private directory, not {value}"
        )
    return value


def _canonical_project_path(path: str | os.PathLike[str]) -> Path:
    project = _absolute_path(path, label="project directory")
    try:
        project = project.resolve(strict=True)
        metadata = project.stat()
    except OSError as exc:
        raise RestartGateError(
            f"cannot resolve project directory: {project}"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise RestartGateError(f"project path is not a directory: {project}")
    return project


def _identity(metadata: os.stat_result) -> list[int]:
    return [int(metadata.st_dev), int(metadata.st_ino)]


def _process_instance_identity(pid: int) -> str:
    if pid <= 0:
        raise RestartGateError("participation service PID must be positive")
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, OSError) as exc:
        raise RestartGateError(
            f"participation service PID is not alive: {pid}"
        ) from exc

    if sys.platform == "linux":
        try:
            raw_stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
            fields_after_command = raw_stat.rsplit(")", 1)[1].split()
            start_ticks = fields_after_command[19]
            boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
                encoding="ascii"
            ).strip()
        except (OSError, IndexError, UnicodeError) as exc:
            raise RestartGateError(
                f"cannot identify participation service process: {pid}"
            ) from exc
        return f"linux:{boot_id}:{start_ticks}"

    try:
        completed = subprocess.run(
            ["/bin/ps", "-o", "lstart=", "-p", str(pid)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            shell=False,
            close_fds=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RestartGateError(
            f"cannot identify participation service process: {pid}"
        ) from exc
    started_at = completed.stdout.strip()
    if completed.returncode != 0 or not started_at:
        raise RestartGateError(
            f"participation service PID is not alive: {pid}"
        )
    return f"darwin:{started_at}"


def _read_private_json(
    path: Path,
    *,
    label: str,
    allow_missing: bool = False,
) -> dict[str, object] | None:
    try:
        fd = os.open(path, RestartGate._open_flags(os.O_RDONLY))
    except FileNotFoundError:
        if allow_missing:
            return None
        raise RestartGateError(f"{label} is missing") from None
    except OSError as exc:
        raise RestartGateError(f"cannot open {label}") from exc

    try:
        os.set_inheritable(fd, False)
        metadata = os.fstat(fd)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) & 0o077
            or metadata.st_size > _MAX_IDENTITY_FILE_BYTES
        ):
            raise RestartGateError(f"invalid {label} file")
        try:
            path_metadata = path.lstat()
        except OSError as exc:
            raise RestartGateError(f"cannot verify {label} path") from exc
        if _identity(path_metadata) != _identity(metadata):
            raise RestartGateError(f"{label} path changed while opening")
        raw = os.read(fd, _MAX_IDENTITY_FILE_BYTES + 1)
    finally:
        os.close(fd)

    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RestartGateError(f"invalid {label} JSON") from exc
    if not isinstance(value, dict):
        raise RestartGateError(f"invalid {label} payload")
    return value


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise OSError("short write")
        offset += written


def _fsync_directory(directory: Path) -> None:
    fd = os.open(
        directory,
        RestartGate._open_flags(
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        ),
    )
    try:
        os.set_inheritable(fd, False)
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish_private_json(
    path: Path,
    value: dict[str, object],
    *,
    label: str,
    create_only: bool = False,
) -> bool:
    payload = (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")
    temp_path = path.parent / f".{path.name}.{_new_token()}.tmp"
    fd: int | None = None
    try:
        fd = os.open(
            temp_path,
            RestartGate._open_flags(os.O_WRONLY | os.O_CREAT | os.O_EXCL),
            0o600,
        )
        os.set_inheritable(fd, False)
        _write_all(fd, payload)
        os.fsync(fd)
        os.close(fd)
        fd = None
        if create_only:
            try:
                os.link(temp_path, path, follow_symlinks=False)
            except FileExistsError:
                return False
            finally:
                temp_path.unlink(missing_ok=True)
        else:
            os.replace(temp_path, path)
        _fsync_directory(path.parent)
        return True
    except OSError as exc:
        if fd is not None:
            os.close(fd)
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            logger.debug("failed to clean temporary %s file", label, exc_info=True)
        raise RestartGateError(f"cannot publish {label}") from exc


def _prepare_private_directory(directory: Path, *, label: str) -> None:
    created = False
    try:
        directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        created = True
    except FileExistsError:
        pass
    except OSError as exc:
        raise RestartGateError(f"cannot create {label}: {directory}") from exc
    try:
        metadata = directory.lstat()
    except OSError as exc:
        raise RestartGateError(f"cannot inspect {label}: {directory}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RestartGateError(f"{label} is not a real directory: {directory}")
    if metadata.st_uid != os.getuid():
        raise RestartGateError(f"{label} is not owned by current user: {directory}")
    if created:
        try:
            os.chmod(directory, 0o700)
            metadata = directory.lstat()
        except OSError as exc:
            raise RestartGateError(f"cannot secure {label}: {directory}") from exc
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise RestartGateError(f"{label} has unsafe permissions: {directory}")


class RestartGate:
    """Checkout-scoped cross-process task/restart gate."""

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        _project_dir: Path | None = None,
        _registry_dir: Path | None = None,
        _expected_owner_id: str | None = None,
    ):
        _require_supported_platform()
        self.directory = _dedicated_gate_path(
            directory,
            label="restart gate directory",
        )
        self._prepare_directory()
        self.admission_path = self.directory / "admission.lock"
        self.drain_path = self.directory / "drain.lock"
        self.generation_path = self.directory / "generation"
        self.owner_path = self.directory / "owner.json"
        self.participation_path = self.directory / "participation.json"
        self.project_dir = _project_dir
        self.registry_dir = _registry_dir
        self._expected_owner_id = _expected_owner_id
        self._pinned_participation_id: str | None = None
        self._task_waiters_canceled = threading.Event()
        # Create the stable lock inodes eagerly.  They are never unlinked.
        identities: list[tuple[int, int]] = []
        for path in (self.admission_path, self.drain_path):
            fd = self._open_lock(path)
            try:
                metadata = os.fstat(fd)
                identities.append((metadata.st_dev, metadata.st_ino))
            finally:
                os.close(fd)
        if identities[0] == identities[1]:
            raise RestartGateError(
                "restart admission and drain locks must use distinct inodes"
            )
        self._directory_identity = _identity(self.directory.lstat())
        self._lock_identities = {
            "admission_identity": [identities[0][0], identities[0][1]],
            "drain_identity": [identities[1][0], identities[1][1]],
        }

    @classmethod
    def for_project(
        cls,
        project_dir: str | os.PathLike[str],
        *,
        override: str | os.PathLike[str] | None = None,
    ) -> "RestartGate":
        """Build the gate bound to a canonical checkout identity.

        The stable locator lives outside the checkout.  Once created it pins
        the checkout to one gate directory, so a later ``.env`` change cannot
        silently move running tasks and restart workers into different lock
        domains.
        """

        project = _canonical_project_path(project_dir)
        registry = cls._registry_directory(project)
        cls._prepare_registry(registry)
        locator_path = registry / "locator.json"
        requested = (
            _dedicated_gate_path(
                override,
                label="restart gate directory override",
            )
            if override
            else None
        )
        if requested == project:
            raise RestartGateError(
                "restart gate directory override must not be the project root"
            )

        locator = cls._read_locator(locator_path, project, allow_missing=True)
        if locator is not None:
            pinned_directory = _dedicated_gate_path(
                str(locator["gate_dir"]),
                label="pinned restart gate directory",
            )
            if requested is not None and requested != pinned_directory:
                raise RestartGateError(
                    "restart gate override conflicts with the checkout's "
                    "pinned gate identity"
                )
            return cls(
                pinned_directory,
                _project_dir=project,
                _registry_dir=registry,
                _expected_owner_id=str(locator["owner_id"]),
            )._bind_owner()

        directory = requested or (registry / "gate")
        gate = cls(
            directory,
            _project_dir=project,
            _registry_dir=registry,
        )._bind_owner()
        locator_value: dict[str, object] = {
            "version": _IDENTITY_VERSION,
            "project_dir": str(project),
            "gate_dir": str(gate.directory),
            "owner_id": gate._expected_owner_id,
        }
        if not _publish_private_json(
            locator_path,
            locator_value,
            label="restart gate locator",
            create_only=True,
        ):
            locator = cls._read_locator(locator_path, project, allow_missing=False)
            assert locator is not None
            if (
                str(locator["gate_dir"]) != str(gate.directory)
                or str(locator["owner_id"]) != gate._expected_owner_id
            ):
                raise RestartGateError(
                    "concurrent restart gate locator bound a different identity"
                )
        return gate

    @classmethod
    def for_worker(
        cls,
        project_dir: str | os.PathLike[str],
        *,
        expected_generation: str,
        configured_override: str | os.PathLike[str] | None = None,
    ) -> "RestartGate":
        """Resolve a detached worker from the participation locator.

        ``configured_override`` is intentionally ignored.  It is accepted so
        callers can pass their current settings without allowing a post-detach
        ``.env`` edit to change the lock identity.
        """

        del configured_override
        gate = cls.from_locator(project_dir)
        proof = gate._validate_participation()
        state = gate._read_generation_state()
        if (
            state["generation"] == _validate_token(expected_generation)
            and state["status"] == _GENERATION_READY
        ):
            gate._pinned_participation_id = str(proof["instance_id"])
        return gate

    @classmethod
    def from_locator(
        cls,
        project_dir: str | os.PathLike[str],
    ) -> "RestartGate":
        """Open the checkout's already-pinned gate without consulting settings."""

        project = _canonical_project_path(project_dir)
        registry = cls._registry_directory(project)
        locator = cls._read_locator(
            registry / "locator.json",
            project,
            allow_missing=False,
        )
        assert locator is not None
        return cls(
            str(locator["gate_dir"]),
            _project_dir=project,
            _registry_dir=registry,
            _expected_owner_id=str(locator["owner_id"]),
        )._bind_owner()

    @staticmethod
    def _registry_directory(project: Path) -> Path:
        digest = hashlib.sha256(os.fsencode(str(project))).hexdigest()[:32]
        return project.parent / ".ghostap-restart-gates" / digest

    @staticmethod
    def _prepare_registry(registry: Path) -> None:
        root = registry.parent
        _prepare_private_directory(root, label="restart gate registry root")
        _prepare_private_directory(registry, label="restart gate registry")

    @staticmethod
    def _read_locator(
        path: Path,
        project: Path,
        *,
        allow_missing: bool,
    ) -> dict[str, object] | None:
        locator = _read_private_json(
            path,
            label="restart gate locator",
            allow_missing=allow_missing,
        )
        if locator is None:
            return None
        if (
            locator.get("version") != _IDENTITY_VERSION
            or locator.get("project_dir") != str(project)
        ):
            raise RestartGateError(
                "restart gate locator belongs to a different checkout"
            )
        try:
            _validate_token(str(locator["owner_id"]))
            _dedicated_gate_path(
                str(locator["gate_dir"]),
                label="pinned restart gate directory",
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RestartGateError("invalid restart gate locator") from exc
        return locator

    def _bind_owner(self) -> "RestartGate":
        owner = _read_private_json(
            self.owner_path,
            label="restart gate owner",
            allow_missing=True,
        )
        if owner is None:
            if self.project_dir is None:
                raise RestartGateError(
                    "restart gate owner is missing for an unbound checkout"
                )
            owner_id = self._expected_owner_id or _new_token()
            candidate: dict[str, object] = {
                "version": _IDENTITY_VERSION,
                "project_dir": str(self.project_dir),
                "gate_dir": str(self.directory),
                "owner_id": owner_id,
            }
            if _publish_private_json(
                self.owner_path,
                candidate,
                label="restart gate owner",
                create_only=True,
            ):
                owner = candidate
            else:
                owner = _read_private_json(
                    self.owner_path,
                    label="restart gate owner",
                    allow_missing=False,
                )
                assert owner is not None

        if (
            owner.get("version") != _IDENTITY_VERSION
            or owner.get("gate_dir") != str(self.directory)
        ):
            raise RestartGateError("invalid restart gate owner identity")
        owner_project = owner.get("project_dir")
        if self.project_dir is not None and owner_project != str(self.project_dir):
            raise RestartGateError(
                "restart gate is already bound to a different checkout"
            )
        try:
            owner_id = _validate_token(str(owner["owner_id"]))
        except (KeyError, TypeError) as exc:
            raise RestartGateError("invalid restart gate owner identity") from exc
        if (
            self._expected_owner_id is not None
            and owner_id != self._expected_owner_id
        ):
            raise RestartGateError("restart gate owner identity changed")
        self._expected_owner_id = owner_id
        if self.project_dir is None:
            try:
                self.project_dir = _canonical_project_path(str(owner_project))
            except (TypeError, RestartGateError) as exc:
                raise RestartGateError("invalid restart gate owner checkout") from exc
        return self

    def _prepare_directory(self) -> None:
        created = False
        try:
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=False)
            created = True
        except FileExistsError:
            pass
        except OSError as exc:
            raise RestartGateError(
                f"cannot create restart gate directory: {self.directory}"
            ) from exc

        try:
            metadata = self.directory.lstat()
        except OSError as exc:
            raise RestartGateError(
                f"cannot inspect restart gate directory: {self.directory}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise RestartGateError(
                f"restart gate path is not a real directory: {self.directory}"
            )
        if metadata.st_uid != os.getuid():
            raise RestartGateError(
                "restart gate directory is not owned by current user: "
                f"{self.directory}"
            )
        if created:
            try:
                os.chmod(self.directory, 0o700)
                metadata = self.directory.lstat()
            except OSError as exc:
                raise RestartGateError(
                    f"cannot secure new restart gate directory: {self.directory}"
                ) from exc
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise RestartGateError(
                "restart gate directory has unsafe permissions: "
                f"{self.directory}"
            )

    @staticmethod
    def _open_flags(base: int) -> int:
        flags = base
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        return flags

    def _open_lock(self, path: Path) -> int:
        flags = self._open_flags(os.O_RDWR)
        created = False
        try:
            fd = os.open(path, flags)
        except FileNotFoundError:
            try:
                fd = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
                created = True
            except FileExistsError:
                try:
                    fd = os.open(path, flags)
                except OSError as exc:
                    raise RestartGateError(
                        f"cannot open restart lock: {path}"
                    ) from exc
            except OSError as exc:
                raise RestartGateError(f"cannot create restart lock: {path}") from exc
        except OSError as exc:
            raise RestartGateError(f"cannot open restart lock: {path}") from exc
        try:
            os.set_inheritable(fd, False)
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise RestartGateError(f"restart lock is not a regular file: {path}")
            if metadata.st_uid != os.getuid():
                raise RestartGateError(
                    f"restart lock is not owned by current user: {path}"
                )
            if metadata.st_nlink != 1:
                raise RestartGateError(
                    f"restart lock must be a single-link file: {path}"
                )
            if created:
                os.fchmod(fd, 0o600)
                metadata = os.fstat(fd)
            elif stat.S_IMODE(metadata.st_mode) & 0o077:
                raise RestartGateError(
                    f"restart lock has unsafe permissions: {path}"
                )
            try:
                path_metadata = path.lstat()
            except OSError as exc:
                raise RestartGateError(
                    f"cannot verify restart lock path: {path}"
                ) from exc
            if (
                path_metadata.st_dev,
                path_metadata.st_ino,
            ) != (
                metadata.st_dev,
                metadata.st_ino,
            ):
                raise RestartGateError(
                    f"restart lock path changed while opening: {path}"
                )
            return fd
        except BaseException:
            os.close(fd)
            raise

    @staticmethod
    def _unlock_and_close(fd: int) -> None:
        """Release a lock without destabilizing an already-terminal task.

        ``close(2)`` releases flock ownership even when an explicit unlock
        reports an error.  Release failures are therefore operational alerts,
        not a reason to rewrite a scheduler terminal state.
        """

        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            logger.exception("failed to explicitly unlock restart gate fd=%s", fd)
        finally:
            try:
                os.close(fd)
            except OSError:
                logger.exception("failed to close restart gate fd=%s", fd)

    @staticmethod
    def _acquire(
        fd: int,
        mode: int,
        deadline: float | None = None,
        canceled: threading.Event | None = None,
    ) -> None:
        while True:
            if canceled is not None and canceled.is_set():
                raise RestartGateError("restart gate task admission canceled")
            try:
                fcntl.flock(fd, mode | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RestartGateTimeout("restart gate deadline expired")
                    time.sleep(min(_LOCK_POLL_SECONDS, remaining))
                else:
                    time.sleep(_LOCK_POLL_SECONDS)
            except OSError as exc:
                if exc.errno == errno.EINTR:
                    continue
                raise RestartGateError("cannot acquire restart gate lock") from exc

    @contextmanager
    def task_guard(self) -> Iterator[None]:
        """Hold the drain reader lock for one complete scheduler run."""

        admission_fd = self._open_lock(self.admission_path)
        drain_fd: int | None = None
        admission_locked = False
        try:
            self._acquire(
                admission_fd,
                fcntl.LOCK_SH,
                canceled=self._task_waiters_canceled,
            )
            admission_locked = True
            if self._task_waiters_canceled.is_set():
                raise RestartGateError("restart gate task admission canceled")
            drain_fd = self._open_lock(self.drain_path)
            self._acquire(
                drain_fd,
                fcntl.LOCK_SH,
                canceled=self._task_waiters_canceled,
            )
            if self._task_waiters_canceled.is_set():
                raise RestartGateError("restart gate task admission canceled")
            admission_locked = False
            self._unlock_and_close(admission_fd)
            yield
        finally:
            try:
                if admission_locked:
                    self._unlock_and_close(admission_fd)
            finally:
                if drain_fd is not None:
                    self._unlock_and_close(drain_fd)

    def cancel_waiters(self) -> None:
        """Fail local task threads that are fenced outside the drain lock.

        This is process-local and does not disturb tasks that already hold the
        drain reader.  It prevents interpreter shutdown from deadlocking on
        executor threads waiting behind a restart worker's admission writer.
        """

        self._task_waiters_canceled.set()

    @contextmanager
    def _exclusive_guard(self, deadline: float) -> Iterator[None]:
        admission_fd = self._open_lock(self.admission_path)
        drain_fd: int | None = None
        admission_locked = False
        try:
            self._acquire(admission_fd, fcntl.LOCK_EX, deadline)
            admission_locked = True
            drain_fd = self._open_lock(self.drain_path)
            self._acquire(drain_fd, fcntl.LOCK_EX, deadline)
            yield
        finally:
            try:
                if drain_fd is not None:
                    self._unlock_and_close(drain_fd)
            finally:
                if admission_locked:
                    self._unlock_and_close(admission_fd)

    def _verify_gate_identity(self) -> None:
        try:
            directory_metadata = self.directory.lstat()
            admission_metadata = self.admission_path.lstat()
            drain_metadata = self.drain_path.lstat()
        except OSError as exc:
            raise RestartGateError("restart gate identity changed") from exc
        if (
            _identity(directory_metadata) != self._directory_identity
            or _identity(admission_metadata)
            != self._lock_identities["admission_identity"]
            or _identity(drain_metadata) != self._lock_identities["drain_identity"]
        ):
            raise RestartGateError("restart gate identity changed")
        if (
            not stat.S_ISDIR(directory_metadata.st_mode)
            or not stat.S_ISREG(admission_metadata.st_mode)
            or not stat.S_ISREG(drain_metadata.st_mode)
        ):
            raise RestartGateError("restart gate identity changed")

    def _validate_participation(self) -> dict[str, object]:
        self._verify_gate_identity()
        self._bind_owner()
        proof = _read_private_json(
            self.participation_path,
            label="restart participation proof",
            allow_missing=False,
        )
        assert proof is not None
        expected_fields = {
            "version": _IDENTITY_VERSION,
            "project_dir": str(self.project_dir),
            "gate_dir": str(self.directory),
            "owner_id": self._expected_owner_id,
            "directory_identity": self._directory_identity,
            **self._lock_identities,
        }
        for field, expected in expected_fields.items():
            if proof.get(field) != expected:
                raise RestartGateError(
                    f"restart participation {field} identity changed"
                )
        try:
            service_pid = int(proof["service_pid"])
            instance_id = _validate_token(str(proof["instance_id"]))
            recorded_process = str(proof["process_instance"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RestartGateError("invalid restart participation proof") from exc
        if self._pinned_participation_id is not None:
            if instance_id != self._pinned_participation_id:
                raise RestartGateError(
                    "restart participation instance identity changed"
                )
        actual_process = _process_instance_identity(service_pid)
        if recorded_process != actual_process:
            raise RestartGateError(
                "restart participation process instance identity changed"
            )
        return proof

    def _read_generation_state(self) -> dict[str, object]:
        state = _read_private_json(
            self.generation_path,
            label="restart generation",
            allow_missing=False,
        )
        assert state is not None
        try:
            generation = _validate_token(str(state["generation"]))
            participation_id = _validate_token(str(state["participation_id"]))
        except (KeyError, TypeError) as exc:
            raise RestartGateError("invalid restart generation state") from exc
        status = state.get("status")
        if (
            state.get("version") != _IDENTITY_VERSION
            or status
            not in {
                _GENERATION_PARTICIPATING,
                _GENERATION_READY,
                _GENERATION_FAILED,
            }
        ):
            raise RestartGateError("invalid restart generation state")
        if status == _GENERATION_FAILED:
            try:
                int(state["exit_code"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RestartGateError("invalid failed restart generation") from exc
        state["generation"] = generation
        state["participation_id"] = participation_id
        return state

    def _publish_generation_state(
        self,
        *,
        generation: str,
        participation_id: str,
        status: str,
        exit_code: int | None,
    ) -> str:
        generation = _validate_token(generation)
        participation_id = _validate_token(participation_id)
        if status not in {
            _GENERATION_PARTICIPATING,
            _GENERATION_READY,
            _GENERATION_FAILED,
        }:
            raise RestartGateError("invalid restart generation status")
        state: dict[str, object] = {
            "version": _IDENTITY_VERSION,
            "generation": generation,
            "participation_id": participation_id,
            "status": status,
            "exit_code": exit_code,
        }
        _publish_private_json(
            self.generation_path,
            state,
            label="restart generation",
        )
        return generation

    def publish_participation(
        self,
        *,
        service_pid: int | None = None,
        instance_id: str | None = None,
    ) -> str:
        """Publish a pre-ready running-service identity proof.

        The service must call :meth:`mark_ready` after its Lark connection is
        established.  Until then snapshot and restart workers fail closed.
        """

        self._verify_gate_identity()
        self._bind_owner()
        pid = os.getpid() if service_pid is None else int(service_pid)
        identity_token = _validate_token(instance_id or _new_token())
        proof: dict[str, object] = {
            "version": _IDENTITY_VERSION,
            "project_dir": str(self.project_dir),
            "gate_dir": str(self.directory),
            "owner_id": self._expected_owner_id,
            "service_pid": pid,
            "instance_id": identity_token,
            "process_instance": _process_instance_identity(pid),
            "directory_identity": self._directory_identity,
            **self._lock_identities,
        }
        _publish_private_json(
            self.participation_path,
            proof,
            label="restart participation proof",
        )
        self._publish_generation_state(
            generation=_new_token(),
            participation_id=identity_token,
            status=_GENERATION_PARTICIPATING,
            exit_code=None,
        )
        return identity_token

    def mark_ready(self, *, service_pid: int | None = None) -> str:
        """Publish readiness for the currently participating service."""

        proof = self._validate_participation()
        expected_pid = os.getpid() if service_pid is None else int(service_pid)
        if int(proof["service_pid"]) != expected_pid:
            raise RestartGateError(
                "restart participation service PID changed before readiness"
            )
        state = self._read_generation_state()
        if state["participation_id"] != proof["instance_id"]:
            raise RestartGateError(
                "restart participation does not own the readiness state"
            )
        if state["status"] == _GENERATION_FAILED:
            raise RestartGateError("restart generation is a failed terminal")
        if state["status"] == _GENERATION_READY:
            return str(state["generation"])
        return self._publish_generation_state(
            generation=_new_token(),
            participation_id=str(proof["instance_id"]),
            status=_GENERATION_READY,
            exit_code=None,
        )

    def ready_generation(self, *, service_pid: int) -> str:
        """Return readiness only for the exact participating service PID."""

        proof = self._validate_participation()
        if int(proof["service_pid"]) != int(service_pid):
            raise RestartGateError(
                "restart participation service PID does not match readiness check"
            )
        state = self._read_generation_state()
        if state["participation_id"] != proof["instance_id"]:
            raise RestartGateError(
                "restart participation does not own the readiness state"
            )
        if state["status"] != _GENERATION_READY:
            raise RestartGateError("restart participation is not ready")
        return str(state["generation"])

    def publish_generation(self) -> str:
        """Atomically publish a fresh generation for a proven live service."""

        proof = self._validate_participation()
        return self._publish_generation_state(
            generation=_new_token(),
            participation_id=str(proof["instance_id"]),
            status=_GENERATION_READY,
            exit_code=None,
        )

    def snapshot(self) -> str:
        """Return a verified ready generation without creating gate state."""

        proof = self._validate_participation()
        state = self._read_generation_state()
        if state["participation_id"] != proof["instance_id"]:
            raise RestartGateError(
                "restart participation does not own the current generation"
            )
        if state["status"] == _GENERATION_FAILED:
            raise RestartGateError("restart generation is a failed terminal")
        if state["status"] != _GENERATION_READY:
            raise RestartGateError("restart participation is not ready")
        return str(state["generation"])

    def run_if_current(
        self,
        expected: str,
        *,
        timeout: float,
        operation: Callable[[float], int],
    ) -> RestartRunResult:
        """Run one restart if ``expected`` is still the current generation.

        The timeout is one absolute budget shared by both exclusive lock
        acquisitions and the operation.  ``operation`` receives the remaining
        seconds and must honor that bound.
        """

        expected = _validate_token(expected)
        if not math.isfinite(timeout) or timeout <= 0:
            return RestartRunResult(RestartRunStatus.TIMED_OUT, EX_TEMPFAIL)
        deadline = time.monotonic() + timeout

        try:
            proof = self._validate_participation()
            with self._exclusive_guard(deadline):
                proof = self._validate_participation()
                current = self._read_generation_state()
                if current["participation_id"] != proof["instance_id"]:
                    raise RestartGateError(
                        "restart participation does not own the current generation"
                    )
                if (
                    current["generation"] != expected
                    or current["status"] != _GENERATION_READY
                ):
                    return RestartRunResult(RestartRunStatus.COALESCED, 0)

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return RestartRunResult(
                        RestartRunStatus.TIMED_OUT,
                        EX_TEMPFAIL,
                    )
                try:
                    exit_code = int(operation(remaining))
                except RestartGateTimeout:
                    return RestartRunResult(
                        RestartRunStatus.TIMED_OUT,
                        EX_TEMPFAIL,
                    )
                if exit_code != 0:
                    after = self._read_generation_state()
                    if (
                        after["generation"] == expected
                        and after["status"] == _GENERATION_READY
                    ):
                        self._publish_generation_state(
                            generation=expected,
                            participation_id=str(current["participation_id"]),
                            status=_GENERATION_FAILED,
                            exit_code=exit_code,
                        )
                    return RestartRunResult(RestartRunStatus.FAILED, exit_code)

                # The newly connected service normally publishes participation
                # and readiness.  Keep a worker-side generation fallback so a
                # successful custom operation cannot trigger a duplicate.
                after = self._read_generation_state()
                if (
                    after["generation"] == expected
                    and after["status"] == _GENERATION_READY
                ):
                    self._publish_generation_state(
                        generation=_new_token(),
                        participation_id=str(current["participation_id"]),
                        status=_GENERATION_READY,
                        exit_code=None,
                    )
                return RestartRunResult(RestartRunStatus.RESTARTED, 0)
        except RestartGateTimeout:
            return RestartRunResult(RestartRunStatus.TIMED_OUT, EX_TEMPFAIL)


def _append_worker_log(path: Path, message: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        os.set_inheritable(fd, False)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        os.write(fd, f"{stamp} [RESTART] {message}\n".encode("utf-8"))
    finally:
        os.close(fd)


def _terminate_process_group(
    process: subprocess.Popen[bytes],
    *,
    grace: float,
) -> None:
    """Terminate and reap a timed-out restart command and all descendants."""

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=max(grace, _MIN_PROCESS_GROUP_PHASE_SECONDS))
        return

    parent_reaped = False
    try:
        process.wait(timeout=max(grace, _MIN_PROCESS_GROUP_PHASE_SECONDS))
        parent_reaped = True
    except subprocess.TimeoutExpired:
        pass

    # The direct script may exit on TERM while a descendant ignores it.  Probe
    # the process group even after reaping the leader, and escalate the whole
    # group when any member remains.
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return
    os.killpg(process.pid, signal.SIGKILL)

    # SIGKILL is not catchable.  Still use a bounded wait so a pathological
    # platform failure cannot strand the restart worker indefinitely.
    if not parent_reaped:
        process.wait(timeout=max(grace, _MIN_PROCESS_GROUP_PHASE_SECONDS))


def run_restart_worker(
    *,
    gate: RestartGate,
    expected_generation: str,
    restart_script: str | os.PathLike[str],
    log_file: str | os.PathLike[str],
    delay: float,
    timeout: float,
    project_dir: str | os.PathLike[str] | None = None,
) -> int:
    """Run a preflight-snapshotted generation through the restart gate."""

    script = _absolute_path(restart_script, label="restart script")
    log_path = _absolute_path(log_file, label="restart log")
    cwd = (
        _absolute_path(project_dir, label="project directory")
        if project_dir is not None
        else script.parent
    )
    expected = _validate_token(expected_generation)
    if not math.isfinite(delay) or delay < 0:
        raise ValueError("restart delay must be >= 0")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("restart timeout must be finite and > 0")

    deadline = time.monotonic() + timeout
    if delay:
        time.sleep(min(delay, timeout))
    if time.monotonic() >= deadline:
        _append_worker_log(
            log_path,
            f"remote worker done status=timed_out exit_code={EX_TEMPFAIL}",
        )
        return EX_TEMPFAIL
    _append_worker_log(log_path, f"remote worker begin generation={expected}")
    remaining_budget = deadline - time.monotonic()
    if remaining_budget <= 0:
        _append_worker_log(
            log_path,
            f"remote worker done status=timed_out exit_code={EX_TEMPFAIL}",
        )
        return EX_TEMPFAIL

    def operation(remaining: float) -> int:
        operation_deadline = time.monotonic() + remaining
        environment = os.environ.copy()
        environment["GHOSTAP_LOG_MODE"] = "append"
        environment["GHOSTAP_RESTART_GATE_DIR"] = str(gate.directory)
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        log_fd = os.open(log_path, flags, 0o600)
        try:
            os.set_inheritable(log_fd, False)
            with os.fdopen(log_fd, "ab", closefd=True) as stream:
                cleanup_reserve = min(
                    _PROCESS_GROUP_CLEANUP_BUDGET_SECONDS,
                    max(
                        _MIN_PROCESS_GROUP_PHASE_SECONDS * 2,
                        remaining * 0.1,
                    ),
                )
                if remaining <= cleanup_reserve:
                    raise RestartGateTimeout(
                        "restart operation has no process execution budget"
                    )
                process = subprocess.Popen(
                    [str(script), "restart"],
                    cwd=str(cwd),
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=stream,
                    shell=False,
                    close_fds=True,
                    start_new_session=True,
                )
                execution_budget = (
                    operation_deadline - time.monotonic() - cleanup_reserve
                )
                if execution_budget <= 0:
                    _terminate_process_group(
                        process,
                        grace=cleanup_reserve / 2,
                    )
                    raise RestartGateTimeout(
                        "restart operation has no process cleanup budget"
                    )
                try:
                    exit_code = process.wait(timeout=execution_budget)
                except subprocess.TimeoutExpired as exc:
                    _terminate_process_group(
                        process,
                        grace=cleanup_reserve / 2,
                    )
                    raise RestartGateTimeout(
                        "restart operation exceeded shared deadline"
                    ) from exc
        except BaseException:
            # fdopen owns the descriptor only after successful construction.
            try:
                os.close(log_fd)
            except OSError:
                pass
            raise
        return int(exit_code)

    result = gate.run_if_current(
        expected,
        timeout=remaining_budget,
        operation=operation,
    )
    _append_worker_log(
        log_path,
        (
            "remote worker done "
            f"status={result.status.value} exit_code={result.exit_code}"
        ),
    )
    return result.exit_code


def _add_worker_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--gate-dir")
    parser.add_argument("--restart-script", required=True)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--delay", required=True, type=float)
    parser.add_argument("--expected-generation", required=True)
    parser.add_argument("--timeout", type=float)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GhostAP safe restart gate")
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker = subparsers.add_parser("worker")
    _add_worker_arguments(worker)

    launch_wrapper = subparsers.add_parser("launch-wrapper")
    _add_worker_arguments(launch_wrapper)
    launch_wrapper.add_argument("--launchd-label", required=True)

    publish = subparsers.add_parser("publish")
    publish.add_argument("--project-dir", required=True)
    publish.add_argument("--gate-dir")
    publish.add_argument("--service-pid", type=int)

    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--project-dir", required=True)
    snapshot.add_argument("--gate-dir")

    ready = subparsers.add_parser("ready")
    ready.add_argument("--project-dir", required=True)
    ready.add_argument("--gate-dir")
    ready.add_argument("--service-pid", required=True, type=int)
    return parser


def _remove_launchd_job(label: str) -> None:
    if not _LAUNCHD_LABEL_PATTERN.fullmatch(label):
        raise RestartGateError("invalid launchd restart label")
    if sys.platform != "darwin":
        return
    try:
        subprocess.run(
            ["/bin/launchctl", "remove", label],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
            close_fds=True,
            timeout=5,
            check=False,
        )
    except Exception:
        logger.exception("failed to remove launchd restart job %s", label)


def _log_cli_failure(args: argparse.Namespace, exc: Exception) -> None:
    log_value = getattr(args, "log_file", None)
    if not log_value:
        return
    try:
        log_path = _absolute_path(log_value, label="restart log")
        detail = str(exc).replace("\r", " ").replace("\n", " ")[:500]
        _append_worker_log(
            log_path,
            f"remote worker bootstrap failed error={type(exc).__name__}: {detail}",
        )
    except Exception:
        logger.exception("failed to append safe restart bootstrap error")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    launch_wrapper = args.command == "launch-wrapper"
    original_cwd = Path.cwd()
    changed_cwd = False
    try:
        project_dir = _absolute_path(
            args.project_dir,
            label="project directory",
        )
        os.chdir(project_dir)
        changed_cwd = True
        # Keep environment/.env interpretation in the central Settings model.
        # This local import avoids a runtime dependency cycle when ws_client
        # imports RestartGate for production scheduler construction.
        from src.config import get_settings

        settings = get_settings()
        configured_dir = args.gate_dir or settings.restart_gate_dir or None
        if args.command in {"worker", "launch-wrapper"}:
            gate = RestartGate.for_worker(
                project_dir,
                expected_generation=args.expected_generation,
                configured_override=configured_dir,
            )
        elif args.command == "ready":
            gate = RestartGate.from_locator(project_dir)
        else:
            gate = RestartGate.for_project(
                project_dir,
                override=configured_dir,
            )
        if args.command == "publish":
            gate.publish_participation(service_pid=args.service_pid)
            return 0
        if args.command == "ready":
            print(
                gate.ready_generation(service_pid=args.service_pid),
                flush=True,
            )
            return 0
        if args.command == "snapshot":
            print(gate.snapshot(), flush=True)
            return 0
        timeout = (
            args.timeout
            if args.timeout is not None
            else settings.restart_gate_timeout
        )
        worker_exit = run_restart_worker(
            gate=gate,
            expected_generation=args.expected_generation,
            project_dir=project_dir,
            restart_script=args.restart_script,
            log_file=args.log_file,
            delay=args.delay,
            timeout=timeout,
        )
        if launch_wrapper:
            _append_worker_log(
                _absolute_path(args.log_file, label="restart log"),
                f"launch wrapper done worker_exit={worker_exit}",
            )
            return 0
        return worker_exit
    except Exception as exc:
        logger.exception("safe restart worker failed")
        _log_cli_failure(args, exc)
        return 0 if launch_wrapper else EX_TEMPFAIL
    finally:
        if changed_cwd:
            try:
                os.chdir(original_cwd)
            except OSError:
                logger.exception(
                    "failed to restore working directory after restart gate CLI"
                )
        if launch_wrapper:
            try:
                _remove_launchd_job(args.launchd_label)
            except Exception:
                logger.exception("failed to clean launchd restart wrapper")


if __name__ == "__main__":  # pragma: no cover - exercised through shell entrypoint
    raise SystemExit(main())
