"""Cross-process admission and drain gate for safe service restarts.

Tasks take the admission and drain locks shared.  A restart takes admission
exclusive before draining active work, so new work cannot enter while the
restart waits.  Lock files are stable and are never replaced or removed.
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
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
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
_MAX_MARKER_BYTES = 8192
_PARTICIPATING = "participating"
_READY = "ready"
_FAILED = "failed"
_VALID_STATES = frozenset({_PARTICIPATING, _READY, _FAILED})
_BROAD_GATE_DIRECTORIES = frozenset(
    {Path("/"), Path("/tmp"), Path("/var/tmp"), Path("/private/tmp"), Path("/dev/shm")}
)


class RestartGateError(RuntimeError):
    """A restart request cannot proceed safely."""


class RestartGateTimeout(RestartGateError):
    """The shared restart deadline expired."""


def _private_flags(base: int) -> int:
    return base | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _validate_token(token: str) -> str:
    if not _TOKEN_PATTERN.fullmatch(token):
        raise RestartGateError("invalid restart generation token")
    return token


def _new_token() -> str:
    # Keep the first character alphanumeric so argparse never treats a token as
    # another option.
    return _validate_token(f"g{secrets.token_urlsafe(23)}")


def _absolute_path(path: str | os.PathLike[str], *, label: str) -> Path:
    value = Path(path)
    if not value.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return Path(os.path.abspath(value))


def _dedicated_gate_path(path: str | os.PathLike[str], *, label: str) -> Path:
    value = _absolute_path(path, label=label)
    if value in _BROAD_GATE_DIRECTORIES or len(value.parts) <= 2 or value == Path.home():
        raise RestartGateError(f"{label} must be a dedicated private directory, not {value}")
    return value


def _canonical_project_path(path: str | os.PathLike[str]) -> Path:
    project = _absolute_path(path, label="project directory")
    try:
        project = project.resolve(strict=True)
        metadata = project.stat()
    except OSError as exc:
        raise RestartGateError(f"cannot resolve project directory: {project}") from exc
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
        raise RestartGateError(f"participation service PID is not alive: {pid}") from exc

    if sys.platform == "linux":
        try:
            raw_stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
            start_ticks = raw_stat.rsplit(")", 1)[1].split()[19]
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
        raise RestartGateError(f"cannot identify participation service process: {pid}") from exc
    started_at = completed.stdout.strip()
    if completed.returncode != 0 or not started_at:
        raise RestartGateError(f"participation service PID is not alive: {pid}")
    return f"darwin:{started_at}"


def _read_private_json(
    path: Path, *, label: str, allow_missing: bool = False
) -> dict[str, object] | None:
    try:
        fd = os.open(path, _private_flags(os.O_RDONLY))
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
            or metadata.st_size > _MAX_MARKER_BYTES
        ):
            raise RestartGateError(f"invalid {label} file")
        try:
            path_metadata = path.lstat()
        except OSError as exc:
            raise RestartGateError(f"cannot verify {label} path") from exc
        if _identity(path_metadata) != _identity(metadata):
            raise RestartGateError(f"{label} path changed while opening")
        raw = os.read(fd, _MAX_MARKER_BYTES + 1)
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
    fd = os.open(directory, _private_flags(os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)))
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
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    temp_path = path.parent / f".{path.name}.{_new_token()}.tmp"
    fd: int | None = None
    try:
        fd = os.open(
            temp_path,
            _private_flags(os.O_WRONLY | os.O_CREAT | os.O_EXCL),
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


def _prepare_private_directory(
    directory: Path, *, label: str, parents: bool = False
) -> None:
    created = False
    try:
        directory.mkdir(mode=0o700, parents=parents, exist_ok=False)
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
    """Checkout-scoped restart state machine and cross-process gate."""

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        project_dir: Path,
        expected_owner_id: str | None = None,
    ) -> None:
        if sys.platform not in _SUPPORTED_PLATFORMS:
            raise RestartGateError(
                f"restart gate requires POSIX flock; unsupported platform: {sys.platform}"
            )
        self.directory = _dedicated_gate_path(directory, label="restart gate directory")
        _prepare_private_directory(
            self.directory, label="restart gate directory", parents=True
        )
        self.project_dir = project_dir
        self.owner_path = self.directory / "owner.json"
        self.state_path = self.directory / "state.json"
        self.admission_path = self.directory / "admission.lock"
        self.drain_path = self.directory / "drain.lock"
        self._expected_owner_id = expected_owner_id
        self._pinned_participation_id: str | None = None

        identities: list[list[int]] = []
        for path in (self.admission_path, self.drain_path):
            fd = self._open_lock(path)
            try:
                identities.append(_identity(os.fstat(fd)))
            finally:
                os.close(fd)
        if identities[0] == identities[1]:
            raise RestartGateError("restart admission and drain locks must use distinct inodes")
        self._directory_identity = _identity(self.directory.lstat())
        self._lock_identities = {
            "admission_identity": identities[0],
            "drain_identity": identities[1],
        }
        self._bind_owner()

    @classmethod
    def for_project(
        cls,
        project_dir: str | os.PathLike[str],
        *,
        override: str | os.PathLike[str] | None = None,
    ) -> RestartGate:
        """Open or create the gate permanently pinned to one checkout."""

        project = _canonical_project_path(project_dir)
        registry = cls._registry_directory(project)
        _prepare_private_directory(
            registry.parent, label="restart gate registry root"
        )
        _prepare_private_directory(registry, label="restart gate registry")
        locator_path = registry / "locator.json"
        requested = (
            _dedicated_gate_path(override, label="restart gate directory override")
            if override
            else None
        )
        if requested == project:
            raise RestartGateError(
                "restart gate directory override must not be the project root"
            )

        locator = cls._read_locator(locator_path, project, allow_missing=True)
        if locator is not None:
            pinned = _dedicated_gate_path(
                str(locator["gate_dir"]), label="pinned restart gate directory"
            )
            if requested is not None and requested != pinned:
                raise RestartGateError(
                    "restart gate override conflicts with the checkout's pinned gate identity"
                )
            return cls(
                pinned,
                project_dir=project,
                expected_owner_id=str(locator["owner_id"]),
            )

        gate = cls(requested or registry / "gate", project_dir=project)
        candidate: dict[str, object] = {
            "version": _IDENTITY_VERSION,
            "project_dir": str(project),
            "gate_dir": str(gate.directory),
            "owner_id": gate._expected_owner_id,
        }
        if not _publish_private_json(
            locator_path,
            candidate,
            label="restart gate locator",
            create_only=True,
        ):
            winner = cls._read_locator(locator_path, project, allow_missing=False)
            assert winner is not None
            if (
                winner.get("gate_dir") != str(gate.directory)
                or winner.get("owner_id") != gate._expected_owner_id
            ):
                raise RestartGateError(
                    "concurrent restart gate locator bound a different identity"
                )
        return gate

    @classmethod
    def from_locator(
        cls,
        project_dir: str | os.PathLike[str],
        *,
        expected_generation: str | None = None,
    ) -> RestartGate:
        """Open the pinned gate without consulting mutable settings."""

        project = _canonical_project_path(project_dir)
        locator = cls._read_locator(
            cls._registry_directory(project) / "locator.json",
            project,
            allow_missing=False,
        )
        assert locator is not None
        gate = cls(
            str(locator["gate_dir"]),
            project_dir=project,
            expected_owner_id=str(locator["owner_id"]),
        )
        if expected_generation is not None:
            expected = _validate_token(expected_generation)
            state = gate._read_state()
            if state["generation"] == expected and state["status"] == _READY:
                gate._pinned_participation_id = str(state["participation_id"])
        return gate

    @staticmethod
    def _registry_directory(project: Path) -> Path:
        digest = hashlib.sha256(os.fsencode(str(project))).hexdigest()[:32]
        return project.parent / ".ghostap-restart-gates" / digest

    @staticmethod
    def _read_locator(
        path: Path, project: Path, *, allow_missing: bool
    ) -> dict[str, object] | None:
        locator = _read_private_json(
            path, label="restart gate locator", allow_missing=allow_missing
        )
        if locator is None:
            return None
        if (
            locator.get("version") != _IDENTITY_VERSION
            or locator.get("project_dir") != str(project)
        ):
            raise RestartGateError("restart gate locator belongs to a different checkout")
        try:
            _validate_token(str(locator["owner_id"]))
            _dedicated_gate_path(
                str(locator["gate_dir"]), label="pinned restart gate directory"
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RestartGateError("invalid restart gate locator") from exc
        return locator

    def _bind_owner(self) -> None:
        owner = _read_private_json(
            self.owner_path, label="restart gate owner", allow_missing=True
        )
        if owner is None:
            candidate: dict[str, object] = {
                "version": _IDENTITY_VERSION,
                "project_dir": str(self.project_dir),
                "gate_dir": str(self.directory),
                "owner_id": self._expected_owner_id or _new_token(),
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
                    self.owner_path, label="restart gate owner", allow_missing=False
                )
                assert owner is not None

        if (
            owner.get("version") != _IDENTITY_VERSION
            or owner.get("project_dir") != str(self.project_dir)
            or owner.get("gate_dir") != str(self.directory)
        ):
            raise RestartGateError("restart gate is bound to a different checkout")
        try:
            owner_id = _validate_token(str(owner["owner_id"]))
        except (KeyError, TypeError) as exc:
            raise RestartGateError("invalid restart gate owner identity") from exc
        if self._expected_owner_id is not None and owner_id != self._expected_owner_id:
            raise RestartGateError("restart gate owner identity changed")
        self._expected_owner_id = owner_id

    def _open_lock(self, path: Path) -> int:
        flags = _private_flags(os.O_RDWR)
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
                    raise RestartGateError(f"cannot open restart lock: {path}") from exc
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
                raise RestartGateError(f"restart lock is not owned by current user: {path}")
            if metadata.st_nlink != 1:
                raise RestartGateError(f"restart lock must be a single-link file: {path}")
            if created:
                os.fchmod(fd, 0o600)
                metadata = os.fstat(fd)
            elif stat.S_IMODE(metadata.st_mode) & 0o077:
                raise RestartGateError(f"restart lock has unsafe permissions: {path}")
            try:
                path_metadata = path.lstat()
            except OSError as exc:
                raise RestartGateError(f"cannot verify restart lock path: {path}") from exc
            if _identity(path_metadata) != _identity(metadata):
                raise RestartGateError(f"restart lock path changed while opening: {path}")
            return fd
        except BaseException:
            os.close(fd)
            raise

    @staticmethod
    def _close_lock(fd: int) -> None:
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
        canceled: Callable[[], bool] | None = None,
    ) -> None:
        while True:
            if canceled is not None and canceled():
                raise RestartGateError("restart gate task admission canceled")
            try:
                fcntl.flock(fd, mode | fcntl.LOCK_NB)
                if canceled is not None and canceled():
                    raise RestartGateError("restart gate task admission canceled")
                return
            except BlockingIOError:
                if deadline is None:
                    time.sleep(_LOCK_POLL_SECONDS)
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RestartGateTimeout("restart gate deadline expired")
                time.sleep(min(_LOCK_POLL_SECONDS, remaining))
            except OSError as exc:
                if exc.errno != errno.EINTR:
                    raise RestartGateError("cannot acquire restart gate lock") from exc

    @contextmanager
    def _lock_pair(
        self,
        *,
        exclusive: bool,
        deadline: float | None = None,
        canceled: Callable[[], bool] | None = None,
    ) -> Iterator[None]:
        admission_fd: int | None = self._open_lock(self.admission_path)
        drain_fd: int | None = None
        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            self._acquire(admission_fd, mode, deadline, canceled)
            drain_fd = self._open_lock(self.drain_path)
            self._acquire(drain_fd, mode, deadline, canceled)
            if not exclusive:
                self._close_lock(admission_fd)
                admission_fd = None
            if canceled is not None and canceled():
                raise RestartGateError("restart gate task admission canceled")
            yield
        finally:
            if drain_fd is not None:
                self._close_lock(drain_fd)
            if admission_fd is not None:
                self._close_lock(admission_fd)

    @contextmanager
    def task_guard(self) -> Iterator[None]:
        """Hold the drain reader lock for one complete scheduler run."""

        with self._lock_pair(exclusive=False):
            yield

    @contextmanager
    def cancellable_task_guard(
        self,
        *,
        canceled: Callable[[], bool],
        deadline: float | None,
    ) -> Iterator[None]:
        """Hold the task lock while honoring one run's cancellation boundary."""

        with self._lock_pair(
            exclusive=False,
            deadline=deadline,
            canceled=canceled,
        ):
            yield

    def _verify_gate_identity(self) -> None:
        try:
            directory = self.directory.lstat()
            admission = self.admission_path.lstat()
            drain = self.drain_path.lstat()
        except OSError as exc:
            raise RestartGateError("restart gate identity changed") from exc
        if (
            _identity(directory) != self._directory_identity
            or _identity(admission) != self._lock_identities["admission_identity"]
            or _identity(drain) != self._lock_identities["drain_identity"]
            or not stat.S_ISDIR(directory.st_mode)
            or not stat.S_ISREG(admission.st_mode)
            or not stat.S_ISREG(drain.st_mode)
        ):
            raise RestartGateError("restart gate identity changed")

    def _validate_state(
        self,
        value: dict[str, object],
        *,
        require_live: bool,
        check_pin: bool,
    ) -> dict[str, object]:
        self._verify_gate_identity()
        self._bind_owner()
        state = dict(value)
        expected = {
            "version": _IDENTITY_VERSION,
            "project_dir": str(self.project_dir),
            "gate_dir": str(self.directory),
            "owner_id": self._expected_owner_id,
            "directory_identity": self._directory_identity,
            **self._lock_identities,
        }
        for field, expected_value in expected.items():
            if state.get(field) != expected_value:
                raise RestartGateError(f"restart state {field} identity changed")
        try:
            state["generation"] = _validate_token(str(state["generation"]))
            state["participation_id"] = _validate_token(
                str(state["participation_id"])
            )
            state["service_pid"] = int(state["service_pid"])
            process_instance = str(state["process_instance"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RestartGateError("invalid restart state") from exc
        status = state.get("status")
        if status not in _VALID_STATES:
            raise RestartGateError("invalid restart state")
        if status == _FAILED:
            try:
                state["exit_code"] = int(state["exit_code"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RestartGateError("invalid failed restart state") from exc
        elif state.get("exit_code") is not None:
            raise RestartGateError("invalid restart state exit code")
        if (
            check_pin
            and self._pinned_participation_id is not None
            and state["participation_id"] != self._pinned_participation_id
        ):
            raise RestartGateError("restart participation instance identity changed")
        if require_live:
            actual_process = _process_instance_identity(int(state["service_pid"]))
            if process_instance != actual_process:
                raise RestartGateError(
                    "restart participation process instance identity changed"
                )
        return state

    def _migrate_legacy_state(self) -> dict[str, object]:
        """Atomically retire the former participation/generation split."""

        proof = _read_private_json(
            self.directory / "participation.json",
            label="legacy restart participation proof",
            allow_missing=True,
        )
        generation = _read_private_json(
            self.directory / "generation",
            label="legacy restart generation",
            allow_missing=True,
        )
        if proof is None and generation is None:
            raise RestartGateError("restart state is missing")
        if proof is None or generation is None:
            raise RestartGateError("legacy restart state is incomplete")
        candidate = {
            **proof,
            "generation": generation.get("generation"),
            "participation_id": generation.get("participation_id"),
            "status": generation.get("status"),
            "exit_code": generation.get("exit_code"),
        }
        candidate = self._validate_state(
            candidate, require_live=True, check_pin=False
        )
        if _publish_private_json(
            self.state_path,
            candidate,
            label="restart state",
            create_only=True,
        ):
            return candidate
        winner = _read_private_json(
            self.state_path, label="restart state", allow_missing=False
        )
        assert winner is not None
        return self._validate_state(winner, require_live=True, check_pin=False)

    def _read_state(
        self, *, require_live: bool = True, check_pin: bool = True
    ) -> dict[str, object]:
        value = _read_private_json(
            self.state_path, label="restart state", allow_missing=True
        )
        if value is None:
            value = self._migrate_legacy_state()
        return self._validate_state(
            value, require_live=require_live, check_pin=check_pin
        )

    def _publish_state(
        self, value: dict[str, object], *, require_live: bool
    ) -> dict[str, object]:
        state = self._validate_state(
            value, require_live=require_live, check_pin=False
        )
        _publish_private_json(self.state_path, state, label="restart state")
        return state

    def publish_participation(self, *, service_pid: int) -> str:
        """Publish a new service identity in the pre-ready state."""

        self._verify_gate_identity()
        self._bind_owner()
        pid = int(service_pid)
        participation_id = _new_token()
        state: dict[str, object] = {
            "version": _IDENTITY_VERSION,
            "project_dir": str(self.project_dir),
            "gate_dir": str(self.directory),
            "owner_id": self._expected_owner_id,
            "directory_identity": self._directory_identity,
            **self._lock_identities,
            "service_pid": pid,
            "process_instance": _process_instance_identity(pid),
            "participation_id": participation_id,
            "generation": _new_token(),
            "status": _PARTICIPATING,
            "exit_code": None,
        }
        self._publish_state(state, require_live=True)
        return participation_id

    def mark_ready(self, *, service_pid: int) -> str:
        """Transition the current participating service to ready."""

        state = self._read_state()
        if int(state["service_pid"]) != int(service_pid):
            raise RestartGateError(
                "restart participation service PID changed before readiness"
            )
        if state["status"] == _FAILED:
            raise RestartGateError("restart generation is a failed terminal")
        if state["status"] == _READY:
            return str(state["generation"])
        state.update(generation=_new_token(), status=_READY, exit_code=None)
        return str(self._publish_state(state, require_live=True)["generation"])

    def _ready_state(self, *, service_pid: int | None = None) -> dict[str, object]:
        state = self._read_state()
        if service_pid is not None and int(state["service_pid"]) != int(service_pid):
            raise RestartGateError(
                "restart participation service PID does not match readiness check"
            )
        if state["status"] == _FAILED:
            raise RestartGateError("restart generation is a failed terminal")
        if state["status"] != _READY:
            raise RestartGateError("restart participation is not ready")
        return state

    def ready_generation(self, *, service_pid: int) -> str:
        return str(self._ready_state(service_pid=service_pid)["generation"])

    def snapshot(self) -> str:
        return str(self._ready_state()["generation"])

    def _fail_if_unchanged(
        self, state: dict[str, object], *, expected: str, exit_code: int
    ) -> None:
        if state["generation"] != expected or state["status"] != _READY:
            return
        state.update(status=_FAILED, exit_code=int(exit_code))
        self._publish_state(state, require_live=False)

    def _run_if_current(
        self,
        expected: str,
        *,
        timeout: float,
        operation: Callable[[float], int],
    ) -> tuple[str, int]:
        """Run exactly one restart and require a new live ready generation."""

        expected = _validate_token(expected)
        if not math.isfinite(timeout) or timeout <= 0:
            return "timed_out", EX_TEMPFAIL
        deadline = time.monotonic() + timeout
        try:
            with self._lock_pair(exclusive=True, deadline=deadline):
                current = self._read_state(check_pin=False)
                if current["generation"] != expected or current["status"] != _READY:
                    return "coalesced", 0
                if (
                    self._pinned_participation_id is not None
                    and current["participation_id"] != self._pinned_participation_id
                ):
                    raise RestartGateError(
                        "restart participation instance identity changed"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return "timed_out", EX_TEMPFAIL
                try:
                    exit_code = int(operation(remaining))
                except RestartGateTimeout:
                    return "timed_out", EX_TEMPFAIL

                after = self._read_state(require_live=False, check_pin=False)
                if exit_code != 0:
                    self._fail_if_unchanged(
                        after, expected=expected, exit_code=exit_code
                    )
                    return "failed", exit_code
                if after["generation"] == expected or after["status"] != _READY:
                    self._fail_if_unchanged(
                        after, expected=expected, exit_code=EX_TEMPFAIL
                    )
                    return "failed", EX_TEMPFAIL
                self._validate_state(after, require_live=True, check_pin=False)
                return "restarted", 0
        except RestartGateTimeout:
            return "timed_out", EX_TEMPFAIL


def _append_worker_log(path: Path, message: str) -> None:
    fd = os.open(
        path,
        _private_flags(os.O_WRONLY | os.O_CREAT | os.O_APPEND),
        0o600,
    )
    try:
        os.set_inheritable(fd, False)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        os.write(fd, f"{stamp} [RESTART] {message}\n".encode("utf-8"))
    finally:
        os.close(fd)


def _terminate_process_group(process: subprocess.Popen[bytes], *, grace: float) -> None:
    """Terminate and reap a timed-out restart command and all descendants."""

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        try:
            process.wait(timeout=max(grace, _MIN_PROCESS_GROUP_PHASE_SECONDS))
        except subprocess.TimeoutExpired as exc:
            raise RestartGateTimeout(
                "restart process did not exit before cleanup deadline"
            ) from exc
        return
    parent_reaped = False
    try:
        process.wait(timeout=max(grace, _MIN_PROCESS_GROUP_PHASE_SECONDS))
        parent_reaped = True
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return
    os.killpg(process.pid, signal.SIGKILL)
    if not parent_reaped:
        try:
            process.wait(timeout=max(grace, _MIN_PROCESS_GROUP_PHASE_SECONDS))
        except subprocess.TimeoutExpired as exc:
            raise RestartGateTimeout(
                "restart process group did not exit before cleanup deadline"
            ) from exc


def run_restart_worker(
    *,
    gate: RestartGate,
    expected_generation: str,
    restart_script: str | os.PathLike[str],
    log_file: str | os.PathLike[str],
    delay: float,
    timeout: float,
    project_dir: str | os.PathLike[str],
) -> int:
    """Run a preflight-snapshotted generation through the restart gate."""

    script = _absolute_path(restart_script, label="restart script")
    log_path = _absolute_path(log_file, label="restart log")
    cwd = _absolute_path(project_dir, label="project directory")
    expected = _validate_token(expected_generation)
    if not math.isfinite(delay) or delay < 0:
        raise ValueError("restart delay must be >= 0")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("restart timeout must be finite and > 0")

    deadline = time.monotonic() + timeout
    _append_worker_log(
        log_path, f"remote worker scheduled generation={expected} delay={delay:.3f}s"
    )
    if delay:
        time.sleep(min(delay, timeout))
    remaining_budget = deadline - time.monotonic()
    if remaining_budget <= 0:
        _append_worker_log(
            log_path, f"remote worker done status=timed_out exit_code={EX_TEMPFAIL}"
        )
        return EX_TEMPFAIL
    _append_worker_log(
        log_path,
        f"remote worker script begin generation={expected} budget={remaining_budget:.3f}s",
    )

    def operation(remaining: float) -> int:
        operation_deadline = time.monotonic() + remaining
        environment = os.environ.copy()
        environment["GHOSTAP_LOG_MODE"] = "append"
        environment["GHOSTAP_RESTART_GATE_DIR"] = str(gate.directory)
        log_fd = os.open(
            log_path,
            _private_flags(os.O_WRONLY | os.O_CREAT | os.O_APPEND),
            0o600,
        )
        try:
            os.set_inheritable(log_fd, False)
            with os.fdopen(log_fd, "ab", closefd=True) as stream:
                cleanup_reserve = min(
                    _PROCESS_GROUP_CLEANUP_BUDGET_SECONDS,
                    max(_MIN_PROCESS_GROUP_PHASE_SECONDS * 2, remaining * 0.1),
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
                    _terminate_process_group(process, grace=cleanup_reserve / 2)
                    raise RestartGateTimeout(
                        "restart operation has no process cleanup budget"
                    )
                try:
                    return int(process.wait(timeout=execution_budget))
                except subprocess.TimeoutExpired as exc:
                    _terminate_process_group(process, grace=cleanup_reserve / 2)
                    raise RestartGateTimeout(
                        "restart operation exceeded shared deadline"
                    ) from exc
        except BaseException:
            try:
                os.close(log_fd)
            except OSError:
                pass
            raise

    status, exit_code = gate._run_if_current(
        expected, timeout=remaining_budget, operation=operation
    )
    _append_worker_log(
        log_path, f"remote worker done status={status} exit_code={exit_code}"
    )
    return exit_code


def _add_worker_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--restart-script", required=True)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--delay", required=True, type=float)
    parser.add_argument("--expected-generation", required=True)
    parser.add_argument("--timeout", type=float)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GhostAP safe restart gate")
    commands = parser.add_subparsers(dest="command", required=True)
    _add_worker_arguments(commands.add_parser("worker"))
    launch_wrapper = commands.add_parser("launch-wrapper")
    _add_worker_arguments(launch_wrapper)
    launch_wrapper.add_argument("--launchd-label", required=True)
    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("--project-dir", required=True)
    ready = commands.add_parser("ready")
    ready.add_argument("--project-dir", required=True)
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
        detail = str(exc).replace("\r", " ").replace("\n", " ")[:500]
        _append_worker_log(
            _absolute_path(log_value, label="restart log"),
            f"remote worker bootstrap failed error={type(exc).__name__}: {detail}",
        )
    except Exception:
        logger.exception("failed to append safe restart bootstrap error")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    launch_wrapper = args.command == "launch-wrapper"
    try:
        project_dir = _canonical_project_path(args.project_dir)
        os.chdir(project_dir)
        expected = getattr(args, "expected_generation", None)
        gate = RestartGate.from_locator(
            project_dir,
            expected_generation=expected if expected is not None else None,
        )
        if args.command == "ready":
            print(gate.ready_generation(service_pid=args.service_pid), flush=True)
            return 0
        if args.command == "snapshot":
            print(gate.snapshot(), flush=True)
            return 0

        timeout = args.timeout
        if timeout is None:
            # Settings remains the single source for the configurable budget;
            # the lock and marker identity always come from the immutable locator.
            from src.config import get_settings

            timeout = get_settings().restart_gate_timeout
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
        if launch_wrapper:
            try:
                _remove_launchd_job(args.launchd_label)
            except Exception:
                logger.exception("failed to clean launchd restart wrapper")


if __name__ == "__main__":  # pragma: no cover - shell entrypoint
    raise SystemExit(main())
