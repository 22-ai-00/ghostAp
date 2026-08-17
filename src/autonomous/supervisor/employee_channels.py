"""Parent-side owner for one fresh Channel interpreter per employee."""

from __future__ import annotations

import json
import logging
import math
import os
import secrets
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Protocol

from src.autonomous.ingress.models import (
    EmployeeIngressAck,
    EmployeeIngressMetadata,
    EmployeeIngressPayload,
)
from src.autonomous.ingress.service import EmployeeIngressService
from src.autonomous.provisioning.channel_protocol import (
    MAX_FRAME_BYTES,
    ChannelBootstrap,
    ChannelFrame,
    FrameType,
    ProtocolError,
    decode_frame,
    encode_bootstrap,
    encode_frame,
    validate_ready_payload,
)
from src.autonomous.supervisor.channel_models import (
    ChannelProcessState,
    EmployeeChannelGenerationChanged,
    EmployeeChannelOutboundError,
    EmployeeChannelOutboundIntegrityError,
    EmployeeChannelOutboundTimeout,
)

logger = logging.getLogger(__name__)

_SANDBOX_METADATA_MAX_BYTES = 4096
_SANDBOX_METADATA_TIMEOUT_SECONDS = 10.0
_EVENT_PIPE_EXIT_GRACE_SECONDS = 0.1
_MACOS_SEATBELT_PROFILE = """
(version 1)
(deny default)
(allow process-exec
    (literal (param "GHOSTAP_PYTHON"))
    (literal (param "GHOSTAP_PYTHON_REAL")))
(allow process-info* (target self))
(allow signal (target self))
(allow sysctl-read)
(allow mach-lookup)
(allow system-socket)
(allow network-outbound)
(allow file-read*
    (subpath (param "GHOSTAP_SOURCE_ROOT"))
    (literal (param "GHOSTAP_WORKER"))
    (subpath (param "GHOSTAP_RUNTIME_PREFIX"))
    (subpath (param "GHOSTAP_BASE_PREFIX"))
    (literal (param "GHOSTAP_PYTHON"))
    (literal (param "GHOSTAP_PYTHON_REAL"))
    (literal (param "GHOSTAP_PYPROJECT"))
    (literal (param "GHOSTAP_UV_LOCK"))
    (subpath (param "GHOSTAP_TEMP"))
    (subpath "/System/Library")
    (subpath "/Library/Frameworks")
    (subpath "/usr/lib")
    (subpath "/usr/share")
    (subpath "/private/etc")
    (subpath "/private/var/db/timezone")
    (literal "/private/var/run/mDNSResponder")
    (subpath "/dev"))
(allow file-write* (subpath (param "GHOSTAP_TEMP")))
""".strip()


class ChannelSandboxUnavailable(RuntimeError):
    """No verified per-employee OS isolation boundary is available."""

    def __init__(self) -> None:
        super().__init__("employee Channel sandbox unavailable")


class ChannelLaunchUnavailable(RuntimeError):
    """One employee Channel launch failed without invalidating other identities."""


@dataclass(frozen=True, slots=True)
class SandboxAttestation:
    pid: int
    verified: bool
    mechanism: str
    details: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _SandboxAttempt:
    prefix: tuple[str, ...]
    mechanism: str
    bwrap_info: bool = False
    seatbelt_proof: bool = False
    fallback: bool = False


@dataclass(frozen=True, slots=True)
class ChannelProcessStatus:
    agent_id: str
    app_id: str
    generation: int
    pid: int
    state: ChannelProcessState
    tenant_key: str = ""
    bot_principal_id: str = ""
    identity: dict[str, Any] = field(default_factory=dict)
    ready_metadata: dict[str, Any] = field(default_factory=dict)
    sandbox: SandboxAttestation | None = None
    started_at: float = field(default_factory=time.time)
    ready_at: float | None = None
    stopped_at: float | None = None
    exit_code: int | None = None
    error_code: str = ""
    stale_frames: int = 0
    delivery_only: bool = False


@dataclass(frozen=True, slots=True)
class ChannelSendReceipt:
    request_id: str
    success: bool
    app_id: str
    generation: int
    connection_id: str
    message_id: str


@dataclass(slots=True)
class _PendingSend:
    completed: threading.Event = field(default_factory=threading.Event)
    success: bool | None = None
    app_id: str = ""
    generation: int = 0
    connection_id: str = ""
    message_id: str = ""
    operation: str = "send"
    expected_message_id: str = ""
    failure_kind: str = ""
    error_code: str = ""


class SandboxAttestor(Protocol):
    def __call__(self, pid: int) -> SandboxAttestation: ...


@dataclass(slots=True)
class _Runtime:
    process: subprocess.Popen[bytes]
    control_fd: int
    event_fd: int
    status: ChannelProcessStatus
    on_event: Callable[[dict[str, Any]], None]
    tenant_key: str = ""
    bot_principal_id: str = ""
    delivery_only: bool = False
    frozen_connection_id: str = ""
    requires_observed_connection: bool = False
    ready: threading.Event = field(default_factory=threading.Event)
    reader: threading.Thread | None = None
    stopping: bool = False
    outbound_sequence: int = 0
    inbound_sequence: int = 0
    pending_lock: threading.Lock = field(default_factory=threading.Lock)
    pending_sends: dict[str, _PendingSend] = field(default_factory=dict)
    control_lock: threading.Lock = field(default_factory=threading.Lock)
    sandbox_temp_dir: Path | None = None


class EmployeeChannelSupervisor:
    """Own employee Channel children without sharing SDK process globals."""

    def __init__(
        self,
        *,
        secret_resolver: Callable[[str, str, str], str],
        ready_timeout: float = 30.0,
        stop_timeout: float = 5.0,
        send_timeout: float = 10.0,
        worker_path: str | Path | None = None,
        launcher: Callable[..., subprocess.Popen[bytes]] | None = None,
        sandbox_attestor: SandboxAttestor | None = None,
        sandbox_prefix: tuple[str, ...] | None = None,
        platform_name: str | None = None,
        ingress_service: EmployeeIngressService | None = None,
        ingress_binding_resolver: Callable[[str, str], tuple[str, str]] | None = None,
        ingress_ack_timeout: float = 1.5,
    ) -> None:
        if (
            isinstance(send_timeout, bool)
            or not isinstance(send_timeout, (int, float))
            or not math.isfinite(float(send_timeout))
            or float(send_timeout) <= 0
        ):
            raise ValueError("send_timeout must be positive and finite")
        self._secret_resolver = secret_resolver
        self._ready_timeout = ready_timeout
        self._stop_timeout = stop_timeout
        self._send_timeout = float(send_timeout)
        self._worker_path = (
            Path(worker_path).resolve()
            if worker_path is not None
            else Path(__file__).resolve().parents[1] / "provisioning" / "channel_worker.py"
        ).resolve()
        self._production_worker = worker_path is None
        if (ingress_service is None) != (ingress_binding_resolver is None):
            raise ValueError("durable ingress service and binding resolver must be configured together")
        if (
            isinstance(ingress_ack_timeout, bool)
            or not isinstance(ingress_ack_timeout, (int, float))
            or not 0 < float(ingress_ack_timeout) < 3.0
        ):
            raise ValueError("invalid employee ingress ACK timeout")
        self._ingress_service = ingress_service
        self._ingress_binding_resolver = ingress_binding_resolver
        self._ingress_ack_timeout = float(ingress_ack_timeout)
        self._launcher = launcher or subprocess.Popen
        if platform_name is not None and platform_name != sys.platform and worker_path is None and launcher is None:
            raise ValueError("platform override requires a test launcher or worker")
        self._platform_name = platform_name or sys.platform
        self._automatic_process_fallback = (
            sandbox_attestor is None and sandbox_prefix is None and self._platform_name == "linux"
        )
        self._sandbox_attestor = sandbox_attestor or attest_process_sandbox
        if sandbox_prefix is not None:
            self._sandbox_kind = "custom"
            self._sandbox_prefix = sandbox_prefix
        elif sandbox_attestor is not None:
            self._sandbox_kind = "custom"
            self._sandbox_prefix = ()
        elif self._platform_name == "linux":
            self._sandbox_kind = "linux-bwrap"
            self._sandbox_prefix = self._bwrap_prefix()
        elif self._platform_name == "darwin":
            self._sandbox_kind = "macos-seatbelt"
            self._sandbox_prefix = self._seatbelt_prefix()
        else:
            self._sandbox_kind = "unavailable"
            self._sandbox_prefix = ()
        self._runtimes: dict[str, _Runtime] = {}
        self._generation_high_watermark: dict[str, int] = {}
        self._lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        self._lifecycle_registry_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._lifecycle_locks: dict[str, threading.RLock] = {}
        self._lifecycle_condition = threading.Condition()
        self._starts_in_flight = 0
        self._closed = False
        self._close_complete = False

    def _bwrap_prefix(self) -> tuple[str, ...]:
        """Build a minimal read-only runtime root with no Vault or project data."""
        repository_root = Path(__file__).resolve().parents[3]
        source_root = repository_root / "src"
        runtime_prefix = Path(sys.prefix).resolve()
        base_runtime_prefix = Path(sys.base_prefix).resolve()
        directory_targets = {Path("/etc"), repository_root}
        worker_is_external = not self._worker_path.is_relative_to(source_root)
        runtime_bindings = {
            (runtime_prefix, runtime_prefix),
            (base_runtime_prefix, base_runtime_prefix),
        }
        python_link = Path(sys.executable)
        while python_link.is_symlink():
            link_target = Path(os.readlink(python_link))
            python_link = (
                link_target
                if link_target.is_absolute()
                else python_link.parent / link_target
            )
        python_real = Path(sys.executable).resolve()
        try:
            python_relative = python_real.relative_to(base_runtime_prefix)
        except ValueError:
            python_relative = None
        if python_relative is not None and python_link.resolve() == python_real:
            runtime_alias = python_link
            for _ in python_relative.parts:
                runtime_alias = runtime_alias.parent
            # Some uv environments keep an absolute interpreter symlink under
            # a logical home path (for example /home) while sys.base_prefix is
            # reported through that path's physical mount (for example
            # /data00/home).  Preserve the lexical target inside the sandbox.
            runtime_bindings.add((base_runtime_prefix, runtime_alias))
        directory_roots = [
            repository_root,
            *(destination for _, destination in runtime_bindings),
        ]
        if worker_is_external:
            # /tmp is replaced by an empty tmpfs below.  Recreate only the
            # destination skeleton for an injected worker before binding the
            # file itself; never expose the host parent directory contents.
            directory_roots.append(self._worker_path.parent)
        for target in directory_roots:
            parent = target
            while parent != parent.parent:
                directory_targets.add(parent)
                parent = parent.parent
        args: list[str] = [
            "/usr/bin/bwrap",
            "--die-with-parent",
            "--new-session",
            "--unshare-user",
            "--unshare-pid",
            "--unshare-uts",
            "--unshare-ipc",
            "--as-pid-1",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
        ]
        for target in sorted(directory_targets, key=lambda item: (len(item.parts), str(item))):
            if target != Path("/"):
                args.extend(("--dir", str(target)))
        for path in (Path("/usr"), Path("/lib"), Path("/lib64")):
            if path.exists():
                args.extend(("--ro-bind", str(path), str(path)))
        for source, destination in sorted(
            runtime_bindings,
            key=lambda binding: (str(binding[1]), str(binding[0])),
        ):
            args.extend(("--ro-bind", str(source), str(destination)))
        if source_root.is_dir():
            args.extend(("--ro-bind", str(source_root), str(source_root)))
        for path in (repository_root / "pyproject.toml", repository_root / "uv.lock"):
            if path.is_file():
                args.extend(("--ro-bind", str(path), str(path)))
        if worker_is_external:
            args.extend(("--ro-bind", str(self._worker_path), str(self._worker_path)))
        for path in (
            Path("/etc/hosts"),
            Path("/etc/nsswitch.conf"),
            Path("/etc/resolv.conf"),
            Path("/etc/ssl"),
        ):
            if path.exists():
                args.extend(("--ro-bind", str(path), str(path)))
        args.extend(("--chdir", "/tmp", "--"))
        return tuple(args)

    def _seatbelt_prefix(self) -> tuple[str, ...]:
        """Build the deny-default macOS Seatbelt launch contract."""
        repository_root = Path(__file__).resolve().parents[3]
        source_root = repository_root / "src"
        if not all((repository_root / name).is_file() for name in ("AGENTS.md", "pyproject.toml", "uv.lock")):
            raise ChannelSandboxUnavailable()
        return (
            "/usr/bin/sandbox-exec",
            "-D",
            f"GHOSTAP_SOURCE_ROOT={source_root}",
            "-D",
            f"GHOSTAP_WORKER={self._worker_path}",
            "-D",
            f"GHOSTAP_RUNTIME_PREFIX={Path(sys.prefix).resolve()}",
            "-D",
            f"GHOSTAP_BASE_PREFIX={Path(sys.base_prefix).resolve()}",
            "-D",
            f"GHOSTAP_PYTHON={sys.executable}",
            "-D",
            f"GHOSTAP_PYTHON_REAL={Path(sys.executable).resolve()}",
            "-D",
            f"GHOSTAP_PYPROJECT={repository_root / 'pyproject.toml'}",
            "-D",
            f"GHOSTAP_UV_LOCK={repository_root / 'uv.lock'}",
            "-p",
            _MACOS_SEATBELT_PROFILE,
        )

    @contextmanager
    def employee_dispatch_guard(self):
        """Freeze live Channel authority without taking the Journal guard."""

        with self._lock:
            yield

    def _agent_lifecycle_lock(self, agent_id: str) -> threading.RLock:
        with self._lifecycle_registry_lock:
            return self._lifecycle_locks.setdefault(
                agent_id,
                threading.RLock(),  # leaf lock: never held while acquiring a LockLevel lock
            )

    def _launch_candidate(
        self,
        *,
        agent_id: str,
        app_id: str,
        generation: int,
        tenant_key: str,
        bot_principal_id: str,
        on_event: Callable[[dict[str, Any]], None],
        sandbox_attempt: _SandboxAttempt,
    ) -> tuple[_Runtime, int, int | None, dict[str, Any] | None, str]:
        child_fds: list[int] = []
        parent_fds: list[int] = []
        metadata_r = -1
        metadata_w = -1
        proof_nonce = ""
        sandbox_temp_dir: Path | None = None
        prefix = sandbox_attempt.prefix
        try:
            bootstrap_r, bootstrap_w = os.pipe()
            child_fds.append(bootstrap_r)
            parent_fds.append(bootstrap_w)
            control_r, control_w = os.pipe()
            child_fds.append(control_r)
            parent_fds.append(control_w)
            event_r, event_w = os.pipe()
            child_fds.append(event_w)
            parent_fds.append(event_r)
            if sandbox_attempt.bwrap_info or sandbox_attempt.seatbelt_proof:
                metadata_r, metadata_w = os.pipe()
                child_fds.append(metadata_w)
                parent_fds.append(metadata_r)
            if sandbox_attempt.bwrap_info:
                prefix = _with_bwrap_info_fd(prefix, metadata_w)
            elif sandbox_attempt.seatbelt_proof:
                proof_nonce = secrets.token_hex(16)
                temp_base = Path("/private/tmp")
                if not temp_base.is_dir():
                    temp_base = Path("/tmp")
                sandbox_temp_dir = Path(
                    tempfile.mkdtemp(
                        prefix="ghostap-employee-channel-",
                        dir=temp_base,
                    )
                )
                sandbox_temp_dir.chmod(0o700)
            env = {"PYTHONUTF8": "1"}
            if sandbox_attempt.seatbelt_proof:
                if sandbox_temp_dir is None:
                    raise ChannelSandboxUnavailable()
                prefix = _with_seatbelt_temp_dir(prefix, sandbox_temp_dir)
                env.update(
                    GHOSTAP_CHANNEL_TMP=str(sandbox_temp_dir),
                    TMPDIR=str(sandbox_temp_dir),
                )
            worker_args = (
                sys.executable,
                "-I",
                str(self._worker_path),
                str(bootstrap_r),
                str(control_r),
                str(event_w),
            )
            pass_fds = (bootstrap_r, control_r, event_w)
            if sandbox_attempt.seatbelt_proof:
                worker_args += (str(metadata_w), proof_nonce)
                pass_fds += (metadata_w,)
            elif sandbox_attempt.bwrap_info:
                pass_fds += (metadata_w,)
            process = self._launcher(
                prefix + worker_args,
                close_fds=True,
                pass_fds=pass_fds,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            for descriptor in (*parent_fds, *child_fds):
                _close_fd(descriptor)
            if sandbox_temp_dir is not None:
                shutil.rmtree(sandbox_temp_dir, ignore_errors=True)
            raise ChannelLaunchUnavailable("employee Channel launch failed") from None
        for descriptor in child_fds:
            _close_fd(descriptor)

        status = ChannelProcessStatus(
            agent_id=agent_id,
            app_id=app_id,
            generation=generation,
            pid=process.pid,
            state=ChannelProcessState.STARTING,
            tenant_key=tenant_key,
            bot_principal_id=bot_principal_id,
        )
        runtime = _Runtime(
            process,
            control_w,
            event_r,
            status,
            on_event,
            tenant_key=tenant_key,
            bot_principal_id=bot_principal_id,
            requires_observed_connection=self._production_worker,
            sandbox_temp_dir=sandbox_temp_dir,
        )
        attestation_pid: int | None = None
        proof: dict[str, Any] | None = None
        if metadata_r >= 0:
            try:
                metadata = _read_sandbox_metadata(metadata_r)
                if sandbox_attempt.bwrap_info:
                    candidate = metadata.get("child-pid")
                    if (
                        isinstance(candidate, int)
                        and not isinstance(candidate, bool)
                        and candidate > 0
                        and candidate != process.pid
                    ):
                        attestation_pid = candidate
                else:
                    proof = metadata
            except (OSError, ValueError):
                pass
            finally:
                _close_fd(metadata_r)
        return runtime, bootstrap_w, attestation_pid, proof, proof_nonce

    def _sandbox_attempts(self) -> list[_SandboxAttempt]:
        if self._sandbox_kind == "linux-bwrap":
            attempts = [
                _SandboxAttempt(
                    self._sandbox_prefix,
                    "bwrap-filesystem",
                    bwrap_info=True,
                )
            ]
            if self._automatic_process_fallback:
                attempts.append(
                    _SandboxAttempt(
                        (),
                        "process-fallback",
                        fallback=True,
                    )
                )
            return attempts
        if self._sandbox_kind == "macos-seatbelt":
            return [
                _SandboxAttempt(
                    self._sandbox_prefix,
                    "seatbelt-filesystem",
                    seatbelt_proof=True,
                )
            ]
        return [_SandboxAttempt(self._sandbox_prefix, self._sandbox_kind)]

    def start(
        self,
        agent_id: str,
        app_id: str,
        credential_ref: str,
        generation: int,
        on_event: Callable[[dict[str, Any]], None],
    ) -> ChannelProcessStatus:
        """Serialize one employee's generations across complete launch/teardown."""

        with self._lifecycle_condition:
            if self._closed:
                raise RuntimeError("employee Channel supervisor is closed")
            self._starts_in_flight += 1
        try:
            with self._agent_lifecycle_lock(agent_id):
                return self._start_serialized(
                    agent_id,
                    app_id,
                    credential_ref,
                    generation,
                    on_event,
                )
        finally:
            with self._lifecycle_condition:
                self._starts_in_flight -= 1
                self._lifecycle_condition.notify_all()

    def start_delivery_only(
        self,
        agent_id: str,
        app_id: str,
        credential_ref: str,
        generation: int,
        *,
        frozen_connection_id: str,
    ) -> ChannelProcessStatus:
        """Restore frozen outbound authority without reopening employee ingress."""

        with self._lifecycle_condition:
            if self._closed:
                raise RuntimeError("employee Channel supervisor is closed")
            self._starts_in_flight += 1
        try:
            with self._agent_lifecycle_lock(agent_id):
                return self._start_serialized(
                    agent_id,
                    app_id,
                    credential_ref,
                    generation,
                    lambda _event: None,
                    delivery_only=True,
                    frozen_connection_id=frozen_connection_id,
                )
        finally:
            with self._lifecycle_condition:
                self._starts_in_flight -= 1
                self._lifecycle_condition.notify_all()

    def _start_serialized(
        self,
        agent_id: str,
        app_id: str,
        credential_ref: str,
        generation: int,
        on_event: Callable[[dict[str, Any]], None],
        *,
        delivery_only: bool = False,
        frozen_connection_id: str = "",
    ) -> ChannelProcessStatus:
        """Launch, attest, bootstrap, and await READY for one employee."""
        self._validate_start(agent_id, app_id, credential_ref, generation, on_event)
        if (
            type(delivery_only) is not bool
            or (
                delivery_only
                and (
                    not isinstance(frozen_connection_id, str)
                    or not frozen_connection_id.startswith("conn_")
                    or frozen_connection_id != frozen_connection_id.strip()
                )
            )
            or (not delivery_only and frozen_connection_id)
        ):
            raise ValueError("invalid delivery-only Channel binding")
        if self._production_worker and self._ingress_service is None:
            raise RuntimeError("durable employee ingress is not configured")
        tenant_key = "tenant-test-unbound"
        bot_principal_id = "bot_test_unbound"
        if self._ingress_binding_resolver is not None:
            tenant_key, bot_principal_id = self._ingress_binding_resolver(agent_id, app_id)
            if (
                not isinstance(tenant_key, str)
                or not tenant_key
                or not isinstance(bot_principal_id, str)
                or not bot_principal_id.startswith("bot_")
            ):
                raise ValueError("invalid durable employee ingress binding")
        with self._lifecycle_condition:
            if self._closed:
                raise RuntimeError("employee Channel supervisor is closed")
        with self._lock:
            existing = self._runtimes.get(agent_id)
            if existing is not None and existing.process.poll() is None:
                if (
                    existing.status.generation == generation
                    and existing.delivery_only == delivery_only
                    and existing.frozen_connection_id == frozen_connection_id
                    and existing.status.state in {ChannelProcessState.STARTING, ChannelProcessState.READY}
                ):
                    return existing.status
        if existing is not None and existing.process.poll() is None:
            self._stop_serialized(agent_id)
            if existing.process.poll() is None:
                raise ChannelLaunchUnavailable("previous employee Channel did not terminate")
        with self._lock:
            high = self._generation_high_watermark.get(agent_id, 0)
            if generation < high or (generation == high and not delivery_only):
                raise ValueError("generation must advance after a worker has stopped")

        attempts = self._sandbox_attempts()
        runtime: _Runtime | None = None
        bootstrap_w = -1
        for attempt_index, sandbox_attempt in enumerate(attempts):
            try:
                (
                    runtime,
                    bootstrap_w,
                    attestation_pid,
                    sandbox_proof,
                    proof_nonce,
                ) = self._launch_candidate(
                    agent_id=agent_id,
                    app_id=app_id,
                    generation=generation,
                    tenant_key=tenant_key,
                    bot_principal_id=bot_principal_id,
                    on_event=on_event,
                    sandbox_attempt=sandbox_attempt,
                )
                runtime.delivery_only = delivery_only
                runtime.frozen_connection_id = frozen_connection_id
                runtime.status = replace(
                    runtime.status,
                    delivery_only=delivery_only,
                )
                with self._lifecycle_condition:
                    supervisor_closed = self._closed
                if supervisor_closed:
                    _close_fd(bootstrap_w)
                    bootstrap_w = -1
                    self._fail_and_reap(runtime, "supervisor-closed")
                    raise RuntimeError("employee Channel supervisor is closed")
            except RuntimeError:
                with self._lifecycle_condition:
                    if self._closed:
                        raise
                if attempt_index + 1 < len(attempts):
                    logger.warning("employee Channel bwrap launch failed; using process fallback")
                    continue
                if self._sandbox_kind == "macos-seatbelt":
                    logger.warning("employee Channel seatbelt launch failed; sandbox unavailable")
                    raise ChannelSandboxUnavailable() from None
                raise
            if sandbox_attempt.fallback:
                attestation = SandboxAttestation(
                    runtime.process.pid,
                    False,
                    "process-fallback",
                    ("bwrap unavailable; no filesystem isolation",),
                )
            elif sandbox_attempt.bwrap_info:
                if attestation_pid is None or runtime.process.poll() is not None:
                    attestation = SandboxAttestation(
                        runtime.process.pid,
                        False,
                        "bwrap-unverified",
                        ("bwrap child metadata unavailable",),
                    )
                else:
                    try:
                        attestation = self._sandbox_attestor(attestation_pid)
                    except Exception:
                        attestation = SandboxAttestation(
                            attestation_pid,
                            False,
                            "attestation-error",
                        )
                    if attestation.verified and runtime.process.poll() is not None:
                        attestation = SandboxAttestation(
                            attestation.pid,
                            False,
                            "bwrap-unverified",
                            ("bwrap monitor exited during attestation",),
                        )
            elif sandbox_attempt.seatbelt_proof:
                attestation = attest_macos_sandbox_proof(
                    sandbox_proof,
                    nonce=proof_nonce,
                    expected_pid=runtime.process.pid,
                )
                if attestation.verified and runtime.process.poll() is not None:
                    attestation = SandboxAttestation(
                        attestation.pid,
                        False,
                        "seatbelt-unverified",
                        ("sandbox-exec monitor exited during attestation",),
                    )
            else:
                try:
                    attestation = self._sandbox_attestor(runtime.process.pid)
                except Exception:
                    attestation = SandboxAttestation(
                        runtime.process.pid,
                        False,
                        "attestation-error",
                    )
            runtime.status = replace(runtime.status, sandbox=attestation)
            if attestation.verified or sandbox_attempt.fallback:
                if sandbox_attempt.fallback:
                    logger.warning("employee Channel is using unverified process fallback")
                break
            _close_fd(bootstrap_w)
            bootstrap_w = -1
            self._fail_and_reap(runtime, "sandbox-unavailable")
            if attempt_index + 1 < len(attempts):
                logger.warning("employee Channel bwrap attestation failed; using process fallback")
                runtime = None
                continue
            with self._lock:
                self._runtimes[agent_id] = runtime
                self._generation_high_watermark[agent_id] = generation
            raise ChannelSandboxUnavailable()
        if runtime is None or bootstrap_w < 0:
            raise ChannelLaunchUnavailable("employee Channel launch failed")
        with self._lifecycle_condition:
            supervisor_closed = self._closed
            if not supervisor_closed:
                with self._lock:
                    high = self._generation_high_watermark.get(agent_id, 0)
                    if generation < high or (generation == high and not delivery_only):
                        _close_fd(bootstrap_w)
                        self._fail_and_reap(runtime, "generation-race")
                        raise ValueError("generation must advance after a worker has stopped")
                    self._runtimes[agent_id] = runtime
                    self._generation_high_watermark[agent_id] = generation
        if supervisor_closed:
            _close_fd(bootstrap_w)
            self._fail_and_reap(runtime, "supervisor-closed")
            raise RuntimeError("employee Channel supervisor is closed")

        try:
            secret = self._secret_resolver(credential_ref, agent_id, app_id)
            bootstrap = encode_bootstrap(
                ChannelBootstrap(
                    agent_id,
                    app_id,
                    generation,
                    secret,
                    tenant_key,
                    bot_principal_id,
                    self._ingress_ack_timeout,
                    delivery_only,
                    frozen_connection_id,
                )
            )
        except Exception:
            _close_fd(bootstrap_w)
            self._fail_and_reap(runtime, "credential-resolution-failed")
            return runtime.status

        runtime.reader = threading.Thread(
            target=self._read_frames,
            args=(runtime,),
            name=f"employee-channel-{agent_id}-{generation}",
            daemon=True,
        )
        runtime.reader.start()
        try:
            _write_all(bootstrap_w, bootstrap)
        except OSError:
            self._fail_and_reap(runtime, "bootstrap-failed")
            return runtime.status
        finally:
            _close_fd(bootstrap_w)

        if not runtime.ready.wait(self._ready_timeout):
            self._fail_and_reap(runtime, "ready-timeout")
        return runtime.status

    def stop(self, agent_id: str) -> ChannelProcessStatus | None:
        """Wait for any in-flight lifecycle operation before returning terminal."""

        with self._agent_lifecycle_lock(agent_id):
            return self._stop_serialized(agent_id)

    def _stop_serialized(self, agent_id: str) -> ChannelProcessStatus | None:
        """Gracefully stop one generation, escalating only after timeout."""
        with self._lock:
            runtime = self._runtimes.get(agent_id)
            if runtime is None:
                return None
            if (
                runtime.status.state
                in {
                    ChannelProcessState.STOPPED,
                    ChannelProcessState.FAILED,
                    ChannelProcessState.CRASHED,
                }
                and runtime.process.poll() is not None
            ):
                return runtime.status
            runtime.stopping = True
            runtime.status = replace(runtime.status, state=ChannelProcessState.STOPPING)
            self._fail_pending_sends(runtime)
        self._send_control(runtime, FrameType.STOP, {})
        _close_fd(self._take_control_fd(runtime))
        self._wait_or_terminate(runtime)
        with self._lock:
            runtime.status = replace(
                runtime.status,
                state=ChannelProcessState.STOPPED,
                stopped_at=time.time(),
                exit_code=runtime.process.poll(),
            )
        self._finish_reader(runtime)
        return runtime.status

    def send(
        self,
        agent_id: str,
        *,
        generation: int,
        target: str,
        message: Any,
        options: Any = None,
        deadline: float | None = None,
    ) -> ChannelSendReceipt:
        """Send through the exact READY employee generation and await its receipt."""
        if not isinstance(target, str) or not target:
            raise ValueError("target is required")
        return self._request_outbound(
            agent_id,
            generation,
            FrameType.SEND,
            {"target": target, "message": message, "options": options},
            deadline=deadline,
        )

    def update_card(
        self,
        agent_id: str,
        *,
        generation: int,
        message_id: str,
        card: dict[str, Any],
        deadline: float | None = None,
    ) -> ChannelSendReceipt:
        """Patch one pre-bound card through the exact READY employee generation."""
        if not isinstance(message_id, str) or not message_id:
            raise ValueError("message_id is required")
        if not isinstance(card, dict):
            raise ValueError("card must be an object")
        return self._request_outbound(
            agent_id,
            generation,
            FrameType.UPDATE_CARD,
            {"message_id": message_id, "card": card},
            expected_message_id=message_id,
            deadline=deadline,
        )

    def _request_outbound(
        self,
        agent_id: str,
        generation: int,
        frame_type: FrameType,
        payload: dict[str, Any],
        *,
        expected_message_id: str = "",
        deadline: float | None = None,
    ) -> ChannelSendReceipt:
        if not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
            raise ValueError("generation must be a positive integer")
        if deadline is not None and (
            isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(float(deadline))
        ):
            raise ValueError("employee Channel outbound deadline is invalid")
        receipt_deadline = time.monotonic() + self._send_timeout
        if deadline is not None:
            receipt_deadline = min(receipt_deadline, float(deadline))
        if receipt_deadline <= time.monotonic():
            raise EmployeeChannelOutboundTimeout("employee Channel outbound deadline exceeded")
        operation = "send" if frame_type is FrameType.SEND else "update_card"
        request_id = f"{operation}_{uuid.uuid4().hex}"
        pending = _PendingSend(operation=operation, expected_message_id=expected_message_id)
        remaining = receipt_deadline - time.monotonic()
        if remaining <= 0 or not self._lock.acquire(timeout=remaining):
            raise EmployeeChannelOutboundTimeout(
                "employee Channel outbound deadline exceeded"
            )
        try:
            runtime = self._runtimes.get(agent_id)
            if runtime is None or runtime.status.state is not ChannelProcessState.READY:
                raise EmployeeChannelOutboundError("employee Channel is not ready")
            if runtime.status.generation != generation:
                raise EmployeeChannelGenerationChanged("employee Channel generation mismatch")
            with runtime.pending_lock:
                runtime.pending_sends[request_id] = pending
            try:
                control_payload = {"request_id": request_id, **payload}
                sent = self._send_control(
                    runtime,
                    frame_type,
                    control_payload,
                    deadline=receipt_deadline,
                )
            except ProtocolError:
                with runtime.pending_lock:
                    runtime.pending_sends.pop(request_id, None)
                raise ValueError(f"unsafe {operation.replace('_', ' ')} payload") from None
            except TimeoutError as exc:
                with runtime.pending_lock:
                    runtime.pending_sends.pop(request_id, None)
                raise EmployeeChannelOutboundTimeout(
                    f"employee Channel {operation.replace('_', ' ')} timed out"
                ) from exc
            if not sent:
                with runtime.pending_lock:
                    runtime.pending_sends.pop(request_id, None)
                raise EmployeeChannelOutboundError(f"employee Channel {operation.replace('_', ' ')} failed")
        finally:
            self._lock.release()
        remaining = max(0.0, receipt_deadline - time.monotonic())
        if not pending.completed.wait(remaining):
            with runtime.pending_lock:
                runtime.pending_sends.pop(request_id, None)
            raise EmployeeChannelOutboundTimeout(f"employee Channel {operation.replace('_', ' ')} receipt timed out")
        with runtime.pending_lock:
            runtime.pending_sends.pop(request_id, None)
        if pending.success is not True:
            message = (
                f"employee Channel {operation.replace('_', ' ')} failed"
                f" ({pending.error_code or 'unknown-worker-failure'})"
            )
            if pending.failure_kind in {"transport", "remote"}:
                raise EmployeeChannelOutboundError(message)
            raise EmployeeChannelOutboundIntegrityError(message)
        return ChannelSendReceipt(
            request_id=request_id,
            success=True,
            app_id=pending.app_id,
            generation=pending.generation,
            connection_id=pending.connection_id,
            message_id=pending.message_id,
        )

    def status(self, agent_id: str) -> ChannelProcessStatus | None:
        """Return a secret-free immutable process snapshot."""
        with self._lock:
            runtime = self._runtimes.get(agent_id)
            if runtime is None:
                return None
            exit_code = runtime.process.poll()
            if (
                exit_code is not None
                and not runtime.stopping
                and runtime.status.state in {ChannelProcessState.STARTING, ChannelProcessState.READY}
            ):
                runtime.status = replace(
                    runtime.status,
                    state=ChannelProcessState.CRASHED,
                    stopped_at=time.time(),
                    exit_code=exit_code,
                    error_code="worker-exited",
                )
                runtime.ready.set()
            return replace(
                runtime.status,
                identity=deepcopy(runtime.status.identity),
                ready_metadata=deepcopy(runtime.status.ready_metadata),
            )

    def close(self) -> None:
        """Stop all owned children and make admission permanently closed."""
        with self._lifecycle_condition:
            if self._closed:
                while not self._close_complete:
                    self._lifecycle_condition.wait()
                return
            self._closed = True
            while self._starts_in_flight:
                self._lifecycle_condition.wait()
        try:
            with self._lock:
                agent_ids = list(self._runtimes)
            for agent_id in agent_ids:
                self.stop(agent_id)
        finally:
            with self._lifecycle_condition:
                self._close_complete = True
                self._lifecycle_condition.notify_all()

    def _validate_start(self, agent_id: str, app_id: str, credential_ref: str, generation: int, on_event: Any) -> None:
        if not all(isinstance(value, str) and value for value in (agent_id, app_id, credential_ref)):
            raise ValueError("agent_id, app_id, and credential_ref are required")
        if not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
            raise ValueError("generation must be a positive integer")
        if not callable(on_event):
            raise TypeError("on_event must be callable")

    def _read_frames(self, runtime: _Runtime) -> None:
        event_fd = self._take_event_fd(runtime)
        if event_fd < 0:
            return
        try:
            with os.fdopen(event_fd, "rb", buffering=0) as stream:
                while True:
                    raw = stream.readline(MAX_FRAME_BYTES + 1)
                    if not raw:
                        break
                    try:
                        frame = decode_frame(raw)
                    except ProtocolError:
                        with self._lock:
                            runtime.status = replace(runtime.status, error_code="protocol-error")
                            self._fail_pending_sends(
                                runtime,
                                failure_kind="internal",
                                error_code="protocol-error",
                            )
                        continue
                    if frame.agent_id != runtime.status.agent_id or frame.generation != runtime.status.generation:
                        with self._lock:
                            runtime.status = replace(runtime.status, stale_frames=runtime.status.stale_frames + 1)
                        continue
                    if frame.sequence <= runtime.inbound_sequence:
                        with self._lock:
                            runtime.status = replace(runtime.status, stale_frames=runtime.status.stale_frames + 1)
                        continue
                    runtime.inbound_sequence = frame.sequence
                    self._accept_frame(runtime, frame)
        finally:
            exit_code = runtime.process.poll()
            if exit_code is None:
                try:
                    exit_code = runtime.process.wait(
                        timeout=min(self._stop_timeout, _EVENT_PIPE_EXIT_GRACE_SECONDS)
                    )
                except subprocess.TimeoutExpired:
                    pass
            should_reap = False
            with self._lock:
                active = runtime.status.state in {
                    ChannelProcessState.STARTING,
                    ChannelProcessState.READY,
                }
                if not runtime.stopping and active:
                    runtime.stopping = True
                    crashed = exit_code is not None
                    runtime.status = replace(
                        runtime.status,
                        state=(ChannelProcessState.CRASHED if crashed else ChannelProcessState.FAILED),
                        ready_at=None,
                        stopped_at=time.time(),
                        exit_code=exit_code,
                        error_code=(
                            "worker-exited-before-ready"
                            if crashed and runtime.status.state is ChannelProcessState.STARTING
                            else "worker-exited"
                            if crashed
                            else "event-pipe-closed"
                        ),
                    )
                    self._fail_pending_sends(runtime)
                    runtime.ready.set()
                    should_reap = not crashed
            if should_reap:
                _close_fd(self._take_control_fd(runtime))
                self._wait_or_terminate(runtime)
                with self._lock:
                    runtime.status = replace(
                        runtime.status,
                        stopped_at=time.time(),
                        exit_code=runtime.process.poll(),
                    )
            if exit_code is not None or should_reap:
                self._cleanup_runtime_temp(runtime)

    def _accept_frame(self, runtime: _Runtime, frame: ChannelFrame) -> None:
        if frame.frame_type is FrameType.READY:
            try:
                validate_ready_payload(frame.payload)
                identity = frame.payload["identity"]
                if identity["app_id"] != runtime.status.app_id:
                    raise ProtocolError("READY app identity does not match runtime")
                if getattr(runtime, "delivery_only", False) and frame.payload["connection_id"] != getattr(
                    runtime, "frozen_connection_id", ""
                ):
                    raise ProtocolError("delivery-only READY connection does not match frozen authority")
            except (KeyError, ProtocolError, TypeError):
                with self._lock:
                    runtime.status = replace(runtime.status, error_code="invalid-ready")
                return
            connection = frame.payload.get("connection")
            if runtime.requires_observed_connection and (
                not isinstance(connection, dict)
                or connection.get("observed") is not True
                or connection.get("secure") is not True
            ):
                with self._lock:
                    runtime.status = replace(runtime.status, error_code="unobserved-connection")
                return
            metadata = {key: deepcopy(value) for key, value in frame.payload.items() if key != "identity"}
            with self._lock:
                runtime.status = replace(
                    runtime.status,
                    state=ChannelProcessState.READY,
                    identity=deepcopy(identity),
                    ready_metadata=metadata,
                    ready_at=time.time(),
                    error_code="",
                )
            runtime.ready.set()
            return
        if frame.frame_type is FrameType.INGRESS:
            if getattr(runtime, "delivery_only", False):
                with self._lock:
                    runtime.status = replace(
                        runtime.status,
                        error_code="delivery-only-ingress-rejected",
                    )
                return
            self._accept_ingress(runtime, frame)
        elif frame.frame_type is FrameType.EVENT:
            if frame.payload.get("event") == "reconnecting":
                with self._lock:
                    runtime.status = replace(
                        runtime.status,
                        state=ChannelProcessState.STARTING,
                        ready_at=None,
                        ready_metadata={"reconnecting_at": time.time()},
                        error_code="channel-reconnecting",
                    )
                    runtime.ready.clear()
            try:
                runtime.on_event(dict(frame.payload))
            except Exception:
                with self._lock:
                    runtime.status = replace(runtime.status, error_code="event-callback-failed")
        elif frame.frame_type is FrameType.ERROR:
            code = frame.payload.get("error_code")
            with self._lock:
                runtime.status = replace(runtime.status, error_code=code if isinstance(code, str) else "worker-error")
        elif frame.frame_type is FrameType.HEALTH:
            operation = frame.payload.get("operation")
            if operation in {"send", "update_card"}:
                request_id = frame.payload.get("request_id")
                success = frame.payload.get("success")
                if isinstance(request_id, str) and isinstance(success, bool):
                    with self._lock:
                        with runtime.pending_lock:
                            pending = runtime.pending_sends.get(request_id)
                            if pending is not None and not pending.completed.is_set():
                                app_id = frame.payload.get("app_id")
                                generation = frame.payload.get("generation")
                                connection_id = frame.payload.get("connection_id")
                                message_id = frame.payload.get("message_id")
                                valid_evidence = (
                                    success is True
                                    and app_id == runtime.status.app_id
                                    and generation == runtime.status.generation
                                    and isinstance(connection_id, str)
                                    and connection_id == runtime.status.ready_metadata.get("connection_id")
                                    and isinstance(message_id, str)
                                    and bool(message_id)
                                    and (pending.operation == "send" or message_id == pending.expected_message_id)
                                )
                                pending.success = valid_evidence
                                if valid_evidence:
                                    pending.app_id = app_id
                                    pending.generation = generation
                                    pending.connection_id = connection_id
                                    pending.message_id = message_id
                                elif success is True:
                                    pending.failure_kind = "internal"
                                    pending.error_code = "invalid-outbound-receipt"
                                    runtime.status = replace(
                                        runtime.status,
                                        error_code=f"invalid-{operation.replace('_', '-')}-receipt",
                                    )
                                else:
                                    failure_kind = frame.payload.get("failure_kind")
                                    error_code = frame.payload.get("error_code")
                                    pending.failure_kind = failure_kind if isinstance(failure_kind, str) else ""
                                    pending.error_code = error_code if isinstance(error_code, str) else ""
                                    runtime.status = replace(
                                        runtime.status,
                                        error_code=(pending.error_code or "invalid-outbound-failure"),
                                    )
                                pending.completed.set()
            with self._lock:
                runtime.status = replace(
                    runtime.status,
                    ready_metadata={**runtime.status.ready_metadata, "health": dict(frame.payload)},
                )

    def _accept_ingress(self, runtime: _Runtime, frame: ChannelFrame) -> None:
        service = self._ingress_service
        try:
            metadata = EmployeeIngressMetadata.from_dict(frame.payload["metadata"])
            payload = EmployeeIngressPayload.from_dict(frame.payload["payload"])
            with self._lock:
                current = self._runtimes.get(runtime.status.agent_id)
                valid = (
                    service is not None
                    and current is runtime
                    and runtime.status.state is ChannelProcessState.READY
                    and metadata.tenant_key == runtime.tenant_key
                    and metadata.agent_id == runtime.status.agent_id
                    and metadata.bot_principal_id == runtime.bot_principal_id
                    and metadata.app_id == runtime.status.app_id
                    and metadata.channel_generation == runtime.status.generation
                    and metadata.connection_id == runtime.status.ready_metadata.get("connection_id")
                    and frame.payload["app_id"] == runtime.status.app_id
                    and frame.payload["connection_id"] == metadata.connection_id
                )
            if not valid or service is None:
                raise ValueError("employee ingress runtime binding mismatch")
            ack = service.accept(
                metadata,
                payload,
                request_id=frame.payload["request_id"],
                action_correlation=frame.payload["action_correlation"],
            )
            if not isinstance(ack, EmployeeIngressAck):
                raise TypeError("employee ingress service returned invalid ACK")
            sent = self._send_control(
                runtime,
                FrameType.INGRESS_ACK,
                {
                    "request_id": ack.request_id,
                    "app_id": ack.app_id,
                    "connection_id": ack.connection_id,
                    "ack": ack.to_dict(),
                },
            )
            if not sent:
                raise BrokenPipeError("employee ingress ACK pipe closed")
            try:
                runtime.on_event(
                    {
                        "event": "durableIngressAccepted",
                        "data": {
                            "acceptance_id": ack.acceptance.acceptance_id,
                            "agent_id": ack.agent_id,
                            "generation": ack.channel_generation,
                        },
                    }
                )
            except Exception:
                with self._lock:
                    runtime.status = replace(
                        runtime.status,
                        error_code="ingress-control-callback-failed",
                    )
        except Exception:
            with self._lock:
                runtime.status = replace(
                    runtime.status,
                    error_code="ingress-not-acknowledged",
                )

    def _send_control(
        self,
        runtime: _Runtime,
        frame_type: FrameType,
        payload: dict[str, Any],
        *,
        deadline: float | None = None,
    ) -> bool:
        abandoned_fd = -1
        if deadline is None:
            runtime.control_lock.acquire()
        else:
            remaining = float(deadline) - time.monotonic()
            if remaining <= 0 or not runtime.control_lock.acquire(timeout=remaining):
                raise TimeoutError("employee Channel IPC write timed out")
        try:
            if runtime.control_fd < 0:
                return False
            runtime.outbound_sequence += 1
            raw = encode_frame(
                ChannelFrame(
                    frame_type,
                    runtime.status.agent_id,
                    runtime.status.generation,
                    runtime.outbound_sequence,
                    payload,
                )
            )
            try:
                if deadline is None:
                    _write_all(runtime.control_fd, raw)
                else:
                    _write_all(runtime.control_fd, raw, deadline=deadline)
            except TimeoutError:
                # A timed-out frame may have been written only partially.
                # Never append another frame to that stream: doing so would
                # let the child parse corrupted control input as a new command.
                abandoned_fd = runtime.control_fd
                runtime.control_fd = -1
                raise
            except OSError:
                if deadline is not None:
                    abandoned_fd = runtime.control_fd
                    runtime.control_fd = -1
                return False
            return True
        finally:
            runtime.control_lock.release()
            _close_fd(abandoned_fd)

    def _fail_and_reap(self, runtime: _Runtime, error_code: str) -> None:
        runtime.stopping = True
        with self._lock:
            self._fail_pending_sends(runtime)
        _close_fd(self._take_control_fd(runtime))
        self._wait_or_terminate(runtime)
        with self._lock:
            runtime.status = replace(
                runtime.status,
                state=ChannelProcessState.FAILED,
                stopped_at=time.time(),
                exit_code=runtime.process.poll(),
                error_code=error_code,
            )
        runtime.ready.set()
        self._finish_reader(runtime)

    @staticmethod
    def _fail_pending_sends(
        runtime: _Runtime,
        *,
        failure_kind: str = "transport",
        error_code: str = "channel-closed",
    ) -> None:
        with runtime.pending_lock:
            for pending in runtime.pending_sends.values():
                if pending.completed.is_set():
                    continue
                pending.success = False
                pending.failure_kind = failure_kind
                pending.error_code = error_code
                pending.completed.set()

    def _wait_or_terminate(self, runtime: _Runtime) -> None:
        try:
            runtime.process.wait(timeout=self._stop_timeout)
        except subprocess.TimeoutExpired:
            runtime.process.terminate()
            try:
                runtime.process.wait(timeout=self._stop_timeout)
            except subprocess.TimeoutExpired:
                runtime.process.kill()
                runtime.process.wait(timeout=self._stop_timeout)

    def _finish_reader(self, runtime: _Runtime) -> None:
        event_fd = self._take_event_fd(runtime)
        if event_fd >= 0:
            _close_fd(event_fd)
        if runtime.reader is not None and runtime.reader is not threading.current_thread():
            runtime.reader.join(timeout=self._stop_timeout)
        self._cleanup_runtime_temp(runtime)

    def _take_event_fd(self, runtime: _Runtime) -> int:
        """Atomically transfer the event-pipe read descriptor to one closer."""

        with self._lock:
            event_fd = runtime.event_fd
            runtime.event_fd = -1
            return event_fd

    @staticmethod
    def _take_control_fd(runtime: _Runtime) -> int:
        """Atomically transfer the control-pipe descriptor to one closer."""

        with runtime.control_lock:
            control_fd = runtime.control_fd
            runtime.control_fd = -1
            return control_fd

    @staticmethod
    def _cleanup_runtime_temp(runtime: _Runtime) -> None:
        path = runtime.sandbox_temp_dir
        runtime.sandbox_temp_dir = None
        if path is None:
            return
        shutil.rmtree(path, ignore_errors=True)


def attest_process_sandbox(pid: int) -> SandboxAttestation:
    """Verify user, mount and PID namespaces plus an absent project secret root."""
    deadline = time.monotonic() + 1.0
    while True:
        try:
            process_start_time = _proc_start_time(pid)
            parent_user_ns = os.readlink("/proc/self/ns/user")
            child_user_ns = os.readlink(f"/proc/{pid}/ns/user")
            parent_mount_ns = os.readlink("/proc/self/ns/mnt")
            child_mount_ns = os.readlink(f"/proc/{pid}/ns/mnt")
            parent_pid_ns = os.readlink("/proc/self/ns/pid")
            child_pid_ns = os.readlink(f"/proc/{pid}/ns/pid")
            repository_root = Path(__file__).resolve().parents[3]
            child_repository = Path(f"/proc/{pid}/root") / repository_root.relative_to("/")
            source_visible = (child_repository / "src").is_dir()
            secrets_hidden = not any((child_repository / name).exists() for name in (".env", ".git", ".Memory"))
            identity_stable = _proc_start_time(pid) == process_start_time
            if (
                child_user_ns != parent_user_ns
                and child_mount_ns != parent_mount_ns
                and child_pid_ns != parent_pid_ns
                and source_visible
                and secrets_hidden
                and identity_stable
            ):
                return SandboxAttestation(
                    pid,
                    True,
                    "bwrap-filesystem",
                    ("user/mount/pid namespaces", "project secrets absent"),
                )
        except (OSError, StopIteration, ValueError):
            if time.monotonic() >= deadline:
                return SandboxAttestation(
                    pid,
                    False,
                    "unverified",
                    ("sandbox inspection failed",),
                )
        if time.monotonic() >= deadline:
            return SandboxAttestation(
                pid,
                False,
                "unverified",
                (
                    "user/mount/pid namespace isolation not attested",
                    "project secret paths are not proven absent",
                ),
            )
        time.sleep(0.01)


def _proc_start_time(pid: int) -> str:
    """Return Linux proc stat field 22 without depending on mutable counters."""
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    command_end = raw.rfind(")")
    if command_end < 0:
        raise ValueError("invalid proc stat")
    fields_after_command = raw[command_end + 2 :].split()
    if len(fields_after_command) < 20:
        raise ValueError("invalid proc stat")
    return fields_after_command[19]


def attest_macos_sandbox_proof(
    proof: dict[str, Any] | None,
    *,
    nonce: str,
    expected_pid: int,
) -> SandboxAttestation:
    """Validate the trusted worker's pre-credential Seatbelt denial proof."""
    if not isinstance(proof, dict) or set(proof) != {
        "schema_version",
        "nonce",
        "pid",
        "source_readable",
        "runtime_readable",
        "repository_canary_errno",
    }:
        return SandboxAttestation(
            0,
            False,
            "seatbelt-unverified",
            ("invalid sandbox proof schema",),
        )
    pid = proof.get("pid")
    denied_errno = proof.get("repository_canary_errno")
    verified = (
        type(proof.get("schema_version")) is int
        and proof.get("schema_version") == 1
        and isinstance(proof.get("nonce"), str)
        and proof.get("nonce") == nonce
        and isinstance(pid, int)
        and not isinstance(pid, bool)
        and pid == expected_pid
        and proof.get("source_readable") is True
        and proof.get("runtime_readable") is True
        and isinstance(denied_errno, int)
        and not isinstance(denied_errno, bool)
        and denied_errno in {1, 13}
    )
    return SandboxAttestation(
        pid if isinstance(pid, int) and not isinstance(pid, bool) else 0,
        verified,
        "seatbelt-filesystem" if verified else "seatbelt-unverified",
        (
            "deny-default Seatbelt profile",
            "repository canary denied before credentials",
        )
        if verified
        else ("sandbox denial proof rejected",),
    )


def _with_bwrap_info_fd(prefix: tuple[str, ...], fd: int) -> tuple[str, ...]:
    try:
        separator = prefix.index("--")
    except ValueError as exc:
        raise ValueError("invalid bwrap launch prefix") from exc
    return prefix[:separator] + ("--info-fd", str(fd)) + prefix[separator:]


def _with_seatbelt_temp_dir(
    prefix: tuple[str, ...],
    path: Path,
) -> tuple[str, ...]:
    try:
        profile_flag = prefix.index("-p")
    except ValueError as exc:
        raise ValueError("invalid seatbelt launch prefix") from exc
    return prefix[:profile_flag] + ("-D", f"GHOSTAP_TEMP={path}") + prefix[profile_flag:]


def _read_sandbox_metadata(fd: int) -> dict[str, Any]:
    """Read exactly one small JSON object from a one-shot sandbox pipe."""
    deadline = time.monotonic() + _SANDBOX_METADATA_TIMEOUT_SECONDS
    raw = bytearray()
    os.set_blocking(fd, False)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError("sandbox metadata timed out")
        readable, _, _ = select.select((fd,), (), (), remaining)
        if not readable:
            raise ValueError("sandbox metadata timed out")
        chunk = os.read(fd, min(1024, _SANDBOX_METADATA_MAX_BYTES + 1 - len(raw)))
        if not chunk:
            break
        raw.extend(chunk)
        if len(raw) > _SANDBOX_METADATA_MAX_BYTES:
            raise ValueError("sandbox metadata is too large")
    try:
        decoded = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid sandbox metadata") from exc
    if not isinstance(decoded, dict):
        raise ValueError("sandbox metadata must be an object")
    return decoded


def _write_all(
    fd: int,
    raw: bytes,
    *,
    deadline: float | None = None,
) -> None:
    if deadline is None:
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise BrokenPipeError("employee Channel IPC closed")
            view = view[written:]
        return

    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(float(deadline)):
        raise ValueError("employee Channel IPC write deadline is invalid")
    deadline = float(deadline)
    was_blocking = os.get_blocking(fd)
    if was_blocking:
        os.set_blocking(fd, False)
    view = memoryview(raw)
    try:
        while view:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("employee Channel IPC write timed out")
            try:
                written = os.write(fd, view)
            except BlockingIOError:
                _, writable, _ = select.select((), (fd,), (), remaining)
                if not writable:
                    raise TimeoutError("employee Channel IPC write timed out")
                continue
            except InterruptedError:
                continue
            if written <= 0:
                raise BrokenPipeError("employee Channel IPC closed")
            view = view[written:]
    finally:
        if was_blocking:
            try:
                os.set_blocking(fd, True)
            except OSError:
                pass


def _close_fd(fd: int) -> None:
    if fd < 0:
        return
    try:
        os.close(fd)
    except OSError:
        pass
