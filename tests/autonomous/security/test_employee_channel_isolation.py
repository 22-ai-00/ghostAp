from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

from src.autonomous.provisioning import channel_worker as channel_worker_module
from src.autonomous.supervisor.employee_channels import (
    ChannelProcessState,
    ChannelSandboxUnavailable,
    EmployeeChannelSupervisor,
    SandboxAttestation,
    _read_sandbox_metadata,
)


def _worker(tmp_path: Path) -> Path:
    path = tmp_path / "security_worker.py"
    path.write_text(
        textwrap.dedent(
            """
            import json
            import os
            import sys

            bootstrap_fd, control_fd, event_fd = map(int, sys.argv[1:])
            bootstrap = json.loads(os.fdopen(bootstrap_fd, 'rb', buffering=0).readline())
            frame = {'v': 1, 'type': 'READY', 'agent_id': bootstrap['agent_id'],
                     'generation': bootstrap['generation'], 'sequence': 1,
                     'payload': {'identity': {'app_id': bootstrap['app_id']},
                                 'environment': dict(os.environ)}}
            os.write(event_fd, (json.dumps(frame, separators=(',', ':')) + '\\n').encode())
            control = os.fdopen(control_fd, 'rb', buffering=0)
            control.readline()
            """
        ),
        encoding="utf-8",
    )
    return path


def _track_current_thread_pipes(
    monkeypatch: pytest.MonkeyPatch,
) -> list[int]:
    """Track only pipes opened by the synchronous launcher under test."""

    owner = threading.get_ident()
    real_pipe = os.pipe
    opened: list[int] = []

    def tracked_pipe() -> tuple[int, int]:
        descriptors = real_pipe()
        if threading.get_ident() == owner:
            opened.extend(descriptors)
        return descriptors

    monkeypatch.setattr(os, "pipe", tracked_pipe)
    return opened


def _assert_pipe_descriptors_closed(descriptors: list[int]) -> None:
    assert descriptors
    for descriptor in descriptors:
        with pytest.raises(OSError) as raised:
            os.fstat(descriptor)
        assert raised.value.errno == errno.EBADF


def _mac_proof_worker(tmp_path: Path, *, denied_errno: int = errno.EACCES) -> Path:
    marker = tmp_path / "proof-emitted"
    path = tmp_path / "mac_proof_worker.py"
    path.write_text(
        textwrap.dedent(
            f"""
            import json
            import os
            import sys

            bootstrap_fd, control_fd, event_fd, proof_fd = map(int, sys.argv[1:5])
            nonce = sys.argv[5]
            {str(marker)!r} and open({str(marker)!r}, 'w').close()
            proof = {{'schema_version': 1, 'nonce': nonce, 'pid': os.getpid(),
                      'source_readable': True, 'runtime_readable': True,
                      'repository_canary_errno': {denied_errno}}}
            os.write(proof_fd, json.dumps(proof, separators=(',', ':')).encode())
            os.close(proof_fd)
            bootstrap = json.loads(os.fdopen(bootstrap_fd, 'rb', buffering=0).readline())
            frame = {{'v': 1, 'type': 'READY', 'agent_id': bootstrap['agent_id'],
                     'generation': bootstrap['generation'], 'sequence': 1,
                     'payload': {{'identity': {{'app_id': bootstrap['app_id']}}}}}}
            os.write(event_fd, (json.dumps(frame, separators=(',', ':')) + '\\n').encode())
            os.fdopen(control_fd, 'rb', buffering=0).readline()
            """
        ),
        encoding="utf-8",
    )
    return path


def _accepted(pid: int) -> SandboxAttestation:
    return SandboxAttestation(pid=pid, verified=True, mechanism="test-fixture")


@pytest.mark.slow
def test_default_sandbox_attestation_fails_closed_before_secret_resolution(tmp_path: Path) -> None:
    resolved = False

    def resolve(*_args: str) -> str:
        nonlocal resolved
        resolved = True
        return "must-not-be-delivered"

    supervisor = EmployeeChannelSupervisor(
        secret_resolver=resolve,
        worker_path=_worker(tmp_path),
        ready_timeout=0.5,
        sandbox_prefix=(),
    )
    try:
        with pytest.raises(ChannelSandboxUnavailable):
            supervisor.start("agt_1", "cli_1", "cred_1", 1, lambda _: None)
        assert resolved is False
        assert supervisor.status("agt_1").state is ChannelProcessState.FAILED
        assert supervisor.status("agt_1").sandbox.verified is False
    finally:
        supervisor.close()


def test_default_launcher_prefers_bwrap_and_falls_back_to_process_isolation(
    tmp_path: Path,
) -> None:
    user_namespaces_work = (
        sys.platform == "linux"
        and Path("/usr/bin/bwrap").is_file()
        and Path("/usr/bin/unshare").is_file()
        and subprocess.run(
            ["/usr/bin/unshare", "-Ur", "true"],
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )
    supervisor = EmployeeChannelSupervisor(
        secret_resolver=lambda *_: "employee-secret",
        worker_path=_worker(tmp_path),
        ready_timeout=1.0,
    )
    try:
        status = supervisor.start(
            "agt_1", "cli_1", "cred_1", 1, lambda _: None
        )
        assert status.state is ChannelProcessState.READY
        assert status.sandbox is not None
        if user_namespaces_work:
            assert status.sandbox.verified is True
        if status.sandbox.verified:
            assert status.sandbox.mechanism == "bwrap-filesystem"
            assert status.sandbox.pid != status.pid
        else:
            assert status.sandbox.mechanism == "process-fallback"
    finally:
        supervisor.close()


def test_secret_and_parent_environment_never_enter_argv_or_child_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "employee-secret-sentinel"
    monkeypatch.setenv("AUTONOMOUS_VAULT_MASTER_KEY", "master-key-sentinel")
    launches: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def launch(argv, **kwargs):
        launches.append((tuple(argv), dict(kwargs)))
        return subprocess.Popen(argv, **kwargs)

    supervisor = EmployeeChannelSupervisor(
        secret_resolver=lambda *_: secret,
        worker_path=_worker(tmp_path),
        launcher=launch,
        sandbox_attestor=_accepted,
        ready_timeout=1.0,
    )
    try:
        status = supervisor.start("agt_1", "cli_1", "cred_1", 1, lambda _: None)

        argv, kwargs = launches[0]
        joined = " ".join(argv)
        assert argv[:2] == (sys.executable, "-I")
        assert Path(argv[2]).is_absolute()
        assert secret not in joined
        assert "cred_1" not in joined
        assert kwargs["close_fds"] is True
        assert len(kwargs["pass_fds"]) == 3
        assert kwargs["env"] == {"PYTHONUTF8": "1"}
        child_env = status.ready_metadata["environment"]
        assert child_env["PYTHONUTF8"] == "1"
        assert "AUTONOMOUS_VAULT_MASTER_KEY" not in child_env
        assert set(child_env) <= {"PYTHONUTF8", "LC_CTYPE"}
    finally:
        supervisor.close()


def test_status_and_failures_never_echo_secret(tmp_path: Path) -> None:
    secret = "do-not-echo-this-secret"

    def broken_attestor(pid: int) -> SandboxAttestation:
        return SandboxAttestation(pid=pid, verified=False, mechanism="none")

    supervisor = EmployeeChannelSupervisor(
        secret_resolver=lambda *_: secret,
        worker_path=_worker(tmp_path),
        sandbox_attestor=broken_attestor,
    )
    try:
        with pytest.raises(ChannelSandboxUnavailable) as raised:
            supervisor.start("agt_1", "cli_1", "cred_secret_name", 1, lambda _: None)
        assert secret not in str(raised.value)
        assert "cred_secret_name" not in repr(supervisor.status("agt_1"))
    finally:
        supervisor.close()


def test_send_payload_rejects_credential_material_before_ipc(tmp_path: Path) -> None:
    supervisor = EmployeeChannelSupervisor(
        secret_resolver=lambda *_: "employee-secret",
        worker_path=_worker(tmp_path),
        sandbox_attestor=_accepted,
        ready_timeout=1.0,
    )
    try:
        supervisor.start("agt_1", "cli_1", "cred_1", 1, lambda _: None)
        with pytest.raises(ValueError, match="unsafe send payload"):
            supervisor.send(
                "agt_1",
                generation=1,
                target="ou_requester",
                message={"app_secret": "must-not-cross-ipc"},
            )
    finally:
        supervisor.close()


def test_launcher_failure_closes_every_parent_and_child_pipe_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = _track_current_thread_pipes(monkeypatch)

    def fail_launch(*_args, **_kwargs):
        raise OSError("injected launcher failure")

    supervisor = EmployeeChannelSupervisor(
        secret_resolver=lambda *_: "must-not-resolve",
        worker_path=_worker(tmp_path),
        launcher=fail_launch,
        sandbox_attestor=_accepted,
    )
    with pytest.raises(RuntimeError, match="launch failed"):
        supervisor.start("agt_1", "cli_1", "cred_1", 1, lambda _: None)

    _assert_pipe_descriptors_closed(opened)
    supervisor.close()


def test_default_linux_launcher_failure_closes_bwrap_info_pipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if sys.platform != "linux":
        pytest.skip("Linux bwrap metadata contract")
    opened = _track_current_thread_pipes(monkeypatch)

    def fail_launch(*_args, **_kwargs):
        raise OSError("injected launcher failure")

    supervisor = EmployeeChannelSupervisor(
        secret_resolver=lambda *_: "must-not-resolve",
        worker_path=_worker(tmp_path),
        launcher=fail_launch,
    )
    with pytest.raises(RuntimeError, match="launch failed"):
        supervisor.start("agt_1", "cli_1", "cred_1", 1, lambda _: None)

    _assert_pipe_descriptors_closed(opened)
    supervisor.close()


def test_macos_temp_setup_failure_closes_all_launch_fds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = _track_current_thread_pipes(monkeypatch)
    resolved = False

    def fail_temp(*_args, **_kwargs):
        raise OSError("injected temp failure")

    def resolve(*_args: str) -> str:
        nonlocal resolved
        resolved = True
        return "must-not-resolve"

    monkeypatch.setattr(
        "src.autonomous.supervisor.employee_channels.tempfile.mkdtemp",
        fail_temp,
    )
    supervisor = EmployeeChannelSupervisor(
        secret_resolver=resolve,
        worker_path=_worker(tmp_path),
        launcher=subprocess.Popen,
        platform_name="darwin",
    )
    with pytest.raises(ChannelSandboxUnavailable):
        supervisor.start("agt_1", "cli_1", "cred_1", 1, lambda _: None)

    assert resolved is False
    _assert_pipe_descriptors_closed(opened)
    supervisor.close()


def test_bwrap_info_metadata_is_bounded_and_strictly_typed() -> None:
    read_fd, write_fd = os.pipe()
    os.write(write_fd, json.dumps({"child-pid": 1234, "future": 1}).encode())
    os.close(write_fd)
    try:
        assert _read_sandbox_metadata(read_fd)["child-pid"] == 1234
    finally:
        os.close(read_fd)

    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"[1234]")
    os.close(write_fd)
    try:
        with pytest.raises(ValueError, match="object"):
            _read_sandbox_metadata(read_fd)
    finally:
        os.close(read_fd)


def test_missing_macos_seatbelt_fails_before_secret_resolution(tmp_path: Path) -> None:
    resolved = False

    def resolve(*_args: str) -> str:
        nonlocal resolved
        resolved = True
        return "must-not-be-delivered"

    supervisor = EmployeeChannelSupervisor(
        secret_resolver=resolve,
        worker_path=_worker(tmp_path),
        platform_name="darwin",
    )
    try:
        with pytest.raises(ChannelSandboxUnavailable):
            supervisor.start("agt_mac", "cli_mac", "cred_mac", 1, lambda _: None)
        assert resolved is False
    finally:
        supervisor.close()


def test_platform_override_cannot_change_production_fallback_policy() -> None:
    other_platform = "darwin" if sys.platform != "darwin" else "linux"
    with pytest.raises(ValueError, match="platform override"):
        EmployeeChannelSupervisor(
            secret_resolver=lambda *_: "unused",
            platform_name=other_platform,
        )


def test_macos_proof_precedes_secret_resolution_and_bootstrap(tmp_path: Path) -> None:
    worker = _mac_proof_worker(tmp_path)
    proof_marker = tmp_path / "proof-emitted"
    launches: list[tuple[tuple[str, ...], dict[str, object]]] = []
    sandbox_temp: Path | None = None

    def launch(argv, **kwargs):
        launches.append((tuple(argv), dict(kwargs)))
        command_start = tuple(argv).index(sys.executable)
        return subprocess.Popen(tuple(argv)[command_start:], **kwargs)

    def resolve(*_args: str) -> str:
        assert proof_marker.is_file()
        return "employee-secret"

    supervisor = EmployeeChannelSupervisor(
        secret_resolver=resolve,
        worker_path=worker,
        launcher=launch,
        platform_name="darwin",
        ready_timeout=1.0,
    )
    try:
        status = supervisor.start(
            "agt_mac",
            "cli_mac",
            "cred_mac",
            1,
            lambda _: None,
        )
        assert status.state is ChannelProcessState.READY
        assert status.sandbox is not None
        assert status.sandbox.verified is True
        assert status.sandbox.mechanism == "seatbelt-filesystem"
        launch_argv, launch_options = launches[0]
        sandbox_temp = Path(launch_options["env"]["GHOSTAP_CHANNEL_TMP"])
        assert sandbox_temp.is_dir()
        assert sandbox_temp.stat().st_mode & 0o777 == 0o700
        assert f"GHOSTAP_TEMP={sandbox_temp}" in launch_argv
    finally:
        supervisor.close()
    assert sandbox_temp is not None
    assert not sandbox_temp.exists()


def test_macos_invalid_denial_proof_never_resolves_secret(tmp_path: Path) -> None:
    resolved = False
    worker = _mac_proof_worker(tmp_path, denied_errno=errno.ENOENT)

    def launch(argv, **kwargs):
        command_start = tuple(argv).index(sys.executable)
        return subprocess.Popen(tuple(argv)[command_start:], **kwargs)

    def resolve(*_args: str) -> str:
        nonlocal resolved
        resolved = True
        return "must-not-be-delivered"

    supervisor = EmployeeChannelSupervisor(
        secret_resolver=resolve,
        worker_path=worker,
        launcher=launch,
        platform_name="darwin",
    )
    try:
        with pytest.raises(ChannelSandboxUnavailable):
            supervisor.start("agt_mac", "cli_mac", "cred_mac", 1, lambda _: None)
        assert resolved is False
    finally:
        supervisor.close()


@pytest.mark.parametrize("ptrace_result", [0, -1])
def test_darwin_worker_hardening_denies_debug_attach(
    monkeypatch: pytest.MonkeyPatch,
    ptrace_result: int,
) -> None:
    calls: list[tuple[int, int, object, int]] = []

    class FakePtrace:
        argtypes: list[object] = []
        restype: object = None

        def __call__(self, request: int, pid: int, address: object, data: int) -> int:
            calls.append((request, pid, address, data))
            return ptrace_result

    class FakeLibC:
        ptrace = FakePtrace()

    limits: list[tuple[int, tuple[int, int]]] = []
    monkeypatch.setattr(channel_worker_module.sys, "platform", "darwin")
    monkeypatch.setattr(channel_worker_module.ctypes, "CDLL", lambda *_args, **_kwargs: FakeLibC())
    monkeypatch.setattr(
        channel_worker_module.resource,
        "setrlimit",
        lambda kind, value: limits.append((kind, value)),
    )

    if ptrace_result == 0:
        channel_worker_module.apply_process_hardening()
    else:
        with pytest.raises(
            channel_worker_module.WorkerSecurityError,
            match="debug attachment",
        ):
            channel_worker_module.apply_process_hardening()

    assert limits == [(channel_worker_module.resource.RLIMIT_CORE, (0, 0))]
    assert calls == [(31, 0, None, 0)]


def test_macos_worker_proof_requires_permission_denial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open = os.open

    def sandboxed_open(path, flags):
        if Path(path).name == "AGENTS.md":
            raise PermissionError(errno.EACCES, "seatbelt denied", str(path))
        return original_open("/dev/null", flags)

    monkeypatch.setattr(channel_worker_module.sys, "platform", "darwin")
    monkeypatch.setattr(channel_worker_module.os, "open", sandboxed_open)
    read_fd, write_fd = os.pipe()

    channel_worker_module.emit_macos_sandbox_proof(write_fd, "c" * 32)
    proof = json.loads(os.read(read_fd, 4096))
    os.close(read_fd)

    assert proof["nonce"] == "c" * 32
    assert proof["repository_canary_errno"] == errno.EACCES
    assert proof["source_readable"] is True
    assert proof["runtime_readable"] is True
