from __future__ import annotations

import fcntl
import json
import math
import multiprocessing
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.tasking import TaskScheduler, TaskSpec, TaskStatus
from src.utils.restart_gate import (
    EX_TEMPFAIL,
    RestartGate,
    RestartGateError,
    RestartRunStatus,
    run_restart_worker,
)
from src.utils.restart_gate import (
    main as restart_gate_main,
)

_PROCESS_TIMEOUT = 10.0
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _ready_gate(
    tmp_path: Path,
    *,
    project_dir: Path | None = None,
    gate_dir: Path | None = None,
) -> RestartGate:
    project = project_dir or (tmp_path / "project")
    project.mkdir(exist_ok=True)
    gate = RestartGate.for_project(
        project,
        override=gate_dir or (tmp_path / "gate"),
    )
    gate.publish_participation(service_pid=os.getpid())
    gate.mark_ready(service_pid=os.getpid())
    return gate


def test_generated_restart_token_is_safe_as_standalone_cli_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.utils.restart_gate as restart_gate

    monkeypatch.setattr(
        restart_gate.secrets,
        "token_urlsafe",
        lambda _size: "-" + ("A" * 30),
    )

    token = restart_gate._new_token()

    assert token.startswith("g-")
    assert 20 <= len(token) <= 64


def _guard_holder(
    gate_dir: str,
    entered,
    release,
) -> None:
    gate = RestartGate(Path(gate_dir))
    with gate.task_guard():
        entered.set()
        if not release.wait(_PROCESS_TIMEOUT):
            raise TimeoutError("guard holder was not released")


def _late_guard(
    gate_dir: str,
    entered,
) -> None:
    gate = RestartGate(Path(gate_dir))
    with gate.task_guard():
        entered.set()


def _restart_holder(
    gate_dir: str,
    expected: str,
    operation_entered,
    operation_release,
    result_queue,
) -> None:
    gate = RestartGate(Path(gate_dir))

    def operation(_remaining: float) -> int:
        operation_entered.set()
        if not operation_release.wait(_PROCESS_TIMEOUT):
            raise TimeoutError("restart operation was not released")
        return 0

    result = gate.run_if_current(
        expected,
        timeout=_PROCESS_TIMEOUT,
        operation=operation,
    )
    result_queue.put((result.status.value, result.exit_code))


def _coalescing_worker(
    gate_dir: str,
    expected: str,
    start,
    operation_log: str,
    result_queue,
) -> None:
    gate = RestartGate(Path(gate_dir))
    if not start.wait(_PROCESS_TIMEOUT):
        raise TimeoutError("coalescing worker did not start")

    def operation(_remaining: float) -> int:
        fd = os.open(operation_log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, f"{os.getpid()}\n".encode())
            os.fsync(fd)
            time.sleep(0.15)
        finally:
            os.close(fd)
        return 0

    result = gate.run_if_current(
        expected,
        timeout=_PROCESS_TIMEOUT,
        operation=operation,
    )
    result_queue.put((result.status.value, result.exit_code))


def _snapshot_worker(gate_dir: str, start, result_queue) -> None:
    gate = RestartGate(Path(gate_dir))
    if not start.wait(_PROCESS_TIMEOUT):
        raise TimeoutError("snapshot worker did not start")
    result_queue.put(gate.snapshot())


def _wait_until_exclusive_holder(path: Path, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        fd = os.open(path, os.O_RDWR)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
        time.sleep(0.01)
    raise AssertionError(f"no exclusive lock observed for {path}")


def _join(process) -> None:
    process.join(_PROCESS_TIMEOUT)
    if process.is_alive():
        process.terminate()
        process.join(2)
        raise AssertionError(f"process {process.pid} did not exit")
    assert process.exitcode == 0


@pytest.mark.skipif(sys.platform not in {"linux", "darwin"}, reason="flock gate is POSIX-only")
def test_restart_fences_admission_then_drains_existing_tasks(tmp_path: Path) -> None:
    ctx = multiprocessing.get_context("spawn")
    gate_dir = tmp_path / "gate"
    gate = _ready_gate(tmp_path, gate_dir=gate_dir)
    expected = gate.snapshot()

    current_entered = ctx.Event()
    current_release = ctx.Event()
    operation_entered = ctx.Event()
    operation_release = ctx.Event()
    late_entered = ctx.Event()
    result_queue = ctx.Queue()

    current = ctx.Process(
        target=_guard_holder,
        args=(str(gate_dir), current_entered, current_release),
    )
    current.start()
    assert current_entered.wait(3)

    restart = ctx.Process(
        target=_restart_holder,
        args=(
            str(gate_dir),
            expected,
            operation_entered,
            operation_release,
            result_queue,
        ),
    )
    restart.start()
    _wait_until_exclusive_holder(gate.admission_path)

    late = ctx.Process(target=_late_guard, args=(str(gate_dir), late_entered))
    late.start()
    assert not late_entered.wait(0.2)
    assert not operation_entered.is_set()

    current_release.set()
    assert operation_entered.wait(3)
    assert not late_entered.wait(0.2)

    operation_release.set()
    assert late_entered.wait(3)

    _join(current)
    _join(restart)
    _join(late)
    assert result_queue.get(timeout=1) == (RestartRunStatus.RESTARTED.value, 0)


@pytest.mark.skipif(
    sys.platform != "linux" or not Path("/proc/self/fd").exists(),
    reason="exec descriptor inspection requires Linux procfs",
)
def test_task_guard_descriptors_are_close_on_exec_even_when_close_fds_is_false(
    tmp_path: Path,
) -> None:
    gate = RestartGate(tmp_path / "gate")

    with gate.task_guard():
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import json, os\n"
                    "paths=[]\n"
                    "for name in os.listdir('/proc/self/fd'):\n"
                    " try: paths.append(os.readlink('/proc/self/fd/'+name))\n"
                    " except OSError: pass\n"
                    "print(json.dumps(paths))\n"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            close_fds=False,
        )

    inherited = json.loads(child.stdout)
    assert str(gate.admission_path) not in inherited
    assert str(gate.drain_path) not in inherited


@pytest.mark.skipif(sys.platform not in {"linux", "darwin"}, reason="flock gate is POSIX-only")
def test_same_generation_restart_workers_coalesce_to_one_operation(
    tmp_path: Path,
) -> None:
    ctx = multiprocessing.get_context("spawn")
    gate_dir = tmp_path / "gate"
    gate = _ready_gate(tmp_path, gate_dir=gate_dir)
    expected = gate.snapshot()
    operation_log = tmp_path / "operations.log"
    start = ctx.Event()
    result_queue = ctx.Queue()

    workers = [
        ctx.Process(
            target=_coalescing_worker,
            args=(
                str(gate_dir),
                expected,
                start,
                str(operation_log),
                result_queue,
            ),
        )
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    start.set()
    for worker in workers:
        _join(worker)

    results = sorted(result_queue.get(timeout=1) for _ in workers)
    assert results == sorted(
        [
            (RestartRunStatus.COALESCED.value, 0),
            (RestartRunStatus.RESTARTED.value, 0),
        ]
    )
    assert len(operation_log.read_text(encoding="utf-8").splitlines()) == 1
    assert gate.snapshot() != expected


@pytest.mark.skipif(sys.platform not in {"linux", "darwin"}, reason="flock gate is POSIX-only")
def test_ready_snapshot_is_concurrent_and_token_is_restricted_ascii(
    tmp_path: Path,
) -> None:
    ctx = multiprocessing.get_context("spawn")
    gate_dir = tmp_path / "gate"
    gate = _ready_gate(tmp_path, gate_dir=gate_dir)
    expected = gate.snapshot()
    start = ctx.Event()
    result_queue = ctx.Queue()
    workers = [
        ctx.Process(
            target=_snapshot_worker,
            args=(str(gate_dir), start, result_queue),
        )
        for _ in range(4)
    ]
    for worker in workers:
        worker.start()
    start.set()
    for worker in workers:
        _join(worker)

    tokens = [result_queue.get(timeout=1) for _ in workers]
    assert len(set(tokens)) == 1
    assert tokens == [expected] * len(tokens)
    assert re.fullmatch(r"[A-Za-z0-9_-]{20,64}", tokens[0])


@pytest.mark.skipif(sys.platform not in {"linux", "darwin"}, reason="flock gate is POSIX-only")
def test_lock_inodes_are_stable_and_symlink_lock_is_rejected(
    tmp_path: Path,
) -> None:
    gate_dir = tmp_path / "gate"
    first = RestartGate(gate_dir)
    inodes = (first.admission_path.stat().st_ino, first.drain_path.stat().st_ino)

    second = RestartGate(gate_dir)
    assert (
        second.admission_path.stat().st_ino,
        second.drain_path.stat().st_ino,
    ) == inodes

    malicious = tmp_path / "malicious"
    malicious.mkdir(mode=0o700)
    target = tmp_path / "target"
    target.write_text("do not follow", encoding="utf-8")
    (malicious / "admission.lock").symlink_to(target)
    with pytest.raises(RestartGateError, match="cannot open restart lock"):
        RestartGate(malicious)


@pytest.mark.skipif(sys.platform not in {"linux", "darwin"}, reason="flock gate is POSIX-only")
def test_existing_gate_directory_must_already_be_private_and_is_not_chmodded(
    tmp_path: Path,
) -> None:
    gate_dir = tmp_path / "unsafe-gate"
    gate_dir.mkdir(mode=0o755)
    gate_dir.chmod(0o755)

    with pytest.raises(RestartGateError, match="unsafe permissions"):
        RestartGate(gate_dir)

    assert stat.S_IMODE(gate_dir.stat().st_mode) == 0o755


@pytest.mark.skipif(sys.platform not in {"linux", "darwin"}, reason="flock gate is POSIX-only")
def test_existing_gate_directory_must_be_owned_by_current_uid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate_dir = tmp_path / "foreign-gate"
    gate_dir.mkdir(mode=0o700)
    gate_dir.chmod(0o700)
    actual_uid = os.getuid()
    monkeypatch.setattr(
        "src.utils.restart_gate.os.getuid",
        lambda: actual_uid + 1,
    )

    with pytest.raises(RestartGateError, match="not owned by current user"):
        RestartGate(gate_dir)


@pytest.mark.skipif(sys.platform not in {"linux", "darwin"}, reason="flock gate is POSIX-only")
@pytest.mark.parametrize("broad_path", [Path("/"), Path("/tmp"), Path("/var/tmp")])
def test_gate_rejects_broad_shared_directories_without_mutating_them(
    broad_path: Path,
) -> None:
    if not broad_path.exists():
        pytest.skip(f"{broad_path} is absent on this platform")
    before = broad_path.stat()

    with pytest.raises(RestartGateError, match="dedicated"):
        RestartGate(broad_path)

    after = broad_path.stat()
    assert stat.S_IMODE(after.st_mode) == stat.S_IMODE(before.st_mode)
    assert after.st_uid == before.st_uid


@pytest.mark.skipif(sys.platform not in {"linux", "darwin"}, reason="flock gate is POSIX-only")
def test_existing_lock_must_be_private_owned_regular_single_link(
    tmp_path: Path,
) -> None:
    gate_dir = tmp_path / "gate"
    gate_dir.mkdir(mode=0o700)
    gate_dir.chmod(0o700)
    target = tmp_path / "target"
    target.write_text("sensitive", encoding="utf-8")
    target.chmod(0o644)
    os.link(target, gate_dir / "admission.lock")

    with pytest.raises(RestartGateError, match="single-link"):
        RestartGate(gate_dir)

    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    assert target.read_text(encoding="utf-8") == "sensitive"


@pytest.mark.skipif(sys.platform not in {"linux", "darwin"}, reason="flock gate is POSIX-only")
def test_existing_lock_permissions_are_rejected_without_fchmod(
    tmp_path: Path,
) -> None:
    gate_dir = tmp_path / "gate"
    gate_dir.mkdir(mode=0o700)
    gate_dir.chmod(0o700)
    lock = gate_dir / "admission.lock"
    lock.write_text("", encoding="utf-8")
    lock.chmod(0o640)

    with pytest.raises(RestartGateError, match="unsafe permissions"):
        RestartGate(gate_dir)

    assert stat.S_IMODE(lock.stat().st_mode) == 0o640


@pytest.mark.skipif(sys.platform not in {"linux", "darwin"}, reason="flock gate is POSIX-only")
def test_existing_lock_must_be_owned_by_current_uid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate_dir = tmp_path / "gate"
    gate_dir.mkdir(mode=0o700)
    gate_dir.chmod(0o700)
    lock = gate_dir / "admission.lock"
    lock.write_text("", encoding="utf-8")
    lock.chmod(0o600)
    actual_uid = os.getuid()
    monkeypatch.setattr(RestartGate, "_prepare_directory", lambda _self: None)
    monkeypatch.setattr(
        "src.utils.restart_gate.os.getuid",
        lambda: actual_uid + 1,
    )

    with pytest.raises(RestartGateError, match="not owned by current user"):
        RestartGate(gate_dir)


@pytest.mark.skipif(sys.platform not in {"linux", "darwin"}, reason="flock gate is POSIX-only")
def test_new_lock_files_are_private_and_use_distinct_inodes(tmp_path: Path) -> None:
    gate = RestartGate(tmp_path / "gate")
    admission = gate.admission_path.stat()
    drain = gate.drain_path.stat()

    assert stat.S_IMODE(admission.st_mode) == 0o600
    assert stat.S_IMODE(drain.st_mode) == 0o600
    assert admission.st_uid == os.getuid()
    assert drain.st_uid == os.getuid()
    assert admission.st_nlink == drain.st_nlink == 1
    assert (admission.st_dev, admission.st_ino) != (drain.st_dev, drain.st_ino)


def test_project_gate_rejects_relative_override(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="override must be an absolute path"):
        RestartGate.for_project(tmp_path, override="shared-gate")


def test_project_gate_rejects_project_root_as_override(tmp_path: Path) -> None:
    with pytest.raises(RestartGateError, match="must not be the project root"):
        RestartGate.for_project(tmp_path, override=tmp_path)


def test_default_gate_lives_outside_checkout_and_binds_canonical_owner(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    alias = tmp_path / "project-alias"
    alias.symlink_to(project, target_is_directory=True)

    gate = RestartGate.for_project(alias)

    assert not gate.directory.is_relative_to(project)
    owner = json.loads((gate.directory / "owner.json").read_text(encoding="utf-8"))
    assert owner["project_dir"] == str(project.resolve())


def test_override_cannot_be_reused_by_a_different_checkout(tmp_path: Path) -> None:
    first_project = tmp_path / "first"
    second_project = tmp_path / "second"
    first_project.mkdir()
    second_project.mkdir()
    shared_override = tmp_path / "shared-gate"

    RestartGate.for_project(first_project, override=shared_override)

    with pytest.raises(RestartGateError, match="different checkout"):
        RestartGate.for_project(second_project, override=shared_override)


def test_snapshot_without_participation_fails_without_creating_proof_or_generation(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    gate = RestartGate.for_project(project, override=tmp_path / "gate")

    with pytest.raises(RestartGateError, match="participation.*missing"):
        gate.snapshot()

    assert not (gate.directory / "participation.json").exists()
    assert not gate.generation_path.exists()


def test_running_instance_explicitly_publishes_verifiable_participation(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    gate = RestartGate.for_project(project, override=tmp_path / "gate")

    participation_id = gate.publish_participation(
        service_pid=os.getpid(),
        instance_id="I" * 24,
    )
    with pytest.raises(RestartGateError, match="not ready"):
        gate.snapshot()
    generation = gate.mark_ready(service_pid=os.getpid())

    assert gate.snapshot() == generation
    proof = json.loads(
        (gate.directory / "participation.json").read_text(encoding="utf-8")
    )
    assert proof["project_dir"] == str(project.resolve())
    assert proof["gate_dir"] == str(gate.directory.resolve())
    assert proof["service_pid"] == os.getpid()
    assert proof["instance_id"] == "I" * 24
    assert participation_id == proof["instance_id"]
    assert proof["directory_identity"]
    assert proof["admission_identity"]
    assert proof["drain_identity"]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("service_pid", 999_999_999),
        ("instance_id", "J" * 24),
        ("process_instance", "stale-process-instance"),
        ("directory_identity", [0, 0]),
        ("admission_identity", [0, 0]),
        ("drain_identity", [0, 0]),
    ],
)
def test_snapshot_rejects_participation_identity_drift(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    gate = RestartGate.for_project(project, override=tmp_path / "gate")
    gate.publish_participation(
        service_pid=os.getpid(),
        instance_id="I" * 24,
    )
    gate.mark_ready(service_pid=os.getpid())
    proof_path = gate.directory / "participation.json"
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof[field] = replacement
    proof_path.write_text(json.dumps(proof), encoding="utf-8")
    proof_path.chmod(0o600)

    with pytest.raises(RestartGateError, match="participation"):
        gate.snapshot()


def test_snapshot_rejects_replaced_lock_inode(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    gate = RestartGate.for_project(project, override=tmp_path / "gate")
    gate.publish_participation(service_pid=os.getpid())
    gate.mark_ready(service_pid=os.getpid())

    old_fd = os.open(gate.admission_path, os.O_RDONLY)
    try:
        gate.admission_path.unlink()
        gate.admission_path.touch(mode=0o600)

        with pytest.raises(RestartGateError, match="identity changed"):
            gate.snapshot()
    finally:
        os.close(old_fd)


def test_worker_rejects_replaced_gate_directory_even_if_files_are_copied(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    gate = RestartGate.for_project(project, override=tmp_path / "gate")
    gate.publish_participation(service_pid=os.getpid())
    generation = gate.mark_ready(service_pid=os.getpid())
    original = gate.directory
    moved = tmp_path / "old-gate"
    original.rename(moved)
    shutil.copytree(moved, original)

    with pytest.raises(RestartGateError, match="participation"):
        RestartGate.for_worker(project, expected_generation=generation)


def test_worker_uses_locator_pinned_gate_when_environment_override_changes(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    original_gate = tmp_path / "original-gate"
    gate = RestartGate.for_project(project, override=original_gate)
    gate.publish_participation(service_pid=os.getpid())
    generation = gate.mark_ready(service_pid=os.getpid())

    worker_gate = RestartGate.for_worker(
        project,
        expected_generation=generation,
        configured_override=tmp_path / "changed-by-env",
    )

    assert worker_gate.directory == original_gate
    assert not (tmp_path / "changed-by-env").exists()


def test_worker_fails_closed_when_pinned_participation_instance_changes(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    gate = RestartGate.for_project(project, override=tmp_path / "gate")
    gate.publish_participation(service_pid=os.getpid())
    generation = gate.mark_ready(service_pid=os.getpid())
    worker_gate = RestartGate.for_worker(
        project,
        expected_generation=generation,
    )

    gate.publish_participation(service_pid=os.getpid())
    gate.mark_ready(service_pid=os.getpid())

    with pytest.raises(RestartGateError, match="instance identity changed"):
        worker_gate.run_if_current(
            generation,
            timeout=1,
            operation=lambda _remaining: pytest.fail("stale worker must not run"),
        )


def test_worker_missing_participation_proof_fails_without_recreating_it(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    gate = RestartGate.for_project(project, override=tmp_path / "gate")
    gate.publish_participation(service_pid=os.getpid())
    generation = gate.mark_ready(service_pid=os.getpid())
    gate.participation_path.unlink()

    with pytest.raises(RestartGateError, match="participation.*missing"):
        RestartGate.for_worker(project, expected_generation=generation)

    assert not gate.participation_path.exists()


def test_ready_generation_requires_marked_ready_and_exact_service_pid(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    gate = RestartGate.for_project(project, override=tmp_path / "gate")
    gate.publish_participation(service_pid=os.getpid())

    with pytest.raises(RestartGateError, match="not ready"):
        gate.ready_generation(service_pid=os.getpid())

    generation = gate.mark_ready(service_pid=os.getpid())
    with pytest.raises(RestartGateError, match="service PID"):
        gate.ready_generation(service_pid=os.getpid() + 1)
    assert gate.ready_generation(service_pid=os.getpid()) == generation


def test_ready_cli_uses_locator_and_prints_only_matching_ready_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import src.config

    project = tmp_path / "project"
    project.mkdir()
    original_gate = tmp_path / "original-gate"
    gate = RestartGate.for_project(project, override=original_gate)
    gate.publish_participation(service_pid=os.getpid())
    generation = gate.mark_ready(service_pid=os.getpid())
    monkeypatch.setattr(
        src.config,
        "get_settings",
        lambda: SimpleNamespace(
            restart_gate_dir=str(tmp_path / "changed-by-env"),
            restart_gate_timeout=9.0,
        ),
    )

    result = restart_gate_main(
        [
            "ready",
            "--project-dir",
            str(project),
            "--service-pid",
            str(os.getpid()),
        ]
    )

    assert result == 0
    assert capsys.readouterr().out.strip() == generation
    assert not (tmp_path / "changed-by-env").exists()


def test_failed_operation_atomically_consumes_generation_and_coalesces_retry(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    gate = RestartGate.for_project(project, override=tmp_path / "gate")
    gate.publish_participation(service_pid=os.getpid())
    expected = gate.mark_ready(service_pid=os.getpid())
    operation_calls = 0

    def operation(_remaining: float) -> int:
        nonlocal operation_calls
        operation_calls += 1
        return 23

    failed = gate.run_if_current(expected, timeout=1, operation=operation)
    retry = gate.run_if_current(expected, timeout=1, operation=operation)

    assert failed.status == RestartRunStatus.FAILED
    assert failed.exit_code == 23
    assert retry.status == RestartRunStatus.COALESCED
    assert retry.exit_code == 0
    assert operation_calls == 1
    with pytest.raises(RestartGateError, match="failed terminal"):
        gate.snapshot()


def test_successful_operation_advances_generation_when_service_does_not_publish(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    gate = RestartGate.for_project(project, override=tmp_path / "gate")
    gate.publish_participation(service_pid=os.getpid())
    expected = gate.mark_ready(service_pid=os.getpid())

    result = gate.run_if_current(
        expected,
        timeout=1,
        operation=lambda _remaining: 0,
    )

    assert result.status == RestartRunStatus.RESTARTED
    assert gate.snapshot() != expected


@pytest.mark.skipif(sys.platform not in {"linux", "darwin"}, reason="flock gate is POSIX-only")
def test_restart_timeout_is_fail_closed_and_preserves_generation(
    tmp_path: Path,
) -> None:
    ctx = multiprocessing.get_context("spawn")
    gate = _ready_gate(tmp_path)
    expected = gate.snapshot()
    entered = ctx.Event()
    release = ctx.Event()
    holder = ctx.Process(
        target=_guard_holder,
        args=(str(gate.directory), entered, release),
    )
    holder.start()
    assert entered.wait(3)

    operation_called = False

    def operation(_remaining: float) -> int:
        nonlocal operation_called
        operation_called = True
        return 0

    try:
        result = gate.run_if_current(expected, timeout=0.15, operation=operation)
    finally:
        release.set()
        _join(holder)

    assert result.status == RestartRunStatus.TIMED_OUT
    assert result.exit_code == EX_TEMPFAIL
    assert not operation_called
    assert gate.snapshot() == expected


@pytest.mark.skipif(sys.platform not in {"linux", "darwin"}, reason="flock gate is POSIX-only")
def test_failed_restart_operation_publishes_failed_terminal(
    tmp_path: Path,
) -> None:
    gate = _ready_gate(tmp_path)
    expected = gate.snapshot()

    result = gate.run_if_current(
        expected,
        timeout=1,
        operation=lambda _remaining: 23,
    )

    assert result.status == RestartRunStatus.FAILED
    assert result.exit_code == 23
    with pytest.raises(RestartGateError, match="failed terminal"):
        gate.snapshot()


def test_restart_worker_uses_bounded_shell_free_close_fds_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _ready_gate(tmp_path)
    expected = gate.snapshot()
    calls: list[dict[str, object]] = []
    monkeypatch.setattr("src.utils.restart_gate.time.sleep", lambda _delay: None)

    class FakeProcess:
        pid = 4242
        returncode = 0

        def __init__(self, command, **kwargs):
            calls.append({"command": command, **kwargs})

        def wait(self, timeout=None):
            calls[-1]["wait_timeout"] = timeout
            return self.returncode

    def fake_popen(command, **kwargs):
        calls.append({"command": command, **kwargs})
        calls.pop()
        return FakeProcess(command, **kwargs)

    monkeypatch.setattr("src.utils.restart_gate.subprocess.Popen", fake_popen)

    result = run_restart_worker(
        gate=gate,
        expected_generation=expected,
        restart_script=tmp_path / "restart.sh",
        log_file=tmp_path / "restart.log",
        delay=0.1,
        timeout=7.0,
    )

    assert result == 0
    assert len(calls) == 1
    call = calls[0]
    assert call["command"] == [str(tmp_path / "restart.sh"), "restart"]
    assert call["shell"] is False
    assert call["close_fds"] is True
    assert call["start_new_session"] is True
    assert 0 < float(call["wait_timeout"]) <= 7.0
    assert call["stdout"] is call["stderr"]
    assert call["env"]["GHOSTAP_LOG_MODE"] == "append"
    assert call["env"]["GHOSTAP_RESTART_GATE_DIR"] == str(gate.directory)


def test_restart_worker_subprocess_timeout_returns_tempfail_without_generation_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _ready_gate(tmp_path)
    expected = gate.snapshot()
    monkeypatch.setattr("src.utils.restart_gate.time.sleep", lambda _delay: None)
    wait_calls: list[float | None] = []
    signals: list[tuple[int, int]] = []

    class TimeoutProcess:
        pid = 4343
        returncode = None

        def wait(self, timeout=None):
            wait_calls.append(timeout)
            if len(wait_calls) < 3:
                raise subprocess.TimeoutExpired(["restart.sh", "restart"], timeout)
            self.returncode = -signal.SIGKILL
            return self.returncode

    monkeypatch.setattr(
        "src.utils.restart_gate.subprocess.Popen",
        lambda *_args, **_kwargs: TimeoutProcess(),
    )
    monkeypatch.setattr(
        "src.utils.restart_gate.os.killpg",
        lambda pgid, signum: signals.append((pgid, signum)),
    )

    result = run_restart_worker(
        gate=gate,
        expected_generation=expected,
        restart_script=tmp_path / "restart.sh",
        log_file=tmp_path / "restart.log",
        delay=0,
        timeout=0.1,
    )

    assert result == EX_TEMPFAIL
    assert gate.snapshot() == expected
    assert signals == [
        (TimeoutProcess.pid, signal.SIGTERM),
        (TimeoutProcess.pid, 0),
        (TimeoutProcess.pid, signal.SIGKILL),
    ]
    assert len(wait_calls) == 3


@pytest.mark.skipif(
    sys.platform != "linux" or not Path("/proc").exists(),
    reason="process-group descendant verification uses Linux procfs",
)
def test_restart_worker_timeout_kills_stubborn_descendant_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _ready_gate(tmp_path)
    expected = gate.snapshot()
    child_pid_file = tmp_path / "child.pid"
    fake_restart = tmp_path / "fake-restart.sh"
    fake_restart.write_text(
        (
            "#!/bin/bash\n"
            "trap '' TERM\n"
            "(\n"
            "  trap '' TERM\n"
            '  printf "%s\\n" "$BASHPID" > "$FAKE_CHILD_PID_FILE"\n'
            "  while :; do sleep 10; done\n"
            ") &\n"
            "while [ ! -s \"$FAKE_CHILD_PID_FILE\" ]; do sleep 0.01; done\n"
            "while :; do sleep 10; done\n"
        ),
        encoding="utf-8",
    )
    fake_restart.chmod(0o700)
    monkeypatch.setenv("FAKE_CHILD_PID_FILE", str(child_pid_file))

    result = run_restart_worker(
        gate=gate,
        expected_generation=expected,
        restart_script=fake_restart,
        log_file=tmp_path / "restart.log",
        delay=0,
        timeout=0.4,
    )

    assert result == EX_TEMPFAIL
    child_pid = int(child_pid_file.read_text(encoding="utf-8").strip())
    deadline = time.monotonic() + 2
    while True:
        proc_stat = Path(f"/proc/{child_pid}/stat")
        try:
            state = proc_stat.read_text(encoding="utf-8").rsplit(")", 1)[1].split()[0]
        except FileNotFoundError:
            break
        if state == "Z":
            break
        if time.monotonic() >= deadline:
            raise AssertionError(f"restart descendant still alive pid={child_pid}")
        time.sleep(0.02)


def test_restart_worker_never_resamples_generation_after_detach(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _ready_gate(tmp_path)
    stale_generation = gate.snapshot()
    gate.publish_generation()
    monkeypatch.setattr(
        "src.utils.restart_gate.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail("stale worker must coalesce"),
    )

    result = run_restart_worker(
        gate=gate,
        expected_generation=stale_generation,
        restart_script=tmp_path / "restart.sh",
        log_file=tmp_path / "restart.log",
        delay=0,
        timeout=1,
    )

    assert result == 0
    assert "status=coalesced" in (tmp_path / "restart.log").read_text(
        encoding="utf-8"
    )


def test_worker_delay_and_initial_log_share_the_restart_timeout_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _ready_gate(tmp_path)
    expected = gate.snapshot()
    clock = iter([10.0, 10.7, 10.8])
    monkeypatch.setattr(
        "src.utils.restart_gate.time.monotonic",
        lambda: next(clock),
    )
    monkeypatch.setattr("src.utils.restart_gate.time.sleep", lambda _delay: None)
    captured: dict[str, float] = {}

    def fake_run_if_current(_expected, *, timeout, operation):
        captured["timeout"] = timeout
        return SimpleNamespace(
            status=RestartRunStatus.COALESCED,
            exit_code=0,
        )

    monkeypatch.setattr(gate, "run_if_current", fake_run_if_current)

    assert (
        run_restart_worker(
            gate=gate,
            expected_generation=expected,
            restart_script=tmp_path / "restart.sh",
            log_file=tmp_path / "restart.log",
            delay=0.7,
            timeout=1.0,
        )
        == 0
    )
    assert captured["timeout"] == pytest.approx(0.2)


def test_worker_cli_executes_only_temp_fake_restart_operation(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    fake_restart = project / "fake-restart.sh"
    calls = tmp_path / "calls"
    fake_restart.write_text(
        (
            "#!/bin/bash\n"
            'printf "%s\\n" "$*" >> "$FAKE_RESTART_CALLS"\n'
            "exit 0\n"
        ),
        encoding="utf-8",
    )
    fake_restart.chmod(0o700)
    gate = _ready_gate(
        tmp_path,
        project_dir=project,
        gate_dir=tmp_path / "gate",
    )
    expected = gate.snapshot()
    log_file = tmp_path / "worker.log"
    environment = os.environ.copy()
    environment["FAKE_RESTART_CALLS"] = str(calls)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.utils.restart_gate",
            "worker",
            "--project-dir",
            str(project),
            "--gate-dir",
            str(gate.directory),
            "--restart-script",
            str(fake_restart),
            "--log-file",
            str(log_file),
            "--delay",
            "0",
            "--expected-generation",
            expected,
            "--timeout",
            "2",
        ],
        cwd=_REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert calls.read_text(encoding="utf-8").splitlines() == ["restart"]
    assert gate.snapshot() != expected
    assert "status=restarted exit_code=0" in log_file.read_text(encoding="utf-8")


def test_snapshot_cli_prints_generation_for_synchronous_rr_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import src.config

    project = tmp_path / "project"
    project.mkdir()
    gate = _ready_gate(tmp_path, project_dir=project)
    monkeypatch.setattr(
        src.config,
        "get_settings",
        lambda: SimpleNamespace(
            restart_gate_dir="",
            restart_gate_timeout=9.0,
        ),
    )
    original_cwd = Path.cwd()
    try:
        result = restart_gate_main(["snapshot", "--project-dir", str(project)])
    finally:
        os.chdir(original_cwd)

    token = capsys.readouterr().out.strip()
    assert result == 0
    assert token == gate.snapshot()


def test_restart_gate_cli_restores_callers_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.config

    caller_cwd = tmp_path / "caller"
    project = tmp_path / "project"
    caller_cwd.mkdir()
    project.mkdir()
    _ready_gate(tmp_path, project_dir=project)
    monkeypatch.chdir(caller_cwd)
    monkeypatch.setattr(
        src.config,
        "get_settings",
        lambda: SimpleNamespace(
            restart_gate_dir="",
            restart_gate_timeout=9.0,
        ),
    )

    assert restart_gate_main(["snapshot", "--project-dir", str(project)]) == 0

    assert Path.cwd() == caller_cwd


def test_worker_bootstrap_failure_is_written_to_restart_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.config

    project = tmp_path / "project"
    project.mkdir()
    gate = _ready_gate(tmp_path, project_dir=project)
    log_file = tmp_path / "worker.log"
    monkeypatch.setattr(
        src.config,
        "get_settings",
        lambda: (_ for _ in ()).throw(RuntimeError("invalid restart config")),
    )

    result = restart_gate_main(
        [
            "worker",
            "--project-dir",
            str(project),
            "--restart-script",
            str(project / "restart.sh"),
            "--log-file",
            str(log_file),
            "--delay",
            "0",
            "--expected-generation",
            gate.snapshot(),
        ]
    )

    assert result == EX_TEMPFAIL
    assert "bootstrap failed" in log_file.read_text(encoding="utf-8")
    assert "invalid restart config" in log_file.read_text(encoding="utf-8")


def test_launch_wrapper_always_returns_zero_logs_result_and_removes_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.config

    project = tmp_path / "project"
    project.mkdir()
    gate = _ready_gate(tmp_path, project_dir=project)
    log_file = tmp_path / "worker.log"
    removed: list[str] = []
    monkeypatch.setattr(
        src.config,
        "get_settings",
        lambda: SimpleNamespace(
            restart_gate_dir="",
            restart_gate_timeout=9.0,
        ),
    )
    monkeypatch.setattr(
        "src.utils.restart_gate.run_restart_worker",
        lambda **_kwargs: 23,
    )
    monkeypatch.setattr(
        "src.utils.restart_gate._remove_launchd_job",
        removed.append,
    )

    result = restart_gate_main(
        [
            "launch-wrapper",
            "--project-dir",
            str(project),
            "--restart-script",
            str(project / "restart.sh"),
            "--log-file",
            str(log_file),
            "--delay",
            "0",
            "--expected-generation",
            gate.snapshot(),
            "--launchd-label",
            "com.ghostap.test.restart.1",
        ]
    )

    assert result == 0
    assert removed == ["com.ghostap.test.restart.1"]
    assert "worker_exit=23" in log_file.read_text(encoding="utf-8")


def test_restart_gate_settings_use_centralized_ghostap_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.config.settings import Settings

    monkeypatch.setenv("GHOSTAP_RESTART_GATE_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("GHOSTAP_RESTART_GATE_TIMEOUT", "42.5")

    settings = Settings(_env_file=None)

    assert settings.restart_gate_dir == str(tmp_path / "shared")
    assert settings.restart_gate_timeout == 42.5


@pytest.mark.parametrize(
    "unsafe",
    ["relative/gate", "/", "/tmp", "/var/tmp"],
)
def test_restart_gate_settings_reject_unsafe_override(unsafe: str) -> None:
    from src.config.settings import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None, restart_gate_dir=unsafe)


def test_restart_gate_settings_rejects_user_home_as_non_dedicated_override() -> None:
    from src.config.settings import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None, restart_gate_dir=str(Path.home()))


@pytest.mark.parametrize("unsafe", [0, -1, math.inf, -math.inf, math.nan])
def test_restart_gate_settings_reject_non_finite_or_non_positive_timeout(
    unsafe: float,
) -> None:
    from src.config.settings import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None, restart_gate_timeout=unsafe)


def test_restart_gate_cli_resolves_override_through_central_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.config

    project = tmp_path / "project"
    project.mkdir()
    override = tmp_path / "shared-gate"
    monkeypatch.setattr(
        src.config,
        "get_settings",
        lambda: SimpleNamespace(
            restart_gate_dir=str(override),
            restart_gate_timeout=9.0,
        ),
    )
    original_cwd = Path.cwd()
    try:
        result = restart_gate_main(
            [
                "publish",
                "--project-dir",
                str(project),
                "--service-pid",
                str(os.getpid()),
            ]
        )
    finally:
        os.chdir(original_cwd)

    assert result == 0
    gate = RestartGate.for_project(project, override=override)
    with pytest.raises(RestartGateError, match="not ready"):
        gate.snapshot()
    assert gate.mark_ready(service_pid=os.getpid())
    assert not (project / ".ghostap-restart").exists()


@pytest.mark.parametrize(
    ("kind", "expected_status"),
    [
        ("success", TaskStatus.SUCCEEDED),
        ("error", TaskStatus.FAILED),
        ("cancel", TaskStatus.CANCELED),
    ],
)
def test_scheduler_run_guard_wraps_terminal_state_and_counter_accounting(
    kind: str,
    expected_status: TaskStatus,
) -> None:
    observations: list[tuple[TaskStatus, int, int]] = []
    scheduler: TaskScheduler

    @contextmanager
    def guard():
        try:
            yield
        finally:
            state = scheduler.get_state(handle.run_id)
            assert state is not None
            observations.append(
                (
                    state.status,
                    scheduler._running_total_normal,
                    scheduler._running_by_key[state.assigned_queue_key],
                )
            )

    scheduler = TaskScheduler(
        max_concurrent=1,
        per_key_concurrency=1,
        run_guard=guard,
    )

    def task(ctx):
        if kind == "error":
            raise RuntimeError("boom")
        if kind == "cancel":
            ctx.cancel_token.cancel()
            ctx.check_canceled()
        return "ok"

    try:
        handle = scheduler.submit(TaskSpec(chat_id="chat", name=kind), task)
        result = handle.wait(timeout=3)
        assert result.status == expected_status
        assert observations == [(expected_status, 0, 0)]
    finally:
        scheduler.stop(wait=True, shutdown_executor=True)


def test_scheduler_guard_enter_failure_converges_failed_and_releases_counters() -> None:
    @contextmanager
    def failing_guard():
        raise RuntimeError("restart gate unavailable")
        yield

    scheduler = TaskScheduler(
        max_concurrent=1,
        per_key_concurrency=1,
        run_guard=failing_guard,
    )
    try:
        handle = scheduler.submit(
            TaskSpec(chat_id="chat", name="guard-error"),
            lambda _ctx: pytest.fail("task must not execute"),
        )
        result = handle.wait(timeout=3)
        state = handle.get_state()

        assert result.status == TaskStatus.FAILED
        assert result.error == "restart gate unavailable"
        assert scheduler._running_total_normal == 0
        assert scheduler._running_by_key[state.assigned_queue_key] == 0
    finally:
        scheduler.stop(wait=True, shutdown_executor=True)


def test_scheduler_guard_factory_failure_converges_failed_and_releases_counters() -> None:
    def failing_factory():
        raise RuntimeError("cannot construct restart guard")

    scheduler = TaskScheduler(
        max_concurrent=1,
        per_key_concurrency=1,
        run_guard=failing_factory,
    )
    try:
        handle = scheduler.submit(
            TaskSpec(chat_id="chat", name="guard-factory-error"),
            lambda _ctx: pytest.fail("task must not execute"),
        )

        result = handle.wait(timeout=3)

        assert result.status == TaskStatus.FAILED
        assert result.error == "cannot construct restart guard"
        assert scheduler._running_total_normal == 0
    finally:
        scheduler.stop(wait=True, shutdown_executor=True)


def test_scheduler_guard_exit_failure_keeps_one_monotonic_terminal_event() -> None:
    events: list[TaskStatus] = []

    @contextmanager
    def release_fails():
        yield
        raise RuntimeError("release failed after close")

    scheduler = TaskScheduler(
        max_concurrent=1,
        per_key_concurrency=1,
        run_guard=release_fails,
    )
    scheduler.add_listener(lambda event: events.append(event.status))
    try:
        handle = scheduler.submit(
            TaskSpec(chat_id="chat", name="guard-release-error"),
            lambda _ctx: "ok",
        )
        result = handle.wait(timeout=3)
        deadline = time.monotonic() + 2
        while handle.get_state().future and not handle.get_state().future.done():
            if time.monotonic() >= deadline:
                raise AssertionError("scheduler future did not settle")
            time.sleep(0.01)

        assert result.status == TaskStatus.SUCCEEDED
        assert events.count(TaskStatus.SUCCEEDED) == 1
        assert events.count(TaskStatus.FAILED) == 0
        assert scheduler._running_total_normal == 0
    finally:
        scheduler.stop(wait=True, shutdown_executor=True)


def test_scheduler_blocking_guard_exit_cannot_race_a_terminal_flip() -> None:
    exit_started = threading.Event()
    release_exit = threading.Event()
    events: list[TaskStatus] = []

    @contextmanager
    def blocking_exit():
        try:
            yield
        finally:
            exit_started.set()
            assert release_exit.wait(3)

    scheduler = TaskScheduler(
        max_concurrent=1,
        per_key_concurrency=1,
        run_guard=blocking_exit,
    )
    scheduler.add_listener(lambda event: events.append(event.status))
    try:
        handle = scheduler.submit(
            TaskSpec(chat_id="chat", name="guard-blocking-exit"),
            lambda _ctx: "ok",
        )
        assert exit_started.wait(2)

        assert handle.get_state().status == TaskStatus.SUCCEEDED
        assert scheduler._running_total_normal == 0
        assert events.count(TaskStatus.SUCCEEDED) == 1
        assert events.count(TaskStatus.FAILED) == 0

        release_exit.set()
        assert handle.wait(timeout=3).status == TaskStatus.SUCCEEDED
        assert events.count(TaskStatus.SUCCEEDED) == 1
        assert events.count(TaskStatus.FAILED) == 0
    finally:
        release_exit.set()
        scheduler.stop(wait=True, shutdown_executor=True)


def test_scheduler_stop_unblocks_task_waiting_at_restart_admission(
    tmp_path: Path,
) -> None:
    gate = RestartGate(tmp_path / "gate")
    scheduler = TaskScheduler(
        max_concurrent=1,
        per_key_concurrency=1,
        run_guard=gate.task_guard,
    )
    callback_called = False

    def callback(_ctx):
        nonlocal callback_called
        callback_called = True

    try:
        with gate._exclusive_guard(time.monotonic() + 3):
            handle = scheduler.submit(
                TaskSpec(chat_id="chat", name="fenced"),
                callback,
            )
            deadline = time.monotonic() + 2
            while handle.get_state().status != TaskStatus.RUNNING:
                if time.monotonic() >= deadline:
                    raise AssertionError("scheduler did not reserve the worker")
                time.sleep(0.01)

            scheduler.stop(wait=True, shutdown_executor=True)
            result = handle.wait(timeout=0.5)

        assert result.status == TaskStatus.CANCELED
        assert not callback_called
        assert scheduler._running_total_normal == 0
    finally:
        scheduler.stop(wait=True, shutdown_executor=True)


def test_ws_scheduler_factory_injects_checkout_scoped_restart_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.feishu import ws_client

    captured: dict[str, object] = {}

    class CapturingScheduler:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(ws_client, "TaskScheduler", CapturingScheduler)
    settings = SimpleNamespace(
        task_scheduler_max_concurrent=3,
        task_scheduler_per_key_concurrency=2,
        system_command_concurrency=1,
        restart_gate_dir="",
    )

    ws_client._build_task_scheduler(settings, project_dir=tmp_path)

    guard = captured["run_guard"]
    assert not guard.__self__.directory.is_relative_to(tmp_path)
    owner = json.loads(
        (guard.__self__.directory / "owner.json").read_text(encoding="utf-8")
    )
    assert owner["project_dir"] == str(tmp_path.resolve())


def test_ws_scheduler_factory_defaults_to_code_checkout_not_process_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.feishu import ws_client

    captured: dict[str, object] = {}

    class FakeGate:
        @contextmanager
        def task_guard(self):
            yield

    def fake_for_project(project_dir, *, override=None):
        captured["project_dir"] = Path(project_dir)
        captured["override"] = override
        return FakeGate()

    class CapturingScheduler:
        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr(
        ws_client.RestartGate,
        "for_project",
        fake_for_project,
    )
    monkeypatch.setattr(ws_client, "TaskScheduler", CapturingScheduler)
    monkeypatch.chdir(tmp_path)
    settings = SimpleNamespace(
        task_scheduler_max_concurrent=3,
        task_scheduler_per_key_concurrency=2,
        system_command_concurrency=1,
        restart_gate_dir="",
    )

    ws_client._build_task_scheduler(settings)

    assert captured["project_dir"] == _REPOSITORY_ROOT
    assert captured["override"] is None
    assert "service_pid" not in captured
