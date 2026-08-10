from __future__ import annotations

import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

import pytest

from src.utils.restart_gate import (
    EX_TEMPFAIL,
    RestartGate,
    RestartGateError,
    RestartGateTimeout,
    run_restart_worker,
)

_PROCESS_TIMEOUT = 10.0


def _ready_gate(tmp_path: Path) -> tuple[Path, RestartGate, str]:
    project = tmp_path / "project"
    project.mkdir()
    gate = RestartGate.for_project(project, override=tmp_path / "gate")
    gate.publish_participation(service_pid=os.getpid())
    generation = gate.mark_ready(service_pid=os.getpid())
    return project, gate, generation


def _write_private_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o600)


def _hold_task(project_dir: str, entered, release) -> None:
    gate = RestartGate.from_locator(project_dir)
    with gate.task_guard():
        entered.set()
        if not release.wait(_PROCESS_TIMEOUT):
            raise TimeoutError("task guard was not released")


def _run_worker(
    project_dir: str,
    generation: str,
    restart_script: str,
    log_file: str,
    start,
    results,
) -> None:
    gate = RestartGate.from_locator(
        project_dir,
        expected_generation=generation,
    )
    if not start.wait(_PROCESS_TIMEOUT):
        raise TimeoutError("restart worker was not started")
    results.put(
        run_restart_worker(
            gate=gate,
            expected_generation=generation,
            restart_script=restart_script,
            log_file=log_file,
            delay=0,
            timeout=_PROCESS_TIMEOUT,
            project_dir=project_dir,
        )
    )


def _join(process) -> None:
    process.join(_PROCESS_TIMEOUT)
    if process.is_alive():
        process.terminate()
        process.join(2)
        raise AssertionError(f"process {process.pid} did not exit")
    assert process.exitcode == 0


def test_generation_requires_a_live_ready_service(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    gate = RestartGate.for_project(project, override=tmp_path / "gate")

    with pytest.raises(RestartGateError, match="state is missing"):
        gate.snapshot()

    gate.publish_participation(service_pid=os.getpid())
    with pytest.raises(RestartGateError, match="not ready"):
        gate.snapshot()

    generation = gate.mark_ready(service_pid=os.getpid())
    assert gate.snapshot() == generation

    state = json.loads(gate.state_path.read_text(encoding="utf-8"))
    state["service_pid"] = 999_999_999
    _write_private_json(gate.state_path, state)
    with pytest.raises(RestartGateError, match="PID is not alive"):
        gate.snapshot()


def test_gate_keeps_lock_inodes_and_checkout_binding(tmp_path: Path) -> None:
    project, gate, _generation = _ready_gate(tmp_path)
    identities = (
        gate.admission_path.stat().st_ino,
        gate.drain_path.stat().st_ino,
    )

    reopened = RestartGate.for_project(project)
    assert reopened.directory == gate.directory
    assert (
        reopened.admission_path.stat().st_ino,
        reopened.drain_path.stat().st_ino,
    ) == identities
    assert identities[0] != identities[1]

    other = tmp_path / "other-project"
    other.mkdir()
    with pytest.raises(RestartGateError, match="different checkout"):
        RestartGate.for_project(other, override=gate.directory)

    old_fd = os.open(gate.admission_path, os.O_RDONLY)
    try:
        gate.admission_path.unlink()
        gate.admission_path.touch(mode=0o600)
        with pytest.raises(RestartGateError, match="identity changed"):
            gate.snapshot()
    finally:
        os.close(old_fd)


@pytest.mark.skipif(
    sys.platform not in {"linux", "darwin"},
    reason="flock gate is POSIX-only",
)
def test_task_guard_blocks_exclusive_pair_until_absolute_deadline(
    tmp_path: Path,
) -> None:
    ctx = multiprocessing.get_context("spawn")
    project, gate, _generation = _ready_gate(tmp_path)
    entered = ctx.Event()
    release = ctx.Event()
    holder = ctx.Process(
        target=_hold_task,
        args=(str(project), entered, release),
    )
    holder.start()
    assert entered.wait(3)

    started = time.monotonic()
    try:
        with pytest.raises(RestartGateTimeout):
            with gate._lock_pair(
                exclusive=True,
                deadline=started + 0.15,
            ):
                pytest.fail("exclusive restart must wait for active work")
    finally:
        release.set()
        _join(holder)

    elapsed = time.monotonic() - started
    assert 0.1 <= elapsed < 1.0


@pytest.mark.skipif(
    sys.platform not in {"linux", "darwin"},
    reason="flock gate is POSIX-only",
)
def test_same_generation_restart_workers_execute_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = multiprocessing.get_context("spawn")
    project, gate, generation = _ready_gate(tmp_path)
    operation_log = tmp_path / "operations.log"
    worker_log = tmp_path / "restart.log"
    monkeypatch.setenv("RESTART_OPERATION_LOG", str(operation_log))
    restart_script = project / "fake-restart.py"
    restart_script.write_text(
        f"""#!{sys.executable}
import json
import os
from pathlib import Path

with Path(os.environ["RESTART_OPERATION_LOG"]).open("a", encoding="utf-8") as stream:
    stream.write("restart\\n")
state_path = Path(os.environ["GHOSTAP_RESTART_GATE_DIR"]) / "state.json"
state = json.loads(state_path.read_text(encoding="utf-8"))
state["generation"] = "g" + "N" * 30
temporary = state_path.with_name(f".state.{{os.getpid()}}.tmp")
temporary.write_text(json.dumps(state), encoding="utf-8")
temporary.chmod(0o600)
os.replace(temporary, state_path)
""",
        encoding="utf-8",
    )
    restart_script.chmod(0o700)
    start = ctx.Event()
    results = ctx.Queue()
    workers = [
        ctx.Process(
            target=_run_worker,
            args=(
                str(project),
                generation,
                str(restart_script),
                str(worker_log),
                start,
                results,
            ),
        )
        for _ in range(2)
    ]

    for worker in workers:
        worker.start()
    start.set()
    for worker in workers:
        _join(worker)

    assert [results.get(timeout=1) for _ in workers] == [0, 0]
    assert operation_log.read_text(encoding="utf-8").splitlines() == ["restart"]
    log = worker_log.read_text(encoding="utf-8")
    assert log.count("status=restarted exit_code=0") == 1
    assert log.count("status=coalesced exit_code=0") == 1
    assert gate.snapshot() != generation


def test_worker_uses_one_absolute_budget_and_logs_milestones(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, gate, generation = _ready_gate(tmp_path)
    clock = iter([10.0, 10.75])
    captured: dict[str, float] = {}
    monkeypatch.setattr("src.utils.restart_gate.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("src.utils.restart_gate.time.sleep", lambda _delay: None)

    def fake_run(_expected, *, timeout, operation):
        captured["timeout"] = timeout
        return "coalesced", 0

    monkeypatch.setattr(gate, "_run_if_current", fake_run)
    log_file = tmp_path / "restart.log"

    assert (
        run_restart_worker(
            gate=gate,
            expected_generation=generation,
            restart_script=tmp_path / "restart.sh",
            log_file=log_file,
            delay=0.75,
            timeout=1.0,
            project_dir=project,
        )
        == 0
    )

    assert captured["timeout"] == pytest.approx(0.25)
    log = log_file.read_text(encoding="utf-8")
    assert "[RESTART] remote worker scheduled" in log
    assert "[RESTART] remote worker script begin" in log


@pytest.mark.skipif(
    sys.platform != "linux" or not Path("/proc").exists(),
    reason="process-group descendant verification uses Linux procfs",
)
def test_timeout_kills_the_restart_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, gate, generation = _ready_gate(tmp_path)
    child_pid_file = tmp_path / "child.pid"
    restart_script = project / "fake-restart.sh"
    restart_script.write_text(
        """#!/bin/bash
trap '' TERM
(
  trap '' TERM
  printf "%s\\n" "$BASHPID" > "$FAKE_CHILD_PID_FILE"
  while :; do sleep 10; done
) &
while [ ! -s "$FAKE_CHILD_PID_FILE" ]; do sleep 0.01; done
while :; do sleep 10; done
""",
        encoding="utf-8",
    )
    restart_script.chmod(0o700)
    monkeypatch.setenv("FAKE_CHILD_PID_FILE", str(child_pid_file))

    result = run_restart_worker(
        gate=gate,
        expected_generation=generation,
        restart_script=restart_script,
        log_file=tmp_path / "restart.log",
        delay=0,
        timeout=0.4,
        project_dir=project,
    )

    assert result == EX_TEMPFAIL
    assert gate.snapshot() == generation
    child_pid = int(child_pid_file.read_text(encoding="utf-8").strip())
    deadline = time.monotonic() + 2
    while True:
        try:
            state = (
                Path(f"/proc/{child_pid}/stat")
                .read_text(encoding="utf-8")
                .rsplit(")", 1)[1]
                .split()[0]
            )
        except FileNotFoundError:
            break
        if state == "Z":
            break
        if time.monotonic() >= deadline:
            raise AssertionError(f"restart descendant still alive pid={child_pid}")
        time.sleep(0.02)


def test_legacy_markers_are_migrated_only_when_state_is_absent(
    tmp_path: Path,
) -> None:
    _project, gate, generation = _ready_gate(tmp_path)
    state = json.loads(gate.state_path.read_text(encoding="utf-8"))
    proof = {
        key: value
        for key, value in state.items()
        if key not in {"generation", "participation_id", "status", "exit_code"}
    }
    legacy_generation = {
        key: state[key]
        for key in ("generation", "participation_id", "status", "exit_code")
    }
    gate.state_path.unlink()
    participation_path = gate.directory / "participation.json"
    generation_path = gate.directory / "generation"
    _write_private_json(participation_path, proof)
    _write_private_json(generation_path, legacy_generation)

    assert gate.snapshot() == generation
    assert gate.state_path.exists()

    generation_path.write_text("not valid JSON", encoding="utf-8")
    assert gate.snapshot() == generation
