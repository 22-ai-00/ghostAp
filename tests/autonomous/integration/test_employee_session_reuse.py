from __future__ import annotations

import threading
from pathlib import Path

from src.autonomous.runtime.employee_actor import (
    EmployeeActorStatus,
    EmployeeAssignment,
)
from src.autonomous.runtime.employee_session import EmployeeSessionBootstrap
from src.autonomous.runtime.employee_supervisor import EmployeeRuntimeSupervisor
from src.autonomous.workforce.identity import AgentIdentity


class _Session:
    def __init__(self, gate: threading.Event | None = None) -> None:
        self.gate = gate
        self.closed = False

    def send_prompt(self, prompt: str, *, timeout: float):
        if self.gate is not None:
            self.gate.wait(timeout)
        return type("Result", (), {"text": prompt})()

    def is_server_healthy(self) -> bool:
        return not self.closed

    def close(self) -> None:
        self.closed = True


def _bootstrap(tmp_path: Path, agent_id: str, *, model: str = "m"):
    workspace = tmp_path / agent_id / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "AGENTS.md").write_text(f"# {agent_id}\n", encoding="utf-8")
    return EmployeeSessionBootstrap.from_agent(
        tenant_key="tenant_1",
        agent=AgentIdentity(
            agent_id=agent_id,
            agent_type="codex",
            model_name=model,
            workspace_path=str(workspace),
            permissions=["file_read"],
            capabilities=["file_read"],
        ),
        project_root=str(tmp_path / "project"),
        identity_version=1,
    )


def test_supervisor_reports_cold_ready_before_first_assignment() -> None:
    supervisor = EmployeeRuntimeSupervisor(session_factory=lambda _bootstrap: None)
    supervisor.ensure_employee("agt_1")
    assert supervisor.status("agt_1") is EmployeeActorStatus.READY_COLD
    supervisor.recycle("agt_1", "identity_changed")
    assert supervisor.status("agt_1") is EmployeeActorStatus.READY_COLD
    supervisor.close()


def test_inspect_missing_employee_does_not_allocate_a_cold_actor() -> None:
    supervisor = EmployeeRuntimeSupervisor(session_factory=lambda _bootstrap: None)

    snapshot = supervisor.inspect("agt_missing")

    assert snapshot.status is EmployeeActorStatus.READY_COLD
    assert snapshot.mailbox_depth == 0
    assert snapshot.active_assignment_id == ""
    assert supervisor._actors == {}  # noqa: SLF001
    supervisor.close()


def test_different_employee_mailboxes_execute_in_parallel(tmp_path: Path) -> None:
    gate = threading.Event()
    entered = threading.Barrier(3)

    class ParallelSession(_Session):
        def send_prompt(self, prompt: str, *, timeout: float):
            entered.wait(timeout=1)
            gate.wait(timeout=1)
            return type("Result", (), {"text": prompt})()

    supervisor = EmployeeRuntimeSupervisor(
        session_factory=lambda _bootstrap: ParallelSession()
    )
    for index in (1, 2):
        bootstrap = _bootstrap(tmp_path, f"agt_{index}")
        supervisor.submit(EmployeeAssignment(f"asgn_{index}", bootstrap, "work", 1))
    entered.wait(timeout=1)
    assert supervisor.status("agt_1") is EmployeeActorStatus.BUSY
    assert supervisor.status("agt_2") is EmployeeActorStatus.BUSY
    gate.set()
    assert supervisor.wait_terminal("asgn_1", timeout=1).status == "completed"
    assert supervisor.wait_terminal("asgn_2", timeout=1).status == "completed"
    supervisor.close()


def test_idle_ttl_recycles_warm_session_to_cold(tmp_path: Path) -> None:
    now = [0.0]
    sessions: list[_Session] = []
    supervisor = EmployeeRuntimeSupervisor(
        session_factory=lambda _bootstrap: sessions.append(_Session()) or sessions[-1],
        idle_ttl_seconds=10,
        monotonic=lambda: now[0],
    )
    bootstrap = _bootstrap(tmp_path, "agt_1")
    supervisor.submit(EmployeeAssignment("asgn_1", bootstrap, "work", 1))
    assert supervisor.wait_terminal("asgn_1", timeout=1).status == "completed"
    assert supervisor.status("agt_1") is EmployeeActorStatus.READY_WARM
    now[0] = 11
    assert supervisor.sweep_idle() == 1
    assert supervisor.status("agt_1") is EmployeeActorStatus.READY_COLD
    assert sessions[0].closed is True
    supervisor.close()


def test_inspect_is_a_pure_atomic_snapshot_and_does_not_reap_idle_session(
    tmp_path: Path,
) -> None:
    now = [0.0]
    sessions: list[_Session] = []
    supervisor = EmployeeRuntimeSupervisor(
        session_factory=lambda _bootstrap: sessions.append(_Session()) or sessions[-1],
        idle_ttl_seconds=10,
        monotonic=lambda: now[0],
    )
    bootstrap = _bootstrap(tmp_path, "agt_1")
    supervisor.submit(EmployeeAssignment("asgn_1", bootstrap, "work", 1))
    assert supervisor.wait_terminal("asgn_1", timeout=1).status == "completed"
    now[0] = 11

    snapshot = supervisor.inspect("agt_1")

    assert snapshot.status is EmployeeActorStatus.READY_WARM
    assert snapshot.mailbox_depth == 0
    assert snapshot.active_assignment_id == ""
    assert sessions[0].closed is False
    assert supervisor.sweep_idle() == 1
    assert sessions[0].closed is True
    supervisor.close()
