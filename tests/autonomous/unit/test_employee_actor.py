from __future__ import annotations

import threading
import time
from concurrent.futures import CancelledError
from pathlib import Path

from src.autonomous.runtime.employee_actor import (
    EmployeeActor,
    EmployeeActorStatus,
    EmployeeAssignment,
)
from src.autonomous.runtime.employee_session import EmployeeSessionBootstrap
from src.autonomous.workforce.identity import AgentIdentity


class _Session:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.closed = False

    def send_prompt(self, prompt: str, *, timeout: float):
        self.prompts.append(prompt)
        time.sleep(0.01)
        return type("Result", (), {"text": prompt})()

    def is_server_healthy(self) -> bool:
        return not self.closed

    def close(self) -> None:
        self.closed = True


def _bootstrap(
    tmp_path: Path,
    *,
    model: str = "m",
    effort: str = "default",
    identity_version: int = 1,
    instruction: str = "# Employee\n",
) -> EmployeeSessionBootstrap:
    workspace = tmp_path / "agent/workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "AGENTS.md").write_text(instruction, encoding="utf-8")
    return EmployeeSessionBootstrap.from_agent(
        tenant_key="tenant_1",
        agent=AgentIdentity(
            agent_id="agt_1",
            agent_type="codex",
            model_name=model,
            reasoning_effort=effort,
            workspace_path=str(workspace),
            permissions=["file_read"],
            capabilities=["file_read"],
            security_profile="employee_v1",
        ),
        project_root=str(tmp_path / "project"),
        identity_version=identity_version,
    )


def test_actor_serializes_mailbox_and_deduplicates_assignment(tmp_path: Path) -> None:
    sessions: list[_Session] = []
    terminals = []

    def factory(_bootstrap):
        session = _Session()
        sessions.append(session)
        return session

    actor = EmployeeActor("agt_1", session_factory=factory, terminal_sink=terminals.append)
    bootstrap = _bootstrap(tmp_path)
    first = EmployeeAssignment("asgn_1", bootstrap, "one", 1)
    second = EmployeeAssignment("asgn_2", bootstrap, "two", 1)
    assert actor.submit(first) == "asgn_1"
    assert actor.submit(first) == "asgn_1"
    assert actor.submit(second) == "asgn_2"
    actor.drain()

    assert [item.assignment_id for item in terminals] == ["asgn_1", "asgn_2"]
    assert len(sessions) == 1
    assert [prompt.rsplit("## ASSIGNMENT\n", 1)[-1] for prompt in sessions[0].prompts] == [
        "one",
        "two",
    ]
    assert actor.status is EmployeeActorStatus.READY_WARM
    actor.close()


def test_cancel_timeout_and_close_each_emit_one_terminal(tmp_path: Path) -> None:
    class ControlledSession(_Session):
        def __init__(self) -> None:
            super().__init__()
            self.canceled = threading.Event()

        def send_prompt(self, prompt: str, *, timeout: float):
            if prompt.endswith("timeout"):
                raise TimeoutError
            self.canceled.wait(1)
            if self.canceled.is_set():
                raise CancelledError
            return super().send_prompt(prompt, timeout=timeout)

        def cancel(self) -> None:
            self.canceled.set()

    terminals = []
    actor = EmployeeActor(
        "agt_1",
        session_factory=lambda _bootstrap: ControlledSession(),
        terminal_sink=terminals.append,
    )
    bootstrap = _bootstrap(tmp_path)
    actor.submit(EmployeeAssignment("asgn_timeout", bootstrap, "timeout", 1))
    actor.submit(EmployeeAssignment("asgn_cancel", bootstrap, "cancel", 1))
    actor.submit(EmployeeAssignment("asgn_close", bootstrap, "close", 1))
    while actor.status is not EmployeeActorStatus.BUSY:
        time.sleep(0.001)
    actor.cancel("asgn_cancel")
    actor.close()

    by_id = {terminal.assignment_id: terminal for terminal in terminals}
    assert len(terminals) == len(by_id) == 3
    assert by_id["asgn_timeout"].status == "timeout"
    assert by_id["asgn_cancel"].status == "canceled"
    assert by_id["asgn_close"].status == "canceled"


def test_actor_recycles_session_when_key_changes(tmp_path: Path) -> None:
    sessions: list[_Session] = []
    actor = EmployeeActor(
        "agt_1",
        session_factory=lambda _bootstrap: sessions.append(_Session()) or sessions[-1],
        terminal_sink=lambda _terminal: None,
    )
    actor.submit(EmployeeAssignment("asgn_1", _bootstrap(tmp_path, model="m1"), "one", 1))
    actor.drain()
    actor.submit(EmployeeAssignment("asgn_2", _bootstrap(tmp_path, model="m2"), "two", 1))
    actor.drain()
    assert len(sessions) == 2
    assert sessions[0].closed is True
    actor.close()


def test_actor_recycles_session_when_instruction_digest_changes(tmp_path: Path) -> None:
    sessions: list[_Session] = []
    actor = EmployeeActor(
        "agt_1",
        session_factory=lambda _bootstrap: sessions.append(_Session()) or sessions[-1],
        terminal_sink=lambda _terminal: None,
    )
    actor.submit(
        EmployeeAssignment(
            "asgn_1",
            _bootstrap(tmp_path, instruction="# Employee v1\n"),
            "one",
            1,
        )
    )
    actor.drain()
    actor.submit(
        EmployeeAssignment(
            "asgn_2",
            _bootstrap(tmp_path, instruction="# Employee v2\n"),
            "two",
            1,
        )
    )
    actor.drain()

    assert len(sessions) == 2
    assert sessions[0].closed is True
    actor.close()


def test_actor_recycles_session_when_identity_or_effort_changes(tmp_path: Path) -> None:
    sessions: list[_Session] = []
    actor = EmployeeActor(
        "agt_1",
        session_factory=lambda _bootstrap: sessions.append(_Session()) or sessions[-1],
        terminal_sink=lambda _terminal: None,
    )
    actor.submit(EmployeeAssignment("asgn_1", _bootstrap(tmp_path), "one", 1))
    actor.drain()
    actor.submit(
        EmployeeAssignment(
            "asgn_2",
            _bootstrap(tmp_path, identity_version=2),
            "two",
            1,
        )
    )
    actor.drain()
    actor.submit(
        EmployeeAssignment(
            "asgn_3",
            _bootstrap(tmp_path, identity_version=2, effort="high"),
            "three",
            1,
        )
    )
    actor.drain()

    assert len(sessions) == 3
    assert all(session.closed for session in sessions[:2])
    actor.close()
