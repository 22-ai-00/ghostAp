from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import pytest

from src.autonomous.journal.anchor import FileAnchor
from src.autonomous.journal.frame import JournalEvent
from src.autonomous.journal.writer import CommitState, JournalWriter
from src.autonomous.runtime.employee_actor import EmployeeAssignmentTerminal
from src.autonomous.runtime.employee_supervisor import EmployeeRuntimeSupervisor


def _writer(tmp_path: Path) -> JournalWriter:
    return JournalWriter.open(
        tmp_path / "journal",
        anchor=FileAnchor(tmp_path / "anchor.json"),
        hmac_key=b"employee-actor-recovery-key-32bytes",
    )


def _commit_queued(writer: JournalWriter, assignment_id: str) -> None:
    aggregate = f"employee-assignment:{assignment_id}"
    event = JournalEvent(
        event_type="employee.actor.assignment_queued",
        aggregate_id=aggregate,
        payload={
            "agent_id": "agt_1",
            "payload_ref": "",
            "prompt_digest": hashlib.sha256(b"secret prompt").hexdigest(),
            "timeout_seconds": 10.0,
            "session_key": {
                "tenant_key": "tenant_1",
                "agent_id": "agt_1",
                "project_root": "/project",
                "backend": "codex",
                "model": "m",
                "profile": "",
            },
        },
    )
    result = writer.commit((event,), {aggregate: 0})
    assert result.state is CommitState.ANCHORED


def _commit_terminal(
    writer: JournalWriter,
    assignment_id: str,
    *,
    status: str,
    error_code: str,
) -> None:
    aggregate = f"employee-assignment:{assignment_id}"
    event = JournalEvent(
        event_type="employee.actor.assignment_terminal",
        aggregate_id=aggregate,
        payload={
            "agent_id": "agt_1",
            "status": status,
            "error_code": error_code,
            "output_digest": "",
        },
    )
    result = writer.commit(
        (event,),
        writer.get_aggregate_versions((aggregate,)),
    )
    assert result.state is CommitState.ANCHORED


def test_recovery_rebuilds_already_anchored_terminal_for_query(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    _commit_queued(writer, "asgn_1")
    _commit_terminal(
        writer,
        "asgn_1",
        status="action_required",
        error_code="employee_session_failed",
    )
    notifications: list[EmployeeAssignmentTerminal] = []
    supervisor = EmployeeRuntimeSupervisor(
        writer=writer,
        terminal_sink=notifications.append,
    )

    try:
        supervisor.recover()
        terminal_reader = getattr(supervisor, "terminal", None)
        assert callable(
            terminal_reader
        ), "recovered actor terminals need a read-only lookup"
        assert terminal_reader("asgn_1") == EmployeeAssignmentTerminal(
            "asgn_1",
            "action_required",
            error_code="employee_session_failed",
        )
        supervisor.recover()
        assert terminal_reader("asgn_1") == EmployeeAssignmentTerminal(
            "asgn_1",
            "action_required",
            error_code="employee_session_failed",
        )
        assert notifications == []
    finally:
        supervisor.close()
        writer.close()


def test_recovery_keeps_richer_live_terminal_when_replaying_its_digest(
    tmp_path: Path,
) -> None:
    from src.autonomous.runtime.employee_actor import EmployeeAssignment
    from tests.autonomous.unit.test_employee_actor import _bootstrap, _Session

    writer = _writer(tmp_path)
    notifications: list[EmployeeAssignmentTerminal] = []
    supervisor = EmployeeRuntimeSupervisor(
        writer=writer,
        session_factory=lambda _bootstrap_value: _Session(),
        terminal_sink=notifications.append,
    )
    try:
        supervisor.submit(
            EmployeeAssignment("asgn_1", _bootstrap(tmp_path), "work", 1)
        )
        live = supervisor.wait_terminal("asgn_1", timeout=1)
        assert live.status == "completed"
        assert live.output

        supervisor.recover()

        assert supervisor.terminal("asgn_1") == live
        assert notifications == [live]
    finally:
        supervisor.close()
        writer.close()


def test_live_terminal_upgrades_replayed_placeholder_and_emits_sink_once(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    _commit_queued(writer, "asgn_1")
    _commit_terminal(writer, "asgn_1", status="completed", error_code="")
    notifications: list[EmployeeAssignmentTerminal] = []
    supervisor = EmployeeRuntimeSupervisor(
        writer=writer,
        terminal_sink=notifications.append,
    )
    live = EmployeeAssignmentTerminal("asgn_1", "completed", output="live result")

    try:
        supervisor.recover()
        assert supervisor.terminal("asgn_1") == EmployeeAssignmentTerminal(
            "asgn_1",
            "completed",
        )

        supervisor._record_terminal(live)

        assert supervisor.wait_terminal("asgn_1", timeout=0) == live
        assert notifications == [live]

        supervisor.recover()
        supervisor._record_terminal(live)
        assert supervisor.terminal("asgn_1") == live
        assert notifications == [live]
    finally:
        supervisor.close()
        writer.close()


def test_recovery_refuses_live_commit_to_sink_window_without_placeholder(
    tmp_path: Path,
) -> None:
    from src.autonomous.runtime.employee_actor import EmployeeAssignment
    from tests.autonomous.unit.test_employee_actor import _bootstrap, _Session

    callback_entered = threading.Event()
    callback_release = threading.Event()

    class PausedTerminalSupervisor(EmployeeRuntimeSupervisor):
        def _record_terminal(self, terminal: EmployeeAssignmentTerminal) -> None:
            callback_entered.set()
            callback_release.wait(timeout=2)
            super()._record_terminal(terminal)

    writer = _writer(tmp_path)
    notifications: list[EmployeeAssignmentTerminal] = []
    supervisor = PausedTerminalSupervisor(
        writer=writer,
        session_factory=lambda _bootstrap_value: _Session(),
        terminal_sink=notifications.append,
    )
    try:
        supervisor.submit(
            EmployeeAssignment("asgn_1", _bootstrap(tmp_path), "work", 1)
        )
        assert callback_entered.wait(timeout=1)

        with pytest.raises(
            RuntimeError,
            match="recovery conflicts with live assignments",
        ):
            supervisor.recover()
        assert supervisor.terminal("asgn_1") is None

        callback_release.set()
        live = supervisor.wait_terminal("asgn_1", timeout=1)
        assert live.output
        assert notifications == [live]

        supervisor.recover()
        assert supervisor.terminal("asgn_1") == live
        assert notifications == [live]
    finally:
        callback_release.set()
        supervisor.close()
        writer.close()


def test_recovery_fences_public_submit_for_full_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.autonomous.runtime.employee_actor import EmployeeAssignment
    from tests.autonomous.unit.test_employee_actor import _bootstrap, _Session

    writer = _writer(tmp_path)
    replay_started = threading.Event()
    replay_release = threading.Event()
    real_replay = writer.replay

    def blocking_replay(from_sequence: int = 1):
        replay_started.set()
        replay_release.wait(timeout=2)
        yield from real_replay(from_sequence)

    monkeypatch.setattr(writer, "replay", blocking_replay)
    supervisor = EmployeeRuntimeSupervisor(
        writer=writer,
        session_factory=lambda _bootstrap_value: _Session(),
    )
    errors: list[BaseException] = []

    def recover() -> None:
        try:
            supervisor.recover()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=recover)
    thread.start()
    try:
        assert replay_started.wait(timeout=1)
        assignment = EmployeeAssignment(
            "asgn_1",
            _bootstrap(tmp_path),
            "work",
            1,
        )
        with pytest.raises(RuntimeError, match="recovery is running"):
            supervisor.submit(assignment)
        assert supervisor.terminal("asgn_1") is None

        replay_release.set()
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert errors == []

        supervisor.submit(assignment)
        assert supervisor.wait_terminal("asgn_1", timeout=1).status == "completed"
    finally:
        replay_release.set()
        thread.join(timeout=1)
        supervisor.close()
        writer.close()


def test_live_terminal_conflict_with_replay_fails_closed(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    _commit_queued(writer, "asgn_1")
    _commit_terminal(writer, "asgn_1", status="completed", error_code="")
    notifications: list[EmployeeAssignmentTerminal] = []
    supervisor = EmployeeRuntimeSupervisor(
        writer=writer,
        terminal_sink=notifications.append,
    )

    try:
        supervisor.recover()
        with pytest.raises(
            RuntimeError,
            match="employee actor terminal result conflicts",
        ):
            supervisor._record_terminal(
                EmployeeAssignmentTerminal(
                    "asgn_1",
                    "action_required",
                    error_code="employee_session_failed",
                )
            )
        assert supervisor.terminal("asgn_1") == EmployeeAssignmentTerminal(
            "asgn_1",
            "completed",
        )
        assert notifications == []
    finally:
        supervisor.close()
        writer.close()


def test_recovery_terminalizes_unresolvable_anchored_mailbox_once(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    _commit_queued(writer, "asgn_1")
    supervisor = EmployeeRuntimeSupervisor(writer=writer)

    assert supervisor.recover() == 1
    assert supervisor.wait_terminal("asgn_1", timeout=0).error_code == (
        "employee_recovery_payload_unavailable"
    )
    assert supervisor.recover() == 0
    events = [
        event
        for frame in writer.replay()
        for event in frame.events
        if event.aggregate_id == "employee-assignment:asgn_1"
    ]
    assert [event.event_type for event in events] == [
        "employee.actor.assignment_queued",
        "employee.actor.assignment_terminal",
    ]
    assert "secret prompt" not in writer.journal_path.read_text(encoding="utf-8")
    supervisor.close()
    writer.close()


def test_backend_effect_is_anchored_before_session_factory(tmp_path: Path) -> None:
    from src.autonomous.runtime.employee_actor import EmployeeAssignment
    from tests.autonomous.unit.test_employee_actor import _bootstrap, _Session

    writer = _writer(tmp_path)
    seen: list[str] = []

    def factory(_bootstrap_value):
        seen.extend(
            event.event_type
            for frame in writer.replay()
            for event in frame.events
        )
        return _Session()

    supervisor = EmployeeRuntimeSupervisor(writer=writer, session_factory=factory)
    supervisor.submit(
        EmployeeAssignment("asgn_1", _bootstrap(tmp_path), "work", 1)
    )
    assert supervisor.wait_terminal("asgn_1", timeout=1).status == "completed"
    assert seen[-2:] == [
        "employee.actor.effect_prepared",
        "employee.actor.effect_executing",
    ]
    supervisor.close()
    writer.close()
