"""Regression tests for CLI backend terminal-state propagation."""

from __future__ import annotations

import io
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import ExitStack, suppress
from unittest.mock import patch

import pytest

from src.agent_session.claude_cli import ClaudeCLIConfig, SyncClaudeCLISession
from src.agent_session.process_cleanup import terminate_and_reap_process_tree


class _CompletedProcess:
    def __init__(self, *, stdout: str, stderr: str, returncode: int) -> None:
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO(stderr)
        self.returncode = returncode

    def poll(self) -> int:
        return self.returncode

    def wait(self, timeout: float = 0) -> int:
        return self.returncode

    def terminate(self) -> None:
        return None


class _DelayedOutputProcess:
    def __init__(self, delay: float = 0.05) -> None:
        self.stderr = io.StringIO("")
        self.returncode: int | None = None
        self._released = threading.Event()
        self._delay = delay
        self.stdout = self._stdout()

    def _stdout(self):
        self._released.wait(self._delay)
        if self.returncode is None:
            yield "late output\n"

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float = 0) -> int:
        if self.returncode is None:
            self.returncode = 0
        self._released.set()
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15
        self._released.set()


class _CancelOnOutputProcess(_DelayedOutputProcess):
    def __init__(self, session: SyncClaudeCLISession) -> None:
        self._session = session
        super().__init__(delay=0)

    def _stdout(self):
        self._session._cancel_event.set()
        yield "partial output\n"


class _StubbornTimeoutProcess:
    def __init__(self) -> None:
        self.stderr = io.StringIO("")
        self.returncode: int | None = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.events: list[str] = []
        self._stdout_released = threading.Event()
        self.stdout = self._stdout()

    def _stdout(self):
        self._stdout_released.wait(timeout=2)
        if False:
            yield ""

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float = 0) -> int:
        self.events.append("wait")
        if self.returncode is None:
            raise subprocess.TimeoutExpired(cmd="stubborn", timeout=timeout)
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.events.append("terminate")
        self._stdout_released.set()

    def kill(self) -> None:
        self.kill_calls += 1
        self.events.append("kill")
        self.returncode = -9
        self._stdout_released.set()


class _UnkillableProcess(_StubbornTimeoutProcess):
    def kill(self) -> None:
        self.kill_calls += 1
        self.events.append("kill")


class _TermIgnoringTimeoutProcess(_StubbornTimeoutProcess):
    """Keep stdout blocked until KILL, like a process ignoring SIGTERM."""

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.events.append("terminate")


def _claude_session() -> SyncClaudeCLISession:
    session = SyncClaudeCLISession(
        cwd="/tmp",
        config=ClaudeCLIConfig(
            command="claude",
            add_dir=False,
            bypass_permissions=False,
        ),
    )
    session.session_id = "session-1"
    return session


def test_claude_timeout_is_not_reported_as_cancelled() -> None:
    session = _claude_session()
    process = _DelayedOutputProcess()

    with patch(
        "src.agent_session.claude_cli.subprocess.Popen",
        return_value=process,
    ):
        result = session.send_prompt("work", timeout=0.01)

    assert result.stop_reason == "timeout"
    assert "超时" in result.text


def test_claude_cancel_is_reported_as_cancelled() -> None:
    session = _claude_session()
    process = _CancelOnOutputProcess(session)

    with patch(
        "src.agent_session.claude_cli.subprocess.Popen",
        return_value=process,
    ):
        result = session.send_prompt("work", timeout=1)

    assert result.stop_reason == "cancelled"


def test_claude_timeout_kills_stubborn_process_and_stays_timeout() -> None:
    session = _claude_session()
    process = _StubbornTimeoutProcess()

    with patch(
        "src.agent_session.claude_cli.subprocess.Popen",
        return_value=process,
    ):
        result = session.send_prompt("work", timeout=0.01)

    assert result.stop_reason == "timeout"
    assert process.terminate_calls >= 1
    assert process.kill_calls == 1
    assert process.poll() == -9
    assert session._proc is None


def test_claude_watchdog_kills_term_ignoring_process_with_blocked_stdout() -> None:
    session = _claude_session()
    process = _TermIgnoringTimeoutProcess()
    observed: list[object] = []

    with patch(
        "src.agent_session.claude_cli.subprocess.Popen",
        return_value=process,
    ):
        worker = threading.Thread(
            target=lambda: observed.append(
                session.send_prompt("work", timeout=0.01)
            ),
            daemon=True,
        )
        worker.start()
        worker.join(timeout=1.5)
        completed_without_external_kill = not worker.is_alive()
        if worker.is_alive():
            process.kill()
            worker.join(timeout=1)

    assert completed_without_external_kill
    assert observed and observed[0].stop_reason == "timeout"
    assert process.terminate_calls >= 1
    assert process.kill_calls == 1
    assert process.events[-2:] == ["kill", "wait"]
    assert session._proc is None


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
@pytest.mark.parametrize(
    ("leader_tail", "expected_stop_reason"),
    [
        ("time.sleep(60)", "timeout"),
        ("", "end_turn"),
    ],
    ids=["leader-running", "leader-exits"],
)
def test_cli_watchdog_kills_descendant_that_inherits_stdout(
    leader_tail: str,
    expected_stop_reason: str,
) -> None:
    """Killing only the CLI leader must not leave stdout held by a child."""
    child_code = (
        "import signal,time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "time.sleep(60)"
    )
    leader_code = (
        "import signal,subprocess,sys,time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}],"
        " stdout=sys.stdout, stderr=sys.stderr);"
        "print('ready', flush=True);"
        f"{leader_tail}"
    )
    real_popen = subprocess.Popen
    spawned: list[subprocess.Popen] = []
    requested_new_session: list[object] = []

    def spawn_descendant(_argv, **kwargs):
        requested_new_session.append(kwargs.get("start_new_session"))
        # Keep a failed pre-fix run safe: isolate the real probe even when the
        # production call has not yet requested its own process group.
        kwargs["start_new_session"] = True
        process = real_popen(
            [sys.executable, "-c", leader_code],
            **kwargs,
        )
        if not leader_tail:
            # Make completion precedence deterministic: the CLI leader has
            # already exited before production computes its prompt deadline,
            # while the descendant still owns the output pipes.
            process.wait(timeout=1)
        spawned.append(process)
        return process

    session = _claude_session()
    patches = (
        patch(
            "src.agent_session.claude_cli.subprocess.Popen",
            side_effect=spawn_descendant,
        ),
        patch(
            "src.agent_session.claude_cli._CLI_TERMINATE_GRACE_S",
            0.1,
        ),
        patch(
            "src.agent_session.claude_cli._CLI_KILL_GRACE_S",
            0.2,
        ),
    )

    observed: list[object] = []
    completed_without_external_kill = False
    prompt_timeout = 0.2 if leader_tail else 1e-9
    try:
        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            worker = threading.Thread(
                target=lambda: observed.append(
                    session.send_prompt("work", timeout=prompt_timeout)
                ),
                daemon=True,
            )
            worker.start()
            worker.join(timeout=1.5)
            completed_without_external_kill = not worker.is_alive()
    finally:
        for process in spawned:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=1)
        if "worker" in locals() and worker.is_alive():
            worker.join(timeout=1)

    assert requested_new_session == [True]
    assert completed_without_external_kill
    assert observed and observed[0].stop_reason == expected_stop_reason
    assert "ready" in observed[0].text
    assert session._proc is None


def test_process_group_receives_full_term_grace_before_kill() -> None:
    process = _CompletedProcess(stdout="", stderr="", returncode=0)
    process_group_id = 123_456_789
    signals: list[int] = []
    term_sent_at = [0.0]

    def signal_group(group_id: int, sig: int) -> None:
        assert group_id == process_group_id
        signals.append(sig)
        if sig == signal.SIGTERM:
            term_sent_at[0] = time.monotonic()
            return
        if sig == 0:
            if time.monotonic() - term_sent_at[0] < 0.03:
                return
            raise ProcessLookupError
        pytest.fail("graceful descendant cleanup escalated to SIGKILL")

    with patch(
        "src.agent_session.process_cleanup.os.killpg",
        side_effect=signal_group,
    ):
        assert terminate_and_reap_process_tree(
            process,
            process_group_id=process_group_id,
            terminate_grace=0.2,
            kill_grace=0.1,
            label="test",
        )

    assert signals[0] == signal.SIGTERM
    assert signal.SIGKILL not in signals


def test_cli_close_failure_keeps_process_handle_and_raises() -> None:
    session = _claude_session()
    process = _UnkillableProcess()
    session._proc = process

    with pytest.raises(RuntimeError, match="failed to terminate"):
        session.close()

    assert session._proc is process
    assert session._force_dead is True
    assert process.terminate_calls >= 1
    assert process.kill_calls >= 1


def test_cli_close_failure_keeps_process_group_handle() -> None:
    session = _claude_session()
    cleanup_path = "src.agent_session.claude_cli._terminate_and_reap_process"

    process = object()
    process_group_id = 987_654_321
    session._proc = process
    session._proc_group_id = process_group_id

    with patch(cleanup_path, return_value=False) as cleanup:
        with pytest.raises(RuntimeError, match="failed to terminate"):
            session.close()

    cleanup.assert_called_once_with(
        process,
        process_group_id=process_group_id,
    )
    assert session._proc is process
    assert session._proc_group_id == process_group_id
    assert session._force_dead is True
