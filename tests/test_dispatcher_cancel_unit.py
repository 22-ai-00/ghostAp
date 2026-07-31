"""Tests for dispatcher _cancel_unit and safe_invoke integration."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.worktree_engine.dispatcher import WorktreeDispatcher
from src.worktree_engine.models import WorktreeUnit, WorktreeUnitStatus


@dataclass
class FakePromptResult:
    stop_reason: str = "end_turn"
    text: str = "done"


class SlowSession:
    """Session that sleeps to trigger pool timeout."""

    def __init__(self, *, provider, tool_name, working_dir, model_name=None, **kw):
        self.provider = provider
        self.tool_name = tool_name
        self.working_dir = working_dir

    def start(self, startup_timeout=60):
        return "session"

    def send_prompt(self, text, on_event=None, timeout=None):
        time.sleep(0.5)  # Long enough to exceed pool timeout
        return FakePromptResult()

    def close(self):
        pass


class FastSession:
    """Session that completes immediately."""

    def __init__(self, *, provider, tool_name, working_dir, model_name=None, **kw):
        pass

    def start(self, startup_timeout=60):
        return "session"

    def send_prompt(self, text, on_event=None, timeout=None):
        return FakePromptResult()

    def close(self):
        pass


def _make_unit(tmp_path: Path, idx: int = 0) -> WorktreeUnit:
    wt = tmp_path / f"wt{idx}"
    wt.mkdir(exist_ok=True)
    unit = WorktreeUnit(unit_id=f"u{idx}", worktree_path=str(wt))
    unit.provider = "test"
    unit.tool_name = "coco"
    unit.task_title = f"task-{idx}"
    unit.task_prompt = f"do something {idx}"
    unit.status = WorktreeUnitStatus.PLANNED
    return unit


class TestPoolTimeoutSetsCancelled:
    """Pool timeout should mark unfinished units as CANCELLED, not FAILED."""

    def test_pool_timeout_sets_cancelled_status(self, tmp_path):
        unit = _make_unit(tmp_path, 0)
        dispatcher = WorktreeDispatcher(session_factory=SlowSession)

        results = dispatcher.execute_units([unit], pool_timeout=0.1)

        assert len(results) == 1
        assert results[0].status == WorktreeUnitStatus.CANCELLED
        assert results[0].to_dict()["cancelled"] is True
        assert results[0].error != ""

    def test_pool_timeout_mixed_fast_and_slow(self, tmp_path):
        """Fast units complete; slow ones get cancelled."""
        fast_unit = _make_unit(tmp_path, 0)
        slow_unit = _make_unit(tmp_path, 1)

        call_count = {"n": 0}

        def session_factory(*, provider, tool_name, working_dir, model_name=None, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return FastSession(provider=provider, tool_name=tool_name, working_dir=working_dir)
            return SlowSession(provider=provider, tool_name=tool_name, working_dir=working_dir)

        dispatcher = WorktreeDispatcher(session_factory=session_factory)
        results = dispatcher.execute_units([fast_unit, slow_unit], pool_timeout=0.2)

        statuses = {u.unit_id: u.status for u in results}
        # Fast unit should complete; slow unit should be cancelled
        assert statuses["u0"] == WorktreeUnitStatus.COMPLETED
        assert statuses["u1"] == WorktreeUnitStatus.CANCELLED


class TestFailUnitUsesSafeInvoke:
    """_fail_unit should use safe_invoke for callbacks (no crash on bad callback)."""

    def test_fail_unit_uses_safe_invoke(self, tmp_path):
        unit = _make_unit(tmp_path, 0)
        dispatcher = WorktreeDispatcher()

        bad_callback = MagicMock(side_effect=RuntimeError("callback exploded"))
        # Should not raise even with a bad callback
        dispatcher._fail_unit(unit, "test error", on_unit_update=bad_callback)

        assert unit.status == WorktreeUnitStatus.FAILED
        assert unit.error == "test error"
        bad_callback.assert_called_once_with(unit)

    def test_fail_unit_none_callback(self, tmp_path):
        unit = _make_unit(tmp_path, 0)
        dispatcher = WorktreeDispatcher()
        # Should not raise with None callback
        dispatcher._fail_unit(unit, "test error", on_unit_update=None)
        assert unit.status == WorktreeUnitStatus.FAILED


class TestCancelUnitNotOverwriteCompleted:
    """_cancel_unit should not overwrite already-completed units."""

    def test_cancel_unit_not_overwrite_completed(self, tmp_path):
        """_run_single_unit skips if status is already CANCELLED."""
        unit = _make_unit(tmp_path, 0)
        unit.status = WorktreeUnitStatus.CANCELLED

        dispatcher = WorktreeDispatcher(session_factory=FastSession)
        # _run_single_unit should return immediately without changing status
        dispatcher._run_single_unit(unit)
        assert unit.status == WorktreeUnitStatus.CANCELLED

    def test_run_single_unit_skips_failed(self, tmp_path):
        """_run_single_unit also skips FAILED status."""
        unit = _make_unit(tmp_path, 0)
        unit.status = WorktreeUnitStatus.FAILED

        dispatcher = WorktreeDispatcher(session_factory=FastSession)
        dispatcher._run_single_unit(unit)
        assert unit.status == WorktreeUnitStatus.FAILED


class TestSessionFactoryException:
    """session_factory raising exception should mark unit as FAILED."""

    def test_session_factory_exception_marks_failed(self, tmp_path):
        """When session_factory raises, unit should be marked FAILED."""
        unit = _make_unit(tmp_path, 0)

        def bad_factory(*, provider, tool_name, working_dir, model_name=None, **kw):
            raise RuntimeError("cannot create session")

        dispatcher = WorktreeDispatcher(session_factory=bad_factory)
        results = dispatcher.execute_units([unit], pool_timeout=10)

        assert len(results) == 1
        assert results[0].status == WorktreeUnitStatus.FAILED
        assert "cannot create session" in results[0].error

    def test_session_start_exception_marks_failed(self, tmp_path):
        """When session.start() raises, unit should be marked FAILED."""
        unit = _make_unit(tmp_path, 0)

        class FailStartSession:
            def __init__(self, **kw):
                pass
            def start(self, startup_timeout=60):
                raise RuntimeError("start failed")
            def close(self):
                pass

        dispatcher = WorktreeDispatcher(session_factory=FailStartSession)
        results = dispatcher.execute_units([unit], pool_timeout=10)

        assert len(results) == 1
        assert results[0].status == WorktreeUnitStatus.FAILED
        assert "start failed" in results[0].error


class TestStopReasonMapping:
    """stop_reason values map correctly to unit status."""

    def test_stop_reason_error_maps_to_failed(self, tmp_path):
        """send_prompt returning stop_reason='error' → unit FAILED."""
        unit = _make_unit(tmp_path, 0)

        class ErrorSession:
            def __init__(self, **kw):
                pass
            def start(self, startup_timeout=60):
                pass
            def send_prompt(self, text, on_event=None, timeout=None):
                return FakePromptResult(stop_reason="error", text="something went wrong")
            def close(self):
                pass

        dispatcher = WorktreeDispatcher(session_factory=ErrorSession)
        results = dispatcher.execute_units([unit], pool_timeout=10)

        assert len(results) == 1
        assert results[0].status == WorktreeUnitStatus.FAILED

    def test_stop_reason_cancelled_maps_to_cancelled(self, tmp_path):
        """send_prompt returning stop_reason='cancelled' preserves cancellation truth."""
        unit = _make_unit(tmp_path, 0)

        class CancelledSession:
            def __init__(self, **kw):
                pass
            def start(self, startup_timeout=60):
                pass
            def send_prompt(self, text, on_event=None, timeout=None):
                return FakePromptResult(stop_reason="cancelled", text="was cancelled")
            def close(self):
                pass

        dispatcher = WorktreeDispatcher(session_factory=CancelledSession)
        results = dispatcher.execute_units([unit], pool_timeout=10)

        assert len(results) == 1
        assert results[0].status == WorktreeUnitStatus.CANCELLED

    @pytest.mark.parametrize("stop_reason", ["", "unknown", "timeout", "max_tokens"])
    def test_unknown_or_incomplete_stop_reason_fails_closed(
        self,
        tmp_path,
        stop_reason,
    ):
        unit = _make_unit(tmp_path, 0)

        class IncompleteSession:
            def __init__(self, **kw):
                pass

            def start(self, startup_timeout=60):
                pass

            def send_prompt(self, text, on_event=None, timeout=None):
                return FakePromptResult(stop_reason=stop_reason, text="partial")

            def close(self):
                pass

        results = WorktreeDispatcher(
            session_factory=IncompleteSession
        ).execute_units([unit], pool_timeout=10)

        assert results[0].status == WorktreeUnitStatus.FAILED

    def test_end_turn_with_pending_plan_fails_closed(self, tmp_path):
        unit = _make_unit(tmp_path, 0)

        class PendingPlanSession:
            def __init__(self, **kw):
                pass

            def start(self, startup_timeout=60):
                pass

            def send_prompt(self, text, on_event=None, timeout=None):
                return SimpleNamespace(
                    stop_reason="end_turn",
                    text="partial",
                    plan=SimpleNamespace(
                        entries=[SimpleNamespace(status="pending")]
                    ),
                    tool_calls=[],
                )

            def close(self):
                pass

        results = WorktreeDispatcher(
            session_factory=PendingPlanSession
        ).execute_units([unit], pool_timeout=10)

        assert results[0].status == WorktreeUnitStatus.FAILED

    def test_stop_reason_end_turn_maps_to_completed(self, tmp_path):
        """send_prompt returning stop_reason='end_turn' → unit COMPLETED."""
        unit = _make_unit(tmp_path, 0)

        dispatcher = WorktreeDispatcher(session_factory=FastSession)
        results = dispatcher.execute_units([unit], pool_timeout=10)

        assert len(results) == 1
        assert results[0].status == WorktreeUnitStatus.COMPLETED

    def test_completed_unit_records_structured_command_provenance(self, tmp_path):
        unit = _make_unit(tmp_path, 0)

        class ProvenanceSession:
            def __init__(self, **kw):
                pass

            def start(self, startup_timeout=60):
                pass

            def send_prompt(self, text, on_event=None, timeout=None):
                return SimpleNamespace(
                    stop_reason="end_turn",
                    text="verified",
                    plan=None,
                    tool_calls=[],
                    tool_results=[
                        {
                            "kind": "execute",
                            "data": {
                                "command": "uv run pytest -q",
                                "exit_code": 0,
                                "stdout": "12 passed",
                            },
                        }
                    ],
                )

            def close(self):
                pass

        result = WorktreeDispatcher(
            session_factory=ProvenanceSession
        ).execute_units([unit], pool_timeout=10)[0]

        assert result.metadata["test_provenance"] == [
            {
                "source": "worktree_dispatcher",
                "command": "uv run pytest -q",
                "exit_code": 0,
                "evidence": "12 passed",
            }
        ]


class TestTTADKProcKillEscalation:
    """proc.kill() escalation when proc.terminate()+wait() times out."""

    def test_proc_kill_called_on_wait_timeout(self):
        """If proc.wait(timeout=5) raises TimeoutExpired after terminate(),
        proc.kill() should be called as escalation.

        Tests by injecting a mock proc into the session and patching
        build_ttadk_subprocess_env to raise, forcing the finally block to run
        with a "still running" process.
        """
        import subprocess as _subprocess
        from unittest.mock import MagicMock

        from src.agent_session.ttadk_cli import SyncTTADKCLISession

        session = SyncTTADKCLISession.__new__(SyncTTADKCLISession)
        session._tool_name = "test_tool"
        session._tool = "test_tool"
        session._model_name = None
        session._cancel_event = MagicMock()
        session._cancel_event.is_set.return_value = False
        session._cancel_event.clear = MagicMock()
        session._last_active = 0
        session._prompt_count = 0
        session._cwd = "/tmp"
        session._agent_type = "ttadk_test"
        session._employee_sandbox = None
        session.session_id = "test_session"
        session.last_active = 0
        session.message_count = 0
        session.last_query = ""

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # process still running
        mock_proc.terminate.return_value = None
        mock_proc.wait.side_effect = _subprocess.TimeoutExpired(cmd="test", timeout=5)
        mock_proc.kill.return_value = None

        # Pre-set _proc so the finally block will try to terminate it
        session._proc = mock_proc

        # Patch build_ttadk_subprocess_env to raise an OSError,
        # which triggers the finally block with _proc still set
        with patch("src.agent_session.ttadk_cli.build_ttadk_subprocess_env", side_effect=OSError("env build failed")):
            session.send_prompt("test prompt")

        # The finally block should have attempted terminate → wait (timeout) → kill
        mock_proc.terminate.assert_called()
        mock_proc.kill.assert_called()


class TestCancelUnitSkipsCompleted:
    """_cancel_unit should not overwrite COMPLETED units — direct unit test."""

    def test_cancel_unit_does_not_overwrite_completed(self, tmp_path):
        """Directly call _cancel_unit on a COMPLETED unit and verify no change."""
        unit = _make_unit(tmp_path, 0)
        unit.status = WorktreeUnitStatus.COMPLETED
        unit.summary = "All done"

        dispatcher = WorktreeDispatcher(session_factory=FastSession)
        dispatcher._cancel_unit(unit, "pool_timeout")

        assert unit.status == WorktreeUnitStatus.COMPLETED
        assert unit.summary == "All done"
        assert not unit._cancel_event.is_set()

    def test_cancel_unit_does_not_overwrite_failed(self, tmp_path):
        """A timeout racing a recorded failure must preserve the first terminal fact."""
        unit = _make_unit(tmp_path, 0)
        unit.status = WorktreeUnitStatus.FAILED
        unit.error = "original error"

        dispatcher = WorktreeDispatcher(session_factory=FastSession)
        dispatcher._cancel_unit(unit, "pool_timeout")

        assert unit.status == WorktreeUnitStatus.FAILED
        assert unit.error == "original error"
        assert not unit._cancel_event.is_set()
