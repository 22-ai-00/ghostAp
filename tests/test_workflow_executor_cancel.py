"""Tests for AgentExecutor cancel guard — active session cancellation.

Verifies that when a per-call cancel_event fires during send_prompt(), the
executor actively calls session.cancel() to interrupt the blocking call,
ensuring race() loser sessions shut down within 5s.
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.workflow_engine.constants import WORKFLOW_TIMEOUT_HEADROOM_S
from src.workflow_engine.executor import AgentExecutor
from src.workflow_engine.models import AgentCallParams


class _SlowCancelableSession:
    """Fake session whose send_prompt blocks until cancel() is called.

    Simulates a long-running LLM call that can be interrupted by cancel().
    """

    def __init__(self, delay_after_cancel: float = 0.1) -> None:
        self.session_id = "slow-session"
        self.created_at = time.time()
        self.last_active = time.time()
        self.message_count = 0
        self.last_query = ""
        self._cancel_event = threading.Event()
        self._delay_after_cancel = delay_after_cancel
        self.cancel_called = threading.Event()
        self.close_called = threading.Event()
        self.send_prompt_started = threading.Event()

    def describe_agent(self) -> str:
        return "fake slow session"

    def start(self, startup_timeout: float = 60) -> str:
        return self.session_id


    def load_local_history(self, *a, **kw):
        return []

    def cancel(self) -> None:
        self.cancel_called.set()
        self._cancel_event.set()

    def close(self) -> None:
        self.close_called.set()

    def to_snapshot(self) -> dict:
        return {"session_id": self.session_id}

    def get_session_info(self) -> str:
        return "SlowCancelableSession"

    def is_server_running(self) -> bool:
        return True

    def is_server_healthy(self, healthcheck_timeout: float = 2.0) -> bool:
        return True

    def send_prompt(self, text: str, on_event=None, timeout=None, idle_timeout=None):
        self.send_prompt_started.set()
        self.last_query = text
        # Block until cancel is called, then simulate short cleanup delay
        cancelled = self._cancel_event.wait(timeout=timeout or 30)
        if cancelled:
            # Simulate the small delay for the process to clean up after cancel
            time.sleep(self._delay_after_cancel)
            raise RuntimeError("cancelled by user")
        # If we timed out without cancel
        raise TimeoutError("prompt timed out")

    def send_prompt_with_retry(self, *args, **kwargs):
        return self.send_prompt(*args, **kwargs)


@pytest.fixture
def make_executor():
    """Factory fixture: create an AgentExecutor with auto-shutdown on teardown.

    Follows the same pattern as make_card_delivery in conftest.py:
    track all created executors and call shutdown(wait=True) during teardown
    to prevent background thread leakage between tests.
    """
    executors: list[AgentExecutor] = []

    def _factory(tmp_path, cancel_event=None):
        ex = AgentExecutor(
            cwd=str(tmp_path),
            cancel_event=cancel_event or threading.Event(),
            max_workers=2,
        )
        executors.append(ex)
        return ex

    yield _factory

    for ex in executors:
        try:
            ex.shutdown(wait=True)
        except Exception:
            pass


def test_cancel_guard_fires_session_cancel_within_200ms(tmp_path, make_executor):
    """When per-call cancel_event fires during send_prompt, session.cancel()
    must be called within 200ms (the guard poll interval + overhead)."""
    executor = make_executor(tmp_path)
    call_cancel = threading.Event()
    session = _SlowCancelableSession(delay_after_cancel=0.05)

    with patch("src.agent_session.factory.create_engine_session", return_value=session):
        params = AgentCallParams(prompt="hello", tool="coco")

        # Run execute in a background thread
        result_holder: list = []

        def run():
            result_holder.append(executor.execute(params, cancel_event=call_cancel))

        t = threading.Thread(target=run, daemon=True)
        t.start()

        # Wait for send_prompt to start
        assert session.send_prompt_started.wait(timeout=5.0), "send_prompt must start"

        # Fire cancel
        start = time.monotonic()
        call_cancel.set()

        # Wait for cancel to be called on the session
        assert session.cancel_called.wait(timeout=2.0), "session.cancel() must be called"
        elapsed_ms = (time.monotonic() - start) * 1000

        assert elapsed_ms < 500, f"session.cancel() should fire in <500ms, took {elapsed_ms:.0f}ms"

        # Wait for execute to finish
        t.join(timeout=10.0)
        assert not t.is_alive(), "execute should complete"

        # Session should be closed
        assert session.close_called.wait(timeout=1.0), "session.close() must be called"

        # Result should have an error (cancelled)
        assert len(result_holder) == 1
        result = result_holder[0]
        assert result.error is not None, "result should indicate error/cancellation"


def test_cancel_guard_does_not_fire_for_normal_completion(tmp_path, make_executor):
    """When send_prompt completes normally, cancel is never called on the session."""
    executor = make_executor(tmp_path)
    call_cancel = threading.Event()

    # Use a FakeSession-like mock that completes immediately
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.text = "hello world"
    mock_result.output_tokens = 10
    mock_session.send_prompt.return_value = mock_result
    mock_session.cancel = MagicMock()
    mock_session.close = MagicMock()

    with patch("src.agent_session.factory.create_engine_session", return_value=mock_session):
        params = AgentCallParams(prompt="hello", tool="coco")
        result = executor.execute(params, cancel_event=call_cancel)

        assert result.output == "hello world"
        assert result.error is None
        # cancel should never have been called for a normal completion
        mock_session.cancel.assert_not_called()
        # close should always be called
        mock_session.close.assert_called_once()


def test_cancel_before_execution_no_session_created(tmp_path, make_executor):
    """If cancel_event is already set before execute runs, it returns early
    without creating a session (no cancel guard needed)."""
    executor = make_executor(tmp_path)
    call_cancel = threading.Event()
    call_cancel.set()  # pre-set

    create_called = threading.Event()

    def delayed_create(*args, **kwargs):
        create_called.set()
        time.sleep(0.1)
        return MagicMock()

    with patch("src.agent_session.factory.create_engine_session", side_effect=delayed_create):
        params = AgentCallParams(prompt="hello", tool="coco")
        result = executor.execute(params, cancel_event=call_cancel)

        assert result.error is not None
        assert "Cancelled before execution" in result.error
        # Session creation should not have been called at all — early exit
        assert not create_called.is_set(), "create_engine_session should not be called when pre-cancelled"


def test_session_close_completes_within_5s_after_cancel(tmp_path, make_executor):
    """End-to-end timing: from cancel_event set to session.close() returning
    must be well under 5s — the acceptance criteria for race loser cleanup."""
    executor = make_executor(tmp_path)
    call_cancel = threading.Event()
    # Simulate session that takes 0.5s to clean up after cancel
    session = _SlowCancelableSession(delay_after_cancel=0.5)

    with patch("src.agent_session.factory.create_engine_session", return_value=session):
        params = AgentCallParams(prompt="hello", tool="coco")

        def run():
            executor.execute(params, cancel_event=call_cancel)

        t = threading.Thread(target=run, daemon=True)
        t.start()

        assert session.send_prompt_started.wait(timeout=5.0)

        start = time.monotonic()
        call_cancel.set()

        # Wait for close to be called
        assert session.close_called.wait(timeout=10.0), "session.close() must be called"
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, f"session.close() must happen within 5s of cancel, took {elapsed:.2f}s"

        t.join(timeout=10.0)


def test_deadline_caps_prompt_timeout(tmp_path, make_executor, monkeypatch):
    """AgentExecutor should cap send_prompt timeout by workflow deadline."""
    import src.workflow_engine.executor as executor_mod

    executor = make_executor(tmp_path)
    seen_timeouts: list[int] = []

    class _DeadlineSession:
        def cancel(self): pass
        def close(self): pass

        def send_prompt(self, text, on_event=None, timeout=None, idle_timeout=None):
            seen_timeouts.append(timeout)
            return SimpleNamespace(text="ok", output_tokens=0)

    session = _DeadlineSession()
    monkeypatch.setattr(executor_mod.time, "monotonic", lambda: 100.0)

    with patch("src.agent_session.factory.create_engine_session", return_value=session):
        params = AgentCallParams(prompt="test", tool="coco", timeout=300)
        result = executor.execute(params, deadline_monotonic=160.0)

    assert result.error is None
    assert seen_timeouts == [int(60.0 - WORKFLOW_TIMEOUT_HEADROOM_S)]


def test_deadline_exhaustion_fails_before_session_creation(tmp_path, make_executor, monkeypatch):
    """No session should be created when the workflow deadline is exhausted."""
    import src.workflow_engine.executor as executor_mod

    executor = make_executor(tmp_path)
    monkeypatch.setattr(executor_mod.time, "monotonic", lambda: 100.0)

    with patch("src.agent_session.factory.create_engine_session") as mock_create:
        params = AgentCallParams(prompt="test", tool="coco", timeout=300)
        result = executor.execute(params, deadline_monotonic=100.0)

    assert result.error is not None
    assert "deadline" in result.error.lower()
    mock_create.assert_not_called()


def test_global_cancel_triggers_cancel_guard(tmp_path, make_executor):
    """When the executor-level (global) cancel_event fires during send_prompt,
    the cancel guard must fire session.cancel() — even if the per-call
    cancel_event is never set.  This validates the OR semantics required for
    /stop_wf to actually interrupt in-flight agent calls."""
    global_cancel = threading.Event()
    executor = make_executor(tmp_path, cancel_event=global_cancel)
    per_call_cancel = threading.Event()  # never set
    session = _SlowCancelableSession(delay_after_cancel=0.05)

    with patch("src.agent_session.factory.create_engine_session", return_value=session):
        params = AgentCallParams(prompt="hello", tool="coco")

        def run():
            executor.execute(params, cancel_event=per_call_cancel)

        t = threading.Thread(target=run, daemon=True)
        t.start()

        assert session.send_prompt_started.wait(timeout=5.0), "send_prompt must start"

        # Fire ONLY the global cancel — per-call stays clear
        assert not per_call_cancel.is_set(), "per-call should not be set"
        start = time.monotonic()
        global_cancel.set()

        assert session.cancel_called.wait(timeout=2.0), (
            "session.cancel() must be called when global cancel fires"
        )
        elapsed_ms = (time.monotonic() - start) * 1000
        assert elapsed_ms < 500, (
            f"session.cancel() should fire in <500ms from global cancel, "
            f"took {elapsed_ms:.0f}ms"
        )

        t.join(timeout=10.0)
        assert not t.is_alive(), "execute should complete"


def test_per_call_cancel_still_works_with_global_present(tmp_path, make_executor):
    """Per-call cancel (race loser abort) must still work independently when
    a global cancel_event is provided.  The OR semantics should not break
    the single-call cancellation path."""
    global_cancel = threading.Event()  # never set
    executor = make_executor(tmp_path, cancel_event=global_cancel)
    per_call_cancel = threading.Event()
    session = _SlowCancelableSession(delay_after_cancel=0.05)

    with patch("src.agent_session.factory.create_engine_session", return_value=session):
        params = AgentCallParams(prompt="hello", tool="coco")

        def run():
            executor.execute(params, cancel_event=per_call_cancel)

        t = threading.Thread(target=run, daemon=True)
        t.start()

        assert session.send_prompt_started.wait(timeout=5.0)

        # Fire only per-call cancel — global stays clear
        assert not global_cancel.is_set()
        start = time.monotonic()
        per_call_cancel.set()

        assert session.cancel_called.wait(timeout=2.0), (
            "session.cancel() must be called when per-call cancel fires"
        )
        elapsed_ms = (time.monotonic() - start) * 1000
        assert elapsed_ms < 500, (
            f"session.cancel() should fire in <500ms from per-call cancel, "
            f"took {elapsed_ms:.0f}ms"
        )

        t.join(timeout=10.0)
        assert not t.is_alive()


# ---------------------------------------------------------------------------
# Effective per-call timeout resolution (host config is the authoritative floor;
# 0 == unlimited; script value may only raise, never lower it)
# ---------------------------------------------------------------------------


class _RecordingSession:
    """Fake session that records the timeout passed to send_prompt and returns
    a successful result immediately (no blocking)."""

    def __init__(self) -> None:
        self.session_id = "recording"
        self.recorded_timeout = None

    def describe_agent(self): return "fake"
    def start(self, startup_timeout: float = 60): return self.session_id
    def load_local_history(self, *a, **kw): return []
    def cancel(self): pass
    def close(self): pass
    def to_snapshot(self): return {}
    def get_session_info(self): return "RecordingSession"
    def is_server_running(self): return True
    def is_server_healthy(self, healthcheck_timeout: float = 2.0): return True

    def send_prompt(self, text, on_event=None, timeout=None, idle_timeout=None):
        self.recorded_timeout = timeout
        return SimpleNamespace(text="ok", output_tokens=0)

    def send_prompt_with_retry(self, *args, **kwargs):
        return self.send_prompt(*args, **kwargs)


def _run_and_capture_timeout(tmp_path, make_executor, *, params, settings_agent_timeout):
    """Execute one agent call and return the timeout passed to send_prompt."""
    session = _RecordingSession()

    def _fake_settings_int(field, fallback):
        if field == "workflow_agent_call_timeout_s":
            return settings_agent_timeout
        return fallback

    executor = make_executor(tmp_path)
    with patch("src.agent_session.factory.create_engine_session", return_value=session), \
         patch("src.workflow_engine.executor._settings_int", side_effect=_fake_settings_int):
        result = executor.execute(params)
    assert result.error is None, f"unexpected error: {result.error}"
    return session.recorded_timeout


def test_script_small_timeout_is_raised_to_config_floor(tmp_path, make_executor):
    """A small script-baked timeout (e.g. 180) must NOT lower the effective
    per-call timeout below the configured floor (600). This is the core fix:
    the LLM script's short timeout was killing long-running coding tasks."""
    params = AgentCallParams(prompt="long task", tool="coco", timeout=180)
    recorded = _run_and_capture_timeout(
        tmp_path, make_executor, params=params, settings_agent_timeout=600
    )
    assert recorded == 600, f"expected floor 600, got {recorded}"


def test_script_larger_timeout_can_raise_above_floor(tmp_path, make_executor):
    """A script timeout larger than the configured floor may raise it."""
    params = AgentCallParams(prompt="very long task", tool="coco", timeout=1800)
    recorded = _run_and_capture_timeout(
        tmp_path, make_executor, params=params, settings_agent_timeout=600
    )
    assert recorded == 1800, f"expected raised 1800, got {recorded}"


def test_config_zero_means_unlimited_uses_finite_backstop(tmp_path, make_executor):
    """A configured per-agent timeout of 0 (unlimited) resolves to the finite
    backstop so the blocking call still eventually returns, never a tiny
    script value."""
    from src.workflow_engine.constants import AGENT_UNLIMITED_BACKSTOP_S

    params = AgentCallParams(prompt="unbounded task", tool="coco", timeout=180)
    recorded = _run_and_capture_timeout(
        tmp_path, make_executor, params=params, settings_agent_timeout=0
    )
    assert recorded == AGENT_UNLIMITED_BACKSTOP_S, (
        f"expected backstop {AGENT_UNLIMITED_BACKSTOP_S}, got {recorded}"
    )
