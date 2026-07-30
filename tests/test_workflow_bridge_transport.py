"""Tests for RuntimeBridge transport-layer behavior (no real Node.js process).

Uses unittest.mock to simulate subprocess, stdout, and executor so we can
exercise error paths, backpressure, lifecycle ordering, and EOF handling
without spawning Node.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

import src.workflow_engine.bridge as bridge_mod
from src.workflow_engine.bridge import RuntimeBridge
from src.workflow_engine.constants import (
    WORKFLOW_TIMEOUT_HEADROOM_S,
    WORKFLOW_TOTAL_TIMEOUT_S,
)
from src.workflow_engine.models import AgentCallResult

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_bridge(tmp_path, **kwargs) -> RuntimeBridge:
    """Construct a bridge with sensible defaults, NOT calling start()."""
    defaults = dict(
        script_path="test_workflow.js",
        cwd=str(tmp_path),
        on_agent_call=lambda params: MagicMock(output="ok"),
    )
    defaults.update(kwargs)
    return RuntimeBridge(**defaults)


def _attach_mock_process(bridge: RuntimeBridge) -> MagicMock:
    """Attach a MagicMock in place of self._process with stdin/stdout stubs."""
    proc = MagicMock()
    proc.poll.return_value = None
    proc.returncode = 0
    proc.stdin = MagicMock()
    proc.stdout = MagicMock()
    proc.stderr = MagicMock()
    bridge._process = proc
    return proc


# ---------------------------------------------------------------------------
# 1. Broken pipe sets done and run() loop exits promptly
# ---------------------------------------------------------------------------


def test_broken_pipe_sets_done(tmp_path):
    """When _send() catches BrokenPipeError, self._done becomes True and
    self._error is populated so run() exits without waiting for the full
    WORKFLOW_TOTAL_TIMEOUT_S."""
    bridge = _make_bridge(tmp_path)
    try:
        proc = _attach_mock_process(bridge)

        # Simulate BrokenPipeError on stdin.write
        proc.stdin.write.side_effect = BrokenPipeError("pipe broken")

        start = time.monotonic()
        bridge._send({"jsonrpc": "2.0", "method": "ping"})
        elapsed = time.monotonic() - start

        assert bridge._done is True, "_done must be set after BrokenPipeError"
        assert bridge._error is not None, "_error must be set after BrokenPipeError"
        assert "connection lost" in bridge._error.lower() or "brokenpipe" in bridge._error.lower()
        # _send should return almost immediately (no retry loop)
        assert elapsed < 1.0, "_send should not block on BrokenPipeError"
    finally:
        bridge.stop()


def test_broken_pipe_run_loop_exits_promptly(tmp_path):
    """After a BrokenPipeError flags _done, run() should exit promptly
    rather than blocking for the total timeout."""
    bridge = _make_bridge(tmp_path)
    try:
        proc = _attach_mock_process(bridge)
        # Provide an executor so that run() doesn't crash on unexpected paths;
        # we don't actually submit anything.
        bridge._executor = MagicMock()
        bridge._executor.shutdown = MagicMock()

        # Arrange: first poll returns None (alive), then _send via some path
        # triggers BrokenPipeError.  We simulate by directly pre-setting _done
        # through the _send path — then calling run() which should return/raise
        # quickly because _done is already True.
        proc.stdin.write.side_effect = BrokenPipeError("pipe broken")
        # Trigger broken pipe via _send (as if some writer did this concurrently)
        bridge._send({"jsonrpc": "2.0", "method": "cancel"})

        assert bridge._done is True

        # run() sees _done True on the first iteration check — wait, actually
        # while not self._done will exit immediately.  Let's just verify by
        # calling run() and timing it.
        start = time.monotonic()
        with pytest.raises(RuntimeError, match="runtime error"):
            bridge.run()
        elapsed = time.monotonic() - start
        assert elapsed < 2.0, (
            f"run() should exit promptly after broken pipe (took {elapsed:.2f}s)"
        )
    finally:
        bridge.stop()


# ---------------------------------------------------------------------------
# 2. stop() kills process BEFORE executor.shutdown
# ---------------------------------------------------------------------------


def test_stop_kills_process_before_shutdown(tmp_path):
    """stop() must kill the Node process before calling executor.shutdown
    so that in-flight agent calls fail fast instead of blocking stop()."""
    bridge = _make_bridge(tmp_path)
    _attach_mock_process(bridge)

    # Use a real ThreadPoolExecutor so we can spy on shutdown ordering
    bridge._executor = ThreadPoolExecutor(max_workers=2)
    bridge._workflow_executor = ThreadPoolExecutor(max_workers=2)

    call_order: list[str] = []

    # Wrap _kill_process to record it was called
    real_kill = bridge._kill_process

    def tracked_kill():
        call_order.append("kill_process")
        real_kill()

    bridge._kill_process = tracked_kill

    # Wrap executor.shutdown to record when it's called
    real_exec_shutdown = bridge._executor.shutdown

    def tracked_exec_shutdown(*args, **kwargs):
        call_order.append("executor_shutdown")
        return real_exec_shutdown(*args, **kwargs)

    bridge._executor.shutdown = tracked_exec_shutdown

    try:
        bridge.stop()

        # After stop(), self._process must be None (killed and cleared)
        assert bridge._process is None, "_process must be None after stop()"
        assert bridge._shutdown_done is True

        # Verify ordering: kill_process appears BEFORE executor_shutdown
        assert "kill_process" in call_order
        assert "executor_shutdown" in call_order
        assert call_order.index("kill_process") < call_order.index("executor_shutdown"), (
            f"kill_process must precede executor_shutdown, got order: {call_order}"
        )
    finally:
        # Safety cleanup (executor is already shut down by stop())
        pass


def test_start_sends_deadline_budget_in_init(tmp_path, monkeypatch):
    """The JS runtime needs the host deadline to fail before hard kill."""
    # Hermetic: pin the total-timeout to the code default regardless of the
    # deployment .env (which may set 0 = unlimited). This test specifically
    # exercises the bounded-deadline propagation path.
    monkeypatch.setattr(
        bridge_mod,
        "_settings_int",
        lambda field, fallback: WORKFLOW_TOTAL_TIMEOUT_S
        if field == "workflow_total_timeout_s"
        else fallback,
    )
    bridge = _make_bridge(tmp_path)
    sent_messages: list[dict] = []

    class FakeStream:
        def __iter__(self):
            return iter([])

        def readline(self):
            return ""

        def read(self):
            return ""

        def close(self):
            pass

    class FakeProcess:
        def __init__(self):
            self.stdin = MagicMock()
            self.stdout = FakeStream()
            self.stderr = FakeStream()
            self.returncode = 0

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(bridge_mod.shutil, "which", lambda name: "/usr/bin/node")
    monkeypatch.setattr(bridge_mod.subprocess, "Popen", lambda *a, **k: FakeProcess())
    monkeypatch.setattr(
        bridge,
        "_wait_for_notification",
        lambda method, timeout=30.0: {"jsonrpc": "2.0", "method": "ready"},
    )
    monkeypatch.setattr(bridge, "_send", lambda msg: sent_messages.append(msg))

    try:
        bridge.start()
    finally:
        bridge.stop()

    init_msg = next(msg for msg in sent_messages if msg.get("method") == "init")
    params = init_msg["params"]
    assert params["total_timeout_s"] == WORKFLOW_TOTAL_TIMEOUT_S
    assert params["deadline_unix_ms"] > params["started_unix_ms"]
    assert params["deadline_unix_ms"] - params["started_unix_ms"] <= (
        WORKFLOW_TOTAL_TIMEOUT_S * 1000
    )


def test_agent_call_timeout_is_capped_by_remaining_workflow_budget(tmp_path, monkeypatch):
    """Host-side cap prevents one late agent() from outliving the run budget."""
    bridge = _make_bridge(tmp_path)
    monkeypatch.setattr(bridge_mod.time, "monotonic", lambda: 100.0)
    bridge._workflow_deadline_monotonic = 160.0

    params = {"prompt": "late call", "tool": "coco", "timeout": 300}
    capped = bridge._cap_agent_timeout_to_remaining_budget(params)

    expected = int(60.0 - WORKFLOW_TIMEOUT_HEADROOM_S)
    assert capped is not params
    assert capped["timeout"] == expected
    assert params["timeout"] == 300


def test_ensure_deadline_uses_settings_total_timeout(tmp_path, monkeypatch):
    """A larger workflow_total_timeout_s from Settings yields a larger deadline."""
    # Simulate an .env override of 7200s (2h) for complex tasks.
    monkeypatch.setattr(
        bridge_mod,
        "_settings_int",
        lambda field, fallback: 7200 if field == "workflow_total_timeout_s" else fallback,
    )
    monkeypatch.setattr(bridge_mod.time, "monotonic", lambda: 1000.0)

    bridge = _make_bridge(tmp_path)
    bridge._ensure_workflow_deadline()

    assert bridge._workflow_total_timeout_s == 7200
    # deadline = started_monotonic (1000) + total_timeout (7200)
    assert bridge._workflow_deadline_monotonic == 1000.0 + 7200
    assert (
        bridge._workflow_deadline_unix_ms - bridge._workflow_started_unix_ms
        == 7200 * 1000
    )


def test_cap_agent_timeout_uses_settings_fallback_when_unset(tmp_path, monkeypatch):
    """When JS omits per-call timeout, the Settings default (larger) is used."""
    monkeypatch.setattr(
        bridge_mod,
        "_settings_int",
        lambda field, fallback: 1200 if field == "workflow_agent_call_timeout_s" else fallback,
    )
    monkeypatch.setattr(bridge_mod.time, "monotonic", lambda: 100.0)
    bridge = _make_bridge(tmp_path)
    # Plenty of budget so the cap does not clamp the requested value.
    bridge._workflow_deadline_monotonic = 100.0 + 10_000

    # No explicit "timeout" in params → falls back to settings value (1200).
    capped = bridge._cap_agent_timeout_to_remaining_budget({"prompt": "x", "tool": "coco"})
    assert capped["timeout"] == 1200


def test_cap_agent_timeout_small_script_value_raised_to_floor(tmp_path, monkeypatch):
    """A small script-baked per-agent timeout must be raised to the configured
    floor, not honored verbatim — this is the fix for long agent() calls being
    killed at the script's tiny timeout."""
    monkeypatch.setattr(
        bridge_mod,
        "_settings_int",
        lambda field, fallback: 600 if field == "workflow_agent_call_timeout_s" else fallback,
    )
    monkeypatch.setattr(bridge_mod.time, "monotonic", lambda: 100.0)
    bridge = _make_bridge(tmp_path)
    # Plenty of budget so the cap does not clamp the floor.
    bridge._workflow_deadline_monotonic = 100.0 + 10_000

    capped = bridge._cap_agent_timeout_to_remaining_budget(
        {"prompt": "x", "tool": "coco", "timeout": 180}
    )
    assert capped["timeout"] == 600, f"expected floor 600, got {capped['timeout']}"


def test_cap_agent_timeout_zero_floor_is_unlimited_backstop(tmp_path, monkeypatch):
    """A configured floor of 0 (unlimited) resolves to the finite backstop when
    no total deadline caps it."""
    from src.workflow_engine.constants import AGENT_UNLIMITED_BACKSTOP_S

    monkeypatch.setattr(
        bridge_mod,
        "_settings_int",
        lambda field, fallback: 0 if field == "workflow_agent_call_timeout_s" else fallback,
    )
    bridge = _make_bridge(tmp_path)
    # Unlimited total deadline (remaining budget is None).
    bridge._workflow_deadline_monotonic = None

    capped = bridge._cap_agent_timeout_to_remaining_budget(
        {"prompt": "x", "tool": "coco", "timeout": 180}
    )
    assert capped["timeout"] == AGENT_UNLIMITED_BACKSTOP_S


def test_start_sends_agent_call_timeout_floor_in_init(tmp_path, monkeypatch):
    """The JS runtime must receive the per-agent timeout floor so its watchdog
    matches the Python executor instead of the script's small baked value."""
    monkeypatch.setattr(
        bridge_mod,
        "_settings_int",
        lambda field, fallback: 900 if field == "workflow_agent_call_timeout_s" else fallback,
    )
    bridge = _make_bridge(tmp_path)
    sent_messages: list[dict] = []

    class FakeStream:
        def __iter__(self):
            return iter([])

        def readline(self):
            return ""

        def read(self):
            return ""

        def close(self):
            pass

    class FakeProcess:
        def __init__(self):
            self.stdin = MagicMock()
            self.stdout = FakeStream()
            self.stderr = FakeStream()
            self.returncode = 0

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(bridge_mod.shutil, "which", lambda name: "/usr/bin/node")
    monkeypatch.setattr(bridge_mod.subprocess, "Popen", lambda *a, **k: FakeProcess())
    monkeypatch.setattr(
        bridge,
        "_wait_for_notification",
        lambda method, timeout=30.0: {"jsonrpc": "2.0", "method": "ready"},
    )
    monkeypatch.setattr(bridge, "_send", lambda msg: sent_messages.append(msg))

    try:
        bridge.start()
    finally:
        bridge.stop()

    init_msg = next(msg for msg in sent_messages if msg.get("method") == "init")
    assert init_msg["params"]["agent_call_timeout_s"] == 900


def test_ensure_deadline_unlimited_when_total_timeout_zero(tmp_path, monkeypatch):
    """workflow_total_timeout_s <= 0 → no total deadline (unlimited mode).

    In unlimited mode the monotonic deadline stays None and the unix-ms value
    sent to JS is 0 (JS treats a falsy deadline as Infinity), while the
    started_* fields are still populated. Remaining-budget helpers must report
    'no deadline' so per-call timeout capping falls back to the Settings value.
    """
    monkeypatch.setattr(
        bridge_mod,
        "_settings_int",
        lambda field, fallback: 0 if field == "workflow_total_timeout_s" else fallback,
    )
    bridge = _make_bridge(tmp_path)
    bridge._ensure_workflow_deadline()

    assert bridge._workflow_total_timeout_s == 0
    assert bridge._workflow_deadline_monotonic is None
    assert bridge._workflow_deadline_unix_ms == 0
    assert bridge._workflow_started_monotonic is not None
    assert bridge._workflow_started_unix_ms is not None
    # No deadline → remaining budget is None and no rejection happens.
    assert bridge._remaining_workflow_budget_s() is None
    bridge._send_error_response = MagicMock()
    assert bridge._reject_if_workflow_budget_exhausted("req-1") is False
    bridge._send_error_response.assert_not_called()


def test_agent_call_rejected_when_no_workflow_budget_remains(tmp_path, monkeypatch):
    bridge = _make_bridge(tmp_path)
    monkeypatch.setattr(bridge_mod.time, "monotonic", lambda: 100.0)
    bridge._workflow_deadline_monotonic = 100.0
    bridge._send_error_response = MagicMock()

    assert bridge._reject_if_workflow_budget_exhausted("req-1") is True
    bridge._send_error_response.assert_called_once()
    assert "deadline" in bridge._send_error_response.call_args.kwargs["message"].lower()


def test_handle_agent_call_passes_capped_timeout_to_callback(tmp_path, monkeypatch):
    seen_timeouts: list[int] = []
    called = threading.Event()

    def on_agent_call(params, **_kwargs):
        seen_timeouts.append(params.timeout)
        called.set()
        return AgentCallResult(output="ok", tool=params.tool)

    bridge = _make_bridge(tmp_path, max_concurrent=1, on_agent_call=on_agent_call)
    try:
        _attach_mock_process(bridge)
        bridge._executor = ThreadPoolExecutor(max_workers=1)
        monkeypatch.setattr(bridge_mod.time, "monotonic", lambda: 100.0)
        bridge._workflow_deadline_monotonic = 160.0

        bridge._handle_agent_call(
            {"prompt": "late call", "tool": "coco", "timeout": 300},
            request_id="req-capped",
        )

        assert called.wait(timeout=2.0)
        assert seen_timeouts == [int(60.0 - WORKFLOW_TIMEOUT_HEADROOM_S)]
    finally:
        bridge.stop()


# ---------------------------------------------------------------------------
# 3. abort_request cancels the corresponding future
# ---------------------------------------------------------------------------


def test_abort_request_cancels_future(tmp_path):
    """When an 'abort_request' notification arrives from JS, the future
    registered under that request_id must be cancelled."""
    bridge = _make_bridge(tmp_path)
    try:
        _attach_mock_process(bridge)

        # Register a pending future (not yet running, so cancel() will succeed)
        fut: Future = Future()
        req_id = "req-abc-123"
        with bridge._request_futures_lock:
            bridge._request_futures[req_id] = fut

        # Dispatch the abort notification like the JS runtime would send it
        bridge._dispatch_message({
            "jsonrpc": "2.0",
            "method": "abort_request",
            "params": {"request_id": req_id},
        })

        assert fut.cancelled(), "Future should be cancelled after abort_request"
        with bridge._request_futures_lock:
            assert req_id not in bridge._request_futures, (
                "aborted request_id should be removed from the map"
            )
    finally:
        bridge.stop()


def test_abort_request_unknown_id_is_noop(tmp_path):
    """An abort_request for an unknown request_id must not raise."""
    bridge = _make_bridge(tmp_path)
    try:
        _attach_mock_process(bridge)

        # No future registered for "does-not-exist" — should be a no-op
        bridge._dispatch_message({
            "jsonrpc": "2.0",
            "method": "abort_request",
            "params": {"request_id": "does-not-exist"},
        })
        # No exception => pass
    finally:
        bridge.stop()


# ---------------------------------------------------------------------------
# 4. Non-JSON lines in stdout do not crash the reader thread
# ---------------------------------------------------------------------------


def test_read_loop_non_json_does_not_crash(tmp_path, caplog):
    """Feeding a mix of non-JSON and valid JSON lines through stdout must
    not crash _read_loop, and after >=10 consecutive non-JSON lines a warning
    is logged, but a subsequent valid JSON line is still queued."""
    bridge = _make_bridge(tmp_path)
    try:
        proc = _attach_mock_process(bridge)

        # Build 12 non-JSON lines, then 1 valid JSON line
        non_json_lines = [f"garbage line {i}\n" for i in range(12)]
        valid_msg = json.dumps({"jsonrpc": "2.0", "method": "log",
                                "params": {"message": "hello"}}) + "\n"

        # stdout must be an iterable for `for line in self._process.stdout`
        proc.stdout = iter(non_json_lines + [valid_msg])

        with caplog.at_level(logging.WARNING, logger="src.workflow_engine.bridge"):
            bridge._read_loop()

        # After _read_loop, _done should be set (stdout reached EOF because
        # our iter was exhausted). But we care more about:
        # - No exception raised (if we got here, we passed)
        # - A warning was logged at least once for >=10 consecutive non-JSON
        warn_records = [r for r in caplog.records
                        if r.levelno >= logging.WARNING and "non-json" in r.getMessage().lower()]
        assert len(warn_records) >= 1, "Expected a warning log for non-JSON output"

        # The valid JSON message must have been enqueued despite the preceding
        # garbage lines.
        with bridge._msg_condition:
            queued_methods = [m.get("method") for m in bridge._msg_queue]
        assert "log" in queued_methods, (
            f"Valid JSON after garbage should still be queued; got methods={queued_methods}"
        )
    finally:
        # stdout was replaced with a plain iterator; clear _process so stop()
        # doesn't try to close iterator streams that lack a close() method.
        bridge._process = None
        bridge.stop()


def test_read_loop_queue_full_fails_runtime_instead_of_dropping(tmp_path, monkeypatch):
    """A full bridge queue must become a visible runtime failure.

    Dropping a JSON-RPC response/notification silently can leave a JS Promise
    pending until the total workflow timeout.
    """
    monkeypatch.setattr(bridge_mod, "MAX_QUEUE_SIZE", 1)

    bridge = _make_bridge(tmp_path)
    try:
        proc = _attach_mock_process(bridge)
        with bridge._msg_condition:
            bridge._msg_queue.append({"jsonrpc": "2.0", "method": "already_full"})

        valid_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"data": "late-response"},
        }) + "\n"
        proc.stdout = iter([valid_msg])

        bridge._read_loop()

        assert bridge._done is True
        assert bridge._error is not None
        assert "message queue full" in bridge._error.lower()
    finally:
        bridge._process = None
        bridge.stop()


# ---------------------------------------------------------------------------
# 5. Reader EOF sets _done
# ---------------------------------------------------------------------------


def test_reader_eof_sets_done(tmp_path):
    """When stdout is exhausted (EOF), _read_loop must set self._done so
    the run() loop can exit."""
    bridge = _make_bridge(tmp_path)
    try:
        proc = _attach_mock_process(bridge)

        # Empty iterable simulates immediate EOF (no lines at all)
        proc.stdout = iter([])

        assert bridge._done is False
        bridge._read_loop()
        assert bridge._done is True, "_done must be set after stdout EOF"
        # On EOF without an explicit terminal frame the reader records a
        # *fallback* diagnostic (kept separate from _error so a queued
        # done/error frame can still win in run()). _error stays None here.
        assert bridge._error is None
        assert bridge._eof_fallback_error is not None
        assert "closed stdout unexpectedly" in bridge._eof_fallback_error
    finally:
        # stdout was replaced with a plain iterator; clear _process so stop()
        # doesn't try to close iterator streams that lack a close() method.
        bridge._process = None
        bridge.stop()


def test_reader_eof_preserves_explicit_error(tmp_path):
    """If _error is already set (e.g. from prior send failure), EOF must
    not overwrite it."""
    bridge = _make_bridge(tmp_path)
    try:
        proc = _attach_mock_process(bridge)
        proc.stdout = iter([])

        bridge._error = "original error"
        bridge._done = True
        bridge._read_loop()
        assert bridge._error == "original error", (
            "Existing _error should not be clobbered by EOF path"
        )
    finally:
        # stdout was replaced with a plain iterator; clear _process so stop()
        # doesn't try to close iterator streams that lack a close() method.
        bridge._process = None
        bridge.stop()


# ---------------------------------------------------------------------------
# 6. Process crash (poll returns non-None) is detected promptly
# ---------------------------------------------------------------------------


def test_process_crash_detected_promptly(tmp_path):
    """If poll() returns non-None while run() is looping, RuntimeError must
    be raised immediately, not waiting for WORKFLOW_TOTAL_TIMEOUT_S."""
    bridge = _make_bridge(tmp_path)
    try:
        proc = _attach_mock_process(bridge)
        bridge._executor = MagicMock()
        bridge._executor.shutdown = MagicMock()

        # Process died immediately with returncode=1
        proc.poll.return_value = 1
        proc.returncode = 1
        proc.stderr.read.return_value = "some stderr output"

        start = time.monotonic()
        with pytest.raises(RuntimeError, match="exited unexpectedly"):
            bridge.run()
        elapsed = time.monotonic() - start

        assert elapsed < 2.0, (
            f"run() must raise promptly on process crash (took {elapsed:.2f}s, "
            f"total timeout is {WORKFLOW_TOTAL_TIMEOUT_S}s)"
        )
    finally:
        bridge.stop()


# ---------------------------------------------------------------------------
# 7. stop() is idempotent
# ---------------------------------------------------------------------------


def test_stop_is_idempotent(tmp_path):
    """Calling stop() twice must not raise, and after the second call the
    bridge remains in a clean shut-down state."""
    bridge = _make_bridge(tmp_path)
    _attach_mock_process(bridge)
    bridge._executor = ThreadPoolExecutor(max_workers=2)
    bridge._workflow_executor = ThreadPoolExecutor(max_workers=2)

    # First stop
    bridge.stop()
    assert bridge._shutdown_done is True
    assert bridge._process is None
    assert bridge._cancel_event.is_set()

    # Second stop — must not raise
    bridge.stop()
    # Third via cleanup() alias — must not raise
    bridge.cleanup()

    assert bridge._shutdown_done is True


# ---------------------------------------------------------------------------
# 8. Backpressure rejects when active futures exceed pressure cap
# ---------------------------------------------------------------------------


def test_backpressure_rejects_when_overwhelmed(tmp_path):
    """When active futures count >= pressure_cap (max(2, max_concurrent*2)),
    _handle_agent_call must send an error response with code -32000 instead
    of submitting to the executor."""
    max_concurrent = 4
    pressure_cap = max(2, max_concurrent * 2)  # = 8

    bridge = _make_bridge(tmp_path, max_concurrent=max_concurrent)
    _attach_mock_process(bridge)

    # Use a real executor but spy on submit to ensure it's NOT called
    bridge._executor = ThreadPoolExecutor(max_workers=max_concurrent)
    try:
        sent_errors: list[dict] = []

        def fake_send_error(request_id, code, message, structured=None):
            sent_errors.append({
                "request_id": request_id,
                "code": code,
                "message": message,
            })

        bridge._send_error_response = fake_send_error
        real_submit = bridge._executor.submit
        submit_calls: list = []

        def tracked_submit(*a, **kw):
            submit_calls.append(1)
            return real_submit(*a, **kw)

        bridge._executor.submit = tracked_submit

        # Saturate _active_futures to exactly the cap
        with bridge._futures_lock:
            for _ in range(pressure_cap):
                bridge._active_futures.add(Future())

        params = {"prompt": "test", "tool": "coco"}
        bridge._handle_agent_call(params, request_id="req-42")

        assert len(sent_errors) == 1, "Expected exactly one error response"
        err = sent_errors[0]
        assert err["code"] == -32000, f"Expected -32000 backpressure code, got {err['code']}"
        assert err["request_id"] == "req-42"
        # Executor.submit must NOT have been called
        assert len(submit_calls) == 0, (
            "Executor.submit should not be called when backpressure is active"
        )
    finally:
        bridge.stop()


def test_backpressure_allows_when_below_cap(tmp_path):
    """Sanity check: below pressure cap, the agent call should be submitted
    normally (we stub the callback to avoid real work)."""
    max_concurrent = 4
    pressure_cap = max(2, max_concurrent * 2)  # = 8

    bridge = _make_bridge(tmp_path, max_concurrent=max_concurrent)
    try:
        _attach_mock_process(bridge)
        bridge._executor = MagicMock()
        mock_future = Future()
        bridge._executor.submit.return_value = mock_future

        # Add only (cap-1) futures so we are just under the threshold
        with bridge._futures_lock:
            for _ in range(pressure_cap - 1):
                bridge._active_futures.add(Future())

        sent_errors: list[dict] = []
        bridge._send_error_response = lambda req_id, code, message, structured=None: sent_errors.append(
            {"code": code}
        )

        params = {"prompt": "test", "tool": "coco"}
        bridge._handle_agent_call(params, request_id="req-under")

        assert sent_errors == [], f"Should NOT reject under pressure cap, got errors: {sent_errors}"
        bridge._executor.submit.assert_called_once()
    finally:
        bridge.stop()


# ---------------------------------------------------------------------------
# 9. abort_request sets per-call cancel_event (race loser abort)
# ---------------------------------------------------------------------------


def test_handle_abort_request_sets_per_call_cancel_event(tmp_path):
    """When _handle_abort_request is called for an in-flight request, the
    per-call cancel_event stored in _request_cancel_events must be set,
    allowing the ACP session's send_prompt loop to exit promptly."""
    bridge = _make_bridge(tmp_path)
    try:
        _attach_mock_process(bridge)
        bridge._executor = MagicMock()
        # Simulate an already-running future (cancel() returns False)
        mock_future = MagicMock(spec=Future)
        mock_future.cancel.return_value = False
        bridge._executor.submit.return_value = mock_future

        # Register a per-call cancel event as if _handle_agent_call had done it
        import threading
        call_cancel = threading.Event()
        with bridge._request_cancel_events_lock:
            bridge._request_cancel_events["req-abort-1"] = call_cancel
        with bridge._request_futures_lock:
            bridge._request_futures["req-abort-1"] = mock_future

        # Sanity: event is not set yet
        assert not call_cancel.is_set()

        # Act: simulate the JS runtime sending abort_request
        bridge._handle_abort_request("req-abort-1")

        # The per-call cancel event must be set
        assert call_cancel.is_set(), "per-call cancel_event must be set by abort_request"
        # The event must still be in the map (find-and-set, not pop) — the worker
        # thread's finally block handles cleanup for running futures.
        with bridge._request_cancel_events_lock:
            assert "req-abort-1" in bridge._request_cancel_events, (
                "cancel_event should remain in map after abort_request for "
                "already-running futures (worker finally block handles cleanup)"
            )
    finally:
        bridge.stop()


def test_handle_agent_call_creates_per_call_cancel_event(tmp_path):
    """_handle_agent_call must create a per-call cancel_event and store it in
    _request_cancel_events so that a later abort_request can interrupt it."""
    bridge = _make_bridge(tmp_path)
    try:
        _attach_mock_process(bridge)
        bridge._executor = MagicMock()
        mock_future = Future()
        bridge._executor.submit.return_value = mock_future

        params = {"prompt": "test", "tool": "coco", "label": "agent-x"}
        bridge._handle_agent_call(params, request_id="req-new-1")

        # Verify per-call cancel event was created and stored
        with bridge._request_cancel_events_lock:
            assert "req-new-1" in bridge._request_cancel_events
            event = bridge._request_cancel_events["req-new-1"]
            assert hasattr(event, "is_set")
            assert hasattr(event, "set")
            assert not event.is_set(), "newly created event must not be set"
    finally:
        bridge.stop()


def test_agent_aborted_notification_dispatches_callback(tmp_path):
    """When the JS runtime sends an 'agent_aborted' notification (e.g. from
    race() loser abort), the on_agent_aborted callback must be invoked with
    the label, reason, and request_id (as kwarg)."""
    import threading

    callback_calls = []
    callback_event = threading.Event()

    def on_agent_aborted(label: str, reason: str, **kwargs) -> None:
        callback_calls.append((label, reason, kwargs.get("request_id")))
        callback_event.set()

    bridge = _make_bridge(tmp_path, on_agent_aborted=on_agent_aborted)
    try:
        # Simulate incoming 'agent_aborted' notification via _dispatch_message
        msg = {
            "jsonrpc": "2.0",
            "method": "agent_aborted",
            "params": {"label": "loser-agent", "reason": "race loser", "request_id": 42},
        }
        bridge._dispatch_message(msg)

        # Wait briefly for callback (should be synchronous)
        assert callback_event.wait(timeout=1.0), "on_agent_aborted callback must be invoked"
        assert len(callback_calls) == 1
        label, reason, req_id = callback_calls[0]
        assert label == "loser-agent"
        assert reason == "race loser"
        assert req_id == 42
    finally:
        bridge.stop()


def test_agent_aborted_backward_compat_no_request_id(tmp_path):
    """agent_aborted notification without request_id must still work
    (backward compatibility with older JS runtimes)."""
    import threading

    callback_calls = []
    callback_event = threading.Event()

    def on_agent_aborted(label: str, reason: str) -> None:
        callback_calls.append((label, reason))
        callback_event.set()

    bridge = _make_bridge(tmp_path, on_agent_aborted=on_agent_aborted)
    try:
        msg = {
            "jsonrpc": "2.0",
            "method": "agent_aborted",
            "params": {"label": "agent-1", "reason": "aborted"},
        }
        bridge._dispatch_message(msg)

        assert callback_event.wait(timeout=1.0)
        assert len(callback_calls) == 1
        assert callback_calls[0] == ("agent-1", "aborted")
    finally:
        bridge.stop()


def test_agent_aborted_without_callback_is_noop(tmp_path):
    """If on_agent_aborted callback is not configured, the notification is
    silently ignored (no error, no crash)."""
    bridge = _make_bridge(tmp_path)  # no on_agent_aborted
    try:
        msg = {
            "jsonrpc": "2.0",
            "method": "agent_aborted",
            "params": {"label": "some-agent", "reason": "race"},
        }
        # Should not raise
        bridge._dispatch_message(msg)
    finally:
        bridge.stop()


def test_abort_request_for_unknown_request_is_safe(tmp_path):
    """Calling _handle_abort_request for a request_id that doesn't exist
    must be a no-op (no error, no crash)."""
    bridge = _make_bridge(tmp_path)
    try:
        _attach_mock_process(bridge)

        # Should not raise
        bridge._handle_abort_request("nonexistent-req-id")

        # Maps should remain empty
        with bridge._request_cancel_events_lock:
            assert len(bridge._request_cancel_events) == 0
        with bridge._request_futures_lock:
            assert len(bridge._request_futures) == 0
    finally:
        bridge.stop()


# ---------------------------------------------------------------------------
# 10. Early abort race condition (abort arrives before worker retrieves event)
# ---------------------------------------------------------------------------


def test_early_abort_still_sets_cancel_event(tmp_path):
    """If abort_request arrives BEFORE the worker thread retrieves the
    cancel_event from the dict (the old pop-and-set pattern would lose the
    signal), the worker must still see a set event and honour the cancellation.

    This is the core regression test for the race() loser cancel bug: the
    abort notification can arrive so fast that it pops the event before the
    worker ever reads it, causing the worker to fall back to the global
    cancel_event and ignore the per-call abort.
    """
    import threading

    bridge = _make_bridge(tmp_path)
    _attach_mock_process(bridge)
    bridge._executor = ThreadPoolExecutor(max_workers=1)

    worker_seen_event = threading.Event()
    worker_event_was_set = threading.Event()
    worker_started = threading.Event()
    slow_start_barrier = threading.Event()

    def slow_handler(params, *, cancel_event=None, **kwargs):
        """Simulate a worker that is slow to start checking cancel_event."""
        worker_started.set()
        # Wait for main thread to send abort BEFORE we check cancel_event
        slow_start_barrier.wait(timeout=5.0)
        # Now check what the worker sees
        if cancel_event is not None:
            worker_seen_event.set()
            if cancel_event.is_set():
                worker_event_was_set.set()
        from src.workflow_engine.models import AgentCallResult
        return AgentCallResult(output="done", tool=params.tool)

    bridge._on_agent_call = slow_handler

    req_id = "req-early-abort"
    bridge._handle_agent_call(
        {"prompt": "test", "tool": "coco", "label": "agent-early"},
        request_id=req_id,
    )

    # Wait for worker thread to start
    assert worker_started.wait(timeout=2.0), "Worker thread should start"

    # Now abort BEFORE the worker checks cancel_event
    bridge._handle_abort_request(req_id)

    # Release the worker to check the event
    slow_start_barrier.set()

    # Wait for worker to finish and report what it saw
    assert worker_seen_event.wait(timeout=5.0), (
        "Worker should have seen a cancel_event (not None)"
    )
    assert worker_event_was_set.wait(timeout=2.0), (
        "Worker should see cancel_event.is_set() == True even though abort "
        "arrived before the worker retrieved the event from the dict"
    )

    bridge.stop()


def test_abort_of_not_yet_started_future_cleans_up_cancel_event(tmp_path):
    """When abort_request cancels a future that hasn't started yet, the
    cancel_event entry must still be cleaned up (since _execute won't run
    and its finally block won't fire)."""
    import threading

    bridge = _make_bridge(tmp_path)
    _attach_mock_process(bridge)
    # Use a pool with 0 active workers so future stays pending
    bridge._executor = ThreadPoolExecutor(max_workers=1)
    # Block the only worker so our test future can't start
    block_evt = threading.Event()
    blocker = bridge._executor.submit(block_evt.wait, 30.0)

    try:
        req_id = "req-pending-future"
        bridge._handle_agent_call(
            {"prompt": "test", "tool": "coco", "label": "agent-pend"},
            request_id=req_id,
        )

        # Sanity: event is registered
        with bridge._request_cancel_events_lock:
            assert req_id in bridge._request_cancel_events

        # Abort before future starts
        bridge._handle_abort_request(req_id)

        # The cancel_event should be cleaned up (future was cancelled before
        # running, so _execute never runs and never cleans up)
        # Give it a brief moment
        import time as _t
        _t.sleep(0.1)

        with bridge._request_cancel_events_lock:
            assert req_id not in bridge._request_cancel_events, (
                "cancel_event should be cleaned up when abort cancels a "
                "not-yet-started future"
            )
    finally:
        block_evt.set()
        blocker.result(timeout=2.0)
        bridge.stop()


# ---------------------------------------------------------------------------
# 11. race() cancellation with real Node.js subprocess
# ---------------------------------------------------------------------------


RACE_CANCEL_SCRIPT = """\
export const meta = {
  name: 'race-cancel-test',
  description: 'Integration test for race() loser cancellation',
  phases: [
    { title: 'race', detail: 'Two contestants, fast wins, slow is cancelled' },
  ],
};

export default async function main() {
  const result = await race([
    { prompt: 'fast prompt', label: 'fast', tool: 'coco' },
    { prompt: 'slow prompt', label: 'slow', tool: 'coco' },
  ]);
  return result;
}
"""


@pytest.mark.skipif(
    not RuntimeBridge.check_node_available(),
    reason="Node.js not available or version too old",
)
def test_race_cancel_real_node_process(tmp_path):
    """race() with a real Node.js subprocess: the fast contestant wins and
    the slow contestant is properly cancelled via abort_request →
    per-call cancel_event → agent_aborted notification chain.

    Verification points:
    - Workflow completes well within 10s (doesn't wait for slow's 30s sleep)
    - slow's per-call cancel_event is set (abort_request reached Python)
    - on_agent_aborted callback is called with label='slow'
    - Final result is the fast contestant's return value
    """
    import threading

    from src.workflow_engine.models import AgentCallResult

    # Write the workflow script into tmp_path
    script_path = tmp_path / "race_cancel_test.js"
    script_path.write_text(RACE_CANCEL_SCRIPT, encoding="utf-8")

    # Project root — needed so that RuntimeBridge can find runtime.js via
    # RUNTIME_JS_PATH (which is relative to cwd).
    import os
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

    # State tracking
    fast_called = threading.Event()
    slow_called = threading.Event()
    slow_cancel_event = None  # will be set by the handler
    slow_cancel_was_set = threading.Event()
    aborted_labels: list[str] = []
    aborted_event = threading.Event()

    def on_agent_call(params, *, cancel_event=None, **_kwargs):
        label = params.label or ""
        if label == "fast":
            fast_called.set()
            # This integration test exercises cancellation of an in-flight
            # loser. Without this condition wait, the immediate winner can
            # legally cancel slow while its future is still queued, which is
            # the separate pending-future path covered above.
            assert slow_called.wait(timeout=3.0), (
                "slow agent should start before the in-flight cancel check"
            )
            # Fast agent returns immediately
            return AgentCallResult(
                output="fast-wins",
                tool=params.tool,
            )
        elif label == "slow":
            nonlocal slow_cancel_event
            slow_cancel_event = cancel_event
            slow_called.set()
            # Slow agent "blocks" for 30s — but we poll cancel_event
            # so we can verify it got set, and exit early if cancelled.
            # We use a tight poll loop so the test finishes fast once
            # cancel_event is set (which should happen within ~1s of
            # the fast agent returning).
            for _ in range(300):  # 30s total if never cancelled
                if cancel_event is not None and cancel_event.is_set():
                    slow_cancel_was_set.set()
                    return AgentCallResult(
                        output="slow-cancelled",
                        error="Cancelled",
                        tool=params.tool,
                    )
                time.sleep(0.1)
            # If we get here, cancellation didn't work
            return AgentCallResult(
                output="slow-done-too-late",
                tool=params.tool,
            )
        else:
            return AgentCallResult(
                output=f"unknown-agent:{label}",
                tool=params.tool,
            )

    def on_agent_aborted(label, reason, **_kwargs):
        aborted_labels.append(label)
        if label == "slow":
            aborted_event.set()

    bridge = RuntimeBridge(
        script_path=str(script_path),
        cwd=project_root,
        max_concurrent=2,
        on_agent_call=on_agent_call,
        on_agent_aborted=on_agent_aborted,
    )

    try:
        bridge.start()

        start = time.monotonic()
        result = bridge.run()
        elapsed = time.monotonic() - start

        # 1. Must finish well before the slow agent's 30s sleep
        assert elapsed < 10.0, (
            f"Workflow took {elapsed:.2f}s — it should have completed "
            f"in <10s because fast wins and slow is cancelled"
        )

        # 2. Both agents were called (race starts both contestants)
        assert fast_called.is_set(), "fast agent should have been called"
        assert slow_called.is_set(), "slow agent should have been called"

        # 3. slow's cancel_event must have been set by abort_request
        assert slow_cancel_event is not None, "slow should have a cancel_event"
        # Wait briefly — the agent_aborted notification may arrive before
        # the Python-side worker thread actually observes cancel_event.is_set()
        # due to thread scheduling. The slow handler polls every 0.1s.
        assert slow_cancel_was_set.wait(timeout=2.0), (
            "slow's per-call cancel_event should have been set by abort_request"
        )

        # 4. on_agent_aborted must have been called for 'slow'
        assert aborted_event.wait(timeout=2.0), (
            "on_agent_aborted callback should be invoked for the race loser"
        )
        assert "slow" in aborted_labels, (
            f"Expected 'slow' in aborted labels, got: {aborted_labels}"
        )

        # 5. Final result should be the fast agent's output
        import json as _json
        result_data = _json.loads(result) if result else None
        assert result_data == "fast-wins", (
            f"Expected result 'fast-wins', got: {result_data!r}"
        )

    finally:
        bridge.stop()
