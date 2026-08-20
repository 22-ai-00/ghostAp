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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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

        def read(self, _size=-1):
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


def test_start_uses_packaged_runtime_not_project_decoy(tmp_path, monkeypatch):
    """A user project must never replace the trusted Workflow runtime."""

    project = tmp_path / "user-project"
    decoy = project / "src" / "workflow_engine" / "runtime" / "runtime.js"
    decoy.parent.mkdir(parents=True)
    decoy.write_text("throw new Error('project decoy executed');", encoding="utf-8")
    bridge = _make_bridge(project)
    captured: dict[str, object] = {}

    class FakeStream:
        def __iter__(self):
            return iter(())

        def readline(self):
            return ""

        def read(self, _size=-1):
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

    def popen(command, **kwargs):
        captured["command"] = command
        captured["cwd"] = kwargs.get("cwd")
        return FakeProcess()

    monkeypatch.setattr(bridge_mod.shutil, "which", lambda _name: "/usr/bin/node")
    monkeypatch.setattr(bridge_mod.subprocess, "Popen", popen)
    monkeypatch.setattr(
        bridge,
        "_wait_for_notification",
        lambda method, timeout=30.0: {"jsonrpc": "2.0", "method": "ready"},
    )

    try:
        bridge.start()
    finally:
        bridge.stop()

    command = captured["command"]
    assert isinstance(command, list)
    runtime = Path(command[1])
    packaged = Path(bridge_mod.__file__).resolve().parent / "runtime" / "runtime.js"
    assert runtime.is_absolute()
    assert runtime == packaged
    assert runtime != decoy
    assert captured["cwd"] == str(project)


def test_start_fails_before_spawn_when_packaged_runtime_is_missing(
    tmp_path,
    monkeypatch,
):
    bridge = _make_bridge(tmp_path)
    popen = MagicMock()
    monkeypatch.setattr(bridge_mod, "RUNTIME_JS_PATH", str(tmp_path / "missing.js"))
    monkeypatch.setattr(bridge_mod.shutil, "which", lambda _name: "/usr/bin/node")
    monkeypatch.setattr(bridge_mod.subprocess, "Popen", popen)

    with pytest.raises(RuntimeError, match="packaged Workflow runtime is missing"):
        bridge.start()

    popen.assert_not_called()


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
    # No deadline means per-call timeout resolution is not workflow-capped.
    assert bridge._remaining_workflow_budget_s() is None


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


def test_start_after_stop_never_spawns_node(tmp_path, monkeypatch):
    """A pre-stopped child bridge must fail before subprocess creation."""
    bridge = _make_bridge(tmp_path)
    popen = MagicMock()
    monkeypatch.setattr(bridge_mod.shutil, "which", lambda _name: "/usr/bin/node")
    monkeypatch.setattr(bridge_mod.subprocess, "Popen", popen)

    bridge.stop()

    with pytest.raises(RuntimeError, match="cancelled before start"):
        bridge.start()

    popen.assert_not_called()


# ---------------------------------------------------------------------------
# 11. race() cancellation with real Node.js subprocess
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not RuntimeBridge.check_node_available(),
    reason="Node.js not available or version too old",
)
def test_packaged_runtime_executes_from_external_project_cwd(tmp_path):
    """The installed runtime must work when the user project is elsewhere."""

    project = tmp_path / "external-project"
    project.mkdir()
    script_path = project / "external_workflow.js"
    script_path.write_text(
        """\
export const meta = {
  name: 'external-cwd-test',
  description: 'Run the packaged Workflow runtime outside the GhostAP checkout',
  phases: [{ title: 'run', detail: 'Return a deterministic result' }],
};

export default async function main() {
  return 'external-cwd-ok';
}
""",
        encoding="utf-8",
    )
    bridge = RuntimeBridge(
        script_path=str(script_path),
        cwd=str(project),
        on_agent_call=lambda _params: AgentCallResult(output="unexpected"),
    )

    try:
        bridge.start()
        assert json.loads(bridge.run()) == "external-cwd-ok"
    finally:
        bridge.stop()


@pytest.mark.skipif(
    not RuntimeBridge.check_node_available(),
    reason="Node.js not available or version too old",
)
def test_generated_workflow_has_direct_node_and_host_environment_access(
    tmp_path, monkeypatch
):
    project = tmp_path / "direct-host-project"
    project.mkdir()
    output = project / "node-host-access.txt"
    helper = project / "host_helper.mjs"
    helper.write_text(
        """\
import { writeFileSync } from 'node:fs';
export function writeMarker(path, value) { writeFileSync(path, value); }
""",
        encoding="utf-8",
    )
    script_path = project / "direct_host_workflow.js"
    monkeypatch.setenv("GHOSTAP_WORKFLOW_HOST_MARKER", "host-visible")
    script_path.write_text(
        f"""\
import {{ writeMarker }} from './host_helper.mjs';

export const meta = {{
  name: 'direct-host-test',
  description: 'Use ordinary Node host capabilities',
  phases: [{{ title: 'run', detail: 'Write a host file' }}],
}};

export default async function main() {{
  console.log('generated workflow host log');
  writeMarker({json.dumps(str(output))}, process.env.GHOSTAP_WORKFLOW_HOST_MARKER);
  return process.env.GHOSTAP_WORKFLOW_HOST_MARKER;
}}
""",
        encoding="utf-8",
    )
    bridge = RuntimeBridge(
        script_path=str(script_path),
        cwd=str(project),
        on_agent_call=lambda _params: AgentCallResult(output="unexpected"),
    )

    try:
        bridge.start()
        assert json.loads(bridge.run()) == "host-visible"
    finally:
        bridge.stop()

    assert output.read_text(encoding="utf-8") == "host-visible"


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

    from src.workflow_engine.models import AgentCallResult

    # Write the workflow script into tmp_path
    script_path = tmp_path / "race_cancel_test.js"
    script_path.write_text(RACE_CANCEL_SCRIPT, encoding="utf-8")

    # Keep the user project as cwd; the trusted runtime is package-relative.
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


def _run_real_node_workflow(
    tmp_path,
    script,
    on_agent_call,
    *,
    max_concurrent=4,
    on_agent_aborted=None,
):
    import os

    script_path = tmp_path / f"runtime_regression_{time.time_ns()}.js"
    script_path.write_text(script, encoding="utf-8")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    bridge = RuntimeBridge(
        script_path=str(script_path),
        cwd=project_root,
        max_concurrent=max_concurrent,
        on_agent_call=on_agent_call,
        on_agent_aborted=on_agent_aborted,
    )
    try:
        bridge.start()
        raw = bridge.run()
        return json.loads(raw) if raw else None
    finally:
        bridge.stop()


@pytest.mark.skipif(
    not RuntimeBridge.check_node_available(),
    reason="Node.js not available or version too old",
)
def test_parallel_sibling_races_abort_only_their_own_losers(tmp_path):
    """A race winner must not cancel requests owned by a sibling race."""
    expected_labels = {"a-fast", "a-slow", "b-fast", "b-slow"}
    started_labels: set[str] = set()
    started_lock = threading.Lock()
    all_started = threading.Event()
    a_slow_cancelled = threading.Event()
    b_fast_returned = threading.Event()
    b_fast_cancelled_early = threading.Event()
    b_slow_cancelled = threading.Event()
    b_slow_cancelled_early = threading.Event()
    aborted_labels: list[str] = []

    def mark_started(label: str) -> None:
        with started_lock:
            started_labels.add(label)
            if started_labels == expected_labels:
                all_started.set()

    def wait_for_cancel(cancel_event, observed: threading.Event) -> AgentCallResult:
        for _ in range(600):
            if cancel_event is not None and cancel_event.is_set():
                observed.set()
                return AgentCallResult(error="cancelled", tool="coco")
            time.sleep(0.005)
        return AgentCallResult(output="unexpected-timeout", tool="coco")

    def on_agent_call(params, *, cancel_event=None, **_kwargs):
        label = params.label or ""
        mark_started(label)
        if label == "a-fast":
            all_started.wait(timeout=2.0)
            return AgentCallResult(output="a-wins", tool=params.tool)
        if label == "a-slow":
            return wait_for_cancel(cancel_event, a_slow_cancelled)
        if label == "b-fast":
            # Keep race B alive until race A has already cancelled its loser.
            # A global interceptor stack incorrectly cancels this request too.
            a_slow_cancelled.wait(timeout=2.0)
            time.sleep(0.05)
            if cancel_event is not None and cancel_event.is_set():
                b_fast_cancelled_early.set()
                return AgentCallResult(error="cancelled early", tool=params.tool)
            b_fast_returned.set()
            return AgentCallResult(output="b-wins", tool=params.tool)
        if label == "b-slow":
            result = wait_for_cancel(cancel_event, b_slow_cancelled)
            if not b_fast_returned.is_set():
                b_slow_cancelled_early.set()
            return result
        return AgentCallResult(error=f"unexpected label: {label}", tool=params.tool)

    def on_agent_aborted(label, _reason, **_kwargs):
        aborted_labels.append(label)

    result = _run_real_node_workflow(
        tmp_path,
        """
export const meta = { name: 'sibling-races', description: 'isolated races', phases: [{ title: 'race', detail: 'run' }] };
export default async function main() {
  return parallel([
    () => race([
      { prompt: 'A fast', label: 'a-fast', tool: 'coco' },
      { prompt: 'A slow', label: 'a-slow', tool: 'coco' },
    ]),
    () => race([
      { prompt: 'B fast', label: 'b-fast', tool: 'coco' },
      { prompt: 'B slow', label: 'b-slow', tool: 'coco' },
    ]),
  ]);
}
""",
        on_agent_call,
        max_concurrent=4,
        on_agent_aborted=on_agent_aborted,
    )

    assert result == ["a-wins", "b-wins"]
    assert a_slow_cancelled.wait(timeout=2.0)
    assert b_slow_cancelled.wait(timeout=2.0)
    assert not b_fast_cancelled_early.is_set()
    assert not b_slow_cancelled_early.is_set()
    assert sorted(aborted_labels) == ["a-slow", "b-slow"]


@pytest.mark.skipif(
    not RuntimeBridge.check_node_available(),
    reason="Node.js not available or version too old",
)
def test_pipeline_collector_inherits_nested_race_without_cancelling_sibling(tmp_path):
    """A pipeline abort reaches descendant requests but not a parallel sibling."""
    nested_one_cancelled = threading.Event()
    nested_two_cancelled = threading.Event()
    sibling_fast_returned = threading.Event()
    sibling_fast_cancelled_early = threading.Event()
    sibling_slow_cancelled = threading.Event()
    sibling_slow_cancelled_early = threading.Event()
    aborted_labels: list[str] = []

    def wait_for_cancel(cancel_event, observed: threading.Event) -> AgentCallResult:
        for _ in range(600):
            if cancel_event is not None and cancel_event.is_set():
                observed.set()
                return AgentCallResult(error="cancelled", tool="coco")
            time.sleep(0.005)
        return AgentCallResult(output="unexpected-timeout", tool="coco")

    def on_agent_call(params, *, cancel_event=None, **_kwargs):
        label = params.label or ""
        if label == "nested-one":
            return wait_for_cancel(cancel_event, nested_one_cancelled)
        if label == "nested-two":
            return wait_for_cancel(cancel_event, nested_two_cancelled)
        if label == "sibling-fast":
            # The pipeline's failing item should first abort both requests in
            # its nested race, while this lexical sibling remains untouched.
            nested_one_cancelled.wait(timeout=2.0)
            nested_two_cancelled.wait(timeout=2.0)
            if cancel_event is not None and cancel_event.is_set():
                sibling_fast_cancelled_early.set()
                return AgentCallResult(error="cancelled early", tool=params.tool)
            sibling_fast_returned.set()
            return AgentCallResult(output="sibling-wins", tool=params.tool)
        if label == "sibling-slow":
            result = wait_for_cancel(cancel_event, sibling_slow_cancelled)
            if not sibling_fast_returned.is_set():
                sibling_slow_cancelled_early.set()
            return result
        return AgentCallResult(error=f"unexpected label: {label}", tool=params.tool)

    def on_agent_aborted(label, _reason, **_kwargs):
        aborted_labels.append(label)

    result = _run_real_node_workflow(
        tmp_path,
        """
export const meta = { name: 'nested-collectors', description: 'nested collector lineage', phases: [{ title: 'pipeline', detail: 'run' }] };
export default async function main() {
  const [pipelineResult, siblingResult] = await parallel([
    async () => {
      try {
        return await pipeline(['nested', 'fail'], async (item) => {
          if (item === 'fail') {
            await new Promise(resolve => setTimeout(resolve, 100));
            throw new Error('pipeline boom');
          }
          return race([
            { prompt: 'nested one', label: 'nested-one', tool: 'coco' },
            { prompt: 'nested two', label: 'nested-two', tool: 'coco' },
          ]);
        });
      } catch (error) {
        return { error: error.message };
      }
    },
    () => race([
      { prompt: 'sibling fast', label: 'sibling-fast', tool: 'coco' },
      { prompt: 'sibling slow', label: 'sibling-slow', tool: 'coco' },
    ]),
  ]);
  return { pipelineResult, siblingResult };
}
""",
        on_agent_call,
        max_concurrent=4,
        on_agent_aborted=on_agent_aborted,
    )

    assert result["pipelineResult"] == {"error": "pipeline boom"}
    assert result["siblingResult"] == "sibling-wins"
    assert nested_one_cancelled.wait(timeout=2.0)
    assert nested_two_cancelled.wait(timeout=2.0)
    assert sibling_slow_cancelled.wait(timeout=2.0)
    assert not sibling_fast_cancelled_early.is_set()
    assert not sibling_slow_cancelled_early.is_set()
    assert sorted(aborted_labels) == ["nested-one", "nested-two", "sibling-slow"]


@pytest.mark.skipif(
    not RuntimeBridge.check_node_available(),
    reason="Node.js not available or version too old",
)
def test_verify_returns_closed_evidence_ledger(tmp_path):
    review = {
        "issues": [
            {
                "severity": "major",
                "description": "Missing rollback handling",
                "evidence": "The result has no rollback field",
            }
        ],
        "approve": False,
    }

    def on_agent_call(params, **_kwargs):
        return AgentCallResult(
            output=json.dumps(review),
            parsed=review,
            tool=params.tool,
        )

    result = _run_real_node_workflow(
        tmp_path,
        """
export const meta = { name: 'verify-ledger', description: 'verify evidence', phases: [{ title: 'verify', detail: 'review' }] };
export default async function main() {
  return verify('candidate', {
    maxRounds: 1,
    timeout: 1,
    verifiers: [{ tool: 'coco', role: 'security-reviewer' }],
  });
}
""",
        on_agent_call,
    )

    assert result["accepted"] is False
    assert result["reviews"] == [
        {
            "round": 1,
            "verifier": "security-reviewer",
            "approve": False,
            "issues": review["issues"],
        }
    ]
    assert result["evidence"][0]["evidence"] == "The result has no rollback field"


@pytest.mark.skipif(
    not RuntimeBridge.check_node_available(),
    reason="Node.js not available or version too old",
)
def test_verify_rejects_non_closed_or_empty_evidence(tmp_path):
    invalid_review = {
        "issues": [
            {
                "severity": "major",
                "description": "Missing rollback handling",
                "evidence": "",
                "unexpected": "must be rejected",
            }
        ],
        "approve": False,
    }

    def on_agent_call(params, **_kwargs):
        return AgentCallResult(
            output=json.dumps(invalid_review),
            parsed=invalid_review,
            tool=params.tool,
        )

    result = _run_real_node_workflow(
        tmp_path,
        """
export const meta = { name: 'verify-closed', description: 'verify schema', phases: [{ title: 'verify', detail: 'review' }] };
export default async function main() {
  return verify('candidate', {
    maxRounds: 1,
    timeout: 1,
    verifiers: [{ tool: 'coco', role: 'strict-reviewer' }],
  });
}
""",
        on_agent_call,
    )

    assert result["accepted"] is False
    assert result["reviews"] == []
    assert result["evidence"] == []
    assert result["failures"][0]["stage"] == "verify_schema"


@pytest.mark.skipif(
    not RuntimeBridge.check_node_available(),
    reason="Node.js not available or version too old",
)
def test_race_clamps_timeout_to_node_safe_timer(tmp_path, monkeypatch):
    monkeypatch.setattr(
        bridge_mod,
        "_settings_int",
        lambda field, fallback: 0 if field == "workflow_total_timeout_s" else fallback,
    )

    def on_agent_call(params, **_kwargs):
        time.sleep(0.05)
        return AgentCallResult(output="winner", tool=params.tool)

    result = _run_real_node_workflow(
        tmp_path,
        """
export const meta = { name: 'safe-timer', description: 'large timeout', phases: [{ title: 'race', detail: 'run' }] };
export default async function main() {
  return race([{ prompt: 'finish', label: 'winner', tool: 'coco' }], { timeout: 2592000 });
}
""",
        on_agent_call,
    )

    assert result == "winner"


@pytest.mark.skipif(
    not RuntimeBridge.check_node_available(),
    reason="Node.js not available or version too old",
)
def test_race_timeout_merges_failures_before_deadline(tmp_path):
    def on_agent_call(params, *, cancel_event=None, **_kwargs):
        if params.label == "early-failure":
            return AgentCallResult(error="early boom", tool=params.tool)
        for _ in range(100):
            if cancel_event is not None and cancel_event.is_set():
                return AgentCallResult(error="cancelled", tool=params.tool)
            time.sleep(0.01)
        return AgentCallResult(output="too late", tool=params.tool)

    result = _run_real_node_workflow(
        tmp_path,
        """
export const meta = { name: 'failure-ledger', description: 'race failures', phases: [{ title: 'race', detail: 'run' }] };
export default async function main() {
  return race([
    { prompt: 'fail now', label: 'early-failure', tool: 'coco' },
    { prompt: 'wait', label: 'slow', tool: 'coco' },
  ], { timeout: 0.15 });
}
""",
        on_agent_call,
        max_concurrent=2,
    )

    assert result["timeout"] is True
    assert any(
        failure.get("index") == 0 and failure.get("error") == "early boom"
        for failure in result["failures"]
    )


@pytest.mark.skipif(
    not RuntimeBridge.check_node_available(),
    reason="Node.js not available or version too old",
)
def test_race_timeout_blocks_delayed_function_dispatch(tmp_path):
    called_labels: list[str] = []

    def on_agent_call(params, *, cancel_event=None, **_kwargs):
        called_labels.append(params.label or "")
        if params.label == "blocking":
            for _ in range(100):
                if cancel_event is not None and cancel_event.is_set():
                    return AgentCallResult(error="cancelled", tool=params.tool)
                time.sleep(0.01)
        return AgentCallResult(output="late", tool=params.tool)

    result = _run_real_node_workflow(
        tmp_path,
        """
export const meta = { name: 'delayed-race', description: 'cancel delayed calls', phases: [{ title: 'race', detail: 'run' }] };
export default async function main() {
  const result = await race([
    async () => {
      await new Promise(resolve => setTimeout(resolve, 80));
      return agent({ prompt: 'must not dispatch', label: 'late', tool: 'coco' });
    },
    { prompt: 'block', label: 'blocking', tool: 'coco' },
  ], { timeout: 0.03 });
  await new Promise(resolve => setTimeout(resolve, 150));
  return result;
}
""",
        on_agent_call,
        max_concurrent=2,
    )

    assert result["timeout"] is True
    assert "blocking" in called_labels
    assert "late" not in called_labels


def test_stderr_helpers_bound_single_line_and_total_bytes():
    line, line_truncated = bridge_mod._clip_utf8("界" * 5000, bridge_mod.STDERR_LINE_MAX_BYTES)
    assert line_truncated is True
    assert len(line.encode("utf-8")) <= bridge_mod.STDERR_LINE_MAX_BYTES
    assert bridge_mod.STDERR_TRUNCATION_MARKER in line

    bounded, total_truncated = bridge_mod._bound_stderr_text(
        "\n".join("x" * 4000 for _ in range(30))
    )
    assert total_truncated is True
    assert len(bounded.encode("utf-8")) <= bridge_mod.STDERR_TOTAL_MAX_BYTES
    assert bridge_mod.STDERR_TRUNCATION_MARKER in bounded


def test_stderr_drain_reads_only_bounded_remainder(tmp_path):
    bridge = _make_bridge(tmp_path)
    try:
        proc = _attach_mock_process(bridge)
        proc.stderr.read.return_value = "z" * (bridge_mod.STDERR_TOTAL_MAX_BYTES + 1)

        result = bridge._drain_stderr()

        proc.stderr.read.assert_called_once_with(bridge_mod.STDERR_TOTAL_MAX_BYTES + 1)
        assert len(result.encode("utf-8")) <= bridge_mod.STDERR_TOTAL_MAX_BYTES
        assert bridge_mod.STDERR_TRUNCATION_MARKER in result
    finally:
        bridge.stop()
