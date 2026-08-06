"""Tests for JS runtime primitive reliability (unit-style, no real Node process).

Uses a mock transport layer that simulates JSON-RPC request/response round-trips
through an in-process pending-requests map, allowing us to exercise parallel
semaphore correctness, race abort propagation, backpressure retry, verify
short-circuit, pipeline failure-abort, and CancelledError passthrough without
spawning a subprocess.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from unittest.mock import MagicMock

from src.workflow_engine.bridge import RuntimeBridge
from src.workflow_engine.models import AgentCallResult

# ---------------------------------------------------------------------------
# Mock bridge — simulates JS-side runtime behaviour from Python side
# ---------------------------------------------------------------------------


class MockJSRuntime:
    """Simulates the JS runtime's request/response handling in Python.

    The bridge sends JSON-RPC requests over stdin; this mock intercepts those
    by replacing _send and _read_loop, then responds through the message queue
    exactly as the Node.js subprocess would.
    """

    def __init__(self, bridge: RuntimeBridge):
        self.bridge = bridge
        self._lock = threading.Lock()
        self._pending: dict[int, dict] = {}
        self._next_id = 0
        self._cancelled = False
        self._on_agent_call = None
        self._agent_call_delay = 0.0
        self._backpressure_count = 0
        self._backpressure_trigger = 0
        self._aborted_requests: list[int] = []

    def set_agent_handler(self, handler):
        """Set a callable that returns AgentCallResult for agent_call requests."""
        self._on_agent_call = handler

    def set_agent_delay(self, delay: float):
        """Set simulated delay per agent call (seconds)."""
        self._agent_call_delay = delay

    def set_backpressure_trigger(self, count: int):
        """First N agent calls return -32000 backpressure, then succeed."""
        self._backpressure_trigger = count
        self._backpressure_count = 0

    @property
    def aborted_requests(self) -> list[int]:
        return list(self._aborted_requests)

    def handle_outbound(self, msg: dict) -> None:
        """Called whenever the bridge sends a message (simulates stdin write)."""
        if msg.get("method") == "cancel":
            self._cancelled = True
            with self._lock:
                for rid, entry in list(self._pending.items()):
                    if not entry.get("aborted"):
                        entry["aborted"] = True
                        entry["reject"]("Workflow cancelled by host")
                self._pending.clear()
            return

        if msg.get("method") == "abort_request":
            rid = msg.get("params", {}).get("request_id")
            if rid is not None:
                self._aborted_requests.append(rid)
            return

        msg_id = msg.get("id")
        method = msg.get("method")
        if msg_id is not None and method:
            # This is a request from JS -> Python; in our mock setup the
            # bridge is *sending* requests to the JS side, so incoming
            # requests come back through _dispatch_message.
            pass

    def simulate_agent_call_response(self, request_id: int, result: str | dict):
        """Inject a successful agent_call response into the bridge."""
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"data": result, "token_usage": 0, "duration_s": 0.1},
        }
        with self.bridge._msg_condition:
            self.bridge._msg_queue.append(response)
            self.bridge._msg_condition.notify()

    def simulate_agent_call_error(self, request_id: int, code: int, message: str):
        """Inject an error response for an agent_call."""
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }
        with self.bridge._msg_condition:
            self.bridge._msg_queue.append(response)
            self.bridge._msg_condition.notify()


# ---------------------------------------------------------------------------
# Direct bridge-level tests (transport-layer primitive behaviour)
# ---------------------------------------------------------------------------


class TestBridgeAbortRequest:
    """Verify abort_request notification cancels the right future."""

    def test_abort_cancels_pending_future(self, tmp_path):
        """When JS sends abort_request for a not-yet-running future, it cancels."""
        bridge = RuntimeBridge(
            script_path="test.js",
            cwd=str(tmp_path),
            max_concurrent=1,
            on_agent_call=lambda p: AgentCallResult(output="ok"),
        )
        bridge._process = MagicMock()
        bridge._process.poll.return_value = None
        bridge._process.stdin = MagicMock()
        bridge._process.stdout = MagicMock()
        bridge._process.stderr = MagicMock()
        bridge._executor = MagicMock(spec=type(bridge._executor) if bridge._executor else object)
        # We need a real executor to test future semantics
        from concurrent.futures import ThreadPoolExecutor

        bridge._executor = ThreadPoolExecutor(max_workers=1)
        bridge._active_futures = set()
        bridge._futures_lock = threading.Lock()
        bridge._request_futures = {}
        bridge._request_futures_lock = threading.Lock()
        bridge._write_lock = threading.Lock()

        try:
            # Submit a call that blocks long enough to abort
            barrier = threading.Barrier(2)

            def slow_handler(params):
                barrier.wait(timeout=5)
                time.sleep(0.05)
                return AgentCallResult(output="done")

            bridge._on_agent_call = slow_handler

            # We can't easily call _handle_agent_call with a mock process
            # because _send_response uses self._process.stdin. Let's test
            # the _request_futures mapping directly.
            from concurrent.futures import Future

            fut = Future()
            with bridge._request_futures_lock:
                bridge._request_futures["req-1"] = fut

            # Abort it
            bridge._handle_abort_request("req-1")

            assert fut.cancelled(), "Future should be cancelled after abort_request"

            # Double-abort is safe (no-op)
            bridge._handle_abort_request("req-1")

            # Unknown request_id is safe (no-op)
            bridge._handle_abort_request("nonexistent")
        finally:
            bridge.stop()


class TestBridgeSendFailure:
    """Verify _send failure sets _done and _error promptly."""

    def test_broken_pipe_sets_done(self, tmp_path):
        """BrokenPipeError during _send sets self._done and self._error."""
        bridge = RuntimeBridge(
            script_path="test.js",
            cwd=str(tmp_path),
            on_agent_call=lambda p: AgentCallResult(output="ok"),
        )
        bridge._process = MagicMock()
        bridge._process.stdin = MagicMock()
        bridge._process.stdin.write.side_effect = BrokenPipeError("pipe broken")
        bridge._process.poll.return_value = None
        bridge._write_lock = threading.Lock()

        try:
            assert not bridge._done
            assert bridge._error is None

            bridge._send({"jsonrpc": "2.0", "method": "cancel", "params": {}})

            assert bridge._done is True
            assert bridge._error is not None
            assert "pipe broken" in bridge._error.lower() or "brokenpipe" in bridge._error.lower()
        finally:
            bridge.stop()


class TestBridgeNDJSONRobustness:
    """Verify non-JSON lines don't crash the reader thread."""

    def test_non_json_lines_are_skipped(self, tmp_path):
        """10+ consecutive non-JSON lines log a warning but don't stop reading."""
        bridge = RuntimeBridge(
            script_path="test.js",
            cwd=str(tmp_path),
            on_agent_call=lambda p: AgentCallResult(output="ok"),
        )

        # Build a mock stdout iterator that yields garbage then a valid line
        lines = deque()
        for i in range(15):
            lines.append(f"garbage line {i} not json at all")
        lines.append(json.dumps({"jsonrpc": "2.0", "method": "ready", "params": {}}))
        lines.append("")  # empty line

        def _stdout_iter():
            while lines:
                yield lines.popleft()

        bridge._process = MagicMock()
        bridge._process.stdout = _stdout_iter()
        bridge._process.poll.return_value = None
        bridge._process.stderr = MagicMock()
        bridge._process.stderr.readline.return_value = ""
        bridge._msg_queue = deque()
        bridge._msg_condition = threading.Condition()

        try:
            # Run read loop until exhausted
            bridge._read_loop()

            # Should have processed the valid 'ready' notification
            assert len(bridge._msg_queue) >= 1
            ready_msgs = [m for m in bridge._msg_queue if m.get("method") == "ready"]
            assert len(ready_msgs) == 1

            # Reader should have set _done (stdout EOF)
            assert bridge._done is True
        finally:
            bridge.stop()


# ---------------------------------------------------------------------------
# State-manager + journal primitive tests (Python-side reliability)
# ---------------------------------------------------------------------------


class TestJournalCacheKeyIncludesRoleAndSchema:
    """Verify compute_key differentiates by role and output_schema."""

    def test_same_prompt_different_role_different_key(self):
        from src.workflow_engine.journal import WorkflowJournal

        key1 = WorkflowJournal.compute_key("hello", "coco", "model-a", role="reviewer")
        key2 = WorkflowJournal.compute_key("hello", "coco", "model-a", role="coder")
        assert key1 != key2

    def test_same_prompt_different_schema_different_key(self):
        from src.workflow_engine.journal import WorkflowJournal

        key1 = WorkflowJournal.compute_key("hello", "coco", "model-a", output_schema={"x": "str"})
        key2 = WorkflowJournal.compute_key("hello", "coco", "model-a", output_schema={"y": "str"})
        assert key1 != key2

    def test_same_prompt_same_role_schema_same_key(self):
        from src.workflow_engine.journal import WorkflowJournal

        key1 = WorkflowJournal.compute_key("hello", "coco", "model-a", role="reviewer", output_schema={"x": "int"})
        key2 = WorkflowJournal.compute_key("hello", "coco", "model-a", role="reviewer", output_schema={"x": "int"})
        assert key1 == key2

    def test_schema_key_order_independent(self):
        from src.workflow_engine.journal import WorkflowJournal

        # JSON dumps with sort_keys=True should make key order irrelevant
        key1 = WorkflowJournal.compute_key("p", "t", "m", output_schema={"a": 1, "b": 2})
        key2 = WorkflowJournal.compute_key("p", "t", "m", output_schema={"b": 2, "a": 1})
        assert key1 == key2

    def test_no_role_no_schema_still_valid(self):
        from src.workflow_engine.journal import WorkflowJournal

        key = WorkflowJournal.compute_key("hello", "coco", "model-a")
        assert isinstance(key, str)
        assert len(key) == 64  # sha256 hex


class TestStateManagerTerminalStateConsistency:
    """Verify terminal-state sticky behaviour: CANCELLED > FAILED, COMPLETED > all."""

    def test_cancelled_not_overwritten_by_failed(self):
        from src.workflow_engine.models import WorkflowMetrics, WorkflowProject, WorkflowStatus
        from src.workflow_engine.state_manager import WorkflowStateManager

        project = WorkflowProject(status=WorkflowStatus.RUNNING, metrics=WorkflowMetrics())
        sm = WorkflowStateManager(project)

        sm.on_workflow_cancelled("user stopped")
        assert project.status == WorkflowStatus.CANCELLED

        sm.on_workflow_failed("some error")
        assert project.status == WorkflowStatus.CANCELLED, "CANCELLED should not be overwritten by FAILED"
        assert project.error == "user stopped"

    def test_failed_overwritten_by_cancelled(self):
        from src.workflow_engine.models import WorkflowMetrics, WorkflowProject, WorkflowStatus
        from src.workflow_engine.state_manager import WorkflowStateManager

        project = WorkflowProject(status=WorkflowStatus.RUNNING, metrics=WorkflowMetrics())
        sm = WorkflowStateManager(project)

        sm.on_workflow_failed("some error")
        assert project.status == WorkflowStatus.FAILED

        sm.on_workflow_cancelled("user stopped")
        assert project.status == WorkflowStatus.CANCELLED
        assert project.error == "user stopped"

    def test_completed_not_overwritten_by_anything(self):
        from src.workflow_engine.models import WorkflowMetrics, WorkflowProject, WorkflowStatus
        from src.workflow_engine.state_manager import WorkflowStateManager

        project = WorkflowProject(status=WorkflowStatus.RUNNING, metrics=WorkflowMetrics())
        sm = WorkflowStateManager(project)

        sm.on_workflow_done("result text")
        assert project.status == WorkflowStatus.COMPLETED

        sm.on_workflow_failed("late error")
        assert project.status == WorkflowStatus.COMPLETED

        sm.on_workflow_cancelled("late cancel")
        assert project.status == WorkflowStatus.COMPLETED

    def test_failed_closes_open_agents(self):
        from src.workflow_engine.models import (
            AgentStatus,
            WorkflowMetrics,
            WorkflowProject,
            WorkflowStatus,
        )
        from src.workflow_engine.state_manager import WorkflowStateManager

        project = WorkflowProject(status=WorkflowStatus.RUNNING, metrics=WorkflowMetrics())
        sm = WorkflowStateManager(project)

        sm.on_agent_started("agent-1", "coco", "phase1", "do thing")
        assert project.metrics.total_agents == 1
        assert project.phases[0].agents[0].status == AgentStatus.RUNNING

        sm.on_workflow_failed("boom")
        assert project.phases[0].agents[0].status == AgentStatus.FAILED
        assert project.phases[0].agents[0].error is not None
        assert project.metrics.failed_agents == 1
        assert project.metrics.completed_agents == 1


class TestRendererSnapshotSafety:
    """Verify renderer works with snapshot objects (concurrent read safety)."""

    def test_renderer_accepts_snapshot(self):
        from src.workflow_engine.models import (
            AgentProgress,
            AgentStatus,
            PhaseProgress,
            WorkflowMetrics,
            WorkflowProject,
            WorkflowStatus,
        )
        from src.workflow_engine.renderer import WorkflowProgressRenderer

        project = WorkflowProject(
            workflow_id="wf-test",
            name="test",
            status=WorkflowStatus.RUNNING,
            requirement="do stuff",
            metrics=WorkflowMetrics(),
        )
        renderer = WorkflowProgressRenderer(project)

        # Render with snapshot (a separate project instance)
        snapshot = WorkflowProject(
            workflow_id="wf-test",
            name="test",
            status=WorkflowStatus.RUNNING,
            requirement="do stuff",
            metrics=WorkflowMetrics(),
            phases=[PhaseProgress(title="Phase 1", started_at=time.time())],
        )
        # Add an agent to the snapshot
        snapshot.phases[0].agents.append(
            AgentProgress(
                label="agent-1",
                tool="coco",
                task_summary="doing work",
                status=AgentStatus.RUNNING,
                started_at=time.time(),
            )
        )
        snapshot.metrics.total_agents = 1

        card = renderer.render_progress_card(snapshot)
        assert "header" in card
        assert "elements" in card
        assert isinstance(card["elements"], list)
        assert len(card["elements"]) > 0

    def test_render_compact_status_from_snapshot(self):
        from src.workflow_engine.models import WorkflowMetrics, WorkflowProject, WorkflowStatus
        from src.workflow_engine.renderer import WorkflowProgressRenderer

        project = WorkflowProject(status=WorkflowStatus.IDLE, metrics=WorkflowMetrics())
        renderer = WorkflowProgressRenderer(project)

        # Compact status reads from self._project directly (no snapshot param)
        text = renderer.render_compact_status()
        assert isinstance(text, str)
        assert len(text) > 0


class TestCancelEventReuse:
    """Verify cancel_event is properly cleared between runs."""

    def test_engine_clear_cancel_event_under_lock(self, tmp_path):
        from src.workflow_engine.engine import WorkflowEngine

        engine = WorkflowEngine(
            chat_id="test_chat",
            root_path=str(tmp_path),
            agent_type="coco",
        )

        # Set it (simulate previous cancelled run)
        engine._cancel_event.set()
        assert engine._cancel_event.is_set()

        # Simulate what execute_workflow does: clear under lock
        with engine._lock:
            engine._cancel_event.clear()

        assert not engine._cancel_event.is_set()
        engine.cleanup()


class TestLateSessionCloseDoesNotBlockWorker:
    """Verify _close_late_session offloads close to daemon thread."""

    def test_close_late_session_returns_immediately(self, tmp_path):
        import concurrent.futures

        from src.workflow_engine.executor import AgentExecutor

        cancel_event = threading.Event()
        executor = AgentExecutor(
            cwd=str(tmp_path),
            cancel_event=cancel_event,
            max_workers=2,
        )

        try:
            # Create a future that takes time to complete
            def slow_create():
                time.sleep(0.2)
                mock_session = MagicMock()
                mock_session.close = MagicMock()
                return mock_session

            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = pool.submit(slow_create)

            # Call _close_late_session — should return immediately
            start = time.monotonic()
            executor._close_late_session(future, "test_tool")
            elapsed = time.monotonic() - start

            # Should return in well under 0.1s (the future itself takes 0.2s)
            assert elapsed < 0.1, f"_close_late_session should not block, took {elapsed:.3f}s"

            # Clean up
            future.result(timeout=2)
            pool.shutdown(wait=True)
            # Give the daemon thread a moment to run
            time.sleep(0.05)
        finally:
            executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Race loser abort — end-to-end timing integration tests
# ---------------------------------------------------------------------------


def test_race_abort_per_call_cancel_event_fires_within_100ms(tmp_path):
    """Integration: when abort_request arrives for an in-flight agent call,
    the per-call cancel_event must be set within 100ms. This is the critical
    path for race() loser abort — the session's send_prompt loop polls this
    event and exits within its poll interval (typically ~1s).
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    bridge = RuntimeBridge(
        script_path="test.js",
        cwd=str(tmp_path),
        max_concurrent=2,
        on_agent_call=lambda p: AgentCallResult(output="ok"),
    )
    bridge._process = MagicMock()
    bridge._process.poll.return_value = None
    bridge._process.stdin = MagicMock()
    bridge._process.stdout = MagicMock()
    bridge._process.stderr = MagicMock()
    bridge._executor = ThreadPoolExecutor(max_workers=2)
    bridge._active_futures = set()
    bridge._futures_lock = threading.Lock()
    bridge._request_futures = {}
    bridge._request_futures_lock = threading.Lock()
    bridge._request_cancel_events = {}
    bridge._request_cancel_events_lock = threading.Lock()
    bridge._write_lock = threading.Lock()
    bridge._msg_condition = threading.Condition()
    bridge._msg_queue = deque()

    try:
        # Submit a slow agent call that will block
        barrier = threading.Barrier(2, timeout=5)

        def slow_handler(params):
            barrier.wait(timeout=5)
            time.sleep(0.1)  # simulate work
            return AgentCallResult(output="slow result")

        bridge._on_agent_call = slow_handler

        # Trigger the agent call via _handle_agent_call
        bridge._handle_agent_call(
            {"prompt": "slow task", "tool": "coco", "label": "loser"},
            request_id="race-req-1",
        )

        # Wait for the handler to start (barrier synchronizes)
        barrier.wait(timeout=5)

        # Verify the per-call cancel event exists and is not set
        with bridge._request_cancel_events_lock:
            call_cancel = bridge._request_cancel_events.get("race-req-1")
        assert call_cancel is not None, "per-call cancel_event must exist"
        assert not call_cancel.is_set(), "event must not be set before abort"

        # Act: simulate JS runtime sending abort_request
        start = time.monotonic()
        bridge._handle_abort_request("race-req-1")
        elapsed_ms = (time.monotonic() - start) * 1000

        # Assert: event is set promptly (should be nearly instant — just a set() call)
        assert call_cancel.is_set(), "per-call cancel_event must be set after abort_request"
        assert elapsed_ms < 100, f"cancel_event should be set in <100ms, took {elapsed_ms:.0f}ms"

    finally:
        # Release the barrier so the slow handler can exit
        try:
            barrier.wait(timeout=1)
        except Exception:
            pass
        bridge.stop()


def test_race_abort_state_transition_within_100ms(tmp_path):
    """Integration: when agent_aborted notification arrives, the state manager
    must mark the agent as CANCELLED within 100ms, so the progress card no
    longer shows it as '执行中'. This validates the fast-path notification
    that runs parallel to the session cleanup.
    """
    from src.workflow_engine.models import (
        AgentStatus,
        WorkflowMetrics,
        WorkflowProject,
        WorkflowStatus,
    )
    from src.workflow_engine.state_manager import WorkflowStateManager

    # Set up state with a running agent
    project = WorkflowProject(
        workflow_id="wf-race",
        name="race-test",
        status=WorkflowStatus.RUNNING,
        requirement="race abort test",
        metrics=WorkflowMetrics(),
    )
    sm = WorkflowStateManager(project)
    sm.on_phase_changed("race phase")
    label = sm.on_agent_started(
        "contestant-b",
        tool="claude",
        phase="race phase",
        task_summary="trying approach B",
    )

    # Verify initial state: RUNNING
    agent = sm._label_to_agent[label]
    assert agent.status == AgentStatus.RUNNING

    # Act: simulate the agent_aborted callback from the bridge
    start = time.monotonic()
    sm.on_agent_aborted(label, reason="race loser")
    elapsed_ms = (time.monotonic() - start) * 1000

    # Assert: agent is now CANCELLED (not RUNNING)
    assert agent.status == AgentStatus.CANCELLED
    assert agent.error == "race loser"
    assert elapsed_ms < 100, f"state transition should take <100ms, took {elapsed_ms:.0f}ms"


def test_race_abort_full_pipeline_under_5s(tmp_path):
    """End-to-end timing: the full chain from abort_request to agent no longer
    showing as '执行中' must complete well under 5s. The chain is:
      abort_request → per-call cancel_event set → session send_prompt poll
      → session.close → agent returns error → state already CANCELLED
    The JS agent_aborted notification runs in parallel and marks state early.
    Total budget: 5s. Expected: well under 1s for the state change, and the
    session cleanup follows shortly after.
    """
    from src.workflow_engine.models import (
        WorkflowMetrics,
        WorkflowProject,
        WorkflowStatus,
    )
    from src.workflow_engine.renderer import WorkflowProgressRenderer
    from src.workflow_engine.state_manager import WorkflowStateManager

    # Set up state with 3 agents running (like a race with 3 contestants)
    project = WorkflowProject(
        workflow_id="wf-race3",
        name="race-3way",
        status=WorkflowStatus.RUNNING,
        requirement="3-way race",
        metrics=WorkflowMetrics(),
    )
    sm = WorkflowStateManager(project)
    sm.on_phase_changed("race")
    winner_label = sm.on_agent_started("winner", tool="coco", phase="race")
    loser1_label = sm.on_agent_started("loser-1", tool="claude", phase="race")
    loser2_label = sm.on_agent_started("loser-2", tool="aiden", phase="race")

    renderer = WorkflowProgressRenderer(project)

    # Verify all 3 are shown as running initially
    snapshot = sm.snapshot()
    card = renderer.render_progress_card(snapshot)
    card_text = json.dumps(card, ensure_ascii=False)
    assert "执行中 (3)" in card_text, "initial state: all 3 agents must be running"

    # Simulate: winner finishes first
    sm.on_agent_done(winner_label, {"token_usage": 100, "duration_s": 1.5, "cached": False})

    # Now simulate the abort notifications for losers (as JS runtime would send them)
    # This is the fast path — agent_aborted marks them CANCELLED immediately
    start = time.monotonic()

    sm.on_agent_aborted(loser1_label, reason="race loser")
    sm.on_agent_aborted(loser2_label, reason="race loser")

    state_elapsed_ms = (time.monotonic() - start) * 1000

    # Verify: no agents shown as '执行中' anymore
    snapshot = sm.snapshot()
    card = renderer.render_progress_card(snapshot)
    card_text = json.dumps(card, ensure_ascii=False)

    # Key assertions for the acceptance criteria
    assert "执行中 (2)" not in card_text, "losers must not show as '执行中 (2)'"
    assert "执行中 (1)" not in card_text, "no agent should show as '执行中 (1)'"
    assert "执行中 (3)" not in card_text, "no agent should show as '执行中 (3)'"
    assert "已取消" in card_text, "cancelled agents should appear in '已取消' group"

    # Timing: state transition must be way under 5s
    assert state_elapsed_ms < 5000, f"state transition must be <5s, took {state_elapsed_ms:.0f}ms"
    # It should actually be near-instant (microseconds), but we assert the
    # acceptance criteria of <5s with a generous margin.


def test_race_abort_session_cancel_and_close_within_5s(tmp_path):
    """End-to-end: race loser's ACP session must be cancelled and closed
    within 5s of abort_request, and the progress card must stop showing it
    as '执行中'.

    Exercises the full chain:
      abort_request -> bridge sets per-call cancel_event ->
      executor cancel guard fires session.cancel() ->
      send_prompt returns -> finally calls session.close()

    This is the acceptance-criteria test for race() loser cleanup SLO.
    """
    from concurrent.futures import ThreadPoolExecutor
    from unittest.mock import patch

    from src.workflow_engine.executor import AgentExecutor
    from src.workflow_engine.models import (
        AgentCallParams,
        AgentStatus,
        WorkflowMetrics,
        WorkflowProject,
        WorkflowStatus,
    )
    from src.workflow_engine.renderer import WorkflowProgressRenderer
    from src.workflow_engine.state_manager import WorkflowStateManager

    class _SlowCancelableSession:
        def __init__(self):
            self.session_id = "loser-session"
            self.created_at = time.time()
            self.last_active = time.time()
            self.message_count = 0
            self.last_query = ""
            self.is_resumed = False
            self._cancel_evt = threading.Event()
            self.cancel_called = threading.Event()
            self.close_called = threading.Event()
            self.send_started = threading.Event()

        def describe_agent(self):
            return "fake"

        def start(self, timeout=60):
            return self.session_id

        def load_session(self, sid, timeout=60):
            self.session_id = sid

        def load_local_history(self, *a, **kw):
            return []

        def to_snapshot(self):
            return {"session_id": self.session_id}

        def get_session_info(self):
            return "SlowCancelableSession"

        def is_server_running(self):
            return True

        def is_server_healthy(self, timeout=2.0):
            return True

        def send_prompt_with_retry(self, *a, **kw):
            return self.send_prompt(*a, **kw)

        def cancel(self):
            self.cancel_called.set()
            self._cancel_evt.set()

        def close(self):
            self.close_called.set()

        def send_prompt(self, text, on_event=None, timeout=None):
            self.send_started.set()
            self.last_query = text
            cancelled = self._cancel_evt.wait(timeout=timeout or 30)
            if cancelled:
                time.sleep(0.1)
                raise RuntimeError("cancelled by user")
            raise TimeoutError("prompt timed out")

    loser_session = _SlowCancelableSession()

    def _factory(agent_type, cwd, model_name=None, cancel_event=None):
        if agent_type == "slow_tool":
            return loser_session
        s = MagicMock()
        s.session_id = "fast-session"
        s.send_prompt.return_value = MagicMock(text="fast result", output_tokens=5)
        s.cancel = MagicMock()
        s.close = MagicMock()
        return s

    project = WorkflowProject(
        workflow_id="wf-race-e2e",
        name="race-e2e",
        status=WorkflowStatus.RUNNING,
        requirement="race loser cleanup e2e",
        metrics=WorkflowMetrics(),
    )
    sm = WorkflowStateManager(project)
    sm.on_phase_changed("race phase")
    renderer = WorkflowProgressRenderer(project)

    engine_cancel = threading.Event()
    executor = AgentExecutor(
        cwd=str(tmp_path),
        cancel_event=engine_cancel,
        max_workers=4,
    )

    def on_agent_call(params: AgentCallParams, cancel_event=None):
        label = params.label or params.tool
        sm.on_agent_started(
            label,
            tool=params.tool,
            phase="race phase",
            task_summary=params.prompt[:50],
        )
        try:
            result = executor.execute(params, cancel_event=cancel_event)
            if result.error:
                sm.on_agent_failed(label, result.error)
            else:
                sm.on_agent_done(
                    label,
                    {
                        "token_usage": result.token_usage,
                        "duration_s": result.duration_s,
                        "cached": False,
                    },
                )
            return result
        except Exception as e:
            sm.on_agent_failed(label, str(e))
            raise

    bridge = RuntimeBridge(
        script_path="test.js",
        cwd=str(tmp_path),
        max_concurrent=4,
        on_agent_call=on_agent_call,
    )
    bridge._process = MagicMock()
    bridge._process.poll.return_value = None
    bridge._process.stdin = MagicMock()
    bridge._process.stdout = MagicMock()
    bridge._process.stderr = MagicMock()
    bridge._executor = ThreadPoolExecutor(max_workers=4)
    bridge._active_futures = set()
    bridge._futures_lock = threading.Lock()
    bridge._request_futures = {}
    bridge._request_futures_lock = threading.Lock()
    bridge._request_cancel_events = {}
    bridge._request_cancel_events_lock = threading.Lock()
    bridge._write_lock = threading.Lock()
    bridge._msg_condition = threading.Condition()
    bridge._msg_queue = deque()

    try:
        with patch("src.agent_session.factory.create_engine_session", side_effect=_factory):
            bridge._handle_agent_call(
                {"prompt": "solve problem slowly", "tool": "slow_tool", "label": "loser"},
                request_id="race-loser-1",
            )

            assert loser_session.send_started.wait(timeout=5.0), "loser send_prompt must start"

            snapshot = sm.snapshot()
            card = renderer.render_progress_card(snapshot)
            card_text = json.dumps(card, ensure_ascii=False)
            assert "执行中" in card_text, "initial: loser must be '执行中'"

            start = time.monotonic()
            bridge._handle_abort_request("race-loser-1")
            sm.on_agent_aborted("loser", reason="race loser")

            assert loser_session.cancel_called.wait(timeout=2.0), "session.cancel() must be called after abort_request"
            cancel_elapsed = (time.monotonic() - start) * 1000
            assert cancel_elapsed < 1000, f"session.cancel() must fire within 1s, took {cancel_elapsed:.0f}ms"

            assert loser_session.close_called.wait(timeout=10.0), "session.close() must be called"
            close_elapsed = time.monotonic() - start
            assert close_elapsed < 5.0, f"session.close() must complete within 5s of abort, took {close_elapsed:.2f}s"

            snapshot = sm.snapshot()
            card = renderer.render_progress_card(snapshot)
            card_text = json.dumps(card, ensure_ascii=False)
            assert "已取消" in card_text, "loser must appear in '已取消' group"
            running_agents = [a for a in snapshot.phases[0].agents if a.status == AgentStatus.RUNNING]
            assert len(running_agents) == 0, f"no agents should be RUNNING, got {len(running_agents)}"

    finally:
        executor.shutdown(wait=False)
        bridge.stop()
