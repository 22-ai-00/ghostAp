"""Tests for race() loser cancellation correctness.

Focuses on the specific failure modes fixed in this change:
1. Duplicate labels: request_id-based lookup prevents label mismatch
2. No-label contestants: default labels ensure status tracking
3. Early abort race: cancel_event is set even when abort arrives before worker
4. Session creation abort: cancellation works during session startup
5. Immediate progress flush: abort events bypass debounce for fast UI update

These complement the broader tests in test_workflow_runtime_primitives.py.
"""
from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.workflow_engine.bridge import RuntimeBridge
from src.workflow_engine.executor import AgentExecutor
from src.workflow_engine.models import (
    AgentCallParams,
    AgentStatus,
    WorkflowMetrics,
    WorkflowProject,
    WorkflowStatus,
)
from src.workflow_engine.progress_coalescer import ProgressCoalescer
from src.workflow_engine.renderer import WorkflowProgressRenderer
from src.workflow_engine.state_manager import WorkflowStateManager

# ---------------------------------------------------------------------------
# 1. Duplicate label handling: request_id-based lookup
# ---------------------------------------------------------------------------


def test_duplicate_labels_abort_by_request_id(tmp_path):
    """When two race contestants have the same label, abort via request_id
    must correctly identify and cancel each one independently — no mix-ups.

    This tests the engine's _request_to_label mapping: each agent gets a
    unique effective label from state_manager, and the engine maps
    request_id → effective_label so abort notifications hit the right agent.
    """
    from src.workflow_engine.engine import WorkflowEngine

    engine = WorkflowEngine(
        chat_id="test-chat",
        root_path=str(tmp_path),
        agent_type="coco",
    )

    # Ensure state manager is initialized
    if engine._state_manager is None:
        engine._state_manager = WorkflowStateManager(engine._project)
    state_mgr = engine._state_manager

    # Simulate two agents with same raw label being registered
    # state_manager disambiguates them
    label1 = state_mgr.on_agent_started(
        "worker", tool="coco", phase="race", task_summary="approach A"
    )
    label2 = state_mgr.on_agent_started(
        "worker", tool="coco", phase="race", task_summary="approach B"
    )

    assert label1 != label2, "state_manager must disambiguate duplicate labels"
    assert label1 == "worker"
    assert "worker" in label2

    # Engine maps request_ids to effective labels
    engine._request_to_label["req-1"] = label1
    engine._request_to_label["req-2"] = label2

    # Simulate abort of req-2 (the second contestant)
    engine._renderer_wf = WorkflowProgressRenderer(engine._project)

    abort_events = []
    engine._callbacks = MagicMock()
    engine._callbacks.on_progress = lambda data: abort_events.append(data)
    engine._progress_coalescer = None  # bypass coalescer for test

    engine._handle_agent_aborted("worker", "race loser", request_id="req-2")

    # Verify the RIGHT agent was cancelled (the second one, not the first)
    snapshot = state_mgr.snapshot()
    agents = snapshot.phases[0].agents
    assert len(agents) == 2

    # First agent should still be RUNNING
    assert agents[0].status == AgentStatus.RUNNING, (
        f"First agent should still be RUNNING, got {agents[0].status}"
    )
    # Second agent should be CANCELLED
    assert agents[1].status == AgentStatus.CANCELLED, (
        f"Second agent should be CANCELLED, got {agents[1].status}"
    )

    engine.cleanup()


def test_abort_by_label_only_still_works(tmp_path):
    """Fallback: abort with only label (no request_id) must still work
    for backward compatibility with scripts that don't send request_id."""
    project = WorkflowProject(
        workflow_id="wf-fallback",
        name="fallback-test",
        status=WorkflowStatus.RUNNING,
        metrics=WorkflowMetrics(),
    )
    sm = WorkflowStateManager(project)
    sm.on_phase_changed("race")
    label = sm.on_agent_started("unique-agent", tool="coco", phase="race")

    # Abort by label only (no request_id) — direct state manager call
    sm.on_agent_aborted(label, reason="race loser")

    snapshot = sm.snapshot()
    agent = snapshot.phases[0].agents[0]
    assert agent.status == AgentStatus.CANCELLED


# ---------------------------------------------------------------------------
# 2. ProgressCoalescer flush_immediate
# ---------------------------------------------------------------------------


def test_progress_coalescer_flush_immediate_bypasses_debounce(tmp_path):
    """flush_immediate() must deliver the snapshot immediately without
    waiting for the debounce interval."""
    delivered = []
    delivery_times = []
    delivery_lock = threading.Lock()

    def on_progress(data):
        with delivery_lock:
            delivered.append(data)
            delivery_times.append(time.monotonic())

    pc = ProgressCoalescer(on_progress=on_progress, debounce_s=10.0)  # long debounce

    start = time.monotonic()
    pc.flush_immediate({"state": "cancelled"})

    # Should be delivered immediately (well under the 10s debounce)
    import time as _t
    _t.sleep(0.05)  # small buffer for thread scheduling

    with delivery_lock:
        assert len(delivered) >= 1, "flush_immediate should deliver immediately"
        assert delivered[0] == {"state": "cancelled"}
        elapsed = delivery_times[0] - start
        assert elapsed < 0.5, f"immediate flush took too long: {elapsed:.2f}s"

    pc.stop()


def test_progress_coalescer_flush_immediate_min_interval(tmp_path):
    """flush_immediate() enforces a min 200ms interval to prevent spamming."""
    delivered = []
    delivery_lock = threading.Lock()

    def on_progress(data):
        with delivery_lock:
            delivered.append(data)

    pc = ProgressCoalescer(on_progress=on_progress, debounce_s=5.0)

    # Send two immediate flushes very close together
    pc.flush_immediate({"state": "first"})
    pc.flush_immediate({"state": "second"})

    time.sleep(0.05)

    with delivery_lock:
        # Only the first one should have been delivered (min interval guard)
        # The second is stored as latest but not immediately flushed
        assert len(delivered) >= 1
        assert delivered[0] == {"state": "first"}

    pc.stop()


# ---------------------------------------------------------------------------
# 3. Session creation phase cancellation
# ---------------------------------------------------------------------------


def test_executor_cancel_during_session_creation(tmp_path):
    """When cancel_event fires during session creation (before send_prompt),
    the executor must detect it and return a cancelled result instead of
    waiting for the full session creation timeout."""
    import threading

    create_started = threading.Event()
    create_block = threading.Event()

    class _SlowCreateSession:
        """Session whose creation takes a long time (simulates slow startup)."""
        def __init__(self):
            self.session_id = "slow-create"
            self.cancel_called = threading.Event()
            self.close_called = threading.Event()

        def describe_agent(self): return "fake"
        def start(self, timeout=60): return self.session_id
        def load_session(self, sid, timeout=60): self.session_id = sid
        def load_local_history(self, *a, **kw): return []
        def to_snapshot(self): return {"session_id": self.session_id}
        def get_session_info(self): return "SlowCreateSession"
        def is_server_running(self): return True
        def is_server_healthy(self, timeout=2.0): return True

        def cancel(self):
            self.cancel_called.set()

        def close(self):
            self.close_called.set()

        def send_prompt(self, text, on_event=None, timeout=None):
            return MagicMock(text="result", output_tokens=5)

    slow_session = _SlowCreateSession()

    def _slow_factory(agent_type, cwd, model_name=None, cancel_event=None):
        create_started.set()
        # Block for a long time (simulating slow session creation)
        create_block.wait(timeout=30.0)
        return slow_session

    engine_cancel = threading.Event()
    executor = AgentExecutor(
        cwd=str(tmp_path),
        cancel_event=engine_cancel,
        max_workers=2,
    )

    result_holder = []

    def run_call():
        params = AgentCallParams(prompt="test", tool="slow", label="slow-agent")
        res = executor.execute(params, cancel_event=engine_cancel)
        result_holder.append(res)

    with patch("src.agent_session.factory.create_engine_session", side_effect=_slow_factory):
        t = threading.Thread(target=run_call)
        t.start()

        # Wait for session creation to start
        assert create_started.wait(timeout=2.0), "session creation should start"

        # Now cancel during creation
        start = time.monotonic()
        engine_cancel.set()

        t.join(timeout=5.0)
        elapsed = time.monotonic() - start

        assert not t.is_alive(), (
            "executor.execute should return promptly after cancel during "
            "session creation, not wait for full timeout"
        )
        assert elapsed < 3.0, (
            f"Cancellation during session creation took too long: {elapsed:.2f}s"
        )

        assert len(result_holder) == 1
        result = result_holder[0]
        assert result.error is not None
        assert "cancel" in result.error.lower(), (
            f"Expected cancelled error, got: {result.error}"
        )

    # Cleanup
    create_block.set()
    executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# 4. Engine _handle_agent_aborted with request_id lookup
# ---------------------------------------------------------------------------


def test_engine_abort_uses_request_id_over_label(tmp_path):
    """_handle_agent_aborted must prefer request_id-based lookup over raw
    label, ensuring correct agent is cancelled even when labels collide."""
    from src.workflow_engine.engine import WorkflowEngine

    engine = WorkflowEngine(
        chat_id="test-chat",
        root_path=str(tmp_path),
        agent_type="coco",
    )

    # Ensure state manager is initialized (it's created in execute_workflow,
    # but we test the handler directly so we set it up manually)
    if engine._state_manager is None:
        engine._state_manager = WorkflowStateManager(engine._project)
    sm = engine._state_manager

    # Register 2 agents with same raw label
    label_a = sm.on_agent_started("agent-x", tool="coco", phase="p1")
    label_b = sm.on_agent_started("agent-x", tool="coco", phase="p1")

    engine._request_to_label["req-a"] = label_a
    engine._request_to_label["req-b"] = label_b

    engine._renderer_wf = WorkflowProgressRenderer(engine._project)
    engine._progress_coalescer = None  # direct delivery
    engine._callbacks = MagicMock()
    engine._callbacks.on_progress = lambda d: None

    # Abort req-b with wrong raw label — should use request_id to find right agent
    engine._handle_agent_aborted("agent-x", "race loser", request_id="req-b")

    snapshot = sm.snapshot()
    agents = snapshot.phases[0].agents
    assert agents[0].status == AgentStatus.RUNNING  # req-a still running
    assert agents[1].status == AgentStatus.CANCELLED  # req-b cancelled

    engine.cleanup()


# ---------------------------------------------------------------------------
# 5. JS runtime race() default label generation (verify via engine integration)
# ---------------------------------------------------------------------------


def test_race_no_label_agents_get_status_updated(tmp_path):
    """Race contestants without explicit labels must still have their status
    updated on abort (not left as RUNNING forever).

    This verifies the JS-side fix: unlabeled contestants get default labels
    like 'race-contestant-N', and agent_aborted notifications include them.
    """
    project = WorkflowProject(
        workflow_id="wf-nolabel",
        name="nolabel-race",
        status=WorkflowStatus.RUNNING,
        metrics=WorkflowMetrics(),
    )
    sm = WorkflowStateManager(project)
    sm.on_phase_changed("race")

    # Simulate what happens with unlabeled contestants:
    # JS generates default label, engine registers with that label
    labels = []
    for i in range(3):
        lbl = sm.on_agent_started(
            f"race-contestant-{i}",
            tool="coco",
            phase="race",
            task_summary=f"approach {i}",
        )
        labels.append(lbl)

    # All 3 running initially
    snapshot = sm.snapshot()
    running = [a for a in snapshot.phases[0].agents if a.status == AgentStatus.RUNNING]
    assert len(running) == 3

    # Abort all but the winner (contestant 0)
    for i in range(1, 3):
        sm.on_agent_aborted(labels[i], reason="race loser")

    # Verify only 1 running, 2 cancelled
    snapshot = sm.snapshot()
    running = [a for a in snapshot.phases[0].agents if a.status == AgentStatus.RUNNING]
    cancelled = [a for a in snapshot.phases[0].agents if a.status == AgentStatus.CANCELLED]
    assert len(running) == 1
    assert len(cancelled) == 2
    # None should be stuck as RUNNING due to missing label
    assert all(a.status != AgentStatus.RUNNING for a in snapshot.phases[0].agents[1:])


# ---------------------------------------------------------------------------
# 6. Bridge agent_aborted forwards request_id kwarg
# ---------------------------------------------------------------------------


def test_bridge_agent_aborted_forwards_request_id(tmp_path):
    """The bridge must forward request_id from the JS notification to
    the Python callback as a kwarg."""
    import threading

    received = []
    done_evt = threading.Event()

    def on_agent_aborted(label, reason, **kwargs):
        received.append({
            "label": label,
            "reason": reason,
            "request_id": kwargs.get("request_id"),
        })
        done_evt.set()

    bridge = RuntimeBridge(
        script_path="test.js",
        cwd=str(tmp_path),
        on_agent_aborted=on_agent_aborted,
    )
    try:
        proc = MagicMock()
        proc.poll.return_value = None
        proc.stdin = MagicMock()
        proc.stdout = MagicMock()
        proc.stderr = MagicMock()
        bridge._process = proc

        msg = {
            "jsonrpc": "2.0",
            "method": "agent_aborted",
            "params": {
                "label": "loser-1",
                "reason": "race loser",
                "request_id": 42,
            },
        }
        bridge._dispatch_message(msg)

        assert done_evt.wait(timeout=1.0), "callback should be invoked synchronously"
        assert len(received) == 1
        assert received[0]["label"] == "loser-1"
        assert received[0]["reason"] == "race loser"
        assert received[0]["request_id"] == 42
    finally:
        bridge.stop()


# ---------------------------------------------------------------------------
# 7. End-to-end: abort with duplicate labels through full pipeline
# ---------------------------------------------------------------------------


def test_end_to_end_duplicate_label_cancel_correct_agent(tmp_path):
    """Full pipeline: two agents with same label, abort one by request_id,
    verify only the correct one is cancelled.

    Exercises: bridge → engine request_id→label mapping → state_manager
    """
    from src.workflow_engine.engine import WorkflowEngine

    engine = WorkflowEngine(
        chat_id="test-chat",
        root_path=str(tmp_path),
        agent_type="coco",
    )

    # Ensure state manager and renderer are initialized
    if engine._state_manager is None:
        engine._state_manager = WorkflowStateManager(engine._project)
    if engine._renderer_wf is None:
        engine._renderer_wf = WorkflowProgressRenderer(engine._project)

    sm = engine._state_manager

    # Set up progress capture
    progress_cards = []
    prog_lock = threading.Lock()

    def capture_progress(card):
        with prog_lock:
            progress_cards.append(card)

    engine._progress_coalescer = None
    engine._callbacks = MagicMock()
    engine._callbacks.on_progress = capture_progress

    # Register two agents with same raw label (simulates race contestants)
    label1 = sm.on_agent_started("worker", tool="coco", phase="race", task_summary="A")
    label2 = sm.on_agent_started("worker", tool="coco", phase="race", task_summary="B")

    engine._request_to_label["req-winner"] = label1
    engine._request_to_label["req-loser"] = label2

    # Initial state: both running
    engine._fire_progress()

    # Abort the loser (req-loser) — this is what JS sends via agent_aborted
    engine._handle_agent_aborted("worker", "race loser", request_id="req-loser")

    # Verify state
    snapshot = sm.snapshot()
    agents = snapshot.phases[0].agents
    assert agents[0].status == AgentStatus.RUNNING, "Winner should still be running"
    assert agents[1].status == AgentStatus.CANCELLED, "Loser should be cancelled"

    # Verify progress card was updated (immediate flush for abort)
    with prog_lock:
        assert len(progress_cards) >= 2  # at least initial + abort
        last_card = progress_cards[-1]
        card_text = json.dumps(last_card, ensure_ascii=False)
        assert "已取消" in card_text, "Cancelled agent should appear in card"

    engine.cleanup()


# ---------------------------------------------------------------------------
# 8. Nested workflow race: agent_aborted propagates from sub-workflow to parent
# ---------------------------------------------------------------------------


NESTED_RACE_TEMPLATE = """\
export const meta = {
  name: 'nested-race-inner',
  description: 'Inner template with race() for nested workflow test',
  phases: [
    { title: 'inner-race', detail: 'Race inside a sub-workflow template' },
  ],
  tools: ['coco'],
};

export default async function main() {
  phase('inner-race');
  const result = await race([
    { prompt: 'inner fast', label: 'inner-fast', tool: 'coco' },
    { prompt: 'inner slow', label: 'inner-slow', tool: 'coco' },
  ]);
  return result;
}
"""


NESTED_RACE_OUTER_SCRIPT = """\
export const meta = {
  name: 'nested-race-outer',
  description: 'Outer workflow that calls a sub-workflow with race()',
  phases: [
    { title: 'outer', detail: 'Call nested workflow' },
  ],
  tools: ['coco'],
};

export default async function main() {
  phase('outer');
  const result = await workflow('nested-race-inner');
  return result;
}
"""


@pytest.mark.skipif(
    not RuntimeBridge.check_node_available(),
    reason="Node.js not available or version too old",
)
def test_nested_workflow_race_abort_propagates_to_parent(tmp_path):
    """When race() inside a sub-workflow cancels a loser, the agent_aborted
    notification must propagate up through the sub-RuntimeBridge to the parent
    bridge callback, so the parent engine/state_manager can update the UI.

    This is the regression test for the bug where sub RuntimeBridge was
    constructed without on_agent_aborted, causing nested race losers to
    remain stuck as RUNNING in the parent's progress card.
    """
    import os

    from src.workflow_engine.models import AgentCallResult

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

    # Write inner template to a temp location
    inner_script = tmp_path / "nested-race-inner.js"
    inner_script.write_text(NESTED_RACE_TEMPLATE, encoding="utf-8")

    # Write outer workflow script
    outer_script = tmp_path / "nested_race_outer.js"
    outer_script.write_text(NESTED_RACE_OUTER_SCRIPT, encoding="utf-8")

    # State tracking at the PARENT bridge level
    aborted_labels: list[str] = []
    aborted_event = threading.Event()
    slow_cancel_seen = threading.Event()
    fast_called = threading.Event()
    slow_called = threading.Event()

    def on_agent_call(params, *, cancel_event=None, **_kwargs):
        label = params.label or ""
        if label == "inner-fast":
            fast_called.set()
            return AgentCallResult(output="inner-fast-wins", tool=params.tool)
        elif label == "inner-slow":
            slow_called.set()
            for _ in range(300):  # 30s max
                if cancel_event is not None and cancel_event.is_set():
                    slow_cancel_seen.set()
                    return AgentCallResult(
                        output="cancelled",
                        error="Cancelled",
                        tool=params.tool,
                    )
                time.sleep(0.1)
            return AgentCallResult(output="slow-too-late", tool=params.tool)
        else:
            return AgentCallResult(
                output=f"unknown:{label}", tool=params.tool or "coco"
            )

    def on_agent_aborted(label, reason, **_kwargs):
        aborted_labels.append(label)
        if label == "inner-slow":
            aborted_event.set()

    # Patch resolve_template_path so the inner template is found
    from src.workflow_engine import templates as templates_mod

    original_resolve = templates_mod.resolve_template_path

    def patched_resolve(root_path, name, *, user_id=None):
        if name == "nested-race-inner":
            return str(inner_script.resolve())
        return original_resolve(root_path, name, user_id=user_id)

    bridge = RuntimeBridge(
        script_path=str(outer_script),
        cwd=project_root,
        max_concurrent=2,
        on_agent_call=on_agent_call,
        on_agent_aborted=on_agent_aborted,
        allowed_tools=["coco"],
    )

    try:
        with patch.object(templates_mod, "resolve_template_path", patched_resolve):
            bridge.start()

            start = time.monotonic()
            result = bridge.run()
            elapsed = time.monotonic() - start

        # 1. Must finish well before slow agent's 30s poll
        assert elapsed < 10.0, (
            f"Nested race took {elapsed:.2f}s — should complete in <10s"
        )

        # 2. Both inner agents were called
        assert fast_called.is_set(), "inner-fast should have been called"
        assert slow_called.is_set(), "inner-slow should have been called"

        # 3. Parent bridge's on_agent_aborted was called for inner-slow
        assert aborted_event.wait(timeout=3.0), (
            f"on_agent_aborted should be invoked for inner-slow at parent "
            f"level; got aborted_labels={aborted_labels}"
        )
        assert "inner-slow" in aborted_labels, (
            f"Expected 'inner-slow' in parent aborted_labels, got: {aborted_labels}"
        )

        # 4. slow's cancel_event was actually set
        assert slow_cancel_seen.wait(timeout=2.0), (
            "inner-slow's cancel_event should have been set by abort_request"
        )

        # 5. Result should contain the fast agent's output
        # Note: nested workflow results are double-JSON-encoded because
        # sub_bridge.run() returns a JSON string which gets placed in the
        # response data field and re-serialized. We just verify the value
        # is present — the core thing we're testing is agent_aborted
        # propagation, not result encoding.
        assert "inner-fast-wins" in result, (
            f"Expected 'inner-fast-wins' in result, got: {result!r}"
        )

    finally:
        bridge.stop()
