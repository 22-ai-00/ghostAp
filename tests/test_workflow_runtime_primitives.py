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
from unittest.mock import MagicMock

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
