"""Tests for bounded queue trimming logic in SlockEngine."""

import threading
import time
from unittest.mock import MagicMock

import pytest

from src.slock_engine.bounded_executor import BoundedExecutor, QueueFullError
from src.slock_engine.engine import SlockEngine
from src.slock_engine.escalation_manager import EscalationManager
from src.slock_engine.models import (
    EscalationLevel,
    EscalationRequest,
    SlockTask,
    TaskStatus,
)
from src.slock_engine.task_board_manager import TaskBoardManager


def _make_engine_mock(tasks=None, escalations=None):
    """Create a minimal mock engine with required attributes for trim methods."""
    engine = MagicMock(spec=SlockEngine)
    engine._lock = threading.Lock()
    engine._tasks = tasks or []
    engine._escalations = escalations or []
    engine._escalation_retry_counts = {}

    # Set up _task_mgr mock with real _trim_done_tasks
    engine._task_mgr = MagicMock(spec=TaskBoardManager)
    engine._task_mgr._tasks = engine._tasks
    engine._task_mgr._lock = engine._lock
    engine._task_mgr._trim_done_tasks = TaskBoardManager._trim_done_tasks.__get__(engine._task_mgr, TaskBoardManager)

    # Set up _escalation_mgr mock with real _trim_escalations
    engine._escalation_mgr = MagicMock(spec=EscalationManager)
    engine._escalation_mgr._escalations = engine._escalations
    engine._escalation_mgr._escalation_retry_counts = engine._escalation_retry_counts
    engine._escalation_mgr._timeout_timers = {}
    engine._escalation_mgr._half_timers = {}
    engine._escalation_mgr._lock = engine._lock
    engine._escalation_mgr._trim_escalations = EscalationManager._trim_escalations.__get__(engine._escalation_mgr, EscalationManager)

    engine._trim_done_tasks = SlockEngine._trim_done_tasks.__get__(engine, SlockEngine)
    engine._trim_escalations = SlockEngine._trim_escalations.__get__(engine, SlockEngine)
    return engine


def _make_done_task(index: int) -> SlockTask:
    """Create a DONE task with claimed_at = index (for ordering)."""
    return SlockTask(
        task_id=f"task-{index}",
        content=f"Task content {index}",
        status=TaskStatus.DONE,
        claimed_by=f"agent-{index}",
        claimed_at=float(index),
        created_in="test-session",
    )


def _make_non_done_task(index: int, status: TaskStatus = TaskStatus.TODO) -> SlockTask:
    """Create a non-DONE task."""
    return SlockTask(
        task_id=f"task-nd-{index}",
        content=f"Non-done task {index}",
        status=status,
        claimed_by=None,
        claimed_at=None,
        created_in="test-session",
    )


def _make_resolved_escalation(index: int) -> EscalationRequest:
    """Create a resolved escalation with resolved_at = index (for ordering)."""
    return EscalationRequest(
        escalation_id=f"esc-{index}",
        agent_id=f"agent-{index}",
        agent_name=f"Agent {index}",
        level=EscalationLevel.WARNING,
        reason=f"Reason {index}",
        options=["option-a", "option-b"],
        resolved=True,
        resolution=f"Resolved {index}",
        resolved_at=float(index),
    )


def _make_unresolved_escalation(index: int) -> EscalationRequest:
    """Create an unresolved escalation."""
    return EscalationRequest(
        escalation_id=f"esc-unresolved-{index}",
        agent_id=f"agent-{index}",
        agent_name=f"Agent {index}",
        level=EscalationLevel.WARNING,
        reason=f"Reason {index}",
        options=["option-a", "option-b"],
        resolved=False,
        resolution=None,
        resolved_at=None,
    )


class TestTrimDoneTasks:
    """Tests for _trim_done_tasks bounded queue logic."""

    def test_trim_done_tasks_at_limit_no_trim(self):
        """Exactly 500 DONE tasks should not trigger any trimming."""
        tasks = [_make_done_task(i) for i in range(500)]
        engine = _make_engine_mock(tasks=tasks)

        with engine._lock:
            engine._trim_done_tasks()

        assert len(engine._tasks) == 500

    def test_trim_done_tasks_over_limit(self):
        """501 DONE tasks should trim to 500, removing the oldest (lowest claimed_at)."""
        tasks = [_make_done_task(i) for i in range(501)]
        engine = _make_engine_mock(tasks=tasks)

        with engine._lock:
            engine._trim_done_tasks()

        assert len(engine._tasks) == 500
        # The oldest task (claimed_at=0) should have been removed
        remaining_ids = {t.task_id for t in engine._tasks}
        assert "task-0" not in remaining_ids
        # The newest task should still be present
        assert "task-500" in remaining_ids

    def test_trim_done_tasks_preserves_non_done(self):
        """Non-DONE tasks must never be removed regardless of count."""
        # Create 501 DONE tasks plus some non-DONE tasks
        done_tasks = [_make_done_task(i) for i in range(501)]
        non_done_tasks = [
            _make_non_done_task(0, TaskStatus.TODO),
            _make_non_done_task(1, TaskStatus.IN_PROGRESS),
            _make_non_done_task(2, TaskStatus.IN_REVIEW),
        ]
        all_tasks = non_done_tasks + done_tasks
        engine = _make_engine_mock(tasks=all_tasks)

        with engine._lock:
            engine._trim_done_tasks()

        # All non-DONE tasks must still be present
        remaining_statuses = [(t.task_id, t.status) for t in engine._tasks if t.status != TaskStatus.DONE]
        assert len(remaining_statuses) == 3
        non_done_ids = {t.task_id for t in engine._tasks if t.status != TaskStatus.DONE}
        assert "task-nd-0" in non_done_ids
        assert "task-nd-1" in non_done_ids
        assert "task-nd-2" in non_done_ids

        # DONE tasks should be trimmed to 500
        done_count = sum(1 for t in engine._tasks if t.status == TaskStatus.DONE)
        assert done_count == 500


class TestTrimEscalations:
    """Tests for _trim_escalations bounded queue logic."""

    def test_trim_escalations_at_limit_no_trim(self):
        """Exactly 100 resolved escalations should not trigger any trimming."""
        escalations = [_make_resolved_escalation(i) for i in range(100)]
        engine = _make_engine_mock(escalations=escalations)

        with engine._lock:
            engine._trim_escalations()

        assert len(engine._escalations) == 100

    def test_trim_escalations_over_limit(self):
        """101 resolved escalations should trim to 100, removing oldest (lowest resolved_at)."""
        escalations = [_make_resolved_escalation(i) for i in range(101)]
        engine = _make_engine_mock(escalations=escalations)

        with engine._lock:
            engine._trim_escalations()

        assert len(engine._escalations) == 100
        # The oldest escalation (resolved_at=0) should have been removed
        remaining_ids = {e.escalation_id for e in engine._escalations}
        assert "esc-0" not in remaining_ids
        # The newest escalation should still be present
        assert "esc-100" in remaining_ids

    def test_trim_escalations_preserves_unresolved(self):
        """Unresolved escalations under the pending cap are preserved."""
        # Create 101 resolved escalations plus some unresolved ones
        resolved = [_make_resolved_escalation(i) for i in range(101)]
        unresolved = [_make_unresolved_escalation(i) for i in range(5)]
        all_escalations = unresolved + resolved
        engine = _make_engine_mock(escalations=all_escalations)

        with engine._lock:
            engine._trim_escalations()

        # All unresolved escalations must still be present
        unresolved_remaining = [e for e in engine._escalations if not e.resolved]
        assert len(unresolved_remaining) == 5
        unresolved_ids = {e.escalation_id for e in unresolved_remaining}
        for i in range(5):
            assert f"esc-unresolved-{i}" in unresolved_ids

        # Resolved escalations should be trimmed to 100
        resolved_count = sum(1 for e in engine._escalations if e.resolved)
        assert resolved_count == 100

    def test_trim_escalations_caps_unresolved(self):
        """Oldest unresolved escalations are capped to avoid unbounded growth."""
        unresolved = [_make_unresolved_escalation(i) for i in range(101)]
        engine = _make_engine_mock(escalations=unresolved)

        with engine._lock:
            engine._trim_escalations()

        assert len(engine._escalations) == 100
        remaining_ids = {e.escalation_id for e in engine._escalations}
        assert "esc-unresolved-0" not in remaining_ids
        assert "esc-unresolved-100" in remaining_ids


class TestBoundedExecutorQueueFull:
    """AC-17: BoundedExecutor rejects submissions when queue is full."""

    def test_shutdown_can_cancel_queued_work_during_service_quiesce(self):
        executor = BoundedExecutor(max_workers=1, max_queue_size=3)
        running = threading.Event()
        release = threading.Event()
        queued_ran = threading.Event()

        def first():
            running.set()
            release.wait(timeout=2)

        first_future = executor.submit(first)
        assert running.wait(timeout=1)
        queued_future = executor.submit(queued_ran.set)

        executor.shutdown(wait=False, cancel_futures=True)
        release.set()
        first_future.result(timeout=1)

        assert queued_future.cancelled()
        assert not queued_ran.is_set()

    def test_queue_full_raises_error(self):
        """Fill worker+queue capacity, then next submit raises QueueFullError."""
        barrier = threading.Barrier(2 + 1)  # 2 workers + main thread signal
        executor = BoundedExecutor(max_workers=2, max_queue_size=3)
        try:
            # Block 2 workers
            def blocking_task():
                barrier.wait(timeout=5)

            executor.submit(blocking_task)
            executor.submit(blocking_task)
            time.sleep(0.1)  # Let workers pick up tasks

            # Fill the queue (pending now = 2 running + 1 queued = 3)
            executor.submit(lambda: None)

            # Next submit should fail
            with pytest.raises(QueueFullError):
                executor.submit(lambda: None)
        finally:
            barrier.abort()
            executor.shutdown(wait=False)

    def test_enqueue_time_attached(self):
        """Each submitted future has an enqueue_time attribute."""
        executor = BoundedExecutor(max_workers=1, max_queue_size=5)
        try:
            before = time.time()
            future = executor.submit(lambda: "result")
            after = time.time()

            assert hasattr(future, "enqueue_time")
            assert before <= future.enqueue_time <= after

            # Verify the task still completes
            assert future.result(timeout=2) == "result"
        finally:
            executor.shutdown(wait=True)

    def test_pending_count_decrements_on_completion(self):
        """pending_count decreases when tasks finish."""
        event = threading.Event()
        executor = BoundedExecutor(max_workers=1, max_queue_size=5)
        try:
            def wait_task():
                event.wait(timeout=5)

            future = executor.submit(wait_task)
            assert executor.pending_count >= 1

            event.set()
            future.result(timeout=2)
            time.sleep(0.05)  # Allow done_callback to fire

            assert executor.pending_count == 0
        finally:
            event.set()
            executor.shutdown(wait=True)

    def test_queue_full_error_message(self):
        """QueueFullError message includes count and limit."""
        executor = BoundedExecutor(max_workers=1, max_queue_size=1)
        blocker = threading.Event()
        try:
            executor.submit(lambda: blocker.wait(timeout=5))
            time.sleep(0.05)

            with pytest.raises(QueueFullError, match="maximum queue size"):
                executor.submit(lambda: None)
        finally:
            blocker.set()
            executor.shutdown(wait=True)

    def test_invalid_constructor_params(self):
        """BoundedExecutor rejects invalid max_workers or max_queue_size."""
        with pytest.raises(ValueError, match="max_workers"):
            BoundedExecutor(max_workers=0, max_queue_size=5)
        with pytest.raises(ValueError, match="max_queue_size"):
            BoundedExecutor(max_workers=2, max_queue_size=0)


class TestEscalationIOExecutorLifecycle:
    """Regression coverage for escalation background I/O thread lifecycle."""

    def test_bounded_io_executor_uses_daemon_thread(self):
        from src.slock_engine.escalation_manager import _BoundedIOExecutor

        executor = _BoundedIOExecutor(max_queue_size=2)
        try:
            assert executor._thread.daemon is True
        finally:
            executor.shutdown(wait=True)
