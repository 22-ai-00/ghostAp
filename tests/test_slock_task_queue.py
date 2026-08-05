"""Tests for slock_engine.task_queue — event-driven bounded queue.

Covers:
- Enqueue/dequeue FIFO ordering
- QueueFullError when at capacity
- notify_idle / wait_for_idle event-driven wake
- cancel by task_id
- get_position and size
- Thread-safety: concurrent enqueue + dequeue
"""

from __future__ import annotations

import threading
import time

import pytest

from src.slock_engine.task_queue import QueuedTask, QueueFullError, TaskQueue

# ============================================================
# Basic operations
# ============================================================


class TestBasicQueueOps:
    """FIFO ordering, size, and position tracking."""

    def test_enqueue_returns_position(self):
        q = TaskQueue(max_size=4)
        t1 = QueuedTask(task_id="t1", text="task 1", chat_id="c1", message_id="m1")
        t2 = QueuedTask(task_id="t2", text="task 2", chat_id="c1", message_id="m2")
        assert q.enqueue(t1) == 1
        assert q.enqueue(t2) == 2

    def test_dequeue_fifo(self):
        q = TaskQueue(max_size=4)
        t1 = QueuedTask(task_id="t1", text="first", chat_id="c1", message_id="m1")
        t2 = QueuedTask(task_id="t2", text="second", chat_id="c1", message_id="m2")
        q.enqueue(t1)
        q.enqueue(t2)
        assert q.dequeue().task_id == "t1"
        assert q.dequeue().task_id == "t2"

    def test_dequeue_empty_returns_none(self):
        q = TaskQueue(max_size=4)
        assert q.dequeue() is None

    def test_size(self):
        q = TaskQueue(max_size=4)
        assert q.size() == 0
        q.enqueue(QueuedTask(task_id="t1", text="x", chat_id="c", message_id="m"))
        assert q.size() == 1
        q.dequeue()
        assert q.size() == 0

    def test_get_position(self):
        q = TaskQueue(max_size=4)
        q.enqueue(QueuedTask(task_id="t1", text="x", chat_id="c", message_id="m1"))
        q.enqueue(QueuedTask(task_id="t2", text="y", chat_id="c", message_id="m2"))
        assert q.get_position("t1") == 1
        assert q.get_position("t2") == 2
        assert q.get_position("t999") is None


# ============================================================
# Bounded capacity
# ============================================================


class TestQueueCapacity:
    """QueueFullError when exceeding max_size."""

    def test_queue_full_raises(self):
        q = TaskQueue(max_size=2)
        q.enqueue(QueuedTask(task_id="t1", text="a", chat_id="c", message_id="m1"))
        q.enqueue(QueuedTask(task_id="t2", text="b", chat_id="c", message_id="m2"))
        with pytest.raises(QueueFullError):
            q.enqueue(QueuedTask(task_id="t3", text="c", chat_id="c", message_id="m3"))

    def test_dequeue_frees_capacity(self):
        q = TaskQueue(max_size=2)
        q.enqueue(QueuedTask(task_id="t1", text="a", chat_id="c", message_id="m1"))
        q.enqueue(QueuedTask(task_id="t2", text="b", chat_id="c", message_id="m2"))
        q.dequeue()
        # Should not raise now
        pos = q.enqueue(QueuedTask(task_id="t3", text="c", chat_id="c", message_id="m3"))
        assert pos == 2


# ============================================================
# Cancel
# ============================================================


class TestQueueCancel:
    """Remove tasks by ID."""

    def test_cancel_existing(self):
        q = TaskQueue(max_size=4)
        q.enqueue(QueuedTask(task_id="t1", text="a", chat_id="c", message_id="m1"))
        q.enqueue(QueuedTask(task_id="t2", text="b", chat_id="c", message_id="m2"))
        assert q.cancel("t1") is True
        assert q.size() == 1
        assert q.dequeue().task_id == "t2"

    def test_cancel_nonexistent(self):
        q = TaskQueue(max_size=4)
        assert q.cancel("no_such_task") is False


# ============================================================
# Event-driven: notify_idle / wait_for_idle
# ============================================================


class TestEventDriven:
    """Condition variable wakeup mechanics."""

    def test_wait_for_idle_returns_true_on_notify(self):
        q = TaskQueue(max_size=4)
        woke = []

        def waiter():
            result = q.wait_for_idle(timeout=2.0)
            woke.append(result)

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.05)  # Let waiter block
        q.notify_idle()
        t.join(timeout=3.0)
        assert woke == [True]

    def test_wait_for_idle_returns_false_on_timeout(self):
        q = TaskQueue(max_size=4)
        result = q.wait_for_idle(timeout=0.05)
        assert result is False

    def test_notify_idle_wakes_multiple_waiters(self):
        q = TaskQueue(max_size=4)
        results = []

        def waiter(idx):
            r = q.wait_for_idle(timeout=2.0)
            results.append((idx, r))

        threads = [threading.Thread(target=waiter, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        time.sleep(0.05)
        q.notify_idle()
        for t in threads:
            t.join(timeout=3.0)
        assert len(results) == 3
        assert all(r is True for _, r in results)


# ============================================================
# Thread-safety
# ============================================================


class TestConcurrency:
    """Concurrent enqueue/dequeue without data corruption."""

    def test_concurrent_enqueue_dequeue(self):
        q = TaskQueue(max_size=100)
        enqueued = []
        dequeued = []

        def producer(start):
            for i in range(20):
                tid = f"t-{start}-{i}"
                try:
                    q.enqueue(QueuedTask(task_id=tid, text="x", chat_id="c", message_id=f"m-{tid}"))
                    enqueued.append(tid)
                except QueueFullError:
                    pass

        def consumer():
            for _ in range(40):
                task = q.dequeue()
                if task:
                    dequeued.append(task.task_id)
                time.sleep(0.001)

        threads = [
            threading.Thread(target=producer, args=(0,)),
            threading.Thread(target=producer, args=(1,)),
            threading.Thread(target=consumer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        # All dequeued items must have been enqueued (no corruption)
        assert all(d in enqueued for d in dequeued)
        # Remaining in queue + dequeued should equal total enqueued
        remaining = q.size()
        assert remaining + len(dequeued) == len(enqueued)


# ============================================================
# New features: bootstrap_pending, peek(), drain_to()
# ============================================================


class TestNewQueueFeatures:
    """Tests for peek() and drain_to()."""

    # ----------------------------------------------------------
    # peek() — returns oldest task without removing it
    # ----------------------------------------------------------

    def test_peek_returns_oldest_task(self):
        q = TaskQueue(max_size=4)
        t1 = QueuedTask(task_id="t1", text="first", chat_id="c1", message_id="m1")
        t2 = QueuedTask(task_id="t2", text="second", chat_id="c1", message_id="m2")
        q.enqueue(t1)
        q.enqueue(t2)
        peeked = q.peek()
        assert peeked is not None
        assert peeked.task_id == "t1"

    def test_peek_does_not_remove_task(self):
        q = TaskQueue(max_size=4)
        t1 = QueuedTask(task_id="t1", text="first", chat_id="c1", message_id="m1")
        q.enqueue(t1)
        q.peek()
        assert q.size() == 1
        assert q.dequeue().task_id == "t1"

    def test_peek_returns_none_when_empty(self):
        q = TaskQueue(max_size=4)
        assert q.peek() is None

    # ----------------------------------------------------------
    # drain_to(callback) — dequeue all tasks in FIFO, pass to cb
    # ----------------------------------------------------------

    def test_drain_to_processes_all_tasks_fifo(self):
        q = TaskQueue(max_size=8)
        for i in range(5):
            q.enqueue(QueuedTask(task_id=f"t{i}", text=f"task {i}", chat_id="c", message_id=f"m{i}"))

        collected = []
        count = q.drain_to(lambda task: collected.append(task.task_id))
        assert count == 5
        assert collected == ["t0", "t1", "t2", "t3", "t4"]

    def test_drain_to_returns_zero_on_empty_queue(self):
        q = TaskQueue(max_size=4)
        count = q.drain_to(lambda task: None)
        assert count == 0

    def test_drain_to_leaves_queue_empty(self):
        q = TaskQueue(max_size=4)
        q.enqueue(QueuedTask(task_id="t1", text="a", chat_id="c", message_id="m1"))
        q.enqueue(QueuedTask(task_id="t2", text="b", chat_id="c", message_id="m2"))
        q.drain_to(lambda task: None)
        assert q.size() == 0
        assert q.dequeue() is None

    def test_drain_to_handles_callback_exception_gracefully(self):
        q = TaskQueue(max_size=8)
        for i in range(4):
            q.enqueue(QueuedTask(task_id=f"t{i}", text=f"task {i}", chat_id="c", message_id=f"m{i}"))

        collected = []

        def flaky_callback(task):
            if task.task_id == "t1":
                raise ValueError("simulated failure")
            collected.append(task.task_id)

        count = q.drain_to(flaky_callback)
        # All 4 tasks should be processed (drained) even if one callback raises
        assert count == 4
        # The tasks that did not raise should have been collected
        assert collected == ["t0", "t2", "t3"]
        # Queue must be empty after drain
        assert q.size() == 0
