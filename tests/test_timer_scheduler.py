"""Tests for TimerScheduler — shared timer infrastructure."""

from __future__ import annotations

import threading
import time

import pytest

from src.card.timers.scheduler import TimerScheduler, _reset_global_scheduler, get_timer_scheduler


@pytest.fixture
def scheduler():
    """Create a fresh TimerScheduler for each test."""
    s = TimerScheduler()
    yield s
    s.shutdown(timeout=2.0)


class TestTimerSchedulerBasic:
    """Basic schedule/cancel/shutdown functionality."""

    def test_shutdown_cancels_distant_timer_and_exits_worker_promptly(self):
        sleep_entered = threading.Event()
        fired = threading.Event()

        class ObservedTimerScheduler(TimerScheduler):
            def _interruptible_sleep(self, duration: float) -> None:
                sleep_entered.set()
                super()._interruptible_sleep(duration)

        scheduler = ObservedTimerScheduler()
        try:
            scheduler.schedule(60.0, fired.set, session_id="distant")
            assert sleep_entered.wait(timeout=1.0)

            started_at = time.monotonic()
            scheduler.shutdown()
            elapsed = time.monotonic() - started_at

            assert elapsed < 0.5
            assert scheduler.is_alive is False
            assert scheduler.pending_count == 0
            assert fired.is_set() is False
        finally:
            # Wake the pre-fix implementation after the assertion so a red
            # test does not leave its daemon worker asleep for 60 seconds.
            scheduler._wake_event.set()
            scheduler._thread.join(timeout=1.0)

    def test_schedule_waiting_on_lock_is_rejected_after_shutdown_starts(self):
        scheduler = TimerScheduler()
        original_lock = scheduler._lock
        schedule_waiting = threading.Event()
        handles = []

        class SignalingLock:
            def __enter__(self):
                if threading.current_thread().name == "late-schedule":
                    schedule_waiting.set()
                original_lock.acquire()
                return self

            def __exit__(self, *_args):
                original_lock.release()

        scheduler._lock = SignalingLock()
        original_lock.acquire()
        worker = threading.Thread(
            target=lambda: handles.append(
                scheduler.schedule(60.0, lambda: None, session_id="late")
            ),
            name="late-schedule",
        )
        try:
            worker.start()
            assert schedule_waiting.wait(timeout=1.0)
            scheduler._shutdown_event.set()
        finally:
            original_lock.release()

        worker.join(timeout=1.0)
        try:
            assert worker.is_alive() is False
            assert len(handles) == 1
            assert handles[0].cancelled is True
            assert scheduler.pending_count == 0
        finally:
            scheduler.shutdown()

    def test_schedule_fires_callback(self, scheduler):
        fired = threading.Event()
        scheduler.schedule(0.05, fired.set, session_id="test")
        assert fired.wait(timeout=2.0), "Callback should fire within 2s"

    def test_cancel_prevents_callback(self, scheduler):
        fired = threading.Event()
        handle = scheduler.schedule(0.1, fired.set, session_id="test")
        scheduler.cancel(handle)
        time.sleep(0.2)
        assert not fired.is_set(), "Cancelled callback should not fire"

    def test_cancel_idempotent(self, scheduler):
        handle = scheduler.schedule(0.1, lambda: None, session_id="test")
        scheduler.cancel(handle)
        scheduler.cancel(handle)  # Should not raise

    def test_cancel_none_handle(self, scheduler):
        scheduler.cancel(None)  # Should not raise

    def test_shutdown_prevents_new_callbacks(self, scheduler):
        scheduler.shutdown()
        fired = threading.Event()
        handle = scheduler.schedule(0.05, fired.set, session_id="test")
        time.sleep(0.1)
        assert handle.cancelled
        assert not fired.is_set()

    def test_pending_count(self, scheduler):
        assert scheduler.pending_count == 0
        scheduler.schedule(1.0, lambda: None, session_id="a")
        scheduler.schedule(1.0, lambda: None, session_id="b")
        assert scheduler.pending_count >= 1  # May fire fast


class TestTimerSchedulerConcurrency:
    """Verify single thread handles many timers."""

    def test_single_daemon_thread_for_100_schedules(self, scheduler):
        """100+ scheduled callbacks should all use the same 1 daemon thread."""
        counter = {"count": 0}
        lock = threading.Lock()
        done = threading.Event()
        target = 100

        def _cb():
            with lock:
                counter["count"] += 1
                if counter["count"] >= target:
                    done.set()

        for i in range(target):
            scheduler.schedule(0.01 * (i % 10), _cb, session_id=f"s{i}")

        assert done.wait(timeout=5.0), f"Only {counter['count']}/{target} fired"
        # Only 1 scheduler thread should exist
        assert scheduler.is_alive
        # The scheduler uses exactly 1 thread internally
        timer_threads = [t for t in threading.enumerate() if t.name == "timer-scheduler"]
        assert len(timer_threads) <= 2  # might be 1 from fixture + 1 from global

    def test_callback_exception_doesnt_kill_scheduler(self, scheduler):
        """A failing callback should not prevent subsequent callbacks."""
        fired = threading.Event()

        def _bad():
            raise RuntimeError("boom")

        scheduler.schedule(0.01, _bad, session_id="fail")
        scheduler.schedule(0.05, fired.set, session_id="ok")

        assert fired.wait(timeout=2.0), "Scheduler should survive callback exception"

    def test_slow_callback_does_not_stall_due_timer_submission(self):
        scheduler = TimerScheduler(callback_workers=2)
        slow_started = threading.Event()
        release_slow = threading.Event()
        fast_fired = threading.Event()

        def _slow():
            slow_started.set()
            release_slow.wait(timeout=2.0)

        try:
            scheduler.schedule(0, _slow, session_id="slow")
            scheduler.schedule(0, fast_fired.set, session_id="fast")

            assert slow_started.wait(timeout=1.0)
            assert fast_fired.wait(timeout=0.5), (
                "a slow callback must not block the timer dequeue thread"
            )
        finally:
            release_slow.set()
            scheduler.shutdown(timeout=1.0)

    def test_shutdown_cancels_submitted_callback_that_has_not_started(self):
        scheduler = TimerScheduler(callback_workers=1)
        first_started = threading.Event()
        release_first = threading.Event()
        queued_fired = threading.Event()

        def _first():
            first_started.set()
            release_first.wait(timeout=2.0)

        try:
            scheduler.schedule(0, _first, session_id="running")
            scheduler.schedule(0, queued_fired.set, session_id="queued")
            assert first_started.wait(timeout=1.0)

            scheduler.shutdown(timeout=0.2)
            release_first.set()
            time.sleep(0.05)

            assert queued_fired.is_set() is False
        finally:
            release_first.set()
            scheduler.shutdown(timeout=1.0)

    def test_saturated_callback_queue_retries_every_accepted_timer(self):
        scheduler = TimerScheduler(callback_workers=1, callback_queue_size=1)
        first_started = threading.Event()
        release_first = threading.Event()
        completed: list[str] = []
        completed_lock = threading.Lock()
        all_done = threading.Event()

        def _record(name: str) -> None:
            with completed_lock:
                completed.append(name)
                if len(completed) == 3:
                    all_done.set()

        def _first() -> None:
            first_started.set()
            release_first.wait(timeout=2.0)
            _record("first")

        try:
            scheduler.schedule(0, _first, session_id="first")
            scheduler.schedule(0, lambda: _record("second"), session_id="second")
            scheduler.schedule(0, lambda: _record("third"), session_id="third")
            assert first_started.wait(timeout=1.0)
            time.sleep(0.05)
            release_first.set()

            assert all_done.wait(timeout=2.0)
            assert set(completed) == {"first", "second", "third"}
        finally:
            release_first.set()
            scheduler.shutdown(timeout=1.0)

    def test_shutdown_timeout_reports_running_callback_then_can_finish(self):
        scheduler = TimerScheduler(callback_workers=1)
        callback_started = threading.Event()
        release_callback = threading.Event()

        def _running() -> None:
            callback_started.set()
            release_callback.wait(timeout=2.0)

        scheduler.schedule(0, _running, session_id="running")
        assert callback_started.wait(timeout=1.0)

        assert scheduler.shutdown(timeout=0.02) is False
        release_callback.set()
        assert scheduler.shutdown(timeout=1.0) is True


class TestTimerSchedulerGlobalSingleton:
    """Test get_timer_scheduler() singleton behavior."""

    def test_singleton_returns_same_instance(self):
        _reset_global_scheduler()
        try:
            s1 = get_timer_scheduler()
            s2 = get_timer_scheduler()
            assert s1 is s2
        finally:
            _reset_global_scheduler()

    def test_reset_creates_new_instance(self):
        _reset_global_scheduler()
        try:
            s1 = get_timer_scheduler()
            _reset_global_scheduler()
            s2 = get_timer_scheduler()
            assert s1 is not s2
        finally:
            _reset_global_scheduler()
