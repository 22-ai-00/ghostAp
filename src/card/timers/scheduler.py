"""Global shared timer scheduler — replaces per-session threading.Timer threads.

Uses a single daemon thread + sched.scheduler to manage all session timers
with O(1) thread overhead regardless of session count.
"""

from __future__ import annotations

import logging
import sched
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

__all__ = ["TimerScheduler", "TimerHandle", "get_timer_scheduler"]


@dataclass
class TimerHandle:
    """Opaque handle returned by schedule(), used for cancellation."""
    _event: sched.Event | None = field(default=None, repr=False)
    _cancelled: bool = field(default=False, repr=False)

    @property
    def cancelled(self) -> bool:
        return self._cancelled


class TimerScheduler:
    """Thread-safe shared timer scheduler using sched.scheduler.

    All callbacks MUST be lightweight (≤10ms). If a callback needs I/O,
    it should submit work to a thread pool rather than blocking the scheduler.
    """

    def __init__(self) -> None:
        self._scheduler = sched.scheduler(timefunc=time.monotonic, delayfunc=self._interruptible_sleep)
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._shutdown_event = threading.Event()
        self._wake_event = threading.Event()  # used to interrupt sleep on new schedule
        self._thread = threading.Thread(
            target=self._run_loop, name="timer-scheduler", daemon=True
        )
        self._thread.start()

    def schedule(self, delay: float, callback: Callable[[], Any], *, session_id: str = "") -> TimerHandle:
        """Schedule a callback to fire after `delay` seconds.

        Args:
            delay: Seconds from now.
            callback: Must complete in ≤10ms (no I/O).
            session_id: For logging/debugging only.

        Returns:
            TimerHandle that can be passed to cancel().
        """
        if self._shutdown_event.is_set():
            handle = TimerHandle()
            handle._cancelled = True
            return handle

        handle = TimerHandle()

        def _wrapped() -> None:
            if handle._cancelled:
                return
            try:
                callback()
            except Exception:
                logger.exception("TimerScheduler callback error (session=%s)", session_id)

        with self._lock:
            # Close the race where schedule() passed the fast-path check just
            # before shutdown began, then waited for the scheduler lock.
            if self._shutdown_event.is_set():
                handle._cancelled = True
                return handle
            event = self._scheduler.enter(delay, 1, _wrapped)
            handle._event = event

        # Wake the scheduler thread so it recalculates sleep
        self._wake_event.set()
        return handle

    def cancel(self, handle: TimerHandle) -> None:
        """Cancel a scheduled timer. No-op if already fired or cancelled."""
        if handle is None or handle._cancelled:
            return
        handle._cancelled = True
        if handle._event is not None:
            with self._lock:
                try:
                    self._scheduler.cancel(handle._event)
                except ValueError:
                    pass  # already fired or removed

    def shutdown(self, timeout: float = 2.0) -> None:
        """Stop the scheduler thread. Pending callbacks are discarded."""
        self._shutdown_event.set()
        # Remove future events before waking the worker.  Otherwise
        # sched.scheduler.run() observes the still-pending event after the
        # first wake and goes straight back to sleep, forcing join() to wait
        # for its timeout and leaving the daemon thread alive.
        with self._lock:
            for event in list(self._scheduler.queue):
                try:
                    self._scheduler.cancel(event)
                except ValueError:
                    pass
        self._wake_event.set()
        self._thread.join(timeout=timeout)

    @property
    def pending_count(self) -> int:
        """Number of pending (not yet fired) events."""
        with self._lock:
            return len(self._scheduler.queue)

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _interruptible_sleep(self, duration: float) -> None:
        """Sleep that can be interrupted by new schedules or shutdown."""
        self._wake_event.wait(timeout=duration)
        self._wake_event.clear()

    def _run_loop(self) -> None:
        """Main loop: run scheduler until shutdown."""
        while not self._shutdown_event.is_set():
            with self._lock:
                has_events = not self._scheduler.empty()
            if has_events:
                # run(blocking=True) will call _interruptible_sleep between events
                self._scheduler.run(blocking=True)
            else:
                # No events — sleep until woken
                self._wake_event.wait()
                self._wake_event.clear()


# Module-level singleton (lazy)
_global_scheduler: TimerScheduler | None = None
_global_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock


def get_timer_scheduler() -> TimerScheduler:
    """Get or create the global TimerScheduler singleton."""
    global _global_scheduler
    if _global_scheduler is None or not _global_scheduler.is_alive:
        with _global_lock:
            if _global_scheduler is None or not _global_scheduler.is_alive:
                _global_scheduler = TimerScheduler()
    return _global_scheduler


def _reset_global_scheduler() -> None:
    """For testing: shut down and reset the global scheduler."""
    global _global_scheduler
    with _global_lock:
        if _global_scheduler is not None:
            _global_scheduler.shutdown()
            _global_scheduler = None
