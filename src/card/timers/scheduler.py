"""Global shared timer scheduler — replaces per-session threading.Timer threads.

Uses a single daemon thread + sched.scheduler to manage all session timers
with O(1) thread overhead regardless of session count.
"""

from __future__ import annotations

import logging
import sched
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

_CALLBACK_RETRY_DELAY_SECONDS = 0.01

__all__ = ["TimerScheduler", "TimerHandle", "get_timer_scheduler"]


@dataclass(eq=False)
class TimerHandle:
    """Opaque handle returned by schedule(), used for cancellation."""
    _event: sched.Event | None = field(default=None, repr=False)
    _future: Future[Any] | None = field(default=None, repr=False)
    _cancelled: bool = field(default=False, repr=False)

    @property
    def cancelled(self) -> bool:
        return self._cancelled


class TimerScheduler:
    """Thread-safe shared timer scheduler using sched.scheduler.

    The timer thread only dequeues due events and submits callbacks to a fixed
    worker pool, so one slow callback cannot stall unrelated session timers.
    """

    def __init__(
        self,
        *,
        callback_workers: int = 4,
        callback_queue_size: int = 1_024,
    ) -> None:
        if callback_workers <= 0:
            raise ValueError("callback_workers must be > 0")
        if callback_queue_size <= 0:
            raise ValueError("callback_queue_size must be > 0")
        self._scheduler = sched.scheduler(timefunc=time.monotonic, delayfunc=self._interruptible_sleep)
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._shutdown_event = threading.Event()
        self._wake_event = threading.Event()  # used to interrupt sleep on new schedule
        self._callback_executor = ThreadPoolExecutor(
            max_workers=callback_workers,
            thread_name_prefix="timer-callback",
        )
        self._callback_slots = threading.BoundedSemaphore(
            callback_workers + callback_queue_size
        )
        self._handles: set[TimerHandle] = set()
        self._futures: set[Future[Any]] = set()
        self._thread = threading.Thread(
            target=self._run_loop, name="timer-scheduler", daemon=True
        )
        self._thread.start()

    def schedule(self, delay: float, callback: Callable[[], Any], *, session_id: str = "") -> TimerHandle:
        """Schedule a callback to fire after `delay` seconds.

        Args:
            delay: Seconds from now.
            callback: Work to execute on the bounded callback worker pool.
            session_id: For logging/debugging only.

        Returns:
            TimerHandle that can be passed to cancel().
        """
        if self._shutdown_event.is_set():
            handle = TimerHandle()
            handle._cancelled = True
            return handle

        handle = TimerHandle()

        def _submit_due_callback() -> None:
            self._submit_callback(handle, callback, session_id)

        with self._lock:
            # Close the race where schedule() passed the fast-path check just
            # before shutdown began, then waited for the scheduler lock.
            if self._shutdown_event.is_set():
                handle._cancelled = True
                return handle
            event = self._scheduler.enter(delay, 1, _submit_due_callback)
            handle._event = event
            self._handles.add(handle)

        # Wake the scheduler thread so it recalculates sleep
        self._wake_event.set()
        return handle

    def cancel(self, handle: TimerHandle) -> None:
        """Cancel a scheduled timer. No-op if already fired or cancelled."""
        if handle is None or handle._cancelled:
            return
        future = None
        with self._lock:
            if handle._cancelled:
                return
            handle._cancelled = True
            if handle._event is not None:
                try:
                    self._scheduler.cancel(handle._event)
                except ValueError:
                    pass  # already fired or removed
                handle._event = None
            if handle._future is not None:
                future = handle._future
            self._handles.discard(handle)
        if future is not None:
            future.cancel()

    def shutdown(self, timeout: float = 2.0) -> bool:
        """Stop scheduling and wait up to one total budget for running callbacks.

        Timers and callbacks that have not begun are cancelled. Already-running
        callbacks are allowed to finish. Returns ``True`` only when both the
        timer thread and every submitted callback have reached a terminal state.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        self._shutdown_event.set()
        # Remove future events before waking the worker.  Otherwise
        # sched.scheduler.run() observes the still-pending event after the
        # first wake and goes straight back to sleep, forcing join() to wait
        # for its timeout and leaving the daemon thread alive.
        futures: set[Future[Any]] = set()
        with self._lock:
            for event in list(self._scheduler.queue):
                try:
                    self._scheduler.cancel(event)
                except ValueError:
                    pass
            for handle in tuple(self._handles):
                handle._cancelled = True
                handle._event = None
                if handle._future is not None:
                    futures.add(handle._future)
            futures.update(self._futures)
        for future in futures:
            future.cancel()
        self._wake_event.set()
        self._thread.join(timeout=max(0.0, deadline - time.monotonic()))
        self._callback_executor.shutdown(wait=False, cancel_futures=True)
        remaining = max(0.0, deadline - time.monotonic())
        if futures:
            _done, not_done = wait(futures, timeout=remaining)
        else:
            not_done = set()
        with self._lock:
            live_futures = {future for future in self._futures if not future.done()}
        return not self._thread.is_alive() and not not_done and not live_futures

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

    def _submit_callback(
        self,
        handle: TimerHandle,
        callback: Callable[[], Any],
        session_id: str,
    ) -> None:
        """Submit one due callback without executing it on the timer thread."""
        with self._lock:
            handle._event = None
            if self._shutdown_event.is_set() or handle._cancelled:
                self._handles.discard(handle)
                return
            if not self._callback_slots.acquire(blocking=False):
                # schedule() already accepted this timer. Preserve at-least-once
                # callback admission by retrying shortly rather than dropping it
                # when the bounded worker queue is momentarily saturated.
                handle._event = self._scheduler.enter(
                    _CALLBACK_RETRY_DELAY_SECONDS,
                    1,
                    lambda: self._submit_callback(handle, callback, session_id),
                )
                return
            try:
                future = self._callback_executor.submit(
                    self._invoke_callback,
                    handle,
                    callback,
                    session_id,
                )
            except RuntimeError:
                self._callback_slots.release()
                handle._cancelled = True
                self._handles.discard(handle)
                return
            handle._future = future
            self._futures.add(future)
        future.add_done_callback(lambda _future: self._callback_finished(handle))

    @staticmethod
    def _invoke_callback(
        handle: TimerHandle,
        callback: Callable[[], Any],
        session_id: str,
    ) -> None:
        if handle._cancelled:
            return
        try:
            callback()
        except Exception:
            logger.exception("TimerScheduler callback error (session=%s)", session_id)

    def _callback_finished(self, handle: TimerHandle) -> None:
        self._callback_slots.release()
        with self._lock:
            if handle._future is not None:
                self._futures.discard(handle._future)
            self._handles.discard(handle)


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
