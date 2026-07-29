"""Bounded executor with queue depth limiting and enqueue time tracking.

Provides a thread-safe wrapper around ThreadPoolExecutor that rejects new
submissions when the pending task count reaches a configured maximum,
preventing unbounded queue growth under load.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

from .exceptions import ExecutorQueueFullError

# Backward compatibility alias (deprecated)
QueueFullError = ExecutorQueueFullError


class BoundedExecutor:
    """ThreadPoolExecutor wrapper with bounded queue depth and enqueue time tracking.

    This executor guarantees that at most *max_queue_size* tasks are pending
    (submitted but not yet completed) at any time.  Attempts to submit beyond
    this limit raise :class:`ExecutorQueueFullError`.

    Each returned :class:`~concurrent.futures.Future` is annotated with an
    ``enqueue_time`` attribute (a :func:`time.time` timestamp) indicating when
    the task was accepted into the queue.

    Thread safety is ensured via an internal lock that makes the
    check-and-submit path atomic, avoiding TOCTOU race conditions.

    Parameters
    ----------
    max_workers:
        Number of worker threads in the underlying pool.
    max_queue_size:
        Maximum number of pending (submitted but incomplete) tasks allowed.
    """

    def __init__(self, max_workers: int, max_queue_size: int) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be >= 1")

        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._max_queue_size = max_queue_size
        self._pending = 0
        self._shutdown = False
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        """Submit a callable for execution if queue capacity allows.

        The submit path is protected by a lock so that the capacity check and
        the actual submission are atomic — no two threads can race past the
        check simultaneously.

        Parameters
        ----------
        fn:
            The callable to execute.
        *args, **kwargs:
            Arguments forwarded to *fn*.

        Returns
        -------
        Future
            A future with an additional ``enqueue_time`` attribute set to the
            :func:`time.time` value at submission.

        Raises
        ------
        ExecutorQueueFullError
            If the number of pending tasks has reached *max_queue_size*.
        """
        with self._lock:
            if self._shutdown:
                raise RuntimeError("executor已关闭")
            if self._pending >= self._max_queue_size:
                raise ExecutorQueueFullError(
                    f"Pending task count ({self._pending}) has reached the "
                    f"maximum queue size ({self._max_queue_size})"
                )

            future: Future = self._executor.submit(fn, *args, **kwargs)
            future.enqueue_time = time.time()  # type: ignore[attr-defined]
            self._pending += 1

        # Register the done callback outside the lock — the callback itself
        # acquires the lock to decrement the counter.
        future.add_done_callback(self._done_callback)
        return future

    def shutdown(
        self,
        wait: bool = True,
        *,
        cancel_futures: bool = False,
    ) -> None:
        """Shut down the underlying thread pool executor.

        Parameters
        ----------
        wait:
            If *True* (default), block until all pending futures complete.
        cancel_futures:
            Cancel work that has not started. Running callables remain
            cooperative and are observed through ``pending_count``.
        """
        with self._lock:
            self._shutdown = True
        self._executor.shutdown(
            wait=wait,
            cancel_futures=cancel_futures,
        )

    @property
    def is_shutdown(self) -> bool:
        """Whether this executor has been shut down."""
        with self._lock:
            return self._shutdown

    @property
    def pending_count(self) -> int:
        """Current number of pending (submitted but incomplete) tasks.

        Useful for observability, metrics, and health checks.
        """
        with self._lock:
            return self._pending

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _done_callback(self, _future: Future) -> None:
        """Decrement pending count when a future completes."""
        with self._lock:
            self._pending -= 1
