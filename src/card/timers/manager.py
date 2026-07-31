"""SessionTimerManager: Extracted timer management from CardSession.

Manages TTL (idle-timeout), prewarning, and terminal-retry timers.
Uses the shared TimerScheduler for O(1) thread overhead across all sessions.
Internal leaf lock protects handle reference reads/writes.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from src.card.timers.scheduler import TimerHandle, get_timer_scheduler

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Maximum retries for TTL expiry lock acquisition
_MAX_TTL_RETRIES = 3


class SessionTimerManager:
    """Manages TTL, prewarning, and retry timers for a CardSession.

    Thread-safety: uses an internal leaf lock (_timer_lock) to protect
    all timer handle reads/writes. This lock is a leaf lock (lowest
    level in the lock hierarchy) and must NOT be held while acquiring
    any LockLevel lock or the owning CardSession's lock.

    Args:
        session_id: Owner session's ID (for logging).
        ttl_seconds: Idle timeout in seconds.
        warn_before_seconds: Seconds before expiry to fire prewarning.
        clock: Monotonic clock callable.
        retry_delay: Delay in seconds for terminal retry timer.
    """

    def __init__(
        self,
        session_id: str,
        ttl_seconds: float = 1800.0,
        clock: Callable[[], float] | None = None,
        retry_delay: float = 3.0,
        timer_factory: Callable | None = None,
        warn_before_seconds: float | None = None,
    ) -> None:
        self._session_id = session_id
        self._timer_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._ttl_seconds = ttl_seconds
        self._warn_before_seconds = warn_before_seconds if warn_before_seconds is not None else ttl_seconds * 0.25
        self._clock = clock or time.monotonic
        self._retry_delay = retry_delay
        # timer_factory is kept for backward compatibility in tests but ignored
        self._timer_factory_legacy = timer_factory

        # Timer handles (protected by _timer_lock)
        self._ttl_handle: TimerHandle | None = None
        self._ttl_prewarning_handle: TimerHandle | None = None
        self._retry_handle: TimerHandle | None = None
        self._terminal_closed: bool = False

        # TTL retry counter (guards against infinite timer spawning)
        self._ttl_retry_count: int = 0

    @property
    def ttl_seconds(self) -> float:
        return self._ttl_seconds

    @property
    def retry_timer(self) -> TimerHandle | None:
        """Backward-compat property (used by tests)."""
        return self._retry_handle

    def reset_ttl_timer(
        self,
        on_expired: Callable[[], None],
        on_prewarning: Callable[[], None],
    ) -> None:
        """Cancel existing TTL timer and schedule a new one (idle-timeout)."""
        scheduler = get_timer_scheduler()
        with self._timer_lock:
            if self._terminal_closed:
                return
            if self._ttl_handle is not None:
                scheduler.cancel(self._ttl_handle)
            if self._ttl_prewarning_handle is not None:
                scheduler.cancel(self._ttl_prewarning_handle)
                self._ttl_prewarning_handle = None

            # Reset retry counter on successful TTL reset
            self._ttl_retry_count = 0

            self._ttl_handle = scheduler.schedule(
                self._ttl_seconds, on_expired, session_id=self._session_id
            )

            # Schedule prewarning: fires warn_before_seconds before expiry
            prewarning_delay = max(0, self._ttl_seconds - self._warn_before_seconds)
            if prewarning_delay > 0:
                self._ttl_prewarning_handle = scheduler.schedule(
                    prewarning_delay, on_prewarning, session_id=self._session_id
                )

    def schedule_ttl_retry(self, on_expired: Callable[[], None]) -> bool:
        """Schedule a retry for TTL expiry lock acquisition failure.

        Returns:
            True if handled (scheduled, or already terminal-fenced); False
            only when an open session has exhausted its retry budget.
        """
        scheduler = get_timer_scheduler()
        with self._timer_lock:
            # True means the request is fully handled.  A terminal-fenced
            # manager needs no retry; False is reserved for retry exhaustion.
            if self._terminal_closed:
                return True
            self._ttl_retry_count += 1
            if self._ttl_retry_count > _MAX_TTL_RETRIES:
                logger.warning(
                    "CardSession %s: TTL retry limit reached (%d), giving up",
                    self._session_id, _MAX_TTL_RETRIES,
                )
                return False

            logger.debug(
                "CardSession %s: TTL timer lock acquisition failed, retry %d/%d",
                self._session_id, self._ttl_retry_count, _MAX_TTL_RETRIES,
            )
            self._ttl_handle = scheduler.schedule(
                5.0, on_expired, session_id=self._session_id
            )
            return True

    def schedule_retry(self, callback: Callable[[], None]) -> None:
        """Schedule a terminal delivery retry timer."""
        scheduler = get_timer_scheduler()
        with self._timer_lock:
            if self._terminal_closed:
                return
            if self._retry_handle is not None:
                scheduler.cancel(self._retry_handle)
            self._retry_handle = scheduler.schedule(
                self._retry_delay, callback, session_id=self._session_id
            )

    def schedule_immediate(self, callback: Callable[[], None]) -> None:
        """Schedule a callback to run immediately (delay=0) on the timer thread."""
        scheduler = get_timer_scheduler()
        with self._timer_lock:
            if self._terminal_closed:
                return
            scheduler.schedule(0, callback, session_id=self._session_id)

    def cancel_all(self) -> None:
        """Cancel all active timers. Thread-safe via internal _timer_lock."""
        scheduler = get_timer_scheduler()
        with self._timer_lock:
            handles = [self._retry_handle, self._ttl_handle, self._ttl_prewarning_handle]
            self._retry_handle = None
            self._ttl_handle = None
            self._ttl_prewarning_handle = None

        # Cancel outside lock
        for h in handles:
            if h is not None:
                scheduler.cancel(h)

    def close(self) -> None:
        """Permanently fence scheduling and cancel every active timer."""
        scheduler = get_timer_scheduler()
        with self._timer_lock:
            self._terminal_closed = True
            handles = [
                self._retry_handle,
                self._ttl_handle,
                self._ttl_prewarning_handle,
            ]
            self._retry_handle = None
            self._ttl_handle = None
            self._ttl_prewarning_handle = None

        for handle in handles:
            if handle is not None:
                scheduler.cancel(handle)
