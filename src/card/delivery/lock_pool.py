"""SessionLockPool: manages per-session RLocks with LRU eviction and drain support."""

from __future__ import annotations

import logging
import threading
import time
import weakref
from collections import OrderedDict
from collections.abc import Callable

logger = logging.getLogger(__name__)


def _eviction_loop_fn(
    weak_pool: weakref.ref, stop_event: threading.Event, interval: float
) -> None:
    """Background eviction thread: periodically cleans stale session locks.

    Uses a weak reference so the thread does not prevent garbage collection.
    """
    while not stop_event.wait(interval):
        pool = weak_pool()
        if pool is None:
            break  # instance GC'd, exit silently
        try:
            pool._periodic_eviction_check()
        except Exception:
            logger.exception("Unexpected error in eviction loop, continuing")
        finally:
            del pool  # release strong ref for next GC cycle


class SessionLockPool:
    """Thread-safe pool of per-session RLocks with LRU/TTL eviction and O(1) drain.

    Public interface:
        acquire(session_id) -> RLock          # get-or-create per-session lock
        release(session_id) -> None           # remove lock entry (on session close)
        try_enter_delivery() -> bool           # atomic admission + in-flight increment
        drain(timeout) -> bool                # wait for all in-flight to finish
        fence() -> None                       # stop accepting new work
        shutdown() -> None                    # stop eviction thread
    """

    def __init__(
        self,
        *,
        max_locks: int = 10_000,
        lock_ttl: float = 600.0,
        eviction_interval: float = 30.0,
        has_active_binding: "Callable[[str], bool] | None" = None,
        purge_callback: "Callable[[], None] | None" = None,
    ) -> None:
        if max_locks <= 0:
            raise ValueError(f"max_session_locks must be > 0, got {max_locks}")
        if lock_ttl <= 0:
            raise ValueError(f"session_lock_ttl must be > 0, got {lock_ttl}")
        if eviction_interval <= 0:
            raise ValueError(f"eviction_interval must be > 0, got {eviction_interval}")

        self._max_locks = max_locks
        self._lock_ttl = lock_ttl

        # Binding checker callback injected at construction (used by eviction logic)
        self._has_active_binding: Callable[[str], bool] = has_active_binding or (lambda sid: False)

        # Optional purge callback invoked during periodic eviction (e.g. TTLSet cleanup)
        self._purge_callback: Callable[[], None] = purge_callback or (lambda: None)

        # Global lock protects _session_locks and _timestamps dicts (O(1) ops only)
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._session_locks: dict[str, threading.RLock] = {}
        self._timestamps: OrderedDict[str, float] = OrderedDict()

        # In-flight counter + condition for O(1) drain
        self._in_flight_condition = threading.Condition(self._lock)
        self._in_flight_count: int = 0

        # Accepting work flag
        self._accepting_work = threading.Event()
        self._accepting_work.set()

        # Background eviction thread
        self._eviction_stop = threading.Event()
        self._eviction_thread = threading.Thread(
            target=_eviction_loop_fn,
            args=(weakref.ref(self), self._eviction_stop, eviction_interval),
            name="session-lock-pool-eviction",
            daemon=True,
        )
        self._eviction_thread.start()

        # Low-watermark full scan: cleans leaked locks even below 50% capacity
        self._full_scan_interval: float = 300.0  # 5 minutes
        self._last_full_scan: float = time.monotonic()

    @property
    def accepting_work(self) -> bool:
        """Whether the pool is accepting new work."""
        return self._accepting_work.is_set()

    def acquire(self, session_id: str) -> threading.RLock:
        """Get or create a per-session RLock, updating LRU timestamp.

        Returns the lock (caller must acquire it themselves for I/O serialization).
        If at capacity and eviction fails, returns a temporary unregistered RLock
        (no-op degradation) and emits a CRITICAL log.
        """
        with self._lock:
            if session_id not in self._session_locks:
                # New session: check capacity
                if len(self._session_locks) >= self._max_locks:
                    self._lru_evict_oldest()
                    if len(self._session_locks) >= self._max_locks:
                        logger.critical(
                            "session lock capacity exhausted (%d/%d) — returning ephemeral lock (no-op degradation)",
                            len(self._session_locks),
                            self._max_locks,
                        )
                        return threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
                self._session_locks[session_id] = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
            # Update LRU timestamp
            self._timestamps[session_id] = time.monotonic()
            self._timestamps.move_to_end(session_id)
            return self._session_locks[session_id]

    def get_existing(self, session_id: str) -> threading.RLock | None:
        """Get existing session lock without creating or updating timestamp."""
        with self._lock:
            return self._session_locks.get(session_id)

    def release(self, session_id: str) -> None:
        """Remove a session lock entry (called on session close)."""
        with self._lock:
            self._session_locks.pop(session_id, None)
            self._timestamps.pop(session_id, None)

    def enter_delivery(self) -> None:
        """Increment in-flight counter for an already-admitted internal caller."""
        with self._in_flight_condition:
            self._in_flight_count += 1

    def try_enter_delivery(self) -> bool:
        """Atomically admit one delivery unless shutdown fencing has begun."""
        with self._in_flight_condition:
            if not self._accepting_work.is_set():
                return False
            self._in_flight_count += 1
            return True

    def exit_delivery(self) -> None:
        """Decrement in-flight counter (call after I/O completes, in finally)."""
        with self._in_flight_condition:
            self._in_flight_count -= 1
            if self._in_flight_count == 0:
                self._in_flight_condition.notify_all()

    def fence(self) -> None:
        """Stop accepting new work (used before drain)."""
        # Admission and fencing share the in-flight condition lock. Therefore
        # a successful drain observes every delivery admitted before this fence,
        # while every later try_enter_delivery() is rejected.
        with self._in_flight_condition:
            self._accepting_work.clear()

    def drain(self, timeout: float = 5.0) -> bool:
        """Wait until all in-flight deliveries complete. O(1) wait.

        Returns True if drained, False on timeout.
        """
        deadline = time.monotonic() + timeout
        with self._in_flight_condition:
            while self._in_flight_count > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.debug(
                        "drain: timeout reached, %d deliveries still in-flight",
                        self._in_flight_count,
                    )
                    return False
                self._in_flight_condition.wait(timeout=remaining)
        return True

    def shutdown(self) -> None:
        """Stop background eviction thread."""
        self._eviction_stop.set()
        if self._eviction_thread.is_alive():
            self._eviction_thread.join(timeout=5.0)

    # ----- Internal eviction logic -----

    def _lru_evict_oldest(self) -> None:
        """Force-evict the oldest session lock entry by LRU order.

        Caller MUST hold self._lock. Skips sessions with active bindings.
        """
        if not self._timestamps:
            return
        skipped: list[tuple[str, float]] = []
        try:
            while self._timestamps:
                oldest_sid, oldest_ts = self._timestamps.popitem(last=False)
                try:
                    has_binding = self._has_active_binding(oldest_sid)
                except Exception:
                    logger.debug("has_active_binding raised for %s, treating as active", oldest_sid)
                    has_binding = True
                if has_binding:
                    skipped.append((oldest_sid, oldest_ts))
                    continue
                # Evict this session
                self._session_locks.pop(oldest_sid, None)
                logger.warning("LRU-evicted session lock for %s (hard cap reached)", oldest_sid)
                return
            logger.warning("LRU eviction failed: all candidates have active bindings")
        finally:
            if skipped:
                for sid, ts in reversed(skipped):
                    self._timestamps[sid] = ts
                    self._timestamps.move_to_end(sid, last=False)

    def _periodic_eviction_check(self) -> None:
        """Called by background thread to check usage and trigger eviction."""
        with self._lock:
            count = len(self._session_locks)
            if count > self._max_locks * 0.9:
                logger.warning(
                    "Session lock usage at %d%% (%d/%d), approaching hard limit",
                    int(count / self._max_locks * 100),
                    count,
                    self._max_locks,
                )
            need_eviction = count > self._max_locks * 0.5
        if need_eviction:
            self._evict_stale_two_phase()
        else:
            self._full_scan_if_needed()
        # Piggyback: invoke purge callback for ancillary cleanup (e.g. TTLSet)
        try:
            self._purge_callback()
        except Exception:
            logger.debug("purge_callback raised, ignoring", exc_info=True)

    def _evict_stale_two_phase(self) -> int:
        """Two-phase eviction: minimizes lock hold time for large session counts.

        Phase 1 (lock held): collect candidate IDs based on timestamp.
        Phase 2 (lock released): check binding status for each candidate.
        Phase 3 (lock re-acquired): re-validate and execute deletion.
        """
        with self._lock:
            now = time.monotonic()
            overflow_count = len(self._session_locks) - int(self._max_locks * 0.8)
            max_evict = min(50, max(1, overflow_count))
            candidates: list[tuple[str, float]] = []

            for sid, ts in list(self._timestamps.items()):
                if len(candidates) >= max_evict:
                    break
                if now - ts > self._lock_ttl:
                    candidates.append((sid, ts))

        if not candidates:
            return 0

        # Phase 2: binding check outside lock
        eligible: list[tuple[str, float]] = []
        for sid, ts in candidates:
            try:
                has_binding = self._has_active_binding(sid)
            except Exception:
                logger.debug("has_active_binding raised for %s in two-phase, skipping", sid)
                continue
            if not has_binding:
                eligible.append((sid, ts))

        if not eligible:
            return 0

        # Phase 3: re-acquire lock, re-validate, delete
        evicted = 0
        with self._lock:
            for sid, original_ts in eligible:
                current_ts = self._timestamps.get(sid)
                if current_ts is None:
                    continue
                if current_ts != original_ts:
                    continue
                self._session_locks.pop(sid, None)
                self._timestamps.pop(sid, None)
                evicted += 1

        if evicted:
            logger.info("Evicted %d stale session lock(s) (two-phase)", evicted)
        return evicted

    def _full_scan_if_needed(self) -> None:
        """Low-frequency full scan for leaked locks below the 50% eviction threshold.

        Runs at most once per _full_scan_interval (default 5 min). Uses the same
        two-phase pattern (collect → check bindings without lock → re-validate).
        Batch size capped at 200 to avoid CPU spikes.
        """
        now = time.monotonic()
        if now - self._last_full_scan < self._full_scan_interval:
            return
        self._last_full_scan = now

        # Phase 1: collect stale candidates (lock held briefly)
        with self._lock:
            scan_now = time.monotonic()
            candidates: list[tuple[str, float]] = []
            for sid, ts in list(self._timestamps.items()):
                if len(candidates) >= 200:
                    break
                if scan_now - ts > self._lock_ttl:
                    candidates.append((sid, ts))

        if not candidates:
            return

        # Phase 2: binding check outside lock
        eligible: list[tuple[str, float]] = []
        for sid, ts in candidates:
            try:
                has_binding = self._has_active_binding(sid)
            except Exception:
                continue
            if not has_binding:
                eligible.append((sid, ts))

        if not eligible:
            return

        # Phase 3: re-acquire lock, re-validate, delete
        evicted = 0
        with self._lock:
            for sid, original_ts in eligible:
                current_ts = self._timestamps.get(sid)
                if current_ts is None or current_ts != original_ts:
                    continue
                self._session_locks.pop(sid, None)
                self._timestamps.pop(sid, None)
                evicted += 1

        if evicted:
            logger.info("Full-scan evicted %d leaked session lock(s)", evicted)
