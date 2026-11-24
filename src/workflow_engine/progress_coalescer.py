"""ProgressCoalescer — debounced async progress updates for workflow cards.

Prevents flooding Feishu with card updates by coalescing rapid-fire progress
events into at most one update per PROGRESS_DEBOUNCE_S interval.

Uses a daemon thread + threading.Event pattern:
- enqueue(snapshot) is fire-and-forget, updates _latest_snapshot under lock
- Daemon thread wakes every PROGRESS_DEBOUNCE_S seconds, takes the latest
  snapshot, renders the card, and calls the on_progress callback
- stop() forces a final flush to prevent losing the last update
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

from .constants import PROGRESS_DEBOUNCE_S

logger = logging.getLogger(__name__)


class ProgressCoalescer:
    """Debounces progress card updates using a daemon thread.

    Thread-safe: multiple agent threads may call ``enqueue()`` concurrently.
    The actual callback fires at most once per ``debounce_s`` seconds,
    delivering the most recent snapshot.
    """

    def __init__(
        self,
        on_progress: Callable[[Any], None],
        debounce_s: float = PROGRESS_DEBOUNCE_S,
    ) -> None:
        self._on_progress = on_progress
        self._debounce_s = debounce_s

        # Thread safety
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._delivery_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock (serializes _on_progress callbacks)
        self._latest_snapshot: Optional[Any] = None
        self._last_flush_at: Optional[float] = None
        self._last_immediate_flush: Optional[float] = None
        self._leading_coalesce_s: float = min(0.01, max(0.0, float(debounce_s) * 0.1))
        self._stop_event = threading.Event()
        self._has_work = threading.Event()

        # Daemon thread — dies with the main process
        self._thread = threading.Thread(
            target=self._run,
            name="ProgressCoalescer",
            daemon=True,
        )
        self._thread.start()

    def enqueue(self, snapshot: Any) -> None:
        """Submit a new progress snapshot. Fire-and-forget, non-blocking.

        Stores the snapshot; the daemon thread will pick it up on its next
        wake cycle. If multiple snapshots are enqueued within one debounce
        window, only the latest is delivered.
        """
        if self._stop_event.is_set():
            return

        with self._lock:
            self._latest_snapshot = snapshot
            self._has_work.set()

    def flush_immediate(self, snapshot: Any) -> None:
        """Flush a snapshot immediately, bypassing the debounce timer.

        Used for abort events where the user needs to see cancelled agents
        disappear from '执行中' quickly. Also updates the latest snapshot
        so subsequent debounced deliveries reflect the same state.

        A short min-interval guard prevents runaway spamming when many
        aborts arrive in quick succession.
        """
        if self._stop_event.is_set():
            return

        now = time.monotonic()
        with self._lock:
            # Min-interval guard: skip immediate delivery if we just flushed within 200ms,
            # leaving the snapshot queued for the daemon thread's trailing edge.
            if self._last_immediate_flush is not None and now - self._last_immediate_flush < 0.2:
                self._latest_snapshot = snapshot
                self._has_work.set()
                return
            self._last_immediate_flush = now
            self._last_flush_at = now
            self._latest_snapshot = None
            self._has_work.clear()

        with self._delivery_lock:
            try:
                self._on_progress(snapshot)
            except Exception:
                logger.debug("ProgressCoalescer immediate flush failed", exc_info=True)

    def stop(self) -> None:
        """Stop the coalescer and force-flush any pending snapshot.

        Called on workflow completion to ensure the final state is rendered.
        Blocks until the daemon thread exits.
        """
        if self._stop_event.is_set():
            return

        self._stop_event.set()
        self._has_work.set()

        # Wait for the daemon thread to exit before flushing the final snapshot
        self._thread.join(timeout=2.0)

        with self._lock:
            snapshot = self._latest_snapshot
            self._latest_snapshot = None

        if snapshot is not None:
            with self._delivery_lock:
                try:
                    self._on_progress(snapshot)
                except Exception:
                    logger.debug("ProgressCoalescer final flush failed", exc_info=True)

    def __enter__(self) -> "ProgressCoalescer":
        """Context manager entry — returns self."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        """Context manager exit — stops the coalescer.

        Does not suppress exceptions (returns False).
        """
        self.stop()
        return False

    def __del__(self) -> None:
        """Destructor fallback — best-effort stop if not properly closed.

        Logs a warning if the coalescer was never stopped, then attempts
        cleanup. Defensive against interpreter shutdown where attributes
        may already be gone.
        """
        try:
            if not hasattr(self, "_stop_event"):
                return
            if not self._stop_event.is_set():
                logger.warning(
                    "ProgressCoalescer was not properly stopped; "
                    "call stop() or use as context manager"
                )
                self.stop()
        except Exception:
            # Best-effort only — never let __del__ raise during shutdown
            pass

    # ------------------------------------------------------------------
    # Internal: Daemon thread loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Main daemon thread loop: leading-edge flush on idle->dirty, then trailing debounce."""
        while not self._stop_event.is_set():
            self._has_work.wait()
            if self._stop_event.is_set():
                return

            with self._lock:
                last_flush = self._last_flush_at
            if last_flush is None:
                wait_s = self._leading_coalesce_s
            else:
                elapsed = time.monotonic() - last_flush
                remaining = self._debounce_s - elapsed
                wait_s = remaining if remaining > 0 else self._leading_coalesce_s

            if wait_s > 0 and self._stop_event.wait(timeout=wait_s):
                return

            # Grab the latest snapshot under lock and clear wakeup gate atomically
            with self._lock:
                snapshot = self._latest_snapshot
                self._latest_snapshot = None
                self._has_work.clear()
                if snapshot is not None:
                    self._last_flush_at = time.monotonic()

            if snapshot is not None:
                with self._delivery_lock:
                    try:
                        self._on_progress(snapshot)
                    except Exception:
                        logger.debug("ProgressCoalescer callback failed", exc_info=True)
