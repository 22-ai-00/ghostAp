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
        self._latest_snapshot: Optional[Any] = None
        self._stop_event = threading.Event()

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

        with self._lock:
            self._latest_snapshot = snapshot

        # Min-interval guard: skip if we just flushed within 200ms
        now = time.monotonic()
        if hasattr(self, "_last_immediate_flush") and now - self._last_immediate_flush < 0.2:
            return
        self._last_immediate_flush = now

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

        # Force flush one last time before exiting
        with self._lock:
            snapshot = self._latest_snapshot
            self._latest_snapshot = None

        if snapshot:
            try:
                self._on_progress(snapshot)
            except Exception:
                logger.debug("ProgressCoalescer final flush failed", exc_info=True)

        # Wait for the daemon thread to exit (should be fast since we set stop_event)
        self._thread.join(timeout=2.0)

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
        """Main daemon thread loop: wake every debounce_s, render latest snapshot."""
        while not self._stop_event.is_set():
            # Sleep in small increments to allow prompt stop
            sleep_start = time.monotonic()
            while time.monotonic() - sleep_start < self._debounce_s:
                if self._stop_event.is_set():
                    return
                time.sleep(min(0.1, self._debounce_s))

            # Grab the latest snapshot under lock
            with self._lock:
                snapshot = self._latest_snapshot
                self._latest_snapshot = None

            if snapshot is not None:
                try:
                    self._on_progress(snapshot)
                except Exception:
                    logger.debug("ProgressCoalescer callback failed", exc_info=True)
