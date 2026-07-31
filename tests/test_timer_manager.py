"""Tests for SessionTimerManager: TTL, prewarning, retry scheduling."""

import threading
import time
from unittest.mock import MagicMock

from src.card.timers.manager import _MAX_TTL_RETRIES, SessionTimerManager


class TestTimerManagerBasic:
    """Basic timer lifecycle tests."""

    def test_reset_ttl_timer_creates_timer(self):
        """reset_ttl_timer schedules TTL and prewarning timers."""
        mgr = SessionTimerManager(session_id="s1", ttl_seconds=10.0)

        on_expired = MagicMock()
        on_prewarning = MagicMock()
        mgr.reset_ttl_timer(on_expired, on_prewarning)

        assert mgr._ttl_handle is not None
        assert mgr._ttl_prewarning_handle is not None
        # Cleanup
        mgr.cancel_all()

    def test_cancel_all_stops_timers(self):
        """cancel_all cancels all active timers."""
        mgr = SessionTimerManager(session_id="s1", ttl_seconds=10.0)

        on_expired = MagicMock()
        on_prewarning = MagicMock()
        mgr.reset_ttl_timer(on_expired, on_prewarning)
        mgr.schedule_retry(MagicMock())

        mgr.cancel_all()

        assert mgr._ttl_handle is None
        assert mgr._ttl_prewarning_handle is None
        assert mgr._retry_handle is None

    def test_schedule_retry_creates_handle(self):
        """schedule_retry creates a timer handle."""
        mgr = SessionTimerManager(session_id="s1", retry_delay=0.5)

        callback = MagicMock()
        mgr.schedule_retry(callback)

        assert mgr._retry_handle is not None
        assert not mgr._retry_handle.cancelled
        mgr.cancel_all()

    def test_terminal_close_permanently_fences_all_future_scheduling(self):
        """Terminal cleanup cannot be undone by a stale callback."""
        mgr = SessionTimerManager(
            session_id="terminal-fence",
            ttl_seconds=10.0,
            retry_delay=0.5,
        )
        mgr.reset_ttl_timer(MagicMock(), MagicMock())
        mgr.schedule_retry(MagicMock())

        mgr.close()
        mgr.reset_ttl_timer(MagicMock(), MagicMock())
        ttl_retry_handled = mgr.schedule_ttl_retry(MagicMock())
        mgr.schedule_retry(MagicMock())

        # True means handled: after terminal close no retry is needed, while
        # False remains reserved for an open session exhausting its retry cap.
        assert ttl_retry_handled is True
        assert mgr._ttl_handle is None
        assert mgr._ttl_prewarning_handle is None
        assert mgr._retry_handle is None


class TestTTLRetry:
    """TTL retry scheduling logic."""

    def test_schedule_ttl_retry_increments_counter(self):
        """Each call increments the retry counter."""
        mgr = SessionTimerManager(session_id="s1")

        result = mgr.schedule_ttl_retry(MagicMock())
        assert result is True
        assert mgr._ttl_retry_count == 1
        mgr.cancel_all()

    def test_schedule_ttl_retry_exceeds_max(self):
        """Returns False when max retries exceeded."""
        mgr = SessionTimerManager(session_id="s1")

        for _ in range(_MAX_TTL_RETRIES):
            result = mgr.schedule_ttl_retry(MagicMock())
            assert result is True

        # Next one should fail
        result = mgr.schedule_ttl_retry(MagicMock())
        assert result is False
        mgr.cancel_all()

    def test_reset_ttl_resets_retry_counter(self):
        """reset_ttl_timer resets the retry counter to 0."""
        mgr = SessionTimerManager(session_id="s1", ttl_seconds=10.0)

        mgr.schedule_ttl_retry(MagicMock())
        mgr.schedule_ttl_retry(MagicMock())
        assert mgr._ttl_retry_count == 2

        mgr.reset_ttl_timer(MagicMock(), MagicMock())
        assert mgr._ttl_retry_count == 0
        mgr.cancel_all()


class TestPrewarningTiming:
    """Verify prewarning is scheduled at 90% of TTL."""

    def test_prewarning_fires_before_expiry(self):
        """Prewarning timer fires before the main TTL timer."""
        mgr = SessionTimerManager(session_id="s1", ttl_seconds=0.2)

        prewarning_called = threading.Event()
        expired_called = threading.Event()

        def on_prewarning():
            prewarning_called.set()

        def on_expired():
            expired_called.set()

        mgr.reset_ttl_timer(on_expired, on_prewarning)

        # Prewarning at 90% = 0.18s, expiry at 0.2s
        prewarning_called.wait(timeout=2.0)
        assert prewarning_called.is_set()
        expired_called.wait(timeout=2.0)
        assert expired_called.is_set()


class TestScheduleRetryCancelsOld:
    """Verify schedule_retry cancels the old timer before creating a new one."""

    def test_consecutive_schedule_retry_cancels_previous(self):
        """Calling schedule_retry twice cancels the first timer's callback."""
        mgr = SessionTimerManager(session_id="s1", retry_delay=0.1)

        first_callback = MagicMock()
        second_callback = MagicMock()

        mgr.schedule_retry(first_callback)
        # Immediately schedule another — old should be cancelled
        mgr.schedule_retry(second_callback)

        # Wait enough for both timers to have fired if not cancelled
        time.sleep(0.3)

        # First callback should NOT have been called (cancelled)
        first_callback.assert_not_called()
        # Second callback should have fired
        second_callback.assert_called_once()
        mgr.cancel_all()

    def test_cancel_all_prevents_callbacks(self):
        """cancel_all prevents pending timers from firing."""
        mgr = SessionTimerManager(session_id="s1", ttl_seconds=100.0)

        on_expired = MagicMock()
        on_prewarning = MagicMock()
        mgr.reset_ttl_timer(on_expired, on_prewarning)
        mgr.schedule_retry(MagicMock())

        # cancel_all should complete without hanging
        mgr.cancel_all()

        # After cancel_all, no callbacks should fire
        time.sleep(0.1)
        on_expired.assert_not_called()
        on_prewarning.assert_not_called()
