"""Tests for SessionRotator: atomic session rotation."""

import threading
import time
from unittest.mock import MagicMock, patch

from src.card.events import CardEvent, CardEventType
from src.card.session.rotator import SessionRotator


class TestSessionRotatorSessionId:
    """Task 34: session_id property delegation."""

    def test_session_id_delegates_to_session(self):
        session = MagicMock()
        session.session_id = "abc-123"
        session.closed = False
        rotator = SessionRotator(session)
        assert rotator.session_id == "abc-123"

    def test_session_id_updates_after_rotate(self):
        old = MagicMock()
        old.session_id = "old-id"
        old.closed = False
        new = MagicMock()
        new.session_id = "new-id"
        new.closed = False

        rotator = SessionRotator(old)
        rotator.rotate(lambda: new)
        assert rotator.session_id == "new-id"


class TestSessionRotatorFactoryReturnsNone:
    """Task 34: factory() returning None keeps old session alive."""

    def _make_mock_session(self):
        session = MagicMock()
        session.closed = False
        session.session_id = "s1"
        return session

    def test_factory_returns_none_keeps_old(self):
        old = self._make_mock_session()
        rotator = SessionRotator(old)
        result = rotator.rotate(lambda: None)
        assert result is None
        assert rotator.current is old

    def test_factory_returns_none_dispatch_still_works(self):
        old = self._make_mock_session()
        rotator = SessionRotator(old)
        rotator.rotate(lambda: None)

        event = MagicMock()
        rotator.dispatch(event)
        old.dispatch.assert_called_with(event)

    def test_factory_returns_none_then_success(self):
        old = self._make_mock_session()
        new = self._make_mock_session()
        new.session_id = "new"
        rotator = SessionRotator(old)

        rotator.rotate(lambda: None)
        assert rotator.current is old

        result = rotator.rotate(lambda: new)
        assert result is new
        assert rotator.current is new


class TestSessionRotatorBasic:
    """Basic rotation flow."""

    def _make_mock_session(self):
        session = MagicMock()
        session.closed = False
        return session

    def test_rotate_new_session_receives_dispatch(self):
        """After rotate, dispatch goes to the new session."""
        old = self._make_mock_session()
        new = self._make_mock_session()

        rotator = SessionRotator(old)
        rotator.rotate(lambda: new)

        event = MagicMock()
        rotator.dispatch(event)

        new.dispatch.assert_called_once_with(event)
        # Old session receives an ARCHIVED event (stale stub)
        old.dispatch.assert_called_once()
        stale_event = old.dispatch.call_args[0][0]
        assert stale_event.type == CardEventType.ARCHIVED

    def test_rotate_returns_new_session(self):
        old = self._make_mock_session()
        new = self._make_mock_session()

        rotator = SessionRotator(old)
        result = rotator.rotate(lambda: new)
        assert result is new

    def test_current_property(self):
        session = self._make_mock_session()
        rotator = SessionRotator(session)
        assert rotator.current is session

    def test_close_idempotent(self):
        """Double close does not raise."""
        session = self._make_mock_session()
        rotator = SessionRotator(session)
        rotator.close()
        rotator.close()
        # close called on the session (may be called multiple times)
        assert session.close.call_count >= 1


class TestSessionRotatorConcurrency:
    """Concurrent dispatch does not raise."""

    def _make_mock_session(self):
        session = MagicMock()
        session.closed = False
        return session

    def test_concurrent_dispatch_no_exception(self):
        """10 threads dispatching concurrently should not raise."""
        session = self._make_mock_session()
        dispatch_count = 0
        count_lock = threading.Lock()

        def record_dispatch(_event):
            nonlocal dispatch_count
            with count_lock:
                dispatch_count += 1

        session.dispatch = record_dispatch
        rotator = SessionRotator(session)

        errors = []

        def dispatch_many():
            try:
                for _ in range(100):
                    rotator.dispatch(MagicMock())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=dispatch_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert dispatch_count == 1000

    def test_concurrent_dispatch_during_rotate(self):
        """Dispatch during rotation should not raise."""
        old = self._make_mock_session()
        new = self._make_mock_session()
        rotator = SessionRotator(old)

        errors = []

        def dispatch_loop():
            try:
                for _ in range(50):
                    rotator.dispatch(MagicMock())
            except Exception as e:
                errors.append(e)

        def rotate_once():
            try:
                rotator.rotate(lambda: new)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=dispatch_loop)
        t2 = threading.Thread(target=rotate_once)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == []


class TestSessionRotatorCloseGuard:
    """Tests for close-guard: dispatch/rotate after close are no-ops."""

    def _make_mock_session(self):
        session = MagicMock()
        session.closed = False
        return session

    def test_dispatch_after_close_is_noop(self):
        """dispatch() after close() should not call session.dispatch."""
        session = self._make_mock_session()
        rotator = SessionRotator(session)
        rotator.close()

        # Reset mock to ignore close-related calls
        session.dispatch.reset_mock()

        rotator.dispatch(MagicMock())
        session.dispatch.assert_not_called()

    def test_rotate_after_close_is_noop(self):
        """rotate() after close() should not call factory and returns None."""
        session = self._make_mock_session()
        rotator = SessionRotator(session)
        rotator.close()

        factory = MagicMock()
        result = rotator.rotate(factory)

        factory.assert_not_called()
        assert result is None


class TestSessionRotatorFactoryFailure:
    """Task 25: Factory failure during rotate keeps old session alive."""

    def _make_mock_session(self):
        session = MagicMock()
        session.closed = False
        return session

    def test_factory_exception_returns_none(self):
        """If factory() raises, rotate returns None."""
        old = self._make_mock_session()
        rotator = SessionRotator(old)

        def bad_factory():
            raise RuntimeError("factory crash")

        result = rotator.rotate(bad_factory)
        assert result is None

    def test_factory_exception_preserves_old_session(self):
        """After factory failure, dispatches still go to old session."""
        old = self._make_mock_session()
        rotator = SessionRotator(old)

        def bad_factory():
            raise ConnectionError("cannot create card")

        rotator.rotate(bad_factory)

        event = MagicMock()
        rotator.dispatch(event)
        old.dispatch.assert_called_with(event)


class TestSessionRotatorContentEventNoRetry:
    """Content events (text_delta etc.) should NOT be retried on rotation boundary."""

    def _make_mock_session(self):
        session = MagicMock()
        session.closed = False
        return session

    def test_content_event_no_retry_on_rotation(self):
        """TEXT_DELTA dispatched during rotation should only go to the original session once."""
        old = self._make_mock_session()
        new = self._make_mock_session()
        rotator = SessionRotator(old)

        # Simulate: dispatch grabs old session, then rotation happens during dispatch
        # We can test this by making old.dispatch trigger the rotation
        rotate_done = [False]

        def dispatch_then_rotate(event):
            if not rotate_done[0]:
                rotate_done[0] = True
                rotator.rotate(lambda: new)

        old.dispatch.side_effect = dispatch_then_rotate

        text_delta_event = CardEvent.text_delta("b1", text="hello")
        rotator.dispatch(text_delta_event)

        # old.dispatch was called (at least once for the original dispatch)
        assert any(
            c[0][0] is text_delta_event for c in old.dispatch.call_args_list
        )
        # new.dispatch should NOT have the text_delta (content events skip retry)
        text_calls = [c for c in new.dispatch.call_args_list if c[0][0].type == CardEventType.TEXT_DELTA]
        assert len(text_calls) == 0

    def test_lifecycle_event_retry_on_rotation(self):
        """COMPLETED dispatched during rotation should retry once to new session."""
        old = self._make_mock_session()
        new = self._make_mock_session()
        rotator = SessionRotator(old)

        # Simulate rotation happening during old's dispatch (only first call)
        rotate_done = [False]

        def dispatch_then_rotate(event):
            if not rotate_done[0]:
                rotate_done[0] = True
                rotator.rotate(lambda: new)

        old.dispatch.side_effect = dispatch_then_rotate

        completed_event = CardEvent.completed()
        rotator.dispatch(completed_event)

        # old.dispatch was called with the completed event
        assert any(
            c[0][0] is completed_event for c in old.dispatch.call_args_list
        )
        # new.dispatch should have received COMPLETED once (lifecycle retry)
        new_completed_calls = [c for c in new.dispatch.call_args_list if c[0][0].type == CardEventType.COMPLETED]
        assert len(new_completed_calls) == 1

    def test_reasoning_delta_no_retry(self):
        """REASONING_DELTA should also not be retried."""
        old = self._make_mock_session()
        new = self._make_mock_session()
        rotator = SessionRotator(old)

        rotate_done = [False]

        def dispatch_then_rotate(event):
            if not rotate_done[0]:
                rotate_done[0] = True
                rotator.rotate(lambda: new)

        old.dispatch.side_effect = dispatch_then_rotate

        event = CardEvent.reasoning_delta("b1", text="thinking...")
        rotator.dispatch(event)

        assert any(c[0][0] is event for c in old.dispatch.call_args_list)
        # Filter out ARCHIVED events — only check if REASONING_DELTA was retried
        reasoning_calls = [c for c in new.dispatch.call_args_list if c[0][0].type == CardEventType.REASONING_DELTA]
        assert len(reasoning_calls) == 0

    def test_factory_exception_does_not_close_rotator(self):
        """Factory failure does not set _closed flag."""
        old = self._make_mock_session()
        rotator = SessionRotator(old)

        rotator.rotate(lambda: (_ for _ in ()).throw(ValueError("boom")))
        # Rotator still operational
        assert rotator.current is old

    def test_factory_failure_then_successful_rotate(self):
        """After a failed rotate, a subsequent rotate can still succeed."""
        old = self._make_mock_session()
        new = self._make_mock_session()
        rotator = SessionRotator(old)

        # First rotate fails
        rotator.rotate(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        assert rotator.current is old

        # Second rotate succeeds
        result = rotator.rotate(lambda: new)
        assert result is new
        assert rotator.current is new


class TestRapidConsecutiveRotation:
    """Task 26: Rapid consecutive rotation does not corrupt state."""

    def _make_mock_session(self):
        session = MagicMock()
        session.closed = False
        return session

    def test_rapid_rotation_all_old_sessions_get_completed(self):
        """Rotating 10 times rapidly: each old session gets ARCHIVED."""
        sessions = [self._make_mock_session() for _ in range(11)]
        rotator = SessionRotator(sessions[0])

        for i in range(1, 11):
            rotator.rotate(lambda s=sessions[i]: s)

        # All sessions except the last should have received ARCHIVED
        for i in range(10):
            assert sessions[i].dispatch.called
            stale_event = sessions[i].dispatch.call_args[0][0]
            assert stale_event.type == CardEventType.ARCHIVED

    def test_rapid_rotation_last_session_receives_dispatch(self):
        """After 10 rotations, dispatch goes to the latest session."""
        sessions = [self._make_mock_session() for _ in range(11)]
        rotator = SessionRotator(sessions[0])

        for i in range(1, 11):
            rotator.rotate(lambda s=sessions[i]: s)

        event = MagicMock()
        rotator.dispatch(event)
        sessions[10].dispatch.assert_called_with(event)

    def test_concurrent_rapid_rotation_no_crash(self):
        """Multiple threads rotating concurrently should not raise."""
        import threading

        initial = self._make_mock_session()
        rotator = SessionRotator(initial)

        errors = []

        def rotate_many():
            try:
                for _ in range(20):
                    new_session = self._make_mock_session()
                    rotator.rotate(lambda s=new_session: s)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=rotate_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []

    def test_closed_property_reflects_internal_state(self):
        """SessionRotator.closed property returns _closed flag accurately."""
        session = self._make_mock_session()
        rotator = SessionRotator(session)
        assert rotator.closed is False

        rotator.close()
        assert rotator.closed is True


class TestSessionRotatorArchiveSequence:
    """AC-13: Archived events carry incrementing sequence numbers."""

    def _make_mock_session(self):
        session = MagicMock()
        session.closed = False
        session.session_id = "s1"
        return session

    def test_first_rotate_sends_sequence_1(self):
        old = self._make_mock_session()
        new = self._make_mock_session()
        rotator = SessionRotator(old)
        rotator.rotate(lambda: new)

        # Old session should receive archived event with sequence=1
        old.dispatch.assert_called()
        archived_event = old.dispatch.call_args[0][0]
        assert archived_event.type == CardEventType.ARCHIVED
        assert archived_event.payload.get("sequence") == 1

    def test_sequence_increments_on_multiple_rotations(self):
        s1 = self._make_mock_session()
        s2 = self._make_mock_session()
        s3 = self._make_mock_session()
        s4 = self._make_mock_session()

        rotator = SessionRotator(s1)
        rotator.rotate(lambda: s2)
        rotator.rotate(lambda: s3)
        rotator.rotate(lambda: s4)

        # Check sequence numbers
        s1_archived = s1.dispatch.call_args[0][0]
        assert s1_archived.payload.get("sequence") == 1

        s2_archived = s2.dispatch.call_args[0][0]
        assert s2_archived.payload.get("sequence") == 2

        s3_archived = s3.dispatch.call_args[0][0]
        assert s3_archived.payload.get("sequence") == 3

    def test_archive_stub_contains_page_numbers(self):
        """Archived stub summary includes 第 N 篇 format."""
        s1 = self._make_mock_session()
        s2 = self._make_mock_session()
        rotator = SessionRotator(s1)
        rotator.rotate(lambda: s2)

        archived_event = s1.dispatch.call_args[0][0]
        summary = archived_event.payload.get("summary", "")
        assert "第 1 篇" in summary
        assert "查看最新卡片" in summary

    def test_archive_stub_page_numbers_increment(self):
        """Multi-rotation page numbers increment correctly."""
        s1 = self._make_mock_session()
        s2 = self._make_mock_session()
        s3 = self._make_mock_session()
        rotator = SessionRotator(s1)
        rotator.rotate(lambda: s2)
        rotator.rotate(lambda: s3)

        s1_summary = s1.dispatch.call_args[0][0].payload.get("summary", "")
        s2_summary = s2.dispatch.call_args[0][0].payload.get("summary", "")
        assert "第 1 篇" in s1_summary
        assert "第 2 篇" in s2_summary


class TestSessionRotatorMaxRotationsTruncation:
    """Tests for max_rotations truncation mode."""

    def _make_mock_session(self):
        session = MagicMock()
        session.closed = False
        session.session_id = "s1"
        return session

    @patch("src.card.session.rotator.get_settings")
    def test_truncation_returns_current_session(self, mock_settings):
        """When max_rotations reached, rotate returns current session."""
        mock_settings.return_value.card.session_max_rotations = 2
        s1 = self._make_mock_session()
        s2 = self._make_mock_session()
        s3 = self._make_mock_session()
        rotator = SessionRotator(s1)

        rotator.rotate(lambda: s2)  # rotation_count = 1
        rotator.rotate(lambda: s3)  # rotation_count = 2

        # Now at max — next rotate returns current session
        s4 = self._make_mock_session()
        result = rotator.rotate(lambda: s4)
        assert result is s3  # Returns current, not s4

    @patch("src.card.session.rotator.get_settings")
    def test_truncation_does_not_call_factory(self, mock_settings):
        """In truncation mode, factory is not called."""
        mock_settings.return_value.card.session_max_rotations = 1
        s1 = self._make_mock_session()
        s2 = self._make_mock_session()
        rotator = SessionRotator(s1)

        rotator.rotate(lambda: s2)  # rotation_count = 1, hits max

        factory = MagicMock()
        rotator.rotate(factory)
        factory.assert_not_called()

    @patch("src.card.session.rotator.get_settings")
    def test_truncation_dispatch_still_works(self, mock_settings):
        """After truncation, dispatch still goes to the last valid session."""
        mock_settings.return_value.card.session_max_rotations = 1
        s1 = self._make_mock_session()
        s2 = self._make_mock_session()
        rotator = SessionRotator(s1)

        rotator.rotate(lambda: s2)  # hits max

        event = MagicMock()
        rotator.dispatch(event)
        s2.dispatch.assert_called_with(event)

    @patch("src.card.session.rotator.get_settings")
    def test_truncation_dispatches_warning_event(self, mock_settings):
        """When max_rotations hit, a warning_updated event is dispatched to current session."""
        mock_settings.return_value.card.session_max_rotations = 1
        s1 = self._make_mock_session()
        s2 = self._make_mock_session()
        rotator = SessionRotator(s1)

        rotator.rotate(lambda: s2)  # rotation_count = 1, hits max

        # Next rotate should dispatch truncation warning to s2
        s2.dispatch.reset_mock()
        rotator.rotate(lambda: self._make_mock_session())
        # Verify dispatch was called with a warning event
        assert s2.dispatch.called
        event = s2.dispatch.call_args[0][0]
        assert event.type.value == "warning_updated"

    @patch("src.card.session.rotator.get_settings")
    def test_capacity_rotation_can_bypass_limit_without_archive_content_hint(
        self,
        mock_settings,
    ):
        """Programming capacity rollover can exceed the semantic rotation cap."""
        mock_settings.return_value.card.session_max_rotations = 1
        s1 = self._make_mock_session()
        s2 = self._make_mock_session()
        s3 = self._make_mock_session()
        rotator = SessionRotator(s1)

        rotator.rotate(lambda: s2)
        s2.dispatch.reset_mock()

        result = rotator.rotate(
            lambda: s3,
            enforce_max_rotations=False,
            archive_with_hint=False,
        )

        assert result is s3
        assert rotator.current is s3
        archived = s2.dispatch.call_args[0][0]
        assert archived.type is CardEventType.ARCHIVED
        assert archived.payload["append_hint"] is False


class TestFactoryWallClockWarning:
    """Verify WARNING is logged when factory() takes >500ms."""

    def _make_mock_session(self):
        s = MagicMock()
        s.session_id = "mock"
        s.closed = False
        s.dispatch = MagicMock()
        return s

    def test_slow_factory_triggers_warning(self, caplog):
        """factory() taking >500ms should emit a WARNING log."""
        s1 = self._make_mock_session()
        rotator = SessionRotator(s1)

        s2 = self._make_mock_session()

        def slow_factory():
            time.sleep(0.6)
            return s2

        import logging
        with caplog.at_level(logging.WARNING, logger="src.card.session_rotator"):
            rotator.rotate(slow_factory)

        assert any("500ms" in rec.message for rec in caplog.records), (
            f"Expected WARNING about >500ms, got: {[r.message for r in caplog.records]}"
        )

    def test_fast_factory_no_warning(self, caplog):
        """factory() taking <500ms should NOT emit a WARNING."""
        s1 = self._make_mock_session()
        rotator = SessionRotator(s1)

        s2 = self._make_mock_session()

        import logging
        with caplog.at_level(logging.WARNING, logger="src.card.session_rotator"):
            rotator.rotate(lambda: s2)

        assert not any("500ms" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# FS-12: old_session.dispatch(archived) exception → new session still works
# ---------------------------------------------------------------------------


class TestArchivedDispatchExceptionIsolation:
    """When old_session.dispatch(archived) raises, new session must still function."""

    def _make_mock_session(self, session_id="s"):
        session = MagicMock()
        session.closed = False
        session.session_id = session_id
        return session

    def test_archived_dispatch_exception_new_session_works(self):
        """old_session.dispatch raises → rotation succeeds, new session usable."""
        old = self._make_mock_session("old_sess")
        # Make dispatch raise on archived event
        old.dispatch.side_effect = RuntimeError("archived dispatch exploded")

        rotator = SessionRotator(old)
        new = self._make_mock_session("new_sess")

        result = rotator.rotate(lambda: new)

        # Rotation should succeed despite old dispatch failure
        assert result is new
        assert rotator.current is new
        assert rotator.session_id == "new_sess"

        # New session should accept dispatches normally
        new.dispatch.reset_mock()
        event = CardEvent.started()
        rotator.dispatch(event)
        new.dispatch.assert_called_once_with(event)


# ---------------------------------------------------------------------------
# Factory called outside lock + orphan cleanup tests
# ---------------------------------------------------------------------------


class TestRotateFactoryOutsideLock:
    """Verify factory() is called outside _lock (non-blocking dispatch)."""

    def _make_mock_session(self, session_id="s"):
        session = MagicMock()
        session.closed = False
        session.session_id = session_id
        return session

    def test_rotate_factory_outside_lock(self):
        """factory() must be invoked when _lock is NOT held."""
        old = self._make_mock_session("old")
        rotator = SessionRotator(old)
        lock_was_held: list[bool] = []

        def recording_factory():
            lock_was_held.append(rotator._lock.locked())
            return self._make_mock_session("new")

        rotator.rotate(recording_factory)

        assert lock_was_held == [False], (
            f"factory() was called with lock held={lock_was_held[0]}, expected False"
        )


class TestRotateOrphanCleanup:
    """Verify orphan session cleanup during concurrent rotate."""

    def _make_mock_session(self, session_id="s"):
        session = MagicMock()
        session.closed = False
        session.session_id = session_id
        return session

    def test_rotate_orphan_cleanup_on_concurrent_rotate(self):
        """When two rotates race, the loser's pre-created session is closed."""
        old = self._make_mock_session("old")
        rotator = SessionRotator(old)

        winner_session = self._make_mock_session("winner")
        loser_session = self._make_mock_session("loser")
        barrier = threading.Barrier(2, timeout=5)

        results = {}

        def rotate_winner():
            results["winner"] = rotator.rotate(lambda: winner_session)

        def rotate_loser():
            def slow_factory():
                # Wait for winner to complete first
                barrier.wait()
                time.sleep(0.05)
                return loser_session
            results["loser"] = rotator.rotate(slow_factory)

        t_loser = threading.Thread(target=rotate_loser)
        t_loser.start()
        time.sleep(0.01)  # Let loser start and enter factory

        # Winner completes while loser is in factory
        barrier.wait()
        rotate_winner()

        t_loser.join(timeout=5)

        # The loser should get current session back (winner's), and close orphan
        assert loser_session.close.called, "Orphan session should have been closed"

    def test_rotate_orphan_cleanup_on_close_during_factory(self):
        """When rotator is closed during factory(), pre-created session is closed."""
        old = self._make_mock_session("old")
        rotator = SessionRotator(old)
        orphan = self._make_mock_session("orphan")

        def factory_that_closes_rotator():
            rotator.close()
            return orphan

        result = rotator.rotate(factory_that_closes_rotator)

        assert result is None
        assert orphan.close.called, "Orphan session should have been closed when rotator was closed"


class TestSessionRotatorRetryExceptionIsolation:
    """Verify retry dispatch path isolates exceptions (does not propagate)."""

    def test_retry_path_exception_does_not_propagate(self):
        """When session rotates mid-dispatch and retry raises, error is swallowed."""
        old_session = MagicMock()
        old_session.closed = False
        old_session.session_id = "old"

        rotator = SessionRotator(old_session)

        # Simulate rotation: after first dispatch, swap to new session
        new_session = MagicMock()
        new_session.closed = False
        new_session.session_id = "new"
        new_session.dispatch.side_effect = RuntimeError("session closed")

        event = CardEvent(type=CardEventType.STARTED)

        def trigger_rotation(*args, **kwargs):
            # Swap session during dispatch to trigger retry path
            rotator._session = new_session

        old_session.dispatch.side_effect = trigger_rotation

        # Should NOT raise — retry exception is isolated
        rotator.dispatch(event)

        # Verify retry was attempted on new session
        new_session.dispatch.assert_called_once_with(event)

    def test_closed_rotator_dispatch_is_noop(self):
        """Dispatch to a closed rotator does nothing."""
        session = MagicMock()
        session.closed = False
        session.session_id = "s1"

        rotator = SessionRotator(session)
        rotator.close()

        event = CardEvent(type=CardEventType.STARTED)
        # Should not raise, should be silently ignored
        rotator.dispatch(event)
        session.dispatch.assert_not_called()
