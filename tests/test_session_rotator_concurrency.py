"""Tests for SessionRotator concurrency safety.

Verifies atomic rotation, close-during-rotate race, and dispatch-after-close.
"""

import threading

from src.card.events import CardEventType
from src.card.session.rotator import SessionRotator


class _FakeSession:
    """Minimal fake CardSession for rotator tests."""

    def __init__(self, session_id="fake"):
        self.session_id = session_id
        self.dispatched = []
        self.closed = False

    def dispatch(self, event):
        self.dispatched.append(event)

    def close(self):
        self.closed = True


class TestSessionRotatorRotation:
    """Rotation correctness."""

    def test_rotate_swaps_session(self):
        """After rotate(), dispatch goes to the new session."""
        old_session = _FakeSession("old")
        new_session = _FakeSession("new")
        rotator = SessionRotator(old_session)

        result = rotator.rotate(lambda: new_session)

        assert result is new_session
        assert rotator.current is new_session

    def test_rotate_dispatches_stale_stub_to_old(self):
        """Old session receives a completion event after rotation."""
        old_session = _FakeSession("old")
        new_session = _FakeSession("new")
        rotator = SessionRotator(old_session)

        rotator.rotate(lambda: new_session)

        assert len(old_session.dispatched) == 1
        assert old_session.dispatched[0].type == CardEventType.ARCHIVED

    def test_rotate_after_close_returns_none(self):
        """rotate() returns None if rotator is already closed."""
        session = _FakeSession()
        rotator = SessionRotator(session)
        rotator.close()

        result = rotator.rotate(lambda: _FakeSession("new"))

        assert result is None


class TestSessionRotatorRaceConditions:
    """Concurrent access safety."""


    def test_close_during_rotate_cleans_up_new_session(self):
        """If close() happens during factory(), new session is cleaned up."""
        old_session = _FakeSession("old")
        rotator = SessionRotator(old_session)

        barrier = threading.Barrier(2, timeout=5)
        new_session = _FakeSession("new")

        def slow_factory():
            barrier.wait()  # Wait for close to happen
            return new_session

        def close_rotator():
            barrier.wait()  # Sync with factory
            rotator.close()

        close_thread = threading.Thread(target=close_rotator)
        close_thread.start()

        result = rotator.rotate(slow_factory)
        close_thread.join()

        # Either close won (result is None, new_session closed)
        # or rotate won (result is new_session)
        # Both are valid outcomes — the key invariant is no crash
        assert result is None or result is new_session
        # Strengthen: if close won the race, new_session must be cleaned up
        if result is None:
            assert new_session.closed, "new_session should be closed when rotate returns None"

    def test_concurrent_rotate_no_orphan_sessions(self):
        """Two concurrent rotate() calls must not produce orphan sessions.

        With the lock-held-factory pattern, only one rotate can proceed at a time,
        so no orphan session should be created.
        """
        old_session = _FakeSession("old")
        rotator = SessionRotator(old_session)

        sessions_created = []
        barrier = threading.Barrier(2, timeout=5)

        def factory():
            s = _FakeSession(f"new_{len(sessions_created)}")
            sessions_created.append(s)
            return s

        def do_rotate():
            barrier.wait()
            rotator.rotate(factory)

        t1 = threading.Thread(target=do_rotate)
        t2 = threading.Thread(target=do_rotate)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        # Both rotates succeed sequentially (lock serializes them)
        # The final active session should be the last one created
        assert rotator.current in sessions_created
        # No leak: every session that was created and is NOT the current
        # must have been cleaned up — either ARCHIVED (lost the swap as "old")
        # or closed (lost the CAS race as orphan).
        for s in sessions_created:
            if s is not rotator.current:
                archived_events = [e for e in s.dispatched if e.type == CardEventType.ARCHIVED]
                was_archived = len(archived_events) == 1
                was_closed = s.closed
                assert was_archived or was_closed, (
                    f"Session {s.session_id} was neither archived nor closed — resource leak"
                )
        # old_session (the initial one) must have been archived by the first rotate
        archived_old = [e for e in old_session.dispatched if e.type == CardEventType.ARCHIVED]
        assert len(archived_old) == 1, "Initial old_session should have received ARCHIVED"
