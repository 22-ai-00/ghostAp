"""Tests for per-session-key lock in ACPSessionManager (TOCTOU prevention)."""
from __future__ import annotations

import threading
import time

import pytest

from src.acp.helper import SessionKeyCodec
from src.acp.manager import ACPSessionManager


class FakeSession:
    """Fake SyncSession for testing."""

    _instance_count = 0
    _count_lock = threading.Lock()

    def __init__(self):
        with FakeSession._count_lock:
            FakeSession._instance_count += 1
        self.session_id = f"fake-{FakeSession._instance_count}"
        self.last_active = time.time()
        self.message_count = 0
        self._closed = False

    def is_server_running(self):
        return not self._closed

    def close(self):
        self._closed = True

    def to_snapshot(self):
        return {"session_id": self.session_id}

    def load_local_history(self, sid):
        pass

    def describe_agent(self):
        return "test-agent"


def _make_starter(delay: float = 0.1):
    """Create a session_starter callable that returns (session, id, diag) tuple."""
    def starter(*, agent_type, cwd, startup_timeout, model_name=None, session_id=None, project_id=None, **kw):
        time.sleep(delay)  # Simulate startup time
        sess = FakeSession()
        return sess, sess.session_id, {}
    return starter


@pytest.fixture
def manager():
    """Create a minimal ACPSessionManager for testing."""
    FakeSession._instance_count = 0
    mgr = ACPSessionManager(
        agent_type="test",
        session_starter=_make_starter(0.05),
    )
    yield mgr
    mgr.cleanup_all()


class TestPerKeyLockSerialization:
    """Concurrent session creation calls for the same key should not leak sessions."""

    def test_concurrent_start_session_same_key_no_leak(self, manager):
        """Multiple threads starting session with same key: only one session survives."""
        results = []
        errors = []

        def start_thread():
            try:
                session = manager._create_session("chat1", project_id="proj1")
                results.append(session)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=start_thread) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Unexpected errors: {errors}"
        # All threads succeeded (each replaces the previous session)
        assert len(results) == 5

        # But only ONE session should be registered in the manager
        with manager._acquire_lock():
            session_count = len(manager._sessions)
        assert session_count == 1, f"Expected 1 active session, got {session_count}"

    def test_concurrent_start_session_different_keys_parallel(self, manager):
        """Different keys should not block each other."""
        start_times = {}
        end_times = {}

        def start_with_key(key_suffix):
            start_times[key_suffix] = time.monotonic()
            manager._create_session(f"chat{key_suffix}", project_id=f"proj{key_suffix}")
            end_times[key_suffix] = time.monotonic()

        threads = [threading.Thread(target=start_with_key, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # All 3 should have completed
        assert len(end_times) == 3

        # With parallelism, total time should be close to single-session time (~0.05s)
        total_wall = max(end_times.values()) - min(start_times.values())
        # Give generous margin for CI — should be well under 3 * 0.05s sequential
        assert total_wall < 1.0, f"Parallel execution took too long: {total_wall:.2f}s"

    def test_key_lock_cleanup_on_session_end(self, manager):
        """Per-key lock should be cleaned up when session ends."""
        manager._create_session("chat1", project_id="proj1")
        key = SessionKeyCodec.encode("chat1", "proj1")

        # The transient start lock reference should be released after start.
        with manager._key_locks_lock:
            assert key not in manager._key_locks

        # End session
        manager.end_session("chat1", project_id="proj1")

        # Ending the session must not recreate or retain a key lock.
        with manager._key_locks_lock:
            assert key not in manager._key_locks

    def test_replacing_existing_session_keeps_same_key_start_serialized(self):
        """Replacing an old session must not drop the key lock while startup is still running."""
        FakeSession._instance_count = 0
        active_starters = 0
        max_active_starters = 0
        active_lock = threading.Lock()
        first_started = threading.Event()

        def starter(*, agent_type, cwd, startup_timeout, model_name=None, session_id=None, project_id=None, **kw):
            nonlocal active_starters, max_active_starters
            with active_lock:
                active_starters += 1
                max_active_starters = max(max_active_starters, active_starters)
                first_started.set()
            time.sleep(0.2)
            with active_lock:
                active_starters -= 1
            sess = FakeSession()
            return sess, sess.session_id, {}

        mgr = ACPSessionManager(agent_type="test", session_starter=starter)
        key = SessionKeyCodec.encode("chat1", "proj1")
        with mgr._acquire_lock():
            mgr._sessions[key] = FakeSession()
        errors: list[BaseException] = []

        def start_replace():
            try:
                mgr._create_session("chat1", project_id="proj1", startup_timeout=5)
            except BaseException as exc:  # pragma: no cover - assertion reports errors
                errors.append(exc)

        try:
            t1 = threading.Thread(target=start_replace)
            t2 = threading.Thread(target=start_replace)
            t1.start()
            assert first_started.wait(timeout=2)
            t2.start()
            t1.join(timeout=5)
            t2.join(timeout=5)

            assert errors == []
            assert max_active_starters == 1
        finally:
            mgr.cleanup_all()

    def test_end_session_does_not_release_start_owned_key_lock_reference(self):
        """External end_session must not delete a key lock owned by an in-flight session creation."""
        FakeSession._instance_count = 0
        mgr = ACPSessionManager(agent_type="test", session_starter=_make_starter(0.01))
        key = SessionKeyCodec.encode("chat1", "proj1")
        with mgr._acquire_lock():
            mgr._sessions[key] = FakeSession()

        first_start_entered = threading.Event()
        allow_first_to_finish = threading.Event()
        active_starts = 0
        max_active_starts = 0
        active_lock = threading.Lock()
        errors: list[BaseException] = []

        def patched_inner(key_arg, *args, **kwargs):
            nonlocal active_starts, max_active_starts
            with active_lock:
                active_starts += 1
                max_active_starts = max(max_active_starts, active_starts)
                if active_starts == 1:
                    first_start_entered.set()
            allow_first_to_finish.wait(timeout=2)
            sess = FakeSession()
            with mgr._acquire_lock():
                mgr._sessions[key_arg] = sess
            with active_lock:
                active_starts -= 1
            return sess

        mgr._start_session_inner = patched_inner

        def do_start():
            try:
                mgr._create_session("chat1", project_id="proj1", startup_timeout=2)
            except BaseException as exc:  # pragma: no cover - assertion reports errors
                errors.append(exc)

        try:
            t1 = threading.Thread(target=do_start)
            t1.start()
            assert first_start_entered.wait(timeout=2)

            mgr.end_session("chat1", project_id="proj1")

            t2 = threading.Thread(target=do_start)
            t2.start()
            time.sleep(0.1)
            allow_first_to_finish.set()
            t1.join(timeout=3)
            t2.join(timeout=3)

            assert errors == []
            assert max_active_starts == 1
        finally:
            allow_first_to_finish.set()
            mgr.cleanup_all()

    def test_resume_retired_session_loads_exact_provider_session(self, manager):
        """A new transport must load the exact provider session it replaces."""
        key = SessionKeyCodec.encode("chat1", "proj1")
        captured: list[tuple[object, ...]] = []
        resumed = FakeSession()
        resumed.session_id = "provider-session-1"

        def start_inner(*args, **_kwargs):
            captured.append(args)
            with manager._acquire_lock():
                manager._sessions[key] = resumed
            return resumed

        manager._start_session_inner = start_inner

        actual = manager.resume_retired_session(
            "chat1",
            cwd="/repo",
            session_id="provider-session-1",
            startup_timeout=2,
            project_id="proj1",
        )

        assert actual is resumed
        assert len(captured) == 1
        assert captured[0][0] == key
        assert captured[0][3] == "provider-session-1"

    def test_resume_retired_session_refuses_concurrent_owner(self, manager):
        """A stale rollover must never replace a newer chat/session owner."""
        key = SessionKeyCodec.encode("chat1", "proj1")
        concurrent = FakeSession()
        with manager._acquire_lock():
            manager._sessions[key] = concurrent

        with pytest.raises(RuntimeError, match="并发新会话已接管"):
            manager.resume_retired_session(
                "chat1",
                cwd="/repo",
                session_id="provider-session-1",
                startup_timeout=2,
                project_id="proj1",
            )

        with manager._acquire_lock():
            assert manager._sessions[key] is concurrent


class TestKeyLockExceptionRecovery:
    """Key lock should be released even when _start_session_inner raises."""

    def test_key_lock_released_on_start_inner_exception(self):
        """If _start_session_inner raises, key lock must be released so next call proceeds."""
        call_count = {"n": 0}

        def patched_inner(self_mgr, key, chat_id, cwd, session_id, startup_timeout,
                          project_id, agent_type_override, model_name, thread_id):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("startup boom")
            # Second call succeeds
            sess = FakeSession()
            with self_mgr._acquire_lock():
                self_mgr._sessions[key] = sess
            return sess

        FakeSession._instance_count = 0
        mgr = ACPSessionManager(agent_type="test", session_starter=_make_starter(0.01))
        try:
            # Patch _start_session_inner to force a raise
            mgr._start_session_inner = lambda *args, **kw: patched_inner(mgr, *args, **kw)

            # First call should fail
            with pytest.raises(RuntimeError, match="startup boom"):
                mgr._create_session("chat1", project_id="proj1")

            # Second call should succeed (lock was released by finally block)
            session = mgr._create_session("chat1", project_id="proj1")
            assert session is not None
        finally:
            mgr.cleanup_all()

    def test_start_end_concurrent_no_crash(self):
        """session creation and end_session called concurrently should not crash or leak."""
        FakeSession._instance_count = 0
        mgr = ACPSessionManager(
            agent_type="test",
            session_starter=_make_starter(0.02),
        )
        errors = []

        def do_start():
            try:
                mgr._create_session("chat1", project_id="proj1")
            except Exception as e:
                errors.append(e)

        def do_end():
            try:
                time.sleep(0.1)  # Let start complete first
                mgr.end_session("chat1", project_id="proj1")
            except Exception as e:
                errors.append(e)

        try:
            t1 = threading.Thread(target=do_start)
            t2 = threading.Thread(target=do_end)
            t1.start()
            t2.start()
            t1.join(timeout=10)
            t2.join(timeout=10)

            assert not errors, f"Unexpected errors: {errors}"
        finally:
            mgr.cleanup_all()

    def test_key_lock_timeout_raises(self):
        """If key_lock cannot be acquired within timeout, TimeoutError is raised."""
        FakeSession._instance_count = 0

        def slow_inner(self_mgr, key, *args, **kw):
            time.sleep(1)  # Hold the key lock for long enough to exceed startup_timeout
            sess = FakeSession()
            with self_mgr._acquire_lock():
                self_mgr._sessions[key] = sess
            return sess

        mgr = ACPSessionManager(agent_type="test", session_starter=_make_starter(0.01))
        mgr._start_session_inner = lambda *args, **kw: slow_inner(mgr, *args, **kw)
        errors = []

        def start_slow():
            try:
                mgr._create_session("chat1", project_id="proj1", startup_timeout=10)
            except Exception:
                pass

        def start_with_timeout():
            time.sleep(0.1)  # Ensure slow thread acquires lock first
            try:
                mgr._create_session("chat1", project_id="proj1", startup_timeout=0.3)
            except TimeoutError as e:
                errors.append(e)

        try:
            t1 = threading.Thread(target=start_slow)
            t2 = threading.Thread(target=start_with_timeout)
            t1.start()
            t2.start()
            t2.join(timeout=3)
            t1.join(timeout=3)

            assert len(errors) == 1
            assert "当前会话正忙" in str(errors[0])
            assert "稍后重试" in str(errors[0])

            # Timeout must release the transient key-lock reference so the same
            # user/session key can retry successfully after the busy operation.
            retry_session = mgr._create_session("chat1", project_id="proj1", startup_timeout=2)
            assert retry_session is not None
        finally:
            mgr.cleanup_all()
