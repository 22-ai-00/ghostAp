"""Lifecycle, cleanup, concurrency, callback, token-capacity, and shutdown tests for RepoLockManager."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Iterator

import pytest

from src.repo_lock import AcquireResult, RepoLockManager
from tests.test_repo_lock import _TMP, isolated_lock_manager


@pytest.fixture()
def mgr() -> Iterator[RepoLockManager]:
    """Create an isolated RepoLockManager (no global singleton side-effects)."""
    with isolated_lock_manager() as manager:
        yield manager


# ---------------------------------------------------------------------------
# Cleanup, idle timeout, and hard-timeout lease tests
# ---------------------------------------------------------------------------


class TestRepoLockCleanup:
    """Cleanup, idle-timeout, and hard-timeout lease renewal tests."""

    def test_idle_timeout_cleanup(self, mgr: RepoLockManager):
        mgr._idle_timeout = 1  # 1 second timeout for testing
        mgr.acquire(f"{_TMP}/repo1", "chat_A")
        # Simulate a leaked lock entry with refcount=0 (e.g. bug in caller)
        with mgr._mu:
            mgr._locks[f"{_TMP}/repo1"].refcount = 0
        time.sleep(1.5)  # Wait for lock to become idle
        mgr._cleanup_idle()
        # Lock should be cleaned up
        result = mgr.acquire(f"{_TMP}/repo1", "chat_B")
        assert result.success is True

    def test_cleanup_skips_active_refcount_on_idle(self, mgr: RepoLockManager):
        """H-1 fix: _cleanup_idle must NOT evict entries with refcount > 0 via idle_timeout.
        Also verifies single refcount case (test_cleanup_idle_skips_active_refcount)."""
        mgr._idle_timeout = 0  # everything is "idle"
        mgr._hard_timeout = 999999  # hard_timeout won't trigger

        # Case 1: refcount=2
        mgr.acquire(f"{_TMP}/repo1", "chat_A")
        mgr.acquire(f"{_TMP}/repo1", "chat_A")  # refcount = 2
        mgr._cleanup_idle()
        locks = mgr.list_locks()
        assert len(locks) == 1
        assert locks[0].refcount == 2
        result = mgr.acquire(f"{_TMP}/repo1", "chat_B")
        assert result.success is False

        # Cleanup for case 2
        mgr.force_release(f"{_TMP}/repo1")

        # Case 2: refcount=1
        mgr.acquire(f"{_TMP}/repo2", "chat_C")  # refcount = 1
        mgr._cleanup_idle()
        locks = mgr.list_locks()
        assert len(locks) == 1
        assert locks[0].refcount == 1
        assert locks[0].chat_id == "chat_C"

    def test_cleanup_evicts_zero_refcount(self, mgr: RepoLockManager):
        """F-01: _cleanup_idle evicts entries with refcount <= 0 and idle."""
        mgr._idle_timeout = 0
        mgr.acquire(f"{_TMP}/repo1", "chat_A")
        mgr.release(f"{_TMP}/repo1", "chat_A")  # refcount = 0, removed by release
        # Re-acquire to get refcount=1, then force refcount=0 without release
        mgr.acquire(f"{_TMP}/repo2", "chat_B")
        # Manually set refcount to 0 for testing
        with mgr._mu:
            mgr._locks[f"{_TMP}/repo2"].refcount = 0
        mgr._cleanup_idle()
        assert mgr.list_locks() == []

    def test_cleanup_idle_skips_recently_touched(self, mgr: RepoLockManager):
        """AC-18: _cleanup_idle must NOT evict entries that are recently touched."""
        mgr._idle_timeout = 1  # 1 second timeout for testing
        mgr.acquire(f"{_TMP}/repo1", "chat_A")
        mgr.acquire(f"{_TMP}/repo1", "chat_A")  # refcount = 2

        time.sleep(0.6)
        mgr.touch(f"{_TMP}/repo1", "chat_A")  # refresh last_active_time
        time.sleep(0.6)
        # Total elapsed > 1s from acquire, but touch was recent (0.6s ago)
        mgr._cleanup_idle()

        # Lock should still be held because touch refreshed last_active_time
        result = mgr.acquire(f"{_TMP}/repo1", "chat_B")
        assert result.success is False  # still held by chat_A

    def test_cleanup_hard_timeout_evicts_active_lock(self, mgr: RepoLockManager, caplog):
        """Leaked active lock with an expired lease is eventually reclaimed."""
        mgr._idle_timeout = 999999  # idle won't trigger
        mgr._hard_timeout = 10
        mgr.acquire(f"{_TMP}/repo1", "chat_A")
        mgr.acquire(f"{_TMP}/repo1", "chat_A")  # refcount = 2
        with mgr._mu:
            entry = mgr._locks[f"{_TMP}/repo1"]
            entry.acquired_at -= 20
            entry.last_active_time -= 20

        with caplog.at_level(logging.CRITICAL, logger="src.repo_lock"):
            mgr._cleanup_idle()

        # Lock should be force-reclaimed by hard_timeout
        assert mgr.list_locks() == []
        result = mgr.acquire(f"{_TMP}/repo1", "chat_B")
        assert result.success is True

        # Verify CRITICAL log emitted for hard-timeout force-reclaim
        critical_msgs = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        assert any("hard-timeout force-reclaimed" in r.message for r in critical_msgs)

    def test_hard_timeout_does_not_reclaim_renewed_long_running_lock(
        self,
        mgr: RepoLockManager,
    ):
        """A heartbeat renews the lease even after total runtime exceeds the cap."""
        mgr._idle_timeout = 300
        mgr._hard_timeout = 3600
        path = f"{_TMP}/repo-long-running"
        mgr.acquire(path, "chat_A", owner_id="run-long")
        with mgr._mu:
            entry = mgr._locks[path]
            entry.acquired_at -= 7200
            entry.last_active_time = time.monotonic()

        mgr._cleanup_idle()

        lock = mgr.get_lock_info(path)
        assert lock is not None
        assert lock.refcount == 1
        conflict = mgr.acquire(
            path,
            "chat_B",
            owner_id="run-other",
        )
        assert conflict.success is False

    def test_heartbeat_interval_is_bounded_by_one_third_of_lease(self):
        with (
            isolated_lock_manager(hard_timeout=3) as short_lease,
            isolated_lock_manager(hard_timeout=3600) as default_lease,
        ):
            assert 0 < short_lease.heartbeat_interval <= 1
            assert default_lease.heartbeat_interval == 30

    def test_cleanup_phase3_warning_log(self, mgr: RepoLockManager, caplog):
        """Phase 3: active lock idle without touch emits WARNING but is NOT evicted."""
        mgr._idle_timeout = 0  # everything is "idle"
        mgr._hard_timeout = 999999  # hard_timeout won't trigger
        mgr.acquire(f"{_TMP}/repo1", "chat_A")  # refcount = 1

        with caplog.at_level(logging.WARNING, logger="src.repo_lock"):
            mgr._cleanup_idle()

        # Lock must NOT be evicted (Phase 3 only warns)
        locks = mgr.list_locks()
        assert len(locks) == 1
        assert locks[0].refcount == 1

        # Verify WARNING log about idle active lock
        warn_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("active lock idle without touch" in r.message for r in warn_msgs)
        assert any("not evicted" in r.message.lower() for r in warn_msgs)

    def test_cleanup_hard_timeout_cleans_token_mapping(self, mgr: RepoLockManager):
        """Phase 2 hard_timeout eviction must also clean path↔token mappings."""
        mgr._idle_timeout = 999999
        mgr._hard_timeout = 0  # everything exceeds hard_timeout
        mgr.acquire(f"{_TMP}/repo1", "chat_A")
        token = mgr.path_to_token(f"{_TMP}/repo1")
        assert token
        assert mgr.token_to_path(token) == f"{_TMP}/repo1"

        mgr._cleanup_idle()

        # Token mapping should be cleaned up after hard-timeout eviction
        assert mgr.token_to_path(token) is None
        with mgr._mu:
            assert f"{_TMP}/repo1" not in mgr._path_to_token


# ---------------------------------------------------------------------------
# Lazy cleanup thread tests
# ---------------------------------------------------------------------------


class TestLazyCleanupThread:
    """Task 23: RepoLockManager cleanup thread is lazily started."""

    def test_lazy_thread_lifecycle(self):
        m = RepoLockManager(idle_timeout=999, cleanup_interval=999)
        try:
            # No thread on construction
            assert m._cleanup_thread is None

            # Thread starts on first acquire
            m.acquire(f"{_TMP}/repo", "chat_A")
            assert m._cleanup_thread is not None
            assert m._cleanup_thread.is_alive()
        finally:
            # Shutdown clears thread ref
            m.shutdown()
            assert m._cleanup_thread is None


# ---------------------------------------------------------------------------
# Concurrency tests
# ---------------------------------------------------------------------------


class TestRepoLockConcurrency:
    """Thread-safety tests for RepoLockManager under concurrent access."""

    def test_concurrent_acquire_same_path(self, mgr: RepoLockManager):
        """Only one chat wins when multiple threads race to acquire the same path."""
        results: dict[str, AcquireResult] = {}
        barrier = threading.Barrier(10)

        def try_acquire(chat_id):
            barrier.wait()
            results[chat_id] = mgr.acquire(f"{_TMP}/contested", chat_id)

        threads = [threading.Thread(target=try_acquire, args=(f"chat_{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        winners = [cid for cid, r in results.items() if r.success]
        losers = [cid for cid, r in results.items() if not r.success]
        assert len(winners) == 1
        assert len(losers) == 9

    def test_concurrent_acquire_release(self, mgr: RepoLockManager):
        """Rapid acquire/release cycles from multiple threads don't corrupt state."""
        errors = []

        def cycle(chat_id, path):
            try:
                for _ in range(50):
                    r = mgr.acquire(path, chat_id)
                    if r.success:
                        mgr.release(path, chat_id)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=cycle, args=(f"chat_{i}", f"{_TMP}/repo_{i % 3}"))
            for i in range(6)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Unexpected errors: {errors}"

    def test_concurrent_hold_execute(self, mgr: RepoLockManager):
        """hold() from multiple threads on different paths is safe."""
        results = []

        def run(idx):
            with mgr.hold(f"{_TMP}/repo_{idx}", f"chat_{idx}"):
                results.append(idx)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sorted(results) == list(range(10))

    def test_cleanup_concurrent_with_acquire_release(self, mgr: RepoLockManager):
        """_cleanup_idle running concurrently with acquire/release must not corrupt state."""
        mgr._idle_timeout = 0
        mgr._hard_timeout = 0
        errors = []

        def cleanup_loop():
            try:
                for _ in range(50):
                    mgr._cleanup_idle()
            except Exception as e:
                errors.append(e)

        def acquire_release_loop(chat_id, path):
            try:
                for _ in range(50):
                    r = mgr.acquire(path, chat_id)
                    if r.success:
                        mgr.touch(path, chat_id)
                        mgr.release(path, chat_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=cleanup_loop)]
        threads += [
            threading.Thread(target=acquire_release_loop, args=(f"chat_{i}", f"{_TMP}/repo_{i % 3}"))
            for i in range(6)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Unexpected errors: {errors}"


class TestCleanupLoopResilience:
    """_cleanup_loop must survive exceptions from _cleanup_idle."""

    def test_cleanup_loop_survives_and_logs_exception(self, mgr: RepoLockManager, caplog):
        """If _cleanup_idle raises, the loop continues and the error is logged."""
        call_count = 0
        original_cleanup_idle = mgr._cleanup_idle

        def flaky_cleanup():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("boom")
            mgr._stop_event.set()
            original_cleanup_idle()

        mgr._cleanup_interval = 0.01
        mgr._cleanup_idle = flaky_cleanup  # type: ignore[assignment]

        with caplog.at_level(logging.ERROR, logger="src.repo_lock"):
            mgr._cleanup_loop()

        assert call_count == 2, f"Expected 2 calls, got {call_count}"
        assert any("RepoLock: cleanup error" in r.message for r in caplog.records)
        assert any("boom" in (r.exc_text or "") for r in caplog.records)


# ---------------------------------------------------------------------------
# Hard-timeout callback tests
# ---------------------------------------------------------------------------


class TestHardTimeoutCallback:
    """Verify on_hard_timeout_reclaim callback invocation and exception safety."""

    def test_callback_invoked_on_hard_timeout(self):
        """Phase 2 hard-timeout eviction invokes the registered callback with (path, chat_id)."""
        reclaimed: list[tuple[str, str]] = []

        def on_reclaim(path: str, chat_id: str) -> None:
            reclaimed.append((path, chat_id))

        with isolated_lock_manager(
            idle_timeout=999999,
            hard_timeout=0,
            on_hard_timeout_reclaim=on_reclaim,
        ) as m:
            m.acquire(f"{_TMP}/repo_cb", "chat_X")
            m._cleanup_idle()

            assert len(reclaimed) == 1
            assert reclaimed[0] == (f"{_TMP}/repo_cb", "chat_X")
            # Lock should be evicted
            assert m.list_locks() == []

    def test_callback_exception_does_not_crash_cleanup(self, caplog):
        """If the callback raises, cleanup still completes and the error is logged."""

        def bad_callback(path: str, chat_id: str) -> None:
            raise RuntimeError("callback exploded")

        with isolated_lock_manager(
            idle_timeout=999999,
            hard_timeout=0,
            on_hard_timeout_reclaim=bad_callback,
        ) as m:
            m.acquire(f"{_TMP}/repo_cb2", "chat_Y")

            with caplog.at_level(logging.ERROR, logger="src.repo_lock"):
                m._cleanup_idle()  # must not raise

            # Lock should still be evicted despite callback failure
            assert m.list_locks() == []

            # Error should be logged
            assert any("SimpleEvent callback failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# AC-R02: Orphan token cleanup + capacity limit
# ---------------------------------------------------------------------------


class TestOrphanTokenCleanup:
    """Verify _cleanup_idle removes orphan token mappings (Phase 4)."""

    def test_cleanup_removes_orphan_tokens_preserves_active(self):
        """Orphan tokens are removed; tokens for active locks survive."""
        with isolated_lock_manager(idle_timeout=999, cleanup_interval=999) as m:
            # Active lock with token
            m.acquire(f"{_TMP}/active_repo", "chatA")
            active_token = m.path_to_token(f"{_TMP}/active_repo")

            # Orphan token (no acquire)
            orphan_token = m.path_to_token(f"{_TMP}/orphan_repo")
            assert orphan_token
            assert m.token_to_path(orphan_token) is not None

            m._cleanup_idle()

            # Active token survives
            assert m.token_to_path(active_token) is not None
            # Orphan is removed
            assert m.token_to_path(orphan_token) is None
            with m._mu:
                assert f"{_TMP}/orphan_repo" not in m._path_to_token


class TestTokenMappingCapacityLimit:
    """Verify path_to_token enforces _TOKEN_MAP_CAPACITY."""

    def test_capacity_limit_purges_orphans_and_respects_active_locks(self):
        with isolated_lock_manager(idle_timeout=999, cleanup_interval=999) as m:
            m._TOKEN_MAP_CAPACITY = 10
            # Fill up with orphan tokens (no acquire)
            for i in range(10):
                m.path_to_token(f"{_TMP}/cap_repo_{i}")
            with m._mu:
                assert len(m._token_to_path) == 10

            # Next token should trigger orphan purge (all are orphans)
            new_token = m.path_to_token(f"{_TMP}/cap_repo_new")
            assert new_token
            with m._mu:
                assert len(m._token_to_path) == 1

            # With active locks: capacity full of non-orphans
            m._TOKEN_MAP_CAPACITY = 5
            for i in range(5):
                m.acquire(f"{_TMP}/locked_repo_{i}", "chatA")
                m.path_to_token(f"{_TMP}/locked_repo_{i}")

            # Adding one more with full capacity - token returned but not cached
            extra_token = m.path_to_token(f"{_TMP}/extra_repo")
            assert extra_token
            assert m.token_to_path(extra_token) is None


class TestRepoLockE2ELifecycle:
    """End-to-end lock lifecycle integration test with a real RepoLockManager."""

    def test_e2e_lock_lifecycle(self, mgr: RepoLockManager):
        """acquire(A) → list_locks visible → release(A) → acquire(B) succeeds."""
        # Step 1: chat_A acquires
        result_a = mgr.acquire(f"{_TMP}/e2e_repo", "chat_A")
        assert result_a.success is True

        # Step 2: lock is visible in list_locks
        locks = mgr.list_locks()
        assert any(lock.chat_id == "chat_A" and f"{_TMP}/e2e_repo" in lock.root_path for lock in locks)

        # Step 3: chat_B cannot acquire while A holds
        result_b_conflict = mgr.acquire(f"{_TMP}/e2e_repo", "chat_B")
        assert result_b_conflict.success is False

        # Step 4: chat_A releases
        mgr.release(f"{_TMP}/e2e_repo", "chat_A")

        # Step 5: lock is no longer visible
        locks_after = mgr.list_locks()
        assert not any(f"{_TMP}/e2e_repo" in lock.root_path for lock in locks_after)

        # Step 6: chat_B can now acquire
        result_b_ok = mgr.acquire(f"{_TMP}/e2e_repo", "chat_B")
        assert result_b_ok.success is True

        # Cleanup
        mgr.release(f"{_TMP}/e2e_repo", "chat_B")

    def test_shutdown_during_active_hold(self):
        """shutdown() while a thread holds a lock must not deadlock or raise."""
        m = RepoLockManager(idle_timeout=300, cleanup_interval=9999)
        errors = []

        def hold_lock():
            try:
                m.acquire(f"{_TMP}/shutdown_repo", "chat_hold")
                time.sleep(2)
                m.release(f"{_TMP}/shutdown_repo", "chat_hold")
            except Exception as e:
                errors.append(e)

        t = threading.Thread(target=hold_lock)
        t.start()
        time.sleep(0.1)  # Ensure thread has acquired the lock

        # shutdown() should not deadlock
        m.shutdown()
        t.join(timeout=5)

        assert not t.is_alive(), "Thread is still alive — possible deadlock"
        assert not errors, f"Unexpected errors during hold: {errors}"


# ---------------------------------------------------------------------------
# Task 16: Blocked chats notification tests
# ---------------------------------------------------------------------------


class TestBlockedChatsNotification:
    """on_release fires with correct blocked chat_ids when lock is released."""

    def test_release_and_force_release_notify_blocked_chats(self, mgr: RepoLockManager):
        """Both release() and force_release() fire on_release with blocked chats."""
        notifications: list[tuple] = []

        def handler(path, chats):
            notifications.append((path, chats))

        mgr.on_release.subscribe(handler)

        # Normal release
        mgr.acquire(f"{_TMP}/notify_repo", "chat_A")
        mgr.acquire(f"{_TMP}/notify_repo", "chat_B")
        mgr.acquire(f"{_TMP}/notify_repo", "chat_C")
        mgr.release(f"{_TMP}/notify_repo", "chat_A")

        assert len(notifications) == 1
        path, chats = notifications[0]
        assert f"{_TMP}/notify_repo" in path or path == os.path.realpath(f"{_TMP}/notify_repo")
        assert "chat_B" in chats
        assert "chat_C" in chats

        # Force release
        notifications.clear()
        mgr.acquire(f"{_TMP}/force_notify", "chat_A")
        mgr.acquire(f"{_TMP}/force_notify", "chat_B")  # blocked
        mgr.force_release(f"{_TMP}/force_notify")

        assert len(notifications) == 1
        assert "chat_B" in notifications[0][1]

    def test_blocked_chats_bounded(self, mgr: RepoLockManager):
        """At most 10 blocked chat_ids are tracked per lock."""
        mgr.acquire(f"{_TMP}/bounded_repo", "holder_chat")

        for i in range(15):
            mgr.acquire(f"{_TMP}/bounded_repo", f"blocked_{i}")

        key = os.path.realpath(f"{_TMP}/bounded_repo")
        with mgr._mu:
            blocked = mgr._blocked_chats.get(key, set())
            assert len(blocked) <= 10


# ---------------------------------------------------------------------------
# Task 17: Singleton thread safety
# ---------------------------------------------------------------------------


class TestSingletonThreadSafety:
    """get_repo_lock_manager() must return the same instance from concurrent threads."""

    def test_concurrent_get_returns_same_instance(self):
        from src.repo_lock import _reset_repo_lock_manager_for_testing, get_repo_lock_manager

        _reset_repo_lock_manager_for_testing()
        try:
            instances: list = []
            barrier = threading.Barrier(10)

            def getter():
                barrier.wait()
                instances.append(id(get_repo_lock_manager()))

            threads = [threading.Thread(target=getter) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All threads should get the same instance
            assert len(set(instances)) == 1
        finally:
            _reset_repo_lock_manager_for_testing()


# ---------------------------------------------------------------------------
# Task 18: Lock after shutdown
# ---------------------------------------------------------------------------


class TestLockAfterShutdown:
    """acquire() after shutdown() must not crash."""

    def test_acquire_after_shutdown_no_crash(self):
        with isolated_lock_manager() as m:
            m.acquire(f"{_TMP}/pre_shutdown", "chat_A")
            m.shutdown()

            # acquire after shutdown should still work (creates new thread)
            result = m.acquire(f"{_TMP}/post_shutdown", "chat_B")
            assert result.success is True


# ---------------------------------------------------------------------------
# Task 37: shutdown_if_active idempotent module-level function
# ---------------------------------------------------------------------------


class TestRepoShutdownIfActive:
    """shutdown_if_active() is idempotent and safe when no instance exists."""

    def test_idempotent_and_no_instance(self):
        from src import repo_lock as _mod

        _orig = _mod._instance
        try:
            # No instance is noop
            _mod._instance = None
            _mod.shutdown_if_active()  # should not raise

            # Double call is idempotent
            with isolated_lock_manager() as m:
                _mod._instance = m
                _mod.shutdown_if_active()
                _mod.shutdown_if_active()  # second call should not raise
        finally:
            _mod._instance = _orig
