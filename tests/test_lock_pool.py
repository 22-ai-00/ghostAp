"""Isolated unit tests for SessionLockPool."""

import threading
import time

import pytest

from src.card.delivery.lock_pool import SessionLockPool


@pytest.fixture
def pool():
    """Create a small pool for testing."""
    p = SessionLockPool(max_locks=5, lock_ttl=1.0, eviction_interval=60.0)
    yield p
    p.shutdown()


class TestAcquire:
    def test_acquire_creates_lock(self, pool: SessionLockPool):
        lock = pool.acquire("s1")
        assert lock is not None
        # Verify it's an RLock by checking it's acquirable
        assert lock.acquire(blocking=False)
        lock.release()
        assert pool.get_existing("s1") is lock

    def test_acquire_same_session_returns_same(self, pool: SessionLockPool):
        lock1 = pool.acquire("s1")
        lock2 = pool.acquire("s1")
        assert lock1 is lock2

    def test_acquire_different_sessions(self, pool: SessionLockPool):
        pool.acquire("s1")
        pool.acquire("s2")
        assert pool.get_existing("s1") is not None
        assert pool.get_existing("s2") is not None

    def test_acquire_at_capacity_evicts_oldest(self, pool: SessionLockPool):
        for i in range(5):
            pool.acquire(f"s{i}")
        # Acquiring a 6th should evict the oldest
        pool.acquire("s5")
        assert pool.get_existing("s0") is None  # oldest evicted
        assert pool.get_existing("s5") is not None

    def test_acquire_at_capacity_with_active_binding_skips(self, pool: SessionLockPool):
        for i in range(5):
            pool.acquire(f"s{i}")
        # Mark s0 as having active binding
        pool._has_active_binding = lambda sid: sid == "s0"
        pool.acquire("s5")
        # s0 should NOT be evicted (active binding), s1 evicted instead
        assert pool.get_existing("s0") is not None
        assert pool.get_existing("s1") is None


class TestRelease:
    def test_release_removes_lock(self, pool: SessionLockPool):
        pool.acquire("s1")
        pool.release("s1")
        assert pool.get_existing("s1") is None

    def test_release_nonexistent_is_noop(self, pool: SessionLockPool):
        pool.release("nonexistent")  # Should not raise
        assert pool.get_existing("nonexistent") is None


class TestFence:
    def test_fence_clears_accepting_work(self, pool: SessionLockPool):
        assert pool.accepting_work is True
        pool.fence()
        assert pool.accepting_work is False


class TestDrain:
    def test_drain_returns_true_when_no_in_flight(self, pool: SessionLockPool):
        assert pool.drain(timeout=1.0) is True

    def test_drain_waits_for_in_flight(self, pool: SessionLockPool):
        pool.enter_delivery()
        result = [None]

        def release_after_delay():
            time.sleep(0.1)
            pool.exit_delivery()

        t = threading.Thread(target=release_after_delay)
        t.start()
        result[0] = pool.drain(timeout=2.0)
        t.join()
        assert result[0] is True

    def test_drain_timeout_returns_false(self, pool: SessionLockPool):
        pool.enter_delivery()
        assert pool.drain(timeout=0.05) is False
        pool.exit_delivery()  # cleanup


class TestEnterExitDelivery:
    def test_enter_exit_balanced(self, pool: SessionLockPool):
        pool.enter_delivery()
        pool.enter_delivery()
        pool.exit_delivery()
        pool.exit_delivery()
        assert pool.drain(timeout=0.1) is True

    def test_exit_notifies_waiters(self, pool: SessionLockPool):
        pool.enter_delivery()
        drained = [False]

        def waiter():
            drained[0] = pool.drain(timeout=2.0)

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.05)
        pool.exit_delivery()
        t.join()
        assert drained[0] is True


class TestShutdown:
    def test_shutdown_stops_eviction_thread(self, pool: SessionLockPool):
        assert pool._eviction_thread.is_alive()
        pool.shutdown()
        assert not pool._eviction_thread.is_alive()

    def test_shutdown_idempotent(self, pool: SessionLockPool):
        pool.shutdown()
        pool.shutdown()  # Should not raise


class TestGetExisting:
    """Tests for get_existing(): retrieve lock without creating."""

    def test_returns_rlock_after_acquire(self, pool: SessionLockPool):
        """After acquire(), get_existing() returns the same RLock."""
        acquired = pool.acquire("s1")
        existing = pool.get_existing("s1")
        assert existing is acquired

    def test_returns_none_when_not_acquired(self, pool: SessionLockPool):
        """get_existing() returns None for unknown session_id."""
        assert pool.get_existing("nonexistent") is None

    def test_returns_none_after_release(self, pool: SessionLockPool):
        """After release(), get_existing() returns None."""
        pool.acquire("s1")
        pool.release("s1")
        assert pool.get_existing("s1") is None


class TestHasActiveBindingCallback:
    """Tests for _has_active_binding callback resilience."""

    def test_callback_exception_does_not_crash_eviction(self):
        """When callback raises, eviction skips that session and returns ephemeral lock (no-op degradation)."""
        def exploding_callback(sid: str) -> bool:
            raise RuntimeError("simulated callback failure")

        pool = SessionLockPool(
            max_locks=3, lock_ttl=1.0, eviction_interval=60.0,
            has_active_binding=exploding_callback,
        )
        try:
            # Fill pool to capacity
            pool.acquire("s0")
            pool.acquire("s1")
            pool.acquire("s2")
            # Attempting to acquire a 4th should trigger LRU eviction
            # The callback raises on every candidate, so eviction fails gracefully
            # Pool returns an ephemeral lock (no-op degradation) instead of raising
            lock = pool.acquire("s3")
            assert lock is not None  # ephemeral lock returned
        finally:
            pool.shutdown()

    def test_callback_returns_non_bool_truthy(self):
        """When callback returns a truthy non-bool (e.g. 1), session is treated as active."""
        def truthy_callback(sid: str):
            return 1  # truthy non-bool

        pool = SessionLockPool(
            max_locks=2, lock_ttl=1.0, eviction_interval=60.0,
            has_active_binding=truthy_callback,
        )
        try:
            pool.acquire("s0")
            pool.acquire("s1")
            # Both return truthy → none can be evicted → ephemeral lock returned
            lock = pool.acquire("s2")
            assert lock is not None  # no-op degradation
        finally:
            pool.shutdown()

    def test_callback_returns_non_bool_falsy(self):
        """When callback returns a falsy non-bool (e.g. 0, None), session can be evicted."""
        def falsy_callback(sid: str):
            return 0  # falsy non-bool

        pool = SessionLockPool(
            max_locks=2, lock_ttl=1.0, eviction_interval=60.0,
            has_active_binding=falsy_callback,
        )
        try:
            pool.acquire("s0")
            pool.acquire("s1")
            # Both return falsy → oldest can be evicted
            pool.acquire("s2")
            assert pool.get_existing("s0") is None  # oldest evicted
            assert pool.get_existing("s2") is not None
        finally:
            pool.shutdown()

    def test_concurrent_eviction_callback_safe(self):
        """Concurrent eviction triggers do not corrupt state when callback is called."""
        import threading

        call_count = {"n": 0}
        lock = threading.Lock()

        def counting_callback(sid: str) -> bool:
            with lock:
                call_count["n"] += 1
            time.sleep(0.001)  # Simulate work
            return False

        pool = SessionLockPool(
            max_locks=5, lock_ttl=0.01, eviction_interval=60.0,
            has_active_binding=counting_callback,
        )
        try:
            # Fill and backdate
            for i in range(5):
                pool.acquire(f"s{i}")
            with pool._lock:
                for i in range(5):
                    pool._timestamps[f"s{i}"] = time.monotonic() - 100

            # Trigger eviction from multiple threads
            errors = []
            def evict_worker():
                try:
                    pool._evict_stale_two_phase()
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=evict_worker) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            assert not errors, f"Concurrent eviction errors: {errors}"
            # Callback was called at least once
            assert call_count["n"] > 0
        finally:
            pool.shutdown()


# ---------------------------------------------------------------------------
# Capacity exhaustion no-op degradation
# ---------------------------------------------------------------------------


class TestCapacityExhaustionNoOpDegradation:
    """Verify pool returns ephemeral lock and logs CRITICAL when capacity exhausted."""

    def test_acquire_at_capacity_returns_lock_not_raises(self, caplog):
        """When all sessions are active and pool is full, acquire returns ephemeral lock."""
        import logging

        def always_active(sid: str) -> bool:
            return True  # all sessions have active bindings

        pool = SessionLockPool(
            max_locks=2, lock_ttl=600.0, eviction_interval=60.0,
            has_active_binding=always_active,
        )
        try:
            pool.acquire("s0")
            pool.acquire("s1")

            with caplog.at_level(logging.CRITICAL):
                # Should NOT raise RuntimeError
                lock = pool.acquire("overflow_session")

            assert lock is not None
            # Verify the ephemeral lock is not registered in the pool
            assert pool.get_existing("overflow_session") is None
            # Verify CRITICAL log was emitted
            assert any("capacity exhausted" in r.message for r in caplog.records if r.levelno == logging.CRITICAL)
        finally:
            pool.shutdown()
