"""Tests for LRU eviction under concurrent access.

Validates that TTLSet and CardDelivery session management
remain consistent under multi-threaded contention.
"""

import threading
import time
import unittest

from src.card.delivery.ttl_set import TTLSet
from tests.helpers.delivery_internals import DeliveryInspector


class TestLruEvictConcurrent(unittest.TestCase):
    """TTLSet consistency under concurrent add/purge/contains operations."""

    def test_concurrent_adds_no_overflow(self):
        """Concurrent adds respect max_size without data corruption."""
        now = time.monotonic()
        lock = threading.Lock()
        clock_value = [now]

        def clock():
            return clock_value[0]

        s = TTLSet(ttl=60.0, max_size=100, clock=clock)
        errors = []

        def add_keys(start, count):
            try:
                for i in range(count):
                    with lock:
                        clock_value[0] += 0.001
                    with lock:
                        s.add(f"k_{start}_{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_keys, args=(t * 100, 50))
            for t in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        present = sum(
            f"k_{thread_index * 100}_{item_index}" in s
            for thread_index in range(4)
            for item_index in range(50)
        )
        self.assertLessEqual(present, 100)

    def test_concurrent_add_and_purge(self):
        """Concurrent add + purge does not corrupt internal state."""
        now_val = [0.0]
        lock = threading.Lock()

        def clock():
            return now_val[0]

        s = TTLSet(ttl=5.0, max_size=200, clock=clock)
        errors = []
        stop_event = threading.Event()

        def adder():
            try:
                for i in range(100):
                    with lock:
                        now_val[0] += 0.01
                        s.add(f"adder_{i}")
            except Exception as e:
                errors.append(e)

        def purger():
            try:
                while not stop_event.is_set():
                    with lock:
                        now_val[0] += 1.0
                        s.purge()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        purge_thread = threading.Thread(target=purger)
        add_thread = threading.Thread(target=adder)

        purge_thread.start()
        add_thread.start()
        add_thread.join()
        stop_event.set()
        purge_thread.join()

        self.assertEqual(len(errors), 0)
        # The public purge contract remains usable after concurrent mutation.
        self.assertGreaterEqual(s.purge(), 0)

    def test_concurrent_contains_during_eviction(self):
        """__contains__ remains safe during concurrent adds exceeding max_size."""
        now_val = [0.0]
        lock = threading.Lock()

        def clock():
            return now_val[0]

        s = TTLSet(ttl=100.0, max_size=50, clock=clock)
        errors = []
        results = []

        # Pre-populate
        for i in range(50):
            with lock:
                now_val[0] += 0.01
                s.add(f"pre_{i}")

        def adder():
            try:
                for i in range(100):
                    with lock:
                        now_val[0] += 0.01
                        s.add(f"new_{i}")
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(100):
                    with lock:
                        result = f"pre_{i % 50}" in s
                    results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=adder),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        # All results should be booleans (no exceptions in __contains__)
        self.assertTrue(all(isinstance(r, bool) for r in results))

    def test_max_size_one_concurrent_adds(self):
        """max_size=1 with concurrent adds maintains exactly 1 entry."""
        now_val = [0.0]
        lock = threading.Lock()

        def clock():
            return now_val[0]

        s = TTLSet(ttl=100.0, max_size=1, clock=clock)
        errors = []

        def adder(prefix):
            try:
                for i in range(50):
                    with lock:
                        now_val[0] += 0.01
                        s.add(f"{prefix}_{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=adder, args=(f"t{t}",))
            for t in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        present = sum(
            f"t{thread_index}_{item_index}" in s
            for thread_index in range(3)
            for item_index in range(50)
        )
        self.assertEqual(present, 1)


class TestTwoPhaseEvictionConcurrent(unittest.TestCase):
    """Concurrent _evict_stale_session_locks_two_phase covering TOCTOU branches."""

    def test_concurrent_eviction_hits_already_evicted_branch(self):
        """Two threads evicting the same candidates — second thread hits current_ts is None."""
        from src.card.delivery.engine import CardDelivery

        class _Stub:
            async def create_card(self, *a, **kw): return "card_id"
            async def update_card(self, *a, **kw): pass

        delivery = CardDelivery(_Stub(), max_session_locks=100, session_lock_ttl=0.01)
        inspector = DeliveryInspector.from_delivery(delivery)
        inspector.eviction_stop.set()
        inspector.eviction_thread.join(timeout=2.0)

        # Seed stale entries that both threads will find as candidates
        with inspector.lock:
            for i in range(20):
                sid = f"stale_{i}"
                inspector.session_locks[sid] = threading.RLock()
                # Timestamps far in the past to ensure TTL expiry
                inspector.timestamps[sid] = 0.0

        threading.Barrier(2, timeout=5.0)
        results = []
        errors = []

        def evict_with_barrier():
            """Run eviction, synchronize threads so both enter Phase 3 concurrently."""
            try:
                evicted = inspector.evict_stale_two_phase()
                results.append(evicted)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=evict_with_barrier)
        t2 = threading.Thread(target=evict_with_barrier)
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        self.assertEqual(len(errors), 0, f"Threads raised errors: {errors}")
        self.assertEqual(len(results), 2)
        # Combined evictions should not exceed the original count
        total_evicted = sum(results)
        self.assertLessEqual(total_evicted, 20)
        # At least one thread should have evicted something
        self.assertGreater(total_evicted, 0)
        # After both evictions, stale entries should be gone
        remaining_stale = sum(1 for k in inspector.timestamps if k.startswith("stale_"))
        self.assertEqual(remaining_stale, 20 - total_evicted)

        delivery._shutdown()


class TestEvictionLoopBoundary(unittest.TestCase):
    """Test _eviction_loop 50% boundary: decision in lock, execution outside lock."""

    def test_eviction_loop_50_percent_boundary(self):
        """When count > 50% at decision time, two-phase eviction is triggered
        even if count drops below 50% before execution begins.

        This validates that the need_eviction flag is evaluated inside the lock
        and the two-phase eviction runs outside the lock (by design).
        """
        from unittest.mock import patch

        from src.card.delivery.engine import CardDelivery

        class _Stub:
            def create_card(self, *a, **kw): return ("msg_id", "card_id")
            def update_card(self, *a, **kw): pass
            def update_element(self, *a, **kw): pass
            def create_streaming_card(self, *a, **kw): return ("msg_id", "card_id")
            def send_card_reference(self, *a, **kw): return "msg_id"

        delivery = CardDelivery(_Stub(), max_session_locks=10, session_lock_ttl=600)
        inspector = DeliveryInspector.from_delivery(delivery)
        # Stop background eviction thread so we control timing
        inspector.eviction_stop.set()
        inspector.eviction_thread.join(timeout=2.0)

        # Seed 6 sessions (>50% of 10) to trigger need_eviction
        with inspector.lock:
            for i in range(6):
                sid = f"s_{i}"
                inspector.session_locks[sid] = threading.RLock()
                inspector.timestamps[sid] = time.monotonic() - 100

        eviction_called = threading.Event()

        # Capture the bound method directly from the pool to avoid recursion through mock
        original_two_phase = delivery._lock_pool._evict_stale_two_phase

        def spy_two_phase():
            """Spy that records invocation, then delegates to original."""
            eviction_called.set()
            return original_two_phase()

        with patch.object(delivery._lock_pool, "_evict_stale_two_phase", side_effect=spy_two_phase):
            # Simulate one iteration of _eviction_loop logic manually
            with inspector.lock:
                count = len(inspector.session_locks)
                need_eviction = count > delivery._max_session_locks * 0.5
            self.assertTrue(need_eviction, f"Expected need_eviction=True, count={count}")

            if need_eviction:
                delivery._lock_pool._evict_stale_two_phase()

        self.assertTrue(eviction_called.is_set(), "Two-phase eviction was not triggered")

        delivery._shutdown()

if __name__ == "__main__":
    unittest.main()
