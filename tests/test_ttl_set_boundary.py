"""Tests for TTLSet boundary conditions.

Validates:
- TTL expiry edge (exact boundary, just expired, not yet expired)
- max_size overflow triggers force eviction
- Refresh semantics (re-add moves to end)
- Invalid constructor args raise ValueError
- purge() returns eviction count
"""

import unittest

from src.card.delivery.ttl_set import TTLSet


class TestTTLSetBoundary(unittest.TestCase):
    """TTLSet boundary and edge-case behavior."""

    def test_key_not_expired_at_exact_ttl(self):
        """Key at exactly ttl seconds should still be considered expired (> not >=)."""
        now = 1000.0
        clock = lambda: now  # noqa: E731
        s = TTLSet(ttl=10.0, clock=clock)
        s.add("a")
        # Advance clock to exactly ttl
        now = 1010.0
        # TTLSet uses `>` so at exactly ttl boundary it's NOT expired
        self.assertIn("a", s)

    def test_key_expired_just_past_ttl(self):
        """Key just past ttl should be expired."""
        now = 1000.0
        clock = lambda: now  # noqa: E731
        s = TTLSet(ttl=10.0, clock=clock)
        s.add("a")
        now = 1010.01
        self.assertNotIn("a", s)

    def test_key_not_expired_before_ttl(self):
        """Key well within ttl is present."""
        now = 1000.0
        clock = lambda: now  # noqa: E731
        s = TTLSet(ttl=10.0, clock=clock)
        s.add("a")
        now = 1005.0
        self.assertIn("a", s)

    def test_max_size_overflow_evicts_oldest(self):
        """Exceeding max_size force-evicts the oldest entry."""
        now = 0.0
        clock = lambda: now  # noqa: E731
        s = TTLSet(ttl=100.0, max_size=3, clock=clock)
        s.add("a")
        now = 1.0
        s.add("b")
        now = 2.0
        s.add("c")
        now = 3.0
        s.add("d")  # Should evict "a"
        self.assertNotIn("a", s)
        self.assertIn("b", s)
        self.assertIn("c", s)
        self.assertIn("d", s)

    def test_refresh_moves_to_end(self):
        """Re-adding an existing key refreshes its timestamp."""
        now = 0.0
        clock = lambda: now  # noqa: E731
        s = TTLSet(ttl=10.0, max_size=3, clock=clock)
        s.add("a")
        now = 1.0
        s.add("b")
        now = 2.0
        s.add("c")
        # Refresh "a" — should move it to end with new timestamp
        now = 3.0
        s.add("a")
        # Now add "d" — should evict "b" (oldest non-refreshed)
        now = 4.0
        s.add("d")
        self.assertNotIn("b", s)
        self.assertIn("a", s)
        self.assertIn("c", s)
        self.assertIn("d", s)

    def test_purge_returns_evicted_count(self):
        """purge() returns how many entries were evicted."""
        now = 0.0
        clock = lambda: now  # noqa: E731
        s = TTLSet(ttl=5.0, clock=clock)
        s.add("a")
        s.add("b")
        now = 1.0
        s.add("c")
        # Expire a and b
        now = 6.0
        count = s.purge()
        self.assertEqual(count, 2)
        self.assertNotIn("a", s)
        self.assertNotIn("b", s)
        self.assertIn("c", s)

    def test_purge_respects_max_evict_batch(self):
        """purge() only evicts up to max_evict_batch per call."""
        now = 0.0
        clock = lambda: now  # noqa: E731
        s = TTLSet(ttl=5.0, max_evict_batch=2, clock=clock)
        for i in range(5):
            s.add(f"k{i}")
        # Expire all
        now = 10.0
        count = s.purge()
        self.assertEqual(count, 2)  # Only 2 evicted due to batch limit
        self.assertEqual(s.purge(), 2)
        self.assertEqual(s.purge(), 1)
        self.assertEqual(s.purge(), 0)

    def test_invalid_ttl_raises(self):
        """ttl <= 0 raises ValueError."""
        with self.assertRaises(ValueError):
            TTLSet(ttl=0)
        with self.assertRaises(ValueError):
            TTLSet(ttl=-1)

    def test_invalid_max_size_raises(self):
        """max_size < 1 raises ValueError."""
        with self.assertRaises(ValueError):
            TTLSet(max_size=0)

    def test_contains_does_not_mutate(self):
        """__contains__ is read-only, does not remove expired entries."""
        now = 0.0
        clock = lambda: now  # noqa: E731
        s = TTLSet(ttl=5.0, clock=clock)
        s.add("x")
        now = 10.0
        # Entry is expired but __contains__ doesn't remove it
        self.assertNotIn("x", s)
        # A later explicit purge still removes it, proving contains was read-only.
        self.assertEqual(s.purge(), 1)


if __name__ == "__main__":
    unittest.main()
