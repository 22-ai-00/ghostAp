"""TTLSet: bounded set with TTL-based expiry and self-contained thread safety.

Extracted from delivery/engine.py for independent testability and reuse.

Thread-safety: TTLSet is internally thread-safe (leaf lock). All public methods
acquire the internal lock before operating. Callers do NOT need to hold an
external lock for individual method calls. However, if callers need atomicity
across multiple TTLSet operations (e.g. add + __contains__), they must use their
own external lock for that compound operation.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable

logger = logging.getLogger(__name__)


class TTLSet:
    """A bounded set where entries expire after a configurable TTL.

    Uses an OrderedDict (insertion-ordered) with monotonic timestamps.
    Expired entries are lazily evicted via:
    - add(): batch-evicts consecutive expired entries from the head.
    - purge(): explicit batch eviction callable by the owner.
    A hard max_size cap ensures memory is bounded even without eviction.

    Thread-safety: Internally holds a threading.Lock (leaf lock — never
    acquired while holding any LockLevel lock). All public methods are safe
    to call concurrently without external synchronization.
    """

    def __init__(
        self,
        ttl: float = 300.0,
        max_size: int = 50_000,
        max_evict_batch: int = 100,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if ttl <= 0:
            raise ValueError("ttl must be positive")
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        if max_evict_batch < 1:
            raise ValueError("max_evict_batch must be >= 1")
        self._ttl = ttl
        self._max_size = max_size
        self._max_evict_batch = max_evict_batch
        self._clock = clock or time.monotonic
        self._entries: OrderedDict[str, float] = OrderedDict()  # key → monotonic timestamp
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

    def add(self, key: str) -> None:
        """Add a key with the current timestamp. Evicts expired/overflow entries."""
        with self._lock:
            # Remove if already present to refresh position
            if key in self._entries:
                self._entries.move_to_end(key)
                self._entries[key] = self._clock()
                return
            self._evict_expired()
            # Hard cap: drop oldest entries if still at capacity
            if len(self._entries) >= self._max_size:
                logger.warning(
                    "TTLSet force-evicting entries, size=%d, max_size=%d",
                    len(self._entries), self._max_size,
                )
            while len(self._entries) >= self._max_size:
                self._entries.popitem(last=False)
            self._entries[key] = self._clock()

    def __contains__(self, key: str) -> bool:
        """Check if key is present and not expired."""
        with self._lock:
            ts = self._entries.get(key)
            if ts is None:
                return False
            if self._clock() - ts > self._ttl:
                return False
            return True

    def purge(self) -> int:
        """Explicitly evict expired entries. Returns the number of entries removed.

        This replaces the previous write-side-effect in __contains__.
        Call this periodically or before capacity-sensitive operations.
        """
        with self._lock:
            return self._evict_expired()

    def _evict_expired(self) -> int:
        """Remove consecutive expired entries from the head (oldest first).

        Limits work per call to max_evict_batch to bound lock-hold time.
        Returns the number of entries evicted.

        NOTE: Caller must hold self._lock.
        """
        now = self._clock()
        evicted = 0
        while self._entries and evicted < self._max_evict_batch:
            # Peek at oldest entry (first item)
            key, ts = next(iter(self._entries.items()))
            if now - ts > self._ttl:
                del self._entries[key]
                evicted += 1
            else:
                break
        return evicted
