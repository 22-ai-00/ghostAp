import logging
import threading
import time
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)


class MessageCache:
    def __init__(
        self,
        ttl: int = 300,
        max_size: int = 1000,
        cleanup_interval: int = 60,
    ):
        self._cache: OrderedDict[str, float] = OrderedDict()
        self._ttl = ttl
        self._max_size = max_size
        self._cleanup_interval = cleanup_interval
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._last_cleanup = time.time()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_event = threading.Event()

    def start_cleanup_thread(self):
        if self._cleanup_thread is not None:
            return
        self._running = True
        self._stop_event.clear()
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True, name="message_cache_cleanup")
        self._cleanup_thread.start()

    def stop_cleanup_thread(self):
        self._running = False
        self._stop_event.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=2)
            self._cleanup_thread = None

    def _cleanup_loop(self):
        while self._running:
            self._stop_event.wait(timeout=self._cleanup_interval)
            if self._running:
                self._do_cleanup()

    def _do_cleanup(self):
        current_time = time.time()
        expired_count = 0

        with self._lock:
            expired_ids = []
            for msg_id, timestamp in self._cache.items():
                if current_time - timestamp > self._ttl:
                    expired_ids.append(msg_id)
                else:
                    break

            for msg_id in expired_ids:
                del self._cache[msg_id]
                expired_count += 1

        if expired_count > 0:
            logger.debug("清理过期消息缓存: %d 条", expired_count)

    def is_duplicate(self, message_id: str) -> bool:
        current_time = time.time()

        with self._lock:
            if current_time - self._last_cleanup > self._cleanup_interval:
                self._quick_cleanup(current_time)
                self._last_cleanup = current_time

            if message_id in self._cache:
                timestamp = self._cache[message_id]
                if current_time - timestamp < self._ttl:
                    return True
                # The per-entry TTL is authoritative.  Periodic cleanup only
                # bounds memory; it must not stretch a 1-second action debounce
                # into the cleanup interval (30 seconds in production).
                del self._cache[message_id]

            self._cache[message_id] = current_time

            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

            return False

    def _quick_cleanup(self, current_time: float):
        cleanup_count = 0
        max_cleanup = 100

        while self._cache and cleanup_count < max_cleanup:
            oldest_id, timestamp = next(iter(self._cache.items()))
            if current_time - timestamp > self._ttl:
                self._cache.pop(oldest_id)
                cleanup_count += 1
            else:
                break

    def clear(self):
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    def contains(self, message_id: str) -> bool:
        with self._lock:
            return message_id in self._cache

    def discard(self, message_id: str) -> None:
        """Remove one reservation after downstream admission fails."""

        with self._lock:
            self._cache.pop(message_id, None)

    def mark_seen(self, message_id: str) -> None:
        """Record a completed message with a fresh duplicate-window timestamp."""

        current_time = time.time()
        with self._lock:
            self._cache.pop(message_id, None)
            self._cache[message_id] = current_time
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
