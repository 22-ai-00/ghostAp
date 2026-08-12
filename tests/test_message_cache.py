import threading
import time

from src.feishu.message_cache import MessageCache


class TestMessageCache:
    def test_is_duplicate_returns_false_for_new_message(self):
        cache = MessageCache(ttl=300, max_size=100)
        assert cache.is_duplicate("msg_001") is False

    def test_is_duplicate_returns_true_for_existing_message(self):
        cache = MessageCache(ttl=300, max_size=100)
        cache.is_duplicate("msg_001")
        assert cache.is_duplicate("msg_001") is True

    def test_size_increases_with_new_messages(self):
        cache = MessageCache(ttl=300, max_size=100)
        assert cache.size() == 0
        cache.is_duplicate("msg_001")
        assert cache.size() == 1
        cache.is_duplicate("msg_002")
        assert cache.size() == 2

    def test_max_size_limit(self):
        cache = MessageCache(ttl=300, max_size=3)
        cache.is_duplicate("msg_001")
        cache.is_duplicate("msg_002")
        cache.is_duplicate("msg_003")
        cache.is_duplicate("msg_004")
        assert cache.size() == 3
        assert cache.contains("msg_001") is False
        assert cache.contains("msg_004") is True

    def test_clear(self):
        cache = MessageCache(ttl=300, max_size=100)
        cache.is_duplicate("msg_001")
        cache.is_duplicate("msg_002")
        assert cache.size() == 2
        cache.clear()
        assert cache.size() == 0

    def test_contains(self):
        cache = MessageCache(ttl=300, max_size=100)
        assert cache.contains("msg_001") is False
        cache.is_duplicate("msg_001")
        assert cache.contains("msg_001") is True

    def test_ttl_expiration(self):
        cache = MessageCache(ttl=1, max_size=100, cleanup_interval=1)
        cache.is_duplicate("msg_001")
        assert cache.contains("msg_001") is True
        time.sleep(1.5)
        cache.is_duplicate("msg_002")
        assert cache.contains("msg_001") is False

    def test_same_key_expires_at_its_ttl_before_periodic_cleanup(self, monkeypatch):
        now = [100.0]
        monkeypatch.setattr("src.feishu.message_cache.time.time", lambda: now[0])
        cache = MessageCache(ttl=1, max_size=100, cleanup_interval=30)

        assert cache.is_duplicate("workflow_add_agent") is False
        now[0] = 101.0

        assert cache.is_duplicate("workflow_add_agent") is False
        assert cache.size() == 1

    def test_zero_ttl_disables_duplicate_window(self, monkeypatch):
        monkeypatch.setattr("src.feishu.message_cache.time.time", lambda: 100.0)
        cache = MessageCache(ttl=0, max_size=100, cleanup_interval=30)

        assert cache.is_duplicate("workflow_add_agent") is False
        assert cache.is_duplicate("workflow_add_agent") is False

    def test_thread_safety(self):
        cache = MessageCache(ttl=300, max_size=1000)
        results = []

        def add_messages(prefix: str, count: int):
            for i in range(count):
                result = cache.is_duplicate(f"{prefix}_{i}")
                results.append((f"{prefix}_{i}", result))

        threads = [
            threading.Thread(target=add_messages, args=("thread1", 100)),
            threading.Thread(target=add_messages, args=("thread2", 100)),
            threading.Thread(target=add_messages, args=("thread3", 100)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert cache.size() == 300
        first_occurrences = [r for r in results if r[1] is False]
        assert len(first_occurrences) == 300

    def test_cleanup_thread_starts_and_stops(self):
        cache = MessageCache(ttl=300, max_size=100, cleanup_interval=1)
        cache.start_cleanup_thread()
        assert cache._cleanup_thread is not None
        assert cache._running is True
        cache.stop_cleanup_thread()
        assert cache._running is False

    def test_quick_cleanup_limits_iterations(self):
        cache = MessageCache(ttl=0, max_size=200, cleanup_interval=0)
        for i in range(150):
            cache.is_duplicate(f"msg_{i}")
        time.sleep(0.1)
        cache.is_duplicate("new_msg")
        assert cache.size() <= 101
