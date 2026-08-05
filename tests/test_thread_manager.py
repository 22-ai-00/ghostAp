"""Tests for thread module — ThreadContext, ThreadContextManager, thread-local helpers."""

import threading
import time

from src.thread.manager import (
    ThreadContextManager,
    _current_thread_id,
    get_current_thread_id,
    get_thread_manager,
    set_current_thread_id,
)
from src.thread.models import ThreadContext


class TestThreadContext:

    def test_creation_with_defaults(self):
        ctx = ThreadContext(thread_root_id="root1", chat_id="chat1", project_id="proj1")
        assert ctx.thread_root_id == "root1"
        assert ctx.chat_id == "chat1"
        assert ctx.project_id == "proj1"
        assert ctx.mode == "smart"
        assert ctx.tool_name is None
        assert ctx.model_name is None
        assert isinstance(ctx.created_at, float)
        assert isinstance(ctx.last_active, float)

    def test_touch_updates_last_active(self):
        ctx = ThreadContext(thread_root_id="root1", chat_id="chat1", project_id="proj1")
        old_active = ctx.last_active
        time.sleep(0.05)
        ctx.touch()
        assert ctx.last_active > old_active

    def test_session_key_suffix(self):
        ctx = ThreadContext(thread_root_id="abc123", chat_id="chat1", project_id="proj1")
        assert ctx.session_key_suffix == "t:abc123"

    def test_session_key_suffix_roundtrip_with_acp_manager(self):
        """ThreadContext.session_key_suffix 应与 ACPSessionManager 的编码/解析协议对齐。"""

        from src.acp.manager import ACPSessionManager

        ctx = ThreadContext(thread_root_id="thread-xyz", chat_id="chat-1", project_id="proj-1")

        # 通过 ACPSessionManager 生成带线程维度的 session_key
        key = ACPSessionManager._session_key(ctx.chat_id, ctx.project_id, thread_id=ctx.thread_root_id)
        chat_id, project_id, thread_id = ACPSessionManager._parse_session_key(key)

        assert chat_id == ctx.chat_id
        assert project_id == ctx.project_id
        assert thread_id == ctx.thread_root_id

    def test_to_dict(self):
        ctx = ThreadContext(
            thread_root_id="root1",
            chat_id="chat1",
            project_id="proj1",
            mode="coco",
            tool_name="claude",
            model_name="claude-3.5-sonnet",
        )
        d = ctx.to_dict()
        assert d["thread_root_id"] == "root1"
        assert d["chat_id"] == "chat1"
        assert d["project_id"] == "proj1"
        assert d["mode"] == "coco"
        assert d["tool_name"] == "claude"
        assert d["model_name"] == "claude-3.5-sonnet"
        assert "created_at" in d
        assert "last_active" in d

    def test_to_dict_from_dict_roundtrip(self):
        ctx = ThreadContext(
            thread_root_id="root1",
            chat_id="chat1",
            project_id="proj1",
            mode="shell",
            tool_name="coco",
            model_name="gpt-4.1",
        )
        d = ctx.to_dict()
        restored = ThreadContext.from_dict(d)
        assert restored.thread_root_id == ctx.thread_root_id
        assert restored.chat_id == ctx.chat_id
        assert restored.project_id == ctx.project_id
        assert restored.mode == ctx.mode
        assert restored.tool_name == ctx.tool_name
        assert restored.model_name == ctx.model_name
        assert restored.created_at == ctx.created_at
        assert restored.last_active == ctx.last_active

    def test_from_dict_with_missing_optional_fields(self):
        data = {
            "thread_root_id": "root1",
            "chat_id": "chat1",
            "project_id": "proj1",
        }
        ctx = ThreadContext.from_dict(data)
        assert ctx.thread_root_id == "root1"
        assert ctx.chat_id == "chat1"
        assert ctx.project_id == "proj1"
        assert ctx.mode == "smart"
        assert ctx.tool_name is None
        assert ctx.model_name is None
        assert isinstance(ctx.created_at, float)
        assert isinstance(ctx.last_active, float)


class TestThreadContextManager:

    def _make_manager(self, ttl: float = 3600, cleanup_interval: float = 999) -> ThreadContextManager:
        return ThreadContextManager(ttl=ttl, cleanup_interval=cleanup_interval)

    def test_register_creates_and_returns_context(self):
        mgr = self._make_manager()
        try:
            ctx = mgr.register("root1", "chat1", "proj1", mode="coco", tool_name="claude")
            assert isinstance(ctx, ThreadContext)
            assert ctx.thread_root_id == "root1"
            assert ctx.chat_id == "chat1"
            assert ctx.project_id == "proj1"
            assert ctx.mode == "coco"
            assert ctx.tool_name == "claude"
            assert mgr.active_count == 1
        finally:
            mgr.close()

    def test_get_returns_context_and_calls_touch(self):
        mgr = self._make_manager()
        try:
            mgr.register("root1", "chat1", "proj1")
            old_active = mgr.get("root1").last_active
            time.sleep(0.05)
            ctx = mgr.get("root1")
            assert ctx is not None
            assert ctx.last_active > old_active
        finally:
            mgr.close()

    def test_get_returns_none_for_unknown_key(self):
        mgr = self._make_manager()
        try:
            assert mgr.get("nonexistent") is None
        finally:
            mgr.close()

    def test_get_by_chat_filters_by_chat_id(self):
        mgr = self._make_manager()
        try:
            mgr.register("root1", "chatA", "proj1")
            mgr.register("root2", "chatA", "proj1")
            mgr.register("root3", "chatB", "proj1")
            results = mgr.get_by_chat("chatA")
            assert len(results) == 2
            root_ids = {c.thread_root_id for c in results}
            assert root_ids == {"root1", "root2"}
        finally:
            mgr.close()

    def test_get_engine_context_requires_exact_thread_root(self):
        mgr = self._make_manager()
        try:
            mgr.register(
                "thread-wf-1",
                "chat1",
                "proj1",
                mode="workflow",
                tool_name="coco",
                model_name="m1",
            )

            ctx = mgr.get_engine_context("thread-wf-1")
            assert ctx is not None
            assert ctx.mode == "workflow"
            assert mgr.get_engine_context("missing-thread") is None
            assert [c.thread_root_id for c in mgr.get_by_chat("chat1")] == ["thread-wf-1"]
        finally:
            mgr.close()

    def test_has_active_engine_ignores_smart_topic(self):
        mgr = self._make_manager()
        try:
            mgr.register("smart-root", "chat1", "proj1", mode="smart")
            mgr.register("wf-root", "chat1", "proj1", mode="workflow")

            assert mgr.has_active_engine("smart-root") is False
            assert mgr.has_active_engine("wf-root") is True
            assert mgr.has_active_engine("missing-root") is False
        finally:
            mgr.close()

    def test_bind_engine_registers_topic_engine_context(self):
        mgr = self._make_manager()
        try:
            ctx = mgr.bind_engine(
                thread_root_id="thread-1",
                chat_id="chat1",
                project_id="proj1",
                mode="workflow",
                tool_name="codex",
                model_name="gpt-5.5",
            )

            assert ctx.thread_root_id == "thread-1"
            assert ctx.mode == "workflow"
            assert ctx.tool_name == "codex"
            assert ctx.model_name == "gpt-5.5"
            assert mgr.get_engine_context("thread-1") is ctx
        finally:
            mgr.close()

    def test_update_mode_success(self):
        mgr = self._make_manager()
        try:
            mgr.register("root1", "chat1", "proj1", mode="smart")
            assert mgr.update_mode("root1", "coco") is True
            ctx = mgr.get("root1")
            assert ctx.mode == "coco"
        finally:
            mgr.close()

    def test_update_mode_failure_unknown_key(self):
        mgr = self._make_manager()
        try:
            assert mgr.update_mode("nonexistent", "coco") is False
        finally:
            mgr.close()

    def test_update_tool_success_both_fields(self):
        mgr = self._make_manager()
        try:
            mgr.register("root1", "chat1", "proj1")
            assert mgr.update_tool("root1", tool_name="claude", model_name="claude-3.5-sonnet") is True
            ctx = mgr.get("root1")
            assert ctx.tool_name == "claude"
            assert ctx.model_name == "claude-3.5-sonnet"
        finally:
            mgr.close()

    def test_update_tool_success_tool_name_only(self):
        mgr = self._make_manager()
        try:
            mgr.register("root1", "chat1", "proj1", tool_name="coco", model_name="gpt-4.1")
            assert mgr.update_tool("root1", tool_name="gemini") is True
            ctx = mgr.get("root1")
            assert ctx.tool_name == "gemini"
            assert ctx.model_name == "gpt-4.1"
        finally:
            mgr.close()

    def test_update_tool_failure_unknown_key(self):
        mgr = self._make_manager()
        try:
            assert mgr.update_tool("nonexistent", tool_name="claude") is False
        finally:
            mgr.close()

    def test_remove_returns_and_removes_context(self):
        mgr = self._make_manager()
        try:
            mgr.register("root1", "chat1", "proj1")
            assert mgr.active_count == 1
            ctx = mgr.remove("root1")
            assert ctx is not None
            assert ctx.thread_root_id == "root1"
            assert mgr.active_count == 0
        finally:
            mgr.close()

    def test_remove_returns_none_for_unknown_key(self):
        mgr = self._make_manager()
        try:
            assert mgr.remove("nonexistent") is None
        finally:
            mgr.close()

    def test_remove_by_chat_removes_all_for_chat_id(self):
        mgr = self._make_manager()
        try:
            mgr.register("root1", "chatA", "proj1")
            mgr.register("root2", "chatA", "proj1")
            mgr.register("root3", "chatB", "proj1")
            removed = mgr.remove_by_chat("chatA")
            assert removed == 2
            assert mgr.active_count == 1
            assert mgr.get("root3") is not None
        finally:
            mgr.close()

    def test_active_count(self):
        mgr = self._make_manager()
        try:
            assert mgr.active_count == 0
            mgr.register("root1", "chat1", "proj1")
            assert mgr.active_count == 1
            mgr.register("root2", "chat2", "proj2")
            assert mgr.active_count == 2
            mgr.remove("root1")
            assert mgr.active_count == 1
        finally:
            mgr.close()

    def test_close_stops_cleanup_thread(self):
        mgr = self._make_manager()
        mgr.close()
        mgr._cleanup_thread.join(timeout=2)
        assert not mgr._cleanup_thread.is_alive()

    def test_ttl_eviction(self):
        mgr = self._make_manager(ttl=0.1, cleanup_interval=999)
        try:
            mgr.register("root1", "chat1", "proj1")
            mgr.register("root2", "chat1", "proj1")
            assert mgr.active_count == 2
            time.sleep(0.2)
            mgr._evict_expired()
            assert mgr.active_count == 0
        finally:
            mgr.close()

    def test_ttl_eviction_partial(self):
        mgr = self._make_manager(ttl=0.15, cleanup_interval=999)
        try:
            mgr.register("root_old", "chat1", "proj1")
            time.sleep(0.1)
            mgr.register("root_new", "chat1", "proj1")
            time.sleep(0.1)
            mgr._evict_expired()
            assert mgr.active_count == 1
            assert mgr.get("root_new") is not None
            assert mgr.get("root_old") is None
        finally:
            mgr.close()


class TestThreadLocalStorage:

    def test_set_and_get_current_thread_id(self):
        set_current_thread_id("thread_abc")
        assert get_current_thread_id() == "thread_abc"
        set_current_thread_id(None)
        assert get_current_thread_id() is None

    def test_default_is_none(self):
        if hasattr(_current_thread_id, "value"):
            delattr(_current_thread_id, "value")
        assert get_current_thread_id() is None

    def test_thread_isolation(self):
        results = {}
        barrier = threading.Barrier(2)

        def worker(name, tid):
            set_current_thread_id(tid)
            barrier.wait(timeout=5)
            results[name] = get_current_thread_id()

        t1 = threading.Thread(target=worker, args=("t1", "id_A"))
        t2 = threading.Thread(target=worker, args=("t2", "id_B"))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        assert results["t1"] == "id_A"
        assert results["t2"] == "id_B"


class TestOnEvictCallback:

    def _make_manager(self, on_evict=None, ttl: float = 3600, cleanup_interval: float = 999) -> ThreadContextManager:
        return ThreadContextManager(ttl=ttl, cleanup_interval=cleanup_interval, on_evict=on_evict)

    def test_remove_calls_on_evict(self):
        from unittest.mock import MagicMock

        cb = MagicMock()
        mgr = self._make_manager(on_evict=cb)
        try:
            ctx = mgr.register("root1", "chat1", "proj1")
            mgr.remove("root1")
            cb.assert_called_once_with(ctx)
        finally:
            mgr.close()

    def test_remove_unknown_does_not_call_on_evict(self):
        from unittest.mock import MagicMock

        cb = MagicMock()
        mgr = self._make_manager(on_evict=cb)
        try:
            mgr.remove("nonexistent")
            cb.assert_not_called()
        finally:
            mgr.close()

    def test_remove_by_chat_calls_on_evict_for_each(self):
        evicted = []
        mgr = self._make_manager(on_evict=lambda ctx: evicted.append(ctx))
        try:
            mgr.register("root1", "chatA", "proj1")
            mgr.register("root2", "chatA", "proj1")
            mgr.remove_by_chat("chatA")
            assert len(evicted) == 2
            evicted_ids = {c.thread_root_id for c in evicted}
            assert evicted_ids == {"root1", "root2"}
        finally:
            mgr.close()

    def test_evict_expired_calls_on_evict(self):
        from unittest.mock import MagicMock

        cb = MagicMock()
        mgr = self._make_manager(on_evict=cb, ttl=0.1, cleanup_interval=999)
        try:
            ctx = mgr.register("root1", "chat1", "proj1")
            time.sleep(0.2)
            mgr._evict_expired()
            cb.assert_called_once_with(ctx)
        finally:
            mgr.close()

    def test_close_calls_on_evict_for_all(self):
        evicted = []
        mgr = self._make_manager(on_evict=lambda ctx: evicted.append(ctx))
        mgr.register("root1", "chat1", "proj1")
        mgr.register("root2", "chat2", "proj2")
        mgr.close()
        assert len(evicted) == 2
        evicted_ids = {c.thread_root_id for c in evicted}
        assert evicted_ids == {"root1", "root2"}

    def test_on_evict_exception_does_not_propagate(self):
        def bad_callback(ctx):
            raise RuntimeError("boom")

        mgr = self._make_manager(on_evict=bad_callback)
        try:
            mgr.register("root1", "chat1", "proj1")
            mgr.remove("root1")
            assert mgr.active_count == 0
        finally:
            mgr.close()

    def test_on_evict_none_does_not_error(self):
        mgr = self._make_manager(on_evict=None)
        try:
            mgr.register("root1", "chat1", "proj1")
            mgr.remove("root1")
            assert mgr.active_count == 0
        finally:
            mgr.close()


class TestGetThreadManagerSingleton:

    def test_returns_same_instance(self):
        import src.thread.manager as mod

        original = mod._manager
        try:
            mod._manager = None
            m1 = get_thread_manager()
            m2 = get_thread_manager()
            assert m1 is m2
            m1.close()
        finally:
            mod._manager = original


class TestDualKeyAlias:

    def _make_manager(self, **kwargs):
        kwargs.setdefault("ttl", 3600)
        kwargs.setdefault("cleanup_interval", 99999)
        return ThreadContextManager(**kwargs)

    def test_get_by_chat_deduplicates(self):
        mgr = self._make_manager()
        try:
            mgr.register("reply1", "c1", "p1", mode="coco", alias_keys=["msg1"])
            result = mgr.get_by_chat("c1")
            assert len(result) == 1
            assert result[0].thread_root_id == "reply1"
        finally:
            mgr.close()

    def test_active_count_deduplicates(self):
        mgr = self._make_manager()
        try:
            mgr.register("reply1", "c1", "p1", mode="coco", alias_keys=["msg1"])
            assert mgr.active_count == 1
            mgr.register("reply2", "c1", "p2", mode="claude", alias_keys=["msg2"])
            assert mgr.active_count == 2
        finally:
            mgr.close()

    def test_remove_by_chat_cleans_aliases(self):
        mgr = self._make_manager()
        try:
            mgr.register("reply1", "c1", "p1", mode="coco", alias_keys=["msg1"])
            count = mgr.remove_by_chat("c1")
            assert count == 1
            assert mgr.get("reply1") is None
            assert mgr.get("msg1") is None
            assert len(mgr._aliases) == 0
        finally:
            mgr.close()

    def test_close_cleans_aliases(self):
        mgr = self._make_manager()
        mgr.register("reply1", "c1", "p1", mode="coco", alias_keys=["msg1"])
        mgr.close()
        assert mgr.get("reply1") is None
        assert mgr.get("msg1") is None
        assert len(mgr._aliases) == 0

    def test_evict_cleans_aliases(self):
        mgr = self._make_manager(ttl=0)
        try:
            mgr.register("reply1", "c1", "p1", mode="coco", alias_keys=["msg1"])
            import time
            time.sleep(0.05)
            mgr._evict_expired()
            assert mgr.get("reply1") is None
            assert mgr.get("msg1") is None
            assert len(mgr._aliases) == 0
        finally:
            mgr.close()

    def test_remove_normalizes_alias_to_canonical(self):
        mgr = self._make_manager()
        try:
            mgr.register("reply1", "c1", "p1", mode="coco", alias_keys=["msg1", "msg2"])
            removed = mgr.remove("msg1")
            assert removed is not None
            assert removed.thread_root_id == "reply1"
            assert mgr.get("reply1") is None
            assert mgr.get("msg1") is None
            assert mgr.get("msg2") is None
            assert mgr.active_count == 0
        finally:
            mgr.close()

    def test_on_evict_called_once_with_alias(self):
        evicted = []
        mgr = self._make_manager(on_evict=lambda ctx: evicted.append(ctx))
        try:
            mgr.register("reply1", "c1", "p1", mode="coco", alias_keys=["msg1"])
            mgr.remove("reply1")
            assert len(evicted) == 1
            assert evicted[0].thread_root_id == "reply1"
        finally:
            mgr.close()
