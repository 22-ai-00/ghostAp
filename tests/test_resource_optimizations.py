"""Regression tests for CPU/memory resource optimizations."""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.acp.client import ACPHistoryStore
from src.acp.manager import ACPSessionManager
from src.card.delivery.channel_client import LarkChannelCardAPIClient
from src.card.delivery.feishu_client import FeishuCardAPIClient
from src.feishu.control_plane import ControlPlane
from src.feishu.message_cache import MessageCache
from src.tasking import TaskPriority, TaskScheduler, TaskSpec, TaskStatus
from src.thread.manager import ThreadContextManager
from src.utils import gc_monitor as gc_monitor_module
from src.utils import memory_tuning, thread_pools
from src.utils.gc_monitor import GCMonitor
from src.workflow_engine.progress_coalescer import ProgressCoalescer


def test_memory_tuning_cross_platform_trim_and_stack(monkeypatch):
    """Memory tuning applies stack limit and safely no-ops glibc calls on macOS."""
    original_stack = threading.stack_size()
    try:
        status = memory_tuning.configure_runtime_memory_tuning(
            stack_size_bytes=1024 * 1024,
            max_arenas=2,
        )
        assert status["stack_size_applied"] is True
        assert threading.stack_size() == 1024 * 1024

        # Non-Linux (e.g. macOS): trim_process_heap must cleanly no-op without loading libc.so.6
        monkeypatch.setattr(memory_tuning.sys, "platform", "darwin")
        monkeypatch.setattr(memory_tuning, "_LIBC_PROBED", False)
        monkeypatch.setattr(memory_tuning, "_LIBC_HANDLE", None)
        with patch("src.utils.memory_tuning.ctypes.CDLL") as mock_cdll:
            assert memory_tuning.trim_process_heap() is False
            mock_cdll.assert_not_called()

        # Linux with working malloc_trim
        fake_trim = MagicMock(return_value=1)
        fake_libc = SimpleNamespace(malloc_trim=fake_trim)
        monkeypatch.setattr(memory_tuning.sys, "platform", "linux")
        monkeypatch.setattr(memory_tuning, "_LIBC_PROBED", True)
        monkeypatch.setattr(memory_tuning, "_LIBC_HANDLE", fake_libc)
        assert memory_tuning.trim_process_heap() is True
        fake_trim.assert_called_once_with(0)
    finally:
        threading.stack_size(original_stack)


def test_gc_monitor_calls_trim_after_collect():
    """GCMonitor triggers heap trimming right after gc.collect()."""
    monitor = GCMonitor(memory_threshold_percent=50.0, check_interval_seconds=0.0)
    with (
        patch("src.utils.gc_monitor.psutil.Process") as mock_proc,
        patch("src.utils.gc_monitor.gc.collect") as mock_collect,
        patch("src.utils.memory_tuning.trim_process_heap") as mock_trim,
    ):
        mock_proc.return_value.memory_percent.return_value = 90.0
        monitor.check_and_collect(label="test")
        mock_collect.assert_called_once()
        mock_trim.assert_called_once()
    assert gc_monitor_module is not None


def test_default_io_workers_scales_with_cpu_count(monkeypatch):
    """I/O pool scales cleanly for 2C VMs instead of hardcoding 16 threads."""
    monkeypatch.delenv("GHOSTAP_IO_POOL_WORKERS", raising=False)
    monkeypatch.setattr(thread_pools.os, "cpu_count", lambda: 2)
    assert thread_pools._default_io_workers() == 4
    monkeypatch.setattr(thread_pools.os, "cpu_count", lambda: 16)
    assert thread_pools._default_io_workers() == 8
    monkeypatch.setenv("GHOSTAP_IO_POOL_WORKERS", "6")
    assert thread_pools._default_io_workers() == 6


def test_task_scheduler_idle_and_stop_without_polling():
    """Idle TaskScheduler executes submitted work cleanly and stops immediately."""
    scheduler = TaskScheduler(max_concurrent=1)
    ran = threading.Event()
    statuses: list[TaskStatus] = []
    executed: list[str] = []
    try:
        # Allow dispatcher to settle into idle condition wait
        time.sleep(0.05)
        handle = scheduler.submit(
            TaskSpec(chat_id="c1", name="ping", task_type="test", priority=TaskPriority.NORMAL),
            lambda _ctx: executed.append("ok") or "ok",
        )
        handle.add_done_callback(lambda ev: (statuses.append(ev.status), ran.set()))
        assert ran.wait(timeout=2.0)
        assert statuses == [TaskStatus.SUCCEEDED]
        assert executed == ["ok"]
        state = scheduler.get_state(handle.run_id)
        assert state is not None and state.status == TaskStatus.SUCCEEDED
    finally:
        t0 = time.monotonic()
        scheduler.stop()
        assert (time.monotonic() - t0) < 1.0


def test_control_plane_wakes_on_deferred_exit_and_stops_cleanly():
    """ControlPlane blocks silently when idle and wakes on deferred exit events."""
    scheduler = MagicMock()
    scheduler.list_tasks.return_value = []
    project_mgr = MagicMock()
    project_mgr.get_project_for_chat.return_value = None
    project_mgr.get_active_project.return_value = None
    submitted: list[tuple] = []
    scheduler.submit.side_effect = lambda spec, fn: submitted.append((spec, fn))

    cp = ControlPlane(scheduler, project_mgr, lambda *_a, **_kw: None)
    try:
        cp.request_deferred_exit(message_id="m1", chat_id="c1", project_id="p1")
        cp.on_scheduler_event(
            SimpleNamespace(
                chat_id="c1",
                project_id="p1",
                run_id="r1",
                task_type="coding",
                status=SimpleNamespace(),  # ignored status
            )
        )
        # Real terminal status wakes loop
        from src.tasking import TaskStatus

        cp.on_scheduler_event(
            SimpleNamespace(
                chat_id="c1",
                project_id="p1",
                run_id="r1",
                task_type="coding",
                status=TaskStatus.SUCCEEDED,
            )
        )
        deadline = time.monotonic() + 2.0
        while not submitted and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(submitted) == 1
    finally:
        t0 = time.monotonic()
        cp.stop()
        assert (time.monotonic() - t0) < 1.0


def test_message_cache_wakes_on_insert_and_stops_fast():
    """MessageCache cleanup thread sleeps silently when empty and exits promptly."""
    cache = MessageCache(ttl=10, max_size=100, cleanup_interval=60)
    cache.start_cleanup_thread()
    try:
        assert cache._cleanup_thread is not None
        assert cache._cleanup_thread.is_alive()
        assert cache.is_duplicate("msg-1") is False
        assert cache.is_duplicate("msg-1") is True
        cache.mark_seen("msg-2")
        assert cache.contains("msg-2") is True
    finally:
        t0 = time.monotonic()
        cache.stop_cleanup_thread()
        assert cache._cleanup_thread is None
        assert (time.monotonic() - t0) < 1.0


def test_acp_manager_lazy_keepalive_thread_lifecycle():
    """ACPSessionManager spawns 0 keepalive threads until a session is tracked."""
    mgr = ACPSessionManager(agent_type="coco", keepalive_interval=0.02)
    try:
        assert mgr._keepalive_thread is None
        fake_session = MagicMock()
        fake_session.session_id = "sess-1"
        fake_session.message_count = 1
        fake_session.last_active = time.time()
        fake_session.is_server_running.return_value = True
        fake_session.to_snapshot.return_value = {"session_id": "sess-1"}

        with mgr._acquire_lock():
            mgr._sessions["k1"] = fake_session
            mgr._ensure_keepalive_started_unlocked()

        assert mgr._keepalive_thread is not None
        assert mgr._keepalive_thread.is_alive()

        # Remove session -> keepalive loop exits cleanly on next check
        with mgr._acquire_lock():
            mgr._sessions.pop("k1", None)
        deadline = time.monotonic() + 1.5
        while mgr._keepalive_thread is not None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert mgr._keepalive_thread is None
    finally:
        mgr.cleanup_all(timeout=5.0)


def test_thread_context_manager_lazy_cleanup_thread():
    """ThreadContextManager avoids spawning a thread before contexts are registered."""
    tcm = ThreadContextManager(ttl=60.0, cleanup_interval=0.02)
    try:
        assert tcm._cleanup_thread is None
        tcm.register("root-1", "chat-1", "proj-1", mode="deep")
        assert tcm._cleanup_thread is not None
        assert tcm._cleanup_thread.is_alive()

        tcm.remove("root-1")
        deadline = time.monotonic() + 1.5
        while tcm._cleanup_thread is not None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert tcm._cleanup_thread is None
    finally:
        tcm.close()


def test_progress_coalescer_event_driven_delivery_and_stop():
    """ProgressCoalescer delivers debounced snapshots without fixed polling."""
    delivered: list[str] = []
    done = threading.Event()

    def _on_progress(snap: str) -> None:
        delivered.append(snap)
        done.set()

    coalescer = ProgressCoalescer(debounce_s=0.02, on_progress=_on_progress)
    try:
        coalescer.enqueue("s1")
        coalescer.enqueue("s2")
        assert done.wait(timeout=1.5)
        assert delivered[-1] == "s2"
    finally:
        t0 = time.monotonic()
        coalescer.stop()
        assert (time.monotonic() - t0) < 1.0


def test_audit_alias_caches_are_lru_bounded(monkeypatch):
    """Audit alias caches in both Feishu clients bound growth via LRU eviction."""
    ch_client = LarkChannelCardAPIClient(MagicMock())
    monkeypatch.setattr(ch_client, "_MAX_AUDIT_ALIASES", 3)
    for i in range(5):
        ch_client._remember_aliases((f"target-{i}",), f"id-{i}")
    assert len(ch_client._audit_aliases_by_target) == 3
    assert "id-0" not in ch_client._audit_aliases_by_target
    assert "id-4" in ch_client._audit_aliases_by_target

    feishu_client = FeishuCardAPIClient(MagicMock())
    monkeypatch.setattr(feishu_client, "_MAX_AUDIT_ALIASES", 3)
    for i in range(5):
        feishu_client._remember_message_audit_aliases(f"msg-{i}", (f"alias-{i}",))
    assert len(feishu_client._audit_aliases_by_message) == 3
    assert "msg-0" not in feishu_client._audit_aliases_by_message
    assert "msg-4" in feishu_client._audit_aliases_by_message


def test_acp_history_store_load_streams_large_files(tmp_path):
    """ACPHistoryStore.load streams lines cleanly while skipping blank/corrupt entries."""
    store = ACPHistoryStore()
    store._base = tmp_path
    session_id = "sess-stream-1"
    p = store._path_for(session_id)
    lines = ["", "{bad-json"]
    for i in range(50):
        lines.append(json.dumps({"role": "user", "seq": i}))
    lines.append("not-json-either")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    recent = store.load(session_id, limit=5)
    assert [item["seq"] for item in recent] == [45, 46, 47, 48, 49]

    all_items = store.load(session_id, limit=0)
    assert len(all_items) == 50
