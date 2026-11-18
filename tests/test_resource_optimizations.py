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


def test_maybe_trim_process_heap_succeeds_during_early_uptime(monkeypatch):
    """Initial maybe_trim_process_heap succeeds even when time.monotonic() < min_interval_s."""
    monkeypatch.setattr(memory_tuning, "_LAST_PERIODIC_TRIM_AT", None)
    clock = iter([5.0, 10.0, 70.0])
    monkeypatch.setattr(memory_tuning.time, "monotonic", lambda: next(clock))
    trims: list[int] = []
    monkeypatch.setattr(
        memory_tuning,
        "trim_process_heap",
        lambda: trims.append(1) or True,
    )

    assert memory_tuning.maybe_trim_process_heap(min_interval_s=60.0) is True
    assert memory_tuning.maybe_trim_process_heap(min_interval_s=60.0) is False
    assert memory_tuning.maybe_trim_process_heap(min_interval_s=60.0) is True
    assert len(trims) == 2


def test_control_plane_deferred_exit_race_and_chat_prefix_wakeup():
    """Deferred exit completes promptly if task already finished and wakes on chat-scoped events."""
    from src.tasking import TaskStatus

    scheduler = MagicMock()
    scheduler.list_tasks.return_value = []
    project_mgr = MagicMock()
    project_mgr.get_project_for_chat.return_value = None
    project_mgr.get_active_project.return_value = None
    submitted: list[tuple] = []
    scheduler.submit.side_effect = lambda spec, fn: submitted.append((spec, fn))

    cp = ControlPlane(scheduler, project_mgr, lambda *_a, **_kw: None)
    try:
        # Case 1: task already finished before request_deferred_exit() -> no spin, completes immediately
        cp.request_deferred_exit(message_id="m-done", chat_id="c1", project_id="p1")
        deadline = time.monotonic() + 1.0
        while not submitted and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(submitted) == 1

        # Case 2: active task finishes with chat-scoped event (project_id=None) -> prefix match wakes "c1:p1"
        submitted.clear()
        running_task = SimpleNamespace(status=TaskStatus.RUNNING)
        scheduler.list_tasks.return_value = [running_task]
        cp.request_deferred_exit(message_id="m-wait", chat_id="c1", project_id="p1")
        time.sleep(0.05)
        assert len(submitted) == 0

        scheduler.list_tasks.return_value = []
        cp.on_scheduler_event(
            SimpleNamespace(
                chat_id="c1",
                project_id=None,
                run_id="r-chat",
                task_type="coding",
                status=TaskStatus.SUCCEEDED,
            )
        )
        deadline = time.monotonic() + 1.0
        while not submitted and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(submitted) == 1
    finally:
        cp.stop()


def test_task_scheduler_next_reap_delay_exact_remaining_and_no_active_poll():
    """TaskScheduler sleeps until exact terminal reap age rather than fixed 15s/0.2s polling."""
    scheduler = TaskScheduler(max_concurrent=1)
    try:
        with scheduler._cv:
            assert scheduler._next_reap_delay_unlocked() is None

        spec = TaskSpec(
            chat_id="c-reap",
            project_id="p-reap",
            name="reap-check",
            task_type="coding",
            priority=TaskPriority.NORMAL,
        )
        done = threading.Event()
        handle = scheduler.submit(spec, lambda _ctx: "ok")
        handle.add_done_callback(lambda _ev: done.set())
        assert done.wait(timeout=2.0)

        deadline = time.monotonic() + 2.0
        state = scheduler.get_state(handle.run_id)
        while state is not None and state.status == TaskStatus.RUNNING and time.monotonic() < deadline:
            time.sleep(0.01)
            state = scheduler.get_state(handle.run_id)
        assert state is not None and state.status == TaskStatus.SUCCEEDED

        with scheduler._cv:
            delay = scheduler._next_reap_delay_unlocked()
            assert delay is not None
            assert 0.05 <= delay <= scheduler._REAP_DEFAULT_MAX_AGE + 1.0
    finally:
        scheduler.stop()


def test_message_cache_adaptive_ttl_wait_only_wakes_on_empty_transition():
    """MessageCache sets _wakeup only when transitioning from empty (or ttl<=0)."""
    cache = MessageCache(ttl=10, max_size=100, cleanup_interval=60)
    assert not cache._wakeup.is_set()
    assert cache.is_duplicate("msg-1") is False
    assert cache._wakeup.is_set()
    cache._wakeup.clear()

    # Subsequent inserts while cache is non-empty do not repeatedly wake the reaper
    assert cache.is_duplicate("msg-2") is False
    assert not cache._wakeup.is_set()


def test_ws_health_monitor_next_watchdog_wait_adapts_to_stale_timeout(monkeypatch):
    """WSHealthMonitor sleeps up to remaining stale timeout when connected and healthy."""
    from src.feishu.ws_health import WSHealthMonitor

    fake_client = SimpleNamespace(_client=None)
    settings = SimpleNamespace(feishu_ws_watchdog_interval=15.0, feishu_ws_stale_timeout=300.0)
    monitor = WSHealthMonitor(fake_client, settings)

    # Disconnected -> uses base watchdog interval (15s)
    assert monitor._next_watchdog_wait_seconds() == 15.0

    # Connected & recently active -> sleeps remaining stale window (~295s)
    monkeypatch.setattr("src.feishu.ws_health.time.time", lambda: 1005.0)
    with monitor._health_lock:
        monitor._last_connect_at = 1000.0
        monitor._last_frame_at = 1000.0
        monitor._last_pong_at = 1000.0
    fake_client._client = SimpleNamespace(_conn=object())
    assert monitor._next_watchdog_wait_seconds() == 295.0

    # Near/past stale deadline -> floored at watchdog interval (15s)
    monkeypatch.setattr("src.feishu.ws_health.time.time", lambda: 1295.0)
    assert monitor._next_watchdog_wait_seconds() == 15.0


def test_chat_lock_and_repo_lock_cleanup_threads_retire_and_restart(monkeypatch):
    """ChatLockManager and RepoLockManager retire empty cleanup threads and restart on re-lock."""
    from src.chat_lock import ChatLockManager
    from src.repo_lock import RepoLockManager

    monkeypatch.setattr(
        "src.chat_lock.get_settings",
        lambda: SimpleNamespace(
            admin_user_ids=("admin-1",),
            chat_lock_max_duration=60.0,
            chat_lock_cleanup_interval=0.02,
        ),
    )
    chat_mgr = ChatLockManager(max_duration=60.0, cleanup_interval=0.02)
    try:
        assert chat_mgr._cleanup_thread is None
        res = chat_mgr.lock_chat("chat-1", "admin-1")
        assert res.success is True
        assert chat_mgr._cleanup_thread is not None and chat_mgr._cleanup_thread.is_alive()

        assert chat_mgr.unlock_chat("chat-1", "admin-1").success is True
        deadline = time.monotonic() + 1.5
        while chat_mgr._cleanup_thread is not None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert chat_mgr._cleanup_thread is None

        # Re-locking cleanly spawns a fresh thread
        assert chat_mgr.lock_chat("chat-2", "admin-1").success is True
        assert chat_mgr._cleanup_thread is not None and chat_mgr._cleanup_thread.is_alive()
    finally:
        chat_mgr.shutdown()

    repo_mgr = RepoLockManager(idle_timeout=0.02, cleanup_interval=0.02, hard_timeout=60)
    try:
        assert repo_mgr._cleanup_thread is None
        acq = repo_mgr.acquire("/tmp/ghostap-test-repo", "chat-1")
        assert acq.success is True
        assert repo_mgr._cleanup_thread is not None and repo_mgr._cleanup_thread.is_alive()
        repo_mgr.release("/tmp/ghostap-test-repo", "chat-1")

        deadline = time.monotonic() + 1.5
        while repo_mgr._cleanup_thread is not None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert repo_mgr._cleanup_thread is None

        acq2 = repo_mgr.acquire("/tmp/ghostap-test-repo", "chat-2")
        assert acq2.success is True
        assert repo_mgr._cleanup_thread is not None and repo_mgr._cleanup_thread.is_alive()
    finally:
        repo_mgr.shutdown()


def test_task_scheduler_threads_start_lazily_and_retire_when_idle():
    """TaskScheduler threads stay dormant at idle, start on demand, and retire cleanly."""
    scheduler = TaskScheduler(max_concurrent=1)
    scheduler._idle_dispatch_timeout = 0.05
    try:
        assert not scheduler._dispatcher.is_alive()
        assert not scheduler._event_dispatcher.is_alive()
        assert not scheduler._completion_dispatcher.is_alive()

        events_seen: list[TaskStatus] = []
        scheduler.add_listener(lambda ev: events_seen.append(ev.status))
        done = threading.Event()
        handle = scheduler.submit(
            TaskSpec(chat_id="lazy-c", name="lazy-task"),
            lambda _ctx: "ok",
        )
        handle.add_done_callback(lambda _ev: done.set())
        assert done.wait(timeout=2.0)

        # Wait for event + completion dispatchers to retire after idle grace timeout
        deadline = time.monotonic() + 2.0
        while (
            scheduler._event_dispatcher.is_alive()
            or scheduler._completion_dispatcher.is_alive()
        ) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not scheduler._event_dispatcher.is_alive()
        assert not scheduler._completion_dispatcher.is_alive()
        assert TaskStatus.SUCCEEDED in events_seen

        # Force immediate reap cycle -> _dispatcher also retires when history clears
        with scheduler._cv:
            st = scheduler._states.get(handle.run_id)
            if st is not None:
                st.ended_at = time.time() - (scheduler._REAP_DEFAULT_MAX_AGE + 10)
            scheduler._cv.notify_all()
        deadline = time.monotonic() + 2.0
        while scheduler._dispatcher.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not scheduler._dispatcher.is_alive()

        # Re-submitting re-spawns dormant threads transparently
        done2 = threading.Event()
        handle2 = scheduler.submit(
            TaskSpec(chat_id="lazy-c", name="lazy-task-2"),
            lambda _ctx: "ok2",
        )
        handle2.add_done_callback(lambda _ev: done2.set())
        assert done2.wait(timeout=2.0)
    finally:
        scheduler.stop(wait=True, shutdown_executor=True)


def test_control_plane_thread_starts_lazily_and_retires_when_empty():
    """ControlPlane starts no thread until request_deferred_exit and retires when empty."""
    from src.tasking import TaskStatus

    scheduler = MagicMock()
    scheduler.list_tasks.return_value = []
    project_mgr = MagicMock()
    submitted: list[tuple] = []
    scheduler.submit.side_effect = lambda spec, fn: submitted.append((spec, fn))

    cp = ControlPlane(scheduler, project_mgr, lambda *_a, **_kw: None)
    try:
        assert cp._thread is None

        # Unrelated completion event while no deferred exit is pending keeps thread dormant
        cp.on_scheduler_event(
            SimpleNamespace(
                chat_id="c1",
                project_id="p1",
                run_id="r1",
                task_type="coding",
                status=TaskStatus.SUCCEEDED,
            )
        )
        assert cp._thread is None

        cp.request_deferred_exit(message_id="m1", chat_id="c1", project_id="p1")
        deadline = time.monotonic() + 1.5
        while not submitted and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(submitted) == 1

        deadline = time.monotonic() + 1.5
        while cp._thread is not None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert cp._thread is None

        # Second deferred exit restarts thread cleanly
        cp.request_deferred_exit(message_id="m2", chat_id="c2", project_id="p2")
        deadline = time.monotonic() + 1.5
        while len(submitted) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(submitted) == 2
    finally:
        cp.stop()


def test_message_cache_lazy_cleanup_retires_and_restarts():
    """MessageCache(lazy_cleanup=True) spawns cleanup thread on demand and retires on empty."""
    cache = MessageCache(ttl=0.05, max_size=10, cleanup_interval=0.02, lazy_cleanup=True)
    try:
        cache.start_cleanup_thread()
        assert cache._cleanup_thread is None

        assert cache.is_duplicate("k1") is False
        assert cache._cleanup_thread is not None and cache._cleanup_thread.is_alive()

        deadline = time.monotonic() + 2.0
        while cache._cleanup_thread is not None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert cache._cleanup_thread is None

        assert cache.is_duplicate("k2") is False
        assert cache._cleanup_thread is not None and cache._cleanup_thread.is_alive()
    finally:
        cache.stop_cleanup_thread()


def test_ws_health_trigger_disconnect_skips_when_loop_stopped(monkeypatch):
    """_trigger_disconnect avoids blocking 5s if SDK loop is not running."""
    from src.feishu.ws_health import WSHealthMonitor

    fake_client = SimpleNamespace(_client=SimpleNamespace(_conn=object(), _disconnect=MagicMock()))
    monitor = WSHealthMonitor(fake_client, SimpleNamespace())
    monkeypatch.setattr("lark_channel.ws.client.loop", SimpleNamespace(is_closed=lambda: False, is_running=lambda: False))
    start = time.monotonic()
    assert monitor._trigger_disconnect(reason="application shutdown") is False
    assert (time.monotonic() - start) < 0.2
    fake_client._client._disconnect.assert_not_called()


def test_feishu_ws_client_init_keeps_single_thread(monkeypatch):
    """Isolated FeishuWSClient initialization starts zero background threads."""
    from src.feishu.ws_client import FeishuWSClient

    before_names = {t.name for t in threading.enumerate()}
    client = FeishuWSClient(message_callback=lambda *_a, **_kw: None)
    try:
        after_names = {t.name for t in threading.enumerate()}
        new_threads = after_names - before_names
        assert not new_threads, f"unexpected startup threads: {new_threads}"
        client._message_cache.start_cleanup_thread()
        client._card_event_cache.start_cleanup_thread()
        assert client._message_cache._cleanup_thread is None
        assert client._card_event_cache._cleanup_thread is None
    finally:
        client.close()


def test_deep_and_spec_handler_lazy_renderer_thread_safe(monkeypatch):
    """Concurrent lazy renderer access instantiates each handler renderer once."""
    from src.feishu.handlers.deep import DeepHandler
    from src.feishu.handlers.spec import SpecHandler

    dh = DeepHandler(MagicMock())
    sh = SpecHandler(MagicMock())
    created: list[tuple[str, object]] = []

    def fake_get_renderer(kind, _handler):
        obj = object()
        created.append((kind, obj))
        return obj

    monkeypatch.setattr("src.feishu.renderers.get_renderer", fake_get_renderer)
    barrier = threading.Barrier(8)
    deep_refs: list[object] = []
    spec_refs: list[object] = []

    def worker(idx: int) -> None:
        barrier.wait()
        if idx % 2 == 0:
            deep_refs.append(dh.renderer)
        else:
            spec_refs.append(sh.renderer)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert len(created) == 2
    assert len({id(ref) for ref in deep_refs}) == 1
    assert len({id(ref) for ref in spec_refs}) == 1


def test_upload_file_rebuilds_request_and_rewinds_on_retry(tmp_path, monkeypatch):
    """FeishuIMClient.upload_file rebuilds CreateFileRequest and rewinds stream on retry."""
    from src.feishu.im_client import FeishuIMClient

    payload = tmp_path / "sample.bin"
    payload.write_bytes(b"payload-bytes")
    seen_reqs: list[object] = []
    seen_contents: list[bytes] = []

    def fake_create(req):
        seen_reqs.append(req)
        seen_contents.append(req.request_body.file.read())
        req.request_body = "mutated_by_sdk"
        if len(seen_reqs) == 1:
            return SimpleNamespace(success=lambda: False, code=500, msg="transient")
        return SimpleNamespace(success=lambda: True, data=SimpleNamespace(file_key="fk_ok"))

    fake_client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(file=SimpleNamespace(create=fake_create))))
    monkeypatch.setattr("time.sleep", lambda *_a, **_kw: None)
    im = FeishuIMClient(
        lambda: fake_client,
        SimpleNamespace(im_api_max_retries=2, im_api_retry_delay_s=0.01),
    )

    assert im.upload_file(str(payload)) == "fk_ok"
    assert len(seen_reqs) == 2
    assert seen_reqs[0] is not seen_reqs[1]
    assert seen_contents == [b"payload-bytes", b"payload-bytes"]
