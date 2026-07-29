import threading
import time
from contextlib import contextmanager

import pytest

from src.tasking import (
    TaskPriority,
    TaskScheduler,
    TaskSpec,
    TaskStatus,
    get_current_task_run_id,
)


def test_scheduler_respects_global_concurrency_limit():
    scheduler = TaskScheduler(max_concurrent=2, per_key_concurrency=1)

    lock = threading.Lock()
    active = 0
    max_active = 0

    def make_task(i: int):
        def _fn(ctx):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.15)
            with lock:
                active -= 1
            return i

        return _fn

    handles = []
    for i in range(6):
        spec = TaskSpec(chat_id=f"chat_{i}", name=f"t{i}", task_type="test")
        handles.append(scheduler.submit(spec, make_task(i)))

    results = [h.wait(timeout=3) for h in handles]
    assert all(r.status == TaskStatus.SUCCEEDED for r in results)
    assert max_active <= 2


def test_scheduler_serializes_tasks_with_same_queue_key():
    scheduler = TaskScheduler(max_concurrent=4, per_key_concurrency=1)

    lock = threading.Lock()
    active = 0
    max_active = 0

    def _fn(ctx):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.1)
        with lock:
            active -= 1
        return "ok"

    chat_id = "chat_same"
    handles = [scheduler.submit(TaskSpec(chat_id=chat_id, queue_key=chat_id, name=f"s{i}"), _fn) for i in range(5)]
    for h in handles:
        r = h.wait(timeout=3)
        assert r.status == TaskStatus.SUCCEEDED
    assert max_active == 1


def test_cancel_queued_task_prevents_execution():
    scheduler = TaskScheduler(max_concurrent=1, per_key_concurrency=1)
    started = threading.Event()
    release = threading.Event()

    def long_task(ctx):
        started.set()
        release.wait(timeout=2)
        return "long"

    def should_not_run(ctx):
        raise AssertionError("canceled task should not execute")

    h1 = scheduler.submit(TaskSpec(chat_id="a", name="long"), long_task)
    assert started.wait(timeout=1)

    h2 = scheduler.submit(TaskSpec(chat_id="b", name="to_cancel", priority=TaskPriority.NORMAL), should_not_run)
    assert h2.cancel() is True

    release.set()
    r1 = h1.wait(timeout=3)
    assert r1.status == TaskStatus.SUCCEEDED

    r2 = h2.wait(timeout=1)
    assert r2.status == TaskStatus.CANCELED


def test_scheduler_can_update_project_id_and_query_by_project():
    scheduler = TaskScheduler(max_concurrent=2, per_key_concurrency=1)

    def _fn(ctx):
        # project is resolved inside task body
        assert scheduler.update_project_id(ctx.run_id, "p1") is True
        ctx.progress("working", percent=30)
        return "ok"

    h = scheduler.submit(TaskSpec(chat_id="c1", name="t1", task_type="test"), _fn)
    r = h.wait(timeout=3)
    assert r.status == TaskStatus.SUCCEEDED

    # include_done=True to include terminal tasks
    tasks = scheduler.list_tasks(project_id="p1", include_done=True)
    assert any(st.run_id == h.run_id for st in tasks)

    # intersection filter
    tasks2 = scheduler.list_tasks(chat_id="c1", project_id="p1", include_done=True)
    assert any(st.run_id == h.run_id for st in tasks2)


def test_system_command_uses_system_queue():
    """System commands should use the SYSTEM queue suffix."""
    spec = TaskSpec(chat_id="chat1", name="help", is_system_command=True)
    assert spec.get_effective_queue_key() == "chat1:SYSTEM"


def test_project_task_uses_project_queue():
    """Project tasks should use the project_id in queue key."""
    spec = TaskSpec(chat_id="chat1", name="task", project_id="proj1")
    assert spec.get_effective_queue_key() == "chat1:proj1"


def test_no_project_task_uses_default_queue():
    """Tasks without project should use DEFAULT queue."""
    spec = TaskSpec(chat_id="chat1", name="task")
    assert spec.get_effective_queue_key() == "chat1:DEFAULT"


def test_explicit_queue_key_takes_precedence():
    """Explicit queue_key should override automatic calculation."""
    spec = TaskSpec(chat_id="chat1", name="task", queue_key="custom:key", project_id="proj1")
    assert spec.get_effective_queue_key() == "custom:key"


def test_system_queue_has_higher_concurrency():
    """System queue should allow higher concurrency than normal queues."""
    scheduler = TaskScheduler(max_concurrent=20, per_key_concurrency=1, system_concurrency=10)

    lock = threading.Lock()
    active_system = 0
    max_active_system = 0
    active_normal = 0
    max_active_normal = 0

    def system_task(ctx):
        nonlocal active_system, max_active_system
        with lock:
            active_system += 1
            max_active_system = max(max_active_system, active_system)
        time.sleep(0.1)
        with lock:
            active_system -= 1
        return "system"

    def normal_task(ctx):
        nonlocal active_normal, max_active_normal
        with lock:
            active_normal += 1
            max_active_normal = max(max_active_normal, active_normal)
        time.sleep(0.1)
        with lock:
            active_normal -= 1
        return "normal"

    handles = []
    for i in range(5):
        handles.append(scheduler.submit(TaskSpec(chat_id="chat1", name=f"sys{i}", is_system_command=True), system_task))
    for i in range(5):
        handles.append(scheduler.submit(TaskSpec(chat_id="chat1", name=f"norm{i}"), normal_task))

    for h in handles:
        r = h.wait(timeout=5)
        assert r.status == TaskStatus.SUCCEEDED

    assert max_active_system >= 2
    assert max_active_normal == 1


def test_system_tasks_not_blocked_by_normal_global_limit():
    """系统任务不应被 normal worker 的全局并发占满而饿死。"""

    scheduler = TaskScheduler(max_concurrent=1, per_key_concurrency=1, system_concurrency=1)

    blocker_started = threading.Event()
    blocker_release = threading.Event()
    sys_started = threading.Event()

    def long_normal(ctx):
        blocker_started.set()
        blocker_release.wait(timeout=2)
        return "normal"

    def quick_system(ctx):
        sys_started.set()
        return "system"

    h_norm = scheduler.submit(TaskSpec(chat_id="chat1", name="normal", project_id="p1"), long_normal)
    assert blocker_started.wait(timeout=1)

    h_sys = scheduler.submit(TaskSpec(chat_id="chat1", name="sys", is_system_command=True), quick_system)

    # system task should start even while normal is blocking
    assert sys_started.wait(timeout=0.3)
    assert h_sys.wait(timeout=2).status == TaskStatus.SUCCEEDED

    blocker_release.set()
    assert h_norm.wait(timeout=2).status == TaskStatus.SUCCEEDED


def test_different_projects_can_run_concurrently():
    """Tasks from different projects should be able to run concurrently."""
    scheduler = TaskScheduler(max_concurrent=10, per_key_concurrency=1)

    lock = threading.Lock()
    active_by_project: dict[str, int] = {"proj1": 0, "proj2": 0}
    max_active_by_project: dict[str, int] = {"proj1": 0, "proj2": 0}
    total_concurrent = 0
    max_total_concurrent = 0

    def make_task(proj_id: str):
        def _fn(ctx):
            nonlocal total_concurrent, max_total_concurrent
            with lock:
                active_by_project[proj_id] += 1
                max_active_by_project[proj_id] = max(max_active_by_project[proj_id], active_by_project[proj_id])
                total_concurrent += 1
                max_total_concurrent = max(max_total_concurrent, total_concurrent)
            time.sleep(0.15)
            with lock:
                active_by_project[proj_id] -= 1
                total_concurrent -= 1
            return proj_id

        return _fn

    handles = []
    for i in range(3):
        handles.append(
            scheduler.submit(TaskSpec(chat_id="chat1", name=f"p1_{i}", project_id="proj1"), make_task("proj1"))
        )
        handles.append(
            scheduler.submit(TaskSpec(chat_id="chat1", name=f"p2_{i}", project_id="proj2"), make_task("proj2"))
        )

    for h in handles:
        r = h.wait(timeout=5)
        assert r.status == TaskStatus.SUCCEEDED

    assert max_active_by_project["proj1"] == 1
    assert max_active_by_project["proj2"] == 1
    assert max_total_concurrent == 2


def test_update_project_id_requeues_queued_task_to_project_queue():
    scheduler = TaskScheduler(max_concurrent=1, per_key_concurrency=1)
    started = threading.Event()
    release = threading.Event()

    def blocker(ctx):
        started.set()
        release.wait(timeout=2)
        return "blocker"

    scheduler.submit(TaskSpec(chat_id="chat1", name="blocker", project_id="p1"), blocker)
    assert started.wait(timeout=1)

    h = scheduler.submit(TaskSpec(chat_id="chat1", name="target"), lambda ctx: "ok")
    state_before = scheduler.get_state(h.run_id)
    assert state_before is not None
    assert state_before.status == TaskStatus.QUEUED
    assert state_before.assigned_queue_key == "chat1:DEFAULT"

    assert scheduler.update_project_id(h.run_id, "p1") is True

    state_after = scheduler.get_state(h.run_id)
    assert state_after is not None
    assert state_after.assigned_queue_key == "chat1:p1"
    assert state_after.project_serial_key == "chat1:p1"

    release.set()
    assert h.wait(timeout=3).status == TaskStatus.SUCCEEDED


def test_stop_cancels_all_queued_tasks_with_terminal_timestamps():
    scheduler = TaskScheduler(max_concurrent=1, per_key_concurrency=1)
    started = threading.Event()
    release = threading.Event()

    def long_task(ctx):
        started.set()
        release.wait(timeout=2)
        return "long"

    h_running = scheduler.submit(TaskSpec(chat_id="c", name="running"), long_task)
    assert started.wait(timeout=1)

    hq1 = scheduler.submit(TaskSpec(chat_id="c", name="q1", project_id="p1"), lambda ctx: "q1")
    hq2 = scheduler.submit(TaskSpec(chat_id="c", name="q2", project_id="p2"), lambda ctx: "q2")

    scheduler.stop(wait=True)

    s1 = scheduler.get_state(hq1.run_id)
    s2 = scheduler.get_state(hq2.run_id)
    assert s1 is not None and s1.status == TaskStatus.CANCELED and s1.ended_at is not None
    assert s2 is not None and s2.status == TaskStatus.CANCELED and s2.ended_at is not None

    release.set()
    assert h_running.wait(timeout=3).status == TaskStatus.SUCCEEDED


def test_dispatch_submit_failure_rolls_back_running_and_sets_failed():
    scheduler = TaskScheduler(max_concurrent=1, per_key_concurrency=1)

    original_submit = scheduler._executor.submit

    def boom(*args, **kwargs):
        raise RuntimeError("submit failed")

    try:
        scheduler._executor.submit = boom
        h = scheduler.submit(TaskSpec(chat_id="c", name="t", project_id="p1"), lambda ctx: "ok")

        result = h.wait(timeout=2)
        assert result.status == TaskStatus.FAILED
        st = h.get_state()
        assert st.error == "submit failed"
        assert scheduler._running_total_normal == 0
        assert scheduler._running_total_system == 0
        assert scheduler._running_by_key[st.assigned_queue_key] == 0
        assert scheduler._running_by_project[st.project_serial_key] == 0
    finally:
        scheduler._executor.submit = original_submit


def test_fence_admission_rejects_new_work_and_cancels_queued_tasks():
    scheduler = TaskScheduler(max_concurrent=1, per_key_concurrency=1)
    started = threading.Event()
    release = threading.Event()

    def blocker(_ctx):
        started.set()
        assert release.wait(timeout=2)
        return "done"

    running = scheduler.submit(
        TaskSpec(chat_id="chat", name="running", queue_key="shared"),
        blocker,
    )
    assert started.wait(timeout=1)
    queued = scheduler.submit(
        TaskSpec(chat_id="chat", name="queued", queue_key="shared"),
        lambda _ctx: pytest.fail("fenced queued work must not execute"),
    )

    scheduler.fence_admission()

    assert queued.wait(timeout=1).status == TaskStatus.CANCELED
    with pytest.raises(RuntimeError, match="admission"):
        scheduler.submit(
            TaskSpec(chat_id="chat", name="late"),
            lambda _ctx: None,
        )

    release.set()
    assert running.wait(timeout=2).status == TaskStatus.SUCCEEDED
    scheduler.stop(wait=True, shutdown_executor=True)


def test_fence_cancels_worker_waiting_at_run_guard_as_canceled():
    attempted = threading.Event()
    canceled = threading.Event()

    class BlockingGuard:
        @contextmanager
        def task_guard(self):
            attempted.set()
            assert canceled.wait(timeout=2)
            raise RuntimeError("restart admission canceled")
            yield

        def cancel_waiters(self):
            canceled.set()

    guard = BlockingGuard()
    scheduler = TaskScheduler(
        max_concurrent=1,
        per_key_concurrency=1,
        run_guard=guard.task_guard,
    )
    callback_called = False

    def callback(_ctx):
        nonlocal callback_called
        callback_called = True

    handle = scheduler.submit(TaskSpec(chat_id="chat", name="waiting"), callback)
    assert attempted.wait(timeout=1)

    scheduler.fence_admission()

    result = handle.wait(timeout=2)
    assert result.status == TaskStatus.CANCELED
    assert not callback_called
    assert scheduler._running_total_normal == 0
    scheduler.stop(wait=True, shutdown_executor=True)


def test_bounded_drain_requests_cooperative_cancel_after_timeout():
    scheduler = TaskScheduler(max_concurrent=1, per_key_concurrency=1)
    started = threading.Event()

    def cooperative(ctx):
        started.set()
        while True:
            ctx.check_canceled()
            time.sleep(0.005)

    handle = scheduler.submit(TaskSpec(chat_id="chat", name="active"), cooperative)
    assert started.wait(timeout=1)
    scheduler.fence_admission()

    assert scheduler.wait_for_idle(timeout=0.02) is False
    assert scheduler.cancel_active() == 1
    assert handle.wait(timeout=2).status == TaskStatus.CANCELED
    assert scheduler.wait_for_idle(timeout=0.2) is True
    scheduler.stop(wait=True, shutdown_executor=True)


def test_base_exception_reaches_failed_terminal_before_guard_release():
    observed: list[TaskStatus] = []
    scheduler: TaskScheduler

    @contextmanager
    def guard():
        try:
            yield
        finally:
            state = scheduler.get_state(handle.run_id)
            assert state is not None
            observed.append(state.status)

    class FatalTaskSignal(BaseException):
        pass

    scheduler = TaskScheduler(
        max_concurrent=1,
        per_key_concurrency=1,
        run_guard=guard,
    )
    handle = scheduler.submit(
        TaskSpec(chat_id="chat", name="fatal"),
        lambda _ctx: (_ for _ in ()).throw(FatalTaskSignal("fatal callback")),
    )

    result = handle.wait(timeout=2)

    assert result.status == TaskStatus.FAILED
    assert result.error == "fatal callback"
    assert observed == [TaskStatus.FAILED]
    assert scheduler._running_total_normal == 0
    scheduler.stop(wait=True, shutdown_executor=True)


def test_guard_enter_base_exception_releases_slot_and_reaches_failed_terminal():
    class FatalGuardAdmission(BaseException):
        pass

    class FatalEnterGuard:
        def __enter__(self):
            raise FatalGuardAdmission("fatal guard admission")

        def __exit__(self, _exc_type, _exc, _traceback):
            return False

    scheduler = TaskScheduler(
        max_concurrent=1,
        per_key_concurrency=1,
        run_guard=FatalEnterGuard,
    )
    handle = scheduler.submit(
        TaskSpec(chat_id="chat", name="fatal-guard-enter"),
        lambda _ctx: pytest.fail("callback must not run after guard rejection"),
    )

    result = handle.wait(timeout=2)

    assert result.status == TaskStatus.FAILED
    assert result.error == "fatal guard admission"
    assert scheduler._running_total_normal == 0
    assert scheduler.wait_for_idle(timeout=0.2) is True
    scheduler.stop(wait=True, shutdown_executor=True)


def test_guard_exit_base_exception_does_not_escape_wait_or_rewrite_success():
    class FatalGuardRelease(BaseException):
        pass

    class FatalExitGuard:
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _traceback):
            raise FatalGuardRelease("fatal guard release")

    scheduler = TaskScheduler(
        max_concurrent=1,
        per_key_concurrency=1,
        run_guard=FatalExitGuard,
    )
    handle = scheduler.submit(
        TaskSpec(chat_id="chat", name="fatal-guard-exit"),
        lambda _ctx: "completed-before-release",
    )
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        future = handle.get_state().future
        if future is not None and future.done():
            break
        time.sleep(0.001)
    assert future is not None and future.done()

    result = handle.wait(timeout=2)

    assert result.status == TaskStatus.SUCCEEDED
    assert result.value == "completed-before-release"
    assert scheduler._running_total_normal == 0
    scheduler.stop(wait=True, shutdown_executor=True)


def test_scheduler_publishes_request_scoped_run_id_inside_callback():
    scheduler = TaskScheduler(max_concurrent=1, per_key_concurrency=1)
    observed: list[str | None] = []

    handle = scheduler.submit(
        TaskSpec(chat_id="chat", name="owner-context"),
        lambda _ctx: observed.append(get_current_task_run_id()),
    )

    assert handle.wait(timeout=2).status == TaskStatus.SUCCEEDED
    assert observed == [handle.run_id]
    assert get_current_task_run_id() is None
    scheduler.stop(wait=True, shutdown_executor=True)


def test_nested_submit_replaces_inherited_parent_run_id():
    scheduler = TaskScheduler(max_concurrent=2, per_key_concurrency=1)
    parent_observed: list[str | None] = []
    child_observed: list[str | None] = []
    child_handles = []

    def parent(_ctx):
        parent_observed.append(get_current_task_run_id())
        child_handles.append(
            scheduler.submit(
                TaskSpec(
                    chat_id="chat",
                    name="child",
                    queue_key="chat:child",
                ),
                lambda _child_ctx: child_observed.append(
                    get_current_task_run_id()
                ),
            )
        )

    parent_handle = scheduler.submit(
        TaskSpec(
            chat_id="chat",
            name="parent",
            queue_key="chat:parent",
        ),
        parent,
    )

    assert parent_handle.wait(timeout=2).status == TaskStatus.SUCCEEDED
    assert child_handles[0].wait(timeout=2).status == TaskStatus.SUCCEEDED
    assert parent_observed == [parent_handle.run_id]
    assert child_observed == [child_handles[0].run_id]
    assert child_observed != parent_observed
    assert get_current_task_run_id() is None
    scheduler.stop(wait=True, shutdown_executor=True)
