"""Focused contracts for the event-driven task scheduler surface."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import pytest

import src.tasking.scheduler as scheduler_module
from src.tasking.scheduler import (
    TaskEvent,
    TaskRunState,
    TaskScheduler,
    TaskSpec,
    TaskStatus,
)
from src.utils.rate_limit import RateLimitExceededException

TERMINAL = frozenset({TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELED})


class EventLog:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._events: list[TaskEvent] = []

    def __call__(self, event: TaskEvent) -> None:
        with self._condition:
            self._events.append(event)
            self._condition.notify_all()

    def wait_for(self, run_id: str, status: TaskStatus, timeout: float = 3.0) -> bool:
        with self._condition:
            return self._condition.wait_for(
                lambda: any(event.run_id == run_id and event.status is status for event in self._events),
                timeout=timeout,
            )

    def for_run(self, run_id: str) -> list[TaskEvent]:
        with self._condition:
            return [event for event in self._events if event.run_id == run_id]


class HoldingExecutor:
    """Executor test double that exposes a pending Future without running it."""

    def __init__(self) -> None:
        self.future: scheduler_module.Future[object] | None = None
        self._fn = None
        self._args: tuple[object, ...] = ()

    def submit(self, fn, *args):
        self.future = scheduler_module.Future()
        self._fn = fn
        self._args = args
        return self.future

    def run(self) -> None:
        assert self.future is not None
        assert self._fn is not None
        if not self.future.set_running_or_notify_cancel():
            return
        self.invoke()

    def invoke(self) -> None:
        assert self.future is not None
        assert self._fn is not None
        try:
            result = self._fn(*self._args)
        except BaseException as exc:
            self.future.set_exception(exc)
        else:
            self.future.set_result(result)

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        del wait
        if cancel_futures and self.future is not None:
            self.future.cancel()


def _scheduler(**kwargs) -> tuple[TaskScheduler, EventLog]:
    scheduler = TaskScheduler(**kwargs)
    events = EventLog()
    scheduler.add_listener(events)
    return scheduler, events


@pytest.mark.parametrize(
    ("initial", "target", "legal"),
    [
        (TaskStatus.QUEUED, TaskStatus.RUNNING, True),
        (TaskStatus.QUEUED, TaskStatus.FAILED, True),
        (TaskStatus.QUEUED, TaskStatus.CANCELED, True),
        (TaskStatus.RUNNING, TaskStatus.SUCCEEDED, True),
        (TaskStatus.RUNNING, TaskStatus.FAILED, True),
        (TaskStatus.RUNNING, TaskStatus.CANCELED, True),
        (TaskStatus.QUEUED, TaskStatus.SUCCEEDED, False),
        (TaskStatus.RUNNING, TaskStatus.QUEUED, False),
        (TaskStatus.SUCCEEDED, TaskStatus.FAILED, False),
        (TaskStatus.FAILED, TaskStatus.RUNNING, False),
        (TaskStatus.CANCELED, TaskStatus.RUNNING, False),
    ],
)
def test_transition_table(initial: TaskStatus, target: TaskStatus, legal: bool) -> None:
    scheduler = TaskScheduler(max_concurrent=1)
    try:
        state = TaskRunState(
            spec=TaskSpec(chat_id="chat", name="transition"),
            run_id="run",
            status=initial,
        )
        with scheduler._lock:
            scheduler._states[state.run_id] = state
            result = scheduler._transition_unlocked(state.run_id, target)

        assert (result is not None) is legal
        assert state.status is (target if legal else initial)
        if legal and target is TaskStatus.RUNNING:
            assert state.started_at is not None
        if legal and target in TERMINAL:
            assert state.ended_at is not None
    finally:
        scheduler.stop(shutdown_executor=True)


def test_terminal_state_fences_cancel_progress_and_late_transition() -> None:
    scheduler, events = _scheduler(max_concurrent=1)
    try:
        handle = scheduler.submit(
            TaskSpec(chat_id="chat", name="terminal", task_id="terminal-task"),
            lambda _ctx: None,
        )
        assert events.wait_for(handle.run_id, TaskStatus.SUCCEEDED)

        assert handle.cancel() is False
        scheduler.update_progress(handle.run_id, message="late", percent=99)
        with scheduler._lock:
            assert scheduler._transition_unlocked(handle.run_id, TaskStatus.FAILED) is None

        state = scheduler.get_state_by_task_id("terminal-task", "chat")
        assert state is not None
        assert state.status is TaskStatus.SUCCEEDED
        assert state.progress_message is None
        assert [event.status for event in events.for_run(handle.run_id)].count(TaskStatus.SUCCEEDED) == 1
    finally:
        scheduler.stop(shutdown_executor=True)


def test_handle_completion_replays_terminal_after_history_reap() -> None:
    scheduler, events = _scheduler(max_concurrent=1, max_terminal_history=1)
    try:
        first = scheduler.submit(
            TaskSpec(chat_id="chat", name="first"),
            lambda _ctx: None,
        )
        assert events.wait_for(first.run_id, TaskStatus.SUCCEEDED)
        second = scheduler.submit(
            TaskSpec(chat_id="chat", name="second"),
            lambda _ctx: None,
        )
        assert events.wait_for(second.run_id, TaskStatus.SUCCEEDED)
        assert scheduler.get_state(first.run_id) is None

        completed = threading.Event()
        observed: list[TaskEvent] = []

        def on_done(event: TaskEvent) -> None:
            observed.append(event)
            completed.set()

        first.add_done_callback(on_done)

        assert completed.wait(timeout=1)
        assert [event.status for event in observed] == [TaskStatus.SUCCEEDED]
        assert observed[0].run_id == first.run_id
    finally:
        scheduler.stop(shutdown_executor=True)


def test_handle_completion_callback_can_reenter_scheduler() -> None:
    scheduler, events = _scheduler(max_concurrent=1)
    callback_done = threading.Event()
    try:
        handle = scheduler.submit(
            TaskSpec(chat_id="chat", name="reentrant"),
            lambda _ctx: None,
        )

        def on_done(event: TaskEvent) -> None:
            assert scheduler.get_state(event.run_id) is not None
            callback_done.set()

        handle.add_done_callback(on_done)

        assert events.wait_for(handle.run_id, TaskStatus.SUCCEEDED)
        assert callback_done.wait(timeout=1)
    finally:
        scheduler.stop(shutdown_executor=True)


def test_poison_completion_callback_cannot_kill_event_dispatcher() -> None:
    scheduler, events = _scheduler(max_concurrent=1)
    survivor_completed = threading.Event()
    try:
        poison = scheduler.submit(
            TaskSpec(chat_id="chat", name="poison"),
            lambda _ctx: None,
        )
        poison.add_done_callback(
            lambda _event: (_ for _ in ()).throw(SystemExit("poison callback"))
        )
        assert events.wait_for(poison.run_id, TaskStatus.SUCCEEDED)

        survivor = scheduler.submit(
            TaskSpec(chat_id="chat", name="survivor"),
            lambda _ctx: None,
        )
        survivor.add_done_callback(lambda _event: survivor_completed.set())

        assert events.wait_for(survivor.run_id, TaskStatus.SUCCEEDED)
        assert survivor_completed.wait(timeout=1)
        assert scheduler._event_dispatcher.is_alive()
    finally:
        scheduler.stop(wait=True, shutdown_executor=True)


def test_completion_quiescence_waits_for_callbacks_without_dispatcher_self_wait() -> None:
    scheduler, events = _scheduler(max_concurrent=1)
    task_started = threading.Event()
    release_task = threading.Event()
    callback_started = threading.Event()
    self_wait_returned = threading.Event()
    release_callback = threading.Event()
    self_wait_results: list[bool] = []
    try:
        handle = scheduler.submit(
            TaskSpec(chat_id="chat", name="completion-quiescence"),
            lambda _ctx: (task_started.set(), release_task.wait(timeout=3)),
        )
        assert task_started.wait(timeout=1)

        def on_done(_event: TaskEvent) -> None:
            callback_started.set()
            try:
                self_wait_results.append(
                    scheduler.wait_for_completion_callbacks(timeout=0.1)
                )
            finally:
                self_wait_returned.set()
            release_callback.wait(timeout=3)

        handle.add_done_callback(on_done)
        release_task.set()
        assert events.wait_for(handle.run_id, TaskStatus.SUCCEEDED)
        assert callback_started.wait(timeout=1)
        assert self_wait_returned.wait(timeout=1)
        assert self_wait_results == [False]

        assert scheduler.wait_for_completion_callbacks(timeout=0) is False
        release_callback.set()
        assert scheduler.wait_for_completion_callbacks(timeout=1) is True
    finally:
        release_task.set()
        release_callback.set()
        scheduler.stop(wait=True, shutdown_executor=True)


def test_completion_quiescence_tracks_terminal_replay_on_registering_thread() -> None:
    scheduler, events = _scheduler(max_concurrent=1)
    callback_started = threading.Event()
    self_wait_returned = threading.Event()
    release_callback = threading.Event()
    self_wait_results: list[bool] = []
    registration_thread: threading.Thread | None = None
    try:
        handle = scheduler.submit(
            TaskSpec(chat_id="chat", name="terminal-replay-quiescence"),
            lambda _ctx: None,
        )
        assert events.wait_for(handle.run_id, TaskStatus.SUCCEEDED)

        def on_done(_event: TaskEvent) -> None:
            callback_started.set()
            try:
                self_wait_results.append(
                    scheduler.wait_for_completion_callbacks(timeout=0.1)
                )
            finally:
                self_wait_returned.set()
            release_callback.wait(timeout=3)

        registration_thread = threading.Thread(
            target=handle.add_done_callback,
            args=(on_done,),
        )
        registration_thread.start()
        assert callback_started.wait(timeout=1)
        assert self_wait_returned.wait(timeout=1)
        assert self_wait_results == [False]
        assert scheduler.wait_for_completion_callbacks(timeout=0) is False

        release_callback.set()
        registration_thread.join(timeout=1)
        assert not registration_thread.is_alive()
        assert scheduler.wait_for_completion_callbacks(timeout=1) is True
    finally:
        release_callback.set()
        if registration_thread is not None:
            registration_thread.join(timeout=1)
        scheduler.stop(wait=True, shutdown_executor=True)


def test_terminal_replay_registration_linearizes_before_quiescence_wait() -> None:
    scheduler, events = _scheduler(max_concurrent=1)
    completion_lock_exited = threading.Event()
    release_completion_exit = threading.Event()
    callback_started = threading.Event()
    release_callback = threading.Event()
    waiter_done = threading.Event()
    wait_results: list[bool] = []
    registration_thread: threading.Thread | None = None
    waiter_thread: threading.Thread | None = None
    try:
        handle = scheduler.submit(
            TaskSpec(chat_id="chat", name="terminal-replay-linearization"),
            lambda _ctx: None,
        )
        assert events.wait_for(handle.run_id, TaskStatus.SUCCEEDED)

        completion = handle._completion
        original_lock = completion._lock

        class CompletionExitBarrier:
            def __enter__(self):
                original_lock.acquire()
                return self

            def __exit__(self, _exc_type, _exc, _traceback) -> None:
                original_lock.release()
                completion_lock_exited.set()
                release_completion_exit.wait(timeout=3)

        completion._lock = CompletionExitBarrier()

        registration_thread = threading.Thread(
            target=handle.add_done_callback,
            args=(lambda _event: (callback_started.set(), release_callback.wait(timeout=3)),),
        )
        registration_thread.start()
        assert completion_lock_exited.wait(timeout=1)

        def wait_for_quiescence() -> None:
            wait_results.append(scheduler.wait_for_completion_callbacks(timeout=0))
            waiter_done.set()

        waiter_thread = threading.Thread(target=wait_for_quiescence)
        waiter_thread.start()
        assert waiter_done.wait(timeout=0.2) is False

        release_completion_exit.set()
        assert callback_started.wait(timeout=1)
        assert waiter_done.wait(timeout=1)
        assert wait_results == [False]

        release_callback.set()
        registration_thread.join(timeout=1)
        waiter_thread.join(timeout=1)
        assert not registration_thread.is_alive()
        assert not waiter_thread.is_alive()
    finally:
        release_completion_exit.set()
        release_callback.set()
        if registration_thread is not None:
            registration_thread.join(timeout=1)
        if waiter_thread is not None:
            waiter_thread.join(timeout=1)
        scheduler.stop(wait=True, shutdown_executor=True)


def test_listener_can_reenter_and_remove_itself_without_deadlock() -> None:
    scheduler, events = _scheduler(max_concurrent=1)
    callback_done = threading.Event()
    observed: list[tuple[str, TaskStatus]] = []

    def listener(event: TaskEvent) -> None:
        observed.append((event.run_id, event.status))
        assert scheduler.get_state(event.run_id) is not None
        if event.status is TaskStatus.SUCCEEDED:
            assert scheduler.remove_listener(listener) is True
            scheduler.submit(
                TaskSpec(chat_id="follow-up", name="follow-up"),
                lambda _ctx: None,
            )
            callback_done.set()

    scheduler.add_listener(listener)
    try:
        handle = scheduler.submit(
            TaskSpec(chat_id="chat", name="reentrant-listener"),
            lambda _ctx: None,
        )
        assert events.wait_for(handle.run_id, TaskStatus.SUCCEEDED)
        assert callback_done.wait(timeout=1)
        assert [status for run_id, status in observed if run_id == handle.run_id] == [
            TaskStatus.QUEUED,
            TaskStatus.RUNNING,
            TaskStatus.SUCCEEDED,
        ]
    finally:
        scheduler.stop(shutdown_executor=True)


def test_executor_prestart_cancel_completes_and_releases_slot_once() -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    executor_blocked = threading.Event()
    release_executor = threading.Event()
    blocker = executor.submit(
        lambda: (executor_blocked.set(), release_executor.wait(timeout=3))
    )
    assert executor_blocked.wait(timeout=1)
    scheduler, events = _scheduler(max_concurrent=1, worker_executor=executor)
    completed = threading.Event()
    terminal_events: list[TaskEvent] = []
    callback_started = threading.Event()
    try:
        handle = scheduler.submit(
            TaskSpec(chat_id="chat", name="prestart-cancel"),
            lambda _ctx: callback_started.set(),
        )
        handle.add_done_callback(
            lambda event: (terminal_events.append(event), completed.set())
        )
        assert events.wait_for(handle.run_id, TaskStatus.RUNNING)

        scheduler.stop(shutdown_executor=True)

        assert completed.wait(timeout=1)
        assert [event.status for event in terminal_events] == [TaskStatus.CANCELED]
        assert not callback_started.is_set()
        assert scheduler.wait_for_idle(timeout=1)
        with scheduler._lock:
            assert scheduler._running_total_normal == 0
            assert scheduler._active_run_ids == set()
            assert scheduler._running_by_key == {}
    finally:
        release_executor.set()
        blocker.result(timeout=1)
        scheduler.stop(shutdown_executor=True)


def test_worker_start_wins_cancel_race_without_double_releasing_slot() -> None:
    executor = HoldingExecutor()
    scheduler, events = _scheduler(
        max_concurrent=1,
        worker_executor=executor,
    )
    callback_started = threading.Event()
    release_callback = threading.Event()
    completed = threading.Event()
    terminal_events: list[TaskEvent] = []
    try:
        handle = scheduler.submit(
            TaskSpec(chat_id="chat", name="start-wins"),
            lambda _ctx: (
                callback_started.set(),
                release_callback.wait(timeout=3),
            ),
        )
        handle.add_done_callback(
            lambda event: (terminal_events.append(event), completed.set())
        )
        assert events.wait_for(handle.run_id, TaskStatus.RUNNING)
        assert executor.future is not None

        worker = threading.Thread(target=executor.run)
        worker.start()
        assert callback_started.wait(timeout=1)
        assert handle.cancel() is True
        release_callback.set()
        worker.join(timeout=1)

        assert events.wait_for(handle.run_id, TaskStatus.CANCELED)
        assert completed.wait(timeout=1)
        assert [event.status for event in terminal_events] == [TaskStatus.CANCELED]
        assert scheduler.wait_for_idle(timeout=1)
        with scheduler._lock:
            assert scheduler._running_total_normal == 0
            assert scheduler._active_run_ids == set()
            assert scheduler._running_by_key == {}
    finally:
        release_callback.set()
        scheduler.stop(shutdown_executor=True)


def test_future_running_before_wrapper_cancel_still_converges_once() -> None:
    executor = HoldingExecutor()
    scheduler, events = _scheduler(
        max_concurrent=1,
        worker_executor=executor,
    )
    callback_started = threading.Event()
    completed = threading.Event()
    terminal_events: list[TaskEvent] = []
    try:
        handle = scheduler.submit(
            TaskSpec(
                chat_id="chat",
                project_id="project",
                name="future-running-wrapper-pending",
            ),
            lambda _ctx: callback_started.set(),
        )
        handle.add_done_callback(
            lambda event: (terminal_events.append(event), completed.set())
        )
        assert events.wait_for(handle.run_id, TaskStatus.RUNNING)
        assert executor.future is not None
        assert executor.future.set_running_or_notify_cancel()

        assert handle.cancel() is True
        worker = threading.Thread(target=executor.invoke)
        worker.start()
        worker.join(timeout=1)

        assert events.wait_for(handle.run_id, TaskStatus.CANCELED)
        assert completed.wait(timeout=1)
        assert [event.status for event in terminal_events] == [TaskStatus.CANCELED]
        assert not callback_started.is_set()
        assert scheduler.wait_for_idle(timeout=1)
        with scheduler._lock:
            assert scheduler._running_total_normal == 0
            assert scheduler._active_run_ids == set()
            assert scheduler._running_by_key == {}
            assert scheduler._running_by_project == {}
    finally:
        scheduler.stop(shutdown_executor=True)


def test_listener_reports_progress_in_order_and_isolates_failures() -> None:
    scheduler, events = _scheduler(max_concurrent=1)

    def broken_listener(_event: TaskEvent) -> None:
        raise RuntimeError("listener failed")

    scheduler.add_listener(broken_listener)

    def work(ctx) -> None:
        ctx.progress("reading", percent=25)
        ctx.progress("writing", percent=75)

    try:
        handle = scheduler.submit(TaskSpec(chat_id="chat", name="observed"), work)
        assert events.wait_for(handle.run_id, TaskStatus.SUCCEEDED)

        run_events = events.for_run(handle.run_id)
        assert run_events[0].status is TaskStatus.QUEUED
        assert run_events[-1].status is TaskStatus.SUCCEEDED
        assert [event.progress_message for event in run_events if event.progress_message] == [
            "reading",
            "writing",
        ]
    finally:
        scheduler.stop(shutdown_executor=True)


def test_task_chat_and_project_queries_are_scoped() -> None:
    scheduler, events = _scheduler(max_concurrent=3, per_key_concurrency=1)
    specs = [
        TaskSpec(chat_id="chat-a", project_id="project-1", task_id="task-alpha-0001", name="a"),
        TaskSpec(chat_id="chat-a", project_id="project-2", task_id="task-beta-0002", name="b"),
        TaskSpec(chat_id="chat-b", project_id="project-1", task_id="task-gamma-0003", name="c"),
    ]
    try:
        handles = [scheduler.submit(spec, lambda _ctx: None) for spec in specs]
        assert all(events.wait_for(handle.run_id, TaskStatus.SUCCEEDED) for handle in handles)

        assert scheduler.list_tasks(chat_id="chat-a") == []
        assert {state.spec.task_id for state in scheduler.list_tasks(chat_id="chat-a", include_done=True)} == {
            "task-alpha-0001",
            "task-beta-0002",
        }
        assert {
            state.spec.task_id
            for state in scheduler.list_tasks(project_id="project-1", include_done=True)
        } == {"task-alpha-0001", "task-gamma-0003"}
        assert {
            state.spec.task_id
            for state in scheduler.list_tasks(
                chat_id="chat-a",
                project_id="project-1",
                include_done=True,
            )
        } == {"task-alpha-0001"}

        assert scheduler.get_state_by_task_id("0002", "chat-a").spec.name == "b"
        assert scheduler.get_state_by_task_id("task-beta-0002", "chat-b") is None
    finally:
        scheduler.stop(shutdown_executor=True)


def test_concurrent_submission_is_thread_safe_and_serial_per_queue() -> None:
    scheduler, events = _scheduler(max_concurrent=6, per_key_concurrency=1)
    lock = threading.Lock()
    active_total = 0
    max_total = 0
    active_by_key: dict[str, int] = defaultdict(int)
    max_by_key: dict[str, int] = defaultdict(int)

    def work(key: str):
        def run(_ctx) -> None:
            nonlocal active_total, max_total
            with lock:
                active_total += 1
                active_by_key[key] += 1
                max_total = max(max_total, active_total)
                max_by_key[key] = max(max_by_key[key], active_by_key[key])
            time.sleep(0.003)
            with lock:
                active_total -= 1
                active_by_key[key] -= 1

        return run

    try:
        handles = []
        for index in range(48):
            queue_key = f"chat-{index % 4}:queue"
            handles.append(
                scheduler.submit(
                    TaskSpec(chat_id=f"chat-{index % 4}", queue_key=queue_key, name=f"task-{index}"),
                    work(queue_key),
                )
            )

        assert all(events.wait_for(handle.run_id, TaskStatus.SUCCEEDED, timeout=5) for handle in handles)
        assert max_total <= 6
        assert max(max_by_key.values()) == 1
        assert len(scheduler.list_tasks(include_done=True, limit=100)) == len(handles)
    finally:
        scheduler.stop(shutdown_executor=True)


def test_concurrent_submissions_with_shared_uuid_prefix_keep_unique_full_run_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_count = 8
    generated_ids = iter(
        scheduler_module.uuid.UUID(f"12345678-9000-4000-8000-{index:012x}") for index in range(task_count)
    )
    id_lock = threading.Lock()

    def next_uuid():
        with id_lock:
            return next(generated_ids)

    monkeypatch.setattr(scheduler_module.uuid, "uuid4", next_uuid)
    scheduler, _events = _scheduler(max_concurrent=1, per_key_concurrency=1)
    submissions_ready = threading.Barrier(task_count, timeout=3)
    release = threading.Event()

    def submit(index: int):
        submissions_ready.wait()
        return scheduler.submit(
            TaskSpec(chat_id="collision-chat", name=f"collision-{index}"),
            lambda _ctx: release.wait(timeout=3),
        )

    try:
        with ThreadPoolExecutor(max_workers=task_count) as pool:
            futures = {index: pool.submit(submit, index) for index in range(task_count)}
            handles = {index: future.result(timeout=3) for index, future in futures.items()}

        assert len({handle.run_id for handle in handles.values()}) == task_count
        assert all(len(handle.run_id) == 32 for handle in handles.values())
        for index, handle in handles.items():
            state = scheduler.get_state(handle.run_id)
            assert state is not None
            assert state.spec.name == f"collision-{index}"
    finally:
        release.set()
        scheduler.stop(shutdown_executor=True)


def test_exact_run_id_collision_retries_without_crossing_cancel_state_or_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duplicate = scheduler_module.uuid.UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    replacement = scheduler_module.uuid.UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
    generated_ids = iter((duplicate, duplicate, replacement))
    monkeypatch.setattr(scheduler_module.uuid, "uuid4", lambda: next(generated_ids))

    scheduler, events = _scheduler(
        max_concurrent=1,
        per_key_concurrency=1,
        system_concurrency=1,
    )
    first_started = threading.Event()
    second_started = threading.Event()
    release_first = threading.Event()
    release_second = threading.Event()
    observed_canceled: dict[str, bool] = {}

    def block(name: str, started: threading.Event, release: threading.Event):
        def run(ctx) -> None:
            started.set()
            release.wait(timeout=3)
            observed_canceled[name] = ctx.cancel_token.is_canceled
            ctx.check_canceled()

        return run

    try:
        first = scheduler.submit(
            TaskSpec(
                chat_id="collision-chat",
                queue_key="collision:first",
                project_id="first-project",
                task_id="collision-first",
                name="first",
            ),
            block("first", first_started, release_first),
        )
        assert first_started.wait(timeout=1)
        second = scheduler.submit(
            TaskSpec(
                chat_id="collision-chat",
                project_id="second-project",
                task_id="collision-second",
                name="second",
                is_system_command=True,
            ),
            block("second", second_started, release_second),
        )
        assert second_started.wait(timeout=1)

        assert first.cancel() is True
        release_second.set()
        assert events.wait_for(second.run_id, TaskStatus.SUCCEEDED)
        release_first.set()
        assert events.wait_for(first.run_id, TaskStatus.CANCELED)
        assert observed_canceled == {"second": False, "first": True}
        assert first.run_id == duplicate.hex
        assert second.run_id == replacement.hex

        first_state = scheduler.get_state_by_task_id("collision-first", "collision-chat")
        second_state = scheduler.get_state_by_task_id("collision-second", "collision-chat")
        assert first_state is not None
        assert second_state is not None
        assert first_state.run_id == first.run_id
        assert second_state.run_id == second.run_id
        assert first_state.status is TaskStatus.CANCELED
        assert second_state.status is TaskStatus.SUCCEEDED
        assert scheduler.wait_for_idle(timeout=1)

        with scheduler._lock:
            assert scheduler._running_total_normal == 0
            assert scheduler._running_total_system == 0
            assert scheduler._active_run_ids == set()
            assert scheduler._running_by_key == {}
            assert scheduler._running_by_project == {}
            assert scheduler._pending_normal == 0
            assert scheduler._pending_system == 0
    finally:
        release_first.set()
        release_second.set()
        scheduler.stop(shutdown_executor=True)


def test_run_id_collision_retry_exhaustion_leaves_scheduler_usable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duplicate = scheduler_module.uuid.UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
    recovery = scheduler_module.uuid.UUID("dddddddd-dddd-4ddd-8ddd-dddddddddddd")
    monkeypatch.setattr(scheduler_module.uuid, "uuid4", lambda: duplicate)
    scheduler, events = _scheduler(max_concurrent=1, per_key_concurrency=1)
    first_started = threading.Event()
    release_first = threading.Event()

    def block_first(_ctx) -> None:
        first_started.set()
        release_first.wait(timeout=3)

    try:
        first = scheduler.submit(
            TaskSpec(chat_id="collision-running", name="collision-running"),
            block_first,
        )
        assert first_started.wait(timeout=1)

        collision_calls = 0

        def repeated_collision():
            nonlocal collision_calls
            collision_calls += 1
            if collision_calls > 100:
                raise AssertionError("run_id allocation did not stop")
            return duplicate

        monkeypatch.setattr(scheduler_module.uuid, "uuid4", repeated_collision)
        with pytest.raises(RuntimeError, match="unique run_id"):
            scheduler.submit(
                TaskSpec(
                    chat_id="collision-rejected",
                    project_id="collision-rejected-project",
                    task_id="collision-rejected-task",
                    name="collision-rejected",
                ),
                lambda _ctx: None,
            )

        with scheduler._lock:
            assert len(scheduler._states) == 1
            assert "collision-rejected" not in scheduler._by_chat
            assert "collision-rejected-project" not in scheduler._by_project
            assert "collision-rejected-task" not in scheduler._by_task_id
            assert scheduler._pending_normal == 0

        monkeypatch.setattr(scheduler_module.uuid, "uuid4", lambda: recovery)
        queued = scheduler.submit(
            TaskSpec(chat_id="collision-recovery", name="collision-recovery"),
            lambda _ctx: None,
        )
        assert queued.cancel() is True
        assert events.wait_for(queued.run_id, TaskStatus.CANCELED)
        assert scheduler.get_state(first.run_id).status is TaskStatus.RUNNING

        release_first.set()
        assert events.wait_for(first.run_id, TaskStatus.SUCCEEDED)
        assert scheduler.wait_for_idle(timeout=1)
    finally:
        release_first.set()
        scheduler.stop(shutdown_executor=True)


def test_queued_and_running_cancellation_reach_terminal_state() -> None:
    scheduler, events = _scheduler(max_concurrent=1, per_key_concurrency=1)
    blocker_started = threading.Event()
    release_blocker = threading.Event()

    def blocker(_ctx) -> None:
        blocker_started.set()
        release_blocker.wait(timeout=2)

    try:
        first = scheduler.submit(TaskSpec(chat_id="chat", name="blocker"), blocker)
        assert blocker_started.wait(timeout=1)
        queued = scheduler.submit(TaskSpec(chat_id="chat", name="queued"), lambda _ctx: None)
        assert queued.cancel() is True
        assert events.wait_for(queued.run_id, TaskStatus.CANCELED)
        release_blocker.set()
        assert events.wait_for(first.run_id, TaskStatus.SUCCEEDED)

        running_started = threading.Event()

        def cooperative(ctx) -> None:
            running_started.set()
            while True:
                ctx.check_canceled()
                time.sleep(0.005)

        running = scheduler.submit(TaskSpec(chat_id="other", name="running"), cooperative)
        assert running_started.wait(timeout=1)
        assert running.cancel() is True
        assert events.wait_for(running.run_id, TaskStatus.CANCELED)
    finally:
        release_blocker.set()
        scheduler.stop(shutdown_executor=True)


def test_terminal_ttl_reap_cleans_state_and_indexes() -> None:
    scheduler, events = _scheduler(max_concurrent=1)
    try:
        handle = scheduler.submit(
            TaskSpec(
                chat_id="chat",
                project_id="project",
                task_id="durable-task-id",
                name="ttl",
            ),
            lambda _ctx: None,
        )
        assert events.wait_for(handle.run_id, TaskStatus.SUCCEEDED)
        assert scheduler.get_state_by_task_id("durable-task-id", "chat") is not None

        with scheduler._cv:
            assert scheduler._reap_completed_states(max_age_seconds=0) == 1

        assert scheduler.get_state_by_task_id("durable-task-id", "chat") is None
        assert scheduler.list_tasks(chat_id="chat", include_done=True) == []
        assert scheduler.list_tasks(project_id="project", include_done=True) == []
    finally:
        scheduler.stop(shutdown_executor=True)


def test_pending_capacity_is_independent_and_rejection_leaves_no_partial_state() -> None:
    assert issubclass(scheduler_module.TaskQueueFullError, RateLimitExceededException)

    scheduler, events = _scheduler(
        max_concurrent=1,
        system_concurrency=1,
        max_pending_normal=1,
        max_pending_system=1,
    )
    normal_started = threading.Event()
    system_started = threading.Event()
    release = threading.Event()

    def block(started: threading.Event):
        def run(_ctx) -> None:
            started.set()
            release.wait(timeout=3)

        return run

    try:
        normal_running = scheduler.submit(
            TaskSpec(chat_id="normal-running", name="normal-running"),
            block(normal_started),
        )
        system_running = scheduler.submit(
            TaskSpec(
                chat_id="system-running",
                name="system-running",
                is_system_command=True,
            ),
            block(system_started),
        )
        assert normal_started.wait(timeout=1)
        assert system_started.wait(timeout=1)

        normal_pending = scheduler.submit(
            TaskSpec(chat_id="normal-pending", name="normal-pending"),
            lambda _ctx: None,
        )
        system_pending = scheduler.submit(
            TaskSpec(
                chat_id="system-pending",
                name="system-pending",
                is_system_command=True,
            ),
            lambda _ctx: None,
        )

        with pytest.raises(scheduler_module.TaskQueueFullError):
            scheduler.submit(
                TaskSpec(
                    chat_id="normal-rejected",
                    project_id="rejected-project",
                    task_id="rejected-task-id",
                    name="normal-rejected",
                ),
                lambda _ctx: None,
            )
        with pytest.raises(scheduler_module.TaskQueueFullError):
            scheduler.submit(
                TaskSpec(
                    chat_id="system-rejected",
                    project_id="rejected-system-project",
                    task_id="rejected-system-task-id",
                    name="system-rejected",
                    is_system_command=True,
                ),
                lambda _ctx: None,
            )

        with scheduler._lock:
            assert all(
                state.spec.chat_id not in {"normal-rejected", "system-rejected"}
                for state in scheduler._states.values()
            )
            assert "normal-rejected" not in scheduler._by_chat
            assert "system-rejected" not in scheduler._by_chat
            assert "rejected-project" not in scheduler._by_project
            assert "rejected-system-project" not in scheduler._by_project
            assert "rejected-task-id" not in scheduler._by_task_id
            assert "rejected-system-task-id" not in scheduler._by_task_id
            assert "normal-rejected:rejected-project" not in scheduler._queues
            assert "system-rejected:SYSTEM" not in scheduler._queues

        release.set()
        for handle in (
            normal_running,
            system_running,
            normal_pending,
            system_pending,
        ):
            assert events.wait_for(handle.run_id, TaskStatus.SUCCEEDED)
    finally:
        release.set()
        scheduler.stop(shutdown_executor=True)


def test_terminal_history_and_idle_keys_stay_bounded_after_sustained_activity() -> None:
    scheduler, events = _scheduler(
        max_concurrent=4,
        per_key_concurrency=1,
        max_pending_normal=100,
        max_terminal_history=3,
    )
    try:
        handles = [
            scheduler.submit(
                TaskSpec(
                    chat_id=f"history-chat-{index}",
                    project_id=f"history-project-{index}",
                    task_id=f"history-task-{index}",
                    name=f"history-{index}",
                ),
                lambda _ctx: None,
            )
            for index in range(30)
        ]
        assert all(
            events.wait_for(handle.run_id, TaskStatus.SUCCEEDED, timeout=5)
            for handle in handles
        )
        assert scheduler.wait_for_idle(timeout=2)

        with scheduler._lock:
            terminal_states = [
                state
                for state in scheduler._states.values()
                if state.status in TERMINAL
            ]
            assert len(terminal_states) <= 3
            assert scheduler._queues == {}
            assert scheduler._running_by_key == {}
            assert scheduler._running_by_project == {}
            assert scheduler._pending_normal == 0
            assert scheduler._pending_system == 0
            assert len(scheduler._by_chat) <= 3
            assert len(scheduler._by_project) <= 3
            assert len(scheduler._by_task_id) <= 3
    finally:
        scheduler.stop(shutdown_executor=True)


def test_terminal_history_does_not_reap_newer_state_behind_active_head() -> None:
    scheduler = TaskScheduler(max_concurrent=1, max_terminal_history=1)
    try:
        old = TaskRunState(
            spec=TaskSpec(chat_id="chat", name="old-active"),
            run_id="old-active",
            status=TaskStatus.SUCCEEDED,
            ended_at=time.time(),
        )
        newer = TaskRunState(
            spec=TaskSpec(chat_id="chat", name="newer-inactive"),
            run_id="newer-inactive",
            status=TaskStatus.SUCCEEDED,
            ended_at=time.time(),
        )
        with scheduler._lock:
            scheduler._states = {old.run_id: old, newer.run_id: newer}
            scheduler._terminal_order = scheduler_module.deque(
                (old.run_id, newer.run_id)
            )
            scheduler._active_run_ids.add(old.run_id)

            assert scheduler._reap_completed_states() == 0
            assert list(scheduler._terminal_order) == [old.run_id, newer.run_id]
            assert set(scheduler._states) == {old.run_id, newer.run_id}

            scheduler._active_run_ids.remove(old.run_id)
            assert scheduler._reap_completed_states() == 1
            assert list(scheduler._terminal_order) == [newer.run_id]
            assert set(scheduler._states) == {newer.run_id}
    finally:
        scheduler.stop(shutdown_executor=True)


def test_reaping_old_duplicate_task_id_preserves_newer_mapping() -> None:
    scheduler, events = _scheduler(max_concurrent=1, max_terminal_history=10)
    release = threading.Event()
    replacement_started = threading.Event()
    try:
        old = scheduler.submit(
            TaskSpec(chat_id="chat", task_id="shared-task-id", name="old"),
            lambda _ctx: None,
        )
        assert events.wait_for(old.run_id, TaskStatus.SUCCEEDED)

        def block(_ctx) -> None:
            replacement_started.set()
            release.wait(timeout=3)

        replacement = scheduler.submit(
            TaskSpec(chat_id="chat", task_id="shared-task-id", name="replacement"),
            block,
        )
        assert replacement_started.wait(timeout=1)

        with scheduler._cv:
            assert scheduler._reap_completed_states(max_age_seconds=0) == 1

        state = scheduler.get_state_by_task_id("shared-task-id", "chat")
        assert state is not None
        assert state.run_id == replacement.run_id
    finally:
        release.set()
        scheduler.stop(shutdown_executor=True)
