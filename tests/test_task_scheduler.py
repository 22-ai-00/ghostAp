"""Focused contracts for the event-driven task scheduler surface."""

from __future__ import annotations

import threading
import time
from collections import defaultdict

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
