"""Lifecycle contracts for TaskScheduler event listeners."""

from __future__ import annotations

import threading

from src.tasking.scheduler import TaskEvent, TaskScheduler, TaskSpec, TaskStatus


def test_remove_listener_is_idempotent_and_stops_future_events() -> None:
    scheduler = TaskScheduler(max_concurrent=1)
    first_succeeded = threading.Event()
    second_succeeded = threading.Event()
    observed_run_ids: list[str] = []

    def listener(event: TaskEvent) -> None:
        observed_run_ids.append(event.run_id)
        if event.status is TaskStatus.SUCCEEDED:
            first_succeeded.set()

    try:
        scheduler.add_listener(listener)
        first = scheduler.submit(
            TaskSpec(chat_id="listener-chat", name="first"),
            lambda _ctx: None,
        )
        assert first_succeeded.wait(timeout=1)

        assert scheduler.remove_listener(listener) is True
        assert scheduler.remove_listener(listener) is False

        scheduler.add_listener(
            lambda event: second_succeeded.set()
            if event.status is TaskStatus.SUCCEEDED
            else None
        )
        second = scheduler.submit(
            TaskSpec(chat_id="listener-chat", name="second"),
            lambda _ctx: None,
        )
        assert second_succeeded.wait(timeout=1)
        assert scheduler.get_state(second.run_id).status is TaskStatus.SUCCEEDED
        assert first.run_id in observed_run_ids
        assert second.run_id not in observed_run_ids
    finally:
        scheduler.stop(wait=True, shutdown_executor=True)
