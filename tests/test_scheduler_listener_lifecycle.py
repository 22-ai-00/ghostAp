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


def test_remove_listener_waits_for_inflight_callback_and_skips_queued_events() -> None:
    scheduler = TaskScheduler(max_concurrent=1)
    callback_entered = threading.Event()
    release_callback = threading.Event()
    remove_returned = threading.Event()
    observed: list[TaskStatus] = []

    def listener(event: TaskEvent) -> None:
        observed.append(event.status)
        if event.status is TaskStatus.QUEUED:
            callback_entered.set()
            release_callback.wait(timeout=3)

    scheduler.add_listener(listener)
    try:
        handle = scheduler.submit(
            TaskSpec(chat_id="listener-chat", name="blocked-listener"),
            lambda _ctx: None,
        )
        assert callback_entered.wait(timeout=1)

        remover = threading.Thread(
            target=lambda: (
                scheduler.remove_listener(listener),
                remove_returned.set(),
            )
        )
        remover.start()
        assert not remove_returned.wait(timeout=0.05)

        release_callback.set()
        assert remove_returned.wait(timeout=1)
        remover.join(timeout=1)
        assert scheduler.get_state(handle.run_id).status is TaskStatus.SUCCEEDED
        assert observed == [TaskStatus.QUEUED]
    finally:
        release_callback.set()
        scheduler.stop(wait=True, shutdown_executor=True)


def test_poison_listener_cannot_kill_following_event_delivery() -> None:
    scheduler = TaskScheduler(max_concurrent=1)
    survivor_completed = threading.Event()

    def poison(_event: TaskEvent) -> None:
        raise SystemExit("poison listener")

    def survivor(event: TaskEvent) -> None:
        if event.status is TaskStatus.SUCCEEDED:
            survivor_completed.set()

    scheduler.add_listener(poison)
    scheduler.add_listener(survivor)
    try:
        scheduler.submit(
            TaskSpec(chat_id="listener-chat", name="survivor"),
            lambda _ctx: None,
        )
        assert survivor_completed.wait(timeout=1)
        assert scheduler._event_dispatcher.is_alive()
    finally:
        scheduler.stop(wait=True, shutdown_executor=True)
