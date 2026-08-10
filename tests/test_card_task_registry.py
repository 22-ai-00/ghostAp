"""Focused contracts for the per-execution card task registry."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from src.card.task_registry import TaskItem, TaskRegistry


def test_register_update_and_snapshot_are_ordered_and_immutable() -> None:
    registry = TaskRegistry()
    original = registry.register("task-2", "second")
    registry.register("task-1", "first", status="in_progress")
    registry.register("task-2", "second revision")

    updated = registry.update_status("task-2", "completed")
    snapshot = registry.get_snapshot()

    assert isinstance(original, TaskItem)
    assert original.name == "second"
    assert updated is not None and updated.status == "completed"
    assert [(item.task_id, item.name, item.status) for item in snapshot] == [
        ("task-2", "second revision", "completed"),
        ("task-1", "first", "in_progress"),
    ]
    assert registry.count == 2
    assert registry.update_status("missing", "failed") is None


def test_current_operation_name_keeps_only_the_latest_value() -> None:
    registry = TaskRegistry()
    registry.register("task", "queued")

    first = registry.update_name("task", "reading repository")
    latest = registry.update_name("task", "running targeted edit")

    assert first is not None and first.name == "reading repository"
    assert latest is not None and latest.name == "running targeted edit"
    assert registry.get("task") == latest
    assert registry.update_name("task", "   ") is None


def test_status_subscription_is_change_only_unsubscribable_and_failure_isolated() -> None:
    registry = TaskRegistry()
    registry.register("task", "work")
    received: list[tuple[str, str]] = []

    def broken(_task_id: str, _status: str) -> None:
        raise RuntimeError("subscriber failed")

    def listener(task_id: str, status: str) -> None:
        received.append((task_id, status))

    registry.subscribe(broken)
    registry.subscribe(listener)
    registry.update_status("task", "in_progress")
    registry.update_status("task", "in_progress")
    registry.update_status("task", "completed", notify=False)
    registry.unsubscribe(listener)
    registry.update_status("task", "failed")

    assert received == [("task", "in_progress")]


def test_registry_remains_consistent_during_concurrent_reads_and_writes() -> None:
    registry = TaskRegistry()
    observed_sizes: list[int] = []
    stop_reader = threading.Event()

    def reader() -> None:
        while not stop_reader.is_set():
            observed_sizes.append(len(registry.get_snapshot()))

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda index: registry.register(f"task-{index}", f"operation-{index}"), range(100)))
            list(
                pool.map(
                    lambda index: registry.update_status(f"task-{index}", "completed"),
                    range(100),
                )
            )
    finally:
        stop_reader.set()
        reader_thread.join(timeout=1)

    snapshot = registry.get_snapshot()
    assert registry.count == 100
    assert len(snapshot) == 100
    assert all(item.status == "completed" for item in snapshot)
    assert all(0 <= size <= 100 for size in observed_sizes)
