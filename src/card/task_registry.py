"""Task registry for task-level card management.

Provides a thread-safe, per-execution TaskRegistry that maintains
the single source of truth (SSOT) for all tasks in a programming session.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable, Literal

from src.card.tool_display import summarize_tool_call_content

if TYPE_CHECKING:
    from src.acp.models import PlanEntryInfo
    from src.spec_engine.models import SpecTask

TaskStatus = Literal["pending", "in_progress", "completed", "failed", "cancelled"]

StatusCallback = Callable[["str", "TaskStatus"], None]


@dataclass(frozen=True)
class TaskItem:
    """Immutable snapshot of a single task."""

    task_id: str
    name: str
    status: TaskStatus = "pending"
    session_id: str | None = None


@dataclass(frozen=True)
class TaskSnapshot:
    """Immutable snapshot of a task for payload/rendering."""

    task_id: str
    name: str
    status: TaskStatus


class TaskRegistry:
    """Thread-safe registry of tasks for a single execution session.

    NOT a process-level singleton — one instance per engine execution.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._tasks: dict[str, TaskItem] = {}
        self._order: list[str] = []  # Insertion order
        self._subscribers: list[StatusCallback] = []

    def register(
        self,
        task_id: str,
        name: str,
        *,
        status: TaskStatus = "pending",
        session_id: str | None = None,
    ) -> TaskItem:
        """Register a new task. Idempotent — updates if already exists."""
        item = TaskItem(task_id=task_id, name=name, status=status, session_id=session_id)
        with self._lock:
            if task_id not in self._tasks:
                self._order.append(task_id)
            self._tasks[task_id] = item
        return item

    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        notify: bool = True,
    ) -> TaskItem | None:
        """Update task status. Returns updated item or None if not found.

        Notifies subscribers after status change unless notify=False.
        """
        with self._lock:
            if task_id not in self._tasks:
                return None
            old = self._tasks[task_id]
            if old.status == status:
                return old
            updated = replace(old, status=status)
            self._tasks[task_id] = updated
            subscribers = self._subscribers.copy() if notify else []

        # Notify outside lock to avoid deadlock
        for cb in subscribers:
            try:
                cb(task_id, status)
            except Exception:
                pass

        return updated

    def update_session_id(self, task_id: str, session_id: str) -> TaskItem | None:
        """Update the session_id binding for a task."""
        with self._lock:
            if task_id not in self._tasks:
                return None
            old = self._tasks[task_id]
            updated = replace(old, session_id=session_id)
            self._tasks[task_id] = updated
        return updated

    def update_name(self, task_id: str, name: str) -> TaskItem | None:
        """Update the display name for a task."""
        name = (name or "").strip()
        if not name:
            return None
        with self._lock:
            if task_id not in self._tasks:
                return None
            old = self._tasks[task_id]
            if old.name == name:
                return old
            updated = replace(old, name=name)
            self._tasks[task_id] = updated
        return updated

    def get(self, task_id: str) -> TaskItem | None:
        """Get a single task by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def get_snapshot(self) -> list[TaskSnapshot]:
        """Get an ordered snapshot of all tasks (immutable, safe to share)."""
        with self._lock:
            return [
                TaskSnapshot(
                    task_id=self._tasks[tid].task_id,
                    name=self._tasks[tid].name,
                    status=self._tasks[tid].status,
                )
                for tid in self._order
                if tid in self._tasks
            ]

    def subscribe(self, callback: StatusCallback) -> None:
        """Subscribe to status change notifications."""
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: StatusCallback) -> None:
        """Unsubscribe from status change notifications."""
        with self._lock:
            try:
                self._subscribers.remove(callback)
            except ValueError:
                pass

    @property
    def count(self) -> int:
        """Number of registered tasks."""
        with self._lock:
            return len(self._tasks)

    def clear(self) -> None:
        """Clear all tasks and subscribers."""
        with self._lock:
            self._tasks.clear()
            self._order.clear()
            self._subscribers.clear()


# ---------------------------------------------------------------------------
# Factory helpers: convert engine-specific plan structures to task dicts
# ---------------------------------------------------------------------------


def tasks_from_plan_entries(entries: list[PlanEntryInfo]) -> list[dict]:
    """Convert ACP PlanEntryInfo list to task dicts for TaskOrchestrator.

    Each entry becomes a task with task_id = "step_{index}" and name = entry.content.
    Only entries with non-empty content are included.
    """
    tasks: list[dict] = []
    for idx, entry in enumerate(entries):
        content = (entry.content or "").strip()
        if not content:
            continue
        # Map PlanEntryInfo.status to TaskStatus
        status = entry.status if entry.status in ("pending", "in_progress", "completed", "failed") else "pending"
        name = summarize_tool_call_content(content, max_chars=120) or content[:120]
        tasks.append({
            "task_id": f"step_{idx}",
            "name": name,
            "status": status,
        })
    return tasks


def tasks_from_spec_tasks(spec_tasks: list[SpecTask]) -> list[dict]:
    """Convert SpecTask list to task dicts for TaskOrchestrator.

    Uses SpecTask.task_id (int) and SpecTask.description as task name.
    """
    tasks: list[dict] = []
    for t in spec_tasks:
        desc = (t.description or "").strip()
        if not desc:
            continue
        # Map SpecTaskStatus to TaskStatus string
        status_map = {"pending": "pending", "in_progress": "in_progress", "completed": "completed", "failed": "failed"}
        status = status_map.get(t.status.value, "pending")
        name = summarize_tool_call_content(desc, max_chars=120) or desc[:120]
        tasks.append({
            "task_id": f"spec_task_{t.task_id}",
            "name": name,
            "status": status,
        })
    return tasks
