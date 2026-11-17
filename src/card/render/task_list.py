"""Task list panel rendering for task-level card management."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.card.state.models import TaskListBlock
from src.card.themes import PANEL_STYLES

if TYPE_CHECKING:
    from src.card.events.payloads import TaskSnapshotPayload

_STATUS_ICONS = {
    "pending": "⏳",
    "in_progress": "🔄",
    "active": "🔄",
    "running": "🔄",
    "completed": "✅",
    "failed": "❌",
    "cancelled": "⊘",
}

_ACTIVE_STATUSES = frozenset({"in_progress", "active", "running"})
_COMPLETED_STATUSES = frozenset({"completed", "done"})
_FAILED_STATUSES = frozenset({"failed"})
_CANCELLED_STATUSES = frozenset({"cancelled"})
_ENDED_STATUSES = _COMPLETED_STATUSES | _FAILED_STATUSES | _CANCELLED_STATUSES

_TASK_BUCKET_VISIBLE_LIMIT = 50


def group_tasks(plan_or_tasks) -> tuple[list, list, list]:
    """Group tasks into v2 buckets: in-progress, ended, pending."""
    tasks = getattr(plan_or_tasks, "entries", plan_or_tasks)
    in_progress: list = []
    completed: list = []
    pending: list = []

    for task in tasks or ():
        status = _task_status(task)
        if status in _ACTIVE_STATUSES:
            in_progress.append(task)
        elif status in _ENDED_STATUSES:
            completed.append(task)
        else:
            pending.append(task)

    return in_progress, completed, pending


def render_task_list_panel(block: TaskListBlock) -> dict | None:
    """Render the task list panel with progress summary.

    Returns None if tasks is empty (no panel rendered).
    Current task is highlighted with bold + arrow prefix.
    """
    tasks = block.tasks
    if not tasks:
        return None

    current_id = block.current_task_id
    total = len(tasks)
    completed_count = sum(1 for t in tasks if str(t.get("status")) in _COMPLETED_STATUSES)

    lines = _build_v2_task_lines(tasks, current_id)
    content = "\n".join(lines)

    # Header with progress summary
    header_title = f"📋 **任务列表** — 进度：{completed_count}/{total} ✅"

    return {
        "tag": "collapsible_panel",
        "expanded": True,
        "header": {
            "title": {"tag": "markdown", "content": header_title},
            "vertical_align": "center",
            "icon": {
                "tag": "standard_icon",
                "token": "down-small-ccm_outlined",
                "size": "16px 16px",
            },
            "icon_position": "follow_text",
            "icon_expanded_angle": -180,
        },
        "border": {"color": PANEL_STYLES["border_task_list"], "corner_radius": PANEL_STYLES["corner_radius"]},
        "vertical_spacing": PANEL_STYLES["vertical_spacing"],
        "padding": PANEL_STYLES["padding_standard"],
        "elements": [{"tag": "markdown", "content": content}],
    }


def _build_v2_task_lines(tasks: tuple[TaskSnapshotPayload, ...], current_id: str) -> list[str]:
    """Build v2 three-bucket task markdown for sticky and full panels."""
    if not tasks:
        return []

    total = len(tasks)
    in_progress, completed, pending = group_tasks(tasks)
    lines: list[str] = []

    lines.append(f"**进行中 ({len(in_progress)})**")
    active_visible = in_progress[:_TASK_BUCKET_VISIBLE_LIMIT]
    for task in active_visible:
        lines.append(_format_task_line(task, current_id, tasks, total))
    if len(active_visible) < len(in_progress):
        lines.append(f"　…还有 {len(in_progress) - len(active_visible)} 个进行中")

    completed_only, failed_count, cancelled_count = _count_ended_statuses(completed)
    if failed_count or cancelled_count:
        ended_icon = "❌" if failed_count else "⊘"
        status_summary = _format_ended_status_summary(
            completed_count=completed_only,
            failed_count=failed_count,
            cancelled_count=cancelled_count,
        )
        ended_label = (
            f"{ended_icon} **已结束 ({len(completed)})** · {status_summary}"
        )
    else:
        ended_label = f"✅ **已完成 ({len(completed)})**"
    lines.append(ended_label)
    done_visible = completed[:_TASK_BUCKET_VISIBLE_LIMIT]
    for task in done_visible:
        lines.append(_format_done_line(task))
    if len(done_visible) < len(completed):
        folded_label = "已结束" if failed_count or cancelled_count else "已完成"
        lines.append(f"　…还有 {len(completed) - len(done_visible)} 个{folded_label}")

    lines.append(f"⏳ **未处理 ({len(pending)})**")
    pending_visible = pending[:_TASK_BUCKET_VISIBLE_LIMIT]
    for task in pending_visible:
        lines.append(_format_pending_line(task))
    if len(pending_visible) < len(pending):
        lines.append(f"　…还有 {len(pending) - len(pending_visible)} 个")
    return lines


def _count_ended_statuses(tasks) -> tuple[int, int, int]:
    """Return completed, failed, and cancelled counts without conflating them."""
    completed_count = sum(
        1 for task in tasks if _task_status(task) in _COMPLETED_STATUSES
    )
    failed_count = sum(
        1 for task in tasks if _task_status(task) in _FAILED_STATUSES
    )
    cancelled_count = sum(
        1 for task in tasks if _task_status(task) in _CANCELLED_STATUSES
    )
    return completed_count, failed_count, cancelled_count


def _format_ended_status_summary(
    *,
    completed_count: int,
    failed_count: int,
    cancelled_count: int,
) -> str:
    parts: list[str] = []
    if completed_count:
        parts.append(f"完成 {completed_count}")
    if failed_count:
        parts.append(f"失败 {failed_count}")
    if cancelled_count:
        parts.append(f"取消 {cancelled_count}")
    return " / ".join(parts)


def _format_task_line(
    task: TaskSnapshotPayload,
    current_id: str,
    tasks: tuple[TaskSnapshotPayload, ...],
    total: int,
) -> str:
    """Format a single task line with step number (step/total).

    Current in_progress task is highlighted with bold + arrow prefix.
    """
    task_id = _task_id(task)
    name = _task_name(task)
    status = _task_status(task)
    icon = _STATUS_ICONS.get(status, "⏳")
    step = next((idx + 1 for idx, item in enumerate(tasks) if _task_id(item) == task_id), 1)
    step_num = f"{step}/{total}"

    if task_id == current_id:
        return f"　{icon} {step_num} **{name}**"
    return f"　{icon} {step_num} {name}"


def _format_done_line(task) -> str:
    status = _task_status(task)
    icon = _STATUS_ICONS.get(status, "✅")
    return f"　{icon} ~~{_task_name(task)}~~"


def _format_pending_line(task) -> str:
    return f"　○ {_task_name(task)}"


def _task_id(task) -> str:
    if isinstance(task, dict):
        return str(task.get("task_id", ""))
    return str(getattr(task, "task_id", ""))


def _task_name(task) -> str:
    if isinstance(task, dict):
        return str(task.get("name") or task.get("title") or "未命名任务")
    return str(getattr(task, "name", None) or getattr(task, "title", None) or "未命名任务")


def _task_status(task) -> str:
    if isinstance(task, dict):
        return str(task.get("status", "pending"))
    return str(getattr(task, "status", "pending"))
