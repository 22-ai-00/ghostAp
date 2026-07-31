"""Project ACP SubAgent lifecycle into an existing Worktree unit summary."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from ..card.tool_display import sanitize_subagent_display_text
from ..utils.callbacks import safe_invoke

if TYPE_CHECKING:
    from ..acp.models import ACPEvent
    from .models import WorktreeUnit


_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
_STATE_STATUS = {
    "pendinginit": "running",
    "pending": "running",
    "running": "running",
    "completed": "completed",
    "shutdown": "completed",
    "errored": "failed",
    "failed": "failed",
    "notfound": "failed",
    "interrupted": "cancelled",
}
_STATUS_LABEL = {
    "running": "执行中",
    "completed": "已完成",
    "failed": "失败",
    "cancelled": "已中断",
}
_STATE_PROGRESS = {
    "pendinginit": "正在初始化",
    "pending": "等待执行",
    "running": "执行中",
    "completed": "已完成",
    "shutdown": "已结束",
    "errored": "执行失败",
    "failed": "执行失败",
    "notfound": "状态不可用",
    "interrupted": "已中断",
}
@dataclass
class _ChildProgress:
    order: int
    status: str = "running"
    progress: str = ""
    activity: str = ""


def _safe_progress_text(value: object, *, opaque_ids: set[str]) -> str:
    """Return one short, card-safe progress line without provider identifiers."""
    try:
        return sanitize_subagent_display_text(
            value,
            fallback="",
            max_chars=180,
            opaque_ids=opaque_ids,
        )
    except Exception:
        return ""


class WorktreeSubagentProgress:
    """Fold provider events by child thread and refresh the existing unit card.

    Child thread ids are used only as ephemeral dictionary keys. They are never
    persisted into ``WorktreeUnit.metadata`` or included in the rendered summary.
    """

    def __init__(
        self,
        unit: "WorktreeUnit",
        *,
        on_unit_update: Callable[["WorktreeUnit"], None] | None,
        label: str = "子Agent",
        base_summary: str = "",
    ) -> None:
        self._unit = unit
        self._on_unit_update = on_unit_update
        self._label = str(label or "子Agent").strip() or "子Agent"
        self._base_summary = str(base_summary or "").strip()
        self._children: dict[str, _ChildProgress] = {}
        self._opaque_ids: set[str] = set()
        self._last_rendered = ""
        self._closed = False
        self._lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock

    def on_event(self, event: "ACPEvent") -> bool:
        tool_call = getattr(event, "tool_call", None)
        if tool_call is None:
            return False

        tool_call_id = str(getattr(tool_call, "id", "") or "").strip()
        with self._lock:
            if self._closed:
                return False
            if tool_call_id:
                self._opaque_ids.add(tool_call_id)

        if getattr(tool_call, "collaboration_tool", None):
            return self._handle_collaboration(tool_call)

        source_id = str(
            getattr(tool_call, "subagent_source_id", None)
            or getattr(event, "source_id", None)
            or ""
        ).strip()
        activity = str(
            getattr(tool_call, "subagent_activity", None) or ""
        ).strip().lower()
        if not source_id or not activity:
            return False
        with self._lock:
            if self._closed:
                return False
            self._opaque_ids.add(source_id)
        return self._handle_activity(
            source_id,
            activity=activity,
            provider_status=str(
                getattr(tool_call, "status", "") or ""
            ).strip().lower(),
        )

    def restore_base_summary(self) -> None:
        """Remove transient review progress without clobbering newer summaries."""
        changed = False
        with self._lock:
            self._closed = True
            if self._last_rendered and self._unit.summary == self._last_rendered:
                self._unit.summary = self._base_summary
                self._last_rendered = ""
                changed = True
        if changed:
            safe_invoke(
                self._on_unit_update,
                self._unit,
                label="on_unit_update",
            )

    def close(self) -> None:
        """Ignore any provider events arriving after the parent prompt ends."""
        with self._lock:
            self._closed = True

    def _handle_collaboration(self, tool_call: object) -> bool:
        states_by_source = {
            str(item.get("source_id") or "").strip(): item
            for item in getattr(tool_call, "subagent_states", ())
            if isinstance(item, dict)
            and str(item.get("source_id") or "").strip()
        }
        source_ids = [
            str(source_id or "").strip()
            for source_id in getattr(tool_call, "collaboration_receivers", ())
            if str(source_id or "").strip()
        ]
        source_ids.extend(
            source_id
            for source_id in states_by_source
            if source_id not in source_ids
        )
        if not source_ids:
            return False

        changed = False
        with self._lock:
            if self._closed:
                return False
            self._opaque_ids.update(source_ids)
            for source_id in source_ids:
                state = states_by_source.get(source_id, {})
                raw_status = str(state.get("status") or "running").strip().lower()
                incoming_status = _STATE_STATUS.get(raw_status, "running")
                child = self._children.get(source_id)
                if child is None:
                    child = _ChildProgress(order=len(self._children) + 1)
                    self._children[source_id] = child
                elif (
                    child.status in _TERMINAL_STATUSES
                    and incoming_status != child.status
                ):
                    continue

                progress = _safe_progress_text(
                    state.get("message"),
                    opaque_ids=self._opaque_ids,
                ) or _STATE_PROGRESS.get(raw_status, "执行中")
                if child.status == incoming_status and child.progress == progress:
                    continue
                child.status = incoming_status
                child.progress = progress
                if incoming_status in _TERMINAL_STATUSES:
                    child.activity = ""
                changed = True
        if changed:
            self._publish()
        return True

    def _handle_activity(
        self,
        source_id: str,
        *,
        activity: str,
        provider_status: str,
    ) -> bool:
        activity_complete = provider_status == "completed"
        activity_failed = provider_status == "failed"
        if activity == "started":
            incoming_status = "running"
            progress = (
                "已启动"
                if activity_complete
                else ("启动未完成" if activity_failed else "正在启动")
            )
        elif activity == "interacted":
            incoming_status = "running"
            progress = (
                "已与主 Agent 交互"
                if activity_complete
                else (
                    "交互未完成"
                    if activity_failed
                    else "正在与主 Agent 交互"
                )
            )
        elif activity == "interrupted":
            incoming_status = "cancelled" if activity_complete else "running"
            progress = (
                "已中断"
                if activity_complete
                else ("中断未完成" if activity_failed else "正在中断")
            )
        else:
            incoming_status = "running"
            progress = "动态已更新"

        changed = False
        with self._lock:
            if self._closed:
                return False
            child = self._children.get(source_id)
            if child is None:
                child = _ChildProgress(order=len(self._children) + 1)
                self._children[source_id] = child
            elif child.status in _TERMINAL_STATUSES:
                return True
            if child.status != incoming_status or child.activity != progress:
                child.status = incoming_status
                child.activity = progress
                changed = True
        if changed:
            self._publish()
        return True

    def _publish(self) -> None:
        with self._lock:
            if self._closed:
                return
            ordered = sorted(self._children.values(), key=lambda child: child.order)
            visible = ordered[:4]
            parts = []
            for child in visible:
                status_label = _STATUS_LABEL.get(child.status, "执行中")
                details = [child.activity, child.progress]
                details = list(dict.fromkeys(detail for detail in details if detail))
                detail = f"（{' · '.join(details)}）" if details else ""
                parts.append(
                    f"{self._label} #{child.order} {status_label}{detail}"
                )
            if len(ordered) > len(visible):
                parts.append(f"另有 {len(ordered) - len(visible)} 个")
            progress_summary = "；".join(parts)
            rendered = (
                f"{self._base_summary} · {progress_summary}"
                if self._base_summary
                else progress_summary
            )
            if not rendered or rendered == self._last_rendered:
                return
            self._unit.summary = rendered
            self._last_rendered = rendered
        safe_invoke(
            self._on_unit_update,
            self._unit,
            label="on_unit_update",
        )
