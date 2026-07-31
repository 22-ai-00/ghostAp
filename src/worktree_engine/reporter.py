from __future__ import annotations

import logging

from ..utils.constants import STATUS_DISPLAY_MAP
from ..utils.ui_text import SPEC_UI_TEXT
from .models import (
    WorktreeInfo,
    WorktreeJourneyStatus,
    WorktreeRuntimeState,
    WorktreeUnit,
    WorktreeUnitStatus,
)

logger = logging.getLogger(__name__)

# Structured reason code → user-facing display text mapping.
# _cancel_unit stores reason codes (e.g. "pool_timeout") in unit.error;
# the display layer maps them to human-friendly text here.
REASON_DISPLAY_MAP: dict[str, str] = {
    "pool_timeout": SPEC_UI_TEXT["timeout_busy_worktree"],
}


class WorktreeReporter:
    def refresh_state(self, state: WorktreeRuntimeState) -> WorktreeRuntimeState:
        state.summary_lines = self.build_unit_summary_lines(state.units)
        state.merge_notes = self.build_merge_notes(state.units, state.base_branch)
        all_required_completed = bool(state.units) and all(
            unit.status == WorktreeUnitStatus.COMPLETED for unit in state.units
        )
        review_passed = bool((state.review_outcome or {}).get("passed"))
        state.merge_entry_ready = (
            all_required_completed
            and review_passed
            and state.journey.status == WorktreeJourneyStatus.COMPLETED
        )
        if state.last_error:
            state.summary_lines.insert(0, f"- 总体错误：{state.last_error}")
        return state

    @staticmethod
    def _get_unit_display_name(unit: WorktreeUnit) -> str:
        if unit.display_name:
            return unit.display_name

        if unit.unit_id:
            # 处理 wt-01 这种标准格式
            if unit.unit_id.startswith("wt-"):
                suffix = unit.unit_id[3:]
                try:
                    idx = int(suffix)
                    if 1 <= idx <= 26:
                        letter = chr(ord("A") + idx - 1)
                        return f"工作空间 {letter}"
                except ValueError:
                    logger.debug("failed to convert index to letter", exc_info=True)

            # 兼容其他带后缀的格式
            suffix = unit.unit_id.split("-")[-1] if "-" in unit.unit_id else unit.unit_id
            if suffix.isdigit():
                return f"单元 {suffix}"

        return "自动分配中"

    def build_unit_summary_lines(self, units: list[WorktreeUnit]) -> list[str]:
        lines: list[str] = []
        status_icon = {"completed": "✅", "failed": "❌", "cancelled": "⏹", "running": "🔄", "planned": "📋", "ready": "⏳"}
        for unit in units:
            status = unit.status or WorktreeUnitStatus.PENDING
            icon = status_icon.get(status, "⏳")
            change_text = "有代码变更" if unit.has_changes else "无代码变更"
            task_text = unit.task_title or "未分配任务"

            display_name = WorktreeReporter._get_unit_display_name(unit)

            if status == WorktreeUnitStatus.CANCELLED:
                reason_text = REASON_DISPLAY_MAP.get(unit.error, unit.error) if unit.error else ""
                summary = f"⏹ 取消原因：{reason_text or '执行已取消，可通过 /wt retry 重试'}"
            elif status == WorktreeUnitStatus.FAILED:
                reason_text = REASON_DISPLAY_MAP.get(unit.error, unit.error) if unit.error else ""
                summary = f"🔍 失败原因：{reason_text or '执行失败，可通过 /wt retry 重试'}"
            else:
                summary = unit.summary or "暂无摘要"
            lines.append(f"- {icon} `{display_name}` · `{STATUS_DISPLAY_MAP.get(status, status)}` · {task_text} · {change_text} · {summary}")
        return lines

    def build_merge_notes(self, units: list[WorktreeUnit], base_branch: str) -> list[dict]:
        notes: list[dict] = []
        target = base_branch or "main"
        for unit in units:
            display_name = WorktreeReporter._get_unit_display_name(unit)
            branch = unit.branch_name or "(未创建)"
            ready = unit.status == WorktreeUnitStatus.COMPLETED
            conflict_disclosure = (
                "冲突时自动优先采用 Worktree 分支变更"
                if ready
                else "单元未完成，禁止合并"
            )
            notes.append({
                "branch": branch,
                "status": "ready" if ready else "blocked",
                "conflict_policy": "worktree_branch_wins",
                "summary": (
                    f"`{display_name}` → 分支 `{branch}` → 建议合并回 `{target}`；"
                    f"{conflict_disclosure}"
                ),
            })
        return notes

    def format_worktree_table(self, entries: list[WorktreeInfo]) -> str:
        """Format worktree list into an aligned table string."""
        if not entries:
            return "(无 worktree)"

        headers = ("路径", "分支", "状态", "最后更新")
        rows: list[tuple[str, str, str, str]] = []
        for entry in entries:
            active_mark = " *" if entry.is_active else ""
            rows.append((
                entry.path,
                entry.branch or "(detached)",
                f"活跃{active_mark}" if entry.is_active else "非活跃",
                entry.last_updated or "-",
            ))

        # Compute column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))

        def _fmt_row(cells: tuple[str, ...]) -> str:
            return "  ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(cells))

        lines = [_fmt_row(headers), "-" * (sum(col_widths) + 2 * (len(headers) - 1))]
        for row in rows:
            lines.append(_fmt_row(row))
        return "\n".join(lines)
