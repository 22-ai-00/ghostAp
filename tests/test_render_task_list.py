"""Tests for src.card.render.task_list module."""
from __future__ import annotations

from typing import get_args, get_type_hints

from src.card.events.payloads import TaskSnapshotPayload
from src.card.render.task_list import render_task_list_panel
from src.card.state.models import TaskListBlock


def _make_block(tasks, current_task_id="t1"):
    return TaskListBlock(
        tasks=tuple(tasks),
        current_task_id=current_task_id,
    )


class TestRenderTaskListPanel:
    def test_basic_rendering(self):
        """Renders tasks with correct status icons."""
        tasks = [
            {"task_id": "t1", "name": "分析需求", "status": "in_progress"},
            {"task_id": "t2", "name": "编写代码", "status": "pending"},
        ]
        block = _make_block(tasks, "t1")
        result = render_task_list_panel(block)

        assert result["tag"] == "collapsible_panel"
        assert result["expanded"] is True
        content = result["elements"][0]["content"]
        assert "🔄" in content
        assert "⏳" in content
        assert "分析需求" in content
        assert "编写代码" in content

    def test_current_task_bold_without_fake_fold_arrow(self):
        """Current task is bolded without fake fold affordances."""
        tasks = [
            {"task_id": "t1", "name": "任务一", "status": "in_progress"},
            {"task_id": "t2", "name": "任务二", "status": "pending"},
        ]
        block = _make_block(tasks, "t1")
        result = render_task_list_panel(block)
        content = result["elements"][0]["content"]

        assert "▶" not in content
        assert "**任务一**" in content
        # Non-current task should NOT be bold
        assert "**任务二**" not in content

    def test_bucket_headers_do_not_look_like_clickable_collapses(self):
        """Inner task buckets are markdown, so they must not show fold arrows."""
        tasks = [
            {"task_id": "t1", "name": "任务一", "status": "in_progress"},
            {"task_id": "t2", "name": "任务二", "status": "in_progress"},
        ]
        block = _make_block(tasks, "t1")

        result = render_task_list_panel(block)
        content = result["elements"][0]["content"]

        assert "▶ **进行中" not in content
        assert "▶ 🔄" not in content

    def test_all_status_icons(self):
        """Core status types (pending/in_progress/completed/failed) render correct icons."""
        tasks = [
            {"task_id": "t1", "name": "A", "status": "pending"},
            {"task_id": "t2", "name": "B", "status": "in_progress"},
            {"task_id": "t3", "name": "C", "status": "completed"},
            {"task_id": "t4", "name": "D", "status": "failed"},
        ]
        block = _make_block(tasks, "t2")
        result = render_task_list_panel(block)
        content = result["elements"][0]["content"]

        assert "⏳" in content  # group header "⏳ 未处理"
        assert "🔄" in content  # in_progress
        assert "✅" in content  # completed
        assert "❌" in content  # failed

    def test_pending_line_no_hourglass(self):
        """Pending lines use ○ without ⏳ emoji (v2 design)."""
        tasks = [
            {"task_id": "t1", "name": "待办任务", "status": "pending"},
        ]
        block = _make_block(tasks, "")
        result = render_task_list_panel(block)
        content = result["elements"][0]["content"]
        lines = content.split("\n")
        pending_lines = [l for l in lines if "待办任务" in l]
        assert len(pending_lines) == 1
        assert "○" in pending_lines[0]
        assert "⏳" not in pending_lines[0]

    def test_failed_task_uses_ended_bucket_not_completed_label(self):
        tasks = [
            {"task_id": "t1", "name": "失败任务", "status": "failed"},
            {"task_id": "t2", "name": "成功任务", "status": "completed"},
        ]
        block = _make_block(tasks, "t1")
        result = render_task_list_panel(block)
        content = result["elements"][0]["content"]

        assert "已结束 (2)" in content
        assert "完成 1 / 失败 1" in content
        assert "已完成 (2)" not in content

    def test_cancelled_task_is_counted_separately_from_failed(self):
        tasks = [
            {"task_id": "t1", "name": "成功任务", "status": "completed"},
            {"task_id": "t2", "name": "取消任务", "status": "cancelled"},
        ]
        block = _make_block(tasks, "t2")

        result = render_task_list_panel(block)
        content = result["elements"][0]["content"]

        assert "已结束 (2)" in content
        assert "完成 1 / 取消 1" in content
        assert "失败" not in content
        assert "⊘ ~~取消任务~~" in content

    def test_failed_and_cancelled_task_counts_are_independent(self):
        tasks = [
            {"task_id": "t1", "name": "成功任务", "status": "completed"},
            {"task_id": "t2", "name": "失败任务", "status": "failed"},
            {"task_id": "t3", "name": "取消任务", "status": "cancelled"},
        ]
        block = _make_block(tasks, "t3")

        result = render_task_list_panel(block)
        content = result["elements"][0]["content"]

        assert "完成 1 / 失败 1 / 取消 1" in content

    def test_empty_tasks(self):
        """Empty tasks list returns None (no panel rendered)."""
        block = _make_block([], "")
        result = render_task_list_panel(block)
        assert result is None

    def test_compact_mode_is_collapsed_neutral_summary(self):
        """Compact child progress stays neutral while the final task is running."""
        tasks = [
            {"task_id": "t1", "name": "探索代码", "status": "completed"},
            {"task_id": "t2", "name": "修复路由", "status": "completed"},
            {"task_id": "t3", "name": "单元测试", "status": "completed"},
            {"task_id": "t4", "name": "集成测试", "status": "completed"},
            {"task_id": "t5", "name": "最终代码审查", "status": "in_progress"},
        ]
        block = _make_block(tasks, "t5")

        result = render_task_list_panel(block, compact=True)

        assert result["tag"] == "collapsible_panel"
        assert result["expanded"] is False
        assert result["border"]["color"] == "grey"
        header_content = result["header"]["title"]["content"]
        assert header_content == (
            "**整体 4/5 · 当前 5/5 最终代码审查**"
        )
        assert "✅" not in header_content

    def test_compact_mode_keeps_in_progress_when_current_id_missing(self):
        """Compact mode still shows in-progress task if current_task_id is stale."""
        tasks = [
            {"task_id": "t1", "name": "已完成", "status": "completed"},
            {"task_id": "t2", "name": "正在执行", "status": "in_progress"},
            {"task_id": "t3", "name": "后续任务", "status": "pending"},
        ]
        block = _make_block(tasks, "missing")

        result = render_task_list_panel(block, compact=True)

        content = result["elements"][0]["content"]
        assert "正在执行" in content
        assert "**正在执行**" in content
        assert "后续任务" in content
        assert "未处理 (1)" in content

    def test_compact_mode_keeps_cancelled_separate_from_failed(self):
        tasks = [
            {"task_id": "t1", "name": "成功任务", "status": "completed"},
            {"task_id": "t2", "name": "取消任务", "status": "cancelled"},
        ]
        block = _make_block(tasks, "t2")

        result = render_task_list_panel(block, compact=True)
        content = result["elements"][0]["content"]

        assert "已结束 (2) · 完成 1 / 取消 1 · 最近 2 项" in content
        assert "失败" not in content

    def test_compact_empty_tasks(self):
        """Compact mode also returns None for empty task lists."""
        block = _make_block([], "")

        assert render_task_list_panel(block, compact=True) is None

    def test_fold_completed_when_over_threshold(self):
        """When > 5 tasks, completed tasks shown in gray compact format."""
        tasks = [
            {"task_id": f"t{i}", "name": f"Task {i}", "status": "completed"}
            for i in range(4)
        ] + [
            {"task_id": "t4", "name": "Running", "status": "in_progress"},
            {"task_id": "t5", "name": "Pending1", "status": "pending"},
        ]
        block = _make_block(tasks, "t4")
        result = render_task_list_panel(block)
        content = result["elements"][0]["content"]

        # Completed tasks shown with strikethrough in compact format
        assert "已完成 (4)" in content
        # Non-completed still shown individually
        assert "Running" in content
        assert "Pending1" in content

    def test_no_fold_under_threshold(self):
        """When <= 5 tasks, all shown individually."""
        tasks = [
            {"task_id": f"t{i}", "name": f"Task {i}", "status": "completed"}
            for i in range(5)
        ]
        block = _make_block(tasks, "t0")
        result = render_task_list_panel(block)
        content = result["elements"][0]["content"]

        # All shown individually, no folded tail summary
        assert "已完成 (5)" in content
        assert "还有" not in content
        for i in range(5):
            assert f"Task {i}" in content

    def test_panel_header(self):
        """Panel header shows task list title with progress summary."""
        block = _make_block([{"task_id": "t1", "name": "X", "status": "pending"}])
        result = render_task_list_panel(block)
        header_content = result["header"]["title"]["content"]
        assert "任务列表" in header_content
        assert "0/1 ✅" in header_content


class TestTaskListAtomOrdering:
    """Verify task_list atoms appear first in rendered card."""

    def test_task_list_first_in_status_section(self):
        from src.card.render.atoms import flatten_to_atoms
        from src.card.render.budget import RenderBudget
        from src.card.render.renderer import _order_atoms_by_section
        from src.card.state.models import ContentBlock, TaskListBlock

        blocks = (
            ContentBlock(kind="text", block_id="txt1", content="Hello"),
            TaskListBlock(tasks=({"task_id": "t1", "name": "A", "status": "pending"},), current_task_id="t1"),
            ContentBlock(kind="phase", block_id="ph1", content="Phase 1"),
        )
        atoms = flatten_to_atoms(blocks, RenderBudget())
        ordered = _order_atoms_by_section(atoms)

        assert ordered[0].kind == "task_list"


def test_task_snapshot_payload_accepts_cancelled_status():
    status_values = get_args(get_type_hints(TaskSnapshotPayload)["status"])

    assert "cancelled" in status_values
