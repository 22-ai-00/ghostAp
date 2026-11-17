from src.card.render.task_list import group_tasks, render_task_list_panel
from src.card.state.models import TaskListBlock


class _Entry:
    def __init__(self, title, status):
        self.title = title
        self.status = status


class _Plan:
    def __init__(self, entries):
        self.entries = entries


def _block(rows, current_task_id="t2"):
    return TaskListBlock(
        current_task_id=current_task_id,
        tasks=tuple({"task_id": tid, "name": name, "status": status} for tid, name, status in rows),
    )


def test_group_tasks_three_buckets_from_plan_entries():
    plan = _Plan([
        _Entry("a", "completed"),
        _Entry("b", "in_progress"),
        _Entry("c", "pending"),
        _Entry("d", "failed"),
    ])

    in_progress, completed, pending = group_tasks(plan)

    assert [task.title for task in in_progress] == ["b"]
    assert [task.title for task in completed] == ["a", "d"]
    assert [task.title for task in pending] == ["c"]


def test_three_groups_always_render():
    atom = render_task_list_panel(_block([
        ("t1", "探索代码", "completed"),
        ("t2", "修复路由", "in_progress"),
        ("t3", "补单测", "pending"),
    ]))
    md = atom["elements"][0]["content"]

    assert "进行中 (1)" in md
    assert "已完成 (1)" in md
    assert "未处理 (1)" in md
    assert "探索代码" in md
    assert "修复路由" in md
    assert "补单测" in md


def test_task_list_keeps_moderate_task_sets_visible():
    rows = [(f"d{i}", f"done{i}", "completed") for i in range(15)]
    rows.append(("now", "now", "in_progress"))
    rows.extend((f"p{i}", f"p{i}", "pending") for i in range(8))

    atom = render_task_list_panel(_block(rows, current_task_id="now"))
    md = atom["elements"][0]["content"]

    assert "已完成 (15)" in md
    assert md.count("~~done") == 15
    assert "p7" in md
    assert "还有" not in md


def test_task_list_folds_only_very_large_buckets():
    rows = [(f"d{i}", f"done{i}", "completed") for i in range(55)]

    atom = render_task_list_panel(_block(rows, current_task_id="missing"))
    md = atom["elements"][0]["content"]

    assert "已完成 (55)" in md
    assert md.count("~~done") == 50
    assert "还有 5 个已完成" in md
