from src.card.render.budget import RenderBudget
from src.card.render.renderer import _render_atoms_to_elements
from src.card.render.tools import build_subagent_dispatch_atom, render_subagent_dispatch_panel
from src.card.state.models import CardState


def _subagents():
    return [
        {
            "label": "测试补齐",
            "sequence": "5.a",
            "tool": "Aiden",
            "model": "claude-haiku-4-5",
            "status": "running",
        },
        {
            "label": "UI 回归",
            "sequence": "5.b",
            "tool": "Codex",
            "model": "gpt-5",
            "status": "completed",
        },
    ]


def test_render_subagent_dispatch_panel_uses_orange_parallel_summary():
    panel = render_subagent_dispatch_panel(_subagents())

    assert panel is not None
    assert panel["expanded"] is True
    assert panel["border"]["color"] == "orange"
    assert "并行子任务" in panel["header"]["title"]["content"]
    assert "**执行中** · 测试补齐 · #5.a · Aiden · claude-haiku-4-5" in panel["elements"][0]["content"]
    assert "**已完成** · UI 回归 · #5.b · Codex · gpt-5" in panel["elements"][0]["content"]


def test_render_subagent_dispatch_panel_does_not_present_unknown_status_as_success():
    panel = render_subagent_dispatch_panel(
        [
            {
                "label": "状态损坏的子任务",
                "tool": "Codex",
                "status": "unexpected-terminal",
            }
        ]
    )

    assert panel is not None
    assert panel["expanded"] is True
    assert panel["border"]["color"] == "orange"
    assert panel["header"]["title"]["content"].startswith("🟠")
    assert "未知 1" in panel["header"]["title"]["content"]
    assert "暂无状态" not in panel["header"]["title"]["content"]


def test_render_subagent_dispatch_panel_does_not_present_cancel_only_as_success():
    panel = render_subagent_dispatch_panel(
        [
            {
                "label": "用户取消的子任务",
                "tool": "Codex",
                "status": "cancelled",
            }
        ]
    )

    assert panel is not None
    title = panel["header"]["title"]["content"]
    assert title.startswith("⚪")
    assert "取消 1" in title
    assert "✅" not in title
    assert "**已取消** · 用户取消的子任务" in panel["elements"][0]["content"]


def test_render_subagent_dispatch_panel_explains_missing_result_and_humanizes_slug():
    panel = render_subagent_dispatch_panel(
        [
            {
                "label": "review__child__reconciliation__fix",
                "tool": "Codex",
                "status": "cancelled",
                "progress": "未收到最终结果，已随主任务停止",
            },
            {
                "label": "子任务",
                "tool": "Codex",
                "status": "completed",
            },
        ]
    )

    assert panel is not None
    content = panel["elements"][0]["content"]
    assert "**已取消** · review child reconciliation fix" in content
    assert "结果：未收到最终结果，已随主任务停止" in content
    assert "**已完成** · 任务说明暂缺" in content
    assert "__" not in content


def test_build_subagent_dispatch_atom_renders_through_registry():
    atom = build_subagent_dispatch_atom(_subagents())

    assert atom is not None
    assert atom.node_count == 4
    elements = _render_atoms_to_elements([atom], CardState(), RenderBudget(), {})

    assert len(elements) == 1
    assert elements[0]["border"]["color"] == "orange"
    assert "并行子任务" in elements[0]["header"]["title"]["content"]
