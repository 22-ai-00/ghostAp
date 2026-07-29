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
    assert "测试补齐 · #5.a · Aiden · claude-haiku-4-5" in panel["elements"][0]["content"]
    assert "UI 回归 · #5.b · Codex · gpt-5" in panel["elements"][0]["content"]


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


def test_build_subagent_dispatch_atom_renders_through_registry():
    atom = build_subagent_dispatch_atom(_subagents())

    assert atom is not None
    assert atom.node_count == 4
    elements = _render_atoms_to_elements([atom], CardState(), RenderBudget(), {})

    assert len(elements) == 1
    assert elements[0]["border"]["color"] == "orange"
    assert "并行子任务" in elements[0]["header"]["title"]["content"]
