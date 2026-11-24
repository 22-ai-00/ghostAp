"""Tests for pure Card JSON 2.0 Workflow config/help/launch builders."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from src.acp.options import ACPModelOption, ACPModelSelectionVariant
from src.card.actions import dispatch as action_ids
from src.card.builders.workflow import (
    WORKFLOW_PRESET_CATALOG,
    WORKFLOW_ROLE_CATALOG,
    WorkflowConfigEditorState,
    _cap_options_preserving,
    build_workflow_config_card,
    build_workflow_help_card,
    build_workflow_preset_launch_card,
    capability_to_acp_options,
)
from src.card.schema import require_card_json_2
from src.card.shared.truncation import count_markdown_table_blocks, count_tagged_nodes
from src.workflow_engine.agent_pool import WorkflowAgentBinding
from src.workflow_engine.tool_registry import ToolCapability


def _walk_dicts(node: object):
    stack = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            yield cur
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)


def _assert_card_2_valid(card: dict) -> None:
    require_card_json_2(card)
    payload_bytes = len(json.dumps(card, ensure_ascii=False).encode("utf-8"))
    assert payload_bytes <= 27 * 1024
    assert count_tagged_nodes(card) <= 180
    table_count = sum(
        count_markdown_table_blocks(str(d.get("content") or ""))
        for d in _walk_dicts(card)
        if d.get("tag") == "markdown"
    ) + sum(1 for d in _walk_dicts(card) if d.get("tag") == "table")
    assert table_count <= 5

    summary = ((card.get("config") or {}).get("summary") or {}).get("content")
    assert isinstance(summary, str)
    assert 8 <= len(summary) <= 60
    assert (card.get("config") or {}).get("update_multi") is True

    for d in _walk_dicts(card):
        if d.get("tag") == "button":
            assert d.get("behaviors") == [{"type": "callback", "value": d["value"]}]
        if d.get("tag") == "select_static":
            assert "width" not in d
            assert d.get("behaviors") == [{"type": "callback", "value": d["value"]}]


def _sample_capabilities() -> dict[str, ToolCapability]:
    return {
        "codex": ToolCapability(
            tool_name="codex",
            display_name="Codex",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=("gpt-5.4", "gpt-5.3-codex"),
            efforts_by_model={
                "gpt-5.4": ("low", "medium", "high", "xhigh"),
                "gpt-5.3-codex": ("low", "medium", "high"),
            },
            default_effort="high",
        ),
        "claude": ToolCapability(
            tool_name="claude",
            display_name="Claude",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=("sonnet", "opus"),
            efforts_by_model={"sonnet": ("low", "medium", "high", "xhigh", "max")},
            default_effort="high",
        ),
        "traex": ToolCapability(
            tool_name="traex",
            display_name="Traex",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=("c_o_new_thinking", "kimi-k2"),
            profiles_by_model={"c_o_new_thinking": ("standard", "max")},
            efforts_by_model={"c_o_new_thinking": ("low", "medium", "high", "xhigh", "max")},
            default_effort="high",
        ),
        "coco": ToolCapability(
            tool_name="coco",
            display_name="Coco",
            available=True,
            supports_model=True,
            supports_effort=False,
            models=("openai/gpt-4o", "default"),
        ),
    }


def test_config_card_multi_effort_coexistence_and_card2_invariants() -> None:
    members = (
        WorkflowAgentBinding(
            agent_id="A1",
            tool_name="codex",
            model_name="gpt-5.4",
            display_name="Scout",
            effort="low",
            role="scout",
        ),
        WorkflowAgentBinding(
            agent_id="A2",
            tool_name="codex",
            model_name="gpt-5.4",
            display_name="Worker",
            effort="xhigh",
            role="worker",
        ),
        WorkflowAgentBinding(
            agent_id="A3",
            tool_name="claude",
            model_name="sonnet",
            display_name="Verifier",
            effort="high",
            role="verifier",
        ),
    )
    state = WorkflowConfigEditorState(
        config_session_id="cfg_1",
        chat_id="oc_test",
        project_id="proj_1",
        preset_id="collaborative",
        members=members,
        orchestrator_agent_id="A2",
        selected_agent_id="A2",
        editor_role="worker",
        editor_tool="codex",
        editor_model="gpt-5.4",
        editor_effort="xhigh",
        requirement="Refactor auth module",
    )
    card = build_workflow_config_card(state, _sample_capabilities())
    _assert_card_2_valid(card)

    raw = json.dumps(card, ensure_ascii=False)
    assert "A1" in raw and "A2" in raw and "A3" in raw
    assert "low" in raw and "xhigh" in raw and "high" in raw
    assert "★ 主编排" in raw
    assert "[正在编辑]" in raw
    assert action_ids.WF_START_WORKFLOW in raw
    assert action_ids.WF_SAVE_TEAM in raw
    assert len(WORKFLOW_PRESET_CATALOG) >= 4
    assert len(WORKFLOW_ROLE_CATALOG) >= 4


def test_config_card_cold_cache_effort_tool_and_model_only_tool() -> None:
    caps = {
        "codex": ToolCapability(
            tool_name="codex",
            display_name="Codex",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=(),
        ),
        "coco": ToolCapability(
            tool_name="coco",
            display_name="Coco",
            available=True,
            supports_model=True,
            supports_effort=False,
            models=(),
        ),
    }

    # Cold-cache effort-capable tool gets unified Model + Effort select_static
    codex_card = build_workflow_config_card(
        WorkflowConfigEditorState(editor_tool="codex", editor_effort="xhigh"),
        caps,
    )
    _assert_card_2_valid(codex_card)
    codex_selects = [d for d in _walk_dicts(codex_card) if d.get("tag") == "select_static"]
    assert any(s.get("name") == "model_name" for s in codex_selects)
    assert any(s.get("name") == "effort" for s in codex_selects)
    effort_sel = next(s for s in codex_selects if s.get("name") == "effort")
    assert effort_sel.get("initial_option") == "xhigh"

    # Model-only tool renders unified single-option default Effort select_static
    # (no invalid effort value in callbacks) and preserves OpenRouter slash model name.
    opts = capability_to_acp_options(caps["coco"], current_model="openai/gpt-4o")
    assert [o.name for o in opts] == ["openai/gpt-4o"]
    coco_card = build_workflow_config_card(
        WorkflowConfigEditorState(editor_tool="coco", editor_model="openai/gpt-4o"),
        caps,
    )
    _assert_card_2_valid(coco_card)
    coco_selects = {s["name"]: s for s in _walk_dicts(coco_card) if s.get("tag") == "select_static"}
    assert "model_name" in coco_selects
    assert [opt["value"] for opt in coco_selects["model_name"]["options"]] == [
        "default",
        "openai/gpt-4o",
    ]
    assert coco_selects["model_name"].get("initial_option") == "openai/gpt-4o"
    assert "effort" in coco_selects
    assert [opt["value"] for opt in coco_selects["effort"]["options"]] == ["default"]
    assert coco_selects["effort"].get("initial_option") == "default"
    for d in _walk_dicts(coco_card):
        if "value" in d and isinstance(d["value"], dict) and "action" in d["value"]:
            assert "effort" not in d["value"]
    assert action_ids.WF_START_WORKFLOW not in json.dumps(coco_card, ensure_ascii=False)


def test_config_card_traex_profile_and_extreme_8_member_60_model_budget() -> None:
    many_models = tuple(f"model-{i}" for i in range(60))
    caps = {
        "traex": ToolCapability(
            tool_name="traex",
            display_name="Traex",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=many_models,
            profiles_by_model={"model-55": ("standard", "max")},
            efforts_by_model={"model-55": ("low", "medium", "high", "xhigh", "max")},
        )
    }
    members = tuple(
        WorkflowAgentBinding(
            agent_id=f"A{i + 1}",
            tool_name="traex",
            model_name="model-55",
            display_name=f"Member {i + 1}",
            profile="max" if i >= 4 else "standard",
            effort=("low", "medium", "high", "xhigh")[i % 4],
            role=("scout", "planner", "worker", "verifier")[i % 4],
        )
        for i in range(8)
    )
    state = WorkflowConfigEditorState(
        preset_id="custom",
        members=members,
        orchestrator_agent_id="A1",
        selected_agent_id="A8",
        editor_tool="traex",
        editor_model="model-55",
        editor_profile="max",
        editor_effort="xhigh",
        requirement="Audit <at id=all></at> & run checks",
        status_notice="Updated\nnotice <font color='red'>x</font>",
    )
    card = build_workflow_config_card(state, caps)
    _assert_card_2_valid(card)

    selects = {s["name"]: s for s in _walk_dicts(card) if s.get("tag") == "select_static"}
    assert "profile" in selects and "effort" in selects
    assert len(selects["model_name"]["options"]) <= 25
    assert selects["model_name"].get("initial_option") == "model-55"
    raw = json.dumps(card, ensure_ascii=False)
    assert "<at" not in raw
    assert "<font" not in raw


def test_help_and_preset_launch_cards_are_valid_card2() -> None:
    help_card = build_workflow_help_card(
        node_requirement_text="Node.js >= 20.0.0",
        project_id="proj_1",
        chat_id="oc_1",
    )
    _assert_card_2_valid(help_card)
    assert action_ids.WF_OPEN_CONFIG in json.dumps(help_card, ensure_ascii=False)

    launch_card = build_workflow_preset_launch_card(
        requirement="Build feature",
        preset_id="collaborative",
        members=(
            WorkflowAgentBinding(
                agent_id="A1",
                tool_name="codex",
                model_name="gpt-5.4",
                display_name="Codex",
                effort="xhigh",
                role="worker",
            ),
        ),
        orchestrator_agent_id="A1",
        chat_id="oc_1",
        project_id="proj_1",
    )
    _assert_card_2_valid(launch_card)
    assert action_ids.WORKFLOW_STOP_RUNNING in json.dumps(launch_card, ensure_ascii=False)
    stop_buttons = [
        d
        for d in _walk_dicts(launch_card)
        if d.get("tag") == "button"
        and (d.get("value") or {}).get("action") == action_ids.WORKFLOW_STOP_RUNNING
    ]
    assert len(stop_buttons) == 1
    assert stop_buttons[0].get("size") == "medium"
    assert "confirm" not in stop_buttons[0]


def test_workflow_builder_import_boundary_clean() -> None:
    path = Path(__file__).resolve().parents[1] / "src" / "card" / "builders" / "workflow.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                assert "delivery" not in parts and "session" not in parts and "feishu" not in parts
        elif isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            assert "delivery" not in parts and "session" not in parts and "feishu" not in parts


def test_workflow_cards_sanitize_lone_surrogates_for_strict_utf8() -> None:
    caps = _sample_capabilities()
    surrogate_member = WorkflowAgentBinding(
        agent_id="A1",
        tool_name="codex",
        model_name="gpt-5.4\ud800",
        display_name="Scout\udfff",
        profile="std\ud800",
        effort="low\ud800",
        role="worker",
    )
    state = WorkflowConfigEditorState(
        config_session_id="cfg_\ud800_1",
        chat_id="oc_\ud800_chat",
        project_id="proj_\ud800_1",
        thread_root_id="om_\ud800_root",
        preset_id="collaborative_\ud800_z",
        members=(surrogate_member,),
        orchestrator_agent_id="A1\ud800",
        selected_agent_id="A1\ud800",
        editor_role="worker",
        editor_tool="codex",
        editor_model="gpt-5.4\ud800",
        editor_profile="std\ud800",
        editor_effort="low\ud800",
        requirement="Fix\ud800bug",
        status_notice="Notice\udfff",
    )
    cfg_card = build_workflow_config_card(state, caps)
    _assert_card_2_valid(cfg_card)
    json.dumps(cfg_card, ensure_ascii=False).encode("utf-8", errors="strict")

    launch_card = build_workflow_preset_launch_card(
        requirement="Req\ud800",
        preset_id="collaborative_\ud800_z",
        members=(surrogate_member,),
        orchestrator_agent_id="A1\ud800",
        chat_id="oc_\ud800_1",
        project_id="proj_\ud800_1",
    )
    _assert_card_2_valid(launch_card)
    json.dumps(launch_card, ensure_ascii=False).encode("utf-8", errors="strict")

    help_card = build_workflow_help_card(
        node_requirement_text="Node\ud800>=20",
        project_id="proj_\ud800_1",
        chat_id="oc_\ud800_1",
    )
    _assert_card_2_valid(help_card)
    json.dumps(help_card, ensure_ascii=False).encode("utf-8", errors="strict")


def test_cap_options_preserving_retains_multiple_keep_items_beyond_limit() -> None:
    items = [f"m{i}" for i in range(30)]
    capped = _cap_options_preserving(items, keep=("m28", "m29"), limit=25)
    assert len(capped) == 25
    assert "m28" in capped
    assert "m29" in capped

    # Tail-overlap regression: keep item at boundary index 24 + item beyond limit
    capped_overlap = _cap_options_preserving(items, keep=("m24", "m28"), limit=25)
    assert len(capped_overlap) <= 25
    assert len(capped_overlap) == 25
    assert len(set(capped_overlap)) == 25
    assert "m24" in capped_overlap
    assert "m28" in capped_overlap

    cap = ToolCapability(
        tool_name="codex",
        display_name="Codex",
        available=True,
        supports_model=True,
        supports_effort=True,
        models=tuple(f"m-{i}" for i in range(28)) + ("base-m", "base-m/high"),
    )
    opts = capability_to_acp_options(cap, current_model="base-m/high")
    opt_names = [opt.name for opt in opts]
    assert len(opt_names) <= 25
    assert "base-m" in opt_names
    assert "base-m/high" in opt_names


def test_config_card_clears_stale_effort_when_custom_options_have_no_efforts() -> None:
    caps = {
        "codex": ToolCapability(
            tool_name="codex",
            display_name="Codex",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=("plain-model",),
        )
    }
    custom_options = {
        "codex": [
            ACPModelOption(
                name="plain-model",
                is_default=True,
                selection_variants=(
                    ACPModelSelectionVariant(
                        name="plain-model",
                        model="plain-model",
                        is_default=True,
                    ),
                ),
            )
        ]
    }
    card = build_workflow_config_card(
        WorkflowConfigEditorState(
            editor_tool="codex",
            editor_model="plain-model",
            editor_effort="xhigh",
        ),
        caps,
        model_options_by_tool=custom_options,
    )
    _assert_card_2_valid(card)
    selects = {s["name"]: s for s in _walk_dicts(card) if s.get("tag") == "select_static"}
    assert "effort" in selects
    assert [opt["value"] for opt in selects["effort"]["options"]] == ["default"]
    assert selects["effort"].get("initial_option") == "default"
    for d in _walk_dicts(card):
        if "value" in d and isinstance(d["value"], dict) and "action" in d["value"]:
            assert "effort" not in d["value"]


def test_config_card_clears_stale_profile_when_cascade_profiles_empty() -> None:
    caps = _sample_capabilities()
    card = build_workflow_config_card(
        WorkflowConfigEditorState(
            chat_id="oc_1",
            preset_id="collaborative",
            editor_role="worker",
            editor_tool="codex",
            editor_model="gpt-5.4",
            editor_profile="max",
            editor_effort="xhigh",
            requirement="Implement feature X",
        ),
        caps,
    )
    _assert_card_2_valid(card)
    selects = [d for d in _walk_dicts(card) if d.get("tag") == "select_static"]
    assert not any(s.get("name") == "profile" for s in selects)
    interactive = [
        d for d in _walk_dicts(card) if d.get("tag") in {"button", "select_static"}
    ]
    assert interactive
    for elem in interactive:
        assert "profile" not in elem["value"]
        assert "profile" not in elem["behaviors"][0]["value"]


def test_deep_collab_preset_ui_and_ux_preview_parity() -> None:
    preset_ids = [pid for pid, _label, _desc in WORKFLOW_PRESET_CATALOG]
    assert "deep_collab" in preset_ids

    caps = _sample_capabilities()
    card = build_workflow_config_card(
        WorkflowConfigEditorState(
            config_session_id="cfg_deep",
            chat_id="oc_1",
            preset_id="deep_collab",
            editor_tool="codex",
            editor_model="gpt-5.4",
            editor_effort="high",
        ),
        caps,
    )
    _assert_card_2_valid(card)
    selects = {s["name"]: s for s in _walk_dicts(card) if s.get("tag") == "select_static"}
    assert any(opt.get("value") == "deep_collab" for opt in selects["preset_id"]["options"])
    assert selects["preset_id"].get("initial_option") == "deep_collab"

    help_card = build_workflow_help_card(node_requirement_text="Node.js >= 20", chat_id="oc_1")
    assert "deep_collab" in json.dumps(help_card, ensure_ascii=False)

    ux_html = (
        Path(__file__).resolve().parents[1] / "ux" / "workflow-team-config.html"
    ).read_text(encoding="utf-8")
    for pid in preset_ids:
        assert f'data-preset="{pid}"' in ux_html
        assert f'value="{pid}"' in ux_html
    assert "deep_collab:" in ux_html


def test_capability_to_acp_options_preserves_openrouter_slash_model_id() -> None:
    from src.card.render.model_cascade import resolve_model_cascade

    models = tuple(f"org/model-{idx}" for idx in range(28)) + ("anthropic/claude-sonnet",)
    cap = ToolCapability(
        tool_name="traex",
        display_name="Traex",
        available=True,
        supports_model=True,
        supports_effort=True,
        models=models,
        efforts_by_model={"anthropic/claude-sonnet": ("low", "high")},
        profiles_by_model={"anthropic/claude-sonnet": ("standard", "max")},
        default_effort="high",
    )

    opts = capability_to_acp_options(
        cap,
        current_model="anthropic/claude-sonnet/max/high",
    )
    opt_names = [opt.name for opt in opts]
    assert len(opt_names) <= 25
    assert "anthropic/claude-sonnet" in opt_names
    assert "anthropic" not in opt_names

    cascade = resolve_model_cascade(
        opts,
        current_model="anthropic/claude-sonnet/max/high",
        selected_model="anthropic/claude-sonnet",
        selected_profile="max",
        selected_effort="high",
    )
    assert cascade.selected_model == "anthropic/claude-sonnet"
    assert cascade.selected_profile == "max"
    assert cascade.selected_effort == "high"

    # Codex/Claude/AGY slash-bearing OpenRouter IDs also retain provider prefix
    for tool_name in ("codex", "claude", "agy"):
        other_cap = ToolCapability(
            tool_name=tool_name,
            display_name=tool_name.upper(),
            available=True,
            supports_model=True,
            supports_effort=True,
            models=models,
            efforts_by_model={"anthropic/claude-sonnet": ("low", "high")},
        )
        other_opts = capability_to_acp_options(
            other_cap,
            current_model="anthropic/claude-sonnet/high",
        )
        other_names = [opt.name for opt in other_opts]
        assert "anthropic/claude-sonnet" in other_names
        assert "anthropic" not in other_names


def test_launch_config_card_places_header_and_quick_launch_cta_before_selectors() -> None:
    members = (
        WorkflowAgentBinding(
            agent_id="A1",
            tool_name="codex",
            model_name="gpt-5.4",
            display_name="Codex",
            effort="low",
            role="scout",
        ),
        WorkflowAgentBinding(
            agent_id="A2",
            tool_name="claude",
            model_name="sonnet",
            display_name="Claude",
            effort="high",
            role="worker",
        ),
    )
    card = build_workflow_config_card(
        WorkflowConfigEditorState(
            config_session_id="cfg_launch",
            chat_id="oc_1",
            project_id="proj_1",
            preset_id="collaborative",
            members=members,
            orchestrator_agent_id="A1",
            selected_agent_id="A1",
            editor_role="scout",
            editor_tool="codex",
            editor_model="gpt-5.4",
            editor_effort="low",
            requirement="分析项目并提出优化方案",
        ),
        _sample_capabilities(),
    )
    _assert_card_2_valid(card)
    assert card["header"]["title"]["content"] == "⚡ Workflow · 配置团队并启动"
    assert (
        card["header"]["subtitle"]["content"]
        == "可直接按当前配置启动，或在下方灵活调整 工具 + 模型 + Effort"
    )

    elements = card["body"]["elements"]
    req_idx = next(
        i
        for i, el in enumerate(elements)
        if el.get("tag") == "markdown" and "任务需求" in el.get("content", "")
    )
    roster_idx = next(
        i
        for i, el in enumerate(elements)
        if el.get("tag") == "markdown" and "当前协作团队" in el.get("content", "")
    )
    cta_idx = next(
        i
        for i, el in enumerate(elements)
        if action_ids.WF_START_WORKFLOW in json.dumps(el, ensure_ascii=False)
        and "按当前团队立即开始" in json.dumps(el, ensure_ascii=False)
    )
    save_top_idx = next(
        i
        for i, el in enumerate(elements)
        if action_ids.WF_SAVE_TEAM in json.dumps(el, ensure_ascii=False)
    )
    first_select_idx = next(
        i for i, el in enumerate(elements) if el.get("tag") == "select_static"
    )
    assert req_idx < roster_idx < cta_idx < first_select_idx
    assert save_top_idx == cta_idx

    cta_cols = elements[cta_idx]["columns"]
    first_btn = cta_cols[0]["elements"][0]
    second_btn = cta_cols[1]["elements"][0]
    assert first_btn["type"] == "primary"
    assert first_btn["value"]["action"] == action_ids.WF_START_WORKFLOW
    assert second_btn["type"] == "default"
    assert second_btn["value"]["action"] == action_ids.WF_SAVE_TEAM


def test_unified_11_tool_model_and_effort_config_card_matrix() -> None:
    caps = {
        "traex": ToolCapability(
            tool_name="traex",
            display_name="Traex",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=("Doubao-Seed-Evolving", "gpt-6.2-sol"),
            profiles_by_model={
                "Doubao-Seed-Evolving": ("standard",),
                "gpt-6.2-sol": ("standard", "max"),
            },
            efforts_by_model={"gpt-6.2-sol": ("low", "medium", "high", "xhigh")},
            authoritative_models=True,
        ),
        "claude": ToolCapability(tool_name="claude", display_name="Claude", available=True, supports_model=True, supports_effort=True, models=("default",)),
        "claude_w": ToolCapability(tool_name="claude_w", display_name="Claude-W", available=True, supports_model=True, supports_effort=True, models=("default",)),
        "codex": ToolCapability(
            tool_name="codex",
            display_name="Codex",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=("gpt-6-astra", "gpt-6-luna"),
            efforts_by_model={"gpt-6-luna": ("low", "medium", "high", "xhigh", "max")},
            default_effort="xhigh",
            authoritative_models=True,
        ),
        "codex_w": ToolCapability(tool_name="codex_w", display_name="Codex-W", available=True, supports_model=True, supports_effort=True, models=("default",)),
        "agy": ToolCapability(
            tool_name="agy",
            display_name="AGY",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=("gemini-4-neon", "gemini-pro-orchestrator-neon", "gpt-oss-120b-medium"),
            efforts_by_model={
                "gemini-4-neon": ("low", "medium", "high", "xhigh"),
                "gemini-pro-orchestrator-neon": ("low", "medium", "high", "xhigh"),
            },
            profiles_by_model={"gemini-pro-orchestrator-neon": ("low", "medium", "high")},
            default_effort="high",
            authoritative_models=True,
        ),
        "dsh": ToolCapability(tool_name="dsh", display_name="DSH", available=True, supports_model=True, supports_effort=True, models=("deepseek/deepseek-reasoner",), efforts_by_model={"deepseek/deepseek-reasoner": ("low", "medium", "high")}),
        "grok": ToolCapability(tool_name="grok", display_name="Grok", available=True, supports_model=True, supports_effort=False, models=()),
        "aiden": ToolCapability(tool_name="aiden", display_name="Aiden", available=True, supports_model=True, supports_effort=False, models=()),
        "gemini": ToolCapability(tool_name="gemini", display_name="Gemini", available=True, supports_model=True, supports_effort=False, models=()),
        "coco": ToolCapability(tool_name="coco", display_name="Coco", available=True, supports_model=True, supports_effort=False, models=("Doubao-Seed-Evolving", "gpt-6.1-sol"), authoritative_models=True),
    }
    model_options = {
        "agy": [
            ACPModelOption(
                name="gemini-4-neon",
                is_default=True,
                reasoning_efforts=("low", "medium", "high", "xhigh"),
                default_reasoning_effort="high",
                selection_variants=(
                    ACPModelSelectionVariant(name="gemini-4-neon/low", model="gemini-4-neon", effort="low"),
                    ACPModelSelectionVariant(name="gemini-4-neon/high", model="gemini-4-neon", effort="high", is_default=True),
                ),
            ),
            ACPModelOption(
                name="gemini-pro-orchestrator-neon",
                selection_variants=(
                    ACPModelSelectionVariant(name="gemini-pro-orchestrator-neon", model="gemini-pro-orchestrator-neon", profile="default", effort=None, is_default=True),
                    ACPModelSelectionVariant(name="gemini-pro-orchestrator-neon/high/high", model="gemini-pro-orchestrator-neon", profile="high", effort="high"),
                ),
            ),
            ACPModelOption(
                name="gpt-oss-120b-medium",
                selection_variants=(ACPModelSelectionVariant(name="gpt-oss-120b-medium", model="gpt-oss-120b-medium", is_default=True),),
            ),
        ],
    }

    for tool_name in caps:
        card = build_workflow_config_card(
            WorkflowConfigEditorState(editor_tool=tool_name),
            caps,
            model_options_by_tool=model_options,
        )
        _assert_card_2_valid(card)
        selects = {s["name"]: s for s in _walk_dicts(card) if s.get("tag") == "select_static"}
        assert "model_name" in selects and "effort" in selects
        assert selects["model_name"]["options"][0]["value"] == "default"
        assert selects["model_name"].get("initial_option") == "default"
        assert selects["effort"]["options"][0]["value"] == "default"

    # AGY standard model shows real catalog + default & effort ladder
    agy_std_card = build_workflow_config_card(
        WorkflowConfigEditorState(editor_tool="agy", editor_model="gemini-4-neon"),
        caps,
        model_options_by_tool=model_options,
    )
    _assert_card_2_valid(agy_std_card)
    agy_std_sels = {s["name"]: s for s in _walk_dicts(agy_std_card) if s.get("tag") == "select_static"}
    assert [o["value"] for o in agy_std_sels["model_name"]["options"]] == [
        "default",
        "gemini-4-neon",
        "gemini-pro-orchestrator-neon",
        "gpt-oss-120b-medium",
    ]
    assert [o["value"] for o in agy_std_sels["effort"]["options"]] == ["default", "low", "high"]
    assert agy_std_sels["effort"].get("initial_option") == "default"
    assert "effort" not in agy_std_sels["effort"]["value"]

    # Explicit Effort choice selects "high" cleanly
    agy_hi_card = build_workflow_config_card(
        WorkflowConfigEditorState(editor_tool="agy", editor_model="gemini-4-neon", editor_effort="high"),
        caps,
        model_options_by_tool=model_options,
    )
    _assert_card_2_valid(agy_hi_card)
    agy_hi_sels = {s["name"]: s for s in _walk_dicts(agy_hi_card) if s.get("tag") == "select_static"}
    assert agy_hi_sels["effort"].get("initial_option") == "high"
    assert agy_hi_sels["effort"]["value"].get("effort") == "high"

    # DSH JSON wire selection on cold/warm model decodes cleanly to provider/model + effort
    dsh_wire_card = build_workflow_config_card(
        WorkflowConfigEditorState(editor_tool="dsh", editor_model='["deepseek","deepseek-reasoner","high"]'),
        caps,
    )
    _assert_card_2_valid(dsh_wire_card)
    dsh_sels = {s["name"]: s for s in _walk_dicts(dsh_wire_card) if s.get("tag") == "select_static"}
    assert dsh_sels["model_name"].get("initial_option") == "deepseek/deepseek-reasoner"
    assert dsh_sels["effort"].get("initial_option") == "high"

    # AGY Orchestrator with profile=default keeps single default Effort; profile=high exposes Effort
    agy_orch_default = build_workflow_config_card(
        WorkflowConfigEditorState(editor_tool="agy", editor_model="gemini-pro-orchestrator-neon", editor_profile="default"),
        caps,
        model_options_by_tool=model_options,
    )
    _assert_card_2_valid(agy_orch_default)
    orch_def_sels = {s["name"]: s for s in _walk_dicts(agy_orch_default) if s.get("tag") == "select_static"}
    assert orch_def_sels["profile"].get("initial_option") == "default"
    assert [o["value"] for o in orch_def_sels["effort"]["options"]] == ["default"]

    agy_orch_high = build_workflow_config_card(
        WorkflowConfigEditorState(editor_tool="agy", editor_model="gemini-pro-orchestrator-neon", editor_profile="high"),
        caps,
        model_options_by_tool=model_options,
    )
    _assert_card_2_valid(agy_orch_high)
    orch_hi_sels = {s["name"]: s for s in _walk_dicts(agy_orch_high) if s.get("tag") == "select_static"}
    assert [o["value"] for o in orch_hi_sels["effort"]["options"]] == ["default", "high"]
    assert orch_hi_sels["effort"].get("initial_option") == "default"

    # Effortless authoritative models on effort-capable backends (codex gpt-6-astra, traex Doubao, agy gpt-oss)
    for t_name, m_name in (
        ("codex", "gpt-6-astra"),
        ("traex", "Doubao-Seed-Evolving"),
        ("agy", "gpt-oss-120b-medium"),
    ):
        card = build_workflow_config_card(
            WorkflowConfigEditorState(editor_tool=t_name, editor_model=m_name),
            caps,
            model_options_by_tool=model_options,
        )
        _assert_card_2_valid(card)
        sels = {s["name"]: s for s in _walk_dicts(card) if s.get("tag") == "select_static"}
        assert [o["value"] for o in sels["effort"]["options"]] == ["default"]
        assert sels["effort"].get("initial_option") == "default"
        assert "effort" not in sels["effort"]["value"]

