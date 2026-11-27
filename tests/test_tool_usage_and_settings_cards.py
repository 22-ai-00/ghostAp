"""Tests for Tool Usage & Settings Card 2.0 builders, runtime settings persistence, and cockpit entries."""

from __future__ import annotations

import ast
import concurrent.futures
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import pytest

from src.acp.models import ACPAvailableCommand, ACPUsageInfo
from src.acp.options import ACPModelOption, ACPModelSelectionVariant
from src.card.actions import dispatch as action_ids
from src.card.builders.diagnostics import DiagnosticsBuilder
from src.card.builders.system import SystemBuilder
from src.card.builders.tool_cockpit import ToolCockpitBuilder
from src.card.schema import require_card_json_2
from src.card.shared.truncation import count_tagged_nodes
from src.project.context import ChatProjectState, ProjectContext


def _walk(obj: Any) -> Iterable[dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for child in obj.values():
            yield from _walk(child)
    elif isinstance(obj, list):
        for child in obj:
            yield from _walk(child)


def _parse_card(result: tuple[str, str]) -> tuple[dict[str, Any], str]:
    msg_type, raw = result
    assert msg_type == "interactive"
    card = json.loads(raw)
    require_card_json_2(card)
    assert (card.get("config") or {}).get("update_multi") is True
    assert count_tagged_nodes(card) <= 180
    assert len(raw.encode("utf-8")) <= 27 * 1024
    return card, raw


def _nodes(card: dict[str, Any], tag: str) -> list[dict[str, Any]]:
    return [n for n in _walk(card) if n.get("tag") == tag]


def _callback_value(node: dict[str, Any]) -> dict[str, Any]:
    val = node.get("value")
    assert isinstance(val, dict)
    assert node.get("behaviors") == [{"type": "callback", "value": val}]
    return val


def _sample_agy_models() -> list[ACPModelOption]:
    return [
        ACPModelOption(
            name="gemini-pro-orchestrator",
            description="AGY Orchestrator",
            is_default=True,
            selection_variants=(
                ACPModelSelectionVariant(
                    name="gemini-pro-orchestrator",
                    model="gemini-pro-orchestrator",
                    profile="default",
                    effort=None,
                    is_default=True,
                ),
                ACPModelSelectionVariant(
                    name="gemini-pro-orchestrator/high/medium",
                    model="gemini-pro-orchestrator",
                    profile="high",
                    effort="medium",
                ),
                ACPModelSelectionVariant(
                    name="gemini-pro-orchestrator/xhigh/high",
                    model="gemini-pro-orchestrator",
                    profile="xhigh",
                    effort="high",
                ),
            ),
        ),
        ACPModelOption(
            name="gemini-4-hydrogen",
            description="Fast AGY model",
            selection_variants=(),
            reasoning_efforts=(),
            default_reasoning_effort=None,
        ),
    ]


def test_tool_cockpit_action_constants_and_frozenset() -> None:
    expected = {
        "OPEN_TOOL_USAGE": "open_tool_usage",
        "REFRESH_TOOL_USAGE": "refresh_tool_usage",
        "OPEN_TOOL_SETTINGS": "open_tool_settings",
        "REFRESH_TOOL_SETTINGS": "refresh_tool_settings",
        "SETTINGS_SELECT_TOOL": "settings_select_tool",
        "SETTINGS_SELECT_MODEL_GROUP": "settings_select_model_group",
        "SETTINGS_SELECT_MODEL_PROFILE": "settings_select_model_profile",
        "SETTINGS_SELECT_MODEL_EFFORT": "settings_select_model_effort",
        "SETTINGS_SELECT_ORCH_PLANNER_MODEL": "settings_select_orch_planner_model",
        "SETTINGS_SELECT_ORCH_EXECUTOR_MODEL": "settings_select_orch_executor_model",
        "SETTINGS_TOGGLE_SHOW_THINKING": "settings_toggle_show_thinking",
        "SETTINGS_SELECT_VERBOSITY": "settings_select_verbosity",
        "SETTINGS_SELECT_AGENT_MODE": "settings_select_agent_mode",
        "SETTINGS_CLEAR_SESSION": "settings_clear_session",
    }
    for attr, value in expected.items():
        assert getattr(action_ids, attr) == value
    assert isinstance(action_ids.TOOL_COCKPIT_ACTIONS, frozenset)
    assert action_ids.TOOL_COCKPIT_ACTIONS == frozenset(expected.values())


def test_chat_project_state_and_context_tool_runtime_settings_roundtrip_and_isolation(
    tmp_path: Path,
) -> None:
    legacy_state = ChatProjectState.from_dict(
        {
            "acp_tool_name": "agy",
            "acp_model_name": "gemini-4-hydrogen",
            "active_modes": ["agy"],
        }
    )
    assert legacy_state.tool_runtime_settings == {}

    legacy_state.tool_runtime_settings["agy"] = {"show_thinking": True, "verbosity": "high"}
    cloned = legacy_state.clone()
    cloned.tool_runtime_settings["agy"]["verbosity"] = "low"
    assert legacy_state.tool_runtime_settings["agy"]["verbosity"] == "high"

    roundtrip_state = ChatProjectState.from_dict(legacy_state.to_dict())
    assert roundtrip_state.tool_runtime_settings == {"agy": {"show_thinking": True, "verbosity": "high"}}
    assert ChatProjectState.from_dict({"tool_runtime_settings": "corrupt"}).tool_runtime_settings == {}

    ctx = ProjectContext("p1", "Demo", str(tmp_path))
    assert ctx.get_tool_runtime_settings("chat-a", "agy") == {}
    saved = ctx.set_tool_runtime_settings(
        "chat-a",
        "agy",
        {"show_thinking": False, "verbosity": "high", "agent_mode": "plan"},
    )
    assert saved == {"show_thinking": False, "verbosity": "high", "agent_mode": "plan"}
    saved["verbosity"] = "low"
    assert ctx.get_tool_runtime_settings("chat-a", "agy") == {
        "show_thinking": False,
        "verbosity": "high",
        "agent_mode": "plan",
    }
    assert ctx.get_tool_runtime_settings("chat-b", "agy") == {}

    # Tool name normalization (claude-w == claude_w) and default chat_id=None fallback
    ctx.set_tool_runtime_settings(None, "claude-w", {"custom_flag": True})
    assert ctx.get_tool_runtime_settings(None, "claude_w") == {"custom_flag": True}

    restored = ProjectContext.from_snapshot(ctx.to_snapshot())
    assert restored.get_tool_runtime_settings("chat-a", "agy") == {
        "show_thinking": False,
        "verbosity": "high",
        "agent_mode": "plan",
    }
    assert restored.get_tool_runtime_settings("chat-b", "agy") == {}
    assert restored.get_tool_runtime_settings(None, "claude_w") == {"custom_flag": True}

    # Clearing via None removes only the targeted tool
    assert restored.set_tool_runtime_settings("chat-a", "agy", None) == {}
    assert restored.get_tool_runtime_settings("chat-a", "agy") == {}

    # Concurrent reads/writes under _chat_lock stay consistent
    def _worker(idx: int) -> None:
        cid = f"chat-{idx % 4}"
        ctx.set_tool_runtime_settings(cid, "agy", {"verbosity": "high", "seq": idx})
        got = ctx.get_tool_runtime_settings(cid, "agy")
        assert "seq" in got

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(_worker, range(24)))


@pytest.mark.parametrize(
    ("remaining", "expected_color"),
    [
        (0.84, "blue"),
        (0.19, "orange"),
        (0.04, "red"),
    ],
)
def test_tool_usage_card_adaptive_header_bars_epoch_credits_and_buttons(
    remaining: float,
    expected_color: str,
) -> None:
    snap = SimpleNamespace(
        quota_groups=[
            SimpleNamespace(
                name="Gemini Models",
                buckets=[
                    SimpleNamespace(
                        id="gemini-weekly",
                        name="Weekly Limit Remaining",
                        window="weekly",
                        remaining_fraction=remaining,
                        reset_time="2026-11-30T19:58:01Z",
                    ),
                    SimpleNamespace(
                        id="gemini-5h",
                        name="Five Hour Limit Remaining",
                        window="5h",
                        remaining_fraction=1.0,
                        reset_time="1970-01-01T00:00:00Z",
                    ),
                ],
            )
        ],
        remaining_credits=25,
        use_g1_credits=True,
        session_active=True,
        session_id="sess-usage-123456",
        model_name="gemini-pro-orchestrator/high/medium",
        turns=7,
        duration_seconds=135.0,
        usage_info=ACPUsageInfo(used=12400, size=200000, cost_amount=0.125, cost_currency="USD"),
        capability_note="Models share 5h & weekly pools.",
    )
    card, raw = _parse_card(
        ToolCockpitBuilder.build_tool_usage_card(
            "agy",
            snap,
            project_name="GhostAP",
            project_id="proj-1",
            thread_root_id="thread-9",
        )
    )
    assert card["header"]["template"] == expected_color
    assert card["header"]["subtitle"]["content"] == "项目：GhostAP"
    assert "█" in raw and "░" in raw
    assert "未消耗（满额）" in raw
    assert "1970-01-01" not in raw
    assert "2026-11-30 19:58 UTC" in raw
    assert "G1 Credits" in raw and "`25`" in raw
    assert "12,400 / 200,000" in raw
    assert "0.125 USD" in raw
    assert "7 轮" in raw

    buttons = _nodes(card, "button")
    callbacks = [_callback_value(btn) for btn in buttons]
    assert [cb["action"] for cb in callbacks] == [
        action_ids.REFRESH_TOOL_USAGE,
        action_ids.OPEN_TOOL_SETTINGS,
        action_ids.RUN_NATIVE_TOOL_COMMAND,
        action_ids.SHOW_TOOL_COMMANDS,
    ]
    assert callbacks[2]["command"] == "/compact"
    assert all(cb["thread_root_id"] == "thread-9" for cb in callbacks)
    assert all(cb["project_id"] == "proj-1" for cb in callbacks)


def test_tool_usage_card_handles_empty_snapshot_and_sanitizes_unsafe_text() -> None:
    empty_card, empty_raw = _parse_card(ToolCockpitBuilder.build_tool_usage_card("codex", None))
    assert empty_card["header"]["template"] == "blue"
    assert "未提供结构化配额窗口" in empty_raw

    dirty_snap = {
        "error": "failed <script>alert(1)</script>\ud800",
        "capability_note": "[click](javascript:alert(1))",
        "session_active": False,
        "turns": 0,
    }
    dirty_card, dirty_raw = _parse_card(ToolCockpitBuilder.build_tool_usage_card("agy", dirty_snap))
    assert "<script>" not in dirty_raw
    assert "javascript:" not in dirty_raw
    dirty_raw.encode("utf-8", errors="strict")
    assert dirty_card["header"]["template"] == "blue"


def test_tool_settings_card_agy_controls_cascade_and_readonly_config() -> None:
    settings_snap = SimpleNamespace(
        show_thinking=True,
        verbosity="high",
        agent_mode="plan",
        backend_config_summary={
            "toolPermission": "request-review",
            "enableTerminalSandbox": False,
            "apiToken": "secret-should-not-leak",
            "trustedWorkspaces": ["/secret/path"],
        },
    )
    card, raw = _parse_card(
        ToolCockpitBuilder.build_tool_settings_card(
            "agy",
            models=_sample_agy_models(),
            current_model="gemini-pro-orchestrator/high/medium",
            available_tools=["agy", "codex", "claude"],
            settings_snapshot=settings_snap,
            project_name="GhostAP",
            project_id="proj-1",
            thread_root_id="thread-2",
            status_banner="✅ 已保存设置",
        )
    )

    assert "secret-should-not-leak" not in raw
    assert "/secret/path" not in raw
    assert "toolPermission" in raw
    assert "已保存设置" in raw
    assert "Planner Effort" in raw
    assert "Executor Effort" in raw

    selects = {node["name"]: node for node in _nodes(card, "select_static")}
    assert set(selects) == {
        "tool_select",
        "model_group",
        "model_profile",
        "model_effort",
        "verbosity",
        "agent_mode",
    }
    assert all("width" not in sel for sel in selects.values())
    assert selects["tool_select"]["initial_option"] == "agy"
    assert selects["model_group"]["initial_option"] == "gemini-pro-orchestrator"
    assert selects["model_profile"]["initial_option"] == "high"
    assert selects["model_effort"]["initial_option"] == "medium"
    assert selects["verbosity"]["initial_option"] == "high"
    assert selects["agent_mode"]["initial_option"] == "plan"

    assert _callback_value(selects["tool_select"])["action"] == action_ids.SETTINGS_SELECT_TOOL
    assert _callback_value(selects["model_group"])["action"] == action_ids.SETTINGS_SELECT_MODEL_GROUP
    assert _callback_value(selects["model_profile"])["action"] == action_ids.SETTINGS_SELECT_MODEL_PROFILE
    assert _callback_value(selects["model_effort"])["action"] == action_ids.SETTINGS_SELECT_MODEL_EFFORT
    assert _callback_value(selects["verbosity"])["action"] == action_ids.SETTINGS_SELECT_VERBOSITY
    assert _callback_value(selects["agent_mode"])["action"] == action_ids.SETTINGS_SELECT_AGENT_MODE

    buttons = _nodes(card, "button")
    callbacks = [_callback_value(btn) for btn in buttons]
    actions_order = [cb["action"] for cb in callbacks]
    assert actions_order == [
        action_ids.SETTINGS_TOGGLE_SHOW_THINKING,
        action_ids.OPEN_TOOL_USAGE,
        action_ids.SHOW_TOOL_COMMANDS,
        action_ids.RUN_NATIVE_TOOL_COMMAND,
        action_ids.SETTINGS_CLEAR_SESSION,
        action_ids.REFRESH_TOOL_SETTINGS,
    ]
    assert callbacks[0]["show_thinking"] is False
    assert callbacks[0]["next_show_thinking"] is False
    assert callbacks[0]["current_show_thinking"] is True
    assert callbacks[3]["command"] == "/compact"


def test_tool_settings_card_agy_defaults_show_thinking_true_and_code_spans_preserve_underscores() -> None:
    default_card, default_raw = _parse_card(
        ToolCockpitBuilder.build_tool_settings_card(
            "agy",
            models=[ACPModelOption(name="c_o_new_thinking")],
            current_model="c_o_new_thinking",
            available_tools=["agy", "claude_w"],
            settings_snapshot=None,
        )
    )
    assert "💭 显示思考过程：已开启" in default_raw
    assert "`c_o_new_thinking`" in default_raw
    assert "c\\_o\\_new\\_thinking" not in default_raw
    toggle_btn = next(
        _callback_value(btn)
        for btn in _nodes(default_card, "button")
        if _callback_value(btn)["action"] == action_ids.SETTINGS_TOGGLE_SHOW_THINKING
    )
    assert toggle_btn["show_thinking"] is False
    assert toggle_btn["next_show_thinking"] is False
    assert toggle_btn["current_show_thinking"] is True

    off_card, off_raw = _parse_card(
        ToolCockpitBuilder.build_tool_settings_card(
            "agy",
            models=[ACPModelOption(name="c_o_new_thinking")],
            current_model="c_o_new_thinking",
            available_tools=["agy"],
            settings_snapshot=SimpleNamespace(show_thinking=False),
        )
    )
    assert "💭 显示思考过程：已关闭" in off_raw
    off_toggle = next(
        _callback_value(btn)
        for btn in _nodes(off_card, "button")
        if _callback_value(btn)["action"] == action_ids.SETTINGS_TOGGLE_SHOW_THINKING
    )
    assert off_toggle["show_thinking"] is True
    assert off_toggle["next_show_thinking"] is True
    assert off_toggle["current_show_thinking"] is False

    _, usage_raw = _parse_card(
        ToolCockpitBuilder.build_tool_usage_card(
            "claude_w",
            SimpleNamespace(model_name="gpt_4o_mini", session_active=True, session_id="sess_abc_1"),
        )
    )
    assert "`gpt_4o_mini`" in usage_raw
    assert "gpt\\_4o\\_mini" not in usage_raw


def test_tool_settings_card_non_agy_empty_models_and_large_catalog_budget() -> None:
    card_empty, raw_empty = _parse_card(
        ToolCockpitBuilder.build_tool_settings_card(
            "codex",
            models=[],
            current_model=None,
            available_tools=["codex", "agy"],
        )
    )
    selects_empty = {node["name"]: node for node in _nodes(card_empty, "select_static")}
    assert set(selects_empty) == {"tool_select", "model_group"}
    assert selects_empty["model_group"]["initial_option"] == "default"
    assert "default（后端托管）" in raw_empty
    assert "思考强度：后端默认（当前模型无独立档位）" in raw_empty
    assert action_ids.SETTINGS_TOGGLE_SHOW_THINKING not in raw_empty

    # Large catalog must stay within 180 tagged nodes and 27KB
    many_models = [ACPModelOption(name=f"model-{i}") for i in range(80)]
    many_tools = [f"tool-{i}" for i in range(40)]
    card_big, _ = _parse_card(
        ToolCockpitBuilder.build_tool_settings_card(
            "codex",
            models=many_models,
            current_model="model-75",
            available_tools=many_tools,
        )
    )
    selects_big = {node["name"]: node for node in _nodes(card_big, "select_static")}
    assert "model-75" in [opt["value"] for opt in selects_big["model_group"]["options"]]


def test_native_commands_ready_and_workspace_cockpit_cards_upgraded_within_budgets() -> None:
    long_desc = "x" * 120
    commands = [
        ACPAvailableCommand("compact", "压缩当前上下文 " + long_desc, ""),
        ACPAvailableCommand("swarm", "协同任务", "<任务>"),
        *[
            ACPAvailableCommand(f"skill:demo-{i}", f"Skill 技能描述 {i} " + long_desc, "")
            for i in range(40)
        ],
    ]
    cmds_card, cmds_raw = _parse_card(
        SystemBuilder.build_tool_native_commands_card(
            "agy",
            commands,
            source="live",
            project_id="proj-1",
            thread_root_id="thread-1",
        )
    )
    assert "🧩 已加载 Skills（共 40 个）" in cmds_raw
    assert ("x" * 80) not in cmds_raw  # descriptions truncated <= 72 chars
    cmd_actions = {_callback_value(btn)["action"] for btn in _nodes(cmds_card, "button")}
    assert action_ids.OPEN_TOOL_USAGE in cmd_actions
    assert action_ids.OPEN_TOOL_SETTINGS in cmd_actions
    assert action_ids.RUN_NATIVE_TOOL_COMMAND in cmd_actions

    ready_card, _ = _parse_card(
        SystemBuilder.build_acp_programming_ready_card("agy", "gemini-4-hydrogen", "proj-1")
    )
    ready_actions = [_callback_value(btn)["action"] for btn in _nodes(ready_card, "button")]
    assert ready_actions == [
        action_ids.OPEN_ACP_MODEL_PICKER,
        action_ids.SHOW_TOOL_COMMANDS,
        action_ids.OPEN_TOOL_USAGE,
        action_ids.OPEN_TOOL_SETTINGS,
    ]

    cockpit_card, _ = _parse_card(
        DiagnosticsBuilder.build_workspace_cockpit_card(
            SimpleNamespace(project_id="proj-1", project_name="Demo", root_path="/tmp/demo"),
            "/tmp/demo",
            git_branch="feat/cockpit-ui",
            direct_info={"mode_key": "agy", "tool_name": "agy", "model_name": "default"},
        )
    )
    cockpit_actions = [_callback_value(btn)["action"] for btn in _nodes(cockpit_card, "button")]
    assert action_ids.OPEN_TOOL_USAGE in cockpit_actions
    assert action_ids.OPEN_TOOL_SETTINGS in cockpit_actions
    assert action_ids.SHOW_BOARD in cockpit_actions


def test_native_commands_delegates_to_partition_commands_and_skills_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.acp.native_commands as native_cmds

    commands = [
        ACPAvailableCommand("compact", "压缩当前上下文", ""),
        ACPAvailableCommand("custom-skill", "Custom helper", ""),
    ]
    monkeypatch.setattr(
        native_cmds,
        "partition_commands_and_skills",
        lambda _tool, cmds: ((cmds[0],), (cmds[1],)),
        raising=False,
    )
    card, raw = _parse_card(
        SystemBuilder.build_tool_native_commands_card("agy", commands, source="live")
    )
    assert "🧩 已加载 Skills（共 1 个）" in raw
    assert "/custom-skill" in raw
    assert action_ids.OPEN_TOOL_USAGE in json.dumps(card, ensure_ascii=False)


def test_tool_cockpit_builder_respects_import_boundaries_and_ux_previews_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    builder_path = root / "src" / "card" / "builders" / "tool_cockpit.py"
    tree = ast.parse(builder_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                assert "delivery" not in parts and "session" not in parts and "feishu" not in parts
                assert "tool_telemetry" not in parts
        elif isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            assert "delivery" not in parts and "session" not in parts and "feishu" not in parts
            assert "tool_telemetry" not in parts

    cockpit_html = (root / "ux" / "tool-usage-and-settings-cockpit.html").read_text(encoding="utf-8")
    native_html = (root / "ux" / "native-tool-commands-card.html").read_text(encoding="utf-8")
    assert "未消耗（满额）" in cockpit_html
    assert "显示思考过程" in cockpit_html
    assert "刷新用量" in cockpit_html
    assert "已加载 Skills" in native_html
    assert "用量 /usage" in native_html
