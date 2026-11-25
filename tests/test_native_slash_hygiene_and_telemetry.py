"""Tests for native slash hygiene, AGY CLI runtime settings/command parsing, factory forwarding, and telemetry."""

from __future__ import annotations

import io
import json
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.acp.execution_instructions import AUTONOMOUS_EXECUTION_PROMPT
from src.acp.models import ACPAvailableCommand, ACPUsageInfo
from src.acp.native_commands import (
    STATIC_TOOL_NATIVE_COMMANDS,
    STRICT_ARG_NATIVE_COMMANDS,
    TUI_ONLY_NATIVE_COMMANDS,
    ZERO_ARG_NATIVE_COMMANDS,
    build_acp_prompt_texts,
    get_tool_native_commands,
    get_tui_only_command_hint,
    is_non_task_native_slash,
    partition_commands_and_skills,
)
from src.acp.tool_telemetry import (
    QuotaBucketSnapshot,
    QuotaGroupSnapshot,
    ToolSettingsSnapshot,
    ToolUsageSnapshot,
    collect_tool_settings_snapshot,
    collect_tool_usage_snapshot,
)
from src.agent_session.agy_cli import AGYCLIConfig, SyncAGYCLISession
from src.agent_session.claude_cli import ClaudeCLIConfig, SyncClaudeCLISession
from src.agent_session.codex_cli import CodexCLIConfig, SyncCodexCLISession
from src.agent_session.factory import _start_base_session, create_programming_session
from src.feishu.handlers.base import BaseHandler
from src.project.unified_context import ContextSourceMode, UnifiedContext


def _completed_process(stdout_lines: list[str] | None = None, stderr: str = "", rc: int = 0) -> MagicMock:
    proc = MagicMock()
    proc.pid = 4321
    proc.returncode = rc
    proc.stdout = iter(stdout_lines or [])
    proc.stderr = io.StringIO(stderr)
    proc.poll.return_value = rc
    proc.wait.return_value = rc
    return proc


# ── 1. native_commands hygiene, TUI hints, merging, and partitioning ────────


def test_expanded_zero_arg_and_strict_arg_sets_and_is_non_task() -> None:
    expected_zero = {
        "compact",
        "status",
        "mcp",
        "skills",
        "review",
        "init",
        "cost",
        "doctor",
        "memory",
        "agents",
        "diff",
        "config",
        "usage",
        "quota",
        "credits",
        "settings",
        "changelog",
        "hooks",
        "permissions",
        "plugin",
        "plugins",
        "context",
        "stats",
        "help",
        "version",
        "learn",
        "todos",
        "bashes",
        "pr-comments",
        "release-notes",
        "security-review",
        "insights",
        "fast",
    }
    assert expected_zero <= ZERO_ARG_NATIVE_COMMANDS

    expected_strict = {
        "usage",
        "quota",
        "credits",
        "config",
        "settings",
        "changelog",
        "hooks",
        "permissions",
        "plugin",
        "plugins",
        "mcp",
        "agents",
        "skills",
        "status",
        "cost",
        "doctor",
        "context",
        "stats",
        "fast",
        "model",
        "effort",
        "help",
        "version",
    }
    assert expected_strict <= STRICT_ARG_NATIVE_COMMANDS

    # Non-task commands (bare & strict-arg parameterized)
    for cmd in (
        "/usage",
        "/quota",
        "/credits",
        "/settings",
        "/config",
        "/config show",
        "/plugin",
        "/plugin list",
        "/plugins list",
        "/mcp",
        "/mcp list",
        "/hooks",
        "/permissions",
        "/agents",
        "/skills",
        "/changelog",
        "/doctor",
        "/cost",
        "/memory",
        "/context",
        "/stats",
        "/help",
        "/help usage",
        "/version",
        "/learn",
        "/todos",
        "/bashes",
        "/pr-comments",
        "/release-notes",
        "/security-review",
        "/insights",
        "/fast",
        "/model gpt-5",
        "/effort high",
        "/compact",
        "/review",
    ):
        assert is_non_task_native_slash(cmd) is True, cmd
        assert build_acp_prompt_texts(cmd) == [cmd], cmd

    # Task-bearing parameterized commands stay wrapped with autonomous prompt at index 1
    for cmd in (
        "/swarm fix auth",
        "/boost deep research",
        "/plan refactor module",
        "/review focus security",
        "/review-branch main",
        "/review-commit abc1234",
        "/goal ship feature",
        "/teamwork build dashboard",
        "/grill-me api design",
        "/browser search docs",
        "/schedule daily check",
        "/compact focus auth",
        "/swarm",
        "/plan",
    ):
        assert is_non_task_native_slash(cmd) is False, cmd
        assert build_acp_prompt_texts(cmd) == [cmd, AUTONOMOUS_EXECUTION_PROMPT], cmd

    assert is_non_task_native_slash(None) is False
    assert is_non_task_native_slash("") is False
    assert is_non_task_native_slash("/usr/bin/python") is False
    assert is_non_task_native_slash("fix bug") is False
    assert build_acp_prompt_texts("fix bug") == [AUTONOMOUS_EXECUTION_PROMPT, "fix bug"]


def test_tui_only_commands_and_hints() -> None:
    expected_tui = {
        "vim",
        "terminal-setup",
        "theme",
        "color",
        "keybindings",
        "ide",
        "fullscreen",
        "clear-screen",
        "login",
        "auth",
        "attach",
    }
    assert expected_tui <= TUI_ONLY_NATIVE_COMMANDS

    for cmd in expected_tui:
        hint = get_tui_only_command_hint(f"/{cmd}")
        assert hint is not None and f"/{cmd}" in hint
        assert "终端" in hint

    agy_ctx_hint = get_tui_only_command_hint("/context", tool_name="agy")
    assert agy_ctx_hint is not None
    assert "/usage" in agy_ctx_hint and "/status" in agy_ctx_hint

    assert get_tui_only_command_hint("/context", tool_name="claude") is None
    assert get_tui_only_command_hint("/context") is None
    assert get_tui_only_command_hint("/usage", tool_name="agy") is None
    assert get_tui_only_command_hint("/swarm fix", tool_name="agy") is None
    assert get_tui_only_command_hint(None) is None


def test_static_catalog_all_11_tools_live_merge_and_partition() -> None:
    expected_tools = {
        "agy",
        "codex",
        "codex_w",
        "traex",
        "coco",
        "claude",
        "claude_w",
        "gemini",
        "aiden",
        "grok",
        "dsh",
    }
    assert set(STATIC_TOOL_NATIVE_COMMANDS.keys()) == expected_tools
    for tool in expected_tools:
        names = {cmd.name for cmd in STATIC_TOOL_NATIVE_COMMANDS[tool]}
        assert {"usage", "settings", "compact", "skills"} <= names, tool

    agy_names = {cmd.name for cmd in STATIC_TOOL_NATIVE_COMMANDS["agy"]}
    assert {"usage", "credits", "settings", "agents", "mcp", "plugin", "hooks", "permissions", "changelog"} <= agy_names

    claude_names = {cmd.name for cmd in STATIC_TOOL_NATIVE_COMMANDS["claude"]}
    assert {"usage", "settings", "doctor", "memory", "agents"} <= claude_names

    # Merge live commands (with skills & omitted input_hint) + static core commands
    live = [
        ACPAvailableCommand("PLAN", "Live plan", ""),
        ACPAvailableCommand("compact", "Live compact", ""),
        ACPAvailableCommand("bytedcli", "Internal platform skill", ""),
        ACPAvailableCommand("firecrawl", "Web scraping skill", ""),
        ACPAvailableCommand("plugin:custom", "Plugin command", ""),
        ACPAvailableCommand("logout", "Blocked", ""),
    ]
    merged, src = get_tool_native_commands("agy", live_commands=live)
    assert src == "live"
    merged_names_lower = [c.name.casefold() for c in merged]
    # Case-insensitive dedupe
    assert merged_names_lower.count("plan") == 1
    assert merged_names_lower.count("compact") == 1
    assert "logout" not in merged_names_lower
    # Static core commands remain discoverable alongside live commands
    for required in ("usage", "credits", "settings", "agents", "mcp", "plugin", "hooks", "permissions", "swarm"):
        assert required in merged_names_lower
    # Missing live input_hint inherits static hint
    plan_cmd = next(c for c in merged if c.name.casefold() == "plan")
    assert plan_cmd.input_hint == "<目标>"

    core_cmds, skill_cmds = partition_commands_and_skills("agy", merged)
    core_names = {c.name.casefold() for c in core_cmds}
    skill_names = {c.name.casefold() for c in skill_cmds}
    assert {"plan", "compact", "usage", "credits", "settings", "skills", "swarm"} <= core_names
    assert {"bytedcli", "firecrawl", "plugin:custom"} == skill_names


# ── 2. Clean slash pass-through across sessions & bridge ────────────────────


def test_clean_slash_passthrough_across_bridge_and_all_cli_sessions(monkeypatch, tmp_path) -> None:
    # BaseHandler.inject_bridge_context
    ctx_store = {}
    uctx = UnifiedContext(project_id="p1")
    uctx.add_conversation("user", "prior context", source_mode=ContextSourceMode.SMART)
    uctx.build_bridge_summary(ContextSourceMode.SMART, ContextSourceMode.AGY)
    ctx_store[("p1", "c1")] = uctx

    handler = object.__new__(BaseHandler)
    handler.ctx = SimpleNamespace(
        context_manager=SimpleNamespace(
            store=SimpleNamespace(get=lambda pid, chat_id="": ctx_store.get((pid, chat_id)))
        )
    )
    project = SimpleNamespace(project_id="p1", project_name="demo")

    for cmd in ("/usage", "/credits", "/settings", "/plugin list", "/mcp list", "/hooks", "/doctor"):
        assert handler.inject_bridge_context(cmd, project, chat_id="c1") == cmd
    assert uctx.last_bridge_summary is not None

    injected = handler.inject_bridge_context("/swarm fix issue", project, chat_id="c1")
    assert injected.startswith("/swarm fix issue\n\n")
    assert "prior context" in injected

    # Codex CLI
    codex_sess = SyncCodexCLISession(cwd=str(tmp_path), config=CodexCLIConfig(add_dir=False))
    assert codex_sess._build_args("/usage", False)[-1] == "/usage"
    assert codex_sess._build_args("/plugin list", False)[-1] == "/plugin list"
    codex_task = codex_sess._build_args("/review focus auth", False)[-1]
    assert codex_task.startswith("/review focus auth") and AUTONOMOUS_EXECUTION_PROMPT in codex_task

    # Claude CLI
    claude_sess = SyncClaudeCLISession(cwd=str(tmp_path), config=ClaudeCLIConfig(add_dir=False))
    claude_sess.session_id = "s-claude"
    usage_argv = claude_sess._build_args("/usage", False)
    assert "--append-system-prompt" not in usage_argv
    assert usage_argv[-2:] == ["--", "/usage"]

    doctor_argv = claude_sess._build_args("  /doctor ", False)
    assert "--append-system-prompt" not in doctor_argv
    assert doctor_argv[-2:] == ["--", "/doctor"]

    review_argv = claude_sess._build_args("/review focus security", False)
    assert "--append-system-prompt" in review_argv
    assert AUTONOMOUS_EXECUTION_PROMPT in review_argv

    plain_argv = claude_sess._build_args("implement feature", False)
    assert "--append-system-prompt" in plain_argv

    # AGY CLI
    agy_sess = SyncAGYCLISession(
        cwd=str(tmp_path),
        model_name="gemini-pro-orchestrator-neon/xhigh/low",
    )
    for cmd in ("/usage", "/credits", "/hooks", "/permissions", "/agents", "/skills", "/help", "/model", "/effort"):
        argv = agy_sess._build_args(cmd, False)
        assert f"--prompt={cmd}" in argv
        assert AUTONOMOUS_EXECUTION_PROMPT not in " ".join(argv)


# ── 3. AGY CLI runtime settings, /plugin rewrite, /mcp, command_result, factory ─


def test_agy_cli_runtime_settings_and_plugin_rewrite(tmp_path) -> None:
    cfg = AGYCLIConfig(command="agy", show_thinking=True, verbosity="low", agent_mode="plan")
    sess = SyncAGYCLISession(
        cwd=str(tmp_path),
        config=cfg,
        model_name="gemini-pro-orchestrator-neon/xhigh/low",
    )
    assert sess.get_runtime_settings() == {
        "show_thinking": True,
        "verbosity": "low",
        "agent_mode": "plan",
    }

    argv1 = sess._build_args("/plugin", False)
    assert "--prompt=/plugin list" in argv1
    assert argv1[argv1.index("--mode") + 1] == "plan"
    settings_json1 = json.loads(argv1[argv1.index("--settings") + 1])
    assert settings_json1["showThinking"] is True
    assert settings_json1["verbosity"] == "low"

    sess.set_runtime_settings({"show_thinking": False, "verbosity": "high", "agent_mode": "accept-edits"})
    assert sess.get_runtime_settings() == {
        "show_thinking": False,
        "verbosity": "high",
        "agent_mode": "accept-edits",
    }
    argv2 = sess._build_args("/plugins", False)
    assert "--prompt=/plugin list" in argv2
    assert argv2[argv2.index("--mode") + 1] == "accept-edits"
    settings_json2 = json.loads(argv2[argv2.index("--settings") + 1])
    assert settings_json2["showThinking"] is False
    assert settings_json2["verbosity"] == "high"

    # Reset optional runtime knobs back to defaults
    sess.set_runtime_settings({"verbosity": "default", "agent_mode": None})
    argv3 = sess._build_args("hello", False)
    assert "--mode" not in argv3
    settings_json3 = json.loads(argv3[argv3.index("--settings") + 1])
    assert "verbosity" not in settings_json3

    # Original AGYCLIConfig passed in was not mutated
    assert cfg.verbosity == "low"
    assert cfg.agent_mode == "plan"
    assert sess.get_available_commands() == list(STATIC_TOOL_NATIVE_COMMANDS["agy"])


def test_agy_cli_mcp_list_subprocess_and_error_handling(monkeypatch, tmp_path) -> None:
    sess = SyncAGYCLISession(
        cwd=str(tmp_path),
        model_name="gemini-pro-orchestrator-neon/xhigh/low",
    )
    monkeypatch.setattr("src.agent_session.agy_cli.shutil.which", lambda _c: "/usr/bin/agy")

    calls: list[tuple[list[str], dict]] = []

    def _fake_run(args, **kwargs):
        calls.append((list(args), dict(kwargs)))
        return SimpleNamespace(returncode=0, stdout="No MCP servers configured.\n", stderr="")

    monkeypatch.setattr("src.agent_session.agy_cli.subprocess.run", _fake_run)
    monkeypatch.setattr(
        "src.agent_session.agy_cli.subprocess.Popen",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("Popen must not run for /mcp")),
    )

    events = []
    res = sess.send_prompt("/mcp", on_event=events.append, timeout=10)
    assert res.stop_reason == "end_turn"
    assert "未配置 MCP" in res.text
    assert calls[-1][0] == ["agy", "mcp", "list"]
    assert "PATH" in calls[-1][1]["env"]
    assert events and "MCP" in (events[0].text or "")

    # /mcp list with configured server output
    monkeypatch.setattr(
        "src.agent_session.agy_cli.subprocess.run",
        lambda *_a, **_kw: SimpleNamespace(returncode=0, stdout="context7  stdio  connected\n", stderr=""),
    )
    res2 = sess.send_prompt("/mcp list")
    assert res2.stop_reason == "end_turn"
    assert "context7" in res2.text

    # Timeout handled gracefully
    monkeypatch.setattr(
        "src.agent_session.agy_cli.subprocess.run",
        lambda *_a, **_kw: (_ for _ in ()).throw(subprocess.TimeoutExpired(cmd="agy", timeout=5)),
    )
    res_to = sess.send_prompt("/mcp")
    assert res_to.stop_reason == "timeout"


@pytest.mark.parametrize(
    ("cmd_name", "data", "response_tsv", "expected_substrings"),
    [
        (
            "usage",
            {
                "description": "Quota details",
                "groups": [
                    {
                        "name": "Gemini Models",
                        "buckets": [
                            {"id": "gemini-weekly", "name": "Weekly Limit Remaining", "window": "weekly", "remaining_fraction": 0.84, "reset_time": "2026-11-30T19:58:01Z"},
                            {"id": "gemini-5h", "name": "Five Hour Limit Remaining", "window": "5h", "remaining_fraction": 1.0, "reset_time": "1970-01-01T00:00:00Z"},
                        ],
                    }
                ],
            },
            "Gemini Models\tWeekly\t84%\n",
            ["Gemini Models", "84%", "未消耗（满额）"],
        ),
        ("credits", {"remaining_credits": 0}, "Remaining credits\t0\n", ["Credits", "`0`"]),
        (
            "config",
            {"config": {"agentMode": "auto", "verbosity": "medium", "showThinking": False, "trustedWorkspaces": ["/secret/path"], "gcp": {"token": "abc"}}},
            "agentMode\tauto\n",
            ["agentMode", "verbosity", "showThinking"],
        ),
        ("hooks", {"hooks": []}, "", ["Hooks", "未配置"]),
        ("permissions", {"permissions": [{"scope": "project"}, {"scope": "global"}]}, "", ["权限配置", "project", "global"]),
        ("agents", {"agents": ["gsd-planner", "gsd-verifier"]}, "", ["gsd-planner", "gsd-verifier"]),
        ("skills", {"skills": [{"name": "bytedcli", "description": "Internal skill", "builtin": False}]}, "", ["bytedcli", "Internal skill"]),
        ("help", {"commands": [{"name": "usage", "aliases": ["quota"], "description": "View quota"}]}, "", ["/usage", "/quota", "View quota"]),
        ("model", {"id": "gemini-pro-orchestrator", "label": "Orchestrator (Carbon)", "is_default": False}, "", ["Orchestrator (Carbon)", "gemini-pro-orchestrator"]),
        ("effort", {"adjustable": False}, "not adjustable\n", ["not adjustable"]),
        ("plugin", {"plugins": [{"name": "swarm", "enabled": True, "description": "Agent swarms"}]}, "", ["swarm", "已启用"]),
    ],
)
def test_agy_cli_command_result_rendering_and_no_secret_leak(
    monkeypatch,
    tmp_path,
    cmd_name: str,
    data: dict,
    response_tsv: str,
    expected_substrings: list[str],
) -> None:
    sess = SyncAGYCLISession(
        cwd=str(tmp_path),
        model_name="gemini-pro-orchestrator-neon/xhigh/low",
    )
    lines = [
        json.dumps({"event": "command_result", "command": {"name": cmd_name, "data": data}}) + "\n",
        json.dumps(
            {
                "event": "result",
                "result": {
                    "conversation_id": "",
                    "status": "SUCCESS",
                    "response": response_tsv,
                    "command": {"name": cmd_name, "data": data},
                },
            }
        )
        + "\n",
    ]
    monkeypatch.setattr("src.agent_session.agy_cli.shutil.which", lambda _c: "/usr/bin/agy")
    monkeypatch.setattr("src.agent_session.agy_cli.subprocess.Popen", lambda *_a, **_kw: _completed_process(lines))

    events = []
    res = sess.send_prompt(f"/{cmd_name}", on_event=events.append)
    assert res.stop_reason == "end_turn"
    for needle in expected_substrings:
        assert needle in res.terminal_text
    # Ensure trustedWorkspaces / secrets never leak from /config
    assert "/secret/path" not in res.terminal_text
    assert "trustedWorkspaces" not in res.terminal_text
    # Ensure command_result + result.command does NOT double-emit
    assert len(events) == 1


def test_agy_cli_result_only_command_fallback(monkeypatch, tmp_path) -> None:
    sess = SyncAGYCLISession(
        cwd=str(tmp_path),
        model_name="gemini-pro-orchestrator-neon/xhigh/low",
    )
    lines = [
        json.dumps(
            {
                "event": "result",
                "result": {
                    "conversation_id": "",
                    "status": "SUCCESS",
                    "response": "",
                    "command": {"name": "hooks", "data": {"hooks": []}},
                },
            }
        )
        + "\n",
    ]
    monkeypatch.setattr("src.agent_session.agy_cli.shutil.which", lambda _c: "/usr/bin/agy")
    monkeypatch.setattr("src.agent_session.agy_cli.subprocess.Popen", lambda *_a, **_kw: _completed_process(lines))
    res = sess.send_prompt("/hooks")
    assert res.stop_reason == "end_turn"
    assert "未配置任何 Hook" in res.text


def test_factory_forwards_runtime_settings_to_supported_sessions(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("src.agent_session.agy_cli.shutil.which", lambda _c: "/usr/bin/agy")
    runtime = {"show_thinking": False, "verbosity": "high", "agent_mode": "plan"}
    session = create_programming_session(
        "agy",
        str(tmp_path),
        model_name="gemini-pro-orchestrator-neon/xhigh/low",
        runtime_settings=runtime,
    )
    assert isinstance(session, SyncAGYCLISession)
    assert session.get_runtime_settings() == runtime

    # Also verifies _start_base_session applies set_runtime_settings for ACP sessions that expose it
    fake_acp = MagicMock()
    with patch("src.acp.sync_adapter.start_session_with_retry", return_value=fake_acp):
        started = _start_base_session(
            "codex",
            str(tmp_path),
            None,
            runtime_settings={"show_thinking": False},
        )
    assert started is fake_acp
    fake_acp.set_runtime_settings.assert_called_once_with({"show_thinking": False})


# ── 4. tool_telemetry snapshots ─────────────────────────────────────────────


def test_collect_tool_usage_snapshot_agy_and_non_agy(monkeypatch, tmp_path) -> None:
    usage_payload = {
        "description": "Weekly & 5-hour quota info",
        "groups": [
            {
                "name": "Gemini Models",
                "description": "Gemini 4",
                "buckets": [
                    {
                        "id": "gemini-weekly",
                        "name": "Weekly Limit Remaining",
                        "window": "weekly",
                        "remaining_fraction": 0.84,
                        "reset_time": "2026-11-30T19:58:01Z",
                    },
                    {
                        "id": "gemini-5h",
                        "name": "Five Hour Limit Remaining",
                        "window": "5h",
                        "remaining_fraction": 0.97,
                        "reset_time": "2026-11-25T12:54:35Z",
                    },
                ],
            },
            {
                "name": "Claude and GPT models",
                "buckets": [
                    {
                        "id": "3p-5h",
                        "name": "Five Hour Limit Remaining",
                        "window": "5h",
                        "remaining_fraction": 1.0,
                        "reset_time": "1970-01-01T00:00:00Z",
                    }
                ],
            },
        ],
    }
    credits_payload = {"remaining_credits": 42}

    def _fake_run(args, **_kwargs):
        cmd_str = " ".join(args)
        if "--prompt=/usage" in cmd_str:
            out = json.dumps({"event": "command_result", "command": {"name": "usage", "data": usage_payload}}) + "\n"
            return SimpleNamespace(returncode=0, stdout=out, stderr="")
        if "--prompt=/credits" in cmd_str:
            out = json.dumps({"event": "result", "result": {"status": "SUCCESS", "command": {"name": "credits", "data": credits_payload}}}) + "\n"
            return SimpleNamespace(returncode=0, stdout=out, stderr="")
        raise AssertionError(f"unexpected probe command: {args}")

    monkeypatch.setattr("src.acp.tool_telemetry.shutil.which", lambda cmd: "/usr/bin/agy" if cmd == "agy" else None)
    monkeypatch.setattr("src.acp.tool_telemetry.subprocess.run", _fake_run)

    fake_session = SimpleNamespace(
        session_id="sess-agy-1",
        message_count=5,
        created_at=time.time() - 90,
        _model_name="gemini-pro-orchestrator-neon/high/low",
        get_usage_info=lambda: ACPUsageInfo(used=12000, size=200000, cost_amount=0.25, cost_currency="USD"),
    )

    snap = collect_tool_usage_snapshot("agy", cwd=str(tmp_path), session=fake_session, timeout_s=3.0)
    assert isinstance(snap, ToolUsageSnapshot)
    assert snap.tool_name == "agy"
    assert snap.session_active is True
    assert snap.session_id == "sess-agy-1"
    assert snap.message_count == 5
    assert snap.duration_seconds is not None and snap.duration_seconds >= 89
    assert snap.context_used == 12000 and snap.context_size == 200000
    assert snap.cost_amount == 0.25 and snap.cost_currency == "USD"
    assert snap.quota_supported is True
    assert snap.credits_known is True and snap.remaining_credits == 42
    assert len(snap.quota_groups) == 2
    assert isinstance(snap.quota_groups[0], QuotaGroupSnapshot)
    assert isinstance(snap.quota_groups[0].buckets[0], QuotaBucketSnapshot)
    assert snap.quota_groups[0].buckets[0].remaining_fraction == pytest.approx(0.84)

    # Non-agy tool handles missing remote quota cleanly
    codex_snap = collect_tool_usage_snapshot("codex", cwd=str(tmp_path), model_name="gpt-5", session=fake_session)
    assert codex_snap.tool_name == "codex"
    assert codex_snap.quota_supported is False
    assert codex_snap.context_used == 12000
    assert codex_snap.capability_note

    # Missing binary or subprocess error never raises
    monkeypatch.setattr("src.acp.tool_telemetry.shutil.which", lambda _cmd: None)
    missing_snap = collect_tool_usage_snapshot("agy", cwd=str(tmp_path))
    assert missing_snap.quota_supported is False
    assert missing_snap.error

    monkeypatch.setattr("src.acp.tool_telemetry.shutil.which", lambda _cmd: "/usr/bin/agy")
    monkeypatch.setattr(
        "src.acp.tool_telemetry.subprocess.run",
        lambda *_a, **_kw: (_ for _ in ()).throw(subprocess.TimeoutExpired("agy", 3.0)),
    )
    timeout_snap = collect_tool_usage_snapshot("agy", cwd=str(tmp_path), timeout_s=3.0)
    assert timeout_snap.quota_supported is False
    assert "超时" in timeout_snap.error


def test_collect_tool_settings_snapshot_sanitizes_and_inspects_acp(monkeypatch, tmp_path) -> None:
    raw_config = {
        "agentMode": "accept-edits",
        "model": "Orchestrator (Carbon)",
        "orchestrator": {"planner": "Gemini 4 Carbon (Medium)", "executor": "Gemini 4 Hydrogen (XHigh)"},
        "showThinking": False,
        "verbosity": "medium",
        "toolPermission": "request-review",
        "trustedWorkspaces": ["/home/secret/repo"],
        "gcp": {"project": "secret", "token": "sk-secret-token-12345"},
        "sandboxProxy": "http://127.0.0.1:8080",
        "apiKey": "sk-1234567890abcdef",
    }
    monkeypatch.setattr("src.acp.tool_telemetry.shutil.which", lambda _c: "/usr/bin/agy")
    monkeypatch.setattr(
        "src.acp.tool_telemetry.subprocess.run",
        lambda *_a, **_kw: SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"event": "command_result", "command": {"name": "config", "data": {"config": raw_config}}}) + "\n",
            stderr="",
        ),
    )

    fake_acp_session = SimpleNamespace(
        config_options=[
            SimpleNamespace(
                root=SimpleNamespace(
                    id="thought_level",
                    name="Thought Level",
                    category="reasoning",
                    current_value="high",
                    options=[SimpleNamespace(value="low"), SimpleNamespace(value="high")],
                )
            )
        ]
    )
    fake_session = SimpleNamespace(
        session_id="sess-1",
        _model_name="gemini-4-neon/high",
        _acp_session=fake_acp_session,
    )

    snap = collect_tool_settings_snapshot(
        "agy",
        cwd=str(tmp_path),
        session=fake_session,
        saved_runtime_settings={"show_thinking": False, "verbosity": "high"},
    )
    assert isinstance(snap, ToolSettingsSnapshot)
    assert snap.runtime_settings == {
        "show_thinking": False,
        "verbosity": "high",
        "agent_mode": "auto",  # default filled for agy
    }
    assert len(snap.acp_config_options) == 1
    assert snap.acp_config_options[0]["id"] == "thought_level"
    assert snap.acp_config_options[0]["options"] == ["low", "high"]
    assert snap.backend_config_summary["agentMode"] == "accept-edits"
    assert snap.backend_config_summary["toolPermission"] == "request-review"
    # No path/token leakage
    summary_dump = json.dumps(snap.backend_config_summary, ensure_ascii=False)
    assert "trustedWorkspaces" not in summary_dump
    assert "/home/secret" not in summary_dump
    assert "sk-secret" not in summary_dump
    assert "sandboxProxy" not in summary_dump
    assert "gcp" not in summary_dump

    # Non-agy defaults cleanly without invoking agy CLI
    codex_snap = collect_tool_settings_snapshot("codex", cwd=str(tmp_path), saved_runtime_settings=None)
    assert codex_snap.runtime_settings == {}
    assert codex_snap.backend_config_summary == {}
    assert codex_snap.capability_note
