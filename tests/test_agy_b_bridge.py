"""Regression and integration tests for the hidden ``/agy-b`` backend and Feishu panel isolation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.acp.helper import list_acp_tools
from src.acp.manager import ACPSessionManager
from src.acp.native_commands import (
    DEFAULT_TOOL_PRIORITY,
    STATIC_TOOL_NATIVE_COMMANDS,
    get_tui_only_command_hint,
)
from src.acp.options import ACPModelOption, ACPModelSelectionVariant
from src.acp.providers import get_providers
from src.acp.startup_utils import AcpRetryStarter, StartupBackend, select_startup_backend
from src.acp.sync_adapter import SyncACPSession
from src.acp.tool_telemetry import (
    collect_tool_settings_snapshot,
    collect_tool_usage_snapshot,
)
from src.agent.intent_recognizer import IntentRecognizer, IntentType
from src.agent_session.agy_cli import AGYCLIConfig, SyncAGYCLISession
from src.agent_session.factory import create_engine_session
from src.agent_session.wrappers import ModelFailureAwareSession
from src.card.builders.system import SystemBuilder
from src.card.builders.tool_cockpit import ToolCockpitBuilder
from src.card.builders.workflow import _extract_current_base_model
from src.config.settings import Settings
from src.engine_base import BaseEngineManager
from src.feishu.handler_context import HandlerContext
from src.feishu.handlers import AGYBModeHandler
from src.feishu.handlers.system import SystemHandler
from src.feishu.main_slash_commands import MAIN_AGENT_COMMANDS
from src.feishu.product_catalog import (
    HIDDEN_SLASH_COMMANDS,
    get_slash_discoverable_actions,
    is_programming_entry_command,
    resolve_command,
)
from src.feishu.slash_command_parser import SlashCommandParser
from src.feishu.ws_client import FeishuWSClient
from src.mode import PROGRAMMING_MODE_VALUES
from src.mode.backends import VISIBLE_BACKEND_SPECS
from src.mode.manager import InteractionMode, ModeManager
from src.project.context import ProjectContext
from src.project.unified_context import ContextSourceMode, ProjectContextManager
from src.utils.engine_identity import resolve_engine_identity
from src.workflow_engine.agent_pool import resolve_effective_model_name
from src.workflow_engine.tool_registry import (
    EFFORT_CAPABLE_TOOLS,
    ToolCapability,
    validate_agent_binding_spec,
)


def _build_ws_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[FeishuWSClient, list[tuple[str, str, str, str]]]:
    fake_home = tmp_path / "home"
    fake_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("APP_SECRET", "test_secret_agy_b")
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        app_id="cli_test_agy_b",
        app_secret="test_secret_agy_b",
        default_working_dir=str(repo_dir),
        projects_dir=str(tmp_path / "projects"),
        projects_config_file=str(fake_home / ".ghostap" / "projects.json"),
    )
    deliveries: list[tuple[str, str, str, str]] = []

    def _run_submitted(_spec, callback):
        callback(SimpleNamespace(run_id="run_agy_b_1"))
        return SimpleNamespace(run_id="run_agy_b_1")

    scheduler = MagicMock()
    scheduler._restart_gate = MagicMock()
    scheduler.list_tasks = MagicMock(return_value=[])
    scheduler.get_state = MagicMock(return_value=None)
    scheduler.submit = MagicMock(side_effect=_run_submitted)

    with (
        patch("src.feishu.ws_client.get_settings", return_value=settings),
        patch("src.config.get_settings", return_value=settings),
        patch("src.feishu.ws_client._build_task_scheduler", return_value=scheduler),
    ):
        client = FeishuWSClient(lambda *_a: None)

    fake_im = SimpleNamespace(
        reply_message=lambda mid, content, msg_type="text", **_kw: (
            deliveries.append(("reply", mid, msg_type, content))
            or SimpleNamespace(
                success=lambda: True,
                code=0,
                msg="ok",
                data=SimpleNamespace(message_id=f"om_reply_{len(deliveries)}"),
            )
        ),
        send_message=lambda _rt, rid, content, msg_type="text", **_kw: (
            deliveries.append(("send", rid, msg_type, content))
            or SimpleNamespace(
                success=lambda: True,
                code=0,
                msg="ok",
                data=SimpleNamespace(message_id=f"om_send_{len(deliveries)}"),
            )
        ),
        patch_message=lambda mid, content, **_kw: (
            deliveries.append(("patch", mid, "interactive", content))
            or SimpleNamespace(success=lambda: True, code=0, msg="ok")
        ),
        add_reaction=lambda *_a, **_kw: SimpleNamespace(
            success=lambda: True, data=SimpleNamespace(reaction_id="r1")
        ),
        delete_reaction=lambda *_a, **_kw: SimpleNamespace(success=lambda: True),
    )

    client._handler_ctx._im_client = fake_im
    for handler in client._handler_ctx.handlers.values():
        handler._im_client = fake_im
        if hasattr(handler, "im_client"):
            try:
                handler.im_client = fake_im
            except AttributeError:
                pass
    return client, deliveries


# ── F8: SSOT, ModeManager, Handler & Identity ──────────────────────


def test_mode_manager_and_handler_register_isolated_agy_b() -> None:
    assert InteractionMode.AGY_B.value == "agy_b"
    assert "agy_b" in PROGRAMMING_MODE_VALUES

    mm = ModeManager()
    mm.enter_agy_b_mode("chat-b")
    assert mm.is_agy_b_mode("chat-b")
    assert not mm.is_agy_mode("chat-b")

    mm.enter_agy_mode("chat-a")
    assert mm.is_agy_mode("chat-a")
    assert not mm.is_agy_b_mode("chat-a")

    assert AGYBModeHandler.mode_key == "agy_b"
    assert AGYBModeHandler.interaction_mode == InteractionMode.AGY_B
    assert AGYBModeHandler.context_source == ContextSourceMode.AGY_B
    handler = AGYBModeHandler.__new__(AGYBModeHandler)
    assert handler._uses_claude_cli() is False


def test_engine_identity_resolves_agy_b_without_agy_prefix_collision() -> None:
    identity = resolve_engine_identity(
        mode=InteractionMode.AGY_B,
        acp_tool_name="agy-b",
        acp_model_name="gemini-4-hydrogen/xhigh",
    )
    assert (identity.engine_name, identity.agent_type, identity.model_name, identity.transport) == (
        "AGY-B",
        "agy_b",
        "gemini-4-hydrogen/xhigh",
        "acp",
    )

    mgr_stub = object.__new__(BaseEngineManager)
    assert mgr_stub._resolve_identity("AGY-B")[1] == "agy_b"
    assert mgr_stub._resolve_identity("agy-b")[1] == "agy_b"
    assert mgr_stub._resolve_identity("agy")[1] == "agy"


# ── F11 / R3: Feishu Slash Panel Exclusion & @bot Resolution ───────


def test_agy_b_command_resolves_but_is_excluded_from_feishu_slash_panel() -> None:
    resolved = resolve_command("/agy-b")
    assert resolved is not None
    assert resolved.action.programming_mode_id == "agy_b"
    assert resolved.action.slash_discoverable is False

    assert resolve_command("/enter_agy_b") is not None
    assert resolve_command("/exit_agy_b") is not None
    assert resolve_command("/end_agy_b") is not None
    assert "/agy-b" in HIDDEN_SLASH_COMMANDS

    discoverable = {action.command for action in get_slash_discoverable_actions()}
    assert "/agy-b" not in discoverable
    assert "/enter_agy_b" not in discoverable
    assert len(discoverable) == 23

    panel = {command.name for command in MAIN_AGENT_COMMANDS}
    assert "/agy-b" not in panel
    assert len(panel) == 23

    assert len(VISIBLE_BACKEND_SPECS) == 9
    assert all(spec.mode_key != "agy_b" for spec in VISIBLE_BACKEND_SPECS)


def test_slash_parser_and_intent_recognizer_handle_agy_b_and_at_bot_suffix() -> None:
    for raw, expected_args in (
        ("/agy-b", ""),
        ("/agy-b fix bug", "fix bug"),
        ("/agy-b@ghostap_bot", ""),
        ("/agy-b@ghostap_bot fix bug", "fix bug"),
    ):
        parsed = SlashCommandParser.parse(raw)
        assert parsed is not None, raw
        assert parsed.command == "/agy-b"
        assert parsed.args == expected_args
        assert parsed.has_args is bool(expected_args)
        assert SystemHandler.is_interceptable_command_match(parsed) is True

    # Unknown native slash with '@' must not be stripped by catalog fallback
    assert resolve_command("/pkg@1.0.0") is None
    assert is_programming_entry_command("/agy-b@ghostap_bot") is True
    assert SystemHandler.is_fast_lane_system_command("/agy-b") is True
    assert SystemHandler.is_fast_lane_system_command("/agy-b@ghostap_bot") is True
    assert SystemHandler.is_fast_lane_system_command("/agy-b fix bug") is False
    assert SystemHandler.is_fast_lane_system_command("/agy-b@ghostap_bot fix bug") is False
    assert SystemHandler.is_exit_command("/exit_agy_b") is True
    assert SystemHandler.is_exit_command("/end_agy_b") is True
    assert SystemHandler.is_exit_command("/agy-b") is False

    recognizer = IntentRecognizer()
    for raw in ("/agy-b", "/agy-b@ghostap_bot", "/enter_agy_b"):
        enter = recognizer._quick_match(raw)
        assert enter is not None, raw
        assert enter.primary_intent == IntentType.ENTER_AGY_B

    for raw in ("/exit_agy_b", "/end_agy_b"):
        ex = recognizer._quick_match(raw)
        assert ex is not None, raw
        assert ex.primary_intent == IntentType.EXIT_MODE

    assert recognizer._get_fallback_intent("agy_b") == IntentType.AGY_B_MESSAGE


def test_tools_list_and_tools_status_exclude_agy_b(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, deliveries = _build_ws_client(tmp_path, monkeypatch)
    system = client._handler_ctx.handlers["system"]
    project = client._project_manager.get_active_project("chat-1")

    monkeypatch.setattr(
        "src.feishu.handlers.system.is_programming_tool_available",
        lambda tool, **_kw: tool in {"agy", "codex"},
    )
    system.show_tools_list("om-tools", "chat-1", project)
    assert deliveries
    tools_payload = deliveries[-1][3]
    assert "AGY-B" not in tools_payload
    assert "/agy-b" not in tools_payload

    deliveries.clear()
    system.show_tools_status("om-status", "chat-1", project)
    assert deliveries
    status_payload = deliveries[-1][3]
    assert "AGY-B" not in status_payload
    assert "/agy-b" not in status_payload


def test_tools_list_and_tools_status_mask_active_or_saved_agy_b_while_cockpit_resolves_via_project_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, deliveries = _build_ws_client(tmp_path, monkeypatch)
    ctx = client._handler_ctx
    system = ctx.handlers["system"]
    ok_proj, _msg, project = client._project_manager.create_project(
        "proj-b", "demo", str(tmp_path / "repo"), chat_id="chat-1"
    )
    assert ok_proj and project is not None

    monkeypatch.setattr(
        "src.feishu.handlers.system.is_programming_tool_available",
        lambda tool, **_kw: tool in {"agy", "agy_b", "codex"},
    )
    monkeypatch.setattr(
        "src.acp.tool_telemetry.shutil.which",
        lambda cmd: f"/usr/local/bin/{cmd}" if cmd in {"agy", "agy-b", "codex"} else None,
    )

    def _collect_button_values(card_json: str) -> list[dict]:
        found: list[dict] = []

        def _walk(node: object) -> None:
            if isinstance(node, dict):
                if node.get("tag") == "button" and isinstance(node.get("value"), dict):
                    found.append(node["value"])
                    return
                for v in node.values():
                    _walk(v)
            elif isinstance(node, list):
                for item in node:
                    _walk(item)

        _walk(json.loads(card_json))
        return found

    for scenario in ("saved", "active"):
        project.set_acp_configuration("agy_b", "gemini-4-hydrogen/high", chat_id="chat-1")
        if scenario == "active":
            project.set_programming_mode("agy_b", True, session_id="sess-b", query_count=2, chat_id="chat-1")
            ctx.mode_manager.enter_agy_b_mode("chat-1", project_id=project.project_id)
        else:
            project.set_programming_mode("agy_b", False, chat_id="chat-1")
            ctx.mode_manager.exit_to_smart("chat-1", project_id=project.project_id)

        deliveries.clear()
        system.show_tools_list("om-tools-hidden", "chat-1", project)
        assert deliveries
        tools_payload = deliveries[-1][3]
        assert "AGY-B" not in tools_payload
        assert "agy_b" not in tools_payload
        assert "/agy-b" not in tools_payload

        tools_btns = _collect_button_values(tools_payload)
        settings_btn = next(b for b in tools_btns if b.get("action") == "open_tool_settings")
        assert "tool_name" not in settings_btn
        assert settings_btn.get("project_id") == project.project_id

        deliveries.clear()
        system.show_tools_status("om-status-hidden", "chat-1", project)
        assert deliveries
        status_payload = deliveries[-1][3]
        assert "AGY-B" not in status_payload
        assert "agy_b" not in status_payload
        assert "/agy-b" not in status_payload

        status_btns = _collect_button_values(status_payload)
        status_settings_btn = next(b for b in status_btns if b.get("action") == "open_tool_settings")
        assert "tool_name" not in status_settings_btn
        assert status_settings_btn.get("project_id") == project.project_id

        # Clicking the bottom cockpit button (project_id only) still resolves agy_b
        with patch.object(system, "show_tool_settings") as settings_spy:
            system.handle_tool_cockpit_action("om-cb", "chat-1", project.project_id, settings_btn)
        settings_spy.assert_called_once()
        assert settings_spy.call_args.args[2] == "agy_b"
        assert settings_spy.call_args.args[3] is project


def test_manual_agy_b_dispatch_routes_to_agy_b_without_touching_agy_manager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _deliveries = _build_ws_client(tmp_path, monkeypatch)
    ctx = client._handler_ctx
    assert isinstance(ctx, HandlerContext)
    assert ctx.agy_manager is not ctx.agy_b_manager
    assert ctx.managers["agy"] is ctx.agy_manager
    assert ctx.managers["agy_b"] is ctx.agy_b_manager

    system = ctx.handlers["system"]
    ok_proj, _msg, project = client._project_manager.create_project("proj-b", "demo", str(tmp_path / "repo"), chat_id="chat-1")
    assert ok_proj and project is not None

    system.show_explicit_acp_model_selection = MagicMock()
    for raw in ("/agy-b", "/agy-b@ghostap_bot"):
        match = SlashCommandParser.parse(raw)
        assert match is not None
        system.handle_intercepted_command("om-1", "chat-1", raw, project, command_match=match)
    assert system.show_explicit_acp_model_selection.call_count == 2
    for call in system.show_explicit_acp_model_selection.call_args_list:
        assert call.args[2] == "agy_b"

    agy_enter_spy = MagicMock()
    agy_b_enter_spy = MagicMock()
    ctx.handlers["agy"].enter_mode = agy_enter_spy
    ctx.handlers["agy_b"].enter_mode = agy_b_enter_spy
    ctx.handlers["agy_b"].handle_message = MagicMock()

    with patch.object(system, "_handle_direct_mode_enter_with_args", return_value=True) as direct_spy:
        match_prompt = SlashCommandParser.parse("/agy-b fix bug")
        assert match_prompt is not None
        system.handle_intercepted_command("om-2", "chat-1", "/agy-b fix bug", project, command_match=match_prompt)
    direct_spy.assert_called_once()
    assert direct_spy.call_args.args[2] == "agy_b"
    assert direct_spy.call_args.args[3] == "fix bug"
    agy_enter_spy.assert_not_called()


# ── F8: ProjectContext & UnifiedContext Isolation ───────────────────


def test_project_context_and_unified_context_isolate_agy_from_agy_b() -> None:
    proj = ProjectContext(project_id="p1", project_name="demo", root_path="/tmp/demo")
    proj.set_programming_mode("agy", True, session_id="sess-agy", query_count=2)
    proj.update_programming_snapshot("agy", query="agy task", query_count=2, session_id="sess-agy")
    proj.set_tool_runtime_settings("chat-1", "agy", {"show_thinking": True, "verbosity": "low"})

    proj.set_programming_mode("agy_b", True, session_id="sess-agy-b", query_count=5)
    proj.update_programming_snapshot("agy_b", query="agy-b task", query_count=5, session_id="sess-agy-b")
    proj.set_tool_runtime_settings("chat-1", "agy_b", {"show_thinking": False, "verbosity": "high"})
    proj.acp_tool_name = "agy_b"
    proj.acp_model_name = "gemini-4-hydrogen/xhigh"

    restored = ProjectContext.from_snapshot(proj.to_snapshot())
    assert restored.acp_tool_name == "agy_b"
    assert restored.acp_model_name == "gemini-4-hydrogen/xhigh"
    assert restored.agy_mode is True
    assert restored.agy_b_mode is True
    assert restored.agy_session_snapshot.session_id == "sess-agy"
    assert restored.agy_b_session_snapshot.session_id == "sess-agy-b"
    assert restored.agy_b_session_snapshot.last_query == "agy-b task"
    assert restored.get_tool_runtime_settings("chat-1", "agy")["verbosity"] == "low"
    assert restored.get_tool_runtime_settings("chat-1", "agy_b")["verbosity"] == "high"

    # Mode transition records AGY -> AGY_B in UnifiedContext
    ctx_mgr = ProjectContextManager()
    handler = AGYBModeHandler.__new__(AGYBModeHandler)
    handler.ctx = SimpleNamespace(context_manager=ctx_mgr)
    handler.record_mode_transition(
        "p1",
        InteractionMode.AGY,
        InteractionMode.AGY_B,
        "manual switch",
        chat_id="chat-1",
    )
    uc = ctx_mgr.store.get("p1", chat_id="chat-1")
    assert uc is not None and uc.entries
    last_entry = uc.entries[-1]
    assert last_entry.metadata.get("from_mode") == ContextSourceMode.AGY.value
    assert last_entry.metadata.get("to_mode") == ContextSourceMode.AGY_B.value


# ── F9: GenericACPProvider & ACP Protocol Parity ───────────────────


def test_agy_b_provider_spawns_agy_b_binary_and_degrades_gracefully() -> None:
    providers = get_providers()
    assert "agy_b" in providers
    provider = providers["agy_b"]
    assert provider.get_serve_command() == ("agy-b", ["acp"])

    listed = {str(getattr(t, "name", t)).strip().lower() for t in list_acp_tools()}
    assert "agy_b" not in listed

    # Cross-platform graceful degradation when agy-b is absent from PATH
    import src.acp.providers as providers_mod

    clear_cache = providers_mod._checkers["agy_b"][2]
    clear_cache()
    try:
        with patch("src.acp.providers.subprocess.run", side_effect=FileNotFoundError("agy-b")):
            assert provider.check_availability() is False
    finally:
        clear_cache()


def test_agy_b_acp_protocol_and_manager_transport_switch(tmp_path: Path) -> None:
    acp_session = SyncACPSession("agy_b", "agy-b", ["acp"], str(tmp_path))
    assert acp_session._uses_agy_acp() is True
    assert acp_session.supports_background_tool_continuation is True

    mgr = ACPSessionManager("agy_b")
    cli_live = SyncAGYCLISession(
        cwd=str(tmp_path),
        config=AGYCLIConfig(command="agy-b"),
        model_name="gemini-pro-orchestrator-neon/xhigh/low",
        agent_type="agy_b",
    )
    cli_live.session_id = "conv-b-1"
    acp_new = MagicMock(session_id="acp-b-new", _model_name="gemini-4-neon/xhigh")
    mgr._sessions[mgr._session_key("chat-1", "proj-b")] = cli_live  # type: ignore[assignment]
    with patch.object(mgr, "_create_session", return_value=acp_new) as create_mock:
        returned = mgr.ensure_session(
            "chat-1",
            cwd=str(tmp_path),
            project_id="proj-b",
            model_name="gemini-4-neon/xhigh",
        )
    assert returned is acp_new
    create_mock.assert_called_once()


# ── F10: AGY CLI, Failover, Native Commands, Telemetry, Cards & Workflow ─


def test_agy_b_cli_factory_retry_and_failover(tmp_path: Path) -> None:
    assert select_startup_backend("agy_b") == StartupBackend.ACP
    assert select_startup_backend("agy_b", model_name="gemini-pro-orchestrator-neon/xhigh/low") == StartupBackend.CLI

    session = SyncAGYCLISession(
        cwd=str(tmp_path),
        config=AGYCLIConfig(command="agy-b"),
        model_name="gemini-pro-orchestrator-neon/xhigh/low",
    )
    assert session._agent_type == "agy_b"
    assert session._display_name == "AGY-B"
    assert session.to_snapshot()["agent_type"] == "agy_b"
    assert session.get_available_commands() == list(STATIC_TOOL_NATIVE_COMMANDS["agy_b"])

    with patch("src.agent_session.agy_cli.SyncAGYCLISession") as cli_cls:
        AcpRetryStarter._build_session(
            backend=StartupBackend.CLI,
            agent_type="agy_b",
            cwd=str(tmp_path),
            model_name="gemini-pro-orchestrator-neon/xhigh/low",
            acp_session_cls=MagicMock(),
            cli_session_cls=MagicMock(),
        )
    cli_cls.assert_called_once()
    assert cli_cls.call_args.kwargs["config"].command == "agy-b"
    assert cli_cls.call_args.kwargs["agent_type"] == "agy_b"

    settings = MagicMock(rate_limit_retry_enabled=False, acp_startup_timeout=20)
    with (
        patch("src.agent_session.factory.get_settings", return_value=settings),
        patch("src.agent_session.factory.SyncAGYCLISession") as factory_cli_cls,
    ):
        create_engine_session(
            "agy_b",
            str(tmp_path),
            model_name="gemini-pro-orchestrator-neon/xhigh/low",
        )
    factory_cli_cls.assert_called_once()
    assert factory_cli_cls.call_args.kwargs["config"].command == "agy-b"
    assert factory_cli_cls.call_args.kwargs["agent_type"] == "agy_b"

    # Failover CLI -> ACP must resolve "agy_b", never "agy"
    cli_base = SyncAGYCLISession(
        cwd=str(tmp_path),
        config=AGYCLIConfig(command="agy-b"),
        model_name="gemini-pro-orchestrator-neon/xhigh/low",
        agent_type="agy_b",
    )
    wrapper = ModelFailureAwareSession(cli_base)
    fake_acp_candidate = MagicMock(session_id="acp-failover-b")
    with (
        patch("src.agent_session.wrappers.resolve_agent_spec", return_value=("agy-b", ["acp"])) as resolve_spy,
        patch.object(wrapper, "_build_sync_acp_candidate", return_value=fake_acp_candidate) as build_acp_spy,
    ):
        ok = wrapper._do_failover(
            from_model="gemini-pro-orchestrator-neon/xhigh/low",
            to_model="gemini-4-neon/xhigh",
        )
    assert ok is True
    resolve_spy.assert_called_once_with("agy_b", model_name="gemini-4-neon/xhigh")
    assert build_acp_spy.call_args.kwargs["agent_type"] == "agy_b"
    assert build_acp_spy.call_args.kwargs["agent_cmd"] == "agy-b"


def test_agy_b_telemetry_cockpit_cards_and_workflow_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert "agy_b" in DEFAULT_TOOL_PRIORITY
    assert "agy_b" in STATIC_TOOL_NATIVE_COMMANDS
    hint = get_tui_only_command_hint("/context", tool_name="agy_b")
    assert hint is not None and "AGY" in hint

    # Missing binary degrades gracefully referencing agy-b
    monkeypatch.setattr("src.acp.tool_telemetry.shutil.which", lambda _cmd: None)
    usage_missing = collect_tool_usage_snapshot("agy_b", cwd=str(tmp_path))
    assert usage_missing.quota_supported is False
    assert "agy-b" in usage_missing.error
    settings_missing = collect_tool_settings_snapshot("agy_b", cwd=str(tmp_path))
    assert "agy-b" in settings_missing.capability_note

    # Available binary probes "agy-b" (never "agy")
    monkeypatch.setattr("src.acp.tool_telemetry.shutil.which", lambda cmd: f"/usr/local/bin/{cmd}")
    probe_commands: list[str] = []

    def _fake_probe(prompt_cmd: str, *, cwd: str, timeout_s: float, command: str = "agy"):
        probe_commands.append(command)
        return {}, "", ""

    monkeypatch.setattr("src.acp.tool_telemetry._run_agy_slash_probe", _fake_probe)
    collect_tool_usage_snapshot("agy_b", cwd=str(tmp_path))
    collect_tool_settings_snapshot("agy_b", cwd=str(tmp_path))
    assert probe_commands and set(probe_commands) == {"agy-b"}

    # Cockpit card renders AGY runtime controls for agy_b
    _, card_json = ToolCockpitBuilder.build_tool_settings_card(
        tool_name="agy_b",
        available_tools=("coco", "agy", "agy_b"),
        models=[
            ACPModelOption(
                name="gemini-pro-orchestrator-neon",
                selection_variants=(
                    ACPModelSelectionVariant(
                        name="gemini-pro-orchestrator-neon/xhigh/low",
                        model="gemini-pro-orchestrator-neon",
                        profile="xhigh",
                        effort="low",
                    ),
                ),
            )
        ],
        current_model="gemini-pro-orchestrator-neon/xhigh/low",
        settings_snapshot=SimpleNamespace(
            runtime_settings={"show_thinking": True, "verbosity": "medium", "agent_mode": "auto"},
            acp_config_options=(),
            backend_config_summary={},
        ),
        project_id="p1",
    )
    card_obj = json.loads(card_json)
    card_str = json.dumps(card_obj, ensure_ascii=False)
    assert "Planner" in card_str
    assert "show_thinking" in card_str

    # SystemBuilder model selection card renders Planner effort for agy_b
    _, sel_json = SystemBuilder.build_acp_model_cascade_card(
        [
            ACPModelOption(
                name="gemini-pro-orchestrator-neon",
                selection_variants=(
                    ACPModelSelectionVariant(
                        name="gemini-pro-orchestrator-neon/xhigh/low",
                        model="gemini-pro-orchestrator-neon",
                        profile="xhigh",
                        effort="low",
                    ),
                ),
            )
        ],
        "agy_b",
        current_model="gemini-pro-orchestrator-neon/xhigh/low",
        project_id="p1",
    )
    assert "Planner" in sel_json

    # Cockpit /usage agy_b & /settings agy_b query agy_b_manager only
    client, deliveries = _build_ws_client(tmp_path, monkeypatch)
    ctx = client._handler_ctx
    ctx.agy_manager.get_session = MagicMock(side_effect=AssertionError("must not query agy_manager"))
    ctx.agy_b_manager.get_session = MagicMock(return_value=None)
    system = ctx.handlers["system"]
    system.show_tool_usage("om-u", "chat-1", "/usage agy-b")
    assert deliveries
    system.show_tool_settings("om-s", "chat-1", "/settings agy_b")
    assert len(deliveries) >= 2

    # Workflow capabilities & validation
    assert "agy_b" in EFFORT_CAPABLE_TOOLS
    cap = ToolCapability(
        tool_name="agy_b",
        display_name="AGY-B",
        available=True,
        supports_model=True,
        supports_effort=True,
        models=("gemini-4-hydrogen", "gemini-pro-orchestrator-neon"),
        efforts_by_model={"gemini-4-hydrogen": ("low", "high"), "gemini-pro-orchestrator-neon": ("low", "high")},
        profiles_by_model={"gemini-pro-orchestrator-neon": ("low", "high")},
        authoritative_models=True,
    )
    assert validate_agent_binding_spec(
        "agy_b",
        model_name="gemini-pro-orchestrator-neon/high/low",
        capabilities={"agy_b": cap},
    ) == (True, None)
    assert _extract_current_base_model(cap, "agy_b", "gemini-pro-orchestrator-neon/high/low") == "gemini-pro-orchestrator-neon"
    assert resolve_effective_model_name("agy_b", "gemini-4-hydrogen", profile=None, effort="high") == "gemini-4-hydrogen/high"


def test_wf_and_wf_config_exclude_agy_b_from_auto_presets_while_cold_cache_discovery_allows_explicit_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.feishu.handlers.workflow_config import WorkflowConfigMixin
    from src.workflow_engine.tool_registry import (
        get_available_tools,
        get_tool_capabilities,
        invalidate_cache,
    )

    invalidate_cache()
    try:
        # Cold cache: get_availability(..., allow_sync_probe=False) returns False,
        # while only "agy-b" (and optionally 1 visible tool "codex") is on PATH.
        with (
            patch(
                "src.acp.providers.tool_registry.get_availability",
                return_value=False,
            ),
            patch(
                "src.workflow_engine.tool_registry.shutil.which",
                side_effect=lambda cmd: f"/usr/local/bin/{cmd}" if cmd in {"agy-b", "codex"} else None,
            ),
            patch(
                "src.acp.helper.get_cached_acp_models",
                return_value=[],
            ),
        ):
            avail = get_available_tools(force_refresh=True, require_available=True)
            assert "agy_b" in avail
            assert avail["agy_b"] == "Google Antigravity 原生 ACP 编程 (AGY-B)"
            assert validate_agent_binding_spec("agy_b", available_tools=avail) == (True, None)

            caps = get_tool_capabilities(force_refresh=True)
            assert caps["agy_b"].available is True

            # Case 1: < 3 visible tools available (only codex + agy_b) -> preset auto-pick excludes agy_b
            members_with_one_visible = WorkflowConfigMixin._default_members_for_preset(
                caps,
                preset_id="collaborative",
            )
            assert [m.tool_name for m in members_with_one_visible] == ["codex"]
            assert all(m.tool_name != "agy_b" for m in members_with_one_visible)

        # Case 2: 0 visible tools available (only agy-b on PATH)
        with (
            patch(
                "src.acp.providers.tool_registry.get_availability",
                return_value=False,
            ),
            patch(
                "src.workflow_engine.tool_registry.shutil.which",
                side_effect=lambda cmd: "/usr/local/bin/agy-b" if cmd == "agy-b" else None,
            ),
            patch(
                "src.acp.helper.get_cached_acp_models",
                return_value=[],
            ),
        ):
            avail_only_b = get_available_tools(force_refresh=True, require_available=True)
            assert "agy_b" in avail_only_b
            caps_only_b = get_tool_capabilities(force_refresh=True)
            assert caps_only_b["agy_b"].available is True

            # _default_members_for_preset must NOT fall back to hidden agy_b/claude_w/codex_w
            members_zero_visible = WorkflowConfigMixin._default_members_for_preset(
                caps_only_b,
                preset_id="collaborative",
            )
            assert all(m.tool_name not in {"agy_b", "claude_w", "codex_w"} for m in members_zero_visible)

            # /wf_config _build_initial_config_state keeps agy_b available in capabilities,
            # while initial auto-populated team and avail_first_tool exclude agy_b.
            client, _deliveries = _build_ws_client(tmp_path, monkeypatch)
            wf_handler = client._handler_ctx.handlers["workflow"]
            ok_proj, _msg, project = client._project_manager.create_project(
                "proj-wf", "demo", str(tmp_path / "repo"), chat_id="chat-1"
            )
            assert ok_proj and project is not None
            state, loaded_caps = wf_handler._build_initial_config_state(
                chat_id="chat-1",
                project=project,
                requirement="test",
            )
            assert loaded_caps["agy_b"].available is True
            assert all(m.tool_name != "agy_b" for m in state.members)
            assert state.editor_tool != "agy_b"

            # /wf _launch_auto_bound_workflow must NOT auto-select agy_b when 0 visible tools exist
            with patch.object(wf_handler, "_reply_workflow_error") as err_spy:
                wf_handler._launch_auto_bound_workflow(
                    message_id="om-wf-1",
                    chat_id="chat-1",
                    requirement="build feature",
                    project=project,
                    root_path=str(tmp_path / "repo"),
                    engine=MagicMock(),
                    admission_owner=None,
                )
            err_spy.assert_called_once()
            assert err_spy.call_args.args[1] == "invalid_state"
    finally:
        invalidate_cache()
