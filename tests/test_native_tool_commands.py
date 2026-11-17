"""Tests for tool-native slash commands, usage tracking, /cmds cards, and smart routing."""

from __future__ import annotations

import asyncio
import io
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from acp.schema import (
    AvailableCommand,
    AvailableCommandInput,
    AvailableCommandsUpdate,
    Cost,
    UnstructuredCommandInput,
    UsageUpdate,
)

from src.acp.client import GhostAPClient
from src.acp.execution_instructions import AUTONOMOUS_EXECUTION_PROMPT
from src.acp.models import ACPAvailableCommand, ACPUsageInfo, PromptResult
from src.acp.native_commands import (
    BLOCKED_NATIVE_COMMANDS,
    build_acp_prompt_texts,
    get_featured_native_commands,
    get_tool_native_commands,
    is_blocked_native_command,
    is_pure_zero_arg_native_slash,
    normalize_acp_available_commands,
    parse_slash_command,
    resolve_smart_mode_native_command,
)
from src.acp.outcome import PromptOutcome
from src.acp.session import ACPSession
from src.acp.sync_adapter import SyncACPSession
from src.agent_session.codex_cli import SyncCodexCLISession
from src.card.actions import dispatch as action_ids
from src.card.builders.system import SystemBuilder
from src.feishu.action_registry import init_action_registry
from src.feishu.dispatcher import MessageDispatcher
from src.feishu.handlers.system import SystemHandler
from src.feishu.product_catalog import resolve_command
from src.feishu.slash_command_parser import SlashCommandParser
from src.mode import PROGRAMMING_MODES, InteractionMode
from src.project.unified_context import ContextSourceMode, UnifiedContext


def test_parse_slash_command_distinguishes_commands_from_posix_paths() -> None:
    assert parse_slash_command("/swarm 修复 auth") == ("swarm", "修复 auth")
    assert parse_slash_command("  /compact ") == ("compact", "")
    assert parse_slash_command("/review-branch main") == ("review-branch", "main")
    assert parse_slash_command("/usr/bin/python -V") is None
    assert parse_slash_command("/data00/home/repo/file.py") is None
    assert parse_slash_command("//comment") is None
    assert parse_slash_command("plain text") is None


def test_blocked_commands_and_prompt_block_ordering() -> None:
    assert "logout" in BLOCKED_NATIVE_COMMANDS
    assert is_blocked_native_command("/logout") is True
    assert is_blocked_native_command("logout now") is True
    assert is_blocked_native_command("/compact") is False

    raw = [
        AvailableCommand(name="swarm", description="Run swarm", input=AvailableCommandInput(root=UnstructuredCommandInput(hint="<task>"))),
        AvailableCommand(name="logout", description="Logout"),
        AvailableCommand(name="compact", description="Compact context"),
    ]
    normalized = normalize_acp_available_commands(raw)
    assert [c.name for c in normalized] == ["swarm", "compact"]
    assert normalized[0].input_hint == "<task>"

    assert is_pure_zero_arg_native_slash("/compact") is True
    assert is_pure_zero_arg_native_slash("/status") is True
    assert is_pure_zero_arg_native_slash("/swarm fix bug") is False

    assert build_acp_prompt_texts("/compact") == ["/compact"]
    assert build_acp_prompt_texts("/status") == ["/status"]
    assert build_acp_prompt_texts("/swarm fix bug") == [
        "/swarm fix bug",
        AUTONOMOUS_EXECUTION_PROMPT,
    ]
    assert build_acp_prompt_texts("fix bug") == [
        AUTONOMOUS_EXECUTION_PROMPT,
        "fix bug",
    ]


def test_resolve_smart_mode_native_command_and_static_catalog() -> None:
    cmds_agy, src_agy = get_tool_native_commands("agy")
    assert src_agy == "static"
    assert any(c.name == "swarm" for c in cmds_agy)
    featured = get_featured_native_commands("agy", limit=4)
    assert len(featured) == 4

    # /swarm uniquely maps to agy
    res_swarm = resolve_smart_mode_native_command(
        "/swarm build feature",
        available_tools=["codex", "agy"],
    )
    assert res_swarm.status == "matched"
    assert res_swarm.tool_name == "agy"

    # /review honors preferred_tool when multiple tools support it
    res_review = resolve_smart_mode_native_command(
        "/review focus auth",
        preferred_tool="traex",
        available_tools=["codex", "traex"],
    )
    assert res_review.status == "matched"
    assert res_review.tool_name == "traex"

    # claude-w alias normalizes cleanly
    cmds_cw, _ = get_tool_native_commands("claude-w")
    assert any(c.name == "compact" for c in cmds_cw)

    assert resolve_smart_mode_native_command("/logout").status == "blocked"
    assert resolve_smart_mode_native_command("/swarm test", available_tools=["codex"]).status == "unavailable"
    assert resolve_smart_mode_native_command("/totally_unknown_xyz").status == "none"


def test_acp_client_and_session_capture_commands_and_usage_and_order() -> None:
    captured_cmds: list[tuple[str, list[ACPAvailableCommand]]] = []
    captured_usage: list[tuple[str, ACPUsageInfo]] = []

    client = GhostAPClient(
        on_event=lambda _ev: None,
        on_available_commands=lambda sid, cmds: captured_cmds.append((sid, cmds)),
        on_usage_update=lambda sid, usage: captured_usage.append((sid, usage)),
    )

    async def _feed() -> None:
        await client.session_update(
            "sess-1",
            AvailableCommandsUpdate(
                session_update="available_commands_update",
                available_commands=[
                    AvailableCommand(name="swarm", description="Swarm"),
                    AvailableCommand(name="logout", description="Blocked"),
                ],
            ),
        )
        await client.session_update(
            "sess-1",
            UsageUpdate(
                session_update="usage_update",
                used=12000,
                size=200000,
                cost=Cost(amount=0.125, currency="USD"),
            ),
        )

    asyncio.run(_feed())
    assert len(captured_cmds) == 1
    assert [c.name for c in captured_cmds[0][1]] == ["swarm"]
    assert captured_usage == [("sess-1", ACPUsageInfo(used=12000, size=200000, cost_amount=0.125, cost_currency="USD"))]

    # ACPSession callback + epoch fencing + SyncACPSession summary
    acp_sess = ACPSession(agent_cmd="agy", agent_args=[], cwd=".")
    acp_sess._session_id = "sess-1"
    acp_sess._on_available_commands("sess-1", [ACPAvailableCommand("swarm", "Swarm")], transport_epoch=acp_sess._transport_epoch, load_epoch=acp_sess._load_epoch)
    acp_sess._on_available_commands("stale", [ACPAvailableCommand("compact", "Old")], transport_epoch=acp_sess._transport_epoch - 1)
    assert [c.name for c in acp_sess.get_available_commands()] == ["swarm"]

    acp_sess._on_usage_update("sess-1", ACPUsageInfo(used=5000, size=100000), transport_epoch=acp_sess._transport_epoch)
    assert acp_sess.get_usage_info() == ACPUsageInfo(used=5000, size=100000)

    sync_sess = object.__new__(SyncACPSession)
    sync_sess._agent_type = "agy"
    sync_sess.session_id = "sess-1"
    sync_sess.created_at = 0.0
    sync_sess.message_count = 3
    sync_sess.is_resumed = False
    sync_sess._model_name = "claude-sonnet-4"
    sync_sess._acp_session = acp_sess
    info = sync_sess.get_session_info()
    assert "claude-sonnet-4" in info
    assert "5,000 / 100,000 (5%)" in info
    assert "原生命令: 1 条" in info


def test_codex_cli_build_args_preserves_native_slash(monkeypatch) -> None:
    captured: list[list[str]] = []

    class _FakeProc:
        def __init__(self) -> None:
            self.pid = 12345
            self.returncode = 0
            self.stdout = iter(['{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}\n'])
            self.stderr = io.StringIO("")

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    def _fake_popen(args, **_kw):
        captured.append(list(args))
        return _FakeProc()

    monkeypatch.setattr("src.agent_session.codex_cli.subprocess.Popen", _fake_popen)
    sess = SyncCodexCLISession(cwd=".")

    sess.send_prompt("/compact")
    assert captured[-1][-1] == "/compact"

    sess.send_prompt("/review focus security")
    assert captured[-1][-1].startswith("/review focus security")
    assert AUTONOMOUS_EXECUTION_PROMPT in captured[-1][-1]


def test_inject_bridge_context_preserves_leading_slash_and_dollar() -> None:
    from src.feishu.handlers.base import BaseHandler

    ctx_store = {}
    uctx = UnifiedContext(project_id="p1")
    uctx.add_conversation("user", "recent changes", source_mode=ContextSourceMode.SMART)
    uctx.build_bridge_summary(ContextSourceMode.SMART, ContextSourceMode.AGY)
    ctx_store[("p1", "c1")] = uctx

    handler = object.__new__(BaseHandler)
    handler.ctx = SimpleNamespace(
        context_manager=SimpleNamespace(
            store=SimpleNamespace(get=lambda pid, chat_id="": ctx_store.get((pid, chat_id)))
        )
    )
    project = SimpleNamespace(project_id="p1", project_name="demo")

    # Zero-arg native slash does not consume bridge or prepend text
    assert handler.inject_bridge_context("/compact", project, chat_id="c1") == "/compact"
    assert uctx.last_bridge_summary is not None

    # Parameterized slash keeps '/' at byte 0 and consumes bridge
    injected_slash = handler.inject_bridge_context("/swarm refactor", project, chat_id="c1")
    assert injected_slash.startswith("/swarm refactor\n\n")
    assert "recent changes" in injected_slash
    assert uctx.last_bridge_summary is None

    # Parameterized '$skill' also stays at byte 0
    uctx.build_bridge_summary(ContextSourceMode.SMART, ContextSourceMode.CODEX)
    injected_dollar = handler.inject_bridge_context("$my-skill inspect", project, chat_id="c1")
    assert injected_dollar.startswith("$my-skill inspect\n\n")


def test_ready_card_and_tool_commands_card_and_actions() -> None:
    _, ready_raw = SystemBuilder.build_acp_programming_ready_card("agy", "default", "p1")
    ready_card = json.loads(ready_raw)
    assert ready_card["schema"] == "2.0"
    raw_ready_dump = json.dumps(ready_card, ensure_ascii=False)
    assert "/swarm" in raw_ready_dump
    assert action_ids.SHOW_TOOL_COMMANDS in raw_ready_dump

    cmds, src = get_tool_native_commands("codex")
    _, cmds_raw = SystemBuilder.build_tool_native_commands_card("codex", cmds, source=src, project_id="p1")
    cmds_card = json.loads(cmds_raw)
    assert cmds_card["schema"] == "2.0"
    cmds_dump = json.dumps(cmds_card, ensure_ascii=False)
    assert action_ids.RUN_NATIVE_TOOL_COMMAND in cmds_dump
    assert "logout" not in cmds_dump


def test_system_handler_and_dispatcher_native_routing_and_fast_lane() -> None:
    assert resolve_command("/cmds") is not None
    assert resolve_command("/commands") is not None

    # Fast-lane queue classification: task-bearing slash prompts stay off system_fast
    assert SystemHandler.is_fast_lane_system_command("/help") is True
    assert SystemHandler.is_fast_lane_system_command("/cmds") is True
    assert SystemHandler.is_fast_lane_system_command("/agy") is True
    assert SystemHandler.is_fast_lane_system_command("/agy /swarm fix") is False
    assert SystemHandler.is_fast_lane_system_command("/codex fix auth") is False
    assert SystemHandler.is_fast_lane_system_command("/swarm fix") is False
    assert SystemHandler.is_fast_lane_system_command("/compact") is False
    assert SystemHandler.is_fast_lane_system_command("/wf build it") is True
    assert SystemHandler.is_fast_lane_system_command("/stop_wf") is True
    assert SystemHandler.is_fast_lane_system_command("/review-branch main") is False
    assert SystemHandler.is_fast_lane_system_command("/goal run tests", in_programming_mode=True) is False
    assert SystemHandler.is_fast_lane_system_command("/goal run tests", in_programming_mode=False) is True

    # Active programming mode passes /goal & /swarm through instead of intercepting
    goal_match = SlashCommandParser.parse("/goal finish work")
    assert goal_match is not None
    assert SystemHandler.is_ghostap_control_command(goal_match, in_programming_mode=True) is False
    assert SystemHandler.is_ghostap_control_command(goal_match, in_programming_mode=False) is True

    # Dispatcher pass-through vs SMART auto-routing
    dispatcher = object.__new__(MessageDispatcher)
    dispatcher.base = MagicMock()
    dispatcher.project = MagicMock()
    dispatcher.handlers = {"coco": dispatcher.base, "system": MagicMock(), "project": dispatcher.project}
    dispatcher.system = dispatcher.handlers["system"]
    dispatcher.system.is_ghostap_control_command = SystemHandler.is_ghostap_control_command
    dispatcher.system.is_interceptable_command_match = SystemHandler.is_interceptable_command_match
    dispatcher._router = MagicMock()
    dispatcher._router.is_exit_command.return_value = False
    dispatcher._router.is_deep_command.return_value = False
    dispatcher._router.is_spec_command.return_value = False
    dispatcher._router.is_workflow_command.return_value = False
    agy_handler = MagicMock()
    dispatcher.client = SimpleNamespace(
        _mode_manager=SimpleNamespace(
            get_mode=lambda *_a, **_kw: InteractionMode.AGY,
            is_programming_mode=lambda *_a, **_kw: True,
        ),
        _get_effective_mode=lambda *_a, **_kw: (InteractionMode.AGY, True),
        _project_manager=SimpleNamespace(get_active_project=lambda _c: SimpleNamespace(project_id="p1", root_path=".")),
        _control_plane=SimpleNamespace(should_defer_exit=lambda **_kw: False),
        _thread_manager=SimpleNamespace(get=lambda _t: None),
        _get_mode_handler=lambda mode: agy_handler if mode == InteractionMode.AGY else None,
        _pending_image_lock=SimpleNamespace(__enter__=lambda *_: None, __exit__=lambda *_: None),
        _pending_image_only=set(),
    )

    project = SimpleNamespace(project_id="p1", root_path=".")
    # Active AGY mode: /swarm and /goal both pass through directly to agy_handler
    dispatcher.process_with_intent("m1", "c1", "/swarm fix tests", project=project)
    dispatcher.process_with_intent("m2", "c1", "/goal ship release", project=project)
    assert agy_handler.handle_message.call_count == 2
    # Blocked /logout never forwards
    dispatcher.process_with_intent("m3", "c1", "/logout", project=project)
    assert agy_handler.handle_message.call_count == 2
    dispatcher.base.reply_error.assert_called_once()


def test_action_registry_and_empty_output_native_slash_completion() -> None:
    from src.feishu.handlers.programming import ProgrammingModeHandler

    # Verify action registry registers SHOW_TOOL_COMMANDS & RUN_NATIVE_TOOL_COMMAND
    handlers_map = {
        mode.value: MagicMock() for mode in PROGRAMMING_MODES
    }
    handlers_map.update(
        {
            "coco": MagicMock(),
            "project": MagicMock(),
            "workflow": MagicMock(),
            "system": MagicMock(),
            "deep": MagicMock(),
            "spec": MagicMock(),
        }
    )
    mock_client = SimpleNamespace(
        _handler_ctx=SimpleNamespace(handlers=handlers_map),
        _project_manager=SimpleNamespace(),
    )
    reg = init_action_registry(mock_client)
    assert action_ids.SHOW_TOOL_COMMANDS in reg
    assert action_ids.RUN_NATIVE_TOOL_COMMAND in reg

    # Synthesize completion summary on empty-text /compact turn
    handler = object.__new__(ProgrammingModeHandler)
    handler.mode_key = "traex"
    handler.mode_name = "traex"
    handler.display_name = "Traex"
    handler.context_source = "traex"
    handler.is_coco = False
    handler.reply_text = MagicMock()
    handler.reply_card = MagicMock()
    handler.add_reaction = MagicMock()
    handler._update_snapshot_on_project = MagicMock()
    handler._get_active_programming_run = MagicMock(return_value=None)
    handler._get_session_manager = MagicMock(return_value=MagicMock())
    handler.ctx = SimpleNamespace(
        settings=SimpleNamespace(
            coco_execution_timeout=60,
            claude_execution_timeout=60,
            acp_max_execution_windows=1,
        ),
        context_manager=SimpleNamespace(update_context=MagicMock()),
        project_manager=SimpleNamespace(save_projects=MagicMock()),
        thread_manager=SimpleNamespace(get=lambda _t: None),
        acp_manager=None,
        card_controller=None,
        hook_manager=None,
        card_factory=None,
    )
    handler._consume_deferred_exit_for_turn = MagicMock(return_value=False)

    prog_session = MagicMock()
    prog_session.get_conclusion_text.return_value = ""
    prog_session.get_final_text.return_value = ""
    prog_session.finish.return_value = SimpleNamespace(succeeded=True)
    result = PromptResult(
        stop_reason="end_turn",
        text="",
        tool_calls=[],
        usage=ACPUsageInfo(used=4000, size=100000),
    )
    assessment = SimpleNamespace(
        outcome=PromptOutcome.COMPLETED,
        stop_reason="end_turn",
        pending_plan_entries=0,
        incomplete_tool_calls=0,
        incomplete_outer_tool_calls=0,
        unresolved_child_tool_calls=0,
        incomplete_tool_diagnostics=(),
    )
    execution_res = SimpleNamespace(
        result=result,
        assessment=assessment,
        entered_finalization=False,
        execution_windows=1,
        window_limit_reached=False,
        automatic_continuations=0,
        windows_used=1,
        continued=False,
        finalization_used=False,
    )

    import src.feishu.handlers.programming as prog_mod
    orig_runner = prog_mod.run_prompt_across_execution_windows
    try:
        prog_mod.run_prompt_across_execution_windows = lambda *_a, **_kw: execution_res
        session = SimpleNamespace(session_id="s1", message_count=1, _force_dead=False)
        project = SimpleNamespace(project_id="p1", root_path=".")
        handler._execute_programming_response(
            message_id="m1",
            chat_id="c1",
            text="/compact",
            session=session,
            project=project,
            cwd=".",
            global_working_dir=".",
            prog_session=prog_session,
        )
    finally:
        prog_mod.run_prompt_across_execution_windows = orig_runner

    prog_session.on_text.assert_called_once()
    written = prog_session.on_text.call_args[0][0]
    assert "/compact" in written
    assert "4,000/100,000 (4%)" in written


def test_system_handler_direct_tool_with_args_and_smart_auto_routing(monkeypatch) -> None:
    handler = object.__new__(SystemHandler)
    handler.reply_error = MagicMock()
    handler.add_reaction = MagicMock()
    handler.handle_enter_acp_saved_selection = MagicMock()
    handler.show_explicit_acp_model_selection = MagicMock()
    handler._resolve_active_programming_mode_key = MagicMock(return_value=None)
    agy_handler = MagicMock()
    handler.get_handler = lambda name: agy_handler if name == "agy" else MagicMock()
    project = SimpleNamespace(project_id="p1", acp_tool_name=None)
    handler._configuration_project = MagicMock(return_value=project)
    handler._resolve_live_commands_for_tool = MagicMock(return_value=None)

    # /agy /logout is blocked
    handler._handle_direct_mode_enter_with_args("m0", "c1", "agy", "/logout", project)
    handler.reply_error.assert_called_once()
    handler.handle_enter_acp_saved_selection.assert_not_called()

    # SMART -> /agy /swarm fix enters saved selection with pending_prompt
    handler._handle_direct_mode_enter_with_args("m1", "c1", "agy", "/swarm fix", project)
    handler.handle_enter_acp_saved_selection.assert_called_once_with(
        "m1",
        "c1",
        "agy",
        project,
        pending_prompt="/swarm fix",
    )

    # Already in AGY mode -> /agy /compact forwards directly without re-entering
    handler._resolve_active_programming_mode_key.return_value = "agy"
    handler._handle_direct_mode_enter_with_args("m2", "c1", "agy", "/compact", project)
    agy_handler.handle_message.assert_called_once_with("m2", "c1", "/compact", project)

    # SMART auto-routing for /swarm -> agy, /unknown_cmd -> False, /logout -> True + error
    handler.handle_enter_acp_saved_selection.reset_mock()
    handler.reply_error.reset_mock()
    monkeypatch.setattr(
        "src.feishu.handlers.system.is_programming_tool_available",
        lambda tool, **_kw: tool == "agy",
    )
    assert handler.try_dispatch_smart_native_command("m3", "c1", "/swarm build feature", project) is True
    handler.handle_enter_acp_saved_selection.assert_called_once_with(
        "m3",
        "c1",
        "agy",
        project,
        pending_prompt="/swarm build feature",
    )
    assert handler.try_dispatch_smart_native_command("m4", "c1", "/logout", project) is True
    handler.reply_error.assert_called_once()
    assert handler.try_dispatch_smart_native_command("m5", "c1", "/unknown_xyz_cmd", project) is False
