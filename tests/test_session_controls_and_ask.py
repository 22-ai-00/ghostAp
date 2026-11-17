"""Tests for Batch 3: running stop button, /stop, /clear, /sh, /ask, and UnifiedContext purity."""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.acp.models import PromptResult
from src.acp.outcome import PromptOutcome
from src.agent.intent_recognizer import IntentRecognizer, IntentType
from src.card.actions import dispatch as action_ids
from src.card.events import CardEvent, CardEventType
from src.card.programming_adapter import build_programming_metadata
from src.card.render.buttons import INTENT_TO_ACTION_ID
from src.card.render.footer import render_footer
from src.card.state.button_intent import ButtonIntent
from src.card.state.models import CardMetadata, CardState, FooterState
from src.card.state.reducers.lifecycle import reduce_lifecycle
from src.card.ui_text import UI_TEXT
from src.feishu.action_registry import init_action_registry
from src.feishu.control_plane import _SYSTEM_GATE_TASK_TYPES
from src.feishu.handlers.base import BaseHandler
from src.feishu.handlers.programming import ProgrammingModeHandler
from src.feishu.handlers.system import SystemHandler
from src.feishu.product_catalog import HIDDEN_SLASH_COMMANDS, resolve_command
from src.feishu.slash_command_parser import SlashCommandParser
from src.mode import PROGRAMMING_MODES, InteractionMode
from src.project.context import ProjectContext
from src.project.unified_context import (
    ContextEntryType,
    ContextSourceMode,
    ProjectContextManager,
)


def test_programming_stop_button_lifecycle_and_20s_footer_hint() -> None:
    assert "system_stop" in _SYSTEM_GATE_TASK_TYPES
    assert ButtonIntent.PROGRAMMING_STOP.value in INTENT_TO_ACTION_ID

    meta = build_programming_metadata(
        "agy",
        tool_name="agy",
        model_name="default",
        project_name="demo",
        working_dir="/tmp/demo",
    )
    assert meta.allow_programming_stop is True

    started = reduce_lifecycle(CardState(metadata=meta), CardEvent(type=CardEventType.STARTED))
    assert len(started.buttons) == 1
    assert started.buttons[0].action_id == ButtonIntent.PROGRAMMING_STOP
    assert started.buttons[0].disabled is False

    stopping = reduce_lifecycle(started, CardEvent(type=CardEventType.STOPPING))
    assert len(stopping.buttons) == 1
    assert stopping.buttons[0].action_id == ButtonIntent.PROGRAMMING_STOP
    assert stopping.buttons[0].disabled is True

    escalated = reduce_lifecycle(stopping, CardEvent(type=CardEventType.STOP_ESCALATED))
    assert len(escalated.buttons) == 1
    assert escalated.buttons[0].type == "danger"
    assert escalated.buttons[0].action_id == ButtonIntent.PROGRAMMING_STOP

    # Non-programming stream card without allow_programming_stop must not show programming stop
    wf_meta = CardMetadata(project_name="wf", programming_text_sections=True)
    wf_started = reduce_lifecycle(CardState(metadata=wf_meta), CardEvent(type=CardEventType.STARTED))
    assert wf_started.buttons == ()

    # Footer renders running controls hint only after >= 20s elapsed
    now_mono = time.monotonic()
    early_state = CardState(
        metadata=replace_started(meta, now_mono - 10.0),
        terminal="running",
        footer=FooterState(status="thinking", status_text="thinking"),
    )
    early_elements = render_footer(early_state)
    assert UI_TEXT["card_footer_running_controls_hint"] not in json.dumps(early_elements, ensure_ascii=False)

    late_state = CardState(
        metadata=replace_started(meta, now_mono - 25.0),
        terminal="running",
        footer=FooterState(status="thinking", status_text="thinking"),
    )
    late_elements = render_footer(late_state)
    assert UI_TEXT["card_footer_running_controls_hint"] in json.dumps(late_elements, ensure_ascii=False)


def replace_started(meta: CardMetadata, started_at: float) -> CardMetadata:
    from dataclasses import replace

    return replace(meta, session_started_at=started_at)


def test_stop_keyword_removed_from_exit_and_stop_command_preserves_mode() -> None:
    assert "停止" not in IntentRecognizer.EXIT_KEYWORDS
    recognizer = IntentRecognizer.__new__(IntentRecognizer)
    assert recognizer._match_programming_exit_keyword("停止", "agy") is None
    assert recognizer._match_programming_exit_keyword("退出", "agy").primary_intent == IntentType.EXIT_MODE

    assert SystemHandler.is_fast_lane_system_command("/stop") is True
    assert SystemHandler.is_fast_lane_system_command("/clear") is True
    assert SystemHandler.is_fast_lane_system_command("/sh git status") is True
    assert SystemHandler.is_fast_lane_system_command("/ask codex explain") is False

    project = ProjectContext(project_id="p1", project_name="demo", root_path="/tmp/demo")
    session = SimpleNamespace(session_id="s1", cancel=MagicMock())
    prog_session = SimpleNamespace(mark_stopping=MagicMock())
    mgr = MagicMock()
    mgr.cancel_session.return_value = True

    prog_handler = object.__new__(ProgrammingModeHandler)
    prog_handler.mode_key = "agy"
    prog_handler._active_programming_runs_lock = threading.Lock()
    prog_handler._active_programming_runs = {}
    prog_handler._get_session_manager = lambda: mgr
    key, state = prog_handler._register_active_programming_run(
        "chat-1", project, task_text="fix", cwd="/tmp/demo"
    )
    state.session = session
    state.prog_session = prog_session

    assert prog_handler.stop_active_run("chat-1", project) is True
    prog_session.mark_stopping.assert_called_once()
    session.cancel.assert_called_once_with(wait=False)
    mgr.cancel_session.assert_called_once()
    prog_handler._unregister_active_programming_run(key, state)

    system = object.__new__(SystemHandler)
    system.ctx = SimpleNamespace(
        handlers={"agy": prog_handler},
        mode_manager=SimpleNamespace(
            get_mode=lambda *_a, **_kw: InteractionMode.AGY,
            is_programming_mode=lambda *_a, **_kw: True,
        ),
        project_manager=SimpleNamespace(get_active_project=lambda _c: project),
        thread_manager=SimpleNamespace(get=lambda _t: None),
    )
    system.reply_text = MagicMock()
    system._resolve_active_programming_mode_key = lambda *_a, **_kw: "agy"
    prog_handler._active_programming_runs[("chat-1", "p1", "")] = state

    system.stop_current_task("m1", "chat-1", project)
    assert UI_TEXT["system_stop_sent_keep_mode"] in system.reply_text.call_args[0][1]

    # Action registry binds PROGRAMMING_STOP
    handlers_map = {m.value: MagicMock() for m in PROGRAMMING_MODES}
    handlers_map.update(
        {
            "coco": MagicMock(),
            "project": MagicMock(),
            "workflow": MagicMock(),
            "system": system,
            "deep": MagicMock(),
            "spec": MagicMock(),
        }
    )
    actions = init_action_registry(
        SimpleNamespace(
            _handler_ctx=SimpleNamespace(handlers=handlers_map),
            _project_manager=SimpleNamespace(get_project_for_chat=lambda *_a: project, get_active_project=lambda *_a: project),
        )
    )
    assert action_ids.PROGRAMMING_STOP in actions


def test_clear_and_sh_commands_keep_mode_and_execute_cleanly() -> None:
    assert resolve_command("/clear") is not None
    assert resolve_command("/reset").action.command == "/clear"
    assert "/sh" in HIDDEN_SLASH_COMMANDS
    assert "/ask" in HIDDEN_SLASH_COMMANDS

    project = ProjectContext(project_id="p1", project_name="demo", root_path="/tmp/demo")
    project.set_programming_mode("codex", True, session_id="s-old", query_count=4)
    project.acp_tool_name = "codex"
    project.acp_model_name = "gpt-5.6-sol"
    project.add_conversation("user", "old turn", "m0")

    mgr = MagicMock()
    store = MagicMock()
    prog_handler = object.__new__(ProgrammingModeHandler)
    prog_handler.mode_key = "codex"
    prog_handler._get_session_manager = lambda: mgr
    prog_handler.ctx = SimpleNamespace(
        context_manager=SimpleNamespace(store=store),
    )

    proj_mgr = MagicMock()
    proj_mgr.get_active_project.return_value = project
    system = object.__new__(SystemHandler)
    system.ctx = SimpleNamespace(
        project_manager=proj_mgr,
        handlers={"codex": prog_handler},
    )
    system.reply_text = MagicMock()
    system.add_reaction = MagicMock()
    system.submit_shell_command = MagicMock()
    system._resolve_active_programming_mode_key = lambda *_a, **_kw: "codex"
    system._resolve_current_acp_tool = lambda *_a, **_kw: "codex"

    # /clear ends live session and wipes conversation/snapshot without changing mode/tool/model
    clear_match = SlashCommandParser.parse("/reset")
    assert SystemHandler.is_ghostap_control_command(clear_match, in_programming_mode=True) is True
    system.handle_clear_command("m1", "chat-1", project)
    mgr.end_session.assert_called_once()
    assert project.conversation_history == []
    assert project.codex_session_snapshot is None
    assert project.codex_mode is True
    assert project.acp_tool_name == "codex"
    assert project.acp_model_name == "gpt-5.6-sol"
    proj_mgr.persist_project_context.assert_called_once_with(project)

    # /sh passes through control gate in active mode and submits shell command
    sh_match = SlashCommandParser.parse("/sh git status")
    assert SystemHandler.is_ghostap_control_command(sh_match, in_programming_mode=True) is True
    system.handle_sh_command("m2", "chat-1", "git status", project)
    system.submit_shell_command.assert_called_once_with("m2", "chat-1", "git status", "/tmp/demo", project)


def test_ask_command_is_zero_side_effect_and_closes_ephemeral_session(monkeypatch: pytest.MonkeyPatch) -> None:
    project = ProjectContext(project_id="p1", project_name="demo", root_path="/tmp/demo")
    project.set_programming_mode("coco", True, session_id="s-coco", query_count=2)
    project.acp_tool_name = "coco"
    project.acp_model_name = "coco-default"

    mode_manager = MagicMock()
    system = object.__new__(SystemHandler)
    system.ctx = SimpleNamespace(
        settings=SimpleNamespace(acp_startup_timeout=20),
        project_manager=SimpleNamespace(get_active_project=lambda _c: project),
        mode_manager=mode_manager,
    )
    system.reply_text = MagicMock()
    system.reply_error = MagicMock()
    system.reply_card = MagicMock()
    system.add_reaction = MagicMock()

    # Blocked /logout inside /ask
    system.handle_ask_command("m0", "chat-1", "codex /logout", project)
    system.reply_error.assert_called_once()

    fake_session = MagicMock()
    fake_session.send_prompt.return_value = PromptResult(stop_reason="end_turn", text="用 lock 即可。")
    closed_calls: list[object] = []

    monkeypatch.setattr(
        "src.feishu.handlers.system.is_programming_tool_available",
        lambda tool, **_kw: tool == "codex",
    )
    monkeypatch.setattr(
        "src.agent_session.create_auxiliary_session",
        lambda **_kw: fake_session,
    )
    monkeypatch.setattr(
        "src.agent_session.close_session_safely",
        lambda sess, **_kw: closed_calls.append(sess),
    )

    system.reply_error.reset_mock()
    system.handle_ask_command("m1", "chat-1", "codex 这里会死锁吗？", project)
    assert len(closed_calls) == 1
    assert closed_calls[0] is fake_session
    system.reply_card.assert_called_once()
    card_payload = system.reply_card.call_args[0][1]
    assert "Codex 旁路咨询" in card_payload
    assert "用 lock 即可" in card_payload
    # Project config and active mode were untouched
    assert project.acp_tool_name == "coco"
    assert project.coco_mode is True
    mode_manager.enter_programming_mode.assert_not_called()


def test_unified_context_purity_raw_text_file_changes_and_no_self_transition() -> None:
    ctx_mgr = ProjectContextManager()
    project = ProjectContext(project_id="p1", project_name="demo", root_path="/tmp/demo")

    # record_mode_transition skips no-op X -> X transitions
    base = object.__new__(BaseHandler)
    base.ctx = SimpleNamespace(context_manager=ctx_mgr)
    base.record_mode_transition("p1", InteractionMode.AGY, InteractionMode.AGY, "resume", chat_id="chat-1")
    uctx = ctx_mgr.get("p1", chat_id="chat-1")
    assert uctx is None or uctx.get_entries_by_type(ContextEntryType.MODE_TRANSITION) == []

    # Real transition records entry & bridge
    base.record_mode_transition("p1", InteractionMode.SMART, InteractionMode.AGY, "enter", chat_id="chat-1")
    uctx = ctx_mgr.get("p1", chat_id="chat-1")
    assert len(uctx.get_entries_by_type(ContextEntryType.MODE_TRANSITION)) == 1

    # _execute_programming_response records raw_task_text (without bridge wrapper) + FILE_CHANGE
    handler = object.__new__(ProgrammingModeHandler)
    handler.mode_key = "agy"
    handler.mode_name = "agy"
    handler.display_name = "AGY"
    handler.context_source = ContextSourceMode.AGY
    handler.is_coco = False
    handler.reply_text = MagicMock()
    handler.reply_card = MagicMock()
    handler.add_reaction = MagicMock()
    handler._update_snapshot_on_project = MagicMock()
    handler._get_active_programming_run = MagicMock(return_value=None)
    handler._get_session_manager = MagicMock(return_value=MagicMock())
    handler._consume_deferred_exit_for_turn = MagicMock(return_value=False)
    handler.ctx = SimpleNamespace(
        settings=SimpleNamespace(coco_execution_timeout=60, claude_execution_timeout=60, acp_max_execution_windows=1),
        context_manager=ctx_mgr,
        project_manager=SimpleNamespace(save_projects=MagicMock()),
        thread_manager=SimpleNamespace(get=lambda _t: None),
        acp_manager=None,
        card_controller=None,
        hook_manager=None,
        card_factory=None,
    )

    prog_session = MagicMock()
    prog_session.get_conclusion_text.return_value = "done"
    prog_session.get_final_text.return_value = "done"
    prog_session.finish.return_value = SimpleNamespace(succeeded=True)
    result = PromptResult(
        stop_reason="end_turn",
        text="done",
        tool_calls=[],
        modified_files={"src/a.py", "tests/test_a.py"},
    )
    execution_res = SimpleNamespace(
        result=result,
        assessment=SimpleNamespace(
            outcome=PromptOutcome.COMPLETED,
            stop_reason="end_turn",
            pending_plan_entries=0,
            incomplete_tool_calls=0,
            incomplete_outer_tool_calls=0,
            unresolved_child_tool_calls=0,
            incomplete_tool_diagnostics=(),
        ),
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
        handler._execute_programming_response(
            message_id="m1",
            chat_id="chat-1",
            text="[Context from previous smart session]\n...\n\n请修复 bug",
            session=session,
            project=project,
            cwd=".",
            global_working_dir=".",
            prog_session=prog_session,
            _finalization_task_text="请修复 bug",
        )
    finally:
        prog_mod.run_prompt_across_execution_windows = orig_runner

    assert project.conversation_history[0].content == "请修复 bug"
    convs = uctx.get_entries_by_type(ContextEntryType.CONVERSATION)
    assert convs[0].content == "请修复 bug"
    changes = uctx.get_entries_by_type(ContextEntryType.FILE_CHANGE)
    assert [c.content for c in changes] == ["src/a.py", "tests/test_a.py"]
