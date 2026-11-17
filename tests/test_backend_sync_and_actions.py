"""Tests for 11-backend SSOT metadata, cached model picker, live sync, switch preservation, and Card 2.0 actions."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.card.actions import dispatch as action_ids
from src.card.backend_meta import (
    ALL_BACKEND_METAS,
    VISIBLE_BACKEND_METAS,
    get_backend_meta,
    resolve_project_programming_mode,
)
from src.card.builders.core import CoreBuilder
from src.card.builders.project import ProjectBuilder
from src.card.themes import MODE_TEMPLATES
from src.feishu.action_registry import init_action_registry
from src.feishu.handlers.project import ProjectHandler
from src.feishu.handlers.system import SystemHandler
from src.mode import PROGRAMMING_MODES, InteractionMode
from src.project.context import ProjectContext, ProjectStatus
from src.utils.signing import sign_command


def _make_handlers_map() -> dict:
    handlers_map = {mode.value: MagicMock() for mode in PROGRAMMING_MODES}
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
    return handlers_map


def test_backend_meta_ssot_covers_all_11_backends() -> None:
    assert len(ALL_BACKEND_METAS) == 11
    assert len(VISIBLE_BACKEND_METAS) == 9
    assert {m.mode_key for m in VISIBLE_BACKEND_METAS} & {"claude_w", "codex_w"} == set()

    for meta in ALL_BACKEND_METAS:
        assert meta.display_name in MODE_TEMPLATES
        assert MODE_TEMPLATES[meta.display_name] == meta.template_color
        assert get_backend_meta(meta.mode_key) == meta
        assert get_backend_meta(meta.cli_name) == meta

    # Header titles and mode resolution cover previously-omitted backends
    for key in ("codex", "aiden", "claude_w", "codex_w", "agy"):
        proj = ProjectContext(project_id="p1", project_name="demo", root_path="/tmp/demo")
        proj.set_programming_mode(key, True, session_id="s1", query_count=4)
        resolved = resolve_project_programming_mode(proj)
        assert resolved == InteractionMode(key)
        header = CoreBuilder._build_header_title(proj)
        assert get_backend_meta(key).display_name in header

    # Project board renders mode info for AGY and attaches Card 2.0 behaviors on all callback buttons
    agy_proj = ProjectContext(project_id="p-agy", project_name="agy-app", root_path="/tmp/agy", last_active=time.time())
    agy_proj.set_programming_mode("agy", True, session_id="s2", query_count=7)
    agy_proj.status = ProjectStatus.IDLE
    idle_proj = ProjectContext(project_id="p-other", project_name="other", root_path="/tmp/o", last_active=time.time() - 10)
    idle_proj.acp_tool_name = "codex"
    idle_proj.acp_model_name = "gpt-5.6-sol"

    _, board_json = ProjectBuilder.build_status_board_card([agy_proj, idle_proj], current_project_id="p-agy")
    board = json.loads(board_json)
    board_str = json.dumps(board, ensure_ascii=False)
    assert "AGY(7)" in board_str
    assert "Codex/gpt-5.6-sol" in board_str

    stack = [board]
    callback_buttons = []
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if node.get("tag") == "button" and "value" in node:
                callback_buttons.append(node)
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    assert callback_buttons
    for btn in callback_buttons:
        assert btn["behaviors"] == [{"type": "callback", "value": btn["value"]}]


def test_open_acp_model_picker_skips_cache_invalidation(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = object.__new__(SystemHandler)
    project = ProjectContext(project_id="p1", project_name="demo", root_path="/tmp/demo")
    handler._action_project = MagicMock(return_value=project)
    handler.show_explicit_acp_model_selection = MagicMock()
    invalidate_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "src.feishu.handlers.system.invalidate_acp_model_cache",
        lambda tool, root: invalidate_calls.append((tool, root)),
    )

    handler.handle_open_acp_model_picker(
        "om-ready",
        "chat-1",
        "p1",
        {"action": action_ids.OPEN_ACP_MODEL_PICKER, "tool_name": "agy"},
    )
    assert invalidate_calls == []
    handler.show_explicit_acp_model_selection.assert_called_once()
    assert handler.show_explicit_acp_model_selection.call_args.kwargs["show_loading"] is False

    handler.handle_refresh_acp_models(
        "om-picker",
        "chat-1",
        "p1",
        {"action": action_ids.REFRESH_ACP_MODELS, "tool_name": "agy"},
    )
    assert invalidate_calls == [("agy", "/tmp/demo")]


def test_model_list_and_live_session_sync_and_acp_tool_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    project = ProjectContext(project_id="p1", project_name="demo", root_path="/tmp/demo")
    project.acp_tool_name = "codex"
    project.acp_model_name = "old-model"

    live_codex = SimpleNamespace(session_id="sess-codex", set_model=MagicMock(return_value=True))
    codex_mgr = MagicMock()
    codex_mgr.get_session.return_value = live_codex
    codex_handler = SimpleNamespace(
        current_model="old-model",
        _enter_mode_on_manager=lambda *_a, **_kw: None,
        _get_session_manager=lambda: codex_mgr,
    )

    live_agy = SimpleNamespace(session_id="sess-agy")
    agy_mgr = MagicMock()
    agy_mgr.get_session.return_value = live_agy
    agy_handler = SimpleNamespace(_get_session_manager=lambda: agy_mgr)

    mode_manager = MagicMock()
    mode_manager.get_mode.return_value = InteractionMode.CODEX
    mode_manager.is_programming_mode.return_value = True

    proj_mgr = MagicMock()
    proj_mgr.commit_acp_configuration.side_effect = (
        lambda proj, *, tool_name, model_name: (setattr(proj, "acp_tool_name", tool_name), setattr(proj, "acp_model_name", model_name), True)[-1]
    )

    system = object.__new__(SystemHandler)
    system.ctx = SimpleNamespace(
        project_manager=proj_mgr,
        mode_manager=mode_manager,
        thread_manager=SimpleNamespace(get=lambda _t: None),
        handlers={"codex": codex_handler, "agy": agy_handler},
    )
    system.reply_text = MagicMock()
    system.reply_error = MagicMock()
    system._fetch_acp_models = MagicMock(
        return_value=[SimpleNamespace(name="old-model", selection_variants=()), SimpleNamespace(name="gpt-5.6-sol", selection_variants=())]
    )
    monkeypatch.setattr("src.thread.get_current_thread_id", lambda: None)

    # /model list displays summary without committing
    system.handle_model_command("m0", "chat-1", "/model list", project)
    proj_mgr.commit_acp_configuration.assert_not_called()
    assert "gpt-5.6-sol" in system.reply_text.call_args[0][1]

    # /model <name> updates live session via set_model
    system.handle_model_command("m1", "chat-1", "/model gpt-5.6-sol", project)
    live_codex.set_model.assert_called_once_with("gpt-5.6-sol")
    assert codex_handler.current_model == "gpt-5.6-sol"

    # /model default retires live session so next turn launches default
    system.handle_model_command("m2", "chat-1", "/model default", project)
    codex_mgr.retire_session.assert_called_once_with("chat-1", project_id="p1")
    assert codex_handler.current_model is None

    # /acp agy while active in codex retires codex session and switches active mode
    monkeypatch.setattr(
        "src.feishu.handlers.system.list_acp_tools",
        lambda: [SimpleNamespace(name="codex"), SimpleNamespace(name="agy")],
    )
    system.handle_acp_command("m3", "chat-1", "/acp agy", project)
    assert project.acp_tool_name == "agy"
    mode_manager.enter_programming_mode.assert_called_once_with("chat-1", InteractionMode.AGY, project_id="p1")


def test_project_preserve_and_switch_respects_all_11_backends_and_acp_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.thread.get_current_thread_id", lambda: None)

    agy_proj = ProjectContext(project_id="p1", project_name="p-agy", root_path="/tmp/p1")
    agy_proj.set_programming_mode("agy", True, session_id="s-old", query_count=1)
    agy_session = SimpleNamespace(
        session_id="s-agy",
        last_query="do work",
        message_count=5,
        to_snapshot=lambda: {"session_id": "s-agy"},
    )
    agy_mgr = SimpleNamespace(get_session=lambda *_a, **_kw: agy_session)
    ctx_mgr = MagicMock()

    handler = object.__new__(ProjectHandler)
    handler.ctx = SimpleNamespace(
        agy_manager=agy_mgr,
        mode_manager=SimpleNamespace(get_mode=lambda *_a, **_kw: InteractionMode.AGY),
        context_manager=ctx_mgr,
        project_manager=MagicMock(),
        handlers={},
    )
    handler.preserve_project_context("chat-1", agy_proj)
    assert agy_proj.agy_session_snapshot.session_id == "s-agy"
    assert agy_proj.agy_session_snapshot.query_count == 5
    ctx_mgr.update_context.assert_called_once()
    assert ctx_mgr.update_context.call_args.kwargs["session_snapshot"]["source_mode"] == "agy"

    # Switching to a project with acp_tool_name='agy' must NOT overwrite with coco
    target_proj = ProjectContext(project_id="p2", project_name="target", root_path="/tmp/p2")
    target_proj.acp_tool_name = "agy"
    target_proj.acp_model_name = "sonnet"
    coco_handler = MagicMock()
    target_agy_handler = MagicMock()

    handler.ctx = SimpleNamespace(
        project_manager=SimpleNamespace(
            find_project_by_name_with_hint=lambda *_a, **_kw: (target_proj, ""),
            validate_project_path=lambda *_a: (True, ""),
            get_active_project=lambda _c: None,
            set_active_project=lambda _c, _p: (True, "ok"),
        ),
        context_manager=MagicMock(),
        mode_manager=MagicMock(),
        handlers={"agy": target_agy_handler},
    )
    handler.reply_card = MagicMock()
    handler.register_message_project = MagicMock()

    handler.switch_project("m1", "chat-1", "target", auto_enter_coco=True, coco_handler=coco_handler)
    coco_handler.enter_mode.assert_not_called()
    target_agy_handler.enter_mode.assert_called_once_with(
        "m1",
        "chat-1",
        project=target_proj,
        model_override="sonnet",
    )

    # Switching to an unconfigured project sends notification card rather than forcing coco
    unconfigured = ProjectContext(project_id="p3", project_name="plain", root_path="/tmp/p3")
    handler.ctx.project_manager.find_project_by_name_with_hint = lambda *_a, **_kw: (unconfigured, "")
    handler.switch_project("m2", "chat-1", "plain", auto_enter_coco=True, coco_handler=coco_handler)
    coco_handler.enter_mode.assert_not_called()
    handler.reply_card.assert_called_once()


def test_action_registry_handles_previously_unregistered_actions_and_signed_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GHOSTAP_SIGNING_KEY", "test-secret-signing-key-32-chars!!")
    handlers_map = _make_handlers_map()
    saved_proj = ProjectContext(project_id="p1", project_name="demo", root_path="/tmp/demo")
    saved_proj.acp_tool_name = "codex"
    client = SimpleNamespace(
        _handler_ctx=SimpleNamespace(handlers=handlers_map),
        _project_manager=SimpleNamespace(
            get_project_for_chat=lambda pid, cid: saved_proj if pid == "p1" else None,
            get_active_project=lambda cid: saved_proj,
        ),
    )

    actions = init_action_registry(client)
    for required in (
        action_ids.OPEN_ACP_MODEL_PICKER,
        action_ids.CONTINUE_DEV,
        action_ids.NEW_PROJECT_PROMPT,
        action_ids.CONFIRM_FORCE_RELEASE,
        action_ids.CONFIRM_LOCK,
        action_ids.RETRY_COMMAND,
    ):
        assert required in actions

    actions[action_ids.CONTINUE_DEV]("m1", "chat-1", "p1", {"action": "continue_dev", "project_id": "p1"})
    handlers_map["system"].handle_enter_acp_saved_selection.assert_called_once_with("m1", "chat-1", "codex", saved_proj)

    saved_proj.acp_tool_name = None
    actions[action_ids.CONTINUE_DEV]("m2", "chat-1", "p1", {"action": "continue_dev", "project_id": "p1"})
    handlers_map["system"].show_tools_list.assert_called_once_with("m2", "chat-1", saved_proj)

    actions[action_ids.NEW_PROJECT_PROMPT]("m3", "chat-1", None, {"action": "new_project_prompt"})
    handlers_map["coco"].reply_text.assert_called_once()

    actions[action_ids.CONFIRM_FORCE_RELEASE]("m4", "chat-1", "p1", {"action": "confirm_force_release"})
    handlers_map["system"].handle_confirm_force_release.assert_called_once()

    actions[action_ids.CONFIRM_LOCK]("m5", "chat-1", "p1", {"action": "confirm_lock"})
    handlers_map["system"].handle_confirm_lock.assert_called_once()

    # Valid v2 chat-bound retry executes once; replay & cross-chat fail closed
    sig = sign_command("/status", "chat-1")
    handlers_map["system"].is_interceptable_command_match.return_value = True
    actions[action_ids.RETRY_COMMAND]("m6", "chat-1", "p1", {"action": "retry_command", "_t": "/status", "_s": sig})
    handlers_map["system"].handle_intercepted_command.assert_called_once()

    # Replay same nonce -> rejected
    actions[action_ids.RETRY_COMMAND]("m7", "chat-1", "p1", {"action": "retry_command", "_t": "/status", "_s": sig})
    assert handlers_map["system"].handle_intercepted_command.call_count == 1
    handlers_map["coco"].reply_error.assert_called()

    # Cross-chat signature -> rejected
    sig_other = sign_command("/status", "chat-other")
    actions[action_ids.RETRY_COMMAND]("m8", "chat-1", "p1", {"action": "retry_command", "_t": "/status", "_s": sig_other})
    assert handlers_map["system"].handle_intercepted_command.call_count == 1
