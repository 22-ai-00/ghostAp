"""Tests for Batch 4: Workspace Cockpit /status, /status <wf_id>, /info, and help/banner updates."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.acp.models import ACPUsageInfo
from src.card.actions import dispatch as action_ids
from src.feishu.handlers.diagnostics import DiagnosticsHandler
from src.feishu.handlers.system import SystemHandler
from src.feishu.product_catalog import HIDDEN_SLASH_COMMANDS, resolve_command
from src.mode import InteractionMode
from src.project.context import ProjectContext


def test_workspace_cockpit_status_and_git_branch_and_buttons(tmp_path) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/feat/cockpit\n", encoding="utf-8")

    # Worktree .git file resolution
    wt_root = tmp_path / "wt"
    wt_root.mkdir()
    wt_gitdir = tmp_path / "actual_gitdir"
    wt_gitdir.mkdir()
    (wt_gitdir / "HEAD").write_text("0123456789abcdef\n", encoding="utf-8")
    (wt_root / ".git").write_text(f"gitdir: {wt_gitdir}\n", encoding="utf-8")

    assert DiagnosticsHandler._read_git_branch(str(repo)) == "feat/cockpit"
    assert DiagnosticsHandler._read_git_branch(str(wt_root)) == "detached@0123456"

    project = ProjectContext(project_id="p1", project_name="cockpit-app", root_path=str(repo))
    project.acp_tool_name = "agy"
    project.acp_model_name = "claude-sonnet-4"

    live_session = SimpleNamespace(
        session_id="sess-cockpit-1234",
        message_count=6,
        _model_name="claude-sonnet-4",
        get_usage_info=lambda: ACPUsageInfo(used=12400, size=200000),
    )
    agy_mgr = SimpleNamespace(get_session=lambda *_a, **_kw: live_session)
    agy_handler = SimpleNamespace(
        _get_active_programming_run=lambda *_a, **_kw: SimpleNamespace(started_at=time.monotonic() - 15.0),
        _get_session_manager=lambda: agy_mgr,
    )

    wf_engine = SimpleNamespace(
        engine_name="Workflow",
        workflow_project=SimpleNamespace(
            workflow_id="wf_91a2c3",
            name="build-api",
            status=SimpleNamespace(value="running"),
            phases=[SimpleNamespace(status=SimpleNamespace(value="completed")), SimpleNamespace(status=SimpleNamespace(value="running"))],
            started_at=time.time(),
            metrics=SimpleNamespace(completed_agents=1, failed_agents=0),
        ),
    )

    diag = object.__new__(DiagnosticsHandler)
    diag.ctx = SimpleNamespace(
        mode_manager=SimpleNamespace(
            get_mode=lambda *_a, **_kw: InteractionMode.AGY,
            is_programming_mode=lambda *_a, **_kw: True,
        ),
        handlers={"agy": agy_handler},
        deep_engine_manager=SimpleNamespace(list_engines=lambda _c: []),
        spec_engine_manager=SimpleNamespace(list_engines=lambda _c: []),
        workflow_engine_manager=SimpleNamespace(list_engines=lambda _c: [wf_engine]),
        chat_lock_manager=None,
        repo_lock_manager=None,
    )
    diag.reply_interactive_card = MagicMock()
    diag.reply_card = MagicMock()
    diag.reply_text = MagicMock()
    diag.get_working_dir = lambda _c: str(repo)

    diag.show_unified_status("m1", "chat-1", "/status", project)
    diag.reply_interactive_card.assert_called_once()
    card = json.loads(diag.reply_interactive_card.call_args[0][1])
    assert card["schema"] == "2.0"
    dump = json.dumps(card, ensure_ascii=False)
    assert "feat/cockpit" in dump
    assert "AGY" in dump
    assert "claude-sonnet-4" in dump
    assert "12,400/200,000 (6%)" in dump
    assert "build-api" in dump
    assert action_ids.OPEN_ACP_MODEL_PICKER in dump
    assert action_ids.SHOW_TOOL_COMMANDS in dump
    assert action_ids.PROGRAMMING_STOP in dump
    assert action_ids.SHOW_BOARD in dump

    # Verify Card 2.0 behaviors on every button
    stack = [card]
    buttons = []
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if node.get("tag") == "button":
                buttons.append(node)
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    assert len(buttons) == 4
    for btn in buttons:
        assert btn["behaviors"] == [{"type": "callback", "value": btn["value"]}]

    # /status <wf_id> shows Workflow task detail card
    diag._show_task_detail("m2", "chat-1", "wf_91a2c3", project)
    diag.reply_card.assert_called_once()
    assert "build-api" in diag.reply_card.call_args[0][1]


def test_universal_info_command_delegates_based_on_mode() -> None:
    assert "/info" in HIDDEN_SLASH_COMMANDS
    assert resolve_command("/info") is not None

    agy_handler = MagicMock()
    diag_handler = MagicMock()
    project = ProjectContext(project_id="p1", project_name="demo", root_path="/tmp/demo")

    system = object.__new__(SystemHandler)
    system.ctx = SimpleNamespace(
        handlers={"agy": agy_handler, "diagnostics": diag_handler},
    )
    system._resolve_active_programming_mode_key = MagicMock(return_value="agy")

    system.handle_info_command("m1", "chat-1", project)
    agy_handler.show_info.assert_called_once_with("m1", "chat-1", project)
    diag_handler.show_unified_status.assert_not_called()

    system._resolve_active_programming_mode_key.return_value = None
    system.handle_info_command("m2", "chat-1", project)
    diag_handler.show_unified_status.assert_called_once_with("m2", "chat-1", "/status", project)
