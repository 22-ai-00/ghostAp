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

    sent_replies: list[tuple[str, str, str]] = []
    im_client = SimpleNamespace(
        reply_message=lambda mid, content, msg_type="text", **_kw: (
            sent_replies.append((mid, msg_type, content))
            or SimpleNamespace(
                success=lambda: True,
                data=SimpleNamespace(message_id=f"om_reply_{len(sent_replies)}"),
            )
        )
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
        settings=SimpleNamespace(default_reply_mode="reply", ref_note_enabled=False),
        im_client=im_client,
        project_manager=SimpleNamespace(get_project_for_diagnostics=lambda _pid: project),
        message_linker=SimpleNamespace(
            resolve_origin=lambda mid: mid,
            get_request_id=lambda _mid: "req_1",
            ensure_request_id=lambda _mid, **_kw: "req_1",
            register_origin=lambda *_a, **_kw: None,
            get_binding=lambda _mid: None,
            link_reply_message=lambda *_a, **_kw: None,
            link_reply=lambda *_a, **_kw: None,
        ),
    )
    diag.im_client = im_client
    diag.get_working_dir = lambda _c: str(repo)

    diag.show_unified_status("m1", "chat-1", "/status", project)
    assert len(sent_replies) == 1
    assert sent_replies[0][:2] == ("m1", "interactive")
    card = json.loads(sent_replies[0][2])
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
    assert len(sent_replies) == 2
    assert sent_replies[1][:2] == ("m2", "interactive")
    assert "build-api" in sent_replies[1][2]


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


def test_base_handler_reply_interactive_card_dispatches_card_and_text() -> None:
    from src.feishu.handlers.base import BaseHandler

    calls: list[tuple[str, str, object]] = []
    handler = object.__new__(BaseHandler)
    handler.reply_card = lambda mid, card, reply_in_thread=None: (
        calls.append(("card", mid, card)) or "om_card"
    )
    handler.reply_text = lambda mid, txt, reply_in_thread=None: (
        calls.append(("text", mid, txt)) or "om_text"
    )

    assert handler.reply_interactive_card("m1", '{"schema":"2.0"}') == "om_card"
    assert handler.reply_interactive_card("m2", "plain fallback", msg_type="text") == "om_text"
    assert calls == [
        ("card", "m1", '{"schema":"2.0"}'),
        ("text", "m2", "plain fallback"),
    ]


def test_ws_client_dispatch_surfaces_attribute_and_lookup_errors(monkeypatch) -> None:
    from src.card.ui_text import UI_TEXT
    from src.feishu.ws_client import FeishuWSClient

    for err in (AttributeError("missing method"), KeyError("missing key")):
        replies: list[tuple[str, str]] = []
        ws = object.__new__(FeishuWSClient)
        ws.settings = SimpleNamespace(thread_programming_enabled=False)
        ws._extract_canonical_ingress_facts = lambda _d: ("m_err", "oc_err", "p2p", "ou_1")
        ws._validate_message = lambda *_a, **_kw: True
        ws._parse_message_content = lambda _m: ("/status", False)
        ws._build_route_decision = lambda *_a, **_kw: SimpleNamespace(
            trusted_project=None,
            auto_enter_mode=None,
            effective_trust=None,
            command_match=None,
        )
        ws._project_manager = SimpleNamespace(
            get_active_project=lambda _c: None,
            get_project_for_chat=lambda *_a, **_kw: None,
        )
        ws._current_trust_can_dispatch = lambda *_a, **_kw: True
        ws._pending_image_lock = __import__("threading").Lock()
        ws._pending_image_only = set()
        ws._pending_image_keys = {}
        ws._handler_ctx = SimpleNamespace(
            handlers={
                "coco": SimpleNamespace(
                    reply_text=lambda mid, txt: replies.append((mid, txt)),
                    ensure_request_id=lambda mid, **_kw: f"req_{mid}",
                )
            }
        )
        ws._dispatch_message_logic = MagicMock(side_effect=err)
        ws._message_ingress_guard = MagicMock()

        event = SimpleNamespace(
            event=SimpleNamespace(
                sender=SimpleNamespace(
                    sender_id=SimpleNamespace(open_id="ou_1", union_id="on_1", user_id="u_1"),
                    sender_type="user",
                ),
                message=SimpleNamespace(
                    message_id="m_err",
                    chat_id="oc_err",
                    chat_type="p2p",
                    message_type="text",
                    content='{"text":"/status"}',
                    root_id=None,
                    parent_id=None,
                    mentions=[],
                ),
            )
        )
        ws._process_message_async(event)
        assert replies == [("m_err", UI_TEXT["ws_message_internal_error"])]
