"""Targeted regression tests for M5 Phase 1 remediation items."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.card.actions import dispatch as action_ids
from src.feishu.handlers.workflow import WorkflowHandler
from src.project import ProjectManager, get_project_workflow_pending_requirement
from src.workflow_engine.bridge import RuntimeBridge
from src.workflow_engine.engine import WorkflowEngine
from src.workflow_engine.models import WorkflowProject, WorkflowStatus
from src.workflow_engine.storage import remove_owned_workflow_artifact
from src.workflow_engine.tool_registry import ToolCapability


def _make_handler(project_manager: ProjectManager, repo_root: Path) -> WorkflowHandler:
    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.ctx.settings = SimpleNamespace(admin_user_ids=[], default_reply_mode="reply")
    handler.ctx.project_manager = project_manager
    handler.reply_text = MagicMock(return_value="om_reply_txt")
    handler.reply_card = MagicMock(return_value="om_reply_card")
    handler.update_card = MagicMock(return_value=True)
    handler.send_card_to_chat = MagicMock(return_value="om_send_card")
    handler.add_reaction = MagicMock()
    handler._resolve_origin = MagicMock(side_effect=lambda mid: mid)
    handler._ensure_topic_engine_context = MagicMock()
    handler.get_working_dir = MagicMock(return_value=str(repo_root))
    handler.get_engine_name = MagicMock(return_value="wf-test")
    return handler


def test_remove_owned_workflow_artifact_cleans_verified_and_confirmed_without_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    temp_dir = tmp_path / "tmp_root"
    temp_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(temp_dir))

    verified = temp_dir / "ghostap-verified-abc123.js"
    confirmed = temp_dir / "ghostap-confirmed-def456.js"
    foreign = temp_dir / "other-script.js"
    outside = outside_dir / "ghostap-verified-escape.js"
    for f in (verified, confirmed, foreign, outside):
        f.write_text("// js\n", encoding="utf-8")

    caplog.set_level(logging.WARNING)
    remove_owned_workflow_artifact(str(verified), root_path=str(tmp_path), warn_on_refusal=True)
    remove_owned_workflow_artifact(str(confirmed), root_path=str(tmp_path), warn_on_refusal=True)
    assert not verified.exists()
    assert not confirmed.exists()
    assert not caplog.records

    remove_owned_workflow_artifact(str(foreign), root_path=str(tmp_path), warn_on_refusal=True)
    remove_owned_workflow_artifact(str(outside), root_path=str(tmp_path), warn_on_refusal=True)
    assert foreign.exists()
    assert outside.exists()
    assert len(caplog.records) == 2


def test_wf_config_cache_miss_rehydration_preserves_orchestrator_on_save_and_start(
    tmp_path: Path,
) -> None:
    mgr = ProjectManager(str(tmp_path / "projects.json"))
    ok, _, proj = mgr.create_project("p1", "Proj1", str(tmp_path / "repo"), chat_id="oc_orch")
    assert ok and proj is not None
    handler = _make_handler(mgr, tmp_path / "repo")

    caps = {
        "codex": ToolCapability(
            tool_name="codex",
            display_name="Codex",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=("gpt-5.4",),
            efforts_by_model={"gpt-5.4": ("low", "high", "xhigh")},
            default_effort="high",
        ),
        "claude": ToolCapability(
            tool_name="claude",
            display_name="Claude",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=("sonnet",),
            efforts_by_model={"sonnet": ("low", "high")},
            default_effort="high",
        ),
    }
    handler._load_workflow_capabilities = MagicMock(return_value=(caps, {"codex": "Codex", "claude": "Claude"}))

    # Build initial card and persist a 2-member team with A1 initially orchestrator
    handler.show_workflow_config("om_cmd", "oc_orch", proj, requirement="Check orchestrator persistence")
    # Save initial team via callback so saved_cfg exists with A1 as orchestrator
    handler.handle_workflow_config_callback_sync(
        "om_reply_card",
        "oc_orch",
        proj.project_id,
        {"action": action_ids.WF_SAVE_TEAM, "chat_id": "oc_orch", "project_id": proj.project_id},
    )

    # Simulate cache eviction / service restart before WF_SAVE_TEAM with A2 in card snapshot
    handler._wf_config_sessions.clear()
    handler._wf_config_message_index.clear()
    _card, toast_type, _toast = handler.handle_workflow_config_callback_sync(
        "om_reply_card",
        "oc_orch",
        proj.project_id,
        {
            "action": action_ids.WF_SAVE_TEAM,
            "chat_id": "oc_orch",
            "project_id": proj.project_id,
            "agent_id": "A2",
            "orchestrator_agent_id": "A2",
            "role": "verifier",
            "tool_name": "claude",
            "model_name": "sonnet",
            "effort": "high",
        },
    )
    assert toast_type == "success"
    saved = proj.get_workflow_team_config("oc_orch")
    assert saved is not None
    assert saved["orchestrator_agent_id"] == "A2"

    # Also verify invalid orchestrator ID is rejected (falls back to saved A2)
    handler._wf_config_sessions.clear()
    handler._wf_config_message_index.clear()
    _card2, toast_type2, _toast2 = handler.handle_workflow_config_callback_sync(
        "om_reply_card",
        "oc_orch",
        proj.project_id,
        {
            "action": action_ids.WF_SELECT_ROLE,
            "chat_id": "oc_orch",
            "project_id": proj.project_id,
            "agent_id": "A1",
            "orchestrator_agent_id": "A99",
            "role": "scout",
            "tool_name": "codex",
            "model_name": "gpt-5.4",
            "effort": "high",
        },
    )
    assert toast_type2 == "info"
    rehydrated = next(iter(handler._wf_config_sessions.values()))
    assert rehydrated.orchestrator_agent_id == "A2"

    # Verify real un-mocked flow: /wf_config <requirement> -> WF_SELECT_AGENT(A2) ->
    # WF_SET_ORCHESTRATOR -> clear caches & reload ProjectManager from disk ->
    # WF_START_WORKFLOW launches cleanly with preserved requirement AND orchestrator_agent_id == "A2".
    def _extract_action_payload(card_obj: dict, target_action: str) -> dict:
        stack: list[object] = [card_obj]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                behaviors = node.get("behaviors")
                if isinstance(behaviors, list):
                    for beh in behaviors:
                        if isinstance(beh, dict) and beh.get("type") == "callback":
                            val = beh.get("value")
                            if isinstance(val, dict) and val.get("action") == target_action:
                                return dict(val)
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
        raise AssertionError(f"Action payload {target_action!r} not found in card")

    # Reset saved orchestrator back to A1 so A2 comes strictly from WF_SET_ORCHESTRATOR + card payload
    mgr.commit_workflow_team_configuration(
        proj,
        {**saved, "orchestrator_agent_id": "A1"},
        chat_id="oc_orch",
    )
    handler._wf_config_sessions.clear()
    handler._wf_config_message_index.clear()
    handler.reply_card.reset_mock()

    handler.handle_workflow_command(
        "om_cmd_real",
        "oc_orch",
        "/wf_config Launch check via real flow",
        proj,
    )
    handler.reply_card.assert_called_once()
    initial_card = handler.reply_card.call_args.args[1]
    select_agent_payload = _extract_action_payload(initial_card, action_ids.WF_SELECT_AGENT)
    cfg_sid = select_agent_payload["config_session_id"]
    assert "requirement" not in select_agent_payload
    assert get_project_workflow_pending_requirement(proj, cfg_sid, "oc_orch") == "Launch check via real flow"

    card_after_sel, toast_sel, _ = handler.handle_workflow_config_callback_sync(
        "om_reply_card",
        "oc_orch",
        proj.project_id,
        {**select_agent_payload, "_option": "A2"},
    )
    assert toast_sel == "info"
    assert card_after_sel is not None

    set_orch_payload = _extract_action_payload(card_after_sel, action_ids.WF_SET_ORCHESTRATOR)
    card_after_orch, toast_orch, _ = handler.handle_workflow_config_callback_sync(
        "om_reply_card",
        "oc_orch",
        proj.project_id,
        set_orch_payload,
    )
    assert toast_orch == "success"
    assert card_after_orch is not None
    start_payload = _extract_action_payload(card_after_orch, action_ids.WF_START_WORKFLOW)
    assert start_payload["orchestrator_agent_id"] == "A2"
    assert "requirement" not in start_payload

    # Evict in-memory config caches AND reload ProjectManager from disk
    handler._wf_config_sessions.clear()
    handler._wf_config_message_index.clear()
    reloaded_mgr = ProjectManager(str(tmp_path / "projects.json"))
    handler.ctx.project_manager = reloaded_mgr
    reloaded_proj = reloaded_mgr.get_project_for_chat(proj.project_id, "oc_orch")
    assert reloaded_proj is not None
    assert get_project_workflow_pending_requirement(reloaded_proj, cfg_sid, "oc_orch") == "Launch check via real flow"

    handler.start_workflow = MagicMock()
    handler.handle_workflow_config_action(
        "om_reply_card",
        "oc_orch",
        reloaded_proj.project_id,
        start_payload,
    )
    handler.start_workflow.assert_called_once()
    assert handler.start_workflow.call_args.args[2] == "Launch check via real flow"
    assert handler.start_workflow.call_args.kwargs["override_orchestrator_id"] == "A2"
    assert handler.start_workflow.call_args.kwargs["reuse_card_message_id"] == "om_reply_card"

    # Pending requirement is consumed on launch so replay after cache eviction does not relaunch
    assert get_project_workflow_pending_requirement(reloaded_proj, cfg_sid, "oc_orch") is None
    handler._wf_config_sessions.clear()
    handler._wf_config_message_index.clear()
    _replay_card, replay_toast_type, _replay_toast, replay_launch = handler._apply_config_transition(
        "om_reply_card",
        "oc_orch",
        reloaded_proj.project_id,
        start_payload,
    )
    assert replay_toast_type == "warning"
    assert replay_launch is None

    # Explicit WF_OPEN_CONFIG without requirement and unrelated /wf_config without requirement stay empty
    handler.reply_card.reset_mock()
    handler.show_workflow_config("om_cmd_empty", "oc_orch", reloaded_proj)
    empty_card = handler.reply_card.call_args.args[1]
    assert "未指定任务" in str(empty_card)

    open_card, open_toast, _, open_launch = handler._apply_config_transition(
        "om_help_card",
        "oc_orch",
        reloaded_proj.project_id,
        {"action": action_ids.WF_OPEN_CONFIG, "chat_id": "oc_orch", "project_id": reloaded_proj.project_id},
    )
    assert open_toast == "info"
    assert open_launch is None
    assert "未指定任务" in str(open_card)

    # Chat & project isolation + bounded LRU eviction for pending requirements
    reloaded_proj.add_chat_id("oc_other")
    for idx in range(40):
        reloaded_mgr.commit_workflow_pending_requirement(
            reloaded_proj,
            f"sid_{idx}",
            f"req_{idx}",
            chat_id="oc_orch",
        )
    assert get_project_workflow_pending_requirement(reloaded_proj, "sid_0", "oc_orch") is None
    assert get_project_workflow_pending_requirement(reloaded_proj, "sid_39", "oc_orch") == "req_39"
    assert get_project_workflow_pending_requirement(reloaded_proj, "sid_39", "oc_other") is None


def test_engine_get_status_text_and_wf_status_surface_generation_failure_and_cancellation(
    tmp_path: Path,
) -> None:
    mgr = ProjectManager(str(tmp_path / "projects.json"))
    ok, _, proj = mgr.create_project("p1", "Proj1", str(tmp_path / "repo"), chat_id="oc_stat")
    assert ok and proj is not None
    handler = _make_handler(mgr, tmp_path / "repo")

    engine = WorkflowEngine.__new__(WorkflowEngine)
    engine._lock = __import__("threading").RLock()
    engine._renderer_wf = None
    engine._state_manager = None
    engine._project = WorkflowProject(
        requirement="test req",
        status=WorkflowStatus.FAILED,
        error="脚本生成超时 api_key=sk-secret123456",
    )
    handler.ctx.workflow_engine_manager = MagicMock()
    handler.ctx.workflow_engine_manager.get.return_value = engine

    failed_text = engine.get_status_text()
    assert failed_text != "Workflow: idle"
    assert "失败" in failed_text
    assert "sk-secret123456" not in failed_text

    handler.show_workflow_status("om_status_fail", "oc_stat", proj)
    handler.reply_text.assert_called_with("om_status_fail", failed_text)

    engine._project.status = WorkflowStatus.CANCELLED
    cancelled_text = engine.get_status_text()
    assert cancelled_text != "Workflow: idle"
    assert "取消" in cancelled_text

    handler.show_workflow_status("om_status_cancel", "oc_stat", proj)
    handler.reply_text.assert_called_with("om_status_cancel", cancelled_text)


def test_runtime_bridge_start_handles_non_utf8_stderr_bytes_without_dropping_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_script = tmp_path / "fake_runtime.js"
    runtime_script.write_text(
        f"""#!{sys.executable}
import json, sys, time
sys.stderr.buffer.write(b"before\\n\\xff\\xfe\\nAFTER_BAD_BYTES\\n")
sys.stderr.buffer.flush()
sys.stdout.write(json.dumps({{"jsonrpc": "2.0", "method": "ready", "params": {{}}}}) + "\\n")
sys.stdout.flush()
time.sleep(0.2)
""",
        encoding="utf-8",
    )
    user_script = tmp_path / "workflow.js"
    user_script.write_text("// dummy\n", encoding="utf-8")

    monkeypatch.setattr("src.workflow_engine.bridge.RUNTIME_JS_PATH", str(runtime_script))
    monkeypatch.setattr("src.workflow_engine.bridge.shutil.which", lambda _name: sys.executable)

    bridge = RuntimeBridge(
        script_path=str(user_script),
        cwd=str(tmp_path),
    )
    try:
        bridge.start()
        stderr_thread = vars(bridge).get("_stderr_thread")
        if stderr_thread is not None:
            stderr_thread.join(timeout=2.0)
        tail = bridge._stderr_tail_text()
        assert "before" in tail
        assert "AFTER_BAD_BYTES" in tail
    finally:
        bridge.stop()


def test_dsh_cold_cache_default_model_with_effort_roundtrips_through_sync_adapter() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.acp.sync_adapter import SyncACPSession
    from src.workflow_engine.agent_pool import WorkflowAgentBinding, resolve_effective_model_name
    from src.workflow_engine.tool_registry import ToolCapability, validate_agent_binding_spec

    effective = resolve_effective_model_name("dsh", "default", effort="high")
    assert effective == '["default","default","high"]'
    assert resolve_effective_model_name("dsh", effective, effort="high") == effective
    assert resolve_effective_model_name("dsh", "default", effort=None) == "default"

    binding = WorkflowAgentBinding(
        agent_id="A1",
        tool_name="dsh",
        model_name="default",
        display_name="DSH",
        effort="high",
    )
    assert binding.effective_model_name() == '["default","default","high"]'

    caps = {
        "dsh": ToolCapability(
            tool_name="dsh",
            display_name="DSH",
            available=True,
            supports_model=True,
            supports_effort=True,
        )
    }
    assert validate_agent_binding_spec("dsh", model_name="default", effort="high", capabilities=caps) == (True, None)

    session = object.__new__(SyncACPSession)
    session._acp_session = AsyncMock()
    session._acp_session.set_config_option = AsyncMock(return_value=True)

    assert asyncio.run(session._apply_dsh_selection(effective)) is True
    session._acp_session.set_config_option.assert_awaited_with(
        "dsh.reasoning_effort", '["effort","high"]'
    )

    session._acp_session.set_config_option.reset_mock()
    assert asyncio.run(session._apply_dsh_selection("default")) is True
    session._acp_session.set_config_option.assert_not_awaited()


def test_traex_default_model_with_profile_and_effort_applies_both_config_options() -> None:
    import asyncio
    from unittest.mock import AsyncMock, call

    from src.acp.sync_adapter import SyncACPSession

    session = object.__new__(SyncACPSession)
    session._acp_session = AsyncMock()
    session._acp_session.set_config_option = AsyncMock(return_value=True)

    assert asyncio.run(session._apply_traex_selection("default/max/high")) is True
    assert session._acp_session.set_config_option.await_args_list == [
        call("profile", "max"),
        call("reasoning_effort", "high"),
    ]

    session._acp_session.set_config_option.reset_mock()
    assert asyncio.run(session._apply_traex_selection("default/max")) is True
    assert session._acp_session.set_config_option.await_args_list == [
        call("profile", "max"),
    ]


def test_workflow_storage_and_cleanup_preserve_symlink_parity(tmp_path: Path) -> None:
    import os

    from src.workflow_engine.storage import (
        workflow_project_storage_root,
        workflow_scripts_dir,
    )

    real_root = tmp_path / "private_var_tmp" / "my_proj"
    real_root.mkdir(parents=True)
    link_parent = tmp_path / "var_tmp_link"
    link_parent.symlink_to(tmp_path / "private_var_tmp")
    link_proj = link_parent / "my_proj"

    assert workflow_project_storage_root(str(link_proj)) == workflow_project_storage_root(
        str(real_root)
    )
    script_dir = Path(workflow_scripts_dir(str(real_root)))
    script_dir.mkdir(parents=True, exist_ok=True)
    artifact = script_dir / "generated-workflow-symlink-test.js"
    artifact.write_text("// test\n", encoding="utf-8")
    try:
        remove_owned_workflow_artifact(str(artifact), root_path=str(link_proj))
        assert not artifact.exists()
    finally:
        if artifact.exists():
            os.remove(artifact)


def test_show_workflow_status_after_terminal_completion_does_not_duplicate_cards(
    tmp_path: Path,
) -> None:
    mgr = ProjectManager(str(tmp_path / "projects.json"))
    ok, _, proj = mgr.create_project("p1", "Proj1", str(tmp_path / "repo"), chat_id="oc_done")
    assert ok and proj is not None
    handler = _make_handler(mgr, tmp_path / "repo")

    engine = MagicMock()
    engine._lock = __import__("threading").RLock()
    engine._workflow_start_owner = None
    engine.project = WorkflowProject(requirement="done req", status=WorkflowStatus.COMPLETED)
    engine.workflow_project = engine.project
    engine.get_status_text.return_value = "✅ 任务: done req | 已完成"
    engine.get_progress_cards.return_value = [
        {"header": {"title": {"tag": "plain_text", "content": "Done"}}, "elements": []}
    ]
    handler.ctx.workflow_engine_manager = MagicMock()
    handler.ctx.workflow_engine_manager.get.return_value = engine

    handler.show_workflow_status("om_status_done", "oc_done", proj)
    handler.send_card_to_chat.assert_not_called()
    handler.reply_card.assert_not_called()
    handler.reply_text.assert_called_once_with("om_status_done", "✅ 任务: done req | 已完成")


def test_runtime_bridge_start_kills_subprocess_when_cancelled_after_init(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    runtime_script = tmp_path / "fake_runtime_cancel.py"
    runtime_script.write_text(
        f"""#!{sys.executable}
import json, sys, time
sys.stdout.write(json.dumps({{"jsonrpc": "2.0", "method": "ready", "params": {{}}}}) + "\\n")
sys.stdout.flush()
for _line in sys.stdin:
    time.sleep(5.0)
""",
        encoding="utf-8",
    )
    user_script = tmp_path / "workflow.js"
    user_script.write_text("// dummy\n", encoding="utf-8")

    monkeypatch.setattr("src.workflow_engine.bridge.RUNTIME_JS_PATH", str(runtime_script))
    monkeypatch.setattr("src.workflow_engine.bridge.shutil.which", lambda _name: sys.executable)

    cancel_event = threading.Event()
    bridge = RuntimeBridge(
        script_path=str(user_script),
        cwd=str(tmp_path),
        cancel_event=cancel_event,
    )
    orig_send = bridge._send

    def send_and_cancel(payload: dict) -> None:
        orig_send(payload)
        if payload.get("method") == "init":
            cancel_event.set()

    bridge._send = send_and_cancel  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="cancelled during start"):
        bridge.start()
    proc = bridge._process
    assert proc is None or proc.poll() is not None


def test_agent_pool_and_registry_default_sentinel_and_openrouter_round_trips() -> None:
    from unittest.mock import patch

    from src.workflow_engine.agent_pool import WorkflowAgentBinding, resolve_effective_model_name
    from src.workflow_engine.tool_registry import validate_agent_binding_spec

    caps = {
        "codex": ToolCapability(tool_name="codex", display_name="Codex", available=True, supports_model=True, supports_effort=True),
        "claude": ToolCapability(tool_name="claude", display_name="Claude", available=True, supports_model=True, supports_effort=True),
        "traex": ToolCapability(tool_name="traex", display_name="Traex", available=True, supports_model=True, supports_effort=True),
        "agy": ToolCapability(tool_name="agy", display_name="AGY", available=True, supports_model=True, supports_effort=True),
        "dsh": ToolCapability(tool_name="dsh", display_name="DSH", available=True, supports_model=True, supports_effort=True),
        "coco": ToolCapability(
            tool_name="coco",
            display_name="Coco",
            available=True,
            supports_model=True,
            supports_effort=False,
            models=("anthropic/claude-sonnet", "openai/gpt-5"),
        ),
    }
    with patch("src.acp.traex_selection.load_traex_model_metadata", return_value=()):
        for tool, expected in (
            ("codex", "default/high"),
            ("claude", "default/high"),
            ("traex", "default/standard/high"),
            ("agy", "default/high"),
            ("dsh", '["default","default","high"]'),
        ):
            assert validate_agent_binding_spec(tool, model_name="default", effort="high", capabilities=caps) == (True, None)
            resolved = resolve_effective_model_name(tool, "default", effort="high")
            assert resolved == expected
            assert resolve_effective_model_name(tool, resolved, effort="high") == expected

    or_binding = WorkflowAgentBinding(
        agent_id="A1",
        tool_name="coco",
        model_name="anthropic/claude-sonnet",
        display_name="Coco",
    )
    assert or_binding.effective_model_name() == "anthropic/claude-sonnet"
    assert validate_agent_binding_spec("coco", model_name="anthropic/claude-sonnet", capabilities=caps) == (True, None)

