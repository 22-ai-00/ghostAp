"""Targeted unit and integration tests for Milestone M4 (F9, F10 & bug fixes)."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.card.actions import dispatch as action_ids
from src.feishu.action_registry import init_action_registry
from src.feishu.handlers.diagnostics_helper import DiagnosticsHelper
from src.feishu.handlers.system import SystemHandler
from src.feishu.handlers.workflow import WorkflowHandler, _WorkflowLifecycleOwner
from src.feishu.handlers.workflow_card_pages import _without_stop_actions
from src.project import ProjectContext, ProjectManager, get_project_workflow_team_config
from src.workflow_engine.agent_pool import WorkflowAgentBinding
from src.workflow_engine.commands import LONG_COMMANDS, SHORT_COMMANDS
from src.workflow_engine.models import (
    AgentProgress,
    AgentStatus,
    PhaseProgress,
    WorkflowProject,
    WorkflowStatus,
)
from src.workflow_engine.process_stream import WorkflowProcessStream
from src.workflow_engine.renderer import WorkflowGenerationRenderer


def _make_workflow_handler() -> WorkflowHandler:
    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.ctx.settings = SimpleNamespace(admin_user_ids=[], default_reply_mode="reply")
    handler.reply_text = MagicMock(return_value="om_reply_txt")
    handler.reply_card = MagicMock(return_value="om_reply_card")
    handler.update_card = MagicMock(return_value=True)
    handler.send_card_to_chat = MagicMock(return_value="om_send_card")
    handler.add_reaction = MagicMock()
    handler._resolve_origin = MagicMock(side_effect=lambda mid: mid)
    handler._ensure_topic_engine_context = MagicMock()
    handler.get_working_dir = MagicMock(return_value="/tmp/repo")
    handler.get_engine_name = MagicMock(return_value="wf-test")
    return handler


def test_f9_workflow_team_config_persistence_isolation_and_rollback(tmp_path: Path) -> None:
    storage_file = tmp_path / "projects.json"
    mgr = ProjectManager(str(storage_file))
    ok, _, proj = mgr.create_project("p1", "Proj1", str(tmp_path / "repo1"), chat_id="oc_chat_1")
    assert ok and proj is not None
    proj.add_chat_id("oc_chat_2")
    mgr.commit_acp_configuration(proj, tool_name="codex", model_name="gpt-5.4", chat_id="oc_chat_1")

    cfg = {
        "preset_id": "review_team",
        "orchestrator_agent_id": "A2",
        "bindings": [
            {
                "agent_id": "A1",
                "tool_name": "codex",
                "model_name": "gpt-5.4",
                "display_name": "Codex Low",
                "effort": "low",
                "role": "worker",
                "roles": ["worker"],
            },
            {
                "agent_id": "A2",
                "tool_name": "codex",
                "model_name": "gpt-5.4",
                "display_name": "Codex XHigh",
                "effort": "xhigh",
                "role": "verifier",
                "roles": ["verifier"],
            },
        ],
        "updated_at": 1700000000.0,
    }
    assert mgr.commit_workflow_team_configuration(proj, cfg, chat_id="oc_chat_1")

    # Chat isolation + ACP isolation
    assert get_project_workflow_team_config(proj, "oc_chat_1") is not None
    assert get_project_workflow_team_config(proj, "oc_chat_2") is None
    assert proj.get_acp_tool_name("oc_chat_1") == "codex"
    assert proj.get_acp_model_name("oc_chat_1") == "gpt-5.4"

    # Deep copy in ChatProjectState.clone()
    cloned = proj.chat_states["oc_chat_1"].clone()
    assert cloned.workflow_team_config == proj.chat_states["oc_chat_1"].workflow_team_config
    cloned.workflow_team_config["preset_id"] = "quick"
    assert proj.get_workflow_team_config("oc_chat_1")["preset_id"] == "review_team"

    # Rollback on _save_projects() failure
    with patch.object(mgr, "_save_projects", return_value=False):
        assert not mgr.commit_workflow_team_configuration(proj, None, chat_id="oc_chat_1")
    assert proj.get_workflow_team_config("oc_chat_1")["preset_id"] == "review_team"

    # Reload from disk preserves multi-effort bindings cleanly
    reloaded = ProjectManager(str(storage_file))
    reloaded_proj = reloaded.get_project_for_chat("p1", "oc_chat_1")
    assert reloaded_proj is not None
    saved = reloaded_proj.get_workflow_team_config("oc_chat_1")
    assert saved is not None
    assert saved["preset_id"] == "review_team"
    assert saved["orchestrator_agent_id"] == "A2"
    assert [b["effort"] for b in saved["bindings"]] == ["low", "xhigh"]


def test_f9_corrupt_and_stale_workflow_team_config_recovers_gracefully() -> None:
    assert ProjectContext.normalize_workflow_team_config("bad-string") is None
    assert ProjectContext.normalize_workflow_team_config({"bindings": "not-a-list"}) is None
    assert ProjectContext.normalize_workflow_team_config({"bindings": []}) is None

    # Unknown tool dropped, invalid effort stripped from valid tool, invalid orch falls back
    recovered = ProjectContext.normalize_workflow_team_config(
        {
            "preset_id": "unknown_preset_xyz",
            "orchestrator_agent_id": "A_MISSING",
            "bindings": [
                {"agent_id": "A0", "tool_name": "nonexistent_tool", "display_name": "Bad"},
                {
                    "agent_id": "A1",
                    "tool_name": "codex",
                    "model_name": None,
                    "display_name": "Codex",
                    "effort": "bogus_effort",
                    "role": "worker",
                },
                {
                    "agent_id": "A1",
                    "tool_name": "claude",
                    "display_name": "DupID",
                },
            ],
        }
    )
    assert recovered is not None
    assert recovered["preset_id"] == "recommended"
    assert recovered["orchestrator_agent_id"] == "A1"
    assert len(recovered["bindings"]) == 1
    assert recovered["bindings"][0]["tool_name"] == "codex"
    assert recovered["bindings"][0]["effort"] is None


def test_f10_commands_and_wf_preset_flag_parsing(tmp_path: Path) -> None:
    assert "/wf_config" in SHORT_COMMANDS
    assert "/workflow_config" in LONG_COMMANDS

    handler = _make_workflow_handler()
    project = ProjectContext(project_id="p1", project_name="P1", root_path=str(tmp_path))
    handler._ensure_project = MagicMock(return_value=project)
    handler.show_workflow_config = MagicMock()
    handler.start_workflow = MagicMock()
    handler._reply_workflow_error = MagicMock()

    # /wf with no args opens /wf_config
    handler.handle_workflow_command("m1", "oc_1", "/wf", project)
    handler.show_workflow_config.assert_called_once_with("m1", "oc_1", project)

    # /wf <req> opens interactive team launch/config card without auto-starting immediately
    handler.show_workflow_config.reset_mock()
    handler.start_workflow.reset_mock()
    with patch("src.workflow_engine.bridge.RuntimeBridge.check_node_available", return_value=True):
        handler.handle_workflow_command("m_int", "oc_1", "/wf Build API", project)
    handler.show_workflow_config.assert_called_once_with(
        "m_int", "oc_1", project, requirement="Build API"
    )
    handler.start_workflow.assert_not_called()

    # /wf --auto <req> (case-insensitive) bypasses config card and launches immediately
    handler.show_workflow_config.reset_mock()
    handler.start_workflow.reset_mock()
    handler.handle_workflow_command("m_auto", "oc_1", "/wf --AUTO Build API", project)
    handler.start_workflow.assert_called_once_with("m_auto", "oc_1", "Build API", project)
    handler.show_workflow_config.assert_not_called()

    # /wf --auto without requirement replies invalid_argument
    handler._reply_workflow_error.reset_mock()
    handler.handle_workflow_command("m_auto_empty", "oc_1", "/wf --auto", project)
    handler._reply_workflow_error.assert_called_once()
    assert handler._reply_workflow_error.call_args.args[:2] == ("m_auto_empty", "invalid_argument")

    # /wf --quick <req> and /wf --quick without requirement
    handler.start_workflow.reset_mock()
    handler.handle_workflow_command("m_quick", "oc_1", "/wf --quick Build API", project)
    handler.start_workflow.assert_called_once_with(
        "m_quick", "oc_1", "Build API", project, preset_id="quick"
    )
    handler.show_workflow_config.reset_mock()
    handler.handle_workflow_command("m_quick_cfg", "oc_1", "/wf --quick", project)
    handler.show_workflow_config.assert_called_once_with(
        "m_quick_cfg", "oc_1", project, preset_id="quick"
    )

    # /wf_config opens config editor (with optional --preset / --auto)
    handler.show_workflow_config.reset_mock()
    handler.handle_workflow_command("m2", "oc_1", "/wf_config --preset quick", project)
    handler.show_workflow_config.assert_called_once_with(
        "m2", "oc_1", project, preset_id="quick", requirement=None
    )
    handler.show_workflow_config.reset_mock()
    handler.handle_workflow_command("m2b", "oc_1", "/wf_config --auto", project)
    handler.show_workflow_config.assert_called_once_with(
        "m2b", "oc_1", project, preset_id="auto", requirement=None
    )

    # /wf --preset <id> with no requirement opens config editor preselected to preset
    handler.show_workflow_config.reset_mock()
    handler.handle_workflow_command("m3", "oc_1", "/wf --preset review_team", project)
    handler.show_workflow_config.assert_called_once_with(
        "m3", "oc_1", project, preset_id="review_team"
    )

    # /wf --preset <id> <req> launches start_workflow with explicit preset_id
    handler.start_workflow.reset_mock()
    handler.handle_workflow_command("m4", "oc_1", "/wf --preset=collaborative Build API", project)
    handler.start_workflow.assert_called_once_with(
        "m4", "oc_1", "Build API", project, preset_id="collaborative"
    )

    # Invalid preset rejected cleanly
    handler._reply_workflow_error.reset_mock()
    handler.handle_workflow_command("m5", "oc_1", "/wf --preset invalid_preset Build API", project)
    handler._reply_workflow_error.assert_called_once()
    assert handler._reply_workflow_error.call_args.args[:2] == ("m5", "invalid_argument")


def test_f10_config_editor_direct_action_registry_validation_and_start(tmp_path: Path) -> None:
    mgr = ProjectManager(str(tmp_path / "projects.json"))
    ok, _, proj = mgr.create_project("p1", "Proj1", str(tmp_path / "repo"), chat_id="oc_1")
    assert ok and proj is not None

    handler = _make_workflow_handler()
    handler.ctx.project_manager = mgr
    handler.start_workflow = MagicMock()

    fake_ws = SimpleNamespace(
        _handler_ctx=SimpleNamespace(
            handlers={
                "coco": MagicMock(),
                "project": MagicMock(),
                "system": MagicMock(),
                "deep": MagicMock(),
                "spec": MagicMock(),
                "workflow": handler,
            }
        )
    )
    registry = init_action_registry(fake_ws)
    for act in action_ids.WF_CONFIG_ACTIONS:
        assert act in registry

    with patch(
        "src.workflow_engine.tool_registry.get_available_tools",
        return_value={"codex": "Codex", "coco": "Coco"},
    ):
        handler.show_workflow_config("om_init", "oc_1", proj, requirement="Optimize cache")
        handler.reply_card.assert_called_once()

        # coco has supports_effort=False so editor ignores/clears stale effort cleanly;
        # direct spec validation still rejects incompatible effort on model-only tools,
        # and effort-capable tools still reject invalid effort values.
        registry[action_ids.WF_SELECT_AGENT]("om_reply_card", "oc_1", "p1", {"action": action_ids.WF_SELECT_AGENT, "_option": "__new__"})
        registry[action_ids.WF_SELECT_TOOL]("om_reply_card", "oc_1", "p1", {"action": action_ids.WF_SELECT_TOOL, "_option": "coco"})
        st = handler._get_wf_config_state(config_session_id="", chat_id="oc_1", project_id="p1", message_id="om_reply_card")
        assert st is not None
        from dataclasses import replace

        from src.workflow_engine.tool_registry import validate_agent_binding_spec
        handler._put_wf_config_state(
            replace(st, editor_tool="coco", editor_model="coco-pro", editor_effort="high"),
            message_id="om_reply_card",
        )
        _c, stale_toast, stale_msg = handler.handle_workflow_config_callback_sync(
            "om_reply_card",
            "oc_1",
            "p1",
            {"action": action_ids.WF_ADD_OR_UPDATE_AGENT},
        )
        assert stale_toast == "success", stale_msg
        st_after_coco = handler._get_wf_config_state(config_session_id="", chat_id="oc_1", project_id="p1", message_id="om_reply_card")
        assert st_after_coco is not None
        assert st_after_coco.members[-1].tool_name == "coco"
        assert st_after_coco.members[-1].effort is None
        ok_direct, err_direct = validate_agent_binding_spec("coco", effort="high", available_tools={"coco": "Coco"})
        assert ok_direct is False and err_direct is not None

        # Reset & launch via WF_START_WORKFLOW
        registry[action_ids.WF_RESET_TEAM]("om_reply_card", "oc_1", "p1", {"action": action_ids.WF_RESET_TEAM})
        registry[action_ids.WF_START_WORKFLOW]("om_reply_card", "oc_1", "p1", {"action": action_ids.WF_START_WORKFLOW})
        handler.start_workflow.assert_called_once()
        assert handler.start_workflow.call_args.args[2] == "Optimize cache"


def test_without_stop_actions_prunes_behaviors_empty_column_column_set_and_hr() -> None:
    pool = (
        WorkflowAgentBinding(agent_id="A1", tool_name="codex", model_name="gpt-5.4", display_name="Codex", effort="high", role="worker"),
    )
    card = WorkflowGenerationRenderer(
        requirement="Check stop pruning",
        agent_pool=pool,
        orchestrator_agent_id="A1",
        chat_id="oc_1",
        project_id="p_1",
    ).render(current_activity="生成中", elapsed_seconds=2)

    # Confirm generation card has stop button + preceding hr inside column_set
    raw_before = json.dumps(card, ensure_ascii=False)
    assert action_ids.WORKFLOW_STOP_RUNNING in raw_before
    assert any(el.get("tag") == "column_set" for el in card["body"]["elements"])

    cleaned = _without_stop_actions(card)
    raw_after = json.dumps(cleaned, ensure_ascii=False)
    assert action_ids.WORKFLOW_STOP_RUNNING not in raw_after
    assert not any(el.get("tag") in {"column", "column_set"} for el in cleaned["body"]["elements"])
    # Trailing hr right before the stop row was pruned, while earlier hr dividers remain
    assert cleaned["body"]["elements"][-1].get("tag") != "hr"
    assert any(el.get("tag") == "hr" for el in cleaned["body"]["elements"])


def test_wf_status_patches_live_card_without_duplicate_and_falls_back_to_text() -> None:
    handler = _make_workflow_handler()
    owner = _WorkflowLifecycleOwner(
        session_key="sess_1",
        initiator_user_id="ou_1",
        chat_id="oc_1",
        project_id="p_1",
        root_path="/tmp/repo",
        card_message_id="om_live_1",
    )
    engine = MagicMock()
    engine._lock = threading.RLock()
    engine._workflow_start_owner = owner
    engine.is_running = True
    engine.get_status_text.return_value = "running status text"
    engine.get_progress_cards.return_value = [
        {
            "header": {"title": {"tag": "plain_text", "content": "Workflow · 运行中"}, "template": "blue"},
            "elements": [{"tag": "markdown", "content": "progress"}],
            "_workflow_page_key": ("status", -1, 0),
        }
    ]
    handler.ctx.workflow_engine_manager.get.return_value = engine
    project = ProjectContext(project_id="p_1", project_name="P1", root_path="/tmp/repo")

    handler.show_workflow_status("om_cmd", "oc_1", project)
    handler.update_card.assert_called_once()
    assert handler.update_card.call_args.args[0] == "om_live_1"
    handler.send_card_to_chat.assert_not_called()

    # If patch fails, falls back to concise text instead of sending duplicate interactive card
    handler.update_card.reset_mock()
    handler.update_card.return_value = False
    handler.show_workflow_status("om_cmd2", "oc_1", project)
    handler.send_card_to_chat.assert_not_called()
    handler.reply_text.assert_called_with("om_cmd2", "running status text")


def test_system_stop_and_diagnostics_phase_progress_integration(tmp_path: Path) -> None:
    now = time.time()
    wp = WorkflowProject(
        workflow_id="wf_int_1",
        name="RealWF",
        status=WorkflowStatus.RUNNING,
        started_at=now,
        phases=[
            PhaseProgress(title="P1", started_at=now, finished_at=now + 1.0, agents=[AgentProgress(label="a1", status=AgentStatus.DONE)]),
            PhaseProgress(title="P2", started_at=now + 1.0, agents=[AgentProgress(label="a2", status=AgentStatus.RUNNING)]),
        ],
    )
    wf_engine = SimpleNamespace(
        engine_name="Workflow",
        workflow_project=wp,
        project=wp,
        is_running=True,
    )
    ctx = SimpleNamespace(
        mode_manager=SimpleNamespace(get_mode=lambda *_a, **_kw: None, is_programming_mode=lambda *_a, **_kw: False),
        handlers={},
        deep_engine_manager=SimpleNamespace(list_engines=lambda _c: []),
        spec_engine_manager=SimpleNamespace(list_engines=lambda _c: []),
        workflow_engine_manager=SimpleNamespace(list_engines=lambda _c: [wf_engine], get=lambda _c, _r: wf_engine),
        chat_lock_manager=None,
        repo_lock_manager=None,
    )
    entries = DiagnosticsHelper.get_all_engine_statuses(ctx, "oc_1", include_done=True)
    assert len(entries) == 1
    assert "1/2 阶段" in entries[0].info

    wf_handler = MagicMock()
    sys_handler = SystemHandler.__new__(SystemHandler)
    sys_handler.ctx = SimpleNamespace(
        mode_manager=SimpleNamespace(get_mode=lambda *_a, **_kw: None),
        workflow_engine_manager=SimpleNamespace(get=lambda _c, _r: wf_engine),
        deep_engine_manager=None,
        spec_engine_manager=None,
    )
    sys_handler.get_handler = lambda name: wf_handler if name == "workflow" else None
    sys_handler.get_working_dir = lambda _c: str(tmp_path)
    sys_handler.reply_text = MagicMock()
    proj = ProjectContext(project_id="p1", project_name="P1", root_path=str(tmp_path))

    sys_handler.stop_current_task("om_stop", "oc_1", proj)
    wf_handler.stop_workflow.assert_called_once_with("om_stop", "oc_1", proj)


def test_process_stream_markers_include_effort_and_role_when_provided() -> None:
    events: list[tuple[str, dict]] = []
    sink = SimpleNamespace(
        started=False,
        healthy=True,
        start=lambda: setattr(sink, "started", True),
        emit=lambda ev: (events.append((ev.type.value, dict(ev.payload))) or True),
        complete=lambda _ev: True,
        abort=lambda: None,
    )
    stream = WorkflowProcessStream(sink)
    assert stream.agent_started(
        "s1",
        "codex",
        meta={"agent_id": "A1", "role": "scout", "model": "gpt-5.4", "effort": "xhigh"},
    )
    assert stream.agent_done(
        "s1",
        {"agent_id": "A1", "role": "scout", "model": "gpt-5.4", "effort": "xhigh", "error": None},
    )
    deltas = [p.get("text", "") for t, p in events if t == "text_delta"]
    assert any("A1" in d and "scout" in d and "xhigh" in d for d in deltas)
    assert any(d.startswith("完成 · s1") and "xhigh" in d and "scout" in d for d in deltas)


def test_stale_config_session_reconstruction_preserves_saved_multi_effort_team(tmp_path: Path) -> None:
    mgr = ProjectManager(str(tmp_path / "projects.json"))
    ok, _, proj = mgr.create_project("p1", "Proj1", str(tmp_path / "repo"), chat_id="oc_1")
    assert ok and proj is not None
    proj.add_chat_id("oc_2")

    saved_cfg = {
        "version": 1,
        "preset_id": "collaborative",
        "orchestrator_agent_id": "A2",
        "bindings": [
            {
                "agent_id": "A1",
                "tool_name": "codex",
                "model_name": "gpt-5.4",
                "display_name": "Codex",
                "effort": "low",
                "role": "worker",
                "roles": ["worker"],
            },
            {
                "agent_id": "A2",
                "tool_name": "codex",
                "model_name": "gpt-5.4",
                "display_name": "Codex",
                "effort": "xhigh",
                "role": "verifier",
                "roles": ["verifier"],
            },
        ],
        "updated_at": 1700000000.0,
    }
    assert mgr.commit_workflow_team_configuration(proj, saved_cfg, chat_id="oc_1")

    handler = _make_workflow_handler()
    handler.ctx.project_manager = mgr

    with patch(
        "src.workflow_engine.tool_registry.get_available_tools",
        return_value={"codex": "Codex", "claude": "Claude", "traex": "Traex"},
    ):
        handler.show_workflow_config("om_init", "oc_1", proj)
        st_before = handler._get_wf_config_state(
            config_session_id="",
            chat_id="oc_1",
            project_id="p1",
            message_id="om_reply_card",
        )
        assert st_before is not None
        cfg_sid = st_before.config_session_id

        # Simulate bot restart / LRU eviction
        handler._wf_config_sessions.clear()
        handler._wf_config_message_index.clear()

        _card, toast_type, _msg = handler.handle_workflow_config_callback_sync(
            "om_reply_card",
            "oc_1",
            "p1",
            {
                "action": action_ids.WF_SELECT_AGENT,
                "config_session_id": cfg_sid,
                "preset_id": "collaborative",
                "_option": "A1",
            },
        )
        assert toast_type != "error"
        reconstructed = handler._get_wf_config_state(
            config_session_id=cfg_sid,
            chat_id="oc_1",
            project_id="p1",
            message_id="om_reply_card",
        )
        assert reconstructed is not None
        assert reconstructed.saved_as_default is True
        assert [m.effort for m in reconstructed.members] == ["low", "xhigh"]
        assert reconstructed.orchestrator_agent_id == "A2"

        # When saved_cfg is absent (oc_2), fallback_preset from payload is honored
        handler._wf_config_sessions.clear()
        handler._wf_config_message_index.clear()
        _c2, toast_2, _m2 = handler.handle_workflow_config_callback_sync(
            "om_card_2",
            "oc_2",
            "p1",
            {
                "action": action_ids.WF_SELECT_AGENT,
                "config_session_id": "evicted_oc2",
                "preset_id": "quick",
                "_option": "A1",
            },
        )
        assert toast_2 != "error"
        reconstructed_2 = handler._get_wf_config_state(
            config_session_id="evicted_oc2",
            chat_id="oc_2",
            project_id="p1",
            message_id="om_card_2",
        )
        assert reconstructed_2 is not None
        assert reconstructed_2.preset_id == "quick"
        assert len(reconstructed_2.members) == 2


def test_invalid_profile_rejected_on_effort_capable_tools(tmp_path: Path) -> None:
    from dataclasses import replace

    from src.workflow_engine.tool_registry import ToolCapability

    mgr = ProjectManager(str(tmp_path / "projects.json"))
    ok, _, proj = mgr.create_project("p1", "Proj1", str(tmp_path / "repo"), chat_id="oc_1")
    assert ok and proj is not None

    handler = _make_workflow_handler()
    handler.ctx.project_manager = mgr

    fake_caps = {
        "traex": ToolCapability(
            tool_name="traex",
            display_name="Traex",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=("gpt-5.4",),
            efforts_by_model={"gpt-5.4": ("low", "medium", "high")},
            default_effort="medium",
            profiles_by_model={"gpt-5.4": ("standard", "max")},
        ),
        "codex": ToolCapability(
            tool_name="codex",
            display_name="Codex",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=("gpt-5.4",),
            efforts_by_model={"gpt-5.4": ("low", "medium", "high", "xhigh")},
            default_effort="medium",
        ),
    }

    with (
        patch("src.workflow_engine.tool_registry.get_tool_capabilities", return_value=fake_caps),
        patch(
            "src.workflow_engine.tool_registry.get_available_tools",
            return_value={"traex": "Traex", "codex": "Codex"},
        ),
    ):
        handler.show_workflow_config("om_init", "oc_1", proj)
        st = handler._get_wf_config_state(
            config_session_id="",
            chat_id="oc_1",
            project_id="p1",
            message_id="om_reply_card",
        )
        assert st is not None
        before_members = st.members

        handler._put_wf_config_state(
            replace(
                st,
                selected_agent_id="__new__",
                editor_tool="traex",
                editor_model="gpt-5.4",
                editor_profile="bogus_profile",
                editor_effort="high",
            ),
            message_id="om_reply_card",
        )
        _card, toast_type, toast_msg = handler.handle_workflow_config_callback_sync(
            "om_reply_card",
            "oc_1",
            "p1",
            {"action": action_ids.WF_ADD_OR_UPDATE_AGENT},
        )
        assert toast_type == "error"
        assert "profile" in toast_msg.lower()
        after = handler._get_wf_config_state(
            config_session_id=st.config_session_id,
            chat_id="oc_1",
            project_id="p1",
            message_id="om_reply_card",
        )
        assert after is not None
        assert after.members == before_members


def test_wf_save_team_surfaces_error_on_invalid_or_colliding_pending_edit(tmp_path: Path) -> None:
    from dataclasses import replace

    from src.workflow_engine.tool_registry import ToolCapability

    mgr = ProjectManager(str(tmp_path / "projects.json"))
    ok, _, proj = mgr.create_project("p1", "Proj1", str(tmp_path / "repo"), chat_id="oc_1")
    assert ok and proj is not None

    handler = _make_workflow_handler()
    handler.ctx.project_manager = mgr

    fake_caps = {
        "codex": ToolCapability(
            tool_name="codex",
            display_name="Codex",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=("gpt-5.4",),
            efforts_by_model={"gpt-5.4": ("low", "medium", "high", "xhigh")},
            default_effort="medium",
        ),
        "traex": ToolCapability(
            tool_name="traex",
            display_name="Traex",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=("gpt-5.4",),
            efforts_by_model={"gpt-5.4": ("low", "medium", "high")},
            default_effort="medium",
            profiles_by_model={"gpt-5.4": ("standard", "max")},
        ),
    }

    with (
        patch("src.workflow_engine.tool_registry.get_tool_capabilities", return_value=fake_caps),
        patch(
            "src.workflow_engine.tool_registry.get_available_tools",
            return_value={"codex": "Codex", "traex": "Traex"},
        ),
    ):
        handler.show_workflow_config("om_init", "oc_1", proj)
        st = handler._get_wf_config_state(
            config_session_id="",
            chat_id="oc_1",
            project_id="p1",
            message_id="om_reply_card",
        )
        assert st is not None

        # 1. Invalid effort pending edit on selected A1
        handler.handle_workflow_config_callback_sync(
            "om_reply_card",
            "oc_1",
            "p1",
            {"action": action_ids.WF_SELECT_EFFORT, "_option": "bogus_effort"},
        )
        _c1, toast_1, msg_1 = handler.handle_workflow_config_callback_sync(
            "om_reply_card",
            "oc_1",
            "p1",
            {"action": action_ids.WF_SAVE_TEAM},
        )
        assert toast_1 == "error"
        assert "bogus_effort" in msg_1
        assert get_project_workflow_team_config(proj, "oc_1") is None

        # 2. Invalid profile pending edit on selected traex A1
        handler._put_wf_config_state(
            replace(
                st,
                selected_agent_id="A1",
                editor_tool="traex",
                editor_model="gpt-5.4",
                editor_profile="bogus_profile",
                editor_effort="high",
            ),
            message_id="om_reply_card",
        )
        _c2, toast_2, msg_2 = handler.handle_workflow_config_callback_sync(
            "om_reply_card",
            "oc_1",
            "p1",
            {"action": action_ids.WF_SAVE_TEAM},
        )
        assert toast_2 == "error"
        assert "profile" in msg_2.lower()
        assert get_project_workflow_team_config(proj, "oc_1") is None

        # 3. Colliding pending edit on A1 matching A2
        m1 = WorkflowAgentBinding(
            agent_id="A1",
            tool_name="codex",
            model_name="gpt-5.4",
            display_name="Codex",
            effort="low",
            role="worker",
            roles=("worker",),
        )
        m2 = WorkflowAgentBinding(
            agent_id="A2",
            tool_name="codex",
            model_name="gpt-5.4",
            display_name="Codex",
            effort="xhigh",
            role="verifier",
            roles=("verifier",),
        )
        handler._put_wf_config_state(
            replace(
                st,
                members=(m1, m2),
                selected_agent_id="A1",
                editor_tool="codex",
                editor_model="gpt-5.4",
                editor_profile=None,
                editor_effort="xhigh",
                editor_role="worker",
            ),
            message_id="om_reply_card",
        )
        _c3, toast_3, msg_3 = handler.handle_workflow_config_callback_sync(
            "om_reply_card",
            "oc_1",
            "p1",
            {"action": action_ids.WF_SAVE_TEAM},
        )
        assert toast_3 == "error"
        assert "duplicate" in msg_3.lower()
        assert get_project_workflow_team_config(proj, "oc_1") is None


def test_wf_status_during_generating_script_never_replies_idle(tmp_path: Path) -> None:
    from src.workflow_engine.engine import WorkflowEngine
    from src.workflow_engine.models import PendingWorkflow

    handler = _make_workflow_handler()
    pool = (
        WorkflowAgentBinding(
            agent_id="A1",
            tool_name="codex",
            model_name="gpt-5.4",
            display_name="Codex",
            effort="high",
            role="worker",
        ),
    )
    wf_proj = WorkflowProject(
        status=WorkflowStatus.GENERATING_SCRIPT,
        requirement="Build M4 workflow",
        pending=PendingWorkflow(
            requirement="Build M4 workflow",
            agent_pool=pool,
            orchestrator_agent_id="A1",
            orchestrator_was_auto=False,
        ),
    )
    engine = WorkflowEngine("oc_1", str(tmp_path))
    engine._project = wf_proj
    assert engine._renderer_wf is None
    assert engine.get_progress_cards() == []
    assert "Workflow: idle" not in engine.get_status_text()
    assert "生成" in engine.get_status_text()

    gen_owner = _WorkflowLifecycleOwner(
        session_key="gen_sess_1",
        initiator_user_id="ou_1",
        chat_id="oc_1",
        project_id="p_1",
        root_path=str(tmp_path),
        card_message_id="om_gen_live",
    )
    engine._script_generation_owner = gen_owner
    handler.ctx.workflow_engine_manager.get.return_value = engine
    project = ProjectContext(project_id="p_1", project_name="P1", root_path=str(tmp_path))

    # 1. Live generation card exists -> patches live card, 0 new cards, no "idle"
    handler.show_workflow_status("om_cmd_1", "oc_1", project)
    handler.update_card.assert_called_once()
    assert handler.update_card.call_args.args[0] == "om_gen_live"
    handler.send_card_to_chat.assert_not_called()
    handler.reply_text.assert_not_called()

    # 2. Patch fails -> falls back to generating status text (never "Workflow: idle")
    handler.update_card.reset_mock()
    handler.update_card.return_value = False
    handler.show_workflow_status("om_cmd_2", "oc_1", project)
    handler.send_card_to_chat.assert_not_called()
    handler.reply_text.assert_called_once()
    assert "Workflow: idle" not in handler.reply_text.call_args.args[1]
    assert "正在生成编排脚本" in handler.reply_text.call_args.args[1]

    # 3. No live card id -> replies generating status text (never "Workflow: idle")
    handler.reply_text.reset_mock()
    engine._script_generation_owner = None
    handler.show_workflow_status("om_cmd_3", "oc_1", project)
    handler.reply_text.assert_called_once()
    assert "Workflow: idle" not in handler.reply_text.call_args.args[1]
    assert "正在生成编排脚本" in handler.reply_text.call_args.args[1]


def test_wf_config_cross_chat_and_cross_project_session_guard(tmp_path: Path) -> None:
    mgr = ProjectManager(str(tmp_path / "projects.json"))
    ok1, _, proj1 = mgr.create_project("p1", "Proj1", str(tmp_path / "repo1"), chat_id="oc_1")
    ok2, _, proj2 = mgr.create_project("p2", "Proj2", str(tmp_path / "repo2"), chat_id="oc_1")
    assert ok1 and ok2 and proj1 and proj2
    proj1.add_chat_id("oc_2")

    handler = _make_workflow_handler()
    handler.ctx.project_manager = mgr

    with patch(
        "src.workflow_engine.tool_registry.get_available_tools",
        return_value={"codex": "Codex", "claude": "Claude"},
    ):
        handler.show_workflow_config("om_init", "oc_1", proj1, requirement="Secret req for chat 1")
        st = handler._get_wf_config_state(
            config_session_id="",
            chat_id="oc_1",
            project_id="p1",
            message_id="om_reply_card",
        )
        assert st is not None

        # Cross-chat lookup rejected
        assert (
            handler._get_wf_config_state(
                config_session_id=st.config_session_id,
                chat_id="oc_2",
                project_id="p1",
                message_id="om_reply_card",
            )
            is None
        )
        # Cross-project lookup rejected
        assert (
            handler._get_wf_config_state(
                config_session_id=st.config_session_id,
                chat_id="oc_1",
                project_id="p2",
                message_id="om_reply_card",
            )
            is None
        )
        # Cross-chat callback does not leak Secret req for chat 1
        card, _t, _m = handler.handle_workflow_config_callback_sync(
            "om_forged",
            "oc_2",
            "p1",
            {"action": action_ids.WF_SELECT_ROLE, "config_session_id": st.config_session_id, "_option": "verifier"},
        )
        assert "Secret req for chat 1" not in json.dumps(card, ensure_ascii=False)


def test_wf_start_workflow_and_override_pool_reject_unavailable_tools(tmp_path: Path) -> None:
    from src.workflow_engine.tool_registry import ToolCapability

    mgr = ProjectManager(str(tmp_path / "projects.json"))
    ok, _, proj = mgr.create_project("p1", "Proj1", str(tmp_path / "repo"), chat_id="oc_1")
    assert ok and proj is not None

    handler = _make_workflow_handler()
    handler.ctx.project_manager = mgr
    handler.start_workflow = MagicMock()

    caps_gemini_gone = {
        "codex": ToolCapability(
            tool_name="codex",
            display_name="Codex",
            available=True,
            supports_model=True,
            supports_effort=True,
        ),
        "gemini": ToolCapability(
            tool_name="gemini",
            display_name="Gemini",
            available=False,
            supports_model=True,
            supports_effort=False,
        ),
    }

    with (
        patch("src.workflow_engine.tool_registry.get_tool_capabilities", return_value=caps_gemini_gone),
        patch(
            "src.workflow_engine.tool_registry.get_available_tools",
            return_value={"codex": "Codex"},
        ),
    ):
        handler.show_workflow_config("om_init", "oc_1", proj, requirement="Run checks")
        st = handler._get_wf_config_state(
            config_session_id="",
            chat_id="oc_1",
            project_id="p1",
            message_id="om_reply_card",
        )
        assert st is not None
        from dataclasses import replace

        stale_gemini = WorkflowAgentBinding(
            agent_id="A1",
            tool_name="gemini",
            model_name=None,
            display_name="Gemini",
            role="worker",
            roles=("worker",),
        )
        handler._put_wf_config_state(
            replace(st, members=(stale_gemini,)),
            message_id="om_reply_card",
        )
        _card, toast_type, toast_msg = handler.handle_workflow_config_callback_sync(
            "om_reply_card",
            "oc_1",
            "p1",
            {"action": action_ids.WF_START_WORKFLOW},
        )
        assert toast_type == "error"
        assert "gemini" in toast_msg.lower()

        handler.handle_workflow_config_action(
            "om_reply_card",
            "oc_1",
            "p1",
            {"action": action_ids.WF_START_WORKFLOW},
        )
        handler.start_workflow.assert_not_called()

        # Also validate direct _launch_auto_bound_workflow override_pool rejection
        handler._reply_workflow_error = MagicMock()
        admission_owner = _WorkflowLifecycleOwner(
            session_key="adm_1",
            chat_id="oc_1",
            project_id="p1",
            root_path=str(tmp_path / "repo"),
        )
        engine = MagicMock()
        handler._launch_auto_bound_workflow(
            message_id="om_launch",
            chat_id="oc_1",
            requirement="Run checks",
            project=proj,
            root_path=str(tmp_path / "repo"),
            engine=engine,
            admission_owner=admission_owner,
            override_pool=(stale_gemini,),
        )
        handler._reply_workflow_error.assert_called_once()
        assert handler._reply_workflow_error.call_args.args[:2] == ("om_launch", "invalid_argument")
        assert admission_owner.done_event.is_set()
        engine.release_lifecycle_owner.assert_called_once_with(admission_owner)


def test_normalize_workflow_team_config_filters_corrupt_entries_before_truncating_to_max() -> None:
    from src.workflow_engine.tool_registry import ToolCapability

    corrupt_prefix = [
        {"agent_id": f"BAD{i}", "tool_name": "nonexistent_tool", "display_name": "Bad"}
        for i in range(10)
    ]
    valid_suffix = [
        {
            "agent_id": f"A{i + 1}",
            "tool_name": "codex",
            "model_name": f"gpt-5.{i}",
            "display_name": "Codex",
            "role": "worker",
            "roles": ["worker"],
        }
        for i in range(8)
    ]
    fake_caps = {
        "codex": ToolCapability(
            tool_name="codex",
            display_name="Codex",
            available=True,
            supports_model=True,
            supports_effort=True,
        )
    }
    with (
        patch("src.workflow_engine.tool_registry.get_tool_capabilities", return_value=fake_caps),
        patch.object(ProjectContext, "normalize_acp_tool_name", side_effect=lambda name: name if name == "codex" else None),
    ):
        recovered = ProjectContext.normalize_workflow_team_config(
            {
                "preset_id": "collaborative",
                "orchestrator_agent_id": "A1",
                "bindings": [*corrupt_prefix, *valid_suffix],
            }
        )
    assert recovered is not None
    assert len(recovered["bindings"]) == 8
    assert [b["agent_id"] for b in recovered["bindings"]] == [f"A{i + 1}" for i in range(8)]


def _extract_card_button_payload(card: dict, target_action: str) -> dict:
    """Recursively find the callback value dict for a button with ``action == target_action``."""
    found: list[dict] = []

    def _walk(node: object) -> None:
        if isinstance(node, dict):
            val = node.get("value")
            if isinstance(val, dict) and val.get("action") == target_action:
                found.append(dict(val))
            for item in node.values():
                _walk(item)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(card)
    assert found, f"Missing button payload for action={target_action}"
    return found[0]


def test_reused_engine_wf_status_during_run2_generating_script_does_not_leak_run1_renderer(
    tmp_path: Path,
) -> None:
    from src.spec_engine.models import ReviewAgentBinding
    from src.workflow_engine.engine import WorkflowEngine, WorkflowEngineCallbacks
    from src.workflow_engine.run_spec import WorkflowRunSpec

    script_path = tmp_path / "run1.js"
    script_path.write_text(
        """
export const meta = { name: "First completed task", phases: [], tools: ["codex"] };
export default async function main() { return "First completed task done"; }
""",
        encoding="utf-8",
    )
    engine = WorkflowEngine("oc_1", str(tmp_path))

    class CompletingBridge:
        @staticmethod
        def check_node_available() -> bool:
            return True

        def __init__(self, *args, **kwargs) -> None:
            return None

        def start(self) -> None:
            return None

        def run(self) -> str:
            return "First completed task done"

        def stop(self) -> None:
            return None

    pool = (
        WorkflowAgentBinding(
            agent_id="A1",
            tool_name="codex",
            model_name="gpt-5.4",
            display_name="Codex",
            effort="high",
            role="worker",
            roles=("worker",),
        ),
    )
    run_spec = WorkflowRunSpec(
        orchestrator=ReviewAgentBinding(
            provider="workflow",
            tool_name="codex",
            display_name="Codex",
            agent_type="codex",
            model_name="gpt-5.4",
            selection_key="A1",
        ),
        reviewers=(),
        tool_model_map={"codex": "gpt-5.4"},
        task="First completed task",
        chat_id="oc_1",
        topic_id=None,
        budget=5,
        deadline=None,
        auto_reviewer=True,
        allowed_tools=("codex",),
        agent_pool=pool,
        orchestrator_agent_id="A1",
    )
    with patch("src.workflow_engine.engine.RuntimeBridge", CompletingBridge):
        run1_result = engine.execute_workflow(
            str(script_path),
            callbacks=WorkflowEngineCallbacks(),
            run_spec=run_spec,
        )
    assert run1_result.status == WorkflowStatus.COMPLETED
    assert engine._renderer_wf is not None
    assert engine.get_progress_cards()

    handler = _make_workflow_handler()
    handler.get_working_dir = MagicMock(return_value=str(tmp_path))
    handler._supersede_incomplete_workflow = MagicMock(return_value=(True, None, None))
    handler._schedule_generate_and_start_workflow = MagicMock()
    handler.ctx.workflow_engine_manager.get_or_create.return_value = engine
    handler.ctx.workflow_engine_manager.get.return_value = engine
    proj = ProjectContext(project_id="p_1", project_name="P1", root_path=str(tmp_path))

    with (
        patch("src.workflow_engine.bridge.RuntimeBridge.check_node_available", return_value=True),
        patch("src.workflow_engine.tool_registry.get_available_tools", return_value={"codex": "Codex"}),
    ):
        handler.start_workflow(
            "om_run2_cmd",
            "oc_1",
            "Second brand new task",
            proj,
            override_pool=pool,
        )

    assert engine.project is not None
    assert engine.project.status == WorkflowStatus.GENERATING_SCRIPT
    assert engine._renderer_wf is None
    assert engine.get_progress_cards() == []
    assert "生成" in engine.get_status_text()

    handler.update_card.reset_mock()
    handler.reply_text.reset_mock()
    handler.show_workflow_status("om_status_2", "oc_1", proj)
    handler.update_card.assert_called_once()
    assert handler.update_card.call_args.args[0] == "om_send_card"
    patched_card = handler.update_card.call_args.args[1]
    patched_json = json.dumps(patched_card, ensure_ascii=False)
    assert "Second brand new task" in patched_json
    assert "First completed task" not in patched_json
    assert "✅ Workflow" not in patched_json
    handler.reply_text.assert_not_called()

    # Defense-in-depth: even if a stale _renderer_wf were present during GENERATING_SCRIPT,
    # get_progress_cards(), get_status_text(), and show_workflow_status() still prioritize Run 2 generation.
    engine._renderer_wf = MagicMock()
    engine._renderer_wf.render_progress_cards.return_value = [{"header": {"title": {"tag": "plain_text", "content": "✅ Workflow"}}, "elements": []}]
    engine._renderer_wf.render_compact_status.return_value = "stale run 1 status"
    assert engine.get_progress_cards() == []
    assert "生成" in engine.get_status_text()
    handler.update_card.reset_mock()
    handler.show_workflow_status("om_status_3", "oc_1", proj)
    handler.update_card.assert_called_once()
    assert "Second brand new task" in json.dumps(handler.update_card.call_args.args[1], ensure_ascii=False)


def test_ui_role_preserves_verifier_across_preset_edit_and_save(tmp_path: Path) -> None:
    from src.feishu.handlers.workflow_config import _ui_role
    from src.workflow_engine.tool_registry import ToolCapability

    assert _ui_role("verifier") == "verifier"

    mgr = ProjectManager(str(tmp_path / "projects.json"))
    ok, _, proj = mgr.create_project("p1", "Proj1", str(tmp_path / "repo"), chat_id="oc_1")
    assert ok and proj is not None

    handler = _make_workflow_handler()
    handler.ctx.project_manager = mgr

    fake_caps = {
        "codex": ToolCapability(
            tool_name="codex",
            display_name="Codex",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=("gpt-5.4",),
            efforts_by_model={"gpt-5.4": ("low", "medium", "high", "xhigh")},
            default_effort="medium",
        ),
        "claude": ToolCapability(
            tool_name="claude",
            display_name="Claude",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=("sonnet",),
            efforts_by_model={"sonnet": ("low", "medium", "high", "xhigh", "max")},
            default_effort="medium",
        ),
    }

    with (
        patch("src.workflow_engine.tool_registry.get_tool_capabilities", return_value=fake_caps),
        patch(
            "src.workflow_engine.tool_registry.get_available_tools",
            return_value={"codex": "Codex", "claude": "Claude"},
        ),
    ):
        handler.show_workflow_config("om_init", "oc_1", proj, preset_id="quick")
        handler.handle_workflow_config_callback_sync(
            "om_reply_card",
            "oc_1",
            "p1",
            {"action": action_ids.WF_SELECT_AGENT, "_option": "A2"},
        )
        st_selected = handler._get_wf_config_state(
            config_session_id="",
            chat_id="oc_1",
            project_id="p1",
            message_id="om_reply_card",
        )
        assert st_selected is not None
        assert st_selected.editor_role == "verifier"

        handler.handle_workflow_config_callback_sync(
            "om_reply_card",
            "oc_1",
            "p1",
            {"action": action_ids.WF_SELECT_EFFORT, "_option": "xhigh"},
        )
        _card, toast_type, _msg = handler.handle_workflow_config_callback_sync(
            "om_reply_card",
            "oc_1",
            "p1",
            {"action": action_ids.WF_SAVE_TEAM},
        )
        assert toast_type == "success"

    saved = get_project_workflow_team_config(proj, "oc_1")
    assert saved is not None
    a2 = next(b for b in saved["bindings"] if b["agent_id"] == "A2")
    assert a2["role"] == "verifier"
    assert a2["roles"] == ["verifier"]
    assert a2["effort"] == "xhigh"


def test_stale_session_rehydration_does_not_leak_profile_or_effort_across_member_or_tool_switches(
    tmp_path: Path,
) -> None:
    from src.workflow_engine.tool_registry import ToolCapability

    mgr = ProjectManager(str(tmp_path / "projects.json"))
    ok, _, proj = mgr.create_project("p1", "Proj1", str(tmp_path / "repo"), chat_id="oc_1")
    assert ok and proj is not None

    fake_caps = {
        "traex": ToolCapability(
            tool_name="traex",
            display_name="Traex",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=("gpt-5.4",),
            efforts_by_model={"gpt-5.4": ("low", "medium", "high")},
            default_effort="medium",
            profiles_by_model={"gpt-5.4": ("standard", "max")},
        ),
        "codex": ToolCapability(
            tool_name="codex",
            display_name="Codex",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=("gpt-5.4",),
            efforts_by_model={"gpt-5.4": ("low", "medium", "high", "xhigh")},
            default_effort="medium",
        ),
        "coco": ToolCapability(
            tool_name="coco",
            display_name="Coco",
            available=True,
            supports_model=True,
            supports_effort=False,
            models=("glm-5",),
        ),
    }

    handler = _make_workflow_handler()
    handler.ctx.project_manager = mgr

    with (
        patch("src.workflow_engine.tool_registry.get_tool_capabilities", return_value=fake_caps),
        patch(
            "src.workflow_engine.tool_registry.get_available_tools",
            return_value={"traex": "Traex", "codex": "Codex", "coco": "Coco"},
        ),
    ):
        # Variant C (cross-member): A1=traex(profile="max", effort="high") + A2=codex(effort="low")
        saved_cross_member = {
            "version": 1,
            "preset_id": "quick",
            "orchestrator_agent_id": "A1",
            "bindings": [
                {
                    "agent_id": "A1",
                    "tool_name": "traex",
                    "model_name": "gpt-5.4",
                    "display_name": "Traex",
                    "profile": "max",
                    "effort": "high",
                    "role": "worker",
                    "roles": ["worker"],
                },
                {
                    "agent_id": "A2",
                    "tool_name": "codex",
                    "model_name": "gpt-5.4",
                    "display_name": "Codex",
                    "effort": "low",
                    "role": "verifier",
                    "roles": ["verifier"],
                },
            ],
            "updated_at": 1700000000.0,
        }
        assert mgr.commit_workflow_team_configuration(proj, saved_cross_member, chat_id="oc_1")

        handler.show_workflow_config("om_init_c", "oc_1", proj)
        card_a2, _, _ = handler.handle_workflow_config_callback_sync(
            "om_reply_card",
            "oc_1",
            "p1",
            {"action": action_ids.WF_SELECT_AGENT, "_option": "A2"},
        )
        assert card_a2 is not None
        save_payload = _extract_card_button_payload(card_a2, action_ids.WF_SAVE_TEAM)
        assert save_payload["agent_id"] == "A2"
        assert save_payload["tool_name"] == "codex"
        assert "profile" not in save_payload

        # Simulate cache eviction and replay the exact rendered WF_SAVE_TEAM button payload
        handler._wf_config_sessions.clear()
        handler._wf_config_message_index.clear()
        _saved_card, toast_c, msg_c = handler.handle_workflow_config_callback_sync(
            "om_reply_card",
            "oc_1",
            "p1",
            save_payload,
        )
        assert toast_c == "success", f"Unexpected Variant C error: {msg_c}"

        # Variant B (cross-tool): A1=codex(effort="xhigh") switched to coco -> cache cleared -> replay WF_ADD_OR_UPDATE_AGENT
        saved_cross_tool = {
            "version": 1,
            "preset_id": "custom",
            "orchestrator_agent_id": "A1",
            "bindings": [
                {
                    "agent_id": "A1",
                    "tool_name": "codex",
                    "model_name": "gpt-5.4",
                    "display_name": "Codex",
                    "effort": "xhigh",
                    "role": "worker",
                    "roles": ["worker"],
                },
            ],
            "updated_at": 1700000001.0,
        }
        assert mgr.commit_workflow_team_configuration(proj, saved_cross_tool, chat_id="oc_1")
        handler._wf_config_sessions.clear()
        handler._wf_config_message_index.clear()

        handler.show_workflow_config("om_init_b", "oc_1", proj)
        card_coco, _, _ = handler.handle_workflow_config_callback_sync(
            "om_reply_card",
            "oc_1",
            "p1",
            {"action": action_ids.WF_SELECT_TOOL, "_option": "coco"},
        )
        assert card_coco is not None
        update_payload = _extract_card_button_payload(card_coco, action_ids.WF_ADD_OR_UPDATE_AGENT)
        assert update_payload["tool_name"] == "coco"
        assert "effort" not in update_payload

        handler._wf_config_sessions.clear()
        handler._wf_config_message_index.clear()
        _updated_card, toast_b, msg_b = handler.handle_workflow_config_callback_sync(
            "om_reply_card",
            "oc_1",
            "p1",
            update_payload,
        )
        assert toast_b == "success", f"Unexpected Variant B error: {msg_b}"
        st_after_b = handler._get_wf_config_state(
            config_session_id="",
            chat_id="oc_1",
            project_id="p1",
            message_id="om_reply_card",
        )
        assert st_after_b is not None
        assert st_after_b.members[0].tool_name == "coco"
        assert st_after_b.members[0].effort is None


def test_override_pool_rejected_when_get_available_tools_returns_empty(tmp_path: Path) -> None:
    from src.workflow_engine.tool_registry import ToolCapability

    handler = _make_workflow_handler()
    proj = ProjectContext(project_id="p1", project_name="P1", root_path=str(tmp_path))
    handler._reply_workflow_error = MagicMock()
    admission_owner = _WorkflowLifecycleOwner(
        session_key="adm_empty",
        chat_id="oc_1",
        project_id="p1",
        root_path=str(tmp_path),
    )
    engine = MagicMock()
    pool = (
        WorkflowAgentBinding(
            agent_id="A1",
            tool_name="codex",
            model_name="gpt-5.4",
            display_name="Codex",
            effort="high",
            role="worker",
        ),
    )
    stale_caps = {
        "codex": ToolCapability(
            tool_name="codex",
            display_name="Codex",
            available=True,
            supports_model=True,
            supports_effort=True,
        ),
    }

    with (
        patch("src.workflow_engine.tool_registry.get_tool_capabilities", return_value=stale_caps),
        patch("src.workflow_engine.tool_registry.get_available_tools", return_value={}),
    ):
        handler._launch_auto_bound_workflow(
            message_id="om_launch_empty",
            chat_id="oc_1",
            requirement="Run checks",
            project=proj,
            root_path=str(tmp_path),
            engine=engine,
            admission_owner=admission_owner,
            override_pool=pool,
        )

    handler._reply_workflow_error.assert_called_once()
    assert handler._reply_workflow_error.call_args.args[:2] == ("om_launch_empty", "invalid_argument")
    assert admission_owner.done_event.is_set()
    engine.release_lifecycle_owner.assert_called_once_with(admission_owner)


def test_wf_start_workflow_double_click_does_not_overwrite_live_card(tmp_path: Path) -> None:
    mgr = ProjectManager(str(tmp_path / "projects.json"))
    ok, _, proj = mgr.create_project("p1", "Proj1", str(tmp_path / "repo"), chat_id="oc_dbl")
    assert ok and proj is not None

    handler = _make_workflow_handler()
    handler.ctx.project_manager = mgr
    handler.start_workflow = MagicMock()

    with patch(
        "src.workflow_engine.tool_registry.get_available_tools",
        return_value={"codex": "Codex", "claude": "Claude"},
    ):
        handler.show_workflow_config("om_init", "oc_dbl", proj, requirement="Double click safety")
        handler.update_card.reset_mock()
        handler.reply_card.reset_mock()

        # Click 1 consumes requirement and launches in-place
        handler.handle_workflow_config_action(
            "om_reply_card",
            "oc_dbl",
            "p1",
            {"action": action_ids.WF_START_WORKFLOW},
        )
        handler.start_workflow.assert_called_once()
        assert handler.start_workflow.call_args.kwargs["reuse_card_message_id"] == "om_reply_card"

        # Click 2 (duplicate click) must NOT overwrite the live generation/running card
        handler.start_workflow.reset_mock()
        handler.handle_workflow_config_action(
            "om_reply_card",
            "oc_dbl",
            "p1",
            {"action": action_ids.WF_START_WORKFLOW},
        )
        handler.start_workflow.assert_not_called()
        handler.update_card.assert_not_called()
        handler.reply_card.assert_not_called()


def test_generation_patch_coalescing_8s_bucket_and_skip_redundant_initial_patch(tmp_path: Path) -> None:
    from src.feishu.handlers import workflow_generation as wf_gen_mod
    from src.workflow_engine.engine import WorkflowEngine
    from src.workflow_engine.models import PendingWorkflow

    assert wf_gen_mod._GENERATION_PROGRESS_DEBOUNCE_S == 8.0

    script_path = tmp_path / "generated.js"
    script_path.write_text(
        "export const meta = { name: 'gen', tools: ['codex'] };\n",
        encoding="utf-8",
    )
    pool = (
        WorkflowAgentBinding(
            agent_id="A1",
            tool_name="codex",
            model_name="gpt-5.4",
            display_name="Codex",
            effort="high",
            role="worker",
        ),
    )
    engine = WorkflowEngine(chat_id="oc_gen", root_path=str(tmp_path))
    owner = _WorkflowLifecycleOwner(
        session_key="sess_gen",
        initiator_user_id="ou_1",
        chat_id="oc_gen",
        project_id="p_1",
        root_path=str(tmp_path),
        source_script_path=str(script_path),
        card_message_id="om_already_sent",
    )
    engine._script_generation_owner = owner
    engine._project = WorkflowProject(
        status=WorkflowStatus.GENERATING_SCRIPT,
        requirement="Calm generation patches",
        pending=PendingWorkflow(
            requirement="Calm generation patches",
            engine_session_key="sess_gen",
            initiator_user_id="ou_1",
            project_id="p_1",
            selected_tools=["codex"],
            agent_pool=pool,
            orchestrator_agent_id="A1",
            preset_id="custom",
        ),
    )

    handler = _make_workflow_handler()
    handler.ctx.workflow_engine_manager.get_or_create.return_value = engine
    handler._remove_owned_workflow_artifact = MagicMock()
    handler._queue_generated_workflow = MagicMock(
        side_effect=lambda **_kw: setattr(engine.project, "status", WorkflowStatus.RUNNING)
    )

    clock = [100.0]
    coalescer_kwargs: list[dict] = []

    class FakeCoalescer:
        def __init__(self, *, on_progress, debounce_s):
            coalescer_kwargs.append({"on_progress": on_progress, "debounce_s": debounce_s})
            self._on_progress = on_progress

        def enqueue(self, hint: str = ""):
            self._on_progress(hint)

        def stop(self):
            return None

    def fake_codegen(*_args, progress_callback=None, **_kwargs):
        # Same bucket (elapsed 1s -> 0) and same status -> delivered once
        clock[0] = 101.0
        progress_callback("正在生成编排脚本...")
        clock[0] = 103.0
        progress_callback("正在生成编排脚本...")
        # Different status text -> delivered even within same 8s bucket
        clock[0] = 104.0
        progress_callback("正在校验脚本结构...")
        # Same status within same 8s bucket -> skipped
        clock[0] = 106.0
        progress_callback("正在校验脚本结构...")
        # Next 8s bucket (elapsed 10s -> 1) -> delivered
        clock[0] = 110.0
        progress_callback("正在校验脚本结构...")
        return str(script_path), {"name": "gen", "tools": ["codex"]}

    handler._generate_script_via_ai = MagicMock(side_effect=fake_codegen)
    proj = ProjectContext(project_id="p_1", project_name="P1", root_path=str(tmp_path))

    with (
        patch("src.feishu.handlers.workflow.time.time", side_effect=lambda: clock[0]),
        patch("src.workflow_engine.progress_coalescer.ProgressCoalescer", FakeCoalescer),
        patch("src.feishu.handlers.workflow.threading.Thread"),
    ):
        handler._generate_and_start_workflow(
            message_id="om_already_sent",
            chat_id="oc_gen",
            requirement="Calm generation patches",
            project=proj,
            root_path=str(tmp_path),
            selected_tools=["codex"],
            expected_session_key="sess_gen",
        )

    assert coalescer_kwargs and coalescer_kwargs[0]["debounce_s"] == 8.0
    # 0 redundant initial PATCH + 3 non-duplicate progress updates = 3 PATCHes (not 6)
    assert len(handler.update_card.call_args_list) == 3
    for call in handler.update_card.call_args_list:
        assert call.args[0] == "om_already_sent"
        assert call.args[1]["config"]["update_multi"] is True


def test_stop_during_generation_transitions_clicked_card_inplace(tmp_path: Path) -> None:
    from src.workflow_engine.engine import WorkflowEngine
    from src.workflow_engine.models import PendingWorkflow

    handler = _make_workflow_handler()
    pool = (
        WorkflowAgentBinding(
            agent_id="A1",
            tool_name="codex",
            model_name="gpt-5.4",
            display_name="Codex",
            effort="high",
            role="worker",
        ),
    )
    engine = WorkflowEngine("oc_stop_gen", str(tmp_path))
    engine._project = WorkflowProject(
        status=WorkflowStatus.GENERATING_SCRIPT,
        requirement="Stop gen inplace",
        pending=PendingWorkflow(
            requirement="Stop gen inplace",
            initiator_user_id="ou_1",
            engine_session_key="sess_stop_gen",
            project_id="p1",
            agent_pool=pool,
            orchestrator_agent_id="A1",
        ),
    )
    owner = _WorkflowLifecycleOwner(
        session_key="sess_stop_gen",
        initiator_user_id="ou_1",
        chat_id="oc_stop_gen",
        project_id="p1",
        root_path=str(tmp_path),
        card_message_id=None,
    )
    engine._script_generation_owner = owner
    handler.ctx.workflow_engine_manager.get.return_value = engine
    proj = ProjectContext(project_id="p1", project_name="P1", root_path=str(tmp_path))
    handler._resolve_project_from_id = MagicMock(return_value=proj)

    with patch("src.thread.get_current_sender_id", return_value="ou_1"):
        handler.handle_workflow_stop_running(
            "om_clicked_card",
            "oc_stop_gen",
            "p1",
            {"action": action_ids.WORKFLOW_STOP_RUNNING, "chat_id": "oc_stop_gen", "project_id": "p1"},
        )

    assert engine.project.status == WorkflowStatus.CANCELLED
    handler.update_card.assert_called_once()
    assert handler.update_card.call_args.args[0] == "om_clicked_card"
    assert "Workflow · 已取消" in json.dumps(handler.update_card.call_args.args[1], ensure_ascii=False)
    handler.send_card_to_chat.assert_not_called()


def test_acp_cache_version_invalidates_stale_tool_capabilities_immediately(tmp_path: Path) -> None:
    from src.acp import helper as acp_helper
    from src.acp.options import ACPModelOption, ACPModelSelectionVariant
    from src.workflow_engine.tool_registry import get_tool_capabilities, invalidate_cache

    cwd = str(tmp_path / "repo")
    Path(cwd).mkdir(parents=True, exist_ok=True)
    invalidate_cache()
    acp_helper.invalidate_acp_model_cache("agy", cwd)

    with (
        patch("src.acp.helper.is_programming_tool_available", return_value=True),
        patch("src.acp.providers.get_providers", return_value={"agy": object()}),
        patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}" if name == "agy" else None),
    ):
        cold_caps = get_tool_capabilities(cwd=cwd)
        assert cold_caps["agy"].models == ()

        key = acp_helper._key("agy", cwd)
        with acp_helper._cache_lock:
            gen = acp_helper._cache_generation.get(key, 0)
        assert acp_helper._store(
            key,
            [
                ACPModelOption(
                    name="gemini-4-neon",
                    is_default=True,
                    reasoning_efforts=("low", "medium", "high", "xhigh"),
                    default_reasoning_effort="high",
                    selection_variants=(
                        ACPModelSelectionVariant(name="gemini-4-neon/high", model="gemini-4-neon", effort="high", is_default=True),
                    ),
                )
            ],
            generation=gen,
        )

        # Without force_refresh=True, version bump invalidates stale empty snapshot
        warm_caps = get_tool_capabilities(cwd=cwd)
        assert warm_caps["agy"].models == ("gemini-4-neon",)
        assert warm_caps["agy"].authoritative_models is True

    # Pre-discovery version stamp check: concurrent _store() during _discover_tool_capabilities()
    # must invalidate the pre-stamped snapshot on the very next read.
    invalidate_cache()
    acp_helper.invalidate_acp_model_cache("agy", cwd)
    from src.workflow_engine import tool_registry as tr_mod

    orig_discover = tr_mod._discover_tool_capabilities

    def _racy_discover(*, cwd: str | None = None):
        caps = orig_discover(cwd=cwd)
        r_key = acp_helper._key("agy", cwd)
        with acp_helper._cache_lock:
            r_gen = acp_helper._cache_generation.get(r_key, 0)
        acp_helper._store(
            r_key,
            [ACPModelOption(name="gemini-4-carbon", is_default=True)],
            generation=r_gen,
        )
        return caps

    with (
        patch("src.acp.helper.is_programming_tool_available", return_value=True),
        patch("src.acp.providers.get_providers", return_value={"agy": object()}),
        patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}" if name == "agy" else None),
    ):
        with patch.object(tr_mod, "_discover_tool_capabilities", side_effect=_racy_discover):
            first_caps = get_tool_capabilities(cwd=cwd)
            assert first_caps["agy"].models == ()
        second_caps = get_tool_capabilities(cwd=cwd)
        assert second_caps["agy"].models == ("gemini-4-carbon",)

    acp_helper.invalidate_acp_model_cache("agy", cwd)
    invalidate_cache()


def test_show_workflow_config_preheats_agy_and_refreshes_late_tools_inplace(tmp_path: Path) -> None:
    from src.acp import helper as acp_helper
    from src.acp.options import ACPModelOption, ACPModelSelectionVariant
    from src.workflow_engine.tool_registry import invalidate_cache

    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    cwd = str(repo)

    mgr = ProjectManager(str(tmp_path / "projects.json"))
    ok, _, proj = mgr.create_project("p1", "Proj1", cwd, chat_id="oc_warm")
    assert ok and proj is not None

    handler = _make_workflow_handler()
    handler.ctx.project_manager = mgr
    handler.get_working_dir = MagicMock(return_value=cwd)
    handler._allow_wf_model_preheat_in_test = True

    codex_release = threading.Event()
    update_done = threading.Event()
    updated_cards: list[tuple[str, dict]] = []

    def _capture_update(mid: str, card: dict) -> bool:
        updated_cards.append((mid, card))
        update_done.set()
        return True

    handler.update_card = MagicMock(side_effect=_capture_update)

    def _fake_probe(tool_name: str, probe_cwd: str | None) -> list[ACPModelOption]:
        key = acp_helper._key(tool_name, probe_cwd)
        with acp_helper._cache_lock:
            gen = acp_helper._cache_generation.get(key, 0)
        if tool_name == "agy":
            opts = [
                ACPModelOption(
                    name="gemini-4-neon",
                    is_default=True,
                    reasoning_efforts=("low", "medium", "high", "xhigh"),
                    default_reasoning_effort="high",
                    selection_variants=(
                        ACPModelSelectionVariant(name="gemini-4-neon/high", model="gemini-4-neon", effort="high", is_default=True),
                    ),
                ),
                ACPModelOption(
                    name="gpt-oss-120b-medium",
                    selection_variants=(ACPModelSelectionVariant(name="gpt-oss-120b-medium", model="gpt-oss-120b-medium", is_default=True),),
                ),
            ]
        elif tool_name == "codex":
            codex_release.wait(timeout=2.0)
            opts = [
                ACPModelOption(
                    name="gpt-6-astra",
                    is_default=True,
                    selection_variants=(ACPModelSelectionVariant(name="gpt-6-astra", model="gpt-6-astra", is_default=True),),
                ),
                ACPModelOption(
                    name="gpt-6-luna",
                    reasoning_efforts=("low", "high", "xhigh"),
                    default_reasoning_effort="xhigh",
                    selection_variants=(
                        ACPModelSelectionVariant(name="gpt-6-luna/xhigh", model="gpt-6-luna", effort="xhigh", is_default=True),
                    ),
                ),
            ]
        else:
            opts = []
        acp_helper._store(key, opts, generation=gen)
        return opts

    invalidate_cache()
    acp_helper.invalidate_acp_model_cache("agy", cwd)
    acp_helper.invalidate_acp_model_cache("codex", cwd)

    try:
        with (
            patch("src.acp.providers.get_providers", return_value={"agy": object(), "codex": object()}),
            patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}" if name in {"agy", "codex"} else None),
            patch("src.acp.helper.is_programming_tool_available", return_value=True),
            patch("src.feishu.handlers.workflow_config._WF_CONFIG_INITIAL_WARMUP_BUDGET_S", 0.5),
            patch("src.feishu.handlers.workflow_config._WF_CONFIG_PROBE_TIMEOUT_S", 2.0),
            patch.object(handler, "_probe_single_workflow_tool_models", side_effect=_fake_probe),
        ):
            handler.show_workflow_config("om_init", "oc_warm", proj, requirement="Inspect models")

            # Initial card already has AGY models after switching editor tool to agy!
            assert handler.reply_card.call_count == 1
            select_agy_card, toast_t, _ = handler.handle_workflow_config_callback_sync(
                "om_reply_card",
                "oc_warm",
                "p1",
                {"action": action_ids.WF_SELECT_TOOL, "_option": "agy"},
            )
            assert toast_t != "error"
            raw_agy = json.dumps(select_agy_card, ensure_ascii=False)
            assert "gemini-4-neon" in raw_agy
            assert "gpt-oss-120b-medium" in raw_agy

            # Switch editor tool to codex while codex probe is still blocked, then release codex
            handler.handle_workflow_config_callback_sync(
                "om_reply_card",
                "oc_warm",
                "p1",
                {"action": action_ids.WF_SELECT_TOOL, "_option": "codex"},
            )
            codex_release.set()
            assert update_done.wait(timeout=2.0), "late codex preheat did not refresh config card"
            for wt in list(getattr(handler, "_wf_config_refresh_threads", ())):
                wt.join(timeout=2.0)
            refreshed_mid, refreshed_card = updated_cards[-1]
            assert refreshed_mid == "om_reply_card"
            raw_codex = json.dumps(refreshed_card, ensure_ascii=False)
            assert "gpt-6-astra" in raw_codex
            assert "gpt-6-luna" in raw_codex

            # Launched session blocks subsequent background refresh overwrites
            handler.handle_workflow_config_callback_sync(
                "om_reply_card",
                "oc_warm",
                "p1",
                {"action": action_ids.WF_START_WORKFLOW},
            )
            assert handler._is_wf_config_launched(
                config_session_id="",
                message_id="om_reply_card",
            )
    finally:
        codex_release.set()
        for wt in list(getattr(handler, "_wf_config_refresh_threads", ())):
            wt.join(timeout=2.0)
        acp_helper.invalidate_acp_model_cache("agy", cwd)
        acp_helper.invalidate_acp_model_cache("codex", cwd)
        invalidate_cache()


def test_default_selection_roundtrip_and_effortless_authoritative_validation(tmp_path: Path) -> None:
    from src.acp import helper as acp_helper
    from src.acp.options import ACPModelOption, ACPModelSelectionVariant
    from src.workflow_engine.agent_pool import resolve_effective_model_name
    from src.workflow_engine.tool_registry import ToolCapability, validate_agent_binding_spec

    caps = {
        "agy": ToolCapability(
            tool_name="agy",
            display_name="AGY",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=("gemini-4-neon", "gpt-oss-120b-medium"),
            efforts_by_model={"gemini-4-neon": ("low", "medium", "high")},
            default_effort="high",
            authoritative_models=True,
        ),
        "codex": ToolCapability(
            tool_name="codex",
            display_name="Codex",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=("gpt-6-astra", "gpt-6-luna"),
            efforts_by_model={"gpt-6-luna": ("low", "high", "xhigh")},
            authoritative_models=True,
        ),
        "traex": ToolCapability(
            tool_name="traex",
            display_name="Traex",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=("Doubao-Seed-Evolving",),
            authoritative_models=True,
        ),
        "coco": ToolCapability(
            tool_name="coco",
            display_name="Coco",
            available=True,
            supports_model=True,
            supports_effort=False,
            models=("Doubao-Seed-Evolving",),
            authoritative_models=True,
        ),
    }

    with patch("src.acp.traex_selection.load_traex_model_metadata", return_value=()):
        assert validate_agent_binding_spec("coco", model_name="default", capabilities=caps) == (True, None)
        assert validate_agent_binding_spec("coco", model_name="default", effort="default", capabilities=caps) == (True, None)
        assert validate_agent_binding_spec("agy", model_name="gpt-oss-120b-medium", effort=None, capabilities=caps) == (True, None)
        assert validate_agent_binding_spec("agy", model_name="gpt-oss-120b-medium", effort="high", capabilities=caps)[0] is False
        assert validate_agent_binding_spec("codex", model_name="gpt-6-astra", effort=None, capabilities=caps) == (True, None)
        assert validate_agent_binding_spec("codex", model_name="gpt-6-astra", effort="xhigh", capabilities=caps)[0] is False
        assert validate_agent_binding_spec("traex", model_name="Doubao-Seed-Evolving", effort="high", capabilities=caps)[0] is False

    assert resolve_effective_model_name("coco", "default") is None
    assert resolve_effective_model_name("agy", "default", effort="default") is None
    assert resolve_effective_model_name("traex", "default", effort="default") is None

    # Effortless default model in cache must not get paired with explicit effort
    effortless_codex = [
        ACPModelOption(
            name="gpt-6-astra",
            is_default=True,
            selection_variants=(ACPModelSelectionVariant(name="gpt-6-astra", model="gpt-6-astra", is_default=True),),
        ),
        ACPModelOption(
            name="gpt-6-luna",
            reasoning_efforts=("low", "high", "xhigh"),
            selection_variants=(ACPModelSelectionVariant(name="gpt-6-luna/high", model="gpt-6-luna", effort="high", is_default=True),),
        ),
    ]
    with patch.object(acp_helper, "get_cached_acp_models", return_value=effortless_codex):
        assert resolve_effective_model_name("codex", None, effort="high") == "gpt-6-luna/high"
    with patch.object(acp_helper, "get_cached_acp_models", return_value=effortless_codex[:1]):
        assert resolve_effective_model_name("codex", None, effort="high") == "default/high"

    mgr = ProjectManager(str(tmp_path / "projects.json"))
    ok, _, proj = mgr.create_project("p1", "Proj1", str(tmp_path / "repo"), chat_id="oc_def")
    assert ok and proj is not None

    handler = _make_workflow_handler()
    handler.ctx.project_manager = mgr

    with (
        patch("src.workflow_engine.tool_registry.get_tool_capabilities", return_value=caps),
        patch(
            "src.workflow_engine.tool_registry.get_available_tools",
            return_value={"agy": "AGY", "codex": "Codex", "traex": "Traex", "coco": "Coco"},
        ),
    ):
        handler.show_workflow_config("om_init", "oc_def", proj, requirement="Check default")
        handler.handle_workflow_config_callback_sync("om_reply_card", "oc_def", "p1", {"action": action_ids.WF_SELECT_AGENT, "_option": "__new__"})
        handler.handle_workflow_config_callback_sync("om_reply_card", "oc_def", "p1", {"action": action_ids.WF_SELECT_TOOL, "_option": "agy"})
        # Select specific model with default_reasoning_effort="high", then pick explicit "default" effort -> must NOT snap back to "high"
        _card_model, _, _ = handler.handle_workflow_config_callback_sync(
            "om_reply_card",
            "oc_def",
            "p1",
            {"action": action_ids.WF_SELECT_MODEL, "_option": "gemini-4-neon"},
        )
        assert _card_model is not None
        card_eff, _, _ = handler.handle_workflow_config_callback_sync(
            "om_reply_card",
            "oc_def",
            "p1",
            {"action": action_ids.WF_SELECT_EFFORT, "_option": "default"},
        )
        eff_sels = [
            s
            for s in (card_eff or {}).get("body", {}).get("elements", [])
            if isinstance(s, dict) and s.get("tag") == "select_static" and s.get("name") == "effort"
        ]
        assert eff_sels and eff_sels[0].get("initial_option") == "default"
        _c, toast_add, msg_add = handler.handle_workflow_config_callback_sync(
            "om_reply_card",
            "oc_def",
            "p1",
            {"action": action_ids.WF_ADD_OR_UPDATE_AGENT},
        )
        assert toast_add == "success", msg_add
        st = handler._get_wf_config_state(config_session_id="", chat_id="oc_def", project_id="p1", message_id="om_reply_card")
        assert st is not None
        added_agy = st.members[-1]
        assert added_agy.tool_name == "agy"
        assert added_agy.model_name == "gemini-4-neon"
        assert added_agy.effort is None
        assert added_agy.effective_model_name() == "gemini-4-neon"

        # Switching to model="default" + effort="default" also roundtrips to None/None
        handler.handle_workflow_config_callback_sync("om_reply_card", "oc_def", "p1", {"action": action_ids.WF_SELECT_AGENT, "_option": "__new__"})
        handler.handle_workflow_config_callback_sync("om_reply_card", "oc_def", "p1", {"action": action_ids.WF_SELECT_TOOL, "_option": "agy"})
        handler.handle_workflow_config_callback_sync("om_reply_card", "oc_def", "p1", {"action": action_ids.WF_SELECT_MODEL, "_option": "default"})
        handler.handle_workflow_config_callback_sync("om_reply_card", "oc_def", "p1", {"action": action_ids.WF_SELECT_EFFORT, "_option": "default"})
        _c2, toast_add2, msg_add2 = handler.handle_workflow_config_callback_sync(
            "om_reply_card",
            "oc_def",
            "p1",
            {"action": action_ids.WF_ADD_OR_UPDATE_AGENT},
        )
        assert toast_add2 == "success", msg_add2
        st2 = handler._get_wf_config_state(config_session_id="", chat_id="oc_def", project_id="p1", message_id="om_reply_card")
        assert st2 is not None
        assert st2.members[-1].model_name is None
        assert st2.members[-1].effort is None
        assert st2.members[-1].effective_model_name() is None

