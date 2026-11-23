"""Regression tests for multi-group, multi-project, and multi-thread AGY context isolation."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.acp.manager import ACPSessionManager
from src.acp.options import ACPModelOption, ACPModelSelectionVariant
from src.card.builders.project import ProjectBuilder
from src.feishu.handlers.base import BaseHandler
from src.feishu.handlers.diagnostics import DiagnosticsHandler
from src.feishu.handlers.programming import AGYModeHandler, TraexModeHandler
from src.feishu.handlers.project import ProjectHandler
from src.feishu.handlers.system import SystemHandler
from src.mode import InteractionMode, ModeManager
from src.project import (
    ProjectContext,
    ProjectContextManager,
    ProjectManager,
    get_project_acp_model_name,
    get_project_acp_tool_name,
    get_project_programming_mode,
    get_project_programming_snapshot,
)


def _make_fake_acp_session(session_id: str, *, model_name: str | None = None):
    sess = MagicMock()
    sess.session_id = session_id
    sess._agent_type = "agy"
    sess._model_name = model_name
    sess._force_dead = False
    sess._is_agy_cli_session = False
    sess.last_active = 10_000_000_000.0
    sess.message_count = 0
    sess.last_query = ""
    sess.is_server_running.return_value = True
    sess.is_server_healthy.return_value = True
    sess.get_usage_info.return_value = None
    sess.to_snapshot.return_value = {"session_id": session_id, "agent_type": "agy"}
    return sess


def test_agy_session_manager_isolates_acp_and_cli_across_chats_projects_and_threads(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started_acp: list[tuple[str, str | None, str | None]] = []

    def _acp_starter(*, agent_type, cwd, model_name=None, project_id=None, **_kw):
        idx = len(started_acp) + 1
        sid = f"acp-agy-{idx}"
        started_acp.append((cwd, model_name, project_id))
        return _make_fake_acp_session(sid, model_name=model_name), sid, {}

    mgr = ACPSessionManager("agy", session_starter=_acp_starter)
    shared_dir = str(tmp_path / "repo-shared")
    other_dir = str(tmp_path / "repo-other")

    s_a_p1 = mgr.ensure_session("chat-a", cwd=shared_dir, project_id="p1")
    s_b_p1 = mgr.ensure_session("chat-b", cwd=shared_dir, project_id="p1")
    s_a_p2 = mgr.ensure_session("chat-a", cwd=other_dir, project_id="p2")
    s_a_p1_t1 = mgr.ensure_session(
        "chat-a",
        cwd=shared_dir,
        project_id="p1",
        thread_id="thread-1",
    )

    assert len({id(s_a_p1), id(s_b_p1), id(s_a_p2), id(s_a_p1_t1)}) == 4
    assert len({s_a_p1.session_id, s_b_p1.session_id, s_a_p2.session_id, s_a_p1_t1.session_id}) == 4

    # Orchestrator custom effort uses SyncAGYCLISession backend, also keyed independently.
    monkeypatch.setattr("shutil.which", lambda cmd: f"/usr/bin/{cmd}")
    cli_mgr = ACPSessionManager("agy")
    cli_a = cli_mgr.ensure_session(
        "chat-a",
        cwd=shared_dir,
        project_id="p1",
        model_name="gemini-pro-orchestrator/high/medium",
    )
    cli_b = cli_mgr.ensure_session(
        "chat-b",
        cwd=shared_dir,
        project_id="p1",
        model_name="gemini-pro-orchestrator/low/low",
    )
    assert cli_a is not cli_b
    assert cli_a._model_name == "gemini-pro-orchestrator/high/medium"
    assert cli_b._model_name == "gemini-pro-orchestrator/low/low"

    # Ending one chat/project session leaves all sibling ACP/CLI sessions intact.
    mgr.end_session("chat-a", project_id="p1")
    assert mgr.get_session("chat-a", project_id="p1") is None
    assert mgr.get_session("chat-b", project_id="p1") is s_b_p1
    assert mgr.get_session("chat-a", project_id="p2") is s_a_p2
    assert mgr.get_session("chat-a", project_id="p1", thread_id="thread-1") is s_a_p1_t1

    cli_mgr.end_session("chat-a", project_id="p1")
    assert cli_mgr.get_session("chat-a", project_id="p1") is None
    assert cli_mgr.get_session("chat-b", project_id="p1") is cli_b


def test_agy_handler_scoped_model_does_not_leak_across_chats_or_projects(tmp_path) -> None:
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    p1 = ProjectContext("p1", "P1", str(proj_dir))
    p2 = ProjectContext("p2", "P2", str(proj_dir))
    p1.add_chat_id("chat-a")
    p2.add_chat_id("chat-b")

    agy_mgr = MagicMock()
    agy_mgr.get_session.return_value = None
    ensure_calls: list[dict] = []

    def _ensure_session(chat_id: str, **kwargs):
        ensure_calls.append({"chat_id": chat_id, **kwargs})
        return _make_fake_acp_session(f"sess-{len(ensure_calls)}", model_name=kwargs.get("model_name"))

    agy_mgr.ensure_session.side_effect = _ensure_session

    ctx = MagicMock()
    ctx.settings = MagicMock(acp_startup_timeout=20, thread_programming_enabled=False)
    ctx.mode_manager = ModeManager()
    ctx.project_manager = MagicMock()
    ctx.project_manager.validate_project_path.return_value = (True, str(proj_dir))
    ctx.project_manager.get_or_create_project_for_path.return_value = (p2, False)
    ctx.working_dirs = {}
    ctx.working_dir_lock = nullcontext()
    ctx.context_manager = ProjectContextManager()
    ctx.agy_manager = agy_mgr

    handler = AGYModeHandler(ctx)
    handler.reply_text = MagicMock()
    handler.reply_card = MagicMock(return_value="card-1")
    handler.add_reaction = MagicMock()
    handler.register_message_project = MagicMock()

    # Chat A selects an AGY model (also sets legacy handler.current_model).
    handler.set_scoped_model("chat-a", "gemini-pro-orchestrator/high/medium", project=p1)
    handler.current_model = "gemini-pro-orchestrator/high/medium"

    # Chat B enters AGY on P2 (and also with project=None) -> must NOT inherit Chat A's model.
    assert handler._get_model_name_override(p2, chat_id="chat-b") is None
    assert handler._get_model_name_override(None, chat_id="chat-b") is None
    assert handler.enter_mode("m-b", "chat-b", project=p2, silent=True) is True
    assert ensure_calls[-1]["chat_id"] == "chat-b"
    assert ensure_calls[-1]["model_name"] is None

    # Chat A without saved ProjectContext config still sees its own scoped model.
    assert handler._get_model_name_override(p1, chat_id="chat-a") == "gemini-pro-orchestrator/high/medium"

    # Rollback on Chat A restores only Chat A and leaves Chat B's scoped model untouched.
    p3 = ProjectContext("p3", "P3", str(proj_dir))
    had_b, prev_b = handler.set_scoped_model("chat-b", "gemini-4-hydrogen/low", project=p3)
    assert had_b is False and prev_b is None
    had_a, prev_a = handler.set_scoped_model("chat-a", "gemini-3-flash/high", project=p1)
    handler.restore_scoped_model("chat-a", had_a, prev_a, project=p1)
    assert handler._get_model_name_override(p1, chat_id="chat-a") == "gemini-pro-orchestrator/high/medium"
    assert handler._get_model_name_override(p3, chat_id="chat-b") == "gemini-4-hydrogen/low"


def test_shared_project_isolates_per_chat_config_modes_snapshots_persistence_and_rollback(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_file = tmp_path / "projects.json"
    proj_dir = tmp_path / "shared_repo"
    proj_dir.mkdir()

    pm = ProjectManager(storage_path=str(storage_file))
    ok, _msg, project = pm.create_project("shared", "Shared", str(proj_dir), chat_id="chat-dm")
    assert ok and project is not None
    project.bound_chat_id = "chat-grp"
    project.add_chat_id("chat-grp")
    project.add_chat_id("chat-other")

    # Chat-grp activates AGY Orchestrator; chat-dm activates AGY Hydrogen; chat-other switches to Traex.
    assert pm.commit_acp_programming_activation(
        project,
        tool_name="agy",
        model_name="gemini-pro-orchestrator/high/medium",
        session_id="sess-grp-1",
        query_count=2,
        activate_mode=True,
        chat_id="chat-grp",
    )
    project.update_programming_snapshot("agy", "fix orchestrator", 4, chat_id="chat-grp")

    assert pm.commit_acp_programming_activation(
        project,
        tool_name="agy",
        model_name="gemini-4-hydrogen/low",
        session_id="sess-dm-1",
        query_count=1,
        activate_mode=True,
        chat_id="chat-dm",
    )
    assert pm.commit_acp_configuration(
        project,
        tool_name="traex",
        model_name="c_o_new/standard/high",
        chat_id="chat-other",
    )
    project.set_programming_mode("traex", True, session_id="sess-other-1", query_count=5, chat_id="chat-other")

    assert get_project_acp_tool_name(project, "chat-grp") == "agy"
    assert get_project_acp_model_name(project, "chat-grp", tool_name="agy") == "gemini-pro-orchestrator/high/medium"
    assert get_project_programming_mode(project, "agy", "chat-grp") is True
    assert get_project_programming_mode(project, "traex", "chat-grp") is False
    grp_snap = get_project_programming_snapshot(project, "agy", "chat-grp")
    assert grp_snap is not None and grp_snap.session_id == "sess-grp-1" and grp_snap.last_query == "fix orchestrator"

    assert get_project_acp_tool_name(project, "chat-dm") == "agy"
    assert get_project_acp_model_name(project, "chat-dm", tool_name="agy") == "gemini-4-hydrogen/low"
    dm_snap = get_project_programming_snapshot(project, "agy", "chat-dm")
    assert dm_snap is not None and dm_snap.session_id == "sess-dm-1"

    assert get_project_acp_tool_name(project, "chat-other") == "traex"
    assert get_project_acp_model_name(project, "chat-other", tool_name="traex") == "c_o_new/standard/high"

    # Exiting / clearing in chat-dm does NOT clear chat-grp's mode or snapshot.
    project.set_programming_mode("agy", False, chat_id="chat-dm")
    project.clear_session_state("agy", chat_id="chat-dm")
    assert get_project_programming_mode(project, "agy", "chat-dm") is False
    assert get_project_programming_snapshot(project, "agy", "chat-dm") is None
    assert get_project_programming_mode(project, "agy", "chat-grp") is True
    assert get_project_programming_snapshot(project, "agy", "chat-grp") is not None

    # Disk persistence & reload keeps every chat's isolated state intact.
    assert pm.persist_project_context(project) is True
    reloaded_pm = ProjectManager(storage_path=str(storage_file))
    reloaded = reloaded_pm.get_project_for_chat("shared", "chat-grp")
    assert reloaded is not None
    assert get_project_acp_model_name(reloaded, "chat-grp", tool_name="agy") == "gemini-pro-orchestrator/high/medium"
    assert get_project_acp_model_name(reloaded, "chat-dm", tool_name="agy") == "gemini-4-hydrogen/low"
    assert get_project_acp_tool_name(reloaded, "chat-other") == "traex"
    assert get_project_programming_mode(reloaded, "agy", "chat-grp") is True
    assert get_project_programming_mode(reloaded, "agy", "chat-dm") is False

    # Persistence failure rolls back only the failing chat's change.
    monkeypatch.setattr(reloaded_pm, "_save_projects", lambda: False)
    assert (
        reloaded_pm.commit_acp_configuration(
            reloaded,
            tool_name="codex",
            model_name="gpt-5",
            chat_id="chat-dm",
        )
        is False
    )
    assert get_project_acp_tool_name(reloaded, "chat-dm") == "agy"
    assert get_project_acp_model_name(reloaded, "chat-dm", tool_name="agy") == "gemini-4-hydrogen/low"
    assert get_project_acp_model_name(reloaded, "chat-grp", tool_name="agy") == "gemini-pro-orchestrator/high/medium"

    assert (
        reloaded_pm.commit_acp_programming_activation(
            reloaded,
            tool_name="codex",
            model_name="gpt-5",
            session_id="sess-fail",
            query_count=1,
            activate_mode=True,
            chat_id="chat-new",
        )
        is False
    )
    assert "chat-new" not in reloaded.chat_states
    assert get_project_acp_model_name(reloaded, "chat-grp", tool_name="agy") == "gemini-pro-orchestrator/high/medium"

    # LRU eviction purges evicted non-owner chat's chat_states entry.
    fake_settings = SimpleNamespace(max_allowed_chat_ids=3, max_evicted_cache=10)
    monkeypatch.setattr("src.config.get_settings", lambda: fake_settings)
    evicted = reloaded.add_chat_id("chat-fourth")
    assert evicted == "chat-grp"
    assert "chat-grp" not in reloaded.chat_states
    assert "chat-other" in reloaded.chat_states


def test_legacy_project_snapshot_and_single_chat_compat(tmp_path) -> None:
    legacy = ProjectContext.from_snapshot(
        {
            "project_id": "legacy",
            "project_name": "Legacy",
            "root_path": str(tmp_path),
            "acp_tool_name": "agy",
            "acp_model_name": "gemini-4-hydrogen/high",
            "allowed_chat_ids": [["chat-1", 100.0]],
        }
    )
    assert get_project_acp_tool_name(legacy, "chat-1") == "agy"
    assert get_project_acp_model_name(legacy, "chat-1", tool_name="agy") == "gemini-4-hydrogen/high"

    # Once chat-2 configures its own tool, chat-1 keeps legacy config cleanly.
    legacy.add_chat_id("chat-2")
    legacy.set_acp_configuration("traex", "c_o_new", chat_id="chat-2")
    assert get_project_acp_tool_name(legacy, "chat-1") == "agy"
    assert get_project_acp_model_name(legacy, "chat-1", tool_name="agy") == "gemini-4-hydrogen/high"
    assert get_project_acp_tool_name(legacy, "chat-2") == "traex"

    # Single-chat direct attribute assignment still mirrors to sole ChatProjectState.
    single = ProjectContext("single", "Single", str(tmp_path))
    single.set_acp_configuration("agy", "gemini-4-hydrogen", chat_id="sole")
    single.acp_model_name = "gemini-pro-orchestrator/max/high"
    assert get_project_acp_model_name(single, "sole", tool_name="agy") == "gemini-pro-orchestrator/max/high"


def test_unified_context_clear_and_thread_isolation() -> None:
    ctx_mgr = ProjectContextManager()
    project = ProjectContext("p1", "Demo", "/tmp/demo")
    base = object.__new__(BaseHandler)
    base.ctx = SimpleNamespace(context_manager=ctx_mgr)

    # Populate chat-a main, chat-a thread t1, chat-b main on p1, and chat-a on p2.
    ctx_mgr.update_context(
        "p1",
        chat_id="chat-a",
        conversation={"role": "user", "content": "main a", "source_mode": "agy"},
    )
    base.record_mode_transition("p1", InteractionMode.SMART, InteractionMode.AGY, "enter", chat_id="chat-a")
    base.record_mode_transition(
        "p1",
        InteractionMode.SMART,
        InteractionMode.AGY,
        "thread-enter",
        chat_id="chat-a",
        thread_id="thread-1",
    )
    ctx_mgr.update_context(
        "p1",
        chat_id="chat-a",
        thread_id="thread-1",
        conversation={"role": "user", "content": "thread task", "source_mode": "agy"},
    )
    ctx_mgr.update_context(
        "p1",
        chat_id="chat-b",
        conversation={"role": "user", "content": "main b", "source_mode": "agy"},
    )
    ctx_mgr.update_context(
        "p2",
        chat_id="chat-a",
        conversation={"role": "user", "content": "other project", "source_mode": "agy"},
    )

    # Injecting bridge context in main chat does NOT consume thread-1's bridge, and vice versa.
    thread_injected = base.inject_bridge_context("do thread work", project, chat_id="chat-a", thread_id="thread-1")
    assert "main a" not in thread_injected
    main_injected = base.inject_bridge_context("do main work", project, chat_id="chat-a")
    assert "main a" in main_injected
    assert "thread task" not in main_injected

    # /clear removes ONLY the target (chat_id, project_id[, thread_id]) stream.
    assert ctx_mgr.clear("p1", chat_id="chat-a") is True
    assert ctx_mgr.get("p1", chat_id="chat-a") is None
    assert ctx_mgr.get("p1", chat_id="chat-a", thread_id="thread-1") is not None
    assert ctx_mgr.get("p1", chat_id="chat-b") is not None
    assert ctx_mgr.get("p2", chat_id="chat-a") is not None


def test_end_to_end_system_commands_and_cards_isolate_shared_project_chats(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_file = tmp_path / "projects.json"
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    pm = ProjectManager(storage_path=str(storage_file))
    ok, _msg, project = pm.create_project("p-shared", "Shared", str(repo_dir), chat_id="chat-a")
    assert ok and project is not None
    project.add_chat_id("chat-b")
    pm.set_active_project("chat-a", "p-shared")
    pm.set_active_project("chat-b", "p-shared")

    mode_mgr = ModeManager()
    mode_mgr.enter_programming_mode("chat-a", InteractionMode.AGY, project_id="p-shared")
    mode_mgr.enter_programming_mode("chat-b", InteractionMode.AGY, project_id="p-shared")

    sess_a = _make_fake_acp_session("sess-a", model_name="gemini-pro-orchestrator/high/medium")
    sess_a.message_count = 3
    sess_b = _make_fake_acp_session("sess-b", model_name="gemini-4-hydrogen/low")
    sess_b.message_count = 1
    agy_mgr = MagicMock()
    agy_mgr.get_session.side_effect = (
        lambda cid, project_id=None, thread_id=None: sess_a if cid == "chat-a" else sess_b
    )

    ctx_mgr = ProjectContextManager()
    ctx_mgr.update_context(
        "p-shared",
        chat_id="chat-a",
        conversation={"role": "user", "content": "secret A", "source_mode": "agy"},
    )
    ctx_mgr.update_context(
        "p-shared",
        chat_id="chat-b",
        conversation={"role": "user", "content": "secret B", "source_mode": "agy"},
    )

    ctx = MagicMock()
    ctx.project_manager = pm
    ctx.mode_manager = mode_mgr
    ctx.context_manager = ctx_mgr
    ctx.settings = MagicMock(acp_startup_timeout=20, default_reply_mode="normal")
    ctx.working_dirs = OrderedDict()
    ctx.working_dir_lock = nullcontext()
    ctx.agy_manager = agy_mgr
    ctx.traex_manager = MagicMock(get_session=MagicMock(return_value=None))

    agy_handler = AGYModeHandler(ctx)
    agy_handler.reply_text = MagicMock()
    agy_handler.reply_card = MagicMock(return_value="c1")
    agy_handler.add_reaction = MagicMock()
    agy_handler.register_message_project = MagicMock()

    traex_handler = TraexModeHandler(ctx)
    traex_handler.reply_text = MagicMock()
    traex_handler.reply_card = MagicMock(return_value="c2")
    traex_handler.add_reaction = MagicMock()
    traex_handler.register_message_project = MagicMock()

    handlers = {"agy": agy_handler, "traex": traex_handler}
    ctx.get_handler.side_effect = lambda name: handlers.get(name)
    ctx.handlers = handlers

    system = SystemHandler(ctx)
    replies: list[tuple[str, str]] = []
    system.reply_text = lambda mid, body, **_kw: replies.append((mid, body))  # type: ignore[method-assign]
    system.reply_error = lambda mid, body, **_kw: replies.append((mid, f"ERR:{body}"))  # type: ignore[method-assign]

    agy_models = [
        ACPModelOption(
            name="gemini-pro-orchestrator",
            description="Gemini Pro Orchestrator",
            selection_variants=(
                ACPModelSelectionVariant(name="gemini-pro-orchestrator", model="gemini-pro-orchestrator", profile="default", effort=None),
                ACPModelSelectionVariant(name="gemini-pro-orchestrator/high/medium", model="gemini-pro-orchestrator", profile="high", effort="medium"),
            ),
        ),
        ACPModelOption(
            name="gemini-4-hydrogen",
            description="Gemini 4 Hydrogen",
            selection_variants=(
                ACPModelSelectionVariant(name="gemini-4-hydrogen/low", model="gemini-4-hydrogen", profile=None, effort="low"),
                ACPModelSelectionVariant(name="gemini-4-hydrogen/high", model="gemini-4-hydrogen", profile=None, effort="high"),
            ),
        ),
    ]
    monkeypatch.setattr(system, "_fetch_acp_models", lambda *_a, **_kw: agy_models)
    monkeypatch.setattr("src.acp.helper.invalidate_acp_model_cache", lambda *_a, **_kw: None)
    monkeypatch.setattr("src.thread.get_current_thread_id", lambda: None)
    monkeypatch.setattr(
        "src.feishu.handlers.system.list_acp_tools",
        lambda: [SimpleNamespace(name="agy"), SimpleNamespace(name="traex")],
    )

    # Chat A selects Orchestrator high/medium; Chat B selects Hydrogen low.
    system.handle_model_command("m-a1", "chat-a", "/model gemini-pro-orchestrator/high/medium", project)
    system.handle_model_command("m-b1", "chat-b", "/model gemini-4-hydrogen/low", project)
    system.handle_effort_command("m-b2", "chat-b", "/effort high", project)

    assert get_project_acp_model_name(project, "chat-a", tool_name="agy") == "gemini-pro-orchestrator/high/medium"
    assert get_project_acp_model_name(project, "chat-b", tool_name="agy") == "gemini-4-hydrogen/high"

    # Chat B switches to /acp traex -> Chat A stays on AGY Orchestrator high/medium!
    system.handle_acp_command("m-b3", "chat-b", "/acp traex", project)
    assert get_project_acp_tool_name(project, "chat-b") == "traex"
    assert get_project_acp_tool_name(project, "chat-a") == "agy"
    assert get_project_acp_model_name(project, "chat-a", tool_name="agy") == "gemini-pro-orchestrator/high/medium"

    # Diagnostics cockpit & ProjectBuilder status cards show per-chat state.
    diag = DiagnosticsHandler(ctx)
    info_a = diag._collect_direct_cockpit_info("chat-a", project)
    info_b = diag._collect_direct_cockpit_info("chat-b", project)
    assert info_a["tool_name"] == "agy"
    assert info_a["model_name"] == "gemini-pro-orchestrator/high/medium"
    assert info_b["tool_name"] == "traex"
    assert info_b["model_name"] == ""

    project.set_programming_mode("agy", True, session_id="sess-a", query_count=3, chat_id="chat-a")
    _t, board_a = ProjectBuilder.build_status_board_card([project], "p-shared", chat_id="chat-a")
    _t, board_b = ProjectBuilder.build_status_board_card([project], "p-shared", chat_id="chat-b")
    assert "AGY(3)" in board_a
    assert "Traex" in board_b and "AGY(" not in board_b

    # /clear in Chat A clears ONLY Chat A's session & UnifiedContext, keeping Chat B untouched.
    system.handle_clear_command("m-a-clear", "chat-a", project)
    agy_mgr.end_session.assert_called_once_with("chat-a", project_id="p-shared", thread_id=None)
    assert ctx_mgr.get("p-shared", chat_id="chat-a") is None
    assert ctx_mgr.get("p-shared", chat_id="chat-b") is not None

    # ProjectHandler.switch_project uses the calling chat's own saved tool/model.
    other_proj = ProjectContext("p-other", "Other", str(repo_dir))
    pm._projects["p-other"] = other_proj
    pm._active_project["chat-a"] = "p-other"
    proj_handler = ProjectHandler(ctx)
    proj_handler.reply_text = MagicMock()
    proj_handler.reply_card = MagicMock(return_value="c-sw")
    proj_handler.register_message_project = MagicMock()
    captured_enter: list[dict] = []
    agy_handler.enter_mode = lambda mid, cid, **kw: captured_enter.append({"cid": cid, **kw}) or True  # type: ignore[method-assign]
    proj_handler.switch_project("m-sw", "chat-a", "Shared", auto_enter_coco=True)
    assert captured_enter[-1]["cid"] == "chat-a"
    assert captured_enter[-1]["model_override"] == "gemini-pro-orchestrator/high/medium"
