"""Vertical contracts for explicit direct-programming sessions.

These tests deliberately use the real handler and ``ACPSessionManager``.  The
transport is replaced only at its process/session factory boundary, where the
recorder can capture the request without making a remote call.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.acp.manager import ACPSessionManager
from src.feishu.handler_context import HandlerContext
from src.feishu.handlers.programming import (
    AidenModeHandler,
    ClaudeModeHandler,
    CocoModeHandler,
    CodexModeHandler,
    GeminiModeHandler,
    TraexModeHandler,
    TTADKModeHandler,
)
from src.feishu.handlers.system import SystemHandler
from src.feishu.slash_command_parser import SlashCommandParser
from src.mode import InteractionMode
from src.project.context import ProjectContext
from src.project.manager import ProjectManager
from src.ttadk.models import ModelListResult, ToolListResult, TTADKModel, TTADKTool
from tests.helpers.session_call_recorder import SessionCallRecorder

BACKENDS = (
    ("coco", CocoModeHandler, "coco-model"),
    ("claude", ClaudeModeHandler, "claude-model"),
    ("aiden", AidenModeHandler, "aiden-model"),
    ("codex", CodexModeHandler, "codex-model"),
    ("gemini", GeminiModeHandler, "gemini-model"),
    ("traex", TraexModeHandler, "traex-model"),
    ("ttadk", TTADKModeHandler, "ttadk-model"),
)


@pytest.mark.parametrize(
    ("backend", "handler_type", "model"),
    (BACKENDS[1], BACKENDS[2], BACKENDS[4]),
)
def test_public_direct_slash_route_then_current_mode_text_uses_one_factory_and_prompt(
    backend: str,
    handler_type: type,
    model: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """The public slash route, not a handler shortcut, owns direct entry."""
    recorder = SessionCallRecorder()
    handler, ctx, manager, project = _make_lane(backend, handler_type, model, recorder, monkeypatch)
    ctx.handlers[backend] = handler
    system = SystemHandler(ctx)
    system.reply_text = MagicMock()
    system.reply_card = MagicMock()

    system.handle_intercepted_command(
        "message-command",
        "chat-direct",
        f"/{backend}",
        project,
        command_match=SlashCommandParser.parse(f"/{backend}"),
    )
    _bind_selected_session(recorder, manager)
    handler.handle_message("message-current-mode", "chat-direct", f"task for {backend}", project)

    expected_backend = backend
    assert recorder.remote_call_topology() == (
        f"factory:{expected_backend}",
        f"prompt:{expected_backend}",
    )
    assert recorder.prompt_calls[0].model == (None if backend == "claude" else model)


@pytest.mark.parametrize(
    ("backend", "handler_type", "model"),
    (BACKENDS[0], BACKENDS[3], BACKENDS[5]),
)
def test_public_model_selection_route_then_callback_uses_selected_backend(
    backend: str,
    handler_type: type,
    model: str,
    monkeypatch: pytest.MonkeyPatch,
):
    recorder = SessionCallRecorder()
    handler, ctx, manager, project = _make_lane(backend, handler_type, model, recorder, monkeypatch)
    ctx.handlers[backend] = handler
    callbacks = []
    ctx.scheduler.submit.side_effect = lambda _spec, callback: callbacks.append(callback)
    system = SystemHandler(ctx)
    system.reply_text = MagicMock()
    system.reply_card = MagicMock()
    system.update_card = MagicMock(return_value=True)
    monkeypatch.setattr(system, "_fetch_acp_models", lambda *_args, **_kwargs: [])

    system.handle_intercepted_command(
        "message-command", "chat-direct", f"/{backend}", project,
        command_match=SlashCommandParser.parse(f"/{backend}"),
    )
    system.handle_select_acp_model("model-card", "chat-direct", backend, model, project)
    assert callbacks[0](MagicMock()) is True
    _bind_selected_session(recorder, manager)
    handler.handle_message("message-current-mode", "chat-direct", f"task for {backend}", project)

    assert recorder.remote_call_topology() == (f"factory:{backend}", f"prompt:{backend}")


def test_public_ttadk_route_and_combined_callback_reaches_ttadk_session(
    monkeypatch: pytest.MonkeyPatch,
):
    recorder = SessionCallRecorder()
    handler, ctx, manager, project = _make_lane("ttadk", TTADKModeHandler, "ttadk-model", recorder, monkeypatch)
    ctx.handlers["ttadk"] = handler
    ttadk = MagicMock()
    ttadk.get_tools.return_value = ToolListResult(tools=[TTADKTool(name="coco")])
    ttadk.get_models.return_value = ModelListResult(models=[TTADKModel(name="ttadk-model")])
    ttadk.get_current_tool.return_value = "coco"
    ttadk.get_current_model.return_value = "ttadk-model"
    ttadk.set_tool.return_value = True
    ttadk.set_model.return_value = True
    monkeypatch.setattr("src.feishu.handlers.ttadk_commands.get_ttadk_manager", lambda: ttadk)
    monkeypatch.setattr("src.feishu.handlers.ttadk_commands.auto_update_ttadk", lambda: None)
    system = SystemHandler(ctx)
    system.reply_text = MagicMock()
    system.reply_card = MagicMock()
    system.update_card = MagicMock(return_value=True)

    system.handle_intercepted_command(
        "message-command", "chat-direct", "/ttadk", project,
        command_match=SlashCommandParser.parse("/ttadk"),
    )
    system.handle_select_ttadk_combined("combined-card", "chat-direct", "coco", "ttadk-model", project)
    _bind_selected_session(recorder, manager)
    handler.handle_message("message-current-mode", "chat-direct", "task for ttadk", project)

    assert recorder.remote_call_topology() == ("factory:ttadk_coco", "prompt:ttadk_coco")
    assert recorder.prompt_calls[0].model == "ttadk-model"


def _context(manager_key: str, manager: ACPSessionManager) -> HandlerContext:
    settings = SimpleNamespace(
        thread_programming_enabled=True,
        project_allowed_roots=[],
        acp_startup_timeout=2,
        coco_execution_timeout=2,
        claude_execution_timeout=2,
        programming_finalization_reserve_s=0,
    )
    managers = {key: MagicMock() for key, _handler, _model in BACKENDS}
    managers[manager_key] = manager
    ctx = HandlerContext(
        settings=settings,
        api_client_factory=MagicMock(),
        message_callback=MagicMock(),
        coco_manager=managers["coco"],
        claude_manager=managers["claude"],
        aiden_manager=managers["aiden"],
        codex_manager=managers["codex"],
        gemini_manager=managers["gemini"],
        traex_manager=managers["traex"],
        ttadk_manager=managers["ttadk"],
        tui2acp_manager=MagicMock(),
        intent_recognizer=MagicMock(),
        scheduler=MagicMock(),
        project_manager=MagicMock(),
        message_mapper=MagicMock(),
        message_linker=MagicMock(),
        mode_manager=MagicMock(),
        context_manager=MagicMock(),
        deep_engine_manager=MagicMock(),
        progress_reporter=MagicMock(),
        spec_engine_manager=MagicMock(),
        spec_reporter=MagicMock(),
        slock_engine_manager=MagicMock(),
        thread_manager=MagicMock(),
        image_handler_factory=MagicMock(),
        working_dirs={"chat-direct": "/tmp/direct-lane"},
        working_dir_lock=threading.Lock(),
        pending_image_keys={},
        pending_image_lock=threading.Lock(),
        enable_streaming=False,
        managers={manager_key: manager},
        handlers={},
        channel_client_factory=None,
    )
    ctx.project_manager.validate_project_path.return_value = (True, "ok")
    ctx.mode_manager.get_mode.return_value = InteractionMode.SMART
    for name in (
        "coco",
        "claude",
        "aiden",
        "codex",
        "gemini",
        "traex",
        "ttadk",
    ):
        getattr(ctx.mode_manager, f"is_{name}_mode").return_value = False
    ctx.context_manager.store.get.return_value = None
    return ctx


def _project(backend: str, model: str) -> ProjectContext:
    project = ProjectContext(
        project_id="project-direct",
        project_name="direct-contract",
        root_path="/tmp/direct-lane",
    )
    if backend == "ttadk":
        project.ttadk_tool_name = "coco"
        project.ttadk_model_name = model
    else:
        project.acp_tool_name = backend
        project.acp_model_name = model
    return project


def _make_lane(
    backend: str,
    handler_type: type,
    model: str,
    recorder: SessionCallRecorder,
    monkeypatch: pytest.MonkeyPatch,
):
    if backend == "claude":
        monkeypatch.setattr("src.acp.manager.SyncClaudeCLISession", recorder.factory_for_backend(backend))
        manager = ACPSessionManager(backend)
    elif backend == "ttadk":
        monkeypatch.setattr("src.acp.manager.SyncTTADKCLISession", recorder.factory_for_backend("ttadk_coco"))
        monkeypatch.setattr("src.ttadk.get_ttadk_manager", lambda: MagicMock())
        monkeypatch.setattr(
            "src.ttadk.startup_common.precheck_ttadk_startup_model",
            lambda **_kwargs: {"model": model},
        )
        manager = ACPSessionManager(backend)
    else:
        manager = ACPSessionManager(backend, session_starter=recorder.session_factory)

    ctx = _context(backend, manager)
    handler = handler_type(ctx)
    handler.reply_text = MagicMock()
    handler.reply_card = MagicMock()
    handler.add_reaction = MagicMock()
    handler.record_mode_transition = MagicMock()
    handler._acquire_repo_lock = MagicMock(return_value=(None, None, False))
    return handler, ctx, manager, _project(backend, model)


def _bind_selected_session(
    recorder: SessionCallRecorder,
    manager: ACPSessionManager,
    *,
    chat_id: str = "chat-direct",
    project_id: str = "project-direct",
    thread_id: str | None = None,
):
    session = manager.get_session(chat_id, project_id=project_id, thread_id=thread_id)
    assert session is not None
    recorder.observe_manager_session_key(
        session,
        chat_id=chat_id,
        project_id=project_id,
        thread_id=thread_id,
        session_key=manager._session_key(chat_id, project_id, thread_id),
    )
    return session


def test_explicit_codex_uses_exactly_one_backend_prompt_and_no_planner(monkeypatch: pytest.MonkeyPatch):
    recorder = SessionCallRecorder()
    handler, ctx, manager, project = _make_lane("codex", CodexModeHandler, "codex-model", recorder, monkeypatch)

    assert handler.enter_mode("message-enter", "chat-direct", project=project)
    _bind_selected_session(recorder, manager)
    handler.handle_message("message-task", "chat-direct", "fix the direct lane", project)

    assert recorder.remote_call_topology() == ("factory:codex", "prompt:codex")
    prompt = recorder.prompt_calls[0]
    assert prompt.backend == "codex"
    assert prompt.model == "codex-model"
    assert prompt.cwd == "/tmp/direct-lane"
    assert prompt.chat_id == "chat-direct"
    assert prompt.project_id == "project-direct"
    assert prompt.thread_id is None
    assert prompt.tool_filter is None
    assert prompt.prompt == "fix the direct lane"
    for collaborator in (
        ctx.intent_recognizer,
        ctx.deep_engine_manager,
        ctx.spec_engine_manager,
        ctx.slock_engine_manager,
    ):
        collaborator.assert_not_called()


def test_explicit_slash_command_enters_real_handler_then_current_mode_prompts(monkeypatch: pytest.MonkeyPatch):
    recorder = SessionCallRecorder()
    handler, ctx, manager, project = _make_lane("claude", ClaudeModeHandler, "claude-model", recorder, monkeypatch)
    ctx.handlers["claude"] = handler
    system = SystemHandler(ctx)
    system.reply_text = MagicMock()
    system.reply_card = MagicMock()

    system.handle_intercepted_command(
        "message-command",
        "chat-direct",
        "/claude",
        project,
        command_match=SlashCommandParser.parse("/claude"),
    )
    _bind_selected_session(recorder, manager)
    handler.handle_message("message-current-mode", "chat-direct", "continue after slash", project)

    assert recorder.remote_call_topology() == ("factory:claude", "prompt:claude")
    assert recorder.prompt_calls[0].prompt == "continue after slash"
    ctx.intent_recognizer.assert_not_called()


@pytest.mark.parametrize(("backend", "handler_type", "model"), BACKENDS)
def test_explicit_backend_keeps_single_target_factory_and_prompt(
    backend: str,
    handler_type: type,
    model: str,
    monkeypatch: pytest.MonkeyPatch,
):
    recorder = SessionCallRecorder()
    handler, _ctx, manager, project = _make_lane(backend, handler_type, model, recorder, monkeypatch)

    assert handler.enter_mode("message-enter", "chat-direct", project=project)
    _bind_selected_session(recorder, manager)
    handler.handle_message("message-task", "chat-direct", f"task for {backend}", project)

    expected_backend = "ttadk_coco" if backend == "ttadk" else backend
    assert recorder.remote_call_topology() == (
        f"factory:{expected_backend}",
        f"prompt:{expected_backend}",
    )
    # Claude's current CLI factory accepts only cwd; its selected model is
    # intentionally recorded as absent here so Task 0.3 can tighten that
    # separate, known transport contract without disguising the baseline.
    assert recorder.prompt_calls[0].model == (None if backend == "claude" else model)


def test_direct_lane_preserves_chat_project_thread_session_key(monkeypatch: pytest.MonkeyPatch):
    recorder = SessionCallRecorder()
    handler, _ctx, manager, project = _make_lane("codex", CodexModeHandler, "codex-model", recorder, monkeypatch)

    with monkeypatch.context() as scoped:
        scoped.setattr("src.thread.get_current_thread_id", lambda: "thread-direct")
        assert handler.enter_mode("message-enter", "chat-direct", project=project, thread_id="thread-direct")
        selected = _bind_selected_session(recorder, manager, thread_id="thread-direct")
        handler.handle_message("message-one", "chat-direct", "first task", project)
        handler.handle_message("message-two", "chat-direct", "continue task", project)

    assert len(recorder.factory_calls) == 1
    expected_key = manager._session_key("chat-direct", "project-direct", "thread-direct")
    assert [call.session_key for call in recorder.prompt_calls] == [expected_key, expected_key]
    assert manager.get_session("chat-direct", project_id="project-direct", thread_id="thread-direct") is selected


def test_backend_start_failure_does_not_persist_selected_project_state(monkeypatch: pytest.MonkeyPatch):
    attempts: list[str] = []

    def failing_factory(*, agent_type: str, **_kwargs):
        attempts.append(agent_type)
        raise RuntimeError("codex unavailable")

    manager = ACPSessionManager("codex", session_starter=failing_factory)
    ctx = _context("codex", manager)
    handler = CodexModeHandler(ctx)
    handler.reply_text = MagicMock()
    handler.reply_card = MagicMock()
    handler.add_reaction = MagicMock()
    handler.record_mode_transition = MagicMock()
    project = _project("coco", "previous-model")
    project.coco_mode = True
    previous_state = (project.acp_tool_name, project.acp_model_name, project.coco_mode, project.codex_mode)

    assert handler.enter_mode("message-enter", "chat-direct", project=project, silent=True) is False

    assert attempts == ["codex"]
    assert (project.acp_tool_name, project.acp_model_name, project.coco_mode, project.codex_mode) == previous_state
    assert ctx.mode_manager.enter_programming_mode.call_count == 0
    assert manager.get_session("chat-direct", project_id="project-direct") is None


def test_scheduled_acp_startup_failure_keeps_previous_project_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    """A real SystemHandler activation cannot pre-commit its selection."""
    def failing_factory(*, agent_type: str, **_kwargs):
        assert agent_type == "codex"
        raise RuntimeError("codex unavailable")

    manager = ACPSessionManager("codex", session_starter=failing_factory)
    ctx = _context("codex", manager)
    project_manager = ProjectManager(str(tmp_path / "projects.json"))
    _, _, project = project_manager.create_project(
        "project-direct",
        "direct-contract",
        str(tmp_path),
        "chat-direct",
    )
    assert project is not None
    project.acp_tool_name = "codex"
    project.acp_model_name = "previous-model"
    project.set_programming_mode("coco", True, "previous-session", 2)
    ctx.project_manager = project_manager

    handler = CodexModeHandler(ctx)
    handler.reply_text = MagicMock()
    handler.reply_card = MagicMock()
    handler.add_reaction = MagicMock()
    handler.record_mode_transition = MagicMock()
    handler.current_model = "previous-handler-model"
    ctx.handlers["codex"] = handler
    system = SystemHandler(ctx)
    system.reply_text = MagicMock()
    system.reply_card = MagicMock()
    system.update_card = MagicMock(return_value=True)
    submitted = []
    ctx.scheduler.submit.side_effect = lambda spec, callback: submitted.append(callback)

    before = (
        project.acp_tool_name,
        project.acp_model_name,
        project.coco_mode,
        project.codex_mode,
        project.coco_session_snapshot,
        project.codex_session_snapshot,
    )
    system.handle_select_acp_model(
        "selection-card",
        "chat-direct",
        "codex",
        "new-model",
        project,
    )

    assert submitted
    assert (
        project.acp_tool_name,
        project.acp_model_name,
        project.coco_mode,
        project.codex_mode,
        project.coco_session_snapshot,
        project.codex_session_snapshot,
    ) == before
    assert handler.current_model == "previous-handler-model"

    assert submitted[0](MagicMock()) is False
    assert (
        project.acp_tool_name,
        project.acp_model_name,
        project.coco_mode,
        project.codex_mode,
        project.coco_session_snapshot,
        project.codex_session_snapshot,
    ) == before
    assert handler.current_model == "previous-handler-model"
    assert manager.get_session("chat-direct", project_id=project.project_id) is None


def test_later_scheduled_selection_owns_project_commit(monkeypatch: pytest.MonkeyPatch, tmp_path):
    recorder = SessionCallRecorder()
    manager = ACPSessionManager("codex", session_starter=recorder.session_factory)
    ctx = _context("codex", manager)
    project_manager = ProjectManager(str(tmp_path / "projects.json"))
    _, _, project = project_manager.create_project(
        "project-direct",
        "direct-contract",
        str(tmp_path),
        "chat-direct",
    )
    assert project is not None
    project.acp_tool_name = "coco"
    project.acp_model_name = "previous-model"
    project.set_programming_mode("coco", True, "previous-session", 1)
    ctx.project_manager = project_manager
    handler = CodexModeHandler(ctx)
    handler.reply_text = MagicMock()
    handler.reply_card = MagicMock()
    handler.add_reaction = MagicMock()
    handler.record_mode_transition = MagicMock()
    ctx.handlers["codex"] = handler
    system = SystemHandler(ctx)
    system.reply_text = MagicMock()
    system.reply_card = MagicMock()
    system.update_card = MagicMock(return_value=True)
    callbacks = []
    ctx.scheduler.submit.side_effect = lambda spec, callback: callbacks.append(callback)

    system.handle_select_acp_model("old-card", "chat-direct", "codex", "old", project)
    system.handle_select_acp_model("new-card", "chat-direct", "codex", "new", project)

    assert callbacks[0](MagicMock()) is False
    assert callbacks[1](MagicMock()) is True
    assert (project.acp_tool_name, project.acp_model_name) == ("codex", "new")
    assert project.codex_mode is True
    assert project.coco_mode is False
    assert [call.model for call in recorder.factory_calls] == ["new"]


def test_interleaved_activation_keeps_newer_session_and_selection(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """A stale callback that has started cannot commit or retain its session."""
    recorder = SessionCallRecorder()
    manager = ACPSessionManager("codex", session_starter=recorder.session_factory)
    ctx = _context("codex", manager)
    projects = ProjectManager(str(tmp_path / "projects.json"))
    _, _, project = projects.create_project("project-direct", "direct", str(tmp_path), "chat-direct")
    assert project is not None
    project.acp_tool_name = "coco"
    project.acp_model_name = "previous"
    project.set_programming_mode("coco", True, "previous-session", 1)
    ctx.project_manager = projects
    handler = CodexModeHandler(ctx)
    handler.reply_text = MagicMock()
    handler.reply_card = MagicMock()
    handler.add_reaction = MagicMock()
    handler.record_mode_transition = MagicMock()
    ctx.handlers["codex"] = handler
    system = SystemHandler(ctx)
    system.update_card = MagicMock(return_value=True)
    callbacks = []
    ctx.scheduler.submit.side_effect = lambda _spec, callback: callbacks.append(callback)

    first_started = threading.Event()
    release_first = threading.Event()
    original_enter = handler.enter_mode
    enter_count = 0

    def gated_enter(*args, **kwargs):
        nonlocal enter_count
        result = original_enter(*args, **kwargs)
        enter_count += 1
        if enter_count == 1:
            first_started.set()
            assert release_first.wait(timeout=2)
        return result

    monkeypatch.setattr(handler, "enter_mode", gated_enter)
    system.handle_select_acp_model("old-card", "chat-direct", "codex", "old", project)
    old_thread = threading.Thread(target=lambda: callbacks[0](MagicMock()))
    old_thread.start()
    assert first_started.wait(timeout=2)

    system.handle_select_acp_model("new-card", "chat-direct", "codex", "new", project)
    assert callbacks[1](MagicMock()) is True
    release_first.set()
    old_thread.join(timeout=2)
    assert not old_thread.is_alive()

    assert (project.acp_tool_name, project.acp_model_name) == ("codex", "new")
    assert manager.get_session("chat-direct", project_id=project.project_id) is recorder.sessions[-1]
    assert [call.model for call in recorder.factory_calls] == ["old", "new"]
    assert system._acp_activation_tokens == {}


def test_released_activation_token_never_aliases_a_later_selection(tmp_path):
    recorder = SessionCallRecorder()
    manager = ACPSessionManager("codex", session_starter=recorder.session_factory)
    ctx = _context("codex", manager)
    project_manager = ProjectManager(str(tmp_path / "projects.json"))
    _, _, project = project_manager.create_project(
        "project-direct",
        "direct-contract",
        str(tmp_path),
        "chat-direct",
    )
    assert project is not None
    ctx.project_manager = project_manager
    handler = CodexModeHandler(ctx)
    handler.reply_text = MagicMock()
    handler.reply_card = MagicMock()
    handler.add_reaction = MagicMock()
    handler.record_mode_transition = MagicMock()
    ctx.handlers["codex"] = handler
    system = SystemHandler(ctx)
    system.update_card = MagicMock(return_value=True)
    callbacks = []
    ctx.scheduler.submit.side_effect = lambda spec, callback: callbacks.append(callback)

    system.handle_select_acp_model("first", "chat-direct", "codex", "first", project)
    assert callbacks[0](MagicMock()) is True
    system.handle_select_acp_model("second", "chat-direct", "codex", "second", project)

    assert callbacks[0](MagicMock()) is False
    assert [call.model for call in recorder.factory_calls] == ["first"]


def test_no_project_activation_keeps_normal_auto_create_mode_entry(tmp_path):
    recorder = SessionCallRecorder()
    manager = ACPSessionManager("codex", session_starter=recorder.session_factory)
    ctx = _context("codex", manager)
    project = ProjectContext("created", "created", str(tmp_path))
    ctx.project_manager.get_active_project.return_value = None
    ctx.project_manager.get_or_create_project_for_path.return_value = (project, False)
    handler = CodexModeHandler(ctx)
    handler.reply_text = MagicMock()
    handler.reply_card = MagicMock()
    handler.add_reaction = MagicMock()
    handler.record_mode_transition = MagicMock()
    ctx.handlers["codex"] = handler
    system = SystemHandler(ctx)
    system.update_card = MagicMock(return_value=True)
    callbacks = []
    ctx.scheduler.submit.side_effect = lambda spec, callback: callbacks.append(callback)

    system.handle_select_acp_model("selection", "chat-direct", "codex", "new", None)

    assert callbacks[0](MagicMock()) is True
    ctx.mode_manager.enter_programming_mode.assert_called_once_with(
        "chat-direct", InteractionMode.CODEX, project_id="created"
    )


def test_model_command_inactive_start_commits_only_after_real_start(tmp_path):
    recorder = SessionCallRecorder()
    manager = ACPSessionManager("codex", session_starter=recorder.session_factory)
    ctx = _context("codex", manager)
    projects = ProjectManager(str(tmp_path / "projects.json"))
    _, _, project = projects.create_project("project-direct", "direct", str(tmp_path), "chat-direct")
    assert project is not None
    project.acp_tool_name = "codex"
    project.acp_model_name = "old"
    project.set_programming_mode("codex", True, "old-session", 1)
    ctx.project_manager = projects
    handler = CodexModeHandler(ctx)
    handler.reply_text = MagicMock()
    handler.reply_card = MagicMock()
    handler.add_reaction = MagicMock()
    handler.record_mode_transition = MagicMock()
    ctx.handlers["codex"] = handler
    system = SystemHandler(ctx)
    system.update_card = MagicMock(return_value=True)

    system.handle_model_command("model", "chat-direct", "/model new", project)

    assert (project.acp_tool_name, project.acp_model_name) == ("codex", "new")
    assert project.codex_mode is True
    assert [call.model for call in recorder.factory_calls] == ["new"]


def test_model_command_start_failure_preserves_real_project_state(tmp_path):
    def fail_start(**_kwargs):
        raise RuntimeError("codex unavailable")

    manager = ACPSessionManager("codex", session_starter=fail_start)
    ctx = _context("codex", manager)
    projects = ProjectManager(str(tmp_path / "projects.json"))
    _, _, project = projects.create_project("project-direct", "direct", str(tmp_path), "chat-direct")
    assert project is not None
    project.acp_tool_name = "codex"
    project.acp_model_name = "old"
    project.set_programming_mode("codex", True, "old-session", 1)
    before = (project.acp_tool_name, project.acp_model_name, project.codex_mode, project.codex_session_snapshot)
    ctx.project_manager = projects
    handler = CodexModeHandler(ctx)
    handler.reply_text = MagicMock()
    handler.reply_card = MagicMock()
    handler.add_reaction = MagicMock()
    handler.record_mode_transition = MagicMock()
    ctx.handlers["codex"] = handler
    system = SystemHandler(ctx)
    system.update_card = MagicMock(return_value=True)

    system.handle_model_command("model", "chat-direct", "/model new", project)

    assert (project.acp_tool_name, project.acp_model_name, project.codex_mode, project.codex_session_snapshot) == before


def _model_command_lane(tmp_path, *, active: bool):
    recorder = SessionCallRecorder()
    manager = ACPSessionManager("codex", session_starter=recorder.session_factory)
    ctx = _context("codex", manager)
    projects = ProjectManager(str(tmp_path / "projects.json"))
    _, _, project = projects.create_project("project-direct", "direct", str(tmp_path), "chat-direct")
    assert project is not None
    project.acp_tool_name = "codex"
    project.acp_model_name = "old"
    if active:
        project.set_programming_mode("codex", True, "old-session", 2)
        ctx.mode_manager.is_codex_mode.return_value = True
        manager.ensure_session("chat-direct", cwd=str(tmp_path), project_id=project.project_id, model_name="old")
    ctx.project_manager = projects
    handler = CodexModeHandler(ctx)
    handler.reply_text = MagicMock()
    handler.reply_card = MagicMock()
    handler.add_reaction = MagicMock()
    handler.record_mode_transition = MagicMock()
    ctx.handlers["codex"] = handler
    system = SystemHandler(ctx)
    system.update_card = MagicMock(return_value=True)
    return recorder, manager, ctx, project, system


def test_model_command_active_protocol_switch_commits_selected_model(tmp_path):
    recorder, manager, _ctx, project, system = _model_command_lane(tmp_path, active=True)

    system.handle_model_command("model", "chat-direct", "/model new", project)

    session = manager.get_session("chat-direct", project_id=project.project_id)
    assert session is recorder.sessions[0]
    assert (project.acp_tool_name, project.acp_model_name, project.codex_mode) == ("codex", "new", True)
    assert [event.kind for event in recorder.events].count("set_model") == 1
    assert [event.kind for event in recorder.events].count("factory") == 1


def test_model_command_active_restart_commits_selected_model(tmp_path):
    recorder, manager, _ctx, project, system = _model_command_lane(tmp_path, active=True)
    recorder.sessions[0].set_model = lambda _model: False

    system.handle_model_command("model", "chat-direct", "/model new", project)

    assert len(recorder.sessions) == 2
    assert manager.get_session("chat-direct", project_id=project.project_id) is recorder.sessions[1]
    assert (project.acp_tool_name, project.acp_model_name, project.codex_mode) == ("codex", "new", True)


def test_model_command_inactive_start_activates_selected_model(tmp_path):
    recorder, manager, _ctx, project, system = _model_command_lane(tmp_path, active=False)

    system.handle_model_command("model", "chat-direct", "/model new", project)

    assert manager.get_session("chat-direct", project_id=project.project_id) is recorder.sessions[0]
    assert (project.acp_tool_name, project.acp_model_name, project.codex_mode) == ("codex", "new", True)


def test_model_command_failed_active_restart_keeps_complete_old_selection(tmp_path):
    recorder, manager, _ctx, project, system = _model_command_lane(tmp_path, active=True)
    recorder.sessions[0].set_model = lambda _model: False
    manager._session_starter = lambda **_kwargs: (_ for _ in ()).throw(TimeoutError("codex unavailable"))
    before = (
        project.acp_tool_name,
        project.acp_model_name,
        project.codex_mode,
        project.codex_session_snapshot,
    )

    system.handle_model_command("model", "chat-direct", "/model new", project)

    assert (
        project.acp_tool_name,
        project.acp_model_name,
        project.codex_mode,
        project.codex_session_snapshot,
    ) == before
    assert [call.backend for call in recorder.factory_calls] == ["codex"]


def test_system_thread_exit_cancels_only_selected_thread_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    recorder = SessionCallRecorder()
    manager = ACPSessionManager("codex", session_starter=recorder.session_factory)
    ctx = _context("codex", manager)
    project_manager = ProjectManager(str(tmp_path / "projects.json"))
    _, _, project = project_manager.create_project(
        "project-direct",
        "direct-contract",
        str(tmp_path),
        "chat-direct",
    )
    assert project is not None
    project.set_programming_mode("coco", True, "top-level-session", 2)
    ctx.project_manager = project_manager
    ctx.mode_manager.get_mode.return_value = InteractionMode.CODEX
    selected = manager.ensure_session(
        "chat-direct",
        cwd=str(tmp_path),
        project_id=project.project_id,
        thread_id="thread-direct",
    )
    handler = CodexModeHandler(ctx)
    handler.reply_text = MagicMock()
    handler.reply_card = MagicMock()
    handler.add_reaction = MagicMock()
    ctx.handlers["codex"] = handler
    system = SystemHandler(ctx)
    system.reply_text = MagicMock()

    with monkeypatch.context() as scoped:
        scoped.setattr("src.thread.get_current_thread_id", lambda: "thread-direct")
        system.exit_current_mode("exit-card", "chat-direct", project)

    assert recorder.cancelled_sessions == [selected]
    assert selected.cancel_count == 1
    assert manager.get_session(
        "chat-direct",
        project_id=project.project_id,
        thread_id="thread-direct",
    ) is None
    assert project.coco_mode is True
    ctx.mode_manager.exit_to_smart.assert_not_called()
