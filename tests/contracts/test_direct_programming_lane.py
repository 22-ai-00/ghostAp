"""Minimal contracts for explicit direct-programming configuration and execution."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.acp.helper import SessionKeyCodec
from src.acp.manager import ACPSessionManager
from src.feishu.handler_context import HandlerContext
from src.feishu.handlers.programming import (
    AidenModeHandler,
    ClaudeModeHandler,
    CocoModeHandler,
    CodexModeHandler,
    GeminiModeHandler,
    TraexModeHandler,
)
from src.feishu.handlers.system import SystemHandler
from src.mode import InteractionMode
from src.project.context import ProjectContext
from src.project.manager import ProjectManager
from tests.helpers.session_call_recorder import SessionCallRecorder

BACKENDS = (
    ("coco", CocoModeHandler, "coco-model"),
    ("claude", ClaudeModeHandler, "claude-model"),
    ("aiden", AidenModeHandler, "aiden-model"),
    ("codex", CodexModeHandler, "codex-model"),
    ("gemini", GeminiModeHandler, "gemini-model"),
    ("traex", TraexModeHandler, "traex-model"),
)


def _context(backend: str, manager) -> HandlerContext:
    managers = {name: MagicMock() for name, _handler, _model in BACKENDS}
    managers[backend] = manager
    mode_manager = MagicMock()
    mode_manager.get_mode.return_value = InteractionMode.SMART
    for name, _handler, _model in BACKENDS:
        getattr(mode_manager, f"is_{name}_mode").return_value = False
    project_manager = MagicMock()
    project_manager.get_active_project.return_value = None
    project_manager.validate_project_path.return_value = (True, "")
    context_manager = MagicMock()
    context_manager.store.get.return_value = None
    return HandlerContext(
        settings=SimpleNamespace(
            thread_programming_enabled=False,
            default_reply_mode="direct",
            card=SimpleNamespace(delivery_api_timeout=1.0),
            coco_execution_timeout=30.0,
            claude_execution_timeout=30.0,
        ),
        api_client_factory=MagicMock(),
        message_callback=MagicMock(),
        coco_manager=managers["coco"],
        claude_manager=managers["claude"],
        aiden_manager=managers["aiden"],
        codex_manager=managers["codex"],
        gemini_manager=managers["gemini"],
        traex_manager=managers["traex"],
        intent_recognizer=MagicMock(),
        scheduler=MagicMock(),
        project_manager=project_manager,
        message_mapper=MagicMock(),
        message_linker=MagicMock(),
        mode_manager=mode_manager,
        context_manager=context_manager,
        deep_engine_manager=MagicMock(),
        progress_reporter=MagicMock(),
        spec_engine_manager=MagicMock(),
        spec_reporter=MagicMock(),
        thread_manager=MagicMock(),
        image_handler_factory=MagicMock(),
        managers=managers,
    )


def _project(backend: str, model: str) -> ProjectContext:
    project = ProjectContext(
        project_id="project-direct",
        project_name="direct-contract",
        root_path="/tmp/direct-lane",
    )
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
        monkeypatch.setattr(
            "src.acp.manager.SyncClaudeCLISession",
            recorder.factory_for_backend(backend),
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


@pytest.mark.parametrize(("backend", "handler_type", "model"), BACKENDS)
def test_explicit_backend_uses_one_selected_factory_and_forwards_the_task(
    backend: str,
    handler_type: type,
    model: str,
    monkeypatch: pytest.MonkeyPatch,
):
    recorder = SessionCallRecorder()
    handler, _ctx, manager, project = _make_lane(
        backend, handler_type, model, recorder, monkeypatch
    )

    assert handler.enter_mode("enter", "chat-direct", project=project)
    session = manager.get_session("chat-direct", project_id=project.project_id)
    assert session is not None
    recorder.observe_manager_session_key(
        session,
        chat_id="chat-direct",
        project_id=project.project_id,
        thread_id=None,
        session_key=SessionKeyCodec.encode("chat-direct", project.project_id, None),
    )
    handler.handle_message("task", "chat-direct", f"task for {backend}", project)

    assert recorder.remote_call_topology() == (
        f"factory:{backend}",
        f"prompt:{backend}",
    )
    assert recorder.prompt_calls[0].model == model
    assert recorder.prompt_calls[0].prompt == f"task for {backend}"


def test_backend_start_failure_keeps_the_previous_project_configuration():
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
    previous = (
        project.acp_tool_name,
        project.acp_model_name,
        project.coco_mode,
        project.codex_mode,
    )

    assert handler.enter_mode("enter", "chat-direct", project=project, silent=True) is False

    assert attempts == ["codex"]
    assert (
        project.acp_tool_name,
        project.acp_model_name,
        project.coco_mode,
        project.codex_mode,
    ) == previous
    assert manager.get_session("chat-direct", project_id=project.project_id) is None


def _configuration_lane(tmp_path):
    storage = tmp_path / "projects.json"
    projects = ProjectManager(str(storage))
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    _, _, first = projects.create_project(
        "first", "first", str(first_root), "chat-first"
    )
    _, _, second = projects.create_project(
        "second", "second", str(second_root), "chat-second"
    )
    assert first is not None and second is not None
    ctx = _context("codex", MagicMock())
    ctx.project_manager = projects
    system = SystemHandler(ctx)
    system.reply_text = MagicMock()
    system.reply_error = MagicMock()
    system.reply_card = MagicMock()
    return storage, projects, first, second, ctx, system


def _install_catalogs(system: SystemHandler, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.feishu.handlers.system.list_acp_tools",
        lambda: [SimpleNamespace(name=name) for name in ("coco", "codex", "claude")],
    )
    system._fetch_acp_models = MagicMock(
        return_value=[SimpleNamespace(name=name) for name in ("old", "new")]
    )


def test_configuration_commands_without_arguments_return_compact_text_summaries(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    _storage, projects, first, _second, _ctx, system = _configuration_lane(tmp_path)
    assert projects.commit_acp_configuration(
        first, tool_name="codex", model_name="old"
    )
    _install_catalogs(system, monkeypatch)

    system.handle_acp_command("acp", "chat-first", "/acp", first)
    system.handle_model_command("model", "chat-first", "/model", first)

    messages = [item.args[1] for item in system.reply_text.call_args_list]
    assert len(messages) == 2
    assert "codex" in messages[0] and "claude" in messages[0]
    assert "codex" in messages[1] and "old" in messages[1] and "new" in messages[1]
    system.reply_card.assert_not_called()
    system.reply_error.assert_not_called()


def test_acp_command_persists_project_locally_and_clears_a_cross_tool_model(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    storage, projects, first, second, _ctx, system = _configuration_lane(tmp_path)
    assert projects.commit_acp_configuration(first, tool_name="coco", model_name="old")
    assert projects.commit_acp_configuration(
        second, tool_name="claude", model_name="second-model"
    )
    _install_catalogs(system, monkeypatch)

    system.handle_acp_command("acp", "chat-first", "/acp codex", first)

    assert (first.acp_tool_name, first.acp_model_name) == ("codex", None)
    assert (second.acp_tool_name, second.acp_model_name) == (
        "claude",
        "second-model",
    )
    reloaded = ProjectManager(str(storage)).get_project_for_chat(
        first.project_id, "chat-first"
    )
    assert reloaded is not None
    assert (reloaded.acp_tool_name, reloaded.acp_model_name) == ("codex", None)


def test_model_command_persists_named_and_default_values_without_cross_project_leakage(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    storage, projects, first, second, _ctx, system = _configuration_lane(tmp_path)
    assert projects.commit_acp_configuration(first, tool_name="codex", model_name="old")
    assert projects.commit_acp_configuration(
        second, tool_name="claude", model_name="second-model"
    )
    _install_catalogs(system, monkeypatch)

    system.handle_model_command("named", "chat-first", "/model new", first)
    assert (first.acp_tool_name, first.acp_model_name) == ("codex", "new")
    system.handle_model_command("default", "chat-first", "/model default", first)

    assert (first.acp_tool_name, first.acp_model_name) == ("codex", None)
    assert (second.acp_tool_name, second.acp_model_name) == (
        "claude",
        "second-model",
    )
    reloaded = ProjectManager(str(storage)).get_project_for_chat(
        first.project_id, "chat-first"
    )
    assert reloaded is not None and reloaded.acp_model_name is None


@pytest.mark.parametrize("command", ("/acp codex", "/model new"))
def test_configuration_save_failure_rolls_back_the_complete_project_selection(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
):
    _storage, projects, first, second, _ctx, system = _configuration_lane(tmp_path)
    assert projects.commit_acp_configuration(first, tool_name="coco", model_name="old")
    assert projects.commit_acp_configuration(
        second, tool_name="claude", model_name="second-model"
    )
    _install_catalogs(system, monkeypatch)
    projects._save_projects = MagicMock(return_value=False)

    if command.startswith("/acp"):
        system.handle_acp_command("config", "chat-first", command, first)
    else:
        system.handle_model_command("config", "chat-first", command, first)

    assert (first.acp_tool_name, first.acp_model_name) == ("coco", "old")
    assert (second.acp_tool_name, second.acp_model_name) == (
        "claude",
        "second-model",
    )
    system.reply_error.assert_called_once()
    system.reply_text.assert_not_called()


def test_saved_configuration_auto_activates_and_forwards_the_pending_task(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    _storage, projects, first, _second, ctx, system = _configuration_lane(tmp_path)
    assert projects.commit_acp_configuration(
        first, tool_name="codex", model_name="saved-model"
    )
    session = SimpleNamespace(session_id="session", message_count=0)
    session_manager = MagicMock()
    session_manager.get_session.return_value = session
    handler = MagicMock()
    handler.current_model = None
    handler._get_session_manager.return_value = session_manager
    ctx.handlers["codex"] = handler
    ctx.scheduler.submit.side_effect = lambda _spec, callback: callback(None)
    system._enter_mode_with_acp_model = MagicMock(return_value=True)
    monkeypatch.setattr("src.thread.get_current_thread_id", lambda: None)

    system.handle_enter_acp_saved_selection(
        "task-message",
        "chat-first",
        "codex",
        first,
        pending_prompt="implement the pending task",
    )

    system._enter_mode_with_acp_model.assert_called_once_with(
        "task-message",
        "chat-first",
        "codex",
        "saved-model",
        first,
        thread_id=None,
    )
    handler.handle_message.assert_called_once_with(
        "task-message",
        "chat-first",
        "implement the pending task",
        first,
    )
    assert first.codex_mode is True
    system.reply_error.assert_not_called()
