"""Minimal contracts for explicit direct-programming configuration and execution."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from lark_channel.event.callback.model.p2_card_action_trigger import (
    P2CardActionTrigger,
)

from src.acp.helper import SessionKeyCodec
from src.acp.manager import ACPSessionManager
from src.acp.options import ACPModelOption, ACPModelSelectionVariant
from src.card.actions import dispatch as action_ids
from src.card.builders.system import SystemBuilder
from src.card.ui_text import UI_TEXT
from src.feishu.action_registry import init_action_registry
from src.feishu.dispatcher import MessageDispatcher
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
from src.feishu.ws_client import FeishuWSClient
from src.mode import InteractionMode
from src.project.context import ProjectContext
from src.project.manager import ProjectManager
from src.thread import set_current_thread_id
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
        return_value=[ACPModelOption(name=name) for name in ("old", "new")]
    )


def test_configuration_commands_keep_acp_summary_and_open_model_selector(
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
    assert len(messages) == 1
    assert "codex" in messages[0] and "claude" in messages[0]
    assert system.reply_card.call_count == 2
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
    system.update_card = MagicMock()
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
    system.update_card.assert_not_called()


@pytest.mark.parametrize("tool_name", tuple(item[0] for item in BACKENDS))
def test_explicit_direct_mode_entry_opens_the_shared_model_selector(
    tmp_path,
    tool_name: str,
):
    _storage, _projects, first, _second, ctx, system = _configuration_lane(tmp_path)
    handler = MagicMock()
    ctx.handlers[tool_name] = handler
    system.show_explicit_acp_model_selection = MagicMock()

    system._handle_direct_mode_enter("entry", "chat-first", tool_name, first)

    system.show_explicit_acp_model_selection.assert_called_once_with(
        "entry",
        "chat-first",
        tool_name,
        first,
    )
    handler.enter_mode.assert_not_called()


def test_model_command_without_arguments_opens_the_shared_selector(
    tmp_path,
):
    _storage, _projects, first, _second, _ctx, system = _configuration_lane(tmp_path)
    first.acp_tool_name = "codex"
    first.acp_model_name = "gpt-5.6-sol/high"
    system.show_explicit_acp_model_selection = MagicMock()

    system.handle_model_command("model", "chat-first", "/model", first)

    system.show_explicit_acp_model_selection.assert_called_once_with(
        "model",
        "chat-first",
        "codex",
        first,
    )
    system.reply_text.assert_not_called()


def test_explicit_selector_builds_one_cascade_card_with_the_saved_model(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    _storage, _projects, first, _second, _ctx, system = _configuration_lane(tmp_path)
    assert system.project_manager.commit_acp_configuration(
        first,
        tool_name="codex",
        model_name="gpt-5.6-sol/high",
    )
    models = [
        ACPModelOption(
            name="gpt-5.6-sol",
            is_default=True,
            selection_variants=(
                ACPModelSelectionVariant(
                    name="gpt-5.6-sol/high",
                    model="gpt-5.6-sol",
                    profile=None,
                    effort="high",
                    is_default=True,
                ),
            ),
        )
    ]
    system._fetch_acp_models = MagicMock(return_value=models)
    monkeypatch.setattr(
        "src.feishu.handlers.system.list_acp_tools",
        lambda: [SimpleNamespace(name="codex")],
    )
    captured: dict = {}

    def build_card(received_models, tool_name, **kwargs):
        captured.update(
            models=received_models,
            tool_name=tool_name,
            **kwargs,
        )
        return "interactive", "selector-card"

    monkeypatch.setattr(
        SystemBuilder,
        "build_acp_model_cascade_card",
        build_card,
        raising=False,
    )
    system.reply_card.return_value = "om-loading"
    system.update_card = MagicMock(return_value=True)

    system.show_explicit_acp_model_selection(
        "entry",
        "chat-first",
        "codex",
        first,
    )

    assert captured["models"] is models
    assert captured["tool_name"] == "codex"
    assert captured["project_id"] == first.project_id
    assert captured["current_model"] == "gpt-5.6-sol/high"
    system.reply_card.assert_called_once()
    assert "模型加载中" in system.reply_card.call_args.args[1]
    system.update_card.assert_called_once_with("om-loading", "selector-card")


@pytest.mark.parametrize("tool_name", tuple(item[0] for item in BACKENDS))
def test_pure_enter_intent_uses_selector_but_pending_task_stays_automatic(
    tmp_path,
    tool_name: str,
):
    _storage, _projects, first, _second, _ctx, system = _configuration_lane(tmp_path)
    handlers = {
        name: MagicMock() for name, _handler_type, _model in BACKENDS
    }
    handlers.update(
        system=system,
        project=MagicMock(),
    )
    client = SimpleNamespace(
        _handler_ctx=SimpleNamespace(handlers=handlers),
        _mode_manager=MagicMock(),
    )
    for name, _handler_type, _model in BACKENDS:
        getattr(client._mode_manager, f"is_{name}_mode").return_value = False
    dispatcher = MessageDispatcher(client)
    system.show_explicit_acp_model_selection = MagicMock()

    if tool_name == "coco":
        dispatcher._handle_enter_coco("entry", "chat-first", first)
    else:
        dispatcher._handle_enter_acp_mode(
            tool_name,
            "entry",
            "chat-first",
            first,
        )

    system.show_explicit_acp_model_selection.assert_called_once_with(
        "entry",
        "chat-first",
        tool_name,
        first,
    )
    handlers[tool_name].enter_mode.assert_not_called()

    system.show_explicit_acp_model_selection.reset_mock()
    if tool_name == "coco":
        dispatcher._handle_enter_coco(
            "task",
            "chat-first",
            first,
            pending_prompt="implement it",
        )
    else:
        dispatcher._handle_enter_acp_mode(
            tool_name,
            "task",
            "chat-first",
            first,
            pending_prompt="implement it",
        )

    system.show_explicit_acp_model_selection.assert_not_called()
    handlers[tool_name].enter_mode.assert_called_once_with(
        "task",
        "chat-first",
        project=first,
    )
    handlers[tool_name].handle_message.assert_called_once_with(
        "task",
        "chat-first",
        "implement it",
        first,
    )


def _action_registry_client(project, system):
    handlers = {
        name: MagicMock() for name, _handler_type, _model in BACKENDS
    }
    handlers.update(
        system=system,
        project=MagicMock(),
        deep=MagicMock(),
        spec=MagicMock(),
        workflow=MagicMock(),
    )
    project_manager = MagicMock()
    project_manager.get_project_for_chat.return_value = project
    project_manager.get_active_project.return_value = project
    return SimpleNamespace(
        _handler_ctx=SimpleNamespace(handlers=handlers),
        _project_manager=project_manager,
    )


@pytest.mark.parametrize("tool_name", tuple(item[0] for item in BACKENDS))
def test_enter_tool_card_buttons_use_the_shared_explicit_controller(
    tmp_path,
    tool_name: str,
):
    _storage, _projects, first, _second, _ctx, system = _configuration_lane(tmp_path)
    system.show_explicit_acp_model_selection = MagicMock()
    client = _action_registry_client(first, system)

    actions = init_action_registry(client)
    actions[f"enter_{tool_name}"](
        "card-message",
        "chat-first",
        first.project_id,
        {},
    )

    system.show_explicit_acp_model_selection.assert_called_once_with(
        "card-message",
        "chat-first",
        tool_name,
        first,
        origin_message_id="card-message",
    )
    client._handler_ctx.handlers[tool_name].handle_card_enter.assert_not_called()


def test_model_actions_are_registered_to_the_system_controller(tmp_path):
    _storage, _projects, first, _second, _ctx, system = _configuration_lane(tmp_path)
    system.handle_select_acp_model = MagicMock()
    system.handle_acp_model_cascade_select = MagicMock()
    system.handle_refresh_acp_models = MagicMock()
    actions = init_action_registry(_action_registry_client(first, system))
    value = {"tool_name": "codex"}

    actions[action_ids.SELECT_ACP_MODEL]("m", "c", first.project_id, value)
    for action in (
        action_ids.SELECT_ACP_MODEL_GROUP,
        action_ids.SELECT_ACP_MODEL_PROFILE,
        action_ids.SELECT_ACP_MODEL_EFFORT,
    ):
        actions[action]("m", "c", first.project_id, {**value, "action": action})
    actions[action_ids.REFRESH_ACP_MODELS]("m", "c", first.project_id, value)

    system.handle_select_acp_model.assert_called_once_with(
        "m", "c", first.project_id, value
    )
    assert system.handle_acp_model_cascade_select.call_count == 3
    system.handle_refresh_acp_models.assert_called_once_with(
        "m", "c", first.project_id, value
    )


def test_cascade_redraw_uses_the_server_visible_option_and_resets_downstream(
    tmp_path,
):
    _storage, _projects, first, _second, _ctx, system = _configuration_lane(tmp_path)
    system.show_explicit_acp_model_selection = MagicMock()

    system.handle_acp_model_cascade_select(
        "card-message",
        "chat-first",
        first.project_id,
        {
            "action": action_ids.SELECT_ACP_MODEL_GROUP,
            "tool_name": "codex",
            "model_group": "stale-model",
            "model_profile": "max",
            "model_effort": "ultra",
            "_option": {"value": "gpt-5.6-sol"},
        },
    )

    system.show_explicit_acp_model_selection.assert_called_once_with(
        "card-message",
        "chat-first",
        "codex",
        first,
        origin_message_id="card-message",
        pending_group="gpt-5.6-sol",
        pending_profile=None,
        pending_effort=None,
        show_loading=False,
    )


def test_final_model_callback_recomposes_from_live_capabilities_and_ignores_raw_model(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    _storage, _projects, first, _second, _ctx, system = _configuration_lane(tmp_path)
    models = [
        ACPModelOption(
            name="gpt-5.6-sol",
            is_default=True,
            selection_variants=(
                ACPModelSelectionVariant(
                    name="gpt-5.6-sol/high",
                    model="gpt-5.6-sol",
                    profile=None,
                    effort="high",
                    is_default=True,
                ),
            ),
        )
    ]
    system._fetch_acp_models = MagicMock(return_value=models)
    system._activate_acp_selection = MagicMock()
    monkeypatch.setattr(
        "src.feishu.handlers.system.list_acp_tools",
        lambda: [SimpleNamespace(name="codex")],
    )
    invalidate = MagicMock()
    monkeypatch.setattr(
        "src.feishu.handlers.system.invalidate_acp_model_cache",
        invalidate,
        raising=False,
    )

    system.handle_select_acp_model(
        "card-message",
        "chat-first",
        first.project_id,
        {
            "tool_name": "codex",
            "model_group": "gpt-5.6-sol",
            "model_profile": None,
            "model_effort": "high",
            "model_name": "attacker-controlled-model",
        },
    )

    invalidate.assert_called_once_with("codex", first.root_path)
    system._activate_acp_selection.assert_called_once_with(
        "card-message",
        "chat-first",
        "codex",
        "gpt-5.6-sol/high",
        first,
        explicit_card=True,
        model_group="gpt-5.6-sol",
        model_profile=None,
        model_effort="high",
    )


def test_final_model_callback_rejects_selection_missing_from_live_capabilities(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    _storage, _projects, first, _second, _ctx, system = _configuration_lane(tmp_path)
    system._fetch_acp_models = MagicMock(
        return_value=[ACPModelOption(name="gpt-5.6-sol", is_default=True)]
    )
    system._activate_acp_selection = MagicMock()
    monkeypatch.setattr(
        "src.feishu.handlers.system.list_acp_tools",
        lambda: [SimpleNamespace(name="codex")],
    )
    monkeypatch.setattr(
        "src.feishu.handlers.system.invalidate_acp_model_cache",
        MagicMock(),
        raising=False,
    )

    system.handle_select_acp_model(
        "card-message",
        "chat-first",
        first.project_id,
        {
            "tool_name": "codex",
            "model_group": "removed-model",
            "model_effort": "ultra",
        },
    )

    system._activate_acp_selection.assert_not_called()
    system.reply_error.assert_called_once()


def test_refresh_models_invalidates_cache_before_redrawing(tmp_path, monkeypatch):
    _storage, _projects, first, _second, _ctx, system = _configuration_lane(tmp_path)
    invalidate = MagicMock()
    monkeypatch.setattr(
        "src.feishu.handlers.system.invalidate_acp_model_cache",
        invalidate,
        raising=False,
    )
    system.show_explicit_acp_model_selection = MagicMock()

    system.handle_refresh_acp_models(
        "card-message",
        "chat-first",
        first.project_id,
        {
            "tool_name": "codex",
            "model_group": "gpt-5.6-sol",
            "model_profile": None,
            "model_effort": "high",
        },
    )

    invalidate.assert_called_once_with("codex", first.root_path)
    system.show_explicit_acp_model_selection.assert_called_once_with(
        "card-message",
        "chat-first",
        "codex",
        first,
        origin_message_id="card-message",
        pending_group="gpt-5.6-sol",
        pending_profile=None,
        pending_effort="high",
    )


def _walk_card_nodes(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_card_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_card_nodes(child)


def _card_dict(card_content):
    if isinstance(card_content, str):
        return json.loads(card_content)
    return card_content


def _card_callback(card_content, action, *, name=None, use_default_model=None):
    for node in _walk_card_nodes(_card_dict(card_content)):
        value = node.get("value")
        if not isinstance(value, dict) or value.get("action") != action:
            continue
        if name is None or node.get("name") == name:
            if (
                use_default_model is not None
                and value.get("use_default_model") is not use_default_model
            ):
                continue
            return dict(value)
    raise AssertionError(f"callback not found: action={action!r}, name={name!r}")


def _card_action_event(*, event_id, message_id, value, option=None, tag="button"):
    action = {
        "tag": tag,
        "name": "model_group" if tag == "select_static" else "",
        "value": value,
    }
    if option is not None:
        action["option"] = option
    return P2CardActionTrigger(
        {
            "schema": "2.0",
            "header": {
                "event_id": event_id,
                "event_type": "card.action.trigger",
                "tenant_key": "tenant-direct-contract",
            },
            "event": {
                "operator": {"open_id": ""},
                "context": {
                    "open_message_id": message_id,
                    "open_chat_id": "chat-first",
                },
                "action": action,
            },
        }
    )


class _ActivationHoldingScheduler:
    def __init__(self, patches):
        self._patches = patches
        self.activation = None
        self.patches_at_submit = None
        self.update_project_id = MagicMock()

    def submit(self, spec, callback):
        assert spec.task_type == "acp_model_activation"
        self.activation = (spec, callback)
        self.patches_at_submit = list(self._patches)
        return SimpleNamespace(run_id="activation-run")


def _direct_model_action_client(*, ctx, projects, scheduler):
    client = SimpleNamespace(
        _handler_ctx=ctx,
        _project_manager=projects,
        _scheduler=scheduler,
        _thread_manager=MagicMock(),
        _chat_lock_gate=MagicMock(),
        settings=SimpleNamespace(thread_programming_enabled=True),
        _resolve_effective_trust=lambda **_kwargs: None,
        _managed_trust_access_decision=lambda _trust: None,
        _current_trust_can_dispatch=lambda _trust: True,
        _get_api_client=MagicMock(),
    )
    client._thread_manager.get.return_value = None
    client._chat_lock_gate.check_card_action.return_value = False
    client._action_handlers = init_action_registry(client)
    return client


def _dispatch_real_card_action(client, event):
    task_ctx = SimpleNamespace(
        run_id="card-action-run",
        spec=SimpleNamespace(
            sender_id="",
            sender_union_id="",
            tenant_key="tenant-direct-contract",
            is_p2p=True,
        ),
    )
    FeishuWSClient._process_card_action_async(
        client,
        event,
        task_ctx=task_ctx,
        effective_trust=None,
    )


@pytest.mark.parametrize("activation_ok", (True, False), ids=("success", "failure"))
def test_model_selector_card_action_flow_patches_one_card_and_finishes_atomically(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    activation_ok: bool,
):
    storage, projects, project, _second, ctx, system = _configuration_lane(tmp_path)
    assert projects.commit_acp_configuration(
        project,
        tool_name="codex",
        model_name="gpt-5.6-sol/ultra",
    )
    models = [
        ACPModelOption(
            name="gpt-5.6-sol",
            is_default=True,
            reasoning_efforts=("high", "ultra"),
            default_reasoning_effort="ultra",
            selection_variants=(
                ACPModelSelectionVariant(
                    name="gpt-5.6-sol/high",
                    model="gpt-5.6-sol",
                    profile=None,
                    effort="high",
                ),
                ACPModelSelectionVariant(
                    name="gpt-5.6-sol/ultra",
                    model="gpt-5.6-sol",
                    profile=None,
                    effort="ultra",
                    is_default=True,
                ),
            ),
        ),
        ACPModelOption(
            name="gpt-5.5",
            reasoning_efforts=("low", "high"),
            default_reasoning_effort="high",
            selection_variants=(
                ACPModelSelectionVariant(
                    name="gpt-5.5/low",
                    model="gpt-5.5",
                    profile=None,
                    effort="low",
                ),
                ACPModelSelectionVariant(
                    name="gpt-5.5/high",
                    model="gpt-5.5",
                    profile=None,
                    effort="high",
                    is_default=True,
                ),
            ),
        ),
    ]
    monkeypatch.setattr(
        "src.feishu.handlers.system.list_acp_tools",
        lambda: [SimpleNamespace(name="codex")],
    )
    monkeypatch.setattr(
        "src.feishu.handlers.system.invalidate_acp_model_cache",
        MagicMock(),
    )
    system._fetch_acp_models = MagicMock(return_value=models)

    handlers = {
        name: MagicMock() for name, _handler_type, _model in BACKENDS
    }
    codex = handlers["codex"]
    codex.current_model = "gpt-5.6-sol/ultra"
    session_manager = MagicMock()
    session_manager.get_session.return_value = SimpleNamespace(
        session_id="session-selected-model",
        message_count=0,
    )
    codex._get_session_manager.return_value = session_manager
    codex._enter_mode_on_manager.side_effect = (
        lambda chat_id, project_id=None: ctx.mode_manager.enter_programming_mode(
            chat_id,
            InteractionMode.CODEX,
            project_id=project_id,
        )
    )
    handlers.update(
        system=system,
        project=MagicMock(),
        deep=MagicMock(),
        spec=MagicMock(),
        workflow=MagicMock(),
    )
    ctx.handlers.clear()
    ctx.handlers.update(handlers)

    patches = []
    system.update_card = MagicMock(
        side_effect=lambda message_id, content: patches.append(
            (message_id, content)
        )
        or True
    )
    scheduler = _ActivationHoldingScheduler(patches)
    ctx.scheduler = scheduler
    system._enter_mode_with_acp_model = MagicMock(return_value=activation_ok)
    client = _direct_model_action_client(
        ctx=ctx,
        projects=projects,
        scheduler=scheduler,
    )

    system.show_explicit_acp_model_selection(
        "entry-message",
        "chat-first",
        "codex",
        project,
    )
    initial_card = system.reply_card.call_args.args[1]
    system.reply_card.reset_mock()
    group_callback = _card_callback(
        initial_card,
        action_ids.SELECT_ACP_MODEL_GROUP,
        name="model_group",
    )
    _dispatch_real_card_action(
        client,
        _card_action_event(
            event_id=f"evt-group-{activation_ok}",
            message_id="om-selector",
            value=group_callback,
            option="gpt-5.5",
            tag="select_static",
        ),
    )

    assert len(patches) == 1
    assert patches[0][0] == "om-selector"
    redrawn_card = patches[0][1]
    group_select = next(
        node
        for node in _walk_card_nodes(_card_dict(redrawn_card))
        if node.get("tag") == "select_static" and node.get("name") == "model_group"
    )
    effort_select = next(
        node
        for node in _walk_card_nodes(_card_dict(redrawn_card))
        if node.get("tag") == "select_static" and node.get("name") == "model_effort"
    )
    assert group_select["initial_option"] == "gpt-5.5"
    assert effort_select["initial_option"] == "high"
    system.reply_card.assert_not_called()

    confirm_callback = _card_callback(
        redrawn_card,
        action_ids.SELECT_ACP_MODEL,
        use_default_model=False,
    )
    assert confirm_callback["model_name"] == "gpt-5.5/high"
    assert "thread_root_id" not in confirm_callback
    set_current_thread_id("omt-stale-card-callback-context")
    try:
        _dispatch_real_card_action(
            client,
            _card_action_event(
                event_id=f"evt-confirm-{activation_ok}",
                message_id="om-selector",
                value=confirm_callback,
            ),
        )
    finally:
        set_current_thread_id(None)

    assert scheduler.activation is not None
    expected_initializing = SystemBuilder.build_acp_programming_initializing_card(
        "codex",
        "gpt-5.5/high",
        project.project_id,
        None,
    )[1]
    problems = []
    submitted_patches = scheduler.patches_at_submit or []
    if len(submitted_patches) != 2:
        problems.append(
            "confirm must PATCH initializing on the original card before scheduler.submit"
        )
    elif submitted_patches[-1][0] != "om-selector" or _card_dict(
        submitted_patches[-1][1]
    ) != _card_dict(expected_initializing):
        problems.append("the immediate confirm PATCH is not the initializing card")

    _activation_spec, activation = scheduler.activation
    activation(SimpleNamespace(run_id="activation-run"))

    expected_terminal = (
        SystemBuilder.build_acp_programming_ready_card(
            "codex",
            "gpt-5.5/high",
            project.project_id,
            None,
        )[1]
        if activation_ok
        else SystemBuilder.build_acp_programming_failed_card(
            "codex",
            "gpt-5.5/high",
            UI_TEXT["system_acp_activation_failed_safe"],
            project.project_id,
            None,
            model_group="gpt-5.5",
            model_profile=None,
            model_effort="high",
        )[1]
    )
    if len(patches) != 3:
        problems.append("activation must PATCH exactly one terminal card after initializing")
    elif patches[-1][0] != "om-selector" or _card_dict(
        patches[-1][1]
    ) != _card_dict(expected_terminal):
        problems.append(
            "successful activation must PATCH ready" if activation_ok else "failed activation must PATCH failed"
        )

    activation_call = system._enter_mode_with_acp_model.call_args
    if activation_call is None or activation_call.kwargs.get("thread_id") is not None:
        problems.append("explicit direct entry inherited a stale callback thread_id")

    reloaded = ProjectManager(str(storage)).get_project_for_chat(
        project.project_id,
        "chat-first",
    )
    assert reloaded is not None
    if activation_ok:
        if (
            project.acp_tool_name,
            project.acp_model_name,
            project.codex_mode,
        ) != ("codex", "gpt-5.5/high", True):
            problems.append("successful activation did not commit the selected persistent mode")
        if (
            reloaded.acp_tool_name,
            reloaded.acp_model_name,
            reloaded.codex_mode,
        ) != ("codex", "gpt-5.5/high", True):
            problems.append("successful activation was not durably persisted")
        if codex._enter_mode_on_manager.call_count != 1:
            problems.append("successful activation did not enter ordinary programming mode")
        session_call = session_manager.get_session.call_args
        if session_call is None or session_call.kwargs.get("thread_id") is not None:
            problems.append("successful activation looked up a topic-scoped session")
    else:
        if (
            project.acp_tool_name,
            project.acp_model_name,
            project.codex_mode,
        ) != ("codex", "gpt-5.6-sol/ultra", False):
            problems.append("failed activation changed the in-memory project selection")
        if (
            reloaded.acp_tool_name,
            reloaded.acp_model_name,
            reloaded.codex_mode,
        ) != ("codex", "gpt-5.6-sol/ultra", False):
            problems.append("failed activation changed the persisted project selection")
        if codex.current_model != "gpt-5.6-sol/ultra":
            problems.append("failed activation did not restore the handler model")
        if codex._enter_mode_on_manager.called:
            problems.append("failed activation entered programming mode")

    if system.reply_error.called:
        problems.append("an updated selector must render failure in-card, not as a second reply")
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    "failure_stage",
    ("session_missing", "commit_rejected", "activation_exception", "submit_exception"),
)
def test_explicit_activation_failure_paths_finish_on_the_same_failed_card(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
):
    _storage, projects, project, _second, ctx, system = _configuration_lane(tmp_path)
    assert projects.commit_acp_configuration(
        project,
        tool_name="codex",
        model_name="old-model",
    )
    codex = MagicMock()
    codex.current_model = "old-model"
    session_manager = MagicMock()
    session_manager.get_session.return_value = (
        None
        if failure_stage == "session_missing"
        else SimpleNamespace(session_id="session-new", message_count=0)
    )
    codex._get_session_manager.return_value = session_manager
    ctx.handlers["codex"] = codex
    system.update_card = MagicMock(return_value=True)
    system._enter_mode_with_acp_model = MagicMock(
        side_effect=(
            RuntimeError("activation exploded")
            if failure_stage == "activation_exception"
            else None
        ),
        return_value=True,
    )
    if failure_stage == "commit_rejected":
        monkeypatch.setattr(
            projects,
            "commit_acp_programming_activation",
            MagicMock(return_value=False),
        )
    if failure_stage == "submit_exception":
        ctx.scheduler.submit.side_effect = RuntimeError("queue unavailable")
    else:
        ctx.scheduler.submit.side_effect = lambda _spec, callback: callback(None)

    system._activate_acp_selection(
        "om-selector",
        "chat-first",
        "codex",
        "gpt-5.6-sol/high",
        project,
        explicit_card=True,
        model_group="gpt-5.6-sol",
        model_profile=None,
        model_effort="high",
    )

    patched_cards = [
        _card_dict(call.args[1]) for call in system.update_card.call_args_list
    ]
    assert len(patched_cards) == 2
    assert "正在初始化" in json.dumps(patched_cards[0], ensure_ascii=False)
    assert patched_cards[1]["header"]["template"] == "red"
    assert "初始化失败" in json.dumps(patched_cards[1], ensure_ascii=False)
    system.reply_error.assert_not_called()
    assert codex.current_model == "old-model"


def test_explicit_default_activation_enters_persistent_mode_and_renders_ready(
    tmp_path,
):
    _storage, projects, project, _second, ctx, system = _configuration_lane(tmp_path)
    codex = MagicMock()
    codex.current_model = "old-model"
    session_manager = MagicMock()
    session_manager.get_session.return_value = SimpleNamespace(
        session_id="session-default",
        message_count=0,
    )
    codex._get_session_manager.return_value = session_manager
    codex._enter_mode_on_manager.side_effect = (
        lambda chat_id, project_id=None: ctx.mode_manager.enter_programming_mode(
            chat_id,
            InteractionMode.CODEX,
            project_id=project_id,
        )
    )
    ctx.handlers["codex"] = codex
    ctx.scheduler.submit.side_effect = lambda _spec, callback: callback(None)
    system.update_card = MagicMock(return_value=True)
    system._enter_mode_with_acp_model = MagicMock(return_value=True)

    system._activate_acp_selection(
        "om-selector",
        "chat-first",
        "codex",
        None,
        project,
        explicit_card=True,
    )

    assert project.acp_tool_name == "codex"
    assert project.acp_model_name is None
    assert project.codex_mode is True
    system._enter_mode_with_acp_model.assert_called_once_with(
        "om-selector",
        "chat-first",
        "codex",
        None,
        project,
        thread_id=None,
    )
    assert "编程模式已就绪" in system.update_card.call_args_list[-1].args[1]


def test_activation_card_patch_failure_falls_back_and_finishes_on_replacement(
    tmp_path,
):
    _storage, projects, project, _second, ctx, system = _configuration_lane(tmp_path)
    codex = MagicMock()
    codex.current_model = None
    manager = MagicMock()
    manager.get_session.return_value = SimpleNamespace(
        session_id="session-fallback",
        message_count=0,
    )
    codex._get_session_manager.return_value = manager
    ctx.handlers["codex"] = codex
    ctx.scheduler.submit.side_effect = lambda _spec, callback: callback(None)
    system._enter_mode_with_acp_model = MagicMock(return_value=True)
    system.update_card = MagicMock(side_effect=(False, True))
    system.reply_card.return_value = "om-replacement"

    system._activate_acp_selection(
        "om-selector",
        "chat-first",
        "codex",
        None,
        project,
        explicit_card=True,
    )

    assert system.update_card.call_args_list[0].args[0] == "om-selector"
    system.reply_card.assert_called_once()
    assert system.update_card.call_args_list[1].args[0] == "om-replacement"


def test_initial_model_discovery_exception_finishes_on_the_loading_card(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    _storage, _projects, project, _second, _ctx, system = _configuration_lane(
        tmp_path
    )
    monkeypatch.setattr(
        "src.feishu.handlers.system.list_acp_tools",
        lambda: [SimpleNamespace(name="codex")],
    )
    system._fetch_acp_models = MagicMock(side_effect=RuntimeError("probe crashed"))
    system.reply_card.return_value = "om-loading"
    system.update_card = MagicMock(return_value=True)

    system.show_explicit_acp_model_selection(
        "entry",
        "chat-first",
        "codex",
        project,
    )

    system.reply_card.assert_called_once()
    assert "模型加载中" in system.reply_card.call_args.args[1]
    system.update_card.assert_called_once()
    assert system.update_card.call_args.args[0] == "om-loading"
    error_card = _card_dict(system.update_card.call_args.args[1])
    assert error_card["header"]["template"] == "red"
    assert _card_callback(error_card, action_ids.REFRESH_ACP_MODELS)


def test_default_failed_activation_retry_is_live_validated_and_stays_explicit(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    _storage, _projects, project, _second, _ctx, system = _configuration_lane(
        tmp_path
    )
    monkeypatch.setattr(
        "src.feishu.handlers.system.list_acp_tools",
        lambda: [SimpleNamespace(name="codex")],
    )
    monkeypatch.setattr(
        "src.feishu.handlers.system.invalidate_acp_model_cache",
        MagicMock(),
    )
    system._fetch_acp_models = MagicMock(
        return_value=[ACPModelOption(name="available-model", is_default=True)]
    )
    system._activate_acp_selection = MagicMock()

    system.handle_select_acp_model(
        "om-failed",
        "chat-first",
        project.project_id,
        {
            "action": action_ids.SELECT_ACP_MODEL,
            "tool_name": "codex",
            "use_default_model": True,
            "model_name": None,
        },
    )

    system._activate_acp_selection.assert_called_once_with(
        "om-failed",
        "chat-first",
        "codex",
        None,
        project,
        explicit_card=True,
        model_group=None,
        model_profile=None,
        model_effort=None,
    )
