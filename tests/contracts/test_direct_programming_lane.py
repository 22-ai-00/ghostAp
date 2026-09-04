"""Minimal contracts for explicit direct-programming configuration and execution."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, call

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
    DSHModeHandler,
    GeminiModeHandler,
    GrokModeHandler,
    TraexModeHandler,
)
from src.feishu.handlers.system import SystemHandler
from src.feishu.ws_client import FeishuWSClient
from src.mode import InteractionMode
from src.project.context import ProjectContext
from src.project.manager import ProjectManager
from src.tasking import TaskStatus
from src.thread import set_current_thread_id
from tests.helpers.session_call_recorder import SessionCallRecorder

BACKENDS = (
    ("coco", CocoModeHandler, "coco-model"),
    ("claude", ClaudeModeHandler, "claude-model"),
    ("aiden", AidenModeHandler, "aiden-model"),
    ("codex", CodexModeHandler, "codex-model"),
    ("gemini", GeminiModeHandler, "gemini-model"),
    ("traex", TraexModeHandler, "traex-model"),
    ("grok", GrokModeHandler, "grok-build"),
    ("dsh", DSHModeHandler, '["deepseek-official","deepseek-v4-flash","high"]'),
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
        grok_manager=managers["grok"],
        dsh_manager=managers["dsh"],
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


def test_backend_start_failure_can_retry_on_the_next_feishu_activation():
    attempts: list[str] = []
    recorder = SessionCallRecorder()

    def flaky_factory(*, agent_type: str, **kwargs):
        attempts.append(agent_type)
        if len(attempts) == 1:
            raise RuntimeError("temporary codex network failure")
        return recorder.session_factory(agent_type=agent_type, **kwargs)

    manager = ACPSessionManager("codex", session_starter=flaky_factory)
    ctx = _context("codex", manager)
    handler = CodexModeHandler(ctx)
    handler.reply_text = MagicMock()
    handler.reply_card = MagicMock()
    handler.add_reaction = MagicMock()
    handler.record_mode_transition = MagicMock()
    project = _project("codex", "codex-model")

    assert handler.enter_mode("first", "chat-direct", project=project, silent=True) is False
    assert manager.get_session("chat-direct", project_id=project.project_id) is None

    assert handler.enter_mode("second", "chat-direct", project=project, silent=True) is True
    assert manager.get_session("chat-direct", project_id=project.project_id) is not None
    assert attempts == ["codex", "codex"]


@pytest.mark.parametrize("model_name", ["gpt-5.6-sol/high", None])
def test_active_mode_without_session_recovers_instead_of_hot_switching(
    model_name: str | None,
):
    """Persistent mode may outlive its transport; model selection must recover it."""
    manager = MagicMock()
    recovered_session = SimpleNamespace(session_id="recovered-session")
    manager.get_session.side_effect = [None, recovered_session]
    ctx = _context("codex", manager)
    ctx.mode_manager.is_codex_mode.return_value = True
    system = SystemHandler(ctx)
    handler = MagicMock()
    handler.current_model = "old-model"
    handler._get_session_manager.return_value = manager
    handler._get_agent_type_override.return_value = None
    handler.enter_mode.return_value = True
    ctx.handlers["codex"] = handler
    project = _project("codex", "old-model")

    assert system._enter_mode_with_acp_model(
        "selector",
        "chat-direct",
        "codex",
        model_name,
        project,
        thread_id=None,
    )

    handler.switch_model.assert_not_called()
    handler.enter_mode.assert_called_once_with(
        "selector",
        "chat-direct",
        project=project,
        silent=True,
        model_override=model_name,
        commit_project_state=False,
        activate_mode=False,
        exit_opposite_mode=False,
        inherit_thread_context=False,
    )


def test_live_direct_session_still_uses_effect_only_hot_switch() -> None:
    manager = MagicMock()
    live_session = SimpleNamespace(
        session_id="live-session",
        set_model=MagicMock(return_value=True),
    )
    manager.get_session.return_value = live_session
    ctx = _context("codex", manager)
    ctx.mode_manager.is_codex_mode.return_value = True
    system = SystemHandler(ctx)
    handler = MagicMock()
    handler.current_model = "old-model"
    handler._get_session_manager.return_value = manager
    handler.switch_model.return_value = True
    ctx.handlers["codex"] = handler
    project = _project("codex", "old-model")

    assert system._enter_mode_with_acp_model(
        "selector",
        "chat-direct",
        "codex",
        "gpt-5.6-sol/high",
        project,
        thread_id=None,
    )

    manager.get_session.assert_called_once_with(
        "chat-direct",
        project_id=project.project_id,
        thread_id=None,
    )
    handler.switch_model.assert_called_once_with(
        "selector",
        "chat-direct",
        "gpt-5.6-sol/high",
        project=project,
        expected_session=live_session,
    )
    handler.enter_mode.assert_not_called()


def test_live_custom_session_resets_to_backend_default_with_cas_replacement(
    tmp_path,
) -> None:
    manager = MagicMock()
    live_session = SimpleNamespace(
        session_id="custom-session",
        _model_name="gpt-5.6-sol/high",
    )
    default_session = SimpleNamespace(
        session_id="default-session",
        _model_name=None,
    )
    manager.get_session.return_value = live_session
    manager.replace_session.return_value = SimpleNamespace(
        session=default_session,
        created=True,
    )
    ctx = _context("codex", manager)
    ctx.settings.acp_startup_timeout = 37
    ctx.mode_manager.is_codex_mode.return_value = True
    system = SystemHandler(ctx)
    handler = MagicMock()
    handler.current_model = "gpt-5.6-sol/high"
    handler._get_session_manager.return_value = manager
    handler._get_agent_type_override.return_value = None
    ctx.handlers["codex"] = handler
    project = _project("codex", "gpt-5.6-sol/high")
    project.root_path = str(tmp_path)

    assert system._enter_mode_with_acp_model(
        "selector",
        "chat-direct",
        "codex",
        None,
        project,
        thread_id=None,
    )

    manager.replace_session.assert_called_once_with(
        "chat-direct",
        cwd=str(tmp_path),
        expected_session=live_session,
        startup_timeout=37,
        project_id=project.project_id,
        agent_type_override=None,
        model_name=None,
        thread_id=None,
    )
    handler.switch_model.assert_not_called()
    handler.enter_mode.assert_not_called()


def test_live_session_without_set_model_uses_cas_replacement(tmp_path) -> None:
    manager = MagicMock()
    live_session = SimpleNamespace(
        session_id="claude-cli-session",
        _model_name="old-model",
    )
    replacement_session = SimpleNamespace(
        session_id="replacement-session",
        _model_name="claude-sonnet-4-5",
    )
    manager.get_session.return_value = live_session
    manager.replace_session.return_value = SimpleNamespace(
        session=replacement_session,
        created=True,
    )
    ctx = _context("claude", manager)
    system = SystemHandler(ctx)
    handler = MagicMock()
    handler.current_model = "old-model"
    handler._get_session_manager.return_value = manager
    handler._get_agent_type_override.return_value = None
    ctx.handlers["claude"] = handler
    project = _project("claude", "old-model")
    project.root_path = str(tmp_path)

    result = system._enter_mode_with_acp_model(
        "selector",
        "chat-direct",
        "claude",
        "claude-sonnet-4-5",
        project,
        thread_id=None,
    )

    assert result.session is replacement_session
    assert result.changed is True
    handler.switch_model.assert_not_called()
    manager.replace_session.assert_called_once_with(
        "chat-direct",
        cwd=str(tmp_path),
        expected_session=live_session,
        startup_timeout=20,
        project_id=project.project_id,
        agent_type_override=None,
        model_name="claude-sonnet-4-5",
        thread_id=None,
    )


def test_topic_activation_replaces_only_the_exact_topic_session() -> None:
    manager = MagicMock()
    topic_session = SimpleNamespace(
        session_id="topic-session",
        _model_name="old-model",
    )
    replacement_session = SimpleNamespace(
        session_id="replacement-topic-session",
        _model_name="gpt-5.6-sol/high",
    )
    manager.get_session.return_value = topic_session
    manager.replace_session.return_value = SimpleNamespace(
        session=replacement_session,
        created=True,
    )
    ctx = _context("codex", manager)
    ctx.mode_manager.is_codex_mode.return_value = True
    system = SystemHandler(ctx)
    handler = MagicMock()
    handler.current_model = "old-model"
    handler._get_session_manager.return_value = manager
    handler._get_agent_type_override.return_value = None
    handler.enter_mode.return_value = True
    ctx.handlers["codex"] = handler
    project = _project("codex", "old-model")

    assert system._enter_mode_with_acp_model(
        "selector",
        "chat-direct",
        "codex",
        "gpt-5.6-sol/high",
        project,
        thread_id="topic-root",
    )

    manager.get_session.assert_called_once_with(
        "chat-direct",
        project_id=project.project_id,
        thread_id="topic-root",
    )
    handler.switch_model.assert_not_called()
    handler.enter_mode.assert_not_called()
    manager.replace_session.assert_called_once_with(
        "chat-direct",
        cwd=project.root_path,
        expected_session=topic_session,
        startup_timeout=20,
        project_id=project.project_id,
        agent_type_override=None,
        model_name="gpt-5.6-sol/high",
        thread_id="topic-root",
    )


def test_topic_custom_session_resets_to_default_on_the_exact_topic_key() -> None:
    manager = MagicMock()
    topic_session = SimpleNamespace(
        session_id="topic-custom",
        _model_name="custom-model",
    )
    default_session = SimpleNamespace(
        session_id="topic-default",
        _model_name=None,
    )
    manager.get_session.return_value = topic_session
    manager.replace_session.return_value = SimpleNamespace(
        session=default_session,
        created=True,
    )
    ctx = _context("codex", manager)
    system = SystemHandler(ctx)
    handler = MagicMock()
    handler.current_model = "custom-model"
    handler._get_session_manager.return_value = manager
    handler._get_agent_type_override.return_value = None
    ctx.handlers["codex"] = handler
    project = _project("codex", "custom-model")

    result = system._enter_mode_with_acp_model(
        "selector",
        "chat-direct",
        "codex",
        None,
        project,
        thread_id="topic-root",
    )

    assert result.session is default_session
    assert result.changed is True
    manager.replace_session.assert_called_once_with(
        "chat-direct",
        cwd=project.root_path,
        expected_session=topic_session,
        startup_timeout=20,
        project_id=project.project_id,
        agent_type_override=None,
        model_name=None,
        thread_id="topic-root",
    )


def test_unknown_live_model_state_is_replaced_when_selecting_default() -> None:
    manager = MagicMock()
    unknown_session = SimpleNamespace(session_id="unknown-model-session")
    default_session = SimpleNamespace(
        session_id="known-default-session",
        _model_name=None,
    )
    manager.get_session.return_value = unknown_session
    manager.replace_session.return_value = SimpleNamespace(
        session=default_session,
        created=True,
    )
    ctx = _context("codex", manager)
    system = SystemHandler(ctx)
    handler = MagicMock()
    handler._get_session_manager.return_value = manager
    handler._get_agent_type_override.return_value = None
    ctx.handlers["codex"] = handler
    project = _project("codex", "old-model")

    result = system._enter_mode_with_acp_model(
        "selector",
        "chat-direct",
        "codex",
        None,
        project,
        thread_id=None,
    )

    assert result.session is default_session
    assert result.changed is True
    manager.replace_session.assert_called_once()


def test_missing_direct_session_recovery_ignores_stale_thread_context(
    tmp_path,
) -> None:
    recorder = SessionCallRecorder()
    manager = ACPSessionManager(
        "codex",
        session_starter=recorder.session_factory,
    )
    _storage, _projects, project, _second, ctx, system = _configuration_lane(
        tmp_path
    )
    ctx.codex_manager = manager
    ctx.managers["codex"] = manager
    ctx.mode_manager.is_codex_mode.return_value = True
    handler = CodexModeHandler(ctx)
    handler.add_reaction = MagicMock()
    handler.reply_card = MagicMock()
    handler.reply_text = MagicMock()
    ctx.handlers["codex"] = handler

    set_current_thread_id("stale-topic-context")
    try:
        assert system._enter_mode_with_acp_model(
            "selector",
            "chat-first",
            "codex",
            "gpt-5.6-sol/high",
            project,
            thread_id=None,
        )
    finally:
        set_current_thread_id(None)

    assert manager.get_session(
        "chat-first",
        project_id=project.project_id,
        thread_id=None,
    ) is not None
    assert manager.get_session(
        "chat-first",
        project_id=project.project_id,
        thread_id="stale-topic-context",
    ) is None
    manager.cleanup_all()


def test_live_switch_failure_is_rendered_only_on_the_selector_card(
    tmp_path,
) -> None:
    """The inner protocol effect must not emit a second generic fatal card."""
    _storage, projects, project, _second, ctx, system = _configuration_lane(
        tmp_path
    )
    assert projects.commit_acp_configuration(
        project,
        tool_name="codex",
        model_name="old-model",
    )
    live_session = SimpleNamespace(
        session_id="live-session",
        message_count=3,
        set_model=MagicMock(return_value=False),
    )
    manager = ctx.managers["codex"]
    manager.get_session.return_value = live_session
    codex = CodexModeHandler(ctx)
    codex.reply_card = MagicMock()
    codex.reply_error = MagicMock()
    ctx.handlers["codex"] = codex
    ctx.mode_manager.is_codex_mode.return_value = True
    ctx.scheduler.submit.side_effect = lambda _spec, callback: callback(None)
    system.update_card = MagicMock(return_value=True)

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

    assert len(system.update_card.call_args_list) == 2
    cards = [
        _card_dict(call.args[1])
        for call in system.update_card.call_args_list
    ]
    assert "正在初始化" in json.dumps(cards[0], ensure_ascii=False)
    assert "初始化失败" in json.dumps(cards[1], ensure_ascii=False)
    assert cards[1]["header"]["template"] == "orange"
    codex.reply_card.assert_not_called()
    codex.reply_error.assert_not_called()
    system.reply_error.assert_not_called()


def test_hot_switch_rejects_a_replaced_session_owner() -> None:
    original = SimpleNamespace(
        session_id="session-a",
        set_model=MagicMock(return_value=True),
    )
    replacement = SimpleNamespace(
        session_id="session-b",
        set_model=MagicMock(return_value=True),
    )
    manager = MagicMock()
    manager.get_session.side_effect = [original, replacement]
    ctx = _context("codex", manager)
    ctx.mode_manager.is_codex_mode.return_value = True
    system = SystemHandler(ctx)
    handler = CodexModeHandler(ctx)
    ctx.handlers["codex"] = handler
    project = _project("codex", "old-model")

    assert not system._enter_mode_with_acp_model(
        "selector",
        "chat-direct",
        "codex",
        "gpt-5.6-sol/high",
        project,
        thread_id=None,
    )

    original.set_model.assert_not_called()
    replacement.set_model.assert_not_called()


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


def test_latest_model_selection_generation_owns_commit_and_terminal_card(
    tmp_path,
) -> None:
    _storage, projects, project, _second, ctx, system = _configuration_lane(
        tmp_path
    )
    assert projects.commit_acp_configuration(
        project,
        tool_name="codex",
        model_name="old-model",
    )
    session = SimpleNamespace(session_id="session", message_count=0)
    manager = MagicMock()
    manager.get_session.return_value = session
    handler = MagicMock()
    handler.current_model = "old-model"
    handler._get_session_manager.return_value = manager
    handler._enter_mode_on_manager.side_effect = (
        lambda chat_id, project_id=None: ctx.mode_manager.enter_programming_mode(
            chat_id,
            InteractionMode.CODEX,
            project_id=project_id,
        )
    )
    ctx.handlers["codex"] = handler
    callbacks = []
    ctx.scheduler.submit.side_effect = (
        lambda _spec, callback: callbacks.append(callback)
        or SimpleNamespace(run_id=f"activation-{len(callbacks)}")
    )
    system.update_card = MagicMock(return_value=True)
    system._enter_mode_with_acp_model = MagicMock(return_value=True)

    system._activate_acp_selection(
        "om-selector",
        "chat-first",
        "codex",
        "model-a",
        project,
        explicit_card=True,
    )
    system._activate_acp_selection(
        "om-selector",
        "chat-first",
        "codex",
        "model-b",
        project,
        explicit_card=True,
    )

    assert len(callbacks) == 2
    assert callbacks[0](None) is False
    assert callbacks[1](None) is True
    assert project.acp_model_name == "model-b"
    rendered_updates = [
        call.args[1] for call in system.update_card.call_args_list
    ]
    assert sum("编程模式已就绪" in card for card in rendered_updates) == 1
    assert "model-b" in rendered_updates[-1]
    assert "model-a" not in rendered_updates[-1]


def test_terminal_activation_releases_generation_state(tmp_path) -> None:
    _storage, projects, project, _second, ctx, system = _configuration_lane(
        tmp_path
    )
    assert projects.commit_acp_configuration(
        project,
        tool_name="codex",
        model_name="old-model",
    )
    session = SimpleNamespace(session_id="session", message_count=0)
    manager = MagicMock()
    manager.get_session.return_value = session
    handler = MagicMock()
    handler.current_model = "old-model"
    handler._get_session_manager.return_value = manager
    handler._enter_mode_on_manager.side_effect = (
        lambda chat_id, project_id=None: ctx.mode_manager.enter_programming_mode(
            chat_id,
            InteractionMode.CODEX,
            project_id=project_id,
        )
    )
    ctx.handlers["codex"] = handler
    ctx.scheduler.submit.side_effect = lambda _spec, callback: callback(None)
    system.update_card = MagicMock(return_value=True)
    system._enter_mode_with_acp_model = MagicMock(return_value=True)

    system._activate_acp_selection(
        "selector",
        "chat-first",
        "codex",
        "new-model",
        project,
        explicit_card=True,
    )

    assert system._acp_activation_generations == {}
    assert system._acp_activation_fence_epochs == {}
    assert system._acp_activation_owners == {}
    assert system._acp_activation_task_owners == {}


def test_topic_activations_use_independent_generations_and_forward_both_tasks(
    tmp_path,
) -> None:
    _storage, projects, project, _second, ctx, system = _configuration_lane(
        tmp_path
    )
    assert projects.commit_acp_configuration(
        project,
        tool_name="codex",
        model_name="saved-model",
    )
    sessions = {
        "topic-a": SimpleNamespace(session_id="session-a", message_count=0),
        "topic-b": SimpleNamespace(session_id="session-b", message_count=0),
    }
    manager = MagicMock()
    manager.get_session.side_effect = (
        lambda _chat_id, *, project_id=None, thread_id=None: sessions[thread_id]
    )
    handler = MagicMock()
    handler.current_model = "saved-model"
    handler._get_session_manager.return_value = manager
    ctx.handlers["codex"] = handler
    callbacks = []
    ctx.scheduler.submit.side_effect = (
        lambda _spec, callback: callbacks.append(callback)
        or SimpleNamespace(run_id=f"topic-activation-{len(callbacks)}")
    )
    system._enter_mode_with_acp_model = MagicMock(return_value=True)

    try:
        set_current_thread_id("topic-a")
        system.handle_enter_acp_saved_selection(
            "message-a",
            "chat-first",
            "codex",
            project,
            pending_prompt="task a",
        )
        set_current_thread_id("topic-b")
        system.handle_enter_acp_saved_selection(
            "message-b",
            "chat-first",
            "codex",
            project,
            pending_prompt="task b",
        )
    finally:
        set_current_thread_id(None)

    assert len(callbacks) == 2
    assert callbacks[0](None) is True
    assert callbacks[1](None) is True
    assert handler.handle_message.call_args_list == [
        call(
            "message-a",
            "chat-first",
            "task a",
            project,
            expected_session=sessions["topic-a"],
        ),
        call(
            "message-b",
            "chat-first",
            "task b",
            project,
            expected_session=sessions["topic-b"],
        ),
    ]


def test_late_topic_activation_cannot_overwrite_new_direct_configuration(
    tmp_path,
) -> None:
    _storage, projects, project, _second, ctx, system = _configuration_lane(
        tmp_path
    )
    assert projects.commit_acp_configuration(
        project,
        tool_name="codex",
        model_name="old-model",
    )
    direct_session = SimpleNamespace(session_id="direct", message_count=0)
    topic_session = SimpleNamespace(session_id="topic", message_count=0)
    manager = MagicMock()
    manager.get_session.side_effect = (
        lambda _chat_id, *, project_id=None, thread_id=None: (
            topic_session if thread_id == "topic-root" else direct_session
        )
    )
    handler = MagicMock()
    handler.current_model = "old-model"
    handler._get_session_manager.return_value = manager
    ctx.handlers["codex"] = handler
    callbacks = []
    ctx.scheduler.submit.side_effect = (
        lambda _spec, callback: callbacks.append(callback)
        or SimpleNamespace(run_id=f"activation-{len(callbacks)}")
    )
    system._enter_mode_with_acp_model = MagicMock(return_value=True)
    system.update_card = MagicMock(return_value=True)

    try:
        set_current_thread_id("topic-root")
        system.handle_enter_acp_saved_selection(
            "topic-message",
            "chat-first",
            "codex",
            project,
            pending_prompt="topic task",
        )
    finally:
        set_current_thread_id(None)
    system._activate_acp_selection(
        "direct-selector",
        "chat-first",
        "codex",
        "new-model",
        project,
        explicit_card=True,
    )

    assert callbacks[1](None) is True
    assert project.acp_model_name == "new-model"
    assert callbacks[0](None) is True
    assert project.acp_model_name == "new-model"
    assert handler.current_model == "new-model"


def test_a_new_selector_card_finishes_the_superseded_card(tmp_path) -> None:
    _storage, projects, project, _second, ctx, system = _configuration_lane(
        tmp_path
    )
    assert projects.commit_acp_configuration(
        project,
        tool_name="codex",
        model_name="old-model",
    )
    session = SimpleNamespace(session_id="session", message_count=0)
    manager = MagicMock()
    manager.get_session.return_value = session
    handler = MagicMock()
    handler.current_model = "old-model"
    handler._get_session_manager.return_value = manager
    ctx.handlers["codex"] = handler
    callbacks = []
    ctx.scheduler.submit.side_effect = (
        lambda _spec, callback: callbacks.append(callback)
        or SimpleNamespace(run_id=f"activation-{len(callbacks)}")
    )
    system.update_card = MagicMock(return_value=True)
    system._enter_mode_with_acp_model = MagicMock(return_value=True)

    system._activate_acp_selection(
        "selector-a",
        "chat-first",
        "codex",
        "model-a",
        project,
        explicit_card=True,
    )
    system._activate_acp_selection(
        "selector-b",
        "chat-first",
        "codex",
        "model-b",
        project,
        explicit_card=True,
    )

    assert callbacks[0](None) is False
    assert callbacks[1](None) is True
    selector_a_updates = [
        call.args[1]
        for call in system.update_card.call_args_list
        if call.args[0] == "selector-a"
    ]
    assert len(selector_a_updates) == 2
    assert "正在初始化" in selector_a_updates[0]
    assert "选择已更新" in selector_a_updates[1]
    assert "已被更新的模型选择取代" in selector_a_updates[1]


def test_scheduler_cancellation_finishes_the_initializing_selector_card(
    tmp_path,
) -> None:
    _storage, _projects, project, _second, ctx, system = _configuration_lane(
        tmp_path
    )

    class _CancelledHandle:
        def add_done_callback(self, callback) -> None:
            callback(SimpleNamespace(status=TaskStatus.CANCELED))

    ctx.scheduler.submit.return_value = _CancelledHandle()
    system.update_card = MagicMock(return_value=True)

    system._activate_acp_selection(
        "selector",
        "chat-first",
        "codex",
        "model-a",
        project,
        explicit_card=True,
    )

    assert len(system.update_card.call_args_list) == 2
    assert "正在初始化" in system.update_card.call_args_list[0].args[1]
    assert "初始化任务已取消" in system.update_card.call_args_list[1].args[1]


def test_exit_during_startup_invalidates_the_activation_before_commit(
    tmp_path,
) -> None:
    _storage, projects, project, _second, ctx, system = _configuration_lane(
        tmp_path
    )
    assert projects.commit_acp_configuration(
        project,
        tool_name="codex",
        model_name="old-model",
    )
    session = SimpleNamespace(session_id="new-session", message_count=0)
    manager = MagicMock()
    manager.get_session.return_value = session
    handler = MagicMock()
    handler.current_model = "old-model"
    handler._get_session_manager.return_value = manager
    ctx.handlers["codex"] = handler
    callback_holder = []
    ctx.scheduler.submit.side_effect = (
        lambda _spec, callback: callback_holder.append(callback)
        or SimpleNamespace(run_id="activation")
    )
    startup_entered = threading.Event()
    allow_startup = threading.Event()

    def _startup(*_args, **_kwargs):
        startup_entered.set()
        assert allow_startup.wait(timeout=2)
        return session

    system._enter_mode_with_acp_model = _startup
    system.update_card = MagicMock(return_value=True)
    system._activate_acp_selection(
        "selector",
        "chat-first",
        "codex",
        "new-model",
        project,
        explicit_card=True,
    )
    activation_result = []
    worker = threading.Thread(
        target=lambda: activation_result.append(callback_holder[0](None))
    )
    worker.start()
    assert startup_entered.wait(timeout=2)

    system.exit_current_mode("exit", "chat-first", project)
    allow_startup.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert activation_result == [False]
    assert project.acp_model_name == "old-model"
    handler._enter_mode_on_manager.assert_not_called()
    manager.retire_session.assert_called_once_with(
        "chat-first",
        project_id=project.project_id,
        thread_id=None,
        expected_session=session,
        timeout=20,
    )
    assert "初始化任务已取消" in system.update_card.call_args_list[-1].args[1]


def test_exit_waits_for_atomic_activation_commit_then_wins_final_mode(
    tmp_path,
) -> None:
    _storage, projects, project, _second, ctx, system = _configuration_lane(
        tmp_path
    )
    assert projects.commit_acp_configuration(
        project,
        tool_name="codex",
        model_name="old-model",
    )
    session = SimpleNamespace(session_id="new-session", message_count=0)
    manager = MagicMock()
    manager.get_session.return_value = session
    handler = MagicMock()
    handler.current_model = "old-model"
    handler._get_session_manager.return_value = manager
    ctx.handlers["codex"] = handler
    callbacks = []
    ctx.scheduler.submit.side_effect = (
        lambda _spec, callback: callbacks.append(callback)
        or SimpleNamespace(run_id="activation")
    )
    system._enter_mode_with_acp_model = MagicMock(return_value=session)
    system.update_card = MagicMock(return_value=True)

    mode = {"value": InteractionMode.SMART}
    events = []
    ctx.mode_manager.get_mode.side_effect = (
        lambda _chat_id, project_id=None: mode["value"]
    )
    handler._enter_mode_on_manager.side_effect = lambda *_args, **_kwargs: (
        events.append("enter"),
        mode.__setitem__("value", InteractionMode.CODEX),
    )
    handler.exit_mode.side_effect = lambda *_args, **_kwargs: (
        events.append("exit"),
        mode.__setitem__("value", InteractionMode.SMART),
    )

    commit_entered = threading.Event()
    allow_commit = threading.Event()
    original_commit = projects.commit_acp_programming_activation

    def _blocking_commit(*args, **kwargs):
        commit_entered.set()
        assert allow_commit.wait(timeout=2)
        committed = original_commit(*args, **kwargs)
        events.append("commit")
        return committed

    projects.commit_acp_programming_activation = _blocking_commit
    system._activate_acp_selection(
        "selector",
        "chat-first",
        "codex",
        "new-model",
        project,
        explicit_card=True,
    )
    activation_worker = threading.Thread(target=lambda: callbacks[0](None))
    activation_worker.start()
    assert commit_entered.wait(timeout=2)

    exit_finished = threading.Event()

    def _exit() -> None:
        system.exit_current_mode("exit", "chat-first", project)
        exit_finished.set()

    exit_worker = threading.Thread(target=_exit)
    exit_worker.start()
    assert not exit_finished.wait(timeout=0.05)
    allow_commit.set()
    activation_worker.join(timeout=2)
    exit_worker.join(timeout=2)

    assert not activation_worker.is_alive()
    assert not exit_worker.is_alive()
    assert events == ["commit", "enter", "exit"]
    assert mode["value"] is InteractionMode.SMART


def test_programming_exit_persists_project_mode_off(tmp_path) -> None:
    storage, projects, project, _second, ctx, _system = _configuration_lane(
        tmp_path
    )
    assert projects.commit_acp_programming_activation(
        project,
        tool_name="codex",
        model_name="model-a",
        session_id="session-a",
    )
    manager = MagicMock()
    manager.get_session.return_value = None
    manager.end_session.return_value = None
    ctx.codex_manager = manager
    ctx.managers["codex"] = manager
    ctx.mode_manager.get_mode.return_value = InteractionMode.CODEX
    handler = CodexModeHandler(ctx)
    handler.add_reaction = MagicMock()
    handler.reply_card = MagicMock()
    handler.reply_text = MagicMock()

    handler.exit_mode("exit", "chat-first", project=project)

    reloaded = ProjectManager(str(storage)).get_project_for_chat(
        project.project_id,
        "chat-first",
    )
    assert reloaded is not None
    assert reloaded.codex_mode is False


def test_programming_exit_reports_when_project_mode_cannot_be_persisted(
    tmp_path,
) -> None:
    _storage, projects, project, _second, ctx, _system = _configuration_lane(
        tmp_path
    )
    assert projects.commit_acp_programming_activation(
        project,
        tool_name="codex",
        model_name="model-a",
        session_id="session-a",
    )
    projects._save_projects = MagicMock(return_value=False)
    manager = MagicMock()
    manager.get_session.return_value = None
    manager.end_session.return_value = None
    ctx.codex_manager = manager
    ctx.managers["codex"] = manager
    ctx.mode_manager.get_mode.return_value = InteractionMode.CODEX
    handler = CodexModeHandler(ctx)
    handler.add_reaction = MagicMock()
    handler.reply_card = MagicMock()
    handler.reply_text = MagicMock()

    handler.exit_mode("exit", "chat-first", project=project)

    assert ctx.mode_manager.exit_to_smart.called
    card = handler.reply_card.call_args.args[1]
    assert "模式退出未完全保存" in card
    assert "服务重启后可能恢复旧模式" in card
    assert "会话已保存" not in card


def test_live_model_switch_retires_session_when_project_commit_fails(
    tmp_path,
) -> None:
    _storage, projects, project, _second, ctx, system = _configuration_lane(
        tmp_path
    )
    assert projects.commit_acp_configuration(
        project,
        tool_name="codex",
        model_name="old-model",
    )
    session = SimpleNamespace(
        session_id="live-session",
        message_count=3,
        _model_name="old-model",
        set_model=MagicMock(return_value=True),
    )
    manager = MagicMock()
    manager.get_session.return_value = session
    handler = CodexModeHandler(ctx)
    handler.current_model = "old-model"
    handler._get_session_manager = MagicMock(return_value=manager)
    ctx.handlers["codex"] = handler
    ctx.scheduler.submit.side_effect = lambda _spec, callback: callback(None)
    system.update_card = MagicMock(return_value=True)
    projects._save_projects = MagicMock(return_value=False)

    system._activate_acp_selection(
        "selector",
        "chat-first",
        "codex",
        "new-model",
        project,
        explicit_card=True,
    )

    session.set_model.assert_called_once()
    manager.retire_session.assert_called_once_with(
        "chat-first",
        project_id=project.project_id,
        thread_id=None,
        expected_session=session,
        timeout=20,
    )
    assert (project.acp_tool_name, project.acp_model_name) == (
        "codex",
        "old-model",
    )
    assert handler.current_model == "old-model"
    assert "初始化失败" in system.update_card.call_args_list[-1].args[1]


def test_typed_programming_exit_routes_through_the_system_fence() -> None:
    system = MagicMock()
    codex = MagicMock()
    handlers = {
        "system": system,
        "project": MagicMock(),
        "coco": MagicMock(),
        "codex": codex,
    }
    client = SimpleNamespace(
        _handler_ctx=SimpleNamespace(handlers=handlers),
        _mode_manager=MagicMock(),
    )
    dispatcher = MessageDispatcher(client)
    project = _project("codex", "model-a")
    task = SimpleNamespace(
        intent=SimpleNamespace(name="EXIT_CODEX"),
        data={},
    )

    dispatcher.execute_single_task(
        "exit-message",
        "chat-direct",
        task,
        "/exit_codex",
        project,
    )

    system.exit_current_mode.assert_called_once_with(
        "exit-message",
        "chat-direct",
        project=project,
    )
    codex.exit_mode.assert_not_called()


def test_programming_card_exit_routes_through_the_system_fence() -> None:
    manager = MagicMock()
    ctx = _context("codex", manager)
    project = _project("codex", "model-a")
    ctx.project_manager.get_project_for_chat.return_value = project
    system = MagicMock()
    ctx.handlers["system"] = system
    handler = CodexModeHandler(ctx)

    handler.handle_card_exit(
        "exit-card",
        "chat-direct",
        project.project_id,
    )

    system.exit_current_mode.assert_called_once_with(
        "exit-card",
        "chat-direct",
        project=project,
    )
    manager.end_session.assert_not_called()


def test_pending_prompt_never_recovers_a_replaced_committed_session() -> None:
    expected = SimpleNamespace(session_id="expected")
    replacement = SimpleNamespace(session_id="replacement")
    manager = MagicMock()
    manager.get_session.return_value = replacement
    ctx = _context("codex", manager)
    handler = CodexModeHandler(ctx)
    handler.enter_mode = MagicMock()
    handler.handle_response = MagicMock()
    project = _project("codex", "model-a")

    handler.handle_message(
        "task-message",
        "chat-direct",
        "implement it",
        project,
        expected_session=expected,
    )

    handler.enter_mode.assert_not_called()
    handler.handle_response.assert_not_called()


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
        expected_session=session,
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


@pytest.mark.parametrize(
    ("tool_name", "mode"),
    [(name, InteractionMode(name)) for name, _handler, _model in BACKENDS],
)
def test_model_command_adapts_to_every_active_programming_backend(
    tmp_path,
    tool_name: str,
    mode: InteractionMode,
):
    _storage, _projects, first, _second, ctx, system = _configuration_lane(tmp_path)
    first.acp_tool_name = None
    first.acp_model_name = None
    ctx.mode_manager.get_mode.return_value = mode
    system.show_explicit_acp_model_selection = MagicMock()

    system.handle_model_command("model", "chat-first", "/model", first)

    system.show_explicit_acp_model_selection.assert_called_once_with(
        "model",
        "chat-first",
        tool_name,
        first,
    )


def _install_single_tool_registration(
    monkeypatch: pytest.MonkeyPatch,
    *,
    registered_tool: str = "codex",
) -> None:
    def _global_scan_is_forbidden():
        raise AssertionError("single-tool model selection must not enumerate ACP tools")

    monkeypatch.setattr(
        "src.feishu.handlers.system.list_acp_tools",
        _global_scan_is_forbidden,
    )
    monkeypatch.setattr(
        "src.feishu.handlers.system.get_providers",
        lambda: {registered_tool: SimpleNamespace(name=registered_tool)},
    )


def test_codex_initial_selector_does_not_enumerate_other_providers(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    _storage, _projects, project, _second, _ctx, system = _configuration_lane(
        tmp_path
    )
    _install_single_tool_registration(monkeypatch)
    models = [ACPModelOption(name="gpt-5.6-sol", is_default=True)]
    system._fetch_acp_models = MagicMock(return_value=models)
    system.reply_card.return_value = "om-loading"
    system.update_card = MagicMock(return_value=True)

    system.show_explicit_acp_model_selection(
        "entry",
        "chat-first",
        "codex",
        project,
    )

    system._fetch_acp_models.assert_called_once_with(
        "codex",
        cwd=project.root_path,
        current_model=None,
    )
    system.update_card.assert_called_once()


def test_codex_cascade_redraw_does_not_enumerate_other_providers(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    _storage, _projects, project, _second, _ctx, system = _configuration_lane(
        tmp_path
    )
    _install_single_tool_registration(monkeypatch)
    system._fetch_acp_models = MagicMock(
        return_value=[ACPModelOption(name="gpt-5.6-sol", is_default=True)]
    )
    system.update_card = MagicMock(return_value=True)

    system.handle_acp_model_cascade_select(
        "om-selector",
        "chat-first",
        project.project_id,
        {
            "action": action_ids.SELECT_ACP_MODEL_GROUP,
            "tool_name": "codex",
            "project_id": project.project_id,
            "_option": "gpt-5.6-sol",
        },
    )

    system._fetch_acp_models.assert_called_once_with(
        "codex",
        cwd=project.root_path,
        current_model=None,
    )
    system.update_card.assert_called_once()


def test_codex_final_confirmation_does_not_enumerate_other_providers(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    _storage, _projects, project, _second, _ctx, system = _configuration_lane(
        tmp_path
    )
    _install_single_tool_registration(monkeypatch)
    monkeypatch.setattr(
        "src.feishu.handlers.system.invalidate_acp_model_cache",
        MagicMock(),
    )
    system._fetch_acp_models = MagicMock(
        return_value=[ACPModelOption(name="gpt-5.6-sol", is_default=True)]
    )
    system._activate_acp_selection = MagicMock()

    system.handle_select_acp_model(
        "om-selector",
        "chat-first",
        project.project_id,
        {
            "action": action_ids.SELECT_ACP_MODEL,
            "tool_name": "codex",
            "project_id": project.project_id,
            "model_name": None,
            "use_default_model": True,
        },
    )

    system._fetch_acp_models.assert_called_once_with(
        "codex",
        cwd=project.root_path,
        current_model=None,
    )
    system._activate_acp_selection.assert_called_once()


def test_forged_unregistered_tool_is_rejected_without_global_provider_scan(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    _storage, _projects, project, _second, _ctx, system = _configuration_lane(
        tmp_path
    )
    _install_single_tool_registration(monkeypatch)
    system._fetch_acp_models = MagicMock()
    system._activate_acp_selection = MagicMock()

    system.handle_select_acp_model(
        "om-selector",
        "chat-first",
        project.project_id,
        {
            "action": action_ids.SELECT_ACP_MODEL,
            "tool_name": "forged-provider",
            "project_id": project.project_id,
            "model_name": None,
            "use_default_model": True,
        },
    )

    system.reply_error.assert_called_once()
    system._fetch_acp_models.assert_not_called()
    system._activate_acp_selection.assert_not_called()


def test_tools_list_uses_programming_availability_without_acp_probe(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    _storage, _projects, project, _second, _ctx, system = _configuration_lane(
        tmp_path
    )
    availability = MagicMock(side_effect=lambda name, **_kwargs: name == "codex")
    monkeypatch.setattr(
        "src.feishu.handlers.system.is_programming_tool_available",
        availability,
    )
    system.reply_interactive_card = MagicMock()

    system.show_tools_list("tools", "chat-first", project)

    assert [call.args[0] for call in availability.call_args_list] == [
        "coco",
        "claude",
        "aiden",
        "codex",
        "gemini",
        "traex",
        "grok",
        "dsh",
    ]
    assert all(
        call.kwargs
        == {"allow_sync_probe": False, "trigger_async_probe": True}
        for call in availability.call_args_list
    )
    system.reply_interactive_card.assert_called_once()


def test_tools_status_uses_programming_availability_without_acp_probe(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    _storage, _projects, project, _second, ctx, system = _configuration_lane(
        tmp_path
    )
    availability = MagicMock(side_effect=lambda name, **_kwargs: name == "codex")
    monkeypatch.setattr(
        "src.feishu.handlers.system.is_programming_tool_available",
        availability,
    )
    for name in ("coco", "claude", "aiden", "codex", "gemini", "traex", "grok", "dsh"):
        getattr(ctx, f"{name}_manager").list_active_sessions.return_value = []
    system.reply_interactive_card = MagicMock()

    system.show_tools_status("tools-status", "chat-first", project)

    assert [call.args[0] for call in availability.call_args_list] == [
        "coco",
        "claude",
        "aiden",
        "codex",
        "gemini",
        "traex",
        "grok",
        "dsh",
    ]
    assert all(
        call.kwargs
        == {"allow_sync_probe": False, "trigger_async_probe": True}
        for call in availability.call_args_list
    )
    system.reply_interactive_card.assert_called_once()


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


def test_generic_exit_card_action_delegates_to_current_mode_controller(tmp_path):
    _storage, _projects, first, _second, _ctx, system = _configuration_lane(tmp_path)
    system.exit_current_mode = MagicMock()
    actions = init_action_registry(_action_registry_client(first, system))

    actions["exit"]("m", "chat-first", first.project_id, {})

    system.exit_current_mode.assert_called_once_with("m", "chat-first", first)


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
    assert patched_cards[1]["header"]["template"] == "orange"
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
