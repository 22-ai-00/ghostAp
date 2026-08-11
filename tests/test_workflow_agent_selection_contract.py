from __future__ import annotations

import json
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.acp.options import ACPModelOption
from src.feishu.action_registry import init_action_registry
from src.feishu.handlers.workflow import WorkflowHandler, _WorkflowLifecycleOwner
from src.workflow_engine.agent_pool import WorkflowAgentBinding
from src.workflow_engine.engine import WorkflowEngine
from src.workflow_engine.models import PendingWorkflow, WorkflowProject, WorkflowStatus
from src.workflow_engine.renderer import WorkflowProgressRenderer

_SELECTION_ACTIONS = {
    "workflow_select_tool",
    "workflow_select_model",
    "workflow_select_profile",
    "workflow_select_effort",
    "workflow_add_agent",
    "workflow_add_recommended_pool",
    "workflow_remove_agent",
    "workflow_clear_agents",
    "workflow_set_orchestrator",
    "workflow_confirm_agents",
}


def _model_catalog() -> list[ACPModelOption]:
    return [
        ACPModelOption(name="fast"),
        ACPModelOption(name="deep"),
        ACPModelOption(name="recommended"),
        ACPModelOption(name="pro"),
    ]


def _binding(agent_id: str, model: str) -> WorkflowAgentBinding:
    return WorkflowAgentBinding(
        agent_id=agent_id,
        tool_name="codex",
        model_name=model,
        display_name=f"Codex {model}",
    )


def _project(tmp_path, project_id: str = "project-1") -> SimpleNamespace:
    return SimpleNamespace(
        project_id=project_id,
        project_name="Project",
        root_path=str(tmp_path),
    )


def _bare_handler(engine: WorkflowEngine, project: SimpleNamespace) -> WorkflowHandler:
    handler = WorkflowHandler.__new__(WorkflowHandler)
    manager = MagicMock()
    manager.get.return_value = engine
    manager.get_or_create.return_value = engine
    handler.ctx = SimpleNamespace(
        workflow_engine_manager=manager,
        settings=SimpleNamespace(admin_user_ids=[]),
    )
    handler._resolve_project_from_id = MagicMock(
        side_effect=lambda project_id, _chat_id: (
            project if project_id == project.project_id else None
        )
    )
    handler._reply_workflow_error = MagicMock()
    handler.update_card = MagicMock(return_value=True)
    handler.send_card_to_chat = MagicMock(return_value="selection-card")
    handler._schedule_generate_and_start_workflow = MagicMock()
    return handler


def _selection_context(tmp_path):
    project = _project(tmp_path)
    engine = WorkflowEngine("chat-1", str(tmp_path))
    pending = PendingWorkflow(
        requirement="implement the workflow",
        initiator_user_id="user-1",
        selection_session_key="selection-1",
        project_id=project.project_id,
        next_agent_sequence=1,
        recommended_agents=[
            {
                "tool_name": "codex",
                "model_name": "recommended",
                "display_name": "Codex Recommended",
            }
        ],
    )
    engine._project = WorkflowProject(
        status=WorkflowStatus.SELECTING_AGENTS,
        pending=pending,
    )
    owner = _WorkflowLifecycleOwner(
        "selection-1",
        "user-1",
        chat_id="chat-1",
        project_id=project.project_id,
        root_path=str(tmp_path),
    )
    engine._workflow_selection_owner = owner
    return _bare_handler(engine, project), engine, project


def _act(
    handler: WorkflowHandler,
    project: SimpleNamespace,
    action: str,
    *,
    user: str = "user-1",
    session: str = "selection-1",
    **values,
) -> None:
    option_key = {
        "workflow_select_tool": "tool",
        "workflow_select_model": "model",
        "workflow_select_profile": "profile",
        "workflow_select_effort": "effort",
    }.get(action)
    selected_option = values.pop(option_key, None) if option_key else None
    payload = {
        "action": action,
        "project_id": project.project_id,
        "selection_session_key": session,
        **values,
    }
    if selected_option is not None:
        payload["_option"] = {"value": selected_option}
    with (
        patch("src.thread.get_current_sender_id", return_value=user),
        patch("src.acp.helper.fetch_acp_models", return_value=_model_catalog()),
        patch(
            "src.workflow_engine.tool_registry.get_available_tools",
            return_value={"codex": "Codex", "gemini": "Gemini"},
        ),
    ):
        handler.handle_workflow_agent_action(
            "selection-card",
            "chat-1",
            project.project_id,
            payload,
        )


def test_wf_enters_selecting_without_generation_session(tmp_path) -> None:
    project = _project(tmp_path)
    engine = WorkflowEngine("chat-1", str(tmp_path))
    handler = _bare_handler(engine, project)
    handler._ensure_project = MagicMock(return_value=project)
    admission_owner = _WorkflowLifecycleOwner("admission", "user-1")
    engine._workflow_selection_owner = admission_owner
    handler._supersede_incomplete_workflow = MagicMock(
        return_value=(True, None, admission_owner)
    )
    handler._ensure_topic_engine_context = MagicMock()
    handler.add_reaction = MagicMock()
    handler.get_engine_name = MagicMock(return_value="codex")
    handler._start_workflow_with_defaults = MagicMock()
    handler.ctx.workflow_engine_manager.get.return_value = None

    with (
        patch(
            "src.workflow_engine.bridge.RuntimeBridge.check_node_available",
            return_value=True,
        ),
        patch("src.thread.get_current_sender_id", return_value="user-1"),
        patch("src.acp.helper.fetch_acp_models", return_value=_model_catalog()),
        patch(
            "src.workflow_engine.tool_registry.get_available_tools",
            return_value={"codex": "Codex", "gemini": "Gemini"},
        ),
    ):
        handler.start_workflow(
            "origin-1",
            "chat-1",
            "implement a reliable workflow",
            project,
        )

    assert engine.project.status is WorkflowStatus.SELECTING_AGENTS
    assert engine.project.pending is not None
    assert engine.project.pending.agent_pool == ()
    assert engine.project.pending.engine_session_key is None
    assert engine._script_generation_owner is None
    assert engine._workflow_selection_owner is not None
    handler._start_workflow_with_defaults.assert_not_called()
    handler.send_card_to_chat.assert_called_once()


def test_selection_add_remove_clear_duplicate_and_stable_ids(tmp_path) -> None:
    handler, engine, project = _selection_context(tmp_path)

    _act(handler, project, "workflow_select_tool", tool="codex")
    _act(handler, project, "workflow_select_model", model="fast")
    _act(handler, project, "workflow_add_agent")
    _act(handler, project, "workflow_select_model", model="deep")
    _act(handler, project, "workflow_add_agent")

    assert [(item.agent_id, item.tool_name, item.model_name) for item in engine.project.pending.agent_pool] == [
        ("A1", "codex", "fast"),
        ("A2", "codex", "deep"),
    ]

    _act(handler, project, "workflow_select_model", model="fast")
    _act(handler, project, "workflow_add_agent")
    assert [item.agent_id for item in engine.project.pending.agent_pool] == ["A1", "A2"]
    assert handler._reply_workflow_error.call_count == 0
    assert engine.project.pending.selection_error == "相同工具和模型的 Agent 已在并发池中。"

    _act(handler, project, "workflow_remove_agent", agent_id="A1")
    assert [item.agent_id for item in engine.project.pending.agent_pool] == ["A2"]
    _act(handler, project, "workflow_clear_agents")
    assert engine.project.pending.agent_pool == ()

    _act(handler, project, "workflow_add_recommended_pool")
    assert [item.agent_id for item in engine.project.pending.agent_pool] == ["A3"]


def test_confirm_auto_or_explicit_orchestrator_freezes_pool_once(tmp_path) -> None:
    handler, engine, project = _selection_context(tmp_path)
    release_owner = MagicMock(wraps=engine.release_lifecycle_owner)
    engine.release_lifecycle_owner = release_owner
    engine.project.pending.agent_pool = (_binding("A1", "fast"), _binding("A2", "deep"))
    engine.project.pending.next_agent_sequence = 3
    selection_owner = engine._workflow_selection_owner

    _act(handler, project, "workflow_set_orchestrator", agent_id="A2")
    assert engine.project.pending.orchestrator_agent_id == "A2"
    _act(handler, project, "workflow_set_orchestrator", agent_id="auto")
    assert engine.project.pending.orchestrator_agent_id is None
    _act(handler, project, "workflow_confirm_agents")

    pending = engine.project.pending
    assert engine.project.status is WorkflowStatus.GENERATING_SCRIPT
    assert pending.orchestrator_agent_id == "A2"
    assert pending.orchestrator_was_auto is True
    assert pending.engine_session_key
    assert pending.engine_session_key != "selection-1"
    assert engine._script_generation_owner is not None
    assert engine._workflow_selection_owner is None
    assert selection_owner.done_event.is_set()
    assert all(item is not selection_owner for item in engine._retired_lifecycle_owners)
    release_owner.assert_any_call(selection_owner)
    handler._schedule_generate_and_start_workflow.assert_called_once()

    _act(handler, project, "workflow_confirm_agents")
    handler._schedule_generate_and_start_workflow.assert_called_once()


def test_repeated_selection_supersede_releases_each_previous_owner(tmp_path) -> None:
    handler, engine, _project_context = _selection_context(tmp_path)
    release_owner = MagicMock(wraps=engine.release_lifecycle_owner)
    engine.release_lifecycle_owner = release_owner

    for _index in range(24):
        previous_owner = engine._workflow_selection_owner
        ok, error, new_owner = handler._supersede_incomplete_workflow(
            engine,
            root_path=str(tmp_path),
            current_user="user-1",
        )
        assert ok is True
        assert error is None
        assert new_owner is engine._workflow_selection_owner
        assert previous_owner.done_event.is_set()
        assert all(
            item is not previous_owner
            for item in engine._retired_lifecycle_owners
        )

    assert engine._retired_lifecycle_owners == []
    assert release_owner.call_count == 24


def test_selection_actions_fail_closed_for_stale_project_session_or_user(tmp_path) -> None:
    handler, engine, project = _selection_context(tmp_path)
    original = engine.project.pending.model_copy(deep=True)

    _act(
        handler,
        project,
        "workflow_add_agent",
        session="stale-selection",
        tool="codex",
        model="fast",
    )
    _act(
        handler,
        project,
        "workflow_add_agent",
        user="other-user",
        tool="codex",
        model="fast",
    )
    handler.handle_workflow_agent_action(
        "selection-card",
        "chat-1",
        "wrong-project",
        {
            "action": "workflow_add_agent",
            "project_id": "wrong-project",
            "selection_session_key": "selection-1",
            "tool": "codex",
            "model": "fast",
        },
    )

    assert engine.project.pending == original
    handler._schedule_generate_and_start_workflow.assert_not_called()
    handler.update_card.assert_not_called()
    assert handler._reply_workflow_error.call_count == 3


def test_selection_renderer_carries_full_pool_auto_and_cas_coordinates(tmp_path) -> None:
    from src.workflow_engine.renderer import WorkflowAgentSelectionRenderer

    pending = PendingWorkflow(
        requirement="build",
        selection_session_key="selection-1",
        project_id="project-1",
        draft_tool_name="codex",
        draft_model_name="fast",
        draft_profile="standard",
        draft_effort="medium",
        agent_pool=(_binding("A-1", "fast"), _binding("A-2", "deep")),
        recommended_agents=[
            {
                "tool_name": "gemini",
                "model_name": "pro",
                "display_name": "Gemini Pro",
            }
        ],
    )
    card = WorkflowAgentSelectionRenderer(
        pending,
        project_id="project-1",
        tool_options={"codex": "Codex", "gemini": "Gemini"},
        model_state=SimpleNamespace(
            model_names=("fast", "deep"),
            profiles=("standard",),
            efforts=("medium",),
        ),
    ).render()
    serialized = json.dumps(card, ensure_ascii=False)

    assert "Auto" in serialized
    for text in ("A-1", "A-2", "codex", "fast", "deep"):
        assert text in serialized

    values: list[dict] = []
    stack = [card]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if isinstance(item.get("value"), dict):
                values.append(item["value"])
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    workflow_values = [value for value in values if value.get("action") in _SELECTION_ACTIONS]
    assert {value["action"] for value in workflow_values} == _SELECTION_ACTIONS - {
        "workflow_select_profile",
        "workflow_select_effort",
    }
    assert all(value.get("project_id") == "project-1" for value in workflow_values)
    assert all(value.get("selection_session_key") == "selection-1" for value in workflow_values)


def test_progress_card_and_run_spec_preserve_pool_and_agent_plan(tmp_path) -> None:
    pool = (_binding("A-1", "fast"), _binding("A-2", "deep"))
    pending = PendingWorkflow(
        requirement="build",
        agent_pool=pool,
        orchestrator_agent_id="A-2",
        selected_tools=["codex"],
        meta={
            "name": "pool-plan",
            "description": "pool plan",
            "phases": [{"title": "Run", "detail": "run"}],
            "tools": ["codex"],
            "agentPlan": [
                {"node": "main", "role": "lead", "agentId": "A-2"},
                {
                    "node": "fanout",
                    "role": "worker",
                    "runtime": True,
                    "candidateAgentIds": ["A-1", "A-2"],
                },
            ],
        },
    )
    project = WorkflowProject(
        status=WorkflowStatus.GENERATING_SCRIPT,
        pending=pending,
    )
    serialized = json.dumps(
        WorkflowProgressRenderer(project).render_progress_card(),
        ensure_ascii=False,
    )
    for text in ("A-1", "A-2", "fast", "deep", "main", "fanout"):
        assert text in serialized

    engine = WorkflowEngine("chat-1", str(tmp_path))
    spec = WorkflowHandler._build_run_spec(
        pending=pending,
        engine=engine,
        task="build",
        chat_id="chat-1",
        topic_id=None,
    )
    assert spec.agent_pool == pool
    assert spec.orchestrator_agent_id == "A-2"


def test_action_registry_wires_all_selection_actions() -> None:
    client = MagicMock()
    workflow = MagicMock()
    handlers = defaultdict(MagicMock)
    handlers["workflow"] = workflow
    client._handler_ctx.handlers = handlers

    actions = init_action_registry(client)

    assert _SELECTION_ACTIONS <= actions.keys()
    assert all(
        actions[action] is workflow.handle_workflow_agent_action
        for action in _SELECTION_ACTIONS
    )


def _workflow_card_nodes(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _workflow_card_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _workflow_card_nodes(child)


def test_generation_renderer_is_read_only_and_shows_complete_agent_bindings() -> None:
    from src.workflow_engine.renderer import WorkflowGenerationRenderer

    pool = (
        WorkflowAgentBinding(
            agent_id="A1",
            tool_name="codex",
            model_name="gpt-5.6-sol",
            display_name="OpenAI Codex",
            profile="standard",
            effort="high",
        ),
        WorkflowAgentBinding(
            agent_id="A2",
            tool_name="traex",
            model_name="openrouter-3o/max/xhigh",
            display_name="TRAE CLI",
            profile="max",
            effort="xhigh",
        ),
    )
    card = WorkflowGenerationRenderer(
        requirement="精简免费模式的安装流程",
        agent_pool=pool,
        orchestrator_agent_id="A2",
        orchestrator_was_auto=True,
    ).render(
        current_activity="A2 正在生成并验证 Workflow 脚本",
        elapsed_seconds=73,
    )
    serialized = json.dumps(card, ensure_ascii=False)
    nodes = list(_workflow_card_nodes(card))
    markdown = "\n".join(
        str(node.get("content") or "")
        for node in nodes
        if node.get("tag") == "markdown"
    )

    for expected in (
        "精简免费模式的安装流程",
        "Auto → A2",
        "A1",
        "A2",
        "codex",
        "traex",
        "gpt-5.6-sol",
        "standard",
        "high",
        "openrouter-3o",
        "max",
        "xhigh",
        "1 分钟 13 秒",
    ):
        assert expected in serialized
    assert serialized.count("A2 正在生成并验证 Workflow 脚本") == 1
    assert "openrouter-3o/max/xhigh" not in markdown
    assert markdown.count("openrouter-3o") == 1
    assert markdown.count("`max`") == 1
    assert markdown.count("`xhigh`") == 1
    assert not any(node.get("tag") in {"select_static", "button"} for node in nodes)
    assert not any("behaviors" in node or "value" in node for node in nodes)


def test_generation_card_formats_wait_time_as_hours_minutes_and_seconds() -> None:
    from src.workflow_engine.renderer import WorkflowGenerationRenderer

    renderer = WorkflowGenerationRenderer(
        requirement="检查 Workflow 卡片",
        agent_pool=(),
        orchestrator_agent_id="A1",
    )

    for elapsed, expected in (
        (32, "已等待 32 秒"),
        (128, "已等待 2 分钟 8 秒"),
        (3789, "已等待 1 小时 3 分钟 9 秒"),
    ):
        serialized = json.dumps(
            renderer.render(elapsed_seconds=elapsed),
            ensure_ascii=False,
        )
        assert expected in serialized


def test_confirm_replaces_selection_card_and_propagates_fallback_message_id(tmp_path) -> None:
    for patch_ok, replacement_id, expected_id in (
        (True, None, "selection-card"),
        (False, "generation-card", "generation-card"),
    ):
        handler, engine, project = _selection_context(tmp_path)
        engine.project.pending.agent_pool = (
            _binding("A1", "fast"),
            _binding("A2", "deep"),
        )
        handler.update_card.return_value = patch_ok
        handler.send_card_to_chat.return_value = replacement_id
        handler.update_card.reset_mock()
        handler.send_card_to_chat.reset_mock()

        _act(handler, project, "workflow_confirm_agents")

        handler.update_card.assert_called_once()
        assert handler.update_card.call_args.args[0] == "selection-card"
        rendered = handler.update_card.call_args.args[1]
        rendered_nodes = list(_workflow_card_nodes(rendered))
        assert not any(
            node.get("tag") in {"select_static", "button"}
            for node in rendered_nodes
        )
        if patch_ok:
            handler.send_card_to_chat.assert_not_called()
        else:
            handler.send_card_to_chat.assert_called_once()
        assert (
            handler._schedule_generate_and_start_workflow.call_args.kwargs["message_id"]
            == expected_id
        )
