from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.acp.options import ACPModelOption, ACPModelSelectionVariant
from src.card.render.model_cascade import resolve_model_cascade
from src.feishu.handlers.workflow import WorkflowHandler, _WorkflowLifecycleOwner
from src.workflow_engine import constants
from src.workflow_engine.agent_pool import WorkflowAgentBinding
from src.workflow_engine.engine import WorkflowEngine
from src.workflow_engine.models import PendingWorkflow, WorkflowProject, WorkflowStatus
from src.workflow_engine.renderer import (
    WorkflowAgentSelectionRenderer,
    WorkflowProgressRenderer,
)


def _catalog(tool: str) -> list[ACPModelOption]:
    if tool == "codex":
        return [
            ACPModelOption(
                name="gpt",
                selection_variants=(
                    ACPModelSelectionVariant(
                        name="gpt/standard/low",
                        model="gpt",
                        profile="standard",
                        effort="low",
                    ),
                    ACPModelSelectionVariant(
                        name="gpt/standard/high",
                        model="gpt",
                        profile="standard",
                        effort="high",
                    ),
                ),
            )
        ]
    return [ACPModelOption(name=f"{tool}-pro")]


def _binding(agent_id: str, tool: str, model: str | None = None) -> WorkflowAgentBinding:
    return WorkflowAgentBinding(
        agent_id=agent_id,
        tool_name=tool,
        model_name=model,
        display_name=f"{tool}:{model or 'default'}",
    )


def _selection(tmp_path, *, pending: PendingWorkflow | None = None):
    project = SimpleNamespace(
        project_id="project-1",
        project_name="Project",
        root_path=str(tmp_path),
    )
    pending = pending or PendingWorkflow(
        requirement="build",
        initiator_user_id="user-1",
        selection_session_key="selection-1",
        project_id=project.project_id,
        draft_tool_name="codex",
    )
    engine = WorkflowEngine("chat-1", str(tmp_path))
    engine._project = WorkflowProject(
        status=WorkflowStatus.SELECTING_AGENTS,
        pending=pending,
    )
    engine._workflow_selection_owner = _WorkflowLifecycleOwner(
        "selection-1",
        "user-1",
        chat_id="chat-1",
        project_id=project.project_id,
        root_path=str(tmp_path),
    )
    handler = WorkflowHandler.__new__(WorkflowHandler)
    manager = MagicMock()
    manager.get.return_value = engine
    handler.ctx = SimpleNamespace(
        workflow_engine_manager=manager,
        settings=SimpleNamespace(admin_user_ids=[]),
    )
    handler._resolve_project_from_id = MagicMock(return_value=project)
    handler._get_root_path = MagicMock(return_value=str(tmp_path))
    handler._reply_workflow_error = MagicMock()
    handler.update_card = MagicMock(return_value=True)
    handler._schedule_generate_and_start_workflow = MagicMock()
    return handler, engine, project


def _act(
    handler: WorkflowHandler,
    project: SimpleNamespace,
    action: str,
    *,
    option: str | None = None,
    tools: dict[str, str] | None = None,
    **values,
) -> None:
    payload = {
        "action": action,
        "project_id": project.project_id,
        "selection_session_key": "selection-1",
        **values,
    }
    if option is not None:
        payload["_option"] = {"value": option}
    with (
        patch("src.thread.get_current_sender_id", return_value="user-1"),
        patch(
            "src.workflow_engine.tool_registry.get_available_tools",
            return_value=tools
            or {"codex": "Codex", "gemini": "Gemini", "traex": "Traex"},
        ),
        patch("src.acp.helper.fetch_acp_models", side_effect=lambda tool, *_a, **_k: _catalog(tool)),
    ):
        handler.handle_workflow_agent_action(
            "selection-card",
            "chat-1",
            project.project_id,
            payload,
        )


def _walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def test_renderer_batches_exact_model_selection_into_one_form_submit() -> None:
    pending = PendingWorkflow(
        requirement="build",
        selection_session_key="selection-1",
        project_id="project-1",
        draft_tool_name="codex",
        draft_model_name="gpt",
        draft_profile="standard",
        draft_effort="high",
        agent_pool=(_binding("A-1", "codex", "gpt/standard/high"),),
    )
    state = resolve_model_cascade(
        _catalog("codex"),
        selected_model="gpt",
        selected_profile="standard",
        selected_effort="high",
    )
    card = WorkflowAgentSelectionRenderer(
        pending,
        project_id="project-1",
        tool_options={"codex": "Codex", "gemini": "Gemini"},
        model_state=state,
    ).render()

    selects = {item.get("name"): item for item in _walk(card) if item.get("tag") == "select_static"}
    assert set(selects) == {"tool_name", "model_selection"}
    assert {option["value"] for option in selects["tool_name"]["options"]} == {
        "codex",
        "gemini",
    }
    assert {option["value"] for option in selects["model_selection"]["options"]} == {
        "default",
        "gpt/standard/low",
        "gpt/standard/high",
    }
    assert selects["model_selection"]["behaviors"] == []
    forms = [item for item in _walk(card) if item.get("tag") == "form"]
    assert len(forms) == 1
    assert forms[0]["name"] == "workflow_agent_binding"
    primary = [item for item in _walk(card) if item.get("type") == "primary"]
    assert len(primary) == 1
    assert primary[0]["text"]["content"] == "使用此池开始编排"
    assert not any(item.get("tag") == "action" for item in _walk(card))
    add = next(
        item
        for item in _walk(card)
        if item.get("tag") == "button" and item.get("value", {}).get("action") == "workflow_add_agent"
    )
    assert add["action_type"] == "form_submit"
    assert add["form_action_type"] == "submit"
    assert add["form_name"] == "workflow_agent_binding"
    assert selects["tool_name"]["behaviors"] == [
        {"type": "callback", "value": selects["tool_name"]["value"]}
    ]


def test_server_consumes_trusted_options_and_builds_same_tool_different_model_pool(tmp_path) -> None:
    handler, engine, project = _selection(tmp_path)

    _act(handler, project, "workflow_select_tool", option="codex")
    _act(
        handler,
        project,
        "workflow_add_agent",
        _form_value={"model_selection": "gpt/standard/high"},
    )
    _act(
        handler,
        project,
        "workflow_add_agent",
        _form_value={"model_selection": "gpt/standard/low"},
    )

    assert [binding.model_name for binding in engine.project.pending.agent_pool] == [
        "gpt/standard/high",
        "gpt/standard/low",
    ]


def test_forged_or_stale_capability_values_fail_closed_with_inline_error(tmp_path) -> None:
    handler, engine, project = _selection(tmp_path)

    _act(handler, project, "workflow_select_tool", option="rogue")
    assert engine.project.pending.draft_tool_name == "codex"
    assert engine.project.pending.selection_error
    assert "rogue" not in json.dumps(handler.update_card.call_args.args[1], ensure_ascii=False)

    engine.project.pending.selection_error = None
    engine.project.pending.draft_model_name = "forged-model"
    _act(handler, project, "workflow_add_agent")
    assert engine.project.pending.agent_pool == ()
    assert engine.project.pending.selection_error

    engine.project.pending.agent_pool = (_binding("A1", "codex", "retired-model"),)
    engine.project.pending.selection_error = None
    _act(handler, project, "workflow_confirm_agents")
    handler._schedule_generate_and_start_workflow.assert_not_called()
    assert engine.project.status is WorkflowStatus.SELECTING_AGENTS
    assert engine.project.pending.selection_error


def test_auto_orchestrator_is_deterministic_under_pool_reordering(tmp_path) -> None:
    resolved: list[str] = []
    for index, pool in enumerate(
        (
            (_binding("A-codex", "codex"), _binding("A-traex", "traex")),
            (_binding("A-traex", "traex"), _binding("A-codex", "codex")),
        )
    ):
        pending = PendingWorkflow(
            requirement="build",
            initiator_user_id="user-1",
            selection_session_key="selection-1",
            project_id="project-1",
            agent_pool=pool,
            recommended_agents=[
                {"tool_name": "traex", "model_name": None, "display_name": "Traex"},
                {"tool_name": "codex", "model_name": None, "display_name": "Codex"},
            ],
        )
        handler, engine, project = _selection(tmp_path / str(index), pending=pending)
        _act(handler, project, "workflow_confirm_agents")
        resolved.append(engine.project.pending.orchestrator_agent_id)

    assert resolved == ["A-traex", "A-traex"]


def test_pool_limit_and_card_action_rows_are_bounded(tmp_path) -> None:
    assert getattr(constants, "MAX_WORKFLOW_AGENT_POOL_SIZE", None) == 8
    catalog = [ACPModelOption(name=f"m{index}") for index in range(1, 10)]
    pending = PendingWorkflow(
        requirement="build",
        initiator_user_id="user-1",
        selection_session_key="selection-1",
        project_id="project-1",
        draft_tool_name="codex",
        draft_model_name="m9",
        agent_pool=tuple(_binding(f"A{index}", "codex", f"m{index}") for index in range(1, 9)),
        next_agent_sequence=9,
    )
    handler, engine, project = _selection(tmp_path, pending=pending)
    with patch("src.acp.helper.fetch_acp_models", return_value=catalog):
        _act(handler, project, "workflow_add_agent")
    assert len(engine.project.pending.agent_pool) == 8
    assert engine.project.pending.selection_error

    state = resolve_model_cascade(catalog, selected_model="m9")
    card = WorkflowAgentSelectionRenderer(
        engine.project.pending,
        project_id="project-1",
        tool_options={"codex": "Codex"},
        model_state=state,
    ).render()
    assert not any(item.get("tag") == "action" for item in _walk(card))
    assert sum(1 for item in _walk(card) if isinstance(item.get("tag"), str)) <= 180
    serialized = json.dumps(card, ensure_ascii=False)
    assert all(f"A{index}" in serialized for index in range(1, 9))
    assert "A-" not in serialized
    assert "最多 8 个" in serialized
    assert len(serialized.encode("utf-8")) <= 27 * 1024


def test_runtime_agent_plan_is_explicitly_marked_as_runtime_assignment() -> None:
    pending = PendingWorkflow(
        requirement="build",
        agent_pool=(_binding("A1", "codex"), _binding("A2", "gemini")),
        orchestrator_agent_id="A1",
        orchestrator_was_auto=True,
        meta={
            "agentPlan": [
                {
                    "node": "fanout",
                    "role": "worker",
                    "runtime": True,
                    "candidateAgentIds": ["A1", "A2"],
                }
            ]
        },
    )
    card = WorkflowProgressRenderer(
        WorkflowProject(status=WorkflowStatus.GENERATING_SCRIPT, pending=pending)
    ).render_progress_card()
    serialized = json.dumps(card, ensure_ascii=False)
    assert "运行时分配" in serialized
    assert "Auto → A1" in serialized


def test_ux_preview_uses_production_agent_ids_and_explicit_pool_cap() -> None:
    preview = Path("ux/workflow-agent-pool.html").read_text(encoding="utf-8")

    assert "A1、A2" in preview
    assert "最多 8 个" in preview
