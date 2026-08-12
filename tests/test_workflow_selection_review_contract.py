from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

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


def test_renderer_uses_callback_selects_without_cardkit_form_submit() -> None:
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
    assert set(selects) == {
        "tool_name",
        "model_group",
        "model_profile",
        "model_effort",
    }
    assert {option["value"] for option in selects["tool_name"]["options"]} == {
        "codex",
        "gemini",
    }
    assert {option["value"] for option in selects["model_group"]["options"]} == {
        "default",
        "gpt",
    }
    assert [option["value"] for option in selects["model_profile"]["options"]] == [
        "standard"
    ]
    assert [option["value"] for option in selects["model_effort"]["options"]] == [
        "low",
        "high",
    ]
    assert selects["model_group"]["initial_option"] == "gpt"
    assert selects["model_profile"]["initial_option"] == "standard"
    assert selects["model_effort"]["initial_option"] == "high"
    assert selects["model_group"]["behaviors"] == [
        {"type": "callback", "value": selects["model_group"]["value"]}
    ]
    forms = [item for item in _walk(card) if item.get("tag") == "form"]
    assert forms == []
    primary = [item for item in _walk(card) if item.get("type") == "primary"]
    assert len(primary) == 1
    assert primary[0]["text"]["content"] == "使用此池开始编排"
    assert not any(item.get("tag") == "action" for item in _walk(card))
    add = next(
        item
        for item in _walk(card)
        if item.get("tag") == "button" and item.get("value", {}).get("action") == "workflow_add_agent"
    )
    assert "action_type" not in add
    assert "form_action_type" not in add
    assert "form_name" not in add
    assert add["behaviors"] == [{"type": "callback", "value": add["value"]}]
    assert selects["tool_name"]["behaviors"] == [
        {"type": "callback", "value": selects["tool_name"]["value"]}
    ]


def test_add_agent_payload_changes_with_the_resolved_draft_state() -> None:
    pending = PendingWorkflow(
        requirement="build",
        selection_session_key="selection-1",
        project_id="project-1",
        draft_tool_name="codex",
        draft_model_name="gpt",
        draft_profile="standard",
        draft_effort="high",
    )
    codex_state = resolve_model_cascade(
        _catalog("codex"),
        selected_model="gpt",
        selected_profile="standard",
        selected_effort="high",
    )

    def add_value(model_state) -> dict:
        card = WorkflowAgentSelectionRenderer(
            pending,
            project_id="project-1",
            tool_options={"codex": "Codex", "traex": "Traex"},
            model_state=model_state,
        ).render()
        return next(
            item["value"]
            for item in _walk(card)
            if item.get("tag") == "button"
            and item.get("value", {}).get("action") == "workflow_add_agent"
        )

    codex_value = add_value(codex_state)
    assert codex_value["_selection_sig"]
    assert add_value(codex_state)["_selection_sig"] == codex_value["_selection_sig"]

    pending.agent_pool = (_binding("A1", "codex", "gpt/standard/high"),)
    pending.next_agent_sequence = 2
    assert add_value(codex_state)["_selection_sig"] != codex_value["_selection_sig"]

    pending.agent_pool = ()
    removed_value = add_value(codex_state)
    assert removed_value["_selection_sig"] != codex_value["_selection_sig"]

    pending.draft_tool_name = "traex"
    pending.draft_model_name = "openrouter-3o"
    pending.draft_profile = "standard"
    pending.draft_effort = "xhigh"
    traex_state = resolve_model_cascade(
        [
            ACPModelOption(
                name="openrouter-3o",
                selection_variants=(
                    ACPModelSelectionVariant(
                        name="openrouter-3o/standard/xhigh",
                        model="openrouter-3o",
                        profile="standard",
                        effort="xhigh",
                    ),
                ),
            )
        ],
        selected_model="openrouter-3o",
        selected_profile="standard",
        selected_effort="xhigh",
    )
    traex_value = add_value(traex_state)

    assert traex_value["_selection_sig"] != codex_value["_selection_sig"]


@pytest.mark.parametrize(
    ("action", "option"),
    [
        ("workflow_select_model", "gpt"),
        ("workflow_select_profile", "standard"),
        ("workflow_select_effort", "high"),
        ("workflow_add_agent", None),
    ],
)
def test_selection_action_reads_the_model_catalog_once(
    tmp_path,
    action: str,
    option: str | None,
) -> None:
    pending = PendingWorkflow(
        requirement="build",
        initiator_user_id="user-1",
        selection_session_key="selection-1",
        project_id="project-1",
        draft_tool_name="codex",
        draft_model_name="gpt",
        draft_profile="standard",
        draft_effort="low",
    )
    handler, _engine, project = _selection(tmp_path, pending=pending)
    payload = {
        "action": action,
        "project_id": project.project_id,
        "selection_session_key": "selection-1",
    }
    if option is not None:
        payload["_option"] = {"value": option}
    fetch_models = MagicMock(return_value=_catalog("codex"))

    with (
        patch("src.thread.get_current_sender_id", return_value="user-1"),
        patch(
            "src.workflow_engine.tool_registry.get_available_tools",
            return_value={"codex": "Codex"},
        ),
        patch("src.acp.helper.fetch_acp_models", fetch_models),
    ):
        handler.handle_workflow_agent_action(
            "selection-card",
            "chat-1",
            project.project_id,
            payload,
        )

    assert fetch_models.call_count == 1


@pytest.mark.parametrize(
    ("action", "option"),
    [
        ("workflow_select_model", "retired-model"),
        ("workflow_select_profile", "retired-profile"),
        ("workflow_select_effort", "retired-effort"),
    ],
)
def test_invalid_cascade_callback_redraws_the_authoritative_draft_state(
    tmp_path,
    action: str,
    option: str,
) -> None:
    pending = PendingWorkflow(
        requirement="build",
        initiator_user_id="user-1",
        selection_session_key="selection-1",
        project_id="project-1",
        draft_tool_name="codex",
        draft_model_name="gpt",
        draft_profile="standard",
        draft_effort="high",
    )
    handler, engine, project = _selection(tmp_path, pending=pending)
    fetch_models = MagicMock(return_value=_catalog("codex"))
    payload = {
        "action": action,
        "project_id": project.project_id,
        "selection_session_key": "selection-1",
        "_option": {"value": option},
    }

    with (
        patch("src.thread.get_current_sender_id", return_value="user-1"),
        patch(
            "src.workflow_engine.tool_registry.get_available_tools",
            return_value={"codex": "Codex"},
        ),
        patch("src.acp.helper.fetch_acp_models", fetch_models),
    ):
        handler.handle_workflow_agent_action(
            "selection-card",
            "chat-1",
            project.project_id,
            payload,
        )

    assert fetch_models.call_count == 1
    assert engine.project.pending.draft_model_name == "gpt"
    assert engine.project.pending.draft_profile == "standard"
    assert engine.project.pending.draft_effort == "high"
    card = handler.update_card.call_args.args[1]
    effort_select = next(
        item
        for item in _walk(card)
        if item.get("tag") == "select_static"
        and item.get("name") == "model_effort"
    )
    assert effort_select["initial_option"] == "high"


@pytest.mark.parametrize(
    (
        "draft_model",
        "draft_profile",
        "draft_effort",
        "expected_effort",
    ),
    [
        ("retired-model", None, None, "low"),
        ("gpt", "retired-profile", "high", "high"),
        ("gpt", "standard", "retired-effort", "low"),
    ],
)
def test_stale_add_reconciles_the_draft_to_the_visible_catalog_once(
    tmp_path,
    draft_model: str,
    draft_profile: str | None,
    draft_effort: str | None,
    expected_effort: str,
) -> None:
    pending = PendingWorkflow(
        requirement="build",
        initiator_user_id="user-1",
        selection_session_key="selection-1",
        project_id="project-1",
        draft_tool_name="codex",
        draft_model_name=draft_model,
        draft_profile=draft_profile,
        draft_effort=draft_effort,
    )
    handler, engine, project = _selection(tmp_path, pending=pending)
    fetch_models = MagicMock(return_value=_catalog("codex"))

    with (
        patch("src.thread.get_current_sender_id", return_value="user-1"),
        patch(
            "src.workflow_engine.tool_registry.get_available_tools",
            return_value={"codex": "Codex"},
        ),
        patch("src.acp.helper.fetch_acp_models", fetch_models),
    ):
        handler.handle_workflow_agent_action(
            "selection-card",
            "chat-1",
            project.project_id,
            {
                "action": "workflow_add_agent",
                "project_id": project.project_id,
                "selection_session_key": "selection-1",
            },
        )

    assert fetch_models.call_count == 1
    assert engine.project.pending.agent_pool == ()
    assert engine.project.pending.selection_error
    assert engine.project.pending.draft_model_name == "gpt"
    assert engine.project.pending.draft_profile == "standard"
    assert engine.project.pending.draft_effort == expected_effort
    card = handler.update_card.call_args.args[1]
    selects = {
        item["name"]: item
        for item in _walk(card)
        if item.get("tag") == "select_static"
    }
    assert selects["model_group"]["initial_option"] == "gpt"
    assert selects["model_profile"]["initial_option"] == "standard"
    assert selects["model_effort"]["initial_option"] == expected_effort
    assert any(
        item.get("tag") == "button"
        and item.get("value", {}).get("action") == "workflow_add_agent"
        for item in _walk(card)
    )


def test_server_consumes_trusted_options_and_builds_same_tool_different_model_pool(tmp_path) -> None:
    handler, engine, project = _selection(tmp_path)

    _act(handler, project, "workflow_select_tool", option="codex")
    _act(handler, project, "workflow_select_model", option="gpt")
    _act(handler, project, "workflow_select_effort", option="high")
    _act(handler, project, "workflow_add_agent")
    _act(handler, project, "workflow_select_effort", option="low")
    _act(handler, project, "workflow_add_agent")

    assert [binding.model_name for binding in engine.project.pending.agent_pool] == [
        "gpt/standard/high",
        "gpt/standard/low",
    ]


def test_tool_callback_atomically_refreshes_the_matching_model_catalog(tmp_path) -> None:
    pending = PendingWorkflow(
        requirement="build",
        initiator_user_id="user-1",
        selection_session_key="selection-1",
        project_id="project-1",
        draft_tool_name="traex",
    )
    handler, engine, project = _selection(tmp_path, pending=pending)

    _act(handler, project, "workflow_select_tool", option="codex")

    assert engine.project.pending.draft_tool_name == "codex"
    card = handler.update_card.call_args.args[1]
    model_select = next(
        item
        for item in _walk(card)
        if item.get("tag") == "select_static"
        and item.get("name") == "model_group"
    )
    assert {option["value"] for option in model_select["options"]} == {
        "default",
        "gpt",
    }
    assert "traex-pro" not in json.dumps(model_select, ensure_ascii=False)
    assert not any(
        item.get("name") in {"model_profile", "model_effort"}
        for item in _walk(card)
    )


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
    assert 'id="modelSelect"' in preview
    assert 'id="profileSelect"' in preview
    assert 'id="effortSelect"' in preview
    assert "选择精确配置" not in preview
