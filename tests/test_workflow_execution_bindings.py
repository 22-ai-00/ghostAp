"""Contracts for freezing and enforcing Workflow execution bindings."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.feishu.handlers.workflow import WorkflowHandler
from src.spec_engine.review_agents import ReviewAgentBinding
from src.workflow_engine.engine import WorkflowEngine
from src.workflow_engine.executor import AgentExecutor
from src.workflow_engine.models import (
    AgentCallParams,
    AgentCallResult,
    PendingConfirmation,
    WorkflowProject,
    WorkflowStatus,
)
from src.workflow_engine.run_spec import WorkflowRunSpec


def _binding(tool: str, model: str | None, *, selection_key: str) -> ReviewAgentBinding:
    return ReviewAgentBinding(
        provider="cli",
        tool_name=tool,
        display_name=tool.title(),
        agent_type=tool,
        model_name=model,
        model_display_name=model,
        selection_key=selection_key,
        use_default_model=model is None,
    )


def _script(tmp_path) -> str:
    path = tmp_path / "binding-contract.js"
    path.write_text(
        """
export const meta = {
  name: "binding-contract",
  description: "binding contract",
  tools: ["coco", "claude"],
};
export default async function workflow() { return "ok"; }
""",
        encoding="utf-8",
    )
    return str(path)


def _run_spec(*, auto_reviewer: bool = False) -> WorkflowRunSpec:
    orchestrator = _binding("coco", "selected-coco-model", selection_key="coco:selected")
    reviewers = () if auto_reviewer else (
        _binding("claude", "selected-claude-model", selection_key="claude:selected"),
    )
    return WorkflowRunSpec(
        orchestrator=orchestrator,
        reviewers=reviewers,
        tool_model_map={
            "coco": "selected-coco-model",
            **({} if auto_reviewer else {"claude": "selected-claude-model"}),
        },
        task="implement the binding contract",
        chat_id="chat-1",
        topic_id="topic-1",
        budget=12,
        deadline=None,
        auto_reviewer=auto_reviewer,
        initiator_user_id="user-1",
        allowed_tools=("coco",) if auto_reviewer else ("coco", "claude"),
    )


def test_workflow_run_spec_is_deeply_frozen() -> None:
    spec = _run_spec()

    with pytest.raises((AttributeError, TypeError)):
        spec.task = "tampered"  # type: ignore[misc]
    with pytest.raises(TypeError):
        spec.tool_model_map["coco"] = "tampered"  # type: ignore[index]

    assert spec.reviewers[0].model_name == "selected-claude-model"
    assert spec.to_dict()["auto_reviewer"] is False

    with pytest.raises(ValueError, match="model"):
        WorkflowRunSpec(
            orchestrator=spec.orchestrator,
            reviewers=spec.reviewers,
            tool_model_map={
                "coco": "different-from-confirmed-selection",
                "claude": "selected-claude-model",
            },
            task=spec.task,
            chat_id=spec.chat_id,
            topic_id=spec.topic_id,
            budget=spec.budget,
            deadline=spec.deadline,
            auto_reviewer=spec.auto_reviewer,
            allowed_tools=spec.allowed_tools,
        )


def test_selected_model_map_reaches_every_agent_call(tmp_path) -> None:
    """Script-provided models cannot override the confirmed immutable map."""
    spec = _run_spec()
    observed: list[AgentCallParams] = []

    class CallingBridge:
        def __init__(self, *args, on_agent_call, allowed_tools, **kwargs):
            self.on_agent_call = on_agent_call
            self.allowed_tools = allowed_tools

        @staticmethod
        def check_node_available() -> bool:
            return True

        def start(self) -> None:
            assert self.allowed_tools == ["coco", "claude"]

        def run(self) -> str:
            first = self.on_agent_call(
                AgentCallParams(
                    prompt="first task",
                    tool="coco",
                    model="script-invented-model",
                    label="first",
                )
            )
            second = self.on_agent_call(
                AgentCallParams(
                    prompt="second task",
                    tool="claude",
                    model=None,
                    label="second",
                )
            )
            return json.dumps({"outputs": [first.output, second.output]})

        def stop(self) -> None:
            return None

    def fake_execute(
        _executor: AgentExecutor,
        params: AgentCallParams,
        **_kwargs,
    ) -> AgentCallResult:
        observed.append(params.model_copy(deep=True))
        return AgentCallResult(
            output=f"{params.tool}:{params.model}",
            stop_reason="end_turn",
            tool=params.tool,
            model=params.model,
        )

    engine = WorkflowEngine(chat_id="chat-1", root_path=str(tmp_path))
    with (
        patch("src.workflow_engine.engine.RuntimeBridge", CallingBridge),
        patch.object(AgentExecutor, "execute", fake_execute),
    ):
        project = engine.execute_workflow(
            script_path=_script(tmp_path),
            run_spec=spec,
        )

    assert project.status == WorkflowStatus.COMPLETED
    assert [(call.tool, call.model) for call in observed[:2]] == [
        ("coco", "selected-coco-model"),
        ("claude", "selected-claude-model"),
    ]
    assert project.run_spec == spec.to_dict()
    assert project.tool_model_map == dict(spec.tool_model_map)


def test_confirm_passes_one_frozen_run_spec_to_engine(tmp_path) -> None:
    """The handler must not fan confirmed bindings back into mutable kwargs."""
    script_text = """
export const meta = {
  name: "handler-binding-contract",
  description: "handler binding contract",
  tools: ["coco", "claude"],
};
export default async function workflow() {
  const result = await agent("work", {
    tool: "coco", model: "invented", timeout: 120
  });
  if (result && result.error) return result;
  return result;
}
"""
    script_path = tmp_path / "generated-workflow-binding.js"
    script_path.write_text(script_text, encoding="utf-8")

    orchestrator = _binding("coco", "selected-coco-model", selection_key="orch")
    reviewer = _binding("claude", "selected-claude-model", selection_key="review")
    engine = WorkflowEngine(chat_id="chat-1", root_path=str(tmp_path))
    engine._project = WorkflowProject(
        status=WorkflowStatus.AWAITING_CONFIRM,
        pending=PendingConfirmation(
            script_path=str(script_path),
            script_hash=hashlib.sha256(script_text.encode()).hexdigest(),
            requirement="implement the binding contract",
            meta={"name": "handler-binding-contract", "tools": ["coco", "claude"]},
            initiator_user_id="user-1",
            engine_session_key="session-1",
            selected_tools=["coco", "claude"],
            orchestrator_binding=orchestrator,
            review_agents=[reviewer],
            auto_reviewer=False,
        ),
    )
    engine.execute_workflow = MagicMock()

    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.ctx.workflow_engine_manager.get.return_value = engine
    handler.ctx.progress_reporter = MagicMock()
    handler._resolve_project_from_id = MagicMock(
        return_value=SimpleNamespace(
            project_id="project-1",
            project_name="project",
            root_path=str(tmp_path),
        )
    )
    handler._get_root_path = MagicMock(return_value=str(tmp_path))
    handler._reply_workflow_error = MagicMock()
    handler._resolve_origin = MagicMock(return_value="topic-root-1")
    handler._show_initial_workflow_progress_card = MagicMock(return_value="progress-1")
    handler._build_workflow_callbacks = MagicMock(return_value=MagicMock())
    handler.get_engine_name = MagicMock(return_value="Coco")
    handler.ensure_request_id = MagicMock(return_value="request-1")
    handler._safe_execute_engine = MagicMock(
        side_effect=lambda *, executor_func, **_kwargs: executor_func()
    )
    handler._submit_engine_task = MagicMock(
        side_effect=lambda run_fn, *_args, **_kwargs: run_fn()
    )

    before = time.monotonic()
    with patch("src.thread.get_current_sender_id", return_value="user-1"):
        handler.handle_workflow_confirm_start(
            "confirm-1",
            "chat-1",
            "project-1",
            {
                "action": "workflow_confirm_start",
                "engine_session_key": "session-1",
            },
        )

    call = engine.execute_workflow.call_args
    assert call is not None
    assert "requirement" not in call.kwargs
    assert "selected_tools" not in call.kwargs
    assert "initiator_user_id" not in call.kwargs
    run_spec = call.kwargs["run_spec"]
    assert isinstance(run_spec, WorkflowRunSpec)
    assert run_spec.orchestrator == orchestrator
    assert run_spec.reviewers == (reviewer,)
    assert dict(run_spec.tool_model_map) == {
        "coco": "selected-coco-model",
        "claude": "selected-claude-model",
    }
    assert run_spec.task == "implement the binding contract"
    assert run_spec.chat_id == "chat-1"
    assert run_spec.topic_id == "topic-root-1"
    assert run_spec.budget > 0
    assert run_spec.deadline is None or run_spec.deadline >= before
    assert run_spec.auto_reviewer is False


def test_run_spec_budget_and_deadline_reach_the_execution_gate(tmp_path) -> None:
    deadline = time.monotonic() + 60
    spec = replace(_run_spec(auto_reviewer=True), budget=1, deadline=deadline)
    executor_deadlines: list[float | None] = []
    bridge_results: dict[str, AgentCallResult] = {}

    class BudgetBridge:
        def __init__(self, *args, on_agent_call, workflow_deadline_monotonic, **kwargs):
            assert workflow_deadline_monotonic == deadline
            self.on_agent_call = on_agent_call

        @staticmethod
        def check_node_available() -> bool:
            return True

        def start(self) -> None:
            return None

        def run(self) -> str:
            bridge_results["first"] = self.on_agent_call(
                AgentCallParams(prompt="first", tool="coco")
            )
            bridge_results["second"] = self.on_agent_call(
                AgentCallParams(prompt="second", tool="coco")
            )
            return json.dumps({"output": bridge_results["first"].output})

        def stop(self) -> None:
            return None

    def fake_execute(_executor, params, *, deadline_monotonic=None, **_kwargs):
        executor_deadlines.append(deadline_monotonic)
        return AgentCallResult(
            output="first completed",
            tool=params.tool,
            model=params.model,
        )

    engine = WorkflowEngine(chat_id="chat-1", root_path=str(tmp_path))
    with (
        patch("src.workflow_engine.engine.RuntimeBridge", BudgetBridge),
        patch.object(AgentExecutor, "execute", fake_execute),
    ):
        project = engine.execute_workflow(
            script_path=_script(tmp_path),
            run_spec=spec,
        )

    assert project.status == WorkflowStatus.COMPLETED
    assert executor_deadlines == [deadline]
    assert bridge_results["second"].error == "Agent call limit exceeded (1)"


def test_expired_run_spec_deadline_fails_closed_before_backend_call(tmp_path) -> None:
    spec = replace(
        _run_spec(auto_reviewer=True),
        deadline=time.monotonic() - 1,
    )

    class DeadlineBridge:
        def __init__(self, *args, on_agent_call, **kwargs):
            self.on_agent_call = on_agent_call

        @staticmethod
        def check_node_available() -> bool:
            return True

        def start(self) -> None:
            return None

        def run(self) -> str:
            result = self.on_agent_call(
                AgentCallParams(prompt="too late", tool="coco")
            )
            return json.dumps({"error": result.error})

        def stop(self) -> None:
            return None

    engine = WorkflowEngine(chat_id="chat-1", root_path=str(tmp_path))
    with (
        patch("src.workflow_engine.engine.RuntimeBridge", DeadlineBridge),
        patch.object(
            AgentExecutor,
            "execute",
            side_effect=AssertionError("expired run reached backend"),
        ),
    ):
        project = engine.execute_workflow(
            script_path=_script(tmp_path),
            run_spec=spec,
        )

    assert project.status == WorkflowStatus.FAILED
    assert "deadline" in (project.error or "").lower()
