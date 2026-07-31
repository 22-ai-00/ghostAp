"""Contracts for explicit Workflow Reviewer execution and evidence."""

from __future__ import annotations

import threading
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
    AgentStatus,
    WorkflowProject,
    WorkflowStatus,
)
from src.workflow_engine.reporting import build_workflow_report_markdown
from src.workflow_engine.run_spec import WorkflowRunSpec
from src.workflow_engine.script_gen import build_script_gen_prompt


def _binding(tool: str, model: str, key: str) -> ReviewAgentBinding:
    return ReviewAgentBinding(
        provider="cli",
        tool_name=tool,
        display_name=tool.title(),
        agent_type=tool,
        model_name=model,
        model_display_name=model,
        selection_key=key,
        use_default_model=False,
    )


def _script(tmp_path) -> str:
    path = tmp_path / "review-contract.js"
    path.write_text(
        """
export const meta = {
  name: "review-contract",
  description: "review contract",
  tools: ["coco", "claude", "codex"],
};
export default async function workflow() { return "deliverable"; }
""",
        encoding="utf-8",
    )
    return str(path)


def _spec(*, auto: bool, reviewers: tuple[ReviewAgentBinding, ...]) -> WorkflowRunSpec:
    orchestrator = _binding("coco", "orch-model", "orch")
    return WorkflowRunSpec(
        orchestrator=orchestrator,
        reviewers=reviewers,
        tool_model_map={
            orchestrator.tool_name: orchestrator.model_name,
            **{reviewer.tool_name: reviewer.model_name for reviewer in reviewers},
        },
        task="produce a reviewed deliverable",
        chat_id="chat-1",
        topic_id="topic-1",
        budget=20,
        deadline=None,
        auto_reviewer=auto,
        initiator_user_id="user-1",
        allowed_tools=tuple(
            dict.fromkeys((orchestrator.tool_name, *(reviewer.tool_name for reviewer in reviewers)))
        ),
    )


def _card_text(value) -> str:
    if isinstance(value, dict):
        return "\n".join(_card_text(item) for item in value.values())
    if isinstance(value, list):
        return "\n".join(_card_text(item) for item in value)
    return str(value)


class _ResultBridge:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def check_node_available() -> bool:
        return True

    def start(self) -> None:
        return None

    def run(self) -> str:
        return "main deliverable"

    def stop(self) -> None:
        return None


def test_auto_reviewer_is_explicit_in_run_spec() -> None:
    spec = _spec(auto=True, reviewers=())

    assert spec.auto_reviewer is True
    assert spec.reviewers == ()
    assert spec.to_dict()["auto_reviewer"] is True

    with pytest.raises(ValueError, match="review"):
        _spec(auto=False, reviewers=())

    with pytest.raises(ValueError, match="Auto"):
        _spec(
            auto=True,
            reviewers=(_binding("claude", "review-model", "review"),),
        )


def test_confirm_card_renders_auto_from_explicit_flag() -> None:
    handler = WorkflowHandler.__new__(WorkflowHandler)
    card = handler._build_confirm_card(
        meta={"name": "auto-review", "tools": ["coco"], "phases": []},
        requirement="small safe task",
        engine_session_key="session-1",
        chat_id="chat-1",
        project_id="project-1",
        selected_tools=["coco"],
        orchestrator_binding=_binding("coco", "orch-model", "orch"),
        review_agents=None,
        auto_reviewer=True,
    )

    text = _card_text(card)
    assert "Auto" in text
    assert "无独立 Reviewer" in text
    assert "编排器负责最终检查" in text
    assert "自评审" not in text


def test_confirm_summary_renders_exact_explicit_reviewer_binding() -> None:
    handler = WorkflowHandler.__new__(WorkflowHandler)
    reviewer = _binding("claude", "claude-review", "review-1")
    card = handler._build_confirm_card(
        meta={"name": "explicit-review", "tools": ["coco", "claude"], "phases": []},
        requirement="high risk task",
        engine_session_key="session-1",
        chat_id="chat-1",
        project_id="project-1",
        selected_tools=["coco", "claude"],
        orchestrator_binding=_binding("coco", "orch-model", "orch"),
        review_agents=[reviewer],
        auto_reviewer=False,
    )

    text = _card_text(card)
    assert "coco" in text
    assert "orch-model" in text
    assert "claude" in text
    assert "claude-review" in text
    assert "Auto" not in text


def test_script_generation_prompt_states_runtime_reviewer_commitment() -> None:
    reviewer = _binding("claude", "claude-review", "review-1")
    prompt = build_script_gen_prompt(
        requirement="high risk task",
        available_tools=["coco", "claude"],
        orchestrator_agent="coco",
        orchestrator_binding=_binding("coco", "orch-model", "orch"),
        review_agents=[reviewer],
        auto_reviewer=False,
    )

    assert "独立 Reviewer" in prompt
    assert "脚本完成后" in prompt
    assert "claude-review" in prompt

    auto_prompt = build_script_gen_prompt(
        requirement="small task",
        available_tools=["coco"],
        orchestrator_agent="coco",
        orchestrator_binding=_binding("coco", "orch-model", "orch"),
        review_agents=[],
        auto_reviewer=True,
    )
    assert "Auto" in auto_prompt
    assert "不承诺独立 Reviewer" in auto_prompt


def test_agent_executor_preserves_backend_stop_reason(tmp_path) -> None:
    """The Reviewer gate must receive the real ACP terminal reason."""
    session = MagicMock()
    session.send_prompt.return_value = SimpleNamespace(
        text="partial review",
        output_tokens=3,
        stop_reason="max_tokens",
    )
    future = MagicMock()
    future.result.return_value = session

    executor = AgentExecutor(cwd=str(tmp_path), cancel_event=threading.Event())
    executor._session_pool.shutdown(wait=False, cancel_futures=True)
    executor._session_pool = MagicMock()
    executor._session_pool.submit.return_value = future
    try:
        result = executor.execute(AgentCallParams(prompt="review", tool="claude"))
    finally:
        executor.shutdown()

    assert result.stop_reason == "max_tokens"


def test_each_explicit_reviewer_is_independently_invoked(tmp_path) -> None:
    reviewers = (
        _binding("claude", "claude-review", "review-1"),
        _binding("codex", "codex-review", "review-2"),
    )
    spec = _spec(auto=False, reviewers=reviewers)
    calls: list[AgentCallParams] = []

    def fake_execute(
        _executor: AgentExecutor,
        params: AgentCallParams,
        **_kwargs,
    ) -> AgentCallResult:
        calls.append(params.model_copy(deep=True))
        return AgentCallResult(
            output=f"independent evidence from {params.tool}",
            tool=params.tool,
            model=params.model,
            token_usage=7,
            stop_reason="end_turn",
        )

    engine = WorkflowEngine(chat_id="chat-1", root_path=str(tmp_path))
    with (
        patch("src.workflow_engine.engine.RuntimeBridge", _ResultBridge),
        patch.object(AgentExecutor, "execute", fake_execute),
    ):
        project = engine.execute_workflow(
            script_path=_script(tmp_path),
            run_spec=spec,
        )

    assert project.status == WorkflowStatus.COMPLETED
    assert [(call.tool, call.model) for call in calls] == [
        ("claude", "claude-review"),
        ("codex", "codex-review"),
    ]
    assert all(call.role == "workflow_reviewer" for call in calls)
    assert all("main deliverable" in call.prompt for call in calls)
    assert [(item.tool, item.model, item.status) for item in project.reviewer_evidence] == [
        ("claude", "claude-review", "completed"),
        ("codex", "codex-review", "completed"),
    ]
    assert all(item.cached is False for item in project.reviewer_evidence)

    reviewer_trace = [
        agent
        for phase in project.phases
        for agent in phase.agents
        if agent.role == "workflow_reviewer"
    ]
    assert [(agent.tool, agent.model) for agent in reviewer_trace] == [
        ("claude", "claude-review"),
        ("codex", "codex-review"),
    ]
    report = build_workflow_report_markdown(project)
    assert "独立评审证据" in report
    assert "claude/claude-review" in report
    assert "codex/codex-review" in report

    restored = WorkflowProject.from_dict(project.to_dict())
    assert restored.reviewer_evidence == project.reviewer_evidence


def test_failed_explicit_reviewer_cannot_publish_reviewed_completion(tmp_path) -> None:
    reviewer = _binding("claude", "claude-review", "review-1")
    spec = _spec(auto=False, reviewers=(reviewer,))

    def fail_review(
        _executor: AgentExecutor,
        params: AgentCallParams,
        **_kwargs,
    ) -> AgentCallResult:
        return AgentCallResult(
            error="review backend unavailable",
            tool=params.tool,
            model=params.model,
        )

    engine = WorkflowEngine(chat_id="chat-1", root_path=str(tmp_path))
    with (
        patch("src.workflow_engine.engine.RuntimeBridge", _ResultBridge),
        patch.object(AgentExecutor, "execute", fail_review),
    ):
        project = engine.execute_workflow(
            script_path=_script(tmp_path),
            run_spec=spec,
        )

    assert project.status == WorkflowStatus.FAILED
    assert "review" in (project.error or "").lower()
    assert len(project.reviewer_evidence) == 1
    assert project.reviewer_evidence[0].status == "failed"
    assert project.reviewer_evidence[0].error == "review backend unavailable"


@pytest.mark.parametrize(
    ("stop_reason", "output"),
    [
        ("max_tokens", "partial review"),
        ("cancelled", "partial review"),
        (None, "review without terminal proof"),
        ("end_turn", ""),
    ],
)
def test_incomplete_or_empty_reviewer_cannot_complete_workflow(
    tmp_path,
    stop_reason: str | None,
    output: str,
) -> None:
    reviewer = _binding("claude", "claude-review", "review-1")
    spec = _spec(auto=False, reviewers=(reviewer,))

    def incomplete_review(
        _executor: AgentExecutor,
        params: AgentCallParams,
        **_kwargs,
    ) -> AgentCallResult:
        return AgentCallResult(
            output=output,
            stop_reason=stop_reason,
            tool=params.tool,
            model=params.model,
        )

    engine = WorkflowEngine(chat_id="chat-1", root_path=str(tmp_path))
    with (
        patch("src.workflow_engine.engine.RuntimeBridge", _ResultBridge),
        patch.object(AgentExecutor, "execute", incomplete_review),
    ):
        project = engine.execute_workflow(
            script_path=_script(tmp_path),
            run_spec=spec,
        )

    assert project.status == WorkflowStatus.FAILED
    assert len(project.reviewer_evidence) == 1
    evidence = project.reviewer_evidence[0]
    assert evidence.status == "failed"
    assert evidence.stop_reason == stop_reason
    assert evidence.error
    reviewer_agents = [
        agent
        for phase in project.phases
        for agent in phase.agents
        if agent.role == "workflow_reviewer"
    ]
    assert len(reviewer_agents) == 1
    assert reviewer_agents[0].status == AgentStatus.FAILED
