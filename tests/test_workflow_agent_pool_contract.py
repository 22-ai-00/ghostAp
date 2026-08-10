"""Strict contracts for Workflow's confirmed tool/model Agent pool."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest

from src.spec_engine.models import ReviewAgentBinding
from src.workflow_engine.bridge import RuntimeBridge
from src.workflow_engine.engine import WorkflowEngine
from src.workflow_engine.models import (
    AgentCallParams,
    AgentCallResult,
    WorkflowAgentBinding,
    WorkflowProject,
)
from src.workflow_engine.run_spec import WorkflowRunSpec
from src.workflow_engine.script_gen import (
    build_script_gen_prompt,
    validate_generated_script,
)


def _agent(
    agent_id: str,
    tool: str,
    model: str | None,
    *,
    display_name: str | None = None,
) -> WorkflowAgentBinding:
    return WorkflowAgentBinding(
        agent_id=agent_id,
        tool_name=tool,
        model_name=model,
        display_name=display_name or agent_id,
    )


def _legacy_binding(tool: str = "codex", model: str | None = "legacy-model") -> ReviewAgentBinding:
    return ReviewAgentBinding(
        provider="workflow",
        tool_name=tool,
        display_name=tool,
        agent_type=tool,
        model_name=model,
        model_display_name=model,
        selection_key=f"{tool}:{model or 'default'}",
        use_default_model=model is None,
    )


def _strict_spec(
    *agent_pool: WorkflowAgentBinding,
    orchestrator_agent_id: str | None = None,
) -> WorkflowRunSpec:
    first = agent_pool[0]
    return WorkflowRunSpec(
        orchestrator=_legacy_binding(first.tool_name, first.model_name),
        reviewers=(),
        tool_model_map={},
        task="exercise the confirmed Agent pool",
        chat_id="chat-agent-pool",
        topic_id="topic-agent-pool",
        budget=20,
        deadline=None,
        auto_reviewer=True,
        allowed_tools=tuple(dict.fromkeys(agent.tool_name for agent in agent_pool)),
        agent_pool=agent_pool,
        orchestrator_agent_id=orchestrator_agent_id,
    )


def _script(body: str, *, agent_plan: str) -> str:
    return f"""
export const meta = {{
  name: "agent-pool-contract",
  description: "strict Agent pool contract",
  phases: [{{ title: "Execute", detail: "Use confirmed bindings" }}],
  maxConcurrent: 3,
  tools: ["codex"],
  patterns: ["fanout"],
  agentPlan: {agent_plan},
}};
export default async function main() {{
  {body}
}}
"""


def test_binding_is_frozen_and_duplicate_tool_model_is_rejected() -> None:
    first = _agent("codex-high", "codex", "gpt-high")
    with pytest.raises(FrozenInstanceError):
        first.model_name = "tampered"  # type: ignore[misc]

    with pytest.raises(ValueError, match="duplicate.*tool.*model"):
        _strict_spec(
            first,
            _agent("codex-high-copy", "codex", "gpt-high"),
        )


def test_same_tool_with_different_models_is_two_callable_agents() -> None:
    high = _agent("codex-high", "codex", "gpt-high")
    fast = _agent("codex-fast", "codex", "gpt-fast")

    spec = _strict_spec(high, fast)

    assert spec.orchestrator_agent_id == "codex-high"
    assert spec.agent_binding("codex-high") == high
    assert spec.agent_binding("codex-fast") == fast


def test_new_run_spec_requires_pool_and_legacy_is_explicit() -> None:
    kwargs = dict(
        orchestrator=_legacy_binding(),
        reviewers=(),
        tool_model_map={"codex": "legacy-model"},
        task="legacy replay",
        chat_id="legacy-chat",
        topic_id=None,
        budget=2,
        deadline=None,
        auto_reviewer=True,
        allowed_tools=("codex",),
    )

    with pytest.raises(ValueError, match="agent_pool"):
        WorkflowRunSpec(**kwargs)

    legacy = WorkflowRunSpec.from_legacy_replay(**kwargs)
    assert legacy.legacy_replay is True
    assert legacy.agent_pool == ()

    with pytest.raises(ValueError, match="unique tool"):
        WorkflowRunSpec.from_legacy_replay(
            **{
                **kwargs,
                "reviewers": (_legacy_binding("codex", "other-model"),),
                "auto_reviewer": False,
            }
        )

    replayed = WorkflowRunSpec.from_dict(
        {
            **kwargs,
            "orchestrator": kwargs["orchestrator"].to_dict(),
            "reviewers": [],
            "allowed_tools": ["codex"],
        }
    )
    assert replayed.legacy_replay is True


def test_unknown_orchestrator_agent_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="orchestrator_agent_id"):
        _strict_spec(
            _agent("codex-high", "codex", "gpt-high"),
            orchestrator_agent_id="outside-pool",
        )


def test_engine_requires_agent_id_and_blocks_tool_model_spoof(tmp_path) -> None:
    high = _agent("codex-high", "codex", "gpt-high")
    fast = _agent("codex-fast", "codex", "gpt-fast")
    engine = WorkflowEngine(chat_id="chat-agent-pool", root_path=str(tmp_path))
    engine._run_spec = _strict_spec(high, fast)
    engine._project = WorkflowProject(selected_tools=["codex"])
    engine._executor = MagicMock()
    engine._executor.execute.side_effect = lambda params, **_kwargs: AgentCallResult(
        output="ok",
        stop_reason="end_turn",
        tool=params.tool,
        model=params.model,
        agent_id=params.agent_id,
    )

    missing = engine._handle_agent_call(AgentCallParams(prompt="missing binding"))
    unknown = engine._handle_agent_call(
        AgentCallParams(prompt="unknown binding", agent_id="outside-pool")
    )
    resolved = engine._handle_agent_call(
        AgentCallParams(
            prompt="bound task",
            agent_id="codex-fast",
            tool="claude",
            model="script-spoofed-model",
        )
    )

    assert "agent_id" in (missing.error or "")
    assert "outside-pool" in (unknown.error or "")
    assert resolved.agent_id == "codex-fast"
    assert resolved.tool == "codex"
    assert resolved.model == "gpt-fast"
    backend_params = engine._executor.execute.call_args.args[0]
    assert (backend_params.agent_id, backend_params.tool, backend_params.model) == (
        "codex-fast",
        "codex",
        "gpt-fast",
    )
    assert engine._executor.execute.call_count == 1


def test_script_prompt_exposes_only_confirmed_agent_pool() -> None:
    pool = (
        _agent("planner", "traex", "seed-pro", display_name="Planner"),
        _agent("reviewer", "codex", "gpt-high", display_name="Reviewer"),
    )

    prompt = build_script_gen_prompt(
        requirement="plan and verify",
        available_tools={"claude": "must stay hidden"},
        agent_pool=pool,
        orchestrator_agent_id="planner",
    )

    assert "planner" in prompt and "traex" in prompt and "seed-pro" in prompt
    assert "reviewer" in prompt and "codex" in prompt and "gpt-high" in prompt
    assert "claude" not in prompt
    assert "agentId" in prompt


def test_validator_requires_pool_binding_for_direct_and_dynamic_workers() -> None:
    pool = (
        _agent("planner", "traex", "seed-pro"),
        _agent("worker-fast", "codex", "gpt-fast"),
        _agent("worker-deep", "codex", "gpt-high"),
    )
    plan = """[
      { node: "plan", role: "planner", agentId: "planner" },
      { node: "implementation", role: "worker", runtime: true,
        candidateAgentIds: ["worker-fast", "worker-deep"] }
    ]"""
    valid = _script(
        """
const results = await fanout("task", [
  { prompt: "fast", agentId: "worker-fast", label: "fast", timeout: 120 },
  { prompt: "deep", agentId: "worker-deep", label: "deep", timeout: 180 },
]);
if (results && results.error) return { error: results.error };
return results;
""",
        agent_plan=plan,
    )
    missing = valid.replace('agentId: "worker-deep", ', "")
    outside = valid.replace('agentId: "worker-deep"', 'agentId: "outside-pool"')

    assert validate_generated_script(valid, agent_pool=pool)[0] is True
    missing_ok, missing_errors = validate_generated_script(missing, agent_pool=pool)
    outside_ok, outside_errors = validate_generated_script(outside, agent_pool=pool)

    assert missing_ok is False
    assert any("agentId" in error for error in missing_errors)
    assert outside_ok is False
    assert any("outside-pool" in error for error in outside_errors)


def test_agent_plan_rejects_ids_outside_pool() -> None:
    pool = (_agent("planner", "traex", "seed-pro"),)
    script = _script(
        """
const result = await agent({
  prompt: "plan", agentId: "planner", label: "plan", timeout: 120
});
if (result && result.error) return { error: result.error };
return result;
""",
        agent_plan='[{ node: "plan", role: "planner", agentId: "outside-pool" }]',
    )

    valid, errors = validate_generated_script(script, agent_pool=pool)

    assert valid is False
    assert any("agentPlan" in error and "outside-pool" in error for error in errors)


@pytest.mark.skipif(
    not RuntimeBridge.check_node_available(),
    reason="Node.js not available or version too old",
)
def test_real_runtime_dynamic_workers_forward_agent_ids(tmp_path) -> None:
    from tests.test_workflow_bridge_transport import _run_real_node_workflow

    observed: list[str | None] = []

    def on_agent_call(params: AgentCallParams, **_kwargs) -> AgentCallResult:
        observed.append(params.agent_id)
        return AgentCallResult(
            output=params.agent_id,
            agent_id=params.agent_id,
            tool=params.tool,
            model=params.model,
        )

    _run_real_node_workflow(
        tmp_path,
        """
export const meta = {
  name: 'pool-workers',
  description: 'pool workers',
  phases: [{ title: 'Fanout', detail: 'Dispatch confirmed pool workers' }],
};
export default async function main() {
  return fanout('task', [
    { prompt: 'fast', agentId: 'codex-fast', label: 'fast', timeout: 1 },
    { prompt: 'deep', agentId: 'codex-deep', label: 'deep', timeout: 1 },
  ], { synthesize: false });
}
""",
        on_agent_call,
    )

    assert observed == ["codex-fast", "codex-deep"]
