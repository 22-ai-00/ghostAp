from __future__ import annotations

from typing import Any

import pytest

from src.workflow_engine.agent_pool import WorkflowAgentBinding
from src.workflow_engine.engine import WorkflowEngine
from src.workflow_engine.models import (
    AgentCallParams,
    AgentCallResult,
    AgentStatus,
    ReviewAgentBinding,
    WorkflowProject,
    WorkflowStatus,
)
from src.workflow_engine.run_spec import WorkflowRunSpec
from src.workflow_engine.script_gen import (
    generate_simple_script,
    validate_generated_script,
)
from src.workflow_engine.state_manager import WorkflowStateManager
from tests.test_workflow_bridge_transport import _run_real_node_workflow


def _review_binding(tool: str, model: str | None) -> ReviewAgentBinding:
    return ReviewAgentBinding(
        provider=tool,
        tool_name=tool,
        display_name=tool.title(),
        agent_type=tool,
        model_name=model,
        model_display_name=model,
        selection_key=f"{tool}:{model or 'default'}",
        use_default_model=model is None,
    )


def _legacy_spec(
    *,
    allowed_tools: tuple[str, ...] = ("codex",),
    tool_model_map: dict[str, str | None] | None = None,
) -> WorkflowRunSpec:
    return WorkflowRunSpec.from_legacy_replay(
        orchestrator=_review_binding("codex", "frozen-model"),
        reviewers=(),
        tool_model_map=(
            {"codex": "frozen-model"}
            if tool_model_map is None
            else tool_model_map
        ),
        task="legacy task",
        chat_id="chat",
        topic_id=None,
        budget=10,
        deadline=None,
        auto_reviewer=True,
        allowed_tools=allowed_tools,
    )


def _pool() -> tuple[WorkflowAgentBinding, ...]:
    return (
        WorkflowAgentBinding(
            agent_id="codex-fast",
            tool_name="codex",
            model_name="fast-model",
            display_name="Codex Fast",
        ),
        WorkflowAgentBinding(
            agent_id="codex-deep",
            tool_name="codex",
            model_name="deep-model",
            display_name="Codex Deep",
        ),
    )


def _script(*, tools: str, agent_plan: str, body: str) -> str:
    return f"""
export const meta = {{
  name: 'review-contract',
  description: 'review contract',
  phases: [{{ title: 'Work', detail: 'Run the work' }}],
  tools: {tools},
  agentPlan: {agent_plan},
}};
export default async function main() {{
  {body}
}}
"""


class _RecordingExecutor:
    def __init__(self) -> None:
        self.calls: list[AgentCallParams] = []

    def execute(self, params: AgentCallParams, *args: Any, **kwargs: Any) -> AgentCallResult:
        self.calls.append(params)
        return AgentCallResult(output="ok")


def _legacy_engine(tmp_path, spec: WorkflowRunSpec) -> tuple[WorkflowEngine, _RecordingExecutor]:
    engine = WorkflowEngine("chat", str(tmp_path), agent_type="codex")
    project = WorkflowProject(name="legacy", status=WorkflowStatus.RUNNING)
    engine._project = project
    engine._state_manager = WorkflowStateManager(project)
    engine._run_spec = spec
    recorder = _RecordingExecutor()
    engine._executor = recorder
    return engine, recorder


def test_legacy_replay_requires_an_exact_frozen_binding_for_every_allowed_tool() -> None:
    with pytest.raises(ValueError, match="allowed|binding|model"):
        _legacy_spec(
            allowed_tools=("codex", "gemini"),
            tool_model_map={"codex": "frozen-model"},
        )

    with pytest.raises(ValueError, match="allowed|binding|extra"):
        _legacy_spec(
            allowed_tools=("codex",),
            tool_model_map={"codex": "frozen-model", "gemini": "other-model"},
        )


def test_legacy_replay_ignores_script_model_and_rejects_extra_tool(tmp_path) -> None:
    engine, recorder = _legacy_engine(tmp_path, _legacy_spec())

    result = engine._handle_agent_call(
        AgentCallParams(
            prompt="work",
            tool="codex",
            model="script-spoof",
            label="worker",
            phase="work",
        ),
        allow_cache=False,
    )

    assert result.error is None
    assert recorder.calls[0].model == "frozen-model"

    rejected = engine._handle_agent_call(
        AgentCallParams(
            prompt="work",
            tool="gemini",
            model="script-model",
            label="extra",
            phase="work",
        ),
        allow_cache=False,
    )
    assert rejected.error
    assert len(recorder.calls) == 1


def test_from_dict_distinguishes_missing_pool_from_explicit_empty_or_invalid() -> None:
    old_payload = _legacy_spec().to_dict()
    assert "agent_pool" not in old_payload
    old_payload.pop("legacy_replay")
    assert WorkflowRunSpec.from_dict(old_payload).legacy_replay is True

    with pytest.raises(ValueError, match="agent_pool"):
        WorkflowRunSpec.from_dict(
            {**old_payload, "agent_pool": [], "legacy_replay": True}
        )

    with pytest.raises(ValueError, match="legacy|agent_pool"):
        WorkflowRunSpec.from_dict({**old_payload, "legacy_replay": False})

    with pytest.raises((TypeError, ValueError), match="agent_pool|agent_id"):
        WorkflowRunSpec.from_dict(
            {
                **old_payload,
                "agent_pool": [{"agent_id": "", "tool_name": "codex"}],
                "legacy_replay": False,
            }
        )


def test_abort_with_unknown_request_id_does_not_fallback_to_raw_label(tmp_path) -> None:
    engine = WorkflowEngine("chat", str(tmp_path), agent_type="codex")
    project = WorkflowProject(name="wf", status=WorkflowStatus.RUNNING)
    manager = WorkflowStateManager(project)
    engine._project = project
    engine._state_manager = manager

    first = manager.on_agent_started("worker", "codex", "phase")
    second = manager.on_agent_started("worker", "codex", "phase")
    assert (first, second) == ("worker", "worker #2")
    engine._request_to_label["known-second"] = second

    engine._handle_agent_aborted(
        "worker",
        "race loser",
        request_id="unknown-request",
    )

    agents = manager.snapshot().phases[0].agents
    assert [agent.status for agent in agents] == [
        AgentStatus.RUNNING,
        AgentStatus.RUNNING,
    ]


def test_state_snapshot_preserves_agent_id() -> None:
    manager = WorkflowStateManager(
        WorkflowProject(name="wf", status=WorkflowStatus.RUNNING)
    )
    manager.on_agent_started(
        "worker",
        "codex",
        "phase",
        agent_id="codex-fast",
    )

    snapshot = manager.snapshot()

    assert snapshot.phases[0].agents[0].agent_id == "codex-fast"


def test_generate_object_worker_inherits_primitive_agent_id(tmp_path) -> None:
    calls: list[AgentCallParams] = []
    script = """
export const meta = {
  name: 'generate-fallback',
  description: 'generate fallback',
  phases: [{ title: 'Generate', detail: 'Generate candidates' }],
};
export default async function main() {
  return generate(
    2,
    { prompt: 'candidate', label: 'generator', timeout: 1 },
    async () => true,
    { agentId: 'codex-fast' },
  );
}
"""

    def on_agent_call(
        request: AgentCallParams,
        **_: Any,
    ) -> AgentCallResult:
        calls.append(request)
        return AgentCallResult(output="candidate", stop_reason="end_turn")

    _run_real_node_workflow(tmp_path, script, on_agent_call)

    assert calls
    assert {call.agent_id for call in calls} == {"codex-fast"}


def test_meta_tools_non_string_returns_structured_validation_error() -> None:
    script = _script(
        tools="['codex', { name: 'spoof' }]",
        agent_plan="[{ node: 'main', role: 'worker', agentId: 'codex-fast' }]",
        body="""
const result = await agent({
  prompt: 'work', agentId: 'codex-fast', timeout: 1,
});
if (result.error) throw new Error(result.error);
return result;
""",
    )

    valid, errors = validate_generated_script(script, agent_pool=_pool())

    assert valid is False
    assert any("meta.tools" in error for error in errors)


@pytest.mark.parametrize(
    "descriptor",
    [
        "{ prompt: \"agentId: 'codex-fast'\", timeout: 1 }",
        "{ prompt: 'work', schema: { description: \"agentId: 'codex-fast'\" }, timeout: 1 }",
    ],
)
def test_nested_prompt_or_schema_text_cannot_satisfy_static_agent_id(
    descriptor: str,
) -> None:
    script = _script(
        tools="['codex']",
        agent_plan="[{ node: 'main', role: 'worker', agentId: 'codex-fast' }]",
        body=f"""
const result = await agent({descriptor});
if (result.error) throw new Error(result.error);
return result;
""",
    )

    valid, errors = validate_generated_script(script, agent_pool=_pool())

    assert valid is False
    assert any("agentId" in error for error in errors)


@pytest.mark.parametrize(
    "agent_plan",
    [
        "[{ node: '', role: 'worker', agentId: 'codex-fast' }]",
        "[{ node: 'main', role: '  ', agentId: 'codex-fast' }]",
        "[{ node: 'fanout', role: 'worker', runtime: true, candidateAgentIds: [] }]",
        "[{ node: 'fanout', role: 'worker', runtime: true, candidateAgentIds: ['codex-fast', 'codex-fast'] }]",
        "[{ node: 'fanout', role: 'worker', candidateAgentIds: ['codex-fast'] }]",
        "[{ node: 'main', role: 'worker', agentId: 'codex-fast', candidateAgentIds: ['codex-deep'] }]",
        "[{ node: 'main', role: 'worker', agentId: 'outside' }]",
    ],
)
def test_agent_plan_rejects_incomplete_ambiguous_or_outside_entries(
    agent_plan: str,
) -> None:
    script = _script(
        tools="['codex']",
        agent_plan=agent_plan,
        body="""
const result = await agent({
  prompt: 'work', agentId: 'codex-fast', timeout: 1,
});
if (result.error) throw new Error(result.error);
return result;
""",
    )

    valid, errors = validate_generated_script(script, agent_pool=_pool())

    assert valid is False
    assert errors
    assert any("agentPlan" in error for error in errors)


def test_simple_script_strict_path_emits_agent_id_and_agent_plan() -> None:
    script = generate_simple_script(
        "implement task",
        agent_pool=_pool(),
        orchestrator_agent_id="codex-deep",
    )

    assert 'agentId: "codex-deep"' in script
    assert "agentPlan" in script
    valid, errors = validate_generated_script(script, agent_pool=_pool())
    assert valid, errors


def test_simple_script_requires_strict_pool_or_explicit_legacy_path() -> None:
    with pytest.raises(ValueError, match="agent_pool|legacy"):
        generate_simple_script("implement task")

    legacy_script = generate_simple_script(
        "implement task",
        selected_tools=["codex"],
        tool_model_map={"codex": "legacy-model"},
        legacy_replay=True,
    )
    assert 'tool: "codex"' in legacy_script
    assert 'model: "legacy-model"' in legacy_script
