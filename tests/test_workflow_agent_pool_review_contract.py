from __future__ import annotations

from typing import Any

import pytest

from src.workflow_engine.agent_pool import WorkflowAgentBinding
from src.workflow_engine.engine import WorkflowEngine
from src.workflow_engine.models import (
    AgentCallParams,
    AgentCallResult,
    AgentStatus,
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


def test_from_dict_rejects_missing_empty_or_invalid_pool() -> None:
    strict_payload = WorkflowRunSpec(
        reviewers=(),
        tool_model_map={},
        task="strict task",
        chat_id="chat",
        topic_id=None,
        budget=4,
        deadline=None,
        auto_reviewer=True,
        orchestrator="codex",
        agent_pool=_pool(),
        orchestrator_agent_id="codex-fast",
    ).to_dict()
    no_pool = dict(strict_payload)
    no_pool.pop("agent_pool")

    with pytest.raises(ValueError, match="agent_pool"):
        WorkflowRunSpec.from_dict(no_pool)

    with pytest.raises(ValueError, match="agent_pool"):
        WorkflowRunSpec.from_dict({**no_pool, "agent_pool": []})

    with pytest.raises((TypeError, ValueError), match="agent_pool|agent_id"):
        WorkflowRunSpec.from_dict(
            {
                **no_pool,
                "agent_pool": [{"agent_id": "", "tool_name": "codex"}],
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


def test_simple_script_requires_strict_pool() -> None:
    with pytest.raises(ValueError, match="agent_pool|empty"):
        generate_simple_script("implement task")


def test_engine_rejects_outside_fallback_agent_ids_upfront(tmp_path) -> None:
    from src.workflow_engine.run_spec import _review_binding

    pool = _pool()
    run_spec = WorkflowRunSpec(
        orchestrator=_review_binding(pool[0]),
        reviewers=(),
        tool_model_map={},
        task="strict fallback validation",
        chat_id="chat",
        topic_id=None,
        budget=4,
        deadline=None,
        auto_reviewer=True,
        agent_pool=pool,
        orchestrator_agent_id="codex-fast",
    )
    engine = WorkflowEngine("chat", str(tmp_path), agent_type="codex")
    engine._run_spec = run_spec
    engine._executor = _RecordingExecutor()

    res = engine._handle_agent_call(
        AgentCallParams(prompt="do work", agent_id="codex-fast", label="w1"),
        fallback_agent_ids=["outside-pool"],
    )

    assert res.error and "outside confirmed agent_pool" in res.error
    assert engine._executor.calls == []


def test_engine_cross_member_fallback_skips_blacklisted_and_protects_reviewer_isolation(
    tmp_path,
) -> None:
    from unittest.mock import patch

    from src.acp.sync_adapter import ACPStartupError
    from src.workflow_engine.executor import AgentExecutor
    from src.workflow_engine.run_spec import _review_binding

    pool = (
        WorkflowAgentBinding(
            agent_id="A1",
            tool_name="traex",
            model_name="kimi-k2",
            display_name="Worker 1",
            role="worker",
        ),
        WorkflowAgentBinding(
            agent_id="A2",
            tool_name="codex",
            model_name="gpt-5.4",
            display_name="Worker 2",
            effort="high",
            role="worker",
        ),
        WorkflowAgentBinding(
            agent_id="A3",
            tool_name="claude",
            model_name="sonnet",
            display_name="Reviewer",
            effort="high",
            role="reviewer",
        ),
    )
    run_spec = WorkflowRunSpec(
        orchestrator=_review_binding(pool[2]),
        reviewers=(),
        tool_model_map={},
        task="fallback contract",
        chat_id="chat",
        topic_id=None,
        budget=6,
        deadline=None,
        auto_reviewer=True,
        agent_pool=pool,
        orchestrator_agent_id="A3",
    )
    engine = WorkflowEngine("chat", str(tmp_path), agent_type="traex")
    engine._run_spec = run_spec
    engine._project = WorkflowProject(name="wf", status=WorkflowStatus.RUNNING)
    engine._state_manager = WorkflowStateManager(engine._project)
    engine._executor = AgentExecutor(cwd=str(tmp_path), cancel_event=engine.cancel_event)

    seen_calls: list[tuple[str, str | None]] = []

    class OkSession:
        def send_prompt(self, _prompt, **_kwargs):
            return type("Resp", (), {"text": "ok", "output_tokens": 1, "stop_reason": "end_turn"})()

        def close(self):
            return None

    def fake_create_session(**kwargs):
        tool = kwargs.get("agent_type")
        model = kwargs.get("model_name")
        seen_calls.append((tool, model))
        if tool in {"traex", "claude"}:
            raise ACPStartupError(f"{tool} binary missing", agent_cmd=tool, agent_args=[], cwd=str(tmp_path))
        return OkSession()

    try:
        with patch("src.agent_session.factory.create_engine_session", side_effect=fake_create_session):
            # Explicit fallback A1 -> A2 succeeds and blacklists A1
            res1 = engine._handle_agent_call(
                AgentCallParams(prompt="w1", agent_id="A1", label="w1"),
                fallback_agent_ids=["A2"],
            )
            assert res1.error is None
            assert res1.agent_id == "A2"
            assert engine._executor.is_blacklisted(
                AgentCallParams(prompt="w1", agent_id="A1", tool="traex", model="kimi-k2", effective_model="kimi-k2")
            )

            # Second call fast-skips blacklisted A1 straight to A2
            before = len(seen_calls)
            res2 = engine._handle_agent_call(
                AgentCallParams(prompt="w2", agent_id="A1", label="w2"),
                fallback_agent_ids=["A2"],
            )
            assert res2.error is None
            assert res2.agent_id == "A2"
            assert seen_calls[before:] == [("codex", "gpt-5.4/high")]

            # Reviewer A3 startup failure must NOT auto-fallback onto worker A2
            res_rev = engine._handle_agent_call(
                AgentCallParams(prompt="review", agent_id="A3", role="reviewer", label="rev"),
            )
            assert res_rev.error is not None
            assert res_rev.agent_id == "A3"
    finally:
        engine._executor.shutdown(wait=True)


def test_engine_explicit_fallback_blocks_reviewer_falling_back_to_worker(
    tmp_path,
) -> None:
    from unittest.mock import patch

    from src.acp.sync_adapter import ACPStartupError
    from src.workflow_engine.executor import AgentExecutor
    from src.workflow_engine.run_spec import _review_binding

    pool = (
        WorkflowAgentBinding(
            agent_id="A1",
            tool_name="codex",
            model_name="gpt-5.4",
            display_name="Worker",
            role="worker",
        ),
        WorkflowAgentBinding(
            agent_id="A2",
            tool_name="claude",
            model_name="sonnet",
            display_name="Reviewer Primary",
            role="reviewer",
        ),
        WorkflowAgentBinding(
            agent_id="A3",
            tool_name="agy",
            model_name="gemini-4-neon",
            display_name="Reviewer Backup",
            role="verifier",
        ),
    )
    run_spec = WorkflowRunSpec(
        orchestrator=_review_binding(pool[1]),
        reviewers=(),
        tool_model_map={},
        task="reviewer explicit fallback isolation",
        chat_id="chat",
        topic_id=None,
        budget=6,
        deadline=None,
        auto_reviewer=True,
        agent_pool=pool,
        orchestrator_agent_id="A2",
    )
    engine = WorkflowEngine("chat", str(tmp_path), agent_type="claude")
    engine._run_spec = run_spec
    engine._project = WorkflowProject(name="wf", status=WorkflowStatus.RUNNING)
    engine._state_manager = WorkflowStateManager(engine._project)
    engine._executor = AgentExecutor(cwd=str(tmp_path), cancel_event=engine.cancel_event)

    executed: list[tuple[str, str, str | None]] = []

    class OkSession:
        def send_prompt(self, _prompt, **_kwargs):
            return type("Resp", (), {"text": "reviewer_ok", "output_tokens": 1, "stop_reason": "end_turn"})()

        def close(self):
            return None

    def fake_create_session(**kwargs):
        tool = kwargs.get("agent_type")
        model = kwargs.get("model_name")
        if tool == "claude":
            executed.append(("A2", tool, model))
            raise ACPStartupError("claude startup failed", agent_cmd="claude", agent_args=[], cwd=str(tmp_path))
        executed.append((tool, tool, model))
        return OkSession()

    try:
        with patch("src.agent_session.factory.create_engine_session", side_effect=fake_create_session):
            res_blocked = engine._handle_agent_call(
                AgentCallParams(prompt="review", agent_id="A2", role="reviewer", label="rev-1"),
                fallback_agent_ids=["A1"],
            )
            assert res_blocked.error is not None
            assert res_blocked.agent_id == "A2"
            assert all(item[0] != "codex" for item in executed)

            executed.clear()
            res_allowed = engine._handle_agent_call(
                AgentCallParams(prompt="review", agent_id="A2", role="reviewer", label="rev-2"),
                fallback_agent_ids=["A1", "A3"],
            )
            assert res_allowed.error is None
            assert res_allowed.agent_id == "A3"
            assert res_allowed.output == "reviewer_ok"
            assert all(item[0] != "codex" for item in executed)
    finally:
        engine._executor.shutdown(wait=True)


