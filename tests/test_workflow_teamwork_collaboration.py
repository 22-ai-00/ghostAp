"""Tests for M2 F4 Teamwork presets, role assignment, and multi-role script generation."""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from src.workflow_engine.agent_pool import WorkflowAgentBinding
from src.workflow_engine.models import AgentCallParams, AgentCallResult
from src.workflow_engine.script_gen import (
    extract_meta_from_script,
    generate_simple_script,
    validate_generated_script,
)
from src.workflow_engine.teamwork import (
    PRESET_ALIASES,
    TEAMWORK_PRESETS,
    TEAMWORK_ROLES,
    assign_teamwork_roles,
    generate_teamwork_script,
    list_teamwork_presets,
    normalize_teamwork_role,
    resolve_teamwork_preset,
)


def _binding(
    agent_id: str,
    tool_name: str,
    *,
    model_name: str | None = None,
    effort: str | None = None,
    profile: str | None = None,
    role: str | None = None,
    roles: tuple[str, ...] = (),
) -> WorkflowAgentBinding:
    return WorkflowAgentBinding(
        agent_id=agent_id,
        tool_name=tool_name,
        model_name=model_name,
        display_name=f"{tool_name}:{agent_id}",
        effort=effort,
        profile=profile,
        role=role,
        roles=roles,
    )


def test_presets_and_aliases_resolve_and_reject_unknown() -> None:
    presets = list_teamwork_presets()
    assert [p.preset_id for p in presets] == [
        "recommended",
        "quick",
        "collaborative",
        "review_team",
        "deep_collab",
    ]
    assert resolve_teamwork_preset(None).preset_id == "recommended"
    assert resolve_teamwork_preset("").preset_id == "recommended"
    for alias, target in PRESET_ALIASES.items():
        assert resolve_teamwork_preset(alias).preset_id == target
    for canonical in TEAMWORK_PRESETS:
        assert resolve_teamwork_preset(canonical.upper()).preset_id == canonical

    with pytest.raises(ValueError, match="Unknown Teamwork preset"):
        resolve_teamwork_preset("nonexistent_preset")


def test_normalize_teamwork_role_covers_canonical_and_aliases() -> None:
    for role in TEAMWORK_ROLES:
        assert normalize_teamwork_role(role) == role
    assert normalize_teamwork_role("explorer") == "scout"
    assert normalize_teamwork_role("orchestrator") == "planner"
    assert normalize_teamwork_role("implementer") == "worker"
    assert normalize_teamwork_role("focused-executor") == "worker"
    assert normalize_teamwork_role("verifier") == "reviewer"
    assert normalize_teamwork_role("adversarial_verifier") == "reviewer"
    assert normalize_teamwork_role("synth") == "synthesizer"
    assert normalize_teamwork_role("   ") is None
    assert normalize_teamwork_role("unknown-custom") is None


def test_assign_teamwork_roles_pairs_effort_tiers_and_enforces_nonself_cross_tool_review() -> None:
    # Same tool + model with low vs xhigh effort, plus distinct review tool
    pool = (
        _binding("A1", "codex", model_name="gpt-5.4", effort="low"),
        _binding("A2", "codex", model_name="gpt-5.4", effort="xhigh"),
        _binding("A3", "claude", model_name="sonnet", effort="high"),
    )
    plan = assign_teamwork_roles(pool, preset="recommended", orchestrator_agent_id="A3")

    assert plan.scout_id == "A1"
    # Primary worker != primary reviewer, and reviewer uses cross-tool claude
    assert plan.reviewer_ids[0] == "A3"
    assert plan.reviewer_ids[0] not in plan.worker_ids
    assert plan.orchestrator_agent_id == "A3"
    assert plan.synthesizer_id == "A3"


def test_assign_teamwork_roles_respects_explicit_role_overrides() -> None:
    pool = (
        _binding("A1", "claude", model_name="sonnet", effort="high", role="worker"),
        _binding("A2", "codex", model_name="gpt-5.4", effort="low", role="verifier"),
        _binding("A3", "traex", model_name="kimi-k2", roles=("scout", "synthesizer")),
    )
    plan = assign_teamwork_roles(pool, preset="deep_collab")

    assert plan.worker_ids[0] == "A1"
    assert plan.reviewer_ids[0] == "A2"
    assert plan.scout_id == "A3"


def test_assign_teamwork_roles_two_same_tool_different_efforts_separates_worker_and_reviewer() -> None:
    pool = (
        _binding("A1", "codex", model_name="gpt-5.4", effort="low"),
        _binding("A2", "codex", model_name="gpt-5.4", effort="xhigh"),
    )
    plan = assign_teamwork_roles(pool, preset="recommended")
    assert plan.worker_ids == ("A1",)
    assert plan.reviewer_ids == ("A2",)
    assert plan.worker_ids[0] != plan.reviewer_ids[0]


@pytest.mark.parametrize("preset_id", ["recommended", "quick", "collaborative", "review_team", "deep_collab"])
@pytest.mark.parametrize("pool_size", [1, 2, 3, 4])
def test_generate_teamwork_script_validates_for_all_presets_and_pool_sizes(
    preset_id: str,
    pool_size: int,
) -> None:
    all_members = (
        _binding("A1", "traex", model_name="kimi-k2", effort="low"),
        _binding("A2", "codex", model_name="gpt-5.4", effort="medium"),
        _binding("A3", "codex", model_name="gpt-5.4", effort="xhigh"),
        _binding("A4", "claude", model_name="sonnet", effort="high"),
    )
    pool = all_members[:pool_size]
    script = generate_teamwork_script(
        "Audit auth module // check https://example.com /* edge */",
        agent_pool=pool,
        preset=preset_id,
        orchestrator_agent_id=pool[-1].agent_id,
    )

    valid, errors = validate_generated_script(script, agent_pool=pool)
    assert valid, f"preset={preset_id}, size={pool_size}, errors={errors}"

    meta = extract_meta_from_script(script)
    assert meta is not None
    assert meta["name"] == f"teamwork-{preset_id}"
    assert meta["phases"]
    assert meta["tools"]
    assert meta["agentPlan"]

    # No backticks or banned routing tokens
    assert "`" not in script
    for banned in ("race(", "await classify(", "candidateTools"):
        assert banned not in script

    # Orchestrator binding is present in meta.agentPlan
    plan_ids = {entry["agentId"] for entry in meta["agentPlan"]}
    assert pool[-1].agent_id in plan_ids

    # 5-component handoff guidance and analysis-only safety note included
    for marker in ("observation", "logic_chain", "caveats", "conclusion", "verification"):
        assert marker in script
    assert "If the user asks for analysis only" in script
    assert "do not change code" in script

    # Explicit non-short timeouts (scout=180, heavy coding/review/synth >= 240)
    timeouts = [int(m.group(1)) for m in re.finditer(r"\btimeout:\s*(\d+)", script)]
    assert timeouts and min(timeouts) >= 180


def test_generate_simple_script_single_vs_multi_agent_delegation() -> None:
    single_pool = (_binding("A1", "coco"),)
    single_script = generate_simple_script("Do focused task", agent_pool=single_pool)
    assert 'name: "automatic-focused-workflow"' in single_script
    assert "function completionEnvelope" in single_script

    multi_pool = (
        _binding("A1", "codex", model_name="gpt-5.4", effort="low"),
        _binding("A2", "claude", model_name="sonnet", effort="high"),
    )
    multi_script = generate_simple_script(
        "Implement and review feature",
        agent_pool=multi_pool,
        orchestrator_agent_id="A2",
    )
    assert 'name: "teamwork-recommended"' in multi_script
    assert 'agentId: "A1"' in multi_script
    assert 'agentId: "A2"' in multi_script
    assert 'reviserAgentId: "A1"' in multi_script
    valid, errors = validate_generated_script(multi_script, agent_pool=multi_pool)
    assert valid, errors


def test_teamwork_generated_script_runs_end_to_end_on_real_node_runtime(tmp_path) -> None:
    from src.workflow_engine.bridge import RuntimeBridge
    from tests.test_workflow_bridge_transport import _run_real_node_workflow

    if not RuntimeBridge.check_node_available():
        pytest.skip("Node.js not available")

    pool = (
        _binding("A1", "traex", model_name="kimi-k2", effort="low", role="scout"),
        _binding("A2", "codex", model_name="gpt-5.4", effort="xhigh", role="worker"),
        _binding("A3", "claude", model_name="sonnet", effort="high", role="reviewer"),
    )
    script = generate_teamwork_script(
        "Build pagination API and verify edge cases",
        agent_pool=pool,
        preset="recommended",
        orchestrator_agent_id="A3",
    )

    calls: list[tuple[str | None, str | None]] = []
    verify_round = [0]

    def on_agent_call(params: AgentCallParams, **_kwargs: Any) -> AgentCallResult:
        calls.append((params.agent_id, params.role))
        if params.role == "reviewer" or (params.label and params.label.startswith("verify-")):
            verify_round[0] += 1
            if verify_round[0] == 1:
                review = {
                    "approve": False,
                    "issues": [
                        {
                            "severity": "major",
                            "description": "Missing negative limit check",
                            "evidence": "limit=-1 is not rejected",
                            "location": "api.py:12",
                        }
                    ],
                }
                return AgentCallResult(output=json.dumps(review), parsed=review, agent_id=params.agent_id)
            ok_review = {"approve": True, "issues": []}
            return AgentCallResult(output=json.dumps(ok_review), parsed=ok_review, agent_id=params.agent_id)
        handoff = {
            "observation": f"obs by {params.agent_id}",
            "logic_chain": "checked bounds",
            "caveats": "none",
            "conclusion": f"done by {params.agent_id}",
            "verification": "pytest passed",
        }
        return AgentCallResult(output=json.dumps(handoff), parsed=handoff, agent_id=params.agent_id)

    result = _run_real_node_workflow(
        tmp_path,
        script,
        on_agent_call,
        agent_pool=[b.to_dict() for b in pool],
        orchestrator_agent_id="A3",
    )

    assert result["card_summary"]["verdict"] == "passed"
    assert result["verification"]["accepted"] is True
    assert result["verification"]["rounds"] == 2
    # Scout -> Worker -> Verifier (reject) -> Reviser (A2!) -> Verifier (approve) -> Synthesizer (A3)
    assert ("A1", "scout") in calls
    assert ("A2", "worker") in calls
    assert ("A2", "reviser") in calls
    assert ("A3", "reviewer") in calls
    assert ("A3", "synthesizer") in calls


def test_teamwork_script_fanout_preserves_dollar_patterns_end_to_end(tmp_path) -> None:
    """generate_teamwork_script() + fanout() deliver $-containing requirements verbatim."""
    from src.workflow_engine.bridge import RuntimeBridge
    from tests.test_workflow_bridge_transport import _run_real_node_workflow

    if not RuntimeBridge.check_node_available():
        raise RuntimeError("Node.js required")

    tricky_req = "TRAP_START ${input} $$ $& $` $' $1 TRAP_END"
    pool = [
        _binding("A1", "traex", model_name="kimi-k2", effort="low"),
        _binding("A2", "codex", model_name="gpt-5.4", effort="high", role="worker"),
        _binding("A3", "claude", model_name="sonnet", effort="high", role="worker"),
        _binding("A4", "agy", model_name="gemini-3", effort="high", role="reviewer"),
    ]
    script = generate_teamwork_script(
        tricky_req,
        agent_pool=pool,
        preset="recommended",
        orchestrator_agent_id="A4",
    )
    worker_prompts: list[str] = []

    def on_agent_call(params: AgentCallParams, **_kwargs) -> AgentCallResult:
        if params.role == "worker":
            worker_prompts.append(params.prompt)
        if params.role == "reviewer":
            review = {"approve": True, "issues": []}
            return AgentCallResult(output=json.dumps(review), parsed=review, tool="agy")
        return AgentCallResult(output="done", tool="codex")

    _run_real_node_workflow(
        tmp_path,
        script,
        on_agent_call,
        agent_pool=[b.to_dict() for b in pool],
        orchestrator_agent_id="A4",
    )
    assert len(worker_prompts) >= 2
    for prompt in worker_prompts:
        assert tricky_req in prompt
