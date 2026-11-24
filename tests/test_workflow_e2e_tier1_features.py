"""Tier 1 opaque-box E2E feature coverage for GhostAP /wf optimization (F1-F10)."""

from __future__ import annotations

import asyncio
import importlib
import json
import re
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from src.acp.models import ACPEvent, ACPEventType, ToolCallInfo
from src.card.events.payloads import filter_workflow_button_value
from src.workflow_engine.agent_pool import WorkflowAgentBinding, freeze_agent_pool
from src.workflow_engine.journal import WorkflowJournal
from src.workflow_engine.models import AgentStatus, WorkflowProject, WorkflowStatus
from src.workflow_engine.progress_coalescer import ProgressCoalescer
from src.workflow_engine.renderer import WorkflowGenerationRenderer
from src.workflow_engine.state_manager import WorkflowStateManager
from tests.helpers.workflow_e2e_harness import (
    FakeACPClientSession,
    FakePopen,
    ScriptedRule,
    SessionInvocation,
    WorkflowSessionRecorder,
    assert_card_2_invariants,
    build_workflow_e2e_graph,
    click_card_element,
    collect_card_buttons,
    collect_card_selects,
    extract_card_text,
    find_card_button,
    find_card_select,
    make_run_spec,
    run_workflow_e2e,
)

# ---------------------------------------------------------------------------
# Area 1: Multi-Tool Scheduling, Capabilities & Role Division (F1, F4, F5)
# ---------------------------------------------------------------------------


def _single_agent_js(name: str, tool: str, role: str, label: str, *, conclusion: str | None = None) -> str:
    ret = (
        f"{{ card_summary: {{ verdict: 'passed', conclusion: {json.dumps(conclusion)} }}, result: r }}"
        if conclusion
        else "{ result: r }"
    )
    return (
        f'export const meta = {{ name: {json.dumps(name)}, description: "e2e", '
        f'phases: [{{ title: "Work", detail: "Run" }}], maxConcurrent: 1, tools: [{json.dumps(tool)}], '
        f'patterns: ["sequence"], agentPlan: [{{ node: {json.dumps(label)}, role: {json.dumps(role)}, agentId: "A1" }}] }};\n'
        f'export default async function main() {{\n'
        f'  const r = await agent("Run", {{ agentId: "A1", role: {json.dumps(role)}, label: {json.dumps(label)}, timeout: 30 }});\n'
        f'  return {ret};\n}}\n'
    )


def test_tier1_tool_capabilities_discover_effort_and_model_only_backends(monkeypatch) -> None:
    from src.workflow_engine.tool_registry import ToolCapability, get_tool_capabilities

    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        "src.acp.helper.is_programming_tool_available",
        lambda *_a, **_kw: True,
    )
    caps = get_tool_capabilities(force_refresh=True)
    for effort_tool in ("codex", "claude", "agy", "traex", "dsh"):
        cap = caps[effort_tool]
        assert isinstance(cap, ToolCapability)
        assert cap.available is True
        assert cap.supports_effort is True
    for plain_tool in ("coco", "aiden", "gemini", "grok"):
        cap = caps[plain_tool]
        assert isinstance(cap, ToolCapability)
        assert cap.supports_effort is False


@pytest.mark.parametrize(
    ("tool_name", "model_name", "profile", "effort"),
    [
        ("codex", "gpt-5.4", None, "xhigh"),
        ("claude", "sonnet", None, "high"),
        ("agy", "gemini-4-hydrogen", None, "high"),
        ("traex", "kimi-k2", "standard", "medium"),
        ("dsh", "deepseek/deepseek-r1", None, "high"),
        ("coco", None, None, None),
    ],
)
def test_tier1_validate_agent_binding_spec_accepts_valid_specs(
    monkeypatch,
    tool_name: str,
    model_name: str | None,
    profile: str | None,
    effort: str | None,
) -> None:
    from src.workflow_engine.tool_registry import invalidate_cache, validate_agent_binding_spec

    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("src.acp.traex_selection.load_traex_model_metadata", lambda *_a, **_kw: ())
    monkeypatch.setattr("src.acp.helper.get_cached_acp_models", lambda *_a, **_kw: ())
    invalidate_cache()
    ok, err = validate_agent_binding_spec(
        tool_name,
        model_name=model_name,
        profile=profile,
        effort=effort,
        available_tools={tool_name: tool_name},
    )
    assert ok is True, err
    assert err is None


def test_tier1_multi_tool_role_division_executes_scout_worker_verifier_synthesizer(tmp_path: Path, monkeypatch) -> None:
    script = """\
export const meta = {
  name: "multi-tool-teamwork", description: "Scout -> Worker -> Verify -> Synthesize",
  phases: [{ title: "Scout", detail: "Inspect" }, { title: "Implement", detail: "Workers" }, { title: "Verify", detail: "Check" }],
  maxConcurrent: 3, tools: ["codex", "claude", "agy", "dsh"], patterns: ["fanout", "verify"],
  agentPlan: [{ node: "scout", role: "scout", agentId: "A1" }, { node: "worker", role: "worker", agentId: "A2" }, { node: "verifier", role: "verifier", agentId: "A3" }, { node: "synth", role: "synthesizer", agentId: "A4" }],
};
export default async function main() {
  await phase("Scout");
  const scoped = await agent("Inspect repo", { agentId: "A1", role: "scout", label: "scout-1", timeout: 30 });
  await phase("Implement");
  const draft = await fanout(scoped, [
    { prompt: "Draft impl: ${input}", agentId: "A2", role: "worker", label: "impl-1", timeout: 30 },
    { prompt: "Draft tests: ${input}", agentId: "A4", role: "worker", label: "impl-2", timeout: 30 },
  ], { synthesizerAgentId: "A4", timeout: 30 });
  await phase("Verify");
  const checked = await verify(draft, { verifiers: [{ agentId: "A3", role: "verifier" }], reviserAgentId: "A2", maxRounds: 2, timeout: 30 });
  return { card_summary: { verdict: checked.accepted ? "passed" : "failed", conclusion: "Multi-tool loop done." }, result: checked };
}
"""
    verify_calls = [0]

    def _resolve(inv: SessionInvocation) -> ScriptedRule:
        if inv.agent_type == "agy":
            verify_calls[0] += 1
            if verify_calls[0] == 1:
                return ScriptedRule(output=json.dumps({"approve": False, "issues": [{"severity": "major", "description": "need test", "evidence": "missing"}]}))
            return ScriptedRule(output=json.dumps({"approve": True, "issues": []}))
        return ScriptedRule(output=f"{inv.agent_type}-output")

    pool = (
        WorkflowAgentBinding(agent_id="A1", tool_name="codex", model_name="gpt-5.4", display_name="Scout", effort="low"),
        WorkflowAgentBinding(agent_id="A2", tool_name="claude", model_name="sonnet", display_name="Worker", effort="high"),
        WorkflowAgentBinding(agent_id="A3", tool_name="agy", model_name="gemini-4-hydrogen", display_name="Verifier", effort="high"),
        WorkflowAgentBinding(agent_id="A4", tool_name="dsh", model_name="deepseek-chat", display_name="Synth", effort="medium"),
    )
    e2e = run_workflow_e2e(tmp_path, monkeypatch, script_content=script, run_spec=make_run_spec(pool=pool, task="Multi-tool teamwork loop"), recorder=WorkflowSessionRecorder(_resolve))
    assert e2e.project.status == WorkflowStatus.COMPLETED
    assert e2e.project.metrics.failed_agents == 0
    assert {inv.agent_type for inv in e2e.recorder.invocations} == {"codex", "claude", "agy", "dsh"}
    assert json.loads(e2e.project.result)["result"]["accepted"] is True


def test_tier1_verify_accepts_extended_review_json_and_routes_reviser_to_worker(tmp_path: Path, monkeypatch) -> None:
    script = """\
export const meta = {
  name: "verify-reviser-route", description: "Verify extended JSON and reviser routing",
  phases: [{ title: "Verify", detail: "Review and revise" }], maxConcurrent: 2, tools: ["codex", "agy"], patterns: ["verify"],
  agentPlan: [{ node: "worker", role: "worker", agentId: "A1" }, { node: "verifier", role: "verifier", agentId: "A2" }],
};
export default async function main() {
  return { result: await verify("initial draft", { verifiers: [{ agentId: "A2", role: "verifier" }], reviserAgentId: "A1", maxRounds: 2, timeout: 30 }) };
}
"""
    round_num = [0]

    def _resolve(inv: SessionInvocation) -> ScriptedRule:
        if inv.agent_type == "agy":
            round_num[0] += 1
            if round_num[0] == 1:
                return ScriptedRule(output=json.dumps({"approve": False, "issues": [{"severity": "major", "description": "edge bug", "evidence": "line 10"}], "summary": "extra field allowed by F5", "confidence": 0.91}))
            return ScriptedRule(output=json.dumps({"approve": True, "issues": [], "summary": "ok"}))
        return ScriptedRule(output="revised draft from worker")

    pool = (
        WorkflowAgentBinding(agent_id="A1", tool_name="codex", model_name="gpt-5.4", display_name="Worker"),
        WorkflowAgentBinding(agent_id="A2", tool_name="agy", model_name="gemini-4-hydrogen", display_name="Verifier"),
    )
    e2e = run_workflow_e2e(tmp_path, monkeypatch, script_content=script, run_spec=make_run_spec(pool=pool, task="Verify reviser routing"), recorder=WorkflowSessionRecorder(_resolve))
    assert e2e.project.status == WorkflowStatus.COMPLETED
    assert [inv.agent_type for inv in e2e.recorder.invocations] == ["agy", "codex", "agy"]
    payload = json.loads(e2e.project.result)["result"]
    assert payload["accepted"] is True and payload["rounds"] == 2


def test_tier1_fanout_and_tournament_synthesize_without_lossy_truncation(tmp_path: Path, monkeypatch) -> None:
    script = """\
export const meta = {
  name: "lossless-synthesis", description: "Verify >2000 char outputs are not truncated",
  phases: [{ title: "Work", detail: "Fanout and tournament" }], maxConcurrent: 2, tools: ["codex", "claude"], patterns: ["fanout", "tournament"],
  agentPlan: [{ node: "w1", role: "worker", agentId: "A1" }, { node: "j1", role: "judge", agentId: "A2" }],
};
export default async function main() {
  const fan = await fanout("topic", [{ prompt: "w1", agentId: "A1", label: "fan-w1", timeout: 30 }, { prompt: "w2", agentId: "A1", label: "fan-w2", timeout: 30 }], { synthesizerAgentId: "A2", timeout: 30 });
  const tour = await tournament([{ prompt: "c1", agentId: "A1", label: "tour-c1", timeout: 30 }, { prompt: "c2", agentId: "A1", label: "tour-c2", timeout: 30 }], null, { judgeAgentId: "A2", timeout: 30 });
  return { fan, tour };
}
"""
    long_w1, long_w2 = ("A" * 2600) + "TAIL_MARKER_W1", ("B" * 2600) + "TAIL_MARKER_W2"
    observed_synth_prompts: list[str] = []

    def _resolve(inv: SessionInvocation) -> ScriptedRule:
        if inv.agent_type == "claude":
            observed_synth_prompts.append(inv.prompt)
            return ScriptedRule(output=json.dumps({"winner": "A", "reasoning": "complete"}))
        return ScriptedRule(output=long_w1 if ("w1" in inv.prompt or "c1" in inv.prompt) else long_w2)

    pool = (
        WorkflowAgentBinding(agent_id="A1", tool_name="codex", model_name="gpt-5.4", display_name="Worker"),
        WorkflowAgentBinding(agent_id="A2", tool_name="claude", model_name="sonnet", display_name="Synth"),
    )
    e2e = run_workflow_e2e(tmp_path, monkeypatch, script_content=script, run_spec=make_run_spec(pool=pool, task="Lossless synthesis"), recorder=WorkflowSessionRecorder(_resolve))
    assert e2e.project.status == WorkflowStatus.COMPLETED
    assert len(observed_synth_prompts) == 2
    for prompt in observed_synth_prompts:
        assert "TAIL_MARKER_W1" in prompt and "TAIL_MARKER_W2" in prompt


# ---------------------------------------------------------------------------
# Area 2: Multi-Effort Coexistence & Backend Passthrough (F2, F3)
# ---------------------------------------------------------------------------


def test_tier1_multi_effort_and_multi_profile_coexist_in_frozen_pool() -> None:
    pool = freeze_agent_pool(
        [
            WorkflowAgentBinding(agent_id="A1", tool_name="codex", model_name="gpt-5.4", display_name="Codex Low", effort="low", role="scout"),
            WorkflowAgentBinding(agent_id="A2", tool_name="codex", model_name="gpt-5.4", display_name="Codex XHigh", effort="xhigh", role="worker"),
            WorkflowAgentBinding(agent_id="A3", tool_name="traex", model_name="kimi-k2", display_name="Traex Std", profile="standard", effort="high", role="reviewer"),
            WorkflowAgentBinding(agent_id="A4", tool_name="traex", model_name="kimi-k2", display_name="Traex Max", profile="max", effort="high", role="synthesizer"),
        ]
    )
    assert len(pool) == 4
    assert len({b.binding_key for b in pool}) == 4
    spec = make_run_spec(pool=pool, task="Coexistence check")
    assert spec.agent_binding("A1").effort == "low"
    assert spec.agent_binding("A2").effort == "xhigh"
    assert spec.agent_binding("A3").profile == "standard"
    assert spec.agent_binding("A4").profile == "max"


@pytest.mark.parametrize(
    ("tool_name", "model_name", "profile", "effort", "expected"),
    [
        ("codex", "gpt-5.4", None, "xhigh", "gpt-5.4/xhigh"),
        ("codex_w", "gpt-5.4", None, "low", "gpt-5.4/low"),
        ("claude", "sonnet", None, "high", "sonnet/high"),
        ("claude_w", "opus[1m]", None, "max", "opus[1m]/max"),
        ("agy", "gemini-4-hydrogen", None, "high", "gemini-4-hydrogen/high"),
        ("traex", "kimi-k2", "max", "high", "kimi-k2/max/high"),
        ("dsh", '["deepseek","deepseek-chat"]', None, "high", '["deepseek","deepseek-chat","high"]'),
        ("coco", "coco-pro", None, None, "coco-pro"),
        ("gemini", "gemini-2.5-pro", None, None, "gemini-2.5-pro"),
    ],
)
def test_tier1_effective_model_name_resolves_canonical_backend_selections(
    tool_name: str,
    model_name: str,
    profile: str | None,
    effort: str | None,
    expected: str,
) -> None:
    binding = WorkflowAgentBinding(
        agent_id="A1",
        tool_name=tool_name,
        model_name=model_name,
        display_name="Agent",
        profile=profile,
        effort=effort,
    )
    assert binding.effective_model_name() == expected


def test_tier1_engine_propagates_distinct_effort_and_isolates_journal_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    script = """\
export const meta = {
  name: "effort-journal-isolation",
  description: "Verify effort passthrough and journal isolation",
  phases: [{ title: "Run", detail: "Execute same prompt across efforts" }],
  maxConcurrent: 1,
  tools: ["codex"],
  patterns: ["sequence"],
  agentPlan: [
    { node: "low", role: "scout", agentId: "A1" },
    { node: "xhigh", role: "worker", agentId: "A2" },
  ],
};
export default async function main() {
  const r1 = await agent("Identical E2E prompt", { agentId: "A1", role: "scout", label: "call-low-1", timeout: 30 });
  const r2 = await agent("Identical E2E prompt", { agentId: "A2", role: "worker", label: "call-xhigh", timeout: 30 });
  const r3 = await agent("Identical E2E prompt", { agentId: "A1", role: "scout", label: "call-low-2", timeout: 30 });
  return { r1, r2, r3 };
}
"""
    recorder = WorkflowSessionRecorder(lambda inv: ScriptedRule(output=f"model={inv.model_name}"))
    pool = (
        WorkflowAgentBinding(agent_id="A1", tool_name="codex", model_name="gpt-5.4", display_name="Low", effort="low"),
        WorkflowAgentBinding(agent_id="A2", tool_name="codex", model_name="gpt-5.4", display_name="XHigh", effort="xhigh"),
    )
    e2e = run_workflow_e2e(
        tmp_path,
        monkeypatch,
        script_content=script,
        run_spec=make_run_spec(pool=pool, task="Effort cache isolation"),
        recorder=recorder,
    )
    assert e2e.project.status == WorkflowStatus.COMPLETED
    # A1 low (miss) + A2 xhigh (miss, NOT colliding with A1!) + A1 repeat (hit)
    assert len(recorder.invocations) == 2
    assert [inv.model_name for inv in recorder.invocations] == ["gpt-5.4/low", "gpt-5.4/xhigh"]
    assert e2e.project.metrics.cached_agents == 1

    # Direct WorkflowJournal.compute_key differentiation for identical model differing only by effort/profile
    assert WorkflowJournal.compute_key(prompt="same", tool="codex", model="gpt-5.4", effort="low") != (
        WorkflowJournal.compute_key(prompt="same", tool="codex", model="gpt-5.4", effort="xhigh")
    )
    assert WorkflowJournal.compute_key(prompt="same", tool="traex", model="kimi-k2", profile="standard", effort="high") != (
        WorkflowJournal.compute_key(prompt="same", tool="traex", model="kimi-k2", profile="max", effort="high")
    )


@pytest.mark.parametrize(
    ("tool_name", "model_name", "profile", "effort", "expected_marker"),
    [
        ("claude", "sonnet", None, "high", "--effort:high"),
        ("claude_w", "opus[1m]", None, "max", "--effort:max"),
        ("codex_w", "gpt-5.4", None, "xhigh", "model_reasoning_effort=xhigh"),
        ("agy", "gemini-pro-orchestrator", "low", "xhigh", "--settings:orchestrator"),
    ],
)
def test_tier1_cli_backend_wire_applies_effort_to_argv(
    tmp_path: Path, monkeypatch, tool_name: str, model_name: str, profile: str | None, effort: str, expected_marker: str
) -> None:
    from src.agent_session.factory import create_engine_session

    monkeypatch.setattr("src.agent_session.claude_cli.SyncClaudeCLISession.start", lambda self, **_kw: "cli-claude")
    monkeypatch.setattr("src.agent_session.codex_cli.SyncCodexCLISession.start", lambda self, **_kw: "cli-codex")
    monkeypatch.setattr("src.agent_session.agy_cli.SyncAGYCLISession.start", lambda self, **_kw: "cli-agy")
    monkeypatch.setattr("subprocess.Popen", lambda cmd, **kwargs: FakePopen(cmd, **kwargs))

    binding = WorkflowAgentBinding(agent_id="A1", tool_name=tool_name, model_name=model_name, display_name=tool_name, profile=profile, effort=effort)
    sess = create_engine_session(tool_name, str(tmp_path), model_name=binding.effective_model_name())
    inner = getattr(getattr(sess, "_inner", sess), "_inner", getattr(sess, "_inner", sess))
    inner.session_id = f"sess-{tool_name}"
    args = inner._build_args("hello", resumed=False)
    sess.close()

    if expected_marker.startswith("--effort:"):
        assert "--effort" in args and args[args.index("--effort") + 1] == expected_marker.split(":", 1)[1]
    elif expected_marker.startswith("--settings:"):
        payload = json.loads(args[args.index("--settings") + 1])["orchestrator"]
        assert "low" in payload["planner"].lower() and "xhigh" in payload["executor"].lower()
    else:
        assert any("model_reasoning_effort" in a and effort in a for a in args)


@pytest.mark.parametrize(
    ("tool_name", "model_name", "profile", "effort", "expected_calls"),
    [
        ("codex", "gpt-5.4", None, "xhigh", [("model", "gpt-5.4"), ("reasoning_effort", "xhigh")]),
        ("agy", "gemini-4-hydrogen", None, "high", [("model", "gemini-4-hydrogen"), ("thought_level", "high")]),
        ("traex", "kimi-k2", "max", "high", [("model", "kimi-k2-max"), ("reasoning_effort", "high")]),
        ("dsh", '["deepseek","deepseek-r1"]', None, "high", [("dsh.model", '["deepseek","deepseek-r1"]'), ("dsh.reasoning_effort", '["effort","high"]')]),
    ],
)
def test_tier1_acp_backend_wire_applies_effort_and_profile_config(
    tmp_path: Path, monkeypatch, tool_name: str, model_name: str, profile: str | None, effort: str, expected_calls: list[tuple[str, str]]
) -> None:
    from src.acp.sync_adapter import SyncACPSession
    from src.acp.traex_selection import TraexModelMetadata, TraexProfileMetadata

    if tool_name == "traex":
        meta = (TraexModelMetadata(config_name="kimi-k2", slug="kimi-k2", profiles=(
            TraexProfileMetadata(profile="standard", backend_model_value="kimi-k2", reasoning_efforts=("low", "high")),
            TraexProfileMetadata(profile="max", backend_model_value="kimi-k2-max", reasoning_efforts=("low", "high")),
        )),)
        monkeypatch.setattr("src.acp.traex_selection.load_traex_model_metadata", lambda *_a, **_kw: meta)

    eff_model = WorkflowAgentBinding(agent_id="A1", tool_name=tool_name, model_name=model_name, display_name=tool_name, profile=profile, effort=effort).effective_model_name()
    assert eff_model is not None
    fake_acp = FakeACPClientSession(agent_type=tool_name, cwd=str(tmp_path), model_name=eff_model)
    acp_sess = SyncACPSession(agent_type=tool_name, cwd=str(tmp_path), agent_cmd=tool_name, agent_args=["--acp"], model_name=eff_model)
    acp_sess._acp_session = fake_acp
    method_map = {
        "codex": acp_sess._apply_official_codex_selection,
        "agy": acp_sess._apply_agy_selection,
        "traex": acp_sess._apply_traex_selection,
        "dsh": acp_sess._apply_dsh_selection,
    }
    assert asyncio.run(method_map[tool_name](eff_model)) is True
    for expected in expected_calls:
        assert expected in fake_acp.config_calls


def test_tier1_agent_progress_and_callbacks_carry_effort_profile_and_role(tmp_path: Path, monkeypatch) -> None:
    sm = WorkflowStateManager(WorkflowProject(workflow_id="wf_sm"))
    sm.on_phase_changed("Work")
    sm.on_agent_started("step-a", "codex", "Work", model="gpt-5.4", role="scout", agent_id="A1", effort="low", profile="turbo", effective_model="gpt-5.4/low")
    sm_agent = sm.snapshot().phases[0].agents[0]
    assert (sm_agent.effort, sm_agent.profile, sm_agent.role, sm_agent.effective_model) == ("low", "turbo", "scout", "gpt-5.4/low")

    script = (
        'export const meta = { name: "progress-effort", description: "Check effort", phases: [{ title: "Work", detail: "Run" }], '
        'maxConcurrent: 2, tools: ["codex", "traex"], patterns: ["sequence"], agentPlan: [{ node: "w1", role: "worker", agentId: "A1" }] };\n'
        'export default async function main() {\n'
        '  const a = await agent("Step A", { agentId: "A1", role: "scout", label: "step-a", timeout: 30 });\n'
        '  const b = await agent("Step B", { agentId: "A2", role: "worker", label: "step-b", timeout: 30 });\n'
        '  return { a, b };\n}\n'
    )
    pool = (
        WorkflowAgentBinding(agent_id="A1", tool_name="codex", model_name="gpt-5.4", display_name="Low", effort="low", role="scout"),
        WorkflowAgentBinding(agent_id="A2", tool_name="traex", model_name="kimi-k2", display_name="Max", profile="max", effort="high", role="worker"),
    )
    e2e = run_workflow_e2e(tmp_path, monkeypatch, script_content=script, run_spec=make_run_spec(pool=pool, task="Progress metadata"), recorder=WorkflowSessionRecorder())
    assert e2e.project.status == WorkflowStatus.COMPLETED
    assert e2e.engine._state_manager is not None
    state_snap = e2e.engine._state_manager.snapshot()
    by_label = {ag.label: ag for ph in state_snap.phases for ag in ph.agents}
    assert (by_label["step-a"].effort, by_label["step-a"].role) == ("low", "scout")
    assert (by_label["step-b"].profile, by_label["step-b"].effort, by_label["step-b"].role) == ("max", "high", "worker")
    assert e2e.progress_snapshots
    done_by_label = {lbl: payload for lbl, payload in e2e.agent_done_payloads}
    assert done_by_label["step-a"].get("effort") == "low"
    assert done_by_label["step-b"].get("effort") == "high" and done_by_label["step-b"].get("profile") == "max"


# ---------------------------------------------------------------------------
# Area 3: Teamwork Presets & Quick Launch (F4)
# ---------------------------------------------------------------------------


def test_tier1_teamwork_presets_catalog_exposes_collaborative_roles_and_bindings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    teamwork_mod = importlib.import_module("src.workflow_engine.teamwork")
    mod_source = Path(teamwork_mod.__file__).read_text(encoding="utf-8")
    for preset in ("auto", "collaborative", "review_team"):
        assert preset in mod_source
    assert any(role in mod_source for role in ("scout", "planner"))
    assert any(role in mod_source for role in ("verifier", "reviewer"))
    assert "synthesizer" in mod_source

    graph = build_workflow_e2e_graph(tmp_path, monkeypatch)
    try:
        with patch(
            "src.workflow_engine.tool_registry.get_available_tools",
            return_value={"codex": "Codex", "claude": "Claude", "agy": "AGY"},
        ):
            graph.send_text("/wf_config")
            card = graph.fake_im.latest_card()
            assert card is not None
            assert_card_2_invariants(card, label="teamwork-catalog-card")
            txt = extract_card_text(card).lower()
            assert any(p in txt for p in ("collaborative", "review_team", "协作", "复核"))
    finally:
        graph.close()


def test_tier1_teamwork_preset_executes_without_ai_codegen_roundtrip(
    tmp_path: Path,
    monkeypatch,
) -> None:
    importlib.import_module("src.workflow_engine.teamwork")

    def _resolve(inv: SessionInvocation) -> ScriptedRule:
        lowered = inv.prompt.lower()
        if "approve" in lowered or "issues" in lowered or "verify" in lowered:
            return ScriptedRule(output=json.dumps({"approve": True, "issues": []}))
        return ScriptedRule(
            output=json.dumps(
                {
                    "observation": "checked files",
                    "logic_chain": "traced token check",
                    "caveats": "none",
                    "conclusion": "auth valid",
                    "verification": "pytest passed",
                }
            )
        )

    recorder = WorkflowSessionRecorder(_resolve).patch(monkeypatch)
    graph = build_workflow_e2e_graph(tmp_path, monkeypatch)
    try:
        with patch(
            "src.workflow_engine.tool_registry.get_available_tools",
            return_value={"codex": "Codex", "claude": "Claude", "agy": "AGY"},
        ):
            graph.send_text("/wf_config")
            cfg_id = graph.fake_im.latest_card_message_id()
            cfg_card = graph.fake_im.latest_card()
            assert cfg_id and cfg_card
            preset_sel = find_card_select(cfg_card, option_values=("review_team", "collaborative"))
            if preset_sel is not None:
                opt_val = next(
                    o["value"] for o in preset_sel["options"] if o.get("value") in {"review_team", "collaborative"}
                )
                _, _, cfg_card = click_card_element(graph, message_id=cfg_id, element=preset_sel, option=opt_val)
            save_btn = find_card_button(cfg_card, action_tokens=("save", "confirm", "apply"), text_tokens=("保存", "应用", "确认"))
            assert save_btn is not None
            click_card_element(graph, message_id=cfg_id, element=save_btn)
            saved_state = graph.project.chat_states.get("oc_e2e")
            assert saved_state is not None and getattr(saved_state, "workflow_team_config", None)
            graph.send_text("/wf Audit auth flow")
            launch_cfg_id = graph.fake_im.latest_card_message_id()
            launch_cfg_card = graph.fake_im.latest_card()
            assert launch_cfg_id and launch_cfg_card
            assert "配置团队并启动" in extract_card_text(launch_cfg_card)
            start_btn = find_card_button(launch_cfg_card, action_tokens=("wf_start_workflow",))
            assert start_btn is not None
            click_card_element(graph, message_id=launch_cfg_id, element=start_btn)

        assert not any("Workflow Script Generation Task" in inv.prompt for inv in recorder.invocations)
        cards = graph.fake_im.all_cards()
        for idx, c in enumerate(cards):
            assert_card_2_invariants(c, label=f"preset-fastpath-{idx}")
        assert len(recorder.invocations) >= 2
        engine = graph.client._handler_ctx.workflow_engine_manager.get("oc_e2e", str(graph.repo_dir))
        assert engine is not None and engine.project.status == WorkflowStatus.COMPLETED
    finally:
        graph.close()


def test_tier1_quick_launch_wf_with_requirement_auto_binds_and_completes(tmp_path: Path, monkeypatch) -> None:
    js_script = _single_agent_js("quick-launch", "codex", "worker", "w1", conclusion="Quick launch done.")

    def _resolve(inv: SessionInvocation) -> ScriptedRule:
        if "Workflow Script Generation Task" in inv.prompt:
            return ScriptedRule(output=js_script)
        lowered = inv.prompt.lower()
        if "approve" in lowered or "issues" in lowered or "verify" in lowered:
            return ScriptedRule(output=json.dumps({"approve": True, "issues": []}))
        return ScriptedRule(output="quick-launch-ok")

    recorder = WorkflowSessionRecorder(_resolve).patch(monkeypatch)
    graph = build_workflow_e2e_graph(tmp_path, monkeypatch)
    try:
        with patch("src.workflow_engine.tool_registry.get_available_tools", return_value={"codex": "Codex", "claude": "Claude"}):
            graph.send_text("/wf --auto 深入分析并优化缓存模块")

        cards = graph.fake_im.all_cards()
        assert len(cards) >= 2
        for idx, card in enumerate(cards):
            assert_card_2_invariants(card, label=f"quick-launch-{idx}")
        assert not any("操作失败" in extract_card_text(c) for c in cards)
        assert recorder.invocations
        assert {inv.agent_type for inv in recorder.invocations}.issubset({"codex", "claude"})
        engine = graph.client._handler_ctx.workflow_engine_manager.get("oc_e2e", str(graph.repo_dir))
        assert engine is not None and engine.project.status == WorkflowStatus.COMPLETED
    finally:
        graph.close()


def test_tier1_interactive_wf_opens_config_adjusts_and_launches_inplace(tmp_path: Path, monkeypatch) -> None:
    def _resolve(inv: SessionInvocation) -> ScriptedRule:
        lowered = inv.prompt.lower()
        if "approve" in lowered or "issues" in lowered or "verify" in lowered:
            return ScriptedRule(output=json.dumps({"approve": True, "issues": []}))
        return ScriptedRule(output=f"interactive-ok:{inv.agent_type}:{inv.model_name}")

    recorder = WorkflowSessionRecorder(_resolve).patch(monkeypatch)
    graph = build_workflow_e2e_graph(tmp_path, monkeypatch)
    try:
        with patch(
            "src.workflow_engine.tool_registry.get_available_tools",
            return_value={"codex": "Codex", "claude": "Claude", "traex": "Traex"},
        ):
            graph.send_text("/wf 重构缓存与认证模块")
            assert not recorder.invocations, "/wf <需求> must open config card instead of auto-starting"
            cfg_id = graph.fake_im.latest_card_message_id()
            cfg_card = graph.fake_im.latest_card()
            assert cfg_id is not None and cfg_card is not None
            assert_card_2_invariants(cfg_card, label="interactive-wf-launch-card")
            assert cfg_card["header"]["title"]["content"] == "⚡ Workflow · 配置团队并启动"
            start_btn = find_card_button(cfg_card, action_tokens=("wf_start_workflow",))
            assert start_btn is not None
            assert "按当前团队立即开始" in (start_btn.get("text") or {}).get("content", "")

            # Adjust A1 to traex + high effort via card callbacks, apply, then launch in-place
            tool_sel = find_card_select(cfg_card, option_values=("traex",), name_tokens=("tool",))
            assert tool_sel is not None
            _, _, cfg_card = click_card_element(graph, message_id=cfg_id, element=tool_sel, option="traex")
            effort_sel = find_card_select(cfg_card, option_values=("high",), name_tokens=("effort",))
            if effort_sel is not None:
                _, _, cfg_card = click_card_element(graph, message_id=cfg_id, element=effort_sel, option="high")
            apply_btn = find_card_button(cfg_card, action_tokens=("wf_add_or_update_agent",))
            assert apply_btn is not None
            _, _, cfg_card = click_card_element(graph, message_id=cfg_id, element=apply_btn)
            assert cfg_card is not None

            before_sends = sum(
                1 for d in graph.fake_im.deliveries if d.op in {"send", "reply"} and d.msg_type == "interactive"
            )
            start_btn_after = find_card_button(cfg_card, action_tokens=("wf_start_workflow",))
            assert start_btn_after is not None
            click_card_element(graph, message_id=cfg_id, element=start_btn_after)
            after_sends = sum(
                1 for d in graph.fake_im.deliveries if d.op in {"send", "reply"} and d.msg_type == "interactive"
            )

        assert after_sends == before_sends, "Launching from config card must patch in-place without sending a second message"
        assert recorder.invocations
        assert any(inv.agent_type == "traex" for inv in recorder.invocations)
        assert any(d.op == "patch" and d.message_id == cfg_id for d in graph.fake_im.deliveries)
        final_on_same_msg = graph.fake_im.latest_card(cfg_id)
        assert final_on_same_msg is not None
        assert "Workflow — 完成" in extract_card_text(final_on_same_msg)
        engine = graph.client._handler_ctx.workflow_engine_manager.get("oc_e2e", str(graph.repo_dir))
        assert engine is not None and engine.project.status == WorkflowStatus.COMPLETED
    finally:
        graph.close()


def test_tier1_preset_flag_bypasses_config_card_and_executes_immediately(tmp_path: Path, monkeypatch) -> None:
    recorder = WorkflowSessionRecorder(
        lambda inv: ScriptedRule(
            output=json.dumps({"approve": True, "issues": []})
            if ("approve" in inv.prompt.lower() or "verify" in inv.prompt.lower())
            else f"preset-flag-ok:{inv.agent_type}"
        )
    ).patch(monkeypatch)
    graph = build_workflow_e2e_graph(tmp_path, monkeypatch)
    try:
        with patch(
            "src.workflow_engine.tool_registry.get_available_tools",
            return_value={"codex": "Codex", "claude": "Claude"},
        ):
            graph.send_text("/wf --preset quick 检查缓存失效问题")

        assert recorder.invocations
        assert not any("配置团队并启动" in extract_card_text(c) for c in graph.fake_im.all_cards())
        engine = graph.client._handler_ctx.workflow_engine_manager.get("oc_e2e", str(graph.repo_dir))
        assert engine is not None and engine.project.status == WorkflowStatus.COMPLETED
    finally:
        graph.close()


def test_tier1_teamwork_role_affinity_assigns_preferred_tools_to_matching_roles(tmp_path: Path, monkeypatch) -> None:
    importlib.import_module("src.workflow_engine.teamwork")
    js_fallback = (
        'export const meta = { name: "role-affinity", description: "Role affinity", phases: [{ title: "Work", detail: "Run" }], '
        'maxConcurrent: 2, tools: ["traex", "codex", "claude"], patterns: ["sequence"], '
        'agentPlan: [{ node: "s", role: "scout", agentId: "A1" }, { node: "w", role: "worker", agentId: "A2" }, { node: "v", role: "verifier", agentId: "A3" }] };\n'
        'export default async function main() {\n'
        '  const s = await agent("Scout", { agentId: "A1", role: "scout", label: "s1", timeout: 30 });\n'
        '  const w = await agent(s, { agentId: "A2", role: "worker", label: "w1", timeout: 30 });\n'
        '  return { result: await verify(w, { verifiers: [{ agentId: "A3", role: "verifier" }], maxRounds: 1, timeout: 30 }) };\n}\n'
    )

    def _resolve(inv: SessionInvocation) -> ScriptedRule:
        if "Workflow Script Generation Task" in inv.prompt:
            return ScriptedRule(output=js_fallback)
        lowered = inv.prompt.lower()
        if "approve" in lowered or "issues" in lowered or "verify" in lowered:
            return ScriptedRule(output=json.dumps({"approve": True, "issues": []}))
        return ScriptedRule(output=f"role-ok:{inv.agent_type}")

    recorder = WorkflowSessionRecorder(_resolve).patch(monkeypatch)
    graph = build_workflow_e2e_graph(tmp_path, monkeypatch)
    try:
        state = graph.project._ensure_chat_state_unlocked("oc_e2e")
        assert hasattr(state, "workflow_team_config"), "ChatProjectState must define workflow_team_config for F9"
        with patch("src.workflow_engine.tool_registry.get_available_tools", return_value={"traex": "Traex", "codex": "Codex", "claude": "Claude"}):
            graph.send_text("/wf_config")
            cfg_id = graph.fake_im.latest_card_message_id()
            cfg_card = graph.fake_im.latest_card()
            assert cfg_id and cfg_card
            preset_sel = find_card_select(cfg_card, option_values=("review_team", "collaborative"))
            if preset_sel is not None:
                opt = next(o["value"] for o in preset_sel["options"] if o.get("value") in {"review_team", "collaborative"})
                _, _, cfg_card = click_card_element(graph, message_id=cfg_id, element=preset_sel, option=opt)
            save_btn = find_card_button(cfg_card, action_tokens=("save", "confirm", "apply"), text_tokens=("保存", "应用", "确认"))
            assert save_btn is not None
            click_card_element(graph, message_id=cfg_id, element=save_btn)
            assert getattr(state, "workflow_team_config", None)
            graph.send_text("/wf 审查并修复边界问题")
            launch_cfg_id = graph.fake_im.latest_card_message_id()
            launch_cfg_card = graph.fake_im.latest_card()
            assert launch_cfg_id and launch_cfg_card
            start_btn = find_card_button(launch_cfg_card, action_tokens=("wf_start_workflow",))
            assert start_btn is not None
            click_card_element(graph, message_id=launch_cfg_id, element=start_btn)
        engine = graph.client._handler_ctx.workflow_engine_manager.get("oc_e2e", str(graph.repo_dir))
        assert engine is not None and engine.project.status == WorkflowStatus.COMPLETED
        assert {"traex", "codex", "claude"}.issubset({inv.agent_type for inv in recorder.invocations})
    finally:
        graph.close()


def test_tier1_teamwork_five_component_handoff_preserved_through_synthesis(
    tmp_path: Path,
    monkeypatch,
) -> None:
    script = """\
export const meta = {
  name: "handoff-synthesis",
  description: "5-component handoff synthesis",
  phases: [{ title: "Collab", detail: "Worker handoffs" }],
  maxConcurrent: 2,
  tools: ["codex", "claude"],
  patterns: ["fanout"],
  agentPlan: [
    { node: "w1", role: "worker", agentId: "A1" },
    { node: "s1", role: "synthesizer", agentId: "A2" },
  ],
};
export default async function main() {
  const synth = await fanout("Build API", [
    { prompt: "Implement endpoint", agentId: "A1", role: "worker", label: "w1", timeout: 30 },
  ], { synthesizerAgentId: "A2", timeout: 30 });
  return { result: synth };
}
"""
    handoff_payload = {
        "observation": "OBS-" + ("X" * 1400) + "OBS_TAIL_MARKER",
        "logic_chain": "Offset limit added cleanly",
        "caveats": "Cursor index deferred",
        "conclusion": "Pagination complete",
        "verification": "VER-" + ("Y" * 1400) + "VER_TAIL_MARKER",
    }

    synth_prompts: list[str] = []

    def _resolve(inv: SessionInvocation) -> ScriptedRule:
        if inv.agent_type == "codex":
            return ScriptedRule(output=json.dumps(handoff_payload))
        synth_prompts.append(inv.prompt)
        return ScriptedRule(output=json.dumps({
            "observation": "OBS_TAIL_MARKER",
            "logic_chain": "ok",
            "caveats": "none",
            "conclusion": "complete",
            "verification": "VER_TAIL_MARKER",
        }))

    pool = (
        WorkflowAgentBinding(agent_id="A1", tool_name="codex", model_name="gpt-5.4", display_name="Worker"),
        WorkflowAgentBinding(agent_id="A2", tool_name="claude", model_name="sonnet", display_name="Synth"),
    )
    e2e = run_workflow_e2e(
        tmp_path,
        monkeypatch,
        script_content=script,
        run_spec=make_run_spec(pool=pool, task="Handoff synthesis"),
        recorder=WorkflowSessionRecorder(_resolve),
    )
    assert e2e.project.status == WorkflowStatus.COMPLETED
    assert e2e.project.metrics.failed_agents == 0
    assert len(synth_prompts) == 1
    assert "OBS_TAIL_MARKER" in synth_prompts[0] and "VER_TAIL_MARKER" in synth_prompts[0]
    for marker in ("observation", "logic_chain", "caveats", "conclusion", "verification", "OBS_TAIL_MARKER", "VER_TAIL_MARKER"):
        assert marker in e2e.project.result


# ---------------------------------------------------------------------------
# Area 4: Feishu Interactive Config, Cascade Callbacks & Persistence (F7, F9, F10)
# ---------------------------------------------------------------------------


def test_tier1_wf_and_wf_config_emit_card2_team_editor_with_presets_and_cascade(
    tmp_path: Path,
    monkeypatch,
) -> None:
    graph = build_workflow_e2e_graph(tmp_path, monkeypatch)
    try:
        with patch(
            "src.workflow_engine.tool_registry.get_available_tools",
            return_value={"codex": "Codex", "claude": "Claude", "traex": "Traex"},
        ):
            graph.send_text("/wf")
            card_wf = graph.fake_im.latest_card()
            assert card_wf is not None
            assert_card_2_invariants(card_wf, label="wf-editor")
            assert collect_card_selects(card_wf), "Expected interactive select_static cascades in /wf config card"
            assert collect_card_buttons(card_wf), "Expected action buttons in /wf config card"

            graph.send_text("/wf_config")
            card_cfg = graph.fake_im.latest_card()
            assert card_cfg is not None
            assert_card_2_invariants(card_cfg, label="wf-config")
    finally:
        graph.close()


def test_tier1_card_cascade_tool_model_profile_effort_updates_synchronously(
    tmp_path: Path,
    monkeypatch,
) -> None:
    importlib.import_module("src.card.builders.workflow")
    graph = build_workflow_e2e_graph(tmp_path, monkeypatch)
    try:
        with patch(
            "src.workflow_engine.tool_registry.get_available_tools",
            return_value={"codex": "Codex", "traex": "Traex"},
        ):
            graph.send_text("/wf_config")
            msg_id = graph.fake_im.latest_card_message_id()
            card = graph.fake_im.latest_card()
            assert msg_id is not None and card is not None

            tool_sel = find_card_select(card, option_values=("traex",), name_tokens=("tool",))
            assert tool_sel is not None
            resp1, _, card = click_card_element(graph, message_id=msg_id, element=tool_sel, option="traex")
            assert resp1 is not None and (resp1.card is not None or resp1.toast is not None)
            assert card is not None

            effort_sel = find_card_select(card, option_values=("xhigh", "high"), name_tokens=("effort",))
            if effort_sel is not None:
                opt = next(o["value"] for o in effort_sel["options"] if o.get("value") in {"xhigh", "high"})
                resp2, _, card = click_card_element(graph, message_id=msg_id, element=effort_sel, option=opt)
                assert resp2 is not None and card is not None
    finally:
        graph.close()


def test_tier1_interactive_add_same_tool_model_with_distinct_effort_coexists_in_editor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    importlib.import_module("src.card.builders.workflow")
    graph = build_workflow_e2e_graph(tmp_path, monkeypatch)
    try:
        with patch(
            "src.workflow_engine.tool_registry.get_available_tools",
            return_value={"codex": "Codex"},
        ):
            graph.send_text("/wf_config")
            cfg_id = graph.fake_im.latest_card_message_id()
            card = graph.fake_im.latest_card()
            assert cfg_id and card is not None
            assert_card_2_invariants(card, label="team-config-initial")

            effort_sel = find_card_select(card, option_values=("low",), name_tokens=("effort",))
            if effort_sel is not None:
                _, _, card = click_card_element(graph, message_id=cfg_id, element=effort_sel, option="low")
            add_btn = find_card_button(card, action_tokens=("add", "append", "new_agent"), text_tokens=("添加", "新增", "+"))
            assert add_btn is not None, "Team config editor must expose an Add Agent button"
            _, _, card = click_card_element(graph, message_id=cfg_id, element=add_btn)
            assert card is not None

            effort_sels = [
                s
                for s in collect_card_selects(card)
                if any(isinstance(o, dict) and str(o.get("value", "")).lower() == "high" for o in (s.get("options") or []))
            ]
            assert effort_sels, "Expected effort select_static with 'high' option"
            _, _, card = click_card_element(graph, message_id=cfg_id, element=effort_sels[-1], option="high")
            assert card is not None
            if "A2" not in extract_card_text(card):
                add_btn2 = find_card_button(card, action_tokens=("add", "append", "new_agent"), text_tokens=("添加", "新增", "+"))
                if add_btn2 is not None:
                    _, _, card = click_card_element(graph, message_id=cfg_id, element=add_btn2)
                    assert card is not None

            assert_card_2_invariants(card, label="team-config-multi-effort")
            txt = extract_card_text(card)
            assert "A1" in txt and "A2" in txt
            assert "low" in txt.lower() and "high" in txt.lower()
    finally:
        graph.close()


def test_tier1_per_chat_project_workflow_team_config_persists_across_reload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    graph = build_workflow_e2e_graph(tmp_path, monkeypatch)
    try:
        with patch(
            "src.workflow_engine.tool_registry.get_available_tools",
            return_value={"codex": "Codex", "claude": "Claude"},
        ):
            graph.send_text("/wf_config")
            cfg_id = graph.fake_im.latest_card_message_id()
            cfg_card = graph.fake_im.latest_card()
            assert cfg_id and cfg_card
            preset_sel = find_card_select(cfg_card, option_values=("collaborative", "review_team"))
            if preset_sel is not None:
                opt = next(o["value"] for o in preset_sel["options"] if o.get("value") in {"collaborative", "review_team"})
                _, _, cfg_card = click_card_element(graph, message_id=cfg_id, element=preset_sel, option=opt)
            save_btn = find_card_button(cfg_card, action_tokens=("save", "confirm", "apply"), text_tokens=("保存", "应用", "确认"))
            assert save_btn is not None
            click_card_element(graph, message_id=cfg_id, element=save_btn)

        in_mem_state = graph.project.chat_states.get("oc_e2e")
        expected_cfg = getattr(in_mem_state, "workflow_team_config", None)
        assert expected_cfg, "Expected non-empty workflow_team_config after saving via /wf_config"

        reloaded = graph.reload_project_manager()
        saved = reloaded.get_project_for_chat(graph.project.project_id, "oc_e2e")
        assert saved is not None
        saved_state = saved.chat_states.get("oc_e2e")
        saved_cfg = getattr(saved_state, "workflow_team_config", None)
        assert saved_cfg and saved_cfg == expected_cfg
    finally:
        graph.close()


def test_tier1_ux_previews_match_card2_workflow_team_and_message_markers() -> None:
    root = Path(__file__).resolve().parents[1]
    team_html = root / "ux" / "workflow-team-config.html"
    msg_html = root / "ux" / "workflow-message-card.html"
    assert team_html.exists(), "ux/workflow-team-config.html must exist for M3/F7"
    assert msg_html.exists()
    team_text = team_html.read_text(encoding="utf-8")
    assert "effort" in team_text.lower()
    assert "preset" in team_text.lower() or "预设" in team_text


def test_tier1_card_action_payload_whitelist_allows_config_fields() -> None:
    raw = {
        "action": "wf_select_effort",
        "chat_id": "oc_e2e",
        "project_id": "demo",
        "agent_id": "A2",
        "preset_id": "collaborative",
        "role": "worker",
        "tool_name": "traex",
        "model_name": "kimi-k2",
        "profile": "max",
        "effort": "high",
        "config_session_id": "cfg_1",
    }
    filtered = filter_workflow_button_value(raw)
    assert filtered == raw


# ---------------------------------------------------------------------------
# Area 5: Progress, COT, Completion Cards & Reporting (F6, F8, F10)
# ---------------------------------------------------------------------------


def test_tier1_progress_coalescer_flushes_leading_edge_immediately_then_coalesces_burst() -> None:
    received: list[str] = []
    first_seen = threading.Event()
    second_seen = threading.Event()

    def _on_progress(snap: str) -> None:
        received.append(snap)
        if snap == "snap-1":
            first_seen.set()
        if snap == "snap-4":
            second_seen.set()

    coalescer = ProgressCoalescer(on_progress=_on_progress, debounce_s=0.4)
    try:
        coalescer.enqueue("snap-1")
        assert first_seen.wait(timeout=0.18), "Leading-edge flush must fire immediately without waiting full debounce_s"
        coalescer.enqueue("snap-2")
        coalescer.enqueue("snap-3")
        coalescer.enqueue("snap-4")
        assert second_seen.wait(timeout=0.9)
        assert received[-1] == "snap-4"
        assert "snap-2" not in received
    finally:
        coalescer.stop()


def test_tier1_generation_card_shows_effort_roles_and_stop_button(tmp_path: Path, monkeypatch) -> None:
    graph = build_workflow_e2e_graph(tmp_path, monkeypatch)
    try:
        wf_handler = graph.client._handler_ctx.handlers["workflow"]
        pool = freeze_agent_pool([
            WorkflowAgentBinding(agent_id="A1", tool_name="codex", model_name="gpt-5.4", display_name="Scout", effort="low", role="scout"),
            WorkflowAgentBinding(agent_id="A2", tool_name="claude", model_name="sonnet", display_name="Reviewer", effort="high", role="verifier"),
        ])
        renderer = WorkflowGenerationRenderer(
            requirement="Audit security",
            agent_pool=pool,
            orchestrator_agent_id="A2",
            orchestrator_was_auto=True,
        )
        frag = renderer.render(current_activity="规划节点中", elapsed_seconds=3)
        wf_handler._inject_workflow_stop_button(frag, "oc_e2e", graph.project.project_id, is_running=True)
        card = wf_handler._build_workflow_card_from_renderer_data(frag)
        assert_card_2_invariants(card, label="gen-card")
        txt = extract_card_text(card)
        assert "low" in txt.lower() and "high" in txt.lower()
        assert any(btn.get("behaviors", [{}])[0].get("value", {}).get("action") == "workflow_stop_running" for btn in collect_card_buttons(card))
    finally:
        graph.close()


def test_tier1_progress_and_completion_cards_display_effort_and_remove_terminal_stop(tmp_path: Path, monkeypatch) -> None:
    script = _single_agent_js("effort-cards", "codex", "worker", "b1", conclusion="Build finished.")
    graph = build_workflow_e2e_graph(tmp_path, monkeypatch)
    try:
        wf_handler = graph.client._handler_ctx.handlers["workflow"]
        WorkflowSessionRecorder(lambda _inv: ScriptedRule(output="built")).patch(monkeypatch)
        pool = freeze_agent_pool([WorkflowAgentBinding(agent_id="A1", tool_name="codex", model_name="gpt-5.4", display_name="Coder", effort="xhigh", role="worker")])
        run_spec = make_run_spec(pool=pool, task="Build with xhigh")
        script_path = tmp_path / "effort_cards.js"
        script_path.write_text(script, encoding="utf-8")
        engine = graph.client._handler_ctx.workflow_engine_manager.get_or_create("oc_e2e", str(graph.repo_dir), engine_name="wf")
        init_msg = wf_handler._show_initial_workflow_progress_card(
            card_message_id=None, chat_id="oc_e2e", wf_project=WorkflowProject(status=WorkflowStatus.RUNNING, requirement=run_spec.task), origin_message_id="om_origin",
        )
        callbacks = wf_handler._build_workflow_callbacks(init_msg, "oc_e2e", graph.project, requirement=run_spec.task)
        project = engine.execute_workflow(str(script_path), callbacks=callbacks, run_spec=run_spec)
        assert project.status == WorkflowStatus.COMPLETED
        assert any(ag.effort == "xhigh" for ph in project.phases for ag in ph.agents)
        cards = graph.fake_im.all_cards()
        for idx, c in enumerate(cards):
            assert_card_2_invariants(c, label=f"exec-card-{idx}")
        final_card = cards[-1]
        assert "xhigh" in extract_card_text(final_card).lower()
        assert not any(btn.get("behaviors", [{}])[0].get("value", {}).get("action") == "workflow_stop_running" for btn in collect_card_buttons(final_card))
    finally:
        graph.close()


def test_tier1_cot_stream_carries_workflow_agent_effort_events(tmp_path: Path, monkeypatch) -> None:
    script = _single_agent_js("cot-effort", "codex", "scout", "a1")
    events = (
        ACPEvent(event_type=ACPEventType.THOUGHT_CHUNK, text="COT_THOUGHT_MARKER"),
        ACPEvent(event_type=ACPEventType.TOOL_CALL_START, tool_call=ToolCallInfo(id="t1", title="rg", kind="search", status="in_progress", content="src/")),
        ACPEvent(event_type=ACPEventType.TOOL_CALL_DONE, tool_call=ToolCallInfo(id="t1", title="rg", kind="search", status="completed", content="found")),
        ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="COT_TEXT_MARKER"),
    )
    graph = build_workflow_e2e_graph(tmp_path, monkeypatch, cot_enabled=True)
    try:
        WorkflowSessionRecorder(lambda _inv: ScriptedRule(output="cot-done", events=events)).patch(monkeypatch)
        wf_handler = graph.client._handler_ctx.handlers["workflow"]
        pool = freeze_agent_pool([WorkflowAgentBinding(agent_id="A1", tool_name="codex", model_name="gpt-5.4", display_name="Scout", effort="xhigh", role="scout")])
        run_spec = make_run_spec(pool=pool, task="COT stream")
        script_path = tmp_path / "cot_effort.js"
        script_path.write_text(script, encoding="utf-8")
        engine = graph.client._handler_ctx.workflow_engine_manager.get_or_create("oc_e2e", str(graph.repo_dir), engine_name="wf-cot")
        init_msg = wf_handler._show_initial_workflow_progress_card(
            card_message_id=None, chat_id="oc_e2e", wf_project=WorkflowProject(status=WorkflowStatus.RUNNING, requirement=run_spec.task), origin_message_id="om_origin",
        )
        callbacks = wf_handler._build_workflow_callbacks(init_msg, "oc_e2e", graph.project, requirement=run_spec.task)
        project = engine.execute_workflow(str(script_path), callbacks=callbacks, run_spec=run_spec)
        assert project.status == WorkflowStatus.COMPLETED
        assert graph.fake_cot.requests
        wire_text = json.dumps(graph.fake_cot.events, ensure_ascii=False)
        assert "COT_THOUGHT_MARKER" in wire_text or "COT_TEXT_MARKER" in wire_text
        assert re.search(r"\bxhigh\b", wire_text, re.IGNORECASE)
        assert "scout" in wire_text.lower()
    finally:
        graph.close()


def test_tier1_html_and_markdown_reports_include_effort_profile_role_and_upload_to_feishu(tmp_path: Path, monkeypatch) -> None:
    script = _single_agent_js("report-effort", "traex", "auditor-role", "r1")
    graph = build_workflow_e2e_graph(tmp_path, monkeypatch)
    try:
        WorkflowSessionRecorder(lambda _inv: ScriptedRule(output="report-body")).patch(monkeypatch)
        wf_handler = graph.client._handler_ctx.handlers["workflow"]
        pool = freeze_agent_pool([WorkflowAgentBinding(agent_id="A1", tool_name="traex", model_name="kimi-k2", display_name="Rev", profile="turbo-prof", effort="xhigh", role="auditor-role")])
        run_spec = make_run_spec(pool=pool, task="Report upload")
        script_path = tmp_path / "report_effort.js"
        script_path.write_text(script, encoding="utf-8")
        engine = graph.client._handler_ctx.workflow_engine_manager.get_or_create("oc_e2e", str(graph.repo_dir), engine_name="wf-rpt")
        init_msg = wf_handler._show_initial_workflow_progress_card(
            card_message_id=None, chat_id="oc_e2e", wf_project=WorkflowProject(status=WorkflowStatus.RUNNING, requirement=run_spec.task), origin_message_id="om_origin",
        )
        callbacks = wf_handler._build_workflow_callbacks(init_msg, "oc_e2e", graph.project, requirement=run_spec.task)
        project = engine.execute_workflow(str(script_path), callbacks=callbacks, run_spec=run_spec)
        assert project.status == WorkflowStatus.COMPLETED
        assert graph.fake_im.uploaded_files and graph.fake_im.file_replies
        uploaded = next(iter(graph.fake_im.uploaded_files.values()))["text"]
        assert "xhigh" in uploaded and "turbo-prof" in uploaded and "auditor-role" in uploaded
        assert any(ag.status == AgentStatus.DONE for ph in project.phases for ag in ph.agents)
    finally:
        graph.close()
