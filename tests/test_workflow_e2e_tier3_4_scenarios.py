"""Tier 3 pairwise combinations and Tier 4 real-world E2E scenarios for /wf."""

from __future__ import annotations

import ast
import importlib
import json
import threading
from pathlib import Path
from unittest.mock import patch

from src.acp.models import ACPEvent, ACPEventType, ToolCallInfo
from src.acp.sync_adapter import ACPStartupError
from src.mode import InteractionMode
from src.workflow_engine.agent_pool import WorkflowAgentBinding, freeze_agent_pool
from src.workflow_engine.models import WorkflowProject, WorkflowStatus
from src.workflow_engine.progress_coalescer import ProgressCoalescer
from src.workflow_engine.renderer import WorkflowGenerationRenderer
from tests.helpers.workflow_e2e_harness import (
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
# Tier 3: Pairwise Cross-Feature Combinations (6 tests)
# ---------------------------------------------------------------------------


def test_tier3_pairwise_multi_tool_multi_effort_verify_loop_and_lossless_fanout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    script = """\
export const meta = {
  name: "t3-multi-tool-effort-verify",
  description: "Pairwise F2+F3+F4+F5",
  phases: [
    { title: "Fanout", detail: "Multi-effort workers" },
    { title: "Verify", detail: "Independent review" },
  ],
  maxConcurrent: 3,
  tools: ["codex", "claude", "agy"],
  patterns: ["fanout", "verify"],
  agentPlan: [
    { node: "w-low", role: "scout", agentId: "A1" },
    { node: "w-xhigh", role: "worker", agentId: "A2" },
    { node: "synth", role: "synthesizer", agentId: "A3" },
    { node: "verify", role: "verifier", agentId: "A4" },
  ],
};
export default async function main() {
  await phase("Fanout");
  const synth = await fanout("Build feature X", [
    { prompt: "Scan ${input}", agentId: "A1", role: "scout", label: "scan", timeout: 30 },
    { prompt: "Implement ${input}", agentId: "A2", role: "worker", label: "impl", timeout: 30 },
  ], { synthesizerAgentId: "A3", timeout: 30 });
  await phase("Verify");
  const checked = await verify(synth, {
    verifiers: [{ agentId: "A4", role: "verifier" }],
    reviserAgentId: "A2",
    maxRounds: 2,
    timeout: 30,
  });
  return { card_summary: { verdict: "passed", conclusion: "Verified synthesis." }, result: checked };
}
"""
    verify_turn = [0]
    synth_prompts: list[str] = []

    def _resolve(inv: SessionInvocation) -> ScriptedRule:
        if inv.agent_type == "agy":
            verify_turn[0] += 1
            if verify_turn[0] == 1:
                return ScriptedRule(
                    output=json.dumps(
                        {
                            "approve": False,
                            "issues": [{"severity": "major", "description": "add guard", "evidence": "fn()"}],
                            "extra_note": "allowed",
                        }
                    )
                )
            return ScriptedRule(output=json.dumps({"approve": True, "issues": []}))
        if inv.agent_type == "claude":
            synth_prompts.append(inv.prompt)
            return ScriptedRule(output="SYNTH_" + inv.prompt[-40:])
        if inv.model_name == "gpt-5.4/low":
            return ScriptedRule(output=("S" * 2400) + "SCOUT_TAIL")
        return ScriptedRule(output=("W" * 2400) + "WORKER_TAIL")

    pool = (
        WorkflowAgentBinding(agent_id="A1", tool_name="codex", model_name="gpt-5.4", display_name="Scout", effort="low", role="scout"),
        WorkflowAgentBinding(agent_id="A2", tool_name="codex", model_name="gpt-5.4", display_name="Worker", effort="xhigh", role="worker"),
        WorkflowAgentBinding(agent_id="A3", tool_name="claude", model_name="sonnet", display_name="Synth", effort="high", role="synthesizer"),
        WorkflowAgentBinding(agent_id="A4", tool_name="agy", model_name="gemini-4-hydrogen", display_name="Verifier", effort="high", role="verifier"),
    )
    e2e = run_workflow_e2e(
        tmp_path,
        monkeypatch,
        script_content=script,
        run_spec=make_run_spec(pool=pool, task="Pairwise F2+F3+F4+F5"),
        recorder=WorkflowSessionRecorder(_resolve),
    )
    assert e2e.project.status == WorkflowStatus.COMPLETED
    assert "SCOUT_TAIL" in synth_prompts[0] and "WORKER_TAIL" in synth_prompts[0]
    models_seen = {inv.model_name for inv in e2e.recorder.invocations}
    assert {"gpt-5.4/low", "gpt-5.4/xhigh", "sonnet/high", "gemini-4-hydrogen/high"}.issubset(models_seen)


def test_tier3_pairwise_saved_preset_cascade_adjustment_and_live_feishu_launch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    exec_script = """\
export const meta = {
  name: "t3-saved-preset-launch",
  description: "Launch saved cascade config",
  phases: [{ title: "Run", detail: "Execute" }],
  maxConcurrent: 2,
  tools: ["traex", "codex"],
  patterns: ["sequence"],
  agentPlan: [{ node: "n1", role: "worker", agentId: "A1" }],
};
export default async function main() {
  const out = await agent("Task", { agentId: "A1", role: "worker", label: "n1", timeout: 30 });
  return { card_summary: { verdict: "passed", conclusion: "Preset launched cleanly." }, result: out };
}
"""

    def _resolve(inv: SessionInvocation) -> ScriptedRule:
        if "Workflow Script Generation Task" in inv.prompt:
            return ScriptedRule(output=exec_script)
        lowered = inv.prompt.lower()
        if "approve" in lowered or "issues" in lowered or "verify" in lowered:
            return ScriptedRule(output=json.dumps({"approve": True, "issues": []}))
        return ScriptedRule(output="saved-preset-ok")

    recorder = WorkflowSessionRecorder(_resolve).patch(monkeypatch)
    graph = build_workflow_e2e_graph(tmp_path, monkeypatch)
    try:
        with patch(
            "src.workflow_engine.tool_registry.get_available_tools",
            return_value={"traex": "Traex", "codex": "Codex"},
        ):
            graph.send_text("/wf_config")
            cfg_msg_id = graph.fake_im.latest_card_message_id()
            cfg_card = graph.fake_im.latest_card()
            assert cfg_msg_id and cfg_card
            assert collect_card_selects(cfg_card)

            # Drive interactive cascade select_static callbacks and Save button via card.action.trigger
            preset_sel = find_card_select(cfg_card, option_values=("collaborative", "review_team"))
            if preset_sel is not None:
                opt = next(o["value"] for o in preset_sel["options"] if o.get("value") in {"collaborative", "review_team"})
                _, _, cfg_card = click_card_element(graph, message_id=cfg_msg_id, element=preset_sel, option=opt)
            tool_sel = find_card_select(cfg_card, option_values=("traex",), name_tokens=("tool",))
            if tool_sel is not None:
                _, _, cfg_card = click_card_element(graph, message_id=cfg_msg_id, element=tool_sel, option="traex")
            effort_sel = find_card_select(cfg_card, option_values=("high", "xhigh"), name_tokens=("effort",))
            if effort_sel is not None:
                eff_opt = next(o["value"] for o in effort_sel["options"] if o.get("value") in {"high", "xhigh"})
                _, _, cfg_card = click_card_element(graph, message_id=cfg_msg_id, element=effort_sel, option=eff_opt)

            save_btn = find_card_button(cfg_card, action_tokens=("save", "confirm", "apply"), text_tokens=("保存", "应用", "确认"))
            assert save_btn is not None, "Team config card must expose a Save button"
            click_card_element(graph, message_id=cfg_msg_id, element=save_btn)

            reloaded = graph.reload_project_manager()
            saved_proj = reloaded.get_project_for_chat(graph.project.project_id, "oc_e2e")
            assert saved_proj is not None
            saved_state = saved_proj.chat_states.get("oc_e2e")
            assert saved_state is not None and getattr(saved_state, "workflow_team_config", None)
            graph.client._project_manager = reloaded
            graph.send_text("/wf 使用已保存的协作团队执行构建")

        assert recorder.invocations
        assert any(inv.agent_type == "traex" for inv in recorder.invocations)
        for idx, c in enumerate(graph.fake_im.all_cards()):
            assert_card_2_invariants(c, label=f"t3-saved-launch-{idx}")
    finally:
        graph.close()


def test_tier3_pairwise_generation_coalescing_stop_button_and_failure_preservation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    graph = build_workflow_e2e_graph(tmp_path, monkeypatch)
    try:
        wf_handler = graph.client._handler_ctx.handlers["workflow"]
        pool = freeze_agent_pool(
            [WorkflowAgentBinding(agent_id="A1", tool_name="codex", model_name="gpt-5.4", display_name="Codex", effort="high", role="worker")]
        )
        gen_renderer = WorkflowGenerationRenderer(
            requirement="Check security",
            agent_pool=pool,
            orchestrator_agent_id="A1",
        )
        gen_frag = gen_renderer.render(current_activity="生成脚本", elapsed_seconds=1)
        wf_handler._inject_workflow_stop_button(gen_frag, "oc_e2e", graph.project.project_id, is_running=True)
        gen_card = wf_handler._build_workflow_card_from_renderer_data(gen_frag)
        assert_card_2_invariants(gen_card, label="t3-gen")
        assert any(
            b.get("behaviors", [{}])[0].get("value", {}).get("action") == "workflow_stop_running"
            for b in collect_card_buttons(gen_card)
        )

        # Leading-edge coalescing + failure conclusion preservation
        flushed: list[str] = []
        first_hit = threading.Event()

        def _on_prog(val: str) -> None:
            flushed.append(val)
            first_hit.set()

        coalescer = ProgressCoalescer(on_progress=_on_prog, debounce_s=0.4)
        try:
            coalescer.enqueue("gen-tick-1")
            assert first_hit.wait(timeout=0.18)
        finally:
            coalescer.stop()

        err_card = wf_handler._build_error_card("internal_error", detail="All pool fallbacks exhausted: codex 503")
        assert_card_2_invariants(err_card, label="t3-err")
        assert not any(
            b.get("behaviors", [{}])[0].get("value", {}).get("action") == "workflow_stop_running"
            for b in collect_card_buttons(err_card)
        )
    finally:
        graph.close()


def test_tier3_pairwise_parallel_concurrency_8_multi_effort_workers_max_concurrent_4(
    tmp_path: Path,
    monkeypatch,
) -> None:
    script = """\
export const meta = {
  name: "t3-parallel-8-workers",
  description: "8 concurrent workers with maxConcurrent=4",
  phases: [{ title: "Parallel", detail: "Barrier test" }],
  maxConcurrent: 4,
  tools: ["codex"],
  patterns: ["parallel"],
  agentPlan: [{ node: "p", role: "worker", agentId: "A1" }],
};
export default async function main() {
  const tasks = [
    { prompt: "job-0", agentId: "A1", label: "j0", timeout: 30 },
    { prompt: "job-1", agentId: "A2", label: "j1", timeout: 30 },
    { prompt: "job-2", agentId: "A3", label: "j2", timeout: 30 },
    { prompt: "job-3", agentId: "A1", label: "j3", timeout: 30 },
    { prompt: "job-4", agentId: "A2", label: "j4", timeout: 30 },
    { prompt: "job-5", agentId: "A3", label: "j5", timeout: 30 },
    { prompt: "job-6", agentId: "A1", label: "j6", timeout: 30 },
    { prompt: "job-7", agentId: "A2", label: "j7", timeout: 30 },
  ];
  const results = await parallel(tasks);
  return { count: results.length, results };
}
"""
    barrier = threading.Barrier(4, timeout=4.0)
    first_wave_lock = threading.Lock()
    first_wave_count = [0]

    def _resolve(inv: SessionInvocation) -> ScriptedRule:
        use_barrier = False
        with first_wave_lock:
            if first_wave_count[0] < 4:
                first_wave_count[0] += 1
                use_barrier = True
        return ScriptedRule(
            output=f"done:{inv.model_name}:{inv.prompt}",
            barrier=barrier if use_barrier else None,
        )

    pool = (
        WorkflowAgentBinding(agent_id="A1", tool_name="codex", model_name="gpt-5.4", display_name="Low", effort="low"),
        WorkflowAgentBinding(agent_id="A2", tool_name="codex", model_name="gpt-5.4", display_name="Med", effort="medium"),
        WorkflowAgentBinding(agent_id="A3", tool_name="codex", model_name="gpt-5.4", display_name="High", effort="high"),
    )
    e2e = run_workflow_e2e(
        tmp_path,
        monkeypatch,
        script_content=script,
        run_spec=make_run_spec(pool=pool, task="8 parallel multi-effort workers"),
        recorder=WorkflowSessionRecorder(_resolve),
    )
    assert e2e.project.status == WorkflowStatus.COMPLETED
    assert len(e2e.recorder.invocations) == 8
    assert {inv.model_name for inv in e2e.recorder.invocations} == {
        "gpt-5.4/low",
        "gpt-5.4/medium",
        "gpt-5.4/high",
    }


def test_tier3_pairwise_cot_failure_fallback_to_keyed_agent_cards_preserving_effort(
    tmp_path: Path,
    monkeypatch,
) -> None:
    script = """\
export const meta = {
  name: "t3-cot-fallback",
  description: "COT create fails -> keyed agent card fallback",
  phases: [{ title: "Run", detail: "Fallback" }],
  maxConcurrent: 1,
  tools: ["codex"],
  patterns: ["sequence"],
  agentPlan: [{ node: "a1", role: "worker", agentId: "A1" }],
};
export default async function main() {
  const out = await agent("Work", { agentId: "A1", role: "worker", label: "a1", timeout: 30 });
  return { result: out };
}
"""
    events = (
        ACPEvent(event_type=ACPEventType.THOUGHT_CHUNK, text="Inspecting auth module"),
        ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(id="tc1", title="rg", kind="search", status="in_progress", content="src/"),
        ),
        ACPEvent(
            event_type=ACPEventType.TOOL_CALL_DONE,
            tool_call=ToolCallInfo(id="tc1", title="rg", kind="search", status="completed", content="ok"),
        ),
        ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="fallback-agent-stream-line"),
    )
    graph = build_workflow_e2e_graph(tmp_path, monkeypatch, cot_enabled=True)
    graph.fake_cot.fail_on_create = True
    try:
        WorkflowSessionRecorder(
            lambda _inv: ScriptedRule(output="fallback-agent-card-ok", events=events)
        ).patch(monkeypatch)
        wf_handler = graph.client._handler_ctx.handlers["workflow"]
        pool = freeze_agent_pool(
            [WorkflowAgentBinding(agent_id="A1", tool_name="codex", model_name="gpt-5.4", display_name="Worker", effort="xhigh", role="worker")]
        )
        run_spec = make_run_spec(pool=pool, task="COT fallback")
        script_path = tmp_path / "t3_cot_fallback.js"
        script_path.write_text(script, encoding="utf-8")
        engine = graph.client._handler_ctx.workflow_engine_manager.get_or_create("oc_e2e", str(graph.repo_dir), engine_name="wf-cot-fb")
        init_card_id = wf_handler._show_initial_workflow_progress_card(
            card_message_id=None,
            chat_id="oc_e2e",
            wf_project=WorkflowProject(status=WorkflowStatus.RUNNING, requirement=run_spec.task),
            origin_message_id="om_origin",
        )
        callbacks = wf_handler._build_workflow_callbacks(init_card_id, "oc_e2e", graph.project, requirement=run_spec.task)
        project = engine.execute_workflow(str(script_path), callbacks=callbacks, run_spec=run_spec)
        assert project.status == WorkflowStatus.COMPLETED
        cards = graph.fake_im.all_cards()
        assert len(cards) >= 2
        for idx, c in enumerate(cards):
            assert_card_2_invariants(c, label=f"cot-fb-{idx}")
        assert any("A1" in extract_card_text(c) and "a1" in extract_card_text(c) for c in cards)
        assert any("xhigh" in extract_card_text(c).lower() for c in cards)
    finally:
        graph.close()


def test_tier3_pairwise_topic_followup_routes_workflow_preserving_chat_codex_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    graph = build_workflow_e2e_graph(tmp_path, monkeypatch)
    try:
        graph.client._mode_manager.enter_mode("oc_e2e", InteractionMode.CODEX, project_id=graph.project.project_id)
        graph.client._thread_manager.register("om_wf_topic_root", "oc_e2e", graph.project.project_id, mode="workflow")
        wf_handler = graph.client._handler_ctx.handlers["workflow"]
        seen_followups: list[str] = []
        with patch.object(wf_handler, "handle_message", side_effect=lambda mid, cid, txt, proj=None: seen_followups.append(txt)):
            graph.send_text("继续补充性能基准测试", root_id="om_wf_topic_root", parent_id="om_wf_topic_root")

        assert seen_followups == ["继续补充性能基准测试"]
        assert graph.client._mode_manager.get_mode("oc_e2e", project_id=graph.project.project_id) == InteractionMode.CODEX
    finally:
        graph.close()


# ---------------------------------------------------------------------------
# Tier 4: Real-World Application Scenarios (6 scenarios)
# ---------------------------------------------------------------------------


def test_tier4_scenario1_full_teamwork_multi_tool_feature_dev_and_review_cycle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    script = """\
export const meta = {
  name: "scenario1-full-teamwork",
  description: "Full multi-tool feature dev and review",
  phases: [
    { title: "Explore", detail: "Scout codebase" },
    { title: "Build", detail: "Parallel implementation" },
    { title: "Review", detail: "Independent verification" },
  ],
  maxConcurrent: 4,
  tools: ["codex", "claude", "agy", "dsh"],
  patterns: ["fanout", "verify"],
  agentPlan: [
    { node: "scout", role: "scout", agentId: "A1" },
    { node: "impl-a", role: "worker", agentId: "A2" },
    { node: "impl-b", role: "worker", agentId: "A3" },
    { node: "verify", role: "verifier", agentId: "A4" },
    { node: "synth", role: "synthesizer", agentId: "A5" },
  ],
};
export default async function main() {
  await phase("Explore");
  const scout = await agent("Map auth module", { agentId: "A1", role: "scout", label: "scout-1", timeout: 30 });
  await phase("Build");
  const impl = await fanout(scout, [
    { prompt: "Implement OAuth: ${input}", agentId: "A2", role: "worker", label: "worker-oauth", timeout: 30 },
    { prompt: "Write tests: ${input}", agentId: "A3", role: "worker", label: "worker-tests", timeout: 30 },
  ], { synthesizerAgentId: "A5", timeout: 30 });
  await phase("Review");
  const reviewed = await verify(impl, {
    verifiers: [{ agentId: "A4", role: "verifier" }],
    reviserAgentId: "A2",
    maxRounds: 2,
    timeout: 30,
  });
  return {
    card_summary: {
      verdict: reviewed.accepted ? "passed" : "needs_attention",
      conclusion: "Feature dev and review cycle completed.",
    },
    result: reviewed,
  };
}
"""

    def _resolve(inv: SessionInvocation) -> ScriptedRule:
        if inv.agent_type == "agy":
            return ScriptedRule(output=json.dumps({"approve": True, "issues": []}))
        return ScriptedRule(output=f"ok:{inv.agent_type}:{inv.model_name}")

    pool = (
        WorkflowAgentBinding(agent_id="A1", tool_name="codex", model_name="gpt-5.4", display_name="Scout", effort="low", role="scout"),
        WorkflowAgentBinding(agent_id="A2", tool_name="codex", model_name="gpt-5.4", display_name="Impl", effort="xhigh", role="worker"),
        WorkflowAgentBinding(agent_id="A3", tool_name="claude", model_name="sonnet", display_name="Tester", effort="high", role="worker"),
        WorkflowAgentBinding(agent_id="A4", tool_name="agy", model_name="gemini-4-hydrogen", display_name="Verifier", effort="high", role="verifier"),
        WorkflowAgentBinding(agent_id="A5", tool_name="dsh", model_name="deepseek-r1", display_name="Synth", effort="high", role="synthesizer"),
    )
    e2e = run_workflow_e2e(
        tmp_path,
        monkeypatch,
        script_content=script,
        run_spec=make_run_spec(pool=pool, task="Full teamwork feature cycle"),
        recorder=WorkflowSessionRecorder(_resolve),
        generate_reports=True,
    )
    assert e2e.project.status == WorkflowStatus.COMPLETED
    agents = [ag for ph in e2e.project.phases for ag in ph.agents]
    assert any(ag.effort == "xhigh" and ag.role == "worker" for ag in agents)
    assert e2e.report_files is not None
    md_text = Path(e2e.report_files.markdown_path).read_text(encoding="utf-8")
    for marker in ("gpt-5.4", "low", "xhigh", "sonnet", "gemini-4-hydrogen", "deepseek-r1"):
        assert marker in md_text


def test_tier4_scenario2_preset_quickstart_vs_custom_feishu_cascade_launch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    wf_js = """\
export const meta = {
  name: "scenario2-dual-track",
  description: "Quickstart and custom config",
  phases: [{ title: "Work", detail: "Execute" }],
  maxConcurrent: 2,
  tools: ["codex", "claude"],
  patterns: ["sequence"],
  agentPlan: [{ node: "w1", role: "worker", agentId: "A1" }],
};
export default async function main() {
  const out = await agent("Execute", { agentId: "A1", role: "worker", label: "w1", timeout: 30 });
  return { card_summary: { verdict: "passed", conclusion: "Dual-track run completed." }, result: out };
}
"""

    def _resolve(inv: SessionInvocation) -> ScriptedRule:
        if "Workflow Script Generation Task" in inv.prompt:
            return ScriptedRule(output=wf_js)
        lowered = inv.prompt.lower()
        if "approve" in lowered or "issues" in lowered or "verify" in lowered:
            return ScriptedRule(output=json.dumps({"approve": True, "issues": []}))
        return ScriptedRule(output="dual-track-ok")

    recorder = WorkflowSessionRecorder(_resolve).patch(monkeypatch)
    graph = build_workflow_e2e_graph(tmp_path, monkeypatch)
    try:
        with patch(
            "src.workflow_engine.tool_registry.get_available_tools",
            return_value={"codex": "Codex", "claude": "Claude"},
        ):
            graph.send_text("/wf 快速修复登录超时问题")
            graph.send_text("/wf_config")
            cfg_id = graph.fake_im.latest_card_message_id()
            cfg_card = graph.fake_im.latest_card()
            assert cfg_id is not None and cfg_card is not None
            selects = collect_card_selects(cfg_card)
            assert selects
            opt = selects[0]["options"][0]["value"]
            click_card_element(graph, message_id=cfg_id, element=selects[0], option=opt)
        assert recorder.invocations
        for idx, c in enumerate(graph.fake_im.all_cards()):
            assert_card_2_invariants(c, label=f"s2-card-{idx}")
    finally:
        graph.close()


def test_tier4_scenario3_cross_member_failover_vs_terminal_failed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    script = """\
export const meta = {
  name: "scenario3-failover",
  description: "Circuit breaker failover and terminal failure",
  phases: [{ title: "Run", detail: "Execute" }],
  maxConcurrent: 1,
  tools: ["traex", "codex"],
  patterns: ["sequence"],
  agentPlan: [{ node: "w1", role: "worker", agentId: "A1" }],
};
export default async function main() {
  const r = await agent("Work", { agentId: "A1", fallbackAgentIds: ["A2"], label: "w1", timeout: 30 });
  if (r && r.error) return { error: r.error, status: "failed" };
  return { result: r };
}
"""
    rec_ok = WorkflowSessionRecorder(lambda inv: ScriptedRule(output=f"ok:{inv.agent_type}"))
    rec_ok.fail_on_start(
        lambda t, _m: t == "traex",
        ACPStartupError("traex unavailable", agent_cmd="traex", agent_args=["--acp"], cwd=str(tmp_path)),
    )
    pool = (
        WorkflowAgentBinding(agent_id="A1", tool_name="traex", model_name="kimi-k2", display_name="Traex"),
        WorkflowAgentBinding(agent_id="A2", tool_name="codex", model_name="gpt-5.4", display_name="Codex", effort="high"),
    )
    spec = make_run_spec(pool=pool, task="Scenario 3 failover")
    res_ok = run_workflow_e2e(tmp_path, monkeypatch, script_content=script, run_spec=spec, recorder=rec_ok)
    assert res_ok.project.status == WorkflowStatus.COMPLETED

    rec_fail = WorkflowSessionRecorder()
    rec_fail.fail_on_start(
        lambda _t, _m: True,
        ACPStartupError("all backends unavailable", agent_cmd="codex", agent_args=["--acp"], cwd=str(tmp_path)),
    )
    res_fail = run_workflow_e2e(tmp_path, monkeypatch, script_content=script, run_spec=spec, recorder=rec_fail)
    assert res_fail.project.status == WorkflowStatus.FAILED
    assert res_fail.project.error


def test_tier4_scenario4_long_synthesis_over_3500_chars_cot_and_html_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    script = """\
export const meta = {
  name: "scenario4-long-synthesis",
  description: "Long output synthesis + COT + HTML attachment",
  phases: [{ title: "Synthesize", detail: "Long workers" }],
  maxConcurrent: 2,
  tools: ["codex", "claude"],
  patterns: ["fanout"],
  agentPlan: [
    { node: "w1", role: "worker", agentId: "A1" },
    { node: "s1", role: "synthesizer", agentId: "A2" },
  ],
};
export default async function main() {
  const synth = await fanout("Audit", [
    { prompt: "Part 1", agentId: "A1", role: "worker", label: "p1", timeout: 30 },
    { prompt: "Part 2", agentId: "A1", role: "worker", label: "p2", timeout: 30 },
  ], { synthesizerAgentId: "A2", timeout: 30 });
  return {
    card_summary: { verdict: "passed", conclusion: "Long synthesis archived in HTML report." },
    result: synth,
  };
}
"""
    long_a = ("SectionA-" * 420) + "LONG_TAIL_ALPHA"
    long_b = ("SectionB-" * 420) + "LONG_TAIL_BETA"
    synth_prompts: list[str] = []

    def _resolve(inv: SessionInvocation) -> ScriptedRule:
        if inv.agent_type == "claude":
            synth_prompts.append(inv.prompt)
            return ScriptedRule(output=inv.prompt)
        return ScriptedRule(output=long_a if "Part 1" in inv.prompt else long_b)

    graph = build_workflow_e2e_graph(tmp_path, monkeypatch, cot_enabled=True)
    try:
        WorkflowSessionRecorder(_resolve).patch(monkeypatch)
        wf_handler = graph.client._handler_ctx.handlers["workflow"]
        pool = freeze_agent_pool(
            [
                WorkflowAgentBinding(agent_id="A1", tool_name="codex", model_name="gpt-5.4", display_name="Worker", effort="high", role="worker"),
                WorkflowAgentBinding(agent_id="A2", tool_name="claude", model_name="sonnet", display_name="Synth", effort="high", role="synthesizer"),
            ]
        )
        run_spec = make_run_spec(pool=pool, task="Long synthesis report")
        script_path = tmp_path / "scenario4.js"
        script_path.write_text(script, encoding="utf-8")
        engine = graph.client._handler_ctx.workflow_engine_manager.get_or_create("oc_e2e", str(graph.repo_dir), engine_name="wf-s4")
        init_card_id = wf_handler._show_initial_workflow_progress_card(
            card_message_id=None,
            chat_id="oc_e2e",
            wf_project=WorkflowProject(status=WorkflowStatus.RUNNING, requirement=run_spec.task),
            origin_message_id="om_origin",
        )
        callbacks = wf_handler._build_workflow_callbacks(init_card_id, "oc_e2e", graph.project, requirement=run_spec.task)
        project = engine.execute_workflow(str(script_path), callbacks=callbacks, run_spec=run_spec)
        assert project.status == WorkflowStatus.COMPLETED
        assert project.metrics.failed_agents == 0
        assert len(synth_prompts) == 1
        assert "LONG_TAIL_ALPHA" in synth_prompts[0] and "LONG_TAIL_BETA" in synth_prompts[0]
        assert graph.fake_im.uploaded_files
        assert graph.fake_im.file_replies
        html_body = next(iter(graph.fake_im.uploaded_files.values()))["text"]
        assert "LONG_TAIL_ALPHA" in html_body and "LONG_TAIL_BETA" in html_body
        for idx, c in enumerate(graph.fake_im.all_cards()):
            assert_card_2_invariants(c, label=f"s4-card-{idx}")
    finally:
        graph.close()


def test_tier4_scenario5_multi_chat_project_isolation_and_topic_persistence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    graph = build_workflow_e2e_graph(tmp_path, monkeypatch, chat_id="oc_alpha")
    try:
        beta_dir = tmp_path / "beta"
        beta_dir.mkdir(parents=True, exist_ok=True)
        graph.client._project_manager.create_project("beta", "BetaProj", str(beta_dir), chat_id="oc_beta")
        proj_beta = graph.client._project_manager.get_project_for_chat("beta", "oc_beta")
        assert proj_beta is not None
        graph.client._project_manager.set_active_project("oc_beta", proj_beta.project_id)

        with patch(
            "src.workflow_engine.tool_registry.get_available_tools",
            return_value={"codex": "Codex", "claude": "Claude"},
        ):
            # Configure and save oc_alpha via /wf_config
            graph.send_text("/wf_config", chat_id="oc_alpha")
            alpha_id = graph.fake_im.latest_card_message_id()
            alpha_card = graph.fake_im.latest_card()
            assert alpha_id and alpha_card
            sel_a = find_card_select(alpha_card, option_values=("review_team",))
            if sel_a is not None:
                _, _, alpha_card = click_card_element(graph, message_id=alpha_id, element=sel_a, option="review_team", chat_id="oc_alpha")
            save_a = find_card_button(alpha_card, action_tokens=("save", "confirm", "apply"), text_tokens=("保存", "应用", "确认"))
            assert save_a is not None
            click_card_element(graph, message_id=alpha_id, element=save_a, chat_id="oc_alpha")

            # Configure and save oc_beta independently via /wf_config
            graph.send_text("/wf_config", chat_id="oc_beta")
            beta_id = graph.fake_im.latest_card_message_id()
            beta_card = graph.fake_im.latest_card()
            assert beta_id and beta_card
            sel_b = find_card_select(beta_card, option_values=("collaborative",))
            if sel_b is not None:
                _, _, beta_card = click_card_element(graph, message_id=beta_id, element=sel_b, option="collaborative", chat_id="oc_beta")
            save_b = find_card_button(beta_card, action_tokens=("save", "confirm", "apply"), text_tokens=("保存", "应用", "确认"))
            assert save_b is not None
            click_card_element(graph, message_id=beta_id, element=save_b, chat_id="oc_beta")

        reloaded = graph.reload_project_manager()
        alpha_saved = reloaded.get_project_for_chat(graph.project.project_id, "oc_alpha").chat_states["oc_alpha"].workflow_team_config
        beta_saved = reloaded.get_project_for_chat(proj_beta.project_id, "oc_beta").chat_states["oc_beta"].workflow_team_config
        assert alpha_saved and beta_saved
        assert alpha_saved != beta_saved or alpha_saved.get("preset") == "review_team"
    finally:
        graph.close()


def test_tier4_scenario6_ux_preview_parity_and_card_import_boundary_audit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    importlib.import_module("src.card.builders.workflow")

    root = Path(__file__).resolve().parents[1]
    card_root = root / "src" / "card"

    # 1. Strict unidirectional card import boundary audit:
    # render/builders NEVER import delivery/session; delivery NEVER imports session.
    for folder in ("render", "builders"):
        for py_file in (card_root / folder).rglob("*.py"):
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        parts = alias.name.split(".")
                        assert "delivery" not in parts and "session" not in parts, f"Illegal import in {py_file}: {alias.name}"
                elif isinstance(node, ast.ImportFrom) and node.module:
                    parts = node.module.split(".")
                    assert "delivery" not in parts and "session" not in parts, f"Illegal import in {py_file}: {node.module}"

    for py_file in (card_root / "delivery").rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "session" not in alias.name.split("."), f"Illegal session import in {py_file}: {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert "session" not in node.module.split("."), f"Illegal session import in {py_file}: {node.module}"

    # 2. UX preview parity with live Card 2.0 /wf_config card
    team_html = (root / "ux" / "workflow-team-config.html").read_text(encoding="utf-8")
    msg_html = (root / "ux" / "workflow-message-card.html").read_text(encoding="utf-8")
    assert "effort" in team_html.lower()
    assert "workflow" in msg_html.lower()

    graph = build_workflow_e2e_graph(tmp_path, monkeypatch)
    try:
        with patch(
            "src.workflow_engine.tool_registry.get_available_tools",
            return_value={"codex": "Codex", "claude": "Claude"},
        ):
            graph.send_text("/wf_config")
            card = graph.fake_im.latest_card()
            assert card is not None
            assert_card_2_invariants(card, label="scenario6-builder")
            assert collect_card_selects(card) and collect_card_buttons(card)
            assert "effort" in extract_card_text(card).lower()
    finally:
        graph.close()
