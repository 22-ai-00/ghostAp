"""Tier 2 boundary, error-handling, and edge-case E2E tests for /wf (F1-F10)."""

from __future__ import annotations

import importlib
import json
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from src.acp.sync_adapter import ACPStartupError
from src.card.events.payloads import filter_workflow_button_value
from src.workflow_engine.agent_pool import WorkflowAgentBinding, freeze_agent_pool
from src.workflow_engine.models import (
    AgentProgress,
    AgentStatus,
    PhaseProgress,
    WorkflowProject,
    WorkflowStatus,
)
from src.workflow_engine.renderer_completion import render_completion_cards
from tests.helpers.workflow_e2e_harness import (
    FakeACPClientSession,
    ScriptedRule,
    SessionInvocation,
    WorkflowSessionRecorder,
    assert_card_2_invariants,
    build_workflow_e2e_graph,
    extract_card_text,
    find_card_button,
    make_run_spec,
    run_workflow_e2e,
)

# ---------------------------------------------------------------------------
# Area 1: Capability, Tool, Effort & Requirement Validation Boundaries (F1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool_name", "which_ok"),
    [
        ("nonexistent_tool", True),
        ("", True),
        ("codex", False),
    ],
)
def test_tier2_validate_binding_rejects_unknown_and_unavailable_tools(
    monkeypatch,
    tool_name: str,
    which_ok: bool,
) -> None:
    from src.workflow_engine.tool_registry import validate_agent_binding_spec

    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}" if which_ok else None)
    available = {"codex": "Codex"} if which_ok else {}
    ok, err = validate_agent_binding_spec(tool_name, available_tools=available)
    assert ok is False
    assert err and isinstance(err, str)


@pytest.mark.parametrize("model_only_tool", ["coco", "gemini", "grok", "aiden"])
def test_tier2_validate_binding_rejects_explicit_effort_on_model_only_backends(
    monkeypatch,
    model_only_tool: str,
) -> None:
    from src.workflow_engine.tool_registry import validate_agent_binding_spec

    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    ok, err = validate_agent_binding_spec(
        model_only_tool,
        model_name="default",
        effort="high",
        available_tools={model_only_tool: model_only_tool},
    )
    assert ok is False
    assert err is not None and ("effort" in err.lower() or "不支持" in err)


@pytest.mark.parametrize("effort_tool", ["codex", "claude", "agy", "traex", "dsh"])
def test_tier2_validate_binding_rejects_invalid_effort_value(
    monkeypatch,
    effort_tool: str,
) -> None:
    from src.workflow_engine.tool_registry import validate_agent_binding_spec

    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    ok, err = validate_agent_binding_spec(
        effort_tool,
        model_name="default",
        effort="ultra_mega_invalid",
        available_tools={effort_tool: effort_tool},
    )
    assert ok is False
    assert err is not None


@pytest.mark.parametrize("bad_req", ["/wf ab", "/wf " + ("x" * 4001)])
def test_tier2_wf_command_rejects_short_and_oversized_requirements(
    tmp_path: Path,
    monkeypatch,
    bad_req: str,
) -> None:
    graph = build_workflow_e2e_graph(tmp_path, monkeypatch)
    try:
        graph.send_text(bad_req)
        cards = graph.fake_im.all_cards()
        assert cards
        for idx, c in enumerate(cards):
            assert_card_2_invariants(c, label=f"bad-req-{idx}")
        txt = extract_card_text(cards[-1])
        assert "过短" in txt or "过长" in txt
    finally:
        graph.close()


def test_tier2_engine_blocks_script_spoofing_unconfirmed_agent_or_tool(
    tmp_path: Path,
    monkeypatch,
) -> None:
    script = """\
export const meta = {
  name: "spoof-check",
  description: "Attempt to invoke unconfirmed agent",
  phases: [{ title: "Run", detail: "Run" }],
  maxConcurrent: 1,
  tools: ["codex"],
  patterns: ["sequence"],
  agentPlan: [{ node: "n1", role: "worker", agentId: "A1" }],
};
export default async function main() {
  const res = await agent("Spoof", { agentId: "A99", tool: "claude", label: "spoof", timeout: 30 });
  if (res && res.error) return { error: res.error, status: "failed" };
  return { result: res };
}
"""
    recorder = WorkflowSessionRecorder()
    pool = (WorkflowAgentBinding(agent_id="A1", tool_name="codex", model_name="gpt-5.4", display_name="Codex"),)
    e2e = run_workflow_e2e(
        tmp_path,
        monkeypatch,
        script_content=script,
        run_spec=make_run_spec(pool=pool, task="Spoof test"),
        recorder=recorder,
    )
    assert e2e.project.status == WorkflowStatus.FAILED
    assert len(recorder.invocations) == 0


# ---------------------------------------------------------------------------
# Area 2: Pool Uniqueness, Duplicate Rejection & Model-Only Safety (F2, F3)
# ---------------------------------------------------------------------------


def test_tier2_freeze_pool_rejects_exact_duplicate_tool_model_profile_effort() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        freeze_agent_pool(
            [
                WorkflowAgentBinding(agent_id="A1", tool_name="codex", model_name="gpt-5.4", display_name="One", effort="high"),
                WorkflowAgentBinding(agent_id="A2", tool_name="codex", model_name="gpt-5.4", display_name="Two", effort="high"),
            ]
        )


def test_tier2_freeze_pool_accepts_same_tool_model_differing_only_by_effort_or_profile() -> None:
    pool = freeze_agent_pool(
        [
            WorkflowAgentBinding(agent_id="A1", tool_name="codex", model_name="gpt-5.4", display_name="Default"),
            WorkflowAgentBinding(agent_id="A2", tool_name="codex", model_name="gpt-5.4", display_name="Low", effort="low"),
            WorkflowAgentBinding(agent_id="A3", tool_name="codex", model_name="gpt-5.4", display_name="High", effort="high"),
            WorkflowAgentBinding(agent_id="A4", tool_name="traex", model_name="kimi-k2", display_name="Std", profile="standard"),
            WorkflowAgentBinding(agent_id="A5", tool_name="traex", model_name="kimi-k2", display_name="Max", profile="max"),
        ]
    )
    assert len(pool) == 5


def test_tier2_freeze_pool_rejects_empty_duplicate_id_and_invalid_agent_id() -> None:
    with pytest.raises(ValueError):
        freeze_agent_pool([], require_nonempty=True)

    with pytest.raises(ValueError):
        freeze_agent_pool(
            [
                WorkflowAgentBinding(agent_id="A1", tool_name="codex", model_name="m1", display_name="1"),
                WorkflowAgentBinding(agent_id="A1", tool_name="claude", model_name="m2", display_name="2"),
            ]
        )

    with pytest.raises(ValueError):
        WorkflowAgentBinding(agent_id="bad id!", tool_name="codex", model_name="m1", display_name="Bad")


def test_tier2_compaction_and_failover_preserve_model_name(tmp_path: Path, monkeypatch) -> None:
    from src.agent_session.model_diagnostics import _apply_compaction_once
    from src.agent_session.wrappers import ModelFailureAwareSession

    captured_models: list[str | None] = []

    def _builder(**kwargs):
        captured_models.append(kwargs.get("model_name"))
        return FakeACPClientSession(
            agent_type=kwargs["agent_type"],
            cwd=kwargs["cwd"],
            model_name=kwargs.get("model_name"),
        )

    old_sess = FakeACPClientSession(agent_type="codex", cwd=str(tmp_path), model_name="gpt-5.4/xhigh")
    rebuilt = _apply_compaction_once(session=old_sess, session_builder=_builder, startup_timeout_s=2.0)
    assert rebuilt is not None
    assert captured_models == ["gpt-5.4/xhigh"]

    # Failover check via ModelFailureAwareSession
    failover_models: list[str | None] = []

    def _fake_sync_acp(**kwargs):
        failover_models.append(kwargs.get("model_name"))
        return FakeACPClientSession(
            agent_type=kwargs["agent_type"],
            cwd=kwargs["cwd"],
            model_name=kwargs.get("model_name"),
        )

    monkeypatch.setattr("src.agent_session.wrappers.SyncACPSession", _fake_sync_acp)
    inner = FakeACPClientSession(agent_type="codex", cwd=str(tmp_path), model_name="gpt-5.4/xhigh")
    inner._agent_args = ["--acp", "--model", "gpt-5.4/xhigh"]
    wrapper = ModelFailureAwareSession(inner=inner)
    ok = wrapper._do_failover(from_model="gpt-5.4/xhigh", to_model="gpt-5.4/low")
    assert ok is True
    assert failover_models == ["gpt-5.4/low"]


def test_tier2_model_only_backends_preserve_model_without_effort_suffix() -> None:
    for tool_name in ("coco", "aiden", "gemini", "grok"):
        binding = WorkflowAgentBinding(
            agent_id="A1",
            tool_name=tool_name,
            model_name="custom-v2",
            display_name=tool_name,
        )
        assert binding.effective_model_name() == "custom-v2"


# ---------------------------------------------------------------------------
# Area 3: Member Failure, Circuit Breaker, Fallback Exhaustion (F4, F5)
# ---------------------------------------------------------------------------


def test_tier2_acp_startup_circuit_breaker_isolates_failing_tool_and_allows_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    script = """\
export const meta = {
  name: "cross-member-fallback",
  description: "Fallback from broken traex to healthy codex",
  phases: [{ title: "Run", detail: "Execute with fallback" }],
  maxConcurrent: 2,
  tools: ["traex", "codex"],
  patterns: ["sequence"],
  agentPlan: [{ node: "w1", role: "worker", agentId: "A1" }],
};
export default async function main() {
  const first = await agent("Do work", { agentId: "A1", fallbackAgentIds: ["A2"], label: "w1", timeout: 30 });
  const second = await agent("Do more", { agentId: "A1", fallbackAgentIds: ["A2"], label: "w2", timeout: 30 });
  return { first, second };
}
"""
    recorder = WorkflowSessionRecorder(lambda inv: ScriptedRule(output=f"recovered-by-{inv.agent_type}"))
    recorder.fail_on_start(
        lambda tool, _m: tool == "traex",
        ACPStartupError("traex binary missing", agent_cmd="traex", agent_args=["--acp"], cwd=str(tmp_path)),
    )
    pool = (
        WorkflowAgentBinding(agent_id="A1", tool_name="traex", model_name="kimi-k2", display_name="Traex"),
        WorkflowAgentBinding(agent_id="A2", tool_name="codex", model_name="gpt-5.4", display_name="Codex", effort="high"),
    )
    e2e = run_workflow_e2e(
        tmp_path,
        monkeypatch,
        script_content=script,
        run_spec=make_run_spec(pool=pool, task="Cross member fallback"),
        recorder=recorder,
    )
    assert e2e.project.status == WorkflowStatus.COMPLETED
    assert any(inv.agent_type == "codex" and inv.model_name == "gpt-5.4/high" for inv in recorder.invocations)


def test_tier2_exhausted_fallback_and_permanent_failure_reach_terminal_failed_and_recover_next_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fail_script = """\
export const meta = {
  name: "exhausted-fail",
  description: "All members fail",
  phases: [{ title: "Run", detail: "Fail" }],
  maxConcurrent: 1,
  tools: ["codex", "claude"],
  patterns: ["sequence"],
  agentPlan: [{ node: "w1", role: "worker", agentId: "A1" }],
};
export default async function main() {
  const res = await agent("Run", { agentId: "A1", fallbackAgentIds: ["A2"], label: "w1", timeout: 30 });
  if (res && res.error) return { error: res.error, status: "failed" };
  return { result: res };
}
"""
    recorder = WorkflowSessionRecorder(
        lambda _inv: ScriptedRule(raise_on_prompt=RuntimeError("permanent backend failure"))
    )
    pool = (
        WorkflowAgentBinding(agent_id="A1", tool_name="codex", model_name="gpt-5.4", display_name="Codex"),
        WorkflowAgentBinding(agent_id="A2", tool_name="claude", model_name="sonnet", display_name="Claude"),
    )
    spec = make_run_spec(pool=pool, task="Exhausted failure")
    e2e1 = run_workflow_e2e(tmp_path, monkeypatch, script_content=fail_script, run_spec=spec, recorder=recorder)
    assert e2e1.project.status == WorkflowStatus.FAILED
    assert e2e1.project.error and "permanent backend failure" in e2e1.project.error

    # Next run on fresh script succeeds cleanly
    ok_script = fail_script.replace("exhausted-fail", "recovered-run")
    e2e2 = run_workflow_e2e(
        tmp_path,
        monkeypatch,
        script_content=ok_script,
        run_spec=spec,
        recorder=WorkflowSessionRecorder(lambda _inv: ScriptedRule(output="healthy-again")),
    )
    assert e2e2.project.status == WorkflowStatus.COMPLETED


def test_tier2_script_gen_fallback_retains_multi_agent_collaboration_when_pool_ge_2(
    tmp_path: Path,
    monkeypatch,
) -> None:
    importlib.import_module("src.workflow_engine.teamwork")
    from src.feishu.handlers.workflow import _WorkflowLifecycleOwner
    from src.workflow_engine.models import PendingWorkflow
    from src.workflow_engine.storage import workflow_scripts_dir

    WorkflowSessionRecorder(lambda _inv: ScriptedRule(output="not valid js")).patch(monkeypatch)
    graph = build_workflow_e2e_graph(tmp_path, monkeypatch)
    try:
        wf_handler = graph.client._handler_ctx.handlers["workflow"]
        pool = freeze_agent_pool(
            [
                WorkflowAgentBinding(agent_id="A1", tool_name="codex", model_name="gpt-5.4", display_name="Codex", effort="low", role="scout"),
                WorkflowAgentBinding(agent_id="A2", tool_name="claude", model_name="sonnet", display_name="Claude", effort="high", role="worker"),
                WorkflowAgentBinding(agent_id="A3", tool_name="agy", model_name="gemini-4-hydrogen", display_name="AGY", effort="high", role="verifier"),
            ]
        )
        engine = graph.client._handler_ctx.workflow_engine_manager.get_or_create("oc_e2e", str(graph.repo_dir), engine_name="wf-fb")
        sdir = Path(workflow_scripts_dir(str(graph.repo_dir)))
        sdir.mkdir(parents=True, exist_ok=True)
        out_path = sdir / "fallback_test.js"
        owner = _WorkflowLifecycleOwner(
            session_key="fb_key",
            initiator_user_id="ou_e2e",
            chat_id="oc_e2e",
            project_id=graph.project.project_id,
            root_path=str(graph.repo_dir),
            source_script_path=str(out_path),
        )
        engine._script_generation_owner = owner
        engine._project = WorkflowProject(
            status=WorkflowStatus.GENERATING_SCRIPT,
            pending=PendingWorkflow(
                requirement="Refactor auth",
                initiator_user_id="ou_e2e",
                engine_session_key="fb_key",
                selected_tools=["codex", "claude", "agy"],
                orchestrator_agent="codex",
                agent_pool=pool,
                orchestrator_agent_id="A1",
            ),
        )
        with patch(
            "src.workflow_engine.tool_registry.get_available_tools",
            return_value={"codex": "Codex", "claude": "Claude", "agy": "AGY"},
        ):
            written_path, _meta = wf_handler._generate_script_via_ai(
                "Refactor auth",
                str(graph.repo_dir),
                engine=engine,
                output_path=str(out_path),
            )
        script = Path(written_path).read_text(encoding="utf-8")
        assert sum(1 for aid in ("A1", "A2", "A3") if aid in script) >= 2
        assert any(prim in script for prim in ("fanout", "verify", "sequence", "parallel"))
    finally:
        graph.close()


def test_tier2_partial_fanout_failure_preserves_partial_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    script = """\
export const meta = {
  name: "partial-fanout",
  description: "One worker fails while two succeed",
  phases: [{ title: "Fanout", detail: "Partial" }],
  maxConcurrent: 3,
  tools: ["codex", "claude"],
  patterns: ["fanout"],
  agentPlan: [{ node: "f1", role: "worker", agentId: "A1" }],
};
export default async function main() {
  const out = await fanout("input", [
    { prompt: "ok-1", agentId: "A1", label: "w-ok-1", timeout: 30 },
    { prompt: "fail-2", agentId: "A2", label: "w-fail-2", timeout: 30 },
    { prompt: "ok-3", agentId: "A1", label: "w-ok-3", timeout: 30 },
  ], { synthesizerAgentId: "A1", timeout: 30 });
  return { result: out };
}
"""

    def _resolve(inv: SessionInvocation) -> ScriptedRule:
        if "fail-2" in inv.prompt:
            return ScriptedRule(raise_on_prompt=RuntimeError("worker-2 crashed"))
        return ScriptedRule(output=f"success:{inv.prompt[:12]}")

    pool = (
        WorkflowAgentBinding(agent_id="A1", tool_name="codex", model_name="gpt-5.4", display_name="Codex"),
        WorkflowAgentBinding(agent_id="A2", tool_name="claude", model_name="sonnet", display_name="Claude"),
    )
    e2e = run_workflow_e2e(
        tmp_path,
        monkeypatch,
        script_content=script,
        run_spec=make_run_spec(pool=pool, task="Partial fanout"),
        recorder=WorkflowSessionRecorder(_resolve),
    )
    assert e2e.project.status == WorkflowStatus.COMPLETED
    payload = json.loads(e2e.project.result)["result"]
    assert payload.get("partialFailures")


def test_tier2_transient_retry_exhaustion_sanitizes_secrets_and_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    script = """\
export const meta = {
  name: "secret-sanitize",
  description: "Sanitize leaked secrets in error",
  phases: [{ title: "Run", detail: "Run" }],
  maxConcurrent: 1,
  tools: ["codex"],
  patterns: ["sequence"],
  agentPlan: [{ node: "n1", role: "worker", agentId: "A1" }],
};
export default async function main() {
  const r = await agent("Run", { agentId: "A1", label: "n1", timeout: 30 });
  if (r && r.error) return { error: r.error, status: "failed" };
  return { result: r };
}
"""
    recorder = WorkflowSessionRecorder(
        lambda _inv: ScriptedRule(
            raise_on_prompt=RuntimeError("503 Service Unavailable token=sk-supersecret999 /home/admin/.env")
        )
    )
    pool = (WorkflowAgentBinding(agent_id="A1", tool_name="codex", model_name="gpt-5.4", display_name="Codex"),)
    e2e = run_workflow_e2e(
        tmp_path,
        monkeypatch,
        script_content=script,
        run_spec=make_run_spec(pool=pool, task="Error sanitization"),
        recorder=recorder,
    )
    assert e2e.project.status == WorkflowStatus.FAILED
    assert "sk-supersecret999" not in (e2e.project.error or "")


# ---------------------------------------------------------------------------
# Area 4: Card 2.0 Budgets, Table Limit, Forgery Stripping & Isolation (F7-F9)
# ---------------------------------------------------------------------------


def test_tier2_card_budget_27kb_and_180_node_ceiling_enforced_under_stress(
    tmp_path: Path,
    monkeypatch,
) -> None:
    graph = build_workflow_e2e_graph(tmp_path, monkeypatch)
    try:
        wf_handler = graph.client._handler_ctx.handlers["workflow"]
        big_findings = [
            {"severity": "high", "text": f"Finding #{i}: " + ("detail " * 40)}
            for i in range(50)
        ]
        proj = WorkflowProject(
            workflow_id="wf_stress_1",
            name="Stress",
            status=WorkflowStatus.COMPLETED,
            result=json.dumps({"card_summary": {"verdict": "needs_attention", "conclusion": "Done", "findings": big_findings}}),
            phases=[
                PhaseProgress(
                    title="Heavy Phase",
                    agents=[
                        AgentProgress(
                            label=f"worker-{i}",
                            agent_id=f"A{(i % 8) + 1}",
                            tool="codex",
                            model="gpt-5.4",
                            status=AgentStatus.DONE,
                            result="X" * 1000,
                        )
                        for i in range(24)
                    ],
                )
            ],
        )
        frags = render_completion_cards(proj)
        for idx, frag in enumerate(frags):
            full_card = wf_handler._build_workflow_card_from_renderer_data(frag)
            assert_card_2_invariants(full_card, label=f"stress-page-{idx}")
    finally:
        graph.close()


def test_tier2_card_table_limit_5_converts_excess_markdown_tables(
    tmp_path: Path,
    monkeypatch,
) -> None:
    graph = build_workflow_e2e_graph(tmp_path, monkeypatch)
    try:
        wf_handler = graph.client._handler_ctx.handlers["workflow"]
        seven_tables = "\n\n".join(
            f"| ColA | ColB |\n|---|---|\n| r{i} | v{i} |" for i in range(7)
        )
        frag = {
            "header": {"title": {"tag": "plain_text", "content": "Workflow · 表格压力"}, "template": "green"},
            "elements": [{"tag": "markdown", "content": seven_tables}],
        }
        guarded = wf_handler._build_workflow_card_from_renderer_data(frag)
        assert_card_2_invariants(guarded, label="table-limit-7")
    finally:
        graph.close()


def test_tier2_filter_workflow_button_value_strips_forged_keys() -> None:
    forged = {
        "action": "wf_save_team",
        "chat_id": "oc_e2e",
        "project_id": "p1",
        "effort": "high",
        "confirmed": True,
        "admin": True,
        "override_budget": 99999,
        "shell_cmd": "rm -rf /",
    }
    cleaned = filter_workflow_button_value(forged)
    assert cleaned["action"] == "wf_save_team"
    assert cleaned["effort"] == "high"
    for bad_key in ("confirmed", "admin", "override_budget", "shell_cmd"):
        assert bad_key not in cleaned


def test_tier2_card_callback_rejects_cross_chat_project_id_and_unauthorized_stop(
    tmp_path: Path,
    monkeypatch,
) -> None:
    graph = build_workflow_e2e_graph(
        tmp_path,
        monkeypatch,
        chat_id="oc_chat_a",
        extra_settings={"allowed_user_ids": "ou_e2e,ou_intruder"},
    )
    try:
        repo_b = tmp_path / "repo_b"
        repo_b.mkdir(parents=True, exist_ok=True)
        graph.client._project_manager.create_project("proj_b", "ProjB", str(repo_b), chat_id="oc_chat_b")
        proj_b = graph.client._project_manager.get_project_for_chat("proj_b", "oc_chat_b")
        assert proj_b is not None

        origin_a = graph.send_text("/wf_help", chat_id="oc_chat_a", sender_id="ou_e2e")
        wf_handler = graph.client._handler_ctx.handlers["workflow"]
        engine_a = graph.client._handler_ctx.workflow_engine_manager.get_or_create("oc_chat_a", str(graph.repo_dir), engine_name="wf-a")
        engine_a._project = WorkflowProject(workflow_id="wf_auth_a", status=WorkflowStatus.RUNNING, initiator_user_id="ou_e2e")
        engine_b = graph.client._handler_ctx.workflow_engine_manager.get_or_create("oc_chat_b", str(repo_b), engine_name="wf-b")
        engine_b._project = WorkflowProject(workflow_id="wf_auth_b", status=WorkflowStatus.RUNNING, initiator_user_id="ou_e2e")

        init_card_id = wf_handler._show_initial_workflow_progress_card(
            card_message_id=None, chat_id="oc_chat_a", wf_project=engine_a.project, origin_message_id=origin_a,
        )
        # 1. Forged card.action.trigger from non-initiator intruder in chat_a + direct handler auth check
        before_cards = len(graph.fake_im.all_cards())
        with patch.object(type(engine_a), "is_running", property(lambda _s: True)):
            _, sub_intruder = graph.click_card(
                message_id=init_card_id,
                chat_id="oc_chat_a",
                operator_id="ou_intruder",
                value={"action": "workflow_stop_running", "chat_id": "oc_chat_a", "project_id": graph.project.project_id},
            )
            assert engine_a.project.status == WorkflowStatus.RUNNING
            assert sub_intruder == 0 or any(
                "只有" in extract_card_text(c) or "拒绝" in extract_card_text(c) or "无权" in extract_card_text(c)
                for c in graph.fake_im.all_cards()[before_cards:]
            )
            with patch("src.thread.get_current_sender_id", return_value="ou_intruder"):
                wf_handler.stop_workflow(origin_a, "oc_chat_a", graph.project)
            assert engine_a.project.status == WorkflowStatus.RUNNING
            assert any("只有" in extract_card_text(c) for c in graph.fake_im.all_cards()[before_cards:])

        # 2. Forged cross-chat project_id in card.action.trigger from chat_a cannot touch chat_b's project or workflow
        with patch.object(type(engine_b), "is_running", property(lambda _s: True)):
            _, sub_cross = graph.click_card(
                message_id=init_card_id,
                chat_id="oc_chat_a",
                operator_id="ou_e2e",
                value={"action": "workflow_stop_running", "chat_id": "oc_chat_a", "project_id": proj_b.project_id},
            )
        assert sub_cross == 1
        assert engine_b.project.status == WorkflowStatus.RUNNING
        assert graph.client._project_manager.get_project_for_chat(proj_b.project_id, "oc_chat_a") is None
    finally:
        graph.close()


def test_tier2_failed_workflow_completion_card_preserves_actionable_conclusion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    graph = build_workflow_e2e_graph(tmp_path, monkeypatch)
    try:
        wf_handler = graph.client._handler_ctx.handlers["workflow"]
        failed_proj = WorkflowProject(
            workflow_id="wf_fail_1",
            name="AuthAudit",
            status=WorkflowStatus.FAILED,
            error="Verification failed: auth token check missing in middleware",
            phases=[
                PhaseProgress(
                    title="Verify",
                    agents=[
                        AgentProgress(
                            label="verify-1",
                            agent_id="A2",
                            tool="agy",
                            model="gemini-4-hydrogen",
                            status=AgentStatus.FAILED,
                            error="auth token check missing",
                        )
                    ],
                )
            ],
        )
        frags = render_completion_cards(failed_proj)
        assert frags
        card = wf_handler._build_workflow_card_from_renderer_data(frags[0])
        assert_card_2_invariants(card, label="failed-conclusion")
        text = extract_card_text(card)
        assert "任务已完成" not in text
        assert "auth token check missing" in text
    finally:
        graph.close()


def test_tier2_corrupt_saved_preset_recovers_cleanly_without_crash(
    tmp_path: Path,
    monkeypatch,
) -> None:
    graph = build_workflow_e2e_graph(tmp_path, monkeypatch)
    try:
        raw = json.loads(graph.projects_config_file.read_text(encoding="utf-8"))
        assert raw.get("projects")
        for p in raw["projects"].values():
            chat_states = p.setdefault("chat_states", {})
            cs = chat_states.setdefault("oc_e2e", {})
            cs["workflow_team_config"] = "corrupt-not-a-dict"
        graph.projects_config_file.write_text(json.dumps(raw), encoding="utf-8")
        assert "corrupt-not-a-dict" in graph.projects_config_file.read_text(encoding="utf-8")

        reloaded = graph.reload_project_manager()
        proj = reloaded.get_project_for_chat(graph.project.project_id, "oc_e2e")
        assert proj is not None
        chat_state = proj.chat_states.get("oc_e2e")
        assert chat_state is not None and hasattr(chat_state, "workflow_team_config")
        assert chat_state.workflow_team_config is None
        graph.client._project_manager = reloaded
        graph.send_text("/wf_config")
        cfg_card = graph.fake_im.latest_card()
        assert cfg_card is not None
        assert_card_2_invariants(cfg_card, label="corrupt-preset-recovery")
        assert not any("未知命令" in extract_card_text(c) for c in [cfg_card])
    finally:
        graph.close()


# ---------------------------------------------------------------------------
# Area 5: Timeouts, Cancellation, /wf_status, /stop, /status, Dedup (F5, F8, F10)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_tier2_schema_repair_timeout_alignment_allows_2x_repair_under_short_floor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    script = """\
export const meta = {
  name: "schema-timeout-align",
  description: "Verify runtime.js requestTimeoutMs honors (1 + SCHEMA_RETRY_MAX)",
  phases: [{ title: "Extract", detail: "Schema call" }],
  maxConcurrent: 1,
  tools: ["codex"],
  patterns: ["sequence"],
  agentPlan: [{ node: "s1", role: "worker", agentId: "A1" }],
};
export default async function main() {
  const res = await agent("Extract JSON", {
    agentId: "A1",
    label: "s1",
    schema: { ok: false },
    timeout: 2,
  });
  return { res };
}
"""
    attempt = [0]

    def _resolve(_inv: SessionInvocation) -> ScriptedRule:
        attempt[0] += 1
        # Total across 1 initial + 2 schema repairs = 13.8s:
        # exceeds pre-M2 requestTimeoutMs() (2s + 10s grace = 12s),
        # fits M2 requestTimeoutMs() (2s * 3 + 10s grace = 16s),
        # while each single prompt (4.6s) stays below bridge effective_s (6s).
        if attempt[0] <= 2:
            return ScriptedRule(output="not json", cancel_wait_s=4.6)
        return ScriptedRule(output=json.dumps({"ok": True}), cancel_wait_s=4.6)

    pool = (WorkflowAgentBinding(agent_id="A1", tool_name="codex", model_name="gpt-5.4", display_name="Codex"),)
    recorder = WorkflowSessionRecorder(_resolve)
    e2e = run_workflow_e2e(
        tmp_path,
        monkeypatch,
        script_content=script,
        run_spec=make_run_spec(pool=pool, task="Schema retry timeout"),
        recorder=recorder,
        short_timeout_floor_s=2,
    )
    assert len(recorder.invocations) == 3
    assert e2e.project.status == WorkflowStatus.COMPLETED
    res = json.loads(e2e.project.result)["res"]
    assert res.get("ok") is True, res


def test_tier2_stop_wf_and_stop_button_cancel_during_execution(tmp_path: Path, monkeypatch) -> None:
    script = (
        'export const meta = { name: "cancel-mid-exec", description: "Cancel", phases: [{ title: "Work", detail: "Block" }], '
        'maxConcurrent: 1, tools: ["codex"], patterns: ["sequence"], agentPlan: [{ node: "w1", role: "worker", agentId: "A1" }] };\n'
        'export default async function main() { return { r: await agent("Long task", { agentId: "A1", label: "w1", timeout: 30 }) }; }\n'
    )
    graph = build_workflow_e2e_graph(tmp_path, monkeypatch, threaded_scheduler=True)
    release = threading.Event()
    try:
        entered = threading.Event()
        WorkflowSessionRecorder(lambda _inv: ScriptedRule(output="never", entered_event=entered, block_event=release)).patch(monkeypatch)
        wf_handler = graph.client._handler_ctx.handlers["workflow"]
        pool = freeze_agent_pool([WorkflowAgentBinding(agent_id="A1", tool_name="codex", model_name="gpt-5.4", display_name="Codex")])
        run_spec = make_run_spec(pool=pool, task="Cancel mid-run")
        script_path = tmp_path / "cancel_mid_exec.js"
        script_path.write_text(script, encoding="utf-8")
        engine = graph.client._handler_ctx.workflow_engine_manager.get_or_create("oc_e2e", str(graph.repo_dir), engine_name="wf-cancel")

        # 1. Cancel via real Card 2.0 `workflow_stop_running` button click (with valid message_linker provenance)
        origin_btn = graph.send_text("/wf_help")
        init_card_id = wf_handler._show_initial_workflow_progress_card(
            card_message_id=None, chat_id="oc_e2e",
            wf_project=WorkflowProject(status=WorkflowStatus.RUNNING, requirement=run_spec.task, initiator_user_id="ou_e2e"),
            origin_message_id=origin_btn,
        )
        cb1 = wf_handler._build_workflow_callbacks(init_card_id, "oc_e2e", graph.project, requirement=run_spec.task)
        t1 = threading.Thread(target=engine.execute_workflow, args=(str(script_path),), kwargs={"callbacks": cb1, "run_spec": run_spec}, daemon=True)
        t1.start()
        assert entered.wait(timeout=4.0)
        stop_btn = None
        deadline = time.monotonic() + 2.5
        while time.monotonic() < deadline and stop_btn is None:
            latest = graph.fake_im.latest_card(init_card_id)
            stop_btn = find_card_button(latest, action_tokens=("workflow_stop_running",)) if latest else None
            if stop_btn is None:
                time.sleep(0.05)
        assert stop_btn is not None, "Running progress card must expose workflow_stop_running button"
        _, submitted = graph.click_card(message_id=init_card_id, value=stop_btn["behaviors"][0]["value"], chat_id="oc_e2e", operator_id="ou_e2e")
        assert submitted == 1
        t1.join(timeout=4.0)
        assert not t1.is_alive() and engine.project.status == WorkflowStatus.CANCELLED
        assert engine._bridge is None or engine._bridge._process is None

        # 2. Cancel via `/stop_wf` text command
        entered.clear()
        init_card_id2 = wf_handler._show_initial_workflow_progress_card(
            card_message_id=None, chat_id="oc_e2e",
            wf_project=WorkflowProject(status=WorkflowStatus.RUNNING, requirement=run_spec.task, initiator_user_id="ou_e2e"),
            origin_message_id=origin_btn,
        )
        cb2 = wf_handler._build_workflow_callbacks(init_card_id2, "oc_e2e", graph.project, requirement=run_spec.task)
        t2 = threading.Thread(target=engine.execute_workflow, args=(str(script_path),), kwargs={"callbacks": cb2, "run_spec": run_spec}, daemon=True)
        t2.start()
        assert entered.wait(timeout=4.0)
        graph.send_text("/stop_wf")
        t2.join(timeout=4.0)
        assert not t2.is_alive() and engine.project.status == WorkflowStatus.CANCELLED
    finally:
        release.set()
        graph.close()


def test_tier2_wf_status_avoids_duplicate_card_spam_when_active_card_exists(tmp_path: Path, monkeypatch) -> None:
    from src.feishu.handlers.workflow import _WorkflowLifecycleOwner

    script = (
        'export const meta = { name: "status-active-card", description: "Status check", phases: [{ title: "Run", detail: "Active" }], '
        'maxConcurrent: 1, tools: ["codex"], patterns: ["sequence"], agentPlan: [{ node: "a1", role: "worker", agentId: "A1" }] };\n'
        'export default async function main() { return { r: await agent("Check", { agentId: "A1", label: "a1", timeout: 30 }) }; }\n'
    )
    graph = build_workflow_e2e_graph(tmp_path, monkeypatch, threaded_scheduler=False)
    entered, release = threading.Event(), threading.Event()
    exec_thread: threading.Thread | None = None
    try:
        WorkflowSessionRecorder(lambda _inv: ScriptedRule(output="done", entered_event=entered, block_event=release)).patch(monkeypatch)
        wf_handler = graph.client._handler_ctx.handlers["workflow"]
        pool = freeze_agent_pool([WorkflowAgentBinding(agent_id="A1", tool_name="codex", model_name="gpt-5.4", display_name="Codex")])
        run_spec = make_run_spec(pool=pool, task="Run checks")
        script_path = tmp_path / "status_active.js"
        script_path.write_text(script, encoding="utf-8")
        engine = graph.client._handler_ctx.workflow_engine_manager.get_or_create("oc_e2e", str(graph.repo_dir), engine_name="wf-stat")
        init_card_id = wf_handler._show_initial_workflow_progress_card(
            card_message_id=None, chat_id="oc_e2e",
            wf_project=WorkflowProject(status=WorkflowStatus.RUNNING, requirement=run_spec.task, initiator_user_id="ou_e2e"),
            origin_message_id="om_origin",
        )
        assert init_card_id is not None
        owner = _WorkflowLifecycleOwner(
            session_key="stat_key", initiator_user_id="ou_e2e", chat_id="oc_e2e",
            project_id=graph.project.project_id, root_path=str(graph.repo_dir), card_message_id=init_card_id,
        )
        callbacks = wf_handler._build_workflow_callbacks(init_card_id, "oc_e2e", graph.project, lifecycle_owner=owner, requirement=run_spec.task)
        exec_thread = threading.Thread(target=engine.execute_workflow, args=(str(script_path),), kwargs={"callbacks": callbacks, "run_spec": run_spec}, daemon=True)
        exec_thread.start()
        assert entered.wait(timeout=4.0)
        with engine._lock:
            if vars(engine).get("_workflow_start_owner") is None:
                engine._workflow_start_owner = owner
        assert engine.get_progress_cards(), "Expected active workflow to produce progress card pages"

        before_deliveries = len(graph.fake_im.deliveries)
        before_sends = sum(1 for d in graph.fake_im.deliveries if d.op in {"send", "reply"} and d.msg_type == "interactive")
        graph.send_text("/wf_status")
        graph.send_text("/wf_status")
        after_sends = sum(1 for d in graph.fake_im.deliveries if d.op in {"send", "reply"} and d.msg_type == "interactive")
        assert after_sends - before_sends == 0, "Expected /wf_status not to spam duplicate new interactive cards"
        assert any(
            (d.op == "patch" and d.message_id == init_card_id) or (d.op in {"send", "reply"} and d.msg_type == "text")
            for d in graph.fake_im.deliveries[before_deliveries:]
        )
    finally:
        release.set()
        if exec_thread is not None:
            exec_thread.join(timeout=4.0)
        graph.close()


def test_tier2_system_stop_and_status_integrate_with_real_workflow_engine_and_phase_progress(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.feishu.handlers.diagnostics_helper import DiagnosticsHelper

    graph = build_workflow_e2e_graph(tmp_path, monkeypatch)
    try:
        engine = graph.client._handler_ctx.workflow_engine_manager.get_or_create("oc_e2e", str(graph.repo_dir), engine_name="wf-diag")
        now = time.time()
        engine._project = WorkflowProject(
            workflow_id="wf_real_diag",
            name="RealPhase",
            status=WorkflowStatus.RUNNING,
            initiator_user_id="ou_e2e",
            started_at=now,
            phases=[
                PhaseProgress(
                    title="Done Phase",
                    started_at=now,
                    finished_at=now + 1.0,
                    agents=[AgentProgress(label="a1", status=AgentStatus.DONE)],
                ),
                PhaseProgress(
                    title="Active Phase",
                    started_at=now + 1.0,
                    agents=[AgentProgress(label="a2", status=AgentStatus.RUNNING)],
                ),
            ],
        )
        statuses = DiagnosticsHelper.get_all_engine_statuses(graph.client._handler_ctx, "oc_e2e", include_done=True)
        wf_entries = [e for e in statuses if e.task_id == "wf_real_diag"]
        assert wf_entries
        assert "1/2 阶段" in wf_entries[0].info

        # /stop must recognize engine.workflow_project.status and stop active Workflow
        wf_handler = graph.client._handler_ctx.handlers["workflow"]
        with patch.object(wf_handler, "stop_workflow", wraps=wf_handler.stop_workflow) as spy_stop:
            graph.send_text("/stop")
            spy_stop.assert_called_once()
    finally:
        graph.close()


def test_tier2_select_static_option_dedup_vs_identical_button_dedup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    graph = build_workflow_e2e_graph(tmp_path, monkeypatch)
    try:
        msg_id = graph.send_text("/wf_help")
        card_id = graph.fake_im.latest_card_message_id()
        assert msg_id and card_id

        # Two different select_static options must NOT be deduplicated as identical clicks
        _, sub1 = graph.click_card(
            message_id=card_id,
            value={"action": "show_status", "project_id": graph.project.project_id},
            option="opt_low",
            tag="select_static",
        )
        _, sub2 = graph.click_card(
            message_id=card_id,
            value={"action": "show_status", "project_id": graph.project.project_id},
            option="opt_high",
            tag="select_static",
        )
        assert sub1 == 1 and sub2 == 1

        # Identical button click twice returns toast dedup ACK without re-dispatching
        _, b1 = graph.click_card(
            message_id=card_id,
            value={"action": "show_tool_commands", "tool_name": "codex", "project_id": graph.project.project_id},
        )
        resp_dup, b2 = graph.click_card(
            message_id=card_id,
            value={"action": "show_tool_commands", "tool_name": "codex", "project_id": graph.project.project_id},
        )
        assert b1 == 1 and b2 == 0
        assert resp_dup.toast is not None
    finally:
        graph.close()


def test_tier2_second_wf_command_rejected_while_workflow_running(
    tmp_path: Path,
    monkeypatch,
) -> None:
    graph = build_workflow_e2e_graph(tmp_path, monkeypatch)
    try:
        engine = graph.client._handler_ctx.workflow_engine_manager.get_or_create("oc_e2e", str(graph.repo_dir), engine_name="wf-busy")
        engine._project = WorkflowProject(workflow_id="wf_busy", status=WorkflowStatus.RUNNING)
        engine._run_state = getattr(engine, "_run_state", None)
        with patch.object(type(engine), "is_running", property(lambda _self: True)):
            graph.send_text("/wf 第二条并发任务")
        card = graph.fake_im.latest_card()
        assert card is not None
        assert_card_2_invariants(card, label="busy-reject")
        assert "执行中" in extract_card_text(card)
    finally:
        graph.close()
