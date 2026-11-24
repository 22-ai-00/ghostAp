"""Self-check tests verifying the Workflow E2E harness works cleanly on HEAD."""

from __future__ import annotations

import json
from pathlib import Path

from src.card.events import CardEvent
from src.feishu.cot import FeishuCOTAPIClient, FeishuCOTStream
from src.workflow_engine.agent_pool import WorkflowAgentBinding
from src.workflow_engine.models import WorkflowStatus
from tests.helpers.workflow_e2e_harness import (
    ScriptedRule,
    SessionInvocation,
    WorkflowSessionRecorder,
    assert_card_2_invariants,
    build_workflow_e2e_graph,
    drive_card_callback,
    make_run_spec,
    run_workflow_e2e,
)

_SIMPLE_WF_JS = """\
export const meta = {
  name: "selfcheck-fanout",
  description: "Selfcheck workflow",
  phases: [{ title: "Work", detail: "Execute workers" }],
  maxConcurrent: 2,
  tools: ["codex", "claude"],
  patterns: ["fanout"],
  agentPlan: [
    { node: "w1", role: "worker", agentId: "A1" },
    { node: "w2", role: "reviewer", agentId: "A2" },
  ],
};

export default async function main() {
  await phase("Work");
  const out = await fanout("hello", [
    { prompt: "task 1", agentId: "A1", role: "worker", label: "w1", timeout: 30 },
    { prompt: "task 2", agentId: "A2", role: "reviewer", label: "w2", timeout: 30 },
  ], { synthesizerAgentId: "A2", timeout: 30 });
  return {
    card_summary: {
      verdict: "passed",
      conclusion: "Selfcheck workflow completed cleanly.",
      findings: [],
      verification: [{ status: "passed", text: "fanout succeeded" }],
      deliverables: [{ type: "code", text: "selfcheck output" }],
      next_steps: [],
    },
    result: out,
  };
}
"""


def test_harness_real_workflow_engine_and_node_runtime_end_to_end(tmp_path: Path, monkeypatch) -> None:
    def _resolve(inv: SessionInvocation) -> ScriptedRule:
        return ScriptedRule(output=json.dumps({"ok": True, "tool": inv.agent_type, "prompt": inv.prompt[:32]}))

    recorder = WorkflowSessionRecorder(_resolve)
    pool = (
        WorkflowAgentBinding(agent_id="A1", tool_name="codex", model_name="gpt-5.4", display_name="Codex"),
        WorkflowAgentBinding(agent_id="A2", tool_name="claude", model_name="sonnet", display_name="Claude"),
    )
    run_spec = make_run_spec(pool=pool, task="Verify real runtime selfcheck")
    e2e = run_workflow_e2e(
        tmp_path,
        monkeypatch,
        script_content=_SIMPLE_WF_JS,
        run_spec=run_spec,
        recorder=recorder,
        generate_reports=True,
    )

    assert e2e.project.status == WorkflowStatus.COMPLETED
    assert len(recorder.invocations) == 3
    assert [inv.agent_type for inv in recorder.invocations] == ["codex", "claude", "claude"]
    assert e2e.report_files is not None
    assert Path(e2e.report_files.html_path).exists()
    assert Path(e2e.report_files.markdown_path).exists()
    assert e2e.progress_snapshots


def test_harness_real_feishu_ws_client_wf_launch_and_card_invariants(tmp_path: Path, monkeypatch) -> None:
    def _resolve(_inv: SessionInvocation) -> ScriptedRule:
        return ScriptedRule(output=json.dumps({"summary": "worker-result-ok", "verdict": "passed"}))

    recorder = WorkflowSessionRecorder(_resolve).patch(monkeypatch)
    graph = build_workflow_e2e_graph(tmp_path, monkeypatch, cot_enabled=False)
    try:
        # 1. Real FeishuWSClient._handle_message -> WorkflowHandler (/wf_help & /wf_status)
        msg_help = graph.send_text("/wf_help")
        assert msg_help.startswith("om_user_")
        graph.send_text("/wf_status")

        # 2. Real WorkflowHandler._build_workflow_callbacks + WorkflowEngine + Node.js runtime
        wf_handler = graph.client._handler_ctx.handlers["workflow"]
        script_path = tmp_path / "selfcheck_feishu_wf.js"
        script_path.write_text(_SIMPLE_WF_JS, encoding="utf-8")
        pool = (
            WorkflowAgentBinding(agent_id="A1", tool_name="codex", model_name="gpt-5.4", display_name="Codex"),
            WorkflowAgentBinding(agent_id="A2", tool_name="claude", model_name="sonnet", display_name="Claude"),
        )
        run_spec = make_run_spec(pool=pool, task="Verify Feishu workflow delivery")
        engine = graph.client._handler_ctx.workflow_engine_manager.get_or_create(
            "oc_e2e",
            str(graph.repo_dir),
            engine_name="Workflow-selfcheck",
        )
        callbacks = wf_handler._build_workflow_callbacks(
            msg_help,
            "oc_e2e",
            graph.project,
            requirement=run_spec.task,
        )
        project = engine.execute_workflow(str(script_path), callbacks=callbacks, run_spec=run_spec)
        assert project.status == WorkflowStatus.COMPLETED

        cards = graph.fake_im.all_cards()
        assert len(cards) >= 2
        for idx, card in enumerate(cards):
            assert_card_2_invariants(card, label=f"selfcheck-card-{idx}")

        assert graph.fake_im.uploaded_files
        assert graph.fake_im.file_replies
        assert any("Selfcheck workflow completed cleanly." in json.dumps(c, ensure_ascii=False) for c in cards)
        assert len(recorder.invocations) == 3
    finally:
        graph.close()


def test_harness_card_callback_and_cot_recording(tmp_path: Path, monkeypatch) -> None:
    graph = build_workflow_e2e_graph(tmp_path, monkeypatch, cot_enabled=True)
    try:
        msg_id = graph.send_text("/wf_help")
        assert msg_id.startswith("om_user_")
        help_card_id = graph.fake_im.latest_card_message_id()
        assert help_card_id is not None

        # Verify Card 2.0 callback path via FeishuWSClient._handle_card_action_callback
        resp, submitted = drive_card_callback(
            graph.client,
            message_id=help_card_id,
            chat_id="oc_e2e",
            operator_id="ou_e2e",
            value={"action": "show_status", "project_id": graph.project.project_id},
        )
        assert resp is not None
        assert submitted == 1
        for idx, card in enumerate(graph.fake_im.all_cards()):
            assert_card_2_invariants(card, label=f"callback-card-{idx}")

        # Verify FakeCOTHttpClient captures FeishuCOTStream wire events
        cot_api = FeishuCOTAPIClient(graph.client._handler_ctx.api_client_factory(), timeout_seconds=0.5)
        stream = FeishuCOTStream(
            cot_api,
            chat_id="oc_e2e",
            origin_message_id=msg_id,
            input_text="selfcheck cot",
        )
        stream.start()
        stream.emit(CardEvent.text_started("b1"))
        stream.emit(CardEvent.text_delta("b1", "hello cot"))
        stream.emit(CardEvent.text_done("b1"))
        assert stream.complete(CardEvent.completed()) is True
        assert graph.fake_cot.requests
        assert graph.fake_cot.events
    finally:
        graph.close()
