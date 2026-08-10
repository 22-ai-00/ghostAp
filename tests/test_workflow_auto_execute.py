"""Contracts for the confirmed-pool Workflow generation path."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.feishu.handlers.workflow import WorkflowHandler, _WorkflowLifecycleOwner
from src.workflow_engine.agent_pool import WorkflowAgentBinding
from src.workflow_engine.models import PendingWorkflow, WorkflowProject, WorkflowStatus

_VALID_SCRIPT = """
export const meta = {
  name: "auto-run",
  description: "automatic workflow",
  phases: [{ title: "Run", detail: "Do work" }],
  tools: ["coco"],
  agentPlan: [{ node: "main", role: "lead", agentId: "A-1" }],
};
export default async function main() {
  const result = await agent({ prompt: "do work", agentId: "A-1", timeout: 120 });
  if (result && result.error) return result;
  return result;
}
"""


def _handler() -> WorkflowHandler:
    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = SimpleNamespace(
        settings=SimpleNamespace(workflow_script_gen_timeout_s=5)
    )
    return handler


def _engine() -> SimpleNamespace:
    owner = _WorkflowLifecycleOwner("generation-1")
    return SimpleNamespace(
        _lock=threading.RLock(),
        _script_generation_owner=owner,
        release_lifecycle_owner=lambda _owner: False,
        project=WorkflowProject(
            status=WorkflowStatus.GENERATING_SCRIPT,
            pending=PendingWorkflow(
                requirement="do work",
                engine_session_key="generation-1",
                orchestrator_agent="coco",
                selected_tools=["coco"],
                agent_pool=(
                    WorkflowAgentBinding(
                        agent_id="A-1",
                        tool_name="coco",
                        model_name=None,
                        display_name="Coco",
                    ),
                ),
                orchestrator_agent_id="A-1",
            ),
        )
    )


def test_script_generation_retries_then_accepts_valid_output(tmp_path) -> None:
    session = MagicMock()
    session.send_prompt.side_effect = [
        SimpleNamespace(stop_reason="end_turn", text="not javascript"),
        SimpleNamespace(stop_reason="end_turn", text=_VALID_SCRIPT),
    ]
    with (
        patch("src.agent_session.create_engine_session", return_value=session),
        patch("src.workflow_engine.tool_registry.get_available_tools", return_value={"coco": "Coco"}),
    ):
        script_path, meta = _handler()._generate_script_via_ai(
            "implement the automatic workflow",
            str(tmp_path),
            ["coco"],
            _engine(),
            output_path=str(tmp_path / ".ghostap" / "workflow_scripts" / "generated.js"),
        )

    assert session.send_prompt.call_count == 2
    assert meta["tools"] == ["coco"]
    assert script_path.endswith(".js")


def test_unsupported_generated_tool_fails_after_bounded_retries(tmp_path) -> None:
    unsupported = _VALID_SCRIPT.replace('["coco"]', '["missing"]', 1)
    session = MagicMock()
    session.send_prompt.return_value = SimpleNamespace(
        stop_reason="end_turn",
        text=unsupported,
    )
    with (
        patch("src.agent_session.create_engine_session", return_value=session),
        patch("src.workflow_engine.tool_registry.get_available_tools", return_value={"coco": "Coco"}),
        pytest.raises(RuntimeError, match="failed|未确认工具|agent_pool"),
    ):
        _handler()._generate_script_via_ai(
            "implement the automatic workflow",
            str(tmp_path),
            ["coco"],
            _engine(),
            output_path=str(tmp_path / ".ghostap" / "workflow_scripts" / "generated.js"),
        )

    assert session.send_prompt.call_count == 3
