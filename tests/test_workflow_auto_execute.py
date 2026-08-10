"""Contracts for the gate-free Workflow generation path."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.feishu.handlers.workflow import WorkflowHandler
from src.workflow_engine.models import PendingWorkflow, WorkflowProject, WorkflowStatus

_VALID_SCRIPT = """
export const meta = {
  name: "auto-run",
  description: "automatic workflow",
  phases: [{ title: "Run", detail: "Do work" }],
  tools: ["coco"],
};
export default async function main() {
  const result = await agent("do work", { tool: "coco", timeout: 120 });
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
    return SimpleNamespace(
        project=WorkflowProject(
            status=WorkflowStatus.GENERATING_SCRIPT,
            pending=PendingWorkflow(
                requirement="do work",
                orchestrator_agent="coco",
                selected_tools=["coco"],
            ),
        )
    )


def test_script_generation_retries_then_accepts_valid_output(tmp_path) -> None:
    session = MagicMock()
    session.send_prompt.side_effect = [
        SimpleNamespace(text="not javascript"),
        SimpleNamespace(text=_VALID_SCRIPT),
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
    session.send_prompt.return_value = SimpleNamespace(text=unsupported)
    with (
        patch("src.agent_session.create_engine_session", return_value=session),
        patch("src.workflow_engine.tool_registry.get_available_tools", return_value={"coco": "Coco"}),
        pytest.raises(RuntimeError, match="3 次尝试"),
    ):
        _handler()._generate_script_via_ai(
            "implement the automatic workflow",
            str(tmp_path),
            ["coco"],
            _engine(),
            output_path=str(tmp_path / ".ghostap" / "workflow_scripts" / "generated.js"),
        )

    assert session.send_prompt.call_count == 3
