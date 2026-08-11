"""Concurrency contract for Workflow progress rendering."""

from __future__ import annotations

import threading

from src.card.state.models import ContentBlock
from src.workflow_engine.models import (
    AgentProgress,
    AgentStatus,
    PhaseProgress,
    WorkflowProject,
    WorkflowStatus,
)
from src.workflow_engine.renderer import (
    WorkflowProgressRenderer,
    render_completion_cards,
)


def _project(marker: str) -> WorkflowProject:
    return WorkflowProject(
        name=f"workflow-{marker}",
        requirement=f"requirement-{marker}",
        status=WorkflowStatus.RUNNING,
        phases=[
            PhaseProgress(
                title=f"phase-{marker}",
                agents=[
                    AgentProgress(
                        label=f"agent-{marker}",
                        agent_id=f"A-{marker}",
                        tool="codex",
                        status=AgentStatus.RUNNING,
                        execution_blocks=[
                            ContentBlock(
                                kind="text",
                                block_id=f"text-{marker}",
                                content=f"stream-{marker}",
                                status="active",
                            )
                        ],
                    )
                ],
            )
        ],
    )


def _all_text(value: object) -> str:
    if isinstance(value, dict):
        return "\n".join(_all_text(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return "\n".join(_all_text(item) for item in value)
    return value if isinstance(value, str) else ""


def test_shared_workflow_renderer_serializes_snapshot_rendering() -> None:
    base = _project("BASE")
    project_a = _project("ONE")
    project_b = _project("TWO")
    first_entered = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()

    class ProbeRenderer(WorkflowProgressRenderer):
        def _render_progress_card_pages_impl(self):
            if threading.current_thread().name == "render-one":
                first_entered.set()
                assert release_first.wait(2)
            else:
                second_entered.set()
            return super()._render_progress_card_pages_impl()

    renderer = ProbeRenderer(base)
    results: dict[str, list[dict]] = {}

    first = threading.Thread(
        target=lambda: results.setdefault(
            "ONE",
            renderer.render_progress_cards(project_a),
        ),
        name="render-one",
    )
    second = threading.Thread(
        target=lambda: results.setdefault(
            "TWO",
            renderer.render_progress_cards(project_b),
        ),
        name="render-two",
    )

    first.start()
    assert first_entered.wait(2)
    second.start()
    try:
        # The second render must not mutate the shared renderer while the first
        # snapshot is between bind and restore.
        assert not second_entered.wait(0.1)
    finally:
        release_first.set()
    first.join(2)
    second.join(2)

    assert not first.is_alive()
    assert not second.is_alive()
    for marker, other in (("ONE", "TWO"), ("TWO", "ONE")):
        rendered = _all_text(results[marker])
        assert f"workflow-{marker}" in rendered
        assert f"stream-{marker}" in rendered
        assert f"workflow-{other}" not in rendered
        assert f"stream-{other}" not in rendered
        assert "workflow-BASE" not in rendered
    assert renderer._project is base


def test_running_renderer_defers_result_ledger_until_terminal() -> None:
    project = _project("ORDER")
    agent = project.phases[0].agents[0]
    agent.status = AgentStatus.DONE
    agent.result = "finished"
    agent.execution_blocks[0] = ContentBlock(
        kind="text",
        block_id="text-order",
        content="stream-ORDER",
        status="completed",
    )

    progress_cards = WorkflowProgressRenderer(project).render_progress_cards(project)
    assert all(card["_workflow_page_key"][0] != "ledger" for card in progress_cards)

    project.status = WorkflowStatus.COMPLETED
    project.result = "workflow finished"
    terminal_cards = render_completion_cards(project)
    kinds = [card["_workflow_page_key"][0] for card in terminal_cards]
    assert kinds[0] == "status"
    assert "agent" in kinds
    assert kinds[-1] == "ledger"
