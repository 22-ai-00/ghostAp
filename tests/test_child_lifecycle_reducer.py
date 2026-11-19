"""Regression contracts for the shared ACP/card child lifecycle reducer."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.acp.child_lifecycle import (
    ChildLifecycleReducer,
    ChildObservationProvenance,
    normalize_provider_child_status,
)
from src.acp.codex_rollout_reconciliation import _normalize_agent_status
from src.acp.collaboration import AuthoritativeChildLifecycleProof
from src.acp.models import ACPEvent, ACPEventType, PromptResult, ToolCallInfo
from src.acp.outcome import PromptOutcome, classify_prompt_result
from src.card.events.projector import (
    UNRESOLVED_AGENT_STATUS,
    finalize_summaries,
    merge_activity_summary,
    merge_collaboration_summaries,
    project_activity,
    project_collaboration,
)


@pytest.mark.parametrize(
    ("raw_status", "expected_status", "expected_outcome"),
    [
        ("pending", "pending", PromptOutcome.INCOMPLETE),
        ("pendinginit", "pending", PromptOutcome.INCOMPLETE),
        ("pending_init", "pending", PromptOutcome.INCOMPLETE),
        ("PENDING_INIT", "pending", PromptOutcome.INCOMPLETE),
        ("running", "running", PromptOutcome.INCOMPLETE),
        ("in_progress", "running", PromptOutcome.INCOMPLETE),
        ("completed", "completed", PromptOutcome.COMPLETED),
        ("failed", "failed", PromptOutcome.COMPLETED),
        ("errored", "failed", PromptOutcome.COMPLETED),
        ("notfound", "failed", PromptOutcome.COMPLETED),
        ("not_found", "failed", PromptOutcome.COMPLETED),
        ("cancelled", "cancelled", PromptOutcome.COMPLETED),
        ("canceled", "cancelled", PromptOutcome.COMPLETED),
        ("interrupted", "cancelled", PromptOutcome.COMPLETED),
        ("shutdown", "cancelled", PromptOutcome.COMPLETED),
    ],
)
def test_outcome_card_and_rollout_agree_on_status_vocabulary(
    raw_status: str,
    expected_status: str,
    expected_outcome: PromptOutcome,
) -> None:
    assert normalize_provider_child_status(raw_status) == expected_status
    assert _normalize_agent_status(raw_status) == expected_status

    tool_call = ToolCallInfo(
        id="list-1",
        title="list_agents",
        kind="other",
        status="completed",
        collaboration_tool="list_agents",
        collaboration_receivers=("child-a",),
        subagent_states=({"source_id": "child-a", "status": raw_status},),
    )
    assessment = classify_prompt_result(
        PromptResult(stop_reason="end_turn", tool_calls=[tool_call])
    )
    assert assessment.outcome is expected_outcome

    projection = project_collaboration(
        ACPEvent(event_type=ACPEventType.TOOL_CALL_DONE, tool_call=tool_call)
    )
    assert projection is not None
    assert len(projection.agents) == 1
    assert projection.agents[0].status == expected_status
    assert projection.agents[0].malformed is False
    if raw_status == "shutdown":
        assert projection.agents[0].progress == "已完成并停止"


def test_generation_reopening_agrees_across_reducer_outcome_proof_and_card() -> None:
    reducer = ChildLifecycleReducer()
    proof = AuthoritativeChildLifecycleProof()
    summaries: dict[str, dict] = {}
    recorded_tools: list[ToolCallInfo] = []

    def step(event_type: ACPEventType, tool_call: ToolCallInfo) -> None:
        nonlocal summaries
        event = ACPEvent(event_type=event_type, tool_call=tool_call)
        reducer.apply(tool_call, event_name=event_type.name)
        proof.observe_event(event)
        collab = project_collaboration(event)
        if collab is not None:
            summaries = merge_collaboration_summaries(
                summaries, collab, parent_sequence=1
            )
        else:
            act = project_activity(event)
            if act is not None:
                summaries = merge_activity_summary(
                    summaries, act, parent_sequence=1
                )
        if event_type is ACPEventType.TOOL_CALL_DONE:
            recorded_tools.append(tool_call)

    # Generation 1: spawn START -> spawn DONE -> list_agents DONE
    spawn_start = ToolCallInfo(
        id="call-spawn-1",
        title="spawn_agent",
        kind="other",
        status="in_progress",
        collaboration_tool="spawn_agent",
        collaboration_receivers=("child-a",),
        subagent_states=({"source_id": "child-a", "status": "running"},),
    )
    spawn_done = ToolCallInfo(
        id="call-spawn-1",
        title="spawn_agent",
        kind="other",
        status="completed",
        collaboration_tool="spawn_agent",
        collaboration_receivers=("child-a",),
        subagent_states=({"source_id": "child-a", "status": "running"},),
    )
    list_gen1 = ToolCallInfo(
        id="call-list-1",
        title="list_agents",
        kind="other",
        status="completed",
        collaboration_tool="list_agents",
        collaboration_receivers=("child-a",),
        subagent_states=({"source_id": "child-a", "status": "completed"},),
    )
    step(ACPEventType.TOOL_CALL_START, spawn_start)
    step(ACPEventType.TOOL_CALL_DONE, spawn_done)
    assert reducer.entries["child-a"].generation == 1
    assert summaries["child-a"]["generation"] == 1
    assert proof.all_observed_children_terminal() is False

    step(ACPEventType.TOOL_CALL_DONE, list_gen1)
    assert reducer.entries["child-a"].generation == 1
    assert reducer.entries["child-a"].status == "completed"
    assert (
        reducer.entries["child-a"].provenance
        is ChildObservationProvenance.AUTHORITATIVE_LIST
    )
    assert proof.all_observed_children_terminal() is True
    assert summaries["child-a"]["status"] == "completed"
    assert (
        classify_prompt_result(
            PromptResult(stop_reason="end_turn", tool_calls=recorded_tools)
        ).outcome
        is PromptOutcome.COMPLETED
    )

    # Generation 2: followup_task reopens terminal child
    followup_done = ToolCallInfo(
        id="call-followup-2",
        title="followup_task",
        kind="other",
        status="completed",
        collaboration_tool="followup_task",
        collaboration_receivers=("child-a",),
        subagent_states=({"source_id": "child-a", "status": "running"},),
    )
    step(ACPEventType.TOOL_CALL_DONE, followup_done)
    assert reducer.entries["child-a"].generation == 2
    assert reducer.entries["child-a"].status == "running"
    assert proof.all_observed_children_terminal() is False
    assert summaries["child-a"]["generation"] == 2
    assert summaries["child-a"]["status"] == "running"
    assert (
        classify_prompt_result(
            PromptResult(stop_reason="end_turn", tool_calls=recorded_tools)
        ).outcome
        is PromptOutcome.INCOMPLETE
    )

    list_gen2 = ToolCallInfo(
        id="call-list-2",
        title="list_agents",
        kind="other",
        status="completed",
        collaboration_tool="list_agents",
        collaboration_receivers=("child-a",),
        subagent_states=({"source_id": "child-a", "status": "completed"},),
    )
    step(ACPEventType.TOOL_CALL_DONE, list_gen2)
    assert reducer.entries["child-a"].generation == 2
    assert proof.all_observed_children_terminal() is True
    assert summaries["child-a"]["status"] == "completed"

    # Generation 3: non-terminal subagent_activity reopens both outcome and card!
    activity_reopen = ToolCallInfo(
        id="call-act-3",
        title="Interact",
        kind="other",
        status="completed",
        subagent_source_id="child-a",
        subagent_activity="interacted",
    )
    step(ACPEventType.TOOL_CALL_DONE, activity_reopen)
    assert reducer.entries["child-a"].generation == 3
    assert reducer.entries["child-a"].status == "running"
    assert (
        reducer.entries["child-a"].provenance
        is ChildObservationProvenance.SUBAGENT_ACTIVITY
    )
    assert proof.all_observed_children_terminal() is False
    assert summaries["child-a"]["generation"] == 3
    assert summaries["child-a"]["status"] == "running"
    assert (
        classify_prompt_result(
            PromptResult(stop_reason="end_turn", tool_calls=recorded_tools)
        ).outcome
        is PromptOutcome.INCOMPLETE
    )


def test_sticky_malformed_status_cannot_be_cleared_across_outcome_or_card() -> None:
    reducer = ChildLifecycleReducer()
    proof = AuthoritativeChildLifecycleProof()
    summaries: dict[str, dict] = {}

    bad_tool = ToolCallInfo(
        id="call-bad",
        title="wait_agent",
        kind="other",
        status="completed",
        collaboration_tool="wait_agent",
        collaboration_receivers=("child-a",),
        subagent_states=({"source_id": "child-a", "status": "future-state"},),
    )
    clean_tool = ToolCallInfo(
        id="call-clean",
        title="list_agents",
        kind="other",
        status="completed",
        collaboration_tool="list_agents",
        collaboration_receivers=("child-a",),
        subagent_states=({"source_id": "child-a", "status": "completed"},),
    )

    for tool in (bad_tool, clean_tool):
        ev = ACPEvent(event_type=ACPEventType.TOOL_CALL_DONE, tool_call=tool)
        reducer.apply(tool, event_name=ev.event_type.name)
        proof.observe_event(ev)
        collab = project_collaboration(ev)
        assert collab is not None
        summaries = merge_collaboration_summaries(
            summaries, collab, parent_sequence=1
        )

    assert reducer.malformed is True
    assert reducer.entries["child-a"].malformed is True
    assert reducer.entries["child-a"].is_terminal is False
    assert reducer.all_terminal() is False
    assert proof.all_observed_children_terminal() is False
    assert (
        classify_prompt_result(
            PromptResult(stop_reason="end_turn", tool_calls=[bad_tool, clean_tool])
        ).outcome
        is PromptOutcome.INCOMPLETE
    )
    assert summaries["child-a"]["malformed"] is True
    assert summaries["child-a"]["status"] == "malformed"
    finalized = finalize_summaries(summaries, "completed")
    assert finalized["child-a"]["status"] == UNRESOLVED_AGENT_STATUS


def test_outer_collaboration_tool_call_done_never_promotes_child_to_terminal() -> None:
    spawn_done = ToolCallInfo(
        id="call-spawn",
        title="spawn_agent",
        kind="other",
        status="completed",
        collaboration_tool="spawn_agent",
        collaboration_receivers=("child-a",),
    )
    activity_done = ToolCallInfo(
        id="call-act",
        title="Start subagent",
        kind="other",
        status="completed",
        subagent_source_id="child-b",
        subagent_activity="started",
    )
    wait_done = ToolCallInfo(
        id="call-wait",
        title="wait_agent",
        kind="other",
        status="completed",
        collaboration_tool="wait_agent",
        collaboration_receivers=("child-c",),
        subagent_states=({"source_id": "child-c", "status": "pending_init"},),
    )

    reducer = ChildLifecycleReducer()
    proof = AuthoritativeChildLifecycleProof()
    summaries: dict[str, dict] = {}
    for tool in (spawn_done, activity_done, wait_done):
        ev = ACPEvent(event_type=ACPEventType.TOOL_CALL_DONE, tool_call=tool)
        reducer.apply(tool, event_name=ev.event_type.name)
        proof.observe_event(ev)
        collab = project_collaboration(ev)
        if collab is not None:
            summaries = merge_collaboration_summaries(
                summaries, collab, parent_sequence=1
            )
        else:
            act = project_activity(ev)
            assert act is not None
            summaries = merge_activity_summary(
                summaries, act, parent_sequence=1
            )

    assert reducer.entries["child-a"].status == "pending"
    assert reducer.entries["child-b"].status == "running"
    assert reducer.entries["child-c"].status == "pending"
    assert reducer.all_terminal() is False
    assert proof.all_observed_children_terminal() is False
    assert summaries["child-a"]["status"] == "pending"
    assert summaries["child-b"]["status"] == "running"
    assert summaries["child-c"]["status"] == "pending"
    assert (
        classify_prompt_result(
            PromptResult(
                stop_reason="end_turn",
                tool_calls=[spawn_done, activity_done, wait_done],
            )
        ).outcome
        is PromptOutcome.INCOMPLETE
    )


def test_acp_child_lifecycle_has_zero_card_imports() -> None:
    acp_dir = Path(__file__).resolve().parents[1] / "src" / "acp"
    for py_file in sorted(acp_dir.rglob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("src.card"), str(py_file)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not module.startswith("src.card"), str(py_file)
                assert not (node.level >= 2 and module.startswith("card")), str(py_file)
