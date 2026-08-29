"""Regression tests for strict Codex rollout child-state reconciliation."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import src.acp.codex_rollout_reconciliation as reconciliation
from src.acp.codex_rollout_reconciliation import (
    enrich_codex_reconciliation_result,
    enrich_codex_terminal_result,
)
from src.acp.models import ACPEventType, PromptResult, ToolCallInfo
from src.acp.outcome import PromptOutcome, classify_prompt_result
from src.acp.session import ACPSession
from src.acp.sync_adapter import SyncACPSession


def _timestamp(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _write_rollout(
    root: Path,
    *,
    session_id: str,
    cwd: Path,
    output_agents: list[dict],
    activities: list[dict[str, object]] | None = None,
    arguments: str = "{}",
    session_timestamp: str = "2026-08-08T09:00:00.000Z",
    call_timestamp: str = "2026-08-08T09:26:54.695Z",
    output_timestamp: str = "2026-08-08T09:26:54.809Z",
) -> Path:
    path = (
        root
        / "sessions"
        / "2026"
        / "08"
        / "08"
        / f"rollout-2026-08-08T09-00-00-{session_id}.jsonl"
    )
    path.parent.mkdir(parents=True)
    events = [
        {
            "timestamp": session_timestamp,
            "type": "session_meta",
            "payload": {"id": session_id, "cwd": str(cwd)},
        },
        *[
            {
                "timestamp": str(activity["timestamp"]),
                "type": "event_msg",
                "payload": {
                    "type": "sub_agent_activity",
                    "event_id": f"activity-{index}",
                    "occurred_at_ms": int(
                        _timestamp(str(activity["timestamp"])) * 1000
                    ),
                    "agent_thread_id": activity["source_id"],
                    "agent_path": activity["agent_path"],
                    "kind": activity.get("kind", "interacted"),
                },
            }
            for index, activity in enumerate(activities or [])
        ],
        {
            "timestamp": call_timestamp,
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "list_agents",
                "namespace": "collaboration",
                "arguments": arguments,
                "call_id": "call-list-agents",
            },
        },
        {
            "timestamp": output_timestamp,
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call-list-agents",
                "output": json.dumps({"agents": output_agents}),
            },
        },
    ]
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    return path


def _write_child_rollout(
    root: Path,
    *,
    parent_session_id: str,
    child_thread_id: str,
    cwd: Path,
    agent_path: str | None,
    lifecycle: list[tuple[str, str, str | None]],
    recorded_parent_id: str | None = None,
) -> Path:
    path = (
        root
        / "sessions"
        / "2026"
        / "08"
        / "08"
        / f"rollout-2026-08-08T09-00-00-{child_thread_id}.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    meta: dict[str, object] = {
        "id": child_thread_id,
        "session_id": parent_session_id,
        "parent_thread_id": recorded_parent_id or parent_session_id,
        "cwd": str(cwd),
        "thread_source": "subagent",
    }
    if agent_path is not None:
        meta["agent_path"] = agent_path
    events: list[dict[str, object]] = [
        {
            "timestamp": "2026-08-08T09:00:00.000Z",
            "type": "session_meta",
            "payload": meta,
        }
    ]
    for timestamp, event_type, turn_id in lifecycle:
        payload = {"type": event_type}
        if turn_id is not None:
            payload["turn_id"] = turn_id
        events.append(
            {
                "timestamp": timestamp,
                "type": "event_msg",
                "payload": payload,
            }
        )
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    return path


def _append_child_lifecycle(
    path: Path,
    *,
    event_type: str,
    turn_id: str,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    event = {
        "timestamp": timestamp,
        "type": "event_msg",
        "payload": {"type": event_type, "turn_id": turn_id},
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def _append_list_agents_output(
    path: Path,
    *,
    call_id: str,
    call_timestamp: str,
    output_timestamp: str,
    raw_output: str,
    arguments: str = "{}",
) -> None:
    events = [
        {
            "timestamp": call_timestamp,
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "list_agents",
                "namespace": "collaboration",
                "arguments": arguments,
                "call_id": call_id,
            },
        },
        {
            "timestamp": output_timestamp,
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": raw_output,
            },
        },
    ]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(event) + "\n" for event in events))


def _append_generic_function_output(
    path: Path,
    *,
    call_id: str,
    call_timestamp: str,
    output_timestamp: str,
    name: str = "wait",
    namespace: str | None = None,
    output: object = None,
) -> None:
    function_call: dict[str, object] = {
        "type": "function_call",
        "name": name,
        "arguments": "{}",
        "call_id": call_id,
    }
    if namespace is not None:
        function_call["namespace"] = namespace
    events = [
        {
            "timestamp": call_timestamp,
            "type": "response_item",
            "payload": function_call,
        },
        {
            "timestamp": output_timestamp,
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": (
                    [{"type": "text", "text": "done"}]
                    if output is None
                    else output
                ),
            },
        },
    ]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(event) + "\n" for event in events))


def _result_with_child_paths() -> PromptResult:
    return PromptResult(
        stop_reason="end_turn",
        tool_calls=[
            ToolCallInfo(
                id=f"activity-{index}",
                title="Interact with subagent",
                kind="other",
                status="completed",
                subagent_source_id=f"thread-{index}",
                subagent_path=f"/root/reviewer-{index}",
                subagent_activity="interacted",
            )
            for index in range(3)
        ],
    )


def _completed_agents(
    paths: tuple[str, ...] | None = None,
) -> list[dict]:
    paths = paths or tuple(f"/root/reviewer-{index}" for index in range(3))
    return [
        {"agent_name": "/root", "agent_status": "running"},
        *[
            {
                "agent_name": path,
                "agent_status": {"completed": "sensitive final answer"},
            }
            for path in paths
        ],
    ]


def _single_child_result(
    *,
    source_id: str = "thread-reviewer",
    path: str = "/root/reviewer",
) -> PromptResult:
    return PromptResult(
        stop_reason="end_turn",
        tool_calls=[
            ToolCallInfo(
                id="activity-reviewer",
                title="Interact with subagent",
                kind="other",
                status="completed",
                subagent_source_id=source_id,
                subagent_path=path,
                subagent_activity="interacted",
            ),
            ToolCallInfo(
                id="stale-list-agents",
                title="list_agents",
                kind="other",
                status="completed",
                collaboration_tool="list_agents",
                collaboration_receivers=(source_id,),
                subagent_states=(
                    {
                        "source_id": source_id,
                        "status": "running",
                        "message": "",
                    },
                ),
            ),
        ],
    )


def test_child_task_complete_overrides_stale_parent_agents_states(
    tmp_path: Path,
) -> None:
    parent_session_id = "parent-session-0001"
    child_thread_id = "child-thread-00001"
    cwd = tmp_path / "repo"
    cwd.mkdir()
    codex_home = tmp_path / "codex-home"
    _write_rollout(
        codex_home,
        session_id=parent_session_id,
        cwd=cwd,
        output_agents=[
            {"agent_name": "/root", "agent_status": "running"},
            {"agent_name": "/root/reviewer", "agent_status": "running"},
        ],
    )
    _write_child_rollout(
        codex_home,
        parent_session_id=parent_session_id,
        child_thread_id=child_thread_id,
        cwd=cwd,
        agent_path="/root/reviewer",
        lifecycle=[
            ("2026-08-08T09:26:54.100Z", "task_started", "turn-review"),
            ("2026-08-08T09:26:55.100Z", "task_complete", "turn-review"),
        ],
    )

    enriched, evidence = enrich_codex_reconciliation_result(
        _single_child_result(source_id=child_thread_id),
        session_id=parent_session_id,
        cwd=str(cwd),
        logical_task_started_at=_timestamp("2026-08-08T09:26:54.000Z"),
        started_at=_timestamp("2026-08-08T09:26:56.000Z"),
        ended_at=_timestamp("2026-08-08T09:27:00.000Z"),
        codex_home=str(codex_home),
    )

    assert evidence is not None
    assert evidence.id.startswith("rollout-child-lifecycle:")
    assert evidence.subagent_states == (
        {
            "source_id": child_thread_id,
            "status": "completed",
            "message": "",
        },
    )
    assert enriched.tool_calls[-1] is evidence
    assert classify_prompt_result(enriched).outcome is PromptOutcome.COMPLETED


def test_child_latest_generation_without_terminal_blocks_parent_fallback(
    tmp_path: Path,
) -> None:
    parent_session_id = "parent-session-0002"
    child_thread_id = "child-thread-00002"
    cwd = tmp_path / "repo"
    cwd.mkdir()
    codex_home = tmp_path / "codex-home"
    _write_rollout(
        codex_home,
        session_id=parent_session_id,
        cwd=cwd,
        output_agents=[
            {"agent_name": "/root", "agent_status": "running"},
            {
                "agent_name": "/root/reviewer",
                "agent_status": {"completed": "stale"},
            },
        ],
    )
    _write_child_rollout(
        codex_home,
        parent_session_id=parent_session_id,
        child_thread_id=child_thread_id,
        cwd=cwd,
        agent_path="/root/reviewer",
        lifecycle=[
            ("2026-08-08T09:26:54.100Z", "task_started", "turn-one"),
            ("2026-08-08T09:26:55.100Z", "task_complete", "turn-one"),
            ("2026-08-08T09:26:56.100Z", "task_started", "turn-two"),
        ],
    )
    original = _single_child_result(source_id=child_thread_id)

    enriched, evidence = enrich_codex_reconciliation_result(
        original,
        session_id=parent_session_id,
        cwd=str(cwd),
        logical_task_started_at=_timestamp("2026-08-08T09:26:54.000Z"),
        started_at=_timestamp("2026-08-08T09:26:56.000Z"),
        ended_at=_timestamp("2026-08-08T09:27:00.000Z"),
        codex_home=str(codex_home),
    )

    assert enriched is original
    assert evidence is None


@pytest.mark.parametrize(
    ("terminal_event", "expected_status"),
    [("task_complete", "completed"), ("turn_aborted", "cancelled")],
)
def test_child_rollout_terminal_event_mapping(
    tmp_path: Path,
    terminal_event: str,
    expected_status: str,
) -> None:
    parent_session_id = "parent-session-0003"
    child_thread_id = "child-thread-00003"
    cwd = tmp_path / "repo"
    cwd.mkdir()
    codex_home = tmp_path / "codex-home"
    _write_rollout(
        codex_home,
        session_id=parent_session_id,
        cwd=cwd,
        output_agents=[
            {"agent_name": "/root", "agent_status": "running"},
            {"agent_name": "/root/reviewer", "agent_status": "running"},
        ],
    )
    _write_child_rollout(
        codex_home,
        parent_session_id=parent_session_id,
        child_thread_id=child_thread_id,
        cwd=cwd,
        agent_path="/root/reviewer",
        lifecycle=[
            ("2026-08-08T09:26:54.100Z", "task_started", "turn-review"),
            ("2026-08-08T09:26:55.100Z", terminal_event, "turn-review"),
        ],
    )

    _, evidence = enrich_codex_reconciliation_result(
        _single_child_result(source_id=child_thread_id),
        session_id=parent_session_id,
        cwd=str(cwd),
        logical_task_started_at=_timestamp("2026-08-08T09:26:54.000Z"),
        started_at=_timestamp("2026-08-08T09:26:56.000Z"),
        ended_at=_timestamp("2026-08-08T09:27:00.000Z"),
        codex_home=str(codex_home),
    )

    assert evidence is not None
    assert evidence.subagent_states[0]["status"] == expected_status


def test_child_rollout_wrong_parent_fails_closed_without_parent_fallback(
    tmp_path: Path,
) -> None:
    parent_session_id = "parent-session-0004"
    child_thread_id = "child-thread-00004"
    cwd = tmp_path / "repo"
    cwd.mkdir()
    codex_home = tmp_path / "codex-home"
    _write_rollout(
        codex_home,
        session_id=parent_session_id,
        cwd=cwd,
        output_agents=[
            {"agent_name": "/root", "agent_status": "running"},
            {
                "agent_name": "/root/reviewer",
                "agent_status": {"completed": "stale"},
            },
        ],
    )
    _write_child_rollout(
        codex_home,
        parent_session_id=parent_session_id,
        child_thread_id=child_thread_id,
        cwd=cwd,
        agent_path="/root/reviewer",
        recorded_parent_id="different-parent-001",
        lifecycle=[
            ("2026-08-08T09:26:54.100Z", "task_started", "turn-review"),
            ("2026-08-08T09:26:55.100Z", "task_complete", "turn-review"),
        ],
    )
    original = _single_child_result(source_id=child_thread_id)

    enriched, evidence = enrich_codex_reconciliation_result(
        original,
        session_id=parent_session_id,
        cwd=str(cwd),
        logical_task_started_at=_timestamp("2026-08-08T09:26:54.000Z"),
        started_at=_timestamp("2026-08-08T09:26:56.000Z"),
        ended_at=_timestamp("2026-08-08T09:27:00.000Z"),
        codex_home=str(codex_home),
    )

    assert enriched is original
    assert evidence is None


def test_rollout_list_agents_terminal_states_are_mapped_without_messages(
    tmp_path: Path,
) -> None:
    session_id = "019fe093-0d77-7fd2-9860-616a2100e71d"
    cwd = tmp_path / "repo"
    cwd.mkdir()
    codex_home = tmp_path / "codex-home"
    _write_rollout(
        codex_home,
        session_id=session_id,
        cwd=cwd,
        output_agents=_completed_agents(),
    )

    enriched, evidence = enrich_codex_reconciliation_result(
        _result_with_child_paths(),
        session_id=session_id,
        cwd=str(cwd),
        started_at=_timestamp("2026-08-08T09:26:54.000Z"),
        ended_at=_timestamp("2026-08-08T09:27:00.000Z"),
        codex_home=str(codex_home),
    )

    assert evidence is not None
    assert [state["status"] for state in evidence.subagent_states] == [
        "completed",
        "completed",
        "completed",
    ]
    assert all(state["message"] == "" for state in evidence.subagent_states)
    assert "sensitive final answer" not in repr(evidence)
    assert enriched.tool_calls[-1] is evidence
    assessment = classify_prompt_result(enriched)
    assert assessment.outcome is PromptOutcome.COMPLETED
    assert assessment.unresolved_child_tool_calls == 0


def test_rollout_evidence_outside_current_turn_is_ignored(
    tmp_path: Path,
) -> None:
    session_id = "019fe093-0d77-7fd2-9860-616a2100e71d"
    cwd = tmp_path / "repo"
    cwd.mkdir()
    codex_home = tmp_path / "codex-home"
    _write_rollout(
        codex_home,
        session_id=session_id,
        cwd=cwd,
        output_agents=_completed_agents(),
    )
    original = _result_with_child_paths()

    enriched, evidence = enrich_codex_reconciliation_result(
        original,
        session_id=session_id,
        cwd=str(cwd),
        started_at=_timestamp("2026-08-08T09:27:30.000Z"),
        ended_at=_timestamp("2026-08-08T09:28:00.000Z"),
        codex_home=str(codex_home),
    )

    assert enriched is original
    assert evidence is None


def test_rollout_unknown_child_path_fails_closed(tmp_path: Path) -> None:
    session_id = "019fe093-0d77-7fd2-9860-616a2100e71d"
    cwd = tmp_path / "repo"
    cwd.mkdir()
    codex_home = tmp_path / "codex-home"
    agents = _completed_agents()
    agents.append(
        {
            "agent_name": "/root/unmapped-reviewer",
            "agent_status": {"completed": "result"},
        }
    )
    _write_rollout(
        codex_home,
        session_id=session_id,
        cwd=cwd,
        output_agents=agents,
    )
    original = _result_with_child_paths()

    enriched, evidence = enrich_codex_reconciliation_result(
        original,
        session_id=session_id,
        cwd=str(cwd),
        started_at=_timestamp("2026-08-08T09:26:54.000Z"),
        ended_at=_timestamp("2026-08-08T09:27:00.000Z"),
        codex_home=str(codex_home),
    )

    assert enriched is original
    assert evidence is None


def test_rollout_running_child_does_not_inject_terminal_evidence(
    tmp_path: Path,
) -> None:
    session_id = "019fe093-0d77-7fd2-9860-616a2100e71d"
    cwd = tmp_path / "repo"
    cwd.mkdir()
    codex_home = tmp_path / "codex-home"
    agents = _completed_agents()
    agents[1] = {
        "agent_name": "/root/reviewer-0",
        "agent_status": "running",
    }
    _write_rollout(
        codex_home,
        session_id=session_id,
        cwd=cwd,
        output_agents=agents,
    )
    original = _result_with_child_paths()

    enriched, evidence = enrich_codex_reconciliation_result(
        original,
        session_id=session_id,
        cwd=str(cwd),
        started_at=_timestamp("2026-08-08T09:26:54.000Z"),
        ended_at=_timestamp("2026-08-08T09:27:00.000Z"),
        codex_home=str(codex_home),
    )

    assert enriched is original
    assert evidence is None


@pytest.mark.parametrize("latest_kind", ["running", "malformed"])
def test_latest_list_agents_output_cannot_fall_back_to_earlier_terminal(
    tmp_path: Path,
    latest_kind: str,
) -> None:
    session_id = "019fe093-0d77-7fd2-9860-616a2100e71d"
    cwd = tmp_path / "repo"
    cwd.mkdir()
    codex_home = tmp_path / "codex-home"
    rollout = _write_rollout(
        codex_home,
        session_id=session_id,
        cwd=cwd,
        output_agents=_completed_agents(),
    )
    if latest_kind == "running":
        latest_agents = _completed_agents()
        latest_agents[1] = {
            "agent_name": "/root/reviewer-0",
            "agent_status": "running",
        }
        raw_output = json.dumps({"agents": latest_agents})
    else:
        raw_output = "{malformed-json"
    _append_list_agents_output(
        rollout,
        call_id="call-list-agents-latest",
        call_timestamp="2026-08-08T09:26:56.000Z",
        output_timestamp="2026-08-08T09:26:56.100Z",
        raw_output=raw_output,
    )
    original = _result_with_child_paths()

    enriched, evidence = enrich_codex_reconciliation_result(
        original,
        session_id=session_id,
        cwd=str(cwd),
        started_at=_timestamp("2026-08-08T09:26:54.000Z"),
        ended_at=_timestamp("2026-08-08T09:27:00.000Z"),
        codex_home=str(codex_home),
    )

    assert enriched is original
    assert evidence is None


def test_rollout_evidence_after_current_turn_is_ignored(tmp_path: Path) -> None:
    session_id = "019fe093-0d77-7fd2-9860-616a2100e71d"
    cwd = tmp_path / "repo"
    cwd.mkdir()
    codex_home = tmp_path / "codex-home"
    _write_rollout(
        codex_home,
        session_id=session_id,
        cwd=cwd,
        output_agents=_completed_agents(),
    )
    original = _result_with_child_paths()

    enriched, evidence = enrich_codex_reconciliation_result(
        original,
        session_id=session_id,
        cwd=str(cwd),
        started_at=_timestamp("2026-08-08T09:26:54.000Z"),
        ended_at=_timestamp("2026-08-08T09:26:54.700Z"),
        codex_home=str(codex_home),
    )

    assert enriched is original
    assert evidence is None


def test_rollout_reconciliation_waits_for_delayed_function_output(
    tmp_path: Path,
) -> None:
    session_id = "019fe093-0d77-7fd2-9860-616a2100e71d"
    cwd = tmp_path / "repo"
    cwd.mkdir()
    codex_home = tmp_path / "codex-home"
    rollout = _write_rollout(
        codex_home,
        session_id=session_id,
        cwd=cwd,
        output_agents=_completed_agents(),
        output_timestamp="2026-08-08T09:27:30.000Z",
    )
    lines = rollout.read_text(encoding="utf-8").splitlines()
    delayed_output = lines.pop()
    rollout.write_text(
        "".join(f"{line}\n" for line in lines),
        encoding="utf-8",
    )

    def append_delayed_output() -> None:
        time.sleep(0.05)
        with rollout.open("a", encoding="utf-8") as handle:
            handle.write(delayed_output + "\n")

    writer = threading.Thread(target=append_delayed_output)
    writer.start()
    try:
        enriched, evidence = enrich_codex_reconciliation_result(
            _result_with_child_paths(),
            session_id=session_id,
            cwd=str(cwd),
            started_at=_timestamp("2026-08-08T09:26:54.000Z"),
            ended_at=_timestamp("2026-08-08T09:28:00.000Z"),
            codex_home=str(codex_home),
        )
    finally:
        writer.join(timeout=1)

    assert evidence is not None
    assert classify_prompt_result(enriched).outcome is PromptOutcome.COMPLETED


def test_official_codex_prompt_streams_child_rollout_terminal_before_return(
    tmp_path: Path,
) -> None:
    parent_session_id = "parent-session-live1"
    child_thread_id = "child-thread-live01"
    cwd = tmp_path / "repo"
    cwd.mkdir()
    codex_home = tmp_path / "codex-home"
    child_rollout = _write_child_rollout(
        codex_home,
        parent_session_id=parent_session_id,
        child_thread_id=child_thread_id,
        cwd=cwd,
        agent_path="/root/live-reviewer",
        lifecycle=[],
    )
    session = ACPSession(
        agent_cmd="npx",
        agent_args=["--yes", "@agentclientprotocol/codex-acp@1.1.7"],
        cwd=str(cwd),
        env={"CODEX_HOME": str(codex_home)},
    )
    session._session_id = parent_session_id

    async def exercise() -> None:
        terminal_seen = asyncio.Event()
        prompt_returned = False
        received = []

        def receive(event) -> None:
            received.append(event)
            tool_call = event.tool_call
            if tool_call is None:
                return
            if any(
                state.get("source_id") == child_thread_id
                and state.get("status") == "completed"
                for state in tool_call.subagent_states
            ):
                assert prompt_returned is False
                terminal_seen.set()

        class Connection:
            async def prompt(self, **_kwargs):
                session._dispatch_event(
                    SimpleNamespace(
                        event_type=ACPEventType.TOOL_CALL_DONE,
                        tool_call=ToolCallInfo(
                            id="spawn-live-reviewer",
                            title="spawn_agent",
                            kind="other",
                            status="completed",
                            subagent_source_id=child_thread_id,
                            subagent_path="/root/live-reviewer",
                            subagent_activity="started",
                            collaboration_tool="spawn_agent",
                            collaboration_receivers=(child_thread_id,),
                            subagent_states=(
                                {
                                    "source_id": child_thread_id,
                                    "status": "running",
                                    "message": "",
                                },
                            ),
                        ),
                    )
                )
                _append_child_lifecycle(
                    child_rollout,
                    event_type="task_started",
                    turn_id="turn-live",
                )
                await asyncio.sleep(0.03)
                _append_child_lifecycle(
                    child_rollout,
                    event_type="task_complete",
                    turn_id="turn-live",
                )
                await asyncio.wait_for(terminal_seen.wait(), timeout=0.8)
                return SimpleNamespace(stop_reason="end_turn")

        session._conn = Connection()
        result = await session.prompt(
            "run live reviewer",
            on_event=receive,
            await_goal_quiescence=False,
        )
        prompt_returned = True

        terminal_calls = [
            tool_call
            for tool_call in result.tool_calls
            if any(
                state.get("source_id") == child_thread_id
                and state.get("status") == "completed"
                for state in tool_call.subagent_states
            )
        ]
        assert terminal_calls
        assert any(
            event.tool_call is not None
            and event.tool_call.id.startswith("rollout-child-lifecycle:")
            for event in received
        )

    asyncio.run(exercise())


def test_sync_official_codex_enrichment_uses_session_home_and_emits_event(
    tmp_path: Path,
) -> None:
    session = object.__new__(SyncACPSession)
    session._agent_type = "codex"
    session._agent_args = ["@agentclientprotocol/codex-acp@1.1.7"]
    session._cwd = str(tmp_path / "repo")
    session.session_id = "019fe093-0d77-7fd2-9860-616a2100e71d"
    session._acp_session = SimpleNamespace(
        _env_override={"HOME": str(tmp_path / "isolated-home")}
    )
    original = _result_with_child_paths()
    evidence = ToolCallInfo(
        id="rollout-list-agents:call-list-agents",
        title="list_agents",
        kind="other",
        status="completed",
        collaboration_tool="list_agents",
        subagent_states=(
            {"source_id": "thread-0", "status": "completed"},
        ),
    )
    enriched = PromptResult(
        stop_reason="end_turn",
        tool_calls=[*original.tool_calls, evidence],
    )
    events = []

    with patch(
        "src.acp.codex_rollout_reconciliation.enrich_codex_reconciliation_result",
        return_value=(enriched, evidence),
    ) as enrich:
        result = SyncACPSession.enrich_child_reconciliation_result(
            session,
            original,
            started_at=1.0,
            ended_at=2.0,
            on_event=events.append,
        )

    assert result is enriched
    enrich.assert_called_once_with(
        original,
        session_id=session.session_id,
        cwd=session._cwd,
        logical_task_started_at=None,
        started_at=1.0,
        ended_at=2.0,
        codex_home=str(tmp_path / "isolated-home" / ".codex"),
    )
    assert len(events) == 1
    assert events[0].event_type is ACPEventType.TOOL_CALL_DONE
    assert events[0].tool_call is evidence


def test_sync_official_terminal_enrichment_emits_all_evidence(
    tmp_path: Path,
) -> None:
    session = object.__new__(SyncACPSession)
    session._agent_type = "codex"
    session._agent_args = ["@agentclientprotocol/codex-acp@1.2.0"]
    session._cwd = str(tmp_path / "repo")
    session.session_id = "019fe093-0d77-7fd2-9860-616a2100e71d"
    session._acp_session = SimpleNamespace(
        _env_override={"HOME": str(tmp_path / "isolated-home")}
    )
    original = PromptResult(
        stop_reason="end_turn",
        tool_calls=[
            ToolCallInfo(
                id="call-generic-wait",
                title="wait",
                kind="other",
                status="in_progress",
            )
        ],
    )
    generic_evidence = ToolCallInfo(
        id="call-generic-wait",
        title="wait",
        kind="other",
        status="completed",
    )
    child_evidence = ToolCallInfo(
        id="rollout-child-lifecycle:019fe093-0d77-7fd2-9860-616a2100e71d",
        title="list_agents",
        kind="other",
        status="completed",
    )
    enriched = PromptResult(
        stop_reason="end_turn",
        tool_calls=[generic_evidence, child_evidence],
    )
    events = []

    with patch(
        "src.acp.codex_rollout_reconciliation.enrich_codex_terminal_result",
        return_value=(enriched, (generic_evidence, child_evidence)),
    ) as enrich:
        result = SyncACPSession.enrich_terminal_evidence_result(
            session,
            original,
            started_at=1.0,
            ended_at=2.0,
            logical_task_started_at=0.5,
            on_event=events.append,
        )

    assert result is enriched
    enrich.assert_called_once_with(
        original,
        session_id=session.session_id,
        cwd=session._cwd,
        logical_task_started_at=0.5,
        started_at=1.0,
        ended_at=2.0,
        codex_home=str(tmp_path / "isolated-home" / ".codex"),
    )
    assert [event.event_type for event in events] == [
        ACPEventType.TOOL_CALL_DONE,
        ACPEventType.TOOL_CALL_DONE,
    ]
    assert [event.tool_call for event in events] == [
        generic_evidence,
        child_evidence,
    ]


def test_sync_non_official_session_does_not_read_codex_rollout(
    tmp_path: Path,
) -> None:
    session = object.__new__(SyncACPSession)
    session._agent_type = "traex"
    session._agent_args = []
    session._cwd = str(tmp_path)
    original = _result_with_child_paths()

    with patch(
        "src.acp.codex_rollout_reconciliation.enrich_codex_reconciliation_result",
    ) as enrich:
        result = SyncACPSession.enrich_child_reconciliation_result(
            session,
            original,
            started_at=1.0,
            ended_at=2.0,
        )

    assert result is original
    enrich.assert_not_called()


def test_sync_official_codex_rollout_error_remains_fail_closed(
    tmp_path: Path,
) -> None:
    session = object.__new__(SyncACPSession)
    session._agent_type = "codex"
    session._agent_args = ["@agentclientprotocol/codex-acp@1.1.7"]
    session._cwd = str(tmp_path)
    session.session_id = "019fe093-0d77-7fd2-9860-616a2100e71d"
    session._acp_session = SimpleNamespace(_env_override={})
    original = _result_with_child_paths()

    with patch(
        "src.acp.codex_rollout_reconciliation.enrich_codex_reconciliation_result",
        side_effect=OSError("rollout unavailable"),
    ):
        result = SyncACPSession.enrich_child_reconciliation_result(
            session,
            original,
            started_at=1.0,
            ended_at=2.0,
        )

    assert result is original



_INCIDENT_PARENT_ID = "parent-activity-session-0001"
_INCIDENT_CHILD_IDS = (
    "child-activity-thread-0001",
    "child-activity-thread-0002",
    "child-activity-thread-0003",
)
_INCIDENT_PATHS = (
    "/root/goal_guardian",
    "/root/visual_risk_review",
    "/root/transaction_test_audit",
)
_INCIDENT_LOGICAL_START = "2026-08-10T07:58:14.000Z"
_INCIDENT_STARTED = "2026-08-10T08:25:03.000Z"
_INCIDENT_ENDED = "2026-08-10T08:25:32.000Z"
_INCIDENT_LIFECYCLE = [
    ("2026-08-10T08:14:40.000Z", "task_started", "review-1"),
    ("2026-08-10T08:17:10.000Z", "turn_aborted", "review-1"),
    ("2026-08-10T08:17:20.000Z", "task_started", "review-2"),
    ("2026-08-10T08:18:05.000Z", "task_complete", "review-2"),
    ("2026-08-10T08:21:45.000Z", "task_started", "review-3"),
    ("2026-08-10T08:22:47.000Z", "task_complete", "review-3"),
]


def _activity_stale_result() -> PromptResult:
    return PromptResult(
        stop_reason="end_turn",
        tool_calls=[
            ToolCallInfo(
                id=f"stale-child-{index}",
                title="child activity",
                kind="other",
                status="completed",
                subagent_source_id=child_id,
                subagent_activity="started",
                subagent_states=(
                    {
                        "source_id": child_id,
                        "status": "running",
                        "message": "",
                    },
                ),
            )
            for index, child_id in enumerate(_INCIDENT_CHILD_IDS)
        ],
    )


def _incident_activities(case: str) -> list[dict[str, object]]:
    activities: list[dict[str, object]] = [
        {
            "timestamp": f"2026-08-10T08:14:{39 + index:02d}.000Z",
            "source_id": child_id,
            "agent_path": agent_path,
            "kind": "started",
        }
        for index, (child_id, agent_path) in enumerate(
            zip(_INCIDENT_CHILD_IDS, _INCIDENT_PATHS, strict=True)
        )
    ]
    if case == "valid":
        activities.extend(
            {
                "timestamp": f"2026-08-10T08:21:{44 + index:02d}.000Z",
                "source_id": child_id,
                "agent_path": agent_path,
                "kind": "interacted",
            }
            for index, (child_id, agent_path) in enumerate(
                zip(_INCIDENT_CHILD_IDS, _INCIDENT_PATHS, strict=True)
            )
        )
    elif case == "conflicting_source":
        activities.append(
            {
                "timestamp": "2026-08-10T08:14:41.000Z",
                "source_id": _INCIDENT_CHILD_IDS[0],
                "agent_path": "/root/conflict",
            }
        )
    elif case == "ambiguous_path":
        activities.append(
            {
                "timestamp": "2026-08-10T08:14:41.000Z",
                "source_id": _INCIDENT_CHILD_IDS[1],
                "agent_path": _INCIDENT_PATHS[0],
            }
        )
    elif case == "unknown_source":
        activities.append(
            {
                "timestamp": "2026-08-10T08:14:41.000Z",
                "source_id": "child-activity-thread-unknown",
                "agent_path": "/root/unknown",
            }
        )
    elif case == "incomplete_set":
        activities.pop()
    elif case == "out_of_window":
        for activity in activities:
            activity["timestamp"] = "2026-08-10T07:57:00.000Z"
    return activities


def _exercise_activity_case(
    tmp_path: Path,
    case: str,
) -> tuple[PromptResult, PromptResult, ToolCallInfo | None]:
    cwd = tmp_path / "repo"
    cwd.mkdir()
    codex_home = tmp_path / "codex-home"
    _write_rollout(
        codex_home,
        session_id=_INCIDENT_PARENT_ID,
        cwd=cwd,
        output_agents=_completed_agents(_INCIDENT_PATHS),
        activities=_incident_activities(case),
        session_timestamp=_INCIDENT_LOGICAL_START,
        call_timestamp="2026-08-10T08:25:19.000Z",
        output_timestamp="2026-08-10T08:25:19.100Z",
    )
    for child_id, agent_path in zip(
        _INCIDENT_CHILD_IDS,
        _INCIDENT_PATHS,
        strict=True,
    ):
        _write_child_rollout(
            codex_home,
            parent_session_id=_INCIDENT_PARENT_ID,
            child_thread_id=child_id,
            cwd=cwd,
            agent_path=agent_path,
            lifecycle=_INCIDENT_LIFECYCLE,
        )
    original = _activity_stale_result()
    enriched, evidence = reconciliation.enrich_codex_reconciliation_result(
        original,
        session_id=_INCIDENT_PARENT_ID,
        cwd=str(cwd),
        logical_task_started_at=_timestamp(_INCIDENT_LOGICAL_START),
        started_at=_timestamp(_INCIDENT_STARTED),
        ended_at=_timestamp(_INCIDENT_ENDED),
        codex_home=str(codex_home),
    )
    return original, enriched, evidence


def test_parent_activity_identity_recovers_stale_children_after_followups(
    tmp_path: Path,
) -> None:
    original, enriched, evidence = _exercise_activity_case(tmp_path, "valid")

    assert classify_prompt_result(original).outcome is PromptOutcome.INCOMPLETE
    assert evidence is not None
    assert {
        (state["source_id"], state["status"])
        for state in evidence.subagent_states
    } == {(child_id, "completed") for child_id in _INCIDENT_CHILD_IDS}
    assert classify_prompt_result(enriched).outcome is PromptOutcome.COMPLETED


def test_terminal_reconciliation_recovers_rejected_spawn_and_children(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "repo"
    cwd.mkdir()
    codex_home = tmp_path / "codex-home"
    rollout = _write_rollout(
        codex_home,
        session_id=_INCIDENT_PARENT_ID,
        cwd=cwd,
        output_agents=_completed_agents(_INCIDENT_PATHS),
        activities=_incident_activities("valid"),
        session_timestamp=_INCIDENT_LOGICAL_START,
        call_timestamp="2026-08-10T08:25:19.000Z",
        output_timestamp="2026-08-10T08:25:19.100Z",
    )
    _append_generic_function_output(
        rollout,
        call_id="call-spawn-rejected",
        call_timestamp="2026-08-10T08:25:23.000Z",
        output_timestamp="2026-08-10T08:25:23.100Z",
        name="spawn_agent",
        namespace="collaboration",
        output=(
            "Full-history forked agents inherit the parent agent type; omit "
            "agent_type, or spawn without a full-history fork."
        ),
    )
    for child_id, agent_path in zip(
        _INCIDENT_CHILD_IDS,
        _INCIDENT_PATHS,
        strict=True,
    ):
        _write_child_rollout(
            codex_home,
            parent_session_id=_INCIDENT_PARENT_ID,
            child_thread_id=child_id,
            cwd=cwd,
            agent_path=agent_path,
            lifecycle=_INCIDENT_LIFECYCLE,
        )
    original = _activity_stale_result()
    original.tool_calls.append(
        ToolCallInfo(
            id="call-spawn-rejected",
            title="spawn_agent",
            kind="other",
            status="in_progress",
        )
    )

    assessment = classify_prompt_result(original)
    assert assessment.incomplete_tool_calls == 4
    assert assessment.incomplete_outer_tool_calls == 1
    assert assessment.unresolved_child_tool_calls == 3

    enriched, evidence = enrich_codex_terminal_result(
        original,
        session_id=_INCIDENT_PARENT_ID,
        cwd=str(cwd),
        logical_task_started_at=_timestamp(_INCIDENT_LOGICAL_START),
        started_at=_timestamp(_INCIDENT_STARTED),
        ended_at=_timestamp(_INCIDENT_ENDED),
        codex_home=str(codex_home),
    )

    assert {item.id for item in evidence} == {
        "call-spawn-rejected",
        f"rollout-child-lifecycle:{_INCIDENT_PARENT_ID}",
    }
    evidence_by_id = {item.id: item for item in evidence}
    assert evidence_by_id["call-spawn-rejected"].status == "failed"
    assert {
        (state["source_id"], state["status"])
        for state in evidence_by_id[
            f"rollout-child-lifecycle:{_INCIDENT_PARENT_ID}"
        ].subagent_states
    } == {(child_id, "completed") for child_id in _INCIDENT_CHILD_IDS}
    final_assessment = classify_prompt_result(enriched)
    assert final_assessment.outcome is PromptOutcome.COMPLETED
    assert final_assessment.incomplete_tool_calls == 0
    assert final_assessment.incomplete_outer_tool_calls == 0
    assert final_assessment.unresolved_child_tool_calls == 0


def test_terminal_reconciliation_recovers_exact_generic_wait(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "repo"
    cwd.mkdir()
    codex_home = tmp_path / "codex-home"
    rollout = _write_rollout(
        codex_home,
        session_id="parent-generic-wait-0001",
        cwd=cwd,
        output_agents=[{"agent_name": "/root", "agent_status": "running"}],
    )
    _append_generic_function_output(
        rollout,
        call_id="call-generic-wait",
        call_timestamp="2026-08-08T09:26:55.000Z",
        output_timestamp="2026-08-08T09:26:55.100Z",
    )
    original = PromptResult(
        stop_reason="end_turn",
        tool_calls=[
            ToolCallInfo(
                id="call-generic-wait",
                title="wait",
                kind="other",
                status="in_progress",
            )
        ],
    )

    enriched, evidence = enrich_codex_terminal_result(
        original,
        session_id="parent-generic-wait-0001",
        cwd=str(cwd),
        started_at=_timestamp("2026-08-08T09:26:54.000Z"),
        ended_at=_timestamp("2026-08-08T09:27:00.000Z"),
        codex_home=str(codex_home),
    )

    assert [item.id for item in evidence] == ["call-generic-wait"]
    assert evidence[0].status == "completed"
    assert classify_prompt_result(enriched).outcome is PromptOutcome.COMPLETED


def test_terminal_reconciliation_ignores_generic_output_after_turn(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "repo"
    cwd.mkdir()
    codex_home = tmp_path / "codex-home"
    rollout = _write_rollout(
        codex_home,
        session_id="parent-generic-after-turn-0001",
        cwd=cwd,
        output_agents=[{"agent_name": "/root", "agent_status": "running"}],
    )
    _append_generic_function_output(
        rollout,
        call_id="call-generic-after-turn",
        call_timestamp="2026-08-08T09:26:55.000Z",
        output_timestamp="2026-08-08T09:27:00.100Z",
    )
    original = PromptResult(
        stop_reason="end_turn",
        tool_calls=[
            ToolCallInfo(
                id="call-generic-after-turn",
                title="wait",
                kind="other",
                status="in_progress",
            )
        ],
    )

    enriched, evidence = enrich_codex_terminal_result(
        original,
        session_id="parent-generic-after-turn-0001",
        cwd=str(cwd),
        started_at=_timestamp("2026-08-08T09:26:54.000Z"),
        ended_at=_timestamp("2026-08-08T09:27:00.000Z"),
        codex_home=str(codex_home),
    )

    assert enriched is original
    assert evidence == ()
    assert classify_prompt_result(enriched).outcome is PromptOutcome.INCOMPLETE


def test_terminal_reconciliation_rejects_whitespace_altered_call_id(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "repo"
    cwd.mkdir()
    codex_home = tmp_path / "codex-home"
    rollout = _write_rollout(
        codex_home,
        session_id="parent-generic-whitespace-0001",
        cwd=cwd,
        output_agents=[{"agent_name": "/root", "agent_status": "running"}],
    )
    _append_generic_function_output(
        rollout,
        call_id="call-generic-whitespace",
        call_timestamp="2026-08-08T09:26:55.000Z",
        output_timestamp="2026-08-08T09:26:55.100Z",
    )
    original = PromptResult(
        stop_reason="end_turn",
        tool_calls=[
            ToolCallInfo(
                id=" call-generic-whitespace ",
                title="wait",
                kind="other",
                status="in_progress",
            )
        ],
    )

    enriched, evidence = enrich_codex_terminal_result(
        original,
        session_id="parent-generic-whitespace-0001",
        cwd=str(cwd),
        started_at=_timestamp("2026-08-08T09:26:54.000Z"),
        ended_at=_timestamp("2026-08-08T09:27:00.000Z"),
        codex_home=str(codex_home),
    )

    assert enriched is original
    assert evidence == ()
    assert classify_prompt_result(enriched).outcome is PromptOutcome.INCOMPLETE


def test_terminal_reconciliation_keeps_unknown_spawn_output_unresolved(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "repo"
    cwd.mkdir()
    codex_home = tmp_path / "codex-home"
    rollout = _write_rollout(
        codex_home,
        session_id="parent-unknown-spawn-0001",
        cwd=cwd,
        output_agents=[{"agent_name": "/root", "agent_status": "running"}],
    )
    _append_generic_function_output(
        rollout,
        call_id="call-unknown-spawn",
        call_timestamp="2026-08-08T09:26:55.000Z",
        output_timestamp="2026-08-08T09:26:55.100Z",
        name="spawn_agent",
        namespace="collaboration",
        output="unrecognized spawn response",
    )
    original = PromptResult(
        stop_reason="end_turn",
        tool_calls=[
            ToolCallInfo(
                id="call-unknown-spawn",
                title="spawn_agent",
                kind="other",
                status="in_progress",
            )
        ],
    )

    enriched, evidence = enrich_codex_terminal_result(
        original,
        session_id="parent-unknown-spawn-0001",
        cwd=str(cwd),
        started_at=_timestamp("2026-08-08T09:26:54.000Z"),
        ended_at=_timestamp("2026-08-08T09:27:00.000Z"),
        codex_home=str(codex_home),
    )

    assert enriched is original
    assert evidence == ()
    assert classify_prompt_result(enriched).outcome is PromptOutcome.INCOMPLETE


def test_terminal_reconciliation_skips_rollout_without_recoverable_state(
    tmp_path: Path,
) -> None:
    original = PromptResult(stop_reason="end_turn")

    with patch.object(reconciliation, "_find_rollout") as find_rollout:
        enriched, evidence = enrich_codex_terminal_result(
            original,
            session_id="parent-no-recovery-0001",
            cwd=str(tmp_path),
            started_at=_timestamp("2026-08-08T09:26:54.000Z"),
            ended_at=_timestamp("2026-08-08T09:27:00.000Z"),
            codex_home=str(tmp_path / "codex-home"),
        )

    assert enriched is original
    assert evidence == ()
    find_rollout.assert_not_called()


def test_terminal_reconciliation_defers_parent_lookup_to_child_reconciliation(
    tmp_path: Path,
) -> None:
    original = _single_child_result()

    with (
        patch.object(reconciliation, "_find_rollout") as find_rollout,
        patch.object(
            reconciliation,
            "enrich_codex_reconciliation_result",
            return_value=(original, None),
        ) as reconcile_children,
    ):
        enriched, evidence = enrich_codex_terminal_result(
            original,
            session_id="parent-child-only-0001",
            cwd=str(tmp_path),
            started_at=_timestamp("2026-08-08T09:26:54.000Z"),
            ended_at=_timestamp("2026-08-08T09:27:00.000Z"),
            codex_home=str(tmp_path / "codex-home"),
        )

    assert enriched is original
    assert evidence == ()
    find_rollout.assert_not_called()
    reconcile_children.assert_called_once()


def test_terminal_reconciliation_ignores_child_terminal_after_turn(
    tmp_path: Path,
) -> None:
    parent_session_id = "parent-child-after-turn-0001"
    child_thread_id = "child-after-turn-0001"
    cwd = tmp_path / "repo"
    cwd.mkdir()
    codex_home = tmp_path / "codex-home"
    _write_rollout(
        codex_home,
        session_id=parent_session_id,
        cwd=cwd,
        output_agents=[
            {"agent_name": "/root", "agent_status": "running"},
            {"agent_name": "/root/reviewer", "agent_status": "running"},
        ],
        activities=[
            {
                "timestamp": "2026-08-08T09:26:55.000Z",
                "source_id": child_thread_id,
                "agent_path": "/root/reviewer",
                "kind": "started",
            }
        ],
    )
    _write_child_rollout(
        codex_home,
        parent_session_id=parent_session_id,
        child_thread_id=child_thread_id,
        cwd=cwd,
        agent_path="/root/reviewer",
        lifecycle=[
            ("2026-08-08T09:26:54.100Z", "task_started", "turn-live"),
            ("2026-08-08T09:27:00.100Z", "task_complete", "turn-live"),
        ],
    )
    original = _single_child_result(
        source_id=child_thread_id,
        path="/root/reviewer",
    )

    enriched, evidence = enrich_codex_terminal_result(
        original,
        session_id=parent_session_id,
        cwd=str(cwd),
        logical_task_started_at=_timestamp("2026-08-08T09:26:54.000Z"),
        started_at=_timestamp("2026-08-08T09:26:54.000Z"),
        ended_at=_timestamp("2026-08-08T09:27:00.000Z"),
        codex_home=str(codex_home),
    )

    assert enriched is original
    assert evidence == ()
    assert classify_prompt_result(enriched).outcome is PromptOutcome.INCOMPLETE


def test_terminal_reconciliation_does_not_upgrade_execute_call(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "repo"
    cwd.mkdir()
    codex_home = tmp_path / "codex-home"
    rollout = _write_rollout(
        codex_home,
        session_id="parent-generic-execute-0001",
        cwd=cwd,
        output_agents=[{"agent_name": "/root", "agent_status": "running"}],
    )
    _append_generic_function_output(
        rollout,
        call_id="call-generic-execute",
        call_timestamp="2026-08-08T09:26:55.000Z",
        output_timestamp="2026-08-08T09:26:55.100Z",
    )
    original = PromptResult(
        stop_reason="end_turn",
        tool_calls=[
            ToolCallInfo(
                id="call-generic-execute",
                title="wait",
                kind="execute",
                status="in_progress",
            )
        ],
    )

    enriched, evidence = enrich_codex_terminal_result(
        original,
        session_id="parent-generic-execute-0001",
        cwd=str(cwd),
        started_at=_timestamp("2026-08-08T09:26:54.000Z"),
        ended_at=_timestamp("2026-08-08T09:27:00.000Z"),
        codex_home=str(codex_home),
    )

    assert enriched is original
    assert evidence == ()
    assert classify_prompt_result(enriched).outcome is PromptOutcome.INCOMPLETE


def test_terminal_reconciliation_rejects_duplicate_generic_pairs(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "repo"
    cwd.mkdir()
    codex_home = tmp_path / "codex-home"
    rollout = _write_rollout(
        codex_home,
        session_id="parent-generic-duplicate-0001",
        cwd=cwd,
        output_agents=[{"agent_name": "/root", "agent_status": "running"}],
    )
    for suffix in ("55", "56"):
        _append_generic_function_output(
            rollout,
            call_id="call-generic-duplicate",
            call_timestamp=f"2026-08-08T09:26:{suffix}.000Z",
            output_timestamp=f"2026-08-08T09:26:{suffix}.100Z",
        )
    original = PromptResult(
        stop_reason="end_turn",
        tool_calls=[
            ToolCallInfo(
                id="call-generic-duplicate",
                title="wait",
                kind="other",
                status="in_progress",
            )
        ],
    )

    enriched, evidence = enrich_codex_terminal_result(
        original,
        session_id="parent-generic-duplicate-0001",
        cwd=str(cwd),
        started_at=_timestamp("2026-08-08T09:26:54.000Z"),
        ended_at=_timestamp("2026-08-08T09:27:00.000Z"),
        codex_home=str(codex_home),
    )

    assert enriched is original
    assert evidence == ()
    assert classify_prompt_result(enriched).outcome is PromptOutcome.INCOMPLETE


def test_terminal_reconciliation_requires_exact_rejected_spawn_call_id(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "repo"
    cwd.mkdir()
    codex_home = tmp_path / "codex-home"
    rollout = _write_rollout(
        codex_home,
        session_id="parent-spawn-ambiguity-0001",
        cwd=cwd,
        output_agents=[{"agent_name": "/root", "agent_status": "running"}],
    )
    _append_generic_function_output(
        rollout,
        call_id="call-spawn-rejected",
        call_timestamp="2026-08-08T09:26:55.000Z",
        output_timestamp="2026-08-08T09:26:55.100Z",
        name="spawn_agent",
        namespace="collaboration",
        output=(
            "Full-history forked agents inherit the parent agent type; omit "
            "agent_type, or spawn without a full-history fork."
        ),
    )
    original = PromptResult(
        stop_reason="end_turn",
        tool_calls=[
            ToolCallInfo(
                id="collab-item-rejected-spawn",
                title="spawn_agent",
                kind="other",
                status="in_progress",
            )
        ],
    )

    enriched, evidence = enrich_codex_terminal_result(
        original,
        session_id="parent-spawn-ambiguity-0001",
        cwd=str(cwd),
        started_at=_timestamp("2026-08-08T09:26:54.000Z"),
        ended_at=_timestamp("2026-08-08T09:27:00.000Z"),
        codex_home=str(codex_home),
    )

    assert enriched is original
    assert evidence == ()
    assert classify_prompt_result(enriched).outcome is PromptOutcome.INCOMPLETE


def test_terminal_reconciliation_requires_matching_generic_call(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "repo"
    cwd.mkdir()
    codex_home = tmp_path / "codex-home"
    rollout = _write_rollout(
        codex_home,
        session_id="parent-generic-output-0001",
        cwd=cwd,
        output_agents=[{"agent_name": "/root", "agent_status": "running"}],
    )
    with rollout.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": "2026-08-08T09:26:54.900Z",
                    "type": "response_item",
                    "payload": {
                        "type": "function_call_output",
                        "call_id": "call-unpaired",
                        "output": [{"type": "text", "text": "done"}],
                    },
                }
            )
            + "\n"
        )
    original = PromptResult(
        stop_reason="end_turn",
        tool_calls=[
            ToolCallInfo(
                id="call-unpaired",
                title="wait",
                kind="other",
                status="in_progress",
            )
        ],
    )

    enriched, evidence = enrich_codex_terminal_result(
        original,
        session_id="parent-generic-output-0001",
        cwd=str(cwd),
        started_at=_timestamp("2026-08-08T09:26:54.000Z"),
        ended_at=_timestamp("2026-08-08T09:27:00.000Z"),
        codex_home=str(codex_home),
    )

    assert enriched is original
    assert evidence == ()
    assert classify_prompt_result(enriched).outcome is PromptOutcome.INCOMPLETE


@pytest.mark.parametrize(
    "case",
    [
        "conflicting_source",
        "ambiguous_path",
        "unknown_source",
        "incomplete_set",
        "out_of_window",
    ],
)
def test_parent_activity_identity_remains_fail_closed(
    tmp_path: Path,
    case: str,
) -> None:
    original, enriched, evidence = _exercise_activity_case(tmp_path, case)

    assert enriched is original
    assert evidence is None
    assert classify_prompt_result(enriched).outcome is PromptOutcome.INCOMPLETE


def test_latest_list_agents_requires_empty_arguments(tmp_path: Path) -> None:
    cwd = tmp_path / "repo"
    cwd.mkdir()
    rollout = _write_rollout(
        tmp_path / "codex-home",
        session_id="parent-activity-session-0003",
        cwd=cwd,
        output_agents=[{"agent_name": "/root", "agent_status": "running"}],
        arguments='{"scope":"all"}',
    )

    assert reconciliation._latest_list_agents_output(
        rollout,
        started_at=_timestamp("2026-08-08T09:26:54.000Z"),
        ended_at=_timestamp("2026-08-08T09:27:00.000Z"),
    ) is None


def test_latest_list_agents_uses_append_order_not_timestamp(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "repo"
    cwd.mkdir()
    rollout = _write_rollout(
        tmp_path / "codex-home",
        session_id="parent-activity-session-0004",
        cwd=cwd,
        output_agents=[{"agent_name": "/root", "agent_status": "running"}],
    )
    _append_list_agents_output(
        rollout,
        call_id="call-appended-second",
        call_timestamp="2026-08-08T09:26:54.500Z",
        output_timestamp="2026-08-08T09:26:54.600Z",
        raw_output=json.dumps(
            {
                "agents": [
                    {"agent_name": "/root", "agent_status": "running"},
                    {
                        "agent_name": "/root/reviewer",
                        "agent_status": "completed",
                    },
                ]
            }
        ),
    )

    latest = reconciliation._latest_list_agents_output(
        rollout,
        started_at=_timestamp("2026-08-08T09:26:54.000Z"),
        ended_at=_timestamp("2026-08-08T09:27:00.000Z"),
    )
    assert latest is not None
    assert latest[0] == "call-appended-second"


@pytest.mark.parametrize(
    "agents",
    [
        [{"agent_name": "/root/reviewer", "agent_status": "completed"}],
        [
            {"agent_name": "/root", "agent_status": "running"},
            {"agent_name": "/root", "agent_status": "running"},
            {"agent_name": "/root/reviewer", "agent_status": "completed"},
        ],
    ],
)
def test_rollout_evidence_requires_exactly_one_root(
    agents: list[dict[str, object]],
) -> None:
    assert reconciliation._build_evidence(
        _single_child_result(
            source_id="child-root-check-0001",
            path="/root/reviewer",
        ),
        call_id="call-root-check",
        agents=agents,
    ) is None


@pytest.mark.parametrize("reader", ["parent", "child"])
@pytest.mark.parametrize(
    "suffix",
    [b'{"timestamp":"2026-08-10T08:25:20.000Z"', b"{malformed-json\n"],
)
def test_corrupt_rollout_tail_never_reuses_earlier_evidence(
    tmp_path: Path,
    reader: str,
    suffix: bytes,
) -> None:
    cwd = tmp_path / "repo"
    cwd.mkdir()
    codex_home = tmp_path / "codex-home"
    if reader == "parent":
        rollout = _write_rollout(
            codex_home,
            session_id="parent-corrupt-tail-0001",
            cwd=cwd,
            output_agents=_completed_agents(),
        )
    else:
        rollout = _write_child_rollout(
            codex_home,
            parent_session_id="parent-corrupt-tail-0002",
            child_thread_id="child-corrupt-tail-0002",
            cwd=cwd,
            agent_path="/root/reviewer",
            lifecycle=[
                ("2026-08-08T09:26:54.100Z", "task_started", "review"),
                ("2026-08-08T09:26:55.100Z", "task_complete", "review"),
            ],
        )
    with rollout.open("ab") as handle:
        handle.write(suffix)

    if reader == "parent":
        observed = reconciliation._latest_list_agents_output(
            rollout,
            started_at=_timestamp("2026-08-08T09:26:54.000Z"),
            ended_at=_timestamp("2026-08-08T09:27:00.000Z"),
        )
    else:
        observed = reconciliation._latest_child_turn_observation(
            rollout,
            logical_task_started_at=_timestamp(
                "2026-08-08T09:26:54.000Z"
            ),
            ended_at=_timestamp("2026-08-08T09:27:00.000Z"),
        )
    assert observed is None


def test_parent_rollout_partial_latest_list_is_retried_until_complete(
    tmp_path: Path,
) -> None:
    parent_id = "parent-corrupt-tail-0003"
    cwd = tmp_path / "repo"
    cwd.mkdir()
    codex_home = tmp_path / "codex-home"
    rollout = _write_rollout(
        codex_home,
        session_id=parent_id,
        cwd=cwd,
        output_agents=_completed_agents(),
    )
    call = json.dumps(
        {
            "timestamp": "2026-08-08T09:26:56.000Z",
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "list_agents",
                "namespace": "collaboration",
                "arguments": "{}",
                "call_id": "call-completed-after-partial",
            },
        }
    )
    output = json.dumps(
        {
            "timestamp": "2026-08-08T09:26:56.100Z",
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call-completed-after-partial",
                "output": json.dumps({"agents": _completed_agents()}),
            },
        }
    )
    split_at = len(call) // 2
    with rollout.open("a", encoding="utf-8") as handle:
        handle.write(call[:split_at])

    def finish_append() -> None:
        time.sleep(0.1)
        with rollout.open("a", encoding="utf-8") as handle:
            handle.write(call[split_at:] + "\n" + output + "\n")

    writer = threading.Thread(target=finish_append)
    writer.start()
    try:
        enriched, evidence = reconciliation.enrich_codex_reconciliation_result(
            _result_with_child_paths(),
            session_id=parent_id,
            cwd=str(cwd),
            started_at=_timestamp("2026-08-08T09:26:54.000Z"),
            ended_at=_timestamp("2026-08-08T09:27:00.000Z"),
            codex_home=str(codex_home),
        )
    finally:
        writer.join(timeout=1)

    assert evidence is not None
    assert classify_prompt_result(enriched).outcome is PromptOutcome.COMPLETED


def _nested_monitor_evidence(
    tmp_path: Path,
    *,
    complete_parent_chain: bool,
) -> ToolCallInfo | None:
    parent_id = "parent-nested-identity-0001"
    direct_id = "child-nested-identity-0002"
    child_id = "child-nested-identity-0003"
    direct_path = "/root/reviewer" if complete_parent_chain else "/root/other"
    child_path = (
        "/root/reviewer/nested" if complete_parent_chain else "/root/direct"
    )
    cwd = tmp_path / "repo"
    cwd.mkdir()
    codex_home = tmp_path / "codex-home"
    terminal = [
        ("2026-08-08T09:26:54.100Z", "task_started", "turn"),
        ("2026-08-08T09:26:55.100Z", "task_complete", "turn"),
    ]
    _write_child_rollout(
        codex_home,
        parent_session_id=parent_id,
        child_thread_id=direct_id,
        cwd=cwd,
        agent_path=direct_path,
        lifecycle=terminal,
    )
    _write_child_rollout(
        codex_home,
        parent_session_id=parent_id,
        child_thread_id=child_id,
        recorded_parent_id=direct_id,
        cwd=cwd,
        agent_path=child_path if complete_parent_chain else None,
        lifecycle=terminal,
    )
    monitor = reconciliation.CodexChildLifecycleMonitor(
        parent_session_id=parent_id,
        cwd=str(cwd),
        logical_task_started_at=_timestamp("2026-08-08T09:26:54.000Z"),
        codex_home=str(codex_home),
    )
    monitor.observe_result(
        PromptResult(
            stop_reason="end_turn",
            tool_calls=[
                ToolCallInfo(
                    id=f"activity-{source_id}",
                    title="sub_agent_activity",
                    kind="other",
                    status="completed",
                    subagent_source_id=source_id,
                    subagent_path=agent_path,
                    subagent_activity="started",
                )
                for source_id, agent_path in (
                    (direct_id, direct_path),
                    (child_id, child_path),
                )
            ],
        )
    )
    return monitor.poll(
        ended_at=_timestamp("2026-08-08T09:27:00.000Z"),
        require_all_terminal=True,
        emit_unchanged=True,
    )


def test_nested_child_cannot_impersonate_direct_child_without_recorded_path(
    tmp_path: Path,
) -> None:
    assert _nested_monitor_evidence(
        tmp_path,
        complete_parent_chain=False,
    ) is None


def test_nested_child_with_complete_parent_path_mapping_is_valid(
    tmp_path: Path,
) -> None:
    evidence = _nested_monitor_evidence(tmp_path, complete_parent_chain=True)

    assert evidence is not None
    assert {state["status"] for state in evidence.subagent_states} == {
        "completed"
    }


def test_authoritative_turn_start_cannot_close_with_unidentified_terminal(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "repo"
    cwd.mkdir()
    rollout = _write_child_rollout(
        tmp_path / "codex-home",
        parent_session_id="parent-turn-id-identity-0001",
        child_thread_id="child-turn-id-identity-0001",
        cwd=cwd,
        agent_path="/root/reviewer",
        lifecycle=[
            ("2026-08-08T09:26:54.100Z", "task_started", "authoritative-turn"),
            ("2026-08-08T09:26:55.100Z", "task_complete", None),
        ],
    )

    observation = reconciliation._latest_child_turn_observation(
        rollout,
        logical_task_started_at=_timestamp("2026-08-08T09:26:54.000Z"),
        ended_at=_timestamp("2026-08-08T09:27:00.000Z"),
    )

    assert observation is not None
    assert observation.status == "running"
