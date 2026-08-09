"""Regression tests for strict Codex rollout child-state reconciliation."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.acp.codex_rollout_reconciliation import (
    enrich_codex_reconciliation_result,
)
from src.acp.models import ACPEventType, PromptResult, ToolCallInfo
from src.acp.outcome import PromptOutcome, classify_prompt_result
from src.acp.sync_adapter import SyncACPSession


def _timestamp(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _write_rollout(
    root: Path,
    *,
    session_id: str,
    cwd: Path,
    output_agents: list[dict],
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
            "timestamp": "2026-08-08T09:00:00.000Z",
            "type": "session_meta",
            "payload": {"id": session_id, "cwd": str(cwd)},
        },
        {
            "timestamp": "2026-08-08T09:26:54.695Z",
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "list_agents",
                "namespace": "collaboration",
                "arguments": "{}",
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


def _append_list_agents_output(
    path: Path,
    *,
    call_id: str,
    call_timestamp: str,
    output_timestamp: str,
    raw_output: str,
) -> None:
    events = [
        {
            "timestamp": call_timestamp,
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "list_agents",
                "namespace": "collaboration",
                "arguments": "{}",
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


def _completed_agents() -> list[dict]:
    return [
        {"agent_name": "/root", "agent_status": "running"},
        *[
            {
                "agent_name": f"/root/reviewer-{index}",
                "agent_status": {"completed": "sensitive final answer"},
            }
            for index in range(3)
        ],
    ]


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
        started_at=1.0,
        ended_at=2.0,
        codex_home=str(tmp_path / "isolated-home" / ".codex"),
    )
    assert len(events) == 1
    assert events[0].event_type is ACPEventType.TOOL_CALL_DONE
    assert events[0].tool_call is evidence


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
