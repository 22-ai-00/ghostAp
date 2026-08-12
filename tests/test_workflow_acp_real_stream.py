"""Real ACP update adaptation into Workflow's full execution stream."""

from __future__ import annotations

import asyncio

import pytest
from acp.schema import ToolCallProgress, ToolCallStart

from src.acp.client import GhostAPClient
from src.acp.models import ACPEvent
from src.card.events import CardEvent
from src.card.state.models import CardState
from src.card.state.reducer import reduce_card_state
from src.card.stream_bridge import ACPStreamBridge


class _StateSink:
    def __init__(self) -> None:
        self.events: list[CardEvent] = []
        self.state = CardState()

    def dispatch(self, event: CardEvent) -> None:
        self.events.append(event)
        self.state = reduce_card_state(self.state, event)


def _tool_start(
    raw_input: object,
    *,
    field_meta: dict | None = None,
) -> ToolCallStart:
    payload = {
            "sessionUpdate": "tool_call",
            "toolCallId": "real-provider-tool-id",
            "title": "bash",
            "kind": "execute",
            "status": "in_progress",
            "rawInput": raw_input,
    }
    if field_meta is not None:
        payload["_meta"] = field_meta
    return ToolCallStart.model_validate(payload)


def _tool_progress(
    *,
    status: str,
    raw_input: object,
    raw_output: object,
    field_meta: dict | None = None,
) -> ToolCallProgress:
    payload = {
            "sessionUpdate": "tool_call_update",
            "toolCallId": "real-provider-tool-id",
            "title": "bash",
            "kind": "execute",
            "status": status,
            "rawInput": raw_input,
            "rawOutput": raw_output,
    }
    if field_meta is not None:
        payload["_meta"] = field_meta
    return ToolCallProgress.model_validate(payload)


def _tool_blocks(sink: _StateSink):
    return [block for block in sink.state.blocks if block.kind == "tool_call"]


def test_real_acp_updates_keep_safe_full_workflow_payload_and_compact_default() -> None:
    input_tail = "REAL_ACP_INPUT_TAIL"
    progress_tail = "REAL_ACP_PROGRESS_OUTPUT_TAIL"
    terminal_tail = "REAL_ACP_TERMINAL_OUTPUT_TAIL"
    secret = "REAL_ACP_SECRET_MUST_NOT_LEAK"
    raw_input = {
        "command": "printf input " + "i" * 5000 + input_tail,
        "api_key": secret,
    }
    progress_output = {
        "stdout": "p" * 5000 + progress_tail,
        "access_token": secret,
    }
    terminal_output = {
        "stdout": "o" * 13000 + terminal_tail,
        "access_token": secret,
    }

    full_sink = _StateSink()
    compact_sink = _StateSink()
    full_bridge = ACPStreamBridge(full_sink, preserve_tool_content=True)
    compact_bridge = ACPStreamBridge(compact_sink)
    normalized: list[ACPEvent] = []

    def on_event(event: ACPEvent) -> None:
        normalized.append(event)
        full_bridge.on_event(event)
        compact_bridge.on_event(event)

    client = GhostAPClient(
        on_event=on_event,
        capture_full_tool_content=True,
    )
    asyncio.run(client.session_update("session", _tool_start(raw_input)))
    asyncio.run(
        client.session_update(
            "session",
            _tool_progress(
                status="in_progress",
                raw_input=raw_input,
                raw_output=progress_output,
            ),
        )
    )
    asyncio.run(
        client.session_update(
            "session",
            _tool_progress(
                status="completed",
                raw_input=raw_input,
                raw_output=terminal_output,
            ),
        )
    )

    assert [event.event_type.value for event in normalized] == [
        "tool_call_start",
        "tool_call_update",
        "tool_call_done",
    ]
    full_tools = _tool_blocks(full_sink)
    assert len(full_tools) == 1
    full_tool = full_tools[0]
    assert full_tool.status == "completed"
    assert input_tail in full_tool.tool_input
    assert progress_tail in full_tool.content
    assert terminal_tail in full_tool.tool_output
    assert secret not in "\n".join(
        (full_tool.tool_input, full_tool.content, full_tool.tool_output)
    )
    assert "<redacted>" in full_tool.tool_input
    assert "<redacted>" in full_tool.content
    assert "<redacted>" in full_tool.tool_output

    # The ordinary bridge keeps the existing bounded projection rather than
    # retaining the Workflow-only full payload.
    compact_tools = _tool_blocks(compact_sink)
    assert len(compact_tools) == 1
    compact_tool = compact_tools[0]
    assert input_tail not in compact_tool.tool_input
    assert progress_tail in compact_tool.content
    assert input_tail not in compact_tool.content
    assert terminal_tail not in compact_tool.tool_output
    assert secret not in "\n".join(
        (compact_tool.tool_input, compact_tool.content, compact_tool.tool_output)
    )
    assert len(compact_tool.tool_input) < 4100
    assert len(compact_tool.tool_output) < 12100


def test_real_acp_workflow_payload_redacts_collaboration_ids_and_paths() -> None:
    child_id = "opaque-child-thread-93f7"
    receiver_id = "opaque-receiver-thread-a18c"
    child_path = "/root/private-child-reviewer"
    receiver_path = "/root/private-receiver-auditor"
    visible_input = "VISIBLE_COLLABORATION_INPUT"
    visible_progress = "VISIBLE_COLLABORATION_PROGRESS"
    visible_output = "VISIBLE_COLLABORATION_OUTPUT"
    field_meta = {
        "codex": {
            "subagent": {
                "threadId": child_id,
                "path": child_path,
                "activity": "started",
            },
            "collaboration": {
                "tool": "wait_agent",
                "receiverThreadIds": [receiver_id],
            },
        }
    }

    def payload(marker: str) -> dict:
        return {
            "prompt": marker,
            "agentThreadId": child_id,
            "subagent_path": child_path,
            "receiverThreadIds": [receiver_id],
            "agentsStates": {
                receiver_id: {
                    "source_id": receiver_id,
                    "path": receiver_path,
                    "status": "running",
                    "message": "visible child status",
                }
            },
            "note": (
                f"known opaque values: {child_id} {receiver_id} "
                f"{child_path} {receiver_path}"
            ),
        }

    sink = _StateSink()
    bridge = ACPStreamBridge(sink, preserve_tool_content=True)
    client = GhostAPClient(
        on_event=bridge.on_event,
        capture_full_tool_content=True,
    )
    raw_input = payload(visible_input)

    asyncio.run(
        client.session_update(
            "session",
            _tool_start(raw_input, field_meta=field_meta),
        )
    )
    asyncio.run(
        client.session_update(
            "session",
            _tool_progress(
                status="in_progress",
                raw_input=payload(visible_progress),
                raw_output=payload(visible_progress),
                field_meta=field_meta,
            ),
        )
    )
    asyncio.run(
        client.session_update(
            "session",
            _tool_progress(
                status="completed",
                raw_input=payload(visible_output),
                raw_output=payload(visible_output),
                field_meta=field_meta,
            ),
        )
    )

    tools = _tool_blocks(sink)
    assert len(tools) == 1
    projected = "\n".join(
        (tools[0].tool_input, tools[0].content, tools[0].tool_output)
    )
    for marker in (visible_input, visible_progress, visible_output):
        assert marker in projected
    assert "visible child status" in projected
    for opaque_value in (
        "real-provider-tool-id",
        child_id,
        receiver_id,
        child_path,
        receiver_path,
    ):
        assert opaque_value not in projected
    assert "<redacted:agent_id>" in projected
    assert "<redacted:agent_path>" in projected


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_real_terminal_only_acp_progress_creates_workflow_tool_block(
    status: str,
) -> None:
    output_tail = "TERMINAL_ONLY_REAL_ACP_TAIL"
    secret = "TERMINAL_ONLY_SECRET_MUST_NOT_LEAK"
    sink = _StateSink()
    bridge = ACPStreamBridge(sink, preserve_tool_content=True)
    client = GhostAPClient(
        on_event=bridge.on_event,
        capture_full_tool_content=True,
    )

    asyncio.run(
        client.session_update(
            "session",
            _tool_progress(
                status=status,
                raw_input={"command": "echo ignored"},
                raw_output={
                    "stdout": "z" * 13000 + output_tail,
                    "password": secret,
                },
            ),
        )
    )

    tools = _tool_blocks(sink)
    assert len(tools) == 1
    assert tools[0].status == status
    assert tools[0].tool_name == "bash"
    assert output_tail in tools[0].tool_output
    assert secret not in tools[0].tool_output
    assert "<redacted>" in tools[0].tool_output


def test_real_in_progress_update_without_start_materializes_active_tool_block() -> None:
    output_tail = "MISSING_START_PROGRESS_OUTPUT_TAIL"
    secret = "MISSING_START_PROGRESS_SECRET"
    sink = _StateSink()
    bridge = ACPStreamBridge(sink, preserve_tool_content=True)
    client = GhostAPClient(
        on_event=bridge.on_event,
        capture_full_tool_content=True,
    )

    asyncio.run(
        client.session_update(
            "session",
            _tool_progress(
                status="in_progress",
                raw_input={"command": "echo stale input"},
                raw_output={
                    "stdout": "u" * 5000 + output_tail,
                    "authorization": secret,
                },
            ),
        )
    )

    tools = _tool_blocks(sink)
    assert len(tools) == 1
    assert tools[0].status == "active"
    assert tools[0].tool_name == "bash"
    assert output_tail in tools[0].content
    assert secret not in tools[0].content
    assert "<redacted>" in tools[0].content
