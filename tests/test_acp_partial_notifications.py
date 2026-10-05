"""Partial adapter lifecycle notifications must reach the SDK client."""

import copy
import sys

import pytest
from acp.schema import SessionNotification
from acp.task import RpcTask, RpcTaskKind
from pydantic import ValidationError

from src.acp.collaboration import merge_tool_call_sequence
from src.acp.models import ACPEventType, ToolCallInfo
from src.acp.session import ACPSession
from src.acp.transport import LateFrameTolerantMessageQueue
from tests.test_acp_stdio_integration import _FAKE_AGENT_CODE


async def _dequeue(message, *, kind=RpcTaskKind.NOTIFICATION):
    queue = LateFrameTolerantMessageQueue()
    original = RpcTask(kind, message)
    await queue.publish(original)
    await queue.close()
    tasks = [task async for task in queue]
    assert len(tasks) == 1
    return original, tasks[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("title", [None, "absent"])
async def test_partial_tool_start_preserves_terminal_status_and_metadata(title):
    update = {
        "sessionUpdate": "tool_call",
        "toolCallId": "child-call",
        "status": "completed",
        "rawOutput": {"result": "done"},
        "_meta": {"codex": {"toolName": "list_agents"}},
    }
    if title is None:
        update["title"] = None
    message = {
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {"sessionId": "test-session", "update": update},
    }
    before = copy.deepcopy(message)
    original, received = await _dequeue(message)

    parsed = SessionNotification.model_validate(received.message["params"])
    assert parsed.update.session_update == "tool_call_update"
    assert parsed.update.status == "completed"
    assert parsed.update.tool_call_id == "child-call"
    assert parsed.update.raw_output == {"result": "done"}
    assert parsed.update.field_meta == update["_meta"]
    assert original.message == before


@pytest.mark.asyncio
async def test_queue_preserves_valid_starts_requests_and_other_validation_errors():
    message = {
        "method": "session/update",
        "params": {
            "sessionId": "test-session",
            "update": {"sessionUpdate": "tool_call", "toolCallId": "call", "title": "wait"},
        },
    }
    original, received = await _dequeue(message)
    assert received is original
    assert SessionNotification.model_validate(received.message["params"]).update.session_update == "tool_call"
    del message["params"]["update"]["title"]
    original, received = await _dequeue(message, kind=RpcTaskKind.REQUEST)
    assert received is original
    with pytest.raises(ValidationError):
        SessionNotification.model_validate(received.message["params"])
    del message["params"]["update"]["toolCallId"]
    _, received = await _dequeue(message)
    with pytest.raises(ValidationError):
        SessionNotification.model_validate(received.message["params"])


@pytest.mark.asyncio
async def test_stdio_partial_terminal_notification_finishes_the_existing_tool(tmp_path, caplog):
    code = _FAKE_AGENT_CODE.replace(
        'await self._conn.session_update(session_id=session_id, update=update_agent_message_text("hello-from-fake"))',
        '''await self._conn._conn.send_notification("session/update", {
                "sessionId": session_id,
                "update": {"sessionUpdate": "tool_call", "toolCallId": "call", "title": "execute", "status": "in_progress"},
            })
            await self._conn._conn.send_notification("session/update", {
                "sessionId": session_id,
                "update": {"sessionUpdate": "tool_call", "toolCallId": "call", "status": "completed"},
            })
            await self._conn.session_update(session_id=session_id, update=update_agent_message_text("hello-from-fake"))''',
    )
    session = ACPSession(agent_cmd=sys.executable, agent_args=["-u", "-c", code], cwd=str(tmp_path))
    events = []
    try:
        await session.start()
        result = await session.prompt("execute the task", on_event=events.append)
        assert result.stop_reason == "end_turn"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].status == "completed"
        assert result.tool_calls[0].title == "execute"
        assert any(
            event.event_type is ACPEventType.TOOL_CALL_START
            and event.tool_call.title == "execute"
            for event in events
        )
        assert any(event.event_type is ACPEventType.TOOL_CALL_DONE for event in events)
        assert "Unhandled error while handling notification" not in caplog.text
    finally:
        await session.close()


def test_partial_title_merge_preserves_same_invocation_without_leaking_across_turns():
    start = ToolCallInfo(id="reused", title="first invocation", kind="execute", status="in_progress")
    done = ToolCallInfo(id="reused", title="", kind="execute", status="completed")
    assert merge_tool_call_sequence([start, done])[0].title == "first invocation"
    assert merge_tool_call_sequence([start, done], generation_boundary_index=1)[0].title == ""
    child_start = ToolCallInfo(id="child", title="review code", kind="other", status="in_progress", collaboration_tool="spawn_agent")
    child_done = ToolCallInfo(id="child", title="", kind="other", status="completed", collaboration_tool="spawn_agent")
    assert merge_tool_call_sequence([child_start, child_done])[0].title == "review code"
