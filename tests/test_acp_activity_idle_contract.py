from __future__ import annotations

import concurrent.futures
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.acp.models import ACPEvent, ACPEventType, PromptResult, ToolCallInfo
from src.acp.sync_adapter import SyncACPSession


class _PromptSource:
    callback = None

    def prompt(self, _text, **kwargs):
        self.callback = kwargs["on_event"]
        return object()


class _Future:
    def __init__(self, source: _PromptSource, clock: list[float], event: object) -> None:
        self.source = source
        self.clock = clock
        self.event = event
        self.calls = 0
        self.cancelled = False

    def result(self, timeout=0):
        del timeout
        if self.cancelled:
            raise concurrent.futures.CancelledError()
        self.calls += 1
        if self.calls == 1:
            self.clock[0] = 4.0
            self.source.callback(self.event)
            raise TimeoutError()
        if self.calls == 2:
            self.clock[0] = 6.0
            raise TimeoutError()
        return PromptResult(stop_reason="end_turn", text="ok")

    def done(self) -> bool:
        return self.cancelled or self.calls >= 3

    def cancel(self) -> bool:
        self.cancelled = True
        return True


def _session(event: object):
    clock = [0.0]
    source = _PromptSource()
    future = _Future(source, clock, event)
    session = SyncACPSession.__new__(SyncACPSession)
    session._acp_session = source
    session._loop = object()
    session._active_future = None
    session._force_dead = False
    session._log_failures = False
    session._agent_type = "test"
    session.last_active = 0.0
    session.message_count = 0
    session.last_query = ""
    session._start_watchdog = lambda: None

    def cancel(wait=True, timeout=2.0):
        del wait, timeout
        return future.cancel()

    session.cancel = cancel
    return session, future, clock


def _meaningful(event: object) -> bool:
    text = str(getattr(event, "text", "") or "").strip()
    event_type = getattr(event, "event_type", None)
    return bool(text) or event_type in {
        ACPEventType.TOOL_CALL_START,
        ACPEventType.TOOL_CALL_UPDATE,
        ACPEventType.TOOL_CALL_DONE,
        ACPEventType.PLAN_UPDATE,
    }


def _send(session, future, clock, *, predicate):
    with (
        patch("src.acp.sync_adapter.asyncio.run_coroutine_threadsafe", return_value=future),
        patch("src.acp.sync_adapter.time.time", side_effect=lambda: clock[0]),
    ):
        return session._send_prompt_once(
            "work",
            timeout=100,
            idle_timeout=5,
            activity_predicate=predicate,
        )


def test_default_activity_semantics_still_count_every_event() -> None:
    heartbeat = SimpleNamespace(event_type="heartbeat", text="")
    session, future, clock = _session(heartbeat)
    assert _send(session, future, clock, predicate=None).text == "ok"


def test_explicit_activity_predicate_ignores_heartbeat_and_empty_events() -> None:
    heartbeat = SimpleNamespace(event_type="heartbeat", text="")
    session, future, clock = _session(heartbeat)
    with pytest.raises(TimeoutError):
        _send(session, future, clock, predicate=_meaningful)


@pytest.mark.parametrize(
    "event",
    [
        ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="working"),
        ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(id="1", title="read", kind="read", status="in_progress"),
        ),
        ACPEvent(event_type=ACPEventType.PLAN_UPDATE),
    ],
)
def test_explicit_activity_predicate_extends_idle_for_meaningful_events(event) -> None:
    session, future, clock = _session(event)
    assert _send(session, future, clock, predicate=_meaningful).text == "ok"

