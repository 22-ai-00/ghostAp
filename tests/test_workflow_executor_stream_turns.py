"""Per-turn ACP stream isolation for Workflow direct Agent calls."""

from __future__ import annotations

import concurrent.futures
import re
import threading
from types import SimpleNamespace
from unittest.mock import patch

from src.acp.models import ACPEvent, ACPEventType, ToolCallInfo
from src.card.events import CardEvent, CardEventType
from src.card.state.models import CardState
from src.card.state.reducer import reduce_card_state
from src.workflow_engine.executor import AgentExecutor
from src.workflow_engine.models import AgentCallParams


def _emit_turn(on_event, marker: str) -> None:
    assert on_event is not None
    on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=f"text-{marker}"))
    on_event(
        ACPEvent(event_type=ACPEventType.THOUGHT_CHUNK, text=f"reason-{marker}")
    )
    # Some providers recycle a tool ID even within one prompt turn. Both
    # logical invocations must survive as separate, chronologically ordered
    # blocks, as must reuse of the same ID in later turns.
    for invocation in ("a", "b"):
        on_event(
            ACPEvent(
                event_type=ACPEventType.TOOL_CALL_START,
                tool_call=ToolCallInfo(
                    id="provider-reused-tool-id",
                    title=f"tool-{marker}-{invocation}",
                    kind="search",
                    status="in_progress",
                    content=f"input-{marker}-{invocation}",
                ),
            )
        )
        on_event(
            ACPEvent(
                event_type=ACPEventType.TOOL_CALL_UPDATE,
                tool_call=ToolCallInfo(
                    id="provider-reused-tool-id",
                    title=f"tool-{marker}-{invocation}",
                    kind="search",
                    status="in_progress",
                    content=f"update-{marker}-{invocation}",
                ),
            )
        )
        on_event(
            ACPEvent(
                event_type=ACPEventType.TOOL_CALL_DONE,
                tool_call=ToolCallInfo(
                    id="provider-reused-tool-id",
                    title=f"tool-{marker}-{invocation}",
                    kind="search",
                    status="completed",
                    content=f"output-{marker}-{invocation}",
                ),
            )
        )


def _emit_late_frame(on_event, marker: str) -> None:
    assert on_event is not None
    on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=f"late-{marker}"))
    on_event(
        ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(
                id="provider-reused-tool-id",
                title=f"late-tool-{marker}",
                kind="search",
                status="in_progress",
                content=f"late-input-{marker}",
            ),
        )
    )


def _assert_two_isolated_turns(
    callbacks: list[object],
    captured: list[CardEvent],
    activities: list[tuple[str, str]],
) -> None:
    assert len(callbacks) == 2
    assert callbacks[0] is not callbacks[1]

    starts = [
        event
        for event in captured
        if event.type
        in {
            CardEventType.TEXT_STARTED,
            CardEventType.REASONING_STARTED,
            CardEventType.TOOL_STARTED,
        }
    ]
    assert [event.type for event in starts] == [
        CardEventType.TEXT_STARTED,
        CardEventType.REASONING_STARTED,
        CardEventType.TOOL_STARTED,
        CardEventType.TOOL_STARTED,
        CardEventType.TEXT_STARTED,
        CardEventType.REASONING_STARTED,
        CardEventType.TOOL_STARTED,
        CardEventType.TOOL_STARTED,
    ]
    block_ids = [str(event.payload["block_id"]) for event in starts]
    assert len(set(block_ids)) == len(block_ids)

    state = CardState()
    for event in captured:
        state = reduce_card_state(state, event)
    assert [block.kind for block in state.blocks] == [
        "text",
        "reasoning",
        "tool_call",
        "tool_call",
        "text",
        "reasoning",
        "tool_call",
        "tool_call",
    ]
    assert all(block.status == "completed" for block in state.blocks)
    rendered = "\n".join(
        str(value)
        for block in state.blocks
        for value in vars(block).values()
        if value
    )
    assert "late-" not in rendered
    for turn in (1, 2):
        for invocation in ("a", "b"):
            for prefix in ("input", "update", "output"):
                assert f"{prefix}-turn-{turn}-{invocation}" in rendered
    assert activities == [
        ("stream-call", "tool-turn-1-a"),
        ("stream-call", "tool-turn-1-a (completed)"),
        ("stream-call", "tool-turn-1-b"),
        ("stream-call", "tool-turn-1-b (completed)"),
        ("stream-call", "tool-turn-2-a"),
        ("stream-call", "tool-turn-2-a (completed)"),
        ("stream-call", "tool-turn-2-b"),
        ("stream-call", "tool-turn-2-b (completed)"),
    ]


class _SchemaRepairSession:
    def __init__(self) -> None:
        self.callbacks: list[object] = []

    def send_prompt(self, _prompt: str, *, on_event=None, **_kwargs):
        self.callbacks.append(on_event)
        turn = len(self.callbacks)
        if turn == 2:
            _emit_late_frame(self.callbacks[0], "schema-retired")
        _emit_turn(on_event, f"turn-{turn}")
        text = '{"wrong": true}' if turn == 1 else '{"ok": true}'
        return SimpleNamespace(text=text, output_tokens=0, stop_reason="end_turn")

    def cancel(self) -> None:
        return None

    def close(self) -> None:
        return None


def test_schema_repair_uses_fresh_retirable_namespaced_stream_turns(tmp_path) -> None:
    session = _SchemaRepairSession()
    captured: list[CardEvent] = []
    activities: list[tuple[str, str]] = []
    executor = AgentExecutor(
        cwd=str(tmp_path),
        cancel_event=threading.Event(),
        on_card_event=lambda _label, event: captured.append(event),
        on_activity=lambda label, activity: activities.append((label, activity)),
    )

    try:
        with patch(
            "src.agent_session.factory.create_engine_session",
            return_value=session,
        ):
            result = executor.execute(
                AgentCallParams(
                    prompt="return json",
                    tool="codex",
                    label="stream-call",
                    schema={"ok": "boolean"},
                )
            )
    finally:
        executor.shutdown(wait=True)

    assert result.error is None
    _assert_two_isolated_turns(session.callbacks, captured, activities)


class _RetrySession:
    def __init__(self, callbacks: list[object], turn: int) -> None:
        self._callbacks = callbacks
        self._turn = turn

    def send_prompt(self, _prompt: str, *, on_event=None, **_kwargs):
        self._callbacks.append(on_event)
        if self._turn == 2:
            _emit_late_frame(self._callbacks[0], "retry-retired")
        _emit_turn(on_event, f"turn-{self._turn}")
        if self._turn == 1:
            raise RuntimeError("503 Service Unavailable")
        return SimpleNamespace(text="done", output_tokens=0, stop_reason="end_turn")

    def cancel(self) -> None:
        return None

    def close(self) -> None:
        return None


def test_outer_retry_uses_fresh_retirable_namespaced_stream_turns(tmp_path) -> None:
    callbacks: list[object] = []
    sessions = [_RetrySession(callbacks, 1), _RetrySession(callbacks, 2)]
    captured: list[CardEvent] = []
    activities: list[tuple[str, str]] = []
    executor = AgentExecutor(
        cwd=str(tmp_path),
        cancel_event=threading.Event(),
        on_card_event=lambda _label, event: captured.append(event),
        on_activity=lambda label, activity: activities.append((label, activity)),
    )

    try:
        with (
            patch(
                "src.agent_session.factory.create_engine_session",
                side_effect=sessions,
            ),
            patch.object(executor, "_sleep_with_backoff"),
        ):
            result = executor.execute(
                AgentCallParams(
                    prompt="work",
                    tool="codex",
                    label="stream-call",
                )
            )
    finally:
        executor.shutdown(wait=True)

    assert result.error is None
    _assert_two_isolated_turns(callbacks, captured, activities)


class _RacingRetirementSession:
    def __init__(self, entered: threading.Event, release: threading.Event) -> None:
        self._entered = entered
        self._release = release
        self.returned = threading.Event()
        self.callback = None
        self.callback_thread: threading.Thread | None = None

    def send_prompt(self, _prompt: str, *, on_event=None, **_kwargs):
        assert on_event is not None
        self.callback = on_event
        self.callback_thread = threading.Thread(
            target=lambda: on_event(
                ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="in-flight")
            )
        )
        self.callback_thread.start()
        assert self._entered.wait(timeout=2)
        self.returned.set()
        return SimpleNamespace(text="done", output_tokens=0, stop_reason="end_turn")

    def cancel(self) -> None:
        return None

    def close(self) -> None:
        if self.callback_thread is not None:
            self.callback_thread.join(timeout=2)


def test_retirement_waits_for_inflight_callback_then_rejects_late_frames(
    tmp_path,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    session = _RacingRetirementSession(entered, release)
    captured: list[CardEvent] = []

    def capture(_label: str, event: CardEvent) -> None:
        captured.append(event)
        if event.type == CardEventType.TEXT_STARTED:
            entered.set()
            assert release.wait(timeout=2)

    executor = AgentExecutor(
        cwd=str(tmp_path),
        cancel_event=threading.Event(),
        on_card_event=capture,
    )

    try:
        with (
            patch(
                "src.agent_session.factory.create_engine_session",
                return_value=session,
            ),
            concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool,
        ):
            future = pool.submit(
                executor.execute,
                AgentCallParams(
                    prompt="work",
                    tool="codex",
                    label="stream-call",
                ),
            )
            assert entered.wait(timeout=2)
            assert session.returned.wait(timeout=2)
            assert not future.done()
            release.set()
            result = future.result(timeout=2)
    finally:
        release.set()
        executor.shutdown(wait=True)

    assert result.error is None
    assert session.callback is not None
    session.callback(
        ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="late-after-retire")
    )
    assert [event.type for event in captured] == [
        CardEventType.TEXT_STARTED,
        CardEventType.TEXT_DELTA,
        CardEventType.TEXT_DONE,
    ]
    assert all("late-after-retire" not in str(event.payload) for event in captured)


class _ConcurrentSession:
    def __init__(self, barrier: threading.Barrier) -> None:
        self._barrier = barrier

    def send_prompt(self, _prompt: str, *, on_event=None, **_kwargs):
        self._barrier.wait(timeout=2)
        _emit_turn(on_event, "concurrent")
        return SimpleNamespace(text="done", output_tokens=0, stop_reason="end_turn")

    def cancel(self) -> None:
        return None

    def close(self) -> None:
        return None


def test_concurrent_calls_with_identical_labels_receive_unique_turn_namespaces(
    tmp_path,
) -> None:
    barrier = threading.Barrier(2)
    captured: list[CardEvent] = []
    captured_lock = threading.Lock()

    def capture(_label: str, event: CardEvent) -> None:
        with captured_lock:
            captured.append(event)

    executor = AgentExecutor(
        cwd=str(tmp_path),
        cancel_event=threading.Event(),
        on_card_event=capture,
    )
    params = AgentCallParams(prompt="work", tool="codex", label="same-label")

    try:
        with (
            patch(
                "src.agent_session.factory.create_engine_session",
                side_effect=lambda **_kwargs: _ConcurrentSession(barrier),
            ),
            concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool,
        ):
            results = list(pool.map(executor.execute, (params, params)))
    finally:
        executor.shutdown(wait=True)

    assert all(result.error is None for result in results)
    starts = [
        event
        for event in captured
        if event.type
        in {
            CardEventType.TEXT_STARTED,
            CardEventType.REASONING_STARTED,
            CardEventType.TOOL_STARTED,
        }
    ]
    assert len(starts) == 8
    block_ids = [str(event.payload["block_id"]) for event in starts]
    assert len(set(block_ids)) == len(block_ids)
    turn_ids = {
        match.group(1)
        for block_id in block_ids
        if (match := re.match(r"_wf_turn_(\d+)_", block_id))
    }
    assert len(turn_ids) == 2
    for turn_id in turn_ids:
        turn_starts = [
            event.type
            for event in starts
            if str(event.payload["block_id"]).startswith(f"_wf_turn_{turn_id}_")
        ]
        assert turn_starts == [
            CardEventType.TEXT_STARTED,
            CardEventType.REASONING_STARTED,
            CardEventType.TOOL_STARTED,
            CardEventType.TOOL_STARTED,
        ]
