"""Regression tests for per-prompt ACP event handler ownership."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from acp.schema import SessionInfoUpdate

import src.acp.session as session_mod
from src.acp.client import GhostAPClient
from src.acp.models import (
    ACPEvent,
    ACPEventType,
    ACPGoalInfo,
    ACPSessionInfo,
)
from src.acp.session import ACPSession


def _text_event(text: str) -> ACPEvent:
    return ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=text)


def _active_goal_info(
    *, control_method: str = "_codex/session/goal_control"
) -> ACPSessionInfo:
    return ACPSessionInfo(
        goal_known=True,
        goal=ACPGoalInfo(
            objective="finish the repository task",
            status="active",
            control_method=control_method,
        ),
        thread_status_known=True,
        thread_status="active",
    )


def _completed_idle_goal_info() -> ACPSessionInfo:
    return ACPSessionInfo(
        goal_known=True,
        goal=ACPGoalInfo(
            objective="finish the repository task",
            status="completed",
            control_method="_codex/session/goal_control",
        ),
        thread_status_known=True,
        thread_status="idle",
    )


def test_prompt_keeps_collector_until_active_goal_is_quiescent(
    tmp_path: Path,
) -> None:
    session = ACPSession(
        agent_cmd="npx",
        agent_args=["-y", "@agentclientprotocol/codex-acp@1.1.7"],
        cwd=str(tmp_path),
    )

    async def exercise() -> None:
        returned = asyncio.Event()

        class Connection:
            async def prompt(self, **_kwargs):
                session._on_session_info("session-goal", _active_goal_info())
                asyncio.get_running_loop().call_later(
                    0.01,
                    session._dispatch_event,
                    _text_event("provider continuation"),
                )
                asyncio.get_running_loop().call_later(
                    0.02,
                    session._on_session_info,
                    "session-goal",
                    _completed_idle_goal_info(),
                )
                return SimpleNamespace(stop_reason="end_turn")

        session._conn = Connection()
        session._session_id = "session-goal"
        task = asyncio.create_task(session.prompt("work"))
        task.add_done_callback(lambda _task: returned.set())
        await asyncio.sleep(0.005)
        assert returned.is_set() is False
        result = await asyncio.wait_for(task, timeout=1)
        assert result.text == "provider continuation"
        assert result.goal is not None
        assert result.goal.status == "completed"
        assert session._event_handler is None

    asyncio.run(exercise())


def test_terminal_goal_waits_for_idle_thread_status(tmp_path: Path) -> None:
    session = ACPSession(agent_cmd="npx", agent_args=[], cwd=str(tmp_path))

    async def exercise() -> None:
        class Connection:
            async def prompt(self, **_kwargs):
                session._on_session_info(
                    "session-goal",
                    ACPSessionInfo(
                        goal_known=True,
                        goal=ACPGoalInfo("finish", "completed"),
                        thread_status_known=True,
                        thread_status="active",
                    ),
                )
                return SimpleNamespace(stop_reason="end_turn")

        session._conn = Connection()
        session._session_id = "session-goal"
        task = asyncio.create_task(session.prompt("work"))
        await asyncio.sleep(0.06)
        assert task.done() is False
        session._on_session_info(
            "session-goal",
            ACPSessionInfo(thread_status_known=True, thread_status="idle"),
        )
        result = await asyncio.wait_for(task, timeout=1)
        assert result.goal is not None
        assert result.goal.status == "completed"

    asyncio.run(exercise())


def test_unknown_goal_status_keeps_prompt_attached_until_trusted_update(
    tmp_path: Path,
) -> None:
    session = ACPSession(
        agent_cmd="npx",
        agent_args=["-y", "@agentclientprotocol/codex-acp@1.1.7"],
        cwd=str(tmp_path),
    )

    async def exercise() -> None:
        class Connection:
            async def prompt(self, **_kwargs):
                session._on_session_info(
                    "session-goal",
                    ACPSessionInfo(
                        goal_known=True,
                        goal=ACPGoalInfo("finish", "running"),
                        thread_status_known=True,
                        thread_status="idle",
                    ),
                )
                return SimpleNamespace(stop_reason="end_turn")

        session._conn = Connection()
        session._session_id = "session-goal"
        prompt = asyncio.create_task(session.prompt("work"))
        await asyncio.sleep(0.15)
        assert prompt.done() is False
        assert session._event_handler is not None
        with pytest.raises(RuntimeError, match="goal status is unknown"):
            await session.has_active_goal()
        session._on_session_info("session-goal", _completed_idle_goal_info())
        result = await asyncio.wait_for(prompt, timeout=1)
        assert result.goal is not None
        assert result.goal.status == "completed"
        assert session._event_handler is None

    asyncio.run(exercise())


def test_terminal_goal_without_known_thread_status_is_quiescent(
    tmp_path: Path,
) -> None:
    session = ACPSession(
        agent_cmd="npx",
        agent_args=["-y", "@agentclientprotocol/codex-acp@1.1.7"],
        cwd=str(tmp_path),
    )
    client = GhostAPClient(
        on_event=session._dispatch_event,
        on_session_info=session._on_session_info,
        root_dir=str(tmp_path),
    )
    session._client = client

    async def exercise() -> None:
        class Connection:
            async def prompt(self, **_kwargs):
                await client.session_update(
                    "session-goal",
                    SessionInfoUpdate.model_validate(
                        {
                            "sessionUpdate": "session_info_update",
                            "_meta": {
                                "codex": {
                                    "goal": {
                                        "objective": "finish",
                                        "status": "completed",
                                    },
                                    "threadStatus": {"type": "busy"},
                                }
                            },
                        }
                    ),
                )
                return SimpleNamespace(stop_reason="end_turn")

        session._conn = Connection()
        session._session_id = "session-goal"
        result = await asyncio.wait_for(session.prompt("work"), timeout=0.2)
        assert result.goal is not None
        assert result.goal.status == "completed"
        assert session._event_handler is None

    asyncio.run(exercise())


def test_malformed_thread_status_cannot_clear_known_active_state(
    tmp_path: Path,
) -> None:
    session = ACPSession(agent_cmd="npx", agent_args=[], cwd=str(tmp_path))
    session._session_id = "session-goal"
    session._on_session_info(
        "session-goal",
        ACPSessionInfo(thread_status_known=True, thread_status="active"),
    )
    session._on_session_info(
        "session-goal",
        ACPSessionInfo(thread_status_observed=True),
    )

    assert session._thread_status_known is True
    assert session._thread_status == "active"

    async def exercise() -> None:
        class Connection:
            async def prompt(self, **_kwargs):
                session._on_session_info(
                    "session-goal",
                    ACPSessionInfo(
                        goal_known=True,
                        goal=ACPGoalInfo("finish", "completed"),
                        thread_status_known=True,
                        thread_status="active",
                    ),
                )
                session._on_session_info(
                    "session-goal",
                    ACPSessionInfo(thread_status_observed=True),
                )
                return SimpleNamespace(stop_reason="end_turn")

        session._conn = Connection()
        session._session_id = "session-goal"
        prompt = asyncio.create_task(session.prompt("work"))
        await asyncio.sleep(0.06)
        assert prompt.done() is False
        session._on_session_info(
            "session-goal",
            ACPSessionInfo(thread_status_known=True, thread_status="idle"),
        )
        await asyncio.wait_for(prompt, timeout=1)

    asyncio.run(exercise())


@pytest.mark.parametrize("thread_status", ["active", "busy"])
def test_known_null_goal_does_not_infer_goal_from_thread_status(
    tmp_path: Path,
    thread_status: str,
) -> None:
    session = ACPSession(
        agent_cmd="npx",
        agent_args=["-y", "@agentclientprotocol/codex-acp@1.1.7"],
        cwd=str(tmp_path),
    )
    client = GhostAPClient(
        on_event=session._dispatch_event,
        on_session_info=session._on_session_info,
        root_dir=str(tmp_path),
    )
    session._client = client

    async def exercise() -> None:
        class Connection:
            async def prompt(self, **_kwargs):
                await client.session_update(
                    "session-goal",
                    SessionInfoUpdate.model_validate(
                        {
                            "sessionUpdate": "session_info_update",
                            "_meta": {
                                "codex": {
                                    "goal": None,
                                    "threadStatus": {"type": thread_status},
                                }
                            },
                        }
                    ),
                )
                return SimpleNamespace(stop_reason="end_turn")

        session._conn = Connection()
        session._session_id = "session-goal"
        result = await asyncio.wait_for(session.prompt("work"), timeout=1)
        assert result.goal is None
        assert session._event_handler is None

    asyncio.run(exercise())


def test_prompt_rechecks_quiescence_before_atomically_detaching_collector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A terminal-to-active transition in the old image window must be observed."""
    session = ACPSession(
        agent_cmd="npx",
        agent_args=["-y", "@agentclientprotocol/codex-acp@1.1.7"],
        cwd=str(tmp_path),
    )
    client = GhostAPClient(
        on_event=session._dispatch_event,
        on_session_info=session._on_session_info,
        root_dir=str(tmp_path),
    )
    session._client = client
    continuation_injected = False

    def inject_continuation(*_args, **_kwargs) -> None:
        nonlocal continuation_injected
        if continuation_injected:
            return
        continuation_injected = True
        session._on_session_info("session-goal", _active_goal_info())
        loop = asyncio.get_running_loop()
        loop.call_soon(
            session._dispatch_event,
            _text_event("continuation from image window"),
        )
        loop.call_soon(
            session._on_session_info,
            "session-goal",
            _completed_idle_goal_info(),
        )

    monkeypatch.setattr(
        session_mod,
        "emit_referenced_changed_local_image_events",
        inject_continuation,
    )

    async def exercise() -> None:
        class Connection:
            async def prompt(self, **_kwargs):
                session._on_session_info(
                    "session-goal",
                    _completed_idle_goal_info(),
                )
                return SimpleNamespace(stop_reason="end_turn")

        session._conn = Connection()
        session._session_id = "session-goal"
        result = await asyncio.wait_for(session.prompt("work"), timeout=1)
        assert result.text == "continuation from image window"
        assert result.goal is not None
        assert result.goal.status == "completed"
        assert session._event_handler is None

    asyncio.run(exercise())


def test_unrelated_session_info_cannot_release_active_goal_waiter(
    tmp_path: Path,
) -> None:
    session = ACPSession(agent_cmd="npx", agent_args=[], cwd=str(tmp_path))
    client = GhostAPClient(
        on_event=session._dispatch_event,
        on_session_info=session._on_session_info,
        root_dir=str(tmp_path),
    )

    async def exercise() -> None:
        class Connection:
            async def prompt(self, **_kwargs):
                session._on_session_info("session-goal", _active_goal_info())
                return SimpleNamespace(stop_reason="end_turn")

        session._client = client
        session._conn = Connection()
        session._session_id = "session-goal"
        task = asyncio.create_task(session.prompt("work"))
        await asyncio.sleep(0.06)
        await client.session_update(
            "session-goal",
            SessionInfoUpdate.model_validate(
                {
                    "sessionUpdate": "session_info_update",
                    "title": "renamed",
                }
            ),
        )
        session._on_session_info("other-session", _completed_idle_goal_info())
        await asyncio.sleep(0)
        assert task.done() is False
        assert await session.has_active_goal() is True
        session._on_session_info("session-goal", _completed_idle_goal_info())
        await asyncio.wait_for(task, timeout=1)

    asyncio.run(exercise())


def test_close_wakes_active_goal_waiter_and_clears_handler(tmp_path: Path) -> None:
    session = ACPSession(agent_cmd="npx", agent_args=[], cwd=str(tmp_path))

    async def exercise() -> None:
        class Connection:
            async def prompt(self, **_kwargs):
                session._on_session_info("session-goal", _active_goal_info())
                return SimpleNamespace(stop_reason="end_turn")

        session._conn = Connection()
        session._session_id = "session-goal"
        task = asyncio.create_task(session.prompt("work"))
        await asyncio.sleep(0.06)
        session._conn = None
        await session.close()
        with pytest.raises(RuntimeError, match="closing"):
            await asyncio.wait_for(task, timeout=1)
        assert session._event_handler is None

    asyncio.run(exercise())


def test_non_codex_prompt_retains_short_tail_drain(tmp_path: Path) -> None:
    session = ACPSession(agent_cmd="test", agent_args=[], cwd=str(tmp_path))

    async def exercise() -> None:
        class Connection:
            async def prompt(self, **_kwargs):
                asyncio.get_running_loop().call_later(
                    0.01,
                    session._dispatch_event,
                    _text_event("late tail"),
                )
                return SimpleNamespace(stop_reason="end_turn")

        session._conn = Connection()
        session._session_id = "session-normal"
        result = await session.prompt("work")
        assert result.text == "late tail"

    asyncio.run(exercise())


def test_cancel_pauses_active_goal_before_cancelling_turn(tmp_path: Path) -> None:
    session = ACPSession(agent_cmd="npx", agent_args=[], cwd=str(tmp_path))
    order: list[str] = []

    async def exercise() -> None:
        class Connection:
            async def ext_method(self, method: str, params: dict):
                order.append(method)
                assert params == {"sessionId": "session-goal", "action": "pause"}
                session._on_session_info(
                    "session-goal",
                    ACPSessionInfo(
                        goal_known=True,
                        goal=ACPGoalInfo("finish", "paused"),
                    ),
                )
                return {}

            async def cancel(self, **_kwargs):
                order.append("cancel")

        session._conn = Connection()
        session._session_id = "session-goal"
        session._on_session_info("session-goal", _active_goal_info())
        await session.cancel()

    asyncio.run(exercise())
    assert order == ["codex/session/goal_control", "cancel"]


def test_cancel_still_cancels_turn_when_goal_pause_fails(tmp_path: Path) -> None:
    session = ACPSession(agent_cmd="npx", agent_args=[], cwd=str(tmp_path))
    order: list[str] = []

    async def exercise() -> None:
        class Connection:
            async def ext_method(self, method: str, _params: dict):
                order.append(method)
                raise RuntimeError("pause failed")

            async def cancel(self, **_kwargs):
                order.append("cancel")

        session._conn = Connection()
        session._session_id = "session-goal"
        session._on_session_info("session-goal", _active_goal_info())
        with pytest.raises(RuntimeError, match="pause failed"):
            await session.cancel()

    asyncio.run(exercise())
    assert order == ["codex/session/goal_control", "cancel"]


def test_untrusted_goal_control_method_is_never_called(tmp_path: Path) -> None:
    session = ACPSession(agent_cmd="npx", agent_args=[], cwd=str(tmp_path))

    async def exercise() -> None:
        class Connection:
            async def ext_method(self, *_args, **_kwargs):
                raise AssertionError("untrusted extension method was called")

        session._conn = Connection()
        session._session_id = "session-goal"
        session._on_session_info(
            "session-goal",
            _active_goal_info(control_method="_evil/delete_everything"),
        )
        assert await session.pause_active_goal() is False

    asyncio.run(exercise())


def test_public_goal_pause_propagates_transport_failure(tmp_path: Path) -> None:
    session = ACPSession(agent_cmd="npx", agent_args=[], cwd=str(tmp_path))

    async def exercise() -> None:
        class Connection:
            async def ext_method(self, *_args, **_kwargs):
                raise ConnectionError("goal control transport lost")

        session._conn = Connection()
        session._session_id = "session-goal"
        session._on_session_info("session-goal", _active_goal_info())

        with pytest.raises(ConnectionError, match="transport lost"):
            await session.pause_active_goal()

    asyncio.run(exercise())


def test_public_goal_pause_rejects_unknown_official_goal_state(
    tmp_path: Path,
) -> None:
    session = ACPSession(
        agent_cmd="npx",
        agent_args=["-y", "@agentclientprotocol/codex-acp@1.1.7"],
        cwd=str(tmp_path),
    )
    session._session_id = "session-goal"
    session._conn = object()

    with pytest.raises(RuntimeError, match="goal state is unknown"):
        asyncio.run(session.pause_active_goal())


def test_public_goal_pause_rejects_state_reset_to_unknown_during_rpc(
    tmp_path: Path,
) -> None:
    session = ACPSession(
        agent_cmd="npx",
        agent_args=["-y", "@agentclientprotocol/codex-acp@1.1.7"],
        cwd=str(tmp_path),
    )

    async def exercise() -> None:
        class Connection:
            async def ext_method(self, *_args, **_kwargs):
                session._reset_lifecycle(goal_known=False)
                return {}

        session._conn = Connection()
        session._session_id = "session-goal"
        session._on_session_info("session-goal", _active_goal_info())
        with pytest.raises(RuntimeError, match="goal state is unknown"):
            await session.pause_active_goal()

    asyncio.run(exercise())


def test_active_goal_cancel_releases_prompt_waiter(tmp_path: Path) -> None:
    session = ACPSession(agent_cmd="npx", agent_args=[], cwd=str(tmp_path))

    async def exercise() -> None:
        class Connection:
            async def prompt(self, **_kwargs):
                session._on_session_info("session-goal", _active_goal_info())
                return SimpleNamespace(stop_reason="end_turn")

            async def ext_method(self, _method: str, _params: dict):
                session._on_session_info(
                    "session-goal",
                    ACPSessionInfo(
                        goal_known=True,
                        goal=ACPGoalInfo("finish", "paused"),
                    ),
                )
                return {}

            async def cancel(self, **_kwargs):
                session._on_session_info(
                    "session-goal",
                    ACPSessionInfo(
                        thread_status_known=True,
                        thread_status="idle",
                    ),
                )

        session._conn = Connection()
        session._session_id = "session-goal"
        task = asyncio.create_task(session.prompt("work"))
        await asyncio.sleep(0.06)
        assert task.done() is False
        await session.cancel(timeout=0.5)
        result = await asyncio.wait_for(task, timeout=1)
        assert result.goal is not None
        assert result.goal.status == "paused"
        assert session._event_handler is None

    asyncio.run(exercise())


def test_official_codex_unknown_goal_state_fails_closed(tmp_path: Path) -> None:
    session = ACPSession(
        agent_cmd="npx",
        agent_args=["-y", "@agentclientprotocol/codex-acp@1.1.7"],
        cwd=str(tmp_path),
    )

    async def exercise() -> None:
        with pytest.raises(RuntimeError, match="goal state is unknown"):
            await session.has_active_goal()

    asyncio.run(exercise())


def test_stale_transport_epoch_cannot_mutate_goal_tracker(tmp_path: Path) -> None:
    session = ACPSession(agent_cmd="npx", agent_args=[], cwd=str(tmp_path))
    session._session_id = "session-goal"
    session._transport_epoch = 2

    session._on_session_info(
        "session-goal",
        _active_goal_info(),
        transport_epoch=1,
    )

    assert asyncio.run(session.has_active_goal()) is False


def test_prompt_exception_clears_handler_and_drops_late_events(
    tmp_path: Path,
) -> None:
    class FailingConnection:
        async def prompt(self, **_kwargs):
            raise RuntimeError("prompt failed")

    received: list[str] = []
    session = ACPSession(agent_cmd="test", agent_args=[], cwd=str(tmp_path))
    session._conn = FailingConnection()
    session._session_id = "session-failure"

    async def exercise() -> None:
        with pytest.raises(RuntimeError, match="prompt failed"):
            await session.prompt(
                "fail",
                on_event=lambda event: received.append(event.text or ""),
            )

        assert session._event_handler is None
        session._dispatch_event(_text_event("late-old-event"))

    asyncio.run(exercise())

    assert received == []


def test_overlapping_prompt_is_rejected_without_rebinding_real_dispatch(
    tmp_path: Path,
) -> None:
    session = ACPSession(agent_cmd="test", agent_args=[], cwd=str(tmp_path))

    class OverlappingConnection:
        def __init__(self) -> None:
            self.calls = 0
            self.first_started = asyncio.Event()
            self.release_first = asyncio.Event()

        async def prompt(self, **_kwargs):
            self.calls += 1
            self.first_started.set()
            await self.release_first.wait()
            return SimpleNamespace(stop_reason="end_turn")

    first_received: list[str] = []
    second_received: list[str] = []

    async def exercise():
        connection = OverlappingConnection()
        session._conn = connection
        session._session_id = "session-overlap"

        first = asyncio.create_task(
            session.prompt(
                "first",
                on_event=lambda event: first_received.append(event.text or ""),
            )
        )
        await connection.first_started.wait()

        with pytest.raises(RuntimeError, match="already running"):
            await session.prompt(
                "second",
                on_event=lambda event: second_received.append(event.text or ""),
            )

        # Exercise the real callback entry point. The rejected prompt must not
        # replace the first prompt's handler and receive this late update.
        session._dispatch_event(_text_event("late-first"))

        connection.release_first.set()
        return await first, connection.calls

    result, calls = asyncio.run(exercise())

    assert first_received == ["late-first"]
    assert second_received == []
    assert result.text == "late-first"
    assert result.stop_reason == "end_turn"
    assert calls == 1
    assert session._event_handler is None


def test_close_transport_failure_keeps_termination_handles(
    tmp_path: Path,
) -> None:
    session = ACPSession(agent_cmd="test", agent_args=[], cwd=str(tmp_path))

    class FailingContext:
        async def __aexit__(self, *_args):
            raise RuntimeError("transport process still alive")

    context = FailingContext()
    connection = object()
    process = object()
    session._ctx_manager = context
    session._conn = connection
    session._proc = process

    with pytest.raises(RuntimeError, match="transport process still alive"):
        asyncio.run(session.close())

    assert session._ctx_manager is context
    assert session._conn is connection
    assert session._proc is process


def test_prompt_emits_changed_referenced_image_before_snapshot_release(
    tmp_path: Path,
) -> None:
    """A final Markdown path must become an image event while attribution is live."""

    image_path = tmp_path / "final-evidence.png"
    session = ACPSession(agent_cmd="test", agent_args=[], cwd=str(tmp_path))
    client = GhostAPClient(
        on_event=session._dispatch_event,
        root_dir=str(tmp_path),
    )
    session._client = client

    class ImageWritingConnection:
        async def prompt(self, **_kwargs):
            image_path.write_bytes(b"\x89PNG\r\n\x1a\nchanged-image")
            session._dispatch_event(
                _text_event(f"完成，证据见 ![result]({image_path})")
            )
            return SimpleNamespace(stop_reason="end_turn")

    session._conn = ImageWritingConnection()
    session._session_id = "session-image"
    observed: list[tuple[ACPEventType, bool]] = []

    def receive(event: ACPEvent) -> None:
        snapshot = client._current_image_snapshot()
        observed.append(
            (
                event.event_type,
                bool(snapshot is not None and snapshot.active),
            )
        )

    result = asyncio.run(session.prompt("create evidence", on_event=receive))

    image_events = [
        snapshot_active
        for event_type, snapshot_active in observed
        if event_type is ACPEventType.IMAGE_CHUNK
    ]
    assert result.stop_reason == "end_turn"
    assert image_events == [True]
    assert client._current_image_snapshot() is None
