"""SyncACPSession close-path cleanup regressions."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from types import SimpleNamespace

from src.acp.models import ACPGoalInfo, ACPSessionInfo
from src.acp.session import ACPSession
from src.acp.sync_adapter import SyncACPSession


def test_close_drains_pending_loop_callbacks_before_loop_close():
    marker: list[str] = []
    loop = asyncio.new_event_loop()
    started = threading.Event()

    def run_loop() -> None:
        asyncio.set_event_loop(loop)

        def keep_loop_responsive() -> None:
            if loop.is_running():
                loop.call_later(0.05, keep_loop_responsive)

        loop.call_soon(keep_loop_responsive)
        started.set()
        loop.run_forever()

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    assert started.wait(timeout=2)

    class FakeACPSession:
        async def close(self) -> None:
            asyncio.get_running_loop().call_soon(marker.append, "pipe-close-callback")

    session = SyncACPSession.__new__(SyncACPSession)
    session._agent_type = "test"
    session._loop = loop
    session._loop_thread = thread
    session._acp_session = FakeACPSession()
    session._watchdog_stop = threading.Event()
    session._watchdog_thread = None

    session.close()

    assert marker == ["pipe-close-callback"]
    assert session._loop is None
    assert not thread.is_alive()


def test_close_cancels_and_forgets_active_prompt_future():
    future: concurrent.futures.Future[None] = concurrent.futures.Future()
    session = SyncACPSession.__new__(SyncACPSession)
    session._active_future = future
    session._acp_session = None
    session._loop = None
    session._watchdog_stop = threading.Event()
    session._watchdog_thread = None

    session.close()

    assert future.cancelled()
    assert session._active_future is None


def test_sync_close_cancels_active_goal_waiter_and_clears_collector(tmp_path):
    loop = asyncio.new_event_loop()
    started = threading.Event()
    prompt_returned = threading.Event()

    def run_loop() -> None:
        asyncio.set_event_loop(loop)

        def keep_loop_responsive() -> None:
            if loop.is_running():
                loop.call_later(0.05, keep_loop_responsive)

        loop.call_soon(keep_loop_responsive)
        started.set()
        loop.run_forever()

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    assert started.wait(timeout=2)

    backend = ACPSession(agent_cmd="npx", agent_args=[], cwd=str(tmp_path))

    class Connection:
        async def prompt(self, **_kwargs):
            backend._on_session_info(
                "session-goal",
                ACPSessionInfo(
                    goal_known=True,
                    goal=ACPGoalInfo("finish", "active"),
                    thread_status_known=True,
                    thread_status="active",
                ),
            )
            prompt_returned.set()
            return SimpleNamespace(stop_reason="end_turn")

    class Context:
        async def __aexit__(self, *_args):
            return None

    backend._conn = Connection()
    backend._ctx_manager = Context()
    backend._session_id = "session-goal"

    session = SyncACPSession.__new__(SyncACPSession)
    session._agent_type = "test"
    session._loop = loop
    session._loop_thread = thread
    session._acp_session = backend
    session._watchdog_stop = threading.Event()
    session._watchdog_thread = None
    session._active_future = asyncio.run_coroutine_threadsafe(
        backend.prompt("work"),
        loop,
    )

    assert prompt_returned.wait(timeout=2)
    session.close()

    assert backend._event_handler is None
    assert session._active_future is None
    assert session._loop is None
    assert not thread.is_alive()
