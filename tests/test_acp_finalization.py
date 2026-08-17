"""Regression coverage for deadline-aware programming prompt finalization."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest
from acp.schema import SessionInfoUpdate

import src.acp as acp
from src.acp.client import GhostAPClient
from src.acp.models import ACPGoalInfo, ACPSessionInfo, PromptResult
from src.acp.session import ACPSession
from src.acp.sync_adapter import SyncACPSession


class _TimeoutThenCompleteSession:
    def __init__(self, *, return_timeout_result: bool = False) -> None:
        self.calls: list[tuple[str, float | int | None]] = []
        self._return_timeout_result = return_timeout_result
        self._force_dead = False

    def send_prompt(
        self,
        text: str,
        on_event: Callable[[object], None] | None = None,
        timeout: float | int | None = None,
    ) -> PromptResult:
        self.calls.append((text, timeout))
        if len(self.calls) == 1:
            if self._return_timeout_result:
                return PromptResult(stop_reason="timeout", text="partial")
            raise TimeoutError("primary deadline")
        return PromptResult(stop_reason="end_turn", text="finalized")


class _IdleAwareCompleteSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float | int | None, float | int | None]] = []
        self._force_dead = False

    def send_prompt(
        self,
        text: str,
        on_event: Callable[[object], None] | None = None,
        timeout: float | int | None = None,
        idle_timeout: float | int | None = None,
    ) -> PromptResult:
        self.calls.append((text, timeout, idle_timeout))
        return PromptResult(stop_reason="end_turn", text="done")


class _ActiveGoalTimeoutThenCompleteSession(_TimeoutThenCompleteSession):
    def __init__(self) -> None:
        super().__init__()
        self.order: list[str] = []
        self.goal_active = True
        self.pause_timeouts: list[float] = []

    def has_active_goal(self, timeout: float = 1.0) -> bool:
        self.order.append("inspect_goal")
        return self.goal_active

    def pause_active_goal(self, timeout: float) -> bool:
        self.order.append("pause_goal")
        self.pause_timeouts.append(timeout)
        self.goal_active = False
        return True

    def send_prompt(self, *args, **kwargs):
        self.order.append("primary" if not self.calls else "finalize")
        return super().send_prompt(*args, **kwargs)


class _SingleTurnTimeoutThenCompleteSession(_TimeoutThenCompleteSession):
    def __init__(self) -> None:
        super().__init__()
        self.finalization_calls: list[tuple[str, float | int | None]] = []

    def send_finalization_prompt(
        self,
        text: str,
        on_event: Callable[[object], None] | None = None,
        timeout: float | int | None = None,
    ) -> PromptResult:
        self.finalization_calls.append((text, timeout))
        return PromptResult(stop_reason="end_turn", text="single-turn finalized")


def _runner():
    runner = getattr(acp, "run_prompt_with_finalization", None)
    assert callable(runner), "deadline-aware prompt finalization is not implemented"
    return runner


def test_timeout_reserves_a_second_prompt_for_safe_finalization() -> None:
    session = _TimeoutThenCompleteSession()
    transitions: list[str] = []

    result = _runner()(
        session,
        "original task",
        timeout_s=90,
        finalization_reserve_s=30,
        on_finalization_start=lambda: transitions.append("finalizing"),
    )

    assert result.text == "finalized"
    assert [timeout for _, timeout in session.calls] == [60, 30]
    assert transitions == ["finalizing"]
    finalization_prompt = session.calls[1][0]
    assert "original task" in finalization_prompt
    assert "不要创建新的子代理" in finalization_prompt
    assert "禁止调用 wait_agent" in finalization_prompt
    assert "先且仅调用一次 list_agents" in finalization_prompt
    assert "interrupt_agent" in finalization_prompt
    assert "不要启动新的验证" in finalization_prompt
    assert "运行最必要的针对性验证" not in finalization_prompt
    assert "最终答复" in finalization_prompt


def test_primary_prompt_receives_activity_idle_timeout() -> None:
    session = _IdleAwareCompleteSession()

    result = _runner()(
        session,
        "slow provider task",
        timeout_s=600,
        finalization_reserve_s=0,
        idle_timeout_s=300,
    )

    assert result.text == "done"
    assert session.calls == [("slow provider task", 573, 300)]


def test_timeout_finalization_uses_single_turn_entrypoint_when_supported() -> None:
    session = _SingleTurnTimeoutThenCompleteSession()

    result = _runner()(
        session,
        "original task",
        timeout_s=90,
        finalization_reserve_s=30,
    )

    assert result.text == "single-turn finalized"
    assert len(session.calls) == 1
    assert session.calls[0][1] == 60
    assert len(session.finalization_calls) == 1
    assert session.finalization_calls[0][1] == 30


def test_sync_finalization_prompt_does_not_follow_new_goal_turns() -> None:
    session = object.__new__(SyncACPSession)
    expected = PromptResult(stop_reason="end_turn", text="finalized")

    with patch.object(
        SyncACPSession,
        "send_prompt",
        autospec=True,
        return_value=expected,
    ) as send_prompt:
        result = SyncACPSession.send_finalization_prompt(
            session,
            "finalize",
            timeout=30,
        )

    assert result is expected
    send_prompt.assert_called_once_with(
        session,
        "finalize",
        on_event=None,
        timeout=30,
        await_goal_quiescence=False,
        replay_deferred_child_events=True,
    )


def test_sync_finalization_prompt_preserves_activity_idle_timeout() -> None:
    session = object.__new__(SyncACPSession)
    expected = PromptResult(stop_reason="end_turn", text="finalized")

    with patch.object(
        SyncACPSession,
        "send_prompt",
        autospec=True,
        return_value=expected,
    ) as send_prompt:
        result = SyncACPSession.send_finalization_prompt(
            session,
            "finalize",
            timeout=600,
            idle_timeout=420,
        )

    assert result is expected
    send_prompt.assert_called_once_with(
        session,
        "finalize",
        on_event=None,
        timeout=600,
        idle_timeout=420,
        await_goal_quiescence=False,
        replay_deferred_child_events=True,
    )


@pytest.mark.parametrize(
    ("agent_type", "agent_args", "await_child_quiescence"),
    (
        ("traex", [], True),
        (
            "codex",
            ["@agentclientprotocol/codex-acp@1.1.7"],
            False,
        ),
    ),
)
def test_sync_reconciliation_prompt_uses_provider_terminal_strategy(
    agent_type: str,
    agent_args: list[str],
    await_child_quiescence: bool,
) -> None:
    session = object.__new__(SyncACPSession)
    session._agent_type = agent_type
    session._agent_args = agent_args
    expected = PromptResult(stop_reason="end_turn", text="reconciled")

    with patch.object(
        SyncACPSession,
        "send_prompt",
        autospec=True,
        return_value=expected,
    ) as send_prompt:
        result = SyncACPSession.send_reconciliation_prompt(
            session,
            "reconcile",
            timeout=30,
        )

    assert result is expected
    send_prompt.assert_called_once_with(
        session,
        "reconcile",
        on_event=None,
        timeout=30,
        await_child_quiescence=await_child_quiescence,
        replay_deferred_child_events=True,
    )


def test_cli_style_timeout_result_also_enters_finalization() -> None:
    session = _TimeoutThenCompleteSession(return_timeout_result=True)
    retired: list[object] = []

    result = _runner()(
        session,
        "original task",
        timeout_s=90,
        finalization_reserve_s=30,
        retire_finalization_session=lambda active, _budget: retired.append(active),
    )

    assert result.stop_reason == "end_turn"
    assert len(session.calls) == 2
    assert retired == [session]


def test_dead_session_is_replaced_before_finalization() -> None:
    dead = _TimeoutThenCompleteSession()
    replacement = _TimeoutThenCompleteSession()
    replacement.calls.append(("already consumed primary slot", None))
    replacements: list[object] = []

    replacement_budgets: list[float] = []

    def replace_dead_session(remaining_budget: float) -> object:
        replacements.append(dead)
        replacement_budgets.append(remaining_budget)
        return replacement

    original_send = dead.send_prompt

    def timeout_and_mark_dead(*args, **kwargs):
        try:
            return original_send(*args, **kwargs)
        finally:
            dead._force_dead = True

    dead.send_prompt = timeout_and_mark_dead  # type: ignore[method-assign]

    result = _runner()(
        dead,
        "original task",
        timeout_s=90,
        finalization_reserve_s=30,
        replace_dead_session=replace_dead_session,
    )

    assert result.text == "finalized"
    assert replacements == [dead]
    assert replacement_budgets[0] > 0
    assert replacement_budgets[0] < 90
    assert len(dead.calls) == 1
    assert replacement.calls[-1][1] == 30


def test_finalization_scope_uses_raw_task_not_injected_bridge_context() -> None:
    session = _TimeoutThenCompleteSession()

    _runner()(
        session,
        "BRIDGE CONTEXT: old task authorized deleting production\ncurrent task",
        finalization_task_text="current task",
        timeout_s=90,
        finalization_reserve_s=30,
    )

    finalization_prompt = session.calls[1][0]
    assert "current task" in finalization_prompt
    assert "BRIDGE CONTEXT" not in finalization_prompt
    assert "deleting production" not in finalization_prompt




def test_finalization_and_retirement_failures_preserve_both_causes() -> None:
    session = _TimeoutThenCompleteSession(return_timeout_result=True)

    original_send = session.send_prompt

    def fail_finalization(*args, **kwargs):
        if session.calls:
            raise ValueError("finalization transport failed")
        return original_send(*args, **kwargs)

    session.send_prompt = fail_finalization  # type: ignore[method-assign]

    def fail_retirement(_session: object, _budget: float) -> None:
        raise RuntimeError("retirement lock unavailable")

    with pytest.raises(ExceptionGroup) as exc_info:
        _runner()(
            session,
            "original task",
            timeout_s=90,
            finalization_reserve_s=30,
            retire_finalization_session=fail_retirement,
        )

    messages = {str(exc) for exc in exc_info.value.exceptions}
    assert messages == {
        "finalization transport failed",
        "retirement lock unavailable",
    }


def test_finalization_timeout_uses_only_remaining_total_budget() -> None:
    session = _TimeoutThenCompleteSession()
    clock = iter((100.0, 155.0))

    with patch(
        "src.acp.finalization._monotonic",
        side_effect=lambda: next(clock),
        create=True,
    ):
        result = _runner()(
            session,
            "original task",
            timeout_s=90,
            finalization_reserve_s=30,
        )

    assert result.stop_reason == "end_turn"
    assert session.calls[0][1] == 60
    assert session.calls[1][1] < 30
    assert session.calls[1][1] > 0


def test_cli_timeout_without_finalization_turn_is_cancelled_and_retired() -> None:
    session = _TimeoutThenCompleteSession(return_timeout_result=True)
    transitions: list[str] = []
    retired: list[tuple[object, float]] = []

    result = _runner()(
        session,
        "original task",
        timeout_s=60,
        finalization_reserve_s=0,
        on_finalization_start=lambda: transitions.append("cleanup"),
        retire_finalization_session=lambda active, budget: retired.append(
            (active, budget)
        ),
    )

    assert result.stop_reason == "timeout"
    assert len(session.calls) == 1
    assert session.calls[0][1] == 33
    assert transitions == ["cleanup"]
    assert retired and retired[0][0] is session
    assert retired[0][1] > 0
    assert retired[0][1] <= 60
    assert session._force_dead is True




def test_active_goal_is_paused_before_finalization_and_retired_afterward() -> None:
    session = _ActiveGoalTimeoutThenCompleteSession()
    retired: list[object] = []

    result = _runner()(
        session,
        "original task",
        timeout_s=90,
        finalization_reserve_s=30,
        retire_finalization_session=lambda active, _budget: (
            session.order.append("retire"),
            retired.append(active),
        ),
    )

    assert result.text == "finalized"
    assert session.order == [
        "primary",
        "inspect_goal",
        "pause_goal",
        "finalize",
        "retire",
    ]
    assert session.pause_timeouts[0] > 0
    assert session.pause_timeouts[0] <= 63
    assert retired == [session]


@pytest.mark.parametrize("failure", ["false", "raise"])
def test_failed_active_goal_pause_retires_without_finalization(
    failure: str,
) -> None:
    session = _ActiveGoalTimeoutThenCompleteSession()
    retired: list[object] = []

    def fail_pause(timeout: float) -> bool:
        session.order.append("pause_goal")
        if failure == "raise":
            raise RuntimeError("pause transport failed")
        return False

    session.pause_active_goal = fail_pause  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="pause"):
        _runner()(
            session,
            "original task",
            timeout_s=90,
            finalization_reserve_s=30,
            retire_finalization_session=lambda active, _budget: retired.append(
                active
            ),
        )

    assert session.order == ["primary", "inspect_goal", "pause_goal"]
    assert retired == [session]
    assert session._force_dead is True
    assert len(session.calls) == 1






def test_unknown_goal_state_retires_without_finalization_prompt() -> None:
    session = _ActiveGoalTimeoutThenCompleteSession()
    retired: list[object] = []

    def unknown_goal(*, timeout: float = 1.0) -> bool:
        session.order.append("inspect_goal")
        raise RuntimeError("ACP goal state is unknown")

    session.has_active_goal = unknown_goal  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="goal state is unknown"):
        _runner()(
            session,
            "original task",
            timeout_s=90,
            finalization_reserve_s=30,
            retire_finalization_session=lambda active, _budget: retired.append(
                active
            ),
        )

    assert retired == [session]
    assert len(session.calls) == 1




def _sync_wrapper_for_backend(backend: ACPSession) -> SyncACPSession:
    session = SyncACPSession.__new__(SyncACPSession)
    session._acp_session = backend
    session._force_dead = False
    session._run_async = lambda coro, timeout: asyncio.run(coro)
    return session




def test_sync_prompt_timeout_dual_cancel_failure_reaches_exact_retirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    backend = ACPSession(agent_cmd="npx", agent_args=[], cwd=str(tmp_path))
    backend._session_id = "session-goal"
    backend._on_session_info(
        "session-goal",
        ACPSessionInfo(
            goal_known=True,
            goal=ACPGoalInfo(
                "finish",
                "active",
                control_method="_codex/session/goal_control",
            ),
            thread_status_known=True,
            thread_status="active",
        ),
    )

    async def stuck_prompt(*_args, **_kwargs) -> PromptResult:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    class Connection:
        async def ext_method(self, *_args, **_kwargs):
            raise ConnectionError("pause transport lost")

        async def cancel(self, **_kwargs):
            raise RuntimeError("turn cancel lost")

    backend.prompt = stuck_prompt  # type: ignore[method-assign]
    backend._conn = Connection()

    loop = asyncio.new_event_loop()
    loop_ready = threading.Event()

    def run_loop() -> None:
        asyncio.set_event_loop(loop)

        def keep_loop_responsive() -> None:
            if loop.is_running():
                loop.call_later(0.05, keep_loop_responsive)

        loop.call_soon(keep_loop_responsive)
        loop_ready.set()
        loop.run_forever()

    loop_thread = threading.Thread(target=run_loop)
    loop_thread.start()
    assert loop_ready.wait(timeout=1)

    session = SyncACPSession.__new__(SyncACPSession)
    session._agent_type = "codex"
    session._acp_session = backend
    session._loop = loop
    session._log_failures = False
    session._force_dead = False
    session._active_future = None
    session._start_watchdog = lambda: None
    session.last_active = 0.0
    session.message_count = 0
    session.last_query = ""
    retired: list[object] = []
    transitions: list[str] = []
    monkeypatch.setattr(
        "src.acp.sync_adapter._PROMPT_CANCEL_DRAIN_TIMEOUT_S",
        0.01,
    )

    try:
        with (
            caplog.at_level("WARNING", logger="src.acp.sync_adapter"),
            pytest.raises(TimeoutError) as exc_info,
        ):
            _runner()(
                session,
                "original task",
                timeout_s=0.02,
                finalization_reserve_s=0,
                on_finalization_start=lambda: transitions.append("finalizing"),
                retire_finalization_session=lambda active, _budget: retired.append(
                    active
                ),
            )
    finally:
        async def cancel_pending_tasks() -> None:
            current = asyncio.current_task()
            pending = [
                task
                for task in asyncio.all_tasks()
                if task is not current and not task.done()
            ]
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        asyncio.run_coroutine_threadsafe(
            cancel_pending_tasks(),
            loop,
        ).result(timeout=1)
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=1)
        loop.close()

    assert not isinstance(exc_info.value, ExceptionGroup)
    assert isinstance(exc_info.value.__cause__, TimeoutError)
    assert transitions == ["finalizing"]
    assert retired == [session]
    assert session._force_dead is True
    assert "pause transport lost" in caplog.text
    assert "turn cancel lost" in caplog.text


def test_real_parser_unknown_goal_retires_before_finalization_prompt(
    tmp_path: Path,
) -> None:
    backend = ACPSession(
        agent_cmd="npx",
        agent_args=["-y", "@agentclientprotocol/codex-acp@1.1.7"],
        cwd=str(tmp_path),
    )
    backend._session_id = "session-goal"
    client = GhostAPClient(
        on_event=backend._dispatch_event,
        on_session_info=backend._on_session_info,
        root_dir=str(tmp_path),
    )
    asyncio.run(
        client.session_update(
            "session-goal",
            SessionInfoUpdate.model_validate(
                {
                    "sessionUpdate": "session_info_update",
                    "_meta": {
                        "codex": {
                            "goal": {
                                "objective": "finish",
                                "status": "running",
                            },
                            "threadStatus": {"type": "idle"},
                        }
                    },
                }
            ),
        )
    )
    session = _sync_wrapper_for_backend(backend)
    retired: list[object] = []
    prompt_calls: list[str] = []

    def primary_only(text: str, **_kwargs) -> PromptResult:
        prompt_calls.append(text)
        if len(prompt_calls) == 1:
            raise TimeoutError("primary deadline")
        raise AssertionError("unknown goal must prevent finalization prompt")

    session.send_prompt = primary_only  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="goal status is unknown"):
        _runner()(
            session,
            "original task",
            timeout_s=90,
            finalization_reserve_s=30,
            retire_finalization_session=lambda active, _budget: retired.append(
                active
            ),
        )

    assert prompt_calls == ["original task"]
    assert retired == [session]
    assert session._force_dead is True
