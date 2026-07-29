"""Regression coverage for deadline-aware programming prompt finalization."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import patch

import pytest

import src.acp as acp
from src.acp.models import PromptResult


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
    assert "最终答复" in finalization_prompt


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


def test_retirement_failure_still_poison_marks_finalization_session() -> None:
    session = _TimeoutThenCompleteSession(return_timeout_result=True)

    def fail_retirement(_session: object, _budget: float) -> None:
        raise RuntimeError("manager lock unavailable")

    try:
        _runner()(
            session,
            "original task",
            timeout_s=90,
            finalization_reserve_s=30,
            retire_finalization_session=fail_retirement,
        )
    except RuntimeError as exc:
        assert "manager lock unavailable" in str(exc)
    else:
        raise AssertionError("retirement failure must not be swallowed")
    assert session._force_dead is True


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


def test_retirement_receives_only_remaining_total_budget() -> None:
    session = _TimeoutThenCompleteSession()
    retirement_budgets: list[float] = []
    clock = iter((100.0, 155.0, 188.0))

    with patch(
        "src.acp.finalization._monotonic",
        side_effect=lambda: next(clock),
    ):
        _runner()(
            session,
            "original task",
            timeout_s=90,
            finalization_reserve_s=30,
            retire_finalization_session=lambda _active, budget: (
                retirement_budgets.append(budget)
            ),
        )

    assert retirement_budgets
    assert retirement_budgets[0] == pytest.approx(2.0)
