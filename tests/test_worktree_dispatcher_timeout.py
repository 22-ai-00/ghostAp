"""
Worktree dispatcher timeout 测试
覆盖 5 个关键场景：pool timeout / inner timeout / callback / fast path
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

from src.agent_session.claude_cli import SyncClaudeCLISession
from src.worktree_engine.dispatcher import WorktreeDispatcher
from src.worktree_engine.models import (
    WorktreeSelectionItem,
    WorktreeUnit,
    WorktreeUnitStatus,
)


def test_pool_timeout_returns_within_bound_and_cancels_session():
    """Pool timeout must not wait for a worker that ignores cancellation."""
    prompt_started = threading.Event()
    release_prompt = threading.Event()
    cancel_called = threading.Event()
    returned = threading.Event()
    captured: dict[str, object] = {}

    class BlockingSession:
        def start(self):
            return None

        def send_prompt(self, *_args, **_kwargs):
            prompt_started.set()
            release_prompt.wait(timeout=5)
            return MagicMock(text="late result", stop_reason="end_turn")

        def cancel(self, *, wait=False, timeout=0):
            del wait, timeout
            cancel_called.set()
            return True

        def close(self):
            return None

    dispatcher = WorktreeDispatcher(session_factory=lambda **_kwargs: BlockingSession())
    unit = WorktreeUnit(
        unit_id="blocked",
        provider="acp",
        tool_name="coco",
        worktree_path="/tmp/blocked-wt",
        task_prompt="run forever",
        status=WorktreeUnitStatus.PLANNED,
    )

    def _execute() -> None:
        started_at = time.monotonic()
        captured["units"] = dispatcher.execute_units(
            [unit],
            pool_timeout=0.05,
            cancel_grace=0.05,
        )
        captured["elapsed"] = time.monotonic() - started_at
        returned.set()

    caller = threading.Thread(target=_execute, daemon=True)
    caller.start()
    assert prompt_started.wait(timeout=0.5)
    try:
        assert returned.wait(timeout=0.3), "execute_units waited for the blocked worker"
        assert cancel_called.is_set()
        assert float(captured["elapsed"]) <= 0.2
        assert unit.status is WorktreeUnitStatus.CANCELLED
        assert unit.cancellation_acknowledged is True
    finally:
        release_prompt.set()
        caller.join(timeout=1)


def test_cancel_ack_wait_is_bounded_when_session_cancel_hangs():
    prompt_started = threading.Event()
    release_prompt = threading.Event()
    release_cancel = threading.Event()
    returned = threading.Event()
    captured: dict[str, object] = {}

    class UnresponsiveCancelSession:
        def start(self):
            return None

        def send_prompt(self, *_args, **_kwargs):
            prompt_started.set()
            release_prompt.wait(timeout=5)
            return MagicMock(text="late result", stop_reason="end_turn")

        def cancel(self, *, wait=False, timeout=0):
            del wait, timeout
            release_cancel.wait(timeout=5)

        def close(self):
            return None

    dispatcher = WorktreeDispatcher(
        session_factory=lambda **_kwargs: UnresponsiveCancelSession()
    )
    unit = WorktreeUnit(
        unit_id="hung-cancel",
        provider="acp",
        tool_name="coco",
        worktree_path="/tmp/hung-cancel-wt",
        task_prompt="run forever",
        status=WorktreeUnitStatus.PLANNED,
    )

    def _execute() -> None:
        started_at = time.monotonic()
        dispatcher.execute_units(
            [unit],
            pool_timeout=0.05,
            cancel_grace=0.05,
        )
        captured["elapsed"] = time.monotonic() - started_at
        returned.set()

    caller = threading.Thread(target=_execute, daemon=True)
    caller.start()
    assert prompt_started.wait(timeout=0.5)
    try:
        assert returned.wait(timeout=0.3), "cancel acknowledgment exceeded grace"
        assert float(captured["elapsed"]) <= 0.2
        assert unit.status is WorktreeUnitStatus.CANCELLED
        assert unit.cancellation_acknowledged is False
        assert unit.metadata["cancellation_error"] == "cancel_ack_timeout"
    finally:
        release_cancel.set()
        release_prompt.set()
        caller.join(timeout=1)


def test_pool_timeout_cancels_every_bound_session():
    release_prompt = threading.Event()
    both_started = threading.Barrier(3)
    cancel_calls: list[str] = []
    returned = threading.Event()

    class BoundSession:
        def __init__(self, name: str):
            self.name = name

        def start(self):
            return None

        def send_prompt(self, *_args, **_kwargs):
            both_started.wait(timeout=1)
            release_prompt.wait(timeout=5)
            return MagicMock(text="late result", stop_reason="end_turn")

        def cancel(self, *, wait=False, timeout=0):
            del wait, timeout
            cancel_calls.append(self.name)
            return True

        def close(self):
            return None

    created = iter([BoundSession("first"), BoundSession("second")])
    dispatcher = WorktreeDispatcher(
        session_factory=lambda **_kwargs: next(created)
    )
    units = [
        WorktreeUnit(
            unit_id=f"bound-{index}",
            provider="acp",
            tool_name="coco",
            worktree_path=f"/tmp/bound-{index}",
            task_prompt="run forever",
            status=WorktreeUnitStatus.PLANNED,
        )
        for index in range(2)
    ]

    caller = threading.Thread(
        target=lambda: (
            dispatcher.execute_units(
                units,
                pool_timeout=0.05,
                cancel_grace=0.05,
            ),
            returned.set(),
        ),
        daemon=True,
    )
    caller.start()
    both_started.wait(timeout=1)
    try:
        assert returned.wait(timeout=0.3)
        assert sorted(cancel_calls) == ["first", "second"]
        assert all(unit.status is WorktreeUnitStatus.CANCELLED for unit in units)
    finally:
        release_prompt.set()
        caller.join(timeout=1)


def test_legacy_cancel_none_is_requested_but_not_acknowledged():
    prompt_started = threading.Event()
    release_prompt = threading.Event()
    returned = threading.Event()

    class LegacyCancelSession(SyncClaudeCLISession):
        def __init__(self):
            super().__init__(cwd="/tmp/legacy-cancel-wt")

        def start(self):
            return None

        def send_prompt(self, *_args, **_kwargs):
            prompt_started.set()
            release_prompt.wait(timeout=5)
            return MagicMock(text="late result", stop_reason="end_turn")

        def close(self):
            return None

    unit = WorktreeUnit(
        unit_id="legacy-cancel",
        provider="cli",
        tool_name="claude",
        worktree_path="/tmp/legacy-cancel-wt",
        task_prompt="run forever",
        status=WorktreeUnitStatus.PLANNED,
    )
    dispatcher = WorktreeDispatcher(
        session_factory=lambda **_kwargs: LegacyCancelSession()
    )
    caller = threading.Thread(
        target=lambda: (
            dispatcher.execute_units(
                [unit],
                pool_timeout=0.05,
                cancel_grace=0.05,
            ),
            returned.set(),
        ),
        daemon=True,
    )
    caller.start()
    assert prompt_started.wait(timeout=0.5)
    try:
        assert returned.wait(timeout=0.3)
        assert unit.cancellation_requested is True
        assert unit.cancellation_acknowledged is False
        assert unit.metadata["cancellation_error"] == "cancel_ack_unknown"
    finally:
        release_prompt.set()
        caller.join(timeout=1)


def test_worktree_dispatcher_pool_timeout():
    """测试 pool-level timeout 场景"""
    # 创建一些会永远运行的单元
    unit1 = WorktreeUnit(unit_id="u1", worktree_path="/tmp/wt1")
    unit2 = WorktreeUnit(unit_id="u2", worktree_path="/tmp/wt2")
    unit3 = WorktreeUnit(unit_id="u3", worktree_path="/tmp/wt3")

    units = [unit1, unit2, unit3]
    tools = [
        WorktreeSelectionItem(provider="acp", tool_name="claude", display_name="Claude"),
        WorktreeSelectionItem(provider="acp", tool_name="gemini", display_name="Gemini"),
        WorktreeSelectionItem(provider="acp", tool_name="coco", display_name="Coco"),
    ]

    # 创建会阻塞但可以被捕获的 session factory
    # 使用异常来模拟超时，而不是真正 sleep 10s
    def slow_session_factory(**kwargs):
        session = MagicMock()
        session.start = MagicMock()

        # 使用可以快速失败的模拟
        def send_prompt_side_effect(*args, **kwargs):
            time.sleep(0.1)  # 稍微延迟但不会太久
            raise TimeoutError("")

        session.send_prompt = MagicMock(side_effect=send_prompt_side_effect)
        session.close = MagicMock()
        return session

    dispatcher = WorktreeDispatcher(session_factory=slow_session_factory)
    planned = dispatcher.plan_user_goal("test goal", units, tools)

    # 用 monkeypatch 来测试 pool timeout 逻辑路径
    # 我们直接测试异常处理逻辑，而不是依赖真实的时间

    # 验证所有单元最终都是 failed 状态（不管是 inner 还是 pool timeout）
    executed = dispatcher.execute_units(planned, timeout=0.1)
    for unit in executed:
        assert unit.status == "failed"
        assert "超时" in unit.error or "繁忙" in unit.error


def test_worktree_dispatcher_inner_timeout():
    """测试 inner timeout（单个单元超时）场景"""
    # 创建一个快速单元和一个慢速单元
    fast_unit = WorktreeUnit(unit_id="fast", worktree_path="/tmp/fast")
    slow_unit = WorktreeUnit(unit_id="slow", worktree_path="/tmp/slow")

    units = [fast_unit, slow_unit]
    tools = [
        WorktreeSelectionItem(provider="acp", tool_name="coco", display_name="Coco"),
        WorktreeSelectionItem(provider="acp", tool_name="claude", display_name="Claude"),
    ]

    call_count = 0

    def session_factory(**kwargs):
        nonlocal call_count
        session = MagicMock()
        session.start = MagicMock()

        if call_count == 0:
            # 快速单元：立即返回
            def fast_send(*args, **kwargs):
                result = MagicMock()
                result.text = "fast done"
                result.stop_reason = "end_turn"
                return result
            session.send_prompt = fast_send
        else:
            # 慢速单元：超时
            def slow_send(*args, **kwargs):
                time.sleep(1)
                raise TimeoutError("")
            session.send_prompt = slow_send

        session.close = MagicMock()
        call_count += 1
        return session

    dispatcher = WorktreeDispatcher(session_factory=session_factory)
    planned = dispatcher.plan_user_goal("test goal", units, tools)

    executed = dispatcher.execute_units(planned, timeout=0.5)

    # 验证 fast_unit 完成，slow_unit 失败
    executed_dict = {unit.unit_id: unit for unit in executed}
    assert executed_dict["fast"].status == "completed"
    assert executed_dict["slow"].status == "failed"
    assert "超时" in executed_dict["slow"].error


def test_worktree_dispatcher_callback_called():
    """测试 on_unit_update callback 正确调用"""
    unit1 = WorktreeUnit(unit_id="u1", worktree_path="/tmp/wt1")
    unit2 = WorktreeUnit(unit_id="u2", worktree_path="/tmp/wt2")

    units = [unit1, unit2]
    tools = [
        WorktreeSelectionItem(provider="acp", tool_name="coco", display_name="Coco"),
        WorktreeSelectionItem(provider="acp", tool_name="claude", display_name="Claude"),
    ]

    callback_calls = []

    def on_unit_update(unit):
        callback_calls.append((unit.unit_id, unit.status))

    def session_factory(**kwargs):
        session = MagicMock()
        session.start = MagicMock()
        result = MagicMock()
        result.text = "done"
        result.stop_reason = "end_turn"
        session.send_prompt = MagicMock(return_value=result)
        session.close = MagicMock()
        return session

    dispatcher = WorktreeDispatcher(session_factory=session_factory)
    planned = dispatcher.plan_user_goal("test goal", units, tools)

    dispatcher.execute_units(planned, on_unit_update=on_unit_update)

    # 验证 callback 至少被调用过（每个单元 running 和完成状态）
    assert len(callback_calls) >= 2
    unit_ids_in_calls = {call[0] for call in callback_calls}
    assert unit_ids_in_calls == {"u1", "u2"}


def test_worktree_dispatcher_fast_path():
    """测试快速路径（无超时，所有单元正常完成）"""
    unit1 = WorktreeUnit(unit_id="u1", worktree_path="/tmp/wt1")
    unit2 = WorktreeUnit(unit_id="u2", worktree_path="/tmp/wt2")
    unit3 = WorktreeUnit(unit_id="u3", worktree_path="/tmp/wt3")

    units = [unit1, unit2, unit3]
    tools = [
        WorktreeSelectionItem(provider="acp", tool_name="claude", display_name="Claude"),
        WorktreeSelectionItem(provider="acp", tool_name="gemini", display_name="Gemini"),
        WorktreeSelectionItem(provider="acp", tool_name="coco", display_name="Coco"),
    ]

    def session_factory(**kwargs):
        session = MagicMock()
        session.start = MagicMock()
        result = MagicMock()
        result.text = f"done with {kwargs.get('tool_name', 'unknown')}"
        result.stop_reason = "end_turn"
        session.send_prompt = MagicMock(return_value=result)
        session.close = MagicMock()
        return session

    dispatcher = WorktreeDispatcher(session_factory=session_factory)
    planned = dispatcher.plan_user_goal("test goal", units, tools)

    # 设置较长的 pool timeout，确保不会触发
    executed = dispatcher.execute_units(planned, pool_timeout=60)

    # 验证所有单元都完成
    for unit in executed:
        assert unit.status == "completed"
        assert unit.error == ""
        assert "done with" in unit.summary


def test_worktree_dispatcher_pool_timeout_status_not_overwritten():
    """测试 pool timeout 后迟到 worker 不会覆盖 cancelled 状态。"""
    unit = WorktreeUnit(unit_id="test", worktree_path="/tmp/test")
    tools = [
        WorktreeSelectionItem(provider="acp", tool_name="coco", display_name="Coco"),
    ]

    slow_calls = 0

    def slow_session_factory(**kwargs):
        nonlocal slow_calls
        slow_calls += 1
        session = MagicMock()
        session.start = MagicMock()

        def slow_send(*args, **kwargs):
            time.sleep(0.5)  # 比 pool timeout 长很多
            # 如果状态已经是 failed，这里不应该执行到
            raise RuntimeError("Should not be called after pool timeout!")

        session.send_prompt = slow_send
        session.close = MagicMock()
        return session

    dispatcher = WorktreeDispatcher(session_factory=slow_session_factory)
    planned = dispatcher.plan_user_goal("test goal", [unit], tools)

    # 设置非常短的 pool timeout
    executed = dispatcher.execute_units(planned, pool_timeout=0.2)

    assert executed[0].status == "cancelled"
    # 即使 _run_single_unit 迟到抛错，generation fence 也会保留取消终态。
