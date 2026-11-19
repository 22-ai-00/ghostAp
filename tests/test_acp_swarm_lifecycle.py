"""Unit tests for AGY native Swarm protocol parsing and lifecycle tracking."""

from __future__ import annotations

import pytest

from src.acp.models import ACPEvent, ACPEventType, PromptResult, ToolCallInfo
from src.acp.swarm_lifecycle import (
    SwarmLifecycleTracker,
    SwarmToolSignal,
    is_auxiliary_swarm_member,
    is_swarm_working_event,
    merge_swarm_tool_signal,
    parse_swarm_tool_signal,
)


def _tool_event(
    event_type: ACPEventType,
    *,
    call_id: str,
    title: str,
    status: str,
    swarm_signal: SwarmToolSignal | None = None,
    content: str = "",
) -> ACPEvent:
    return ACPEvent(
        event_type=event_type,
        tool_call=ToolCallInfo(
            id=call_id,
            title=title,
            kind="other",
            status=status,
            content=content,
            swarm_signal=swarm_signal,
        ),
    )


def _text_event(text: str) -> ACPEvent:
    return ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=text)


def test_parse_and_merge_swarm_signals_across_agy_wire_formats() -> None:
    add_start = parse_swarm_tool_signal(
        "Run swarm",
        {"action": "add", "name": "worker", "tasks": ["part1", "part2"]},
        None,
    )
    add_progress = parse_swarm_tool_signal(
        "",
        None,
        'Created At: 2026-11-18T00:00:00Z\nCompleted At: 2026-11-18T00:00:01Z\n{"status":"added","name":"worker"}',
    )
    merged_add = merge_swarm_tool_signal(add_start, add_progress)
    assert merged_add is not None
    assert merged_add.tool == "swarm"
    assert merged_add.action == "add"
    assert merged_add.added_members == ("worker-1", "worker-2", "worker")

    group_n_signal = parse_swarm_tool_signal(
        "contrib:swarm",
        {"action": "add", "name": "shard", "n": 3},
        None,
    )
    assert group_n_signal is not None
    assert group_n_signal.added_members == ("shard-1", "shard-2", "shard-3")

    status_signal = parse_swarm_tool_signal(
        "swarm",
        {"action": "status"},
        {
            "members": [
                {"name": "monitor", "state": "waiting"},
                {"name": "narrator-2", "state": "working"},
                {"name": "worker-1", "state": "working"},
                {"name": "worker-2", "state": "waiting"},
            ]
        },
    )
    assert status_signal is not None
    assert status_signal.action == "status"
    assert status_signal.roster_members == (
        ("monitor", "waiting"),
        ("narrator-2", "working"),
        ("worker-1", "working"),
        ("worker-2", "waiting"),
    )

    done_post = parse_swarm_tool_signal(
        "post_message",
        {"text": "[done] All tests green"},
        None,
    )
    assert done_post is not None
    assert done_post.is_done_post is True
    assert done_post.mentions_worker is False

    wake_post = parse_swarm_tool_signal(
        "post_message",
        {"text": "@worker-1 please check #4"},
        None,
    )
    assert wake_post is not None
    assert wake_post.is_done_post is False
    assert wake_post.mentions_worker is True

    feedback_post = parse_swarm_tool_signal(
        "post_message",
        {"text": "@narrator user feedback: keep concise"},
        None,
    )
    assert feedback_post is not None
    assert feedback_post.is_done_post is False
    assert feedback_post.mentions_worker is False

    board_read = parse_swarm_tool_signal(
        "read_board",
        {"since": 0},
        "#4 worker-1: [done] finished cleanly",
    )
    assert board_read is not None
    assert board_read.board_has_done is True

    assert is_auxiliary_swarm_member("monitor") is True
    assert is_auxiliary_swarm_member("narrator-2") is True
    assert is_auxiliary_swarm_member("coordinator") is True
    assert is_auxiliary_swarm_member("all") is True
    assert is_auxiliary_swarm_member("worker-1") is False


def test_is_swarm_working_event_ignores_auxiliary_and_failed_calls() -> None:
    add_monitor_done = _tool_event(
        ACPEventType.TOOL_CALL_DONE,
        call_id="c_mon",
        title="swarm",
        status="completed",
        swarm_signal=SwarmToolSignal(tool="swarm", action="add", added_members=("monitor",)),
    )
    assert is_swarm_working_event(add_monitor_done) is False

    add_worker_failed = _tool_event(
        ACPEventType.TOOL_CALL_DONE,
        call_id="c_fail",
        title="swarm",
        status="failed",
        swarm_signal=SwarmToolSignal(tool="swarm", action="add", added_members=("worker-1",)),
    )
    assert is_swarm_working_event(add_worker_failed) is False

    add_worker_done = _tool_event(
        ACPEventType.TOOL_CALL_DONE,
        call_id="c_ok",
        title="swarm",
        status="completed",
        swarm_signal=SwarmToolSignal(tool="swarm", action="add", added_members=("worker-1",)),
    )
    assert is_swarm_working_event(add_worker_done) is True

    wait_start = _tool_event(
        ACPEventType.TOOL_CALL_START,
        call_id="c_wait",
        title="swarm_wait",
        status="in_progress",
    )
    assert is_swarm_working_event(wait_start) is True

    status_working = _tool_event(
        ACPEventType.TOOL_CALL_DONE,
        call_id="c_status",
        title="swarm",
        status="completed",
        swarm_signal=SwarmToolSignal(
            tool="swarm",
            action="status",
            roster_members=(("monitor", "working"), ("worker-1", "working")),
        ),
    )
    assert is_swarm_working_event(status_working) is True

    # Hand-crafted DONE with only raw JSON content fallback
    raw_content_done = _tool_event(
        ACPEventType.TOOL_CALL_DONE,
        call_id="c_raw",
        title="swarm",
        status="completed",
        content='{"status":"added","name":"impl"}',
    )
    assert is_swarm_working_event(raw_content_done) is True


def test_swarm_tracker_ordinary_prompt_and_failed_add_do_not_wait() -> None:
    tracker = SwarmLifecycleTracker()
    assert tracker.observe_event(
        _tool_event(
            ACPEventType.TOOL_CALL_DONE,
            call_id="c1",
            title="read_file",
            status="completed",
        ),
        10.0,
    ) is False
    assert tracker.observe_event(_text_event("普通回复"), 10.1) is False
    tracker.on_prompt_response_returned(10.2)
    assert tracker.evaluate_wait(10.2) == (False, None)

    failed_tracker = SwarmLifecycleTracker()
    failed_tracker.observe_event(
        _tool_event(
            ACPEventType.TOOL_CALL_START,
            call_id="c_add",
            title="swarm",
            status="in_progress",
            swarm_signal=SwarmToolSignal(tool="swarm", action="add", added_members=("worker-1",)),
        ),
        20.0,
    )
    failed_tracker.observe_event(
        _tool_event(
            ACPEventType.TOOL_CALL_DONE,
            call_id="c_add",
            title="swarm",
            status="failed",
        ),
        20.1,
    )
    failed_tracker.on_prompt_response_returned(20.2)
    assert failed_tracker.active is False
    assert failed_tracker.evaluate_wait(20.2) == (False, None)


def test_swarm_tracker_multi_turn_done_lifecycle_and_plan_handoff() -> None:
    tracker = SwarmLifecycleTracker(
        done_text_settle_s=0.1,
        done_quiet_settle_s=0.2,
        turn_gap_reset_s=0.15,
    )
    # Turn 1: add monitor + worker + interim text -> PromptResponse returns
    tracker.observe_event(
        _tool_event(
            ACPEventType.TOOL_CALL_DONE,
            call_id="c_mon",
            title="swarm",
            status="completed",
            swarm_signal=SwarmToolSignal(tool="swarm", action="add", added_members=("monitor",)),
        ),
        1.0,
    )
    assert tracker.active is False
    tracker.observe_event(
        _tool_event(
            ACPEventType.TOOL_CALL_DONE,
            call_id="c_w1",
            title="swarm",
            status="completed",
            swarm_signal=SwarmToolSignal(tool="swarm", action="add", added_members=("worker-1",)),
        ),
        1.1,
    )
    tracker.observe_event(_text_event("已组建 swarm，等待 worker-1"), 1.2)
    tracker.on_prompt_response_returned(1.2)
    assert tracker.evaluate_wait(1.3) == (True, None)

    # Turn 2: reactive wakeup posts follow-up to worker, writes interim update, parks with swarm_wait
    tracker.observe_event(
        _tool_event(
            ACPEventType.TOOL_CALL_DONE,
            call_id="c_post",
            title="post_message",
            status="completed",
            swarm_signal=SwarmToolSignal(tool="post_message", mentions_worker=True),
        ),
        5.0,
    )
    tracker.observe_event(_text_event("阶段进展..."), 5.1)
    assert tracker.evaluate_wait(5.2) == (True, None)
    tracker.observe_event(
        _tool_event(
            ACPEventType.TOOL_CALL_DONE,
            call_id="c_wait",
            title="swarm_wait",
            status="completed",
        ),
        5.2,
    )
    assert tracker.evaluate_wait(6.0) == (True, None)

    # Turn 3: mid-swarm exit_plan_mode handoff holds cleanly across turn boundary
    tracker.observe_event(
        _tool_event(
            ACPEventType.TOOL_CALL_DONE,
            call_id="c_exit",
            title="exit_plan_mode",
            status="completed",
        ),
        8.0,
    )
    assert tracker.evaluate_wait(9.0) == (True, None)

    # Turn 4: kill monitor, post [done], emit final summary
    tracker.observe_event(
        _tool_event(
            ACPEventType.TOOL_CALL_DONE,
            call_id="c_done",
            title="post_message",
            status="completed",
            swarm_signal=SwarmToolSignal(tool="post_message", is_done_post=True),
        ),
        10.0,
    )
    tracker.observe_event(_text_event("全部优化与测试已完成。"), 10.05)
    should_wait, timeout = tracker.evaluate_wait(10.08)
    assert should_wait is True
    assert timeout == pytest.approx(0.07, abs=0.01)

    should_wait_done, timeout_done = tracker.evaluate_wait(10.20)
    assert should_wait_done is False
    assert timeout_done is None
    assert tracker.final_turn_text == "全部优化与测试已完成。"


def test_swarm_tracker_completes_when_all_task_workers_killed_or_roster_idle() -> None:
    kill_tracker = SwarmLifecycleTracker(all_killed_text_settle_s=0.1, turn_gap_reset_s=0.1)
    kill_tracker.observe_event(
        _tool_event(
            ACPEventType.TOOL_CALL_DONE,
            call_id="c_grp",
            title="swarm",
            status="completed",
            swarm_signal=SwarmToolSignal(
                tool="swarm",
                action="add",
                added_members=("impl-1", "impl-2"),
            ),
        ),
        1.0,
    )
    kill_tracker.observe_event(_text_event("正在执行"), 1.1)
    kill_tracker.on_prompt_response_returned(1.1)

    # Prefix kill clears both impl-1 and impl-2, leaving monitor out of concern
    kill_tracker.observe_event(
        _tool_event(
            ACPEventType.TOOL_CALL_DONE,
            call_id="c_kill",
            title="swarm",
            status="completed",
            swarm_signal=SwarmToolSignal(tool="swarm", action="kill", killed_targets=("impl",)),
        ),
        1.2,
    )
    kill_tracker.observe_event(_text_event("Swarm 已收尾完成。"), 1.25)
    assert kill_tracker.evaluate_wait(1.30)[0] is True
    assert kill_tracker.evaluate_wait(1.40) == (False, None)
    assert kill_tracker.final_turn_text == "Swarm 已收尾完成。"

    idle_tracker = SwarmLifecycleTracker(idle_roster_text_settle_s=0.1, turn_gap_reset_s=0.2)
    idle_tracker.observe_event(
        _tool_event(
            ACPEventType.TOOL_CALL_DONE,
            call_id="c_add",
            title="swarm",
            status="completed",
            swarm_signal=SwarmToolSignal(tool="swarm", action="add", added_members=("worker-a",)),
        ),
        2.0,
    )
    idle_tracker.observe_event(_text_event("启动"), 2.1)
    idle_tracker.on_prompt_response_returned(2.1)
    idle_tracker.observe_event(
        _tool_event(
            ACPEventType.TOOL_CALL_DONE,
            call_id="c_st",
            title="swarm",
            status="completed",
            swarm_signal=SwarmToolSignal(
                tool="swarm",
                action="status",
                roster_members=(("monitor", "waiting"), ("worker-a", "waiting")),
            ),
        ),
        3.0,
    )
    idle_tracker.observe_event(_text_event("所有 worker 已空闲，总结如下。"), 3.1)
    assert idle_tracker.evaluate_wait(3.15)[0] is True
    assert idle_tracker.evaluate_wait(3.25) == (False, None)
    assert idle_tracker.final_turn_text == "所有 worker 已空闲，总结如下。"


def test_continuation_checks_swarm_final_text_instead_of_interim_question() -> None:
    from src.acp.continuation import _requests_explicit_user_input, merge_prompt_results

    merged = merge_prompt_results(
        PromptResult(stop_reason="end_turn", text="是否继续?", swarm_used=True),
        PromptResult(
            stop_reason="end_turn",
            text="是否继续?\n全部任务已完成。",
            swarm_used=True,
            swarm_completed=True,
            swarm_final_text="全部任务已完成。",
        ),
    )
    assert merged.swarm_used is True
    assert merged.swarm_completed is True
    assert merged.swarm_final_text == "全部任务已完成。"
    assert _requests_explicit_user_input(merged) is False


def test_swarm_tracker_ignores_non_main_source_events() -> None:
    tracker = SwarmLifecycleTracker(done_text_settle_s=0.1, turn_gap_reset_s=0.1)
    tracker.observe_event(
        _tool_event(
            ACPEventType.TOOL_CALL_DONE,
            call_id="c_add",
            title="swarm",
            status="completed",
            swarm_signal=SwarmToolSignal(tool="swarm", action="add", added_members=("worker-1",)),
        ),
        1.0,
    )
    tracker.on_prompt_response_returned(1.05)
    tracker.observe_event(
        _tool_event(
            ACPEventType.TOOL_CALL_DONE,
            call_id="c_done",
            title="post_message",
            status="completed",
            swarm_signal=SwarmToolSignal(tool="post_message", is_done_post=True),
        ),
        2.0,
    )
    tracker.observe_event(_text_event("主协调者终答"), 2.01)
    activity_before_child = tracker.last_activity_monotonic

    # Subagent TEXT_CHUNK and TOOL_CALL_* must not pollute final_turn_text,
    # erase has_text_since_last_tool, open child tool IDs, or extend settle timer.
    assert tracker.observe_event(
        ACPEvent(
            event_type=ACPEventType.TEXT_CHUNK,
            text="子代理噪声",
            source_id="child-subagent-1",
        ),
        2.05,
    ) is False
    assert tracker.observe_event(
        ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            source_id="child-subagent-1",
            tool_call=ToolCallInfo(
                id="child-tool-1",
                title="read_file",
                kind="read",
                status="in_progress",
            ),
        ),
        2.08,
    ) is False
    assert is_swarm_working_event(
        ACPEvent(
            event_type=ACPEventType.TOOL_CALL_DONE,
            source_id="child-subagent-1",
            tool_call=ToolCallInfo(
                id="child-swarm-call",
                title="swarm",
                kind="other",
                status="completed",
                swarm_signal=SwarmToolSignal(tool="swarm", action="add", added_members=("helper",)),
            ),
        )
    ) is False

    assert tracker.final_turn_text == "主协调者终答"
    assert tracker.has_text_since_last_tool is True
    assert tracker.last_activity_monotonic == activity_before_child
    assert tracker.evaluate_wait(2.12) == (False, None)
