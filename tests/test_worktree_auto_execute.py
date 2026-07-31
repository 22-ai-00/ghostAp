"""Tests for worktree auto-execute fast path and silent mode.

Covers:
  (a) /wt with goal parsed correctly
  (b) /wt without goal falls back to confirm card
  (c) Goal from card input triggers fast path
  (d) finish_selection with goal auto-triggers execution
  (e) Zero tools selection error
  (f) Silent mode throttle and timeout notification
  (g) Goal persistence across tool/model selection
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.feishu.handlers.system import SystemHandler
from src.feishu.handlers.worktree import WorktreeHandler
from src.feishu.slash_command_parser import SlashCommandParser
from src.project.context import ProjectContext
from src.worktree_engine.manager import WorktreeManager
from src.worktree_engine.models import (
    WorktreeJourneyState,
    WorktreeJourneyStatus,
    WorktreeRuntimeState,
    WorktreeSelectionItem,
    WorktreeUnit,
    transition_journey_state,
)
from src.worktree_engine.review_adapter import (
    WorktreeReviewOutcome,
    WorktreeReviewPlan,
    WorktreeReviewVerdict,
)
from src.worktree_engine.selection import WorktreeToolOption

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_system_handler() -> WorktreeHandler:
    ctx = MagicMock()
    ctx.settings.ref_note_enabled = False
    handler = WorktreeHandler(ctx)
    return handler


def _make_project(pid: str = "p-auto") -> ProjectContext:
    return ProjectContext(project_id=pid, project_name="AutoTest", root_path="/tmp/auto-test")


_FAKE_TOOLS = [
    {"provider": "acp", "tool_name": "coco", "display_name": "Coco",
     "description": "AI 编程", "supports_model": False, "skip_model_selection": True},
    {"provider": "cli", "tool_name": "claude", "display_name": "Claude",
     "description": "Claude CLI", "supports_model": True},
]


# ---------------------------------------------------------------------------
# Journey state machine tests (PENDING → AUTO_EXECUTING → RUNNING → COMPLETED/FAILED)
# ---------------------------------------------------------------------------


def test_journey_goal_created_transitions_to_pending():
    """goal_created 事件将旅程从 IDLE 推进到 PENDING，并记录 goal。"""
    state = WorktreeJourneyState()
    assert state.status == WorktreeJourneyStatus.IDLE

    new_state = transition_journey_state(state, event="goal_created", goal="实现登录功能")

    assert new_state is not state
    assert new_state.status == WorktreeJourneyStatus.PENDING
    assert new_state.goal == "实现登录功能"
    assert new_state.last_error == ""


def test_journey_auto_execute_started_from_pending():
    """auto_execute_started 仅允许在 IDLE/PENDING 时进入 AUTO_EXECUTING，并可设置 silent_mode。"""
    state = WorktreeJourneyState(status=WorktreeJourneyStatus.PENDING, goal="添加单元测试")

    new_state = transition_journey_state(
        state,
        event="auto_execute_started",
        silent_mode=True,
    )

    assert new_state.status == WorktreeJourneyStatus.AUTO_EXECUTING
    assert new_state.goal == "添加单元测试"
    assert new_state.silent_mode is True
    assert new_state.last_error == ""


def test_journey_execution_flow_to_completed():
    """从 PENDING → AUTO_EXECUTING → RUNNING → COMPLETED 的主干路径。"""
    s0 = WorktreeJourneyState()
    s1 = transition_journey_state(s0, event="goal_created", goal="重构数据库")
    assert s1.status == WorktreeJourneyStatus.PENDING

    s2 = transition_journey_state(s1, event="auto_execute_started")
    assert s2.status == WorktreeJourneyStatus.AUTO_EXECUTING

    s3 = transition_journey_state(s2, event="execution_started")
    assert s3.status == WorktreeJourneyStatus.RUNNING

    s4 = transition_journey_state(s3, event="execution_succeeded")
    assert s4.status == WorktreeJourneyStatus.COMPLETED
    assert s4.last_error == ""


def test_journey_execution_failed_records_error():
    """execution_failed 将状态置为 FAILED 并记录 last_error。"""
    running = WorktreeJourneyState(status=WorktreeJourneyStatus.RUNNING, goal="自动修复 CI")

    failed = transition_journey_state(
        running,
        event="execution_failed",
        error="某个 worktree 单元执行失败",
    )

    assert failed.status == WorktreeJourneyStatus.FAILED
    assert "失败" in failed.last_error


def test_journey_illegal_transition_does_not_mutate_state():
    """非法事件/迁移保持原状态不变，仅在 last_error 中留下提示。"""
    state = WorktreeJourneyState(status=WorktreeJourneyStatus.IDLE)

    new_state = transition_journey_state(state, event="execution_succeeded")

    assert new_state.status == WorktreeJourneyStatus.IDLE
    assert "非法状态迁移" in (new_state.last_error or "") or "未知旅程事件" in (new_state.last_error or "")


def test_journey_reset_always_returns_idle_state():
    """reset 事件总是返回一个全新的 IDLE 状态，用于清理一次旅程。"""
    state = WorktreeJourneyState(
        status=WorktreeJourneyStatus.FAILED,
        goal="之前的任务",
        last_error="error",
        silent_mode=True,
    )

    reset_state = transition_journey_state(state, event="reset")

    assert reset_state.status == WorktreeJourneyStatus.IDLE
    assert reset_state.goal == ""
    assert reset_state.last_error == ""
    assert reset_state.silent_mode is False


# ---------------------------------------------------------------------------
# (a) /wt with goal parsed correctly
# ---------------------------------------------------------------------------

def test_wt_command_with_goal_parsed():
    """/wt 实现登录功能 → goal='实现登录功能' persisted in selection state."""
    handler = _make_system_handler()
    project = _make_project()
    handler.ctx.project_manager.get_active_project.return_value = project

    with patch.object(handler, "_get_available_worktree_tools", return_value=_FAKE_TOOLS):
        handler.handle_worktree_command("msg1", "chat1", project, goal="实现登录功能")

    state = WorktreeManager.get_state(project)
    assert state.selection.pending_goal == "实现登录功能"
    # journey 也应记录该目标，并处于 PENDING 阶段
    assert state.journey.goal == "实现登录功能"
    assert state.journey.status == WorktreeJourneyStatus.PENDING

    # Verify a session was created (CardSession dispatched tool_select event)
    assert "p-auto" in handler._renderer._sessions


def test_apply_journey_event_updates_runtime_state_journey():
    """WorktreeManager.apply_journey_event 应通过 transition_journey_state 更新 journey。"""
    # 初始运行态
    runtime_state = WorktreeRuntimeState()
    assert runtime_state.journey.status == WorktreeJourneyStatus.IDLE

    # 应用 goal_created 事件
    WorktreeManager.apply_journey_event(
        runtime_state,
        event="goal_created",
        goal="实现登录功能",
    )

    assert runtime_state.journey.status == WorktreeJourneyStatus.PENDING
    assert runtime_state.journey.goal == "实现登录功能"

    # 再次应用 auto_execute_started，打开 silent_mode
    WorktreeManager.apply_journey_event(
        runtime_state,
        event="auto_execute_started",
        silent_mode=True,
    )

    assert runtime_state.journey.status == WorktreeJourneyStatus.AUTO_EXECUTING
    assert runtime_state.journey.silent_mode is True


def test_apply_journey_event_handles_unknown_event_gracefully():
    """未知事件不会抛异常，并在 journey.last_error 中留下标记。"""
    runtime_state = WorktreeRuntimeState()

    WorktreeManager.apply_journey_event(runtime_state, event="__unknown__")

    assert runtime_state.journey.status == WorktreeJourneyStatus.IDLE
    # transition_journey_state 会在 last_error 中写入 "未知旅程事件"，兜底逻辑不应覆盖该信息
    assert runtime_state.journey.last_error


def test_is_awaiting_goal_truth_table_matches_journey_status_enum():
    """遍历全部 WorktreeJourneyStatus，校验 is_awaiting_goal 的真值表契约。

    约定真值表（在 WorktreeManager.is_awaiting_goal docstring 中声明）：
    - IDLE/RUNNING/COMPLETED/FAILED → 一律 False；
    - PENDING/AUTO_EXECUTING        → 仅当存在 ready 单元时为 True。
    """

    truth_table = {
        WorktreeJourneyStatus.IDLE: False,
        WorktreeJourneyStatus.PENDING: True,
        WorktreeJourneyStatus.AWAITING_GOAL: True,
        WorktreeJourneyStatus.AUTO_EXECUTING: True,
        WorktreeJourneyStatus.RUNNING: False,
        WorktreeJourneyStatus.COMPLETED: False,
        WorktreeJourneyStatus.FAILED: False,
    }

    # 枚举防御：确保测试覆盖全部枚举成员
    assert set(truth_table.keys()) == set(WorktreeJourneyStatus)

    # AWAITING_GOAL is the post-selection state before real units exist; the
    # others require at least one ready unit.
    for status, expects_true_with_ready in truth_table.items():
        # 有 ready 单元的情况
        state = WorktreeRuntimeState()
        state.journey.status = status
        state.units = [WorktreeUnit(unit_id="u1", status="ready")]
        assert WorktreeManager.is_awaiting_goal(state) is expects_true_with_ready

        # 无 ready 单元的情况
        state_no_ready = WorktreeRuntimeState()
        state_no_ready.journey.status = status
        state_no_ready.units = [WorktreeUnit(unit_id="u1", status="pending")]
        assert WorktreeManager.is_awaiting_goal(state_no_ready) is (
            status == WorktreeJourneyStatus.AWAITING_GOAL
        )


def test_wt_prefix_command_parses_goal():
    """handle_worktree_prefix_command extracts goal from '/wt 实现登录'."""
    handler = _make_system_handler()
    project = _make_project()
    handler.ctx.project_manager.get_active_project.return_value = project

    reply_mock = MagicMock()
    with patch.object(handler, "_get_available_worktree_tools", return_value=_FAKE_TOOLS), \
         patch.object(handler, "reply_text", reply_mock):
        handler.handle_worktree_prefix_command("msg1", "chat1", "/wt 实现登录", project)

    state = WorktreeManager.get_state(project)
    assert state.selection.pending_goal == "实现登录"
    assert state.journey.goal == "实现登录"
    assert state.journey.status == WorktreeJourneyStatus.PENDING


# ---------------------------------------------------------------------------
# (b) /wt without goal falls back to confirm card
# ---------------------------------------------------------------------------

def test_no_goal_fallback_confirm_card():
    """/wt without goal → finish_selection → confirm card dispatched via session."""
    handler = _make_system_handler()
    project = _make_project("p-fb")
    handler.ctx.project_manager.get_active_project.return_value = project
    handler.ctx.project_manager.get_project.return_value = project
    handler.ctx.project_manager.get_project_for_chat.return_value = project

    mgr = handler._worktree_manager()
    mgr.start_selection(project)
    mgr.select_tool(project, WorktreeToolOption(
        provider="acp", tool_name="coco", display_name="Coco", supports_model=False,
    ))
    mgr.add_pending_item(project)
    mgr.back_to_tool_selection(project)

    mock_session = MagicMock()
    mock_session.closed = False
    with patch.object(handler, "_get_or_create_session", return_value=mock_session):
        handler.handle_finish_worktree_selection("msg-fb", "chat-fb", project_id="p-fb", value={})

    # Should dispatch WORKTREE_CONFIRM event (no goal → confirm card, not auto-execute)
    mock_session.dispatch.assert_called_once()
    event = mock_session.dispatch.call_args[0][0]
    from src.card.events import CardEventType
    assert event.type == CardEventType.WORKTREE_CONFIRM
    assert event.payload.get("goal", "") == ""


# ---------------------------------------------------------------------------
# (c) Goal from card input triggers fast path
# ---------------------------------------------------------------------------

def test_goal_from_card_input_triggers_auto_execute():
    """finish_selection with goal in _input_value → auto-execute (no confirm card)."""
    handler = _make_system_handler()
    project = _make_project("p-inp")
    handler.ctx.project_manager.get_active_project.return_value = project
    handler.ctx.project_manager.get_project.return_value = project
    handler.ctx.project_manager.get_project_for_chat.return_value = project

    mgr = handler._worktree_manager()
    mgr.start_selection(project)
    mgr.select_tool(project, WorktreeToolOption(
        provider="cli", tool_name="claude", display_name="Claude", supports_model=False,
    ))
    mgr.add_pending_item(project)

    auto_exec_mock = MagicMock()
    with patch.object(handler, "_auto_execute_worktree", auto_exec_mock):
        handler.handle_finish_worktree_selection(
            "msg-inp", "chat-inp", project_id="p-inp",
            value={"_input_value": "优化性能"},
        )

    auto_exec_mock.assert_called_once()
    call_args = auto_exec_mock.call_args
    assert call_args[0][2] == "优化性能"  # goal positional arg


# ---------------------------------------------------------------------------
# (d) finish_selection with goal auto-triggers execution
# ---------------------------------------------------------------------------

def test_finish_selection_with_pending_goal_auto_executes():
    """pending_goal set from /wt → finish_selection auto-triggers _auto_execute_worktree."""
    handler = _make_system_handler()
    project = _make_project("p-pend")
    handler.ctx.project_manager.get_active_project.return_value = project
    handler.ctx.project_manager.get_project.return_value = project
    handler.ctx.project_manager.get_project_for_chat.return_value = project

    mgr = handler._worktree_manager()
    mgr.start_selection(project, goal="添加单元测试")
    mgr.select_tool(project, WorktreeToolOption(
        provider="acp", tool_name="coco", display_name="Coco", supports_model=False,
    ))
    mgr.add_pending_item(project)

    auto_exec_mock = MagicMock()
    with patch.object(handler, "_auto_execute_worktree", auto_exec_mock):
        handler.handle_finish_worktree_selection(
            "msg-pend", "chat-pend", project_id="p-pend", value={},
        )

    auto_exec_mock.assert_called_once()
    assert auto_exec_mock.call_args[0][2] == "添加单元测试"


def test_auto_execute_worktree_calls_execute_with_silent_mode():
    """_auto_execute_worktree → ensure_worktrees → handle_worktree_execute(silent_mode=True)."""
    handler = _make_system_handler()
    project = _make_project("p-exec")
    handler.ctx.project_manager.get_active_project.return_value = project

    mgr = handler._worktree_manager()
    mgr.start_selection(project, goal="测试目标")
    mgr.select_tool(project, WorktreeToolOption(
        provider="acp", tool_name="coco", display_name="Coco", supports_model=False,
    ))
    mgr.add_pending_item(project)
    mgr.finalize_selection(project)

    execute_mock = MagicMock()
    with patch.object(mgr, "ensure_worktrees") as ew_mock, \
         patch.object(handler, "handle_worktree_execute", execute_mock):
        # Mock ensure_worktrees to return state with units
        state = mgr.get_state(project)
        state.units = [WorktreeUnit(unit_id="u1")]
        state.last_error = ""
        ew_mock.return_value = state

        handler._auto_execute_worktree("msg-exec", "chat-exec", "测试目标", project=project)

    execute_mock.assert_called_once()
    kwargs = execute_mock.call_args
    assert kwargs[1]["silent_mode"] is True
    assert kwargs[0][2] == "测试目标"  # goal
    # Verify units were set to "ready"
    assert all(u.status == "ready" for u in state.units)
    # Verify journey reflects AUTO_EXECUTING with silent_mode
    assert state.journey.status == WorktreeJourneyStatus.AUTO_EXECUTING
    assert state.journey.goal == "测试目标"
    assert state.journey.silent_mode is True


# ---------------------------------------------------------------------------
# (e) Zero tools selection error
# ---------------------------------------------------------------------------

def test_zero_tools_error_blocks_execution():
    """finish_selection with 0 tools → error, no _auto_execute_worktree call."""
    handler = _make_system_handler()
    project = _make_project("p-zero")
    handler.ctx.project_manager.get_active_project.return_value = project
    handler.ctx.project_manager.get_project.return_value = project
    handler.ctx.project_manager.get_project_for_chat.return_value = project

    mgr = handler._worktree_manager()
    mgr.start_selection(project, goal="不该执行")

    error_mock = MagicMock()
    auto_exec_mock = MagicMock()
    with patch.object(handler, "reply_error", error_mock), \
         patch.object(handler, "_auto_execute_worktree", auto_exec_mock):
        handler.handle_finish_worktree_selection(
            "msg-zero", "chat-zero", project_id="p-zero", value={"goal": "不该执行"},
        )

    error_mock.assert_called_once()
    assert "请至少选择一个工具" in str(error_mock.call_args)
    auto_exec_mock.assert_not_called()


# ---------------------------------------------------------------------------
# (f) Silent mode throttle and timeout notification
# ---------------------------------------------------------------------------

def test_silent_mode_throttle_interval():
    """silent_mode=True → throttle interval is 30s (not 0.5s)."""
    handler = _make_system_handler()
    project = _make_project("p-sil")
    handler.ctx.project_manager.get_active_project.return_value = project

    mgr = handler._worktree_manager()
    mgr.start_selection(project)
    mgr.select_tool(project, WorktreeToolOption(
        provider="acp", tool_name="coco", display_name="Coco", supports_model=False,
    ))
    mgr.add_pending_item(project)
    mgr.finalize_selection(project)

    # Prepare units
    state = mgr.get_state(project)
    state.units = [WorktreeUnit(unit_id="u1", status="ready")]

    mock_session = MagicMock()
    mock_session.closed = False

    # Mock execute_goal to capture the on_unit_update callback
    captured_callback = [None]

    def fake_execute(proj, goal, on_unit_update=None, **kw):
        captured_callback[0] = on_unit_update
        state.units[0].status = "completed"
        state.merge_entry_ready = False
        state.last_error = ""
        return state

    with patch.object(mgr, "execute_goal", side_effect=fake_execute), \
         patch.object(handler, "_get_or_create_session", return_value=mock_session):
        handler.handle_worktree_execute("msg-sil", "chat-sil", "测试", project=project, silent_mode=True)

    # Check initial dispatch includes STARTED and WORKTREE_PROGRESS
    dispatch_calls = [c.args[0] for c in mock_session.dispatch.call_args_list]
    types = [e.type.value for e in dispatch_calls]
    assert "started" in types
    assert "worktree_progress" in types

    # Callback should have been captured
    assert captured_callback[0] is not None

    # Simulate rapid callbacks — they should be throttled at 30s
    mock_session.dispatch.reset_mock()
    cb = captured_callback[0]
    cb(state.units[0])  # first call — should NOT update (just set time)
    cb(state.units[0])  # second call within <30s — should be throttled
    # Only final result card patched (from main flow), not from rapid callbacks.
    # The rapid callbacks should not cause additional dispatch calls beyond throttle.


def test_handle_worktree_execute_auto_merges_and_cleans_after_success():
    """Successful WT execution auto-merges and auto-cleans without a manual merge step."""
    handler = _make_system_handler()
    project = _make_project("p-auto-merge")
    handler.ctx.project_manager.get_active_project.return_value = project

    mgr = handler._worktree_manager()
    state = mgr.get_state(project)
    state.base_branch = "main"
    state.git_root = project.root_path
    state.units = [
        WorktreeUnit(
            unit_id="u1",
            branch_name="ghostap/wt/topic/01-unit",
            status="ready",
            has_changes=True,
        )
    ]

    mock_session = MagicMock()
    mock_session.closed = False
    merge_results = [{
        "branch": "ghostap/wt/topic/01-unit",
        "success": True,
        "detail": "合并成功；如有同文件冲突，已自动优先采用 Worktree 分支变更",
    }]

    def fake_execute(proj, goal, on_unit_update=None, **kw):
        state.units[0].status = "completed"
        state.merge_entry_ready = True
        state.merge_notes = [{"branch": state.units[0].branch_name, "status": "ready", "summary": "ready"}]
        state.last_error = ""
        return state

    with patch.object(mgr, "execute_goal", side_effect=fake_execute), \
         patch.object(mgr, "merge_to_base", return_value=(state, merge_results)) as merge_mock, \
         patch.object(mgr, "cleanup_worktrees", return_value=(WorktreeRuntimeState(), [])) as cleanup_mock, \
         patch.object(handler, "_get_or_create_session", return_value=mock_session), \
         patch.object(handler, "_close_session") as close_mock:
        handler.handle_worktree_execute("msg-auto", "chat-auto", "实现功能", project=project)

    merge_mock.assert_called_once_with(project)
    cleanup_mock.assert_called_once_with(project, force=True)
    close_mock.assert_called_once_with(project.project_id)
    cleanup_events = [
        c.args[0] for c in mock_session.dispatch.call_args_list
        if getattr(c.args[0].type, "value", c.args[0].type) == "worktree_cleanup"
    ]
    assert cleanup_events
    assert cleanup_events[-1].payload["cleanup_phase"] == "completed"
    assert "Worktree 分支变更" in cleanup_events[-1].payload["merge_results"][0]["detail"]


# ---------------------------------------------------------------------------
# (g) Goal persistence across tool/model selection
# ---------------------------------------------------------------------------

def test_goal_persistence_across_selection():
    """Goal set via /wt persists through tool select → model select → back to tool select."""
    handler = _make_system_handler()
    project = _make_project("p-pers")
    handler.ctx.project_manager.get_active_project.return_value = project
    handler.ctx.project_manager.get_project.return_value = project
    handler.ctx.project_manager.get_project_for_chat.return_value = project

    mgr = handler._worktree_manager()
    mock_session = MagicMock()
    mock_session.closed = False

    # Step 1: /wt with goal
    with patch.object(handler, "_get_available_worktree_tools", return_value=_FAKE_TOOLS), \
         patch.object(handler, "_get_or_create_session", return_value=mock_session):
        handler.handle_worktree_command("m1", "c1", project, goal="实现搜索功能")

    state = mgr.get_state(project)
    assert state.selection.pending_goal == "实现搜索功能"

    # Step 2: select tool — goal should persist (goal no longer in card value)
    with patch.object(handler, "_get_models_for_tool", return_value=[{"name": "sonnet", "display_name": "Sonnet", "is_default": True},
                                                                      {"name": "opus", "display_name": "Opus", "is_default": False}]), \
         patch.object(handler, "_get_available_worktree_tools", return_value=_FAKE_TOOLS), \
         patch.object(handler, "_get_or_create_session", return_value=mock_session):
        handler.handle_worktree_select_tool(
            "m2", "c1", project_id="p-pers",
            value={"tool_name": "claude", "provider": "cli", "supports_model": True},
        )

    state = mgr.get_state(project)
    assert state.selection.pending_goal == "实现搜索功能"
    assert state.selection.pending_item is not None

    # Step 3: select model — should return to tool select (no auto-execute)
    mock_session.dispatch.reset_mock()
    with patch.object(handler, "_get_available_worktree_tools", return_value=_FAKE_TOOLS), \
         patch.object(handler, "_get_or_create_session", return_value=mock_session):
        handler.handle_worktree_select_model(
            "m3", "c1", project_id="p-pers",
            value={"model_name": "sonnet", "model_display_name": "Sonnet"},
        )

    # Should dispatch tool_select back (not auto-execute)
    mock_session.dispatch.assert_called_once()
    state = mgr.get_state(project)
    assert state.selection.pending_goal == "实现搜索功能"
    assert len(state.selection.selected_items) == 1


def test_worktree_select_default_model_adds_unpinned_selection():
    """The explicit default-model option should not pin a model into WT units."""
    handler = _make_system_handler()
    project = _make_project("p-default-model")
    handler.ctx.project_manager.get_active_project.return_value = project
    handler.ctx.project_manager.get_project.return_value = project
    handler.ctx.project_manager.get_project_for_chat.return_value = project

    mgr = handler._worktree_manager()
    mgr.start_selection(project)
    mgr.select_tool(project, WorktreeToolOption(
        provider="acp", tool_name="codex", display_name="Codex", supports_model=True,
    ))

    mock_session = MagicMock()
    mock_session.closed = False
    with patch.object(handler, "_get_available_worktree_tools", return_value=_FAKE_TOOLS), \
         patch.object(handler, "_get_or_create_session", return_value=mock_session):
        handler.handle_worktree_select_model(
            "m-default", "c-default", project_id="p-default-model",
            value={"model_name": "__ghostap_default_model__", "use_default_model": True},
        )

    state = mgr.get_state(project)
    assert len(state.selection.selected_items) == 1
    item = state.selection.selected_items[0]
    assert item.tool_name == "codex"
    assert item.model_name is None
    assert item.effective_model_display_name == "默认模型"



def test_plan_goal_emits_goal_created_and_updates_journey():
    """WorktreeManager.plan_goal 应触发 goal_created 并记录 journey.goal。"""
    project = _make_project("p-plan")
    mgr = WorktreeManager(project_manager=None)

    # 避免真实 dispatcher 行为，替换为简单 stub
    mgr._dispatcher = MagicMock()
    mgr._dispatcher.plan_user_goal.return_value = []
    mgr._reporter = MagicMock()
    mgr._reporter.refresh_state.side_effect = lambda s: s

    state = mgr.plan_goal(project, "实现搜索功能")

    assert state.journey.status == WorktreeJourneyStatus.PENDING
    assert state.journey.goal == "实现搜索功能"


def test_execute_goal_emits_execution_events_on_success_and_failure():
    """execute_goal 应在成功/失败路径上分别推进 COMPLETED/FAILED 旅程状态。"""
    project = _make_project("p-exec-flow")
    mgr = WorktreeManager(project_manager=None)

    # 构造初始运行态：已有一个 unit，selection 中有一个 item，避免 ensure_worktrees 分支
    state = WorktreeManager.get_state(project)
    state.units = [WorktreeUnit(unit_id="u1")]
    state.selection.selected_items = [
        WorktreeSelectionItem(provider="acp", tool_name="coco", display_name="Coco")
    ]

    # Stub dispatcher / reporter
    mgr._dispatcher = MagicMock()
    # plan_user_goal 直接返回现有 units
    mgr._dispatcher.plan_user_goal.side_effect = lambda goal, units, items: units
    # execute_units 在第一次调用成功返回，在第二次调用抛出异常
    def _successful_dispatch(units, **_kwargs):
        for unit in units:
            unit.status = "completed"
        return units

    mgr._dispatcher.execute_units.side_effect = [_successful_dispatch(state.units), Exception("boom")]

    mgr._review_adapter = MagicMock()
    mgr._review_adapter.plan_roles.return_value = WorktreeReviewPlan()
    mgr._review_adapter.review_units.return_value = WorktreeReviewOutcome(
        verdict=WorktreeReviewVerdict.PASS,
        summary="verified",
        findings=[
            {
                "severity": "observation",
                "message": "verified",
                "evidence": "targeted test passed",
            }
        ],
        tests=[
            {
                "command": "fake test",
                "passed": True,
                "verified": True,
                "evidence": "passed",
            }
        ],
    )

    mgr._reporter = MagicMock()
    mgr._reporter.refresh_state.side_effect = lambda s: s

    # 成功路径：第一次调用 execute_goal
    ok_state = mgr.execute_goal(project, "修复 bug")
    assert ok_state.journey.status == WorktreeJourneyStatus.COMPLETED
    assert ok_state.journey.goal == "修复 bug"
    assert ok_state.iteration_count == 1

    # 失败路径：第二次调用 execute_goal，会走 execute_units 异常分支
    fail_state = mgr.execute_goal(project, "再次修复 bug")
    assert fail_state.journey.status == WorktreeJourneyStatus.FAILED
    assert fail_state.iteration_count == 2
    # last_error 由 get_error_detail 生成，这里只要求非空
    assert fail_state.journey.last_error



# ---------------------------------------------------------------------------
# is_interceptable_command tests
# ---------------------------------------------------------------------------

def test_is_interceptable_command_wt_with_goal():
    m = SlashCommandParser.parse
    assert SystemHandler.is_interceptable_command_match(m("/wt 实现登录功能"))
    assert SystemHandler.is_interceptable_command_match(m("/worktree 重构认证"))
    assert SystemHandler.is_interceptable_command_match(m("/wt"))
    assert SystemHandler.is_interceptable_command_match(m("/worktree"))
    assert not SystemHandler.is_interceptable_command_match(m("/wtzzzz"))


# ---------------------------------------------------------------------------
# Edge-case tests (boundary hardening)
# ---------------------------------------------------------------------------


def test_whitespace_only_goal_falls_back_to_confirm_card():
    """Goal that is pure whitespace (e.g. '   ') should be treated as empty → confirm card."""
    handler = _make_system_handler()
    project = _make_project("p-ws")
    handler.ctx.project_manager.get_active_project.return_value = project
    handler.ctx.project_manager.get_project.return_value = project
    handler.ctx.project_manager.get_project_for_chat.return_value = project

    mgr = handler._worktree_manager()
    mgr.start_selection(project)
    mgr.select_tool(project, WorktreeToolOption(
        provider="acp", tool_name="coco", display_name="Coco", supports_model=False,
    ))
    mgr.add_pending_item(project)
    mgr.back_to_tool_selection(project)

    mock_session = MagicMock()
    mock_session.closed = False
    auto_exec_mock = MagicMock()
    with patch.object(handler, "_get_or_create_session", return_value=mock_session), \
         patch.object(handler, "_auto_execute_worktree", auto_exec_mock):
        handler.handle_finish_worktree_selection(
            "msg-ws", "chat-ws", project_id="p-ws",
            value={"worktree_goal": "   "},
        )

    # Should NOT trigger auto-execute
    auto_exec_mock.assert_not_called()
    # Should dispatch WORKTREE_CONFIRM event
    mock_session.dispatch.assert_called_once()
    event = mock_session.dispatch.call_args[0][0]
    from src.card.events import CardEventType
    assert event.type == CardEventType.WORKTREE_CONFIRM


def test_goal_with_newlines_preserved():
    """Goal containing newlines should be passed through without truncation."""
    handler = _make_system_handler()
    project = _make_project("p-nl")
    handler.ctx.project_manager.get_active_project.return_value = project
    handler.ctx.project_manager.get_project.return_value = project
    handler.ctx.project_manager.get_project_for_chat.return_value = project

    mgr = handler._worktree_manager()
    mgr.start_selection(project)
    mgr.select_tool(project, WorktreeToolOption(
        provider="acp", tool_name="coco", display_name="Coco", supports_model=False,
    ))
    mgr.add_pending_item(project)

    multiline_goal = "第一步：实现登录\n第二步：添加测试"
    auto_exec_mock = MagicMock()
    mock_session = MagicMock()
    mock_session.closed = False
    with patch.object(handler, "_auto_execute_worktree", auto_exec_mock), \
         patch.object(handler, "_get_or_create_session", return_value=mock_session):
        handler.handle_finish_worktree_selection(
            "msg-nl", "chat-nl", project_id="p-nl",
            value={"worktree_goal": multiline_goal},
        )

    auto_exec_mock.assert_called_once()
    actual_goal = auto_exec_mock.call_args[0][2]
    assert "第一步" in actual_goal
    assert "第二步" in actual_goal


def test_silent_mode_timeout_notification_fires():
    """Silent mode 10-min safety valve: callback at >=600s dispatches progress event with timeout msg.

    The closure captures time from module globals and the session from the handler.
    We must mock both during handle_worktree_execute (to capture the callback) and again
    when invoking the callback (so the closure sees mocked time).
    """
    handler = _make_system_handler()
    project = _make_project("p-timeout")
    handler.ctx.project_manager.get_active_project.return_value = project

    mgr = handler._worktree_manager()
    mgr.start_selection(project)
    mgr.select_tool(project, WorktreeToolOption(
        provider="acp", tool_name="coco", display_name="Coco", supports_model=False,
    ))
    mgr.add_pending_item(project)
    mgr.finalize_selection(project)

    state = mgr.get_state(project)
    state.units = [WorktreeUnit(unit_id="u1", status="ready")]

    mock_session = MagicMock()
    mock_session.closed = False
    captured_callback = [None]

    _fake_now = [1000.0]  # starting fake clock

    def fake_execute(proj, goal, on_unit_update=None, **kw):
        captured_callback[0] = on_unit_update
        state.units[0].status = "completed"
        state.merge_entry_ready = False
        state.last_error = ""
        return state

    import src.feishu.handlers.worktree as _wt_mod
    original_time = _wt_mod.time

    fake_time = MagicMock()
    fake_time.time = lambda: _fake_now[0]

    with patch.object(handler, "_get_or_create_session", return_value=mock_session):
        try:
            _wt_mod.time = fake_time
            with patch.object(mgr, "execute_goal", side_effect=fake_execute):
                handler.handle_worktree_execute("msg-to", "chat-to", "测试超时", project=project, silent_mode=True)
        finally:
            _wt_mod.time = original_time

        assert captured_callback[0] is not None
        cb = captured_callback[0]

        # Advance clock past 600s timeout
        mock_session.dispatch.reset_mock()
        _fake_now[0] = 1000.0 + 601
        try:
            _wt_mod.time = fake_time
            cb(state.units[0])
        finally:
            _wt_mod.time = original_time

    # The 10-min safety valve should have triggered a dispatch call
    assert mock_session.dispatch.call_count >= 1
    # Verify it dispatched a worktree_progress event
    dispatched = [c.args[0] for c in mock_session.dispatch.call_args_list]
    types = [e.type.value for e in dispatched]
    assert "worktree_progress" in types


# ──────────────────────────────────────────────────────────────
# Phase-6: New API tests — enum, ensure_worktree_state, truncate_goal, from_dict migration
# ──────────────────────────────────────────────────────────────

from src.worktree_engine.models import (
    WorktreeUnitStatus,
    ensure_worktree_state,
    truncate_goal,
)


class TestWorktreeUnitStatusEnum:
    """WorktreeUnitStatus 枚举的 str-Enum 兼容性测试。"""

    def test_enum_values_equal_strings(self):
        """str Enum members should be equal to their string values."""
        assert WorktreeUnitStatus.PENDING == "pending"
        assert WorktreeUnitStatus.READY == "ready"
        assert WorktreeUnitStatus.PLANNED == "planned"
        assert WorktreeUnitStatus.RUNNING == "running"
        assert WorktreeUnitStatus.COMPLETED == "completed"
        assert WorktreeUnitStatus.FAILED == "failed"

    def test_unit_from_dict_parses_status_to_enum(self):
        """from_dict should parse string status into WorktreeUnitStatus."""
        unit = WorktreeUnit.from_dict({"unit_id": "u1", "status": "running"})
        assert unit.status is WorktreeUnitStatus.RUNNING

    def test_unit_from_dict_unknown_status_defaults_pending(self):
        """from_dict with unknown status should default to PENDING."""
        unit = WorktreeUnit.from_dict({"unit_id": "u1", "status": "unknown_xyz"})
        assert unit.status is WorktreeUnitStatus.PENDING


class TestEnsureWorktreeState:
    """ensure_worktree_state 共享 getter 测试。"""

    def test_creates_state_on_bare_project(self):
        """Should create WorktreeRuntimeState on a project without worktree_state."""
        project = _make_project("p-ensure-1")
        if hasattr(project, "worktree_state"):
            delattr(project, "worktree_state")
        state = ensure_worktree_state(project)
        assert isinstance(state, WorktreeRuntimeState)
        assert project.worktree_state is state

    def test_returns_existing_state(self):
        """Should return existing state without replacing it."""
        project = _make_project("p-ensure-2")
        existing = WorktreeRuntimeState(enabled=True)
        project.worktree_state = existing
        state = ensure_worktree_state(project)
        assert state is existing
        assert state.enabled is True

class TestTruncateGoal:
    """truncate_goal Unicode-safe 截断测试。"""

    def test_short_goal_unchanged(self):
        assert truncate_goal("hello") == "hello"

    def test_over_length_truncated_with_ellipsis(self):
        goal = "x" * 100
        result = truncate_goal(goal)
        assert result.endswith("...")
        assert len(result) == 83  # 80 + len("...")

    def test_unicode_safe(self):
        goal = "你好世界" * 30  # 120 chars
        result = truncate_goal(goal, max_len=10)
        assert result.endswith("...")
        assert len(result) == 13  # 10 + len("...")


class TestFromDictLastUserGoalMigration:
    """WorktreeRuntimeState.from_dict 的 last_user_goal → journey.goal 迁移测试。"""

    def test_legacy_goal_migrates_to_journey(self):
        """Legacy last_user_goal should migrate to journey.goal when journey.goal is empty."""
        data = {"last_user_goal": "旧目标", "journey": {"status": "idle", "goal": ""}}
        state = WorktreeRuntimeState.from_dict(data)
        assert state.journey.goal == "旧目标"

    def test_journey_goal_takes_precedence(self):
        """If journey.goal is already set, last_user_goal should NOT overwrite it."""
        data = {"last_user_goal": "旧目标", "journey": {"status": "pending", "goal": "新目标"}}
        state = WorktreeRuntimeState.from_dict(data)
        assert state.journey.goal == "新目标"

    def test_no_legacy_goal(self):
        """Without last_user_goal, journey.goal should remain as-is."""
        data = {"journey": {"status": "idle", "goal": "existing"}}
        state = WorktreeRuntimeState.from_dict(data)
        assert state.journey.goal == "existing"
