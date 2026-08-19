from __future__ import annotations

import json
import logging
from contextlib import ExitStack, contextmanager, nullcontext
from unittest.mock import MagicMock, call, patch

import pytest

from src.acp.continuation import PromptContinuationResult
from src.acp.models import (
    ACPEvent,
    ACPEventType,
    ACPGoalInfo,
    ACPImageInfo,
    PlanEntryInfo,
    PlanInfo,
    PromptResult,
    ToolCallInfo,
)
from src.acp.outcome import (
    PromptAssessment,
    PromptOutcome,
    classify_prompt_result,
)
from src.feishu.handlers.programming import (
    ClaudeModeHandler,
    _ActiveProgrammingRun,
    _log_prompt_execution,
)


class _FakeProgrammingCardSession:
    last = None

    def __init__(self, *_args, **kwargs):
        self.failed_text = None
        self.fail_kwargs = {}
        self.finished = False
        self.finish_kwargs = {}
        self.cancelled_reason = None
        self.continuation_boundaries = 0
        self.kwargs = kwargs
        self.process_sink = kwargs.get("process_sink")
        self.process_sink_activation_calls = 0
        type(self).last = self

    def start(self):
        return None

    def get_message_id(self):
        return None

    def activate_process_sink(self):
        self.process_sink_activation_calls += 1
        sink = self.process_sink
        if sink is None:
            return False
        try:
            if not sink.started:
                sink.start()
            return sink.started and bool(getattr(sink, "healthy", True))
        except Exception:
            sink.abort()
            return False

    def get_process_message_id(self):
        if self.process_sink is None:
            return None
        return self.process_sink.message_id

    def wait_until_visible(self, _timeout):
        return True

    def abort(self):
        return None

    def wait_delivery_idle(self, _timeout):
        return True

    def terminal_delivery_succeeded(self):
        return True

    def on_event(self, _event):
        return None

    def on_text(self, text):
        self.final_text = text

    def get_final_text(self):
        return getattr(self, "final_text", "")

    def finish(self, **kwargs):
        self.finished = True
        self.finish_kwargs = kwargs

    def fail(self, text, **kwargs):
        self.failed_text = text
        self.fail_kwargs = kwargs

    def cancel(self, *, reason):
        self.cancelled_reason = reason

    def begin_continuation_turn(self):
        self.continuation_boundaries += 1


class _FakeProcessSink:
    def __init__(
        self,
        *,
        message_id: str | None = "om-cot-process",
        start_error: Exception | None = None,
        call_order: list[str] | None = None,
    ) -> None:
        self.message_id = message_id
        self.started = False
        self.healthy = True
        self.start_error = start_error
        self.call_order = call_order
        self.start_calls = 0
        self.abort_calls = 0

    def start(self) -> None:
        self.start_calls += 1
        if self.call_order is not None:
            self.call_order.append("cot_start")
        if self.start_error is not None:
            raise self.start_error
        self.started = True

    def abort(self) -> None:
        self.abort_calls += 1
        self.started = False


class _QueuedPromptSession:
    def __init__(
        self,
        *results: PromptResult,
        first_event: ACPEvent | None = None,
    ):
        self._results = list(results)
        self._first_event = first_event
        self.calls: list[tuple[str, object, float | int | None]] = []
        self.idle_timeouts: list[float | int | None] = []
        self._force_dead = False
        self.session_id = "queued-session"
        self.message_count = 1

    def send_prompt(self, text, on_event=None, timeout=None, idle_timeout=None):
        self.calls.append((text, on_event, timeout))
        self.idle_timeouts.append(idle_timeout)
        if len(self.calls) == 1 and self._first_event is not None:
            on_event(self._first_event)
        result = self._results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def _pending_result(text: str = "partial") -> PromptResult:
    return PromptResult(
        stop_reason="end_turn",
        text=text,
        plan=PlanInfo(
            entries=[
                PlanEntryInfo(content="implementation", status="completed"),
                PlanEntryInfo(content="verification", status="pending"),
            ]
        ),
    )


def _completed_result(text: str = "done") -> PromptResult:
    return PromptResult(
        stop_reason="end_turn",
        text=text,
        plan=PlanInfo(
            entries=[
                PlanEntryInfo(content="implementation", status="completed"),
                PlanEntryInfo(content="verification", status="completed"),
            ]
        ),
    )


def _incomplete_execution(
    *,
    entered_finalization: bool,
    goal_status: str | None = None,
    execution_windows: int = 1,
    window_limit_reached: bool = False,
) -> PromptContinuationResult:
    goal = (
        ACPGoalInfo(objective="finish", status=goal_status)
        if goal_status is not None
        else None
    )
    result = PromptResult(
        stop_reason="end_turn",
        text="partial result",
        goal=goal,
    )
    assessment = PromptAssessment(
        outcome=PromptOutcome.INCOMPLETE,
        stop_reason="end_turn",
        detail="仍有 3 个计划项未完成",
        pending_plan_entries=3,
    )
    return PromptContinuationResult(
        result=result,
        assessment=assessment,
        automatic_continuations=0,
        entered_finalization=entered_finalization,
        execution_windows=execution_windows,
        window_limit_reached=window_limit_reached,
    )


def _completed_execution(text: str = "done") -> PromptContinuationResult:
    result = PromptResult(
        stop_reason="end_turn",
        text=text,
        plan=PlanInfo(
            entries=[PlanEntryInfo(content="finish", status="completed")]
        ),
    )
    return PromptContinuationResult(
        result=result,
        assessment=classify_prompt_result(result),
        automatic_continuations=0,
        entered_finalization=False,
    )


def _make_handler():
    ctx = MagicMock()
    ctx.settings = MagicMock()
    ctx.settings.claude_execution_timeout = 600
    ctx.settings.coco_execution_timeout = 600
    ctx.settings.programming_agent_idle_timeout_s = 420
    ctx.settings.app_id = "cli_test"
    ctx.settings.repo_lock_hard_timeout = 3600
    ctx.settings.feishu_cot_enabled = False
    ctx.api_client_factory = MagicMock()
    ctx.pending_image_lock = nullcontext()
    ctx.pending_image_keys = {}
    ctx.message_linker = MagicMock()
    ctx.context_manager = MagicMock()

    handler = ClaudeModeHandler(ctx)
    handler.reply_text = MagicMock()
    handler.add_reaction = MagicMock()
    handler.register_message_project = MagicMock()
    handler.ensure_request_id = MagicMock(return_value="req-1")
    return handler


def test_handler_helper_matches_claude_initialization_contract():
    handler = _make_handler()

    assert handler._current_model is None
    assert handler._get_model_name_override() is None


def test_btw_without_running_task_forwards_only_the_question():
    handler = _make_handler()
    handler.handle_message = MagicMock()

    handler.handle_btw("btw-message", "chat", "顺便解释一下", None)

    handler.handle_message.assert_called_once_with(
        "btw-message", "chat", "顺便解释一下", None
    )


def test_btw_during_running_task_uses_read_only_auxiliary_session():
    handler = _make_handler()
    handler.settings.acp_startup_timeout = 20
    handler.handle_message = MagicMock()
    key = handler._active_run_key("chat", None)
    state = _ActiveProgrammingRun(
        task_text="修复截图识别问题",
        cwd="/tmp/project",
        agent_type="codex",
        model_name="gpt-test",
    )
    state.observe(
        ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(
                id="view-1",
                title="查看 bug 截图",
                kind="read",
                status="in_progress",
                content="检查截图细节",
            ),
        )
    )
    handler._active_programming_runs[key] = state
    sidecar = MagicMock()
    sidecar.send_prompt.return_value = PromptResult(
        stop_reason="end_turn",
        text="反复查看是为了确认不同区域的细节。",
    )

    with (
        patch(
            "src.agent_session.create_auxiliary_session",
            return_value=sidecar,
        ) as create_auxiliary,
        patch("src.agent_session.close_session_safely") as close_session,
    ):
        handler.handle_btw(
            "btw-message", "chat", "为什么反复查看？", None
        )

    handler.handle_message.assert_not_called()
    create_auxiliary.assert_called_once_with(
        agent_type="codex",
        cwd="/tmp/project",
        model_name="gpt-test",
        thread_id="btw-observer",
        startup_timeout=20.0,
        startup_retries=1,
        startup_log_failures=False,
    )
    prompt = sidecar.send_prompt.call_args.args[0]
    assert "修复截图识别问题" in prompt
    assert "查看 bug 截图" in prompt
    assert "为什么反复查看？" in prompt
    assert sidecar.send_prompt.call_args.kwargs == {"timeout": 90}
    close_session.assert_called_once_with(sidecar)
    replies = [call.args[1] for call in handler.reply_text.call_args_list]
    assert "主任务不会被中断" in replies[0]
    assert "反复查看是为了确认不同区域的细节" in replies[1]


def test_btw_auxiliary_failure_keeps_main_task_running_and_reports_snapshot():
    handler = _make_handler()
    handler.settings.acp_startup_timeout = 20
    key = handler._active_run_key("chat", None)
    state = _ActiveProgrammingRun(
        task_text="主任务",
        cwd="/tmp/project",
        agent_type="codex",
        model_name=None,
    )
    state.observe(
        ACPEvent(
            event_type=ACPEventType.THOUGHT_CHUNK,
            text="正在定位重复读取来源",
        )
    )
    handler._active_programming_runs[key] = state

    with patch(
        "src.agent_session.create_auxiliary_session",
        side_effect=RuntimeError("observer unavailable"),
    ):
        handler.handle_btw("btw-message", "chat", "现在进展如何？", None)

    replies = [call.args[1] for call in handler.reply_text.call_args_list]
    assert "主任务不会被中断" in replies[0]
    assert "主任务未被中断" in replies[1]
    assert "正在定位重复读取来源" in replies[1]
    assert state.claim_btw() is True
    state.release_btw()


def test_btw_rejects_a_second_parallel_side_question():
    handler = _make_handler()
    key = handler._active_run_key("chat", None)
    state = _ActiveProgrammingRun(
        task_text="主任务",
        cwd="/tmp/project",
        agent_type="codex",
        model_name=None,
    )
    assert state.claim_btw() is True
    handler._active_programming_runs[key] = state

    handler.handle_btw("second", "chat", "第二个问题", None)

    assert "已有一个旁路问题" in handler.reply_text.call_args.args[1]
    state.release_btw()


def test_btw_snapshot_is_bounded_and_preserves_task_header():
    state = _ActiveProgrammingRun(
        task_text="必须保留的主任务",
        cwd="/tmp/project",
        agent_type="codex",
        model_name=None,
    )
    state.observe(ACPEvent(event_type=ACPEventType.THOUGHT_CHUNK, text="第一段"))
    state.observe(ACPEvent(event_type=ACPEventType.THOUGHT_CHUNK, text="第二段"))
    for index in range(80):
        state.observe(
            ACPEvent(
                event_type=ACPEventType.TOOL_CALL_UPDATE,
                tool_call=ToolCallInfo(
                    id=f"tool-{index}",
                    title=f"操作 {index}",
                    kind="read",
                    status="in_progress",
                    content="x" * 500,
                ),
            )
        )

    snapshot = state.snapshot()

    assert snapshot.startswith("主任务: 必须保留的主任务")
    assert len(snapshot) <= 12_000
    assert sum("Agent 分析:" in event for event in state.events) <= 1


def test_btw_during_handle_message_does_not_reacquire_repository_lock():
    handler = _make_handler()
    handler.settings.acp_startup_timeout = 20
    handler.get_working_dir = MagicMock(return_value="/tmp/project")
    handler._get_session_manager().get_session.return_value = MagicMock(
        session_id="main-session"
    )
    handler._acquire_repo_lock = MagicMock(return_value=(None, None, False))
    sidecar = MagicMock()
    sidecar.send_prompt.return_value = PromptResult(
        stop_reason="end_turn", text="这是旁路回答"
    )

    def run_main(*_args, **_kwargs):
        handler.handle_btw(
            "btw-message", "chat", "主任务为什么在读图？", None
        )

    handler.handle_response = MagicMock(side_effect=run_main)
    with (
        patch(
            "src.agent_session.create_auxiliary_session",
            return_value=sidecar,
        ),
        patch("src.agent_session.close_session_safely"),
    ):
        handler.handle_message("main-message", "chat", "修复截图问题", None)

    handler._acquire_repo_lock.assert_called_once_with(None, "chat")
    assert "这是旁路回答" in handler.reply_text.call_args.args[1]


def test_prompt_execution_log_identifies_child_only_incompleteness(
    caplog: pytest.LogCaptureFixture,
) -> None:
    result = PromptResult(
        stop_reason="end_turn",
        tool_calls=[
            ToolCallInfo(
                id="opaque-provider-call-id",
                title="secret command must not be logged",
                kind="other",
                status="completed",
                content="secret prompt must not be logged",
                collaboration_tool="wait_agent",
                subagent_states=(
                    {
                        "source_id": "opaque-child-id",
                        "status": "running",
                        "message": "secret child output must not be logged",
                    },
                ),
            )
        ],
    )
    assessment = classify_prompt_result(result)
    execution = PromptContinuationResult(
        result=result,
        assessment=assessment,
        automatic_continuations=0,
        entered_finalization=False,
    )

    with caplog.at_level(
        logging.INFO,
        logger="src.feishu.handlers.programming",
    ):
        _log_prompt_execution("Codex", execution)

    assert "incomplete_outer_tool_calls=0" in caplog.text
    assert "unresolved_child_tool_calls=1" in caplog.text
    assert "unresolved_tools=wait_agent:completed[running]" in caplog.text
    assert "opaque-provider-call-id" not in caplog.text
    assert "opaque-child-id" not in caplog.text
    assert "secret" not in caplog.text


def test_prompt_execution_log_allowlists_provider_controlled_categories(
    caplog: pytest.LogCaptureFixture,
) -> None:
    result = PromptResult(
        stop_reason="end_turn",
        tool_calls=[
            ToolCallInfo(
                id="opaque-call",
                title="opaque-title",
                kind="sk-proj-kind-secret",
                status="sk-proj-outer-secret",
                collaboration_tool="sk-proj-tool-secret",
                subagent_states=(
                    {
                        "source_id": "opaque-child",
                        "status": "sk-proj-child-secret",
                    },
                ),
            )
        ],
    )
    assessment = classify_prompt_result(result)
    execution = PromptContinuationResult(
        result=result,
        assessment=assessment,
        automatic_continuations=0,
        entered_finalization=False,
    )

    with caplog.at_level(
        logging.INFO,
        logger="src.feishu.handlers.programming",
    ):
        _log_prompt_execution("Codex", execution)

    assert "unresolved_tools=unknown:unknown[unknown]" in caplog.text
    assert "sk-proj" not in caplog.text


def test_prompt_execution_log_allowlists_provider_goal_status(
    caplog: pytest.LogCaptureFixture,
) -> None:
    result = PromptResult(
        stop_reason="end_turn",
        goal=ACPGoalInfo(
            objective="secret objective must not be logged",
            status="sk-proj-goal-secret",
        ),
    )
    assessment = classify_prompt_result(result)
    execution = PromptContinuationResult(
        result=result,
        assessment=assessment,
        automatic_continuations=0,
        entered_finalization=False,
    )

    with caplog.at_level(
        logging.INFO,
        logger="src.feishu.handlers.programming",
    ):
        _log_prompt_execution("Codex", execution)

    assert "goal_status=unknown" in caplog.text
    assert "sk-proj" not in caplog.text
    assert "secret objective" not in caplog.text


@contextmanager
def _streaming_environment(
    adapter_cls=_FakeProgrammingCardSession,
    *,
    delivery=None,
):
    delivery = delivery or MagicMock()
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "src.card.delivery.factory.create_card_delivery",
                return_value=delivery,
            )
        )
        stack.enter_context(
            patch(
                "src.card.delivery.channel_client.LarkChannelCardAPIClient",
                return_value=MagicMock(),
            )
        )
        stack.enter_context(
            patch("src.card.session.CardSession", return_value=MagicMock())
        )
        stack.enter_context(
            patch(
                "src.card.programming_adapter.ProgrammingCardSession",
                adapter_cls,
            )
        )
        stack.enter_context(
            patch(
                "src.feishu.handlers.programming.classify_prompt_result",
                side_effect=AssertionError(
                    "handler must consume continuation assessment directly"
                ),
                create=True,
            )
        )
        yield delivery


def test_programming_owned_delivery_shuts_down_after_success():
    handler = _make_handler()
    handler.ctx.channel_client_factory = MagicMock(return_value=object())
    session = _QueuedPromptSession(_completed_result())
    delivery = MagicMock()
    delivery.shutdown.return_value = True

    with _streaming_environment(delivery=delivery):
        handler.handle_response(
            "msg-owned-success",
            "chat-1",
            "finish",
            session,
            None,
            "/tmp",
            "/tmp",
        )

    delivery.shutdown.assert_called_once()


def test_programming_owned_delivery_shuts_down_after_initial_card_fallback():
    handler = _make_handler()
    handler.ctx.channel_client_factory = MagicMock(return_value=object())
    call_order: list[str] = []
    handler.reply_text.side_effect = lambda *_args, **_kwargs: (
        call_order.append("notice") or "notice-message"
    )
    handler._handle_response_non_streaming = MagicMock(
        side_effect=lambda *_args, **_kwargs: call_order.append("fallback")
    )
    delivery = MagicMock()
    delivery.shutdown.side_effect = lambda **_kwargs: (
        call_order.append("shutdown") or True
    )

    class _InvisibleProgrammingCardSession(_FakeProgrammingCardSession):
        def wait_until_visible(self, _timeout):
            return False

    with _streaming_environment(
        _InvisibleProgrammingCardSession,
        delivery=delivery,
    ):
        handler.handle_response(
            "msg-owned-fallback",
            "chat-1",
            "finish",
            MagicMock(),
            None,
            "/tmp",
            "/tmp",
        )

    delivery.shutdown.assert_called_once()
    handler._handle_response_non_streaming.assert_called_once()
    assert call_order == ["shutdown", "notice", "fallback"]
    notice = handler.reply_text.call_args.args[1]
    assert "已切换为普通文本" in notice
    assert "任务仍在执行" in notice
    assert "cardkit:card:write" in notice
    assert (
        "https://open.feishu.cn/app/cli_test/auth"
        "?q=cardkit:card:write&op_from=openapi&token_type=tenant"
    ) in notice


def test_programming_owned_delivery_shuts_down_when_execution_raises():
    handler = _make_handler()
    handler.ctx.channel_client_factory = MagicMock(return_value=object())
    handler._execute_programming_response = MagicMock(
        side_effect=RuntimeError("unexpected execution failure")
    )
    delivery = MagicMock()
    delivery.shutdown.return_value = True

    with (
        _streaming_environment(delivery=delivery),
        pytest.raises(RuntimeError, match="unexpected execution failure"),
    ):
        handler.handle_response(
            "msg-owned-exception",
            "chat-1",
            "finish",
            MagicMock(),
            None,
            "/tmp",
            "/tmp",
        )

    delivery.shutdown.assert_called_once()


def test_streaming_pending_plan_continues_on_same_session_and_finishes():
    handler = _make_handler()
    handler.ctx.channel_client_factory = MagicMock(return_value=object())
    session = _QueuedPromptSession(_pending_result(), _completed_result())

    with _streaming_environment():
        handler.handle_response(
            "msg-1",
            "chat-1",
            "finish the task",
            session,
            None,
            "/tmp",
            "/tmp",
        )

    adapter = _FakeProgrammingCardSession.last
    assert len(session.calls) == 2
    assert session.idle_timeouts == [420, 420]
    assert session.calls[0][0] == "finish the task"
    assert "自动续做指令" in session.calls[1][0]
    assert adapter.continuation_boundaries == 1
    assert adapter.finished is True
    assert adapter.failed_text is None
    assert callable(adapter.kwargs["session_factory"])
    assert adapter.kwargs["continuation_visibility_timeout"] >= 2.0


def test_streaming_retries_pending_plan_three_times_then_reports_incomplete():
    handler = _make_handler()
    handler.ctx.channel_client_factory = MagicMock(return_value=object())
    session = _QueuedPromptSession(
        _pending_result("first partial"),
        _pending_result("second partial"),
        _pending_result("third partial"),
        _pending_result("fourth partial"),
    )

    with _streaming_environment():
        handler.handle_response(
            "msg-1",
            "chat-1",
            "finish the task",
            session,
            None,
            "/tmp",
            "/tmp",
        )

    adapter = _FakeProgrammingCardSession.last
    assert len(session.calls) == 4
    assert adapter.continuation_boundaries == 3
    assert adapter.finished is False
    assert adapter.failed_text is not None
    assert adapter.fail_kwargs["details"]
    assert adapter.fail_kwargs["detail_action"]["action"] == "show_error_details"


def test_streaming_stale_child_history_becomes_completed_with_diagnostic() -> None:
    handler = _make_handler()
    handler.ctx.channel_client_factory = MagicMock(return_value=object())
    session = MagicMock()
    session.session_id = "stale-child-history"
    session.message_count = 1
    result = PromptResult(
        stop_reason="end_turn",
        text="完整审计结果",
        tool_calls=[
            ToolCallInfo(
                id="stale-child-activity",
                title="subagent activity",
                kind="other",
                status="completed",
                subagent_source_id="child-a",
                subagent_activity="interacted",
            )
        ],
    )
    execution = PromptContinuationResult(
        result=result,
        assessment=classify_prompt_result(result),
        automatic_continuations=1,
        entered_finalization=False,
    )

    proof = MagicMock()
    proof.all_observed_children_terminal.return_value = True

    with (
        _streaming_environment(_FakeProgrammingCardSession),
        patch(
            "src.feishu.handlers.programming.AuthoritativeChildLifecycleProof",
            return_value=proof,
        ),
        patch(
            "src.feishu.handlers.programming.run_prompt_across_execution_windows",
            return_value=execution,
        ),
    ):
        handler.handle_response(
            "msg-stale-child",
            "chat-1",
            "audit",
            session,
            None,
            "/tmp",
            "/tmp",
        )

    adapter = _FakeProgrammingCardSession.last
    assert adapter.finished is True
    assert adapter.failed_text is None
    assert "不影响" in adapter.finish_kwargs["warning"]
    assert adapter.finish_kwargs["details"]
    assert adapter.finish_kwargs["detail_action"]["action"] == "show_error_details"


def test_non_streaming_pending_plan_continues_and_replies_with_success():
    handler = _make_handler()
    handler.reply_card = MagicMock()
    handler.upload_acp_image = MagicMock()
    session = _QueuedPromptSession(_pending_result(), _completed_result())

    with patch(
        "src.feishu.handlers.programming.classify_prompt_result",
        side_effect=AssertionError(
            "handler must consume continuation assessment directly"
        ),
        create=True,
    ):
        handler._handle_response_non_streaming(
            "msg-1",
            "chat-1",
            "finish the task",
            session,
            None,
            "/tmp",
        )

    assert len(session.calls) == 2
    handler.reply_card.assert_not_called()
    handler.reply_text.assert_called_once()
    assert "done" in handler.reply_text.call_args.args[1]
    handler.add_reaction.assert_called_once()


def test_admin_project_task_wraps_acp_execution_in_trusted_personal_lease():
    handler = _make_handler()
    handler.reply_card = MagicMock()
    handler.upload_acp_image = MagicMock()
    handler.settings.acp_trusted_personal_mode = True
    handler.settings.acp_trusted_personal_ack = True
    handler.settings.admin_user_ids = frozenset({"ou_owner"})
    project = MagicMock()
    project.project_id = "project-1"
    project.root_path = "/tmp/project-1"
    session = MagicMock()
    session.session_id = "trusted-session"
    session.message_count = 1
    session._force_dead = False
    session.set_trusted_personal_permissions.return_value = True

    def execute_first_window(initial_session, initial_prompt, **kwargs):
        return kwargs["execute_window"](initial_session, initial_prompt)

    with (
        patch("src.thread.get_current_sender_id", return_value="ou_owner"),
        patch(
            "src.feishu.handlers.programming.run_prompt_across_execution_windows",
            side_effect=execute_first_window,
        ),
        patch(
            "src.feishu.handlers.programming.run_prompt_with_continuation",
            return_value=_completed_execution("pushed"),
        ),
    ):
        handler._handle_response_non_streaming(
            "msg-trusted",
            "chat-1",
            "push current branch",
            session,
            project,
            "/tmp/project-1",
        )

    assert session.set_trusted_personal_permissions.call_args_list == [
        call(True),
        call(False),
    ]
    assert session._force_dead is False
    assert "pushed" in handler.reply_text.call_args.args[1]


def test_non_streaming_retries_pending_plan_then_renders_incomplete():
    handler = _make_handler()
    handler.reply_card = MagicMock()
    handler.upload_acp_image = MagicMock(return_value="img-waiting")
    image = ACPImageInfo(
        image_id="waiting-image",
        mime_type="image/png",
        data="unused-by-mocked-uploader",
        name="waiting.png",
    )
    session = _QueuedPromptSession(
        _pending_result("first partial"),
        _pending_result("second partial"),
        _pending_result("third partial"),
        _pending_result("fourth partial"),
        first_event=ACPEvent(
            event_type=ACPEventType.IMAGE_CHUNK,
            image=image,
        ),
    )

    handler._handle_response_non_streaming(
        "msg-1",
        "chat-1",
        "finish the task",
        session,
        None,
        "/tmp",
    )

    assert len(session.calls) == 4
    assert "自动续做指令" in session.calls[1][0]
    handler.upload_acp_image.assert_called_once_with(image)
    handler.reply_text.assert_not_called()
    handler.reply_card.assert_called_once()
    card = json.loads(handler.reply_card.call_args.args[1])
    markdown = "\n".join(
        str(element.get("content") or "")
        for element in card["body"]["elements"]
        if element.get("tag") == "markdown"
    )
    assert "等待用户确认" not in markdown
    assert "未完成" in markdown
    assert "仍有 1 个计划项未完成" in markdown
    handler.add_reaction.assert_not_called()


@pytest.mark.parametrize("entered_finalization", [True, False])
def test_streaming_finalization_provenance_controls_incomplete_copy(
    entered_finalization: bool,
) -> None:
    handler = _make_handler()
    handler.ctx.channel_client_factory = MagicMock(return_value=object())
    session = MagicMock()
    session.session_id = "streaming-provenance"
    session.message_count = 1

    with (
        _streaming_environment(),
        patch(
            "src.feishu.handlers.programming.run_prompt_across_execution_windows",
            return_value=_incomplete_execution(
                entered_finalization=entered_finalization
            ),
        ),
    ):
        handler.handle_response(
            "msg-finalization",
            "chat-1",
            "finish the task",
            session,
            None,
            "/tmp",
            "/tmp",
        )

    failed_text = _FakeProgrammingCardSession.last.failed_text
    assert "仍有 3 个计划项未完成" in failed_text
    assert ("执行窗口已耗尽" in failed_text) is entered_finalization


@pytest.mark.parametrize("entered_finalization", [True, False])
def test_non_streaming_finalization_provenance_controls_incomplete_copy(
    entered_finalization: bool,
) -> None:
    handler = _make_handler()
    handler.reply_card = MagicMock()
    handler.upload_acp_image = MagicMock()
    session = MagicMock()

    with patch(
        "src.feishu.handlers.programming.run_prompt_across_execution_windows",
        return_value=_incomplete_execution(
            entered_finalization=entered_finalization
        ),
    ):
        handler._handle_response_non_streaming(
            "msg-finalization",
            "chat-1",
            "finish the task",
            session,
            None,
            "/tmp",
        )

    response = handler.reply_text.call_args.args[1]
    assert "仍有 3 个计划项未完成" in response
    assert ("执行窗口已耗尽" in response) is entered_finalization


def test_timeout_incomplete_resumes_and_finishes_in_second_window() -> None:
    handler = _make_handler()
    handler.ctx.channel_client_factory = MagicMock(return_value=object())
    handler.settings.programming_max_execution_windows = 4
    manager = MagicMock()
    handler.ctx.claude_manager = manager
    first_session = MagicMock()
    first_session.session_id = "provider-session-1"
    first_session.message_count = 1
    resumed_session = MagicMock()
    resumed_session.session_id = "provider-session-1"
    resumed_session.message_count = 2
    manager.resume_retired_session.return_value = resumed_session

    with (
        _streaming_environment(),
        patch(
            "src.feishu.handlers.programming.run_prompt_with_continuation",
            side_effect=[
                _incomplete_execution(entered_finalization=True),
                _completed_execution("finished after rollover"),
            ],
        ),
    ):
        handler.handle_response(
            "msg-window-rollover",
            "chat-1",
            "finish the task",
            first_session,
            None,
            "/tmp",
            "/tmp",
        )

    adapter = _FakeProgrammingCardSession.last
    assert adapter.finished is True
    assert adapter.failed_text is None
    assert adapter.continuation_boundaries == 1
    manager.resume_retired_session.assert_called_once()
    resume_kwargs = manager.resume_retired_session.call_args.kwargs
    assert resume_kwargs["session_id"] == "provider-session-1"


def test_window_limit_uses_total_window_terminal_copy() -> None:
    handler = _make_handler()
    handler.ctx.channel_client_factory = MagicMock(return_value=object())
    session = MagicMock()
    session.session_id = "provider-session-limit"
    session.message_count = 4

    with (
        _streaming_environment(),
        patch(
            "src.feishu.handlers.programming.run_prompt_across_execution_windows",
            return_value=_incomplete_execution(
                entered_finalization=True,
                execution_windows=4,
                window_limit_reached=True,
            ),
        ),
    ):
        handler.handle_response(
            "msg-window-limit",
            "chat-1",
            "finish the task",
            session,
            None,
            "/tmp",
            "/tmp",
        )

    failed_text = _FakeProgrammingCardSession.last.failed_text
    assert "已自动执行 4 个窗口" in failed_text
    assert "达到配置的安全上限" in failed_text
    assert "单个执行窗口已耗尽" not in failed_text


@pytest.mark.parametrize("goal_status", ["paused", "blocked"])
def test_provider_goal_fails_after_automatic_recovery_is_exhausted(
    goal_status: str,
) -> None:
    handler = _make_handler()
    handler.ctx.channel_client_factory = MagicMock(return_value=object())
    session = MagicMock()
    session.session_id = f"provider-goal-{goal_status}"
    session.message_count = 1

    with (
        _streaming_environment(),
        patch(
            "src.feishu.handlers.programming.run_prompt_with_continuation",
            return_value=_incomplete_execution(
                entered_finalization=False,
                goal_status=goal_status,
            ),
        ),
    ):
        handler.handle_response(
            "msg-provider-goal",
            "chat-1",
            "finish the task",
            session,
            None,
            "/tmp",
            "/tmp",
        )

    adapter = _FakeProgrammingCardSession.last
    assert adapter.finished is False
    assert adapter.failed_text is not None
    assert adapter.continuation_boundaries == 0


@pytest.mark.parametrize("goal_status", ["paused", "blocked"])
def test_provider_goal_after_finalization_fails_with_timeout_truth(
    goal_status: str,
) -> None:
    handler = _make_handler()
    handler.ctx.channel_client_factory = MagicMock(return_value=object())
    session = MagicMock()
    session.session_id = f"retired-provider-goal-{goal_status}"
    session.message_count = 1

    with (
        _streaming_environment(),
        patch(
            "src.feishu.handlers.programming.run_prompt_across_execution_windows",
            return_value=_incomplete_execution(
                entered_finalization=True,
                goal_status=goal_status,
            ),
        ),
    ):
        handler.handle_response(
            "msg-provider-goal-finalization",
            "chat-1",
            "finish the task",
            session,
            None,
            "/tmp",
            "/tmp",
        )

    adapter = _FakeProgrammingCardSession.last
    assert adapter.finished is False
    assert adapter.continuation_boundaries == 0
    assert "执行窗口已耗尽" in adapter.failed_text
    assert "仍有 3 个计划项未完成" in adapter.failed_text
    if goal_status == "paused":
        assert "Codex Goal 已确认暂停" in adapter.failed_text
    else:
        assert "Codex Goal 已确认暂停" not in adapter.failed_text


def test_programming_handle_response_builds_channel_card_client():
    handler = _make_handler()
    handler.settings.feishu_cot_enabled = False
    channel = object()
    handler.ctx.channel_client_factory = MagicMock(return_value=channel)
    session = MagicMock()
    session.session_id = "sid-1"
    session.message_count = 1
    session.send_prompt.side_effect = RuntimeError("stop after transport setup")
    channel_adapter = MagicMock(name="channel_card_adapter")
    delivery_factory = MagicMock(return_value=MagicMock())

    with (
        patch(
            "src.card.delivery.factory.create_card_delivery",
            delivery_factory,
        ),
        patch(
            "src.card.delivery.channel_client.LarkChannelCardAPIClient",
            return_value=channel_adapter,
        ) as channel_client_cls,
        patch(
            "src.card.delivery.feishu_client.FeishuCardAPIClient",
            return_value=MagicMock(name="legacy_card_adapter"),
        ),
        patch("src.feishu.cot.FeishuCOTAPIClient") as cot_api_cls,
        patch("src.feishu.cot.FeishuCOTSession") as cot_session_cls,
        patch("src.card.session.CardSession", return_value=MagicMock()),
        patch("src.card.session.factory.CardSessionFactory", return_value=MagicMock()),
        patch("src.card.programming_adapter.ProgrammingCardSession", _FakeProgrammingCardSession),
    ):
        handler.handle_response(
            "msg-1",
            "chat-1",
            "hello",
            session,
            None,
            "/tmp",
            "/tmp",
        )

    channel_client_cls.assert_called_once()
    assert channel_client_cls.call_args.args == (channel,)
    assert channel_client_cls.call_args.kwargs["preallocate_cards"] is True
    delivery_factory.assert_called_once()
    assert delivery_factory.call_args.args == (channel_adapter,)
    assert callable(delivery_factory.call_args.kwargs["payload_transform"])
    assert callable(delivery_factory.call_args.kwargs["trust_revision_provider"])
    cot_api_cls.assert_not_called()
    cot_session_cls.assert_not_called()


def test_programming_starts_cot_only_after_card_is_visible_and_links_both_messages():
    handler = _make_handler()
    handler.settings.feishu_cot_enabled = True
    handler.settings.default_reply_mode = "thread"
    handler.ctx.channel_client_factory = MagicMock(return_value=object())
    session = _QueuedPromptSession(_completed_result())
    call_order: list[str] = []
    process_sink = _FakeProcessSink(call_order=call_order)

    class _LinkedProgrammingCardSession(_FakeProgrammingCardSession):
        last = None

        def wait_until_visible(self, _timeout):
            call_order.append("card_visible")
            return True

        def get_message_id(self):
            return "om-card-summary"

    cot_api = MagicMock(name="cot_api")
    with (
        _streaming_environment(_LinkedProgrammingCardSession),
        patch(
            "src.feishu.cot.FeishuCOTAPIClient",
            return_value=cot_api,
        ) as cot_api_cls,
        patch(
            "src.feishu.cot.FeishuCOTSession",
            return_value=process_sink,
        ) as cot_session_cls,
    ):
        handler.handle_response(
            "msg-cot-success",
            "chat-1",
            "injected task text",
            session,
            None,
            "/tmp",
            "/tmp",
            _finalization_task_text="raw user task",
        )

    adapter = _LinkedProgrammingCardSession.last
    assert call_order[:2] == ["card_visible", "cot_start"]
    assert adapter.process_sink is process_sink
    assert adapter.process_sink_activation_calls == 1
    assert process_sink.start_calls == 1
    assert adapter.finished is True
    assert len(session.calls) == 1
    cot_api_cls.assert_called_once()
    assert cot_api_cls.call_args.args == (handler.ctx.api_client_factory.return_value,)
    cot_session_cls.assert_called_once_with(
        cot_api,
        chat_id="chat-1",
        origin_message_id="msg-cot-success",
        reply_in_thread=True,
        input_text="raw user task",
        request_timeout=2.0,
    )
    handler.ctx.message_linker.link_reply.assert_any_call(
        "msg-cot-success",
        "om-card-summary",
    )
    handler.ctx.message_linker.link_reply.assert_any_call(
        "msg-cot-success",
        "om-cot-process",
    )


def test_programming_cot_activation_failure_keeps_streaming_card_execution():
    handler = _make_handler()
    handler.settings.feishu_cot_enabled = True
    handler.ctx.channel_client_factory = MagicMock(return_value=object())
    handler._handle_response_non_streaming = MagicMock()
    session = _QueuedPromptSession(_completed_result())
    call_order: list[str] = []
    process_sink = _FakeProcessSink(
        message_id=None,
        start_error=RuntimeError("COT unavailable"),
        call_order=call_order,
    )

    class _LinkedProgrammingCardSession(_FakeProgrammingCardSession):
        last = None

        def wait_until_visible(self, _timeout):
            call_order.append("card_visible")
            return True

        def get_message_id(self):
            return "om-card-fallback"

    with (
        _streaming_environment(_LinkedProgrammingCardSession),
        patch("src.feishu.cot.FeishuCOTAPIClient", return_value=MagicMock()),
        patch("src.feishu.cot.FeishuCOTSession", return_value=process_sink),
    ):
        handler.handle_response(
            "msg-cot-failure",
            "chat-1",
            "finish",
            session,
            None,
            "/tmp",
            "/tmp",
        )

    adapter = _LinkedProgrammingCardSession.last
    assert call_order == ["card_visible", "cot_start"]
    assert adapter.process_sink_activation_calls == 1
    assert process_sink.start_calls == 1
    assert process_sink.abort_calls == 1
    assert adapter.finished is True
    assert len(session.calls) == 1
    handler._handle_response_non_streaming.assert_not_called()
    handler.ctx.message_linker.link_reply.assert_called_once_with(
        "msg-cot-failure",
        "om-card-fallback",
    )


def test_programming_aborts_active_cot_card_session_when_execution_entry_raises():
    handler = _make_handler()
    handler.settings.feishu_cot_enabled = True
    handler.ctx.channel_client_factory = MagicMock(return_value=object())
    handler._handle_response_non_streaming = MagicMock()
    failure = RuntimeError("execution entry failed")
    handler._execute_programming_response = MagicMock(side_effect=failure)
    process_sink = _FakeProcessSink()
    delivery = MagicMock()
    delivery.shutdown.return_value = True

    class _AbortTrackingProgrammingCardSession(_FakeProgrammingCardSession):
        last = None

        def __init__(self, *_args, **kwargs):
            super().__init__(*_args, **kwargs)
            self.aborted = False
            type(self).last = self

        def get_message_id(self):
            return "om-card-before-execution"

        def abort(self):
            self.aborted = True
            if self.process_sink is not None and self.process_sink.started:
                self.process_sink.abort()

    with (
        _streaming_environment(
            _AbortTrackingProgrammingCardSession,
            delivery=delivery,
        ),
        patch("src.feishu.cot.FeishuCOTAPIClient", return_value=MagicMock()),
        patch("src.feishu.cot.FeishuCOTSession", return_value=process_sink),
        pytest.raises(RuntimeError, match="execution entry failed") as exc_info,
    ):
        handler.handle_response(
            "msg-cot-execution-failure",
            "chat-1",
            "finish",
            MagicMock(),
            None,
            "/tmp",
            "/tmp",
        )

    adapter = _AbortTrackingProgrammingCardSession.last
    assert exc_info.value is failure
    assert adapter.process_sink_activation_calls == 1
    assert process_sink.start_calls == 1
    assert adapter.aborted is True
    assert process_sink.abort_calls == 1
    handler._handle_response_non_streaming.assert_not_called()
    delivery.shutdown.assert_called_once()


def test_programming_handle_response_falls_back_when_channel_is_unavailable():
    handler = _make_handler()
    handler.ctx.channel_client_factory = None
    handler._handle_response_non_streaming = MagicMock()
    session = MagicMock()

    handler.handle_response(
        "msg-1",
        "chat-1",
        "hello",
        session,
        None,
        "/tmp",
        "/tmp",
    )

    handler._handle_response_non_streaming.assert_called_once_with(
        "msg-1",
        "chat-1",
        "hello",
        session,
        None,
        "/tmp",
        _repo_lock_mgr=None,
        _root_path=None,
        _finalization_task_text=None,
    )
    handler.reply_text.assert_called_once()
    assert "任务仍在执行" in handler.reply_text.call_args.args[1]
    session.send_prompt.assert_not_called()


def test_programming_falls_back_when_initial_async_channel_card_is_not_visible():
    handler = _make_handler()
    handler.settings.feishu_cot_enabled = True
    handler.ctx.channel_client_factory = MagicMock(return_value=object())
    handler._handle_response_non_streaming = MagicMock()
    session = MagicMock()
    process_sink = _FakeProcessSink()

    class _InvisibleProgrammingCardSession(_FakeProgrammingCardSession):
        last = None

        def __init__(self, *_args, **_kwargs):
            super().__init__(*_args, **_kwargs)
            self.aborted = False
            type(self).last = self

        def wait_until_visible(self, _timeout):
            return False

        def abort(self):
            self.aborted = True

    with (
        patch("src.card.delivery.factory.create_card_delivery", return_value=MagicMock()),
        patch(
            "src.card.delivery.channel_client.LarkChannelCardAPIClient",
            return_value=MagicMock(),
        ),
        patch("src.feishu.cot.FeishuCOTAPIClient", return_value=MagicMock()),
        patch(
            "src.feishu.cot.FeishuCOTSession",
            return_value=process_sink,
        ) as cot_session_cls,
        patch("src.card.session.CardSession", return_value=MagicMock()),
        patch("src.card.session.factory.CardSessionFactory", return_value=MagicMock()),
        patch(
            "src.card.programming_adapter.ProgrammingCardSession",
            _InvisibleProgrammingCardSession,
        ),
    ):
        handler.handle_response(
            "msg-1",
            "chat-1",
            "hello",
            session,
            None,
            "/tmp",
            "/tmp",
        )

    assert _InvisibleProgrammingCardSession.last.aborted is True
    assert _InvisibleProgrammingCardSession.last.process_sink is process_sink
    assert _InvisibleProgrammingCardSession.last.process_sink_activation_calls == 0
    assert process_sink.start_calls == 0
    cot_session_cls.assert_called_once()
    handler._handle_response_non_streaming.assert_called_once()
    session.send_prompt.assert_not_called()


def test_programming_terminal_delivery_failure_aborts_retry_and_replies_text():
    handler = _make_handler()
    handler.ctx.channel_client_factory = MagicMock(return_value=object())
    handler._update_snapshot_on_project = MagicMock()
    project = MagicMock()
    project.project_name = "ghostAp"
    project.root_path = "/tmp/ghostAp"
    project.project_id = "project-1"
    session = MagicMock()
    session.session_id = "sid-1"
    session.message_count = 1
    session.send_prompt.side_effect = RuntimeError("terminal delivery test")

    class _FailedTerminalProgrammingCardSession(_FakeProgrammingCardSession):
        last = None

        def __init__(self, *_args, **_kwargs):
            super().__init__()
            self.aborted = False
            self.final_text = "early-main-before-exception"
            type(self).last = self

        def terminal_delivery_succeeded(self):
            return False

        def abort(self):
            self.aborted = True

    with (
        patch("src.card.delivery.factory.create_card_delivery", return_value=MagicMock()),
        patch(
            "src.card.delivery.channel_client.LarkChannelCardAPIClient",
            return_value=MagicMock(),
        ),
        patch("src.card.session.CardSession", return_value=MagicMock()),
        patch("src.card.session.factory.CardSessionFactory", return_value=MagicMock()),
        patch(
            "src.card.programming_adapter.ProgrammingCardSession",
            _FailedTerminalProgrammingCardSession,
        ),
    ):
        handler.handle_response(
            "msg-1",
            "chat-1",
            "hello",
            session,
            project,
            "/tmp",
            "/tmp",
        )

    assert _FailedTerminalProgrammingCardSession.last.aborted is True
    handler.reply_text.assert_called_once()
    assert handler.reply_text.call_args.args[0] == "msg-1"
    assert "early-main-before-exception" in handler.reply_text.call_args.args[1]
    assert "terminal delivery test" in handler.reply_text.call_args.args[1]
    assistant_conversation = next(
        call
        for call in project.add_conversation.call_args_list
        if call.args[0] == "assistant"
    )
    assert "early-main-before-exception" in assistant_conversation.args[1]
    assert "terminal delivery test" in assistant_conversation.args[1]


def test_programming_timeout_fallback_keeps_earlier_main_transcript():
    handler = _make_handler()
    handler.ctx.channel_client_factory = MagicMock(return_value=object())
    handler._retire_finalization_session = MagicMock()
    session = MagicMock()
    session.session_id = "sid-timeout"
    session.message_count = 1
    session.send_prompt.side_effect = TimeoutError("stream deadline")

    class _TranscriptTimeoutProgrammingCardSession(_FakeProgrammingCardSession):
        last = None

        def __init__(self, *_args, **_kwargs):
            super().__init__()
            self.aborted = False
            self.final_text = "early-main-before-timeout"
            type(self).last = self

        def terminal_delivery_succeeded(self):
            return False

        def abort(self):
            self.aborted = True

    with _streaming_environment(_TranscriptTimeoutProgrammingCardSession):
        handler.handle_response(
            "msg-timeout",
            "chat-1",
            "hello",
            session,
            None,
            "/tmp",
            "/tmp",
        )

    assert _TranscriptTimeoutProgrammingCardSession.last.aborted is True
    handler.reply_text.assert_called_once()
    fallback = handler.reply_text.call_args.args[1]
    assert "early-main-before-timeout" in fallback
    assert "stream deadline" in fallback


@pytest.mark.parametrize(
    "reconciliation_result",
    [
        TimeoutError("reconciliation deadline"),
        PromptResult(stop_reason="timeout", text="reconciliation timed out"),
    ],
)
def test_reconciliation_timeout_retirement_cannot_repersist_snapshot(
    reconciliation_result: PromptResult | BaseException,
) -> None:
    handler = _make_handler()
    handler.ctx.channel_client_factory = MagicMock(return_value=object())
    handler._retire_finalization_session = MagicMock()
    handler._clear_snapshot_for_session = MagicMock()
    handler._update_snapshot_on_project = MagicMock()
    project = MagicMock()
    project.project_name = "ghostAp"
    project.root_path = "/tmp/ghostAp"
    project.project_id = "project-1"
    child_running = PromptResult(
        stop_reason="end_turn",
        tool_calls=[
            ToolCallInfo(
                id="review-running",
                title="list_agents",
                kind="other",
                status="completed",
                subagent_states=(
                    {"source_id": "reviewer", "status": "running"},
                ),
            )
        ],
    )
    session = _QueuedPromptSession(child_running, reconciliation_result)

    with _streaming_environment():
        handler.handle_response(
            "msg-reconciliation-timeout",
            "chat-1",
            "finish the task",
            session,
            project,
            "/tmp",
            "/tmp",
        )

    handler._retire_finalization_session.assert_called_once()
    handler._update_snapshot_on_project.assert_not_called()
    assert handler._clear_snapshot_for_session.call_count >= 2
