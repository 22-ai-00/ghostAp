from __future__ import annotations

import json
import logging
from contextlib import ExitStack, contextmanager, nullcontext
from unittest.mock import MagicMock, patch

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
        type(self).last = self

    def start(self):
        return None

    def get_message_id(self):
        return None

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

class _QueuedPromptSession:
    def __init__(
        self,
        *results: PromptResult,
        first_event: ACPEvent | None = None,
    ):
        self._results = list(results)
        self._first_event = first_event
        self.calls: list[tuple[str, object, float | int | None]] = []
        self._force_dead = False
        self.session_id = "queued-session"
        self.message_count = 1

    def send_prompt(self, text, on_event=None, timeout=None):
        self.calls.append((text, on_event, timeout))
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
    ctx.settings.repo_lock_hard_timeout = 3600
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
    assert call_order == ["shutdown", "fallback"]


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
    session.send_prompt.assert_not_called()


def test_programming_falls_back_when_initial_async_channel_card_is_not_visible():
    handler = _make_handler()
    handler.ctx.channel_client_factory = MagicMock(return_value=object())
    handler._handle_response_non_streaming = MagicMock()
    session = MagicMock()

    class _InvisibleProgrammingCardSession(_FakeProgrammingCardSession):
        last = None

        def __init__(self, *_args, **_kwargs):
            super().__init__()
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
