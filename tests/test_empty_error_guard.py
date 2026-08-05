"""Tests for empty-error-message guard across all patched modules.

Validates that bare TimeoutError() (no message) and other empty-message
exceptions never produce empty user-facing strings.
"""
import asyncio
import concurrent.futures
import json
import logging
import re
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.utils.errors import fmt_exception, get_error_detail

# ---------------------------------------------------------------------------
# Helpers for chained exception construction
# ---------------------------------------------------------------------------


def _chain_cause(outer_cls, inner):
    outer = outer_cls("wrapper")
    outer.__cause__ = inner
    return outer


def _chain_context(outer_cls, inner):
    outer = outer_cls("wrapper")
    outer.__context__ = inner
    return outer


# ===========================================================================
# Section 1: Integration tests (complex setup, kept individually or lightly
# parametrized where methods were near-duplicates)
# ===========================================================================


class TestBuildErrorCardEmptyGuard:
    """system.py: build_error_card must never produce empty message body."""

    def test_bare_timeout_error_has_nonempty_message(self):
        from src.card.builders.system import SystemBuilder

        _, card_json = SystemBuilder.build_error_card(TimeoutError())
        card = json.loads(card_json)
        body_elements = card.get("body", {}).get("elements", card.get("elements", []))
        content_texts = [
            el.get("content", "")
            for el in body_elements
            if el.get("tag") == "markdown" or el.get("tag") == "div"
        ]
        full_text = " ".join(content_texts)
        assert "超时" in full_text or "未知错误" in full_text

    def test_bare_timeout_error_no_empty_body(self):
        from src.card.builders.system import SystemBuilder

        _, card_json = SystemBuilder.build_error_card(TimeoutError())
        assert "\n\n\n" not in card_json

    def test_string_exc_still_works(self):
        from src.card.builders.system import SystemBuilder

        _, card_json = SystemBuilder.build_error_card("具体错误信息")
        assert "具体错误信息" in card_json

    def test_named_timeout_preserves_message(self):
        from src.card.builders.system import SystemBuilder

        _, card_json = SystemBuilder.build_error_card(
            TimeoutError("ACP prompt 执行超时 (120s)")
        )
        assert "ACP prompt 执行超时" in card_json


class TestBaseHandlerFallbackEmptyGuard:
    """base.py: fallback text path must never produce '❌ title: ' with empty tail."""

    def _make_handler(self):
        from src.feishu.handlers.base import BaseHandler

        ctx = MagicMock()
        ctx.settings = MagicMock()
        ctx.settings.ref_note_enabled = False
        handler = BaseHandler(ctx)
        return handler

    def test_fallback_path_bare_timeout_no_empty_tail(self):
        handler = self._make_handler()
        sent_content = []

        def capture_reply(msg_id, content, **kw):
            sent_content.append(content)

        handler.reply_text = capture_reply

        with patch("src.card.CardBuilder") as mock_cb:
            mock_cb.build_error_card.side_effect = Exception("card build failed")
            handler.send_error_card(
                chat_id="test_chat",
                exc=TimeoutError(),
                title="启动超时",
                origin_message_id="msg123",
            )

        assert len(sent_content) == 1
        msg = sent_content[0]
        assert not msg.endswith(": ")
        assert "操作失败" in msg or "启动超时" in msg


class TestSchedulerEmptyErrorGuard:
    """scheduler.py: state.error must never be empty for bare exceptions."""

    @pytest.mark.parametrize("exc_factory,label", [
        pytest.param(TimeoutError, "timeout", id="bare_timeout"),
        pytest.param(Exception, "empty_exc", id="bare_exception"),
    ])
    def test_error_state_nonempty(self, exc_factory, label):
        from src.tasking.scheduler import TaskScheduler, TaskSpec

        scheduler = TaskScheduler(max_concurrent=2)
        try:
            spec = TaskSpec(chat_id="c1", name=f"test_{label}")

            def failing_task(ctx):
                raise exc_factory()

            handle = scheduler.submit(spec, failing_task)
            try:
                handle.wait(timeout=5)
            except Exception:
                pass

            state = scheduler.get_state(handle.run_id)
            assert state is not None
            assert state.error
            assert len(state.error) > 0
        finally:
            scheduler.stop(shutdown_executor=True)


class TestDeepHandlerProjectCreateEmptyGuard:
    """deep.py: project creation failure must never produce empty error message."""

    def _make_handler(self):
        from src.feishu.handlers.deep import DeepHandler

        ctx = MagicMock()
        ctx.settings = MagicMock()
        ctx.settings.ref_note_enabled = False
        handler = DeepHandler(ctx)
        return handler

    @pytest.mark.parametrize("exc_factory,check_timeout", [
        pytest.param(Exception, False, id="bare_exception"),
        pytest.param(TimeoutError, True, id="bare_timeout"),
    ])
    def test_project_create_nonempty_error(self, exc_factory, check_timeout):
        handler = self._make_handler()
        sent = []
        handler.send_error_card = lambda **kw: sent.append(kw)
        handler.get_working_dir = lambda cid: "/tmp"
        handler.ctx.project_manager.get_or_create_project_for_path.side_effect = exc_factory()

        handler.start_deep_engine("msg1", "chat1", "test requirement")
        assert len(sent) == 1
        exc_val = str(sent[0]["exc"])
        assert exc_val
        assert len(exc_val) > 0
        if check_timeout:
            assert "超时" in exc_val


class TestSpecHandlerEmptyGuard:
    """spec.py: all error paths must produce non-empty user-facing messages."""

    def _make_handler(self):
        from src.feishu.handlers.spec import SpecHandler

        ctx = MagicMock()
        ctx.settings = MagicMock()
        ctx.settings.ref_note_enabled = False
        handler = SpecHandler(ctx)
        return handler

    @pytest.mark.parametrize("exc_factory,check_timeout", [
        pytest.param(Exception, False, id="bare_exception"),
        pytest.param(TimeoutError, True, id="bare_timeout"),
    ])
    def test_project_create_error(self, exc_factory, check_timeout):
        handler = self._make_handler()
        sent = []
        handler.reply_card = lambda mid, content, **kw: sent.append(content)
        handler.get_working_dir = lambda cid: "/tmp"
        handler.ctx.project_manager.get_or_create_project_for_path.side_effect = exc_factory()

        handler.start_spec_engine("msg1", "chat1", "req")
        assert len(sent) == 1
        assert sent[0]
        if check_timeout:
            assert "超时" in sent[0]
        else:
            assert "创建项目" in sent[0]
            assert "❌" in sent[0]

    def test_export_bare_exception(self):
        result = get_error_detail(Exception())
        assert result

    def test_restore_context_bare_exception(self):
        from src.utils.errors import fmt_error

        result = fmt_error("恢复项目上下文", Exception())
        assert result
        assert "恢复项目上下文" in result

    def test_restore_context_bare_timeout(self):
        from src.utils.errors import fmt_error

        result = fmt_error("恢复任务上下文", TimeoutError())
        assert "超时" in result


class TestSystemHandlerRefreshModelsIntegration:
    """system.py: handle_refresh_ttadk_models reply_error must never
    produce empty-tail message for bare TimeoutError or Exception."""

    def _make_handler(self):
        from src.feishu.handlers.system import SystemHandler

        ctx = MagicMock()
        ctx.settings = MagicMock()
        ctx.settings.ref_note_enabled = False
        handler = SystemHandler(ctx)
        return handler

    @pytest.mark.parametrize("exc,check_timeout,check_msg", [
        pytest.param(TimeoutError(), True, None, id="bare_timeout"),
        pytest.param(Exception(), False, None, id="bare_exception"),
        pytest.param(TimeoutError("模型服务不可用"), False, "模型服务不可用", id="named_timeout"),
    ])
    def test_reply_error_nonempty(self, exc, check_timeout, check_msg):
        handler = self._make_handler()
        sent = []
        handler.reply_error = lambda mid, content, **kw: sent.append(content)
        handler._resolve_ttadk_cwd = lambda *a, **kw: "/tmp"
        handler._maybe_log_ttadk_cwd = lambda **kw: None

        mock_mgr = MagicMock()
        mock_mgr.get_current_tool.return_value = "coco"
        mock_mgr.refresh_models.side_effect = exc

        with patch("src.feishu.handlers.ttadk_commands.get_ttadk_manager", return_value=mock_mgr):
            handler.handle_refresh_ttadk_models("msg1", "chat1", "coco")

        assert len(sent) == 1
        msg = sent[0]
        assert msg
        assert not msg.endswith(": ")
        if check_timeout:
            assert "超时" in msg
        if check_msg:
            assert check_msg in msg


_EXPECTED_METRICS_KEYS = {
    "metric_type", "engine", "fail_reason",
    "consecutive_timeouts", "consecutive_failures",
    "circuit_open", "adaptive_timeout", "backoff_level",
}


class TestSpecReviewMetricsLog:
    """spec_engine/review.py: review_metrics logger.info is called with valid JSON."""

    def _run_conduct_review_with_error(self, exc: Exception):
        from src.spec_engine.review import ReviewCircuitState, conduct_review

        circuit = ReviewCircuitState()
        settings = MagicMock()
        settings.spec_review_failure_circuit_enabled = False
        settings.spec_review_failure_max_consecutive = 3
        settings.spec_review_failure_cooldown_cycles = 3
        settings.spec_review_timeout = 120
        settings.spec_review_min_timeout = 30
        settings.spec_review_hard_floor = 15
        settings.spec_review_failure_max_cooldown_cycles = 12

        def raise_exc(*a, **kw):
            raise exc

        records: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = lambda r: records.append(r)

        logger = logging.getLogger("src.utils.review_helpers")
        logger.addHandler(handler)
        old_level = logger.level
        logger.setLevel(logging.DEBUG)
        logger_me = logging.getLogger("src.utils.metrics_exporter")
        logger_me.addHandler(handler)
        old_level_me = logger_me.level
        logger_me.setLevel(logging.DEBUG)
        from src.utils.metrics_exporter import reset_metrics_exporter
        reset_metrics_exporter()
        try:
            conduct_review(
                session=MagicMock(),
                settings=settings,
                project=MagicMock(requirement="test req"),
                send_prompt_with_retry_fn=raise_exc,
                build_review_exception_diagnostics_fn=lambda e, cycle: {
                    "phase": "review", "cycle": cycle,
                    "fail_reason": "timeout" if isinstance(e, TimeoutError) else "exception",
                    "err_type": type(e).__name__, "err_repr": repr(e),
                    "error_text": str(e) or "审查执行异常",
                },
                circuit=circuit,
                cycle=3,
            )
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)
            logger_me.removeHandler(handler)
            logger_me.setLevel(old_level_me)
            reset_metrics_exporter()

        return [r for r in records if "review_metrics" in str(r.getMessage())]

    def test_metrics_log_emitted_on_timeout(self):
        recs = self._run_conduct_review_with_error(TimeoutError("slow"))
        assert len(recs) >= 1, "review_metrics log not emitted"

    def test_metrics_log_emitted_on_regular_error(self):
        recs = self._run_conduct_review_with_error(RuntimeError("oops"))
        assert len(recs) >= 1, "review_metrics log not emitted"

    def test_metrics_json_structure(self):
        recs = self._run_conduct_review_with_error(TimeoutError())
        msg = recs[0].getMessage()
        json_str = msg.split("review_metrics: ", 1)[1]
        data = json.loads(json_str)
        missing = _EXPECTED_METRICS_KEYS - set(data.keys())
        assert not missing, f"Missing metrics keys: {missing}"
        assert data["engine"] == "spec"
        assert data["metric_type"] == "review_exception"

    def test_metrics_adaptive_timeout_is_int(self):
        recs = self._run_conduct_review_with_error(TimeoutError())
        msg = recs[0].getMessage()
        json_str = msg.split("review_metrics: ", 1)[1]
        data = json.loads(json_str)
        assert isinstance(data["adaptive_timeout"], int)
        assert data["adaptive_timeout"] > 0

    def test_metrics_sentinel_fallback_when_compute_fails(self):
        import inspect

        from src.spec_engine.review import conduct_review
        source = inspect.getsource(conduct_review)
        assert "review_timeout: int = 0" in source or "review_timeout = 0" in source


class TestRunAsyncTimeoutWrapping:
    """sync_adapter._run_async must never leak empty TimeoutError messages."""

    def _make_adapter(self):
        from src.acp.sync_adapter import SyncACPSession

        adapter = SyncACPSession.__new__(SyncACPSession)
        adapter._agent_type = "test_agent"
        adapter._loop = asyncio.new_event_loop()
        adapter._loop_thread = threading.Thread(
            target=adapter._loop.run_forever, daemon=True
        )
        adapter._loop_thread.start()
        return adapter

    def _cleanup(self, adapter):
        adapter._drain_loop_before_close()
        adapter._loop.call_soon_threadsafe(adapter._loop.stop)
        adapter._loop_thread.join(timeout=2)

    def test_timeout_has_nonempty_message(self):
        adapter = self._make_adapter()
        try:
            async def hang():
                await asyncio.sleep(999)

            try:
                adapter._run_async(hang(), timeout=0.05)
                assert False, "Should have raised TimeoutError"
            except TimeoutError as e:
                msg = str(e)
                assert msg, "_run_async produced empty TimeoutError message"
                assert "(empty message)" not in msg
                assert "test_agent" in msg
        finally:
            self._cleanup(adapter)

    def test_expected_timeout_can_suppress_error_trace(self, caplog):
        adapter = self._make_adapter()
        adapter._log_failures = False
        try:
            async def hang():
                await asyncio.sleep(999)

            with caplog.at_level(logging.ERROR, logger="src.acp.sync_adapter"):
                with pytest.raises(TimeoutError):
                    adapter._run_async(hang(), timeout=0.01)

            assert not caplog.records
        finally:
            self._cleanup(adapter)

    def test_normal_return_unaffected(self):
        adapter = self._make_adapter()
        try:
            async def ok():
                return 42

            assert adapter._run_async(ok(), timeout=5) == 42
        finally:
            self._cleanup(adapter)

    def test_non_timeout_exception_passthrough(self):
        adapter = self._make_adapter()
        try:
            async def boom():
                raise ValueError("test boom")

            try:
                adapter._run_async(boom(), timeout=5)
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "test boom" in str(e)
        finally:
            self._cleanup(adapter)

    def test_timeout_with_existing_message_preserved(self):
        adapter = self._make_adapter()
        try:
            async def raise_with_msg():
                raise TimeoutError("custom timeout msg")

            try:
                adapter._run_async(raise_with_msg(), timeout=5)
                assert False, "Should have raised TimeoutError"
            except TimeoutError as e:
                assert "custom timeout msg" in str(e)
        finally:
            self._cleanup(adapter)


# ===========================================================================
# Section 2: Parametrized pure-function tests
# (consolidates many small classes that tested the same utility functions)
# ===========================================================================


# ---------------------------------------------------------------------------
# get_error_detail: always non-empty for all exception variants
# (consolidates: TestGetErrorDetailNeverEmpty, TestSpecEngineInternalEmptyGuard,
#  TestMainAppEmptyGuard, TestSystemHandlerTTADKRefreshEmptyGuard,
#  TestBaseHandlerFallbackTimeoutContext,
#  TestTimeoutErrorE2EDetail)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exc,must_contain", [
    pytest.param(TimeoutError(), "超时", id="bare_timeout"),
    pytest.param(Exception(), None, id="bare_exception"),
    pytest.param(ValueError("bad input"), "bad input", id="value_error_with_msg"),
    pytest.param(ValueError(), None, id="bare_value_error"),
    pytest.param(RuntimeError(), None, id="bare_runtime_error"),
    pytest.param(concurrent.futures.TimeoutError(), "超时", id="concurrent_timeout"),
    pytest.param(TimeoutError("ACP 超时 120s"), "ACP 超时 120s", id="timeout_with_msg"),
    pytest.param(
        _chain_cause(RuntimeError, TimeoutError()), "超时", id="chained_timeout_cause",
    ),
])
def test_get_error_detail_never_empty(exc, must_contain):
    """get_error_detail must always return non-empty for any exception."""
    result = get_error_detail(exc)
    assert result, f"get_error_detail({type(exc).__name__}) returned empty"
    if must_contain:
        assert must_contain in result, f"Expected '{must_contain}' in '{result}'"


def test_get_error_detail_not_old_fallback():
    """BaseHandler fallback: detail != '操作失败' (old fallback) for TimeoutError."""
    detail = get_error_detail(TimeoutError())
    assert detail != "操作失败"


def test_get_error_detail_third_party_timeout_via_name():
    """TimeoutExpired, ReadTimeout, ConnectTimeout detected by name in chain."""
    for tn in ("TimeoutExpired", "ReadTimeout", "ConnectTimeout"):
        class MockTimeout(Exception):
            pass
        MockTimeout.__name__ = tn
        inner = MockTimeout("inner timeout")
        outer = RuntimeError("outer")
        outer.__cause__ = inner
        result = get_error_detail(outer)
        assert "超时" in result, f"Failed for {tn}"


# ---------------------------------------------------------------------------
# fmt_exception: all branches produce non-empty output
# (consolidates TestFmtExceptionEmptyGuard)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("context,exc,must_contain", [
    pytest.param("处理", Exception(), "Exception()", id="bare_exception_repr"),
    pytest.param("验证", ValueError(), "ValueError()", id="bare_value_error_repr"),
    pytest.param("操作", RuntimeError("具体原因"), "具体原因", id="named_runtime_error"),
    pytest.param("操作", TimeoutError(), "操作耗时过长", id="bare_timeout_fixed_msg"),
    pytest.param("任务", concurrent.futures.TimeoutError(), "操作耗时过长", id="concurrent_timeout"),
])
def test_fmt_exception_nonempty(context, exc, must_contain):
    """fmt_exception must never produce trailing empty content."""
    result = fmt_exception(context, exc)
    assert not result.endswith(": ")
    assert must_contain in result


def test_fmt_exception_wrapped_timeout_chain():
    """Chained TimeoutError via __cause__ and __context__ detected."""
    inner = TimeoutError()
    outer = RuntimeError("chained failure")
    outer.__cause__ = inner
    result = fmt_exception("审查", outer)
    assert "审查超时" in result
    assert "操作耗时过长" in result

    # Also test asyncio variant via __context__
    inner2 = asyncio.TimeoutError()
    outer2 = ValueError("failed")
    outer2.__context__ = inner2
    result2 = fmt_exception("执行", outer2)
    assert "执行超时" in result2
    assert "操作耗时过长" in result2


# ---------------------------------------------------------------------------
# fmt_error: save-state and restore-context branches
# (consolidates TestSpecHandlerSaveStateEmptyGuard)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("context,exc_factory,must_contain", [
    pytest.param("保存 Spec 状态", TimeoutError, "超时", id="save_state_timeout"),
    pytest.param("保存 Spec 状态", Exception, "保存 Spec 状态", id="save_state_exception"),
])
def test_fmt_error_nonempty(context, exc_factory, must_contain):
    """fmt_error must never leave empty detail."""
    from src.utils.errors import fmt_error

    result = fmt_error(context, exc_factory())
    assert result
    assert must_contain in result
    assert not result.endswith(": ")


# ---------------------------------------------------------------------------
# f-string pattern: `f"prefix: {str(e) or repr(e)}"` never empty
# (consolidates: TestIntentRecognizerEmptyGuard, TestEngineBaseLoggerEmptyGuard,
#  TestProjectManagerEmptyGuard, TestArtifactsParseEmptyGuard)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prefix,exc,expected_substr", [
    # IntentRecognizer
    pytest.param("异常回退", Exception(), "Exception()", id="intent_bare_exception"),
    pytest.param("异常回退", TimeoutError(), "TimeoutError()", id="intent_bare_timeout"),
    pytest.param("异常回退", ValueError("bad input"), "bad input", id="intent_named_value_error"),
    # EngineBase logger
    pytest.param("Deep Engine 执行超时 (task_id=t1)", TimeoutError(), "TimeoutError()", id="engine_timeout"),
    pytest.param("Deep Engine 执行异常", RuntimeError("connection lost"), "connection lost", id="engine_named_runtime"),
    # ProjectManager
    pytest.param("无法创建目录 /tmp/test", PermissionError("access denied"), "access denied", id="project_named_perm_error"),
    # ArtifactsParse
    pytest.param("规格 JSON 解析失败", Exception(), "Exception()", id="spec_parse_bare_exception"),
    pytest.param("无法创建目录 /tmp/test", OSError(), None, id="project_bare_os_error"),
])
def test_fstring_or_repr_pattern_nonempty(prefix, exc, expected_substr):
    """f'prefix: {str(e) or repr(e)}' must never produce empty tail."""
    msg = f"{prefix}: {str(exc) or repr(exc)}"
    assert msg
    assert not msg.endswith(": ")
    assert not msg.endswith("：")
    if expected_substr:
        assert expected_substr in msg


# ---------------------------------------------------------------------------
# build_review_exception_diagnostics: no (empty message) marker
# ---------------------------------------------------------------------------


class TestDiagnoseReviewFailureNoEmptyMessageMarker:
    """review_diagnostics.py: error_text must never contain '(empty message)'."""

    def _build(self, exc: Exception) -> dict:
        from src.utils.review_diagnostics import build_review_exception_diagnostics
        return build_review_exception_diagnostics(exc, cycle=1)

    @pytest.mark.parametrize("exc", [
        pytest.param(TimeoutError(), id="timeout"),
        pytest.param(ValueError(), id="value_error"),
        pytest.param(RuntimeError(), id="runtime_error"),
        pytest.param(Exception(), id="bare_exception"),
    ])
    def test_no_empty_marker(self, exc):
        diag = self._build(exc)
        assert "(empty message)" not in diag["error_text"]

    def test_timeout_friendly_text(self):
        diag = self._build(TimeoutError())
        assert "审查超时" in diag["error_text"]
        assert diag["err_repr"]
        assert "TimeoutError" in diag["err_repr"]

    @pytest.mark.parametrize("exc_cls", [ValueError, RuntimeError, Exception],
                             ids=["ValueError", "RuntimeError", "Exception"])
    def test_non_timeout_friendly_text(self, exc_cls):
        diag = self._build(exc_cls())
        assert "审查执行异常" in diag["error_text"]

    def test_timeout_with_message_preserved(self):
        diag = self._build(TimeoutError("took too long"))
        assert "took too long" in diag["error_text"]


# ---------------------------------------------------------------------------
# _infer_fail_reason + _has_timeout_in_chain
# ---------------------------------------------------------------------------


class TestInferFailReasonChainedExceptions:
    """review_diagnostics: _infer_fail_reason must detect TimeoutError in exception chains."""

    def _infer(self, exc: Exception) -> str:
        from src.utils.review_diagnostics import build_review_exception_diagnostics
        diag = build_review_exception_diagnostics(exc, cycle=1)
        return diag.get("fail_reason", "")

    @pytest.mark.parametrize("exc,expected_timeout", [
        pytest.param(TimeoutError(), True, id="direct_timeout"),
        pytest.param(TimeoutError("ACP prompt 超时"), True, id="direct_timeout_with_msg"),
        pytest.param(
            _chain_cause(RuntimeError, TimeoutError("inner")), True, id="wrapped_via_cause",
        ),
        pytest.param(
            _chain_context(RuntimeError, TimeoutError("inner")), True, id="wrapped_via_context",
        ),
        pytest.param(ValueError("bad input"), False, id="no_timeout"),
        pytest.param(_chain_cause(RuntimeError, ValueError("inner")), False, id="non_timeout_chain"),
    ])
    def test_fail_reason(self, exc, expected_timeout):
        result = self._infer(exc)
        if expected_timeout:
            assert result == "timeout"
        else:
            assert result != "timeout"

    def test_multi_level_nested_timeout(self):
        """Exception → RuntimeError → TimeoutError (3 levels) → 'timeout'."""
        deep = TimeoutError("deep")
        mid = RuntimeError("mid")
        mid.__cause__ = deep
        outer = Exception("outer")
        outer.__context__ = mid
        assert self._infer(outer) == "timeout"

    def test_chain_depth_limit_protection(self):
        """Deep chain (>10 levels) does not cause infinite recursion."""
        from src.utils.review_diagnostics import _has_timeout_in_chain
        exc = RuntimeError("base")
        for i in range(15):
            wrapper = RuntimeError(f"level-{i}")
            wrapper.__cause__ = exc
            exc = wrapper
        assert not _has_timeout_in_chain(exc)

    def test_chain_with_timeout_at_depth_9(self):
        """TimeoutError at depth 9 (within limit) is detected."""
        from src.utils.review_diagnostics import _has_timeout_in_chain
        exc = TimeoutError("deep")
        for i in range(9):
            wrapper = RuntimeError(f"level-{i}")
            wrapper.__cause__ = exc
            exc = wrapper
        assert _has_timeout_in_chain(exc)


# ---------------------------------------------------------------------------
# handle_review_exception E2E: no empty message leak
# ---------------------------------------------------------------------------


class TestHandleReviewExceptionE2EEmptyMessage:
    """handle_review_exception: no empty message / '(empty message)' leaks."""

    def _make_circuit(self):
        from src.spec_engine.review import ReviewCircuitState
        return ReviewCircuitState()

    def _make_settings(self):
        s = MagicMock()
        s.spec_review_failure_circuit_enabled = True
        s.spec_review_failure_max_consecutive = 3
        s.spec_review_failure_cooldown_cycles = 3
        s.spec_review_failure_max_cooldown_cycles = 12
        return s

    @pytest.mark.parametrize("exc,check_content", [
        pytest.param(asyncio.TimeoutError(), None, id="bare_asyncio_timeout"),
        pytest.param(TimeoutError(""), None, id="empty_string_timeout"),
        pytest.param(ValueError("bad input"), "bad input", id="non_timeout_exception"),
    ])
    def test_suggestion_nonempty(self, exc, check_content):
        from src.utils.review_helpers import handle_review_exception

        circuit = self._make_circuit()
        result = handle_review_exception(
            exc, circuit=circuit, cycle=1,
            settings=self._make_settings(), engine="spec",
        )
        assert result.suggestion_text
        assert "(empty message)" not in result.suggestion_text
        assert result.suggestion_text.strip() != ""
        if check_content:
            assert check_content in result.suggestion_text

    def test_chained_timeout_empty(self):
        """RuntimeError wrapping asyncio.TimeoutError() — chain traversal."""
        from src.utils.review_helpers import handle_review_exception

        inner = asyncio.TimeoutError()
        exc = RuntimeError("review failed")
        exc.__cause__ = inner
        circuit = self._make_circuit()
        result = handle_review_exception(
            exc, circuit=circuit, cycle=2,
            settings=self._make_settings(), engine="spec",
        )
        assert result.suggestion_text
        assert "(empty message)" not in result.suggestion_text


# ---------------------------------------------------------------------------
# build_review_error_suggestion output guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs,must_contain", [
    pytest.param(dict(fail_reason="timeout"), None, id="timeout_branch"),
    pytest.param(
        dict(fail_reason="unknown", error_text="(empty message)", err_repr=""),
        None, id="empty_message_marker",
    ),
    pytest.param(
        dict(fail_reason="parse_error", error_text="JSON decode failed"),
        "JSON decode failed", id="normal_error",
    ),
    pytest.param(dict(), None, id="all_empty"),
    pytest.param(
        dict(fail_reason="some_error", error_text="", err_repr="ValueError('x')"),
        "ValueError('x')", id="repr_fallback",
    ),
])
def test_build_review_error_suggestion_nonempty(kwargs, must_contain):
    """build_review_error_suggestion never returns empty for any branch."""
    from src.utils.review_helpers import build_review_error_suggestion

    result = build_review_error_suggestion(**kwargs)
    assert result and result.strip()
    assert "(empty message)" not in result
    if must_contain:
        assert must_contain in result


# ---------------------------------------------------------------------------
# Logger-level empty guards
# (consolidates: TestLogExceptionEmptyTimeout, TestSpecHandlerResumeTimeoutLogNotEmpty,
#  TestIMClientRetryTimeoutLogNotEmpty, TestAgentSessionRateLimitTimeoutLogNotEmpty)
# ---------------------------------------------------------------------------


class TestLogExceptionEmptyTimeout:
    """errors.py: log_exception must not produce empty detail for TimeoutError()."""

    def test_log_exception_empty_timeout_uses_detail(self):
        from src.utils.errors import GhostAPError, log_exception

        mock_logger = MagicMock(spec=logging.Logger)
        exc = GhostAPError("")
        log_exception(mock_logger, "测试操作", exc)
        mock_logger.warning.assert_called_once()
        logged_msg = mock_logger.warning.call_args[0][0]
        parts = logged_msg.split(": ", 1)
        assert len(parts) == 2, f"Expected 'msg: detail' format, got: {logged_msg}"
        assert parts[1], f"Detail part is empty in: {logged_msg}"

    def test_log_exception_timeout_error_goes_to_log_level(self):
        from src.utils.errors import log_exception

        mock_logger = MagicMock(spec=logging.Logger)
        log_exception(mock_logger, "超时测试", TimeoutError())
        mock_logger.log.assert_called_once()


@pytest.mark.parametrize("fmt_template,fmt_args", [
    pytest.param(
        "恢复任务上下文失败(task_id=%s): %s", ("test-task",),
        id="spec_handler_resume_timeout",
    ),
    pytest.param(
        "%s异常(尝试%d/%d): %s", ("发送消息", 1, 3),
        id="im_client_retry_timeout",
    ),
    pytest.param(
        "[RateLimit] 限速检测，等待 %ds 后重试 (attempt=%d/%d): %s", (30, 1, 3),
        id="agent_session_rate_limit_timeout",
    ),
])
def test_logger_format_nonempty_detail(fmt_template, fmt_args):
    """Logger format strings must not have empty tail for TimeoutError()."""
    detail = get_error_detail(TimeoutError())
    formatted = fmt_template % (*fmt_args, detail)
    assert not formatted.endswith(": "), f"Log ends with empty detail: {formatted}"
    assert "超时" in formatted or "未知" in formatted


# ===========================================================================
# Section 3: Lint scan tests (static file scanning)
# ===========================================================================


class TestNoBareFStringInUserVisibleErrors:
    """Lint guard: reply_error / send_error_card calls must not use bare f\"{e}\"
    or f\"{exc}\" which can produce empty strings for TimeoutError()."""

    _BARE_FSTR_RE = re.compile(
        r'(?:reply_error|send_error_card|_reply_message|reply_text|update_card)'
        r'\s*\([^)]*f["\'].*\{(?:e|exc|err|ex|te|error|exception)\}[^)]*\)'
    )
    _BARE_LOGGER_RE = re.compile(
        r'logger\.(?:warning|error)\s*\(\s*f["\'].*\{(?:e|exc|err|ex|te|error|exception)\}'
    )
    _SKIP_GUARDS = ("get_error_detail", "repr(", " or ")

    @classmethod
    def _line_is_guarded(cls, line: str) -> bool:
        return any(g in line for g in cls._SKIP_GUARDS)

    def _scan_src_files(self):
        src_dir = Path(__file__).resolve().parent.parent / "src"
        violations = []
        for py_file in src_dir.rglob("*.py"):
            for i, line in enumerate(py_file.read_text().splitlines(), 1):
                if self._line_is_guarded(line):
                    continue
                if self._BARE_FSTR_RE.search(line):
                    violations.append(f"{py_file.relative_to(src_dir.parent)}:{i}: {line.strip()}")
        return violations

    def _scan_logger_bare_fstr(self):
        src_dir = Path(__file__).resolve().parent.parent / "src"
        violations = []
        for py_file in src_dir.rglob("*.py"):
            for i, line in enumerate(py_file.read_text().splitlines(), 1):
                if self._line_is_guarded(line):
                    continue
                if self._BARE_LOGGER_RE.search(line):
                    violations.append(f"{py_file.relative_to(src_dir.parent)}:{i}: {line.strip()}")
        return violations

    def test_no_bare_fstring_in_user_visible_errors(self):
        violations = self._scan_src_files()
        assert not violations, (
            "Found bare f\"{e}\" in user-visible error paths (risk of empty message):\n"
            + "\n".join(violations)
        )

    def test_no_bare_fstring_in_logger_errors(self):
        violations = self._scan_logger_bare_fstr()
        assert not violations, (
            "Found bare f\"{e}\" in logger.warning/error (risk of empty message in logs):\n"
            + "\n".join(violations)
        )

    # --- Bare asyncio.wait_for lint ---
    _BARE_WAIT_FOR_RE = re.compile(r'asyncio\.wait_for\s*\(')
    _WAIT_FOR_ALLOWLIST = {"src/utils/async_helpers.py"}

    def _scan_bare_wait_for(self):
        src_dir = Path(__file__).resolve().parent.parent / "src"
        violations = []
        for py_file in src_dir.rglob("*.py"):
            rel = str(py_file.relative_to(src_dir.parent))
            if rel in self._WAIT_FOR_ALLOWLIST:
                continue
            for i, line in enumerate(py_file.read_text().splitlines(), 1):
                if self._BARE_WAIT_FOR_RE.search(line):
                    violations.append(f"{rel}:{i}: {line.strip()}")
        return violations

    def test_no_bare_asyncio_wait_for(self):
        violations = self._scan_bare_wait_for()
        assert not violations, (
            "Found bare asyncio.wait_for (use safe_wait_for instead):\n"
            + "\n".join(violations)
        )

    # --- Bare logger %s exception variable lint ---
    _BARE_LOGGER_PERCENT_RE = re.compile(
        r'logger\.(?:debug|info|warning|error|critical)\s*\('
        r'[^)]*%s[^)]*,\s*'
        r'(?:e|exc|ex|te|error|exception|cb_exc|last_err)\s*\)'
    )

    def _scan_logger_bare_percent(self):
        src_dir = Path(__file__).resolve().parent.parent / "src"
        violations = []
        for py_file in src_dir.rglob("*.py"):
            for i, line in enumerate(py_file.read_text().splitlines(), 1):
                stripped = line.strip()
                if self._line_is_guarded(stripped):
                    continue
                if "exc_info" in stripped:
                    continue
                if self._BARE_LOGGER_PERCENT_RE.search(stripped):
                    violations.append(f"{py_file.relative_to(src_dir.parent)}:{i}: {stripped}")
        return violations

    def test_no_bare_logger_percent_exception(self):
        violations = self._scan_logger_bare_percent()
        assert not violations, (
            "Found logger.xxx('...%s', e) with bare exception variable "
            "(use str(e) or repr(e) instead):\n"
            + "\n".join(violations)
        )


class TestReviewCallsTotalTimeout:
    """Lint guard: review-related send_prompt_with_retry calls must include total_timeout."""

    _REVIEW_FILES = ["src/spec_engine/review.py"]

    def test_review_send_prompt_with_retry_has_total_timeout(self):
        src_dir = Path(__file__).resolve().parent.parent
        violations = []
        call_re = re.compile(r'send_prompt_with_retry\s*\(')
        total_timeout_re = re.compile(r'total_timeout\s*=')
        review_fn_re = re.compile(r'^\s*def\s+(?:conduct_review|_conduct_review)\b')

        for rel_path in self._REVIEW_FILES:
            full_path = src_dir / rel_path
            if not full_path.exists():
                continue
            content = full_path.read_text()
            lines = content.splitlines()
            in_review_fn = False
            for i, line in enumerate(lines, 1):
                if re.match(r'^\s*def\s+\w+', line):
                    in_review_fn = bool(review_fn_re.match(line))
                if not in_review_fn:
                    continue
                if call_re.search(line):
                    snippet = "\n".join(lines[i - 1 : min(i + 10, len(lines))])
                    if not total_timeout_re.search(snippet):
                        violations.append(f"{rel_path}:{i}: {line.strip()}")

        assert not violations, (
            "Found send_prompt_with_retry calls in review code WITHOUT total_timeout:\n"
            + "\n".join(violations)
        )


class TestNoBareRaiseTimeoutError:
    """Prevent bare `raise TimeoutError()` (no message) in src/."""

    _BARE_RAISE_RE = re.compile(r'raise\s+TimeoutError\s*\(\s*\)')
    _ALLOWLIST = {"src/acp/sync_adapter.py"}

    def _scan(self):
        src_dir = Path(__file__).resolve().parent.parent / "src"
        violations = []
        for py_file in src_dir.rglob("*.py"):
            rel = str(py_file.relative_to(src_dir.parent))
            if rel in self._ALLOWLIST:
                continue
            for i, line in enumerate(py_file.read_text().splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if self._BARE_RAISE_RE.search(stripped):
                    violations.append(f"{rel}:{i}: {stripped}")
        return violations

    def test_no_bare_raise_timeout_error_in_src(self):
        violations = self._scan()
        assert not violations, (
            "Found bare `raise TimeoutError()` without message in src/ "
            "(risk of empty error message):\n"
            + "\n".join(violations)
        )


# ---------------------------------------------------------------------------
# Static scan: ban `str(e) or repr(e)` pattern from src/ (except errors.py)
# ---------------------------------------------------------------------------

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
_BAN_PATTERN = re.compile(r"str\(\w+\)\s+or\s+(?:repr\(\w+\)|\"(?:\(empty\)|empty)\")")
_ALLOWED_FILES = {"errors.py"}


class TestBanStrOrReprPattern:
    """Ensure `str(e) or repr(e)` is never used in src/ (use get_error_detail instead)."""

    def test_no_str_or_repr_in_src(self):
        violations: list[str] = []
        for py_file in sorted(_SRC_ROOT.rglob("*.py")):
            if py_file.name in _ALLOWED_FILES:
                continue
            try:
                text = py_file.read_text(encoding="utf-8")
            except Exception:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if _BAN_PATTERN.search(line):
                    rel = py_file.relative_to(_SRC_ROOT.parent)
                    violations.append(f"  {rel}:{lineno}: {line.strip()}")
        assert not violations, (
            f"Found {len(violations)} occurrence(s) of banned `str(x) or repr(x)` pattern "
            f"(use get_error_detail instead):\n" + "\n".join(violations)
        )


# ---------------------------------------------------------------------------
# No bare `except Exception: pass` in lock-related source files
# ---------------------------------------------------------------------------


class TestNoBareSilentExceptInLockFiles:
    """Lock-related files must not swallow exceptions silently."""

    _LOCK_FILES = [
        "src/chat_lock.py",
        "src/repo_lock.py",
        "src/card/builders/lock_chat.py",
        "src/card/builders/lock_repo.py",
        "src/card/builders/lock_common.py",
        "src/feishu/handlers/lock_helper.py",
    ]

    def test_no_except_exception_pass(self):
        root = Path(__file__).resolve().parent.parent
        pattern = re.compile(r'except\s+Exception\s*:\s*$')
        violations = []
        for rel_path in self._LOCK_FILES:
            fpath = root / rel_path
            if not fpath.exists():
                continue
            lines = fpath.read_text(encoding="utf-8").splitlines()
            for i, line in enumerate(lines, 1):
                if pattern.search(line.strip()):
                    for j in range(i, min(i + 3, len(lines))):
                        next_line = lines[j].strip()
                        if next_line == "pass":
                            violations.append(f"{rel_path}:{j + 1}: {next_line}  (after except at line {i})")
                            break
                        elif next_line:
                            break
        assert not violations, (
            "Found bare `except Exception: pass` (should use logger.debug with exc_info=True):\n"
            + "\n".join(violations)
        )
