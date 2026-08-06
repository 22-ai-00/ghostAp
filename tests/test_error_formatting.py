import asyncio
import json

import pytest

from src.utils.errors import (
    _has_timeout_in_chain,
    classify_timeout,
    fmt_error,
    fmt_exception,
    get_error_detail,
)


def _chain(outer, *, cause=None, context=None):
    """Helper: attach __cause__/__context__ and return the outer exception."""
    if cause is not None:
        outer.__cause__ = cause
    if context is not None:
        outer.__context__ = context
    return outer


class TestErrorFormatting:
    def test_fmt_error_with_string(self):
        """Test formatting a simple string error."""
        result = fmt_error("测试", "发生错误")
        assert result == "❌ 测试失败: 发生错误"

    def test_fmt_error_with_empty_string(self):
        """Test formatting an empty string (should just show action failed)."""
        result = fmt_error("测试")
        assert result == "❌ 测试失败"

    def test_fmt_error_with_exception(self):
        """Test formatting a general exception."""
        exc = ValueError("无效值")
        result = fmt_error("测试", exc)
        assert result == "❌ 测试失败: 无效值"

    def test_fmt_error_with_timeout_empty_message(self):
        """Test formatting a TimeoutError with empty message (the main fix)."""
        exc = asyncio.TimeoutError()
        result = fmt_error("测试", exc)
        assert "操作超时，请稍后重试" in result
        assert "❌ 测试失败" in result

    def test_fmt_error_with_timeout_custom_message(self):
        """Test formatting a TimeoutError with a custom message (should be preserved)."""
        # Note: asyncio.TimeoutError() doesn't usually take args in older python,
        # but built-in TimeoutError does. asyncio.TimeoutError is an alias in 3.11+.
        # Let's test with built-in TimeoutError which is compatible.
        exc = TimeoutError("连接超时")
        result = fmt_error("测试", exc)
        # Expecting: ❌ 测试失败: 操作超时 (连接超时)
        assert "操作超时 (连接超时)" in result


class TestErrorDiagnosticContext:
    def test_register_and_resolve_diagnostic_details_are_sanitized_and_truncated(self):
        from src.card.error_diagnostics import ErrorDiagnosticStore

        store = ErrorDiagnosticStore(ttl_seconds=60, max_details_chars=80)
        token = store.register(
            title="Codex 暂不可用",
            summary="cli unavailable",
            details="command: /home/alice/project/.venv/bin/coco --token SECRET_TOKEN=abc123\n"
            "stderr: failed at /data00/home/alice/work/ghostAp/src/main.py\n"
            + "x" * 200,
            chat_id="c1",
            origin_message_id="m1",
            request_id="req-1",
        )

        rendered = store.render(token, chat_id="c1", origin_message_id="m1", request_id="req-1")

        assert "Codex 暂不可用" in rendered
        assert "cli unavailable" in rendered
        assert "/home/alice" not in rendered
        assert "/data00/home/alice" not in rendered
        assert "SECRET_TOKEN=abc123" not in rendered
        assert "<path>" in rendered
        assert "<redacted>" in rendered
        assert "已截断" in rendered
        assert len(rendered) < 500

    def test_missing_diagnostic_token_returns_expired_feedback(self):
        from src.card.error_diagnostics import ErrorDiagnosticStore

        store = ErrorDiagnosticStore(ttl_seconds=60)

        assert store.render("missing-token") == "⚠️ 诊断详情已过期或不存在，请重新触发操作获取最新摘要。"


class TestFuturesUnfinishedSanitization:
    """Verify stdlib concurrent.futures 'N (of M) futures unfinished' is sanitized."""

    def test_fmt_error_sanitizes_futures_unfinished(self):
        exc = TimeoutError("1 (of 5) futures unfinished")
        result = fmt_error("操作", exc)
        assert "futures unfinished" not in result
        assert "操作超时" in result

    def test_fmt_error_sanitizes_multiple_futures_unfinished(self):
        exc = TimeoutError("3 (of 5) futures unfinished")
        result = fmt_error("审查", exc)
        assert "futures unfinished" not in result
        assert "操作超时" in result

    def test_get_error_detail_sanitizes_futures_unfinished(self):
        exc = TimeoutError("1 (of 5) futures unfinished")
        result = get_error_detail(exc)
        assert "futures unfinished" not in result
        assert "操作超时" in result

    def test_get_error_detail_pure_futures_becomes_friendly(self):
        """When the entire message is just 'N (of M) futures unfinished', fallback to friendly text."""
        exc = TimeoutError("2 (of 3) futures unfinished")
        result = get_error_detail(exc)
        assert "futures unfinished" not in result
        assert "请稍后重试" in result

    def test_fmt_error_preserves_custom_timeout_message(self):
        """Custom TimeoutError messages that don't match the pattern are kept."""
        exc = TimeoutError("连接超时")
        result = fmt_error("操作", exc)
        assert "连接超时" in result
        assert "操作超时" in result

    def test_fmt_error_mixed_message_sanitized(self):
        """Message containing futures unfinished mixed with other text."""
        exc = TimeoutError("timeout after 30s: 1 (of 5) futures unfinished")
        result = fmt_error("操作", exc)
        assert "futures unfinished" not in result
        assert "timeout after 30s" in result


class TestClassifyTimeout:
    """Tests for classify_timeout — single source of truth for timeout classification."""

    def test_classify_timeout_direct_timeout(self):
        assert classify_timeout(TimeoutError("boom")) is True

    def test_classify_timeout_asyncio_timeout(self):
        assert classify_timeout(asyncio.TimeoutError()) is True

    def test_classify_timeout_chained_timeout(self):
        inner = TimeoutError("inner")
        outer = RuntimeError("outer")
        outer.__cause__ = inner
        assert classify_timeout(outer) is True

    def test_classify_timeout_non_timeout(self):
        assert classify_timeout(ValueError("nope")) is False

    # --- Heuristic: "Internal error" + elapsed/timeout threshold ---

    def test_heuristic_internal_error_near_timeout(self):
        """97.5% of timeout threshold + "Internal error" -> True."""
        assert classify_timeout(
            RuntimeError("Internal error"), elapsed_s=19.5, timeout_s=20
        ) is True

    def test_heuristic_internal_error_below_threshold(self):
        """Only 50% of timeout threshold -> False."""
        assert classify_timeout(
            RuntimeError("Internal error"), elapsed_s=10, timeout_s=20
        ) is False

    def test_heuristic_no_internal_error_near_timeout(self):
        """Near timeout but error message is not "Internal error" -> False."""
        assert classify_timeout(
            RuntimeError("some other error"), elapsed_s=19.5, timeout_s=20
        ) is False

    def test_heuristic_internal_error_case_insensitive(self):
        """Case-insensitive match on "internal error" -> True."""
        assert classify_timeout(
            RuntimeError("internal error"), elapsed_s=19.5, timeout_s=20
        ) is True

    def test_heuristic_chained_internal_error(self):
        """Outer RuntimeError wrapping inner "Internal error" -> True."""
        inner = RuntimeError("Internal error")
        outer = RuntimeError("session start failed")
        outer.__cause__ = inner
        assert classify_timeout(outer, elapsed_s=19.5, timeout_s=20) is True

    def test_heuristic_no_elapsed_no_timeout(self):
        """No elapsed/timeout provided -> no heuristic, "Internal error" alone -> False."""
        assert classify_timeout(RuntimeError("Internal error")) is False

    def test_heuristic_only_one_param_provided(self):
        """Only one of elapsed/timeout provided -> heuristic not applied."""
        assert classify_timeout(
            RuntimeError("Internal error"), elapsed_s=19.5
        ) is False
        assert classify_timeout(
            RuntimeError("Internal error"), timeout_s=20
        ) is False


class TestFmtException:
    """Tests for fmt_exception — uses classify_timeout as single source of truth."""

    def test_fmt_exception_direct_timeout(self):
        exc = TimeoutError("boom")
        result = fmt_exception("连接", exc)
        assert result == "❌ 连接超时: 操作耗时过长，请重试"

    def test_fmt_exception_asyncio_timeout(self):
        exc = asyncio.TimeoutError()
        result = fmt_exception("请求", exc)
        assert result == "❌ 请求超时: 操作耗时过长，请重试"

    def test_fmt_exception_chained_timeout(self):
        inner = TimeoutError("inner")
        outer = RuntimeError("outer")
        outer.__cause__ = inner
        result = fmt_exception("操作", outer)
        assert result == "❌ 操作超时: 操作耗时过长，请重试"

    def test_fmt_exception_non_timeout(self):
        exc = ValueError("bad value")
        result = fmt_exception("解析", exc)
        assert result == "❌ 解析异常: bad value"

    def test_fmt_exception_non_timeout_empty_message(self):
        exc = RuntimeError()
        result = fmt_exception("处理", exc)
        assert "❌ 处理异常:" in result


class TestHasOnErrorProtocol:
    """Verify HasOnError Protocol and _format_engine_error type safety."""

    def test_spec_engine_callbacks_satisfies_protocol(self):
        from src.engine_base import HasOnError
        from src.spec_engine.engine import SpecEngineCallbacks

        cb = SpecEngineCallbacks()
        assert isinstance(cb, HasOnError)

    def test_deep_engine_callbacks_satisfies_protocol(self):
        from src.deep_engine.engine import DeepEngineCallbacks
        from src.engine_base import HasOnError

        cb = DeepEngineCallbacks()
        assert isinstance(cb, HasOnError)

    def test_format_engine_error_calls_on_error(self):
        """on_error callback should be invoked with the formatted message."""
        from dataclasses import dataclass, field
        from typing import Callable, Optional

        from src.engine_base import BaseEngine

        @dataclass
        class FakeCallbacks:
            on_error: Optional[Callable[[str], None]] = None
            errors: list = field(default_factory=list)

            def __post_init__(self):
                self.on_error = lambda msg: self.errors.append(msg)

        engine = BaseEngine(chat_id="test", root_path="/tmp")
        cb = FakeCallbacks()
        result = engine._format_engine_error(
            ValueError("boom"), "测试", callbacks=cb
        )
        assert "测试异常" in result
        assert "boom" in result
        assert len(cb.errors) == 1
        assert cb.errors[0] == result

    def test_format_engine_error_callbacks_none(self):
        """callbacks=None should not raise."""
        from src.engine_base import BaseEngine

        engine = BaseEngine(chat_id="test", root_path="/tmp")
        result = engine._format_engine_error(
            ValueError("x"), "测试", callbacks=None
        )
        assert "测试异常" in result

    def test_format_engine_error_on_error_none(self):
        """callbacks with on_error=None should not raise."""
        from dataclasses import dataclass
        from typing import Callable, Optional

        from src.engine_base import BaseEngine

        @dataclass
        class NullCallbacks:
            on_error: Optional[Callable[[str], None]] = None

        engine = BaseEngine(chat_id="test", root_path="/tmp")
        result = engine._format_engine_error(
            ValueError("x"), "测试", callbacks=NullCallbacks()
        )
        assert "测试异常" in result


class TestErrorCardPathContract:
    """Task 30: base/spec/engine error-card paths share detail/retry payload contract."""

    def test_system_error_card_preserves_detail_and_retry_action_payloads(self):
        from src.card import CardBuilder

        detail_action = {
            "action": "show_error_details",
            "engine_type": "spec",
            "request_id": "req-1",
            "engine_project_id": "proj-1",
        }
        retry_action = {
            "action": "spec_resume",
            "request_id": "req-1",
            "engine_project_id": "proj-1",
        }

        _, card_json = CardBuilder.build_error_card(
            ValueError("boom"),
            title="Spec 执行失败",
            details="engine=Spec; action=spec; request_id=req-1; project=proj-1",
            detail_action=detail_action,
            retry_action=retry_action,
        )
        card = json.loads(card_json)

        def _button_values(node):
            if isinstance(node, dict):
                if node.get("tag") == "button":
                    yield node.get("value")
                for value in node.values():
                    yield from _button_values(value)
            elif isinstance(node, list):
                for item in node:
                    yield from _button_values(item)

        values = list(_button_values(card.get("body", {}).get("elements", [])))

        detail_values = [value for value in values if value.get("action") == "show_error_details"]
        assert detail_values
        detail_value = detail_values[0]
        assert detail_value["request_id"] == "req-1"
        assert detail_value.get("diagnostic_token")
        assert "engine_type" not in detail_value
        assert "engine_project_id" not in detail_value
        assert "details" not in detail_value
        assert {"action": "spec_resume", "request_id": "req-1"} in values

    def test_degraded_error_card_uses_fixed_safe_summary_and_hides_raw_exception(self):
        """Degraded cards must not expose caller-composed raw exception text."""
        from src.card import CardBuilder

        raw_error = (
            "RuntimeError: cmd=/data00/home/alice/work/ghostAp/.venv/bin/coco "
            "TOKEN=secret\nTraceback at /data00/home/alice/work/ghostAp/src/feishu/handlers/programming.py"
        )

        _, card_json = CardBuilder.build_error_card(
            raw_error,
            title="Claude 启动失败",
            summary=raw_error,
            details=raw_error,
            severity="degraded",
            continue_action={"degraded_to": "Aiden"},
            retry_action={"original_mode": "Claude", "retry_mode": "Claude", "degraded_to": "Aiden"},
            detail_action={"chat_id": "c1", "origin_message_id": "card-mid"},
        )
        card = json.loads(card_json)
        rendered = json.dumps(card, ensure_ascii=False)

        assert "操作未能按原模式完成，已进入安全降级路径。" in rendered
        assert "cmd=" not in rendered
        assert "TOKEN=secret" not in rendered
        assert "Traceback" not in rendered
        assert "/data00/home/alice" not in rendered
        assert "Claude 启动失败" in rendered


class TestGetErrorDetailChain:
    """Migrated from test_utils_errors_timeout.py — get_error_detail chain coverage."""

    @pytest.mark.parametrize(
        "exc_factory, expect_timeout, extra_substr",
        [
            pytest.param(
                lambda: _chain(RuntimeError("wrapper"), cause=TimeoutError("inner timeout")),
                True,
                None,
                id="wrapped_timeout_via_cause",
            ),
            pytest.param(
                lambda: _chain(Exception("context wrap"), context=TimeoutError()),
                True,
                None,
                id="wrapped_timeout_via_context",
            ),
            pytest.param(
                lambda: _chain(
                    Exception("outer"),
                    context=_chain(RuntimeError("mid"), cause=TimeoutError("deep")),
                ),
                True,
                None,
                id="multi_level_chain",
            ),
            pytest.param(
                lambda: _chain(RuntimeError("wrapper"), cause=ValueError("inner")),
                False,
                "wrapper",
                id="no_timeout_in_chain",
            ),
        ],
    )
    def test_get_error_detail_chain(self, exc_factory, expect_timeout, extra_substr):
        result = get_error_detail(exc_factory())
        if expect_timeout:
            assert "超时" in result
        else:
            assert "超时" not in result
        if extra_substr is not None:
            assert extra_substr in result

    def test_get_error_detail_direct_timeout_regression(self):
        """Regression: direct TimeoutError() still returns '超时' text."""
        result = get_error_detail(TimeoutError())
        assert "超时" in result
        assert result == "操作超时，请稍后重试"


class TestFmtErrorWrappedTimeout:
    """Migrated from test_utils_errors_timeout.py — fmt_error wrapping behavior."""

    def test_fmt_error_wrapped_timeout_via_cause(self):
        """fmt_error with wrapped TimeoutError → timeout formatting."""
        outer = _chain(RuntimeError("rte wrap"), cause=TimeoutError("inner"))
        result = fmt_error("测试", outer)
        assert "超时" in result
        assert "rte wrap" in result

    def test_fmt_error_wrapped_bare_timeout_empty_outer(self):
        """fmt_error with wrapping exc that has empty str → fallback timeout text."""
        outer = _chain(RuntimeError(), cause=TimeoutError())
        result = fmt_error("测试", outer)
        assert "超时" in result
        assert "操作超时，请稍后重试" in result


class TestHasTimeoutInChain:
    """Migrated from test_utils_errors_timeout.py — _has_timeout_in_chain behavior."""

    @pytest.mark.parametrize(
        "exc_factory, expected",
        [
            pytest.param(
                lambda: _chain(RuntimeError("wrap"), cause=asyncio.TimeoutError()),
                True,
                id="asyncio_timeout_via_cause",
            ),
            pytest.param(
                lambda: _chain(RuntimeError("wrap"), context=TimeoutError("t")),
                True,
                id="builtin_timeout_via_context",
            ),
            pytest.param(
                lambda: _chain(
                    RuntimeError("wrap"),
                    cause=type("TimeoutExpired", (Exception,), {})("cmd timed out"),
                ),
                True,
                id="timeout_expired_by_name",
            ),
            pytest.param(
                lambda: _chain(
                    RuntimeError(),
                    cause=type("ReadTimeout", (Exception,), {})(),
                ),
                True,
                id="read_timeout_by_name",
            ),
            pytest.param(
                lambda: _chain(
                    RuntimeError(),
                    context=type("ConnectTimeout", (IOError,), {})(),
                ),
                True,
                id="connect_timeout_by_name",
            ),
            pytest.param(
                lambda: _chain(RuntimeError("wrap"), cause=ValueError("v")),
                False,
                id="no_timeout_returns_false",
            ),
        ],
    )
    def test_has_timeout_in_chain(self, exc_factory, expected):
        assert _has_timeout_in_chain(exc_factory()) is expected

    def test_chain_depth_limit(self):
        """Chain deeper than _CHAIN_MAX_DEPTH (10) does not recurse infinitely."""
        # Build a 15-level chain ending with TimeoutError
        current = TimeoutError("deep")
        for _ in range(15):
            parent = RuntimeError("level")
            parent.__cause__ = current
            current = parent
        # Might or might not find it depending on depth — just must not hang/crash
        _has_timeout_in_chain(current)

    def test_chain_consistency_with_review_diagnostics(self):
        """_has_timeout_in_chain from errors.py is the same function used by review_diagnostics."""
        from src.utils.review_diagnostics import _has_timeout_in_chain as rd_fn

        assert rd_fn is _has_timeout_in_chain
