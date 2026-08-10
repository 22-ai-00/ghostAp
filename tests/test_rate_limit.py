"""Tests for rate-limit detection and RateLimitAwareSession retry logic."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.acp.models import PromptResult
from src.agent_session.claude_cli import SyncClaudeCLISession
from src.agent_session.model_diagnostics import (
    _apply_compaction_once,
    _detect_rate_limit,
    _extract_model_from_agent_args,
    _replace_model_in_agent_args,
    classify_model_failure,
)
from src.agent_session.wrappers import (
    ModelFailureAwareSession,
    RateLimitAwareSession,
)

# ======================================================================
# _detect_rate_limit() tests
# ======================================================================


class TestDetectRateLimit:
    """Test rate-limit error detection from exception messages."""

    def test_detects_rate_limit_keyword(self):
        err = Exception("Request failed: rate_limit exceeded")
        assert _detect_rate_limit(err) is not None

    def test_detects_rate_limit_with_space(self):
        err = Exception("rate limit reached")
        assert _detect_rate_limit(err) is not None

    def test_detects_429_status(self):
        err = Exception("HTTP 429 Too Many Requests")
        result = _detect_rate_limit(err)
        assert result is not None

    def test_429_not_matched_as_substring(self):
        """429 must be a word boundary, not part of another number."""
        err = Exception("Error code 14291")
        assert _detect_rate_limit(err) is None

    def test_detects_too_many_requests(self):
        err = Exception("too many requests, please slow down")
        assert _detect_rate_limit(err) is not None

    def test_detects_overloaded(self):
        err = Exception("The server is overloaded")
        assert _detect_rate_limit(err) is not None

    def test_extracts_retry_after_seconds(self):
        err = Exception("rate_limit: retry_after=60")
        result = _detect_rate_limit(err)
        assert result == 60

    def test_extracts_retry_after_with_colon(self):
        err = Exception("Rate limit hit. Retry-After: 120")
        result = _detect_rate_limit(err)
        assert result == 120

    def test_clamps_retry_after_to_max_600(self):
        err = Exception("rate_limit retry_after=9999")
        result = _detect_rate_limit(err)
        assert result == 600

    def test_clamps_retry_after_to_min_1(self):
        err = Exception("rate_limit retry_after=0")
        result = _detect_rate_limit(err)
        assert result == 1

    def test_returns_zero_when_no_retry_after(self):
        err = Exception("rate limit exceeded")
        result = _detect_rate_limit(err)
        assert result == 0

    def test_returns_none_for_non_rate_limit_error(self):
        err = Exception("Connection refused")
        assert _detect_rate_limit(err) is None

    def test_returns_none_for_empty_message(self):
        err = Exception("")
        assert _detect_rate_limit(err) is None

    def test_case_insensitive(self):
        err = Exception("RATE LIMIT")
        assert _detect_rate_limit(err) is not None


class TestModelFailureClassifier:
    def test_need_compaction_detected_and_model_extracted(self):
        err = Exception("Model failed: model 'gpt-5.2': receive message: need compaction")
        info = classify_model_failure(error=err)
        assert info.get("reason") == "need_compaction"
        assert info.get("fail_phase") == "model_compaction"
        assert info.get("failed_model") == "gpt-5.2"

    def test_loop_detected(self):
        err = Exception("loop detected")
        info = classify_model_failure(error=err)
        assert info.get("reason") == "loop_detected"
        assert info.get("fail_phase") == "model_loop"

    def test_failover_to_extracted_even_when_reason_unknown(self):
        err = Exception("Failing over to: gpt-5.1")
        info = classify_model_failure(error=err)
        assert info.get("failover_to") == "gpt-5.1"

    def test_snippet_fields_are_included_in_blob(self):
        class _E(Exception):
            pass

        e = _E("")
        e.stderr_snippet = "Model failed: model 'gpt-5.2': receive message: need compaction"
        info = classify_model_failure(error=e)
        assert info.get("reason") == "need_compaction"
        assert info.get("failed_model") == "gpt-5.2"


class TestModelFailoverHelpers:
    def test_extract_model_from_agent_args_coco_style(self):
        assert _extract_model_from_agent_args(["acp", "serve", "-c", "model.name=gpt-5.2"]) == "gpt-5.2"


    def test_replace_model_in_agent_args_coco_style(self):
        args, ok = _replace_model_in_agent_args(["acp", "serve", "-c", "model.name=gpt-5.2"], "gpt-5.1")
        assert ok is True
        assert "model.name=gpt-5.1" in args


def test_apply_compaction_once_rebuilds_session_with_same_cmd_args(monkeypatch):
    """compaction builder 只重建 candidate；旧 session 由 wrapper 事务关闭。"""
    created = []

    class _Old:
        def __init__(self):
            self._agent_type = "coco"
            self._cwd = "/tmp"
            self._agent_cmd = "coco"
            self._agent_args = ["acp", "serve", "-c", "model.name=gpt-5.2"]
            self.closed = False

        def close(self):
            self.closed = True

        def start(self, startup_timeout: float = 60) -> str:
            return "old"

    class _New:
        def __init__(self, **kw):
            created.append(dict(kw))
            self._agent_type = kw.get("agent_type")
            self._cwd = kw.get("cwd")
            self._agent_cmd = kw.get("agent_cmd")
            self._agent_args = kw.get("agent_args")
            self.session_id = "new"

        def start(self, startup_timeout: float = 60) -> str:
            self.started_timeout = startup_timeout
            return "new"

        def close(self):
            return None

    old = _Old()
    new = _apply_compaction_once(session=old, session_builder=lambda **kw: _New(**kw), startup_timeout_s=1.0)
    assert old.closed is False
    assert new is not None
    assert created and created[-1]["agent_cmd"] == "coco"
    assert "model.name=gpt-5.2" in " ".join(created[-1]["agent_args"])


def test_model_failure_aware_session_need_compaction_compacts_then_retries(monkeypatch, caplog):
    """ModelFailureAwareSession：need compaction 时应执行 compaction 并重试一次。"""
    import logging

    caplog.set_level(logging.WARNING)

    class _Inner:
        def __init__(self):
            self.session_id = "sid"
            self.created_at = 0.0
            self.last_active = 0.0
            self.message_count = 0
            self.last_query = ""
            self.calls = 0

        def describe_agent(self):
            return "dummy"

        def start(self, startup_timeout: float = 60):
            return "sid"


        def load_local_history(self, session_id=None, limit: int = 200):
            return []

        def cancel(self):
            return None

        def close(self):
            return None

        def to_snapshot(self):
            return {}

        def get_session_info(self):
            return ""

        def is_server_running(self):
            return True

        def is_server_healthy(self, healthcheck_timeout: float = 2.0):
            return True

        def send_prompt(self, text: str, on_event=None, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("Model failed: model 'gpt-5.2': receive message: need compaction")
            return type("R", (), {"stop_reason": "end_turn", "text": "ok"})()

    compactions = {"n": 0}

    def _compaction_action(sess):
        compactions["n"] += 1
        return sess  # return same inner for test

    s = ModelFailureAwareSession(inner=_Inner(), compaction_action=_compaction_action)
    r = s.send_prompt("hi")
    assert getattr(r, "text", "") == "ok"
    assert compactions["n"] == 1
    joined = "\n".join([x.getMessage() for x in caplog.records])
    assert "action=compaction" in joined
    assert "reason=need_compaction" in joined
    assert "fail_phase=model_compaction" in joined


def test_model_failure_aware_session_compaction_loop_suppresses_compaction(monkeypatch, caplog):
    """loop 检测：窗口内 compaction 次数达到阈值时，应抑制 compaction 并直接抛错。"""
    import logging

    caplog.set_level(logging.WARNING)

    class _Inner:
        def __init__(self):
            self.session_id = "sid"
            self.created_at = 0.0
            self.last_active = 0.0
            self.message_count = 0
            self.last_query = ""
            self.calls = 0

        def describe_agent(self):
            return "dummy"

        def start(self, startup_timeout: float = 60):
            return "sid"


        def load_local_history(self, session_id=None, limit: int = 200):
            return []

        def cancel(self):
            return None

        def close(self):
            return None

        def to_snapshot(self):
            return {}

        def get_session_info(self):
            return ""

        def is_server_running(self):
            return True

        def is_server_healthy(self, healthcheck_timeout: float = 2.0):
            return True

        def send_prompt(self, text: str, on_event=None, timeout=None):
            self.calls += 1
            raise RuntimeError("Model failed: model 'gpt-5.2': receive message: need compaction")

    class _Settings:
        model_failure_compaction_enabled = True
        model_failure_compaction_loop_window_s = 999.0
        model_failure_compaction_loop_max = 1  # 1 次即判 loop

    monkeypatch.setattr("src.agent_session.wrappers.get_settings", lambda: _Settings())

    called = {"n": 0}

    def _compaction_action(sess):
        called["n"] += 1
        return sess

    s = ModelFailureAwareSession(inner=_Inner(), compaction_action=_compaction_action)
    with pytest.raises(RuntimeError):
        s.send_prompt("hi")
    assert called["n"] == 0
    joined = "\n".join([x.getMessage() for x in caplog.records]).lower()
    assert "action=suppress" in joined
    assert "fail_phase=model_loop" in joined


def test_model_failure_aware_session_loop_detected_triggers_failover(monkeypatch, caplog):
    """loop detected：应触发一次 failover（gpt-5.2 -> gpt-5.1）并重试。"""
    import logging

    caplog.set_level(logging.WARNING)

    class _Inner:
        def __init__(self):
            self.session_id = "sid"
            self.created_at = 0.0
            self.last_active = 0.0
            self.message_count = 0
            self.last_query = ""
            self._agent_type = "coco"
            self._cwd = "/tmp"
            self._agent_cmd = "coco"
            self._agent_args = ["acp", "serve", "-c", "model.name=gpt-5.2"]
            self.calls = 0

        def describe_agent(self):
            return "dummy"

        def start(self, startup_timeout: float = 60):
            return "sid"


        def load_local_history(self, session_id=None, limit: int = 200):
            return []

        def cancel(self):
            return None

        def close(self):
            return None

        def to_snapshot(self):
            return {}

        def get_session_info(self):
            return ""

        def is_server_running(self):
            return True

        def is_server_healthy(self, healthcheck_timeout: float = 2.0):
            return True

        def send_prompt(self, text: str, on_event=None, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("loop detected\nFailing over to: gpt-5.1")
            return type("R", (), {"stop_reason": "end_turn", "text": "ok"})()

    class _New:
        def __init__(self, **kw):
            self.session_id = "new"
            self.created_at = 0.0
            self.last_active = 0.0
            self.message_count = 0
            self.last_query = ""
            self._agent_type = kw.get("agent_type")
            self._cwd = kw.get("cwd")
            self._agent_cmd = kw.get("agent_cmd")
            self._agent_args = kw.get("agent_args")
            self._inner = _Inner()  # not used

        def start(self, startup_timeout: float = 60):
            return "new"

        def describe_agent(self):
            return "dummy"


        def load_local_history(self, session_id=None, limit: int = 200):
            return []

        def cancel(self):
            return None

        def close(self):
            return None

        def to_snapshot(self):
            return {}

        def get_session_info(self):
            return ""

        def is_server_running(self):
            return True

        def is_server_healthy(self, healthcheck_timeout: float = 2.0):
            return True

        def send_prompt(self, text: str, on_event=None, timeout=None):
            return type("R", (), {"stop_reason": "end_turn", "text": "ok"})()

    # patch SyncACPSession so _do_failover can rebuild
    monkeypatch.setattr("src.agent_session.wrappers.SyncACPSession", lambda **kw: _New(**kw))

    s = ModelFailureAwareSession(inner=_Inner())
    r = s.send_prompt("hi")
    assert getattr(r, "text", "") == "ok"
    logs = "\n".join([x.getMessage() for x in caplog.records]).lower()
    assert "action=failover" in logs
    assert "fail_phase=model_loop" in logs


# ======================================================================
# RateLimitAwareSession tests
# ======================================================================


def _make_mock_session(**overrides):
    """Create a mock SyncSession with sensible defaults."""
    s = MagicMock()
    s.session_id = overrides.get("session_id", "test-session-id")
    s.created_at = overrides.get("created_at", 1000.0)
    s.last_active = overrides.get("last_active", 1000.0)
    s.message_count = overrides.get("message_count", 0)
    s.last_query = overrides.get("last_query", "")
    s.send_prompt.return_value = PromptResult(stop_reason="end_turn", text="ok")
    return s


class TestRateLimitAwareSession:
    """Test RateLimitAwareSession wrapper."""


    def test_delegates_cancel(self):
        inner = _make_mock_session()
        cancel_ev = threading.Event()
        wrapped = RateLimitAwareSession(inner, cancel_event=cancel_ev)
        wrapped.cancel()
        inner.cancel.assert_called_once()
        assert cancel_ev.is_set()


    @patch("src.agent_session.wrappers.get_settings")
    def test_send_prompt_bypasses_retry_when_disabled(self, mock_settings):
        settings = MagicMock()
        settings.rate_limit_retry_enabled = False
        mock_settings.return_value = settings

        inner = _make_mock_session()
        inner.send_prompt.side_effect = Exception("rate_limit exceeded")
        wrapped = RateLimitAwareSession(inner)

        with pytest.raises(Exception, match="rate_limit exceeded"):
            wrapped.send_prompt("test")

    @patch("src.agent_session.wrappers.get_settings")
    def test_retries_on_rate_limit_then_succeeds(self, mock_settings):
        settings = MagicMock()
        settings.rate_limit_retry_enabled = True
        settings.rate_limit_max_retries = 3
        settings.rate_limit_max_wait = 300
        settings.rate_limit_base_wait = 1  # fast for test
        mock_settings.return_value = settings

        inner = _make_mock_session()
        expected = PromptResult(stop_reason="end_turn", text="ok")
        inner.send_prompt.side_effect = [
            Exception("rate_limit exceeded"),
            expected,
        ]

        callback = MagicMock()
        wrapped = RateLimitAwareSession(inner, on_rate_limit=callback)
        result = wrapped.send_prompt("test")

        assert result == expected
        assert inner.send_prompt.call_count == 2
        callback.assert_called_once()

    @patch("src.agent_session.wrappers.get_settings")
    def test_raises_after_max_retries_exhausted(self, mock_settings):
        settings = MagicMock()
        settings.rate_limit_retry_enabled = True
        settings.rate_limit_max_retries = 2
        settings.rate_limit_max_wait = 300
        settings.rate_limit_base_wait = 1
        mock_settings.return_value = settings

        inner = _make_mock_session()
        inner.send_prompt.side_effect = Exception("rate_limit exceeded")

        wrapped = RateLimitAwareSession(inner)
        with pytest.raises(Exception, match="rate_limit exceeded"):
            wrapped.send_prompt("test")

        # 1 initial + 2 retries = 3 total
        assert inner.send_prompt.call_count == 3

    @patch("src.agent_session.wrappers.get_settings")
    def test_non_rate_limit_error_not_retried(self, mock_settings):
        settings = MagicMock()
        settings.rate_limit_retry_enabled = True
        settings.rate_limit_max_retries = 3
        settings.rate_limit_max_wait = 300
        settings.rate_limit_base_wait = 1
        mock_settings.return_value = settings

        inner = _make_mock_session()
        inner.send_prompt.side_effect = Exception("Connection refused")

        wrapped = RateLimitAwareSession(inner)
        with pytest.raises(Exception, match="Connection refused"):
            wrapped.send_prompt("test")

        assert inner.send_prompt.call_count == 1

    @patch("src.agent_session.wrappers.get_settings")
    def test_cancel_interrupts_rate_limit_wait(self, mock_settings):
        settings = MagicMock()
        settings.rate_limit_retry_enabled = True
        settings.rate_limit_max_retries = 3
        settings.rate_limit_max_wait = 300
        settings.rate_limit_base_wait = 60  # long wait
        mock_settings.return_value = settings

        inner = _make_mock_session()
        inner.send_prompt.side_effect = Exception("rate_limit exceeded")

        cancel_ev = threading.Event()
        wrapped = RateLimitAwareSession(inner, cancel_event=cancel_ev)

        # Cancel after a short delay
        def _cancel():
            time.sleep(0.3)
            cancel_ev.set()

        t = threading.Thread(target=_cancel, daemon=True)
        t.start()

        start = time.monotonic()
        with pytest.raises(Exception, match="rate_limit"):
            wrapped.send_prompt("test")
        elapsed = time.monotonic() - start

        # Should have been interrupted well before the 60s wait
        assert elapsed < 5

    @patch("src.agent_session.wrappers.get_settings")
    def test_callback_exception_does_not_break_retry(self, mock_settings):
        settings = MagicMock()
        settings.rate_limit_retry_enabled = True
        settings.rate_limit_max_retries = 3
        settings.rate_limit_max_wait = 300
        settings.rate_limit_base_wait = 1
        mock_settings.return_value = settings

        inner = _make_mock_session()
        expected = PromptResult(stop_reason="end_turn", text="ok")
        inner.send_prompt.side_effect = [
            Exception("rate_limit exceeded"),
            expected,
        ]

        def bad_callback(wait_secs):
            raise RuntimeError("callback exploded")

        wrapped = RateLimitAwareSession(inner, on_rate_limit=bad_callback)
        result = wrapped.send_prompt("test")

        # Should still succeed despite callback error
        assert result == expected

    @patch("src.agent_session.wrappers.get_settings")
    def test_rate_limit_until_set_during_wait(self, mock_settings):
        settings = MagicMock()
        settings.rate_limit_retry_enabled = True
        settings.rate_limit_max_retries = 3
        settings.rate_limit_max_wait = 300
        settings.rate_limit_base_wait = 2
        mock_settings.return_value = settings

        inner = _make_mock_session()
        expected = PromptResult(stop_reason="end_turn", text="ok")
        inner.send_prompt.side_effect = [
            Exception("rate_limit exceeded"),
            expected,
        ]

        wrapped = RateLimitAwareSession(inner)

        # Check rate_limit_until is set during the wait window
        observed_until = []

        def _observer():
            time.sleep(0.2)
            observed_until.append(wrapped.rate_limit_until)

        t = threading.Thread(target=_observer, daemon=True)
        t.start()

        result = wrapped.send_prompt("test")
        t.join(timeout=5)

        assert result == expected
        # During the wait, rate_limit_until should have been a non-None monotonic deadline
        assert len(observed_until) == 1
        assert observed_until[0] is not None

    def test_to_snapshot_delegates(self):
        inner = _make_mock_session()
        inner.to_snapshot.return_value = {"session_id": "x", "backend": "cli"}
        wrapped = RateLimitAwareSession(inner)
        assert wrapped.to_snapshot() == {"session_id": "x", "backend": "cli"}

    def test_get_session_info_delegates(self):
        inner = _make_mock_session()
        inner.get_session_info.return_value = "info"
        wrapped = RateLimitAwareSession(inner)
        assert wrapped.get_session_info() == "info"

    def test_is_server_running_delegates(self):
        inner = _make_mock_session()
        inner.is_server_running.return_value = True
        wrapped = RateLimitAwareSession(inner)
        assert wrapped.is_server_running() is True

    def test_is_server_healthy_delegates(self):
        inner = _make_mock_session()
        inner.is_server_healthy.return_value = False
        wrapped = RateLimitAwareSession(inner)
        assert wrapped.is_server_healthy(healthcheck_timeout=5.0) is False
        inner.is_server_healthy.assert_called_once_with(healthcheck_timeout=5.0)


# ======================================================================
# create_engine_session wrapping tests
# ======================================================================


class TestCreateEngineSession:
    """Test that create_engine_session wraps with RateLimitAwareSession.

    说明：当前 create_engine_session 会在最外层套一层 ModelFailureAwareSession，
    因此这里需要检查“内层是否包含 RateLimitAwareSession”。
    """

    @patch("src.agent_session.factory.get_settings")
    @patch("src.agent_session.factory.SyncClaudeCLISession")
    def test_claude_wrapped_when_enabled(self, MockCLI, mock_settings):
        settings = MagicMock()
        settings.rate_limit_retry_enabled = True
        settings.acp_startup_timeout = 20
        mock_settings.return_value = settings

        mock_session = MagicMock()
        MockCLI.return_value = mock_session

        from src.agent_session import create_engine_session

        result = create_engine_session("claude", "/tmp")

        assert isinstance(result, ModelFailureAwareSession)
        assert isinstance(getattr(result, "_inner", None), RateLimitAwareSession)
        mock_session.start.assert_called_once()

    @patch("src.agent_session.factory.get_settings")
    @patch("src.agent_session.factory.SyncClaudeCLISession")
    def test_claude_not_wrapped_when_disabled(self, MockCLI, mock_settings):
        settings = MagicMock()
        settings.rate_limit_retry_enabled = False
        settings.acp_startup_timeout = 20
        mock_settings.return_value = settings

        mock_session = MagicMock()
        MockCLI.return_value = mock_session

        from src.agent_session import create_engine_session

        result = create_engine_session("claude", "/tmp")

        assert isinstance(result, ModelFailureAwareSession)
        assert not isinstance(getattr(result, "_inner", None), RateLimitAwareSession)

    @patch("src.acp.sync_adapter.start_session_with_retry")
    @patch("src.agent_session.factory.get_settings")
    @patch("src.agent_session.factory.SyncClaudeCLISession")
    def test_claude_uses_acp_when_tool_filter_required(self, MockCLI, mock_settings, mock_start):
        settings = MagicMock()
        settings.rate_limit_retry_enabled = False
        settings.acp_startup_timeout = 20
        mock_settings.return_value = settings
        acp_session = MagicMock()
        mock_start.return_value = acp_session

        from src.agent_session import create_engine_session

        result = create_engine_session("claude", "/tmp", require_tool_filter=True)

        assert isinstance(result, ModelFailureAwareSession)
        assert getattr(result, "_inner", None) is acp_session
        MockCLI.assert_not_called()
        mock_start.assert_called_once_with(
            agent_type="claude",
            cwd="/tmp",
            startup_timeout=20,
            model_name=None,
            auto_approve=None,  # No lane override: preserve the Settings fallback.
        )


def test_model_failure_failover_map_default_in_settings(monkeypatch):
    """配置回归：Settings 应提供默认 failover 映射 gpt-5.2:gpt-5.1。"""
    from src.config import Settings

    s = Settings()
    assert "gpt-5.2" in (s.model_failure_failover_map or "")
    assert "gpt-5.1" in (s.model_failure_failover_map or "")


def test_model_failure_aware_session_send_prompt_with_retry_success_after_retry(monkeypatch):
    class _Inner:
        def __init__(self):
            self.session_id = "sid"
            self.created_at = 0.0
            self.last_active = 0.0
            self.message_count = 0
            self.last_query = ""
            self.calls = 0

        def describe_agent(self):
            return "dummy"

        def start(self, startup_timeout: float = 60):
            return "sid"


        def load_local_history(self, session_id=None, limit: int = 200):
            return []

        def cancel(self):
            return None

        def close(self):
            return None

        def to_snapshot(self):
            return {}

        def get_session_info(self):
            return ""

        def is_server_running(self):
            return True

        def is_server_healthy(self, healthcheck_timeout: float = 2.0):
            return True

        def send_prompt(self, text: str, on_event=None, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("internal error")
            return PromptResult(stop_reason="end_turn", text="ok")

    class _S:
        model_failure_compaction_enabled = False

    monkeypatch.setattr("src.agent_session.wrappers.get_settings", lambda: _S())
    s = ModelFailureAwareSession(inner=_Inner())
    out = s.send_prompt_with_retry("hello")
    assert out.text == "ok"


def test_sync_claude_cli_send_prompt_with_retry_uses_retry_policy():
    s = SyncClaudeCLISession(cwd="/tmp")
    calls = {"n": 0}

    def _flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("internal error")
        return PromptResult(stop_reason="end_turn", text="done")

    s.send_prompt = _flaky  # type: ignore[assignment]
    out = s.send_prompt_with_retry("hello")
    assert out.text == "done"
