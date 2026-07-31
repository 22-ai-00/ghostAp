"""Tests for ACP sync_adapter auto-update logic."""

from __future__ import annotations

import asyncio
import concurrent.futures
import re
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest

from src.acp import diagnostics as diag
from src.acp import sync_adapter as sa
from src.acp.models import PromptResult


def _cancel_test_session() -> sa.SyncACPSession:
    session = sa.SyncACPSession.__new__(sa.SyncACPSession)
    session._acp_session = SimpleNamespace(cancel=lambda: asyncio.sleep(0))
    session._loop = object()
    return session


def test_cancel_wait_returns_explicit_transport_ack(monkeypatch):
    session = _cancel_test_session()

    def submit(coro, _loop):
        coro.close()
        future: concurrent.futures.Future[None] = concurrent.futures.Future()
        future.set_result(None)
        return future

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", submit)

    assert session.cancel(wait=True, timeout=0.1) is True


def test_cancel_wait_timeout_returns_not_acknowledged(monkeypatch):
    session = _cancel_test_session()

    def submit(coro, _loop):
        coro.close()
        future: concurrent.futures.Future[None] = concurrent.futures.Future()
        future.set_exception(TimeoutError("cancel ack timeout"))
        return future

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", submit)

    assert session.cancel(wait=True, timeout=0.1) is False


def test_send_prompt_rejects_concurrent_threads_before_replacing_active_future(
    tmp_path: Path,
    monkeypatch,
):
    session = sa.SyncACPSession(
        agent_type="test",
        cwd=str(tmp_path),
        agent_cmd="test",
    )
    session._acp_session = SimpleNamespace(
        prompt=lambda *_args, **_kwargs: asyncio.sleep(0),
    )
    session._loop = object()
    session._start_watchdog = lambda: None

    first_future: concurrent.futures.Future[PromptResult] = (
        concurrent.futures.Future()
    )
    submitted = threading.Event()
    submit_calls = 0

    def submit(coro, _loop):
        nonlocal submit_calls
        submit_calls += 1
        coro.close()
        submitted.set()
        return first_future

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", submit)
    first_result: list[PromptResult] = []
    first_error: list[BaseException] = []

    def run_first() -> None:
        try:
            first_result.append(session.send_prompt("first", timeout=2))
        except BaseException as exc:  # pragma: no cover - diagnostic capture
            first_error.append(exc)

    thread = threading.Thread(target=run_first)
    thread.start()
    assert submitted.wait(timeout=1)

    with pytest.raises(RuntimeError, match="already running"):
        session.send_prompt("second", timeout=0.1)

    first_future.set_result(PromptResult(stop_reason="end_turn", text="done"))
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert first_error == []
    assert [result.text for result in first_result] == ["done"]
    assert submit_calls == 1
    assert session._active_future is None


def test_expected_prompt_connection_close_does_not_warn(monkeypatch, caplog):
    session = sa.SyncACPSession.__new__(sa.SyncACPSession)
    session._agent_type = "coco"
    session._acp_session = SimpleNamespace(prompt=lambda *_args, **_kwargs: asyncio.sleep(0))
    session._loop = object()
    session._log_failures = False
    session._active_future = None
    session._start_watchdog = lambda: None
    session.last_active = 0.0
    session.message_count = 0
    session.last_query = ""

    def submit(coro, _loop):
        coro.close()
        future = concurrent.futures.Future()
        future.set_exception(ConnectionError("Connection closed"))
        return future

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", submit)
    with caplog.at_level("WARNING", logger="src.acp.sync_adapter"):
        with pytest.raises(ConnectionError, match="Connection closed"):
            session.send_prompt("classify", timeout=0.1)

    assert not caplog.records
    assert session._force_dead is True


def test_close_timeout_keeps_transport_handles_and_raises(monkeypatch):
    session = sa.SyncACPSession.__new__(sa.SyncACPSession)
    session._agent_type = "codex"
    session._force_dead = False
    session._active_future = None
    session._stop_watchdog = lambda: None
    backend = SimpleNamespace(close=lambda: asyncio.sleep(0))
    session._acp_session = backend

    class _Loop:
        def stop(self):
            return None

        def call_soon_threadsafe(self, _callback):
            return None

        def close(self):
            return None

    loop = _Loop()
    session._loop = loop
    session._loop_thread = None
    session._drain_loop_before_close = lambda: None

    close_future: concurrent.futures.Future[None] = concurrent.futures.Future()
    close_future.set_exception(TimeoutError("transport close timed out"))

    def submit(coro, _loop):
        coro.close()
        return close_future

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", submit)

    with pytest.raises(TimeoutError, match="transport close timed out"):
        session.close()

    assert session._force_dead is True
    assert session._acp_session is backend
    assert session._loop is loop


def test_close_refuses_to_clear_loop_when_loop_thread_does_not_stop(
    monkeypatch,
):
    session = sa.SyncACPSession.__new__(sa.SyncACPSession)
    session._agent_type = "codex"
    session._force_dead = False
    session._active_future = None
    session._stop_watchdog = lambda: None
    backend = SimpleNamespace(close=lambda: asyncio.sleep(0))
    session._acp_session = backend

    class _Loop:
        def __init__(self):
            self.close_called = False

        def stop(self):
            return None

        def call_soon_threadsafe(self, _callback):
            return None

        def close(self):
            self.close_called = True

    class _Thread:
        def is_alive(self):
            return True

        def join(self, timeout):
            assert timeout == 5

    loop = _Loop()
    session._loop = loop
    session._loop_thread = _Thread()
    session._drain_loop_before_close = lambda: None

    def submit(coro, _loop):
        coro.close()
        future: concurrent.futures.Future[None] = concurrent.futures.Future()
        future.set_result(None)
        return future

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", submit)

    with pytest.raises(RuntimeError, match="event loop thread did not stop"):
        session.close()

    assert session._force_dead is True
    assert session._acp_session is backend
    assert session._loop is loop
    assert loop.close_called is False


@pytest.mark.asyncio
async def test_official_codex_startup_applies_explicit_model_before_ready(monkeypatch):
    """The official adapter ignores Zed CLI flags, so startup must set its config option."""
    session = sa.SyncACPSession.__new__(sa.SyncACPSession)
    session._agent_type = "codex"
    session._agent_cmd = "npx"
    session._agent_args = ["--yes", "@agentclientprotocol/codex-acp@1.1.2"]
    session._model_name = "gpt-5.6-sol"
    session._cwd = "/tmp"
    fake_acp_session = AsyncMock()
    fake_acp_session.start.return_value = "session-1"
    fake_acp_session.set_model.return_value = True
    monkeypatch.setattr(sa, "ACPSession", lambda **_kwargs: fake_acp_session)

    assert await session._start_session() == "session-1"

    fake_acp_session.set_model.assert_awaited_once_with("gpt-5.6-sol")


@pytest.mark.asyncio
async def test_official_codex_startup_applies_model_and_reasoning_effort(monkeypatch):
    session = sa.SyncACPSession.__new__(sa.SyncACPSession)
    session._agent_type = "codex"
    session._agent_cmd = "npx"
    session._agent_args = ["--yes", "@agentclientprotocol/codex-acp@1.1.2"]
    session._model_name = "gpt-5.6-sol/max"
    session._cwd = "/tmp"
    fake_acp_session = AsyncMock()
    fake_acp_session.start.return_value = "session-1"
    fake_acp_session.set_config_option.return_value = True
    monkeypatch.setattr(sa, "ACPSession", lambda **_kwargs: fake_acp_session)

    assert await session._start_session() == "session-1"

    assert fake_acp_session.set_config_option.await_args_list == [
        call("model", "gpt-5.6-sol"),
        call("reasoning_effort", "max"),
    ]


@pytest.mark.asyncio
async def test_official_codex_startup_closes_when_reasoning_effort_is_rejected(monkeypatch):
    session = sa.SyncACPSession.__new__(sa.SyncACPSession)
    session._agent_type = "codex"
    session._agent_cmd = "npx"
    session._agent_args = ["--yes", "@agentclientprotocol/codex-acp@1.1.2"]
    session._model_name = "gpt-5.6-sol/ultra"
    session._cwd = "/tmp"
    fake_acp_session = AsyncMock()
    fake_acp_session.start.return_value = "session-1"
    fake_acp_session.set_config_option.side_effect = [True, False]
    monkeypatch.setattr(sa, "ACPSession", lambda **_kwargs: fake_acp_session)

    with pytest.raises(RuntimeError, match="gpt-5.6-sol/ultra"):
        await session._start_session()

    fake_acp_session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_official_codex_startup_closes_and_fails_when_selected_model_is_rejected(monkeypatch):
    session = sa.SyncACPSession.__new__(sa.SyncACPSession)
    session._agent_type = "codex"
    session._agent_cmd = "npx"
    session._agent_args = ["--yes", "@agentclientprotocol/codex-acp@1.1.2"]
    session._model_name = "gpt-5.6-sol"
    session._cwd = "/tmp"
    fake_acp_session = AsyncMock()
    fake_acp_session.start.return_value = "session-1"
    fake_acp_session.set_model.return_value = False
    monkeypatch.setattr(sa, "ACPSession", lambda **_kwargs: fake_acp_session)

    with pytest.raises(RuntimeError, match="gpt-5.6-sol"):
        await session._start_session()

    fake_acp_session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_official_codex_startup_keeps_default_without_redundant_model_rpc(monkeypatch):
    session = sa.SyncACPSession.__new__(sa.SyncACPSession)
    session._agent_type = "codex"
    session._agent_cmd = "npx"
    session._agent_args = ["--yes", "@agentclientprotocol/codex-acp@1.1.2"]
    session._model_name = None
    session._cwd = "/tmp"
    fake_acp_session = AsyncMock()
    fake_acp_session.start.return_value = "session-1"
    monkeypatch.setattr(sa, "ACPSession", lambda **_kwargs: fake_acp_session)

    assert await session._start_session() == "session-1"

    fake_acp_session.set_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_traex_startup_applies_profile_and_reasoning_effort(monkeypatch):
    from src.acp.traex_selection import TraexRuntimeSelection

    session = sa.SyncACPSession.__new__(sa.SyncACPSession)
    session._agent_type = "traex"
    session._agent_cmd = "traex"
    session._agent_args = ["acp", "serve"]
    session._model_name = "c_o_new_thinking/max/max"
    session._cwd = "/tmp"
    fake_acp_session = AsyncMock()
    fake_acp_session.start.return_value = "session-1"
    fake_acp_session.set_config_option.return_value = True
    monkeypatch.setattr(sa, "ACPSession", lambda **_kwargs: fake_acp_session)
    monkeypatch.setattr(
        "src.acp.traex_selection.resolve_traex_runtime_selection",
        lambda _selection: TraexRuntimeSelection(
            model_id="c_o_new_thinking",
            backend_model_value="c_o_new_thinking__max",
            profile="max",
            effort="max",
        ),
    )

    assert await session._start_session() == "session-1"

    assert fake_acp_session.set_config_option.await_args_list == [
        call("model", "c_o_new_thinking__max"),
        call("reasoning_effort", "max"),
    ]


@pytest.mark.asyncio
async def test_traex_startup_closes_when_effort_is_rejected(monkeypatch):
    from src.acp.traex_selection import TraexRuntimeSelection

    session = sa.SyncACPSession.__new__(sa.SyncACPSession)
    session._agent_type = "traex"
    session._agent_cmd = "traex"
    session._agent_args = ["acp", "serve"]
    session._model_name = "c_o_new_thinking/max/max"
    session._cwd = "/tmp"
    fake_acp_session = AsyncMock()
    fake_acp_session.start.return_value = "session-1"
    fake_acp_session.set_config_option.side_effect = [True, False]
    monkeypatch.setattr(sa, "ACPSession", lambda **_kwargs: fake_acp_session)
    monkeypatch.setattr(
        "src.acp.traex_selection.resolve_traex_runtime_selection",
        lambda _selection: TraexRuntimeSelection(
            model_id="c_o_new_thinking",
            backend_model_value="c_o_new_thinking__max",
            profile="max",
            effort="max",
        ),
    )

    with pytest.raises(RuntimeError, match="Traex ACP rejected"):
        await session._start_session()

    fake_acp_session.close.assert_awaited_once()


def test_sync_adapter_startup_fail_log_has_err_type_and_err_repr(monkeypatch, caplog):
    """防回归：sync_adapter 启动失败且 str(err)=='' 时日志仍可定位。"""

    class _Cfg:
        acp_diagnostics_redact_enabled = True
        acp_diagnostics_redact_patterns = [r"(?i)token\s*[:=]\s*[^\s]+", r"(?i)api[_-]?key\s*[:=]\s*[^\s]+"]
        acp_diagnostics_redact_replacement = "***REDACTED***"
        acp_diagnostics_args_limit = 80
        acp_diagnostics_snippet_limit = 120
        acp_diagnostics_total_limit = 400

    monkeypatch.setattr(sa, "get_settings", lambda: _Cfg())

    class _EmptyStrErr(Exception):
        def __str__(self):
            return ""

    class _FakeSession:
        def __init__(self, agent_type: str, cwd: str, model_name=None, ttadk_use_pty=None):
            self._agent_cmd = "ttadk"
            self._agent_args = ["acp", "serve"]

        def start(self, startup_timeout: float = 60, **kwargs):
            raise _EmptyStrErr()

        def describe_agent(self):
            return "fake"

        def close(self):
            return None

    caplog.set_level("WARNING")
    with pytest.raises(sa.ACPStartupError):
        sa.start_session_with_retry(
            agent_type="ttadk_codex",
            cwd="/tmp",
            startup_timeout=0.1,
            model_name="gpt-5.2",
            session_cls=_FakeSession,
            log_failures=True,
        )

    logs = "\n".join([r.getMessage() for r in caplog.records])
    assert "Engine session start failed" in logs
    assert "err_type=" in logs
    assert "err_repr=" in logs


def test_sync_adapter_startup_fail_diagnostics_summary_redacted_and_truncated(monkeypatch, caplog):
    """防回归：sync_adapter 的 diagnostics_summary 必须脱敏且遵循 total_limit 截断。"""

    class _Cfg:
        acp_diagnostics_redact_enabled = True
        acp_diagnostics_redact_patterns = [r"(?i)token\s*[:=]\s*[^\s]+", r"(?i)api[_-]?key\s*[:=]\s*[^\s]+"]
        acp_diagnostics_redact_replacement = "***REDACTED***"
        acp_diagnostics_args_limit = 40
        acp_diagnostics_snippet_limit = 60
        acp_diagnostics_total_limit = 220

    monkeypatch.setattr(sa, "get_settings", lambda: _Cfg())

    class _EmptyStrErr(Exception):
        def __str__(self):
            return ""

    class _FakeSession:
        def __init__(self, agent_type: str, cwd: str, model_name=None, ttadk_use_pty=None):
            self._agent_cmd = "ttadk"
            # include sensitive args for redaction
            self._agent_args = ["acp", "serve", "--token=abc123", "--api_key=sk-secret-1234567890"]

        def start(self, startup_timeout: float = 60, **kwargs):
            e = _EmptyStrErr()
            e.stderr = "token=abc123 api_key=sk-secret-1234567890 " + "x" * 1000
            raise e

        def describe_agent(self):
            return "fake"

        def close(self):
            return None

    caplog.set_level("WARNING")
    with pytest.raises(sa.ACPStartupError):
        sa.start_session_with_retry(
            agent_type="ttadk_codex",
            cwd="/tmp",
            startup_timeout=0.1,
            model_name="gpt-5.2",
            session_cls=_FakeSession,
            log_failures=True,
        )

    logs = "\n".join([r.getMessage() for r in caplog.records])
    assert "diagnostics_summary=" in logs
    assert "abc123" not in logs
    assert "sk-secret" not in logs
    assert "***REDACTED***" in logs
    assert len(logs) <= 2000


def test_build_startup_diagnostics_has_error_text_and_fail_reason(monkeypatch):
    """防回归：diagnostics 必须包含 error_text/fail_reason，且 error_text 永不为空。"""

    class _Cfg:
        acp_diagnostics_redact_enabled = True
        acp_diagnostics_redact_patterns = []
        acp_diagnostics_redact_replacement = "***REDACTED***"
        acp_diagnostics_args_limit = 80
        acp_diagnostics_snippet_limit = 120
        acp_diagnostics_total_limit = 400

    monkeypatch.setattr(sa, "get_settings", lambda: _Cfg())

    class _EmptyStrErr(Exception):
        def __str__(self):
            return ""

    e = _EmptyStrErr()
    # Ensure classification sees invalid_model
    e.stderr = "Invalid model: foo. Model must be one of: bar,baz"

    d = sa.build_startup_diagnostics(
        agent_type="ttadk_codex",
        cwd="/tmp",
        model_name="foo",
        session=None,
        error=e,
        attempt=1,
        retries=2,
        timeout_s=0.1,
    )
    assert isinstance(d, dict)
    assert "error_text" in d
    assert str(d.get("error_text") or "").strip() != ""
    assert "fail_reason" in d
    assert str(d.get("fail_reason") or "").strip() in ("invalid_model", "start_failed")


def test_normalize_startup_diagnostics_empty_error_text_fallbacks(monkeypatch):
    """防回归：normalize 必须保证 error_text/fail_reason 非空，并在缺字段时兜底。"""

    class _Cfg:
        acp_diagnostics_redact_enabled = False
        acp_diagnostics_redact_patterns = []
        acp_diagnostics_redact_replacement = "***REDACTED***"
        acp_diagnostics_args_limit = 80
        acp_diagnostics_snippet_limit = 80
        acp_diagnostics_total_limit = 200

    d = diag.normalize_startup_diagnostics(
        {"cmd": "", "args": None, "stderr_snippet": "", "error": "", "error_text": "", "fail_reason": ""},
        get_settings_fn=lambda: _Cfg(),
    )
    assert isinstance(d, dict)
    assert str(d.get("error_text") or "").strip() != ""
    assert str(d.get("fail_reason") or "").strip() != ""
    assert isinstance(d.get("args"), list)


def test_normalize_startup_diagnostics_redacts_and_truncates(monkeypatch):
    """防回归：normalize 必须执行脱敏与截断（按 config 上限）。"""

    class _Cfg:
        acp_diagnostics_redact_enabled = True
        # Cover common API key shapes including hyphens.
        acp_diagnostics_redact_patterns = [r"(?i)token\s*[:=]\s*[^\s]+", r"sk-[^\s]{10,}"]
        acp_diagnostics_redact_replacement = "***REDACTED***"
        acp_diagnostics_args_limit = 40
        acp_diagnostics_snippet_limit = 60
        acp_diagnostics_total_limit = 200

    raw = {
        "cmd": "ttadk",
        "args": ["--token=abc123", "--x=" + ("y" * 200)],
        "stderr_snippet": "token=abc123 sk-secret-1234567890 " + ("x" * 2000),
        "error_text": "token=abc123 " + ("z" * 2000),
        "fail_reason": "invalid_model",
    }
    out = diag.normalize_startup_diagnostics(raw, get_settings_fn=lambda: _Cfg())
    blob = "\n".join(
        [
            str(out.get("cmd") or ""),
            " ".join([str(x) for x in (out.get("args") or [])]),
            str(out.get("stderr_snippet") or ""),
            str(out.get("error_text") or ""),
        ]
    )
    assert "abc123" not in blob
    assert "sk-secret" not in blob
    assert "***REDACTED***" in blob
    assert len(str(out.get("stderr_snippet") or "")) <= 200
    assert len(str(out.get("error_text") or "")) <= 200


def test_start_session_with_retry_ttadk_startup_error_empty_message_has_fail_reason_and_error_text(monkeypatch, caplog):
    """回归A：TTADKStartupError('') 时，启动失败日志必须包含非空 fail_reason/error_text（不再出现空串）。"""

    class _Cfg:
        acp_startup_retries = 1
        acp_diagnostics_redact_enabled = False
        acp_diagnostics_redact_patterns = []
        acp_diagnostics_redact_replacement = "***REDACTED***"
        acp_diagnostics_args_limit = 80
        acp_diagnostics_snippet_limit = 120
        acp_diagnostics_total_limit = 500

    monkeypatch.setattr(sa, "get_settings", lambda: _Cfg())

    from src.ttadk.manager import TTADKStartupError

    class _FakeSession:
        def __init__(self, agent_type: str, cwd: str, model_name=None, ttadk_use_pty=None):
            self._agent_cmd = "ttadk"
            self._agent_args = ["acp", "serve"]

        def start(self, startup_timeout: float = 60, **kwargs):
            # Empty message on purpose
            raise TTADKStartupError("")

        def describe_agent(self):
            return "fake"

        def close(self):
            return None

    caplog.set_level("WARNING")
    with pytest.raises(sa.ACPStartupError):
        sa.start_session_with_retry(
            agent_type="ttadk_claude",
            cwd="/tmp",
            startup_timeout=0.1,
            model_name="gpt-5.2",
            session_cls=_FakeSession,
            log_failures=True,
        )

    logs = "\n".join([r.getMessage() for r in caplog.records])
    assert "Engine session start failed" in logs
    assert "fail_reason=" in logs
    assert "error_text=" in logs


def test_start_session_with_retry_honors_single_attempt_override(monkeypatch):
    class _Cfg:
        acp_startup_retries = 4

    attempts: list[float] = []

    class _FakeSession:
        def __init__(self, agent_type: str, cwd: str, ttadk_use_pty=None):
            self._agent_cmd = agent_type
            self._agent_args = []

        def start(self, startup_timeout: float = 60, **_kwargs):
            attempts.append(startup_timeout)
            raise RuntimeError("startup failed")

        def describe_agent(self):
            return "fake"

        def close(self):
            return None

    monkeypatch.setattr(sa, "get_settings", lambda: _Cfg())

    with pytest.raises(sa.ACPStartupError):
        sa.start_session_with_retry(
            agent_type="coco",
            cwd="/tmp",
            startup_timeout=0.25,
            session_cls=_FakeSession,
            log_failures=False,
            retries=1,
        )

    assert attempts == [0.25]


def test_start_agent_session_with_diagnostics_attaches_non_empty_error_text(monkeypatch):
    """冻结：通用启动器失败时，异常上必须携带 diagnostics 且 error_text 非空。"""

    class _Cfg:
        acp_startup_retries = 1
        acp_diagnostics_redact_enabled = False
        acp_diagnostics_redact_patterns = []
        acp_diagnostics_redact_replacement = "***REDACTED***"
        acp_diagnostics_args_limit = 80
        acp_diagnostics_snippet_limit = 120
        acp_diagnostics_total_limit = 400

    monkeypatch.setattr(sa, "get_settings", lambda: _Cfg())

    class _EmptyStrErr(Exception):
        def __str__(self):
            return ""

    class _FakeSession:
        def __init__(self, agent_type: str, cwd: str, model_name=None, ttadk_use_pty=None):
            self.session_id = ""

        def start(self, startup_timeout: float = 60, **kwargs):
            raise _EmptyStrErr()

        def close(self):
            return None

        def describe_agent(self):
            return "fake"

    with pytest.raises(Exception) as ei:
        sa.start_agent_session_with_diagnostics(
            agent_type="ttadk_codex",
            cwd="/tmp",
            startup_timeout=0.1,
            model_name="gpt-5.2",
            session_cls=_FakeSession,
            log_failures=False,
        )

    d = getattr(ei.value, "diagnostics", None)
    assert isinstance(d, dict)
    assert str(d.get("error_text") or "").strip() != ""
    assert str(d.get("fail_reason") or "").strip() != ""


def test_start_session_with_retry_runtime_error_empty_message_has_fail_reason_and_error_text(monkeypatch, caplog):
    """回归B：RuntimeError('')（无 stdout/stderr/rc）时，日志仍必须包含非空 fail_reason/error_text。"""

    class _Cfg:
        acp_startup_retries = 1
        acp_diagnostics_redact_enabled = False
        acp_diagnostics_redact_patterns = []
        acp_diagnostics_redact_replacement = "***REDACTED***"
        acp_diagnostics_args_limit = 80
        acp_diagnostics_snippet_limit = 120
        acp_diagnostics_total_limit = 500

    monkeypatch.setattr(sa, "get_settings", lambda: _Cfg())

    class _FakeSession:
        def __init__(self, agent_type: str, cwd: str, model_name=None, ttadk_use_pty=None):
            self._agent_cmd = "ttadk"
            self._agent_args = ["acp", "serve"]

        def start(self, startup_timeout: float = 60, **kwargs):
            raise RuntimeError("")

        def describe_agent(self):
            return "fake"

        def close(self):
            return None

    caplog.set_level("WARNING")
    with pytest.raises(sa.ACPStartupError):
        sa.start_session_with_retry(
            agent_type="ttadk_claude",
            cwd="/tmp",
            startup_timeout=0.1,
            model_name="gpt-5.2",
            session_cls=_FakeSession,
            log_failures=True,
        )

    logs = "\n".join([r.getMessage() for r in caplog.records])
    assert "Engine session start failed" in logs
    assert "fail_reason=" in logs
    assert "error_text=" in logs


def test_no_cross_module_private_import_from_acp_diagnostics():
    """防回归：禁止在 src/ 下跨模块导入 src.acp.diagnostics 的私有 `_xxx`。

    约束说明：只检查 `import`/`from ... import ...` 行，避免误伤注释/字符串。
    """

    root = Path(__file__).resolve().parents[1]
    src_dir = root / "src"
    assert src_dir.is_dir()

    # Match only import lines to reduce false positives.
    pat = re.compile(r"^\s*from\s+(src\.acp\.diagnostics|\.diagnostics)\s+import\s+.*\b_\w+", re.IGNORECASE)
    alias_import_pat = re.compile(
        r"^\s*import\s+src\.acp\.diagnostics\s+as\s+(?P<alias>[A-Za-z_]\w*)\s*$",
        re.IGNORECASE,
    )
    alias_from_pat = re.compile(
        r"^\s*from\s+src\.acp\s+import\s+diagnostics\s+as\s+(?P<alias>[A-Za-z_]\w*)\s*$",
        re.IGNORECASE,
    )

    # Positive sanity checks: valid public API imports/usages should NOT match.
    assert pat.search("from src.acp.diagnostics import redact_text") is None
    assert re.search(r"\bd\._\w+\b", "d.redact_text('x')") is None

    offenders: list[str] = []
    for p in src_dir.rglob("*.py"):
        if p.name == "diagnostics.py":
            continue
        try:
            lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        aliases: set[str] = set()
        for i, line in enumerate(lines, start=1):
            # ignore commented-out imports
            if line.lstrip().startswith("#"):
                continue
            if pat.search(line):
                offenders.append(f"{p.relative_to(root)}:{i}:{line.strip()}")
                break

            m = alias_import_pat.search(line)
            if m:
                aliases.add(m.group("alias"))
                continue
            m = alias_from_pat.search(line)
            if m:
                aliases.add(m.group("alias"))
                continue

        # Also prevent `import src.acp.diagnostics as d; d._xxx` private usages.
        if aliases:
            for i, line in enumerate(lines, start=1):
                if line.lstrip().startswith("#"):
                    continue
                for a in aliases:
                    if re.search(rf"\b{re.escape(a)}\._\w+\b", line):
                        offenders.append(f"{p.relative_to(root)}:{i}:{line.strip()}")
                        aliases.clear()
                        break
                if not aliases:
                    break

    assert offenders == [], "发现跨模块私有导入（请改用 diagnostics 公共 API）：\n" + "\n".join(offenders)


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset module-level state between tests."""
    sa._update_attempted.clear()
    try:
        sa._supports_acp_serve.cache_clear()
    except Exception:
        pass
    yield
    sa._update_attempted.clear()
    try:
        sa._supports_acp_serve.cache_clear()
    except Exception:
        pass


def _fake_settings(**overrides):
    defaults = {
        "acp_auto_update": True,
        "acp_startup_retries": 2,
    }
    defaults.update(overrides)

    class FakeSettings:
        def __getattr__(self, name):
            if name in defaults:
                return defaults[name]
            if name == "get_acp_command":
                return lambda agent_type: ("", [])
            raise AttributeError(name)

    return FakeSettings()


# ── _auto_update_agent ──────────────────────────────────────────────


class TestAutoUpdateAgent:
    def test_successful_update(self, monkeypatch):
        """Auto-update returns True when subprocess exits 0."""
        monkeypatch.setattr(sa, "get_settings", lambda: _fake_settings())
        monkeypatch.setattr(
            sa.subprocess,
            "run",
            lambda *a, **kw: SimpleNamespace(returncode=0, stdout="Updated to v1.2.3", stderr=""),
        )
        assert sa._auto_update_agent("coco") is True

    def test_failed_update(self, monkeypatch):
        """Auto-update returns False when subprocess exits non-zero."""
        monkeypatch.setattr(sa, "get_settings", lambda: _fake_settings())
        monkeypatch.setattr(
            sa.subprocess,
            "run",
            lambda *a, **kw: SimpleNamespace(returncode=1, stdout="", stderr="network error"),
        )
        assert sa._auto_update_agent("coco") is False

    def test_update_exception(self, monkeypatch):
        """Auto-update returns False when subprocess raises."""
        monkeypatch.setattr(sa, "get_settings", lambda: _fake_settings())
        monkeypatch.setattr(
            sa.subprocess,
            "run",
            lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError("coco not found")),
        )
        assert sa._auto_update_agent("coco") is False

    def test_dedup_skips_second_attempt(self, monkeypatch):
        """Same command is only updated once per process lifecycle."""
        call_count = {"n": 0}

        def fake_run(*a, **kw):
            call_count["n"] += 1
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        monkeypatch.setattr(sa, "get_settings", lambda: _fake_settings())
        monkeypatch.setattr(sa.subprocess, "run", fake_run)

        assert sa._auto_update_agent("coco") is True
        assert sa._auto_update_agent("coco") is False  # deduped
        assert call_count["n"] == 1

    def test_config_disabled(self, monkeypatch):
        """Auto-update respects acp_auto_update=False."""
        monkeypatch.setattr(sa, "get_settings", lambda: _fake_settings(acp_auto_update=False))
        call_count = {"n": 0}

        def fake_run(*a, **kw):
            call_count["n"] += 1
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        monkeypatch.setattr(sa.subprocess, "run", fake_run)
        assert sa._auto_update_agent("coco") is False
        assert call_count["n"] == 0  # subprocess never called


# ── _resolve_with_auto_update ────────────────────────────────────────


class TestResolveWithAutoUpdate:
    def test_already_supported(self, monkeypatch):
        """No update attempted when ACP is already supported."""
        monkeypatch.setattr(sa, "_supports_acp_serve", lambda cmd: True)
        # Patch _auto_update_agent to track if it was called
        called = {"n": 0}
        orig = sa._auto_update_agent

        def tracking(*a, **kw):
            called["n"] += 1
            return orig(*a, **kw)

        monkeypatch.setattr(sa, "_auto_update_agent", tracking)
        assert sa._resolve_with_auto_update("coco") is True
        assert called["n"] == 0

    def test_update_fixes_support(self, monkeypatch):
        """After auto-update, re-probe succeeds."""
        iter([False, True])  # first call False, second True

        # Need to work with the lru_cache-decorated function
        monkeypatch.setattr(sa, "get_settings", lambda: _fake_settings())

        call_idx = {"n": 0}

        def fake_run(cmd, **kw):
            call_idx["n"] += 1
            if cmd[1:] == ["update"]:
                return SimpleNamespace(returncode=0, stdout="updated", stderr="")
            # acp serve -h probe
            if call_idx["n"] <= 1:
                return SimpleNamespace(returncode=1, stdout="", stderr="unknown command")
            return SimpleNamespace(returncode=0, stdout="Start the ACP server", stderr="")

        monkeypatch.setattr(sa.subprocess, "run", fake_run)
        assert sa._resolve_with_auto_update("coco") is True

    def test_update_fails_still_unsupported(self, monkeypatch):
        """Auto-update fails → returns False."""
        monkeypatch.setattr(sa, "get_settings", lambda: _fake_settings())
        monkeypatch.setattr(
            sa.subprocess,
            "run",
            lambda cmd, **kw: SimpleNamespace(returncode=1, stdout="", stderr="fail"),
        )
        assert sa._resolve_with_auto_update("coco") is False


# ── resolve_agent_spec ───────────────────────────────────────────────


class TestResolveAgentSpec:
    def test_coco_with_auto_update(self, monkeypatch):
        """resolve_agent_spec returns coco spec after successful auto-update."""
        monkeypatch.setattr(sa, "get_settings", lambda: _fake_settings())
        monkeypatch.setattr(sa, "_resolve_with_auto_update", lambda cmd: cmd == "coco")
        assert sa.resolve_agent_spec("coco") == ("coco", ["acp", "serve"])

    def test_coco_fails_after_update(self, monkeypatch):
        """resolve_agent_spec raises after auto-update fails."""
        monkeypatch.setattr(sa, "get_settings", lambda: _fake_settings())
        from src.acp.provider import ToolRegistry
        from src.acp.providers import tool_registry

        class FakeProvider:
            def __init__(self):
                self.name = "coco"
            def get_serve_command(self, model_name=None):
                return "coco", ["acp", "serve"]
            def get_fallback_command(self, model_name=None):
                return None
            def check_availability(self, model_name=None):
                return False

        # Pre-populate cache so it's a synchronous failure in get_serve_command
        # Patch the class boundary. Patching this inherited method on the singleton
        # makes monkeypatch restore it as an instance attribute, which then shadows
        # later class-level patches and leaks ordering into the next test.
        monkeypatch.setattr(ToolRegistry, "get_provider", lambda _self, _name: FakeProvider())
        tool_registry._set_availability_cache("coco", False)

        with pytest.raises(RuntimeError, match="not available for ACP mode|does not appear to support ACP server mode"):
            sa.resolve_agent_spec("coco")

    def test_config_override_bypasses_detection(self, monkeypatch):
        """Config overrides skip detection and auto-update entirely."""
        settings = _fake_settings()
        settings.get_acp_command = lambda agent_type: ("/custom/coco", ["serve"])
        monkeypatch.setattr(sa, "get_settings", lambda: settings)
        assert sa.resolve_agent_spec("coco") == ("/custom/coco", ["serve"])

    def test_ttadk_coco_no_model(self, monkeypatch):
        """resolve_agent_spec returns ttadk spec for ttadk_coco without model."""
        monkeypatch.setattr(sa, "get_settings", lambda: _fake_settings())
        assert sa.resolve_agent_spec("ttadk_coco") == (
            "python3",
            ["-m", "src.ttadk.wrapper", "ttadk", "code", "-t", "coco", "-a", "acp", "-a", "serve"],
        )

    def test_ttadk_claude_no_model(self, monkeypatch):
        """resolve_agent_spec returns ttadk spec for ttadk_claude without model."""
        monkeypatch.setattr(sa, "get_settings", lambda: _fake_settings())
        assert sa.resolve_agent_spec("ttadk_claude") == (
            "python3",
            ["-m", "src.ttadk.wrapper", "ttadk", "code", "-t", "claude", "-a", "acp", "-a", "serve"],
        )

    def test_ttadk_with_model(self, monkeypatch):
        """resolve_agent_spec returns ttadk spec with model parameter."""
        monkeypatch.setattr(sa, "get_settings", lambda: _fake_settings())
        assert sa.resolve_agent_spec("ttadk_coco", model_name="gpt-4") == (
            "python3",
            ["-m", "src.ttadk.wrapper", "ttadk", "code", "-t", "coco", "-m", "gpt-4", "-a", "acp", "-a", "serve"],
        )

    def test_ttadk_case_insensitive(self, monkeypatch):
        """resolve_agent_spec handles case-insensitive ttadk prefix."""
        monkeypatch.setattr(sa, "get_settings", lambda: _fake_settings())
        assert sa.resolve_agent_spec("TTADK_COCO") == (
            "python3",
            ["-m", "src.ttadk.wrapper", "ttadk", "code", "-t", "coco", "-a", "acp", "-a", "serve"],
        )


def test_build_startup_diagnostics_timeout_s_unparseable_is_none():
    class _Err(Exception):
        pass

    d1 = sa.build_startup_diagnostics(agent_type="coco", cwd=".", model_name=None, error=_Err("x"), timeout_s="abc")
    assert d1.get("timeout_s") is None
    d2 = sa.build_startup_diagnostics(agent_type="coco", cwd=".", model_name=None, error=_Err("x"), timeout_s="")
    assert d2.get("timeout_s") is None
    d3 = sa.build_startup_diagnostics(agent_type="coco", cwd=".", model_name=None, error=_Err("x"), timeout_s=object())
    assert d3.get("timeout_s") is None


def test_build_startup_diagnostics_timeout_s_parseable_is_float():
    class _Err(Exception):
        pass

    d1 = sa.build_startup_diagnostics(agent_type="coco", cwd=".", model_name=None, error=_Err("x"), timeout_s="2.0")
    assert isinstance(d1.get("timeout_s"), float)
    assert d1.get("timeout_s") == 2.0
    d2 = sa.build_startup_diagnostics(agent_type="coco", cwd=".", model_name=None, error=_Err("x"), timeout_s=1)
    assert isinstance(d2.get("timeout_s"), float)
    assert d2.get("timeout_s") == 1.0
    d3 = sa.build_startup_diagnostics(agent_type="coco", cwd=".", model_name=None, error=_Err("x"), timeout_s=1.5)
    assert isinstance(d3.get("timeout_s"), float)
    assert d3.get("timeout_s") == 1.5


def test_startup_diagnostics_builder_is_extracted_entrypoint():
    class _Err(Exception):
        pass

    error = _Err("boom")
    builder = sa.StartupDiagnosticsBuilder(
        agent_type="coco",
        cwd="/tmp/project",
        model_name="model-a",
        error=error,
        attempt=1,
        retries=2,
        timeout_s=3,
    )

    assert builder.build() == sa.build_startup_diagnostics(
        agent_type="coco",
        cwd="/tmp/project",
        model_name="model-a",
        error=error,
        attempt=1,
        retries=2,
        timeout_s=3,
    )


def test_startup_diagnostics_redaction_enabled_masks_tokens(monkeypatch):
    """脱敏开启：args/stdout/stderr 中的敏感片段应被掩码。"""
    settings = _fake_settings(
        acp_diagnostics_redact_enabled=True,
        acp_diagnostics_redact_replacement="***REDACTED***",
        acp_diagnostics_redact_patterns=[r"(?i)bearer\s+[^\s]+", r"sk-[A-Za-z0-9]{10,}"],
        acp_diagnostics_args_limit=200,
        acp_diagnostics_snippet_limit=200,
        acp_diagnostics_total_limit=2000,
    )
    monkeypatch.setattr(sa, "get_settings", lambda: settings)

    class _Err(Exception):
        pass

    e = _Err("boom")
    # simulate error carrying stdout/stderr
    e.stdout = "hello Bearer abcdef0123456789"  # type: ignore[attr-defined]
    e.stderr = "sk-0123456789abcdefTOKEN"  # type: ignore[attr-defined]

    d = sa.build_startup_diagnostics(
        agent_type="coco",
        cwd=".",
        model_name=None,
        error=e,
        session=None,
    )
    out = sa.format_startup_diagnostics(d)
    assert "Bearer abcdef" not in out
    assert "sk-012345" not in out
    assert "***REDACTED***" in out


def test_startup_diagnostics_redaction_disabled_keeps_tokens(monkeypatch):
    """脱敏关闭：输出允许包含原始片段（仍受长度截断）。"""
    token1 = "Bearer abcdef0123456789"
    token2 = "sk-0123456789abcdefTOKEN"
    settings = _fake_settings(
        acp_diagnostics_redact_enabled=False,
        acp_diagnostics_redact_replacement="***REDACTED***",
        acp_diagnostics_redact_patterns=[r"(?i)bearer\s+[^\s]+", r"sk-[A-Za-z0-9]{10,}"],
        acp_diagnostics_args_limit=500,
        acp_diagnostics_snippet_limit=500,
        acp_diagnostics_total_limit=5000,
    )
    monkeypatch.setattr(sa, "get_settings", lambda: settings)

    class _Err(Exception):
        pass

    e = _Err("boom")
    e.stdout = f"hello {token1}"  # type: ignore[attr-defined]
    e.stderr = token2  # type: ignore[attr-defined]

    d = sa.build_startup_diagnostics(agent_type="coco", cwd=".", model_name=None, error=e, session=None)
    out = sa.format_startup_diagnostics(d)
    assert token1 in out
    assert token2 in out


def test_startup_diagnostics_truncation_marks_truncated(monkeypatch):
    """超长输出：应出现截断标记。"""
    settings = _fake_settings(
        acp_diagnostics_redact_enabled=True,
        acp_diagnostics_redact_patterns=[],
        acp_diagnostics_redact_replacement="***REDACTED***",
        acp_diagnostics_args_limit=50,
        acp_diagnostics_snippet_limit=20,
        acp_diagnostics_total_limit=200,
    )
    monkeypatch.setattr(sa, "get_settings", lambda: settings)

    class _Err(Exception):
        pass

    e = _Err("boom")
    e.stdout = "x" * 200  # type: ignore[attr-defined]
    d = sa.build_startup_diagnostics(agent_type="coco", cwd=".", model_name=None, error=e, session=None)
    out = sa.format_startup_diagnostics(d)
    assert "truncated" in out


def test_diagnostics_public_api_config_injection(monkeypatch):
    """公共 API：get_diagnostics_config(get_settings_fn=...) 必须可注入且异常安全。"""

    class Fake:
        acp_diagnostics_redact_enabled = True
        acp_diagnostics_redact_patterns = [r"sk-[A-Za-z0-9]{10,}"]
        acp_diagnostics_redact_replacement = "XXX"
        acp_diagnostics_args_limit = "123"
        acp_diagnostics_snippet_limit = 7
        acp_diagnostics_total_limit = 9

    cfg = diag.get_diagnostics_config(get_settings_fn=lambda: Fake())
    assert cfg.redact_enabled is True
    assert cfg.redact_replacement == "XXX"
    assert cfg.args_limit == 123
    assert cfg.snippet_limit == 7
    assert cfg.total_limit == 9
    assert cfg.redact_patterns == [r"sk-[A-Za-z0-9]{10,}"]


@pytest.mark.parametrize(
    ("settings_overrides", "expected"),
    [
        ({}, (600, 240, 2000)),
        (
            {
                "acp_diagnostics_args_limit": "not-int",
                "acp_diagnostics_snippet_limit": "bad",
                "acp_diagnostics_total_limit": object(),
            },
            (600, 240, 2000),
        ),
        (
            {
                "acp_diagnostics_args_limit": -1,
                "acp_diagnostics_snippet_limit": -2,
                "acp_diagnostics_total_limit": -3,
            },
            (600, 240, 2000),
        ),
        (
            {
                "acp_diagnostics_args_limit": 0,
                "acp_diagnostics_snippet_limit": 0,
                "acp_diagnostics_total_limit": 0,
            },
            (0, 0, 0),
        ),
        (
            {
                "acp_diagnostics_args_limit": "123",
                "acp_diagnostics_snippet_limit": 45,
                "acp_diagnostics_total_limit": 678,
            },
            (123, 45, 678),
        ),
    ],
)
def test_diagnostics_config_limit_boundaries(settings_overrides, expected):
    """SSOT 常量迁移后，配置缺失/非法/负数/0/显式值边界保持不变。"""
    cfg = diag.get_diagnostics_config(get_settings_fn=lambda: _fake_settings(**settings_overrides))

    assert (cfg.args_limit, cfg.snippet_limit, cfg.total_limit) == expected


@pytest.mark.parametrize(
    ("settings_overrides", "expected_stdout"),
    [
        ({}, "x" * 240 + "…(truncated)"),
        ({"acp_diagnostics_snippet_limit": "bad"}, "x" * 240 + "…(truncated)"),
        ({"acp_diagnostics_snippet_limit": -1}, "x" * 240 + "…(truncated)"),
        ({"acp_diagnostics_snippet_limit": 0}, "x" * 240 + "…(truncated)"),
        ({"acp_diagnostics_snippet_limit": 12}, "x" * 12 + "…(truncated)"),
    ],
)
def test_sync_adapter_diagnostics_snippet_limit_boundaries(monkeypatch, settings_overrides, expected_stdout):
    """sync_adapter 通过 diagnostics SSOT 获取 snippet 默认值并保持边界语义。"""
    monkeypatch.setattr(sa, "get_settings", lambda: _fake_settings(**settings_overrides))

    class _Err(Exception):
        pass

    e = _Err("boom")
    e.stdout = "x" * 300  # type: ignore[attr-defined]

    d = sa.build_startup_diagnostics(agent_type="coco", cwd=".", model_name=None, error=e, session=None)

    assert d["stdout_snippet"] == expected_stdout


def test_diagnostics_public_api_redact_text_illegal_pattern_is_safe():
    """公共 API：非法正则不应抛异常。"""
    s = diag.redact_text("hello sk-0123456789abcdefTOKEN", patterns=["("], replacement="***")
    # best-effort: may or may not redact, but must return a string and not raise
    assert isinstance(s, str)


def test_diagnostics_redact_text_compile_cache_reuses_compiled_patterns(monkeypatch):
    """性能护栏：相同 patterns+replacement 多次调用不应重复编译。"""
    # best-effort: tests should not rely on private internals, but cache clear makes it stable
    try:
        diag._compile_redaction_patterns.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass

    orig_compile = diag.re.compile
    count = {"n": 0}

    def _counting_compile(p, *a, **kw):
        count["n"] += 1
        return orig_compile(p, *a, **kw)

    monkeypatch.setattr(diag.re, "compile", _counting_compile)

    patterns = [r"Bearer\s+[^\s]+", r"sk-[A-Za-z0-9]{10,}"]
    s1 = diag.redact_text("Bearer abc sk-0123456789abcdefTOKEN", patterns=patterns, replacement="X")
    s2 = diag.redact_text("Bearer abc sk-0123456789abcdefTOKEN", patterns=patterns, replacement="X")
    assert isinstance(s1, str)
    assert isinstance(s2, str)
    assert count["n"] == len(patterns)


def test_diagnostics_redact_text_compile_cache_replacement_change_does_not_recompile(monkeypatch):
    """性能护栏：patterns 不变、仅 replacement 变化时不应重复编译。"""
    try:
        diag._compile_redaction_patterns.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass

    orig_compile = diag.re.compile
    count = {"n": 0}

    def _counting_compile(p, *a, **kw):
        count["n"] += 1
        return orig_compile(p, *a, **kw)

    monkeypatch.setattr(diag.re, "compile", _counting_compile)

    patterns = [r"Bearer\s+[^\s]+", r"sk-[A-Za-z0-9]{10,}"]
    _ = diag.redact_text("Bearer abc sk-0123456789abcdefTOKEN", patterns=patterns, replacement="X")
    _ = diag.redact_text("Bearer abc sk-0123456789abcdefTOKEN", patterns=patterns, replacement="Y")
    assert count["n"] == len(patterns)


def test_diagnostics_redact_text_mixed_valid_invalid_patterns_still_redacts():
    """鲁棒性：混合合法/非法 pattern 时，合法规则仍应生效且不抛异常。"""
    text = "hello Bearer abcdef0123456789"
    out = diag.redact_text(text, patterns=["(", r"(?i)bearer\s+[^\s]+"], replacement="***")
    assert isinstance(out, str)
    assert "Bearer abcdef" not in out


def test_diagnostics_safe_extract_returns_default_and_logs_debug(caplog):
    """diagnostics 场景的安全提取 helper 应保留容错语义并留下可追溯日志。"""
    caplog.set_level("DEBUG", logger="src.acp.diagnostics")

    def boom():
        raise RuntimeError("diagnostic source failed")

    assert diag.safe_extract(boom, default="fallback", log_msg="collect session failed") == "fallback"
    assert "collect session failed" in caplog.text
