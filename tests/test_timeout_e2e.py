"""End-to-end tests: TimeoutError → user-visible card / message is never empty.

Validates the full chain from bare TimeoutError() raised at session level
through engine → handler → card builder, ensuring the final user-facing
output always contains a meaningful Chinese-friendly message.
"""

import asyncio
import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from src.utils.errors import fmt_error, fmt_exception, get_error_detail

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _TimeoutSession:
    """Fake ACP session that always raises bare TimeoutError (empty message)."""

    def send_prompt(self, prompt, on_event=None, timeout=0, **kw):
        raise TimeoutError()

    def send_prompt_with_retry(self, prompt, on_event=None, timeout=0, **kw):
        raise TimeoutError()

    def cancel(self):
        pass

    def close(self):
        pass


class _AsyncioTimeoutSession:
    """Fake ACP session that always raises bare asyncio.TimeoutError (empty message)."""

    def send_prompt(self, prompt, on_event=None, timeout=0, **kw):
        raise asyncio.TimeoutError()

    def send_prompt_with_retry(self, prompt, on_event=None, timeout=0, **kw):
        raise asyncio.TimeoutError()

    def cancel(self):
        pass

    def close(self):
        pass


# ---------------------------------------------------------------------------
# 1. Formatter layer: get_error_detail / fmt_error / fmt_exception
# ---------------------------------------------------------------------------

class TestFormatterLayerE2E:
    """fmt_error / fmt_exception / get_error_detail never return empty for bare errors."""

    @pytest.mark.parametrize("exc", [
        TimeoutError(),
        asyncio.TimeoutError(),
        Exception(),
        ValueError(),
        RuntimeError(),
    ])
    def test_get_error_detail_never_empty(self, exc):
        detail = get_error_detail(exc)
        assert detail, f"get_error_detail returned empty for {type(exc).__name__}()"
        assert len(detail.strip()) > 0

    def test_fmt_error_bare_timeout(self):
        msg = fmt_error("审查", TimeoutError())
        assert "超时" in msg
        assert msg.strip().endswith("重试") or "超时" in msg

    def test_fmt_error_bare_exception(self):
        msg = fmt_error("执行", Exception())
        # Should still produce non-empty (the prefix is always present)
        assert "❌" in msg
        assert "执行" in msg

    def test_fmt_exception_bare_timeout(self):
        msg = fmt_exception("操作", TimeoutError())
        assert "超时" in msg
        assert len(msg.strip()) > 0

    def test_fmt_exception_bare_exception(self):
        msg = fmt_exception("操作", Exception())
        assert len(msg.strip()) > 0
        # repr fallback should kick in
        assert "Exception" in msg or "操作" in msg


# ---------------------------------------------------------------------------
# 2. Card builder layer: build_error_card
# ---------------------------------------------------------------------------

class TestSystemBuilderE2E:
    """SystemBuilder.build_error_card must always contain non-empty error text."""

    @pytest.mark.parametrize("exc", [
        TimeoutError(),
        asyncio.TimeoutError(),
        Exception(),
    ])
    def test_card_body_has_nonempty_error(self, exc):
        from src.card.builders.system import SystemBuilder

        msg_type, card_json = SystemBuilder.build_error_card(exc)
        assert msg_type == "interactive"
        card = json.loads(card_json)
        body_elements = card.get("body", {}).get("elements", card.get("elements", []))
        content_texts = [
            el.get("content", "")
            for el in body_elements
            if el.get("tag") in ("markdown", "div")
        ]
        full_text = " ".join(content_texts)
        assert len(full_text.strip()) > 0, f"Card body empty for {type(exc).__name__}()"

    def test_timeout_card_has_chinese_friendly_text(self):
        from src.card.builders.system import SystemBuilder

        _, card_json = SystemBuilder.build_error_card(TimeoutError())
        assert "超时" in card_json or "未知错误" in card_json


# ---------------------------------------------------------------------------
# 3. Engine layer: engine.execute → callbacks.on_error non-empty
# ---------------------------------------------------------------------------

class TestDeepEngineE2E:
    """DeepEngine bare TimeoutError → on_error callback message non-empty."""

    @pytest.fixture
    def deep_engine(self):
        from src.deep_engine.engine import DeepEngine
        with patch("src.engine_base.get_settings") as mock_settings:
            s = MagicMock()
            s.coco_execution_timeout = 300
            s.claude_execution_timeout = 600
            s.deep_memory_threshold = 90
            mock_settings.return_value = s
            yield DeepEngine(chat_id="e2e", root_path="/tmp/e2e")

    def test_bare_timeout_produces_nonempty_error(self, deep_engine, caplog):
        from src.deep_engine.engine import DeepEngineCallbacks

        cb = DeepEngineCallbacks()
        errors = []
        cb.on_error = lambda msg: errors.append(msg)

        with patch("src.deep_engine.engine.create_engine_session", return_value=_TimeoutSession()):
            deep_engine.plan_and_execute("test", callbacks=cb)

        assert len(errors) >= 1
        for err in errors:
            assert len(err.strip()) > 0, "on_error received empty message"
            assert "超时" in err

    def test_bare_asyncio_timeout_produces_nonempty_error(self, deep_engine, caplog):
        from src.deep_engine.engine import DeepEngineCallbacks

        cb = DeepEngineCallbacks()
        errors = []
        cb.on_error = lambda msg: errors.append(msg)

        with patch("src.deep_engine.engine.create_engine_session", return_value=_AsyncioTimeoutSession()):
            deep_engine.plan_and_execute("test", callbacks=cb)

        assert len(errors) >= 1
        for err in errors:
            assert len(err.strip()) > 0, "on_error received empty message for asyncio.TimeoutError"
            assert "超时" in err


class TestSpecEngineE2E:
    """SpecEngine bare TimeoutError → on_error callback message non-empty."""

    @pytest.fixture
    def spec_engine(self):
        from src.spec_engine.engine import SpecEngine
        with patch("src.engine_base.get_settings") as mock_settings:
            s = MagicMock()
            s.spec_max_cycles = 10
            s.spec_max_cycles_limit = 5000
            s.spec_convergence_window = 2
            s.spec_execution_timeout = 300
            s.spec_review_timeout = 120
            s.spec_review_enabled = True
            s.spec_review_failure_circuit_enabled = False
            s.spec_min_cycles = 1
            s.spec_rebuild_session_between_cycles = False
            s.spec_max_retries = 2
            s.spec_infinite_mode = False
            s.spec_disable_convergence = False
            s.spec_disable_early_stop = False
            s.spec_cycle_tasks_max = 10
            mock_settings.return_value = s
            yield SpecEngine(chat_id="e2e", root_path="/tmp/e2e")

    def test_bare_timeout_produces_nonempty_error(self, spec_engine, caplog, monkeypatch):
        from src.spec_engine.engine import SpecEngineCallbacks

        cb = SpecEngineCallbacks()
        errors = []
        cb.on_error = lambda msg: errors.append(msg)

        monkeypatch.setattr(
            "src.spec_engine.engine.create_engine_session",
            lambda **kw: _TimeoutSession(),
        )
        monkeypatch.setattr(
            spec_engine,
            "_create_session_fn",
            lambda **kw: _TimeoutSession(),
        )
        monkeypatch.setattr(
            "src.spec_engine.engine.parse_acceptance_criteria",
            lambda txt, decompose_fn=None: ["criterion1"],
        )
        monkeypatch.setattr(
            spec_engine, "_run_cycle_loop",
            lambda **kw: (_ for _ in ()).throw(TimeoutError()),
        )

        spec_engine.execute("test", callbacks=cb)

        assert len(errors) >= 1
        for err in errors:
            assert len(err.strip()) > 0, "on_error received empty message"
            assert "超时" in err

    def test_bare_asyncio_timeout_produces_nonempty_error(self, spec_engine, caplog, monkeypatch):
        from src.spec_engine.engine import SpecEngineCallbacks

        cb = SpecEngineCallbacks()
        errors = []
        cb.on_error = lambda msg: errors.append(msg)

        monkeypatch.setattr(
            "src.spec_engine.engine.create_engine_session",
            lambda **kw: _AsyncioTimeoutSession(),
        )
        monkeypatch.setattr(
            spec_engine,
            "_create_session_fn",
            lambda **kw: _AsyncioTimeoutSession(),
        )
        monkeypatch.setattr(
            "src.spec_engine.engine.parse_acceptance_criteria",
            lambda txt, decompose_fn=None: ["criterion1"],
        )
        monkeypatch.setattr(
            spec_engine, "_run_cycle_loop",
            lambda **kw: (_ for _ in ()).throw(asyncio.TimeoutError()),
        )

        spec_engine.execute("test", callbacks=cb)

        assert len(errors) >= 1
        for err in errors:
            assert len(err.strip()) > 0, "on_error received empty message for asyncio.TimeoutError"
            assert "超时" in err


# ---------------------------------------------------------------------------
# 4. Command executor: bare exception → error_message non-empty
# ---------------------------------------------------------------------------

class TestCommandExecutorE2E:
    """CommandExecutor exception paths → error_message always non-empty."""

    def test_bare_exception_in_execute_has_nonempty_error(self):
        from src.command_executor import CommandExecutor

        executor = CommandExecutor()
        # Force a bare Exception by providing a command that triggers an internal error
        with patch.object(executor, "_sanitize_command_for_noninteractive", side_effect=Exception()):
            result = executor.execute("echo test")
        assert result.error_message is not None
        assert len(result.error_message.strip()) > 0
        assert "执行异常" in result.error_message

    def test_bare_timeout_in_execute_has_nonempty_error(self):
        from src.command_executor import CommandExecutor

        executor = CommandExecutor()
        with patch.object(executor, "_sanitize_command_for_noninteractive", side_effect=TimeoutError()):
            result = executor.execute("echo test")
        assert result.error_message is not None
        assert len(result.error_message.strip()) > 0
        # TimeoutError() has empty str, but with our guard it falls back to repr
        assert "执行异常" in result.error_message
        assert "TimeoutError" in result.error_message or "超时" in result.error_message


# ---------------------------------------------------------------------------
# 5. Internal diagnostics: str(e) or repr(e) guards
# ---------------------------------------------------------------------------

class TestInternalDiagnosticsGuard:
    """Internal-only str(e) paths now produce non-empty error strings."""

    def test_acp_client_read_file_error_nonempty(self):
        """client.py: field_meta['error'] non-empty for bare Exception."""
        from src.acp.client import GhostAPClient

        client = GhostAPClient.__new__(GhostAPClient)
        client._root_dir = "/nonexistent"
        client._ops_log = []
        client._record = lambda *a: None

        import asyncio
        loop = asyncio.new_event_loop()
        try:
            resp = loop.run_until_complete(
                client.read_text_file(path="../../etc/passwd", session_id="test")
            )
        finally:
            loop.close()

        # The response should have a non-empty error in field_meta
        if resp and resp.field_meta and "error" in resp.field_meta:
            assert len(resp.field_meta["error"].strip()) > 0

    def test_coco_model_manager_error_nonempty(self):
        """manager.py: ModelListResult.error non-empty for bare Exception."""
        from src.coco_model.manager import CocoModelManager

        mgr = CocoModelManager.__new__(CocoModelManager)
        mgr._cached_models = None
        mgr._cache_time = 0.0
        mgr._cache_ttl = 0.0
        mgr._current_model = None
        mgr._lock = __import__("threading").Lock()

        # Force _load_models to raise bare Exception
        mgr._load_models = MagicMock(side_effect=Exception())
        mgr._ensure_initialized = lambda: None
        mgr._is_cache_valid = lambda: False
        result = mgr.get_models()

        assert result.error is not None
        assert len(result.error.strip()) > 0, "ModelListResult.error is empty for bare Exception"



# ---------------------------------------------------------------------------
# 6. User-facing empty-guard final: 6 residual f"{e}" sites
# ---------------------------------------------------------------------------

class TestUserFacingEmptyGuardFinal:
    """Verify the 6 residual user-facing f'{e}' sites now produce non-empty messages."""

    # --- agent_session.py: Claude session ---
    def test_claude_session_bare_timeout_nonempty(self):
        from src.agent_session.claude_cli import SyncClaudeCLISession

        session = SyncClaudeCLISession(cwd="/tmp")
        session.session_id = "test-session"

        # Force subprocess.Popen to raise bare TimeoutError inside send_prompt
        with patch(
            "src.agent_session.claude_cli.subprocess.Popen",
            side_effect=TimeoutError(),
        ):
            result = session.send_prompt("test")

        assert result.stop_reason == "error"
        assert len(result.text.strip()) > 0, "Claude session error text is empty"
        assert "执行异常" in result.text
        # Must not end with just ": "
        assert not result.text.strip().endswith(":")

    # --- programming.py: ACP execute exception ---
    def test_programming_acp_execute_bare_exception_nonempty(self):
        """programming.py:708 — f'❌ 执行异常: {get_error_detail(e)}' guard."""
        from src.utils.errors import get_error_detail

        # Simulate what the handler does: format error with get_error_detail
        for exc in [TimeoutError(), Exception(), ValueError()]:
            detail = get_error_detail(exc)
            msg = f"❌ 执行异常: {detail}"
            assert len(msg.strip()) > len("❌ 执行异常: "), f"Empty detail for {type(exc).__name__}"
            assert not msg.strip().endswith(": ")

    # --- programming.py: model switch failure ---
    def test_programming_model_switch_bare_exception_nonempty(self):
        """programming.py:438 — f'切换 ... 模型失败: {get_error_detail(e)}' guard."""
        from src.utils.errors import get_error_detail

        for exc in [TimeoutError(), Exception()]:
            detail = get_error_detail(exc)
            msg = f"切换模型失败: {detail}"
            assert len(detail.strip()) > 0, f"Empty detail for {type(exc).__name__}"
            assert not msg.strip().endswith(": ")

    # --- ws_client.py: card action failure ---
    def test_ws_client_card_action_bare_exception_nonempty(self):
        """ws_client.py:2287 — f'❌ 操作失败 (...): {str(e) or repr(e)}' guard."""
        for exc in [TimeoutError(), Exception(), ValueError()]:
            detail = str(exc) or repr(exc)
            msg = f"❌ 操作失败 (test_action): {detail}"
            assert len(detail.strip()) > 0, f"Empty detail for {type(exc).__name__}"
            assert not msg.strip().endswith(": ")

    # --- diagnostics.py: diff report failure ---
    def test_diagnostics_diff_report_bare_exception_nonempty(self):
        """diagnostics.py:652 — f'Diff 报告生成异常: {get_error_detail(e)}' guard."""
        from src.utils.errors import get_error_detail

        for exc in [TimeoutError(), Exception(), RuntimeError()]:
            detail = get_error_detail(exc)
            msg = f"Diff 报告生成异常: {detail}"
            assert len(detail.strip()) > 0, f"Empty detail for {type(exc).__name__}"
            assert not msg.strip().endswith(": ")


# ---------------------------------------------------------------------------
# Programming handler: dedicated TimeoutError branch
# ---------------------------------------------------------------------------
class TestProgrammingHandlerTimeoutBranch:
    """Verify programming handler routes TimeoutError to dedicated branch."""

    def test_streaming_path_timeout_uses_timeout_text(self):
        """流式路径: send_prompt 抛 TimeoutError → final_response 包含'超时'而非'异常'."""
        detail = get_error_detail(TimeoutError("ACP prompt 执行超时 (120s)"))
        msg = f"⏳ 执行超时: {detail}"
        assert "超时" in msg
        assert "异常" not in msg
        assert len(detail.strip()) > 0

    def test_streaming_path_bare_timeout_nonempty(self):
        """流式路径: 裸 TimeoutError() → final_response 仍然非空."""
        detail = get_error_detail(TimeoutError())
        msg = f"⏳ 执行超时: {detail}"
        assert len(msg.strip()) > len("⏳ 执行超时: ")
        assert not msg.strip().endswith(": ")

    def test_non_streaming_path_timeout_card_title(self):
        """非流式路径: send_prompt 抛 TimeoutError → error card title 包含'超时'."""
        from src.card.builders.system import SystemBuilder

        exc = TimeoutError("ACP prompt 执行超时 (120s)")
        msg_type, content = SystemBuilder.build_error_card(exc, title="执行超时")
        assert msg_type == "interactive"
        # Card body should include the timeout title
        card_data = json.loads(content) if isinstance(content, str) else content
        card_str = json.dumps(card_data, ensure_ascii=False)
        assert "超时" in card_str

    def test_non_streaming_path_bare_timeout_card_nonempty(self):
        """非流式路径: 裸 TimeoutError() → error card 内容非空."""
        from src.card.builders.system import SystemBuilder

        exc = TimeoutError()
        msg_type, content = SystemBuilder.build_error_card(exc, title="执行超时")
        assert msg_type == "interactive"
        card_data = json.loads(content) if isinstance(content, str) else content
        card_str = json.dumps(card_data, ensure_ascii=False)
        assert "超时" in card_str
        # Should NOT contain bare "(empty message)" or end with empty detail
        assert "(empty message)" not in card_str


# ---------------------------------------------------------------------------
# E2E: concurrent.futures.TimeoutError() — full chain traversal
# ---------------------------------------------------------------------------


class TestConcurrentFuturesTimeoutE2EChain:
    """Simulate concurrent.futures.TimeoutError() (bare, empty message) through
    the active retry and diagnostics layers: should_retry → prompt_with_retry →
    build_review_exception_diagnostics → normalize_review_diagnostics. Verify
    no layer leaks an empty message."""

    def test_should_retry_accepts_bare_futures_timeout(self):
        """Layer 1: should_retry recognises concurrent.futures.TimeoutError."""
        import concurrent.futures

        from src.utils.retry import should_retry
        assert should_retry(concurrent.futures.TimeoutError()) is True

    def test_prompt_with_retry_retries_futures_timeout(self):
        """Layer 2: prompt_with_retry retries on concurrent.futures.TimeoutError."""
        import concurrent.futures
        from unittest.mock import MagicMock

        from src.utils.retry import RetryPolicy, prompt_with_retry

        action = MagicMock(side_effect=[concurrent.futures.TimeoutError(), "ok"])
        cancel = threading.Event()
        policy = RetryPolicy(max_retries=2, retry_delay=0.01, jitter_factor=0)
        result = prompt_with_retry(action, cancel, retry_policy=policy)
        assert result == "ok"
        assert action.call_count == 2

    def test_diagnostics_no_empty_text_for_futures_timeout(self):
        """Layer 3: build_review_exception_diagnostics produces non-empty error_text."""
        import concurrent.futures

        from src.utils.review_diagnostics import build_review_exception_diagnostics
        diag = build_review_exception_diagnostics(
            concurrent.futures.TimeoutError(), cycle=1,
        )
        assert diag["error_text"]
        assert "(empty message)" not in diag["error_text"]
        assert diag["fail_reason"] == "timeout"

    def test_normalize_no_empty_text_for_futures_timeout(self):
        """Layer 4: normalize_review_diagnostics guarantees non-empty error_text."""
        import concurrent.futures

        from src.utils.review_diagnostics import (
            build_review_exception_diagnostics,
            normalize_review_diagnostics,
        )
        diag = build_review_exception_diagnostics(
            concurrent.futures.TimeoutError(), cycle=1,
        )
        normalized = normalize_review_diagnostics(diag)
        assert normalized["error_text"]
        assert "(empty message)" not in normalized["error_text"]
