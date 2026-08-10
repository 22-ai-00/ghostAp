"""Tests: engine TimeoutError handling in top-level execution paths.

Validates that TimeoutError is caught separately from generic Exception, logged at
WARNING level (not ERROR), and produces error messages containing '超时'.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.acp.models import PromptResult
from src.acp.outcome import classify_prompt_result
from src.deep_engine.engine import DeepEngine, DeepEngineCallbacks
from src.deep_engine.models import DeepProjectStatus
from src.engine_base import EngineRunState
from src.spec_engine.engine import SpecEngine, SpecEngineCallbacks
from src.spec_engine.models import SpecProjectStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _TimeoutSession:
    """A fake session that raises TimeoutError on every prompt call."""

    def send_prompt(self, prompt, on_event=None, timeout=0, **kw):
        raise TimeoutError()

    def send_prompt_with_retry(self, prompt, on_event=None, timeout=0, **kw):
        raise TimeoutError()

    def cancel(self):
        pass

    def close(self):
        pass


@pytest.fixture
def deep_engine():
    with patch("src.engine_base.get_settings") as mock_settings:
        s = MagicMock()
        s.coco_execution_timeout = 300
        s.claude_execution_timeout = 600
        s.deep_memory_threshold = 90
        mock_settings.return_value = s
        yield DeepEngine(chat_id="t", root_path="/tmp/test")


@pytest.fixture
def spec_engine():
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
        yield SpecEngine(chat_id="t", root_path="/tmp/test")


# ===========================================================================
# Deep Engine
# ===========================================================================

class TestDeepEngineTimeout:

    def test_plan_and_execute_timeout(self, deep_engine, caplog):
        """plan_and_execute: TimeoutError → WARNING log + '执行超时' + FAILED status."""
        cb = DeepEngineCallbacks()
        errors = []
        cb.on_error = lambda msg: errors.append(msg)

        with patch("src.deep_engine.engine.create_engine_session", return_value=_TimeoutSession()):
            project = deep_engine.plan_and_execute("do something", callbacks=cb)

        assert project.status == DeepProjectStatus.FAILED
        assert any("执行超时" in e for e in errors)
        assert deep_engine.run_state == EngineRunState.IDLE

        # Verify WARNING, not ERROR
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING and "执行超时" in r.message]
        assert len(warning_records) >= 1

    def test_drain_pending_context_timeout(self, deep_engine, caplog):
        """A timed-out follow-up fails closed and keeps its pending guidance."""
        deep_engine._run_state = EngineRunState.RUNNING
        deep_engine._session = _TimeoutSession()
        deep_engine._pending_context = ["extra context"]

        last = PromptResult(stop_reason="end_turn")
        result, assessment = deep_engine._drain_pending_context(
            on_event=lambda e: None,
            timeout=10,
            last_result=last,
            last_assessment=classify_prompt_result(last),
        )

        assert result.stop_reason == "pending_context_timeout"
        assert assessment.stop_reason == "pending_context_timeout"
        assert deep_engine._pending_context == ["extra context"]
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING and "超时" in r.message]
        assert len(warning_records) >= 1

# ===========================================================================
# Spec Engine
# ===========================================================================

class TestSpecEngineTimeout:

    def test_execute_timeout(self, spec_engine, caplog, monkeypatch):
        """execute: TimeoutError → WARNING + 'Spec执行超时' + ABORTED status."""
        cb = SpecEngineCallbacks()
        errors = []
        cb.on_error = lambda msg: errors.append(msg)

        monkeypatch.setattr(spec_engine, "_create_session_fn", lambda **kw: _TimeoutSession())
        # Stub criteria parsing to avoid LLM
        monkeypatch.setattr(
            "src.spec_engine.engine.parse_acceptance_criteria",
            lambda txt, decompose_fn=None: ["criterion1"],
        )
        # Stub _run_cycle_loop to directly raise TimeoutError (avoids deep settings deps)
        monkeypatch.setattr(
            spec_engine, "_run_cycle_loop",
            lambda **kw: (_ for _ in ()).throw(TimeoutError("ACP prompt 执行超时 (300s)")),
        )

        project = spec_engine.execute("do something", callbacks=cb)

        assert project.status == SpecProjectStatus.FAILED
        assert any("Spec执行超时" in e for e in errors)
        assert spec_engine.run_state == EngineRunState.IDLE

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING and "Spec执行超时" in r.message]
        assert len(warning_records) >= 1
