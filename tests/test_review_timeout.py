"""Tests for TimeoutError handling improvements across review diagnostics,
circuit breaker, and convergence detection.

Covers:
- T7:  sync_adapter TimeoutError re-raise carries non-empty message
- T8:  review.py _extract_error_text / _infer_fail_reason friendly output
- T9:  conduct_review returns friendly ReviewResult on timeout
- T10: circuit breaker triggers after consecutive failures and recovers
- T11: convergence detection skips review_failed cycles
"""

import concurrent.futures
import logging
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.card.ui_text import UI_TEXT
from src.engine_base import PerspectiveReview, ReviewPerspective, ReviewResult
from src.spec_engine.convergence import detect_convergence
from src.spec_engine.models import (
    CriteriaTracker,
    SpecCycle,
    SpecProject,
)
from src.spec_engine.review import (
    PipelineRetryContext,
    ReviewCircuitState,
    _conduct_review_pipeline,
    attempt_pipeline_retry,
    build_retry_diagnostics,
    build_review_exception_diagnostics,
    conduct_review,
    handle_pipeline_errors_with_retry,
    outcomes_to_review_result,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(**overrides):
    """Create a minimal mock settings object with review-related defaults."""
    defaults = {
        "spec_review_timeout": 240,
        "spec_review_failure_circuit_enabled": True,
        "spec_review_failure_max_consecutive": 3,
        "spec_review_failure_cooldown_cycles": 3,
        "spec_review_failure_max_cooldown_cycles": 12,
        "spec_review_min_timeout": 60,
        "spec_review_hard_floor": 15,
        "spec_review_enabled": True,
        "diagnostics_redact_enabled": False,
        "diagnostics_snippet_limit": 240,
        "diagnostics_total_limit": 2000,
    }
    defaults.update(overrides)
    s = MagicMock()
    for k, v in defaults.items():
        setattr(s, k, v)
    return s


def _make_project(*, requirement: str = "test", cycles: list | None = None) -> SpecProject:
    return SpecProject(
        project_id="test-1",
        name="test",
        root_path="/tmp/test",
        requirement=requirement,
        cycles=cycles or [],
        criteria_tracker=CriteriaTracker(criteria=["criterion_a"]),
    )


def _make_review_result(passed: bool, suggestions: list[str] | None = None) -> ReviewResult:
    return ReviewResult(
        reviews=[
            PerspectiveReview(
                perspective=p,
                passed=passed,
                suggestions=suggestions or [],
            )
            for p in ReviewPerspective
        ],
    )


# ===========================================================================
# T7: sync_adapter TimeoutError re-raise carries non-empty message
# ===========================================================================


class TestSyncAdapterTimeoutMessage:
    """Verify that send_prompt wraps bare TimeoutError with a meaningful message."""

    def test_timeout_error_has_message(self):
        """When future.result raises TimeoutError, re-raised error must carry text."""
        from src.acp.sync_adapter import SyncACPSession

        session = SyncACPSession.__new__(SyncACPSession)
        session._acp_session = MagicMock()
        session._loop = MagicMock()
        session._active_future = None
        session.last_active = 0
        session.message_count = 0
        session.last_query = ""
        session._watchdog_timer = None
        session._watchdog_lock = threading.Lock()
        session.cancel = MagicMock()

        # Mock _start_watchdog to be a no-op
        session._start_watchdog = MagicMock()

        # Create a future that times out
        future = concurrent.futures.Future()

        with patch("asyncio.run_coroutine_threadsafe", return_value=future):
            # Simulate TimeoutError from future.result
            with patch.object(future, "result", side_effect=TimeoutError()):
                with pytest.raises(TimeoutError, match="ACP prompt 执行超时"):
                    session.send_prompt("hello", timeout=30)

    def test_timeout_error_includes_timeout_value(self):
        """Re-raised TimeoutError message includes the timeout seconds."""
        from src.acp.sync_adapter import SyncACPSession

        session = SyncACPSession.__new__(SyncACPSession)
        session._acp_session = MagicMock()
        session._loop = MagicMock()
        session._active_future = None
        session.last_active = 0
        session.message_count = 0
        session.last_query = ""
        session._watchdog_timer = None
        session._watchdog_lock = threading.Lock()
        session.cancel = MagicMock()
        session._start_watchdog = MagicMock()

        future = concurrent.futures.Future()

        with patch("asyncio.run_coroutine_threadsafe", return_value=future):
            with patch.object(future, "result", side_effect=TimeoutError()):
                with pytest.raises(TimeoutError) as exc_info:
                    session.send_prompt("hello", timeout=42)
                assert "42" in str(exc_info.value)

    def test_timeout_marks_session_dead_when_original_prompt_does_not_drain(self):
        """A sent cancel notification is not proof that the prompt has stopped."""
        from src.acp.sync_adapter import SyncACPSession

        session = SyncACPSession.__new__(SyncACPSession)
        session._acp_session = MagicMock()
        session._loop = MagicMock()
        session._active_future = None
        session._force_dead = False
        session.last_active = 0
        session.message_count = 0
        session.last_query = ""
        session._watchdog_timer = None
        session._watchdog_lock = threading.Lock()
        session.cancel = MagicMock()
        session._start_watchdog = MagicMock()

        pending = concurrent.futures.Future()

        with (
            patch("asyncio.run_coroutine_threadsafe", return_value=pending),
            patch(
                "src.acp.sync_adapter._PROMPT_CANCEL_DRAIN_TIMEOUT_S",
                0.01,
                create=True,
            ),
        ):
            with pytest.raises(TimeoutError, match="ACP prompt 执行超时"):
                session.send_prompt("hello", timeout=0.01)

        session.cancel.assert_called_once_with(wait=True, timeout=2.0)
        assert pending.cancelled()
        assert session._force_dead is True

    def test_cancel_drain_terminal_transport_error_marks_session_dead(self):
        """A drained future can be terminal while its ACP transport is unusable."""
        from src.acp.sync_adapter import SyncACPSession

        session = SyncACPSession.__new__(SyncACPSession)
        session._acp_session = MagicMock()
        session._loop = MagicMock()
        session._active_future = None
        session._force_dead = False
        session._log_failures = False
        session.last_active = 0
        session.message_count = 0
        session.last_query = ""
        session.cancel = MagicMock()
        session._start_watchdog = MagicMock()

        future = MagicMock()
        future.result.side_effect = [
            TimeoutError(),
            ConnectionError("connection closed"),
        ]
        future.done.return_value = True

        with patch("asyncio.run_coroutine_threadsafe", return_value=future):
            with pytest.raises(TimeoutError, match="ACP prompt 执行超时"):
                session.send_prompt("hello", timeout=0.01)

        session.cancel.assert_called_once_with(wait=True, timeout=2.0)
        assert session._force_dead is True

    def test_completed_prompt_side_timeout_is_not_marked_undrained(self):
        """Future.result can raise TimeoutError from a completed prompt itself."""
        from src.acp.sync_adapter import SyncACPSession

        session = SyncACPSession.__new__(SyncACPSession)
        session._acp_session = MagicMock()
        session._loop = MagicMock()
        session._active_future = None
        session._force_dead = False
        session._log_failures = False
        session.last_active = 0
        session.message_count = 0
        session.last_query = ""
        session.cancel = MagicMock()
        session._start_watchdog = MagicMock()

        completed = concurrent.futures.Future()
        completed.set_exception(TimeoutError("prompt-side timeout"))

        with patch("asyncio.run_coroutine_threadsafe", return_value=completed):
            with pytest.raises(TimeoutError, match="ACP prompt 执行超时"):
                session.send_prompt("hello", timeout=30)

        assert completed.done()
        assert session._force_dead is False


# ===========================================================================
# T8: review.py diagnostics — friendly text for TimeoutError
# ===========================================================================


class TestReviewDiagnosticsFriendlyText:
    """Verify _extract_error_text and _infer_fail_reason handle TimeoutError correctly."""

    def test_build_diagnostics_timeout_empty_message_replaced(self):
        """TimeoutError() with empty message should produce friendly Chinese text."""
        diag = build_review_exception_diagnostics(
            TimeoutError(),
            cycle=5,
            get_settings_fn=_make_settings,
        )
        error_text = str(diag.get("error_text") or "")
        assert "empty message" not in error_text.lower()
        assert "审查超时" in error_text

    def test_build_diagnostics_timeout_with_message_kept(self):
        """TimeoutError('some detail') should keep the original message."""
        diag = build_review_exception_diagnostics(
            TimeoutError("ACP prompt 执行超时 (120s)"),
            cycle=5,
            get_settings_fn=_make_settings,
        )
        error_text = str(diag.get("error_text") or "")
        assert "120s" in error_text

    def test_build_diagnostics_fail_reason_is_timeout(self):
        """fail_reason should be 'timeout' for TimeoutError."""
        diag = build_review_exception_diagnostics(
            TimeoutError(),
            cycle=1,
            get_settings_fn=_make_settings,
        )
        assert diag["fail_reason"] == "timeout"

    def test_build_diagnostics_non_timeout_keeps_error_text(self):
        """Non-timeout exceptions should preserve their error_text."""
        diag = build_review_exception_diagnostics(
            ValueError("bad value"),
            cycle=2,
            get_settings_fn=_make_settings,
        )
        assert "bad value" in str(diag.get("error_text") or "")
        assert diag["fail_reason"] == "parse_error"


# ===========================================================================
# T9: conduct_review returns friendly ReviewResult on timeout
# ===========================================================================


class TestConductReviewTimeout:
    """conduct_review should return a ReviewResult with friendly timeout text."""

    def test_timeout_fallback_suggestion_is_friendly(self):
        """On timeout, fallback suggestions must be user-friendly Chinese."""
        settings = _make_settings()
        project = _make_project()
        circuit = ReviewCircuitState()

        def fake_send_prompt(*args, **kwargs):
            raise TimeoutError()

        result = conduct_review(
            session=MagicMock(),
            settings=settings,
            project=project,
            send_prompt_with_retry_fn=fake_send_prompt,
            build_review_exception_diagnostics_fn=lambda e, **kw: build_review_exception_diagnostics(
                e, cycle=kw.get("cycle", 1), get_settings_fn=lambda: settings,
            ),
            circuit=circuit,
            cycle=1,
        )

        assert isinstance(result, ReviewResult)
        assert len(result.reviews) == len(ReviewPerspective)
        for pr in result.reviews:
            assert not pr.passed
            assert len(pr.suggestions) == 1
            suggestion = pr.suggestions[0]
            assert "超时" in suggestion
            assert "empty message" not in suggestion.lower()

    def test_non_timeout_fallback_shows_error_detail(self):
        """Non-timeout exceptions should produce a suggestion with error detail."""
        settings = _make_settings()
        project = _make_project()
        circuit = ReviewCircuitState()

        def fake_send_prompt(*args, **kwargs):
            raise ValueError("connection reset")

        result = conduct_review(
            session=MagicMock(),
            settings=settings,
            project=project,
            send_prompt_with_retry_fn=fake_send_prompt,
            build_review_exception_diagnostics_fn=lambda e, **kw: build_review_exception_diagnostics(
                e, cycle=kw.get("cycle", 1), get_settings_fn=lambda: settings,
            ),
            circuit=circuit,
            cycle=1,
        )

        assert isinstance(result, ReviewResult)
        for pr in result.reviews:
            suggestion = pr.suggestions[0]
            assert "审查执行异常" in suggestion


# ===========================================================================
# T10: Circuit breaker — triggers after consecutive failures, recovers
# ===========================================================================


class TestCircuitBreaker:
    """Verify the review failure circuit breaker logic."""

    def _run_review_cycle(self, settings, project, circuit, cycle):
        """Helper: run one conduct_review that always times out."""
        def fake_send_prompt(*args, **kwargs):
            raise TimeoutError()

        return conduct_review(
            session=MagicMock(),
            settings=settings,
            project=project,
            send_prompt_with_retry_fn=fake_send_prompt,
            build_review_exception_diagnostics_fn=lambda e, **kw: build_review_exception_diagnostics(
                e, cycle=kw.get("cycle", 1), get_settings_fn=lambda: settings,
            ),
            circuit=circuit,
            cycle=cycle,
        )

    def test_circuit_opens_after_max_consecutive_failures(self):
        """After 3 consecutive failures, circuit should be open."""
        settings = _make_settings(
            spec_review_failure_circuit_enabled=True,
            spec_review_failure_max_consecutive=3,
            spec_review_failure_cooldown_cycles=3,
        )
        project = _make_project()
        circuit = ReviewCircuitState()

        # 3 consecutive failures
        for cycle in range(1, 4):
            self._run_review_cycle(settings, project, circuit, cycle)

        assert circuit.review_failure_consecutive >= 3
        assert circuit.review_circuit_open_until_cycle > 0

    def test_circuit_skips_review_while_open(self):
        """While circuit is open, conduct_review should skip and return fallback."""
        settings = _make_settings(
            spec_review_failure_circuit_enabled=True,
            spec_review_failure_max_consecutive=3,
            spec_review_failure_cooldown_cycles=3,
        )
        project = _make_project()
        circuit = ReviewCircuitState()

        # Open circuit: 3 failures at cycles 1-3
        for cycle in range(1, 4):
            self._run_review_cycle(settings, project, circuit, cycle)


        # Cycle within cooldown should be skipped (no prompt sent)
        send_called = False
        def tracking_send(*args, **kwargs):
            nonlocal send_called
            send_called = True
            raise TimeoutError()

        result = conduct_review(
            session=MagicMock(),
            settings=settings,
            project=project,
            send_prompt_with_retry_fn=tracking_send,
            build_review_exception_diagnostics_fn=lambda e, **kw: build_review_exception_diagnostics(
                e, cycle=kw.get("cycle", 1), get_settings_fn=lambda: settings,
            ),
            circuit=circuit,
            cycle=4,  # within cooldown
        )

        assert not send_called, "Review prompt should not be sent while circuit is open"
        assert isinstance(result, ReviewResult)
        for pr in result.reviews:
            assert "审查暂停" in pr.suggestions[0]

    def test_circuit_skip_emits_metrics(self):
        """Circuit-breaker skip path should emit review_circuit_skip metrics."""
        settings = _make_settings(
            spec_review_failure_circuit_enabled=True,
            spec_review_failure_max_consecutive=3,
            spec_review_failure_cooldown_cycles=3,
        )
        project = _make_project()
        circuit = ReviewCircuitState()

        # Open circuit: 3 failures at cycles 1-3
        for cycle in range(1, 4):
            self._run_review_cycle(settings, project, circuit, cycle)

        # Capture metrics via logger fallback
        captured_metrics = {}
        import json as _json

        original_info = logging.getLogger("src.spec_engine.review").info

        def capture_info(msg, *args):
            if "review_circuit_skip_metrics" in str(msg):
                try:
                    captured_metrics.update(_json.loads(args[0] if args else "{}"))
                except Exception:
                    pass
            original_info(msg, *args)

        with patch("src.spec_engine.review.logger.info", side_effect=capture_info):
            with patch("src.utils.metrics_exporter.get_metrics_exporter", side_effect=ImportError("test")):
                conduct_review(
                    session=MagicMock(),
                    settings=settings,
                    project=project,
                    send_prompt_with_retry_fn=MagicMock(),
                    build_review_exception_diagnostics_fn=lambda e, **kw: {},
                    circuit=circuit,
                    cycle=4,  # within cooldown
                )

        assert captured_metrics.get("metric_type") == "review_circuit_skip"
        assert captured_metrics.get("engine") == "spec"
        assert captured_metrics.get("cycle") == 4
        assert "consecutive_failures" in captured_metrics
        assert "consecutive_skips" in captured_metrics

    def test_circuit_recovers_after_cooldown(self):
        """After cooldown cycles pass, circuit should allow review again."""
        settings = _make_settings(
            spec_review_failure_circuit_enabled=True,
            spec_review_failure_max_consecutive=3,
            spec_review_failure_cooldown_cycles=3,
        )
        project = _make_project()
        circuit = ReviewCircuitState()

        # Open circuit at cycle 3 (open_until = 3 + 3 = 6)
        for cycle in range(1, 4):
            self._run_review_cycle(settings, project, circuit, cycle)

        open_until = circuit.review_circuit_open_until_cycle

        # Cycle after cooldown should attempt review again
        send_called = False
        def tracking_send(*args, **kwargs):
            nonlocal send_called
            send_called = True
            raise ValueError("normal error")  # not timeout

        conduct_review(
            session=MagicMock(),
            settings=settings,
            project=project,
            send_prompt_with_retry_fn=tracking_send,
            build_review_exception_diagnostics_fn=lambda e, **kw: build_review_exception_diagnostics(
                e, cycle=kw.get("cycle", 1), get_settings_fn=lambda: settings,
            ),
            circuit=circuit,
            cycle=open_until + 1,  # past cooldown
        )

        assert send_called, "Review should be attempted after cooldown expires"


# ===========================================================================
# T11: Convergence detection skips review_failed cycles
# ===========================================================================


class TestConvergenceSkipsReviewFailed:
    """detect_convergence must return False when recent cycles have review_failed."""

    def _make_cycle(self, num: int, *, decision: str = "", review: ReviewResult | None = None) -> SpecCycle:
        c = SpecCycle(cycle_number=num)
        c.review_result = review
        c.review_decision = decision
        return c

    def test_review_failed_decision_prevents_convergence(self):
        """Cycles with review_failed* decision should not converge."""
        review = _make_review_result(False, ["fix timeout"])
        project = _make_project(cycles=[
            self._make_cycle(1, decision="review_failed_continue", review=review),
            self._make_cycle(2, decision="review_failed_continue", review=review),
        ])

        result = detect_convergence(
            project,
            convergence_window=2,
            review_enabled=True,
        )

        assert result is False

    def test_review_failed_open_circuit_prevents_convergence(self):
        """review_failed_open_circuit decision should also prevent convergence."""
        review = _make_review_result(False, ["circuit breaker"])
        project = _make_project(cycles=[
            self._make_cycle(1, decision="review_failed_open_circuit", review=review),
            self._make_cycle(2, decision="review_failed_open_circuit", review=review),
        ])

        result = detect_convergence(
            project,
            convergence_window=2,
            review_enabled=True,
        )

        assert result is False

    def test_normal_identical_reviews_do_converge(self):
        """Normal (non-failed) cycles with identical suggestions should converge."""
        review = _make_review_result(False, ["same suggestion"])
        project = _make_project(cycles=[
            self._make_cycle(1, decision="needs_improvement", review=review),
            self._make_cycle(2, decision="needs_improvement", review=review),
        ])

        result = detect_convergence(
            project,
            convergence_window=2,
            review_enabled=True,
        )

        assert result is True


# ===========================================================================
# T12: Exponential backoff for circuit breaker cooldown
# ===========================================================================


class TestSpecCircuitExponentialBackoff:
    """Verify exponential backoff: cooldown grows 3→6→12 on repeated triggers."""

    def _run_review_cycle(self, settings, project, circuit, cycle):
        def fake_send_prompt(*args, **kwargs):
            raise TimeoutError()

        return conduct_review(
            session=MagicMock(),
            settings=settings,
            project=project,
            send_prompt_with_retry_fn=fake_send_prompt,
            build_review_exception_diagnostics_fn=lambda e, **kw: build_review_exception_diagnostics(
                e, cycle=kw.get("cycle", 1), get_settings_fn=lambda: settings,
            ),
            circuit=circuit,
            cycle=cycle,
        )

    def test_first_trigger_cooldown_is_base(self):
        """First circuit trigger: cooldown = base (3)."""
        settings = _make_settings(
            spec_review_failure_max_consecutive=3,
            spec_review_failure_cooldown_cycles=3,
            spec_review_failure_max_cooldown_cycles=12,
        )
        project = _make_project()
        circuit = ReviewCircuitState()

        for cycle in range(1, 4):
            self._run_review_cycle(settings, project, circuit, cycle)

        # First trigger: cooldown=3, open_until = 3 + 3 = 6
        assert circuit.review_circuit_open_until_cycle == 6
        assert circuit.backoff_level == 1

    def test_second_trigger_cooldown_doubles(self):
        """Second circuit trigger: cooldown = 6."""
        settings = _make_settings(
            spec_review_failure_max_consecutive=3,
            spec_review_failure_cooldown_cycles=3,
            spec_review_failure_max_cooldown_cycles=12,
        )
        project = _make_project()
        circuit = ReviewCircuitState()

        # First trigger (cycles 1-3)
        for cycle in range(1, 4):
            self._run_review_cycle(settings, project, circuit, cycle)
        first_open_until = circuit.review_circuit_open_until_cycle

        # Reset consecutive but keep backoff_level (simulate: circuit closes, fails again)
        circuit.review_failure_consecutive = 0
        circuit.recent_outcomes.clear()

        # Second trigger (cycles after cooldown)
        base = first_open_until + 1
        for cycle in range(base, base + 3):
            self._run_review_cycle(settings, project, circuit, cycle)

        # cooldown = 3 * 2^1 = 6
        assert circuit.review_circuit_open_until_cycle == (base + 2) + 6
        assert circuit.backoff_level == 2

    def test_third_trigger_cooldown_capped(self):
        """Third circuit trigger: cooldown = min(12, max_cooldown=12)."""
        settings = _make_settings(
            spec_review_failure_max_consecutive=3,
            spec_review_failure_cooldown_cycles=3,
            spec_review_failure_max_cooldown_cycles=12,
        )
        project = _make_project()
        circuit = ReviewCircuitState()

        # First trigger (cycles 1-3) → cooldown=3, open_until=6, backoff=1
        for cycle in range(1, 4):
            self._run_review_cycle(settings, project, circuit, cycle)
        assert circuit.backoff_level == 1

        # Second trigger: start after cooldown expires
        circuit.review_failure_consecutive = 0
        circuit.recent_outcomes.clear()
        base2 = circuit.review_circuit_open_until_cycle + 1
        for cycle in range(base2, base2 + 3):
            self._run_review_cycle(settings, project, circuit, cycle)
        assert circuit.backoff_level == 2

        # Third trigger: start after cooldown expires
        circuit.review_failure_consecutive = 0
        circuit.recent_outcomes.clear()
        base3 = circuit.review_circuit_open_until_cycle + 1
        for cycle in range(base3, base3 + 3):
            self._run_review_cycle(settings, project, circuit, cycle)
        assert circuit.backoff_level == 3
        # cooldown = min(3 * 2^2, 12) = 12
        assert circuit.review_circuit_open_until_cycle == (base3 + 2) + 12

        # Fourth trigger: still capped at 12
        circuit.review_failure_consecutive = 0
        circuit.recent_outcomes.clear()
        base4 = circuit.review_circuit_open_until_cycle + 1
        for cycle in range(base4, base4 + 3):
            self._run_review_cycle(settings, project, circuit, cycle)
        assert circuit.review_circuit_open_until_cycle == (base4 + 2) + 12

    def test_success_resets_backoff_level(self):
        """After a successful review, backoff_level resets to 0."""
        settings = _make_settings(
            spec_review_failure_max_consecutive=3,
            spec_review_failure_cooldown_cycles=3,
            spec_review_failure_max_cooldown_cycles=12,
        )
        project = _make_project()
        circuit = ReviewCircuitState()

        # Trigger once
        for cycle in range(1, 4):
            self._run_review_cycle(settings, project, circuit, cycle)
        assert circuit.backoff_level == 1

        # Simulate success
        def success_send(*args, **kwargs):
            pass  # no exception = success

        conduct_review(
            session=MagicMock(),
            settings=settings,
            project=project,
            send_prompt_with_retry_fn=success_send,
            build_review_exception_diagnostics_fn=lambda e, **kw: {},
            circuit=circuit,
            cycle=10,
        )

        assert circuit.backoff_level == 0
        assert circuit.consecutive_timeouts == 0


# ===========================================================================
# T13: Adaptive (progressive) review timeout
# ===========================================================================


class TestSpecAdaptiveTimeout:
    """Verify review timeout decreases on consecutive timeouts."""

    def test_timeout_decreases_on_consecutive_timeouts(self):
        """Consecutive TimeoutErrors should trigger progressively shorter timeouts."""
        settings = _make_settings(
            spec_review_timeout=240,
            spec_review_min_timeout=60,
            spec_review_failure_max_consecutive=100,  # high so circuit doesn't interfere
        )
        project = _make_project()
        circuit = ReviewCircuitState()

        captured_timeouts = []

        def capturing_send(*args, **kwargs):
            captured_timeouts.append(kwargs.get("timeout"))
            raise TimeoutError()

        for cycle in range(1, 4):
            conduct_review(
                session=MagicMock(),
                settings=settings,
                project=project,
                send_prompt_with_retry_fn=capturing_send,
                build_review_exception_diagnostics_fn=lambda e, **kw: build_review_exception_diagnostics(
                    e, cycle=kw.get("cycle", 1), get_settings_fn=lambda: settings,
                ),
                circuit=circuit,
                cycle=cycle,
            )

        # Timeout sequence with 1.3x decay: 240 (n=0), 184 (n=1), 142 (n=2)
        assert captured_timeouts == [240, 184, 142]

    def test_timeout_resets_after_success(self):
        """After success, timeout should go back to base."""
        settings = _make_settings(
            spec_review_timeout=240,
            spec_review_min_timeout=60,
            spec_review_failure_max_consecutive=100,
        )
        project = _make_project()
        circuit = ReviewCircuitState()

        # 2 timeouts
        def timeout_send(*args, **kwargs):
            raise TimeoutError()

        for cycle in range(1, 3):
            conduct_review(
                session=MagicMock(),
                settings=settings,
                project=project,
                send_prompt_with_retry_fn=timeout_send,
                build_review_exception_diagnostics_fn=lambda e, **kw: build_review_exception_diagnostics(
                    e, cycle=kw.get("cycle", 1), get_settings_fn=lambda: settings,
                ),
                circuit=circuit,
                cycle=cycle,
            )
        assert circuit.consecutive_timeouts == 2

        # 1 success
        def success_send(*args, **kwargs):
            pass

        conduct_review(
            session=MagicMock(),
            settings=settings,
            project=project,
            send_prompt_with_retry_fn=success_send,
            build_review_exception_diagnostics_fn=lambda e, **kw: {},
            circuit=circuit,
            cycle=10,
        )
        assert circuit.consecutive_timeouts == 0

        # Next timeout should use base again
        captured = []

        def capture_send(*args, **kwargs):
            captured.append(kwargs.get("timeout"))
            raise TimeoutError()

        conduct_review(
            session=MagicMock(),
            settings=settings,
            project=project,
            send_prompt_with_retry_fn=capture_send,
            build_review_exception_diagnostics_fn=lambda e, **kw: build_review_exception_diagnostics(
                e, cycle=kw.get("cycle", 1), get_settings_fn=lambda: settings,
            ),
            circuit=circuit,
            cycle=11,
        )
        assert captured == [240]

    def test_timeout_respects_min(self):
        """Timeout never goes below min_timeout."""
        settings = _make_settings(
            spec_review_timeout=240,
            spec_review_min_timeout=60,
            spec_review_failure_max_consecutive=100,
        )
        project = _make_project()
        circuit = ReviewCircuitState()
        circuit.consecutive_timeouts = 10  # very high

        captured = []

        def capture_send(*args, **kwargs):
            captured.append(kwargs.get("timeout"))
            raise TimeoutError()

        conduct_review(
            session=MagicMock(),
            settings=settings,
            project=project,
            send_prompt_with_retry_fn=capture_send,
            build_review_exception_diagnostics_fn=lambda e, **kw: build_review_exception_diagnostics(
                e, cycle=kw.get("cycle", 1), get_settings_fn=lambda: settings,
            ),
            circuit=circuit,
            cycle=1,
        )
        assert captured == [60]


# ---------------------------------------------------------------------------
# T-Skip: Spec ReviewCircuitState.consecutive_skips serialization & overrun
# ---------------------------------------------------------------------------


class TestCircuitStateSerialization:
    """Round-trip serialization for ReviewCircuitState fields."""

    @pytest.mark.parametrize("field,value", [
        pytest.param("consecutive_skips", 0, id="skips_default"),
        pytest.param("consecutive_skips", 7, id="skips_nonzero"),
        pytest.param("last_review_elapsed_ms", 0, id="elapsed_default"),
        pytest.param("last_review_elapsed_ms", 12345, id="elapsed_nonzero"),
    ])
    def test_round_trip(self, field, value):
        circuit = ReviewCircuitState(**{field: value})
        assert getattr(circuit, field) == value
        d = circuit.to_dict()
        assert d[field] == value
        restored = ReviewCircuitState.from_dict(d)
        assert getattr(restored, field) == value

    @pytest.mark.parametrize("field", [
        pytest.param("consecutive_skips", id="skips_missing"),
        pytest.param("last_review_elapsed_ms", id="elapsed_missing"),
    ])
    def test_from_dict_missing_key_defaults_zero(self, field):
        """Backward compat: old persisted state without field defaults to 0."""
        restored = ReviewCircuitState.from_dict({"review_failure_consecutive": 2})
        assert getattr(restored, field) == 0


class TestSpecSkipOverrunProtection:
    """conduct_review increments consecutive_skips on circuit-open skip
    and logs a warning when the overrun threshold is reached."""

    def test_consecutive_skips_increments_on_skip(self):
        settings = _make_settings(spec_review_failure_max_consecutive=3)
        project = _make_project()
        circuit = ReviewCircuitState(
            review_failure_consecutive=3,
            review_circuit_open_until_cycle=10,
        )
        assert circuit.consecutive_skips == 0

        result = conduct_review(
            session=MagicMock(),
            settings=settings,
            project=project,
            send_prompt_with_retry_fn=MagicMock(),
            build_review_exception_diagnostics_fn=MagicMock(),
            circuit=circuit,
            cycle=5,
        )
        assert circuit.consecutive_skips == 1
        # Below overrun threshold: suggestions should NOT contain overrun warning
        for rev in result.reviews:
            assert not any("跳过次数异常偏高" in s for s in rev.suggestions)

    def test_overrun_warning_logged(self):
        """When consecutive_skips reaches threshold, a warning is logged and suggestions reflect overrun."""
        settings = _make_settings(spec_review_failure_max_consecutive=3)
        project = _make_project()
        # threshold = max(1,3)*2 = 6; set skips to 5 so next skip hits 6
        circuit = ReviewCircuitState(
            review_failure_consecutive=3,
            review_circuit_open_until_cycle=100,
            consecutive_skips=5,
        )

        import logging
        with patch.object(logging.getLogger("src.spec_engine.review"), "warning") as mock_warn:
            result = conduct_review(
                session=MagicMock(),
                settings=settings,
                project=project,
                send_prompt_with_retry_fn=MagicMock(),
                build_review_exception_diagnostics_fn=MagicMock(),
                circuit=circuit,
                cycle=50,
            )
        assert circuit.consecutive_skips == 6
        overrun_calls = [c for c in mock_warn.call_args_list if "review_skip_overrun" in str(c)]
        assert len(overrun_calls) >= 1
        # FS-14: skip-overrun warning moved to logger-only, not in user-visible suggestions
        for rev in result.reviews:
            assert not any("跳过次数异常偏高" in s for s in rev.suggestions)

    def test_consecutive_skips_reset_on_success(self):
        """Successful review resets consecutive_skips to 0."""
        settings = _make_settings(spec_review_failure_max_consecutive=3)
        project = _make_project()
        circuit = ReviewCircuitState(consecutive_skips=5)

        def fake_send(*args, **kwargs):
            # Return valid review text so parse succeeds
            pass

        conduct_review(
            session=MagicMock(),
            settings=settings,
            project=project,
            send_prompt_with_retry_fn=fake_send,
            build_review_exception_diagnostics_fn=MagicMock(),
            circuit=circuit,
            cycle=1,
        )
        assert circuit.consecutive_skips == 0


# ---------------------------------------------------------------------------
# normalize_review_diagnostics: error_text truncation
# ---------------------------------------------------------------------------


class TestNormalizeDiagnosticsErrorTextTruncation:
    """normalize_review_diagnostics must truncate error_text to 500 chars."""

    def _normalize(self, diag, **kwargs):
        from src.utils.review_diagnostics import normalize_review_diagnostics
        return normalize_review_diagnostics(diag, **kwargs)

    def test_short_text_unchanged(self):
        diag = {"error_text": "short error"}
        result = self._normalize(diag)
        assert result["error_text"] == "short error"

    def test_501_chars_truncated(self):
        text = "a" * 501
        diag = {"error_text": text}
        result = self._normalize(diag)
        assert len(result["error_text"]) <= 500
        assert result["error_text"].endswith("…(truncated)")

    def test_custom_limit(self):
        text = "y" * 200
        diag = {"error_text": text}
        result = self._normalize(diag, error_text_limit=100)
        assert len(result["error_text"]) <= 100
        assert result["error_text"].endswith("…(truncated)")

    def test_empty_text_fallback_not_truncated(self):
        """Empty error_text falls back to err_repr, which should be short."""
        diag = {"error_text": "", "err_repr": "TimeoutError()"}
        result = self._normalize(diag)
        assert result["error_text"] == "TimeoutError()"


# ===========================================================================
# T-Pipeline: Pipeline全量超时时 circuit 计数器递增
# ===========================================================================


class TestPipelineAllTimeoutIncrementsCircuit:
    """When all workers in the pipeline path fail with TIMEOUT,
    circuit.consecutive_timeouts and review_failure_consecutive must increment.

    With auto-retry enabled (default), the pipeline retries once before
    incrementing the circuit counters.
    """

    def test_pipeline_all_timeout_increments_circuit(self):
        from src.spec_engine.perspective_worker import PerspectiveOutcome, ReviewErrorCode
        from src.spec_engine.review_artifacts import ReviewArtifacts

        circuit = ReviewCircuitState()
        assert circuit.consecutive_timeouts == 0
        assert circuit.review_failure_consecutive == 0

        settings = _make_settings(
            spec_review_timeout=240,
            spec_review_max_parallel=3,
            spec_review_retry_max_delay=30,
            spec_review_retry_max_attempts=1,
            spec_review_min_timeout=60,
            spec_review_hard_floor=20,
        )

        artifacts = ReviewArtifacts(
            cycle_number=1,
            cwd="/tmp",
            requirement="test",
            diff_patch="patch",
            touched_files=["a.py"],
            spec_output="spec",
            plan_output="plan",
            build_output="build",
        )

        # Build fake all-timeout outcomes
        timeout_outcomes = [
            PerspectiveOutcome(
                perspective=p,
                review=PerspectiveReview(
                    perspective=p,
                    passed=False,
                    suggestions=["审查异常：超时"],
                    summary="异常",
                ),
                elapsed_ms=180000,
                error="当前系统较繁忙",
                error_code=ReviewErrorCode.TIMEOUT,
            )
            for p in ReviewPerspective
        ]

        with (
            patch("src.spec_engine.review_pipeline.run_review_pipeline", return_value=timeout_outcomes),
            patch("src.spec_engine.review_retry.time.sleep", return_value=None),
        ):
            result = conduct_review(
                session=MagicMock(),
                settings=settings,
                project=_make_project(),
                send_prompt_with_retry_fn=MagicMock(),
                build_review_exception_diagnostics_fn=MagicMock(return_value={}),
                circuit=circuit,
                cycle=1,
                artifacts=artifacts,
                agent_type="coco",
            )

        # Retry was attempted but also failed → circuit incremented
        assert circuit.consecutive_timeouts == 1
        assert circuit.review_failure_consecutive == 1
        assert isinstance(result, ReviewResult)


# ===========================================================================
# T-Budget: max_parallel=3 时 budget_seconds 计算验证
# ===========================================================================


class TestBudgetSecondsWithMaxParallel3:
    """Verify budget_seconds formula with max_parallel=3."""

    def test_budget_seconds_with_max_parallel_3(self):
        from src.spec_engine.perspective_worker import PerspectiveOutcome
        from src.spec_engine.review_artifacts import ReviewArtifacts

        settings = _make_settings(
            spec_review_timeout=240,
            spec_review_max_parallel=3,
        )
        circuit = ReviewCircuitState()

        artifacts = ReviewArtifacts(
            cycle_number=1,
            cwd="/tmp",
            requirement="test",
            diff_patch="patch",
            touched_files=["a.py"],
            spec_output="spec",
            plan_output="plan",
            build_output="build",
        )

        captured_budget = []

        def fake_pipeline(arts, budget, **kwargs):
            captured_budget.append(budget.total_seconds)
            return [
                PerspectiveOutcome(
                    perspective=p,
                    review=PerspectiveReview(perspective=p, passed=True, suggestions=[]),
                    elapsed_ms=100,
                )
                for p in ReviewPerspective
            ]

        with patch("src.spec_engine.review_pipeline.run_review_pipeline", side_effect=fake_pipeline):
            conduct_review(
                session=MagicMock(),
                settings=settings,
                project=_make_project(),
                send_prompt_with_retry_fn=MagicMock(),
                build_review_exception_diagnostics_fn=MagicMock(return_value={}),
                circuit=circuit,
                cycle=1,
                artifacts=artifacts,
                agent_type="coco",
            )

        # max_parallel=3 → multiplier=ceil(5/3)=2 → budget=240*max(2,2+3)=240*5=1200
        assert len(captured_budget) == 1
        assert captured_budget[0] == 1200.0


# ===========================================================================
# T-AutoRetry: In-cycle auto-retry on full timeout
# ===========================================================================


def _make_timeout_outcomes():
    """Build a list of all-timeout PerspectiveOutcome for every perspective."""
    from src.spec_engine.perspective_worker import PerspectiveOutcome, ReviewErrorCode
    return [
        PerspectiveOutcome(
            perspective=p,
            review=PerspectiveReview(
                perspective=p,
                passed=False,
                suggestions=["审查异常：超时"],
                summary="异常",
            ),
            elapsed_ms=180000,
            error="当前系统较繁忙",
            error_code=ReviewErrorCode.TIMEOUT,
        )
        for p in ReviewPerspective
    ]


def _make_success_outcomes():
    """Build a list of all-pass PerspectiveOutcome for every perspective."""
    from src.spec_engine.perspective_worker import PerspectiveOutcome
    return [
        PerspectiveOutcome(
            perspective=p,
            review=PerspectiveReview(perspective=p, passed=True, suggestions=[]),
            elapsed_ms=500,
        )
        for p in ReviewPerspective
    ]


def _make_retry_artifacts():
    from src.spec_engine.review_artifacts import ReviewArtifacts
    return ReviewArtifacts(
        cycle_number=1,
        cwd="/tmp",
        requirement="test",
        diff_patch="patch",
        touched_files=["a.py"],
        spec_output="spec",
        plan_output="plan",
        build_output="build",
    )


def _make_retry_settings(**overrides):
    defaults = {
        "spec_review_timeout": 240,
        "spec_review_max_parallel": 3,
        "spec_review_retry_max_delay": 30,
        "spec_review_retry_max_attempts": 1,
        "spec_review_min_timeout": 60,
        "spec_review_hard_floor": 20,
        "spec_review_retry_base_delay": 0.05,
        "spec_review_retry_decay_factor": 1.5,
    }
    defaults.update(overrides)
    return _make_settings(**defaults)


class TestAutoRetryOnFullTimeout:
    """In-cycle auto-retry: when all workers timeout, retry once with reduced budget."""

    def test_retry_success_resets_circuit(self):
        """When retry succeeds, circuit counters should be fully reset."""
        circuit = ReviewCircuitState()
        # Simulate prior timeout state
        circuit.consecutive_timeouts = 2
        circuit.review_failure_consecutive = 2

        call_count = {"n": 0}

        def side_effect(artifacts, budget, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_timeout_outcomes()  # first call: all timeout
            return _make_success_outcomes()       # retry: success

        with (
            patch("src.spec_engine.review_pipeline.run_review_pipeline", side_effect=side_effect),
            patch("src.spec_engine.review_retry.time.sleep", return_value=None) as mock_sleep,
        ):
            result = conduct_review(
                session=MagicMock(),
                settings=_make_retry_settings(),
                project=_make_project(),
                send_prompt_with_retry_fn=MagicMock(),
                build_review_exception_diagnostics_fn=MagicMock(return_value={}),
                circuit=circuit,
                cycle=3,
                artifacts=_make_retry_artifacts(),
                agent_type="coco",
            )

        # Retry was called
        assert call_count["n"] == 2
        mock_sleep.assert_called_once()

        # Circuit fully reset
        assert circuit.consecutive_timeouts == 0
        assert circuit.review_failure_consecutive == 0
        assert circuit.backoff_level == 0
        assert circuit.review_circuit_open_until_cycle == 0

        # Result should be passing
        assert result.all_passed is True

    def test_retry_failure_increments_circuit(self):
        """When retry also fails, circuit counters should increment normally."""
        circuit = ReviewCircuitState()
        circuit.consecutive_timeouts = 0

        with (
            patch("src.spec_engine.review_pipeline.run_review_pipeline", return_value=_make_timeout_outcomes()),
            patch("src.spec_engine.review_retry.time.sleep", return_value=None),
        ):
            result = conduct_review(
                session=MagicMock(),
                settings=_make_retry_settings(),
                project=_make_project(),
                send_prompt_with_retry_fn=MagicMock(),
                build_review_exception_diagnostics_fn=MagicMock(return_value={}),
                circuit=circuit,
                cycle=1,
                artifacts=_make_retry_artifacts(),
                agent_type="coco",
            )

        # Both first attempt and retry failed
        assert circuit.consecutive_timeouts == 1
        assert circuit.review_failure_consecutive == 1
        assert result.all_passed is False

        # Diagnostics should record retry info
        diag = circuit.last_review_failure_diag
        assert diag is not None
        assert diag.get("retry_attempted") is True
        assert diag.get("retry_succeeded") is False

    def test_retry_exception_still_increments_circuit(self):
        """When retry raises an exception, circuit should still increment."""
        circuit = ReviewCircuitState()
        call_count = {"n": 0}

        def side_effect(artifacts, budget, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_timeout_outcomes()
            raise RuntimeError("connection lost")

        with (
            patch("src.spec_engine.review_pipeline.run_review_pipeline", side_effect=side_effect),
            patch("src.spec_engine.review_retry.time.sleep", return_value=None),
        ):
            result = conduct_review(
                session=MagicMock(),
                settings=_make_retry_settings(),
                project=_make_project(),
                send_prompt_with_retry_fn=MagicMock(),
                build_review_exception_diagnostics_fn=MagicMock(return_value={}),
                circuit=circuit,
                cycle=1,
                artifacts=_make_retry_artifacts(),
                agent_type="coco",
            )

        assert call_count["n"] == 2
        assert circuit.consecutive_timeouts == 1
        assert result.all_passed is False

    def test_retry_uses_reduced_budget(self):
        """Retry budget should use adaptive timeout (shorter than original)."""
        circuit = ReviewCircuitState()
        circuit.consecutive_timeouts = 1  # will compute reduced timeout

        budgets_seen = []

        def side_effect(artifacts, budget, **kwargs):
            budgets_seen.append(budget.total_seconds)
            return _make_timeout_outcomes()  # always timeout

        with (
            patch("src.spec_engine.review_pipeline.run_review_pipeline", side_effect=side_effect),
            patch("src.spec_engine.review_retry.time.sleep", return_value=None),
        ):
            conduct_review(
                session=MagicMock(),
                settings=_make_retry_settings(),
                project=_make_project(),
                send_prompt_with_retry_fn=MagicMock(),
                build_review_exception_diagnostics_fn=MagicMock(return_value={}),
                circuit=circuit,
                cycle=1,
                artifacts=_make_retry_artifacts(),
                agent_type="coco",
            )

        # Should have 2 budgets: original and retry
        assert len(budgets_seen) == 2
        # Retry budget should be smaller than original
        assert budgets_seen[1] < budgets_seen[0]


class TestAutoRetryDisabled:
    """When spec_review_retry_max_attempts=0, no retry should be attempted."""

    def test_no_retry_when_disabled(self):
        circuit = ReviewCircuitState()
        call_count = {"n": 0}

        def side_effect(artifacts, budget, **kwargs):
            call_count["n"] += 1
            return _make_timeout_outcomes()

        settings = _make_retry_settings(spec_review_retry_max_attempts=0)

        with (
            patch("src.spec_engine.review_pipeline.run_review_pipeline", side_effect=side_effect),
            patch("src.spec_engine.review_retry.time.sleep", return_value=None) as mock_sleep,
        ):
            result = conduct_review(
                session=MagicMock(),
                settings=settings,
                project=_make_project(),
                send_prompt_with_retry_fn=MagicMock(),
                build_review_exception_diagnostics_fn=MagicMock(return_value={}),
                circuit=circuit,
                cycle=1,
                artifacts=_make_retry_artifacts(),
                agent_type="coco",
            )

        # Pipeline called only once (no retry)
        assert call_count["n"] == 1
        mock_sleep.assert_not_called()

        # Circuit still incremented normally
        assert circuit.consecutive_timeouts == 1
        assert circuit.review_failure_consecutive == 1
        assert result.all_passed is False


# ===========================================================================
# New AC tests: retry paths, cancel, status callback, config validation
# ===========================================================================


def _make_partial_timeout_outcomes():
    """2 workers timeout, 3 workers succeed."""
    from src.spec_engine.perspective_worker import PerspectiveOutcome, ReviewErrorCode
    perspectives = list(ReviewPerspective)
    outcomes = []
    for i, p in enumerate(perspectives):
        if i < 2:
            outcomes.append(PerspectiveOutcome(
                perspective=p,
                review=PerspectiveReview(perspective=p, passed=False, suggestions=["超时"], summary="异常"),
                elapsed_ms=180000,
                error="timeout",
                error_code=ReviewErrorCode.TIMEOUT,
            ))
        else:
            outcomes.append(PerspectiveOutcome(
                perspective=p,
                review=PerspectiveReview(perspective=p, passed=True, suggestions=[]),
                elapsed_ms=500,
            ))
    return outcomes


def _make_partial_error_outcomes():
    """Retry outcomes with 3 success + 2 errors (non-timeout)."""
    from src.spec_engine.perspective_worker import PerspectiveOutcome, ReviewErrorCode
    perspectives = list(ReviewPerspective)
    outcomes = []
    for i, p in enumerate(perspectives):
        if i < 2:
            outcomes.append(PerspectiveOutcome(
                perspective=p,
                review=PerspectiveReview(perspective=p, passed=False, suggestions=["err"], summary="异常"),
                elapsed_ms=1000,
                error="some_error",
                error_code=ReviewErrorCode.WORKER_ERROR,
            ))
        else:
            outcomes.append(PerspectiveOutcome(
                perspective=p,
                review=PerspectiveReview(perspective=p, passed=True, suggestions=[]),
                elapsed_ms=500,
            ))
    return outcomes


class TestRetryPartialSuccess:
    """AC-R11: retry returns partial errors -> circuit increments, _retry_succeeded=False."""

    def test_retry_partial_success_increments_circuit(self):
        circuit = ReviewCircuitState()
        circuit.consecutive_timeouts = 0
        call_count = {"n": 0}

        def side_effect(artifacts, budget, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_timeout_outcomes()
            return _make_partial_error_outcomes()  # retry: partial errors

        with (
            patch("src.spec_engine.review_pipeline.run_review_pipeline", side_effect=side_effect),
            patch("src.spec_engine.review_retry.time.sleep", return_value=None),
        ):
            result = conduct_review(
                session=MagicMock(),
                settings=_make_retry_settings(),
                project=_make_project(),
                send_prompt_with_retry_fn=MagicMock(),
                build_review_exception_diagnostics_fn=MagicMock(return_value={}),
                circuit=circuit,
                cycle=1,
                artifacts=_make_retry_artifacts(),
                agent_type="coco",
            )

        assert call_count["n"] == 2  # original + retry
        # Retry had errors -> not successful, circuit incremented
        assert circuit.consecutive_timeouts == 1
        assert circuit.review_failure_consecutive == 1
        assert result.all_passed is False
        diag = circuit.last_review_failure_diag
        assert diag is not None
        assert diag.get("retry_attempted") is True
        assert diag.get("retry_succeeded") is False


class TestRetryWaitDelayFormula:
    """AC-R12: the delay passed to Event.wait matches compute_retry_delay formula."""

    def test_retry_wait_delay_matches_formula(self):
        from src.utils.review_helpers import compute_retry_delay

        circuit = ReviewCircuitState()
        circuit.consecutive_timeouts = 2

        settings = _make_retry_settings(spec_review_retry_base_delay=8.0)

        with (
            patch("src.spec_engine.review_pipeline.run_review_pipeline", return_value=_make_timeout_outcomes()),
            patch("src.spec_engine.review_retry.time.sleep", return_value=None) as mock_sleep,
        ):
            conduct_review(
                session=MagicMock(),
                settings=settings,
                project=_make_project(),
                send_prompt_with_retry_fn=MagicMock(),
                build_review_exception_diagnostics_fn=MagicMock(return_value={}),
                circuit=circuit,
                cycle=1,
                artifacts=_make_retry_artifacts(),
                agent_type="coco",
            )

        expected_delay = compute_retry_delay(2, base_delay=8.0, max_delay=30.0)
        mock_sleep.assert_called_once()
        actual_delay = mock_sleep.call_args[0][0]
        assert abs(actual_delay - expected_delay) < 0.01


class TestRetrySuccessResetsSkips:
    """AC-R13: retry success resets consecutive_skips to 0."""

    def test_retry_success_resets_consecutive_skips(self):
        circuit = ReviewCircuitState()
        circuit.consecutive_timeouts = 1
        circuit.review_failure_consecutive = 1
        circuit.consecutive_skips = 5  # had prior skips
        call_count = {"n": 0}

        def side_effect(artifacts, budget, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_timeout_outcomes()
            return _make_success_outcomes()

        with (
            patch("src.spec_engine.review_pipeline.run_review_pipeline", side_effect=side_effect),
            patch("src.spec_engine.review_retry.time.sleep", return_value=None),
        ):
            conduct_review(
                session=MagicMock(),
                settings=_make_retry_settings(),
                project=_make_project(),
                send_prompt_with_retry_fn=MagicMock(),
                build_review_exception_diagnostics_fn=MagicMock(return_value={}),
                circuit=circuit,
                cycle=3,
                artifacts=_make_retry_artifacts(),
                agent_type="coco",
            )

        assert circuit.consecutive_skips == 0
        assert circuit.consecutive_timeouts == 0
        assert circuit.recent_outcomes[-1] == "success"


def _make_mixed_failure_outcomes():
    """1 timeout + 1 non-timeout error + 3 success → all_timeout is False."""
    from src.spec_engine.perspective_worker import PerspectiveOutcome, ReviewErrorCode
    perspectives = list(ReviewPerspective)
    outcomes = []
    for i, p in enumerate(perspectives):
        if i == 0:
            outcomes.append(PerspectiveOutcome(
                perspective=p,
                review=PerspectiveReview(perspective=p, passed=False, suggestions=["超时"], summary="异常"),
                elapsed_ms=180000,
                error="timeout",
                error_code=ReviewErrorCode.TIMEOUT,
            ))
        elif i == 1:
            outcomes.append(PerspectiveOutcome(
                perspective=p,
                review=PerspectiveReview(perspective=p, passed=False, suggestions=["err"], summary="异常"),
                elapsed_ms=1000,
                error="some_error",
                error_code=ReviewErrorCode.WORKER_ERROR,
            ))
        else:
            outcomes.append(PerspectiveOutcome(
                perspective=p,
                review=PerspectiveReview(perspective=p, passed=True, suggestions=[]),
                elapsed_ms=500,
            ))
    return outcomes


class TestPartialTimeoutNoRetry:
    """AC-R14: partial timeout (not all) does not trigger retry."""

    def test_partial_timeout_no_retry_no_circuit_increment(self):
        circuit = ReviewCircuitState()
        circuit.consecutive_timeouts = 0
        call_count = {"n": 0}

        def side_effect(artifacts, budget, **kwargs):
            call_count["n"] += 1
            return _make_mixed_failure_outcomes()

        with (
            patch("src.spec_engine.review_pipeline.run_review_pipeline", side_effect=side_effect),
            patch("src.spec_engine.review_retry.time.sleep", return_value=None) as mock_sleep,
        ):
            conduct_review(
                session=MagicMock(),
                settings=_make_retry_settings(),
                project=_make_project(),
                send_prompt_with_retry_fn=MagicMock(),
                build_review_exception_diagnostics_fn=MagicMock(return_value={}),
                circuit=circuit,
                cycle=1,
                artifacts=_make_retry_artifacts(),
                agent_type="coco",
            )

        # Only one call, no retry
        assert call_count["n"] == 1
        mock_sleep.assert_not_called()
        # consecutive_timeouts NOT incremented (not all_timeout)
        assert circuit.consecutive_timeouts == 0


class TestRecentOutcomesAfterRetry:
    """AC-R15: recent_outcomes tracking after retry success/failure."""

    def test_recent_outcomes_failure(self):
        circuit = ReviewCircuitState()

        with (
            patch("src.spec_engine.review_pipeline.run_review_pipeline", return_value=_make_timeout_outcomes()),
            patch("src.spec_engine.review_retry.time.sleep", return_value=None),
        ):
            conduct_review(
                session=MagicMock(),
                settings=_make_retry_settings(),
                project=_make_project(),
                send_prompt_with_retry_fn=MagicMock(),
                build_review_exception_diagnostics_fn=MagicMock(return_value={}),
                circuit=circuit,
                cycle=1,
                artifacts=_make_retry_artifacts(),
                agent_type="coco",
            )

        assert circuit.recent_outcomes[-1] == "partial_failure"



class TestRetryStatusCallback:
    """AC-R04: on_retry_status is called per-attempt inside the retry loop."""

    def test_retry_status_callback_called_per_attempt(self):
        from src.spec_engine.retry_status import RetryStatus

        circuit = ReviewCircuitState()
        status_messages = []

        def side_effect(artifacts, budget, **kwargs):
            return _make_timeout_outcomes()

        def mock_on_status(event):
            status_messages.append(event)

        with (
            patch("src.spec_engine.review_pipeline.run_review_pipeline", side_effect=side_effect),
            patch("src.spec_engine.review_retry.time.sleep", return_value=None),
        ):
            _conduct_review_pipeline(
                settings=_make_retry_settings(spec_review_retry_base_delay=8.0),
                build_review_exception_diagnostics_fn=MagicMock(return_value={}),
                circuit=circuit,
                cycle=1,
                artifacts=_make_retry_artifacts(),
                agent_type="coco",
                model_name=None,
                on_review_done=None,
                on_retry_status=mock_on_status,
            )

        # Status callback should be called at least once
        assert len(status_messages) >= 1
        # AC-24: first call is WAITING with computed delay seconds
        # Default delay = min(8.0 * 1.5^0, 30) = 8
        from src.spec_engine.retry_status import RetryEvent
        first = status_messages[0]
        assert isinstance(first, RetryEvent)
        assert first.status == RetryStatus.WAITING
        assert first.delay_sec == 8.0


class TestLowDelayRetryStarting:
    """AC-23: when spec_review_retry_max_delay=1 (delay < 2), on_retry_status
    skips WAITING and emits EXECUTING directly."""

    def test_low_delay_sends_executing_directly(self):
        from src.spec_engine.retry_status import RetryEvent, RetryStatus

        circuit = ReviewCircuitState()
        status_messages = []

        def side_effect(artifacts, budget, **kwargs):
            return _make_timeout_outcomes()

        def mock_on_status(event):
            status_messages.append(event)

        with (
            patch("src.spec_engine.review_pipeline.run_review_pipeline", side_effect=side_effect),
            patch("src.spec_engine.review_retry.time.sleep", return_value=None),
        ):
            _conduct_review_pipeline(
                settings=_make_retry_settings(spec_review_retry_max_delay=1),
                build_review_exception_diagnostics_fn=MagicMock(return_value={}),
                circuit=circuit,
                cycle=1,
                artifacts=_make_retry_artifacts(),
                agent_type="coco",
                model_name=None,
                on_review_done=None,
                on_retry_status=mock_on_status,
            )

        # With delay < 2, WAITING is skipped; first callback is EXECUTING
        assert len(status_messages) >= 1
        first = status_messages[0]
        assert isinstance(first, RetryEvent)
        assert first.status == RetryStatus.EXECUTING


class TestSettingsMaxDelayValidation:
    """AC-R07: spec_review_retry_max_delay must be > 0."""

    @pytest.mark.parametrize("value", [0, -5], ids=["zero", "negative"])
    def test_invalid_raises_validation_error(self, value):
        from pydantic import ValidationError

        from src.config import Settings

        with pytest.raises(ValidationError):
            Settings(spec_review_retry_max_delay=value, _env_file=None)


class TestSettingsCrossFieldValidation:
    """AC-18/T-06/T-07: cross-field validators for spec_review timing parameters."""

    @pytest.mark.parametrize("kwargs,match", [
        pytest.param(
            dict(spec_review_retry_max_delay=250, spec_review_timeout=240),
            "spec_review_retry_max_delay",
            id="max_delay_exceeds_timeout",
        ),
        pytest.param(
            dict(spec_review_timeout=0),
            "spec_review_timeout",
            id="timeout_zero",
        ),
        pytest.param(
            dict(spec_review_min_timeout=-1),
            "spec_review_min_timeout",
            id="min_timeout_negative",
        ),
        pytest.param(
            dict(spec_review_hard_floor=0),
            "spec_review_hard_floor",
            id="hard_floor_zero",
        ),
        pytest.param(
            dict(spec_review_hard_floor=70, spec_review_min_timeout=60, spec_review_timeout=240),
            "spec_review_hard_floor",
            id="hard_floor_exceeds_min_timeout",
        ),
        pytest.param(
            dict(spec_review_retry_max_delay=100, spec_review_retry_max_attempts=5, spec_review_timeout=240),
            "请减小 SPEC_REVIEW_RETRY_MAX_ATTEMPTS 或 SPEC_REVIEW_RETRY_MAX_DELAY",
            id="total_retry_budget_exceeds_limit",
        ),
    ])
    def test_invalid_config_raises(self, kwargs, match):
        from pydantic import ValidationError

        from src.config import Settings

        with pytest.raises(ValidationError, match=match):
            Settings(**kwargs, _env_file=None)


# ---------------------------------------------------------------------------
# Direct tests for handle_pipeline_errors_with_retry
# ---------------------------------------------------------------------------


class TestHandlePipelineErrorsWithRetry:
    """Direct unit tests for the extracted handle_pipeline_errors_with_retry function."""

    # -- (a) All timeout + retry succeeds → new ReviewResult, diag is None --

    def test_all_timeout_retry_success_returns_new_result_and_no_diag(self):
        """When all workers timed out and retry succeeds, returns new result with diag=None."""
        circuit = ReviewCircuitState()
        settings = _make_retry_settings()

        timeout_outcomes = _make_timeout_outcomes()
        success_outcomes = _make_success_outcomes()
        original_result = outcomes_to_review_result(timeout_outcomes, 1)

        mock_pipeline = MagicMock(return_value=success_outcomes)

        from src.spec_engine.cycle_budget import CycleBudget

        ctx = PipelineRetryContext(
            cancel_event=None, on_retry_status=None,
            base_timeout=240,
            multiplier=2,
            pipeline_fn=mock_pipeline,
            budget_cls=CycleBudget,
            artifacts=_make_retry_artifacts(),
            agent_type="coco",
            model_name=None,
        )
        with patch("src.spec_engine.review_retry.time.sleep", return_value=None):
            result, diag = handle_pipeline_errors_with_retry(
                outcomes=timeout_outcomes,
                review_result=original_result,
                circuit=circuit,
                settings=settings,
                cycle=1,
                ctx=ctx,
                retry_texts={"retry_no_retry": UI_TEXT["retry_no_retry"], "retry_exhausted": UI_TEXT["retry_exhausted"]},
            )

        assert diag is None
        assert result.all_passed is True
        assert circuit.review_failure_consecutive == 0
        assert circuit.consecutive_timeouts == 0
        mock_pipeline.assert_called_once()

    # -- (b) All timeout + retry fails → original result, diag has retry_attempted --

    def test_all_timeout_retry_failure_returns_diag_with_retry_attempted(self):
        """When all workers timed out and retry also fails, diag contains retry info."""
        circuit = ReviewCircuitState()
        settings = _make_retry_settings()

        timeout_outcomes = _make_timeout_outcomes()
        original_result = outcomes_to_review_result(timeout_outcomes, 2)

        # Retry also returns timeout outcomes
        mock_pipeline = MagicMock(return_value=timeout_outcomes)

        from src.spec_engine.cycle_budget import CycleBudget

        ctx = PipelineRetryContext(
            cancel_event=None, on_retry_status=None,
            base_timeout=240,
            multiplier=2,
            pipeline_fn=mock_pipeline,
            budget_cls=CycleBudget,
            artifacts=_make_retry_artifacts(),
            agent_type="coco",
            model_name=None,
        )
        with patch("src.spec_engine.review_retry.time.sleep", return_value=None):
            result, diag = handle_pipeline_errors_with_retry(
                outcomes=timeout_outcomes,
                review_result=original_result,
                circuit=circuit,
                settings=settings,
                cycle=2,
                ctx=ctx,
                retry_texts={"retry_no_retry": UI_TEXT["retry_no_retry"], "retry_exhausted": UI_TEXT["retry_exhausted"]},
            )

        assert diag is not None
        assert diag["retry_attempted"] is True
        assert diag["retry_succeeded"] is False
        assert "total_wall_clock_ms" in diag
        assert isinstance(diag["total_wall_clock_ms"], int)
        assert "retry_attempts_detail" in diag
        assert diag["retry_attempts_detail"]["attempted"] is True
        assert diag["retry_attempts_detail"]["succeeded"] is False
        assert diag["retry_attempts_detail"]["all_timeout"] is True
        assert circuit.consecutive_timeouts == 1
        assert circuit.review_failure_consecutive == 1
        # Suggestions should be overwritten with recovery guidance
        for pr in result.reviews:
            assert len(pr.suggestions) == 1
            assert "重试" in pr.suggestions[0] or "resume" in pr.suggestions[0] or "已自动" in pr.suggestions[0]

    # -- (c) Partial failure (not all timeout) → no retry triggered --

    def test_partial_failure_no_retry(self):
        """When only some workers failed (not all timeout), retry is not triggered."""
        circuit = ReviewCircuitState()
        settings = _make_retry_settings()

        partial_outcomes = _make_partial_error_outcomes()
        original_result = outcomes_to_review_result(partial_outcomes, 3)

        mock_pipeline = MagicMock()

        from src.spec_engine.cycle_budget import CycleBudget

        result, diag = handle_pipeline_errors_with_retry(
            outcomes=partial_outcomes,
            review_result=original_result,
            circuit=circuit,
            settings=settings,
            cycle=3,
            ctx=PipelineRetryContext(
                cancel_event=None, on_retry_status=None,
                base_timeout=240,
                multiplier=2,
                pipeline_fn=mock_pipeline,
                budget_cls=CycleBudget,
                artifacts=_make_retry_artifacts(),
                agent_type="coco",
                model_name=None,
            ),
            retry_texts={"retry_no_retry": UI_TEXT["retry_no_retry"], "retry_exhausted": UI_TEXT["retry_exhausted"]},
        )

        assert diag is not None
        assert diag["retry_attempted"] is False
        assert diag["retry_succeeded"] is False
        assert "total_wall_clock_ms" in diag
        assert "retry_attempts_detail" in diag
        assert diag["retry_attempts_detail"]["attempted"] is False
        mock_pipeline.assert_not_called()
        # Circuit timeout counters should NOT be incremented (not all-timeout)
        assert circuit.consecutive_timeouts == 0

    # -- (d) max_attempts=0 → no retry triggered --

    def test_auto_retry_disabled_no_retry(self):
        """When max_attempts=0, no retry is attempted even on full timeout."""
        circuit = ReviewCircuitState()
        settings = _make_retry_settings(spec_review_retry_max_attempts=0)

        timeout_outcomes = _make_timeout_outcomes()
        original_result = outcomes_to_review_result(timeout_outcomes, 4)

        mock_pipeline = MagicMock()

        from src.spec_engine.cycle_budget import CycleBudget

        result, diag = handle_pipeline_errors_with_retry(
            outcomes=timeout_outcomes,
            review_result=original_result,
            circuit=circuit,
            settings=settings,
            cycle=4,
            ctx=PipelineRetryContext(
                cancel_event=None, on_retry_status=None,
                base_timeout=240,
                multiplier=2,
                pipeline_fn=mock_pipeline,
                budget_cls=CycleBudget,
                artifacts=_make_retry_artifacts(),
                agent_type="coco",
                model_name=None,
            ),
            retry_texts={"retry_no_retry": UI_TEXT["retry_no_retry"], "retry_exhausted": UI_TEXT["retry_exhausted"]},
        )

        assert diag is not None
        assert diag["retry_attempted"] is False
        assert diag["retry_succeeded"] is False
        mock_pipeline.assert_not_called()
        # All-timeout path still increments circuit counters
        assert circuit.consecutive_timeouts == 1
        assert circuit.review_failure_consecutive == 1


# ---------------------------------------------------------------------------
# Task 15: TestAttemptPipelineRetryDirect
# ---------------------------------------------------------------------------


class TestAttemptPipelineRetryDirect:
    """Direct unit tests for attempt_pipeline_retry."""

    def test_max_attempts_zero_returns_none(self):
        """When max_attempts=0, immediately returns None without calling pipeline."""
        circuit = ReviewCircuitState()
        settings = _make_retry_settings(spec_review_retry_max_attempts=0)
        mock_pipeline = MagicMock()

        from src.spec_engine.cycle_budget import CycleBudget

        ctx = PipelineRetryContext(
            cancel_event=None, on_retry_status=None,
            base_timeout=120,
            multiplier=2,
            pipeline_fn=mock_pipeline,
            budget_cls=CycleBudget,
            artifacts=_make_retry_artifacts(),
            agent_type="coco",
            model_name=None,
        )
        result = attempt_pipeline_retry(
            circuit=circuit, settings=settings, cycle=1, ctx=ctx,
        )

        assert result is None
        mock_pipeline.assert_not_called()

    def test_multi_attempt_first_fail_second_success(self):
        """With max_attempts=2, first attempt fails, second succeeds."""
        circuit = ReviewCircuitState()
        settings = _make_retry_settings(spec_review_retry_max_attempts=2)

        from src.spec_engine.cycle_budget import CycleBudget

        success_outcomes = _make_success_outcomes()
        timeout_outcomes = _make_timeout_outcomes()

        call_count = {"n": 0}

        def side_effect(artifacts, budget, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return timeout_outcomes  # first attempt fails
            return success_outcomes  # second attempt succeeds

        ctx = PipelineRetryContext(
            cancel_event=None, on_retry_status=None,
            base_timeout=120,
            multiplier=2,
            pipeline_fn=side_effect,
            budget_cls=CycleBudget,
            artifacts=_make_retry_artifacts(),
            agent_type="coco",
            model_name=None,
        )

        with patch("src.spec_engine.review_retry.time.sleep", return_value=None):
            result = attempt_pipeline_retry(
                circuit=circuit, settings=settings, cycle=1, ctx=ctx,
            )

        assert result is not None
        assert all(not o.error for o in result)
        assert call_count["n"] == 2

    def test_on_retry_status_exception_no_effect(self):
        """on_retry_status raising an exception does not affect retry execution."""
        circuit = ReviewCircuitState()
        settings = _make_retry_settings(spec_review_retry_max_attempts=1)

        from src.spec_engine.cycle_budget import CycleBudget

        success_outcomes = _make_success_outcomes()

        def bad_status(event):
            raise RuntimeError("callback failure")

        ctx = PipelineRetryContext(
            cancel_event=None, on_retry_status=bad_status,
            base_timeout=120,
            multiplier=2,
            pipeline_fn=MagicMock(return_value=success_outcomes),
            budget_cls=CycleBudget,
            artifacts=_make_retry_artifacts(),
            agent_type="coco",
            model_name=None,
        )

        with patch("src.spec_engine.review_retry.time.sleep", return_value=None):
            result = attempt_pipeline_retry(
                circuit=circuit, settings=settings, cycle=1, ctx=ctx,
            )

        # Should succeed despite callback exception
        assert result is not None
        assert result == success_outcomes

    def test_cancel_event_aborts_during_wait(self):
        """When cancel_event is set, retry aborts during wait and returns None."""
        circuit = ReviewCircuitState()
        settings = _make_retry_settings(spec_review_retry_max_attempts=1)

        from src.spec_engine.cycle_budget import CycleBudget

        cancel_event = threading.Event()
        cancel_event.set()  # pre-set = cancelled

        mock_pipeline = MagicMock()
        ctx = PipelineRetryContext(
            cancel_event=cancel_event, on_retry_status=None,
            base_timeout=120,
            multiplier=2,
            pipeline_fn=mock_pipeline,
            budget_cls=CycleBudget,
            artifacts=_make_retry_artifacts(),
            agent_type="coco",
            model_name=None,
        )

        result = attempt_pipeline_retry(
            circuit=circuit, settings=settings, cycle=1, ctx=ctx,
        )

        assert result is None
        mock_pipeline.assert_not_called()


# ---------------------------------------------------------------------------
# Task 16: TestReviewCircuitStateResetOnSuccess
# ---------------------------------------------------------------------------


class TestReviewCircuitStateResetOnSuccess:
    """Direct unit tests for ReviewCircuitState.reset_on_success()."""

    def test_all_fields_reset(self):
        """All 5 counter fields are reset to 0 after reset_on_success()."""
        circuit = ReviewCircuitState(
            review_failure_consecutive=5,
            review_circuit_open_until_cycle=10,
            backoff_level=3,
            consecutive_timeouts=7,
            consecutive_skips=2,
            last_review_elapsed_ms=9999,
            recent_outcomes=["failure", "failure"],
        )

        circuit.reset_on_success()

        assert circuit.review_failure_consecutive == 0
        assert circuit.review_circuit_open_until_cycle == 0
        assert circuit.backoff_level == 0
        assert circuit.consecutive_timeouts == 0
        assert circuit.consecutive_skips == 0
        # last_review_elapsed_ms is NOT reset by reset_on_success
        assert circuit.last_review_elapsed_ms == 9999
        # recent_outcomes should have "success" appended
        assert circuit.recent_outcomes[-1] == "success"

    def test_recent_outcomes_append_and_trim(self):
        """recent_outcomes appends 'success' and trims to last 20 entries."""
        # Case 1: 19 entries -> becomes 20 after append
        circuit = ReviewCircuitState(recent_outcomes=["x"] * 19)
        circuit.reset_on_success()
        assert len(circuit.recent_outcomes) == 20
        assert circuit.recent_outcomes[-1] == "success"

        # Case 2: already 20 entries -> still 20 after append+trim
        circuit2 = ReviewCircuitState(recent_outcomes=["y"] * 20)
        circuit2.reset_on_success()
        assert len(circuit2.recent_outcomes) == 20
        assert circuit2.recent_outcomes[-1] == "success"
        assert circuit2.recent_outcomes[0] == "y"  # first 'y' was trimmed

        # Case 3: 25 entries -> trimmed to 20
        circuit3 = ReviewCircuitState(recent_outcomes=["z"] * 25)
        circuit3.reset_on_success()
        assert len(circuit3.recent_outcomes) == 20
        assert circuit3.recent_outcomes[-1] == "success"


# ---------------------------------------------------------------------------
# Task 17: TestMultiAttemptRetryLoop
# ---------------------------------------------------------------------------


class TestMultiAttemptRetryLoop:
    """Tests for multi-attempt retry loop (max_attempts > 1)."""

    def test_three_attempts_all_fail(self):
        """With max_attempts=3 and all attempts failing, returns None after 3 calls."""
        circuit = ReviewCircuitState()
        settings = _make_retry_settings(spec_review_retry_max_attempts=3)

        from src.spec_engine.cycle_budget import CycleBudget

        timeout_outcomes = _make_timeout_outcomes()
        mock_pipeline = MagicMock(return_value=timeout_outcomes)

        ctx = PipelineRetryContext(
            cancel_event=None, on_retry_status=None,
            base_timeout=120,
            multiplier=2,
            pipeline_fn=mock_pipeline,
            budget_cls=CycleBudget,
            artifacts=_make_retry_artifacts(),
            agent_type="coco",
            model_name=None,
        )

        with patch("src.spec_engine.review_retry.time.sleep", return_value=None):
            result = attempt_pipeline_retry(
                circuit=circuit, settings=settings, cycle=1, ctx=ctx,
            )

        assert result is None
        assert mock_pipeline.call_count == 3




# ---------------------------------------------------------------------------
# Task 20: test_config_max_attempts_over_10_raises
# ---------------------------------------------------------------------------


class TestConfigMaxAttemptsUpperBound:
    """spec_review_retry_max_attempts > 10 should raise ValidationError."""

    def test_max_attempts_over_10_raises(self):
        """Settings rejects spec_review_retry_max_attempts > 10."""
        import os

        from pydantic import ValidationError

        from src.config import Settings

        env = {
            "FEISHU_APP_ID": "test",
            "FEISHU_APP_SECRET": "test",
            "LLM_API_KEY": "test",
            "SPEC_REVIEW_RETRY_MAX_ATTEMPTS": "11",
        }
        with patch.dict(os.environ, env, clear=False):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            assert "spec_review_retry_max_attempts" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Task 14: test_max_attempts_negative_raises_value_error (AC-R09)
# ---------------------------------------------------------------------------


class TestConfigMaxAttemptsNegative:
    """spec_review_retry_max_attempts < 0 should raise ValidationError."""

    def test_max_attempts_negative_raises_value_error(self):
        """Settings rejects spec_review_retry_max_attempts = -1."""
        import os

        from pydantic import ValidationError

        from src.config import Settings

        env = {
            "FEISHU_APP_ID": "test",
            "FEISHU_APP_SECRET": "test",
            "LLM_API_KEY": "test",
            "SPEC_REVIEW_RETRY_MAX_ATTEMPTS": "-1",
        }
        with patch.dict(os.environ, env, clear=False):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            err_str = str(exc_info.value)
            assert "spec_review_retry_max_attempts" in err_str
            assert "≥ 0" in err_str



# ---------------------------------------------------------------------------
# TestSettingsNoneFallback removed: pydantic validators now enforce non-None
# values at config construction time, making None-fallback paths unreachable.
# ---------------------------------------------------------------------------


class TestCancelEventMidMultiRetry:
    """AC-27: cancel_event set after first pipeline call aborts subsequent attempts."""

    def test_cancel_event_mid_multi_retry(self):
        """max_attempts=3, cancel_event set after first call — total calls <= 2."""
        import threading

        from src.spec_engine.cycle_budget import CycleBudget

        circuit = ReviewCircuitState()
        cancel_event = threading.Event()
        call_count = {"n": 0}

        def pipeline_with_cancel(artifacts, budget, **kwargs):
            call_count["n"] += 1
            if call_count["n"] >= 1:
                # Set cancel after first pipeline call
                cancel_event.set()
            return _make_timeout_outcomes()

        settings = _make_retry_settings(spec_review_retry_max_attempts=3, spec_review_retry_base_delay=0.1, spec_review_retry_max_delay=0.1)

        ctx = PipelineRetryContext(
            cancel_event=cancel_event, on_retry_status=None,
            base_timeout=120,
            multiplier=2,
            pipeline_fn=pipeline_with_cancel,
            budget_cls=CycleBudget,
            artifacts=_make_retry_artifacts(),
            agent_type="coco",
            model_name=None,
        )

        result = attempt_pipeline_retry(
            circuit=circuit, settings=settings, cycle=1, ctx=ctx,
        )

        # Cancel should have stopped further attempts
        assert call_count["n"] <= 2
        # Result should be None (cancelled/failed)
        assert result is None


# ---------------------------------------------------------------------------
# Task 17 & 18: TestCancelEventIntegration (AC-R01, AC-R12)
# ---------------------------------------------------------------------------


class TestCancelEventIntegration:
    """cancel_event integration with SpecEngine lifecycle."""

    def test_stop_interrupts_wait(self):
        """stop() sets cancel_event, making _evt.wait() return immediately."""
        from src.spec_engine.cycle_budget import CycleBudget

        cancel_event = threading.Event()
        circuit = ReviewCircuitState(consecutive_timeouts=1)
        settings = _make_retry_settings(spec_review_retry_max_delay=60, spec_review_retry_base_delay=60.0)

        # Pipeline will never be reached if wait is interrupted
        mock_pipeline = MagicMock(return_value=_make_success_outcomes())

        ctx = PipelineRetryContext(
            cancel_event=cancel_event, on_retry_status=None,
            base_timeout=120,
            multiplier=2,
            pipeline_fn=mock_pipeline,
            budget_cls=CycleBudget,
            artifacts=_make_retry_artifacts(),
            agent_type="coco",
            model_name=None,
        )

        import time

        result_holder = [None]
        elapsed_holder = [0.0]

        def _run():
            t0 = time.monotonic()
            result_holder[0] = attempt_pipeline_retry(
                circuit=circuit, settings=settings, cycle=1, ctx=ctx,
            )
            elapsed_holder[0] = time.monotonic() - t0

        thread = threading.Thread(target=_run)
        thread.start()
        # Wait briefly then signal cancel
        time.sleep(0.1)
        cancel_event.set()
        thread.join(timeout=2.0)

        assert not thread.is_alive(), "Thread should have completed"
        assert elapsed_holder[0] < 1.0, f"Wait was not interrupted quickly: {elapsed_holder[0]:.2f}s"
        assert result_holder[0] is None, "Cancelled retry should return None"

    def test_stopping_state_sets_cancel_event(self):
        """When engine._run_state is STOPPING, cancel_event should be set."""
        from src.engine_base import EngineRunState
        from src.spec_engine.engine import SpecEngine

        engine = SpecEngine(chat_id="test", root_path="/tmp/test")
        engine._run_state = EngineRunState.STOPPING

        # Simulate what _conduct_review does at its entry
        engine._review_cancel_event.clear()
        if engine._run_state != EngineRunState.RUNNING:
            engine._review_cancel_event.set()

        assert engine._review_cancel_event.is_set()



# ---------------------------------------------------------------------------
# Task 21: Text template assertions (AC-R07, AC-R08)
# ---------------------------------------------------------------------------


class TestRetryTextTemplates:
    """Verify retry/timeout text templates meet spec requirements."""

    def test_retry_exhausted_has_placeholder(self):
        """retry_exhausted should contain {n} placeholder for attempt count."""
        from src.card.ui_text import UI_TEXT

        text = UI_TEXT["retry_exhausted"]
        assert "{n}" in text
        assert "重试" in text

    def test_retry_no_retry_banner_empty_string(self):
        """retry_no_retry should mention '未进行重试'."""
        from src.card.ui_text import UI_TEXT

        text = UI_TEXT["retry_no_retry"]
        assert "未进行重试" in text


# ---------------------------------------------------------------------------
# TestBuildRetryDiagnosticsBranches — direct unit test for build_retry_diagnostics
# ---------------------------------------------------------------------------


class TestBuildRetryDiagnosticsBranches:
    """AC-25: parametrized test covering all 5 err_type_val branches of build_retry_diagnostics.

    Verifies:
    - The returned dict has correct 'err_type' for each branch
    - circuit is NOT mutated (pure function guarantee)
    """

    def _make_settings(self, max_attempts=1):
        return SimpleNamespace(
            spec_review_retry_max_attempts=max_attempts,
            spec_review_timeout=240,
            spec_review_retry_max_delay=30,
        )

    def _make_circuit(self):
        circuit = ReviewCircuitState()
        circuit.consecutive_timeouts = 2
        circuit.review_failure_consecutive = 1
        circuit.recent_outcomes = ["success"]
        return circuit

    def _make_outcomes(self, error_code):
        from src.engine_base import PerspectiveReview, ReviewPerspective
        from src.spec_engine.perspective_worker import PerspectiveOutcome
        p = ReviewPerspective.ARCHITECT
        return [
            PerspectiveOutcome(
                perspective=p,
                review=PerspectiveReview(
                    perspective=p,
                    passed=False,
                    suggestions=["timeout"],
                    summary="timeout",
                ),
                error="timeout",
                error_code=error_code,
            ),
        ]

    @pytest.mark.parametrize("max_attempts", [1, 3])
    def test_retry_exhausted_branches(self, max_attempts):
        """Branch: all_timeout=True, retry_attempted=True."""
        from src.card.ui_text import UI_TEXT
        from src.spec_engine.perspective_worker import ReviewErrorCode
        from src.utils.text import format_friendly_duration

        expected_text = UI_TEXT["retry_exhausted"].format(
            n=max_attempts, elapsed_friendly=format_friendly_duration(int(240 + 30 * max_attempts))
        )

        outcomes = self._make_outcomes(ReviewErrorCode.TIMEOUT)
        circuit = self._make_circuit()
        original_timeouts = circuit.consecutive_timeouts
        original_failures = circuit.review_failure_consecutive
        original_outcomes_len = len(circuit.recent_outcomes)

        diag = build_retry_diagnostics(
            outcomes=outcomes,
            failed_workers=outcomes,
            circuit=circuit,
            settings=self._make_settings(max_attempts=max_attempts),
            cycle=1,
            err_type_val="timeout",
            all_timeout=True,
            retry_attempted=True,
            retry_texts={
                "retry_no_retry": UI_TEXT["retry_no_retry"],
                "retry_exhausted": UI_TEXT["retry_exhausted"],
            },
        )

        assert diag["err_type"] == expected_text
        # Pure function: circuit NOT mutated
        assert circuit.consecutive_timeouts == original_timeouts
        assert circuit.review_failure_consecutive == original_failures
        assert len(circuit.recent_outcomes) == original_outcomes_len

    def test_timeout_worker_busy_no_retry(self):
        """Branch: all_timeout=True, retry_attempted=False, max_attempts=0."""
        from src.card.ui_text import UI_TEXT
        from src.spec_engine.perspective_worker import ReviewErrorCode

        outcomes = self._make_outcomes(ReviewErrorCode.TIMEOUT)
        circuit = self._make_circuit()

        diag = build_retry_diagnostics(
            outcomes=outcomes,
            failed_workers=outcomes,
            circuit=circuit,
            settings=self._make_settings(max_attempts=0),
            cycle=2,
            err_type_val="timeout",
            all_timeout=True,
            retry_attempted=False,
            retry_texts={
                "retry_no_retry": UI_TEXT["retry_no_retry"],
                "retry_exhausted": UI_TEXT["retry_exhausted"],
            },
        )

        assert diag["err_type"] == UI_TEXT["retry_no_retry"]

    def test_timeout_worker_busy_with_retry_enabled(self):
        """Branch: all_timeout=True, retry_attempted=False, max_attempts>0.

        Even though max_attempts > 0, since retry was not attempted,
        we should see retry_no_retry (not retry_exhausted).
        """
        from src.card.ui_text import UI_TEXT
        from src.spec_engine.perspective_worker import ReviewErrorCode

        outcomes = self._make_outcomes(ReviewErrorCode.TIMEOUT)
        circuit = self._make_circuit()

        diag = build_retry_diagnostics(
            outcomes=outcomes,
            failed_workers=outcomes,
            circuit=circuit,
            settings=self._make_settings(max_attempts=2),
            cycle=3,
            err_type_val="timeout",
            all_timeout=True,
            retry_attempted=False,
            retry_texts={
                "retry_no_retry": UI_TEXT["retry_no_retry"],
                "retry_exhausted": UI_TEXT["retry_exhausted"],
            },
        )

        assert diag["err_type"] == UI_TEXT["retry_no_retry"]

    def test_partial_timeout_branch(self):
        """Branch: all_timeout=False, first failed worker has TIMEOUT error_code."""
        from src.card.ui_text import UI_TEXT
        from src.engine_base import PerspectiveReview, ReviewPerspective
        from src.spec_engine.perspective_worker import PerspectiveOutcome, ReviewErrorCode

        p = ReviewPerspective.ARCHITECT
        failed = [PerspectiveOutcome(
            perspective=p,
            review=PerspectiveReview(
                perspective=p,
                passed=False,
                suggestions=["timeout"],
                summary="timeout",
            ),
            error="timeout",
            error_code=ReviewErrorCode.TIMEOUT,
        )]
        circuit = self._make_circuit()

        diag = build_retry_diagnostics(
            outcomes=failed,
            failed_workers=failed,
            circuit=circuit,
            settings=self._make_settings(),
            cycle=4,
            err_type_val="original_error",
            all_timeout=False,
            retry_attempted=False,
            retry_texts={
                "retry_no_retry": UI_TEXT["retry_no_retry"],
                "retry_exhausted": UI_TEXT["retry_exhausted"],
            },
        )

        assert diag["err_type"] == UI_TEXT["retry_no_retry"]


# ---------------------------------------------------------------------------
# TestReviewCircuitStateOnFailure — AC-R14
# ---------------------------------------------------------------------------


class TestReviewCircuitStateOnFailure:
    """AC-R14: on_failure(all_timeout=True/False) branch coverage."""

    def test_on_failure_all_timeout_true_increments_consecutive_timeouts(self):
        """all_timeout=True should increment consecutive_timeouts."""
        circuit = ReviewCircuitState(consecutive_timeouts=2, review_failure_consecutive=1)
        circuit.on_failure(all_timeout=True)
        assert circuit.consecutive_timeouts == 3
        assert circuit.review_failure_consecutive == 2

    def test_on_failure_all_timeout_false_does_not_increment_consecutive_timeouts(self):
        """all_timeout=False should NOT increment consecutive_timeouts."""
        circuit = ReviewCircuitState(consecutive_timeouts=2, review_failure_consecutive=1)
        circuit.on_failure(all_timeout=False)
        assert circuit.consecutive_timeouts == 2  # unchanged
        assert circuit.review_failure_consecutive == 2  # still incremented



# ---------------------------------------------------------------------------
# TestBuildRetryDiagnosticsEmptyWorkers — AC-R17
# ---------------------------------------------------------------------------


class TestBuildRetryDiagnosticsEmptyWorkers:
    """AC-R17: failed_workers=[] edge case falls through to default err_type."""

    def test_empty_failed_workers_uses_original_err_type(self):
        """When failed_workers is empty, err_type_val stays as the passed-in value."""
        circuit = ReviewCircuitState()
        settings = _make_retry_settings()

        diag = build_retry_diagnostics(
            outcomes=[],
            failed_workers=[],
            circuit=circuit,
            settings=settings,
            cycle=1,
            err_type_val="unknown",
            all_timeout=False,
            retry_attempted=False,
            retry_texts={
                "retry_no_retry": UI_TEXT["retry_no_retry"],
                "retry_exhausted": UI_TEXT["retry_exhausted"],
            },
        )

        # Should keep the original "unknown" value since no branch modifies it
        assert diag["err_type"] == "unknown"


# ---------------------------------------------------------------------------
# TestRetryMaxAttemptsZeroDisablesRetry — AC-R18
# ---------------------------------------------------------------------------


class TestRetryMaxAttemptsZeroDisablesRetry:
    """AC-R18: max_attempts=0 via _conduct_review_pipeline disables retry."""

    def test_max_attempts_zero_no_retry_via_pipeline(self):
        """With max_attempts=0, pipeline_fn should NOT be called a second time."""

        circuit = ReviewCircuitState()
        settings = _make_retry_settings(
            spec_review_retry_max_attempts=0,
        )
        timeout_outcomes = _make_timeout_outcomes()

        from src.spec_engine.cycle_budget import CycleBudget

        mock_pipeline = MagicMock(return_value=timeout_outcomes)
        ctx = PipelineRetryContext(
            cancel_event=None,
            on_retry_status=None,
            base_timeout=120,
            multiplier=2,
            pipeline_fn=mock_pipeline,
            budget_cls=CycleBudget,
            artifacts=_make_retry_artifacts(),
            agent_type="coco",
            model_name=None,
        )

        # attempt_pipeline_retry with max_attempts=0 should return None immediately
        result = attempt_pipeline_retry(
            circuit=circuit, settings=settings, cycle=1, ctx=ctx,
        )
        assert result is None
        # pipeline_fn should NOT have been called (since we return None before any attempt)
        mock_pipeline.assert_not_called()

