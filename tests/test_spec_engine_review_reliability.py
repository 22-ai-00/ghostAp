"""Integration tests for spec engine review reliability.

Covers:
- classify_timeout heuristic for startup-phase "Internal error" failures
- RoleReviewWorker startup failure detection and error message differentiation
- Startup retry with exponential backoff
- Non-blocking role degradation (skip on startup failure)
- Full pipeline: startup failure -> retry -> skip non-blocking roles
"""

from unittest.mock import MagicMock, patch

import pytest

from src.engine_base import ReviewPerspective
from src.spec_engine.adaptive_review import (
    RoleReviewWorker,
    _outcomes_to_review_result,
    run_adaptive_role_review_pipeline,
)
from src.spec_engine.review_aggregation import RoleReviewOutcome
from src.spec_engine.review_artifacts import ReviewArtifacts
from src.spec_engine.review_roles import ReviewRoleSpec
from src.utils.errors import classify_timeout

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_role(role_id: str = "test_role", *, blocking: bool = True, category: str = "software") -> ReviewRoleSpec:
    return ReviewRoleSpec(
        role_id=role_id,
        display_name=f"Test Role {role_id}",
        category=category,
        mission="Test mission",
        review_focus=["test focus"],
        must_check=["test check"],
        evidence_policy="Test evidence",
        blocking=blocking,
        base_perspective=ReviewPerspective.ARCHITECT,
    )


def _make_artifacts() -> ReviewArtifacts:
    return ReviewArtifacts(
        cycle_number=1,
        requirement="Test requirement",
        cwd="/tmp/test",
        diff_patch="diff --git a/test.py b/test.py\n+new line\n",
        touched_files=["test.py"],
    )


# ---------------------------------------------------------------------------
# classify_timeout heuristic
# ---------------------------------------------------------------------------


class TestClassifyTimeoutStartupHeuristic:
    """Verify the "Internal error" + elapsed-time heuristic."""

    def test_eighty_percent_threshold(self):
        """At exactly 80% of timeout, should be True."""
        exc = RuntimeError("Internal error")
        assert classify_timeout(exc, elapsed_s=16.0, timeout_s=20) is True

    def test_below_eighty_percent(self):
        exc = RuntimeError("Internal error")
        assert classify_timeout(exc, elapsed_s=15.9, timeout_s=20) is False


# ---------------------------------------------------------------------------
# RoleReviewWorker — startup vs execution failure distinction
# ---------------------------------------------------------------------------


class TestRoleReviewWorkerFailureTypes:
    """Verify RoleReviewWorker distinguishes startup from execution failures."""

    def test_startup_failure_detected_via_exception_attrs(self):
        """Fast startup failure (not a timeout) is blocking."""
        role = _make_role("architect")
        worker = RoleReviewWorker(role, timeout=240.0)
        artifacts = _make_artifacts()

        def failing_runner(prompt, on_event, timeout):
            exc = RuntimeError("connection refused")
            exc.startup_elapsed_s = 2.0
            exc.startup_timeout_s = 30.0
            exc.startup_failed = True
            raise exc

        outcome = worker.run(artifacts, failing_runner)
        assert outcome.passed is False
        assert outcome.blocking is True
        assert "启动失败" in outcome.summary
        assert outcome.suggestions[0].severity == "observation"
        assert "startup failed" in outcome.suggestions[0].evidence

    def test_blocking_role_startup_timeout_stays_blocking(self):
        """Blocking role with startup timeout stays blocking (fail-closed)."""
        role = _make_role("architect")
        worker = RoleReviewWorker(role, timeout=240.0)
        artifacts = _make_artifacts()

        def failing_runner(prompt, on_event, timeout):
            exc = RuntimeError("Internal error: session start timed out")
            exc.startup_elapsed_s = 29.5
            exc.startup_timeout_s = 30.0
            exc.startup_failed = True
            raise exc

        outcome = worker.run(artifacts, failing_runner)
        assert outcome.passed is False
        assert outcome.blocking is True
        assert "启动超时" in outcome.summary
        assert outcome.suggestions[0].severity == "major"
        assert "startup timed out" in outcome.suggestions[0].evidence

    def test_blocking_role_execution_timeout_stays_blocking(self):
        """Blocking role with execution timeout stays blocking (fail-closed)."""
        role = _make_role("architect")
        worker = RoleReviewWorker(role, timeout=240.0)
        artifacts = _make_artifacts()

        def failing_runner(prompt, on_event, timeout):
            raise TimeoutError("prompt execution timed out")

        outcome = worker.run(artifacts, failing_runner)
        assert outcome.passed is False
        assert outcome.blocking is True
        assert "审查超时" in outcome.summary
        assert "阻断" in outcome.summary
        assert "已降级" not in outcome.summary
        assert "timeout_blocking" in outcome.error

    def test_execution_failure_not_startup(self):
        role = _make_role("architect")
        worker = RoleReviewWorker(role, timeout=240.0)
        artifacts = _make_artifacts()

        def failing_runner(prompt, on_event, timeout):
            raise RuntimeError("something broke during execution")

        outcome = worker.run(artifacts, failing_runner)
        assert outcome.passed is False
        assert outcome.blocking is True
        assert "审查异常" in outcome.summary
        assert outcome.skipped is False

    def test_successful_run_returns_parsed_outcome(self):
        role = _make_role("architect")
        worker = RoleReviewWorker(role, timeout=240.0)
        artifacts = _make_artifacts()

        def success_runner(prompt, on_event, timeout):
            return '{"role_id": "architect", "verdict": "PASS", "summary": "all good", "suggestions": []}'

        outcome = worker.run(artifacts, success_runner)
        assert outcome.passed is True
        assert outcome.role_id == "architect"
        assert outcome.skipped is False


# ---------------------------------------------------------------------------
# RoleReviewWorker — non-blocking role degradation
# ---------------------------------------------------------------------------


class TestRoleReviewWorkerDegradation:
    """Verify non-blocking roles are skipped on startup failure."""

    def test_non_blocking_role_skipped_on_startup_failure(self):
        role = _make_role("doc_writer", blocking=False, category="writing")
        worker = RoleReviewWorker(role, timeout=240.0)
        artifacts = _make_artifacts()

        def failing_runner(prompt, on_event, timeout):
            exc = RuntimeError("Internal error")
            exc.startup_elapsed_s = 28.0
            exc.startup_timeout_s = 30.0
            exc.startup_failed = True
            raise exc

        outcome = worker.run(artifacts, failing_runner)
        assert outcome.passed is True
        assert outcome.skipped is True
        assert outcome.blocking is False
        assert "跳过" in outcome.summary
        assert "启动失败" in outcome.summary
        assert outcome.suggestions[0].severity == "observation"
        assert outcome.suggestions[0].blocking is False

    def test_blocking_role_remains_blocking_on_startup_timeout(self):
        """Blocking roles stay blocking on startup timeout (fail-closed semantics)."""
        role = _make_role("security_audit", blocking=True, category="security")
        worker = RoleReviewWorker(role, timeout=240.0)
        artifacts = _make_artifacts()

        def failing_runner(prompt, on_event, timeout):
            exc = RuntimeError("Internal error")
            exc.startup_elapsed_s = 28.0
            exc.startup_timeout_s = 30.0
            exc.startup_failed = True
            raise exc

        outcome = worker.run(artifacts, failing_runner)
        assert outcome.passed is False
        assert outcome.skipped is False
        assert outcome.blocking is True
        assert "启动超时" in outcome.summary

    def test_non_blocking_role_not_skipped_on_execution_failure(self):
        """Only startup failures trigger degradation, not execution failures."""
        role = _make_role("doc_writer", blocking=False, category="writing")
        worker = RoleReviewWorker(role, timeout=240.0)
        artifacts = _make_artifacts()

        def failing_runner(prompt, on_event, timeout):
            raise RuntimeError("model returned bad output")

        outcome = worker.run(artifacts, failing_runner)
        assert outcome.passed is False
        assert outcome.skipped is False


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------


class TestReviewPipelineReliability:
    """End-to-end tests with mock prompt runners simulating startup failures."""

    def test_pipeline_with_startup_failure_on_blocking_role(self):
        """A blocking role with fast startup failure (non-timeout) fails the review."""
        artifacts = _make_artifacts()
        roles = [
            _make_role("architect", blocking=True),
            _make_role("security", blocking=True),
        ]

        call_count = {"architect": 0, "security": 0}

        def factory(role):
            def runner(prompt, on_event, timeout):
                call_count[role.role_id] += 1
                if role.role_id == "architect":
                    # Fast failure, not a timeout → blocking=True
                    exc = RuntimeError("connection refused")
                    exc.startup_elapsed_s = 2.0
                    exc.startup_timeout_s = 30.0
                    exc.startup_failed = True
                    raise exc
                return '{"role_id": "%s", "verdict": "PASS", "summary": "ok", "suggestions": []}' % role.role_id

            return runner

        result = run_adaptive_role_review_pipeline(
            artifacts,
            roles,
            prompt_runner_factory=factory,
            max_parallel=1,
            timeout=240.0,
        )
        assert result.blocking_review_passed is False
        arch_outcome = next(o for o in result.role_outcomes if o.role_id == "architect")
        assert arch_outcome.passed is False
        assert arch_outcome.blocking is True
        assert "启动失败" in arch_outcome.summary
        assert result.skipped_roles_count == 0

    def test_pipeline_with_startup_failure_on_non_blocking_role(self):
        """A non-blocking role failing at startup should be skipped, review still passes."""
        artifacts = _make_artifacts()
        roles = [
            _make_role("architect", blocking=True),
            _make_role("doc_writer", blocking=False, category="writing"),
        ]

        def factory(role):
            def runner(prompt, on_event, timeout):
                if role.role_id == "doc_writer":
                    exc = RuntimeError("Internal error: backend down")
                    exc.startup_elapsed_s = 28.5
                    exc.startup_timeout_s = 30.0
                    exc.startup_failed = True
                    raise exc
                return '{"role_id": "%s", "verdict": "PASS", "summary": "ok", "suggestions": []}' % role.role_id

            return runner

        result = run_adaptive_role_review_pipeline(
            artifacts,
            roles,
            prompt_runner_factory=factory,
            max_parallel=1,
            timeout=240.0,
        )
        # The blocking architect role passes, so overall review passes
        assert result.blocking_review_passed is True
        assert result.skipped_roles_count == 1
        doc_outcome = next(o for o in result.role_outcomes if o.role_id == "doc_writer")
        assert doc_outcome.skipped is True
        assert doc_outcome.passed is True

    def test_outcomes_to_result_counts_skipped(self):
        outcomes = [
            RoleReviewOutcome(
                role_id="r1",
                role_display_name="R1",
                role_category="software",
                passed=True,
                skipped=False,
                summary="ok",
            ),
            RoleReviewOutcome(
                role_id="r2",
                role_display_name="R2",
                role_category="writing",
                passed=True,
                skipped=True,
                summary="skipped",
            ),
            RoleReviewOutcome(
                role_id="r3",
                role_display_name="R3",
                role_category="research",
                passed=True,
                skipped=True,
                summary="skipped",
            ),
        ]
        result = _outcomes_to_review_result(outcomes, iteration=1)
        assert result.skipped_roles_count == 2

    def test_pipeline_all_roles_pass(self):
        """Baseline: all roles pass, no skipped."""
        artifacts = _make_artifacts()
        roles = [
            _make_role("architect", blocking=True),
            _make_role("security", blocking=True),
        ]

        def factory(role):
            def runner(prompt, on_event, timeout):
                return '{"role_id": "%s", "verdict": "PASS", "summary": "all good", "suggestions": []}' % role.role_id

            return runner

        result = run_adaptive_role_review_pipeline(
            artifacts,
            roles,
            prompt_runner_factory=factory,
            max_parallel=1,
            timeout=240.0,
        )
        assert result.blocking_review_passed is True
        assert result.skipped_roles_count == 0
        assert all(o.passed for o in result.role_outcomes)


# ---------------------------------------------------------------------------
# Startup retry with exponential backoff
# ---------------------------------------------------------------------------


class TestStartupRetryBackoff:
    """Verify _run_with_startup_retry actually sleeps and retries on startup failure."""

    def test_first_startup_fails_second_succeeds_calls_sleep(self):
        """First EphemeralReviewSession.__enter__ raises; retry succeeds after sleep."""
        from src.spec_engine.review_strategy import _run_with_startup_retry

        role = _make_role("test_role")
        call_count = {"enter": 0, "send_prompt": 0}

        class FakeSession:
            def send_prompt(self, prompt, on_event=None, timeout=None):
                call_count["send_prompt"] += 1
                return "ok result"

            def close(self):
                pass

        original_eph = None

        def fake_eph_init(self, *args, **kwargs):
            nonlocal original_eph
            self._agent_type = kwargs.get("agent_type", args[0] if args else "coco")
            self._cwd = kwargs.get("cwd", args[1] if len(args) > 1 else ".")
            self._model_name = kwargs.get("model_name", args[2] if len(args) > 2 else None)
            self._startup_timeout = kwargs.get("startup_timeout", args[3] if len(args) > 3 else None)
            self._session = None
            self.startup_elapsed_s = 0.0
            self.session_started = False

        def fake_eph_enter(self):
            call_count["enter"] += 1
            import time as _time

            t0 = _time.perf_counter()
            try:
                if call_count["enter"] <= 1:
                    raise RuntimeError("connection refused on first attempt")
                self._session = FakeSession()
                self.session_started = True
                return self._session
            finally:
                self.startup_elapsed_s = _time.perf_counter() - t0 + 1.0

        def fake_eph_exit(self, *exc):
            if self._session is None:
                return
            try:
                close = getattr(self._session, "close", None)
                if callable(close):
                    close()
            finally:
                self._session = None

        with (
            patch("src.spec_engine.review_strategy.EphemeralReviewSession.__init__", fake_eph_init),
            patch("src.spec_engine.review_strategy.EphemeralReviewSession.__enter__", fake_eph_enter),
            patch("src.spec_engine.review_strategy.EphemeralReviewSession.__exit__", fake_eph_exit),
            patch("src.spec_engine.review_strategy.time.sleep") as mock_sleep,
        ):
            result = _run_with_startup_retry(
                role,
                "coco",
                None,
                "test prompt",
                None,
                240.0,
                30.0,
                "/tmp/test",
            )

        assert result == "ok result"
        assert call_count["enter"] == 2
        assert call_count["send_prompt"] == 1
        assert mock_sleep.call_count >= 1

    def test_execution_failure_not_retried(self):
        """Session starts successfully but send_prompt fails — no retry, no startup annotation."""
        from src.spec_engine.review_strategy import _run_with_startup_retry

        role = _make_role("test_role")
        call_count = {"enter": 0, "send_prompt": 0}

        class FakeSession:
            def send_prompt(self, prompt, on_event=None, timeout=None):
                call_count["send_prompt"] += 1
                raise RuntimeError("model error during execution")

            def close(self):
                pass

        def fake_eph_init(self, *args, **kwargs):
            self._agent_type = kwargs.get("agent_type", args[0] if args else "coco")
            self._cwd = kwargs.get("cwd", args[1] if len(args) > 1 else ".")
            self._model_name = kwargs.get("model_name", args[2] if len(args) > 2 else None)
            self._startup_timeout = kwargs.get("startup_timeout", args[3] if len(args) > 3 else None)
            self._session = None
            self.startup_elapsed_s = 0.0
            self.session_started = False

        def fake_eph_enter(self):
            call_count["enter"] += 1
            import time as _time

            t0 = _time.perf_counter()
            try:
                self._session = FakeSession()
                self.session_started = True
                return self._session
            finally:
                self.startup_elapsed_s = _time.perf_counter() - t0

        def fake_eph_exit(self, *exc):
            if self._session is None:
                return
            try:
                close = getattr(self._session, "close", None)
                if callable(close):
                    close()
            finally:
                self._session = None

        with (
            patch("src.spec_engine.review_strategy.EphemeralReviewSession.__init__", fake_eph_init),
            patch("src.spec_engine.review_strategy.EphemeralReviewSession.__enter__", fake_eph_enter),
            patch("src.spec_engine.review_strategy.EphemeralReviewSession.__exit__", fake_eph_exit),
            patch("src.spec_engine.review_strategy.time.sleep") as mock_sleep,
        ):
            with pytest.raises(RuntimeError, match="model error during execution") as exc_info:
                _run_with_startup_retry(
                    role,
                    "coco",
                    None,
                    "test prompt",
                    None,
                    240.0,
                    30.0,
                    "/tmp/test",
                )

        assert call_count["enter"] == 1
        assert call_count["send_prompt"] == 1
        assert mock_sleep.call_count == 0
        assert not getattr(exc_info.value, "startup_failed", False)


# ---------------------------------------------------------------------------
# EphemeralReviewSession session_started field
# ---------------------------------------------------------------------------


class TestEphemeralReviewSessionStarted:
    """Verify EphemeralReviewSession.session_started semantics."""

    def test_session_started_false_after_init(self):
        from src.agent_session.factory import EphemeralReviewSession

        eph = EphemeralReviewSession("coco", "/tmp", None, startup_timeout=30.0)
        assert eph.session_started is False

    def test_session_started_true_after_successful_enter(self):
        from src.agent_session.factory import EphemeralReviewSession

        fake_session = MagicMock()
        fake_session.close = MagicMock()

        with patch("src.agent_session.factory.create_review_session", return_value=fake_session):
            eph = EphemeralReviewSession("coco", "/tmp", None, startup_timeout=30.0)
            with eph as session:
                assert eph.session_started is True
                assert session is fake_session
            # __exit__ sets _session=None but session_started stays True
            assert eph.session_started is True

    def test_session_started_false_after_failed_enter(self):
        from src.agent_session.factory import EphemeralReviewSession

        with patch("src.agent_session.factory.create_review_session", side_effect=RuntimeError("boom")):
            eph = EphemeralReviewSession("coco", "/tmp", None, startup_timeout=30.0)
            with pytest.raises(RuntimeError, match="boom"):
                with eph:
                    pass
            assert eph.session_started is False


# ---------------------------------------------------------------------------
# Blocking role infrastructure failure fails overall review
# ---------------------------------------------------------------------------


class TestBlockingRoleInfrastructureFailure:
    """Blocking role startup failure must keep blocking_review_passed=False."""

    def test_blocking_role_startup_timeout_fails_overall_review(self):
        artifacts = _make_artifacts()
        roles = [
            _make_role("architect", blocking=True),
            _make_role("security", blocking=True),
        ]

        def factory(role):
            def runner(prompt, on_event, timeout):
                if role.role_id == "architect":
                    exc = RuntimeError("Internal error")
                    exc.startup_elapsed_s = 28.5
                    exc.startup_timeout_s = 30.0
                    exc.startup_failed = True
                    raise exc
                return '{"role_id": "%s", "verdict": "PASS", "summary": "ok", "suggestions": []}' % role.role_id

            return runner

        result = run_adaptive_role_review_pipeline(
            artifacts,
            roles,
            prompt_runner_factory=factory,
            max_parallel=1,
            timeout=240.0,
        )
        assert result.blocking_review_passed is False
        arch_outcome = next(o for o in result.role_outcomes if o.role_id == "architect")
        assert arch_outcome.passed is False
        assert arch_outcome.blocking is True
        assert arch_outcome.skipped is False
        assert result.skipped_roles_count == 0
