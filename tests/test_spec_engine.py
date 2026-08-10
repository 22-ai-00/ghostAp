"""Tests for spec_engine — ACP-driven SpecEngine with structured methodology."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from src.acp.models import ACPEvent, ACPEventType
from src.engine_base import EngineRunState, PerspectiveReview, ReviewPerspective, ReviewResult
from src.spec_engine.engine import SpecEngine, SpecEngineCallbacks
from src.spec_engine.manager import SpecEngineManager
from src.spec_engine.models import (
    ReviewCircuitState,
    ReviewContext,
    ReviewRoleSpec,
    SpecCycle,
    SpecProject,
    SpecProjectStatus,
    SpecTask,
    SpecTaskStatus,
)
from src.spec_engine.review import ReviewOrchestrator, conduct_review
from src.spec_engine.review_artifacts import ReviewArtifacts
from src.utils.review_diagnostics import (
    build_review_exception_diagnostics,
    normalize_review_diagnostics,
)

# ---------------------------------------------------------------------------
# Shared spec settings factory — avoids 50-line duplication across test classes
# ---------------------------------------------------------------------------

def _make_spec_settings(**overrides):
    """Build a mock settings object with spec_engine defaults.

    Callers can override any field via keyword arguments.
    """
    s = MagicMock()
    defaults = dict(
        spec_max_cycles=1,
        spec_max_cycles_limit=5000,
        spec_convergence_window=1,
        spec_execution_timeout=300,
        spec_review_enabled=True,
        spec_infinite_mode=False,
        spec_disable_convergence=False,
        spec_disable_early_stop=False,
        spec_min_cycles=1,
        spec_max_retries=1,
        spec_cycle_tasks_max=50,
        spec_cycle_output_max_chars=4000,
        spec_state_filename=".spec_engine_state.json",
        spec_artifacts_dirname=".spec_engine",
        spec_persist_phase_artifacts=True,
        spec_persist_every_phase=True,
        spec_discovery_enabled=False,
        spec_discovery_max_questions=3,
        spec_discovery_force_nonempty=True,
        spec_generated_specs_per_cycle=1,
        spec_discovery_gate_on_satisfied=True,
        spec_discovery_max_pending=5,
        spec_discovery_cooldown_cycles=3,
        spec_backlog_stuck_window=3,
        spec_success_ignore_backlog=True,
        spec_allow_resume_from_disk=True,
        spec_history_log_filename="history.jsonl",
        spec_phase_output_persist_max_chars=20000,
        spec_cycle_artifact_retention=50,
        spec_generated_specs_retention=1000,
        spec_review_failure_circuit_enabled=False,
        spec_review_failure_max_consecutive=3,
        spec_review_failure_cooldown_cycles=3,
        spec_review_timeout=120,
        spec_review_min_timeout=30,
        spec_review_hard_floor=15,
        spec_review_max_parallel=4,
        spec_review_retry_max_attempts=1,
        spec_review_retry_max_delay=30,
        spec_state_cycles_tail=50,
        spec_state_work_items_tail=200,
        spec_state_metrics_tail=200,
        spec_rebuild_session_between_cycles=False,
    )
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(s, k, v)
    return s






def _run_review_with_error(error: Exception, *, timeout: float = 42):
    observed_timeouts = []
    role = ReviewRoleSpec(
        role_id="reviewer",
        display_name="Reviewer",
        category="software",
        mission="review",
        review_focus=["correctness"],
        must_check=["errors"],
    )

    def runner_factory(_role):
        def run(_prompt, _on_event, role_timeout):
            observed_timeouts.append(role_timeout)
            raise error

        return run

    context = ReviewContext(
        cycle=1,
        session=None,
        settings=_make_spec_settings(
            spec_completion_gate_enabled=False,
            spec_review_failure_circuit_enabled=False,
            spec_review_retry_max_attempts=0,
            spec_review_max_parallel=1,
            spec_review_role_timeout_multipliers={},
            spec_review_timeout=timeout,
        ),
        project=None,
        send_prompt_with_retry_fn=lambda *args, **kwargs: None,
        build_review_exception_diagnostics_fn=build_review_exception_diagnostics,
        circuit=ReviewCircuitState(),
        artifacts=ReviewArtifacts(cycle_number=1, requirement="test", cwd="/tmp"),
        prompt_runner_factory=runner_factory,
        role_plan_override=[role],
    )
    return conduct_review(context), observed_timeouts


def test_review_worker_error_uses_diagnostics_snippet():
    class EmptyError(RuntimeError):
        def __str__(self):
            return ""

    error = EmptyError()
    error.stderr_snippet = "E: invalid params"

    result, _ = _run_review_with_error(error)
    suggestions = [text for review in result.reviews for text in review.suggestions]

    assert any("invalid params" in text.lower() for text in suggestions)


def test_review_worker_error_redacts_diagnostics_snippet():
    class EmptyError(RuntimeError):
        def __str__(self):
            return ""

    error = EmptyError()
    error.stderr_snippet = "token=SECRET_TOKEN"

    result, _ = _run_review_with_error(error)
    suggestions = [text for review in result.reviews for text in review.suggestions]

    assert suggestions
    assert all("SECRET_TOKEN" not in text for text in suggestions)
    assert any("REDACTED" in text for text in suggestions)


def test_try_switch_model_claude_returns_false_without_switch(monkeypatch, tmp_path):
    from src.spec_engine.engine import SpecEngine

    engine = SpecEngine(chat_id="c1", root_path=str(tmp_path), agent_type="claude")
    engine._models_tried = []
    engine._current_model = None

    # claude CLI 模式不应进入 ACP 模型切换分支
    assert engine._try_switch_model(callbacks=MagicMock()) is False


# ======================================================================
# TestSpecModels — enums, creation, serialization, lifecycle
# ======================================================================


class TestSpecModels:


    def test_task_status_enum(self):
        assert SpecTaskStatus.PENDING.value == "pending"
        assert SpecTaskStatus.COMPLETED.value == "completed"

    def test_spec_task_to_dict_from_dict(self):
        task = SpecTask(
            task_id=2, description="Add tests", dependencies=[1], status=SpecTaskStatus.COMPLETED, output="ok"
        )
        d = task.to_dict()
        assert d["task_id"] == 2
        assert d["dependencies"] == [1]
        assert d["status"] == "completed"

        restored = SpecTask.from_dict(d)
        assert restored.task_id == 2
        assert restored.dependencies == [1]
        assert restored.status == SpecTaskStatus.COMPLETED
        assert restored.output == "ok"







# ======================================================================
# TestPhaseTracker — event processing
# ======================================================================




# ======================================================================
# TestSpecReporter — content formatters and title helpers
# ======================================================================




# ======================================================================
# TestSpecEngine — core engine behavior
# ======================================================================


class TestSpecEngine:
    @patch("src.engine_base.get_settings")
    def _make_engine(self, mock_settings, **kwargs):
        s = MagicMock()
        s.spec_max_cycles = 10
        s.spec_max_cycles_limit = 5000
        s.spec_convergence_window = 2
        s.spec_execution_timeout = 300
        s.spec_cycle_tasks_max = 50
        s.spec_cycle_output_max_chars = 4000
        s.spec_state_filename = ".spec_engine_state.json"
        s.spec_artifacts_dirname = ".spec_engine"
        s.spec_persist_phase_artifacts = True
        s.spec_persist_every_phase = True
        # Keep legacy unit tests stable: discovery is tested separately.
        s.spec_discovery_enabled = False
        s.spec_discovery_max_questions = 3
        s.spec_discovery_force_nonempty = True
        s.spec_generated_specs_per_cycle = 1
        s.spec_discovery_gate_on_satisfied = True
        s.spec_discovery_max_pending = 5
        s.spec_discovery_cooldown_cycles = 3
        s.spec_backlog_stuck_window = 3
        s.spec_success_ignore_backlog = True
        s.spec_allow_resume_from_disk = True
        s.spec_infinite_mode = False
        s.spec_disable_convergence = False
        s.spec_disable_early_stop = False
        s.spec_min_cycles = 1
        mock_settings.return_value = s
        return SpecEngine(chat_id="c1", root_path="/tmp/test", **kwargs)




    def test_max_cycles_with_core_acceptance_completes_with_backlog(self):
        engine = self._make_engine()
        project = SpecProject.create(root_path="/tmp/test")
        project.acceptance_criteria = ["核心验收"]
        project.criteria_tracker.init_criteria(project.acceptance_criteria)
        project.criteria_tracker.batch_update({0: True}, 1)
        project.review_pass_streak = 1
        engine._project = project
        engine._last_review = MagicMock(all_passed=True)
        engine.settings.spec_review_enabled = True
        engine.settings.spec_review_pass_streak_required = 1

        engine._handle_max_cycles_termination(3)

        assert project.status is SpecProjectStatus.COMPLETED
        assert project.completed_at is not None





    @pytest.mark.parametrize("state", [EngineRunState.RUNNING, EngineRunState.STOPPING])
    def test_cleanup_while_active_only_requests_stop(self, state):
        engine = self._make_engine()
        engine._run_state = state
        engine._session = MagicMock()
        project = MagicMock()
        engine._project = project

        engine.cleanup()

        assert engine.run_state == EngineRunState.STOPPING
        engine._session.cancel.assert_called_once()
        # 活跃态 cleanup 不应立即清空 project，避免并发线程访问 self._project 失败
        assert engine._project is project


    def test_try_switch_model_returns_false_when_not_running(self):
        engine = self._make_engine()
        engine._run_state = EngineRunState.STOPPING

        callbacks = SpecEngineCallbacks()
        assert engine._try_switch_model(callbacks) is False


















    def test_detect_convergence_not_enough_cycles(self):
        engine = self._make_engine()
        engine._project = SpecProject.create(root_path="/tmp")
        assert not engine._detect_convergence()

    def test_detect_convergence_triggered(self):
        engine = self._make_engine()
        engine._project = SpecProject.create(root_path="/tmp")
        engine._project.criteria_tracker.init_criteria(["C1", "C2"])

        # 2 cycles, criteria satisfied count stays the same (0), review suggestions stay the same (1)
        def _make_review(iteration):
            return ReviewResult(
                reviews=[
                    PerspectiveReview(
                        perspective=ReviewPerspective.ARCHITECT, passed=False, suggestions=["S1"], summary="1条建议"
                    ),
                    PerspectiveReview(
                        perspective=ReviewPerspective.PRODUCT, passed=True, suggestions=[], summary="通过"
                    ),
                    PerspectiveReview(perspective=ReviewPerspective.USER, passed=True, suggestions=[], summary="通过"),
                    PerspectiveReview(
                        perspective=ReviewPerspective.TESTER, passed=True, suggestions=[], summary="通过"
                    ),
                ],
                iteration=iteration,
            )

        engine._project.cycles = [
            SpecCycle(cycle_number=1, build_output="x" * 100, review_result=_make_review(1)),
            SpecCycle(cycle_number=2, build_output="y" * 100, review_result=_make_review(2)),
        ]
        assert engine._detect_convergence()

    def test_detect_convergence_not_triggered(self):
        engine = self._make_engine()
        engine._project = SpecProject.create(root_path="/tmp")
        engine._project.criteria_tracker.init_criteria(["C1", "C2"])

        # Simulate criteria progress in the window
        engine._project.criteria_tracker.update(0, True, 1)
        engine._project.criteria_tracker.update(1, True, 2)

        review_pass = ReviewResult(
            reviews=[
                PerspectiveReview(perspective=p, passed=True, suggestions=[], summary="通过") for p in ReviewPerspective
            ],
            iteration=1,
        )

        engine._project.cycles = [
            SpecCycle(cycle_number=1, build_output="x" * 100, review_result=review_pass),
            SpecCycle(cycle_number=2, build_output="y" * 100, review_result=review_pass),
        ]
        assert not engine._detect_convergence()







        # No exception expected





# ======================================================================
# ======================================================================
# Review cancellation
# ======================================================================


class TestReviewCancellation:
    def test_orchestrator_reset_tracks_engine_state(self):
        orchestrator = ReviewOrchestrator()
        orchestrator.signal_stop()

        assert orchestrator.reset_cancel_event(is_running=True) is True
        assert not orchestrator.cancel_event.is_set()
        assert orchestrator.reset_cancel_event(is_running=False) is False
        assert orchestrator.cancel_event.is_set()

    def test_engine_stop_signals_review_workers(self):
        engine = SpecEngine(chat_id="c1", root_path="/tmp/test")
        engine._run_state = EngineRunState.RUNNING
        engine._session = MagicMock()

        engine.stop()

        assert engine._review_orchestrator.cancel_event.is_set()


# TestSpecEngineManager — get_or_create, active, cleanup
# ======================================================================


class TestSpecEngineManager:
    @patch("src.engine_base.get_settings")
    def test_get_or_create(self, mock_settings):
        s = MagicMock()
        s.spec_max_cycles = 10
        s.spec_convergence_window = 2
        s.spec_execution_timeout = 300
        mock_settings.return_value = s

        mgr = SpecEngineManager()
        e1 = mgr.get_or_create("chat1", "/tmp/a")
        e2 = mgr.get_or_create("chat1", "/tmp/a")
        assert e1 is e2  # Same instance

    @patch("src.engine_base.get_settings")
    def test_get_different_paths(self, mock_settings):
        s = MagicMock()
        s.spec_max_cycles = 10
        s.spec_convergence_window = 2
        s.spec_execution_timeout = 300
        mock_settings.return_value = s

        mgr = SpecEngineManager()
        e1 = mgr.get_or_create("chat1", "/tmp/a")
        e2 = mgr.get_or_create("chat1", "/tmp/b")
        assert e1 is not e2

    @patch("src.engine_base.get_settings")
    def test_get_active_engine(self, mock_settings):
        s = MagicMock()
        s.spec_max_cycles = 10
        s.spec_convergence_window = 2
        s.spec_execution_timeout = 300
        mock_settings.return_value = s

        mgr = SpecEngineManager()
        e = mgr.get_or_create("chat1", "/tmp/a")
        assert mgr.get_active_engine("chat1") is None

        e._run_state = EngineRunState.RUNNING
        assert mgr.get_active_engine("chat1") is e

    @patch("src.engine_base.get_settings")
    def test_engine_name_switch(self, mock_settings):
        s = MagicMock()
        s.spec_max_cycles = 10
        s.spec_convergence_window = 2
        s.spec_execution_timeout = 300
        mock_settings.return_value = s

        mgr = SpecEngineManager()
        e1 = mgr.get_or_create("chat1", "/tmp/a", engine_name="Coco")
        assert e1.engine_name == "Coco"
        e2 = mgr.get_or_create("chat1", "/tmp/a", engine_name="Claude")
        assert e2.engine_name == "Claude"
        assert e1 is not e2  # New instance because name changed

    @patch("src.engine_base.get_settings")
    def test_engine_name_switch_blocked_while_running(self, mock_settings):
        s = MagicMock()
        s.spec_max_cycles = 10
        s.spec_convergence_window = 2
        s.spec_execution_timeout = 300
        mock_settings.return_value = s

        mgr = SpecEngineManager()
        e1 = mgr.get_or_create("chat1", "/tmp/a", engine_name="Coco")
        e1._run_state = EngineRunState.RUNNING
        e2 = mgr.get_or_create("chat1", "/tmp/a", engine_name="Claude")
        assert e2 is e1  # Not replaced because still running

    @patch("src.engine_base.get_settings")
    def test_cleanup_all(self, mock_settings):
        s = MagicMock()
        s.spec_max_cycles = 10
        s.spec_convergence_window = 2
        s.spec_execution_timeout = 300
        mock_settings.return_value = s

        mgr = SpecEngineManager()
        mgr.get_or_create("chat1", "/tmp/a")
        mgr.get_or_create("chat2", "/tmp/b")
        mgr.cleanup_all()
        assert mgr.list_engines() == []

    @patch("src.engine_base.get_settings")
    def test_cleanup_all_keeps_running_engine(self, mock_settings):
        s = MagicMock()
        s.spec_max_cycles = 10
        s.spec_convergence_window = 2
        s.spec_execution_timeout = 300
        mock_settings.return_value = s

        mgr = SpecEngineManager()
        engine = mgr.get_or_create("chat1", "/tmp/a")
        engine._run_state = EngineRunState.RUNNING
        mgr.cleanup_all()
        assert mgr.get("chat1", "/tmp/a") is engine
        assert engine.run_state == EngineRunState.STOPPING




    @patch("src.engine_base.get_settings")
    def test_get_none_for_missing(self, mock_settings):
        s = MagicMock()
        mock_settings.return_value = s
        mgr = SpecEngineManager()
        assert mgr.get("chat1", "/tmp/a") is None
        assert mgr.get_active_engine("chat1") is None

    @patch("src.engine_base.get_settings")
    def test_list_engines(self, mock_settings):
        s = MagicMock()
        s.spec_max_cycles = 10
        s.spec_convergence_window = 2
        s.spec_execution_timeout = 300
        mock_settings.return_value = s

        mgr = SpecEngineManager()
        mgr.get_or_create("c1", "/tmp/a")
        mgr.get_or_create("c1", "/tmp/b")
        mgr.get_or_create("c2", "/tmp/c")
        assert len(mgr.list_engines()) == 3
        assert len(mgr.list_engines("c1")) == 2
        assert len(mgr.list_engines("c2")) == 1

    @patch("src.engine_base.get_settings")
    def test_get_active_engines(self, mock_settings):
        s = MagicMock()
        s.spec_max_cycles = 10
        s.spec_convergence_window = 2
        s.spec_execution_timeout = 300
        mock_settings.return_value = s

        mgr = SpecEngineManager()
        e1 = mgr.get_or_create("c1", "/tmp/a")
        mgr.get_or_create("c1", "/tmp/b")
        e1._run_state = EngineRunState.RUNNING
        active = mgr.get_active_engines("c1")
        assert len(active) == 1
        assert active[0] is e1


# ======================================================================
# TestSpecHandler — command routing
# ======================================================================




# ======================================================================
# TestSystemHandler — is_spec_command predicate
# ======================================================================


class TestSystemHandlerSpec:
    def test_is_spec_command(self):
        from src.feishu.handlers.system import SystemHandler

        assert SystemHandler.is_spec_command("/spec build auth")
        assert SystemHandler.is_spec_command("/spec_status")
        assert SystemHandler.is_spec_command("/stop_spec")
        assert SystemHandler.is_spec_command("/spec_guide focus")
        assert SystemHandler.is_spec_command("/spec_export")
        assert not SystemHandler.is_spec_command("/deep build")
        assert not SystemHandler.is_spec_command("/deep do stuff")
        assert not SystemHandler.is_spec_command("hello")


# ======================================================================
# TestIntentRecognizer — spec intents
# ======================================================================


class TestIntentRecognizerSpec:
    def test_spec_intent_types_exist(self):
        from src.agent.intent_recognizer import IntentType

        assert hasattr(IntentType, "ENTER_SPEC")
        assert hasattr(IntentType, "SPEC_STATUS")
        assert hasattr(IntentType, "STOP_SPEC")
        assert hasattr(IntentType, "SPEC_GUIDE")

    def test_spec_exact_commands(self):
        from src.agent.intent_recognizer import IntentRecognizer

        recognizer = IntentRecognizer()
        # Quick match: /spec_status
        result = recognizer.recognize("/spec_status", "smart")
        from src.agent.intent_recognizer import IntentType

        assert result.primary_intent == IntentType.SPEC_STATUS

    def test_spec_guide_quick_match(self):
        from src.agent.intent_recognizer import IntentRecognizer, IntentType

        recognizer = IntentRecognizer()
        result = recognizer.recognize("/spec_guide focus on tests", "smart")
        assert result.primary_intent == IntentType.SPEC_GUIDE


# ======================================================================
# TestConfig — spec engine settings
# ======================================================================


class TestConfigSpec:
    def test_spec_settings_defaults(self):
        from src.config import Settings

        s = Settings(app_id="", app_secret="", _env_file=None)
        assert s.spec_max_cycles == 1000
        assert s.spec_max_cycles_limit >= 5000
        assert s.spec_execution_timeout == 7200
        assert s.spec_convergence_window == 2
        assert s.spec_review_enabled is True
        assert s.spec_discovery_enabled is True

    def test_spec_review_circuit_breaker_defaults(self):
        """FS-17: review circuit breaker settings have sensible defaults."""
        from src.config import Settings

        s = Settings(app_id="", app_secret="", _env_file=None)
        assert s.spec_review_failure_circuit_enabled is True
        assert s.spec_review_failure_max_consecutive == 4
        assert s.spec_review_failure_cooldown_cycles == 2
        assert s.spec_review_failure_max_cooldown_cycles == 12
        assert s.spec_review_timeout == 240
        assert s.spec_review_min_timeout == 60
        assert s.spec_review_hard_floor == 20

    def test_lock_settings_defaults(self):
        """FS-17: lock-related settings have sensible defaults."""
        from src.config import Settings

        s = Settings(app_id="", app_secret="", _env_file=None)
        assert s.repo_lock_idle_timeout == 300
        assert s.repo_lock_cleanup_interval == 60
        assert s.repo_lock_hard_timeout == 3600
        assert s.chat_lock_max_duration == 86400
        assert s.chat_lock_cleanup_interval == 60


# ======================================================================
# TestSpecEngineExecution — integration tests for execute/resume/review
# ======================================================================


class TestSpecEngineExecution:
    """Integration tests for execute, resume, review, criteria evaluation."""

    def _mock_settings(self):
        return _make_spec_settings()

    def _make_mock_session(self, text_responses):
        """Mock session that returns text_responses sequentially via on_event."""
        session = MagicMock()
        call_index = [0]
        responses = list(text_responses)

        def fake_send_prompt(prompt, on_event=None, timeout=None, **kwargs):
            idx = call_index[0]
            call_index[0] += 1
            text = responses[idx] if idx < len(responses) else ""
            if on_event and text:
                on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=text))
            return MagicMock(stop_reason="end_turn")

        session.send_prompt = fake_send_prompt
        session.send_prompt_with_retry = fake_send_prompt
        return session




    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_execute_stop_mid_cycle(self, mock_settings, mock_create):
        """Stop during SPEC phase -> cycle saved as failed, project cancelled."""
        mock_settings.return_value = self._mock_settings()

        def fake_send_prompt(prompt, on_event=None, timeout=None, **kwargs):
            if on_event:
                on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="partial"))

        session = MagicMock()
        session.send_prompt = fake_send_prompt
        session.send_prompt_with_retry = fake_send_prompt
        mock_create.return_value = session

        engine = SpecEngine(chat_id="c1", root_path="/tmp/test")

        # Stop after first phase completes
        original = engine._run_phase

        def stop_after_first(cycle_num, phase, prompt, callbacks, timeout):
            result = original(cycle_num, phase, prompt, callbacks, timeout)
            engine._run_state = EngineRunState.STOPPING
            return result

        engine._run_phase = stop_after_first

        project = engine.execute("- test requirement")

        assert project.status == SpecProjectStatus.CANCELLED
        assert len(project.cycles) == 1
        assert project.cycles[0].status == "failed"
        assert engine.run_state == EngineRunState.IDLE

    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_execute_exception_handling(self, mock_settings, mock_create):
        """Exception during session creation → ABORTED + on_error called."""
        mock_settings.return_value = self._mock_settings()
        mock_create.side_effect = RuntimeError("connection failed")

        engine = SpecEngine(chat_id="c1", root_path="/tmp/test")
        error_msgs = []
        callbacks = SpecEngineCallbacks(on_error=lambda e: error_msgs.append(e))

        project = engine.execute("- test req", callbacks)

        assert project.status == SpecProjectStatus.FAILED
        assert len(error_msgs) == 1
        assert "connection failed" in error_msgs[0]
        assert engine.run_state == EngineRunState.IDLE











    def test_convergence_with_stagnant_review_suggestions(self):
        """Convergence detects stagnant review suggestions across window."""
        with patch("src.engine_base.get_settings") as mock_settings:
            s = MagicMock()
            s.spec_max_cycles = 10
            s.spec_convergence_window = 2
            s.spec_execution_timeout = 300
            mock_settings.return_value = s

            engine = SpecEngine(chat_id="c1", root_path="/tmp/test")
            engine._project = SpecProject.create(root_path="/tmp/test")
            engine._project.criteria_tracker.init_criteria(["C1", "C2"])

            # 2 cycles with same non-zero suggestion count → converge
            def _make_review(n_suggestions, iteration):
                return ReviewResult(
                    reviews=[
                        PerspectiveReview(
                            perspective=ReviewPerspective.ARCHITECT,
                            passed=False,
                            suggestions=[f"S{i}" for i in range(n_suggestions)],
                            summary=f"{n_suggestions}条建议",
                        ),
                        PerspectiveReview(
                            perspective=ReviewPerspective.PRODUCT, passed=True, suggestions=[], summary="通过"
                        ),
                        PerspectiveReview(
                            perspective=ReviewPerspective.USER, passed=True, suggestions=[], summary="通过"
                        ),
                        PerspectiveReview(
                            perspective=ReviewPerspective.TESTER, passed=True, suggestions=[], summary="通过"
                        ),
                    ],
                    iteration=iteration,
                )

            engine._project.cycles = [
                SpecCycle(cycle_number=1, build_output="x" * 100, review_result=_make_review(1, 1)),
                SpecCycle(cycle_number=2, build_output="y" * 100, review_result=_make_review(1, 2)),
            ]
            assert engine._detect_convergence()

    def test_convergence_not_triggered_when_improving(self):
        """Convergence NOT triggered when suggestions are decreasing."""
        with patch("src.engine_base.get_settings") as mock_settings:
            s = MagicMock()
            s.spec_max_cycles = 10
            s.spec_convergence_window = 2
            s.spec_execution_timeout = 300
            mock_settings.return_value = s

            engine = SpecEngine(chat_id="c1", root_path="/tmp/test")
            engine._project = SpecProject.create(root_path="/tmp/test")
            engine._project.criteria_tracker.init_criteria(["C1", "C2"])

            def _make_review(n_suggestions, iteration):
                return ReviewResult(
                    reviews=[
                        PerspectiveReview(
                            perspective=ReviewPerspective.ARCHITECT,
                            passed=False,
                            suggestions=[f"S{i}" for i in range(n_suggestions)],
                            summary=f"{n_suggestions}条建议",
                        ),
                        PerspectiveReview(
                            perspective=ReviewPerspective.PRODUCT, passed=True, suggestions=[], summary="通过"
                        ),
                        PerspectiveReview(
                            perspective=ReviewPerspective.USER, passed=True, suggestions=[], summary="通过"
                        ),
                        PerspectiveReview(
                            perspective=ReviewPerspective.TESTER, passed=True, suggestions=[], summary="通过"
                        ),
                    ],
                    iteration=iteration,
                )

            engine._project.cycles = [
                SpecCycle(cycle_number=1, build_output="x" * 100, review_result=_make_review(3, 1)),
                SpecCycle(cycle_number=2, build_output="y" * 100, review_result=_make_review(1, 2)),
            ]
            assert not engine._detect_convergence()


    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_discovery_generates_spec_files_and_backlog(self, mock_settings, mock_create, tmp_path):
        """每轮循环后触发问题发现→生成 spec 文件→加入 backlog，并能被下一轮加载执行。"""
        s = self._mock_settings()
        s.spec_max_cycles = 2
        s.spec_review_enabled = False
        s.spec_discovery_enabled = True
        s.spec_discovery_max_questions = 1
        s.spec_generated_specs_per_cycle = 1
        s.spec_discovery_gate_on_satisfied = True
        s.spec_discovery_max_pending = 5
        s.spec_discovery_cooldown_cycles = 3
        s.spec_backlog_stuck_window = 3
        s.spec_success_ignore_backlog = True
        s.spec_convergence_window = 0
        # Keep artifacts tiny for test
        s.spec_cycle_artifact_retention = 1
        mock_settings.return_value = s

        spec_json = """```json\n{\"goals\":[\"G\"],\"functional_spec\":[\"F\"],\"non_functional_requirements\":[],\"acceptance_criteria\":[\"实现登录功能\"],\"out_of_scope\":[],\"risks\":[],\"clarification_questions\":[],\"decisions\":[],\"version\":\"1.0\"}\n```"""
        plan_json = """```json\n{\"architecture\":\"A\",\"tech_stack\":[],\"steps\":[\"S\"],\"file_changes\":[],\"test_plan\":[],\"risks\":[],\"version\":\"1.0\"}\n```"""
        discovery1 = (
            """```json\n[{"id":"Q-1","question":"如何提升错误提示可用性？","why":"用户体验","priority":"P1"}]\n```"""
        )
        gen1 = """```json\n[{"id":"Q-1","spec":{"goals":["提升错误提示"],"functional_spec":["完善错误提示"],"non_functional_requirements":[],"acceptance_criteria":["错误提示清晰可读"],"out_of_scope":[],"risks":[],"clarification_questions":[],"decisions":[],"version":"1.0"}}]\n```"""

        # Cycle 1: spec, plan, task, build, criteria(FAIL), discovery, gen
        # Cycle 2: (spec loaded from file), plan, task, build, criteria(PASS)
        #          discovery 被门控跳过（all_satisfied=True + gate_on_satisfied=True）
        session = self._make_mock_session(
            [
                spec_json,
                plan_json,
                "1. T1 (依赖: 无)",
                "build ok",
                "CRITERIA_1: FAIL",
                discovery1,
                gen1,
                plan_json,
                "1. T2 (依赖: 无)",
                "build ok 2",
                "CRITERIA_1: PASS",
            ]
        )
        mock_create.return_value = session

        engine = SpecEngine(chat_id="c1", root_path=str(tmp_path))
        project = engine.execute("- 实现登录功能")

        # 修复后行为：all_satisfied + review_passed 时 ignore_backlog=True
        # → 直接 success，不再被 backlog 阻塞
        assert project.status == SpecProjectStatus.COMPLETED
        assert len(project.cycles) == 2
        assert len(project.work_items) >= 1
        # The first generated item should have been consumed in cycle 2
        assert project.work_items[0].used_in_cycle in (1, 2)
        assert os.path.exists(project.work_items[0].spec_path)



# ======================================================================
# TestSpecEngineProjectTypes — web/api/script variants
# ======================================================================




# ======================================================================
# TestLooseReviewParsing — parse_review_output_loose
# ======================================================================




# ======================================================================
# TestSpecReporterNewMethods — status_line, duration_line, criteria_section
# ======================================================================





# ======================================================================
# TimeoutError 改进: 诊断 / 文案 / 配置项
# ======================================================================


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (TimeoutError(), "审查超时"),
        (RuntimeError(), "审查执行异常"),
    ],
)
def test_review_exception_diagnostics_has_friendly_text(error, expected):
    diagnostics = normalize_review_diagnostics(
        build_review_exception_diagnostics(error, cycle=1)
    )

    assert expected in diagnostics["error_text"]


def test_spec_review_timeout_config_exists_and_defaults():
    from src.config import Settings

    settings = Settings(feishu_app_id="x", feishu_app_secret="x")

    assert settings.spec_review_timeout == 240


def test_review_timeout_is_passed_to_prompt_runner():
    _, observed_timeouts = _run_review_with_error(RuntimeError("bad input"), timeout=42)

    assert observed_timeouts
    assert set(observed_timeouts) == {42}


class TestReviewCircuitStatePersistence:
    """Verify circuit state survives save → load round-trip."""

    def test_save_load_roundtrip(self, tmp_path):
        """Circuit counters survive save_engine_state → load_engine_state."""
        from src.spec_engine.persistence import load_engine_state, save_engine_state
        from src.spec_engine.review import ReviewCircuitState

        # Prepare a minimal SpecProject
        proj = SpecProject(
            project_id="p1", name="test", root_path=str(tmp_path),
            requirement="test req",
        )
        circuit = ReviewCircuitState(
            review_failure_consecutive=2,
            review_circuit_open_until_cycle=5,
            backoff_level=1,
            consecutive_timeouts=3,
        )

        settings = MagicMock()
        settings.spec_state_filename = ".spec_state.json"
        fp = str(tmp_path / ".spec_state.json")

        save_engine_state(
            project=proj, settings=settings, root_path=str(tmp_path),
            chat_id="c1",
            build_runtime_context_fn=lambda: {},
            project_to_compact_dict_fn=proj.to_dict,
            filepath=fp,
            review_circuit=circuit.to_dict(),
        )

        loaded_proj, rc_dict = load_engine_state(fp)
        assert loaded_proj is not None
        restored = ReviewCircuitState.from_dict(rc_dict)
        assert restored.review_failure_consecutive == 2
        assert restored.review_circuit_open_until_cycle == 5
        assert restored.backoff_level == 1
        assert restored.consecutive_timeouts == 3

    def test_load_old_format_without_circuit(self, tmp_path):
        """Old snapshots (no review_circuit key) return default values."""
        from src.spec_engine.persistence import load_engine_state
        from src.spec_engine.review import ReviewCircuitState

        proj = SpecProject(
            project_id="p2", name="old", root_path=str(tmp_path),
        )
        # Simulate old-format state file (no review_circuit key)
        old_state = {
            "chat_id": "c1",
            "root_path": str(tmp_path),
            "project": proj.to_dict(),
            "saved_at": 1.0,
        }
        fp = str(tmp_path / "old_state.json")
        with open(fp, "w") as f:
            json.dump(old_state, f)

        loaded_proj, rc_dict = load_engine_state(fp)
        assert loaded_proj is not None
        assert rc_dict == {}
        restored = ReviewCircuitState.from_dict(rc_dict) if rc_dict else ReviewCircuitState()
        assert restored.backoff_level == 0
        assert restored.consecutive_timeouts == 0
        assert restored.review_failure_consecutive == 0






class TestSpecEngineCycleResilience:
    """Tests for cycle-level exception digestion: exceptions inside a cycle
    should NOT abort the engine but instead mark the cycle failed and continue."""

    _SPEC_JSON = '```json\n{"goals":["G"],"functional_spec":["F"],"non_functional_requirements":[],"acceptance_criteria":["实现功能"],"out_of_scope":[],"risks":[],"clarification_questions":[],"decisions":[],"version":"1.0"}\n```'
    _PLAN_JSON = '```json\n{"architecture":"A","tech_stack":[],"steps":["S1"],"file_changes":[],"test_plan":[],"risks":[],"version":"1.0"}\n```'

    def _mock_settings(self):
        return _make_spec_settings(
            spec_max_cycles=2,
            spec_review_enabled=False,
            spec_persist_phase_artifacts=False,
            spec_persist_every_phase=False,
            spec_max_consecutive_failures=3,
            spec_backlog_stuck_window=0,
            spec_allow_resume_from_disk=False,
            spec_model_switch_enabled=False,
            spec_review_timeout=60,
            spec_review_max_parallel=2,
            spec_review_failure_max_cooldown_cycles=12,
            review_circuit_window_size=10,
            review_circuit_success_rate_threshold=0.3,
            review_circuit_lint_fallback_enabled=False,
            review_circuit_lint_timeout=10,
        )

    def _make_mock_session(self, send_fn):
        session = MagicMock()
        session.send_prompt = send_fn
        session.send_prompt_with_retry = send_fn
        return session

    def _apply_engine_mocks(self, engine):
        """Prevent real session creation / model switching in tests."""
        engine._recreate_session_best_effort = lambda: None
        engine._try_switch_model = lambda callbacks: False

    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_cycle_exception_digested_continues_next_cycle(self, mock_settings, mock_create, tmp_path):
        """Cycle 1 raises RuntimeError in SPEC phase → cycle marked failed → cycle 2 succeeds → COMPLETED."""
        s = self._mock_settings()
        s.spec_max_cycles = 2
        s.spec_success_ignore_backlog = True
        mock_settings.return_value = s

        call_count = [0]
        criteria_text = "CRITERIA_1: PASS"

        def fake_send(prompt, on_event=None, timeout=None, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("模型切换失败")
            # Cycle 2: spec, plan, task, build, criteria
            texts = [self._SPEC_JSON, self._PLAN_JSON, "1. T1 (依赖: 无)", "build done " * 10, criteria_text]
            idx = call_count[0] - 2
            text = texts[idx] if idx < len(texts) else "ok"
            if on_event and text:
                on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=text))
            return MagicMock(stop_reason="end_turn")

        session = self._make_mock_session(fake_send)
        mock_create.return_value = session

        engine = SpecEngine(chat_id="c1", root_path=str(tmp_path))
        self._apply_engine_mocks(engine)
        project = engine.execute("- 实现功能")

        assert project.status == SpecProjectStatus.COMPLETED
        assert len(project.cycles) == 2
        assert project.cycles[0].status == "failed"
        assert project.cycles[0].error_message is not None
        assert "模型切换失败" in project.cycles[0].error_message
        assert project.cycles[1].status == "completed"

    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_consecutive_failures_aborts_engine(self, mock_settings, mock_create, tmp_path):
        """All cycles fail → consecutive_failures termination → ABORTED."""
        s = self._mock_settings()
        s.spec_max_cycles = 10
        s.spec_max_consecutive_failures = 2
        mock_settings.return_value = s

        def always_fail(prompt, on_event=None, timeout=None, **kw):
            raise RuntimeError("always fail")

        session = self._make_mock_session(always_fail)
        mock_create.return_value = session

        engine = SpecEngine(chat_id="c1", root_path=str(tmp_path))
        self._apply_engine_mocks(engine)
        project = engine.execute("- test req")

        assert project.status == SpecProjectStatus.FAILED
        assert "连续异常终止" in (project.error or "")
        assert len(project.cycles) == 2
        assert all(c.status == "failed" for c in project.cycles)

    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_successful_cycle_resets_failure_counter(self, mock_settings, mock_create, tmp_path):
        """Fail → success → fail → should NOT trigger consecutive_failures (max=2)."""
        s = self._mock_settings()
        s.spec_max_cycles = 3
        s.spec_max_consecutive_failures = 2
        s.spec_success_ignore_backlog = True
        mock_settings.return_value = s

        call_count = [0]
        criteria_text = "CRITERIA_1: PASS"

        def fake_send(prompt, on_event=None, timeout=None, **kw):
            call_count[0] += 1
            # Cycle 1 (calls 1): fail at SPEC
            if call_count[0] == 1:
                raise RuntimeError("cycle 1 fail")
            # Cycle 2 (calls 2-6): succeed — spec, plan, task, build, criteria
            if 2 <= call_count[0] <= 6:
                texts = [self._SPEC_JSON, self._PLAN_JSON, "1. T1 (依赖: 无)", "build done " * 10, criteria_text]
                idx = call_count[0] - 2
                text = texts[idx] if idx < len(texts) else "ok"
                if on_event and text:
                    on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=text))
                return MagicMock(stop_reason="end_turn")
            # Cycle 3 would fail but should not get here — cycle 2 should succeed and terminate
            raise RuntimeError("cycle 3 fail")

        session = self._make_mock_session(fake_send)
        mock_create.return_value = session

        engine = SpecEngine(chat_id="c1", root_path=str(tmp_path))
        self._apply_engine_mocks(engine)
        project = engine.execute("- 实现功能")

        # Cycle 2 should succeed and terminate engine
        assert project.status == SpecProjectStatus.COMPLETED
        assert project.cycles[0].status == "failed"
        assert project.cycles[1].status == "completed"

    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_timeout_error_digested_in_cycle(self, mock_settings, mock_create, tmp_path):
        """TimeoutError should be digested inside the cycle, not bubble to execute()."""
        s = self._mock_settings()
        s.spec_max_cycles = 2
        s.spec_success_ignore_backlog = True
        mock_settings.return_value = s

        call_count = [0]
        criteria_text = "CRITERIA_1: PASS"

        def fake_send(prompt, on_event=None, timeout=None, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                raise TimeoutError("phase timeout")
            texts = [self._SPEC_JSON, self._PLAN_JSON, "1. T1 (依赖: 无)", "build done " * 10, criteria_text]
            idx = call_count[0] - 2
            text = texts[idx] if idx < len(texts) else "ok"
            if on_event and text:
                on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=text))
            return MagicMock(stop_reason="end_turn")

        session = self._make_mock_session(fake_send)
        mock_create.return_value = session

        engine = SpecEngine(chat_id="c1", root_path=str(tmp_path))
        self._apply_engine_mocks(engine)
        project = engine.execute("- 实现功能")

        assert project.status == SpecProjectStatus.COMPLETED
        assert project.cycles[0].status == "failed"
        assert project.cycles[0].error_message is not None
        assert project.cycles[1].status == "completed"

    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_session_recreated_after_cycle_exception(self, mock_settings, mock_create, tmp_path):
        """After cycle exception, _recreate_session_best_effort should be called."""
        s = self._mock_settings()
        s.spec_max_cycles = 2
        s.spec_max_consecutive_failures = 3
        s.spec_success_ignore_backlog = True
        mock_settings.return_value = s

        call_count = [0]
        criteria_text = "CRITERIA_1: PASS"

        def fake_send(prompt, on_event=None, timeout=None, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("fail")
            texts = [self._SPEC_JSON, self._PLAN_JSON, "1. T1 (依赖: 无)", "build done " * 10, criteria_text]
            idx = call_count[0] - 2
            text = texts[idx] if idx < len(texts) else "ok"
            if on_event and text:
                on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=text))
            return MagicMock(stop_reason="end_turn")

        session = self._make_mock_session(fake_send)
        mock_create.return_value = session

        engine = SpecEngine(chat_id="c1", root_path=str(tmp_path))
        engine._try_switch_model = lambda callbacks: False

        recreate_calls = []

        def tracked_recreate():
            recreate_calls.append(1)

        engine._recreate_session_best_effort = tracked_recreate

        engine.execute("- 实现功能")

        # At least one recreate call after cycle 1 failure
        assert len(recreate_calls) >= 1

    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_stopping_during_exception_is_cancelled(self, mock_settings, mock_create, tmp_path):
        """If engine is STOPPING when exception occurs, should result in CANCELLED, not digest."""
        s = self._mock_settings()
        s.spec_max_cycles = 5
        s.spec_max_consecutive_failures = 10
        mock_settings.return_value = s

        def fail_then_stop(prompt, on_event=None, timeout=None, **kw):
            raise RuntimeError("session cancelled")

        session = self._make_mock_session(fail_then_stop)
        mock_create.return_value = session

        engine = SpecEngine(chat_id="c1", root_path=str(tmp_path))
        self._apply_engine_mocks(engine)

        # Set engine to STOPPING before the exception is caught
        original_run_phase = engine._run_phase

        def patched_run_phase(*args, **kwargs):
            engine._run_state = EngineRunState.STOPPING
            return original_run_phase(*args, **kwargs)

        engine._run_phase = patched_run_phase

        project = engine.execute("- test")

        assert project.status == SpecProjectStatus.CANCELLED

    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_error_message_field_persisted_on_failed_cycle(self, mock_settings, mock_create, tmp_path):
        """SpecCycle.error_message should be set and serializable."""
        s = self._mock_settings()
        s.spec_max_cycles = 1
        s.spec_max_consecutive_failures = 5
        mock_settings.return_value = s

        def fail_send(prompt, on_event=None, timeout=None, **kw):
            raise RuntimeError("test error detail xyz")

        session = self._make_mock_session(fail_send)
        mock_create.return_value = session

        engine = SpecEngine(chat_id="c1", root_path=str(tmp_path))
        self._apply_engine_mocks(engine)
        project = engine.execute("- test req")

        assert len(project.cycles) == 1
        cycle = project.cycles[0]
        assert cycle.status == "failed"
        assert cycle.error_message is not None
        assert "test error detail xyz" in cycle.error_message

        # Verify serialization roundtrip
        d = cycle.to_dict()
        assert "error_message" in d
        restored = SpecCycle.from_dict(d)
        assert restored.error_message == cycle.error_message


# ---------------------------------------------------------------------------
# Tests merged from test_spec_engine_di.py
# ---------------------------------------------------------------------------



    # We can check that the retry_delay and backoff_multiplier were read from mock_retry_policy
    # indirectly if we patch RetryPolicy inside engine._run_phase, but knowing it's stored
    # and accessed is usually sufficient for DI unit tests.


# ---------------------------------------------------------------------------
# Tests merged from test_spec_gc.py
# ---------------------------------------------------------------------------
