"""Configuration validation tests — cross-field constraints and friendly error output.

Covers:
- AC-R14: min_timeout > timeout is rejected
- AC-R14: hard_floor > min_timeout is rejected
- AC-R14: retry_max_delay > timeout is rejected
- AC-R14: total retry budget exceeded is rejected
- AC-R15: spec_review_max_parallel boundary values (0, -1, 21) rejected
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.config import Settings


class TestMinTimeoutGtTimeoutRejected:
    """AC-R14: spec_review_min_timeout > spec_review_timeout must raise ValidationError."""

    def test_min_timeout_exceeds_timeout_raises(self):
        with pytest.raises(Exception) as exc_info:
            Settings(spec_review_min_timeout=150, spec_review_timeout=120)
        error_str = str(exc_info.value)
        assert "spec_review_min_timeout" in error_str or "min_timeout" in error_str

    def test_min_timeout_equals_timeout_ok(self):
        """Equal values should be acceptable (with compatible retry budget)."""
        s = Settings(
            spec_review_min_timeout=120,
            spec_review_timeout=120,
            spec_review_retry_max_attempts=0,  # Disable retry to avoid budget constraint
        )
        assert s.spec_review_min_timeout == 120

    def test_min_timeout_less_than_timeout_ok(self):
        s = Settings(spec_review_min_timeout=30, spec_review_timeout=120)
        assert s.spec_review_min_timeout == 30


class TestHardFloorGtMinTimeoutRejected:
    """AC-R14: spec_review_hard_floor > spec_review_min_timeout must raise ValidationError."""

    def test_hard_floor_exceeds_min_timeout_raises(self):
        with pytest.raises(Exception) as exc_info:
            Settings(spec_review_hard_floor=50, spec_review_min_timeout=30)
        error_str = str(exc_info.value)
        assert "hard_floor" in error_str or "min_timeout" in error_str

    def test_hard_floor_equals_min_timeout_ok(self):
        s = Settings(spec_review_hard_floor=30, spec_review_min_timeout=30)
        assert s.spec_review_hard_floor == 30


class TestRetryMaxDelayGtTimeoutRejected:
    """AC-R14: spec_review_retry_max_delay > spec_review_timeout must raise ValidationError."""

    def test_retry_max_delay_exceeds_timeout_raises(self):
        with pytest.raises(Exception) as exc_info:
            Settings(spec_review_retry_max_delay=150, spec_review_timeout=120)
        error_str = str(exc_info.value)
        assert "retry_max_delay" in error_str or "timeout" in error_str

    def test_retry_max_delay_equals_timeout_ok(self):
        s = Settings(
            spec_review_retry_max_delay=120,
            spec_review_timeout=120,
            spec_review_retry_max_attempts=0,  # Disable retry to avoid budget constraint
        )
        assert s.spec_review_retry_max_delay == 120


class TestRetryBudgetExceededRejected:
    """AC-R14: (max_delay + min_timeout) * max_attempts > timeout * 2 must raise."""

    def test_budget_exceeded_raises(self):
        # (60 + 60) * 3 = 360 > 120 * 2 = 240
        with pytest.raises(Exception) as exc_info:
            Settings(
                spec_review_retry_max_delay=60,
                spec_review_min_timeout=60,
                spec_review_hard_floor=20,
                spec_review_timeout=120,
                spec_review_retry_max_attempts=3,
            )
        error_str = str(exc_info.value)
        assert "budget" in error_str.lower() or "预算" in error_str or "retry" in error_str.lower()

    def test_budget_within_limit_ok(self):
        # (30 + 30) * 2 = 120 <= 120 * 2 = 240
        s = Settings(
            spec_review_retry_max_delay=30,
            spec_review_min_timeout=30,
            spec_review_hard_floor=20,
            spec_review_timeout=120,
            spec_review_retry_max_attempts=2,
        )
        assert s.spec_review_retry_max_attempts == 2


class TestMaxParallelBoundaryRejected:
    """AC-R15: spec_review_max_parallel must be in [1, 20]."""

    def test_zero_raises(self):
        with pytest.raises(Exception) as exc_info:
            Settings(spec_review_max_parallel=0)
        assert "spec_review_max_parallel" in str(exc_info.value)

    def test_negative_raises(self):
        with pytest.raises(Exception) as exc_info:
            Settings(spec_review_max_parallel=-1)
        assert "spec_review_max_parallel" in str(exc_info.value)

    def test_twenty_one_raises(self):
        with pytest.raises(Exception) as exc_info:
            Settings(spec_review_max_parallel=21)
        assert "spec_review_max_parallel" in str(exc_info.value)

    def test_one_ok(self):
        s = Settings(spec_review_max_parallel=1)
        assert s.spec_review_max_parallel == 1

    def test_twenty_ok(self):
        s = Settings(spec_review_max_parallel=20)
        assert s.spec_review_max_parallel == 20


class TestMainCatchesConfigurationError:
    """AC-R03: main() catches ConfigurationError and exits gracefully."""

    def test_main_prints_error_and_exits(self):
        from unittest.mock import patch

        from src.config import ConfigurationError

        with patch("src.main.Application", side_effect=ConfigurationError("test config error")):
            with pytest.raises(SystemExit) as exc_info:
                from src.main import main
                main()

        assert exc_info.value.code == 1


class TestPostValidateWarnings:
    """_post_validate_warnings reports soft retry budget notes without startup warning noise."""

    def test_post_validate_warnings_emits_info_log(self, caplog):
        """When (retry_max_delay + timeout) * max_attempts > timeout * 2, an INFO note is emitted."""
        import logging

        from src.config import _post_validate_warnings

        # Construct settings where realistic_budget > budget_limit but within hard validator:
        # (40 + 30) * 2 = 140 > 30 * 2 = 60 — triggers soft warning
        # But (40 + 30) * 2 = 140 <= hard limit check uses min_timeout not base_timeout
        # Use values that pass hard validation: budget check uses (max_delay + min_timeout) * max_attempts <= timeout * 2
        # Hard validator: (retry_max_delay + min_timeout) * max_attempts <= timeout * 2
        # We need: (retry_max_delay + min_timeout) * max_attempts <= timeout * 2 (pass hard)
        # AND: (retry_max_delay + timeout) * max_attempts > timeout * 2 (trigger soft)
        # Use: retry_max_delay=50, timeout=120, min_timeout=30, max_attempts=2
        # Hard: (50 + 30) * 2 = 160 <= 120 * 2 = 240 ✓
        # Soft: (50 + 120) * 2 = 340 > 120 * 2 = 240 ✓
        s = Settings(
            spec_review_retry_max_delay=50,
            spec_review_timeout=120,
            spec_review_min_timeout=30,
            spec_review_hard_floor=20,
            spec_review_retry_max_attempts=2,
        )

        with caplog.at_level(logging.INFO, logger="src.config"):
            _post_validate_warnings(s)

        assert any("重试实际预算可能超限" in r.message for r in caplog.records), (
            f"Expected info log not found. Records: {[r.message for r in caplog.records]}"
        )


# ---------------------------------------------------------------------------
# Task 22: Config validation uses uppercase field names
# ---------------------------------------------------------------------------


class TestConfigValidationUppercase:
    """Validators for lock/chat fields emit uppercase field names in error messages."""

    def test_max_allowed_chat_ids_error_uppercase(self):
        """max_allowed_chat_ids validator error message contains uppercase field name."""
        with pytest.raises(Exception) as exc_info:
            Settings(max_allowed_chat_ids=0)
        error_str = str(exc_info.value)
        assert "MAX_ALLOWED_CHAT_IDS" in error_str

    def test_repo_lock_idle_timeout_error_uppercase(self):
        """repo_lock_idle_timeout validator error message contains uppercase field name."""
        with pytest.raises(Exception) as exc_info:
            Settings(repo_lock_idle_timeout=0)
        error_str = str(exc_info.value)
        assert "REPO_LOCK_IDLE_TIMEOUT" in error_str

    def test_chat_lock_max_duration_error_uppercase(self):
        """chat_lock_max_duration validator error message contains uppercase field name."""
        with pytest.raises(Exception) as exc_info:
            Settings(chat_lock_max_duration=0)
        error_str = str(exc_info.value)
        assert "CHAT_LOCK_MAX_DURATION" in error_str


# ---------------------------------------------------------------------------
# Step-01: lock_confirm_timeout and max_evicted_cache > 0 validators
# ---------------------------------------------------------------------------


class TestLockConfirmTimeoutValidation:
    """AC-R16: lock_confirm_timeout must be > 0."""

    def test_zero_raises(self):
        with pytest.raises(Exception) as exc_info:
            Settings(lock_confirm_timeout=0)
        assert "LOCK_CONFIRM_TIMEOUT" in str(exc_info.value)

    def test_negative_raises(self):
        with pytest.raises(Exception) as exc_info:
            Settings(lock_confirm_timeout=-1)
        assert "LOCK_CONFIRM_TIMEOUT" in str(exc_info.value)

    def test_positive_ok(self):
        s = Settings(lock_confirm_timeout=60)
        assert s.lock_confirm_timeout == 60


class TestMaxEvictedCacheValidation:
    """AC-R16: max_evicted_cache must be > 0."""

    def test_zero_raises(self):
        with pytest.raises(Exception) as exc_info:
            Settings(max_evicted_cache=0)
        assert "MAX_EVICTED_CACHE" in str(exc_info.value)

    def test_negative_raises(self):
        with pytest.raises(Exception) as exc_info:
            Settings(max_evicted_cache=-1)
        assert "MAX_EVICTED_CACHE" in str(exc_info.value)

    def test_positive_ok(self):
        s = Settings(max_evicted_cache=100)
        assert s.max_evicted_cache == 100


# ---------------------------------------------------------------------------
# Step-02: lock_backend removed — residual env var silently ignored
# ---------------------------------------------------------------------------


class TestLockBackendRemoved:
    """AC-R09: lock_backend field removed; residual LOCK_BACKEND env var is ignored."""

    def test_residual_lock_backend_env_ignored(self):
        with patch.dict("os.environ", {"LOCK_BACKEND": "memory"}, clear=False):
            s = Settings()
        assert not hasattr(s, "lock_backend")


class TestWorkflowTimeoutSettings:
    """Workflow (/wf) timeout knobs: raised defaults + lower-bound validation."""

    def test_defaults(self):
        # Hermetic: ignore the deployment .env so this asserts the code-level
        # defaults, not whatever the local .env overrides them to.
        s = Settings(_env_file=None)
        assert s.workflow_total_timeout_s == 3600
        assert s.workflow_agent_call_timeout_s == 600
        assert s.workflow_script_gen_timeout_s == 600
        assert s.workflow_session_create_timeout_s == 120

        from src.workflow_engine.constants import SCRIPT_GEN_TIMEOUT_S

        assert SCRIPT_GEN_TIMEOUT_S == s.workflow_script_gen_timeout_s

    def test_env_override_applied(self):
        s = Settings(
            workflow_total_timeout_s=7200,
            workflow_agent_call_timeout_s=1200,
            workflow_script_gen_timeout_s=300,
            workflow_session_create_timeout_s=240,
        )
        assert s.workflow_total_timeout_s == 7200
        assert s.workflow_agent_call_timeout_s == 1200
        assert s.workflow_script_gen_timeout_s == 300
        assert s.workflow_session_create_timeout_s == 240

    def test_total_timeout_zero_allowed_means_unlimited(self):
        # 0 disables the total deadline entirely (unlimited long-running /wf).
        s = Settings(workflow_total_timeout_s=0)
        assert s.workflow_total_timeout_s == 0

    def test_total_timeout_negative_rejected(self):
        with pytest.raises(Exception):
            Settings(workflow_total_timeout_s=-1)  # ge=0

    def test_agent_call_timeout_zero_allowed_means_unlimited(self):
        # 0 disables the per-agent deadline entirely (unlimited long-running
        # agent() calls); the value is otherwise the authoritative floor.
        s = Settings(workflow_agent_call_timeout_s=0)
        assert s.workflow_agent_call_timeout_s == 0

    def test_agent_call_timeout_negative_rejected(self):
        with pytest.raises(Exception):
            Settings(workflow_agent_call_timeout_s=-1)  # ge=0

    def test_script_gen_timeout_below_floor_rejected(self):
        with pytest.raises(Exception):
            Settings(workflow_script_gen_timeout_s=5)  # ge=10

    def test_session_create_timeout_below_floor_rejected(self):
        with pytest.raises(Exception):
            Settings(workflow_session_create_timeout_s=5)  # ge=10


class TestProgrammingExecutionWindowSettings:
    def test_default_allows_four_bounded_windows(self):
        settings = Settings(_env_file=None)

        assert settings.programming_max_execution_windows == 4

    @pytest.mark.parametrize("value", [0, -1, 25])
    def test_rejects_unsafe_window_limits(self, value: int):
        with pytest.raises(Exception):
            Settings(
                _env_file=None,
                programming_max_execution_windows=value,
            )


class TestAutonomousTeamAndContextRetrySettings:
    def test_defaults_are_typed_and_discoverable(self):
        settings = Settings(_env_file=None)

        assert settings.autonomous_context_retry_base_seconds == 1.0
        assert settings.autonomous_context_retry_max_seconds == 30.0
        assert settings.autonomous_team_step_timeout_seconds == 600.0

    @pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), -1.0, 0.0])
    @pytest.mark.parametrize(
        "field",
        [
            "autonomous_context_retry_base_seconds",
            "autonomous_context_retry_max_seconds",
            "autonomous_team_step_timeout_seconds",
        ],
    )
    def test_rejects_invalid_numeric_values(self, field, value):
        with pytest.raises(Exception):
            Settings(_env_file=None, **{field: value})

    def test_context_retry_base_cannot_exceed_maximum(self):
        with pytest.raises(Exception, match="context retry"):
            Settings(
                _env_file=None,
                autonomous_context_retry_base_seconds=31.0,
                autonomous_context_retry_max_seconds=30.0,
            )


class TestTaskSchedulerBoundedSettings:
    def test_defaults_are_positive_and_bounded(self):
        settings = Settings(_env_file=None)

        assert settings.task_scheduler_max_pending_normal == 1000
        assert settings.task_scheduler_max_pending_system == 100
        assert settings.task_scheduler_max_terminal_history == 5000

    @pytest.mark.parametrize(
        "field",
        [
            "task_scheduler_max_pending_normal",
            "task_scheduler_max_pending_system",
            "task_scheduler_max_terminal_history",
        ],
    )
    @pytest.mark.parametrize("value", [0, -1])
    def test_non_positive_limits_are_rejected(self, field: str, value: int):
        with pytest.raises(Exception):
            Settings(_env_file=None, **{field: value})

    def test_scheduler_builder_wires_all_bounded_limits(self):
        from src.feishu.ws_client import _build_task_scheduler

        settings = SimpleNamespace(
            restart_gate_dir="",
            task_scheduler_max_concurrent=2,
            task_scheduler_per_key_concurrency=1,
            system_command_concurrency=1,
            task_scheduler_max_pending_normal=17,
            task_scheduler_max_pending_system=5,
            task_scheduler_max_terminal_history=23,
        )
        gate = MagicMock()
        gate.task_guard = nullcontext
        gate.cancel_waiters = MagicMock()

        with patch(
            "src.feishu.ws_client.RestartGate.for_project",
            return_value=gate,
        ):
            scheduler = _build_task_scheduler(settings)

        try:
            assert scheduler._max_pending_normal == 17
            assert scheduler._max_pending_system == 5
            assert scheduler._max_terminal_history == 23
        finally:
            scheduler.stop(wait=True, shutdown_executor=True)


class TestFeishuCOTSettings:
    def test_cot_is_enabled_by_default(self):
        assert Settings(_env_file=None).feishu_cot_enabled is True

    def test_cot_emergency_switch_can_disable_transport(self):
        assert Settings(_env_file=None, feishu_cot_enabled=False).feishu_cot_enabled is False
