import threading
from unittest.mock import MagicMock

import pytest

from src.utils.circuit_breaker import CircuitBreaker, CircuitBreakerOpenException
from src.utils.retry import (
    RetryPolicy,
    get_retry_delay,
    prompt_with_retry,
    should_retry,
)


class TestRetryPolicy:
    def test_default_values(self):
        policy = RetryPolicy()
        assert policy.max_retries == 3
        assert policy.retry_delay == 2.0
        assert policy.backoff_multiplier == 1.5
        assert policy.max_delay == 60.0
        assert policy.jitter_factor == 0.25


class TestShouldRetry:
    @pytest.mark.parametrize(
        "error_msg",
        [
            "Invalid params: xyz",
            "Connection timeout",
            "connection reset by peer",
            "rate limit exceeded",
            "ratelimit hit",
            "too many requests",
            "server is overloaded",
            "server error 500",
            "internal error occurred",
        ],
    )
    def test_retryable_errors(self, error_msg):
        assert should_retry(error_msg) is True

    @pytest.mark.parametrize(
        "error_msg",
        [
            "directory /tmp/xyz not found",
            "directory does not exist",
            "Permission denied: /etc/passwd",
            "Authentication failed for user",
        ],
    )
    def test_non_retryable_errors(self, error_msg):
        assert should_retry(error_msg) is False

    def test_unknown_error_returns_false(self):
        assert should_retry("unknown error") is False

    def test_exception_input(self):
        assert should_retry(Exception("timeout")) is True
        assert should_retry(ValueError("permission denied")) is False


class TestGetRetryDelay:
    def test_delay_calculation_no_jitter(self):
        policy = RetryPolicy(retry_delay=2.0, backoff_multiplier=1.5, jitter_factor=0)
        assert get_retry_delay(0, policy) == 2.0
        assert get_retry_delay(1, policy) == 3.0
        assert get_retry_delay(2, policy) == 4.5

    def test_with_different_policy_no_jitter(self):
        policy = RetryPolicy(retry_delay=1.0, backoff_multiplier=2.0, jitter_factor=0)
        assert get_retry_delay(0, policy) == 1.0
        assert get_retry_delay(1, policy) == 2.0
        assert get_retry_delay(2, policy) == 4.0
        assert get_retry_delay(3, policy) == 8.0

    def test_max_delay_cap(self):
        policy = RetryPolicy(retry_delay=10.0, backoff_multiplier=3.0, max_delay=20.0, jitter_factor=0)
        assert get_retry_delay(0, policy) == 10.0
        assert get_retry_delay(1, policy) == 20.0
        assert get_retry_delay(2, policy) == 20.0

    def test_jitter_within_bounds(self):
        policy = RetryPolicy(retry_delay=10.0, backoff_multiplier=1.0, jitter_factor=0.25)
        for _ in range(100):
            delay = get_retry_delay(0, policy)
            assert 7.5 <= delay <= 12.5

    def test_zero_jitter_deterministic(self):
        policy = RetryPolicy(retry_delay=5.0, backoff_multiplier=2.0, jitter_factor=0)
        assert get_retry_delay(0, policy) == 5.0
        assert get_retry_delay(0, policy) == 5.0


class TestPromptWithRetry:
    def test_success_first_attempt(self):
        action = MagicMock(return_value="ok")
        cancel = threading.Event()
        result = prompt_with_retry(action, cancel)
        assert result == "ok"
        assert action.call_count == 1

    def test_retries_on_retryable_error(self):
        action = MagicMock(side_effect=[RuntimeError("timeout"), "ok"])
        cancel = threading.Event()
        policy = RetryPolicy(max_retries=2, retry_delay=0.01, jitter_factor=0)
        result = prompt_with_retry(action, cancel, retry_policy=policy)
        assert result == "ok"
        assert action.call_count == 2

    def test_raises_on_non_retryable_error(self):
        action = MagicMock(side_effect=RuntimeError("permission denied"))
        cancel = threading.Event()
        policy = RetryPolicy(max_retries=3, retry_delay=0.01, jitter_factor=0)
        with pytest.raises(RuntimeError, match="permission denied"):
            prompt_with_retry(action, cancel, retry_policy=policy)
        assert action.call_count == 1

    def test_raises_after_max_retries(self):
        action = MagicMock(side_effect=RuntimeError("timeout"))
        cancel = threading.Event()
        policy = RetryPolicy(max_retries=2, retry_delay=0.01, jitter_factor=0)
        with pytest.raises(RuntimeError, match="timeout"):
            prompt_with_retry(action, cancel, retry_policy=policy)
        assert action.call_count == 3

    def test_before_retry_callback(self):
        action = MagicMock(side_effect=[RuntimeError("timeout"), "ok"])
        cancel = threading.Event()
        policy = RetryPolicy(max_retries=2, retry_delay=0.01, jitter_factor=0)
        callback = MagicMock()
        prompt_with_retry(action, cancel, retry_policy=policy, before_retry=callback)
        callback.assert_called_once()
        assert callback.call_args[0][0] == 1

    def test_cancel_event_aborts_retry(self):
        call_count = 0

        def action():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("timeout")

        cancel = threading.Event()
        cancel.set()
        policy = RetryPolicy(max_retries=3, retry_delay=0.01, jitter_factor=0)
        with pytest.raises(RuntimeError, match="timeout"):
            prompt_with_retry(action, cancel, retry_policy=policy)
        assert call_count == 1

    def test_circuit_breaker_integration(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60.0)
        action = MagicMock(side_effect=[RuntimeError("timeout"), RuntimeError("timeout"), "ok"])
        cancel = threading.Event()
        policy = RetryPolicy(max_retries=5, retry_delay=0.01, jitter_factor=0)
        with pytest.raises(CircuitBreakerOpenException):
            prompt_with_retry(action, cancel, retry_policy=policy, circuit_breaker=cb)

    def test_circuit_breaker_success(self):
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
        action = MagicMock(return_value="ok")
        cancel = threading.Event()
        result = prompt_with_retry(action, cancel, circuit_breaker=cb)
        assert result == "ok"


# ---------------------------------------------------------------------------
# Task 4: should_retry isinstance short-circuit + retry observability logs
# ---------------------------------------------------------------------------

import asyncio
import logging


class TestShouldRetryTimeoutShortCircuit:
    """should_retry must return True for bare TimeoutError/asyncio.TimeoutError
    even when str(error) is empty (no 'timeout' substring to match)."""

    def test_bare_timeout_error(self):
        assert should_retry(TimeoutError()) is True

    def test_bare_asyncio_timeout_error(self):
        assert should_retry(asyncio.TimeoutError()) is True

    def test_timeout_error_empty_string(self):
        assert should_retry(TimeoutError("")) is True

    def test_timeout_error_with_message(self):
        assert should_retry(TimeoutError("ACP prompt 超时")) is True

    def test_non_timeout_exception_not_affected(self):
        """instanceof short-circuit must not change behavior for non-TimeoutError."""
        assert should_retry(ValueError("unknown issue")) is False

    def test_non_retryable_string_still_rejected(self):
        """String inputs still follow pattern matching, not isinstance."""
        assert should_retry("permission denied") is False


class TestPromptWithRetryObservabilityLogs:
    """prompt_with_retry must emit retry observability logs with expected fields."""

    def test_retry_log_emitted_with_expected_fields(self, caplog):
        """Verify retry log contains attempt, elapsed_ms, remaining_budget."""
        action = MagicMock(side_effect=[TimeoutError(), "ok"])
        cancel = threading.Event()
        policy = RetryPolicy(max_retries=2, retry_delay=0.01, jitter_factor=0)

        with caplog.at_level(logging.INFO, logger="src.utils.retry"):
            result = prompt_with_retry(action, cancel, retry_policy=policy)

        assert result == "ok"
        retry_logs = [r for r in caplog.records if "prompt_with_retry: retry" in r.getMessage()]
        assert len(retry_logs) >= 1
        msg = retry_logs[0].getMessage()
        assert "attempt=1/2" in msg
        assert "elapsed_ms=" in msg
        assert "remaining_budget=unlimited" in msg

    def test_retry_log_with_total_timeout_shows_budget(self, caplog):
        """When total_timeout is set, remaining_budget shows seconds."""
        action = MagicMock(side_effect=[TimeoutError(), "ok"])
        cancel = threading.Event()
        policy = RetryPolicy(max_retries=2, retry_delay=0.01, jitter_factor=0)

        with caplog.at_level(logging.INFO, logger="src.utils.retry"):
            result = prompt_with_retry(
                action, cancel, retry_policy=policy, total_timeout=300.0,
            )

        assert result == "ok"
        retry_logs = [r for r in caplog.records if "prompt_with_retry: retry" in r.getMessage()]
        assert len(retry_logs) >= 1
        msg = retry_logs[0].getMessage()
        assert "remaining_budget=" in msg
        assert "unlimited" not in msg  # should show actual seconds
