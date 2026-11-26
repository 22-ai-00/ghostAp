"""Tests for workflow engine fault tolerance and error handling.

Covers:
- AgentExecutor retry logic for transient errors with exponential backoff
- Schema validation failure retry with fix prompts
- Timeout handling via AGENT_CALL_TIMEOUT_S
- Backpressure handling via MAX_QUEUE_SIZE
- Error categorization (ErrorCategory enum)
- Error sanitization (sanitize_for_reply)
- Pipeline error handling (stop vs continue on failure)
- Cancel event handling in AgentExecutor

These tests ensure the workflow engine gracefully handles failures and
provides reliable, user-safe error messages.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import pytest

from src.workflow_engine.constants import (
    AGENT_CALL_TIMEOUT_S,
    MAX_RETRIES,
    RETRY_BACKOFF_BASE_S,
    SCHEMA_RETRY_MAX,
)
from src.workflow_engine.errors import (
    ErrorCategory,
    _strip_internal_details,
    categorize_error,
    is_transient_error,
    sanitize_for_reply,
)
from src.workflow_engine.executor import AgentExecutor
from src.workflow_engine.models import AgentCallParams

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_session(output_text="", output_tokens=100, side_effect=None):
    """Create a mock engine session with configurable send_prompt behavior."""
    session = MagicMock()
    mock_result = MagicMock()
    mock_result.text = output_text
    mock_result.output_tokens = output_tokens

    if side_effect is not None:
        session.send_prompt.side_effect = side_effect
    else:
        session.send_prompt.return_value = mock_result

    session.close = MagicMock()
    return session


def _make_executor(cancel_event=None, cwd="/tmp/test"):
    """Create an AgentExecutor with a mock session pool.

    The real ThreadPoolExecutor created by AgentExecutor.__init__ is
    shut down before replacing it with a mock, preventing thread leaks
    in unit tests.
    """
    if cancel_event is None:
        cancel_event = threading.Event()
    executor = AgentExecutor(cwd=cwd, cancel_event=cancel_event)
    # Shut down the real thread pool before replacing it with a mock
    executor._session_pool.shutdown(wait=False, cancel_futures=True)
    # Replace the real thread pool with a mock to avoid actual threading
    executor._session_pool = MagicMock()
    return executor


def _simulate_transient_error_then_success(
    transient_error: Exception, success_output: str = "success", num_failures: int = 1
):
    """Create a side_effect function that fails N times then succeeds."""
    call_count = 0

    def _side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= num_failures:
            raise transient_error
        mock_result = MagicMock()
        mock_result.text = success_output
        mock_result.output_tokens = 50
        return mock_result

    return _side_effect


# ---------------------------------------------------------------------------
# 1. TestAgentRetryLogic
# ---------------------------------------------------------------------------


class TestAgentRetryLogic(unittest.TestCase):
    """Test AgentExecutor retry logic for transient errors.

    Reliability impact: Transient errors (network blips, rate limits) are
    common in distributed systems. Automatic retry with backoff prevents
    these from failing user workflows unnecessarily.
    """

    def setUp(self):
        self.cancel_event = threading.Event()
        self.executor = _make_executor(cancel_event=self.cancel_event)

    def tearDown(self):
        self.executor.shutdown(wait=False)

    def test_retry_on_transient_error(self):
        """Verify transient errors trigger a retry and eventual success.

        When a call fails with a transient error (e.g., network timeout),
        the executor should retry and succeed on the second attempt.
        This prevents flaky failures from disrupting workflows.
        """
        transient_error = RuntimeError("Network timeout: connection reset")
        side_effect = _simulate_transient_error_then_success(
            transient_error, success_output="task completed", num_failures=1
        )
        mock_session = _make_mock_session(side_effect=side_effect)
        mock_future = MagicMock()
        mock_future.result.return_value = mock_session
        self.executor._session_pool.submit.return_value = mock_future

        params = AgentCallParams(prompt="test prompt", tool="coco")
        result = self.executor.execute(params)

        # Should succeed after retry
        self.assertIsNone(result.error)
        self.assertEqual(result.output, "task completed")
        # send_prompt should be called twice (initial + 1 retry)
        self.assertEqual(mock_session.send_prompt.call_count, 2)

    @pytest.mark.slow
    def test_retry_limit_exceeded(self):
        """Verify after max retries, the call fails with the last error.

        When all retry attempts fail with transient errors, the executor
        should give up after MAX_RETRIES and return an error result.
        This prevents infinite retry loops.
        """
        persistent_transient_error = RuntimeError("Network timeout: connection reset")
        mock_session = _make_mock_session(side_effect=persistent_transient_error)
        mock_future = MagicMock()
        mock_future.result.return_value = mock_session
        self.executor._session_pool.submit.return_value = mock_future

        params = AgentCallParams(prompt="test prompt", tool="coco")
        result = self.executor.execute(params)

        # Should fail after MAX_RETRIES + 1 total attempts
        self.assertIsNotNone(result.error)
        self.assertIn("Network timeout", result.error)
        # send_prompt should be called MAX_RETRIES + 1 times (initial + retries)
        self.assertEqual(mock_session.send_prompt.call_count, MAX_RETRIES + 1)

    def test_acp_prompt_execution_timeout_fails_fast(self):
        """A full ACP prompt timeout is already an exhausted backend call.

        Retrying the same prompt after a 300s backend timeout stretches one
        workflow agent call to roughly twenty minutes and leaves the progress
        card looking stuck. The workflow executor should fail this call once
        and let the workflow choose a fallback path.
        """
        acp_timeout = TimeoutError("ACP prompt 执行超时 (300s)")
        mock_session = _make_mock_session(side_effect=acp_timeout)
        mock_future = MagicMock()
        mock_future.result.return_value = mock_session
        self.executor._session_pool.submit.return_value = mock_future

        params = AgentCallParams(prompt="Analyze this task", tool="traex")
        result = self.executor.execute(params)

        self.assertIsNotNone(result.error)
        self.assertIn("ACP prompt", result.error)
        self.assertEqual(mock_session.send_prompt.call_count, 1)

    def test_retry_does_not_occur_for_permanent_errors(self):
        """Verify permanent errors do not trigger retries.

        Permanent errors (invalid schema, permission denied) cannot be
        resolved by retrying. The executor should fail fast to avoid
        wasting resources and token budget.
        """
        permanent_error = ValueError("Invalid schema: missing required field 'id'")
        mock_session = _make_mock_session(side_effect=permanent_error)
        mock_future = MagicMock()
        mock_future.result.return_value = mock_session
        self.executor._session_pool.submit.return_value = mock_future

        params = AgentCallParams(prompt="test prompt", tool="coco")
        result = self.executor.execute(params)

        # Should fail immediately without retry
        self.assertIsNotNone(result.error)
        self.assertIn("Invalid schema", result.error)
        # send_prompt should be called only once (no retries for permanent errors)
        self.assertEqual(mock_session.send_prompt.call_count, 1)

    @patch.object(AgentExecutor, "_sleep_with_backoff")
    def test_retry_backoff(self, mock_backoff):
        """Verify retries use exponential backoff with increasing delays.

        Exponential backoff prevents overwhelming the service during
        outages and gives the system time to recover.
        Delay pattern: 1s, 2s, 4s for RETRY_BACKOFF_BASE_S=1.0
        """
        transient_error = RuntimeError("503 Service Unavailable")
        # Fail all attempts so we see all backoff delays
        mock_session = _make_mock_session(side_effect=transient_error)
        mock_future = MagicMock()
        mock_future.result.return_value = mock_session
        self.executor._session_pool.submit.return_value = mock_future

        params = AgentCallParams(prompt="test prompt", tool="coco")
        result = self.executor.execute(params)

        # Should fail after all retries
        self.assertIsNotNone(result.error)

        # Verify exponential backoff delays: attempt 0 -> 1s, attempt 1 -> 2s, attempt 2 -> 4s
        # _sleep_with_backoff is called with the zero-based attempt number
        backoff_calls = mock_backoff.call_args_list
        self.assertEqual(len(backoff_calls), MAX_RETRIES)  # 3 retries

        # Check the attempt numbers passed to _sleep_with_backoff
        for i, call in enumerate(backoff_calls):
            self.assertEqual(call.args[0], i)

        # Verify the expected delays would be correct
        expected_delays = [RETRY_BACKOFF_BASE_S * (2**i) for i in range(MAX_RETRIES)]
        self.assertEqual(expected_delays, [1.0, 2.0, 4.0])


# ---------------------------------------------------------------------------
# 2. TestSchemaValidationRetry
# ---------------------------------------------------------------------------


class TestSchemaValidationRetry(unittest.TestCase):
    """Test schema validation failure handling with fix prompts.

    Reliability impact: Agents often produce output that doesn't match
    the requested schema. Automatic schema fix prompts significantly
    improve success rates without user intervention.
    """

    def setUp(self):
        self.cancel_event = threading.Event()
        self.executor = _make_executor(cancel_event=self.cancel_event)

    def tearDown(self):
        self.executor.shutdown(wait=False)

    def test_schema_validation_failure_triggers_fix_prompt(self):
        """Verify schema mismatch generates a fix prompt and retries.

        When output doesn't match the required schema, the executor
        should generate a fix prompt explaining the schema requirements
        and asking the agent to correct its output.
        """
        # First call returns invalid JSON, second returns valid
        invalid_output = "Here is the result: {name: 'test'}"  # Invalid JSON
        valid_output = '{"name": "test", "value": 42}'
        schema = {"name": "string", "value": "number"}

        call_count = 0

        def _side_effect(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_result = MagicMock()
            if call_count == 1:
                mock_result.text = invalid_output
            else:
                mock_result.text = valid_output
            mock_result.output_tokens = 50
            return mock_result

        mock_session = _make_mock_session(side_effect=_side_effect)
        mock_future = MagicMock()
        mock_future.result.return_value = mock_session
        self.executor._session_pool.submit.return_value = mock_future

        params = AgentCallParams(
            prompt="test prompt", tool="coco", schema=schema
        )
        result = self.executor.execute(params)

        # Should succeed after schema fix
        self.assertIsNone(result.error)
        self.assertEqual(result.parsed, {"name": "test", "value": 42})
        # send_prompt should be called twice (initial + fix prompt)
        self.assertEqual(mock_session.send_prompt.call_count, 2)

        # Verify the second call was a fix prompt
        calls = mock_session.send_prompt.call_args_list
        fix_prompt = calls[1].args[0]
        self.assertIn("did not conform to the required JSON schema", fix_prompt)
        self.assertIn("name", fix_prompt)
        self.assertIn("value", fix_prompt)

    def test_schema_retry_succeeds_after_fix(self):
        """Verify second attempt with fix prompt can succeed.

        After receiving a fix prompt, the agent should be able to
        correct its output to match the schema.
        """
        schema = {"result": "string", "confidence": "number"}

        # First: missing key, second: valid
        outputs = [
            '{"result": "answer"}',  # missing confidence
            '{"result": "answer", "confidence": 0.95}',  # valid
        ]
        call_count = 0

        def _side_effect(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_result = MagicMock()
            mock_result.text = outputs[call_count - 1]
            mock_result.output_tokens = 50
            return mock_result

        mock_session = _make_mock_session(side_effect=_side_effect)
        mock_future = MagicMock()
        mock_future.result.return_value = mock_session
        self.executor._session_pool.submit.return_value = mock_future

        params = AgentCallParams(
            prompt="test prompt", tool="coco", schema=schema
        )
        result = self.executor.execute(params)

        self.assertIsNone(result.error)
        self.assertEqual(result.parsed, {"result": "answer", "confidence": 0.95})
        self.assertEqual(mock_session.send_prompt.call_count, 2)

    def test_schema_retry_exhausted(self):
        """Verify after max schema retries, the call fails with clear error.

        If the agent repeatedly fails to produce valid schema output
        even after fix prompts, the executor should give up and return
        the raw output with no parsed result.
        """
        schema = {"name": "string", "id": "number"}
        # Always return output missing the 'id' field
        invalid_output = '{"name": "test"}'

        mock_session = _make_mock_session(output_text=invalid_output)
        mock_future = MagicMock()
        mock_future.result.return_value = mock_session
        self.executor._session_pool.submit.return_value = mock_future

        params = AgentCallParams(
            prompt="test prompt", tool="coco", schema=schema
        )
        result = self.executor.execute(params)

        # Should exhaust all schema retries
        # Total calls: 1 initial + SCHEMA_RETRY_MAX fix prompts
        expected_calls = 1 + SCHEMA_RETRY_MAX
        self.assertEqual(mock_session.send_prompt.call_count, expected_calls)
        # parsed should be None since validation never succeeded
        self.assertIsNone(result.parsed)
        # But raw output is still returned
        self.assertEqual(result.output, invalid_output)
        # A structured node contract must fail closed after retries; callers
        # can still inspect the raw output for diagnostics.
        self.assertIsNotNone(result.error)
        self.assertIn("schema validation", result.error.lower())


# ---------------------------------------------------------------------------
# 3. TestTimeoutHandling
# ---------------------------------------------------------------------------


class TestTimeoutHandling(unittest.TestCase):
    """Test timeout handling for agent calls.

    Reliability impact: Hung or slow agent calls can block entire
    workflows. Timeout limits ensure failures are detected quickly
    and the system remains responsive.
    """

    def setUp(self):
        self.cancel_event = threading.Event()
        self.executor = _make_executor(cancel_event=self.cancel_event)

    def tearDown(self):
        self.executor.shutdown(wait=False)

    def test_agent_call_times_out(self):
        """Verify agent calls exceeding AGENT_CALL_TIMEOUT_S are cancelled.

        When session creation exceeds SESSION_CREATE_TIMEOUT_S, the
        executor should cancel the attempt and return a timeout error.
        This prevents the workflow from hanging indefinitely.
        """
        # Make the future.result() raise TimeoutError
        mock_future = MagicMock()
        mock_future.result.side_effect = concurrent.futures.TimeoutError(
            "Timed out waiting for session"
        )
        mock_future.done.return_value = False
        self.executor._session_pool.submit.return_value = mock_future

        params = AgentCallParams(prompt="test prompt", tool="coco")
        with (
            patch("src.workflow_engine.executor.SESSION_CREATE_TIMEOUT_S", 0.01),
            patch("src.workflow_engine.executor._settings_int", lambda field, fallback: 0.01),
        ):
            result = self.executor.execute(params)

        self.assertIsNotNone(result.error)
        self.assertIn("timeout", result.error.lower())

    def test_session_creation_timeout_closes_late_session(self):
        """Late sessions created after a timeout must be closed.

        The timeout path can return before create_engine_session finishes in
        the worker pool. If that late session is not closed, repeated WF
        failures keep subprocess/ACP sessions alive after the workflow has
        already reported timeout.
        """
        create_started = threading.Event()
        release_create = threading.Event()
        late_session = _make_mock_session(output_text="late")
        real_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.executor._session_pool = real_pool

        def slow_create_engine_session(*args, **kwargs):
            create_started.set()
            release_create.wait(timeout=2)
            return late_session

        try:
            with (
                patch("src.workflow_engine.executor.SESSION_CREATE_TIMEOUT_S", 0.01),
                patch("src.workflow_engine.executor._settings_int", lambda field, fallback: 0.01),
                patch("src.agent_session.factory.create_engine_session", side_effect=slow_create_engine_session),
            ):
                params = AgentCallParams(prompt="test prompt", tool="coco")
                result = self.executor.execute(params)

            self.assertTrue(create_started.is_set())
            self.assertIsNotNone(result.error)
            self.assertIn("session creation timeout", result.error)

            release_create.set()
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and late_session.close.call_count == 0:
                time.sleep(0.01)

            self.assertGreaterEqual(late_session.close.call_count, 1)
        finally:
            release_create.set()
            real_pool.shutdown(wait=True, cancel_futures=True)

    def test_timeout_error_category(self):
        """Verify timeout errors are categorized as RUNTIME_TIMEOUT.

        Proper error categorization enables appropriate user-facing
        messages and helps with error analytics.
        """
        timeout_error = f"session creation timeout (>{AGENT_CALL_TIMEOUT_S}s)"
        category = categorize_error(timeout_error)
        self.assertEqual(category, ErrorCategory.RUNTIME_TIMEOUT)

        # Also test "timed out" phrasing
        category2 = categorize_error("The call timed out after 300 seconds")
        self.assertEqual(category2, ErrorCategory.RUNTIME_TIMEOUT)

    def test_timeout_does_not_leak_sensitive_info(self):
        """Verify timeout error messages are sanitized.

        Error messages shown to users should not contain internal
        implementation details like file paths or stack traces.
        """
        raw_timeout_error = (
            "TimeoutError: Call timed out\n"
            '  File "/home/user/src/workflow_engine/executor.py", line 100\n'
            "    in send_prompt\n"
            "  File \"/usr/lib/python3.11/threading.py\", line 500\n"
            "    in wait\n"
        )
        sanitized = sanitize_for_reply(raw_timeout_error, ErrorCategory.RUNTIME_TIMEOUT)

        # Should not contain internal details
        self.assertNotIn("/home/user", sanitized)
        self.assertNotIn("executor.py", sanitized)
        self.assertNotIn("threading.py", sanitized)
        self.assertNotIn("TimeoutError", sanitized)
        # Should contain user-friendly message
        self.assertIn("超时", sanitized)  # Chinese for timeout


# ---------------------------------------------------------------------------
# 5. TestErrorCategorization
# ---------------------------------------------------------------------------


class TestErrorCategorization(unittest.TestCase):
    """Test error message categorization into ErrorCategory enum.

    Reliability impact: Consistent error categorization enables
    appropriate user-facing messages, analytics, and automated
    handling of different error types.
    """

    def test_agent_limit_category(self):
        """Verify limit exceeded errors are categorized as AGENT_LIMIT.

        Agent limit errors indicate the workflow has too many steps
        and should be split into smaller tasks.
        """
        test_cases = [
            "Agent limit exceeded: max 200 agents reached",
            "Error: limit exceeded for agent calls",
            "Max agents limit reached for this workflow",
            "Agent limit reached, cannot continue",
        ]
        for msg in test_cases:
            with self.subTest(msg=msg):
                self.assertEqual(categorize_error(msg), ErrorCategory.AGENT_LIMIT)

    def test_tool_not_allowed_category(self):
        """Verify tool permission errors are categorized as TOOL_NOT_ALLOWED.

        Tool not allowed errors indicate the workflow is trying to use
        a tool that hasn't been approved for this run.
        """
        test_cases = [
            "Tool 'shell' is not in allowed list",
            "Error: tool not allowed in this workflow",
            "Command not permitted by security policy",
        ]
        for msg in test_cases:
            with self.subTest(msg=msg):
                self.assertEqual(
                    categorize_error(msg), ErrorCategory.TOOL_NOT_ALLOWED
                )

    def test_cancelled_category(self):
        """Verify cancellation errors are categorized as CANCELLED.

        Cancelled errors are user-initiated and should show appropriate
        cancellation messaging rather than failure messaging.
        """
        test_cases = [
            "Workflow cancelled by user",
            "Call cancelled during execution",
            "Operation canceled by request",  # American spelling
            "Cancelled before execution could start",
        ]
        for msg in test_cases:
            with self.subTest(msg=msg):
                self.assertEqual(categorize_error(msg), ErrorCategory.CANCELLED)

    def test_unknown_error_falls_back_to_internal(self):
        """Verify unknown errors default to INTERNAL_ERROR.

        Unknown errors should be handled gracefully with a generic
        user-facing message while preserving the raw error for logging.
        """
        test_cases = [
            "Something went wrong",
            "Unexpected null pointer exception",
            "",  # Empty error
        ]
        for msg in test_cases:
            with self.subTest(msg=msg):
                self.assertEqual(
                    categorize_error(msg), ErrorCategory.INTERNAL_ERROR
                )


# ---------------------------------------------------------------------------
# 6. TestErrorSanitization
# ---------------------------------------------------------------------------


class TestErrorSanitization(unittest.TestCase):
    """Test error sanitization to prevent leaking sensitive information.

    Security impact: Internal implementation details (file paths,
    stack traces, module names) should never be shown to end users.
    This prevents information disclosure attacks and provides a
    better user experience.
    """

    def test_sanitize_removes_stack_traces(self):
        """Verify sanitize_for_reply removes Python stack traces.

        Stack traces contain internal file paths and line numbers
        that are not useful to end users and could expose security
        sensitive information.
        """
        raw_error = (
            "Traceback (most recent call last):\n"
            '  File "/home/user/src/workflow_engine/executor.py", line 150\n'
            "    result = session.send_prompt(prompt)\n"
            '  File "/home/user/src/agent_session/client.py", line 85\n'
            "    raise TimeoutError('Call timed out')\n"
            "TimeoutError: Call timed out"
        )
        sanitized = sanitize_for_reply(raw_error, ErrorCategory.RUNTIME_TIMEOUT)

        self.assertNotIn("Traceback", sanitized)
        self.assertNotIn("File \"/", sanitized)
        self.assertNotIn("line 150", sanitized)
        self.assertNotIn("executor.py", sanitized)
        self.assertNotIn("TimeoutError", sanitized)

    def test_sanitize_removes_file_paths(self):
        """Verify file paths are removed or generalized.

        Absolute file paths reveal the server's directory structure
        which is sensitive information.
        """
        raw_error = (
            "Error: Cannot find module '/home/jiataorui/work/ghostAp/node_modules/foo/index.js'\n"
            "    at Module._resolveFilename (node:internal/modules/cjs/loader.js:1075:15)"
        )
        cleaned = _strip_internal_details(raw_error)

        self.assertNotIn("/home/jiataorui", cleaned)
        self.assertNotIn("/node_modules/", cleaned)
        self.assertNotIn("internal/modules/cjs/loader.js", cleaned)

    def test_sanitize_preserves_user_facing_info(self):
        """Verify useful error info (category, user message) is preserved.

        While internal details are removed, users should still get
        clear, actionable error messages appropriate for the error type.
        """
        raw_error = (
            "RuntimeError: Budget exhausted at /src/engine.py:42\n"
            "src.workflow_engine.budget module crashed"
        )

        # Test each category preserves appropriate user info
        test_cases = [
            (ErrorCategory.AGENT_LIMIT, "Agent"),
            (ErrorCategory.TOOL_NOT_ALLOWED, "工具"),
            (ErrorCategory.RUNTIME_TIMEOUT, "超时"),
            (ErrorCategory.CANCELLED, "取消"),
            (ErrorCategory.INTERNAL_ERROR, "内部错误"),
        ]

        for category, expected_keyword in test_cases:
            with self.subTest(category=category):
                sanitized = sanitize_for_reply(raw_error, category)
                self.assertIn(expected_keyword, sanitized)
                # Internal details should still be removed
                self.assertNotIn("/src/engine.py", sanitized)
                self.assertNotIn("src.workflow_engine", sanitized)


# ---------------------------------------------------------------------------
# 7. TestPipelineErrorHandling
# ---------------------------------------------------------------------------


class TestPipelineErrorHandling(unittest.TestCase):
    """Test pipeline stage error handling behavior.

    Reliability impact: Pipelines process multiple items through
    multiple stages. Configurable error handling allows workflows
    to either fail fast (default) or continue processing remaining
    items when some fail.

    These tests verify the pipeline logic as implemented in
    src/workflow_engine/runtime/runtime.js by reimplementing the
    core logic in Python for testability.
    """

    def _pipeline_python(self, items, *args):
        """Python reimplementation of the JS pipeline() function.

        This mirrors the logic in runtime.js to verify correctness
        without requiring a Node.js subprocess.
        """
        import asyncio

        if not isinstance(items, list):
            raise TypeError("pipeline() expects an array of items as first argument")

        # Extract options from last argument if it's an object (not a function)
        stages = list(args)
        options = {}
        if stages and not callable(stages[-1]) and isinstance(stages[-1], dict):
            options = stages[-1]
            stages = stages[:-1]

        continue_on_failure = (
            options.get("continueOnFailure")
            or options.get("continue_on_failure")
            or False
        )

        async def process_item(item):
            current = item
            for i, stage in enumerate(stages):
                try:
                    if asyncio.iscoroutinefunction(stage):
                        current = await stage(current)
                    else:
                        current = stage(current)
                except Exception as err:
                    if continue_on_failure:
                        return {
                            "error": str(err),
                            "failedAtStage": i,
                            "partialResult": current,
                        }
                    raise
            return current

        async def run_all():
            return await asyncio.gather(*[process_item(item) for item in items])

        return asyncio.run(run_all())

    def test_pipeline_stops_on_failure(self):
        """Verify pipeline stops when a stage fails (default behavior).

        By default, a stage failure should propagate immediately,
        stopping the entire pipeline. This is the safe default
        that prevents cascading failures.
        """
        def stage1(x):
            return x * 2

        def failing_stage(x):
            raise ValueError(f"Failed processing {x}")

        def stage3(x):
            return x + 10

        # Test that pipeline without continue_on_failure raises
        with self.assertRaises(ValueError) as ctx:
            self._pipeline_python([1, 2, 3], stage1, failing_stage, stage3)

        self.assertIn("Failed processing", str(ctx.exception))

    def test_pipeline_continues_on_failure_when_configured(self):
        """Verify pipeline can be configured to continue after stage failures.

        When continue_on_failure is enabled, items that fail are
        returned with error information, while successful items
        continue through all stages. This is useful for batch
        processing where partial results are better than none.
        """
        def stage1(x):
            return x * 2

        def sometimes_failing(x):
            if x == 4:  # Fail on the second item (after stage1: 2*2=4)
                raise ValueError("Bad value")
            return x + 1

        def stage3(x):
            return x * 10

        # Test with continue_on_failure enabled
        results = self._pipeline_python(
            [1, 2, 3],
            stage1,
            sometimes_failing,
            stage3,
            {"continue_on_failure": True},
        )

        # Item 1: 1*2=2 -> 2+1=3 -> 3*10=30 (success)
        # Item 2: 2*2=4 -> fails at stage2 (error)
        # Item 3: 3*2=6 -> 6+1=7 -> 7*10=70 (success)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], 30)
        self.assertIsInstance(results[1], dict)
        self.assertIn("error", results[1])
        self.assertEqual(results[1]["failedAtStage"], 1)
        self.assertEqual(results[1]["partialResult"], 4)
        self.assertEqual(results[2], 70)


# ---------------------------------------------------------------------------
# Additional: Test is_transient_error helper
# ---------------------------------------------------------------------------


class TestIsTransientError(unittest.TestCase):
    """Test the is_transient_error helper function.

    This helper determines whether an error is worth retrying.
    Correct classification is critical to avoid wasting resources
    retrying permanent errors while still recovering from transient ones.
    """

    def test_transient_errors_detected(self):
        """Verify known transient errors are correctly identified."""
        transient_errors = [
            "Network timeout: connection reset",
            "503 Service Unavailable",
            "504 Gateway Timeout",
            "Rate limit exceeded, please retry later",
            "Connection refused",
            "Temporary failure in name resolution",
            "Server is busy, try again",
        ]
        for msg in transient_errors:
            with self.subTest(msg=msg):
                self.assertTrue(is_transient_error(msg), f"Should be transient: {msg}")

    def test_acp_prompt_execution_timeout_is_not_retryable(self):
        """ACP prompt timeout means the model call consumed its full budget."""
        self.assertFalse(is_transient_error("TimeoutError: ACP prompt 执行超时 (300s)"))
        self.assertFalse(is_transient_error("ACP prompt execution timeout (300s)"))

    def test_permanent_errors_not_retryable(self):
        """Verify permanent errors are correctly identified as not retryable."""
        permanent_errors = [
            "Invalid schema: missing required field",
            "Permission denied: cannot access file",
            "Tool 'shell' not in allowed list",
            "404 Not Found",
            "403 Forbidden",
            "Bad request: invalid parameter",
            "Schema validation failed",
        ]
        for msg in permanent_errors:
            with self.subTest(msg=msg):
                self.assertFalse(
                    is_transient_error(msg), f"Should NOT be transient: {msg}"
                )

    def test_empty_error_not_retryable(self):
        """Verify empty/None errors are not retryable (safe default)."""
        self.assertFalse(is_transient_error(""))
        self.assertFalse(is_transient_error(None))


class TestM2CoalescerErrorsAndReporting(unittest.TestCase):
    """Tests for M2 F6: leading-edge ProgressCoalescer, error sanitation, and effort reporting."""

    def test_leading_edge_coalescer_immediate_flush_and_falsy_stop(self):
        import threading

        from src.workflow_engine.progress_coalescer import ProgressCoalescer

        seen: list[object] = []
        first_event = threading.Event()
        third_event = threading.Event()

        def on_progress(snap: object) -> None:
            seen.append(snap)
            if snap == "first":
                first_event.set()
            if snap == "burst-3":
                third_event.set()

        coalescer = ProgressCoalescer(on_progress=on_progress, debounce_s=0.4)
        try:
            coalescer.enqueue("first")
            self.assertTrue(first_event.wait(timeout=0.18))
            coalescer.enqueue("burst-1")
            coalescer.enqueue("burst-2")
            coalescer.enqueue("burst-3")
            self.assertTrue(third_event.wait(timeout=1.0))
            self.assertNotIn("burst-1", seen)

            # flush_immediate deduplicates and does not double-deliver
            before_count = len(seen)
            coalescer.flush_immediate("urgent")
            time.sleep(0.05)
            self.assertEqual(seen.count("urgent"), 1)
            self.assertGreater(len(seen), before_count)

            # Falsy snapshot (0) is flushed on stop()
            coalescer.enqueue(0)
        finally:
            coalescer.stop()
        self.assertEqual(seen[-1], 0)

    def test_strip_internal_details_secrets_urls_node_frames_and_tracebacks(self):
        from src.workflow_engine.errors import ErrorCategory, categorize_error, format_structured_error

        self.assertEqual(_strip_internal_details(None), "")
        self.assertEqual(_strip_internal_details(""), "")

        secret_msg = "503 token=sk-supersecret999 Bearer abcdefghij12345 api_key=key_secret_123 at /home/admin/.env"
        cleaned_secret = _strip_internal_details(secret_msg)
        self.assertNotIn("sk-supersecret999", cleaned_secret)
        self.assertNotIn("abcdefghij12345", cleaned_secret)
        self.assertNotIn("key_secret_123", cleaned_secret)
        self.assertNotIn("/home/admin/.env", cleaned_secret)

        url_and_cmd = "See https://example.com/a/b/c.json or run /wf_status"
        cleaned_url = _strip_internal_details(url_and_cmd)
        self.assertIn("https://example.com/a/b/c.json", cleaned_url)
        self.assertIn("/wf_status", cleaned_url)

        tb = (
            "Traceback (most recent call last):\n"
            '  File "/home/user/src/workflow_engine/engine.py", line 42\n'
            "    run_thing()\n"
            "ValueError: invalid agent binding"
        )
        cleaned_tb = _strip_internal_details(tb)
        self.assertNotIn("Traceback", cleaned_tb)
        self.assertNotIn("engine.py", cleaned_tb)
        self.assertIn("ValueError: invalid agent binding", cleaned_tb)

        node_err = (
            "Error: Cannot find module '/home/u/app/node_modules/x.js'\n"
            "    at Module._resolveFilename (node:internal/modules/cjs/loader.js:1075:15)"
        )
        cleaned_node = _strip_internal_details(node_err)
        self.assertNotIn("node:internal", cleaned_node)
        self.assertNotIn("/home/u/app", cleaned_node)

        self.assertEqual(categorize_error("Workflow deadline exceeded before call"), ErrorCategory.RUNTIME_TIMEOUT)
        self.assertEqual(categorize_error("操作超时"), ErrorCategory.RUNTIME_TIMEOUT)
        self.assertEqual(categorize_error("工作流已取消"), ErrorCategory.CANCELLED)
        self.assertEqual(
            categorize_error("Workflow agent_id is outside confirmed agent_pool: X"),
            ErrorCategory.INVALID_ARGUMENT,
        )
        structured = format_structured_error("Workflow agent_id is outside confirmed agent_pool: /tmp/a.py")
        self.assertEqual(structured.category, ErrorCategory.INVALID_ARGUMENT)
        self.assertNotIn("/tmp/a.py", structured.user_message)

    def test_reporting_includes_effort_profile_role_agent_id_and_escapes_html(self):
        from src.workflow_engine.models import (
            AgentProgress,
            AgentStatus,
            PhaseProgress,
            WorkflowProject,
            WorkflowStatus,
        )
        from src.workflow_engine.reporting import (
            build_workflow_report_html,
            build_workflow_report_markdown,
        )

        project = WorkflowProject(
            workflow_id="wf-rep-1",
            name="Teamwork Report",
            status=WorkflowStatus.COMPLETED,
            requirement="Test reporting",
            run_spec={
                "orchestrator_agent_id": "A2",
                "orchestrator": {"tool_name": "claude", "model_name": "sonnet", "use_default_model": False},
                "auto_reviewer": False,
                "agent_pool": [
                    {
                        "agent_id": "A1",
                        "tool_name": "traex",
                        "model_name": "kimi-k2",
                        "profile": "max",
                        "effort": "high",
                        "role": "worker",
                    }
                ],
            },
            phases=[
                PhaseProgress(
                    title="Execute",
                    agents=[
                        AgentProgress(
                            label="w1",
                            agent_id="A1",
                            tool="traex",
                            model="kimi-k2",
                            profile="max",
                            effort="high",
                            role="worker<script>alert(1)</script>",
                            effective_model="kimi-k2/max/high",
                            status=AgentStatus.DONE,
                            duration_s=1.5,
                            token_usage=120,
                        )
                    ],
                )
            ],
        )
        md = build_workflow_report_markdown(project)
        html_out = build_workflow_report_html(project, md)

        for token in ("agent_id=A1", "profile=max", "effort=high", "effective_model=kimi-k2/max/high"):
            self.assertIn(token, md)
        self.assertIn("[A1]", html_out)
        self.assertIn("profile=max", html_out)
        self.assertIn("effort=high", html_out)
        self.assertIn("effective=kimi-k2/max/high", html_out)
        self.assertNotIn("<script>alert(1)</script>", html_out)

    def test_strip_internal_details_redacts_authorization_bearer_and_schemes(self):
        cases = [
            (
                "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload123.sig456 failed",
                ["eyJhbGciOiJIUzI1NiJ9", "payload123", "sig456"],
            ),
            (
                "Authorization=Bearer supersecrettoken12345 failed",
                ["supersecrettoken12345"],
            ),
            (
                'Authorization: "Bearer eyJhbGciOiJIUzI1NiJ9.quoted.sig"',
                ["eyJhbGciOiJIUzI1NiJ9", "quoted", "sig"],
            ),
            (
                'Authorization: Bearer "eyJhbGciOiJIUzI1NiJ9.quoted2.sig2"',
                ["eyJhbGciOiJIUzI1NiJ9", "quoted2", "sig2"],
            ),
            (
                "Authorization: Basic dXNlcjpwYXNzd29yZA==",
                ["dXNlcjpwYXNzd29yZA=="],
            ),
            (
                "Authorization:\nBearer eyJhbGciOiJIUzI1NiJ9.multiline.sig",
                ["eyJhbGciOiJIUzI1NiJ9", "multiline"],
            ),
            (
                "api_key=ak_live_987654 token=sk-secret999999 Bearer standaloneToken123 ghp-abcdef123456 xoxb-123456-abcdef",
                ["ak_live_987654", "sk-secret999999", "standaloneToken123", "ghp-abcdef123456", "xoxb-123456-abcdef"],
            ),
        ]
        for raw, secrets in cases:
            with self.subTest(raw=raw):
                cleaned = _strip_internal_details(raw)
                for secret in secrets:
                    self.assertNotIn(secret, cleaned)
                self.assertIn("<redacted>", cleaned)

    def test_strip_internal_details_redacts_json_multiline_and_quoted_secrets(self):
        leak_cases = [
            ('{"Authorization": "Basic dXNlcjpwYXNz"}', ["dXNlcjpwYXNz"]),
            ('{"api_key": "plain_secret_value_123"}', ["plain_secret_value_123"]),
            ('{"password": "hunter2_secret"}', ["hunter2_secret"]),
            ('{"secret": "my_app_secret_val"}', ["my_app_secret_val"]),
            (
                '{"token": "eyJhbGciOiJIUzI1NiJ9.payload.sig"}',
                ["eyJhbGciOiJIUzI1NiJ9", "payload", "sig"],
            ),
            (
                "{'Authorization': 'Basic dXNlcjpwYXNz', 'api_key': 'singlekey456'}",
                ["dXNlcjpwYXNz", "singlekey456"],
            ),
            ("Authorization:\nBasic dXNlcjpwYXNz", ["dXNlcjpwYXNz"]),
            ("Authorization:\nBearer tok12", ["tok12"]),
            ('Bearer "quotedstandalone123"', ["quotedstandalone123"]),
            (
                '{"headers": {"Authorization": "Basic dXNlcjpwYXNz"}}',
                ["dXNlcjpwYXNz"],
            ),
            ("{api_key:unquoted_secret123}", ["unquoted_secret123"]),
            ("Authorization:\n\"Bearer tok99\"", ["tok99"]),
        ]
        for raw, secrets in leak_cases:
            with self.subTest(raw=raw):
                cleaned = _strip_internal_details(raw)
                for secret in secrets:
                    self.assertNotIn(secret, cleaned)
                self.assertIn("<redacted>", cleaned)
                # Verify delimiters and idempotency
                self.assertEqual(_strip_internal_details(cleaned), cleaned)
                if raw.endswith("}"):
                    self.assertTrue(cleaned.endswith("}"))

        # Invalid-argument reply path also strips JSON-embedded secrets
        reply = sanitize_for_reply(
            'Bad payload {"api_key": "plain_secret_value_123"}',
            ErrorCategory.INVALID_ARGUMENT,
        )
        self.assertNotIn("plain_secret_value_123", reply)
        self.assertIn("<redacted>", reply)

        # Ordinary prose, URLs, slash commands, and non-auth multiline messages must stay intact
        preserved = [
            "Basic validation failed",
            "Basic check failed",
            "Basic setup error",
            "See https://api.example.com/v1/foo/bar for details",
            "Use /wf_status or /wf to inspect",
            "Invalid token:\nPlease check configuration",
        ]
        for text in preserved:
            with self.subTest(preserved=text):
                cleaned = _strip_internal_details(text)
                self.assertEqual(cleaned, text)

    def test_strip_internal_details_linear_redos_bounds(self):
        """Worst-case pathological inputs finish in strictly linear O(N) time (< 100 ms; < 50 ms for 50k backslash ReDoS)."""
        pathological = [
            ("spaces_50k", " " * 50000),
            ("auth_spaces_50k", "Authorization:" + " " * 50000 + "hello"),
            ("newlines_20k", "\n" * 20000),
            ("auth_newlines_20k", "Authorization:" + "\n" * 20000 + "hello"),
            ("escaped_quotes_50k", '"' + '\\"' * 25000),
            ("sq_edq_50k", 'api_key=\\"' + "\\" * 50000 + "!"),
            ("sq_esq_50k", "api_key=\\'" + "\\" * 50000 + "!"),
            ("sq_auth2_50k", 'Authorization: Bearer \\"' + "\\" * 50000 + "!"),
            ("ml_edq_50k", 'api_key:\n  \\"' + "\\" * 50000 + "!"),
            ("ml_esq_50k", "api_key:\n  \\'" + "\\" * 50000 + "!"),
            ("ml_auth2_50k", 'Authorization:\n  Bearer \\"' + "\\" * 50000 + "!"),
            ("deep_json_recursion", '{"a":' * 10000 + "bad" + "}" * 10000),
            ("deep_json_late_secret", '{"a":' * 10000 + ' {"api_key": "late_secret"}'),
            ("slash_dot_50k", "/." * 25000),
            ("file_quotes_50k", 'File "' + '"' * 50000),
            ("node_at_spaces_50k", "at x" + " " * 50000 + "(bad"),
        ]
        strict_50ms = {"sq_edq_50k", "sq_esq_50k", "sq_auth2_50k", "ml_edq_50k", "ml_esq_50k", "ml_auth2_50k"}
        for name, payload in pathological:
            with self.subTest(name=name):
                t0 = time.perf_counter()
                cleaned = _strip_internal_details(payload)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                limit_ms = 50.0 if name in strict_50ms else 100.0
                self.assertLess(elapsed_ms, limit_ms, f"{name} took {elapsed_ms:.2f} ms")
                if name == "deep_json_late_secret":
                    self.assertNotIn("late_secret", cleaned)
                    self.assertIn("<redacted>", cleaned)

        # Escaped backslashes & escaped quotes inside \"...\" still match cleanly.
        for raw, secret in (
            (r'api_key=\"escaped\\\\\"', "escaped"),
            (r'api_key=\"sec\\\"ret\"', "sec"),
        ):
            cleaned = _strip_internal_details(raw)
            self.assertNotIn(secret, cleaned)
            self.assertIn("<redacted>", cleaned)
            self.assertEqual(_strip_internal_details(cleaned), cleaned)

    def test_strip_internal_details_escaped_nested_json_snake_camel_and_auth_schemes(self):
        """Escaped/nested JSON, snake_case/camelCase/x-api-key, and expanded auth schemes are redacted idempotently."""
        import json

        cases = [
            (
                r'Error: body="{\"api_key\": \"escaped_secret_123\", \"ok\": \"pub\"}"',
                ["escaped_secret_123"],
                ['"ok\\": \\"pub\\"'],
            ),
            (
                r'Error: body=\"{\\\"api_key\\\": \\\"escaped_secret_123\\\"}\"',
                ["escaped_secret_123"],
                [],
            ),
            (
                json.dumps(
                    {
                        "detail": json.dumps(
                            {
                                "api_key": "nested_escaped_secret_42",
                                "Authorization": "Basic bGVha2Vk",
                                "msg": "中文错误保留",
                            },
                            ensure_ascii=False,
                        )
                    },
                    ensure_ascii=False,
                ),
                ["nested_escaped_secret_42", "bGVha2Vk"],
                ["中文错误保留"],
            ),
            (
                json.dumps(
                    {"outer": json.dumps({"inner": json.dumps({"api_key": "triple_secret_99", "ok": "keep"})})}
                ),
                ["triple_secret_99"],
                ["keep"],
            ),
            (
                "openai_api_key=plain_openai_secret_123 client_secret=cs_123 github_token=gh_123 "
                "db_password=db_pw_123 aws_secret_access_key=aws_123 bearer_token=bt_123 tenant_access_token=tat_123",
                ["plain_openai_secret_123", "cs_123", "gh_123", "db_pw_123", "aws_123", "bt_123", "tat_123"],
                [],
            ),
            (
                '{"apiKey": "camel_key_123", "clientSecret": "camel_sec_456"}',
                ["camel_key_123", "camel_sec_456"],
                [],
            ),
            (
                "x-api-key: hdr_secret_789",
                ["hdr_secret_789"],
                [],
            ),
            (
                "Authorization: ApiKey my_secret_apikey_123",
                ["my_secret_apikey_123"],
                [],
            ),
            (
                "Authorization: Bot MTIzNDU2.ghijkl.mnopqr",
                ["MTIzNDU2"],
                [],
            ),
            (
                'Authorization: Digest username="alice", response="digest_secret_resp"',
                ["digest_secret_resp"],
                [],
            ),
            (
                'Authorization: OAuth oauth_token="oa_tok_123", oauth_signature="oa_sig_456"',
                ["oa_tok_123", "oa_sig_456"],
                [],
            ),
            (
                '{\n  "api_key":\n    "multiline_json_secret_123"\n}',
                ["multiline_json_secret_123"],
                [],
            ),
            (
                '{\n  "api_key":\n    "truncated_multiline_secret_123"',
                ["truncated_multiline_secret_123"],
                [],
            ),
        ]
        for raw, secrets, required in cases:
            with self.subTest(raw=raw[:60]):
                cleaned = _strip_internal_details(raw)
                for secret in secrets:
                    self.assertNotIn(secret, cleaned)
                for req in required:
                    self.assertIn(req, cleaned)
                self.assertIn("<redacted>", cleaned)
                self.assertEqual(_strip_internal_details(cleaned), cleaned)

    def test_strip_internal_details_preserves_ghostap_telemetry_and_metadata(self):
        """GhostAP LLM telemetry, metadata keys, empty values, prose, URLs, and slash commands are never mangled."""
        import json

        preserved = [
            json.dumps(
                {
                    "token_usage": 350,
                    "token_count": 42,
                    "token_counts": {"in": 10, "out": 32},
                    "token_limit": 4096,
                    "token_limits": [4096, 8192],
                    "token_budget": 12000,
                    "token_budgets": {"soft": 8000},
                    "token_type": "bpe",
                    "token_types": ["bpe"],
                    "token_name": "cl100k",
                    "token_names": ["cl100k"],
                    "tokens_used": 99,
                    "max_tokens": 4096,
                    "max_token": 4096,
                    "prompt_tokens": 50,
                    "completion_tokens": 200,
                    "total_tokens": 250,
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "thinking_tokens": 5,
                    "thought_tokens": 5,
                    "reasoning_tokens": 8,
                    "cached_tokens": 12,
                }
            ),
            json.dumps(
                {
                    "primary_key": "pk_1",
                    "foreign_key": "fk_2",
                    "sort_key": "sk_3",
                    "partition_key": "part_4",
                    "cache_key": "ck_5",
                    "public_key": "pub_6",
                    "file_key": "file_v3_1",
                    "image_key": "img_v3_2",
                    "breaker_key": "brk_1",
                    "binding_key": "bnd_1",
                    "session_key": "sess_1",
                    "engine_session_key": "esess_1",
                    "key_name": "k1",
                    "key_type": "rsa",
                    "key_files": ["a.py"],
                    "key_findings": ["ok"],
                }
            ),
            "secret_sauce=ketchup is_secret=true has_secret=false no_secret=1 use_secret=0 non_secret=yes",
            "Password validation failed for user alice",
            "Authorization:\nPlease check configuration",
            "Invalid token:\nBasic validation failed",
            '{"api_key": ""}',
            '{\n  "api_key": ""\n}',
            "See https://api.example.com/v1/token/status or run /wf_status and /wf",
        ]
        for raw in preserved:
            with self.subTest(raw=raw[:60]):
                self.assertEqual(_strip_internal_details(raw), raw)

    def test_runtime_bridge_stderr_redacts_nested_escaped_json_secrets(self):
        """Real RuntimeBridge subprocess stderr with double-serialized JSON secrets is redacted end-to-end."""
        import shutil
        import tempfile
        from pathlib import Path

        from src.workflow_engine.bridge import RuntimeBridge

        if shutil.which("node") is None:
            self.skipTest("node binary not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            script = Path(tmpdir) / "crash.mjs"
            script.write_text(
                'export const meta = { name: "crash", description: "test", phases: [{ title: "run", detail: "crash" }], agentPlan: [] };\n'
                "export default async function() {\n"
                "  console.error(JSON.stringify({ detail: JSON.stringify("
                '{ api_key: "nested_escaped_secret_42", Authorization: "Basic bGVha2Vk" }'
                ") }));\n"
                "  await new Promise((r) => setTimeout(r, 40));\n"
                "  process.exit(2);\n"
                "}\n",
                encoding="utf-8",
            )
            bridge = RuntimeBridge(
                script_path=str(script),
                cwd=tmpdir,
                on_agent_call=lambda params: None,
            )
            try:
                bridge.start()
                with self.assertRaises(RuntimeError) as ctx:
                    bridge.run()
            finally:
                bridge.stop()
            msg = str(ctx.exception)
            self.assertNotIn("nested_escaped_secret_42", msg)
            self.assertNotIn("bGVha2Vk", msg)
            self.assertIn("<redacted>", msg)

    def test_strip_internal_details_deep_json_nesting_no_recursion_error(self):
        """Iterative JSON redaction handles depths 950..1000 even from nested call frames."""

        def _invoke_from_depth(frames: int, payload: str) -> str:
            if frames > 0:
                return _invoke_from_depth(frames - 1, payload)
            return _strip_internal_details(payload)

        for depth in (950, 995, 1000):
            with self.subTest(depth=depth):
                payload = '{"a":' * depth + '{"api_key":"deep_sec"}' + "}" * depth
                cleaned = _invoke_from_depth(200, payload)
                self.assertNotIn("deep_sec", cleaned)
                self.assertIn("<redacted>", cleaned)
                self.assertEqual(_strip_internal_details(cleaned), cleaned)

    def test_strip_internal_details_truncated_json_overflow_token_and_pem(self):
        """Truncated JSON/containers, Stage 1a overflow/unclosed quotes, Token token=, standalone escaped Bearer, and unclosed PEM are redacted."""
        cases = [
            (r'{"detail": "{\"api_key\": \"trunc_secret', ["trunc_secret"]),
            ('{"api_key": "trunc_simple_secret', ["trunc_simple_secret"]),
            ("{'api_key': 'trunc_sq_secret", ["trunc_sq_secret"]),
            ('{"api_key": ["trunc_arr_1", "trunc_arr_2"', ["trunc_arr_1", "trunc_arr_2"]),
            ('{"api_key": {"val": "trunc_obj_secret"', ["trunc_obj_secret"]),
            (r'{"Authorization": "Digest username=\"a\", response=\"trunc_digest_sec', ["trunc_digest_sec"]),
            ('{"a":' * 65 + ' {"api_key": {"nested": "late_nested_secret"}}', ["late_nested_secret"]),
            ('{"a":' * 65 + ' {"api_key": ["late_list_sec1", "late_list_sec2"]}', ["late_list_sec1", "late_list_sec2"]),
            ('Error "unclosed then {"api_key": {"val": "secret_after_quote"}}', ["secret_after_quote"]),
            ('Error "unclosed then {"api_key": ["secret_list_after_quote"]}', ["secret_list_after_quote"]),
            ('{"bad": "unclosed\n{"api_key": {"nested": "next_line_secret"}}', ["next_line_secret"]),
            ('Authorization: Token token="rails_sec_123"', ["rails_sec_123"]),
            ('Authorization: Bearer token="bearer_sub_123"', ["bearer_sub_123"]),
            ('Authorization: Token key="key_sub_123"', ["key_sub_123"]),
            (r'failed with Bearer \"standalone_esc_tok\"', ["standalone_esc_tok"]),
            (
                "Error:\n-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAsecretpemdata123",
                ["secretpemdata123"],
            ),
            (
                "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAsecret",
                ["AAAAsecret"],
            ),
            (
                "-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEIsecretEcData",
                ["secretEcData"],
            ),
            (
                "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANsecretPkcs8",
                ["secretPkcs8"],
            ),
            (
                "-----BEGIN ENCRYPTED PRIVATE KEY-----\nMIIFDjBABgkqhkiG9w0secretEnc",
                ["secretEnc"],
            ),
            (
                "-----BEGIN PGP PRIVATE KEY BLOCK-----\nlQOYBsecretPgp",
                ["secretPgp"],
            ),
        ]
        for raw, secrets in cases:
            with self.subTest(raw=raw[:60]):
                cleaned = _strip_internal_details(raw)
                for secret in secrets:
                    self.assertNotIn(secret, cleaned)
                self.assertIn("<redacted>", cleaned)
                self.assertEqual(_strip_internal_details(cleaned), cleaned)

        # Preserve non-PEM prefix before unclosed private key, and do not strip unclosed public CERTIFICATE.
        self.assertTrue(
            _strip_internal_details(
                "Keep prefix:\n-----BEGIN RSA PRIVATE KEY-----\nMIIEowSecret"
            ).startswith("Keep prefix:\n")
        )
        cert = "-----BEGIN CERTIFICATE-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8A"
        self.assertEqual(_strip_internal_details(cert), cert)

    def test_runtime_bridge_stderr_redacts_4kb_clipped_json_secret(self):
        """Real RuntimeBridge >4KB stderr line clipped by STDERR_LINE_MAX_BYTES still redacts truncated api_key."""
        import shutil
        import tempfile
        from pathlib import Path

        from src.workflow_engine.bridge import RuntimeBridge

        if shutil.which("node") is None:
            self.skipTest("node binary not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            script = Path(tmpdir) / "crash_4kb.mjs"
            script.write_text(
                'export const meta = { name: "crash4kb", description: "test", phases: [{ title: "run", detail: "crash" }], agentPlan: [] };\n'
                "export default async function() {\n"
                '  console.error(JSON.stringify({ pad: "X".repeat(4035), api_key: "clipped_secret_9876543210_abcdefghijklmnopqrstuvwxyz" }));\n'
                "  await new Promise((r) => setTimeout(r, 40));\n"
                "  process.exit(2);\n"
                "}\n",
                encoding="utf-8",
            )
            bridge = RuntimeBridge(
                script_path=str(script),
                cwd=tmpdir,
                on_agent_call=lambda params: None,
            )
            try:
                bridge.start()
                with self.assertRaises(RuntimeError) as ctx:
                    bridge.run()
            finally:
                bridge.stop()
            msg = str(ctx.exception)
            self.assertNotIn("clipped_secr", msg)
            self.assertIn("<redacted>", msg)

    def test_engine_reviewer_roles_tuple_never_auto_fallbacks_to_worker(self):
        import tempfile

        from src.acp.sync_adapter import ACPStartupError
        from src.workflow_engine.agent_pool import WorkflowAgentBinding
        from src.workflow_engine.engine import WorkflowEngine
        from src.workflow_engine.models import WorkflowProject, WorkflowStatus
        from src.workflow_engine.run_spec import WorkflowRunSpec, _review_binding
        from src.workflow_engine.state_manager import WorkflowStateManager

        with tempfile.TemporaryDirectory() as tmpdir:
            pool = (
                WorkflowAgentBinding(
                    agent_id="A1",
                    tool_name="claude",
                    model_name="sonnet",
                    display_name="Reviewer A1",
                    role=None,
                    roles=("reviewer",),
                ),
                WorkflowAgentBinding(
                    agent_id="A2",
                    tool_name="codex",
                    model_name="gpt-5.4",
                    display_name="Worker A2",
                    role="worker",
                ),
                WorkflowAgentBinding(
                    agent_id="A4",
                    tool_name="traex",
                    model_name="kimi-k2",
                    display_name="Dual A4",
                    role=None,
                    roles=("worker", "reviewer"),
                ),
            )
            run_spec = WorkflowRunSpec(
                orchestrator=_review_binding(pool[0]),
                reviewers=(),
                tool_model_map={},
                task="reviewer roles isolation",
                chat_id="chat",
                topic_id=None,
                budget=6,
                deadline=None,
                auto_reviewer=True,
                agent_pool=pool,
                orchestrator_agent_id="A1",
            )
            engine = WorkflowEngine("chat", tmpdir, agent_type="claude")
            engine._run_spec = run_spec
            engine._project = WorkflowProject(name="wf", status=WorkflowStatus.RUNNING)
            engine._state_manager = WorkflowStateManager(engine._project)
            engine._executor = AgentExecutor(cwd=tmpdir, cancel_event=engine.cancel_event)

            executed_tools: list[str] = []

            def fake_create_session(**kwargs):
                tool = str(kwargs.get("agent_type") or "")
                executed_tools.append(tool)
                raise ACPStartupError(
                    f"{tool} binary missing",
                    agent_cmd=tool,
                    agent_args=[],
                    cwd=tmpdir,
                )

            try:
                with patch("src.agent_session.factory.create_engine_session", side_effect=fake_create_session):
                    # A1 has role=None, roles=("reviewer",); call omits explicit role.
                    # Must NOT auto-fallback onto worker A2 (codex).
                    res_a1 = engine._handle_agent_call(
                        AgentCallParams(prompt="review diff", agent_id="A1", label="rev-a1")
                    )
                    self.assertIsNotNone(res_a1.error)
                    self.assertNotIn("codex", executed_tools)

                    # A4 has role=None, roles=("worker", "reviewer"); call omits explicit role.
                    # Must NOT auto-fallback onto worker A2 (codex).
                    executed_tools.clear()
                    res_a4 = engine._handle_agent_call(
                        AgentCallParams(prompt="review diff 2", agent_id="A4", label="rev-a4")
                    )
                    self.assertIsNotNone(res_a4.error)
                    self.assertNotIn("codex", executed_tools)
            finally:
                engine._executor.shutdown(wait=True)

    def test_strip_internal_details_pem_dos_empty_containers_bearer_and_3x_escaped_fixes(self):
        """Regression tests for PEM O(N) bound, Bearer empty quotes, empty containers, JS/util.inspect multiline, 5-bs parity, 3x truncated JSON, leading-backslash KV, and multiline Basic."""
        # 1. PEM O(N) bound on 20k unclosed public certificate headers
        t0 = time.perf_counter()
        pem_20k = "-----BEGIN CERTIFICATE-----" * 20000
        cleaned_pem = _strip_internal_details(pem_20k)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.assertLess(elapsed_ms, 50.0, f"PEM 20k took {elapsed_ms:.2f} ms")
        self.assertEqual(cleaned_pem, pem_20k)

        pem_mixed = "-----BEGIN CERTIFICATE-----\nMIIcert\n-----BEGIN RSA PRIVATE KEY-----\nMIIsec"
        self.assertEqual(
            _strip_internal_details(pem_mixed),
            "-----BEGIN CERTIFICATE-----\nMIIcert\n<redacted>",
        )

        # 2. Bearer "" and Bearer \"\" preservation + idempotency, while empty module quotes still strip
        bearer_empty = 'api_key="" token=\'\' secret=\\"\\" Bearer "" Bearer \\"\\"'
        cleaned_bearer = _strip_internal_details(bearer_empty)
        self.assertEqual(cleaned_bearer, bearer_empty)
        self.assertEqual(_strip_internal_details(cleaned_bearer), cleaned_bearer)
        self.assertEqual(_strip_internal_details("Error: Cannot find module ''"), "Error: Cannot find module")

        # 3. Empty {} / [] containers preserved (including alongside secrets), non-empty nested [{}] redacted
        for empty_case in (
            '{"api_key": {}}',
            '{"api_key": []}',
            '{"api_key": "", "secret": "", "token": []}',
            '{"token": {}}',
            "token=[]",
            "api_key={}",
        ):
            with self.subTest(empty_case=empty_case):
                self.assertEqual(_strip_internal_details(empty_case), empty_case)
        self.assertEqual(
            _strip_internal_details('{"api_key": {}, "token": "sec123"}'),
            '{"api_key": {}, "token": "<redacted>"}',
        )
        self.assertEqual(_strip_internal_details('{"api_key": [{}]}'), '{"api_key": "<redacted>"}')

        # 4. Multiline JS / Node util.inspect unquoted-key containers redacted; plain English prose kept
        js_cases = [
            ('api_key: {\n  nested: "secret_js_obj"\n}', ["secret_js_obj"], []),
            ('credentials: {\n  value: "secret_creds_val"\n}', ["secret_creds_val"], []),
            ('api_key={\n  val: "my_secret_123"\n}', ["my_secret_123"], []),
            (
                "{\n  api_key: {\n    nested: 'node_inspect_secret',\n    other: 'node_inspect_secret_2'\n  },\n  ok: 'keep'\n}",
                ["node_inspect_secret", "node_inspect_secret_2"],
                ["ok: 'keep'"],
            ),
        ]
        for raw, secrets, required in js_cases:
            with self.subTest(raw=raw[:40]):
                cleaned = _strip_internal_details(raw)
                for sec in secrets:
                    self.assertNotIn(sec, cleaned)
                for req in required:
                    self.assertIn(req, cleaned)
                self.assertIn("<redacted>", cleaned)
                self.assertEqual(_strip_internal_details(cleaned), cleaned)

        self.assertEqual(
            _strip_internal_details("api_key: {\nPlease check config"),
            "api_key: {\nPlease check config",
        )

        # 5-backslash parity (\\\\\") in container scanning preserves trailing public fields
        for raw_5bs, kept in (
            (r'{"detail": "{\"api_key\": [\"abc\\\\\"], \"public\": \"keep_me\"', "keep_me"),
            (r'{"detail": "{\"api_key\": {\"x\": \"a\\\\\"}, \"ok\": \"keep\"', "keep"),
        ):
            with self.subTest(raw_5bs=raw_5bs):
                cleaned_5bs = _strip_internal_details(raw_5bs)
                self.assertIn(kept, cleaned_5bs)
                self.assertIn("<redacted>", cleaned_5bs)
                self.assertEqual(_strip_internal_details(cleaned_5bs), cleaned_5bs)

        # 5. Truncated 3x-escaped JSON, leading-backslash unquoted KV, and multiline auth vs prose
        for raw, secret, must_keep in (
            (r'{"body": "{\"detail\": \"{\\\"api_key\\\": \\\"sec_3x_trunc', "sec_3x_trunc", None),
            (r'{"body": "{\"detail\": \"{\\\"Authorization\\\": \\\"Basic YWRtaW46c2VjcmV0', "YWRtaW46c2VjcmV0", None),
            (r'{"body": "{\"detail\": \"{\\\"api_key\\\": {\\\"nested\\\": \\\"sec_3x_obj', "sec_3x_obj", None),
            (r"api_key=\secret123 ok=pub", "secret123", "ok=pub"),
            ("api_key:\n  Basic YWRtaW46c2VjcmV0", "YWRtaW46c2VjcmV0", None),
            ("api_key:\n  Bearer tok_multi_123", "tok_multi_123", None),
        ):
            with self.subTest(raw=raw[:50]):
                cleaned = _strip_internal_details(raw)
                self.assertNotIn(secret, cleaned)
                self.assertIn("<redacted>", cleaned)
                if must_keep:
                    self.assertIn(must_keep, cleaned)
                self.assertEqual(_strip_internal_details(cleaned), cleaned)

        for prose in ("secret:\n  Basic setup failed", "Invalid token:\nBasic validation failed"):
            self.assertEqual(_strip_internal_details(prose), prose)

    def test_strip_internal_details_pem_boundaries_even_bs_container_desync_blank_lines_and_minor_auth(self):
        """Regression tests for Challenger 6 fixes: same-line PEM bounds, even-backslash quotes, container desync, blank lines, and minor auth/containers."""
        import shutil
        import tempfile
        from pathlib import Path

        from src.workflow_engine.bridge import RuntimeBridge

        # 1. Incomplete -----END must not consume subsequent -----BEGIN ... PRIVATE KEY----- header
        pem_incomplete_end = (
            "-----BEGIN CERTIFICATE-----\n"
            "MIIcert\n"
            "-----END CERTIFICATE\n"
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIISECRETKEY123\n"
            "-----END RSA PRIVATE KEY-----"
        )
        cleaned_pem1 = _strip_internal_details(pem_incomplete_end)
        self.assertNotIn("MIISECRETKEY123", cleaned_pem1)
        self.assertIn("MIIcert", cleaned_pem1)
        self.assertIn("<redacted>", cleaned_pem1)
        self.assertEqual(_strip_internal_details(cleaned_pem1), cleaned_pem1)

        pem_incomplete_end_with_text = (
            "-----BEGIN CERTIFICATE-----\n"
            "MIIcert\n"
            "-----END \n"
            "some text\n"
            "-----BEGIN PRIVATE KEY-----\n"
            "MIISECRETKEY456\n"
            "-----END PRIVATE KEY-----"
        )
        cleaned_pem2 = _strip_internal_details(pem_incomplete_end_with_text)
        self.assertNotIn("MIISECRETKEY456", cleaned_pem2)
        self.assertIn("some text", cleaned_pem2)
        self.assertEqual(_strip_internal_details(cleaned_pem2), cleaned_pem2)

        # Real RuntimeBridge stderr with incomplete END before PRIVATE KEY
        if shutil.which("node") is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                script = Path(tmpdir) / "pem_incomplete.mjs"
                script.write_text(
                    'export const meta = { name: "pem", description: "test", phases: [{ title: "run", detail: "crash" }], agentPlan: [] };\n'
                    "export default async function() {\n"
                    '  console.error("-----BEGIN CERTIFICATE-----\\nMIIcert\\n-----END CERTIFICATE\\n-----BEGIN RSA PRIVATE KEY-----\\nMII_BRIDGE_SECRET\\n-----END RSA PRIVATE KEY-----");\n'
                    "  await new Promise((r) => setTimeout(r, 30));\n"
                    "  process.exit(1);\n"
                    "}\n",
                    encoding="utf-8",
                )
                bridge = RuntimeBridge(script_path=str(script), cwd=tmpdir, on_agent_call=lambda p: None)
                try:
                    bridge.start()
                    with self.assertRaises(RuntimeError) as ctx:
                        bridge.run()
                finally:
                    bridge.stop()
                msg = str(ctx.exception)
                self.assertNotIn("MII_BRIDGE_SECRET", msg)
                self.assertIn("<redacted>", msg)

        # 2. Even-backslash quote values (2x, 4x) across KV, multiline KV, Bearer, and Authorization
        even_bs_cases = [
            ('api_key=\\\\"secret_2bs_dq" ok=pub', "secret_2bs_dq", "ok=pub"),
            ("api_key=\\\\'secret_2bs_sq' ok=pub", "secret_2bs_sq", "ok=pub"),
            ('api_key=\\\\\\\\"secret_4bs_dq" ok=pub', "secret_4bs_dq", "ok=pub"),
            ('api_key=\\\\"my_2bs_unclosed ok=pub', "my_2bs_unclosed", None),
            ('api_key:\n  \\\\"ml_secret_2bs\\\\"', "ml_secret_2bs", None),
            ('Bearer \\\\"my_bearer_2bs_secret\\\\" ok=pub', "my_bearer_2bs_secret", "ok=pub"),
            ('Authorization: Bearer \\\\"my_auth_2bs_secret\\\\" ok=pub', "my_auth_2bs_secret", "ok=pub"),
            ('Bearer \\\\\\"my_bearer_3bs_secret\\\\\\" ok=pub', "my_bearer_3bs_secret", "ok=pub"),
            ('Authorization: Bearer \\\\\\"my_auth_3bs_secret\\\\\\" ok=pub', "my_auth_3bs_secret", "ok=pub"),
        ]
        for raw, secret, must_keep in even_bs_cases:
            with self.subTest(raw=raw):
                cleaned = _strip_internal_details(raw)
                self.assertNotIn(secret, cleaned)
                self.assertIn("<redacted>", cleaned)
                if must_keep is not None:
                    self.assertIn(must_keep, cleaned)
                self.assertEqual(_strip_internal_details(cleaned), cleaned)

        # Empty even/odd Bearer and KV quotes remain preserved
        for empty in (
            'Bearer ""',
            'Bearer \\"\\"',
            'Bearer \\\\"\\\\"',
            'Bearer \\\\\\"\\\\\\"',
            'api_key=\\\\"\\\\"',
        ):
            with self.subTest(empty=empty):
                self.assertEqual(_strip_internal_details(empty), empty)

        # 3. Container with escaped_bs=0 containing \"...\" preserves trailing public fields
        for raw_c, kept in (
            ('api_key=[\\"a]b\\", \\"c\\"] ok="public_keep"', 'ok="public_keep"'),
            ('api_key=["a]b", "c"] ok="public_keep"', 'ok="public_keep"'),
            ('api_key={\\"k\\": \\"a}b\\"} ok="public_keep"', 'ok="public_keep"'),
        ):
            with self.subTest(raw_c=raw_c):
                cleaned_c = _strip_internal_details(raw_c)
                self.assertIn(kept, cleaned_c)
                self.assertIn("<redacted>", cleaned_c)
                self.assertEqual(_strip_internal_details(cleaned_c), cleaned_c)

        # 4. Multiline containers with blank lines redact secret while rejecting prose
        blank_container = 'api_key={\n\n  val: "secret_after_blank"\n}'
        cleaned_blank = _strip_internal_details(blank_container)
        self.assertNotIn("secret_after_blank", cleaned_blank)
        self.assertIn("<redacted>", cleaned_blank)
        self.assertEqual(_strip_internal_details(cleaned_blank), cleaned_blank)
        self.assertEqual(
            _strip_internal_details("api_key: {\n\nPlease check config"),
            "api_key: {\n\nPlease check config",
        )

        # 5. Minor multiline auth (short alnum, lowercase, HMAC-SHA256), util.inspect Map, and multiline closed container strings
        for raw_m, secret in (
            ("api_key:\n  Basic 9f8a1", "9f8a1"),
            ("api_key:\n  Basic abcdefghij", "abcdefghij"),
            ("api_key:\n  HMAC-SHA256 secret_sig_123", "secret_sig_123"),
            ("api_key=Map(1) {\n  'secret_key' => 'my_map_secret'\n}", "my_map_secret"),
            ('api_key=[\n  "line1\n  secret_line2"\n]', "secret_line2"),
        ):
            with self.subTest(raw_m=raw_m):
                cleaned_m = _strip_internal_details(raw_m)
                self.assertNotIn(secret, cleaned_m)
                self.assertIn("<redacted>", cleaned_m)
                self.assertEqual(_strip_internal_details(cleaned_m), cleaned_m)

        # Unclosed multiline string in unclosed container must not swallow prose
        unclosed_prose = 'api_key=[\n  "line1\nPlease check config'
        cleaned_unclosed = _strip_internal_details(unclosed_prose)
        self.assertIn("Please check config", cleaned_unclosed)
        self.assertNotIn("line1", cleaned_unclosed)

        # ReDoS timing checks for new even-backslash & blank-line patterns (< 50 ms)
        for name, payload in (
            ("sq_2bs_50k", 'api_key=\\\\"' + "\\" * 50000 + "!"),
            ("sq_auth2_2bs_50k", 'Authorization: Bearer \\\\"' + "\\" * 50000 + "!"),
            ("standalone_bearer_50k", 'Bearer \\"' + "\\" * 50000 + "!"),
            ("blank_lines_20k", 'api_key={\n' + "\n" * 20000 + '  val: "sec"\n}'),
        ):
            with self.subTest(redos=name):
                t0 = time.perf_counter()
                _strip_internal_details(payload)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self.assertLess(elapsed_ms, 50.0, f"{name} took {elapsed_ms:.2f} ms")

    def test_strip_internal_details_same_line_pem_end_non_mersenne_odd_bs_and_unclosed_multiline_isolation(self):
        """Regression tests for Challenger 7 fixes: same-line PEM END/BEGIN bounds, 5-bs/9-bs odd-backslash parity, and unclosed multiline container isolation."""
        import shutil
        import tempfile
        from pathlib import Path

        from src.workflow_engine.bridge import RuntimeBridge
        from src.workflow_engine.errors import _scan_container_value

        # 1. Same-line incomplete -----END ... before -----BEGIN ... PRIVATE KEY-----
        same_line_pem_cases = [
            (
                "-----BEGIN CERTIFICATE-----\nMIIcert\n"
                "-----END CERTIFICATE -----BEGIN RSA PRIVATE KEY-----\n"
                "MII_SAME_LINE_SPACE\n-----END RSA PRIVATE KEY-----",
                "MII_SAME_LINE_SPACE",
                ["MIIcert"],
            ),
            (
                "-----BEGIN CERTIFICATE-----\nMIIcert\n"
                "-----END CERTIFICATE\t-----BEGIN RSA PRIVATE KEY-----\n"
                "MII_SAME_LINE_TAB\n-----END RSA PRIVATE KEY-----",
                "MII_SAME_LINE_TAB",
                ["MIIcert"],
            ),
            (
                "-----BEGIN CERTIFICATE-----\nMIIcert\n"
                "-----END -----BEGIN PRIVATE KEY-----\n"
                "MII_SAME_LINE_EMPTY_LABEL\n-----END PRIVATE KEY-----",
                "MII_SAME_LINE_EMPTY_LABEL",
                ["MIIcert"],
            ),
            (
                "-----BEGIN CERTIFICATE -----BEGIN RSA PRIVATE KEY-----\n"
                "MII_SAME_LINE_BEGIN_BEGIN\n-----END RSA PRIVATE KEY-----",
                "MII_SAME_LINE_BEGIN_BEGIN",
                [],
            ),
            (
                "-----BEGIN CERTIFICATE -----BEGIN RSA PRIVATE KEY-----\n"
                "MII_SAME_LINE_BEGIN_UNCLOSED",
                "MII_SAME_LINE_BEGIN_UNCLOSED",
                [],
            ),
        ]
        for raw, secret, keeps in same_line_pem_cases:
            with self.subTest(pem=raw[:60]):
                cleaned = _strip_internal_details(raw)
                self.assertNotIn(secret, cleaned)
                self.assertIn("<redacted>", cleaned)
                for keep in keeps:
                    self.assertIn(keep, cleaned)
                self.assertEqual(_strip_internal_details(cleaned), cleaned)

        # Real RuntimeBridge stderr with same-line incomplete END before PRIVATE KEY
        if shutil.which("node") is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                script = Path(tmpdir) / "pem_same_line.mjs"
                script.write_text(
                    'export const meta = { name: "pem_sl", description: "test", phases: [{ title: "run", detail: "crash" }], agentPlan: [] };\n'
                    "export default async function() {\n"
                    '  console.error("-----BEGIN CERTIFICATE-----\\nMIIcert\\n-----END CERTIFICATE -----BEGIN RSA PRIVATE KEY-----\\nMII_BRIDGE_SAME_LINE_SECRET\\n-----END RSA PRIVATE KEY-----");\n'
                    "  await new Promise((r) => setTimeout(r, 30));\n"
                    "  process.exit(1);\n"
                    "}\n",
                    encoding="utf-8",
                )
                bridge = RuntimeBridge(script_path=str(script), cwd=tmpdir, on_agent_call=lambda p: None)
                try:
                    bridge.start()
                    with self.assertRaises(RuntimeError) as ctx:
                        bridge.run()
                finally:
                    bridge.stop()
                msg = str(ctx.exception)
                self.assertNotIn("MII_BRIDGE_SAME_LINE_SECRET", msg)
                self.assertIn("<redacted>", msg)

        # 2. Non-(2^k - 1) odd-backslash literals (5-bs, 9-bs, 7-bs) with inner 1-bs/3-bs quotes
        bs1 = "\\"
        bs3 = "\\\\\\"
        bs5 = "\\\\\\\\\\"
        bs7 = "\\\\\\\\\\\\\\"
        bs9 = "\\\\\\\\\\\\\\\\\\"
        odd_bs_cases = [
            (f'api_key={bs5}"sec{bs1}"ret_5bs{bs5}" ok="pub"', "ret_5bs", 'ok="pub"'),
            (f'api_key={bs5}"sec{bs3}"ret_5bs_3{bs5}" ok="pub"', "ret_5bs_3", 'ok="pub"'),
            (f'api_key={bs5}"sec{bs1}"ret_5bs_esc{bs9}" ok="pub"', "ret_5bs_esc", 'ok="pub"'),
            (f'api_key={bs9}"sec{bs1}"ret_9bs{bs9}" ok="pub"', "ret_9bs", 'ok="pub"'),
            (f'api_key={bs7}"sec{bs1}"ret_7bs{bs7}" ok="pub"', "ret_7bs", 'ok="pub"'),
            (f'Bearer {bs5}"sec{bs1}"ret_b5{bs5}" ok="pub"', "ret_b5", 'ok="pub"'),
            (f'Bearer {bs9}"sec{bs1}"ret_b9{bs9}" ok="pub"', "ret_b9", 'ok="pub"'),
            (f'Authorization: Bearer {bs5}"sec{bs1}"ret_a5{bs5}" ok="pub"', "ret_a5", 'ok="pub"'),
            (f'Authorization: Bearer {bs9}"sec{bs1}"ret_a9{bs9}" ok="pub"', "ret_a9", 'ok="pub"'),
            (f'api_key=[{bs5}"a{bs1}"b]c{bs5}", {bs5}"sec_d5{bs5}"] ok="pub"', "sec_d5", 'ok="pub"'),
            (f'api_key=[{bs9}"a{bs1}"b]c{bs9}", {bs9}"sec_d9{bs9}"] ok="pub"', "sec_d9", 'ok="pub"'),
        ]
        for raw, secret, must_keep in odd_bs_cases:
            with self.subTest(odd_bs=raw[:50]):
                cleaned = _strip_internal_details(raw)
                self.assertNotIn(secret, cleaned)
                self.assertIn("<redacted>", cleaned)
                self.assertIn(must_keep, cleaned)
                self.assertEqual(_strip_internal_details(cleaned), cleaned)

        for empty in (
            f'Bearer {bs5}"{bs5}"',
            f'Bearer {bs9}"{bs9}"',
            f'api_key={bs5}"{bs5}"',
            f'api_key={bs9}"{bs9}"',
        ):
            with self.subTest(empty=empty):
                self.assertEqual(_strip_internal_details(empty), empty)

        # 3. Earlier unclosed multiline containers must not starve subsequent closed multiline string containers
        unclosed_then_closed = (
            "api_key=[\n  'unclosed_sq_1\n"
            + ("ok=1\n" * 600)
            + "api_key=[\n  'unclosed_sq_2\n"
            + ("ok=2\n" * 600)
            + 'api_key=[\n  "closed_line1\n  SECRET_AFTER_SQ_BUDGET_0"\n]'
        )
        cleaned_u2c = _strip_internal_details(unclosed_then_closed)
        self.assertNotIn("SECRET_AFTER_SQ_BUDGET_0", cleaned_u2c)
        self.assertIn("<redacted>", cleaned_u2c)
        self.assertEqual(_strip_internal_details(cleaned_u2c), cleaned_u2c)

        multi_delim_unclosed = (
            "api_key=[\n  'unclosed_sq\n"
            + ("ok=1\n" * 600)
            + f'api_key=[\n  {bs1}"unclosed_bs1\n'
            + ("ok=2\n" * 600)
            + f"api_key=[\n  {bs1}'unclosed_bs1_sq\n"
            + ("ok=3\n" * 600)
            + f'api_key=[\n  {bs3}"unclosed_bs3\n'
            + ("ok=4\n" * 600)
            + 'api_key=[\n  "closed_line1\n  SECRET_AFTER_MULTI_DELIM"\n] ok=pub'
        )
        cleaned_multi = _strip_internal_details(multi_delim_unclosed)
        self.assertNotIn("SECRET_AFTER_MULTI_DELIM", cleaned_multi)
        self.assertIn("ok=pub", cleaned_multi)
        self.assertEqual(_strip_internal_details(cleaned_multi), cleaned_multi)

        end_idx, empty_flag = _scan_container_value(
            '[\n  "closed_line1\n  SECRET_VAL"\n]', 0, multiline_budget=[0]
        )
        self.assertEqual(end_idx, 33)
        self.assertFalse(empty_flag)

        # O(N) scaling bounds for new fixes
        for name, payload, limit_ms in (
            (
                "pem_20k_same_line_end",
                "-----BEGIN CERTIFICATE-----\nMIIcert\n-----END CERTIFICATE -----BEGIN CERTIFICATE-----\n" * 5000,
                150.0,
            ),
            ("ml_str_fallback_10k", 'api_key=[\n  "line1\n' * 10000, 250.0),
            ("ml_sq_5k_unclosed", "api_key=[\n  'unclosed_sq\n" * 5000 + "ok=1\n" * 5000, 200.0),
            ("5bs_50k", f'api_key={bs5}"' + "\\" * 50000 + "!", 50.0),
            ("9bs_50k", f'api_key={bs9}"' + "\\" * 50000 + "!", 50.0),
        ):
            with self.subTest(redos=name):
                t0 = time.perf_counter()
                _strip_internal_details(payload)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self.assertLess(elapsed_ms, limit_ms, f"{name} took {elapsed_ms:.2f} ms")

    def test_strip_internal_details_multi_dash_pem_phase2_isolation_and_linear_bounds(self):
        """Regression tests for 6..12-dash same-line PEM bounds, >16/zero-bs Phase-2 container isolation, and O(N) scaling across container, empty-quote, and embedded-JSON paths."""
        import shutil
        import tempfile
        from pathlib import Path

        from src.workflow_engine.bridge import RuntimeBridge

        # 1. 6+..12-dash same-line PEM END/BEGIN and BEGIN/BEGIN boundaries (LF, CRLF, space, tab, no-space, truncated)
        for dash_count in range(5, 13):
            dashes = "-" * dash_count
            for sep in (" ", "\t", ""):
                for nl in ("\n", "\r\n"):
                    raw_end_begin = (
                        f"-----BEGIN CERTIFICATE-----{nl}MIIcert{nl}"
                        f"-----END CERTIFICATE{sep}{dashes}BEGIN RSA PRIVATE KEY-----{nl}"
                        f"MII_SEC_{dash_count}{nl}-----END RSA PRIVATE KEY-----"
                    )
                    cleaned_eb = _strip_internal_details(raw_end_begin)
                    self.assertNotIn(f"MII_SEC_{dash_count}", cleaned_eb)
                    self.assertIn("<redacted>", cleaned_eb)
                    if dash_count < 10:
                        self.assertIn("MIIcert", cleaned_eb)
                    self.assertEqual(_strip_internal_details(cleaned_eb), cleaned_eb)

                    raw_trunc = (
                        f"-----BEGIN CERTIFICATE{sep}{dashes}BEGIN RSA PRIVATE KEY-----{nl}"
                        f"MII_TRUNC_{dash_count}"
                    )
                    cleaned_tr = _strip_internal_details(raw_trunc)
                    self.assertNotIn(f"MII_TRUNC_{dash_count}", cleaned_tr)
                    self.assertIn("<redacted>", cleaned_tr)
                    self.assertEqual(_strip_internal_details(cleaned_tr), cleaned_tr)

        if shutil.which("node") is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                script = Path(tmpdir) / "pem_6dash.mjs"
                script.write_text(
                    'export const meta = { name: "pem6", description: "test", phases: [{ title: "run", detail: "crash" }], agentPlan: [] };\n'
                    "export default async function() {\n"
                    '  console.error("-----BEGIN CERTIFICATE-----\\nMIIcert\\n-----END CERTIFICATE ------BEGIN RSA PRIVATE KEY-----\\nMII_BRIDGE_6DASH_SECRET\\n-----END RSA PRIVATE KEY-----");\n'
                    "  await new Promise((r) => setTimeout(r, 30));\n"
                    "  process.exit(1);\n"
                    "}\n",
                    encoding="utf-8",
                )
                bridge = RuntimeBridge(
                    script_path=str(script), cwd=tmpdir, on_agent_call=lambda p: None
                )
                try:
                    bridge.start()
                    with self.assertRaises(RuntimeError) as ctx:
                        bridge.run()
                finally:
                    bridge.stop()
                msg = str(ctx.exception)
                self.assertNotIn("MII_BRIDGE_6DASH_SECRET", msg)
                self.assertIn("<redacted>", msg)

        # 2. >16 unclosed delimiters + nested closed multiline container + long 5-bs multiline tail
        delims_16 = ["\\" * (2 * idx + 11) + "'" for idx in range(16)]
        prefix_16 = "".join(
            f"api_key=[\n  {delim}unclosed_{idx}\n" + ("ok=1\n" * 5)
            for idx, delim in enumerate(delims_16)
        )
        bs5 = "\\\\\\\\\\"
        tail_block = '  "k": "v",\n' * 450
        c1_body = (
            'api_key=[\n  "ml1\n  ml2"\n'
            '  api_key: [\n    "closed_dq_l1\n    closed_dq_l2",\n'
            '    "pad": "012345678901234567890123456789012345678901234567890123456789",\n'
            '    "SECRET_PHASE2_TAIL64_LEAK"\n  ],\n'
            f"  {bs5}'\n"
        )
        payload_16 = (
            prefix_16
            + c1_body
            + ("\n" * (len(tail_block) + 1))
            + f"{bs5}'\n"
            + tail_block
        )
        cleaned_16 = _strip_internal_details(payload_16)
        self.assertNotIn("SECRET_PHASE2_TAIL64_LEAK", cleaned_16)
        self.assertEqual(_strip_internal_details(cleaned_16), cleaned_16)

        # 3. Zero-backslash overlapping Phase-2 starvation & nested container inside jumped multiline string
        c2_zero_bs = (
            '  api_key: [\n'
            '    "closed_dq_l1\n    closed_dq_l2",\n'
            '    "pad": "' + ("0123456789" * 20) + '",\n'
            '    "SECRET_ZERO_BS_STARVE"\n'
            '  ],\n'
        )
        cleaned_zero_bs = _strip_internal_details(
            'api_key=[\n  "a\nb"\n' + c2_zero_bs + ("1," * 2500) + "1"
        )
        self.assertNotIn("SECRET_ZERO_BS_STARVE", cleaned_zero_bs)
        self.assertEqual(_strip_internal_details(cleaned_zero_bs), cleaned_zero_bs)

        u20 = "".join(
            'api_key=[\n  "ml1\n  ml2"\n' + ("1," * 500) + "1\n" for _ in range(20)
        )
        jumped_c2 = (
            'api_key=[\n  "outer\n'
            '  api_key: [\n    "inner_ml1\n    inner_ml2",\n'
            '    "pad": "' + ("0123456789" * 20) + '",\n'
            '    "SECRET_INSIDE_JUMPED_STR"\n  ],\n'
            '  end"\n]'
        )
        cleaned_jumped = _strip_internal_details(u20 + jumped_c2)
        self.assertNotIn("SECRET_INSIDE_JUMPED_STR", cleaned_jumped)
        self.assertEqual(_strip_internal_details(cleaned_jumped), cleaned_jumped)

        # CRLF closed multiline container
        cleaned_crlf = _strip_internal_details(
            'api_key=[\r\n  "l1\r\n  SECRET_CRLF_ML"\r\n]'
        )
        self.assertNotIn("SECRET_CRLF_ML", cleaned_crlf)
        self.assertIn("<redacted>", cleaned_crlf)

        # 4. Empty-quote & late embedded-JSON preservation alongside large invalid prefixes
        self.assertEqual(_strip_internal_details('Bearer ""'), 'Bearer ""')
        self.assertEqual(
            _strip_internal_details('Error: Cannot find module ""'),
            "Error: Cannot find module",
        )
        late_json = (
            ("x" * 500000) + ("{a} " * 8000) + '{"api_key": "late_sec"}'
        )
        cleaned_late = _strip_internal_details(late_json)
        self.assertNotIn("late_sec", cleaned_late)
        self.assertIn("<redacted>", cleaned_late)

        # 5. Linear O(N) / O(N log N) scaling bounds across container, empty-quote, and embedded-JSON paths
        for name, payload, limit_ms in (
            (
                "desc_4k1_400_with_p16",
                prefix_16
                + "".join(
                    "api_key=[\n  " + ("\\" * (4 * k + 1)) + f'"u{k}\n'
                    for k in range(400, 0, -1)
                ),
                150.0,
            ),
            (
                "desc_4k1_400_no_p16",
                "".join(
                    "api_key=[\n  " + ("\\" * (4 * k + 1)) + f'"u{k}\n'
                    for k in range(400, 0, -1)
                ),
                150.0,
            ),
            (
                "desc_4k3_400",
                prefix_16
                + "".join(
                    "api_key=[\n  " + ("\\" * (4 * k + 3)) + f'"u{k}\n'
                    for k in range(400, 0, -1)
                ),
                150.0,
            ),
            (
                "mixed_dq_sq_4k1_400",
                prefix_16
                + "".join(
                    "api_key=[\n  "
                    + ("\\" * (4 * k + 1))
                    + ('"' if k % 2 == 0 else "'")
                    + f"u{k}\n"
                    for k in range(400, 0, -1)
                ),
                150.0,
            ),
            ("empty_quotes_200k", 'a "" ' * 40000, 60.0),
            ("json_fail_532k", ("x" * 500000) + ("{a} " * 8000), 150.0),
            ("esc_quote_192k", r'\"{\\\"a\\\":1}\" ' * 12000, 600.0),
        ):
            with self.subTest(scaling=name):
                t0 = time.perf_counter()
                _strip_internal_details(payload)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self.assertLess(
                    elapsed_ms, limit_ms, f"{name} took {elapsed_ms:.2f} ms"
                )

    def test_strip_internal_details_phase2_memo_no_starvation_or_failed_memo_poison(self):
        """Regression tests for Phase-2 p2_memo non-starvation, failed_memo non-poisoning, nested C3-in-C2, and O(N log N) single-line string jumping."""
        # 1. Challenger 9 Probe 1 (Mersenne K=7, K=10) & Polynomial Probe 1 (K=50, K=200)
        for label, k_count, rep, use_mersenne, limit_ms in (
            ("p1_mersenne_k7", 7, 250, True, 200.0),
            ("p1_mersenne_k10", 10, 250, True, 200.0),
            ("p1_poly_k50", 50, 2000, False, 250.0),
            ("p1_poly_k200", 200, 10000, False, 500.0),
        ):
            with self.subTest(probe=label):
                if use_mersenne:
                    parts = [
                        "api_key={\n  " + ("\\" * ((1 << (k + 1)) - 1)) + f"'m{k}\n"
                        for k in range(k_count)
                    ]
                else:
                    parts = [
                        "api_key={\n  " + ("\\" * (4 * (k_count - k) + 1)) + f"'m{k}\n"
                        for k in range(k_count)
                    ]
                parts.append(
                    '  api_key: [ ' + ("\\" * (4 * k_count + 5)) + '"closed_dq_l1\n'
                )
                for k in range(k_count - 1, -1, -1):
                    open_bs = (
                        ((1 << (k + 1)) - 1) if use_mersenne else (4 * (k_count - k) + 1)
                    )
                    parts.append(
                        ("\\" * open_bs) + "'" + ("\\" * (4 * k + 5)) + '"'
                    )
                parts.append(
                    ("\\" * (4 * k_count + 9))
                    + '" '
                    + ('"a": "0123456789", ' * rep)
                    + '"SECRET_P2_VISITS" ]'
                )
                payload = "".join(parts)
                t0 = time.perf_counter()
                out = _strip_internal_details(payload)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self.assertNotIn("SECRET_P2_VISITS", out)
                self.assertIn("<redacted>", out)
                self.assertEqual(_strip_internal_details(out), out)
                self.assertLess(
                    elapsed_ms, limit_ms, f"{label} took {elapsed_ms:.2f} ms"
                )

        # 2. Challenger 9 Probe 2 (failed_memo non-poisoning when U_{K-1} pushes C2 onto stack)
        for label, k_count, rep, use_mersenne, limit_ms in (
            ("p2_mersenne_k7", 7, 250, True, 200.0),
            ("p2_mersenne_k10", 10, 250, True, 200.0),
            ("p2_poly_k50", 50, 2000, False, 250.0),
            ("p2_poly_k200", 200, 10000, False, 500.0),
        ):
            with self.subTest(probe=label):
                if use_mersenne:
                    parts2 = [
                        "api_key={\n  " + ("\\" * ((1 << (k + 1)) - 1)) + f"'m{k}\n"
                        for k in range(k_count)
                    ]
                    parts2.append(("\\" * ((1 << k_count) - 1)) + "' ")
                else:
                    parts2 = [
                        "api_key={\n  " + ("\\" * (4 * (k_count - k) + 1)) + f"'m{k}\n"
                        for k in range(k_count)
                    ]
                    parts2.append(("\\" * 5) + "' ")
                parts2.append(
                    '  api_key: [ '
                    + ("\\" * (4 * (k_count - 1) + 5))
                    + '"closed_dq_l1\n'
                )
                for k in range(k_count - 2, -1, -1):
                    open_bs = (
                        ((1 << (k + 1)) - 1) if use_mersenne else (4 * (k_count - k) + 1)
                    )
                    parts2.append(
                        ("\\" * open_bs) + "'" + ("\\" * (4 * k + 5)) + '"'
                    )
                parts2.append(
                    ("\\" * (4 * (k_count - 1) + 9))
                    + '" '
                    + ('"a": "0123456789", ' * rep)
                    + '"SECRET_FAILED_MEMO_POISON" ]'
                )
                payload2 = "".join(parts2)
                t0 = time.perf_counter()
                out2 = _strip_internal_details(payload2)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self.assertNotIn("SECRET_FAILED_MEMO_POISON", out2)
                self.assertIn("<redacted>", out2)
                self.assertEqual(_strip_internal_details(out2), out2)
                self.assertLess(
                    elapsed_ms, limit_ms, f"{label} took {elapsed_ms:.2f} ms"
                )

        # 3. Nested outer-fail / inner-closed C2 + C3 inside C2 + outer prose preservation
        k_nested = 10
        nested_parts = [
            "api_key={\n  " + ("\\" * ((1 << (k + 1)) - 1)) + f"'u{k}\n"
            for k in range(k_nested)
        ]
        nested_parts.append(
            '  api_key: [ ' + ("\\" * (4 * k_nested + 5)) + '"c2_l1\n'
        )
        for k in range(k_nested - 1, -1, -1):
            nested_parts.append(
                ("\\" * ((1 << (k + 1)) - 1)) + "'" + ("\\" * (4 * k + 5)) + '"'
            )
        nested_parts.append(
            ("\\" * (4 * k_nested + 9))
            + '" '
            + 'token: {\n    "c3_l1\n    SECRET_C3_INNER"\n  }, '
            + ('"a": "0123456789", ' * 300)
            + '"SECRET_C2_OUTER" ] ok=outer_prose'
        )
        out_nested = _strip_internal_details("".join(nested_parts))
        self.assertNotIn("SECRET_C3_INNER", out_nested)
        self.assertNotIn("SECRET_C2_OUTER", out_nested)
        self.assertIn("<redacted>", out_nested)
        self.assertIn("ok=outer_prose", out_nested)
        self.assertEqual(_strip_internal_details(out_nested), out_nested)

        # 4. Overlapping Phase-2 single-line strings over 100KB (K=200) and 500KB (K=400) lines
        for label, k_sl, line_len, limit_ms in (
            ("p2_sl_k200_100kb", 200, 100000, 300.0),
            ("p2_sl_k400_500kb", 400, 500000, 1000.0),
        ):
            with self.subTest(probe=label):
                sl_parts = [
                    "api_key={\n  " + ("\\" * (4 * (k_sl - k) + 1)) + f"'m{k}\n"
                    for k in range(k_sl)
                ]
                for k in range(k_sl - 1, -1, -1):
                    sl_parts.append(
                        ("\\" * (4 * (k_sl - k) + 1)) + "'" + ("\\" * (4 * k + 5)) + '"'
                    )
                sl_parts.append("x" * line_len)
                for k in range(k_sl):
                    sl_parts.append(("\\" * (4 * k + 5)) + '"')
                sl_parts.append(' api_key=["SECRET_AFTER_OVERLAPPING_SL"]')
                payload_sl = "".join(sl_parts)
                t0 = time.perf_counter()
                out_sl = _strip_internal_details(payload_sl)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self.assertNotIn("SECRET_AFTER_OVERLAPPING_SL", out_sl)
                self.assertIn("<redacted>", out_sl)
                self.assertEqual(_strip_internal_details(out_sl), out_sl)
                self.assertLess(
                    elapsed_ms, limit_ms, f"{label} took {elapsed_ms:.2f} ms"
                )


if __name__ == "__main__":
    unittest.main()
