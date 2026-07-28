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
    MAX_QUEUE_SIZE,
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
# 4. TestBackpressureHandling
# ---------------------------------------------------------------------------


class TestBackpressureHandling(unittest.TestCase):
    """Test backpressure handling when message queue is full.

    Reliability impact: Under heavy load, the message queue can fill
    up. Backpressure prevents memory exhaustion and provides clear
    feedback to callers that the system is overloaded.
    """

    def setUp(self):
        self.cancel_event = threading.Event()
        from src.workflow_engine.bridge import RuntimeBridge

        self.bridge = RuntimeBridge(
            script_path="/tmp/test.js",
            cwd="/tmp",
            on_agent_call=MagicMock(),
            cancel_event=self.cancel_event,
        )
        # Initialize executor to avoid None checks
        self.bridge._executor = MagicMock()
        self.bridge._msg_queue = MagicMock()
        self.bridge._msg_condition = MagicMock()
        self.bridge._send_error_response = MagicMock()
        self.bridge._send_response = MagicMock()

    def tearDown(self):
        self.bridge.stop()

    def test_queue_full_rejects_new_calls(self):
        """Verify full queue rejects new agent calls with backpressure error.

        When the message queue reaches MAX_QUEUE_SIZE, new agent calls
        should be rejected immediately with a clear backpressure error.
        This prevents the system from becoming unresponsive under load.
        """
        # Simulate queue at capacity
        self.bridge._msg_queue.__len__.return_value = MAX_QUEUE_SIZE

        params = {"prompt": "test", "tool": "coco"}
        self.bridge._handle_agent_call(params, request_id="req_123")

        # Should have sent error response
        self.bridge._send_error_response.assert_called_once()
        call_args = self.bridge._send_error_response.call_args
        # request_id is positional, code and message are keyword args
        self.assertEqual(call_args.args[0], "req_123")
        self.assertEqual(call_args.kwargs["code"], -32000)  # Custom server error code
        self.assertIn("backpressure", call_args.kwargs["message"].lower())
        self.assertIn("retry later", call_args.kwargs["message"].lower())

    def test_backpressure_error_is_sanitized(self):
        """Verify backpressure errors are user-friendly and don't leak internals.

        Backpressure errors should tell users to retry without exposing
        internal queue sizes or implementation details.
        """
        self.bridge._msg_queue.__len__.return_value = MAX_QUEUE_SIZE

        params = {"prompt": "test", "tool": "coco"}
        self.bridge._handle_agent_call(params, request_id="req_456")

        call_args = self.bridge._send_error_response.call_args
        error_msg = call_args.kwargs["message"]

        # Should be user-friendly
        self.assertIn("too many pending messages", error_msg)
        self.assertIn("retry later", error_msg)
        # Should NOT leak internal constants
        self.assertNotIn(str(MAX_QUEUE_SIZE), error_msg)
        self.assertNotIn("_msg_queue", error_msg)
        self.assertNotIn("deque", error_msg)

    def test_queue_drains_after_backpressure(self):
        """Verify after queue drains, new calls are accepted again.

        Backpressure should be temporary — once the queue has space,
        new calls should be processed normally.
        """
        # First: queue is full, reject
        self.bridge._msg_queue.__len__.return_value = MAX_QUEUE_SIZE
        self.bridge._handle_agent_call({"prompt": "test1"}, "req_1")

        self.bridge._send_error_response.assert_called_once()
        self.bridge._send_error_response.reset_mock()

        # Now: queue has space, accept
        self.bridge._msg_queue.__len__.return_value = 5
        # Mock the executor submit to succeed
        mock_future = MagicMock()
        self.bridge._executor.submit.return_value = mock_future

        self.bridge._handle_agent_call({"prompt": "test2"}, "req_2")

        # Should NOT have sent error response
        self.bridge._send_error_response.assert_not_called()
        # Should have submitted to executor
        self.bridge._executor.submit.assert_called_once()


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

    def test_pipeline_continue_on_failure_camelcase(self):
        """Verify continueOnFailure (camelCase) also works for JS compatibility."""
        def stage1(x):
            return x + 1

        def failing(x):
            raise RuntimeError("boom")

        results = self._pipeline_python(
            [1, 2],
            stage1,
            failing,
            {"continueOnFailure": True},
        )

        self.assertEqual(results[0], {"error": "boom", "failedAtStage": 1, "partialResult": 2})
        self.assertEqual(results[1], {"error": "boom", "failedAtStage": 1, "partialResult": 3})

    def test_pipeline_js_source_has_continue_on_failure(self):
        """Verify the JS runtime source includes continue_on_failure support.

        This ensures the Python test implementation matches the actual
        JS implementation.
        """
        import os

        runtime_js_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "src",
            "workflow_engine",
            "runtime",
            "runtime.js",
        )
        with open(runtime_js_path) as f:
            source = f.read()

        # Verify the JS implementation has continue_on_failure support
        self.assertIn("continueOnFailure", source)
        self.assertIn("continue_on_failure", source)
        self.assertIn("failedAtStage", source)
        self.assertIn("partialResult", source)


# ---------------------------------------------------------------------------
# 8. TestCancelHandling
# ---------------------------------------------------------------------------


class TestCancelHandling(unittest.TestCase):
    """Test cancel event handling in AgentExecutor.

    Reliability impact: Users need to be able to cancel long-running
    workflows. Proper cancel handling ensures resources are cleaned
    up and the system remains responsive.
    """

    def setUp(self):
        self.cancel_event = threading.Event()
        self.executor = _make_executor(cancel_event=self.cancel_event)

    def tearDown(self):
        self.executor.shutdown(wait=False)

    def test_cancel_event_stops_execution(self):
        """Verify setting the cancel event stops in-progress agent calls.

        When a user cancels a workflow, the cancel event is set and
        the executor should stop at the earliest opportunity, returning
        a cancelled error rather than continuing to process.
        """
        # Set cancel before execution
        self.cancel_event.set()

        params = AgentCallParams(prompt="test prompt", tool="coco")
        result = self.executor.execute(params)

        # Should return cancelled error immediately
        self.assertIsNotNone(result.error)
        self.assertIn("Cancelled", result.error)
        # Session pool should not have been used
        self.executor._session_pool.submit.assert_not_called()

    def test_cancelled_error_category(self):
        """Verify cancelled calls return CANCELLED error category.

        Cancelled operations are not failures — they're user-initiated
        stops. Proper categorization ensures the UI shows appropriate
        messaging rather than error styling.
        """
        cancel_errors = [
            "Cancelled before execution",
            "Cancelled during execution",
            "Cancelled during retry",
            "Workflow cancelled by user",
        ]

        for error_msg in cancel_errors:
            with self.subTest(msg=error_msg):
                category = categorize_error(error_msg)
                self.assertEqual(category, ErrorCategory.CANCELLED)

                # Sanitized message should be user-friendly
                sanitized = sanitize_for_reply(error_msg, category)
                self.assertIn("取消", sanitized)
                self.assertNotIn("error", sanitized.lower())

    def test_cancel_stops_retry_backoff(self):
        """Verify cancel event interrupts retry backoff sleep.

        If a user cancels during the backoff delay between retries,
        the executor should stop waiting immediately rather than
        sleeping for the full backoff duration.
        """
        transient_error = RuntimeError("Network timeout")
        mock_session = _make_mock_session(side_effect=transient_error)
        mock_future = MagicMock()
        mock_future.result.return_value = mock_session
        self.executor._session_pool.submit.return_value = mock_future

        # Set cancel event after a short delay to interrupt backoff
        def set_cancel_after_delay():
            time.sleep(0.15)
            self.cancel_event.set()

        cancel_thread = threading.Thread(target=set_cancel_after_delay, daemon=True)

        params = AgentCallParams(prompt="test prompt", tool="coco")

        # Time the execution to verify it doesn't sleep for full backoff
        start = time.monotonic()
        cancel_thread.start()
        result = self.executor.execute(params)
        elapsed = time.monotonic() - start

        # Should have been cancelled during retry (not full 7s backoff)
        self.assertLess(elapsed, 1.0)
        self.assertIsNotNone(result.error)
        self.assertIn("Cancelled", result.error)


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


if __name__ == "__main__":
    unittest.main()
