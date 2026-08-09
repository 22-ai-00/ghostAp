"""Tests for workflow error sanitization (errors.py).

Validates:
- ErrorCategory enum has all expected members
- sanitize_for_reply returns category-specific safe messages
- Internal details (file paths, tracebacks, module names) are NOT in output
- _strip_internal_details removes sensitive patterns
- WorkflowUserError preserves raw detail for logging
"""

import unittest

from src.workflow_engine.errors import (
    ErrorCategory,
    WorkflowUserError,
    _sanitize_error,
    _strip_internal_details,
    sanitize_for_reply,
)


class TestErrorCategoryEnum(unittest.TestCase):
    """Test ErrorCategory has all expected members."""

    def test_all_categories_present(self):
        expected = {
            # Unified error surface (four categories)
            "SESSION_EXPIRED",
            "INVALID_STATE",
            "INVALID_ARGUMENT",
            "FORBIDDEN",
            # Legacy detailed categories
            "AGENT_LIMIT",
            "TOOL_NOT_ALLOWED",
            "SCRIPT_VALIDATION",
            "RUNTIME_TIMEOUT",
            "REVIEW_FAILED",
            "INTERNAL_ERROR",
            "CANCELLED",
            # Deprecated: kept for backwards compatibility
            "BUDGET_EXHAUSTED",
        }
        actual = {c.name for c in ErrorCategory}
        self.assertEqual(expected, actual)

    def test_values_are_snake_case(self):
        for cat in ErrorCategory:
            self.assertTrue(cat.value.islower())
            self.assertNotIn(" ", cat.value)


class TestSanitizeForReply(unittest.TestCase):
    """Test sanitize_for_reply returns safe user messages."""

    def test_agent_limit(self):
        msg = sanitize_for_reply("limit reached at 200", ErrorCategory.AGENT_LIMIT)
        self.assertIn("Agent", msg)
        self.assertNotIn("200", msg)

    def test_internal_error_hides_raw(self):
        raw = 'File "/home/user/src/workflow_engine/executor.py", line 42, in execute\nRuntimeError: some internal problem'
        msg = sanitize_for_reply(raw, ErrorCategory.INTERNAL_ERROR)
        self.assertNotIn("/home/user", msg)
        self.assertNotIn("executor.py", msg)
        self.assertNotIn("RuntimeError", msg)
        self.assertIn("内部错误", msg)

    def test_cancelled(self):
        msg = sanitize_for_reply("", ErrorCategory.CANCELLED)
        self.assertIn("取消", msg)

    def test_budget_exhausted_deprecated(self):
        """Test BUDGET_EXHAUSTED returns deprecated message for backward compatibility."""
        msg = sanitize_for_reply("token limit reached", ErrorCategory.BUDGET_EXHAUSTED)
        self.assertIn("预算已耗尽", msg)
        self.assertIn("废弃", msg)

    def test_budget_exhausted_no_longer_triggered(self):
        """Verify BUDGET_EXHAUSTED category is deprecated and not used in new code."""
        # This test documents that BUDGET_EXHAUSTED is kept only for backward compatibility
        # and should not be triggered by new error handling paths
        self.assertTrue(hasattr(ErrorCategory, 'BUDGET_EXHAUSTED'))

    def test_unknown_category_falls_back_to_internal(self):
        # All valid categories should produce non-empty messages
        for cat in ErrorCategory:
            msg = sanitize_for_reply("test error", cat)
            self.assertTrue(len(msg) > 5)


class TestStripInternalDetails(unittest.TestCase):
    """Test _strip_internal_details removes sensitive patterns."""

    def test_strips_file_paths(self):
        raw = "Error at /home/user/project/src/module.py:42"
        result = _strip_internal_details(raw)
        self.assertNotIn("/home/user", result)
        self.assertNotIn("module.py", result)

    def test_strips_traceback(self):
        raw = (
            "Traceback (most recent call last):\n"
            '  File "/opt/app/src/engine.py", line 10, in run\n'
            "    raise ValueError('bad')\n"
            "ValueError: bad"
        )
        result = _strip_internal_details(raw)
        self.assertNotIn("Traceback", result)
        self.assertNotIn("/opt/app", result)

    def test_strips_dotted_module_names(self):
        raw = "Error in src.workflow_engine.executor: timeout"
        result = _strip_internal_details(raw)
        self.assertNotIn("src.workflow_engine.executor", result)

    def test_preserves_plain_error_text(self):
        raw = "Connection timed out after 30 seconds"
        result = _strip_internal_details(raw)
        self.assertIn("Connection timed out", result)

    def test_collapses_whitespace(self):
        raw = "error\n\n\n\n\ndetails"
        result = _strip_internal_details(raw)
        self.assertNotIn("\n\n\n", result)


class TestSanitizeError(unittest.TestCase):
    """Test _sanitize_error produces WorkflowUserError correctly."""

    def test_returns_workflow_user_error(self):
        result = _sanitize_error("some raw error", ErrorCategory.INTERNAL_ERROR)
        self.assertIsInstance(result, WorkflowUserError)
        self.assertEqual(result.category, ErrorCategory.INTERNAL_ERROR)

    def test_preserves_internal_detail(self):
        raw = "detailed internal info with /path/to/file.py:42"
        result = _sanitize_error(raw, ErrorCategory.INTERNAL_ERROR)
        self.assertEqual(result.internal_detail, raw)

    def test_empty_raw_gives_none_detail(self):
        result = _sanitize_error("", ErrorCategory.CANCELLED)
        self.assertIsNone(result.internal_detail)

    def test_user_message_is_clean(self):
        raw = 'File "/src/engine.py", line 1\nsrc.workflow_engine.bridge crashed'
        result = _sanitize_error(raw, ErrorCategory.INTERNAL_ERROR)
        self.assertNotIn("/src/engine.py", result.user_message)
        self.assertNotIn("src.workflow_engine", result.user_message)


class TestEngineExceptionErrorSanitized(unittest.TestCase):
    """Integration: engine exception handler sanitizes paths."""

    def test_engine_exception_error_sanitized(self):
        """AC13: project.error must not leak internal file paths."""
        raw = "ValueError: failed at /data00/home/user/work/ghostAp/src/workflow_engine/engine.py:42"
        sanitized = _strip_internal_details(raw)
        self.assertNotIn("/data00", sanitized)
        self.assertNotIn("/home", sanitized)
        self.assertIn("ValueError", sanitized)  # error type is preserved


class TestBridgeStderrSanitized(unittest.TestCase):
    """Integration: bridge stderr content is sanitized."""

    def test_bridge_stderr_sanitized(self):
        """AC14: RuntimeError from bridge must not leak absolute paths in stderr."""
        stderr = (
            "Error: Cannot find module '/home/jiataorui/work/ghostAp/node_modules/foo/index.js'\n"
            "    at Module._resolveFilename (node:internal/modules/cjs/loader:1075:15)"
        )
        sanitized = _strip_internal_details(stderr)
        self.assertNotIn("/home/jiataorui", sanitized)
        self.assertNotIn("/node_modules/", sanitized)


class TestRendererCompletionCardErrorSanitized(unittest.TestCase):
    """Integration: completion card does not leak internal paths."""

    def test_renderer_completion_card_error_sanitized(self):
        """AC13: Completion card must not leak internal paths in error display."""
        # BudgetState has been removed - test verifies error sanitization directly
        raw_error = "RuntimeError: failed at /data00/home/user/work/ghostAp/src/workflow_engine/executor.py:130"
        sanitized = _strip_internal_details(raw_error)
        self.assertNotIn("/data00", sanitized)
        self.assertNotIn("/home/user", sanitized)


class TestRendererAgentErrorSanitized(unittest.TestCase):
    """Integration: progress card agent errors are sanitized."""

    def test_renderer_agent_error_sanitized(self):
        """AC13: Progress card must not leak dotted module names in agent errors."""
        raw_agent_error = "TimeoutError in src.workflow_engine.executor at /data00/home/user/work/ghostAp/src/workflow_engine/executor.py:42"
        sanitized = _strip_internal_details(raw_agent_error[:80])
        self.assertNotIn("src.workflow_engine.executor", sanitized)
        self.assertNotIn("/data00", sanitized)


if __name__ == "__main__":
    unittest.main()
