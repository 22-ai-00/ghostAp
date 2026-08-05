"""AC17: Parametrized test verifying all UI_TEXT format templates don't raise KeyError.

Iterates over all UI_TEXT keys containing {} placeholders and calls .format()
with sample kwargs to ensure no missing parameters.
"""

import re
import unittest

import pytest

from src.card.render.footer import _format_idle_timeout
from src.card.ui_text import UI_TEXT

# Extract all format field names from a string
_FORMAT_FIELD_RE = re.compile(r"\{(\w+)\}")

# Sample values for all known format parameters across UI_TEXT
_SAMPLE_KWARGS = {
    "task_count": 3,
    "task_list": "  1. Task A\n  2. Task B\n  3. Task C",
    "task_name": "修复登录",
    "seq": 1,
    "next_seq": 2,
    "link_text": "查看最新卡片",
    "msg_id": "om_test_message",
    "sid_short": "abc123",
    "rotation_count": 2,
    "page": 3,
    "independent_count": 3,
    "merged_count": 2,
    "completed": 3,
    "engine_cmd": "deep",
    "project_name": "TestProject",
    "root_path": "/home/user/project",
    "tool_calls_count": 5,
    "iteration": 3,
    "total": 10,
    "status_icon": "✅",
    "cycle_num": 2,
    "tool_count": 8,
    "file_count": 4,
    "summary": "完成了重构",
    "n": 7,
    "context_lines": "some context",
    "error": "something went wrong",
    "reason": "timeout",
    "timeout": 30,
    "elapsed": 15,
    "remaining": 120,
    "model": "gpt-4",
    "provider": "openai",
    "name": "test_tool",
    "status": "completed",
    "count": 5,
    "max": 10,
    "cmd": "deep_status",
    "minutes": 5,
    "seconds": 30,
    "version": "1.0.0",
    "commands": "deep 或 loop",
    "duration": "3m 20s",
    "branch": "main",
    "tool_name": "read",
    "input_preview": "...",
    "output_preview": "...",
    "file_path": "/src/main.py",
    "line": 42,
    "message": "success",
    "chat_id": "chat_123",
    "user_name": "张三",
    "session_id": "sess_abc",
    "page_count": 2,
    "lock_holder": "user_x",
    "project": "my_project",
    "time_str": "14:30",
    "warn_minutes": 5,
}


def _get_format_keys() -> list[tuple[str, str]]:
    """Return list of (key, template) for all UI_TEXT entries with format placeholders."""
    results = []
    for key, value in UI_TEXT.items():
        if isinstance(value, str) and "{" in value:
            results.append((key, value))
    return results


_FORMAT_ENTRIES = _get_format_keys()


def test_ui_text_format_no_key_error():
    """All UI_TEXT templates with {} placeholders should format without KeyError."""
    failures: list[str] = []
    for key, template in _FORMAT_ENTRIES:
        # Extract required field names
        fields = _FORMAT_FIELD_RE.findall(template)
        if not fields:
            # Template has { but no named fields (e.g., literal braces) — skip
            continue

        # Build kwargs from sample values; use placeholder for unknown fields
        kwargs = {}
        for field in fields:
            kwargs[field] = _SAMPLE_KWARGS.get(field, f"<{field}>")

        # Attempt format and collect any errors
        try:
            result = template.format(**kwargs)
            if not isinstance(result, str) or len(result) == 0:
                failures.append(f"{key}: format returned empty or non-string")
        except Exception as exc:
            failures.append(f"{key}: {type(exc).__name__}: {exc}")

    assert not failures, (
        f"{len(failures)} UI_TEXT format failure(s):\n" + "\n".join(failures)
    )


class TestUITextFrozenProxy:
    """AC16: UI_TEXT is frozen (MappingProxyType) and raises TypeError on mutation."""

    def test_frozen_proxy_raises_on_assignment(self):
        """UI_TEXT['key'] = 'x' should raise TypeError."""
        with pytest.raises(TypeError):
            UI_TEXT["orch_plan_archived"] = "tampered"  # type: ignore[index]

# ---------------------------------------------------------------------------
# Strict UI_TEXT placeholder validation (merged from test_ui_text_placeholder_strict.py)
# ---------------------------------------------------------------------------

class TestUITextPlaceholderStrict(unittest.TestCase):
    """Strict validation of UI_TEXT entries."""

    def test_no_empty_values(self):
        """No UI_TEXT entry should have an empty string value."""
        empty_keys = [k for k, v in UI_TEXT.items() if isinstance(v, str) and v.strip() == ""]
        self.assertEqual(empty_keys, [], f"Empty UI_TEXT values: {empty_keys}")

    def test_no_unbalanced_braces(self):
        """Format strings should have balanced { } braces."""
        unbalanced = []
        for key, value in UI_TEXT.items():
            if not isinstance(value, str):
                continue
            # Count single braces (not escaped {{ or }})
            stripped = value.replace("{{", "").replace("}}", "")
            opens = stripped.count("{")
            closes = stripped.count("}")
            if opens != closes:
                unbalanced.append(f"{key}: opens={opens}, closes={closes}")
        self.assertEqual(unbalanced, [], "Unbalanced braces:\n" + "\n".join(unbalanced))

# ---------------------------------------------------------------------------
# Timeout format tests (merged from test_timeout_format.py)
# ---------------------------------------------------------------------------


class TestFormatIdleTimeout:
    """Test _format_idle_timeout edge cases."""

    def test_representative_formats(self):
        cases = [
            (60, "1 分钟"),         # test_minimum_value_60
            (300, "5 分钟"),        # test_300_seconds
            (350, "6 分钟"),        # test_non_60_divisible (rounds up)
            (1800, "30 分钟"),      # test_1800_seconds (default)
            (3600, "1 小时"),       # test_3600_seconds (exact hour)
            (7200, "2 小时"),       # test_7200_seconds (exact hours)
        ]
        for seconds, expected in cases:
            assert _format_idle_timeout(seconds) == expected, seconds

        for seconds in (4500, 5400):
            result = _format_idle_timeout(seconds)
            assert "约" in result, seconds
            assert "小时" in result, seconds
