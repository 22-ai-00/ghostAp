"""Tests for display-safe tool-call metadata."""

import pytest

from src.acp.models import ToolCallInfo
from src.card import tool_display
from src.card.tool_display import extract_tool_call_label


def test_agent_tool_name_rejects_escaped_source_fragment():
    call = ToolCallInfo(
        id="call_internal",
        title="agent",
        kind="other",
        status="in_progress",
        content='子代理：\\" not in ordinary_output\\",\\n',
    )

    assert tool_display.extract_agent_tool_name(call) == "agent"


def test_task_label_rejects_opaque_call_identifier():
    call = ToolCallInfo(
        id="call_internal",
        title="task",
        kind="other",
        status="in_progress",
        content="call_usOANvwWFgpuBkmHB",
    )

    assert extract_tool_call_label(call, generic_labels={"task"}) == "子任务"


def test_agent_tool_name_keeps_clean_marker_before_escaped_newline():
    call = ToolCallInfo(
        id="call_internal",
        title="agent",
        kind="other",
        status="in_progress",
        content="子代理：Explore\\nignored metadata",
    )

    assert tool_display.extract_agent_tool_name(call) == "Explore"


def test_agent_tool_name_uses_safe_non_generic_title_before_fallback():
    call = ToolCallInfo(
        id="call_internal",
        title="Review Agent",
        kind="other",
        status="in_progress",
        content="",
    )

    assert tool_display.extract_agent_tool_name(call) == "Review Agent"


def test_agent_tool_name_splits_actual_control_whitespace():
    call = ToolCallInfo(
        id="call_internal",
        title="agent",
        kind="other",
        status="in_progress",
        content="子代理：Explore\tignored metadata",
    )

    assert tool_display.extract_agent_tool_name(call) == "Explore"


def test_agent_tool_name_rejects_inline_raw_json_fragment():
    call = ToolCallInfo(
        id="call_internal",
        title="agent",
        kind="other",
        status="in_progress",
        content='子代理：Explore raw JSON: {"model":"x"}',
    )

    assert tool_display.extract_agent_tool_name(call) == "agent"


def test_task_label_rejects_prefixed_opaque_call_identifier():
    call = ToolCallInfo(
        id="call_internal",
        title="task",
        kind="other",
        status="in_progress",
        content="prefix-call_secret",
    )

    assert extract_tool_call_label(call, generic_labels={"task"}) == "子任务"


def test_task_label_uses_only_first_line_of_json_description():
    call = ToolCallInfo(
        id="call_internal",
        title="task",
        kind="other",
        status="in_progress",
        content='{"description":"修复路由\\nassert false"}',
    )

    assert extract_tool_call_label(call, generic_labels={"task"}) == "修复路由"


def test_task_label_rejects_unterminated_json_fragment():
    call = ToolCallInfo(
        id="call_internal",
        title="task",
        kind="other",
        status="in_progress",
        content='{"description":"Fix card"',
    )

    assert extract_tool_call_label(call, generic_labels={"task"}) == "子任务"


def test_task_label_rejects_truncated_formatted_output_fragment():
    call = ToolCallInfo(
        id="call_internal",
        title="task",
        kind="other",
        status="in_progress",
        content='"formatted_output": "FFFFFF',
    )

    assert extract_tool_call_label(call, generic_labels={"task"}) == "子任务"


@pytest.mark.parametrize(
    "label",
    [
        "[P0] 修复安全回归",
        "[1] 修复安全回归",
        "支持 raw JSON 输入",
    ],
)
def test_task_label_keeps_bracketed_or_json_named_human_text(label):
    call = ToolCallInfo(
        id="call_internal",
        title="task",
        kind="other",
        status="in_progress",
        content=label,
    )

    assert extract_tool_call_label(call, generic_labels={"task"}) == label


def test_failure_detail_extracts_nested_error_without_structured_stdout():
    detail = tool_display.sanitize_tool_failure_detail(
        {
            "call_id": "call_private_123",
            "stdout": "SECRET_STDOUT_MUST_NOT_LEAK",
            "result": {
                "message": "call_private_123 transport timed out",
            },
        }
    )

    assert detail == "transport timed out"
    assert "SECRET_STDOUT_MUST_NOT_LEAK" not in detail


def test_failure_detail_removes_arbitrary_structured_ids_and_commands():
    current_call_id = "550e8400-e29b-41d4-a716-446655440000"
    payload_call_id = "toolu_01JZ8G7R4M"
    nested_id = "request-opaque-987"
    command = "uv run secret-command --token hidden"

    detail = tool_display.sanitize_tool_failure_detail(
        {
            "call_id": payload_call_id,
            "command": ["/bin/zsh", "-lc", command],
            "result": {
                "id": nested_id,
                "error": (
                    f"{current_call_id} {payload_call_id} {nested_id} "
                    f"{command} timed out"
                ),
            },
        },
        opaque_ids=(current_call_id,),
    )

    assert "timed out" in detail
    assert current_call_id not in detail
    assert payload_call_id not in detail
    assert nested_id not in detail
    assert command not in detail


def test_failure_detail_removes_short_id_without_corrupting_other_numbers():
    detail = tool_display.sanitize_tool_failure_detail(
        {
            "id": 1,
            "error": "request 1 failed after 10 attempts",
        }
    )

    assert detail == "request failed after 10 attempts"


def test_failure_detail_removes_known_id_without_corrupting_word_substrings():
    detail = tool_display.sanitize_tool_failure_detail(
        {
            "id": "test",
            "error": "latest test failed",
        }
    )

    assert detail == "latest failed"


def test_failure_detail_removes_full_ansi_and_redacts_secrets():
    secret = "sk-0123456789abcdef"

    detail = tool_display.sanitize_tool_failure_detail(
        (
            "\x1b]8;;https://evil.example/private\x07click\x1b]8;;\x07 "
            "\x1b[31mAPI_TOKEN="
            f"{secret} timed out\x1b[0m\u202e"
        )
    )

    assert "timed out" in detail
    assert secret not in detail
    assert "evil.example" not in detail
    assert "\x1b" not in detail
    assert "\x07" not in detail
    assert "\u202e" not in detail
