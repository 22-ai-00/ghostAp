"""Card event payload TypedDict definitions."""

from __future__ import annotations

import re
from typing import Any, Literal, NotRequired, TypedDict

from src.card.actions.dispatch import WORKFLOW_STOP_RUNNING

# Private transport metadata carried with logicalized tool events.  It is not
# rendered by CardState; process transports use it only to redact a provider's
# raw tool-call identifier from outbound content.
OPAQUE_TOOL_CALL_ID_KEY = "_opaque_raw_tool_call_id"

# ---------------------------------------------------------------------------
# Lifecycle payload TypedDicts
# ---------------------------------------------------------------------------

class CompletedPayload(TypedDict, total=False):
    """Payload for COMPLETED event."""
    summary: str
    duration_seconds: float
    warning: str
    details: str
    detail_action: dict[str, Any]


class FailedPayload(TypedDict):
    """Payload for FAILED event."""
    error: str
    details: NotRequired[str]
    detail_action: NotRequired[dict[str, Any]]
    retry_action: NotRequired[dict[str, Any]]
    duration_seconds: NotRequired[float]


# ---------------------------------------------------------------------------
# Content block payload TypedDicts
# ---------------------------------------------------------------------------

class TextBlockPayload(TypedDict):
    """Payload for TEXT_STARTED/TEXT_DELTA/TEXT_DONE events."""
    block_id: str
    text: NotRequired[str]
    source_kind: NotRequired[Literal["main", "subagent"]]
    source_sequence: NotRequired[str | None]
    source_label: NotRequired[str | None]
    source_ref: NotRequired[str]


class ReasoningBlockPayload(TypedDict):
    """Payload for REASONING_STARTED/REASONING_DELTA/REASONING_DONE events."""
    block_id: str
    text: NotRequired[str]


class ToolStartedPayload(TypedDict):
    """Payload for TOOL_STARTED event."""
    block_id: str
    tool_name: str
    tool_input: str
    _opaque_raw_tool_call_id: NotRequired[str]


class ToolDeltaPayload(TypedDict):
    """Payload for TOOL_DELTA event."""
    block_id: str
    content: str
    _opaque_raw_tool_call_id: NotRequired[str]


class ImagePayload(TypedDict):
    """Payload for IMAGE_ADDED/IMAGE_FAILED events."""
    image_id: str
    image_key: NotRequired[str]
    alt: str


# ---------------------------------------------------------------------------
# Meta payload TypedDicts
# ---------------------------------------------------------------------------

class ToolModelChangedPayload(TypedDict, total=False):
    """Payload for TOOL_MODEL_CHANGED event."""
    tool_name: str | None
    model_name: str | None
    unit_label: str | None
    live_ticker_frame: str | None
    subagents: tuple[dict, ...]


class ProgressPayload(TypedDict):
    """Payload for PROGRESS_UPDATED event."""
    current: int
    total: int
    label: str


class CardSplitPayload(TypedDict):
    """Payload for CARD_SPLIT event."""
    reason: str
    hint: str
    bridge_phrase: NotRequired[str]


# ---------------------------------------------------------------------------
# Engine lifecycle payload TypedDicts
# ---------------------------------------------------------------------------

class CycleStartedPayload(TypedDict):
    """Payload for CYCLE_STARTED event."""
    cycle_num: int
    max_cycles: int


class CycleDonePayload(TypedDict):
    """Payload for CYCLE_DONE event."""
    cycle_num: int
    status: str


class PhaseStartedPayload(TypedDict):
    """Payload for PHASE_STARTED event."""
    cycle_num: int
    phase: str
    subtitle: NotRequired[str]
    content: NotRequired[str]


class PhaseDonePayload(TypedDict):
    """Payload for PHASE_DONE event."""
    cycle_num: int
    phase: str
    output: str
    subtitle: NotRequired[str]


class SpecPlanUpdatedPayload(TypedDict):
    """Payload for SPEC_PLAN_UPDATED event."""
    cycle_num: int
    plan: dict


class SpecTasksUpdatedPayload(TypedDict):
    """Payload for SPEC_TASKS_UPDATED event."""
    cycle_num: int
    tasks: list[dict]


class ReviewResultUpdatedPayload(TypedDict):
    """Payload for REVIEW_RESULT_UPDATED event."""
    cycle_num: int
    roles: list[dict]


class ReviewRetryPayload(TypedDict):
    """Payload for REVIEW_RETRY event."""
    cycle_num: int
    attempt: int
    max_attempts: int
    status: str
    delay_sec: float


class CriteriaUpdatedPayload(TypedDict):
    """Payload for CRITERIA_UPDATED event."""
    content: str
    satisfied_count: int
    total_count: int


class WarningPayload(TypedDict):
    """Payload for WARNING_UPDATED event."""
    warning: str


# ---------------------------------------------------------------------------
# Task-level card management payload TypedDicts
# ---------------------------------------------------------------------------

class TaskSnapshotPayload(TypedDict):
    """Single task item in task list payload."""
    task_id: str
    name: str
    status: Literal["pending", "in_progress", "completed", "failed", "cancelled"]


# ---------------------------------------------------------------------------
# Allowed fields for Workflow card callbacks.
_WORKFLOW_BUTTON_FIELDS: frozenset[str] = frozenset(
    {
        "action",
        "chat_id",
        "project_id",
    }
)

_WORKFLOW_CONFIG_ID_FIELDS: frozenset[str] = frozenset(
    {
        "action",
        "chat_id",
        "project_id",
        "thread_root_id",
        "config_id",
        "config_session_id",
        "preset_id",
        "agent_id",
        "orchestrator_agent_id",
        "role",
        "tool_name",
        "profile",
        "effort",
        "model_profile",
        "model_effort",
    }
)

_WORKFLOW_CONFIG_TEXT_FIELDS: frozenset[str] = frozenset(
    {
        "model_name",
        "model_group",
        "display_name",
        "_option",
    }
)

_WORKFLOW_ACTION_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_WORKFLOW_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_WORKFLOW_STOP_STR_MAX = 128
_WORKFLOW_TEXT_STR_MAX = 256
_WORKFLOW_OPTIONS_MAX = 16


def _has_control_chars(text: str) -> bool:
    return any(
        ord(ch) < 32 or ord(ch) == 127 or 0xD800 <= ord(ch) <= 0xDFFF
        for ch in text
    )


def _sanitize_workflow_text(value: Any, *, max_len: int) -> str | None:
    if not isinstance(value, str):
        return None
    if not value or len(value) > max_len or _has_control_chars(value):
        return None
    return value


def _sanitize_workflow_id(value: Any) -> str | None:
    text = _sanitize_workflow_text(value, max_len=128)
    if text is None:
        return None
    if not _WORKFLOW_ID_RE.fullmatch(text):
        return None
    return text


def filter_workflow_button_value(value: dict[str, Any]) -> dict[str, Any]:
    """Return a button-value dict stripped of any field not in the schema.

    Safety: a Feishu callback can carry arbitrary keys injected into the
    button payload by a compromised client. We strip unknown fields,
    non-strings, control characters, and oversized strings at the handler
    boundary so downstream code cannot read forged fields such as
    ``"confirmed"``, ``"admin"``, ``"override_budget"``, or ``"shell_cmd"``.

    When ``action == WORKFLOW_STOP_RUNNING``, only ``action``, ``chat_id``,
    and ``project_id`` are permitted.
    """
    if not isinstance(value, dict):
        return {}

    raw_action = value.get("action")
    if raw_action is not None:
        if not isinstance(raw_action, str) or not _WORKFLOW_ACTION_RE.fullmatch(raw_action):
            return {}

    if raw_action == WORKFLOW_STOP_RUNNING:
        cleaned_stop: dict[str, Any] = {}
        for key in _WORKFLOW_BUTTON_FIELDS:
            if key not in value:
                continue
            item = value[key]
            if key == "action":
                cleaned_stop[key] = raw_action
            elif isinstance(item, str) and len(item) <= _WORKFLOW_STOP_STR_MAX and not _has_control_chars(item):
                cleaned_stop[key] = item
        return cleaned_stop

    cleaned: dict[str, Any] = {}
    for key, raw in value.items():
        if raw is None:
            continue
        if key in _WORKFLOW_CONFIG_ID_FIELDS:
            if key in {"chat_id", "project_id", "thread_root_id"}:
                safe_val = _sanitize_workflow_text(raw, max_len=_WORKFLOW_STOP_STR_MAX)
                if safe_val is not None and _WORKFLOW_ID_RE.fullmatch(safe_val):
                    cleaned[key] = safe_val
            else:
                safe_id = _sanitize_workflow_id(raw)
                if safe_id is not None:
                    cleaned[key] = safe_id
        elif key in _WORKFLOW_CONFIG_TEXT_FIELDS:
            safe_text = _sanitize_workflow_text(raw, max_len=_WORKFLOW_TEXT_STR_MAX)
            if safe_text is not None:
                cleaned[key] = safe_text
        elif key == "_options" and isinstance(raw, list):
            safe_opts: list[str] = []
            for item in raw[:_WORKFLOW_OPTIONS_MAX]:
                safe_item = _sanitize_workflow_text(item, max_len=_WORKFLOW_TEXT_STR_MAX)
                if safe_item is not None:
                    safe_opts.append(safe_item)
            if safe_opts:
                cleaned[key] = safe_opts
    return cleaned
