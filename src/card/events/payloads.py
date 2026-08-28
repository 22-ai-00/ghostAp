"""Card event payload TypedDict definitions."""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

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


class ToolDonePayload(TypedDict):
    """Payload for TOOL_DONE event."""
    block_id: str
    tool_output: NotRequired[str]
    tool_summary: NotRequired[str]
    tool_name: NotRequired[str]
    _opaque_raw_tool_call_id: NotRequired[str]


class ToolFailedPayload(TypedDict):
    """Payload for TOOL_FAILED event."""
    block_id: str
    error: str
    tool_name: NotRequired[str]
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
# Allowed fields for the remaining Workflow stop callback.
_WORKFLOW_BUTTON_FIELDS: set[str] = {
    "action",
    "chat_id",
    "project_id",
}


def filter_workflow_button_value(value: dict[str, Any]) -> dict[str, Any]:
    """Return a button-value dict stripped of any field not in the schema.

    Safety: a Feishu callback can carry arbitrary keys injected into the
    button payload by a compromised client. We strip unknown fields at the
    handler boundary so downstream code cannot read forged fields such as
    ``"confirmed"``, ``"admin"``, or ``"override_budget"``.
    """
    if not isinstance(value, dict):
        return {}
    return {k: v for k, v in value.items() if k in _WORKFLOW_BUTTON_FIELDS}
