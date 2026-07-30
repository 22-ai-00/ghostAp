"""Card event payload TypedDict definitions."""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

# ---------------------------------------------------------------------------
# Lifecycle payload TypedDicts
# ---------------------------------------------------------------------------

class CompletedPayload(TypedDict, total=False):
    """Payload for COMPLETED event."""
    summary: str
    duration_seconds: float


class FailedPayload(TypedDict):
    """Payload for FAILED event."""
    error: str
    duration_seconds: NotRequired[float]


class BlockedPayload(TypedDict, total=False):
    """Payload for BLOCKED event."""
    reason: str


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


class ToolDeltaPayload(TypedDict):
    """Payload for TOOL_DELTA event."""
    block_id: str
    content: str


class ToolDonePayload(TypedDict):
    """Payload for TOOL_DONE event."""
    block_id: str
    tool_output: str
    tool_summary: str


class ToolFailedPayload(TypedDict):
    """Payload for TOOL_FAILED event."""
    block_id: str
    error: str


class ImagePayload(TypedDict):
    """Payload for IMAGE_ADDED/IMAGE_FAILED events."""
    image_id: str
    image_key: NotRequired[str]
    alt: str


class PlanUpdatedPayload(TypedDict):
    """Payload for PLAN_UPDATED event."""
    content: str


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
# Worktree payload TypedDicts
# ---------------------------------------------------------------------------

class WorktreeToolItem(TypedDict, total=False):
    """Single tool item in worktree tool selection."""
    name: str
    display_name: str
    provider: str
    available: bool


class WorktreeMergeNote(TypedDict, total=False):
    """Single merge note describing a branch to merge."""
    branch: str
    worktree_path: str
    status: str
    summary: str
    unit_id: str


class WorktreeSelectedItem(TypedDict, total=False):
    """A selected tool-model combination for worktree execution."""
    tool: str
    model: str
    display_name: str


class WorktreeMergeResult(TypedDict, total=False):
    """Result of a single merge operation."""
    branch: str
    status: str
    message: str


class WorktreeUnitPayload(TypedDict, total=False):
    """Single work unit in worktree progress. Flexible schema — only 'status' is required."""
    status: str  # required (total=False still lets us annotate intent)
    name: str
    unit_id: str
    display_name: str
    summary: str


class WorktreeProgressPayload(TypedDict, total=False):
    """Payload for WORKTREE_PROGRESS event."""
    units: list[WorktreeUnitPayload]
    project_id: str
    message: str
    silent: bool
    iteration: int
    thread_root_id: str


class WorktreeToolSelectPayload(TypedDict):
    """Payload for WORKTREE_TOOL_SELECT event."""
    tools: list[dict]  # tool dicts vary by source (ACP vs TTADK)
    selected: list[str] | list[dict]
    project_id: str
    message: str
    select_action: NotRequired[str]
    pending_tool: NotRequired[str]
    thread_root_id: NotRequired[str]
    mode_label: NotRequired[str]
    tool_select_title: NotRequired[str]
    model_select_title: NotRequired[str]
    auto_action: NotRequired[str]
    auto_text: NotRequired[str]
    auto_description: NotRequired[str]
    finish_action: NotRequired[str]
    remove_action: NotRequired[str]
    clear_action: NotRequired[str]
    back_action: NotRequired[str]
    show_stepper: NotRequired[bool]
    # Model-select pagination: current page + tool identity so page-nav buttons
    # can re-trigger the tool-select action and re-fetch the model list.
    model_page: NotRequired[int]
    page_action: NotRequired[str]
    page_tool_name: NotRequired[str]
    page_provider: NotRequired[str]


class WorktreeConfirmPayload(TypedDict):
    """Payload for WORKTREE_CONFIRM event."""
    selected_items: list[dict]
    goal: str
    project_id: str
    message: str
    thread_root_id: NotRequired[str]


class WorktreeCleanupPayload(TypedDict):
    """Payload for WORKTREE_CLEANUP event."""
    merge_notes: list[dict]
    base_branch: str
    merge_results: NotRequired[list[dict] | None]
    project_id: str
    units: NotRequired[list[dict] | None]
    cleanup_phase: Literal["summary", "actions", "completed"]
    thread_root_id: NotRequired[str]


class WorktreeMergePayload(TypedDict):
    """Payload for WORKTREE_MERGE event."""
    merge_notes: list[dict]
    base_branch: str
    project_id: str
    thread_root_id: NotRequired[str]


class WorktreeCompletedNoChangePayload(TypedDict):
    """Payload for WORKTREE_COMPLETED_NO_CHANGE event."""
    units: list[WorktreeUnitPayload]
    project_id: str
    message: str
    iteration: NotRequired[int]
    thread_root_id: NotRequired[str]


# ---------------------------------------------------------------------------
# Task-level card management payload TypedDicts
# ---------------------------------------------------------------------------

class TaskSnapshotPayload(TypedDict):
    """Single task item in task list payload."""
    task_id: str
    name: str
    status: Literal["pending", "in_progress", "completed", "failed"]


class TaskListUpdatedPayload(TypedDict):
    """Payload for TASK_LIST_UPDATED event."""
    tasks: list[TaskSnapshotPayload]
    current_task_id: str


# ---------------------------------------------------------------------------
# Workflow engine payload TypedDicts
# ---------------------------------------------------------------------------

class WorkflowProgressPayload(TypedDict):
    """Payload for WORKFLOW_PROGRESS event — full card JSON from renderer.

    ``card`` is required; callers must always supply the rendered card JSON
    so downstream consumers can read payload["card"] without defensive fallbacks.
    ``compact_status`` is optional and carries a short human-readable summary.

    Deprecated fields (budget_consumed, budget_remaining): These are kept only for
    backwards compatibility with older clients. New code should not use these fields
    as budget control has been removed from Workflow mode. These fields will be
    removed in a future version.
    """
    card: dict  # Feishu card JSON with header + elements
    compact_status: NotRequired[str]
    # Deprecated: budget control has been removed. Kept for backwards compatibility.
    # These fields should always return 0, not None, to avoid client errors.
    # Will be removed in v2.0.
    budget_consumed: NotRequired[int]
    budget_remaining: NotRequired[int]


class WorkflowPhasePayload(TypedDict):
    """Payload for WORKFLOW_PHASE event."""
    title: str


class WorkflowAgentStartedPayload(TypedDict):
    """Payload for WORKFLOW_AGENT_STARTED event."""
    label: str
    tool: str
    phase: str


class WorkflowAgentDonePayload(TypedDict):
    """Payload for WORKFLOW_AGENT_DONE event."""
    label: str
    token_usage: NotRequired[int]
    duration_s: NotRequired[float]
    cached: NotRequired[bool]


class WorkflowAgentFailedPayload(TypedDict):
    """Payload for WORKFLOW_AGENT_FAILED event."""
    label: str
    error: str


class WorkflowLogPayload(TypedDict):
    """Payload for WORKFLOW_LOG event."""
    message: str


class WorkflowPhaseItem(TypedDict):
    """Structured phase item for WorkflowConfirmPayload.phases."""
    title: str
    detail: NotRequired[str]


class WorkflowRefItem(TypedDict, total=False):
    """Structured sub-workflow reference for WorkflowConfirmPayload.workflow_refs.

    Normalized contract: ``{ name, description?, args?, failure_policy? }``.
    Legacy string refs are supported for backward compatibility but should be
    converted to dict at boundaries. ``script_path`` is *not* accepted at
    runtime — the Python bridge resolves templates by name only so refs
    always go through validate_template_name + resolve_template_path.

    Fields:
      name (str, required): Template identifier. Matches a built-in name,
          a user template, a project template, or a global allowlisted
          template name (without the .js extension).
      description (str, optional): Human-readable description surfaced in
          the confirmation card and in injected comments.
      args (dict, optional): Arguments forwarded to the sub-workflow as
          ``workflow(name, args)``. Must be JSON-serializable.
      failure_policy (str, optional): One of ``"skip"`` (default — catch
          and continue) or ``"fail_fast"`` (let the exception propagate to
          the parent script).
    """

    name: str
    description: NotRequired[str]
    args: NotRequired[dict[str, Any]]
    failure_policy: NotRequired[str]
    # Legacy path/hash fields are preserved here to gracefully parse older card
    # payloads that still embed them. They are intentionally typed as NotRequired
    # and callers must not rely on them for sub-workflow resolution.
    path: NotRequired[str]
    hash: NotRequired[str]


class WorkflowRefItemLegacy(TypedDict, total=False):
    """Legacy sub-workflow reference payload fields — kept as a distinct
    type for reading older payloads without polluting the canonical schema.
    """

    name: str
    path: str
    hash: str


class WorkflowConfirmPayloadRequired(TypedDict):
    """Required fields for WORKFLOW_CONFIRM event.

    Security-critical: initiator_user_id and engine_session_key are required
    to prevent cross-user confirmation hijacking.
    """
    script_name: str
    description: str
    phases: list["WorkflowPhaseItem"]
    tools: list[str]
    requirement: str
    initiator_user_id: str
    engine_session_key: str


class WorkflowConfirmPayloadOptional(TypedDict, total=False):
    """Optional fields for WORKFLOW_CONFIRM event."""
    project_id: str
    chat_id: str
    is_fallback: bool
    workflow_refs: list["WorkflowRefItem"]
    dependency_graph: dict  # {phase: [dep_phases]}
    phase_tool_mapping: dict  # {phase: [tools]}
    script_preview: str  # truncated script for user review
    # Deprecated: budget control has been removed. Kept for backwards compatibility.
    # Should always return 0, not None. Will be removed in v2.0.
    budget_total: int


class WorkflowConfirmPayload(WorkflowConfirmPayloadRequired, WorkflowConfirmPayloadOptional):
    """Payload for WORKFLOW_CONFIRM event — preview before execution.

    Security-critical: initiator_user_id and engine_session_key are required
    to prevent cross-user confirmation hijacking.
    """

    pass


# ---------------------------------------------------------------------------
# Workflow button callback payload TypedDicts
#
# These describe the ``value`` dict attached to every Feishu button on a
# workflow confirm card. Handlers MUST validate incoming callback payloads
# against these schemas; any unexpected field should be ignored to prevent
# injection of forged fields (e.g. ``value.confirmed``).
# ---------------------------------------------------------------------------


class _WorkflowBaseButtonValueRequired(TypedDict):
    """Fields required on every workflow button at the type level.

    Separating required fields into their own TypedDict gives us type-level
    enforcement of action/chat_id/project_id/engine_session_key — omitting
    any of them will now surface as a type-checking error, not just a
    runtime warning. Downstream button values inherit both the required
    fields and the optional root_id via ``_WorkflowBaseButtonValue`` below.
    """

    action: str
    chat_id: str
    project_id: str
    engine_session_key: str


class _WorkflowBaseButtonValueOptional(TypedDict, total=False):
    """Fields that may or may not appear on a workflow button.

    ``root_id`` carries the stable filesystem root associated with the
    pending engine. It is populated when the originating button is aware
    of the root path (e.g. agent-selection cards initialized before any
    project context is available); otherwise it is omitted.
    """

    root_id: str


class _WorkflowBaseButtonValue(
    _WorkflowBaseButtonValueRequired,
    _WorkflowBaseButtonValueOptional,
):
    """Combined TypedDict for workflow button callback values.

    Inherits the required fields (action/chat_id/project_id/engine_session_key)
    from :class:`_WorkflowBaseButtonValueRequired` and the optional
    ``root_id`` from :class:`_WorkflowBaseButtonValueOptional`. This
    mirrors the runtime expectation: every callback must identify the
    originating chat, project, and engine session, while root_id is
    filled in only when the caller knows the filesystem root.
    """

    pass


class WorkflowSelectToolButtonValue(_WorkflowBaseButtonValue):
    """``WORKFLOW_SELECT_TOOL`` button payload."""

    tool_name: str


class WorkflowGenericButtonValue(_WorkflowBaseButtonValue):
    """Shared confirm/cancel/regen/back-to-tools payload shape.

    Matches buttons: CONFIRM_START, CANCEL, REGENERATE_SCRIPT,
    FILL_MISSING_TOOLS, BACK_TO_TOOLS.
    """

    pass


class WorkflowConfirmCardValue(TypedDict, total=False):
    """Union type annotation for a workflow button ``value`` dict.

    Fields mirror ``_WORKFLOW_BUTTON_FIELDS`` below so the button parser and
    the TypedDict contract cannot drift. Adding a new field here requires
    adding the same key to ``_WORKFLOW_BUTTON_FIELDS`` — otherwise the
    filter helper will strip it.
    """

    action: str
    chat_id: str
    project_id: str
    engine_session_key: str
    root_id: str
    tool_name: str
    role_id: str
    initiator_user_id: str
    ref_index: int
    template_name: str
    # Orchestrator combined card fields (tool+model selection, remove/clear)
    provider: str
    display_name: str
    supports_model: bool
    model_name: str
    name: str
    use_default_model: bool
    _option: str
    selection_key: str
    model_page: int


# ---------------------------------------------------------------------------
# Specialised per-action workflow button TypedDicts
#
# These narrow :class:`WorkflowConfirmCardValue` down to the fields relevant
# for a single button action. They give callers precise typing for callback
# payloads without requiring the broad ``WorkflowConfirmCardValue`` every-
# where. Keep each subset consistent with the fields actually attached by
# the matching button builder in :mod:`src.card.render.buttons`.
# ---------------------------------------------------------------------------


class WorkflowOrchestratorSelectToolButtonValue(_WorkflowBaseButtonValue):
    """``WORKFLOW_ORCHESTRATOR_SELECT_TOOL`` button payload."""

    tool_name: str
    provider: str
    display_name: str
    supports_model: bool
    selection_key: str
    model_page: int


class WorkflowOrchestratorSelectModelButtonValue(_WorkflowBaseButtonValue):
    """``WORKFLOW_ORCHESTRATOR_SELECT_MODEL`` button payload."""

    model_name: str
    name: str
    model_group: str
    model_profile: str
    model_effort: str
    use_default_model: bool
    _option: str
    selection_key: str


class WorkflowOrchestratorRemoveButtonValue(_WorkflowBaseButtonValue):
    """``WORKFLOW_ORCHESTRATOR_REMOVE`` button payload."""

    selection_key: str


class WorkflowReviewSelectToolButtonValue(_WorkflowBaseButtonValue):
    """``WORKFLOW_REVIEW_SELECT_TOOL`` button payload."""

    tool_name: str
    provider: str
    display_name: str
    supports_model: bool
    selection_key: str
    model_page: int


class WorkflowReviewSelectModelButtonValue(_WorkflowBaseButtonValue):
    """``WORKFLOW_REVIEW_SELECT_MODEL`` button payload."""

    model_name: str
    name: str
    model_group: str
    model_profile: str
    model_effort: str
    use_default_model: bool
    _option: str
    selection_key: str


class WorkflowReviewRemoveButtonValue(_WorkflowBaseButtonValue):
    """``WORKFLOW_REVIEW_REMOVE`` button payload."""

    selection_key: str


# ``_WORKFLOW_BUTTON_FIELDS`` is the single source of truth for allowed
# button payload keys. It mirrors :class:`WorkflowConfirmCardValue` 1:1 — any
# new field must appear here and in the TypedDict. This set is consumed by
# :func:`filter_workflow_button_value` to drop unknown payload keys.
_WORKFLOW_BUTTON_FIELDS: set[str] = {
    "action",
    "chat_id",
    "project_id",
    "engine_session_key",
    "root_id",
    "tool_name",
    "role_id",
    "initiator_user_id",
    "ref_index",
    "template_name",
    # Orchestrator combined card fields (tool+model selection, remove/clear)
    "provider",
    "display_name",
    "supports_model",
    "model_name",
    "model_group",
    "model_profile",
    "model_effort",
    "name",
    "use_default_model",
    "_option",
    "selection_key",
    "model_page",
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
