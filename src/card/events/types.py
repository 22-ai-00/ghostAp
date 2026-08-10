"""Card event type enumeration."""

from enum import Enum


class CardEventType(str, Enum):
    """All card event types."""
    # Lifecycle
    STARTED = "started"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ARCHIVED = "archived"
    PAUSED = "paused"
    RESUMED = "resumed"
    BLOCKED = "blocked"
    # Content
    TEXT_STARTED = "text_started"
    TEXT_DELTA = "text_delta"
    TEXT_DONE = "text_done"
    REASONING_STARTED = "reasoning_started"
    REASONING_DELTA = "reasoning_delta"
    REASONING_DONE = "reasoning_done"
    TOOL_STARTED = "tool_started"
    TOOL_DELTA = "tool_delta"
    TOOL_DONE = "tool_done"
    TOOL_FAILED = "tool_failed"
    IMAGE_ADDED = "image_added"
    IMAGE_FAILED = "image_failed"
    PLAN_UPDATED = "plan_updated"
    # Meta
    TOOL_MODEL_CHANGED = "tool_model_changed"
    PROGRESS_UPDATED = "progress_updated"
    # Engine lifecycle (Spec)
    CYCLE_STARTED = "cycle_started"
    CYCLE_DONE = "cycle_done"
    PHASE_STARTED = "phase_started"
    PHASE_DONE = "phase_done"
    SPEC_PLAN_UPDATED = "spec_plan_updated"
    SPEC_TASKS_UPDATED = "spec_tasks_updated"
    REVIEW_RESULT_UPDATED = "review_result_updated"
    REVIEW_RETRY = "review_retry"
    CRITERIA_UPDATED = "criteria_updated"
    WARNING_UPDATED = "warning_updated"
    # UI control
    MODE_TOGGLED = "mode_toggled"
    STOP_ESCALATED = "stop_escalated"
    # Task-level card management
    TASK_LIST_UPDATED = "task_list_updated"
    SECTION_SEPARATOR = "section_separator"
    CARD_SPLIT = "card_split"
