"""Action dispatch: centralized action_id constants + action registries.

All action_id strings used in reducers, handlers, and action registries
are defined here. Registry factories map action_ids to CardEvent constructors
for CardSession injection.
"""
from __future__ import annotations

from collections.abc import Callable

from src.card.events import CardEvent, CardEventType

# ---------------------------------------------------------------------------
# Approval actions
# ---------------------------------------------------------------------------
APPROVE_ACTION = "approve_action"  # User approves a pending approval request
REJECT_ACTION = "reject_action"  # User rejects a pending approval request

# ---------------------------------------------------------------------------
# Spec engine actions
# ---------------------------------------------------------------------------
SPEC_STOP = "spec_stop"  # Force-stop current Spec engine execution
SPEC_SKIP_RETRY = "spec_skip_retry"  # Skip retry and accept current cycle result
SPEC_RESUME = "spec_resume"  # Resume/retry Spec engine after failure
SPEC_REVIEW_USE_AUTO = "spec_review_use_auto"  # Keep Spec review on the main agent/model
SPEC_REVIEW_FINISH_SELECTION = "spec_review_finish_selection"  # Confirm selected Spec review agents
SPEC_REVIEW_SELECT_TOOL = "spec_review_select_tool"  # Select a tool for Spec multi-agent review
SPEC_REVIEW_SELECT_MODEL = "spec_review_select_model"  # Select a model for pending Spec review tool
SPEC_REVIEW_REMOVE_ITEM = "spec_review_remove_item"  # Remove a selected Spec review agent
SPEC_REVIEW_CLEAR_ITEMS = "spec_review_clear_items"  # Clear selected Spec review agents
SPEC_RESTORE_RUN = "spec_restore_run"  # Restore a persisted Spec run from cache
SHOW_SPEC_REVIEW_MENU = "show_spec_review_menu"  # Return to Spec review tool selection

# ---------------------------------------------------------------------------
# Deep engine actions
# ---------------------------------------------------------------------------
DEEP_RESUME = "deep_resume"  # Resume/retry Deep engine after failure
DEEP_STOP = "deep_stop"  # Force-stop current Deep engine execution

# ---------------------------------------------------------------------------
# Generic engine actions
# ---------------------------------------------------------------------------
ENGINE_STOP = "engine_stop"  # Generic stop — routed by engine_type at dispatch time
ENGINE_RESTART = "engine_restart"  # Restart engine after TTL timeout or completion
TTL_KEEP_ALIVE = "ttl_keep_alive"  # User requests to keep session alive (reset idle timer)
MODE_FULL = "mode_full"  # Switch card to full (detailed) view mode
MODE_COMPACT = "mode_compact"  # Switch card to compact (minimal) view mode

# ---------------------------------------------------------------------------
# Workflow actions
# ---------------------------------------------------------------------------
# Design decision: Workflow card interactions are handled directly by the
# WorkflowHandler (src/feishu/handlers/workflow.py) rather than through the
# CardSession event pipeline. This is intentional — Workflow's confirmation
# and tool selection cards are built and updated by the handler layer without
# a build_workflow_action_registry() factory. The rationale is that Workflow's
# execution model (isolated JS runtime + bridge) differs from Spec's
# CardSession-driven lifecycle, and adding a registry would require a
# CardSession instance that doesn't naturally fit the workflow's fire-and-forget
# execution pattern.
WORKFLOW_CONFIRM_TOOLS = "workflow_confirm_tools"  # Confirm tool selection and generate script
WORKFLOW_CONFIRM_START = "workflow_confirm_start"  # Confirm generated script and start execution
WORKFLOW_CANCEL = "workflow_cancel"  # Cancel workflow before execution starts
WORKFLOW_STOP_RUNNING = "workflow_stop_running"  # Stop a running workflow from the progress card
WORKFLOW_SELECT_TOOL = "workflow_select_tool"  # Toggle tool selection for workflow execution
WORKFLOW_REGENERATE_SCRIPT = "workflow_regenerate_script"  # Regenerate script with current tool selection
WORKFLOW_FILL_MISSING_TOOLS = "workflow_fill_missing_tools"  # Auto-add tools the script needs but user hasn't selected
WORKFLOW_BACK_TO_TOOLS = "workflow_back_to_tools"  # Return to tool selection screen
WORKFLOW_VIEW_WORKFLOW_REF = "workflow_view_workflow_ref"  # View details of a workflow reference
WORKFLOW_REMOVE_WORKFLOW_REF = "workflow_remove_workflow_ref"  # Remove a workflow reference
WORKFLOW_ADD_WORKFLOW_REF = "workflow_add_workflow_ref"  # Add a workflow reference
# Workflow orchestrator selection (two-step flow, combined card)
WORKFLOW_ORCHESTRATOR_SELECT_TOOL = "workflow_orchestrator_select_tool"
WORKFLOW_ORCHESTRATOR_SELECT_MODEL_GROUP = "workflow_orchestrator_select_model_group"
WORKFLOW_ORCHESTRATOR_SELECT_MODEL_PROFILE = "workflow_orchestrator_select_model_profile"
WORKFLOW_ORCHESTRATOR_SELECT_MODEL_EFFORT = "workflow_orchestrator_select_model_effort"
WORKFLOW_ORCHESTRATOR_SELECT_MODEL = "workflow_orchestrator_select_model"
WORKFLOW_ORCHESTRATOR_REMOVE = "workflow_orchestrator_remove"
WORKFLOW_ORCHESTRATOR_CLEAR = "workflow_orchestrator_clear"
WORKFLOW_ORCHESTRATOR_FINISH = "workflow_orchestrator_finish"
# Workflow review agent selection
WORKFLOW_REVIEW_SELECT_TOOL = "workflow_review_select_tool"
WORKFLOW_REVIEW_SELECT_MODEL_GROUP = "workflow_review_select_model_group"
WORKFLOW_REVIEW_SELECT_MODEL_PROFILE = "workflow_review_select_model_profile"
WORKFLOW_REVIEW_SELECT_MODEL_EFFORT = "workflow_review_select_model_effort"
WORKFLOW_REVIEW_SELECT_MODEL = "workflow_review_select_model"
WORKFLOW_REVIEW_FINISH = "workflow_review_finish"
WORKFLOW_REVIEW_REMOVE = "workflow_review_remove"
WORKFLOW_REVIEW_CLEAR = "workflow_review_clear"
WORKFLOW_REVIEW_TOGGLE_AUTO = "workflow_review_toggle_auto"
SHOW_WORKFLOW_MENU = "show_workflow_menu"  # Show workflow menu / start workflow flow
WORKFLOW_LIST_TEMPLATES = "workflow_list_templates"  # List available workflow templates
WORKFLOW_SHOW_HELP = "workflow_show_help"  # Show workflow help

# ---------------------------------------------------------------------------
# Global / status actions
# ---------------------------------------------------------------------------
SHOW_STATUS = "show_status"  # Show current session status card
SHOW_BOARD = "show_board"  # Show project dashboard
REFRESH_BOARD = "refresh_board"  # Refresh dashboard data
SWITCH_PROJECT = "switch_project"  # Switch active project context
SWITCH_BOARD_PAGE = "switch_board_page"  # Navigate between dashboard pages
SHOW_DETAIL = "show_detail"  # Show detailed info for a specific item
SWITCH_TO = "switch_to"  # Switch to a different programming mode
CONTINUE_DEV = "continue_dev"  # Continue development in current session
LIST_FILES = "list_files"  # List project files
NEW_PROJECT_PROMPT = "new_project_prompt"  # Prompt user to create a new project
SHOW_HELP_MENU = "show_help_menu"  # Display help menu card
ENTER_DEEP_PROMPT = "enter_deep_prompt"  # Quick-enter Deep mode from status card
SHOW_DEEP_STATUS = "show_deep_status"  # Show Deep engine execution status
RETRY_COMMAND = "retry_command"  # Retry the last failed command
CONTINUE_DEGRADED = "continue_degraded"  # Continue with the available degraded capability
SHOW_ERROR_DETAILS = "show_error_details"  # Show diagnostics for a degraded/recoverable error card
RETRY_ORIGINAL = "retry_original"  # Retry the original mode/action that produced a degraded card
HELP_CATEGORY = "help_category"  # Navigate to a specific help category

# ---------------------------------------------------------------------------
# ACP actions
# ---------------------------------------------------------------------------
SHOW_ACP_MENU = "show_acp_menu"  # Show ACP tool/model selection menu
SELECT_ACP_TOOL = "select_acp_tool"  # Select an ACP-capable tool
SELECT_ACP_MODEL = "select_acp_model"  # Select a model for the ACP tool
REFRESH_ACP_MODELS = "refresh_acp_models"  # Refresh available ACP model list
# Cascade model-select dropdowns for normal programming mode. Changing the
# family/profile/effort dropdown only re-renders the card (does not enter the
# mode); the final SELECT_ACP_MODEL button commits the choice.
SELECT_ACP_MODEL_GROUP = "select_acp_model_group"  # Model-family dropdown change
SELECT_ACP_MODEL_PROFILE = "select_acp_model_profile"  # Profile dropdown change
SELECT_ACP_MODEL_EFFORT = "select_acp_model_effort"  # Effort dropdown change

# ---------------------------------------------------------------------------
# Team actions
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Lock actions
# ---------------------------------------------------------------------------
FORCE_RELEASE_REPO_LOCK = "force_release_repo_lock"  # Force-release repo lock (may interrupt other user)
CONFIRM_LOCK = "confirm_lock"  # Confirm acquiring a contested lock
CANCEL_LOCK = "cancel_lock"  # Cancel lock acquisition request
CONFIRM_FORCE_RELEASE = "confirm_force_release"  # Double-confirm force release (danger action)
CANCEL_FORCE_RELEASE = "cancel_force_release"  # Cancel force release request


def build_common_action_registry() -> dict[str, Callable[[dict], CardEvent]]:
    """Build action registry entries shared across all engine sessions.

    Includes: mode toggle, generic engine stop.
    """
    return {
        MODE_FULL: lambda p: CardEvent.mode_toggled(compact=False),
        MODE_COMPACT: lambda p: CardEvent.mode_toggled(compact=True),
        ENGINE_STOP: lambda p: CardEvent(type=CardEventType.STOPPING),
    }
