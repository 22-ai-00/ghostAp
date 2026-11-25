"""Action dispatch: centralized action_id constants + action registries.

All action_id strings used in reducers, handlers, and action registries
are defined here. Registry factories map action_ids to CardEvent constructors
for CardSession injection.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.card.events import CardEvent

# ---------------------------------------------------------------------------
# Spec engine actions
# ---------------------------------------------------------------------------
SPEC_STOP = "spec_stop"  # Force-stop current Spec engine execution

# ---------------------------------------------------------------------------
# Generic engine actions
# ---------------------------------------------------------------------------
ENGINE_STOP = "engine_stop"  # Generic stop — routed by engine_type at dispatch time
PROGRAMMING_STOP = "programming_stop"  # Stop active ordinary programming turn without exiting mode
ENGINE_RESTART = "engine_restart"  # Restart engine after TTL timeout or completion
TTL_KEEP_ALIVE = "ttl_keep_alive"  # User requests to keep session alive (reset idle timer)
MODE_FULL = "mode_full"  # Switch card to full (detailed) view mode
MODE_COMPACT = "mode_compact"  # Switch card to compact (minimal) view mode

# ---------------------------------------------------------------------------
# Workflow actions
# ---------------------------------------------------------------------------
WORKFLOW_STOP_RUNNING = "workflow_stop_running"  # Stop a running workflow from the progress card
SHOW_WORKFLOW_MENU = "show_workflow_menu"  # Show Workflow usage from the global command menu
WORKFLOW_SELECT_TOOL = "workflow_select_tool"
WORKFLOW_SELECT_MODEL = "workflow_select_model"
WORKFLOW_SELECT_PROFILE = "workflow_select_profile"
WORKFLOW_SELECT_EFFORT = "workflow_select_effort"
WORKFLOW_ADD_AGENT = "workflow_add_agent"
WORKFLOW_ADD_RECOMMENDED_POOL = "workflow_add_recommended_pool"
WORKFLOW_REMOVE_AGENT = "workflow_remove_agent"
WORKFLOW_CLEAR_AGENTS = "workflow_clear_agents"
WORKFLOW_SET_ORCHESTRATOR = "workflow_set_orchestrator"
WORKFLOW_CONFIRM_AGENTS = "workflow_confirm_agents"

WORKFLOW_AGENT_SELECTION_ACTIONS = frozenset(
    {
        WORKFLOW_SELECT_TOOL,
        WORKFLOW_SELECT_MODEL,
        WORKFLOW_SELECT_PROFILE,
        WORKFLOW_SELECT_EFFORT,
        WORKFLOW_ADD_AGENT,
        WORKFLOW_ADD_RECOMMENDED_POOL,
        WORKFLOW_REMOVE_AGENT,
        WORKFLOW_CLEAR_AGENTS,
        WORKFLOW_SET_ORCHESTRATOR,
        WORKFLOW_CONFIRM_AGENTS,
    }
)

# Interactive Teamwork /wf_config Card 2.0 actions (F7/F10)
WF_OPEN_CONFIG = "wf_open_config"
WF_SELECT_PRESET = "wf_select_preset"
WF_SELECT_AGENT = "wf_select_agent"
WF_SELECT_ROLE = "wf_select_role"
WF_SELECT_TOOL = "wf_select_tool"
WF_SELECT_MODEL = "wf_select_model"
WF_SELECT_PROFILE = "wf_select_profile"
WF_SELECT_EFFORT = "wf_select_effort"
WF_ADD_OR_UPDATE_AGENT = "wf_add_or_update_agent"
WF_REMOVE_AGENT = "wf_remove_agent"
WF_SET_ORCHESTRATOR = "wf_set_orchestrator"
WF_SAVE_TEAM = "wf_save_team"
WF_RESET_TEAM = "wf_reset_team"
WF_START_WORKFLOW = "wf_start_workflow"

WF_CONFIG_ACTIONS = frozenset(
    {
        WF_OPEN_CONFIG,
        WF_SELECT_PRESET,
        WF_SELECT_AGENT,
        WF_SELECT_ROLE,
        WF_SELECT_TOOL,
        WF_SELECT_MODEL,
        WF_SELECT_PROFILE,
        WF_SELECT_EFFORT,
        WF_ADD_OR_UPDATE_AGENT,
        WF_REMOVE_AGENT,
        WF_SET_ORCHESTRATOR,
        WF_SAVE_TEAM,
        WF_RESET_TEAM,
        WF_START_WORKFLOW,
    }
)

# ---------------------------------------------------------------------------
# Explicit ACP model selection actions
# ---------------------------------------------------------------------------
SELECT_ACP_MODEL = "select_acp_model"
OPEN_ACP_MODEL_PICKER = "open_acp_model_picker"
REFRESH_ACP_MODELS = "refresh_acp_models"
SELECT_ACP_MODEL_GROUP = "select_acp_model_group"
SELECT_ACP_MODEL_PROFILE = "select_acp_model_profile"
SELECT_ACP_MODEL_EFFORT = "select_acp_model_effort"
SHOW_TOOL_COMMANDS = "show_tool_commands"
RUN_NATIVE_TOOL_COMMAND = "run_native_tool_command"

# ---------------------------------------------------------------------------
# Tool cockpit (/usage & /settings) actions
# ---------------------------------------------------------------------------
OPEN_TOOL_USAGE = "open_tool_usage"
REFRESH_TOOL_USAGE = "refresh_tool_usage"
OPEN_TOOL_SETTINGS = "open_tool_settings"
REFRESH_TOOL_SETTINGS = "refresh_tool_settings"
SETTINGS_SELECT_TOOL = "settings_select_tool"
SETTINGS_SELECT_MODEL_GROUP = "settings_select_model_group"
SETTINGS_SELECT_MODEL_PROFILE = "settings_select_model_profile"
SETTINGS_SELECT_MODEL_EFFORT = "settings_select_model_effort"
SETTINGS_TOGGLE_SHOW_THINKING = "settings_toggle_show_thinking"
SETTINGS_SELECT_VERBOSITY = "settings_select_verbosity"
SETTINGS_SELECT_AGENT_MODE = "settings_select_agent_mode"
SETTINGS_CLEAR_SESSION = "settings_clear_session"

TOOL_COCKPIT_ACTIONS = frozenset(
    {
        OPEN_TOOL_USAGE,
        REFRESH_TOOL_USAGE,
        OPEN_TOOL_SETTINGS,
        REFRESH_TOOL_SETTINGS,
        SETTINGS_SELECT_TOOL,
        SETTINGS_SELECT_MODEL_GROUP,
        SETTINGS_SELECT_MODEL_PROFILE,
        SETTINGS_SELECT_MODEL_EFFORT,
        SETTINGS_TOGGLE_SHOW_THINKING,
        SETTINGS_SELECT_VERBOSITY,
        SETTINGS_SELECT_AGENT_MODE,
        SETTINGS_CLEAR_SESSION,
    }
)

# ---------------------------------------------------------------------------
# Global / status actions
# ---------------------------------------------------------------------------
EXIT_PROGRAMMING_MODE = "exit"  # Exit whichever programming backend is active
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
HELP_CATEGORY = "help_category"  # Navigate to a specific help category

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
    from src.card.events import CardEvent, CardEventType

    return {
        MODE_FULL: lambda p: CardEvent.mode_toggled(compact=False),
        MODE_COMPACT: lambda p: CardEvent.mode_toggled(compact=True),
        ENGINE_STOP: lambda p: CardEvent(type=CardEventType.STOPPING),
        PROGRAMMING_STOP: lambda p: CardEvent(type=CardEventType.STOPPING),
    }
