"""Action dispatch: centralized action_id constants + action registries.

All action_id strings used in reducers, handlers, and action registries
are defined here. Registry factories map action_ids to CardEvent constructors
for CardSession injection.
"""
from __future__ import annotations

from collections.abc import Callable

from src.card.events import CardEvent, CardEventType

# ---------------------------------------------------------------------------
# Spec engine actions
# ---------------------------------------------------------------------------
SPEC_STOP = "spec_stop"  # Force-stop current Spec engine execution

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
WORKFLOW_STOP_RUNNING = "workflow_stop_running"  # Stop a running workflow from the progress card
SHOW_WORKFLOW_MENU = "show_workflow_menu"  # Show Workflow usage from the global command menu

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
    return {
        MODE_FULL: lambda p: CardEvent.mode_toggled(compact=False),
        MODE_COMPACT: lambda p: CardEvent.mode_toggled(compact=True),
        ENGINE_STOP: lambda p: CardEvent(type=CardEventType.STOPPING),
    }
