"""Main Bot Slash discovery contracts."""

from src.feishu.main_slash_commands import MAIN_AGENT_COMMANDS
from src.feishu.product_catalog import (
    HIDDEN_SLASH_COMMANDS,
    get_slash_discoverable_actions,
    resolve_command,
)

# The deliberately trimmed Feishu slash panel: the ~19 commands owners use
# day-to-day. Everything else stays reachable when typed explicitly but is
# hidden from auto-completion.
EXPECTED_PANEL_COMMANDS = {
    "/help",
    "/menu",
    "/coco",
    "/claude",
    "/aiden",
    "/codex",
    "/gemini",
    "/traex",
    "/grok",
    "/dsh",
    "/model",
    "/exit",
    "/projects",
    "/new",
    "/switch",
    "/status",
    "/deep",
    "/spec",
    "/wf",
}


def test_slash_panel_is_trimmed_to_common_commands() -> None:
    registered = {command.name for command in MAIN_AGENT_COMMANDS}
    discoverable = {action.command for action in get_slash_discoverable_actions()}

    assert discoverable == EXPECTED_PANEL_COMMANDS
    assert len(discoverable) == 19
    # MAIN_AGENT_COMMANDS is a pure projection of the discoverable set.
    assert registered == discoverable


def test_auxiliary_spec_controls_are_hidden_but_still_resolve() -> None:
    registered = {command.name for command in MAIN_AGENT_COMMANDS}
    spec_controls = {
        "/spec",
        "/spec_status",
        "/spec_history",
        "/spec_metrics",
        "/spec_config",
        "/spec_export",
        "/spec_save",
        "/spec_guide",
        "/stop_spec",
    }

    # Only the /spec entry point remains in the panel ...
    assert registered & spec_controls == {"/spec"}
    # ... while every auxiliary stays reachable when typed explicitly.
    for command in spec_controls:
        assert resolve_command(command) is not None


def test_hidden_commands_remain_typable_but_leave_the_panel() -> None:
    for command in HIDDEN_SLASH_COMMANDS:
        resolved = resolve_command(command)
        assert resolved is not None, command
        assert resolved.action.slash_discoverable is False
