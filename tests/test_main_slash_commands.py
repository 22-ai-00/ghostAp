"""Main Bot Slash discovery contracts."""

from src.feishu.main_slash_commands import MAIN_AGENT_COMMANDS
from src.feishu.product_catalog import resolve_command


def test_spec_export_is_compatible_but_not_advertised_as_chat_export() -> None:
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

    assert registered & spec_controls == spec_controls - {"/spec_export"}
    assert resolve_command("/spec_export") is not None
