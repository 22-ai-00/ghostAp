"""Transport-aware availability contracts for programming tool discovery."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.acp.helper import list_acp_tools


def _tool_names() -> list[str]:
    return [tool.name for tool in list_acp_tools()]


def test_claude_cli_on_path_is_available_without_acp_probe_or_auto_update() -> None:
    auto_update = MagicMock(return_value=True)
    claude_acp_check = MagicMock(
        side_effect=lambda: auto_update("claude")
    )
    codex_acp_check = MagicMock(return_value=False)
    registry_availability = MagicMock(
        side_effect=lambda name, **_kwargs: name == "codex"
    )

    with patch(
        "src.acp.helper.get_providers",
        return_value={
            "claude": SimpleNamespace(check_availability=claude_acp_check),
            "codex": SimpleNamespace(check_availability=codex_acp_check),
        },
    ), patch(
        "shutil.which",
        side_effect=lambda name: "/usr/bin/claude" if name == "claude" else None,
    ), patch(
        "src.acp.providers.tool_registry.get_availability",
        side_effect=registry_availability,
    ):
        names = _tool_names()

    assert names == ["claude", "codex"]
    claude_acp_check.assert_not_called()
    auto_update.assert_not_called()
    codex_acp_check.assert_not_called()
    assert [call.args[0] for call in registry_availability.call_args_list] == [
        "codex"
    ]


def test_missing_claude_cli_is_unavailable_without_acp_probe_or_auto_update() -> None:
    auto_update = MagicMock(return_value=True)
    claude_acp_check = MagicMock(
        side_effect=lambda: auto_update("claude")
    )
    registry_availability = MagicMock(
        side_effect=lambda name, **_kwargs: name == "codex"
    )

    with patch(
        "src.acp.helper.get_providers",
        return_value={
            "claude": SimpleNamespace(check_availability=claude_acp_check),
            "codex": SimpleNamespace(check_availability=MagicMock(return_value=False)),
        },
    ), patch(
        "shutil.which",
        return_value=None,
    ), patch(
        "src.acp.providers.tool_registry.get_availability",
        side_effect=registry_availability,
    ):
        names = _tool_names()

    assert names == ["codex"]
    claude_acp_check.assert_not_called()
    auto_update.assert_not_called()
    assert [call.args[0] for call in registry_availability.call_args_list] == [
        "codex"
    ]
