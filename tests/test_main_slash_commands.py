"""Main Bot Slash Command catalog and startup synchronization contracts."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.feishu.handlers.system import SystemHandler
from src.feishu.main_slash_commands import (
    MAIN_AGENT_COMMANDS,
    reconcile_main_agent_slash_commands,
)
from src.feishu.slash_command_parser import SlashCommandParser
from src.feishu.ws_client import FeishuWSClient
from src.slock_engine.slash_commands import (
    SlockCommandAction,
    parse_slock_command,
)
from src.workflow_engine.commands import TOPIC_ENGINE_COMMANDS

SYSTEM_COMMANDS = frozenset(
    {
        "/help",
        "/menu",
        "/coco",
        "/claude",
        "/aiden",
        "/codex",
        "/gemini",
        "/traex",
        "/ttadk",
        "/tui2acp",
        "/acp",
        "/model",
        "/exit",
        "/coco_status",
        "/coco_info",
        "/claude_info",
        "/aiden_info",
        "/codex_info",
        "/gemini_info",
        "/traex_info",
        "/ttadk_info",
        "/tui2acp_info",
        "/ttadk_refresh",
        "/tools",
        "/tools_status",
        "/projects",
        "/new",
        "/new-chat",
        "/switch",
        "/close",
        "/status",
        "/tasks",
        "/diff",
        "/trace",
        "/lock",
        "/unlock",
        "/setadmin",
        "/btw",
    }
)

DEEP_COMMANDS = frozenset(
    {
        "/deep",
        "/deep_status",
        "/deep_update",
        "/stop_deep",
    }
)

SPEC_COMMANDS = frozenset(
    {
        "/spec",
        "/spec_recover",
        "/spec_status",
        "/spec_history",
        "/spec_metrics",
        "/spec_config",
        "/spec_export",
        "/spec_save",
        "/stop_spec",
        "/spec_pause",
        "/spec_resume",
        "/spec_guide",
    }
)

WORKFLOW_COMMANDS = frozenset(
    {
        "/wf",
        "/wf_status",
        "/wf_help",
        "/stop_wf",
        "/wf_save",
        "/wf_list",
        "/wf_delete",
        "/wf_history",
    }
)

SLOCK_COMMANDS = frozenset(
    {
        "/slock",
        "/slocks",
        "/new-team",
        "/new-role",
        "/hire",
        "/fire",
        "/employees",
        "/history",
        "/employee-memory",
        "/council",
        "/discuss",
        "/memory",
        "/role",
        "/task",
        "/team",
        "/plan",
    }
)

EXPECTED_MAIN_AGENT_COMMANDS = (
    SYSTEM_COMMANDS
    | DEEP_COMMANDS
    | SPEC_COMMANDS
    | WORKFLOW_COMMANDS
    | SLOCK_COMMANDS
)


def _catalog_names() -> set[str]:
    return {f"/{item.canonical().command}" for item in MAIN_AGENT_COMMANDS}


def test_main_agent_catalog_covers_primary_supported_commands() -> None:
    assert _catalog_names() == EXPECTED_MAIN_AGENT_COMMANDS


def test_main_agent_catalog_is_unique_and_within_feishu_limit() -> None:
    canonical = [item.canonical() for item in MAIN_AGENT_COMMANDS]

    assert len(canonical) <= 100
    assert len({item.command for item in canonical}) == len(canonical)
    assert all(item.description for item in canonical)


def test_cataloged_system_commands_are_routable() -> None:
    for command in sorted(SYSTEM_COMMANDS):
        match = SlashCommandParser.parse(command)
        assert SystemHandler.is_interceptable_command_match(match), (
            f"{command!r} was not interceptable"
        )


def test_cataloged_workflow_commands_are_routable() -> None:
    for command in sorted(WORKFLOW_COMMANDS):
        assert command in TOPIC_ENGINE_COMMANDS, f"{command!r} was not routable"


def test_cataloged_slock_commands_are_routable() -> None:
    for command in sorted(SLOCK_COMMANDS):
        assert parse_slock_command(command).action is not SlockCommandAction.UNKNOWN, (
            f"{command!r} was not routable"
        )


@pytest.mark.asyncio
async def test_reconcile_uses_official_adapter_and_exact_main_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.feishu.main_slash_commands as module

    client = object()
    api = object()
    verified = object()
    reconciler = MagicMock()
    reconciler.reconcile = AsyncMock(return_value=verified)
    api_factory = MagicMock(return_value=api)
    reconciler_factory = MagicMock(return_value=reconciler)
    monkeypatch.setattr(module, "LarkSlashCommandAPI", api_factory)
    monkeypatch.setattr(module, "SlashCommandReconciler", reconciler_factory)

    result = await reconcile_main_agent_slash_commands(client)

    assert result is verified
    api_factory.assert_called_once_with(client)
    reconciler_factory.assert_called_once_with(
        api,
        desired=MAIN_AGENT_COMMANDS,
    )


def test_main_slash_sync_starts_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FeishuWSClient.__new__(FeishuWSClient)
    client._slash_command_sync_thread = None
    calls: list[str] = []
    monkeypatch.setattr(
        client,
        "_sync_main_slash_commands",
        lambda: calls.append("sync"),
        raising=False,
    )

    client._start_main_slash_command_sync()
    first_thread = client._slash_command_sync_thread
    first_thread.join(timeout=1)
    client._start_main_slash_command_sync()

    assert calls == ["sync"]
    assert client._slash_command_sync_thread is first_thread
    assert first_thread.name == "main-slash-command-sync"
    assert first_thread.daemon is True


def test_main_slash_sync_failure_is_actionable_and_does_not_propagate(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import src.feishu.ws_client as module

    client = FeishuWSClient.__new__(FeishuWSClient)
    client._get_api_client = MagicMock(return_value=object())
    monkeypatch.setattr(
        module,
        "reconcile_main_agent_slash_commands",
        AsyncMock(side_effect=RuntimeError("tenant-access-token=super-secret")),
        raising=False,
    )

    with caplog.at_level(logging.WARNING):
        client._sync_main_slash_commands()

    assert "application:app_slash_command:read" in caplog.text
    assert "application:app_slash_command:write" in caplog.text
    assert "super-secret" not in caplog.text


def test_main_slash_sync_logs_verified_counts(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import src.feishu.ws_client as module

    client = FeishuWSClient.__new__(FeishuWSClient)
    client._get_api_client = MagicMock(return_value=object())
    verified = SimpleNamespace(
        observed=tuple(range(80)),
        created=("help", "deep"),
        updated=("status",),
        deleted=("retired",),
    )
    monkeypatch.setattr(
        module,
        "reconcile_main_agent_slash_commands",
        AsyncMock(return_value=verified),
        raising=False,
    )

    with caplog.at_level(logging.INFO):
        client._sync_main_slash_commands()

    assert (
        "Main Agent Slash Commands ready: total=80 created=2 updated=1 deleted=1"
        in caplog.text
    )
