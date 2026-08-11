"""Regression contracts for the first-class Grok Build ACP integration."""

from __future__ import annotations

from unittest.mock import Mock, patch

import src.acp.providers as providers_mod
from src.agent.intent_recognizer import IntentRecognizer, IntentType
from src.feishu.product_catalog import PUBLIC_ACTIONS
from src.mode import ModeManager
from src.project.context import ProjectContext
from src.project.unified_context import ContextSourceMode


def teardown_function() -> None:
    providers_mod._reset_providers_for_testing()


def test_grok_provider_uses_native_stdio_command_and_global_model_option() -> None:
    providers_mod._reset_providers_for_testing()
    provider = providers_mod.get_providers()["grok"]

    assert provider.get_serve_command() == ("grok", ["agent", "stdio"])
    assert provider.get_serve_command("grok-build") == (
        "grok",
        ["agent", "--model", "grok-build", "stdio"],
    )


def test_grok_availability_probe_targets_official_agent_help() -> None:
    providers_mod._reset_providers_for_testing()
    completed = Mock(stdout="Usage: grok agent\nCommands:\n  stdio", stderr="")
    with patch.object(providers_mod.subprocess, "run", return_value=completed) as run:
        assert providers_mod.get_providers()["grok"].check_availability() is True

    assert run.call_args.args[0] == ["grok", "agent", "--help"]


def test_grok_mode_is_a_persistent_programming_mode() -> None:
    manager = ModeManager()

    manager.enter_grok_mode("chat", project_id="project")

    assert manager.is_grok_mode("chat", project_id="project") is True
    assert manager.is_programming_mode("chat", project_id="project") is True
    assert manager.get_mode_display_name("chat", project_id="project") == "🌌 Grok 编程模式"


def test_grok_commands_and_context_source_are_public() -> None:
    recognizer = IntentRecognizer()

    enter = recognizer._quick_match("/grok")
    info = recognizer._quick_match("/grok_info")

    assert enter is not None and enter.primary_intent is IntentType.ENTER_GROK
    assert info is not None and info.primary_intent is IntentType.GROK_MESSAGE
    assert info.primary_data["command"] == "info"
    assert ContextSourceMode.GROK.value == "grok"
    assert any(
        action.command == "/grok"
        and action.programming_mode_id == "grok"
        and action.enters_programming_mode
        for action in PUBLIC_ACTIONS
    )


def test_grok_project_session_round_trips() -> None:
    context = ProjectContext("project", "Project", "/tmp/project")
    context.set_programming_mode("grok", True, session_id="session-grok", query_count=2)
    context.update_programming_snapshot("grok", "continue", 3)

    restored = ProjectContext.from_snapshot(context.to_snapshot())

    assert restored.grok_mode is True
    assert restored.grok_session_snapshot is not None
    assert restored.grok_session_snapshot.session_id == "session-grok"
    assert restored.grok_session_snapshot.query_count == 3
    assert restored.grok_session_snapshot.last_query == "continue"
