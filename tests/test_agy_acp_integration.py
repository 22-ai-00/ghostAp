"""Regression contracts for the first-class AGY native ACP integration."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import src.acp.helper as helper_mod
import src.acp.providers as providers_mod
import src.feishu.handlers.programming as programming_mod
from src.acp.agy_selection import compose_agy_model_selection, split_agy_model_selection
from src.acp.sync_adapter import SyncACPSession
from src.agent.intent_recognizer import IntentRecognizer, IntentType
from src.agent_session.backend_resolver import is_cli_backend
from src.card.programming_adapter import build_programming_metadata
from src.card.render.model_cascade import resolve_model_cascade
from src.card.shared import build_mode_buttons
from src.feishu.dispatcher import MessageDispatcher
from src.feishu.product_catalog import PUBLIC_ACTIONS
from src.mode import InteractionMode, ModeManager
from src.project.context import ProjectContext
from src.project.unified_context import ContextSourceMode
from src.utils.engine_identity import resolve_engine_identity
from src.utils.env import build_clean_env
from src.workflow_engine.constants import TOOL_DESCRIPTIONS
from src.workflow_engine.tool_registry import get_available_tools, invalidate_cache


def teardown_function() -> None:
    providers_mod._reset_providers_for_testing()


def test_agy_provider_uses_native_acp_stdio_without_cli_fallback() -> None:
    providers_mod._reset_providers_for_testing()
    providers = providers_mod.get_providers()

    assert "agy" in providers
    assert providers["agy"].get_serve_command() == ("agy", ["acp"])
    assert providers["agy"].get_serve_command("gemini-4-hydrogen/xhigh") == (
        "agy",
        ["acp"],
    )
    assert providers["agy"].get_fallback_command("gemini-4-hydrogen/xhigh") is None
    assert is_cli_backend("agy") is False


def test_agy_availability_probe_requires_acp_stdio_subcommand() -> None:
    providers_mod._reset_providers_for_testing()
    completed = Mock(
        stdout="Usage: agy acp [flags]\nStart the Agent Client Protocol (ACP) stdio server",
        stderr="",
    )
    with patch.object(providers_mod.subprocess, "run", return_value=completed) as run:
        assert providers_mod.get_providers()["agy"].check_availability() is True

    assert run.call_args.args[0] == ["agy", "acp", "--help"]

    providers_mod._reset_providers_for_testing()
    missing_stdio = Mock(stdout="Usage: agy acp [flags]\n", stderr="")
    with patch.object(providers_mod.subprocess, "run", return_value=missing_stdio):
        assert providers_mod.get_providers()["agy"].check_availability() is False


def test_build_clean_env_strips_parent_antigravity_session_guards() -> None:
    env = build_clean_env(
        {
            "HOME": "/tmp/ghostap-test-home",
            "CLAUDECODE": "1",
            "ANTIGRAVITY_AGENT": "1",
            "ANTIGRAVITY_LS_ADDRESS": "localhost:27493",
            "ANTIGRAVITY_CSRF_TOKEN": "secret-token",
            "ANTIGRAVITY_CONVERSATION_ID": "conv-1",
            "ANTIGRAVITY_TRAJECTORY_ID": "traj-1",
            "ANTIGRAVITY_SOURCE_METADATA": '{"tool":{}}',
            "ANTIGRAVITY_APP_DATA_DIR": "/tmp/custom-agy-app-data",
        }
    )

    for key in (
        "CLAUDECODE",
        "ANTIGRAVITY_AGENT",
        "ANTIGRAVITY_LS_ADDRESS",
        "ANTIGRAVITY_CSRF_TOKEN",
        "ANTIGRAVITY_CONVERSATION_ID",
        "ANTIGRAVITY_TRAJECTORY_ID",
        "ANTIGRAVITY_SOURCE_METADATA",
    ):
        assert key not in env
    assert env["ANTIGRAVITY_APP_DATA_DIR"] == "/tmp/custom-agy-app-data"


def test_agy_mode_and_project_session_round_trip() -> None:
    manager = ModeManager()
    manager.enter_agy_mode("chat", project_id="project")

    assert manager.is_agy_mode("chat", project_id="project") is True
    assert manager.is_programming_mode("chat", project_id="project") is True
    assert manager.get_mode_display_name("chat", project_id="project") == "🪐 AGY 编程模式"

    context = ProjectContext("project", "Project", "/tmp/project")
    context.set_programming_mode("agy", True, session_id="session-agy", query_count=2)
    context.update_programming_snapshot("agy", "continue", 3)

    restored = ProjectContext.from_snapshot(context.to_snapshot())

    assert restored.agy_mode is True
    assert restored.agy_session_snapshot is not None
    assert restored.agy_session_snapshot.session_id == "session-agy"
    assert restored.agy_session_snapshot.query_count == 3
    assert restored.agy_session_snapshot.last_query == "continue"


def test_agy_commands_catalog_and_shared_exit_contract() -> None:
    recognizer = IntentRecognizer()

    enter = recognizer._quick_match("/agy")
    alias = recognizer._quick_match("/enter_agy")
    info = recognizer._quick_match("/agy_info")
    keyword = recognizer._quick_match("进入agy")

    assert enter is not None and enter.primary_intent is IntentType.ENTER_AGY
    assert alias is not None and alias.primary_intent is IntentType.ENTER_AGY
    assert info is not None and info.primary_intent is IntentType.AGY_MESSAGE
    assert info.primary_data["command"] == "info"
    assert keyword is not None and keyword.primary_intent is IntentType.ENTER_AGY
    assert ContextSourceMode.AGY.value == "agy"
    assert any(
        action.command == "/agy"
        and action.programming_mode_id == "agy"
        and action.enters_programming_mode
        and action.slash_discoverable
        for action in PUBLIC_ACTIONS
    )

    exit_action = next(action for action in PUBLIC_ACTIONS if action.command == "/exit")
    integration_doc = (
        Path(__file__).resolve().parents[1] / "docs" / "agy_acp_integration.md"
    ).read_text(encoding="utf-8")

    assert not hasattr(IntentType, "EXIT_AGY")
    assert "/exit_agy" not in exit_action.aliases
    assert "/exit_agy" not in integration_doc


def test_agy_model_discovery_expands_per_model_thought_levels_and_omits_unsupported() -> None:
    discover = getattr(helper_mod, "discover_agy_model_options", None)
    assert callable(discover)

    def option(value: str, name: str) -> SimpleNamespace:
        return SimpleNamespace(value=value, name=name, description=None)

    def response(
        current_model: str,
        current_effort: str | None,
        efforts: tuple[str, ...],
    ) -> SimpleNamespace:
        config_options = [
            SimpleNamespace(
                id="model",
                category="model",
                current_value=current_model,
                options=[
                    option("gemini-4-carbon", "Gemini 4 Carbon"),
                    option("gemini-4-hydrogen", "Gemini 4 Hydrogen"),
                    option("gemini-pro-orchestrator", "Orchestrator (Carbon)"),
                ],
            )
        ]
        if efforts:
            config_options.append(
                SimpleNamespace(
                    id="thought_level",
                    category="thought_level",
                    current_value=current_effort,
                    options=[option(effort, effort.title()) for effort in efforts],
                )
            )
        return SimpleNamespace(
            session_id="session-agy",
            config_options=config_options,
        )

    initial = response("gemini-4-carbon", "medium", ("low", "medium", "high"))

    class Connection:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str]] = []

        async def set_config_option(self, *, config_id: str, session_id: str, value: str):
            self.calls.append((config_id, session_id, value))
            if value == "gemini-4-hydrogen":
                return response(value, "medium", ("low", "medium", "high", "xhigh"))
            return response(value, None, ())

    connection = Connection()
    models = asyncio.run(discover(connection, initial))

    assert [model.name for model in models] == [
        "gemini-4-carbon",
        "gemini-4-hydrogen",
        "gemini-pro-orchestrator",
    ]
    assert [variant.name for variant in models[0].selection_variants] == [
        "gemini-4-carbon/low",
        "gemini-4-carbon/medium",
        "gemini-4-carbon/high",
    ]
    assert [variant.name for variant in models[1].selection_variants] == [
        "gemini-4-hydrogen/low",
        "gemini-4-hydrogen/medium",
        "gemini-4-hydrogen/high",
        "gemini-4-hydrogen/xhigh",
    ]
    assert [variant.name for variant in models[2].selection_variants] == [
        "gemini-pro-orchestrator",
    ]
    assert models[0].is_default is True
    assert connection.calls == [
        ("model", "session-agy", "gemini-4-hydrogen"),
        ("model", "session-agy", "gemini-pro-orchestrator"),
    ]

    hydrogen_state = resolve_model_cascade(models, selected_model="gemini-4-hydrogen", selected_effort="xhigh")
    assert hydrogen_state.efforts == ("low", "medium", "high", "xhigh")
    assert hydrogen_state.selection == "gemini-4-hydrogen/xhigh"

    orchestrator_state = resolve_model_cascade(models, selected_model="gemini-pro-orchestrator")
    assert orchestrator_state.efforts == ()
    assert orchestrator_state.selected_effort is None
    assert orchestrator_state.selection == "gemini-pro-orchestrator"


def test_agy_runtime_selection_applies_model_and_thought_level_and_fails_closed() -> None:
    class Session:
        def __init__(self, *, reject: bool = False) -> None:
            self.calls: list[tuple[str, str]] = []
            self.reject = reject

        async def set_config_option(self, config_id: str, value: str) -> bool:
            self.calls.append((config_id, value))
            return not self.reject

    wrapper = SyncACPSession(
        "agy",
        "/tmp",
        agent_cmd="agy",
        agent_args=["acp"],
    )
    session = Session()
    wrapper._acp_session = session  # type: ignore[assignment]

    assert asyncio.run(
        wrapper._apply_agy_selection(compose_agy_model_selection("gemini-4-hydrogen", "xhigh"))  # type: ignore[attr-defined]
    ) is True
    assert session.calls == [
        ("model", "gemini-4-hydrogen"),
        ("thought_level", "xhigh"),
    ]

    bare_session = Session()
    wrapper._acp_session = bare_session  # type: ignore[assignment]
    assert asyncio.run(wrapper._apply_agy_selection("gemini-pro-orchestrator")) is True  # type: ignore[attr-defined]
    assert bare_session.calls == [("model", "gemini-pro-orchestrator")]

    reject_session = Session(reject=True)
    wrapper._acp_session = reject_session  # type: ignore[assignment]
    assert asyncio.run(wrapper._apply_agy_selection("gemini-4-hydrogen/xhigh")) is False  # type: ignore[attr-defined]
    assert asyncio.run(wrapper._apply_agy_selection("gemini-4-hydrogen/bogus")) is False  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "value",
    ["", "   ", "gemini-4-hydrogen/", "/high", "gemini-4-hydrogen/bogus", "a/b/high"],
)
def test_agy_selection_rejects_ambiguous_or_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="invalid AGY model selection"):
        split_agy_model_selection(value)


def test_agy_handler_dispatch_identity_card_and_workflow_pool() -> None:
    handler_type = getattr(programming_mod, "AGYModeHandler", None)
    assert handler_type is not None
    assert handler_type.interaction_mode is InteractionMode.AGY
    assert handler_type.mode_key == "agy"
    assert "agy" in MessageDispatcher._PROGRAMMING_MODES

    identity = resolve_engine_identity(
        mode=InteractionMode.AGY,
        acp_tool_name="agy",
        acp_model_name="gemini-4-hydrogen/xhigh",
    )
    assert identity.engine_name == "AGY"
    assert identity.agent_type == "agy"
    assert identity.transport == "acp"
    assert identity.model_name == "gemini-4-hydrogen/xhigh"

    metadata = build_programming_metadata("agy")
    assert (metadata.mode_emoji, metadata.mode_name, metadata.tool_name) == (
        "🪐",
        "AGY",
        "agy",
    )
    actions = [
        button["behaviors"][0]["value"]["action"]
        for button in build_mode_buttons(InteractionMode.AGY, "project")
    ]
    assert actions == ["exit", "switch_project"]

    invalidate_cache()
    assert TOOL_DESCRIPTIONS["agy"] == "Google Antigravity 原生 ACP 编程"
    assert get_available_tools()["agy"] == "Google Antigravity 原生 ACP 编程"
