"""Regression contracts for the first-class DSH ACP integration."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import src.acp.helper as helper_mod
import src.acp.providers as providers_mod
import src.feishu.handlers.programming as programming_mod
from src.acp.dsh_selection import decode_dsh_model_value, split_dsh_model_selection
from src.acp.sync_adapter import SyncACPSession
from src.agent.intent_recognizer import IntentRecognizer, IntentType
from src.card.programming_adapter import build_programming_metadata
from src.card.shared import build_mode_buttons
from src.feishu.dispatcher import MessageDispatcher
from src.feishu.product_catalog import PUBLIC_ACTIONS
from src.mode import InteractionMode, ModeManager
from src.project.context import ProjectContext
from src.project.unified_context import ContextSourceMode
from src.utils.engine_identity import resolve_engine_identity
from src.workflow_engine.constants import TOOL_DESCRIPTIONS
from src.workflow_engine.tool_registry import get_available_tools, invalidate_cache


def teardown_function() -> None:
    providers_mod._reset_providers_for_testing()


def test_dsh_provider_uses_native_profile_stdio_command() -> None:
    providers_mod._reset_providers_for_testing()
    providers = providers_mod.get_providers()

    assert "dsh" in providers
    assert providers["dsh"].get_serve_command() == ("dsh", ["--profile", "acp"])
    assert providers["dsh"].get_serve_command('["deepseek-official","deepseek-v4-flash"]') == (
        "dsh",
        ["--profile", "acp"],
    )


def test_dsh_availability_probe_requires_acp_profile_plugin() -> None:
    providers_mod._reset_providers_for_testing()
    completed = Mock(stdout="@dsh-enhanced/acp link:./plugins/acp", stderr="")
    with patch.object(providers_mod.subprocess, "run", return_value=completed) as run:
        assert providers_mod.get_providers()["dsh"].check_availability() is True

    assert run.call_args.args[0] == [
        "dsh",
        "plugin",
        "--profile",
        "acp",
        "list",
        "--depth",
        "0",
    ]


def test_dsh_mode_is_a_persistent_programming_mode() -> None:
    manager = ModeManager()

    manager.enter_dsh_mode("chat", project_id="project")

    assert manager.is_dsh_mode("chat", project_id="project") is True
    assert manager.is_programming_mode("chat", project_id="project") is True
    assert manager.get_mode_display_name("chat", project_id="project") == "🧭 DSH 编程模式"


def test_dsh_commands_and_context_source_are_public() -> None:
    recognizer = IntentRecognizer()

    enter = recognizer._quick_match("/dsh")
    info = recognizer._quick_match("/dsh_info")

    assert enter is not None and enter.primary_intent is IntentType.ENTER_DSH
    assert info is not None and info.primary_intent is IntentType.DSH_MESSAGE
    assert info.primary_data["command"] == "info"
    assert ContextSourceMode.DSH.value == "dsh"
    assert any(
        action.command == "/dsh"
        and action.programming_mode_id == "dsh"
        and action.enters_programming_mode
        for action in PUBLIC_ACTIONS
    )


def test_dsh_project_session_round_trips() -> None:
    context = ProjectContext("project", "Project", "/tmp/project")
    context.set_programming_mode("dsh", True, session_id="session-dsh", query_count=2)
    context.update_programming_snapshot("dsh", "continue", 3)

    restored = ProjectContext.from_snapshot(context.to_snapshot())

    assert restored.dsh_mode is True
    assert restored.dsh_session_snapshot is not None
    assert restored.dsh_session_snapshot.session_id == "session-dsh"
    assert restored.dsh_session_snapshot.query_count == 3
    assert restored.dsh_session_snapshot.last_query == "continue"


def test_dsh_custom_model_protocol_has_provider_specific_discovery() -> None:
    discover = getattr(helper_mod, "discover_dsh_model_options", None)
    assert callable(discover)

    def option(value: str, name: str) -> SimpleNamespace:
        return SimpleNamespace(value=value, name=name, description=None)

    def response(
        current_model: str,
        current_effort: str,
        efforts: tuple[str, ...],
    ) -> SimpleNamespace:
        return SimpleNamespace(
            session_id="session-dsh",
            config_options=[
                SimpleNamespace(
                    id="dsh.model",
                    category="model",
                    current_value=current_model,
                    options=[
                        SimpleNamespace(
                            group="deepseek-official",
                            name="DeepSeek",
                            options=[
                                option('["deepseek-official","flash"]', "Flash"),
                                option('["deepseek-official","pro"]', "Pro"),
                            ],
                        )
                    ],
                ),
                SimpleNamespace(
                    id="dsh.reasoning_effort",
                    category="thought_level",
                    current_value=json.dumps(["effort", current_effort], separators=(",", ":")),
                    options=[
                        option('["default"]', "Provider default"),
                        *(
                            option(
                                json.dumps(["effort", effort], separators=(",", ":")),
                                effort.title(),
                            )
                            for effort in efforts
                        ),
                    ],
                ),
            ],
        )

    initial = response('["deepseek-official","flash"]', "high", ("off", "high"))

    class Connection:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str]] = []

        async def set_config_option(self, *, config_id: str, session_id: str, value: str):
            self.calls.append((config_id, session_id, value))
            return response(value, "off", ("off", "high"))

    connection = Connection()
    models = asyncio.run(discover(connection, initial))

    assert [model.name for model in models] == [
        "deepseek-official/flash",
        "deepseek-official/pro",
    ]
    assert [variant.effort for variant in models[0].selection_variants] == [
        "default",
        "off",
        "high",
    ]
    assert models[0].selection_variants[-1].is_default is True
    assert connection.calls == [
        ("dsh.model", "session-dsh", '["deepseek-official","pro"]')
    ]


def test_dsh_runtime_selection_uses_namespaced_config_options() -> None:
    class Session:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        async def set_config_option(self, config_id: str, value: str) -> bool:
            self.calls.append((config_id, value))
            return True

    wrapper = SyncACPSession(
        "dsh",
        "/tmp",
        agent_cmd="dsh",
        agent_args=["--profile", "acp"],
    )
    session = Session()
    wrapper._acp_session = session  # type: ignore[assignment]

    applied = asyncio.run(
        wrapper._apply_dsh_selection(  # type: ignore[attr-defined]
            '["deepseek-official","deepseek-v4-flash","high"]'
        )
    )

    assert applied is True
    assert session.calls == [
        ("dsh.model", '["deepseek-official","deepseek-v4-flash"]'),
        ("dsh.reasoning_effort", '["effort","high"]'),
    ]


@pytest.mark.parametrize(
    ("decoder", "value"),
    [
        (decode_dsh_model_value, '["provider","model","unexpected"]'),
        (decode_dsh_model_value, '["provider",""]'),
        (split_dsh_model_selection, '["provider","model",""]'),
    ],
)
def test_dsh_selection_rejects_ambiguous_or_empty_wire_values(
    decoder,
    value: str,
) -> None:
    with pytest.raises(ValueError, match="invalid DSH model selection"):
        decoder(value)


def test_dsh_handler_dispatch_identity_and_card_metadata_are_wired() -> None:
    handler_type = getattr(programming_mod, "DSHModeHandler", None)
    assert handler_type is not None
    assert handler_type.interaction_mode is InteractionMode.DSH
    assert handler_type.mode_key == "dsh"
    assert "dsh" in MessageDispatcher._PROGRAMMING_MODES

    identity = resolve_engine_identity(
        mode=InteractionMode.DSH,
        acp_tool_name="dsh",
        acp_model_name='["deepseek-official","deepseek-v4-flash","high"]',
    )
    assert identity.agent_type == "dsh"
    assert identity.transport == "acp"
    assert identity.model_name == '["deepseek-official","deepseek-v4-flash","high"]'

    metadata = build_programming_metadata("dsh")
    assert (metadata.mode_emoji, metadata.mode_name, metadata.tool_name) == (
        "🧭",
        "DSH",
        "dsh",
    )
    actions = [
        button["behaviors"][0]["value"]["action"]
        for button in build_mode_buttons(InteractionMode.DSH, "project")
    ]
    assert actions == ["exit", "switch_project"]


def test_dsh_reuses_shared_model_and_exit_commands() -> None:
    exit_action = next(action for action in PUBLIC_ACTIONS if action.command == "/exit")
    integration_doc = (
        Path(__file__).resolve().parents[1] / "docs" / "dsh_acp_integration.md"
    ).read_text(encoding="utf-8")

    assert not hasattr(IntentType, "EXIT_DSH")
    assert "/exit_dsh" not in exit_action.aliases
    assert "/exit_dsh" not in integration_doc
    for tool_name in ("Codex", "Traex", "DSH"):
        assert tool_name in integration_doc


def test_dsh_is_available_to_workflow_agent_pools() -> None:
    invalidate_cache()
    assert TOOL_DESCRIPTIONS["dsh"] == "DeepSeek Harness 原生 ACP 编程"
    assert get_available_tools()["dsh"] == "DeepSeek Harness 原生 ACP 编程"
