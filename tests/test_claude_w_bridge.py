"""Regression tests for the hidden ``/claude-w`` local CLI bridge command.

The ``claude_w`` mode is a sibling of ``claude``:

- it bridges to a locally installed ``claude-w`` executable via the same
  ``SyncClaudeCLISession`` shell transport (not ACP);
- it is a full programming mode (mode/intent/handler/project snapshot);
- it stays hidden from the Feishu slash-command discovery surface while
  remaining resolvable when typed explicitly (``/claude-w``).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.acp.startup_utils import AcpRetryStarter, StartupBackend, select_startup_backend
from src.agent.intent_recognizer import IntentRecognizer, IntentType
from src.agent_session.backend_resolver import (
    agent_type_for_cli_command,
    cli_command_for_agent,
    is_cli_backend,
)
from src.agent_session.factory import create_engine_session
from src.feishu.handlers import ClaudeWModeHandler
from src.feishu.main_slash_commands import MAIN_AGENT_COMMANDS
from src.feishu.product_catalog import (
    get_slash_discoverable_actions,
    resolve_command,
)
from src.mode import PROGRAMMING_MODE_VALUES
from src.mode.manager import InteractionMode, ModeManager
from src.project.context import ProjectContext
from src.project.unified_context import ContextSourceMode


# ── backend resolution ──────────────────────────────────────────────


def test_cli_backend_resolver_maps_claude_w() -> None:
    assert is_cli_backend("claude_w")
    assert is_cli_backend("CLAUDE_W")
    assert cli_command_for_agent("claude_w") == "claude-w"
    assert agent_type_for_cli_command("claude-w") == "claude_w"
    # claude keeps its canonical binary and round-trips
    assert cli_command_for_agent("claude") == "claude"
    assert agent_type_for_cli_command("claude") == "claude"
    # unknown commands fall back to claude rather than crashing
    assert agent_type_for_cli_command("something-else") == "claude"
    # ACP backends are not CLI backends
    assert not is_cli_backend("codex")


def test_startup_backend_selects_cli_for_claude_w() -> None:
    assert select_startup_backend("claude_w") == StartupBackend.CLI
    assert select_startup_backend("claude") == StartupBackend.CLI
    assert select_startup_backend("codex") == StartupBackend.ACP


def test_engine_factory_spawns_claude_w_binary() -> None:
    settings = MagicMock()
    settings.rate_limit_retry_enabled = False
    settings.acp_startup_timeout = 20
    with (
        patch("src.agent_session.factory.get_settings", return_value=settings),
        patch("src.agent_session.factory.SyncClaudeCLISession") as cli_session,
    ):
        create_engine_session("claude_w", "/tmp", model_name=None)

    cli_session.assert_called_once()
    kwargs = cli_session.call_args.kwargs
    assert kwargs["cwd"] == "/tmp"
    assert kwargs["config"].command == "claude-w"


def test_retry_starter_builds_claude_w_session_with_claude_w_binary() -> None:
    cli_session_cls = MagicMock()
    AcpRetryStarter._build_session(
        backend=StartupBackend.CLI,
        agent_type="claude_w",
        cwd="/tmp",
        model_name=None,
        acp_session_cls=MagicMock(),
        cli_session_cls=cli_session_cls,
    )
    cli_session_cls.assert_called_once()
    assert cli_session_cls.call_args.kwargs["config"].command == "claude-w"


def test_cli_snapshot_records_claude_w_agent_type() -> None:
    from src.agent_session.claude_cli import ClaudeCLIConfig, SyncClaudeCLISession

    session = SyncClaudeCLISession(
        cwd="/tmp",
        config=ClaudeCLIConfig(command="claude-w", add_dir=False),
    )
    assert session.to_snapshot()["agent_type"] == "claude_w"


# ── mode / intent / handler wiring ──────────────────────────────────


def test_mode_manager_registers_claude_w() -> None:
    assert InteractionMode.CLAUDE_W.value == "claude_w"
    assert "claude_w" in PROGRAMMING_MODE_VALUES
    mm = ModeManager()
    mm.enter_claude_w_mode("chat-1")
    assert mm.is_claude_w_mode("chat-1")
    assert not mm.is_claude_mode("chat-1")


def test_claude_w_handler_mirrors_claude_cli_bridge() -> None:
    assert ClaudeWModeHandler.mode_key == "claude_w"
    assert ClaudeWModeHandler.interaction_mode == InteractionMode.CLAUDE_W
    assert ClaudeWModeHandler.context_source == ContextSourceMode.CLAUDE_W
    # _uses_claude_cli does not touch construction state; bypass __init__(ctx)
    handler = ClaudeWModeHandler.__new__(ClaudeWModeHandler)
    assert handler._uses_claude_cli() is True


def test_intent_recognizer_routes_claude_w() -> None:
    recognizer = IntentRecognizer()

    enter = recognizer._quick_match("/claude-w")
    assert enter is not None
    assert enter.primary_intent == IntentType.ENTER_CLAUDE_W

    enter_alias = recognizer._quick_match("/enter_claude_w")
    assert enter_alias is not None
    assert enter_alias.primary_intent == IntentType.ENTER_CLAUDE_W

    exit_alias = recognizer._quick_match("/exit_claude_w")
    assert exit_alias is not None
    assert exit_alias.primary_intent == IntentType.EXIT_MODE

    fallback = recognizer._get_fallback_intent("claude_w")
    assert fallback == IntentType.CLAUDE_W_MESSAGE


# ── project snapshot ────────────────────────────────────────────────


def test_project_context_persists_claude_w_snapshot() -> None:
    ctx = ProjectContext(project_id="p1", project_name="n", root_path="/tmp")
    ctx.set_claude_w_mode(True, session_id="sess-1", query_count=1)
    ctx.update_claude_w_snapshot(query="hello", query_count=3, session_id="sess-1")

    restored = ProjectContext.from_snapshot(ctx.to_snapshot())
    assert restored.claude_w_mode is True
    assert restored.claude_w_session_snapshot is not None
    assert restored.claude_w_session_snapshot.session_id == "sess-1"
    assert restored.claude_w_session_snapshot.query_count == 3
    assert restored.claude_w_session_snapshot.last_query == "hello"


# ── hidden slash command ────────────────────────────────────────────


def test_claude_w_command_resolves_but_is_hidden_from_discovery() -> None:
    resolved = resolve_command("/claude-w")
    assert resolved is not None
    assert resolved.action.programming_mode_id == "claude_w"
    assert resolved.action.slash_discoverable is False

    assert resolve_command("/enter_claude_w") is not None
    assert resolve_command("/exit_claude_w") is not None

    discoverable = {action.command for action in get_slash_discoverable_actions()}
    assert "/claude-w" not in discoverable
    assert "/enter_claude_w" not in discoverable

    panel = {command.name for command in MAIN_AGENT_COMMANDS}
    assert "/claude-w" not in panel


def test_system_handler_recognizes_claude_w_slashes() -> None:
    from src.feishu.handlers.system import SystemHandler
    from src.feishu.slash_command_parser import SlashCommandParser

    for raw in ("/claude-w", "/enter_claude_w", "/end_claude_w", "/exit_claude_w"):
        match = SlashCommandParser.parse(raw)
        assert SystemHandler.is_interceptable_command_match(match) is True

    assert SystemHandler.is_exit_command("/exit_claude_w") is True
    assert SystemHandler.is_exit_command("/end_claude_w") is True
    assert SystemHandler.is_exit_command("/claude-w") is False


def test_claude_w_registered_as_provider_but_hidden_from_tools_list() -> None:
    from src.acp.helper import list_acp_tools
    from src.acp.providers import get_providers

    assert "claude_w" in {name.lower() for name in get_providers()}

    listed = set()
    for tool in list_acp_tools():
        name = getattr(tool, "name", tool)
        listed.add(str(name).strip().lower())
    assert "claude_w" not in listed


# ── model-card activation path ─────────────────────────────────────


def test_model_card_activation_accepts_claude_w_tool() -> None:
    """Regression: selecting a model on the /claude-w card used to fail with
    "不支持的 ACP 工具: claude_w" because the activation handler map omitted it."""
    from src.feishu.handlers.system import SystemHandler

    project = ProjectContext(
        project_id="p1",
        project_name="n",
        root_path="/tmp",
    )
    project.acp_tool_name = "claude_w"

    manager = MagicMock()
    fresh_session = SimpleNamespace(session_id="fresh-session")
    manager.get_session.side_effect = [None, fresh_session]

    handler = MagicMock()
    handler._get_session_manager.return_value = manager
    handler.enter_mode.return_value = True

    system = SystemHandler.__new__(SystemHandler)
    system.ctx = SimpleNamespace(project_manager=MagicMock())
    system.get_handler = MagicMock(return_value=handler)
    system.reply_error = MagicMock()

    effect = system._enter_mode_with_acp_model(
        "selector",
        "chat-1",
        "claude_w",
        None,
        project,
    )

    # No "unsupported tool" error must be emitted.
    system.reply_error.assert_not_called()
    handler.enter_mode.assert_called_once()
    enter_kwargs = handler.enter_mode.call_args.kwargs
    assert enter_kwargs["silent"] is True
    assert effect is not None
    assert getattr(effect, "session", None) is fresh_session
    assert getattr(effect, "changed", None) is True

