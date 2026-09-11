"""Regression tests for the hidden ``/codex-w`` local CLI bridge command.

The ``codex_w`` mode is a sibling of ``codex`` and mirrors ``claude_w``:

- it bridges to a locally installed ``codex-w`` executable via a per-prompt
  ``codex exec`` shell transport (not ACP);
- it is a full programming mode (mode/intent/handler/project snapshot);
- it stays hidden from the Feishu slash-command discovery surface while
  remaining resolvable when typed explicitly (``/codex-w``);
- the model card synthesises a model/effort matrix even though the CLI exposes
  no ACP catalog, and both overrides are emitted before the ``exec``
  subcommand (a codex CLI quirk).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.acp import helper
from src.acp.startup_utils import AcpRetryStarter, StartupBackend, select_startup_backend
from src.agent.intent_recognizer import IntentRecognizer, IntentType
from src.agent_session.backend_resolver import (
    agent_type_for_cli_command,
    cli_command_for_agent,
    is_cli_backend,
    is_codex_cli_backend,
)
from src.agent_session.factory import create_engine_session
from src.feishu.handlers import CodexWModeHandler
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


def test_cli_backend_resolver_maps_codex_w() -> None:
    assert is_cli_backend("codex_w")
    assert is_cli_backend("CODEX_W")
    assert is_codex_cli_backend("codex_w")
    assert cli_command_for_agent("codex_w") == "codex-w"
    assert agent_type_for_cli_command("codex-w") == "codex_w"
    # codex (ACP) is neither a CLI backend nor the codex-CLI family
    assert not is_cli_backend("codex")
    assert not is_codex_cli_backend("codex")
    # claude-w shares the CLI bridge but uses the Claude argv family
    assert is_cli_backend("claude_w")
    assert not is_codex_cli_backend("claude_w")


def test_startup_backend_selects_cli_for_codex_w() -> None:
    assert select_startup_backend("codex_w") == StartupBackend.CLI
    assert select_startup_backend("codex") == StartupBackend.ACP


def test_engine_factory_spawns_codex_w_binary() -> None:
    settings = MagicMock()
    settings.rate_limit_retry_enabled = False
    settings.acp_startup_timeout = 20
    with (
        patch("src.agent_session.factory.get_settings", return_value=settings),
        patch("src.agent_session.factory.SyncCodexCLISession") as cli_session,
    ):
        create_engine_session("codex_w", "/tmp", model_name=None)

    cli_session.assert_called_once()
    kwargs = cli_session.call_args.kwargs
    assert kwargs["cwd"] == "/tmp"
    assert kwargs["config"].command == "codex-w"


def test_retry_starter_builds_codex_w_session_with_codex_w_binary() -> None:
    # The codex branch ignores the injected cli_session_cls and imports the
    # concrete Codex session directly, so patch it at its source module.
    with patch("src.agent_session.codex_cli.SyncCodexCLISession") as cli_session:
        AcpRetryStarter._build_session(
            backend=StartupBackend.CLI,
            agent_type="codex_w",
            cwd="/tmp",
            model_name=None,
            acp_session_cls=MagicMock(),
            cli_session_cls=MagicMock(),
        )
    cli_session.assert_called_once()
    assert cli_session.call_args.kwargs["config"].command == "codex-w"


def test_cli_snapshot_records_codex_w_agent_type() -> None:
    from src.agent_session.codex_cli import CodexCLIConfig, SyncCodexCLISession

    session = SyncCodexCLISession(
        cwd="/tmp",
        config=CodexCLIConfig(command="codex-w", add_dir=False),
    )
    assert session.to_snapshot()["agent_type"] == "codex_w"


# ── real codex exec argv ────────────────────────────────────────────


def _completed_process(stdout_lines: list[str] | None = None) -> MagicMock:
    process = MagicMock()
    process.stdout = iter(stdout_lines or [])
    process.stderr = MagicMock()
    process.stderr.read.return_value = ""
    process.returncode = 0
    process.poll.return_value = 0
    process.wait.return_value = None
    process.pid = 1234
    return process


_THREAD_STARTED = '{"type":"thread.started","thread_id":"thread-1"}'
_AGENT_MESSAGE = '{"type":"item.completed","item":{"type":"agent_message","text":"hello"}}'


def _run_with_mocked_process(session, prompt: str) -> MagicMock:
    popen_patch = patch(
        "src.agent_session.codex_cli.subprocess.Popen",
        return_value=_completed_process([_THREAD_STARTED + "\n", _AGENT_MESSAGE + "\n"]),
    )
    with (
        patch("src.agent_session.codex_cli.shutil.which", return_value="/usr/bin/codex-w"),
        patch("src.utils.env.build_clean_env", return_value={}),
        popen_patch as popen,
    ):
        result = session.send_prompt(prompt)
    assert result.stop_reason == "end_turn"
    return popen


def test_codex_exec_argv_shape_for_plain_prompt() -> None:
    from src.agent_session.codex_cli import CodexCLIConfig, SyncCodexCLISession

    session = SyncCodexCLISession(
        cwd="/repo",
        config=CodexCLIConfig(command="codex-w", add_dir=False),
    )
    popen = _run_with_mocked_process(session, "do the thing")

    argv = popen.call_args.args[0]
    assert argv == [
        "codex-w",
        "exec",
        "--json",
        "--skip-git-repo-check",
        "-C",
        "/repo",
        "do the thing",
    ]
    # first prompt establishes the thread from thread.started
    assert session.session_id == "thread-1"
    assert session.is_resumed is True


def test_model_and_effort_overrides_precede_exec_subcommand() -> None:
    from src.agent_session.codex_cli import CodexCLIConfig, SyncCodexCLISession

    session = SyncCodexCLISession(
        cwd="/repo",
        model_name="gpt-5/high",
        config=CodexCLIConfig(command="codex-w", add_dir=False),
    )
    popen = _run_with_mocked_process(session, "think harder")

    argv = popen.call_args.args[0]
    assert argv[:6] == [
        "codex-w",
        "-c",
        'model="gpt-5"',
        "-c",
        'model_reasoning_effort="high"',
        "exec",
    ]
    # Regression guard: every -c override MUST precede exec, otherwise codex
    # mis-resolves model metadata and routes to a ChatGPT account (HTTP 400).
    exec_index = argv.index("exec")
    assert all(i < exec_index for i, tok in enumerate(argv) if tok == "-c")


def test_default_model_token_omits_model_but_keeps_effort() -> None:
    from src.agent_session.codex_cli import CodexCLIConfig, SyncCodexCLISession

    session = SyncCodexCLISession(
        cwd="/repo",
        model_name="default/high",
        config=CodexCLIConfig(command="codex-w", add_dir=False),
    )
    popen = _run_with_mocked_process(session, "pick wrapper default model")

    argv = popen.call_args.args[0]
    assert not any(tok.startswith("model=") for tok in argv)
    assert 'model_reasoning_effort="high"' in argv
    assert argv.index('model_reasoning_effort="high"') < argv.index("exec")


def test_plain_default_selection_emits_no_overrides() -> None:
    from src.agent_session.codex_cli import CodexCLIConfig, SyncCodexCLISession

    session = SyncCodexCLISession(
        cwd="/repo",
        model_name="default",
        config=CodexCLIConfig(command="codex-w", add_dir=False),
    )
    popen = _run_with_mocked_process(session, "pure defaults")

    argv = popen.call_args.args[0]
    assert "-c" not in argv


def test_resumed_prompt_uses_exec_resume_and_retransmits_effort() -> None:
    from src.agent_session.codex_cli import CodexCLIConfig, SyncCodexCLISession

    session = SyncCodexCLISession(
        cwd="/repo",
        model_name="gpt-5/low",
        config=CodexCLIConfig(command="codex-w", add_dir=False),
    )
    session.session_id = "prior-thread"
    session.is_resumed = True
    popen = _run_with_mocked_process(session, "continue")

    argv = popen.call_args.args[0]
    assert argv[:2] == ["codex-w", "-c"]
    resume_at = argv.index("resume")
    assert argv[resume_at : resume_at + 2] == ["resume", "prior-thread"]
    assert resume_at > argv.index("exec")
    assert argv[-1] == "continue"


# ── synthetic model/effort card catalog ─────────────────────────────


def _variant_names(option) -> list[str]:
    return [v.name for v in option.selection_variants]


def test_codex_cli_catalog_defaults_to_pseudo_model_with_effort_ladder() -> None:
    from src.acp.model_selection import (
        CODEX_DEFAULT_MODEL_TOKEN,
        CODEX_REASONING_EFFORT_ORDER,
    )

    options = helper.fetch_acp_models("codex_w", None, current_model=None)

    assert len(options) == 1
    only = options[0]
    assert only.name == CODEX_DEFAULT_MODEL_TOKEN
    assert _variant_names(only) == [
        CODEX_DEFAULT_MODEL_TOKEN,
        *[f"{CODEX_DEFAULT_MODEL_TOKEN}/{level}" for level in CODEX_REASONING_EFFORT_ORDER],
    ]
    assert only.reasoning_efforts == CODEX_REASONING_EFFORT_ORDER
    assert only.is_default is True
    assert only.selection_variants[0].is_default is True


def test_codex_cli_catalog_roundtrips_saved_model_and_effort() -> None:
    options = helper.fetch_acp_models("codex_w", None, current_model="gpt-5/ultra")

    names = {option.name for option in options}
    assert names == {"default", "gpt-5"}

    saved = next(o for o in options if o.name == "gpt-5")
    ultra = next(v for v in saved.selection_variants if v.name == "gpt-5/ultra")
    assert ultra.is_default is True
    assert ultra.effort == "ultra"
    assert saved.is_default is True

    pseudo = next(o for o in options if o.name == "default")
    assert pseudo.is_default is False


# ── mode / intent / handler wiring ──────────────────────────────────


def test_mode_manager_registers_codex_w() -> None:
    assert InteractionMode.CODEX_W.value == "codex_w"
    assert "codex_w" in PROGRAMMING_MODE_VALUES
    mm = ModeManager()
    mm.enter_codex_w_mode("chat-1")
    assert mm.is_codex_w_mode("chat-1")
    assert not mm.is_codex_mode("chat-1")


def test_codex_w_handler_mirrors_codex_bridge() -> None:
    assert CodexWModeHandler.mode_key == "codex_w"
    assert CodexWModeHandler.interaction_mode == InteractionMode.CODEX_W
    assert CodexWModeHandler.context_source == ContextSourceMode.CODEX_W
    assert CodexWModeHandler.mode_emoji == "🌩️"


def test_intent_recognizer_routes_codex_w() -> None:
    recognizer = IntentRecognizer()

    enter = recognizer._quick_match("/codex-w")
    assert enter is not None
    assert enter.primary_intent == IntentType.ENTER_CODEX_W

    enter_alias = recognizer._quick_match("/enter_codex_w")
    assert enter_alias is not None
    assert enter_alias.primary_intent == IntentType.ENTER_CODEX_W

    exit_alias = recognizer._quick_match("/exit_codex_w")
    assert exit_alias is not None
    assert exit_alias.primary_intent == IntentType.EXIT_MODE

    fallback = recognizer._get_fallback_intent("codex_w")
    assert fallback == IntentType.CODEX_W_MESSAGE


# ── project snapshot ────────────────────────────────────────────────


def test_project_context_persists_codex_w_snapshot() -> None:
    ctx = ProjectContext(project_id="p1", project_name="n", root_path="/tmp")
    ctx.set_codex_w_mode(True, session_id="sess-1", query_count=1)
    ctx.update_codex_w_snapshot(query="hello", query_count=3, session_id="sess-1")

    restored = ProjectContext.from_snapshot(ctx.to_snapshot())
    assert restored.codex_w_mode is True
    assert restored.codex_w_session_snapshot is not None
    assert restored.codex_w_session_snapshot.session_id == "sess-1"
    assert restored.codex_w_session_snapshot.query_count == 3
    assert restored.codex_w_session_snapshot.last_query == "hello"


# ── hidden slash command ────────────────────────────────────────────


def test_codex_w_command_resolves_but_is_hidden_from_discovery() -> None:
    resolved = resolve_command("/codex-w")
    assert resolved is not None
    assert resolved.action.programming_mode_id == "codex_w"
    assert resolved.action.slash_discoverable is False

    assert resolve_command("/enter_codex_w") is not None
    assert resolve_command("/exit_codex_w") is not None

    discoverable = {action.command for action in get_slash_discoverable_actions()}
    assert "/codex-w" not in discoverable
    assert "/enter_codex_w" not in discoverable

    panel = {command.name for command in MAIN_AGENT_COMMANDS}
    assert "/codex-w" not in panel


def test_system_handler_recognizes_codex_w_slashes() -> None:
    from src.feishu.handlers.system import SystemHandler
    from src.feishu.slash_command_parser import SlashCommandParser

    for raw in ("/codex-w", "/enter_codex_w", "/end_codex_w", "/exit_codex_w"):
        match = SlashCommandParser.parse(raw)
        assert SystemHandler.is_interceptable_command_match(match) is True

    assert SystemHandler.is_exit_command("/exit_codex_w") is True
    assert SystemHandler.is_exit_command("/end_codex_w") is True
    assert SystemHandler.is_exit_command("/codex-w") is False


def test_codex_w_registered_as_provider_but_hidden_from_tools_list() -> None:
    from src.acp.helper import list_acp_tools
    from src.acp.providers import get_providers

    assert "codex_w" in {name.lower() for name in get_providers()}

    listed = set()
    for tool in list_acp_tools():
        name = getattr(tool, "name", tool)
        listed.add(str(name).strip().lower())
    assert "codex_w" not in listed


# ── model-card activation path ─────────────────────────────────────


def test_model_card_activation_accepts_codex_w_tool() -> None:
    """The activation handler tool map must include codex_w, otherwise the
    model card fails with "不支持的 ACP 工具: codex_w"."""
    from src.feishu.handlers.system import SystemHandler

    project = ProjectContext(
        project_id="p1",
        project_name="n",
        root_path="/tmp",
    )
    project.acp_tool_name = "codex_w"

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
        "codex_w",
        None,
        project,
    )

    system.reply_error.assert_not_called()
    handler.enter_mode.assert_called_once()
    assert handler.enter_mode.call_args.kwargs["silent"] is True
    assert effect is not None
    assert getattr(effect, "session", None) is fresh_session
    assert getattr(effect, "changed", None) is True
