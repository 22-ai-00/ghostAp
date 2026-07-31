"""Regression tests for truthful Claude CLI model process arguments."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.agent_session.claude_cli import ClaudeCLIConfig, SyncClaudeCLISession
from src.agent_session.factory import create_engine_session, create_sync_session


def _completed_process() -> MagicMock:
    process = MagicMock()
    process.stdout = iter([])
    process.stderr = MagicMock()
    process.stderr.read.return_value = ""
    process.returncode = 0
    process.poll.return_value = 0
    process.wait.return_value = None
    process.pid = 1234
    return process


def test_selected_claude_model_reaches_real_cli_argv() -> None:
    session = SyncClaudeCLISession(
        cwd="/tmp",
        model_name="claude-sonnet-4-5",
        config=ClaudeCLIConfig(add_dir=False, bypass_permissions=False),
    )
    session.session_id = "session-1"

    with (
        patch(
            "src.agent_session.claude_cli.subprocess.Popen",
            return_value=_completed_process(),
        ) as popen,
        patch("src.utils.env.build_clean_env", return_value={}),
    ):
        result = session.send_prompt("implement the task")

    assert result.stop_reason == "end_turn"
    assert session.to_snapshot()["model_name"] == "claude-sonnet-4-5"
    argv = popen.call_args.args[0]
    assert argv[argv.index("--model") + 1] == "claude-sonnet-4-5"
    assert argv[-2:] == ["--", "implement the task"]


def test_sync_factory_passes_selected_model_to_claude_cli() -> None:
    with patch("src.agent_session.factory.SyncClaudeCLISession") as cli_session:
        create_sync_session("claude", "/tmp", model_name="claude-sonnet-4-5")

    cli_session.assert_called_once_with(
        cwd="/tmp",
        model_name="claude-sonnet-4-5",
    )


def test_engine_factory_passes_selected_model_to_claude_cli() -> None:
    settings = MagicMock()
    settings.rate_limit_retry_enabled = False
    settings.acp_startup_timeout = 20
    with (
        patch("src.agent_session.factory.get_settings", return_value=settings),
        patch("src.agent_session.factory.SyncClaudeCLISession") as cli_session,
    ):
        create_engine_session(
            "claude",
            "/tmp",
            model_name="claude-opus-4-8[1m]",
        )

    cli_session.assert_called_once_with(
        cwd="/tmp",
        model_name="claude-opus-4-8[1m]",
        employee_process_env=None,
    )
