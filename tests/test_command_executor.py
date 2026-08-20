from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.command_executor import CommandExecutor


def _settings(**overrides):
    values = {"command_timeout": 30, "command_max_output_length": 4_000}
    values.update(overrides)
    return SimpleNamespace(**values)


def test_command_executor_forwards_arbitrary_shell_text_without_policy_checks(
    monkeypatch,
) -> None:
    process = MagicMock(returncode=0, stdout="ok", stderr="")
    runner = MagicMock()
    runner.run.return_value = process
    monkeypatch.setenv("SHELL", "/bin/sh")
    monkeypatch.setenv("GHOSTAP_DIRECT_HOST_MARKER", "visible")
    command = "printf safe; rm -rf /do-not-run-this-test"

    result = CommandExecutor(
        settings=_settings(), subprocess_executor=runner
    ).execute(command, cwd="/tmp", interactive=False)

    assert result.success is True
    args = runner.run.call_args.args[0]
    assert args == ["/bin/sh", "-l", "-c", command]
    assert runner.run.call_args.kwargs["env"]["GHOSTAP_DIRECT_HOST_MARKER"] == "visible"


def test_command_executor_retains_timeout_and_output_bounds() -> None:
    runner = MagicMock()
    runner.run.side_effect = subprocess.TimeoutExpired(["/bin/sh"], 3)

    result = CommandExecutor(
        settings=_settings(command_timeout=3), subprocess_executor=runner
    ).execute("long-running")

    assert result.success is False
    assert result.return_code == -1
    assert result.error_message == "命令执行超时（3秒）"
