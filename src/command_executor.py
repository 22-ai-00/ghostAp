"""Direct host command execution for the Shell and ACP callback lanes."""

from __future__ import annotations

import logging
import os
import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import get_settings
from .utils.errors import get_error_detail
from .utils.redact import redact_sensitive
from .utils.text import truncate_output

logger = logging.getLogger(__name__)


class SubprocessExecutor(ABC):
    """Small subprocess seam used by command-execution tests."""

    @abstractmethod
    def run(
        self,
        cmd_args: List[str],
        shell: bool = False,
        capture_output: bool = True,
        text: bool = True,
        timeout: Optional[float] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> Any:
        raise NotImplementedError


class DefaultSubprocessExecutor(SubprocessExecutor):
    def run(
        self,
        cmd_args: List[str],
        shell: bool = False,
        capture_output: bool = True,
        text: bool = True,
        timeout: Optional[float] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> Any:
        return subprocess.run(
            cmd_args,
            shell=shell,
            capture_output=capture_output,
            text=text,
            timeout=timeout,
            cwd=cwd,
            env=env,
        )


@dataclass
class CommandExecutionResult:
    success: bool
    stdout: str
    stderr: str
    return_code: int
    error_message: Optional[str] = None

    def to_message(self) -> str:
        if self.error_message:
            return f"❌ 执行失败: {self.error_message}"
        parts = []
        if self.stdout:
            parts.append(f"📤 输出:\n```\n{self.stdout}\n```")
        if self.stderr:
            parts.append(f"⚠️ 错误输出:\n```\n{self.stderr}\n```")
        if not parts:
            parts.append("✅ 命令执行成功（无输出）")
        parts.append(f"🔢 返回码: {self.return_code}")
        return "\n".join(parts)


class CommandExecutor:
    """Execute a command directly without GhostAP policy checks or isolation."""

    def __init__(self, settings=None, subprocess_executor: Optional[SubprocessExecutor] = None):
        self.settings = settings if settings is not None else get_settings()
        self.subprocess_executor = subprocess_executor or DefaultSubprocessExecutor()

    def execute(
        self,
        command: str,
        cwd: Optional[str] = None,
        interactive: bool = True,
        chat_id: Optional[str] = None,
        env_overrides: Optional[Dict[str, str]] = None,
    ) -> CommandExecutionResult:
        del chat_id
        try:
            command = self._sanitize_command_for_noninteractive(command)
            env = os.environ.copy()
            env.update(
                {
                    "GIT_PAGER": "cat",
                    "PAGER": "cat",
                    "MANPAGER": "cat",
                    "SYSTEMD_PAGER": "cat",
                    "LESS": "FRX",
                    "GIT_TERMINAL_PROMPT": "0",
                    "TERM": "dumb",
                }
            )
            if env_overrides:
                env.update(env_overrides)

            shell_path = os.environ.get("SHELL", "/bin/bash")
            cmd_args = [shell_path, "-i" if interactive else "-l", "-c", command]
            process = self.subprocess_executor.run(
                cmd_args,
                shell=False,
                capture_output=True,
                text=True,
                timeout=self.settings.command_timeout,
                cwd=cwd,
                env=env,
            )
            stdout = process.stdout
            stderr = process.stderr
            if stderr:
                ignored = (
                    "no job control in this shell",
                    "cannot set terminal process group",
                    "Inappropriate ioctl for device",
                    "bash: cannot set terminal process group",
                    "The input device is not a TTY",
                )
                stderr = "\n".join(
                    line
                    for line in stderr.splitlines()
                    if not any(pattern in line for pattern in ignored)
                )
            stdout = redact_sensitive(
                truncate_output(stdout, self.settings.command_max_output_length)
            )
            stderr = redact_sensitive(
                truncate_output(
                    stderr,
                    self.settings.command_max_output_length,
                    label="错误输出被截断",
                )
            )
            return CommandExecutionResult(
                success=process.returncode == 0,
                stdout=stdout,
                stderr=stderr,
                return_code=process.returncode,
            )
        except subprocess.TimeoutExpired:
            return CommandExecutionResult(
                success=False,
                stdout="",
                stderr="",
                return_code=-1,
                error_message=f"命令执行超时（{self.settings.command_timeout}秒）",
            )
        except Exception as exc:
            return CommandExecutionResult(
                success=False,
                stdout="",
                stderr="",
                return_code=-1,
                error_message=f"执行异常: {get_error_detail(exc)}",
            )

    @staticmethod
    def _sanitize_command_for_noninteractive(command: str) -> str:
        cmd = command.strip()
        if not cmd:
            return command
        lowered = cmd.lower()
        if (
            "--no-pager" in lowered
            or "--paginate" in lowered
            or re.search(r"(^|\s)git\s+-p(\s|$)", lowered)
        ):
            return command
        if re.match(r"^\s*(?:sudo\s+)?git\b", cmd):
            return re.sub(
                r"^(\s*(?:sudo\s+)?)git\b",
                r"\1git --no-pager",
                command,
                count=1,
            )
        return command


__all__ = [
    "CommandExecutionResult",
    "CommandExecutor",
    "DefaultSubprocessExecutor",
    "SubprocessExecutor",
]
