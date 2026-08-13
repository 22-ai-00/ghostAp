import logging
import re
import shlex
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config import get_settings
from ..utils.errors import get_error_detail
from ..utils.redact import redact_sensitive
from ..utils.text import truncate_output

logger = logging.getLogger(__name__)


class SubprocessExecutor(ABC):
    """子进程执行抽象接口，便于测试模拟"""

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
        pass


class DefaultSubprocessExecutor(SubprocessExecutor):
    """默认的子进程执行实现，直接调用 subprocess.run"""

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


class SecurityCheckStrategy(ABC):
    """安全检查策略抽象基类"""

    @abstractmethod
    def check(self, command: str, settings) -> tuple[bool, Optional[str]]:
        """
        检查命令是否安全

        Returns:
            (is_safe, reason): 第一个元素表示是否安全，第二个元素表示拒绝原因（如果不安全）
        """
        pass


class WhitelistCheckStrategy(SecurityCheckStrategy):
    """白名单检查策略"""

    def check(self, command: str, settings) -> tuple[bool, Optional[str]]:
        if not settings.sandbox_use_whitelist:
            return True, None

        whitelist = settings.command_whitelist
        if not whitelist:
            return False, "白名单模式已启用但白名单为空"

        # 首先检查是否包含 shell 控制字符，这些字符可能用于执行多个命令
        shell_control_chars = [';', '&&', '||', '|', '`', '$(']
        for char in shell_control_chars:
            if char in command:
                return False, f"命令包含不允许的控制字符: {char}"

        # 检查括号等特殊字符
        if '(' in command or ')' in command or '{' in command or '}' in command:
            return False, "命令包含不允许的括号字符"

        try:
            # 使用 shlex.split() 安全解析命令，获取第一个 token（实际执行的命令）
            tokens = shlex.split(command.strip())
            if not tokens:
                return False, "命令为空"

            cmd_name = tokens[0].lower()

            # 检查命令名是否匹配白名单
            for allowed in whitelist:
                allowed_lower = allowed.strip().lower()
                if allowed_lower and (cmd_name == allowed_lower):
                    # 白名单匹配通过，继续后续检查
                    break
            else:
                return False, f"命令不在白名单中: {command}"
        except ValueError as e:
            # shlex.split() 在处理不匹配的引号时会抛出 ValueError
            return False, f"命令解析失败: {str(e)}"

        return True, None


class DangerousPatternCheckStrategy(SecurityCheckStrategy):
    """危险模式检查策略"""

    DANGEROUS_PATTERNS = [
        r"rm\s+(-[rf]+\s+)?/(?:['\"])?(?:$|\s|[;&|])",
        r"rm\s+(-[rf]+\s+)?/\*",
        r"mkfs\.",
        r"dd\s+if=",
        r">\s*/dev/sd[a-z]",
        r"shutdown",
        r"reboot",
        r"halt\b",
        r"poweroff",
        r"init\s+[06]",
        r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;",
        r"fork\s*bomb",
        r"chmod\s+(-[rR]+\s+)?777\s+/",
        r"chown\s+.*\s+/($|\s)",
    ]

    def __init__(self):
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.DANGEROUS_PATTERNS]

    def check(self, command: str, settings) -> tuple[bool, Optional[str]]:
        for pattern in self._compiled_patterns:
            if pattern.search(command):
                return False, f"命令包含危险操作模式: {pattern.pattern}"
        return True, None


class BlacklistCheckStrategy(SecurityCheckStrategy):
    """黑名单检查策略"""

    def check(self, command: str, settings) -> tuple[bool, Optional[str]]:
        for blacklisted in settings.command_blacklist:
            if blacklisted in command:
                return False, f"命令包含黑名单内容: {blacklisted}"
        return True, None


@dataclass
class ExecutionResult:
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


class SandboxExecutor:
    def __init__(self, settings=None, subprocess_executor: Optional[SubprocessExecutor] = None, security_strategies: Optional[List[SecurityCheckStrategy]] = None):
        self.settings = settings if settings is not None else get_settings()
        self.subprocess_executor = subprocess_executor if subprocess_executor is not None else DefaultSubprocessExecutor()

        # 如果未提供策略，则使用默认策略链
        if security_strategies is None:
            self.security_strategies = [
                WhitelistCheckStrategy(),
                BlacklistCheckStrategy(),
            ]
        else:
            self.security_strategies = security_strategies

    def is_command_safe(self, command: str) -> tuple[bool, Optional[str]]:
        # 依次执行所有安全检查策略
        for strategy in self.security_strategies:
            is_safe, reason = strategy.check(command, self.settings)
            if not is_safe:
                return False, reason
        return True, None

    def execute(
        self,
        command: str,
        cwd: Optional[str] = None,
        interactive: bool = True,
        chat_id: Optional[str] = None,
        env_overrides: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        is_safe, reason = self.is_command_safe(command)
        if not is_safe:
            return ExecutionResult(
                success=False, stdout="", stderr="", return_code=-1, error_message=f"安全检查未通过: {reason}"
            )

        # Debug sentinel: handler layer should already hold the repo lock.
        # Log a warning if we detect the lock is held by a different chat.
        # When sandbox_strict_lock_mode is True, raise LockConflictError
        # instead of just warning (AC-R05).
        if cwd and chat_id:
            try:
                from ..config import get_settings as _get_settings
                from ..repo_lock import LockConflictError, get_repo_lock_manager
                _rlm = get_repo_lock_manager()
                _info = _rlm.get_lock_info(cwd)
                if _info and _info.chat_id != chat_id:
                    _strict = True  # fail-close default
                    try:
                        _strict = _get_settings().sandbox_strict_lock_mode
                    except Exception:
                        logger.debug("failed to get strict_lock_mode setting", exc_info=True)
                    if _strict:
                        raise LockConflictError(
                            holder_chat_id=_info.chat_id,
                            locked_since=_info.acquired_at,
                            last_active=_info.last_active_time,
                        )
                    logger.warning(
                        "SandboxExecutor: repo %s locked by chat=%s but executing for chat=%s — "
                        "handler layer may have missed lock acquisition",
                        cwd, _info.chat_id[:12] if _info.chat_id else "?",
                        chat_id[:12] if chat_id else "?",
                    )
            except LockConflictError:
                raise  # re-raise strict-mode conflict
            except Exception:
                logger.debug("best-effort sentinel check failed", exc_info=True)

        return self._execute_command(command, cwd, interactive, env_overrides)

    def _execute_command(
        self,
        command: str,
        cwd: Optional[str],
        interactive: bool,
        env_overrides: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """Inner command execution logic (subprocess call)."""
        try:
            # 1) 尽可能把“会打开 pager 的命令”变成非交互式
            command = self._sanitize_command_for_noninteractive(command)

            # 2) 构建环境变量：禁用各种 pager，避免命令阻塞
            import os

            env = os.environ.copy()
            env.update(
                {
                    # git / man / systemd 等常见 pager
                    "GIT_PAGER": "cat",
                    "PAGER": "cat",
                    "MANPAGER": "cat",
                    "SYSTEMD_PAGER": "cat",
                    # less：F(一屏退出) R(支持颜色) X(不清屏)
                    "LESS": "FRX",
                    # 禁用 git 交互式提示（例如需要输入用户名密码时）
                    "GIT_TERMINAL_PROMPT": "0",
                    # 终端类型设为 dumb，尽量减少交互/控制序列
                    "TERM": "dumb",
                }
            )
            if env_overrides:
                env.update(env_overrides)

            # Detect shell and use it in interactive mode to load profiles/aliases
            shell_path = os.environ.get("SHELL", "/bin/bash")
            if interactive:
                cmd_args = [shell_path, "-i", "-c", command]
            else:
                # Use login shell (-l) to ensure environment variables (PATH, nvm, etc.) are loaded
                # even in non-interactive mode. This fixes "command not found" for user-installed tools.
                cmd_args = [shell_path, "-l", "-c", command]

            process = self.subprocess_executor.run(
                cmd_args,
                shell=False,
                capture_output=True,
                text=True,
                timeout=self.settings.sandbox_timeout,
                cwd=cwd,
                env=env,
            )

            stdout = process.stdout
            stderr = process.stderr
            max_len = self.settings.sandbox_max_output_length

            # Filter out interactive shell noise
            if stderr:
                ignore_patterns = [
                    "no job control in this shell",
                    "cannot set terminal process group",
                    "Inappropriate ioctl for device",
                    "bash: cannot set terminal process group",
                    "The input device is not a TTY",
                ]
                stderr = "\n".join(
                    [line for line in stderr.splitlines() if not any(p in line for p in ignore_patterns)]
                )

            stdout = truncate_output(stdout, max_len)
            stderr = truncate_output(stderr, max_len, label="错误输出被截断")

            stdout = redact_sensitive(stdout)
            stderr = redact_sensitive(stderr)

            return ExecutionResult(
                success=process.returncode == 0,
                stdout=stdout,
                stderr=stderr,
                return_code=process.returncode,
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                return_code=-1,
                error_message=f"命令执行超时（{self.settings.sandbox_timeout}秒）",
            )
        except Exception as e:
            return ExecutionResult(
                success=False, stdout="", stderr="", return_code=-1, error_message=f"执行异常: {get_error_detail(e)}"
            )

    def _sanitize_command_for_noninteractive(self, command: str) -> str:
        """Best-effort rewrite to avoid interactive pagers.

        目前主要覆盖 git：在检测到 `git ...` 且未显式要求分页时，自动加上 `--no-pager`。

        注意：此函数不做复杂 shell 解析，只做保守的前缀改写。
        """
        cmd = command.strip()
        if not cmd:
            return command

        # 用户显式要求分页/或已禁用 pager 时，不做改写
        lowered = cmd.lower()
        if "--no-pager" in lowered or "--paginate" in lowered or re.search(r"(^|\s)git\s+-p(\s|$)", lowered):
            return command

        # 常见形态：git xxx... / sudo git xxx...
        if re.match(r"^\s*(?:sudo\s+)?git\b", cmd):
            return re.sub(r"^(\s*(?:sudo\s+)?)git\b", r"\1git --no-pager", command, count=1)

        return command
