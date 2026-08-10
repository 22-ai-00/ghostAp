"""Claude Code CLI session backend."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Callable, Optional

from ..acp.claude_capabilities import strip_1m_suffix
from ..acp.client import (
    emit_referenced_changed_local_image_events,
    snapshot_local_image_artifacts,
)
from ..acp.models import ACPEvent, ACPEventType, PromptResult
from ..config import get_settings
from ..utils.errors import get_error_detail
from .employee_cli_sandbox import EmployeeCLISandbox
from .process_cleanup import terminate_and_reap_process_tree
from .protocol import _PromptRetryMixin

logger = logging.getLogger(__name__)

_CLI_TERMINATE_GRACE_S = 5.0
_CLI_KILL_GRACE_S = 3.0


def _terminate_and_reap_process(
    proc: subprocess.Popen,
    *,
    process_group_id: int | None = None,
) -> bool:
    """Bounded TERM→KILL cleanup for one CLI subprocess tree."""
    return terminate_and_reap_process_tree(
        proc,
        process_group_id=process_group_id,
        terminate_grace=_CLI_TERMINATE_GRACE_S,
        kill_grace=_CLI_KILL_GRACE_S,
        label="ClaudeCLI",
    )


@dataclass
class ClaudeCLIConfig:
    """Configuration knobs for Claude Code CLI backend."""

    command: str = "claude"
    add_dir: bool = True
    bypass_permissions: Optional[bool] = None  # None → use config.claude_cli_skip_permissions


class SyncClaudeCLISession(_PromptRetryMixin):
    """Claude Code CLI backend.

    - Uses `claude -p` (print and exit) per prompt.
    - Uses `--session-id` for the first prompt and `--resume <id>` afterwards.
    - Emits TEXT_CHUNK ACP events only (no plan/tool events).
    """

    def __init__(
        self,
        cwd: str,
        config: Optional[ClaudeCLIConfig] = None,
        *,
        model_name: Optional[str] = None,
        employee_process_env: Mapping[str, str] | None = None,
    ):
        self._cwd = cwd
        self._cfg = config or ClaudeCLIConfig()
        self._model_name = (model_name or "").strip() or None
        self._proc: Optional[subprocess.Popen] = None
        self._proc_group_id: int | None = None
        self._cancel_event = threading.Event()
        self._force_dead = False
        self._tool_filter = None
        self._employee_sandbox = (
            EmployeeCLISandbox(cwd=cwd, process_env=employee_process_env)
            if employee_process_env is not None
            else None
        )

        self.session_id: str = ""
        self.created_at: float = time.time()
        self.last_active: float = time.time()
        self.message_count: int = 0
        self.last_query: str = ""
        self.is_resumed: bool = False

    def describe_agent(self) -> str:
        return f"cmd={self._cfg.command} cwd={self._cwd} backend=cli"

    def start(self, startup_timeout: float = 60) -> str:
        # No long-running server here; just validate executable and mint a session id.
        if not shutil.which(self._cfg.command):
            raise RuntimeError(f"未找到 Claude CLI 可执行文件: {self._cfg.command}")
        if not self.session_id:
            self.session_id = str(uuid.uuid4())
        return self.session_id

    def load_session(self, session_id: str, timeout: float) -> None:
        # Claude CLI uses local persistence; we just switch to target session id.
        del timeout
        self.session_id = session_id
        self.is_resumed = True

    def load_local_history(self, session_id: Optional[str] = None, limit: int = 200) -> list[dict]:
        # Claude CLI manages its own history; GhostAP doesn't parse it here.
        return []

    def is_server_running(self) -> bool:
        # Per-prompt spawn — no persistent server to check.
        return True

    def is_server_healthy(self, healthcheck_timeout: float = 2.0) -> bool:
        return True

    @property
    def employee_process_env(self) -> dict[str, str] | None:
        sandbox = self._employee_sandbox
        return None if sandbox is None else sandbox.process_env

    def set_tool_filter(self, tool_filter) -> None:
        self._tool_filter = tool_filter

    def get_tool_filter(self):
        return self._tool_filter

    def configure_employee_sandbox(
        self,
        *,
        read_only_roots: Sequence[str],
        writable_roots: Sequence[str],
    ) -> None:
        if self._employee_sandbox is None:
            raise RuntimeError("employee CLI environment is unavailable")
        self._employee_sandbox.configure(
            command=self._cfg.command,
            read_only_roots=read_only_roots,
            writable_roots=writable_roots,
        )

    def _resolve_bypass_permissions(self) -> bool:
        """Allow permission bypass only inside the managed employee sandbox."""
        if self._cfg.bypass_permissions is not None:
            requested = self._cfg.bypass_permissions
        else:
            requested = get_settings().claude_cli_skip_permissions
        if requested and self._employee_sandbox is None:
            raise RuntimeError(
                "Claude 权限绕过仅允许在受控员工沙箱中使用"
            )
        return requested

    def send_prompt(
        self,
        text: str,
        on_event: Optional[Callable[[ACPEvent], None]] = None,
        timeout: Optional[int] = None,
    ) -> PromptResult:
        if not self.session_id:
            self.start()

        self._cancel_event.clear()
        self.last_active = time.time()
        self.message_count += 1
        self.last_query = text
        image_snapshot = (
            snapshot_local_image_artifacts(self._cwd)
            if on_event is not None
            else {}
        )
        media_references: list[str] = []

        def _build_args(resumed: bool) -> list[str]:
            args: list[str] = [self._cfg.command, "-p"]
            if self._employee_sandbox is not None:
                args.append("--bare")
            if self._cfg.add_dir:
                args += ["--add-dir", self._cwd]
            if self._resolve_bypass_permissions():
                args.append("--dangerously-skip-permissions")
            if self._model_name:
                args += ["--model", strip_1m_suffix(self._model_name)]

            if resumed:
                args += ["--resume", self.session_id]
            else:
                args += ["--session-id", self.session_id]

            args.append("--")
            args.append(text)
            return args

        def _run_once(resumed: bool) -> tuple[int, str, str, str]:
            """Run one claude invocation and return (returncode, stdout, stderr, state)."""
            args = _build_args(resumed)
            chunks: list[str] = []
            ensure_process_stopped: Callable[[], bool] | None = None
            try:
                # Claude Code CLI refuses to launch inside another Claude Code session.
                # Our process may run under Claude Code / other wrappers, so we must
                # explicitly unset the guard env to avoid nested-session crash.
                if self._employee_sandbox is not None:
                    env = self._employee_sandbox.process_env
                    args = self._employee_sandbox.wrap_argv(args)
                else:
                    from ..utils.env import build_clean_env

                    env = build_clean_env()
                from ..utils.env import apply_anthropic_betas

                apply_anthropic_betas(env, self._model_name)

                self._proc = subprocess.Popen(
                    args,
                    cwd=self._cwd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=(os.name == "posix"),
                )
                self._proc_group_id = (
                    getattr(self._proc, "pid", None)
                    if os.name == "posix"
                    else None
                )

                deadline = (time.monotonic() + timeout) if timeout else None
                assert self._proc.stdout is not None

                # Watchdog thread: terminates the process on timeout or cancel
                # since blocking readline cannot check these conditions.
                terminated_reason: list[str] = []
                proc_ref = self._proc
                proc_group_id = self._proc_group_id
                cleanup_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
                cleanup_done = threading.Event()
                cleanup_succeeded = [False]

                def _ensure_process_stopped() -> bool:
                    with cleanup_lock:
                        if cleanup_done.is_set():
                            return cleanup_succeeded[0]
                        cleanup_succeeded[0] = _terminate_and_reap_process(
                            proc_ref,
                            process_group_id=proc_group_id,
                        )
                        cleanup_done.set()
                        if not cleanup_succeeded[0]:
                            self._force_dead = True
                        return cleanup_succeeded[0]

                ensure_process_stopped = _ensure_process_stopped

                def _watchdog():
                    while True:
                        if self._cancel_event.is_set():
                            terminated_reason.append("cancelled")
                            _ensure_process_stopped()
                            return
                        try:
                            leader_exited = proc_ref.poll() is not None
                        except Exception:
                            _ensure_process_stopped()
                            return
                        if leader_exited:
                            # A descendant may still own stdout/stderr after the
                            # CLI leader exits. Converge the whole process group
                            # before letting the blocking reader wait for EOF.
                            _ensure_process_stopped()
                            return
                        if deadline and time.monotonic() > deadline:
                            terminated_reason.append("timeout")
                            _ensure_process_stopped()
                            return
                        wait_timeout = 0.1
                        if deadline is not None:
                            wait_timeout = min(
                                wait_timeout,
                                max(0.0, deadline - time.monotonic()),
                            )
                        self._cancel_event.wait(timeout=wait_timeout)

                watchdog_thread = threading.Thread(target=_watchdog, daemon=True)
                watchdog_thread.start()

                for line in self._proc.stdout:
                    if terminated_reason:
                        break
                    if self._cancel_event.is_set():
                        _ensure_process_stopped()
                        return (1, "".join(chunks), "", "cancelled")
                    if deadline and time.monotonic() > deadline:
                        try:
                            leader_running = proc_ref.poll() is None
                        except Exception:
                            leader_running = True
                        if leader_running:
                            _ensure_process_stopped()
                            return (1, "".join(chunks), "", "timeout")
                    chunks.append(line)
                    if on_event:
                        on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=line))

                if terminated_reason:
                    _ensure_process_stopped()
                    return (1, "".join(chunks), "", terminated_reason[0])
                try:
                    self._proc.wait(timeout=_CLI_TERMINATE_GRACE_S)
                except subprocess.TimeoutExpired:
                    reaped = _ensure_process_stopped()
                    if not reaped:
                        self._force_dead = True
                    if deadline and time.monotonic() >= deadline:
                        return (1, "".join(chunks), "", "timeout")
                    raise

                rc = int(self._proc.returncode or 0)
                err = (self._proc.stderr.read() or "").strip("\n") if self._proc.stderr else ""
                return (rc, "".join(chunks).strip("\n"), err, "ok")
            finally:
                proc = self._proc
                proc_group_id = self._proc_group_id
                if proc is None:
                    self._proc = None
                    self._proc_group_id = None
                elif (
                    ensure_process_stopped()
                    if ensure_process_stopped is not None
                    else _terminate_and_reap_process(
                        proc,
                        process_group_id=proc_group_id,
                    )
                ):
                    self._proc = None
                    self._proc_group_id = None
                else:
                    self._force_dead = True
                    self._proc = proc
                    self._proc_group_id = proc_group_id

        try:
            rc, out, err, state = _run_once(resumed=self.is_resumed)
            media_references.extend((out, err))

            if state == "cancelled":
                self.is_resumed = True
                return PromptResult(stop_reason="cancelled", text=out)
            if state == "timeout":
                self.is_resumed = True
                timeout_text = (out + "\n❌ Claude 执行超时").strip()
                return PromptResult(stop_reason="timeout", text=timeout_text)

            output = out
            if rc != 0 and err:
                output = (output + "\n" + err).strip("\n")
                if on_event:
                    on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="\n" + err))

            self.is_resumed = True
            stop_reason = "end_turn" if rc == 0 else "failed"
            return PromptResult(stop_reason=stop_reason, text=output)

        except (subprocess.SubprocessError, OSError, TimeoutError) as e:
            self.is_resumed = True
            return PromptResult(stop_reason="error", text=f"❌ Claude 执行异常: {get_error_detail(e)}")
        finally:
            if on_event is not None:
                try:
                    emit_referenced_changed_local_image_events(
                        self._cwd,
                        image_snapshot,
                        media_references,
                        on_event,
                    )
                except Exception:
                    logger.warning(
                        "[ClaudeCLI] local image artifact discovery failed",
                        exc_info=True,
                    )

    def cancel(self, wait: bool = False, timeout: float = 2.0) -> None:
        """Signal cancellation — the streaming loop will terminate the process."""
        del wait, timeout  # CLI transport has no acknowledgment protocol.
        self._cancel_event.set()
        proc = self._proc
        if proc:
            try:
                proc.terminate()
            except Exception:
                logger.debug("SyncClaudeCLISession.cancel: terminate failed", exc_info=True)

    def close(self) -> None:
        proc = self._proc
        if proc is None:
            return
        if _terminate_and_reap_process(
            proc,
            process_group_id=self._proc_group_id,
        ):
            self._proc = None
            self._proc_group_id = None
            return
        self._force_dead = True
        self._proc = proc
        raise RuntimeError("Claude CLI failed to terminate subprocess")

    def to_snapshot(self) -> dict:
        return {
            "session_id": self.session_id,
            "agent_type": "claude",
            "cwd": self._cwd,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "message_count": self.message_count,
            "last_query": self.last_query,
            "is_resumed": self.is_resumed,
            "backend": "cli",
            "model_name": self._model_name,
        }

    def get_session_info(self) -> str:
        duration = int(time.time() - self.created_at)
        minutes, seconds = divmod(duration, 60)
        resumed_info = " (已恢复)" if self.is_resumed else ""
        return (
            f"📊 Claude 会话信息{resumed_info} (CLI):\n"
            f"- 会话ID: {self.session_id}\n"
            f"- 消息数: {self.message_count}\n"
            f"- 持续时间: {minutes}分{seconds}秒"
        )
