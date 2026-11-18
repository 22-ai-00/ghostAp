"""Shared lifecycle implementation for per-prompt CLI session backends."""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Optional

from ..acp.client import (
    emit_referenced_changed_local_image_events,
    snapshot_local_image_artifacts,
)
from ..acp.models import ACPEvent, PromptResult
from ..acp.prompt_generation import PromptGenerationTracker
from ..utils.errors import get_error_detail
from .protocol import _PromptRetryMixin


@dataclass(frozen=True)
class _CLIModuleHooks:
    """Late-bound module helpers so tests can patch subclass module globals."""

    popen: Callable[..., subprocess.Popen]
    which: Callable[[str], Optional[str]]
    terminate_and_reap: Callable[..., bool]
    terminate_grace_s: float
    logger: logging.Logger


class _BaseCLISession(_PromptRetryMixin, PromptGenerationTracker):
    """Shared process/watchdog/retry lifecycle for CLI print-and-exit backends."""

    _display_name: str = "CLI"
    _log_prefix: str = "CLI"
    _missing_binary_prefix: str = "CLI"
    _stdin_devnull: bool = False

    def __init__(
        self,
        cwd: str,
        config: Any,
        *,
        model_name: Optional[str] = None,
        employee_process_env: Mapping[str, str] | None = None,
    ) -> None:
        self._cwd = cwd
        self._cfg = config
        self._model_name = (model_name or "").strip() or None
        self._proc: Optional[subprocess.Popen] = None
        self._proc_group_id: int | None = None
        self._cancel_event = threading.Event()
        self._prompt_generation_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._prompt_generation = 0
        self._active_prompt_generation: int | None = None
        self._user_cancel_generation: int | None = None
        self._force_dead = False
        self._employee_process_env = (
            dict(employee_process_env) if employee_process_env is not None else None
        )

        self.session_id: str = ""
        self.created_at: float = time.time()
        self.last_active: float = time.time()
        self.message_count: int = 0
        self.last_query: str = ""
        self.is_resumed: bool = False

    def _module_hooks(self) -> _CLIModuleHooks:
        raise NotImplementedError

    def _build_args(self, text: str, resumed: bool) -> list[str]:
        raise NotImplementedError

    def _prepare_env(self, env: dict[str, str]) -> None:
        del env

    def _ensure_session_ready(self, which: Callable[[str], Optional[str]]) -> None:
        del which

    def _consume_stdout_line(
        self,
        line: str,
        output_chunks: list[str],
        emit: Callable[[ACPEvent], None],
    ) -> str:
        raise NotImplementedError

    def _filter_stderr(self, err: str) -> str:
        return err

    def _handle_process_exit(
        self,
        *,
        rc: int,
        out: str,
        err: str,
        state: str,
        backend_error: str,
        on_event: Optional[Callable[[ACPEvent], None]],
    ) -> PromptResult:
        raise NotImplementedError

    def _mark_resumed_after_error(self) -> None:
        self.is_resumed = True

    def describe_agent(self) -> str:
        return f"cmd={self._cfg.command} cwd={self._cwd} backend=cli"

    def start(self, startup_timeout: float = 60) -> str:
        del startup_timeout
        hooks = self._module_hooks()
        if not hooks.which(self._cfg.command):
            raise RuntimeError(f"未找到 {self._missing_binary_prefix} 可执行文件: {self._cfg.command}")
        return self.session_id

    def load_session(self, session_id: str, timeout: float) -> None:
        del timeout
        self.session_id = session_id
        self.is_resumed = True

    def load_local_history(self, session_id: Optional[str] = None, limit: int = 200) -> list[dict]:
        del session_id, limit
        return []

    def is_server_running(self) -> bool:
        return True

    def is_server_healthy(self, healthcheck_timeout: float = 2.0) -> bool:
        del healthcheck_timeout
        return True

    @property
    def employee_process_env(self) -> dict[str, str] | None:
        env = self._employee_process_env
        return None if env is None else dict(env)

    def send_prompt(
        self,
        text: str,
        on_event: Optional[Callable[[ACPEvent], None]] = None,
        timeout: Optional[float] = None,
        idle_timeout: Optional[float] = None,
        activity_predicate: Optional[Callable[[ACPEvent], bool]] = None,
    ) -> PromptResult:
        self._cancel_event.clear()
        prompt_generation = self._begin_prompt_generation()
        generation_consumed = False
        try:
            result = self._send_prompt_once(
                text,
                on_event=on_event,
                timeout=timeout,
                idle_timeout=idle_timeout,
                activity_predicate=activity_predicate,
            )
            user_cancelled = self._consume_prompt_generation(prompt_generation)
            generation_consumed = True
            if str(result.stop_reason or "").strip().casefold() in {"cancelled", "canceled"}:
                if user_cancelled:
                    result.cancellation_source = "user"
                elif result.cancellation_source is None:
                    result.cancellation_source = "provider"
            return result
        finally:
            if not generation_consumed:
                self._consume_prompt_generation(prompt_generation)

    def _send_prompt_once(
        self,
        text: str,
        on_event: Optional[Callable[[ACPEvent], None]] = None,
        timeout: Optional[float] = None,
        idle_timeout: Optional[float] = None,
        activity_predicate: Optional[Callable[[ACPEvent], bool]] = None,
    ) -> PromptResult:
        hooks = self._module_hooks()
        self._ensure_session_ready(hooks.which)

        self.last_active = time.time()
        self.message_count += 1
        self.last_query = text
        image_snapshot = snapshot_local_image_artifacts(self._cwd) if on_event is not None else {}
        media_references: list[str] = []

        try:
            rc, out, err, state, backend_error = self._run_once(
                text=text,
                resumed=self.is_resumed,
                hooks=hooks,
                on_event=on_event,
                timeout=timeout,
                idle_timeout=idle_timeout,
                activity_predicate=activity_predicate,
            )
            media_references.extend((out, err))
            if backend_error and backend_error not in media_references:
                media_references.append(backend_error)

            if state == "cancelled":
                self.is_resumed = True
                return PromptResult(stop_reason="cancelled", text=out)
            if state in {"timeout", "idle_timeout"}:
                self.is_resumed = True
                reason = "空闲超时" if state == "idle_timeout" else "执行超时"
                return PromptResult(stop_reason="timeout", text=(out + f"\n❌ {self._display_name} {reason}").strip())

            return self._handle_process_exit(
                rc=rc,
                out=out,
                err=err,
                state=state,
                backend_error=backend_error,
                on_event=on_event,
            )
        except (subprocess.SubprocessError, OSError, TimeoutError) as e:
            self._mark_resumed_after_error()
            return PromptResult(stop_reason="error", text=f"❌ {self._display_name} 执行异常: {get_error_detail(e)}")
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
                    hooks.logger.warning(
                        "[%s] local image artifact discovery failed",
                        self._log_prefix,
                        exc_info=True,
                    )

    def _run_once(
        self,
        *,
        text: str,
        resumed: bool,
        hooks: _CLIModuleHooks,
        on_event: Optional[Callable[[ACPEvent], None]],
        timeout: Optional[float],
        idle_timeout: Optional[float],
        activity_predicate: Optional[Callable[[ACPEvent], bool]],
    ) -> tuple[int, str, str, str, str]:
        args = self._build_args(text, resumed)
        output_chunks: list[str] = []
        ensure_process_stopped: Callable[[], bool] | None = None
        backend_error = ""

        try:
            from ..utils.env import build_clean_env

            env = build_clean_env(
                dict(self._employee_process_env) if self._employee_process_env is not None else None
            )
            self._prepare_env(env)

            popen_kwargs: dict[str, Any] = {
                "cwd": self._cwd,
                "env": env,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "start_new_session": (os.name == "posix"),
            }
            if self._stdin_devnull:
                popen_kwargs["stdin"] = subprocess.DEVNULL

            self._proc = hooks.popen(args, **popen_kwargs)
            self._proc_group_id = getattr(self._proc, "pid", None) if os.name == "posix" else None

            started_at = time.monotonic()
            deadline = (started_at + timeout) if timeout else None
            effective_idle_timeout = (
                float(idle_timeout) if idle_timeout is not None and float(idle_timeout) > 0 else 0.0
            )
            last_activity_at = [started_at]

            def _emit(event: ACPEvent) -> None:
                if on_event is None:
                    return
                counts_as_activity = True
                if activity_predicate is not None:
                    try:
                        counts_as_activity = bool(activity_predicate(event))
                    except Exception:
                        counts_as_activity = False
                        hooks.logger.warning(
                            "%s activity predicate failed closed",
                            self._display_name,
                            exc_info=True,
                        )
                if counts_as_activity:
                    last_activity_at[0] = time.monotonic()
                on_event(event)

            assert self._proc.stdout is not None
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
                    cleanup_succeeded[0] = hooks.terminate_and_reap(
                        proc_ref,
                        process_group_id=proc_group_id,
                    )
                    cleanup_done.set()
                    if not cleanup_succeeded[0]:
                        self._force_dead = True
                    return cleanup_succeeded[0]

            ensure_process_stopped = _ensure_process_stopped

            def _watchdog() -> None:
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
                    now = time.monotonic()
                    if deadline and now > deadline:
                        terminated_reason.append("timeout")
                        _ensure_process_stopped()
                        return
                    if effective_idle_timeout > 0 and now - last_activity_at[0] >= effective_idle_timeout:
                        terminated_reason.append("idle_timeout")
                        _ensure_process_stopped()
                        return
                    wait_timeout = 0.1
                    if deadline is not None:
                        wait_timeout = min(wait_timeout, max(0.0, deadline - now))
                    if effective_idle_timeout > 0:
                        wait_timeout = min(wait_timeout, max(0.0, effective_idle_timeout - (now - last_activity_at[0])))
                    self._cancel_event.wait(timeout=wait_timeout)

            threading.Thread(target=_watchdog, daemon=True).start()

            for line in self._proc.stdout:
                if terminated_reason:
                    break
                if self._cancel_event.is_set():
                    _ensure_process_stopped()
                    return (1, "".join(output_chunks), "", "cancelled", "")
                if deadline and time.monotonic() > deadline:
                    try:
                        leader_running = proc_ref.poll() is None
                    except Exception:
                        leader_running = True
                    if leader_running:
                        _ensure_process_stopped()
                        return (1, "".join(output_chunks), "", "timeout", "")

                event_error = self._consume_stdout_line(line, output_chunks, _emit)
                if event_error and not backend_error:
                    backend_error = event_error

            if terminated_reason:
                _ensure_process_stopped()
                return (1, "".join(output_chunks), "", terminated_reason[0], "")
            try:
                self._proc.wait(timeout=hooks.terminate_grace_s)
            except subprocess.TimeoutExpired:
                if not _ensure_process_stopped():
                    self._force_dead = True
                if deadline and time.monotonic() >= deadline:
                    return (1, "".join(output_chunks), "", "timeout", "")
                raise

            rc = int(self._proc.returncode or 0)
            err = (self._proc.stderr.read() or "").strip("\n") if self._proc.stderr else ""
            err = self._filter_stderr(err)
            return (rc, "".join(output_chunks).strip("\n"), err, "ok", backend_error)
        finally:
            proc = self._proc
            proc_group_id = self._proc_group_id
            if proc is None:
                self._proc = None
                self._proc_group_id = None
            elif (
                ensure_process_stopped()
                if ensure_process_stopped is not None
                else hooks.terminate_and_reap(proc, process_group_id=proc_group_id)
            ):
                self._proc = None
                self._proc_group_id = None
            else:
                self._force_dead = True
                self._proc = proc
                self._proc_group_id = proc_group_id

    def cancel(self, wait: bool = False, timeout: float = 2.0) -> None:
        """Signal cancellation — the streaming loop will terminate the process."""
        del wait, timeout
        self._cancel_event.set()
        proc = self._proc
        if proc:
            try:
                proc.terminate()
            except Exception:
                self._module_hooks().logger.debug("%s.cancel: terminate failed", type(self).__name__, exc_info=True)

    def close(self) -> None:
        proc = self._proc
        if proc is None:
            return
        if self._module_hooks().terminate_and_reap(proc, process_group_id=self._proc_group_id):
            self._proc = None
            self._proc_group_id = None
            return
        self._force_dead = True
        self._proc = proc
        raise RuntimeError(f"{self._display_name} CLI failed to terminate subprocess")

    def to_snapshot(self) -> dict:
        from .backend_resolver import agent_type_for_cli_command

        return {
            "session_id": self.session_id,
            "agent_type": agent_type_for_cli_command(self._cfg.command),
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
            f"📊 {self._display_name} 会话信息{resumed_info} (CLI):\n"
            f"- 会话ID: {self.session_id}\n"
            f"- 消息数: {self.message_count}\n"
            f"- 持续时间: {minutes}分{seconds}秒"
        )
