"""Codex CLI (``codex-w`` bridge) session backend."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable, Optional

from ..acp.client import (
    emit_referenced_changed_local_image_events,
    snapshot_local_image_artifacts,
)
from ..acp.model_selection import (
    CODEX_DEFAULT_MODEL_TOKEN,
    split_codex_model_selection,
)
from ..acp.models import ACPEvent, ACPEventType, PromptResult
from ..acp.prompt_generation import PromptGenerationTracker
from ..utils.errors import get_error_detail
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
        label="CodexCLI",
    )


@dataclass
class CodexCLIConfig:
    """Configuration knobs for the Codex CLI bridge backend."""

    command: str = "codex"
    add_dir: bool = True


class SyncCodexCLISession(_PromptRetryMixin, PromptGenerationTracker):
    """Codex CLI backend bridged through the local ``codex-w`` executable.

    - Uses ``codex exec`` (print and exit) per prompt.
    - The first prompt starts a fresh thread; later prompts call
      ``codex exec resume <thread_id>``.
    - Emits TEXT_CHUNK ACP events for assistant ``agent_message`` items only.

    Two ``codex`` CLI quirks shape the argument layout (both verified against
    codex-cli 0.154.0 through the ``codex-w`` gateway wrapper):

    1. ``-c key=value`` overrides must be placed *before* the ``exec``
       subcommand.  Overrides placed after ``exec`` make codex fail to resolve
       model metadata and fall back to a ChatGPT-account route that rejects the
       LLMBox model with HTTP 400.
    2. A working directory outside a trusted git repository requires
       ``--skip-git-repo-check``; passing it unconditionally is harmless inside
       a repository.
    """

    def __init__(
        self,
        cwd: str,
        config: Optional[CodexCLIConfig] = None,
        *,
        model_name: Optional[str] = None,
        employee_process_env: Mapping[str, str] | None = None,
    ):
        self._cwd = cwd
        self._cfg = config or CodexCLIConfig()
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

        # The Codex thread id is assigned by the backend and surfaces in the
        # first ``thread.started`` JSONL event; it stays empty until then.
        self.session_id: str = ""
        self.created_at: float = time.time()
        self.last_active: float = time.time()
        self.message_count: int = 0
        self.last_query: str = ""
        self.is_resumed: bool = False

    def describe_agent(self) -> str:
        return f"cmd={self._cfg.command} cwd={self._cwd} backend=cli"

    def start(self, startup_timeout: float = 60) -> str:
        # No long-running server; just validate the executable is on PATH.
        del startup_timeout
        if not shutil.which(self._cfg.command):
            raise RuntimeError(f"未找到 Codex CLI 可执行文件: {self._cfg.command}")
        return self.session_id

    def load_session(self, session_id: str, timeout: float) -> None:
        # Codex CLI keeps its own rollout persistence; just target the thread.
        del timeout
        self.session_id = session_id
        self.is_resumed = True

    def load_local_history(self, session_id: Optional[str] = None, limit: int = 200) -> list[dict]:
        return []

    def is_server_running(self) -> bool:
        # Per-prompt spawn — no persistent server to check.
        return True

    def is_server_healthy(self, healthcheck_timeout: float = 2.0) -> bool:
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
            if str(result.stop_reason or "").strip().casefold() in {
                "cancelled",
                "canceled",
            }:
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
        if not shutil.which(self._cfg.command):
            raise RuntimeError(f"未找到 Codex CLI 可执行文件: {self._cfg.command}")

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
            # Global ``-c`` overrides MUST precede the ``exec`` subcommand;
            # codex ignores/mis-resolves them afterwards (model metadata 400).
            args: list[str] = [self._cfg.command]
            base_model, effort = split_codex_model_selection(self._model_name)
            if base_model and base_model != CODEX_DEFAULT_MODEL_TOKEN:
                args += ["-c", f'model="{base_model}"']
            if effort:
                args += ["-c", f'model_reasoning_effort="{effort}"']

            args += [
                "exec",
                "--json",
                "--skip-git-repo-check",
                "-C",
                self._cwd,
            ]
            if resumed and self.session_id:
                args += ["resume", self.session_id]
            args.append(text)
            return args

        def _run_once(resumed: bool) -> tuple[int, str, str, str, str]:
            """Run one codex invocation.

            Returns (returncode, assistant_text, stderr, state, backend_error).
            """
            args = _build_args(resumed)
            ensure_process_stopped: Callable[[], bool] | None = None
            assistant_chunks: list[str] = []
            backend_error = ""
            try:
                from ..utils.env import build_clean_env

                env = build_clean_env(
                    dict(self._employee_process_env)
                    if self._employee_process_env is not None
                    else None
                )

                self._proc = subprocess.Popen(
                    args,
                    cwd=self._cwd,
                    env=env,
                    stdin=subprocess.DEVNULL,
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

                started_at = time.monotonic()
                deadline = (started_at + timeout) if timeout else None
                effective_idle_timeout = (
                    float(idle_timeout)
                    if idle_timeout is not None and float(idle_timeout) > 0
                    else 0.0
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
                            logger.warning(
                                "Codex CLI activity predicate failed closed",
                                exc_info=True,
                            )
                    if counts_as_activity:
                        last_activity_at[0] = time.monotonic()
                    on_event(event)

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
                        if (
                            effective_idle_timeout > 0
                            and time.monotonic() - last_activity_at[0]
                            >= effective_idle_timeout
                        ):
                            terminated_reason.append("idle_timeout")
                            _ensure_process_stopped()
                            return
                        wait_timeout = 0.1
                        if deadline is not None:
                            wait_timeout = min(
                                wait_timeout,
                                max(0.0, deadline - time.monotonic()),
                            )
                        if effective_idle_timeout > 0:
                            wait_timeout = min(
                                wait_timeout,
                                max(
                                    0.0,
                                    effective_idle_timeout
                                    - (time.monotonic() - last_activity_at[0]),
                                ),
                            )
                        self._cancel_event.wait(timeout=wait_timeout)

                watchdog_thread = threading.Thread(target=_watchdog, daemon=True)
                watchdog_thread.start()

                for line in self._proc.stdout:
                    if terminated_reason:
                        break
                    if self._cancel_event.is_set():
                        _ensure_process_stopped()
                        return (1, "".join(assistant_chunks), "", "cancelled", "")
                    if deadline and time.monotonic() > deadline:
                        try:
                            leader_running = proc_ref.poll() is None
                        except Exception:
                            leader_running = True
                        if leader_running:
                            _ensure_process_stopped()
                            return (1, "".join(assistant_chunks), "", "timeout", "")

                    payload = line.strip()
                    if payload:
                        event_error = self._consume_event(
                            payload,
                            assistant_chunks,
                            lambda text: _emit(
                                ACPEvent(
                                    event_type=ACPEventType.TEXT_CHUNK,
                                    text=text,
                                )
                            ),
                        )
                        if event_error and not backend_error:
                            backend_error = event_error

                if terminated_reason:
                    _ensure_process_stopped()
                    return (
                        1,
                        "".join(assistant_chunks),
                        "",
                        terminated_reason[0],
                        "",
                    )
                try:
                    self._proc.wait(timeout=_CLI_TERMINATE_GRACE_S)
                except subprocess.TimeoutExpired:
                    reaped = _ensure_process_stopped()
                    if not reaped:
                        self._force_dead = True
                    if deadline and time.monotonic() >= deadline:
                        return (1, "".join(assistant_chunks), "", "timeout", "")
                    raise

                rc = int(self._proc.returncode or 0)
                err = (self._proc.stderr.read() or "").strip("\n") if self._proc.stderr else ""
                # Keep codex's noisy Rust tracing out of the surfaced stderr.
                err = "\n".join(
                    ln
                    for ln in err.splitlines()
                    if not ln.startswith(("20", "Reading additional input"))
                ).strip("\n")
                if not backend_error and rc != 0:
                    backend_error = err or f"codex exec 退出码 {rc}"
                state = "ok" if rc == 0 and not backend_error else "failed"
                return (rc, "".join(assistant_chunks).strip("\n"), err, state, backend_error)
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
            rc, out, err, state, backend_error = _run_once(resumed=self.is_resumed)
            media_references.extend((out, err))

            if state == "cancelled":
                self.is_resumed = True
                return PromptResult(stop_reason="cancelled", text=out)
            if state in {"timeout", "idle_timeout"}:
                self.is_resumed = True
                reason = "空闲超时" if state == "idle_timeout" else "执行超时"
                timeout_text = (out + f"\n❌ Codex {reason}").strip()
                return PromptResult(stop_reason="timeout", text=timeout_text)

            output = out
            if state == "failed":
                detail = backend_error or err
                if detail:
                    output = (output + "\n" + detail).strip("\n")
                    if on_event:
                        on_event(
                            ACPEvent(
                                event_type=ACPEventType.TEXT_CHUNK,
                                text="\n" + detail,
                            )
                        )
                self.is_resumed = bool(self.session_id)
                return PromptResult(stop_reason="failed", text=output)

            # A successful turn establishes/resumes a backend thread.
            self.is_resumed = bool(self.session_id)
            del rc
            return PromptResult(stop_reason="end_turn", text=output)

        except (subprocess.SubprocessError, OSError, TimeoutError) as e:
            self.is_resumed = bool(self.session_id)
            return PromptResult(
                stop_reason="error",
                text=f"❌ Codex 执行异常: {get_error_detail(e)}",
            )
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
                        "[CodexCLI] local image artifact discovery failed",
                        exc_info=True,
                    )

    def _consume_event(
        self,
        payload: str,
        assistant_chunks: list[str],
        emit_text: Callable[[str], None],
    ) -> str:
        """Parse one ``--json`` JSONL event.

        Returns a non-empty human-readable message for a fatal backend error,
        or an empty string otherwise.
        """
        try:
            event = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return ""
        event_type = str(event.get("type") or "")

        if event_type == "thread.started":
            thread_id = str(event.get("thread_id") or "").strip()
            if thread_id and not self.session_id:
                self.session_id = thread_id
            return ""

        if event_type in {"error", "turn.failed"}:
            message = self._extract_error_message(event)
            if message:
                logger.warning("[CodexCLI] backend error: %s", message)
            return message

        if event_type != "item.completed":
            return ""

        item = event.get("item") or {}
        if str(item.get("type") or "") != "agent_message":
            return ""
        text = str(item.get("text") or "")
        if not text:
            return ""
        assistant_chunks.append(text)
        emit_text(text)
        return ""

    @staticmethod
    def _extract_error_message(event: dict) -> str:
        """Pull a human-readable message out of error / turn.failed payloads."""
        raw = event.get("error")
        if isinstance(raw, dict):
            inner = raw.get("message")
            if isinstance(inner, str):
                # Codex sometimes double-encodes a JSON error body as a string.
                try:
                    parsed = json.loads(inner)
                    nested = (
                        parsed.get("error", {}).get("message")
                        if isinstance(parsed, dict)
                        else None
                    )
                    if isinstance(nested, str) and nested:
                        return nested
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass
                return inner
        message = event.get("message")
        if isinstance(message, str):
            return message
        item = event.get("item")
        if isinstance(item, dict):
            inner_message = item.get("message")
            if isinstance(inner_message, str):
                return inner_message
        return ""

    def cancel(self, wait: bool = False, timeout: float = 2.0) -> None:
        """Signal cancellation — the streaming loop will terminate the process."""
        del wait, timeout  # CLI transport has no acknowledgment protocol.
        self._cancel_event.set()
        proc = self._proc
        if proc:
            try:
                proc.terminate()
            except Exception:
                logger.debug("SyncCodexCLISession.cancel: terminate failed", exc_info=True)

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
        raise RuntimeError("Codex CLI failed to terminate subprocess")

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
            f"📊 Codex 会话信息{resumed_info} (CLI):\n"
            f"- 会话ID: {self.session_id}\n"
            f"- 消息数: {self.message_count}\n"
            f"- 持续时间: {minutes}分{seconds}秒"
        )
