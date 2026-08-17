"""Synchronous adapter for ACPSession.

Existing GhostAP code is synchronous (threading-based). This adapter runs
an asyncio event loop in a dedicated daemon thread and exposes synchronous
methods that bridge to the async ACPSession.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import logging
import subprocess
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

from ..config import get_settings
from ..employee_session_scope import record_employee_session_outcome
from ..utils.errors import get_error_detail, sanitize_futures_msg
from .client import ACPHistoryStore
from .diagnostics import (
    DEFAULT_DIAGNOSTICS_SNIPPET_LIMIT,
    get_diagnostics_config,
    normalize_startup_diagnostics,
    safe_str,
    truncate_text,
)
from .diagnostics import (
    format_startup_diagnostics_summary as format_startup_diagnostics,
)
from .models import ACPEvent, PromptResult
from .prompt_generation import PromptGenerationTracker
from .session import ACPResumeRejected, ACPSession, ACPStartupError
from .startup_utils import initial_startup_diagnostics

logger = logging.getLogger(__name__)

_PROMPT_CANCEL_DRAIN_TIMEOUT_S = 5.0

def build_startup_diagnostics(
    *,
    agent_type: str,
    cwd: str,
    model_name: Optional[str],
    session: object = None,
    error: Exception,
    attempt: Optional[int] = None,
    retries: Optional[int] = None,
    timeout_s: Optional[float] = None,
    snippet_limit: int = DEFAULT_DIAGNOSTICS_SNIPPET_LIMIT,
) -> dict:
    """Collect startup evidence; normalization is owned by diagnostics.py."""
    diag = initial_startup_diagnostics(
        agent_type=agent_type,
        cwd=cwd,
        model_name=model_name,
        error=error,
        attempt=attempt,
        retries=retries,
        timeout_s=timeout_s,
    )
    try:
        cfg_limit = int(
            get_diagnostics_config(get_settings_fn=get_settings).snippet_limit or 0
        )
    except (TypeError, ValueError):
        cfg_limit = 0
    limit = cfg_limit or int(snippet_limit or DEFAULT_DIAGNOSTICS_SNIPPET_LIMIT)

    def _args(value: object) -> list[str]:
        if isinstance(value, (str, bytes, bytearray)):
            return [safe_str(value)] if value else []
        try:
            return [safe_str(item) for item in (value or [])]
        except (TypeError, ValueError):
            return []

    session_cmd = getattr(session, "_agent_cmd", "") if session is not None else ""
    session_args = getattr(session, "_agent_args", ()) if session is not None else ()
    diag["cmd"] = safe_str(session_cmd or getattr(error, "agent_cmd", ""))
    diag["args"] = _args(session_args or getattr(error, "agent_args", ()))
    if not diag["cmd"] and not diag["args"]:
        try:
            command, arguments = resolve_agent_spec(agent_type, model_name=model_name)
            diag["cmd"], diag["args"] = safe_str(command), _args(arguments)
        except (AgentSpecResolveError, OSError, RuntimeError, ValueError):
            diag["cmd"] = safe_str(agent_type)

    returncode = getattr(error, "returncode", None)
    if returncode is None and session is not None:
        acp_session = getattr(session, "_acp_session", None)
        returncode = getattr(getattr(acp_session, "_proc", None), "returncode", None)
    try:
        diag["rc"] = int(returncode) if returncode is not None else None
    except (TypeError, ValueError):
        diag["rc"] = None

    stdout = getattr(error, "stdout_snippet", "") or getattr(error, "stdout", "")
    stderr = getattr(error, "stderr_snippet", "") or getattr(error, "stderr", "")
    diag["stdout_snippet"] = truncate_text(safe_str(stdout), limit)
    diag["stderr_snippet"] = truncate_text(safe_str(stderr), limit)

    phase = safe_str(getattr(error, "fail_phase", "")).strip()
    if not phase:
        blob = "\n".join(
            (safe_str(error), diag["stdout_snippet"], diag["stderr_snippet"])
        ).casefold()
        if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)):
            phase = "timeout"
        elif "stdin is not a terminal" in blob or "stdin-not-tty" in blob:
            phase = "stdin_not_tty"
        elif any(
            marker in blob
            for marker in ("invalid model", "model must be one of", "unknown model")
        ) or ("invalid value" in blob and "--model" in blob):
            phase = "invalid_model"
        else:
            phase = "start_failed"
    diag["fail_phase"] = phase
    diag["fail_reason"] = safe_str(
        getattr(error, "fail_reason", "") or phase
    )

    if session is not None:
        try:
            diag["spec"] = truncate_text(safe_str(session.describe_agent()), 400)
        except (AttributeError, TypeError, ValueError):
            pass
    diag["agent_spec"] = diag["spec"]

    message = safe_str(error).strip()
    if not message or message in {"(empty)", "None"}:
        message = diag["stderr_snippet"] or diag["stdout_snippet"]
    cause = getattr(error, "__cause__", None) or getattr(error, "__context__", None)
    if len(message) < 8 and cause is not None and cause is not error:
        cause_text = safe_str(cause).strip()
        if cause_text:
            message = f"{message}\n" if message else ""
            message += f"cause={type(cause).__name__}: {cause_text}"
    if not message:
        message = f"<{type(error).__name__}> (empty output)"
    hints = [f"phase={phase}"]
    if diag["rc"] is not None:
        hints.insert(0, f"rc={diag['rc']}")
    diag["error_text"] = truncate_text(
        f"{message}\n{' '.join(hints)}", 400
    )
    diag["error"] = diag["error_text"]
    return normalize_startup_diagnostics(diag, get_settings_fn=get_settings)


class AgentSpecResolveError(ACPStartupError):
    """解析 agent spec 失败（统一可诊断异常协议）。

    说明：该错误属于启动前阶段（fail_phase=agent_spec_resolve），用于避免进入 ACP handshake 超时。
    """

    def __init__(
        self,
        message: str,
        *,
        agent_cmd: str = "",
        agent_args: Optional[list[str]] = None,
        returncode: Optional[int] = None,
        stdout_snippet: str = "",
        stderr_snippet: str = "",
    ):
        super().__init__(
            message,
            agent_cmd=str(agent_cmd or ""),
            agent_args=[str(x) for x in (agent_args or [])],
            cwd="",
            returncode=returncode,
            stdout_snippet=str(stdout_snippet or ""),
            stderr_snippet=str(stderr_snippet or ""),
            fail_phase="agent_spec_resolve",
            cause=None,
        )


@lru_cache(maxsize=64)
def _probe_acp_serve_help(command: str) -> tuple[bool, Optional[int], str, str]:
    """探测 `<command> acp serve --help` 是否可用，并返回 (ok, rc, stdout_snip, stderr_snip)。

    - ok=True 仅表示该命令支持 ACP server 启动（可用 `acp serve`）。
    - 该探测避免对不支持 ACP 的工具进入 handshake 超时。
    """
    cmd = (command or "").strip()
    if not cmd:
        return False, None, "", ""
    try:
        # Claude Code 等可能因嵌套会话 guard 拒绝启动；探测时移除该 env，提升稳健性。
        from ..utils.env import build_clean_env
        env = build_clean_env()
        p = subprocess.run(
            [cmd, "acp", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=5,
            env=env,
        )
        out = p.stdout or ""
        err = p.stderr or ""
        blob = (out + "\n" + err).lower()
        ok = bool(
            p.returncode == 0
            and (("acp serve" in blob and "usage" in blob) or ("acp" in blob and "server" in blob))
        )
        # 片段截断，避免日志/异常过大
        return ok, int(p.returncode), (out or "")[-200:], (err or "")[-200:]
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        return False, None, "", (str(e) or type(e).__name__)[:200]


@lru_cache(maxsize=32)
def _supports_acp_serve(command: str) -> bool:
    """Best-effort detection whether a binary supports `acp serve`.

    We avoid hard-failing on environments where the agent CLI differs.

    Note: Results are cached per command name. The cache is cleared after a
    successful auto-update so upgraded binaries are detected without restart.
    """
    try:
        # Some agent CLIs (notably Claude Code) refuse to launch when `CLAUDECODE`
        # is set (nested-session guard). Since this probe is executed inside our
        # service process, explicitly drop it to keep detection robust.
        from ..utils.env import build_clean_env
        env = build_clean_env()
        p = subprocess.run(
            [command, "acp", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=5,
            env=env,
        )
        out = (getattr(p, "stdout", "") or "") + "\n" + (getattr(p, "stderr", "") or "")
        out_lower = out.lower()

        # Some tests/fakes don't provide returncode; treat it as success.
        rc = getattr(p, "returncode", 0)
        if rc not in (0, None):
            return False

        # Preferred: explicit subcommand usage for `acp serve`.
        if "acp serve" in out_lower and "usage:" in out_lower:
            return True

        # Backward-compatible heuristic: many CLIs print "Start the ACP server".
        if "acp" in out_lower and "server" in out_lower:
            return True
        return False
    except (OSError, subprocess.SubprocessError, ValueError):
        return False


# Track which agent CLIs have already been auto-updated in this process
# to avoid repeated update attempts.
_update_attempted: set[str] = set()


def _auto_update_agent(command: str) -> bool:
    """Attempt to auto-update an agent CLI binary.

    Runs ``<command> update`` and returns True if the update process exits
    successfully. Each command is only updated once per process lifecycle.
    """
    if command in _update_attempted:
        logger.debug("[ACP] Auto-update already attempted for %s, skipping", command)
        return False
    _update_attempted.add(command)

    settings = get_settings()
    if not settings.acp_auto_update:
        logger.debug("[ACP] Auto-update disabled by config (acp_auto_update=False)")
        return False

    auto_update_timeout = getattr(settings, "acp_auto_update_timeout", 120)

    logger.info("[ACP] %s does not support ACP server mode, attempting auto-update...", command)
    try:
        p = subprocess.run(
            [command, "update"],
            capture_output=True,
            text=True,
            timeout=auto_update_timeout,
        )
        stdout = (p.stdout or "").strip()
        stderr = (p.stderr or "").strip()
        if p.returncode == 0:
            logger.info("[ACP] %s auto-update succeeded. stdout=%s", command, stdout[-200:] if stdout else "(empty)")
            return True
        else:
            logger.warning(
                "[ACP] %s auto-update failed (rc=%d). stderr=%s",
                command,
                p.returncode,
                stderr[-200:] if stderr else "(empty)",
            )
            return False
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("[ACP] %s auto-update error: %s", command, get_error_detail(e), exc_info=True)
        return False


def _resolve_with_auto_update(command: str) -> bool:
    """Check ACP support, auto-update if needed, return final support status."""
    if _supports_acp_serve(command):
        return True
    # Try auto-update then re-probe
    if _auto_update_agent(command):
        _supports_acp_serve.cache_clear()
        try:
            _probe_acp_serve_help.cache_clear()
        except AttributeError:
            logger.debug("_resolve_with_auto_update: cache_clear not available", exc_info=True)
        if _supports_acp_serve(command):
            return True
    return False


def resolve_agent_spec(agent_type: str, model_name: Optional[str] = None) -> tuple[str, list[str]]:
    """Resolve (command, args) for spawning an ACP agent process over stdio."""
    agent_type = (agent_type or "").lower()

    settings = get_settings()
    override_cmd, override_args = settings.get_acp_command(agent_type)
    if override_cmd:
        return override_cmd, override_args

    # Delegate to ToolRegistry for registered tools, or fallback.
    # NOTE: this also triggers a best-effort async preheat so that common
    # tools (coco/aiden) are probed in the background instead of blocking
    # the first real session startup.
    try:
        from .providers import get_providers, tool_registry

        get_providers()

        try:
            # Best-effort: warms availability cache via a daemon thread.
            # Safe to call multiple times and safe to ignore all failures.
            tool_registry.preheat_async()
        except (OSError, RuntimeError):
            logger.debug("resolve_agent_spec: preheat_async failed", exc_info=True)

        # Traex model/profile/effort is one ACP config cascade. Starting the
        # process with a partially normalized CLI model and then applying the
        # full selection again after ``new_session`` creates two authorities
        # that can disagree (for example ``model/standard`` rejects a later
        # reasoning_effort). Start model-neutral; ``_apply_traex_selection`` is
        # the single runtime selection boundary.
        startup_model = None if agent_type == "traex" else model_name
        return tool_registry.get_serve_command(agent_type, startup_model)
    except Exception as e:
        raise RuntimeError(
            f"{agent_type} does not appear to support ACP server mode. Please set *_ACP_CMD/*_ACP_ARGS overrides. Details: {get_error_detail(e)}"
        )


def start_session_with_retry(
    agent_type: str,
    cwd: str,
    startup_timeout: float = 60,
    model_name: Optional[str] = None,
    session_cls: Optional[type["SyncACPSession"]] = None,
    log_failures: bool = True,
    env: Optional[dict[str, str]] = None,
    retries: Optional[int] = None,
    auto_approve: bool | None = None,
    capture_full_tool_content: bool = False,
) -> SyncACPSession:
    """Start an ACP session with retry and progressive timeout.

    Extracts the retry logic from ACPSessionManager so that Deep/Spec engines
    can benefit from the same robustness without per-chat session management.
    """
    settings = get_settings()
    retries = max(
        1,
        int(
            retries
            if retries is not None
            else (getattr(settings, "acp_startup_retries", 2) or 2)
        ),
    )

    last_err: Exception | None = None
    session: SyncACPSession | None = None
    last_diag: dict | None = None

    if session_cls is None:
        session_cls = SyncACPSession

    def construct_session(**kwargs: object) -> SyncACPSession:
        optional_kwargs: dict[str, object] = {
            "auto_approve": auto_approve,
            "capture_full_tool_content": bool(capture_full_tool_content),
        }
        while True:
            try:
                return session_cls(**kwargs, **optional_kwargs)
            except TypeError as exc:
                unsupported = next(
                    (
                        name
                        for name in optional_kwargs
                        if name in str(exc)
                    ),
                    None,
                )
                if unsupported is None:
                    raise
                optional_kwargs.pop(unsupported)
                logger.debug(
                    "session_cls does not accept %s, using legacy signature",
                    unsupported,
                    exc_info=True,
                )

    for attempt in range(1, retries + 1):
        try:
            if env is not None:
                session = construct_session(
                    agent_type=agent_type,
                    cwd=cwd,
                    model_name=model_name,
                    env=dict(env),
                )
                try:
                    setattr(session, "_log_failures", bool(log_failures))
                except (AttributeError, TypeError):
                    logger.debug(
                        "session does not expose startup log policy",
                        exc_info=True,
                    )
                effective_timeout = float(startup_timeout) * (
                    1.0 + 0.5 * (attempt - 1)
                )
                session.start(startup_timeout=effective_timeout)
                logger.info(
                    "[ACP:%s] Engine session started (attempt=%d/%d)",
                    agent_type.upper(),
                    attempt,
                    retries,
                )
                return session
            # Backward-compatible construction: allow fakes/older signatures without model_name kw.
            if model_name:
                try:
                    session = construct_session(
                        agent_type=agent_type,
                        cwd=cwd,
                        model_name=model_name,
                    )
                except TypeError:
                    logger.debug("session_cls does not accept model_name, using minimal signature", exc_info=True)
                    session = construct_session(agent_type=agent_type, cwd=cwd)
            else:
                session = construct_session(agent_type=agent_type, cwd=cwd)
            try:
                setattr(session, "_log_failures", bool(log_failures))
            except (AttributeError, TypeError):
                logger.debug(
                    "session does not expose startup log policy",
                    exc_info=True,
                )
            effective_timeout = float(startup_timeout) * (1.0 + 0.5 * (attempt - 1))
            session.start(startup_timeout=effective_timeout)
            logger.info("[ACP:%s] Engine session started (attempt=%d/%d)", agent_type.upper(), attempt, retries)
            return session
        except AgentSpecResolveError:
            # Agent spec resolution failures cannot benefit from retries.
            raise
        except Exception as e:
            last_err = e
            spec = ""
            try:
                spec = session.describe_agent() if session else ""
            except (AttributeError, TypeError):
                spec = ""

            # Best-effort structured diagnostics for startup failures.
            diag = build_startup_diagnostics(
                agent_type=agent_type,
                cwd=cwd,
                model_name=model_name,
                session=session,
                error=e,
                attempt=int(attempt),
                retries=int(retries),
            )
            last_diag = dict(diag or {})
            # 补充：保留可读 spec 以便快速复现
            if spec and not diag.get("spec"):
                try:
                    diag["spec"] = truncate_text(spec, 400)
                except (TypeError, ValueError):
                    logger.debug("start_session_with_retry: spec truncation failed", exc_info=True)

            if bool(log_failures):
                try:
                    from .diagnostics import format_startup_failure_log_line

                    logger.warning(
                        format_startup_failure_log_line(
                            agent_type=agent_type,
                            event="Engine session start failed",
                            attempt=int(attempt),
                            retries=int(retries),
                            error=e,
                            diag=diag if isinstance(diag, dict) else None,
                            attempts=(diag.get("attempts") if isinstance(diag, dict) else None),
                            get_settings_fn=get_settings,
                        )
                    )
                except (ImportError, TypeError, ValueError):
                    # fallback to legacy message
                    logger.warning(
                        "[ACP:%s] Engine session start failed: %s",
                        agent_type.upper(),
                        format_startup_diagnostics(diag),
                    )
            try:
                if session:
                    session.close()
            except (OSError, RuntimeError):
                logger.debug("start_session_with_retry: session close failed", exc_info=True)
            session = None
            if attempt < retries:
                time.sleep(min(2.0, 0.3 * attempt))

    spec = ""
    try:
        spec = f" ({resolve_agent_spec(agent_type)})"
    except Exception:
        logger.debug("start_session_with_retry: resolve_agent_spec for error message failed", exc_info=True)

    # 诊断载体契约（SSOT=build_startup_diagnostics）：
    # - 上层（ACPSessionManager / engines）需要稳定读取 cmd/args/rc/stdout_snippet/stderr_snippet
    # - 这里用 ACPStartupError 作为“可诊断异常”，避免仅抛 RuntimeError 导致信息丢失/日志为空
    agent_cmd = ""
    agent_args: list[str] = []
    stdout_snip = ""
    stderr_snip = ""
    rc: Optional[int] = None
    try:
        if isinstance(last_diag, dict):
            agent_cmd = safe_str(last_diag.get("cmd") or "")
            agent_args = [str(x) for x in (last_diag.get("args") or [])]
            stdout_snip = safe_str(last_diag.get("stdout_snippet") or "")
            stderr_snip = safe_str(last_diag.get("stderr_snippet") or "")
            _rc = last_diag.get("rc")
            if _rc is not None:
                rc = int(_rc)
    except Exception:
        logger.debug("start_session_with_retry: diagnostics extraction failed", exc_info=True)

    if not agent_cmd:
        try:
            # 注意：resolve_agent_spec 可能失败（例如 agent 不存在），因此 best-effort。
            cmd, args = resolve_agent_spec(agent_type, model_name=model_name)
            agent_cmd = safe_str(cmd or "")
            agent_args = [str(x) for x in (args or [])]
        except Exception:
            logger.debug("start_session_with_retry: fallback resolve_agent_spec failed", exc_info=True)
            agent_cmd = safe_str(agent_type or "")

    if rc is None:
        try:
            _rc = getattr(last_err, "returncode", None)
            if _rc is not None:
                rc = int(_rc)
        except Exception:
            logger.debug("start_session_with_retry: returncode extraction failed", exc_info=True)
            rc = None

    # 最后兜底：若 snippet 为空，尽量从异常上提取一点点（不做全量输出）
    if not stdout_snip:
        try:
            stdout_snip = truncate_text(
                safe_str(getattr(last_err, "stdout_snippet", "") or getattr(last_err, "stdout", "") or ""), DEFAULT_DIAGNOSTICS_SNIPPET_LIMIT
            )
        except Exception:
            logger.debug("start_session_with_retry: stdout snippet fallback failed", exc_info=True)
            stdout_snip = ""
    if not stderr_snip:
        try:
            stderr_snip = truncate_text(
                safe_str(getattr(last_err, "stderr_snippet", "") or getattr(last_err, "stderr", "") or ""), DEFAULT_DIAGNOSTICS_SNIPPET_LIMIT
            )
        except Exception:
            logger.debug("start_session_with_retry: stderr snippet fallback failed", exc_info=True)
            stderr_snip = ""

    raise ACPStartupError(
        f"启动 {agent_type} ACP Server 失败{spec}（已重试 {retries} 次）",
        agent_cmd=agent_cmd or safe_str(agent_type or ""),
        agent_args=list(agent_args or []),
        cwd=cwd,
        returncode=rc,
        stdout_snippet=stdout_snip,
        stderr_snippet=stderr_snip,
        fail_phase="retry_exhausted",
        cause=last_err,
    ) from last_err


class SyncACPSession(PromptGenerationTracker):
    """Synchronous wrapper for ACPSession.

    Runs an asyncio event loop in a background thread and provides blocking
    methods for the synchronous codebase.
    """

    def __init__(
        self,
        agent_type: str,
        cwd: str,
        agent_args: Optional[list[str]] = None,
        agent_cmd: Optional[str] = None,
        model_name: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        auto_approve: bool | None = None,
        capture_full_tool_content: bool = False,
    ):
        self._agent_type = agent_type
        self._cwd = cwd
        if agent_cmd is not None:
            self._agent_cmd = agent_cmd
            self._agent_args = agent_args or []
        else:
            cmd, args = resolve_agent_spec(agent_type, model_name=model_name)
            self._agent_cmd = cmd
            self._agent_args = agent_args or args
        self._model_name = (model_name or "").strip() or None
        self._explicit_env = dict(env) if env is not None else None
        self._auto_approve = auto_approve
        self._capture_full_tool_content = bool(capture_full_tool_content)
        self._log_failures = True
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._acp_session: Optional[ACPSession] = None
        self._started = threading.Event()
        self._prompt_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._prompt_generation_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._prompt_generation = 0
        self._active_prompt_generation: int | None = None
        self._user_cancel_generation: int | None = None

        # Persistent watchdog: monitors active prompt future for process death
        self._active_future: Optional[asyncio.Future] = None
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: Optional[threading.Thread] = None
        # Terminal-state marker: set True when a prompt detects the session is
        # irrecoverably dead so that is_server_running() immediately returns False
        # without relying on asyncio process reaping.
        self._force_dead: bool = False

        # Public state (compatible with old BaseSession interface)
        self.session_id: str = ""
        self.created_at: float = time.time()
        self.last_active: float = time.time()
        self.message_count: int = 0
        self.last_query: str = ""
        self.is_resumed: bool = False
        # Local history loaded from ~/.ghostap/acp_history/<session_id>.jsonl
        self.history: list[dict] = []

    def describe_agent(self) -> str:
        """Human-readable agent command spec for debugging."""
        try:
            args = " ".join(str(x) for x in (self._agent_args or []))
            return f"cmd={self._agent_cmd} args={args} cwd={self._cwd}"
        except (AttributeError, TypeError):
            return f"agent={self._agent_type}"

    def start(self, startup_timeout: float = 60) -> str:
        """Start event loop thread + ACP session. Returns session_id.

        Args:
            startup_timeout: Seconds to wait for ACP server process + handshake.
        """
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name=f"acp-{self._agent_type}",
        )
        self._loop_thread.start()
        if not self._started.wait(timeout=min(5.0, float(startup_timeout or 60))):
            # Fail fast: event loop thread did not start.
            self.close()
            logger.error("[ACP:%s] 事件循环启动超时 (timeout=%ss)", self._agent_type, min(5.0, float(startup_timeout or 60)))
            raise TimeoutError(f"ACP 事件循环启动超时: agent={self._agent_type}")

        # Start ACP session (spawns agent `acp serve` process and initializes protocol)
        try:
            session_id = self._run_async(self._start_session(), timeout=startup_timeout)
            self.session_id = session_id
            return session_id
        except Exception:
            # Best-effort cleanup on startup failure.
            try:
                self.close()
            except (OSError, RuntimeError):
                pass
            raise

    def is_server_running(self) -> bool:
        """Best-effort check whether the ACP agent process is still alive."""
        # Fast path: if a previous prompt detected terminal-state death, skip
        # expensive process introspection.
        if getattr(self, "_force_dead", False):
            return False
        try:
            if not self._acp_session:
                return False
            proc = getattr(self._acp_session, "_proc", None)
            if proc is None:
                return False
            # asyncio.subprocess.Process has `returncode`, while subprocess.Popen has `poll()`.
            rc = getattr(proc, "returncode", None)
            if rc is not None:
                return False
            poll = getattr(proc, "poll", None)
            if callable(poll):
                return poll() is None
            return True
        except (OSError, AttributeError):
            return False

    def is_server_healthy(self, healthcheck_timeout: float = 2.0) -> bool:
        """More accurate ACP server health check.

        - Ensures process is alive
        - Ensures ACP connection can respond to a lightweight request
        """
        if not self.is_server_running():
            return False
        if not self._acp_session:
            return False
        try:
            # Run a lightweight RPC (list_sessions) with a short timeout.
            return bool(
                self._run_async(
                    self._acp_session.health_check(timeout=healthcheck_timeout), timeout=healthcheck_timeout + 1.0
                )
            )
        except (TimeoutError, OSError, RuntimeError):
            return False

    def _uses_official_codex_acp(self) -> bool:
        """Whether this session was started through the official Codex ACP fallback."""
        if (self._agent_type or "").strip().lower() != "codex":
            return False
        return any(
            "@agentclientprotocol/codex-acp" in str(arg)
            for arg in (self._agent_args or [])
        )

    def _uses_traex_acp(self) -> bool:
        return (self._agent_type or "").strip().lower() == "traex"

    def _uses_dsh_acp(self) -> bool:
        return (self._agent_type or "").strip().lower() == "dsh"

    async def _apply_official_codex_selection(self, selection: str) -> bool:
        """Apply a persisted Codex model/Effort selection over ACP config options."""
        if not self._acp_session:
            return False
        from .model_selection import split_codex_model_selection

        model_id, reasoning_effort = split_codex_model_selection(selection)
        if not model_id:
            return False
        if reasoning_effort is None:
            return bool(await self._acp_session.set_model(model_id))
        if not await self._acp_session.set_config_option("model", model_id):
            return False
        return bool(
            await self._acp_session.set_config_option(
                "reasoning_effort",
                reasoning_effort,
            )
        )

    async def _apply_traex_selection(self, selection: str) -> bool:
        if not self._acp_session:
            return False
        from .traex_selection import resolve_traex_runtime_selection

        try:
            resolved = resolve_traex_runtime_selection(selection)
        except ValueError as exc:
            logger.warning(
                "[ACP:TRAEX] invalid model selection: %s",
                get_error_detail(exc),
            )
            return False
        if not await self._acp_session.set_config_option(
            "model",
            resolved.backend_model_value,
        ):
            return False
        if resolved.effort is None:
            return True
        return bool(
            await self._acp_session.set_config_option(
                "reasoning_effort",
                resolved.effort,
            )
        )

    async def _apply_dsh_selection(self, selection: str) -> bool:
        if not self._acp_session:
            return False
        from .dsh_selection import (
            DSH_MODEL_CONFIG_ID,
            DSH_REASONING_CONFIG_ID,
            split_dsh_model_selection,
        )

        try:
            model_value, reasoning_value = split_dsh_model_selection(selection)
        except ValueError as exc:
            logger.warning(
                "[ACP:DSH] invalid model selection: %s",
                get_error_detail(exc),
            )
            return False
        if not await self._acp_session.set_config_option(
            DSH_MODEL_CONFIG_ID,
            model_value,
        ):
            return False
        if reasoning_value is None:
            return True
        return bool(
            await self._acp_session.set_config_option(
                DSH_REASONING_CONFIG_ID,
                reasoning_value,
            )
        )

    async def _start_session(self) -> str:
        env_override = (
            dict(self._explicit_env)
            if getattr(self, "_explicit_env", None) is not None
            else None
        )
        # Anthropic 1M-context beta: when the user picked a `[1m]`-suffixed
        # claude model, set ANTHROPIC_BETAS as a defensive fallback in case
        # the wrapper drops the `--model <id>[1m]` suffix.  Safe no-op for
        # all other tools / models.
        #
        # IMPORTANT: ACPSession.start() calls build_clean_env(base=env_override)
        # which REPLACES os.environ entirely when base is provided (it does not
        # merge). So if we hand it a single-key {'ANTHROPIC_BETAS': ...} dict,
        # the spawned Claude subprocess loses HOME, USER, ANTHROPIC_API_KEY,
        # OAuth tokens cached under ~/.claude, locale, proxies, etc — which can
        # silently flip the user onto a different identity / billing context.
        # Therefore, when env_override is None and we still need to inject
        # betas, seed it with a real os.environ copy (via build_clean_env).
        from ..utils.env import apply_anthropic_betas, build_clean_env
        from .claude_capabilities import is_1m_variant
        if env_override is None and is_1m_variant((self._model_name or "").strip()):
            env_override = build_clean_env()
        env_override = apply_anthropic_betas(dict(env_override or {}), self._model_name)
        if not env_override:
            env_override = None

        self._acp_session = ACPSession(
            agent_cmd=self._agent_cmd,
            agent_args=self._agent_args,
            cwd=self._cwd,
            env=env_override,
            auto_approve=self._auto_approve,
            capture_full_tool_content=self._capture_full_tool_content,
        )
        session_id = await self._acp_session.start()

        # The official adapter intentionally does not parse Codex CLI ``-c``
        # arguments. Apply an explicit selection over ACP before exposing this
        # session as ready; otherwise a model card can claim one model while the
        # first prompt silently uses the user's default model.
        if self._uses_official_codex_acp() and self._model_name:
            applied = await self._apply_official_codex_selection(self._model_name)
            if not applied:
                with contextlib.suppress(Exception):
                    await self._acp_session.close()
                raise RuntimeError(
                    f"Codex ACP rejected selected model: {self._model_name}"
                )
        if self._uses_traex_acp() and self._model_name:
            applied = await self._apply_traex_selection(self._model_name)
            if not applied:
                with contextlib.suppress(Exception):
                    await self._acp_session.close()
                raise RuntimeError(
                    f"Traex ACP rejected selected model: {self._model_name}"
                )
        if self._uses_dsh_acp() and self._model_name:
            applied = await self._apply_dsh_selection(self._model_name)
            if not applied:
                with contextlib.suppress(Exception):
                    await self._acp_session.close()
                raise RuntimeError(
                    f"DSH ACP rejected selected model: {self._model_name}"
                )
        return session_id

    def load_session(self, session_id: str, timeout: float) -> None:
        """Load an existing session (for resume)."""
        if not self._acp_session:
            raise RuntimeError("Session not started")
        bounded_timeout = float(timeout)
        if bounded_timeout <= 0:
            raise TimeoutError("ACP resume load budget exhausted")
        try:
            self._run_async(
                self._acp_session.load_session(session_id),
                timeout=bounded_timeout,
            )
        except BaseException as exc:
            if not isinstance(exc, ACPResumeRejected):
                self._force_dead = True
            raise
        self.session_id = session_id
        self.is_resumed = True
        self.load_local_history(session_id)

    def load_local_history(self, session_id: Optional[str] = None, limit: int = 200) -> list[dict]:
        """Load persisted local history for a given ACP session id.

        Handles missing/corrupt history files by returning an empty list.
        """
        sid = session_id or self.session_id
        try:
            store = ACPHistoryStore()
            self.history = store.load(sid, limit=limit)
        except (OSError, json.JSONDecodeError, ValueError) as e:
            logger.debug("[ACP] load_local_history failed for %s: %s", sid, str(e))
            self.history = []
        return list(self.history)

    def _start_watchdog(self) -> None:
        """Start a persistent watchdog thread that monitors active prompt futures."""
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            return
        self._watchdog_stop.clear()

        def _watchdog_loop():
            while not self._watchdog_stop.wait(timeout=5.0):
                fut = self._active_future
                if fut is None or fut.done():
                    continue
                if not self.is_server_running():
                    logger.warning("[ACP:%s] Agent process died mid-prompt, cancelling", self._agent_type)
                    fut.cancel()

        self._watchdog_thread = threading.Thread(
            target=_watchdog_loop,
            daemon=True,
            name=f"acp-watchdog-{self._agent_type}",
        )
        self._watchdog_thread.start()

    def _stop_watchdog(self) -> None:
        """Stop the persistent watchdog thread."""
        self._watchdog_stop.set()
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=2)
        self._watchdog_thread = None

    def send_prompt(
        self,
        text: str,
        on_event: Optional[Callable[[ACPEvent], None]] = None,
        timeout: Optional[int] = None,
        idle_timeout: Optional[float] = None,
        activity_predicate: Optional[Callable[[ACPEvent], bool]] = None,
        await_goal_quiescence: bool = True,
        await_child_quiescence: bool = False,
        replay_deferred_child_events: bool = False,
    ) -> PromptResult:
        """Send one prompt, rejecting concurrent callers on this wrapper."""
        prompt_lock = getattr(self, "_prompt_lock", None)
        if prompt_lock is None:
            prompt_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
            self._prompt_lock = prompt_lock
        if not prompt_lock.acquire(blocking=False):
            raise RuntimeError("ACP prompt is already running for this session")
        prompt_generation = self._begin_prompt_generation()
        generation_consumed = False
        try:
            result = self._send_prompt_once(
                text,
                on_event=on_event,
                timeout=timeout,
                idle_timeout=idle_timeout,
                activity_predicate=activity_predicate,
                await_goal_quiescence=await_goal_quiescence,
                await_child_quiescence=await_child_quiescence,
                replay_deferred_child_events=replay_deferred_child_events,
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
            prompt_lock.release()

    def send_finalization_prompt(
        self,
        text: str,
        on_event: Optional[Callable[[ACPEvent], None]] = None,
        timeout: Optional[int] = None,
    ) -> PromptResult:
        """Run the terminal cleanup turn without following new Goal turns."""
        return self.send_prompt(
            text,
            on_event=on_event,
            timeout=timeout,
            await_goal_quiescence=False,
            replay_deferred_child_events=True,
        )

    def send_continuation_prompt(
        self,
        text: str,
        on_event: Optional[Callable[[ACPEvent], None]] = None,
        timeout: Optional[int] = None,
    ) -> PromptResult:
        """Continue the same logical task with deferred child evidence."""
        return self.send_prompt(
            text,
            on_event=on_event,
            timeout=timeout,
            replay_deferred_child_events=True,
        )

    def send_reconciliation_prompt(
        self,
        text: str,
        on_event: Optional[Callable[[ACPEvent], None]] = None,
        timeout: Optional[int] = None,
    ) -> PromptResult:
        """Keep collecting until every observed transient child is terminal."""
        return self.send_prompt(
            text,
            on_event=on_event,
            timeout=timeout,
            await_child_quiescence=not self._uses_official_codex_acp(),
            replay_deferred_child_events=True,
        )

    def enrich_child_reconciliation_result(
        self,
        result: PromptResult,
        *,
        started_at: float,
        ended_at: float,
        logical_task_started_at: float | None = None,
        on_event: Optional[Callable[[ACPEvent], None]] = None,
    ) -> PromptResult:
        """Recover strict list_agents evidence omitted by official Codex ACP."""
        if not self._uses_official_codex_acp():
            return result
        from .codex_rollout_reconciliation import (
            enrich_codex_reconciliation_result,
        )
        from .models import ACPEventType

        acp_session = getattr(self, "_acp_session", None)
        session_id = str(
            getattr(self, "session_id", None)
            or getattr(acp_session, "_session_id", None)
            or ""
        )
        env_override = getattr(acp_session, "_env_override", None)
        codex_home: str | None = None
        if isinstance(env_override, dict):
            raw_codex_home = str(
                env_override.get("CODEX_HOME") or ""
            ).strip()
            if raw_codex_home:
                codex_home = raw_codex_home
            else:
                raw_home = str(env_override.get("HOME") or "").strip()
                if raw_home:
                    codex_home = str(Path(raw_home).expanduser() / ".codex")
        try:
            enriched, evidence = enrich_codex_reconciliation_result(
                result,
                session_id=session_id,
                cwd=self._cwd,
                logical_task_started_at=logical_task_started_at,
                started_at=started_at,
                ended_at=ended_at,
                codex_home=codex_home,
            )
        except Exception:
            logger.warning(
                "[ACP:CODEX] rollout child reconciliation failed closed",
                exc_info=True,
            )
            return result
        if evidence is not None and on_event is not None:
            try:
                on_event(
                    ACPEvent(
                        event_type=ACPEventType.TOOL_CALL_DONE,
                        tool_call=evidence,
                    )
                )
            except Exception:
                logger.warning(
                    "[ACP:CODEX] reconciled child event callback failed",
                    exc_info=True,
                )
        return enriched

    def _send_prompt_once(
        self,
        text: str,
        on_event: Optional[Callable[[ACPEvent], None]] = None,
        timeout: Optional[int] = None,
        idle_timeout: Optional[float] = None,
        activity_predicate: Optional[Callable[[ACPEvent], bool]] = None,
        await_goal_quiescence: bool = True,
        await_child_quiescence: bool = False,
        replay_deferred_child_events: bool = False,
    ) -> PromptResult:
        """Send prompt synchronously, blocking until completion.

        A persistent watchdog thread monitors for agent process death and
        cancels the future early instead of waiting for the full timeout.

        When *idle_timeout* is set, the timeout becomes activity-based: as long
        as ACP events keep arriving (indicating the agent is working), the
        deadline is extended.  The agent is only timed out after *idle_timeout*
        seconds of silence.  *timeout* then acts as a hard cap to prevent
        infinite runs.
        """
        if not self._acp_session:
            raise RuntimeError("Session not started")

        effective_timeout = timeout if timeout is not None else 600.0
        effective_idle_timeout = idle_timeout if idle_timeout is not None else 0.0

        self.last_active = time.time()
        self.message_count += 1
        self.last_query = text

        # Activity tracker — updated by _activity_on_event wrapper
        last_activity_ts = [time.time()]

        def _activity_on_event(ev: ACPEvent) -> None:
            counts_as_activity = True
            if activity_predicate is not None:
                try:
                    counts_as_activity = bool(activity_predicate(ev))
                except Exception:
                    counts_as_activity = False
                    logger.warning("ACP activity predicate failed closed", exc_info=True)
            if counts_as_activity:
                last_activity_ts[0] = time.time()
            if on_event:
                on_event(ev)

        use_adaptive = effective_idle_timeout > 0
        event_cb = _activity_on_event if use_adaptive else on_event

        future = asyncio.run_coroutine_threadsafe(
            self._acp_session.prompt(
                text,
                on_event=event_cb,
                await_goal_quiescence=await_goal_quiescence,
                await_child_quiescence=await_child_quiescence,
                replay_deferred_child_events=replay_deferred_child_events,
            ),
            self._loop,
        )
        self._active_future = future
        self._start_watchdog()

        try:
            if not use_adaptive:
                return future.result(timeout=effective_timeout)

            # Adaptive polling: check idle vs hard cap
            poll_interval = 5.0
            hard_deadline = time.time() + effective_timeout
            while True:
                remaining = hard_deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError(
                        f"ACP prompt 执行超时 (hard cap {effective_timeout}s)"
                    )
                wait_time = min(poll_interval, remaining)
                try:
                    return future.result(timeout=wait_time)
                except TimeoutError:
                    if future.done():
                        return future.result(timeout=0)
                    idle_elapsed = time.time() - last_activity_ts[0]
                    if idle_elapsed >= effective_idle_timeout:
                        raise TimeoutError(
                            f"ACP prompt 空闲超时 ({idle_elapsed:.0f}s 无活动, "
                            f"idle_timeout={effective_idle_timeout}s)"
                        )
                    # Agent still active — continue waiting
        except (asyncio.CancelledError, concurrent.futures.CancelledError):
            record_employee_session_outcome("canceled")
            self._force_dead = True
            raise RuntimeError("ACP agent 进程在执行过程中意外终止")
        except TimeoutError as e:
            record_employee_session_outcome("timeout")
            agent_type_str = getattr(self, "_agent_type", "unknown")
            if getattr(self, "_log_failures", True):
                logger.error("[ACP:%s] prompt 执行超时 (timeout=%ss): %s", agent_type_str, effective_timeout, get_error_detail(e), exc_info=True)
            else:
                logger.debug(
                    "[ACP:%s] expected prompt timeout (timeout=%ss)",
                    agent_type_str,
                    effective_timeout,
                )
            try:
                self.cancel(wait=True, timeout=2.0)
            except Exception as cancel_error:
                future.cancel()
                self._force_dead = True
                logger.warning(
                    "[ACP:%s] prompt timeout cleanup failed; session marked dead: %s",
                    agent_type_str,
                    get_error_detail(cancel_error),
                    exc_info=True,
                )
            try:
                future.result(timeout=_PROMPT_CANCEL_DRAIN_TIMEOUT_S)
            except (
                asyncio.CancelledError,
                concurrent.futures.CancelledError,
            ):
                pass
            except TimeoutError:
                if not future.done():
                    future.cancel()
                    self._force_dead = True
                    logger.warning(
                        "[ACP:%s] prompt cancel did not drain within %.1fs; "
                        "session marked dead",
                        agent_type_str,
                        _PROMPT_CANCEL_DRAIN_TIMEOUT_S,
                    )
                # A completed future may itself carry TimeoutError. In that
                # case prompt ownership has already drained successfully.
            except Exception as drain_error:
                # A prompt-side exception after cancel is still a terminal future:
                # its finally block has released prompt ownership and event routing.
                # Transport-terminal errors additionally make the session unsafe
                # to reuse even though the future itself is complete.
                drain_detail = str(drain_error).casefold()
                if (
                    isinstance(drain_error, (ConnectionError, BrokenPipeError))
                    or "terminal state" in drain_detail
                    or "broken pipe" in drain_detail
                    or ("connection" in drain_detail and "closed" in drain_detail)
                ):
                    self._force_dead = True
            raise TimeoutError(f"ACP prompt 执行超时 ({effective_timeout}s)") from e
        except Exception as e:
            err_detail = str(e).lower()
            if "terminal state" in err_detail or "broken pipe" in err_detail or "connection" in err_detail and "closed" in err_detail:
                self._force_dead = True
                if getattr(self, "_log_failures", True):
                    logger.warning(
                        "[ACP:%s] Session marked dead after prompt error: %s",
                        getattr(self, "_agent_type", "unknown"),
                        str(e)[:120],
                    )
                else:
                    logger.debug(
                        "[ACP:%s] expected prompt session closure",
                        getattr(self, "_agent_type", "unknown"),
                    )
            raise
        finally:
            # Retain an undrained future so close() can make one more cancellation
            # attempt while evicting the force-dead session.
            if future.done() or not getattr(self, "_force_dead", False):
                self._active_future = None

    def set_model(self, model_id: str, timeout: float = 10.0) -> bool:
        """Switch model on the running ACP session via session/setModel.

        Returns True if the agent accepted the model switch, False otherwise.
        Falls back gracefully for agents that don't support the method.
        """
        if not self._acp_session or not self._loop:
            return False
        try:
            if self._uses_traex_acp():
                operation = self._apply_traex_selection(model_id)
            elif self._uses_dsh_acp():
                operation = self._apply_dsh_selection(model_id)
            elif self._uses_official_codex_acp():
                operation = self._apply_official_codex_selection(model_id)
            else:
                operation = self._acp_session.set_model(model_id)
            future = asyncio.run_coroutine_threadsafe(
                operation,
                self._loop,
            )
            applied = bool(future.result(timeout=float(timeout or 10.0)))
            if applied:
                self._model_name = str(model_id or "").strip() or None
            return applied
        except (TimeoutError, OSError, RuntimeError) as e:
            logger.warning("[ACP] set_model failed: %s", get_error_detail(e), exc_info=True)
            return False

    def set_tool_filter(self, filter_fn: "Callable[[str, dict | None], bool]") -> None:
        """Install a per-session tool filter for least-privilege execution.

        The filter_fn receives (tool_name, args) and returns True to allow.
        This is stored locally and checked by the engine before tool execution.
        """
        self._tool_filter = filter_fn
        if self._acp_session is not None:
            self._acp_session.set_tool_filter(filter_fn)

    def get_tool_filter(self) -> "Optional[Callable[[str, dict | None], bool]]":
        """Return the currently installed tool filter, or None."""
        return getattr(self, "_tool_filter", None)

    def has_active_goal(self, timeout: float = 1.0) -> bool:
        """Inspect provider-owned goal state on the session event loop."""
        if not self._acp_session:
            return False
        bounded_timeout = float(timeout)
        if bounded_timeout <= 0:
            raise TimeoutError("ACP goal inspection budget exhausted")
        return bool(
            self._run_async(
                self._acp_session.has_active_goal(),
                timeout=bounded_timeout,
            )
        )

    def pause_active_goal(self, timeout: float) -> bool:
        """Pause a provider-owned goal within the caller's absolute budget."""
        if not self._acp_session:
            return False
        bounded_timeout = float(timeout)
        if bounded_timeout <= 0:
            raise TimeoutError("ACP goal pause budget exhausted")
        return bool(
            self._run_async(
                self._acp_session.pause_active_goal(),
                timeout=bounded_timeout,
            )
        )

    def cancel(self, wait: bool = False, timeout: float = 2.0) -> bool | None:
        """Cancel current prompt.

        When wait=True, block (up to `timeout` s) until the agent has acknowledged
        the cancel. This prevents the race where a follow-up `send_prompt` lands
        at the agent before cancel is processed, causing a `-32602 Invalid params`
        rejection because the session is still mid-cancel.
        """
        if not (self._acp_session and self._loop):
            return False
        fut = asyncio.run_coroutine_threadsafe(
            self._acp_session.cancel(timeout=timeout),
            self._loop,
        )
        if not wait:
            return None
        try:
            fut.result(timeout=timeout)
            return True
        except TimeoutError as e:
            fut.cancel()
            self._force_dead = True
            logger.debug("[ACP] cancel wait timed out: %s", get_error_detail(e))
            return False
        except (OSError, RuntimeError) as e:
            logger.debug("[ACP] cancel wait skipped: %s", get_error_detail(e))
            return False

    def close(self) -> None:
        """Close session and stop event loop."""
        active_future = getattr(self, "_active_future", None)
        close_future = None
        try:
            self._stop_watchdog()
            if active_future is not None and not active_future.done():
                active_future.cancel()
            if self._acp_session is not None and self._loop is None:
                raise RuntimeError(
                    "ACP transport cannot close without its event loop"
                )
            if self._acp_session and self._loop:
                future = asyncio.run_coroutine_threadsafe(
                    self._acp_session.close(),
                    self._loop,
                )
                close_future = future
                future.result(timeout=10)

            if self._loop:
                self._drain_loop_before_close()
                self._loop.call_soon_threadsafe(self._loop.stop)
                if self._loop_thread and self._loop_thread.is_alive():
                    self._loop_thread.join(timeout=5)
                    if self._loop_thread.is_alive():
                        raise RuntimeError(
                            "ACP event loop thread did not stop"
                        )
                self._loop.close()
        except BaseException:
            self._force_dead = True
            if close_future is not None and not close_future.done():
                close_future.cancel()
            logger.warning(
                "ACP session close did not confirm transport termination",
                exc_info=True,
            )
            raise
        else:
            self._active_future = None
            self._loop = None
            self._loop_thread = None
            self._acp_session = None

    def _drain_loop_before_close(self) -> None:
        """Run pending subprocess pipe callbacks before closing the loop."""
        loop = self._loop
        if not loop or not loop.is_running():
            return
        if self._loop_thread and threading.current_thread() is self._loop_thread:
            return

        async def _drain() -> None:
            for _ in range(3):
                await asyncio.sleep(0)
            with contextlib.suppress(Exception):
                await loop.shutdown_asyncgens()

        try:
            future = asyncio.run_coroutine_threadsafe(_drain(), loop)
            future.result(timeout=2.0)
        except (TimeoutError, OSError, RuntimeError) as e:
            raise RuntimeError(
                f"ACP event loop drain failed: {get_error_detail(e)}"
            ) from e

    def to_snapshot(self) -> dict:
        """Return session snapshot for persistence."""
        return {
            "session_id": self.session_id,
            "agent_type": self._agent_type,
            "cwd": self._cwd,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "message_count": self.message_count,
            "last_query": self.last_query,
            "is_resumed": self.is_resumed,
        }

    def get_session_info(self) -> str:
        """Return human-readable session info."""
        duration = int(time.time() - self.created_at)
        minutes, seconds = divmod(duration, 60)
        agent_name = self._agent_type.capitalize()
        resumed_info = " (已恢复)" if self.is_resumed else ""
        return (
            f"📊 {agent_name} 会话信息{resumed_info}:\n"
            f"- 会话ID: {self.session_id}\n"
            f"- 消息数: {self.message_count}\n"
            f"- 持续时间: {minutes}分{seconds}秒"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        """Run asyncio event loop in background thread."""
        asyncio.set_event_loop(self._loop)
        self._started.set()
        self._loop.run_forever()

    def _run_async(self, coro, timeout: float = 60) -> Any:
        """Run async coroutine in background loop, blocking until done."""
        if not self._loop:
            raise RuntimeError("Event loop not started")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except TimeoutError as e:
            future.cancel()
            msg = sanitize_futures_msg(str(e))
            if not msg or msg == "操作超时，请稍后重试":
                msg = f"ACP 异步操作超时 ({timeout}s): agent={self._agent_type}"
            if getattr(self, "_log_failures", True):
                logger.error(
                    "[ACP:%s] _run_async 超时 (timeout=%ss): %s",
                    self._agent_type,
                    timeout,
                    get_error_detail(e),
                    exc_info=True,
                )
            else:
                logger.debug(
                    "[ACP:%s] expected _run_async timeout (timeout=%ss)",
                    self._agent_type,
                    timeout,
                )
            raise TimeoutError(msg) from e
