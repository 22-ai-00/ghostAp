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
from typing import Any, Callable, Optional

from ..config import get_settings
from ..employee_session_scope import record_employee_session_outcome
from ..ttadk.env_sandbox import build_ttadk_subprocess_env
from ..utils.errors import get_error_detail, sanitize_futures_msg
from .client import ACPHistoryStore
from .diagnostics import (
    DEFAULT_DIAGNOSTICS_SNIPPET_LIMIT,
    DEFAULT_DIAGNOSTICS_TOTAL_LIMIT,
    get_diagnostics_config,
    normalize_startup_diagnostics,
    redact_text,
    safe_extract,
    safe_str,
    truncate_text,
)
from .models import ACPEvent, PromptResult
from .session import ACPSession, ACPStartupError
from .startup_utils import initial_startup_diagnostics, safe_float_or_none

logger = logging.getLogger(__name__)

_PROMPT_CANCEL_DRAIN_TIMEOUT_S = 5.0

# 供 resolve_agent_spec 内部 best-effort 读取 manager 缓存时使用，避免引入额外锁实现。
_NULL_LOCK = contextlib.nullcontext()


def _resolve_startup_snippet_limit(snippet_limit: int) -> int:
    """Resolve effective startup diagnostics snippet limit from config with compat fallback."""
    cfg = get_diagnostics_config(get_settings_fn=get_settings)
    try:
        snippet_limit_eff = int(cfg.snippet_limit or 0)
    except Exception:
        logger.debug("build_startup_diagnostics: snippet_limit config conversion failed", exc_info=True)
        snippet_limit_eff = 0
    if snippet_limit_eff <= 0:
        try:
            snippet_limit_eff = int(snippet_limit or DEFAULT_DIAGNOSTICS_SNIPPET_LIMIT)
        except Exception:
            logger.debug("build_startup_diagnostics: snippet_limit argument conversion failed", exc_info=True)
            snippet_limit_eff = DEFAULT_DIAGNOSTICS_SNIPPET_LIMIT
    return snippet_limit_eff


def _initial_startup_diagnostics(
    *,
    agent_type: str,
    cwd: str,
    model_name: Optional[str],
    error: Exception,
    attempt: Optional[int],
    retries: Optional[int],
    timeout_s: Optional[float],
) -> dict:
    """Create the stable startup diagnostics container before best-effort enrichment."""
    return initial_startup_diagnostics(
        agent_type=agent_type,
        cwd=cwd,
        model_name=model_name,
        error=error,
        attempt=attempt,
        retries=retries,
        timeout_s=timeout_s,
    )


def classify_startup_fail_phase(*, error: Exception, error_blob: str) -> str:
    """Best-effort classify startup failure phase.

    Contract:
    - Never raises
    - Returns one of: invalid_model | stdin_not_tty | timeout | start_failed

    Notes:
    - Prefer reusing TTADK-side matchers when available.
    - Uses minimal string fallbacks to remain functional when TTADK module is unavailable.
    """
    try:
        # Timeout variants
        try:
            if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)):
                return "timeout"
        except Exception:
            logger.debug("classify_startup_fail_phase: timeout check failed", exc_info=True)

        blob = str(error_blob or "")
        lower = blob.lower()

        # Prefer TTADK matchers (best-effort, avoid hard dependency / circular import by delaying import).
        try:
            import importlib

            m = importlib.import_module("src.ttadk.models")
            is_invalid = getattr(m, "is_invalid_model_error", None)
            is_tty = getattr(m, "is_stdin_not_tty_error", None)
            try:
                if callable(is_tty) and bool(is_tty(blob)):
                    return "stdin_not_tty"
            except Exception:
                logger.debug("classify_startup_fail_phase: is_tty check failed", exc_info=True)
            try:
                if callable(is_invalid) and bool(is_invalid(blob)):
                    return "invalid_model"
            except Exception:
                logger.debug("classify_startup_fail_phase: is_invalid check failed", exc_info=True)
        except Exception:
            logger.debug("classify_startup_fail_phase: TTADK matcher import failed", exc_info=True)

        # Minimal string fallbacks
        if "stdin is not a terminal" in lower or "stdin-not-tty" in lower:
            return "stdin_not_tty"
        if (
            "invalid model" in lower
            or "model must be one of" in lower
            or "unknown model" in lower
            or ("invalid value" in lower and "--model" in lower)
        ):
            return "invalid_model"

        return "start_failed"
    except Exception:
        logger.debug("classify_startup_fail_phase: unexpected error", exc_info=True)
        return "start_failed"


class StartupDiagnosticsBuilder:
    """Builder for stable startup failure diagnostics.

    ``build_startup_diagnostics`` remains the public compatibility function;
    this class owns the construction path so future enrichment can be split
    into focused methods without growing the compatibility wrapper.
    """

    def __init__(
        self,
        *,
        agent_type: str,
        cwd: str,
        model_name: Optional[str],
        error: Exception,
        session: object = None,
        attempt: Optional[int] = None,
        retries: Optional[int] = None,
        timeout_s: Optional[float] = None,
        snippet_limit: int = DEFAULT_DIAGNOSTICS_SNIPPET_LIMIT,
    ) -> None:
        self.agent_type = agent_type
        self.cwd = cwd
        self.model_name = model_name
        self.session = session
        self.error = error
        self.attempt = attempt
        self.retries = retries
        self.timeout_s = timeout_s
        self.snippet_limit = snippet_limit

    def build(self) -> dict:
        return _build_startup_diagnostics_impl(
            agent_type=self.agent_type,
            cwd=self.cwd,
            model_name=self.model_name,
            session=self.session,
            error=self.error,
            attempt=self.attempt,
            retries=self.retries,
            timeout_s=self.timeout_s,
            snippet_limit=self.snippet_limit,
        )


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
    """Compatibility entrypoint for startup diagnostics construction."""
    return StartupDiagnosticsBuilder(
        agent_type=agent_type,
        cwd=cwd,
        model_name=model_name,
        session=session,
        error=error,
        attempt=attempt,
        retries=retries,
        timeout_s=timeout_s,
        snippet_limit=snippet_limit,
    ).build()


def _build_startup_diagnostics_impl(
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
    """构造稳定可序列化的启动失败诊断信息。

    目标：无论错误对象/会话对象携带的信息是否完整，最终日志字段都稳定存在，
    便于定位“启动失败但日志为空/极少”的问题。

    必含字段：cmd/args/rc/stdout_snippet/stderr_snippet。
    """
    # NOTE: `build_startup_diagnostics` is a compat entry.
    # New SSOT for non-empty fallbacks/redaction/truncation is
    # `src.acp.diagnostics.normalize_startup_diagnostics`.
    snippet_limit_eff = _resolve_startup_snippet_limit(snippet_limit)

    diag: dict = _initial_startup_diagnostics(
        agent_type=agent_type,
        cwd=cwd,
        model_name=model_name,
        error=error,
        attempt=attempt,
        retries=retries,
        timeout_s=timeout_s,
    )

    # error_repr (best-effort; later redaction+truncation handled by normalize_startup_diagnostics)
    diag["error_repr"] = safe_extract(
        lambda: truncate_text(repr(error) if error is not None else "", DEFAULT_DIAGNOSTICS_SNIPPET_LIMIT),
        default="",
        log_msg="build_startup_diagnostics: error_repr extraction failed",
    )

    # cmd/args: prefer session then error (标准协议优先：ACPStartupError.agent_cmd/agent_args)
    try:
        cmd = safe_str(getattr(session, "_agent_cmd", "") or "") if session is not None else ""
        args = list(getattr(session, "_agent_args", []) or []) if session is not None else []
        if cmd or args:
            diag["cmd"] = cmd
            diag["args"] = [str(x) for x in args]
    except Exception:
        logger.debug("build_startup_diagnostics: cmd/args from session failed", exc_info=True)

    if not diag.get("cmd") and not diag.get("args"):
        try:
            diag["cmd"] = safe_str(getattr(error, "agent_cmd", "") or "")
            diag["args"] = [str(x) for x in (getattr(error, "agent_args", []) or [])]
        except Exception:
            logger.debug("build_startup_diagnostics: cmd/args from error failed", exc_info=True)

    # 迁移期兜底（可删除条件：全链路启动失败仅抛 ACPStartupError/其子类，并稳定设置 agent_cmd/agent_args）。
    # Extra compatibility: some errors may use `cmd/args` instead of `agent_cmd/agent_args`.
    if (not diag.get("cmd")) and (not (diag.get("args") or [])):
        try:
            diag["cmd"] = safe_str(getattr(error, "cmd", "") or "")
            diag["args"] = [str(x) for x in (getattr(error, "args", []) or [])]
        except Exception:
            logger.debug("build_startup_diagnostics: cmd/args compat extraction failed", exc_info=True)

    # return code
    try:
        rc = getattr(error, "returncode", None)
        if rc is not None:
            diag["rc"] = int(rc)
    except Exception:
        logger.debug("build_startup_diagnostics: returncode extraction failed", exc_info=True)

    # 迁移期兜底（可删除条件同上）：部分历史错误用 `.rc` 表示 returncode。
    # Extra compatibility: some subprocess-like errors may use `.rc`.
    if diag.get("rc") is None:
        try:
            rc = getattr(error, "rc", None)
            if rc is not None:
                diag["rc"] = int(rc)
        except Exception:
            logger.debug("build_startup_diagnostics: rc compat extraction failed", exc_info=True)

    # Optional: fail_phase from ACPStartupError (for log aggregation)
    try:
        phase = safe_str(getattr(error, "fail_phase", "") or "")
        if phase:
            diag["fail_phase"] = truncate_text(phase, 80)
    except Exception:
        logger.debug("build_startup_diagnostics: fail_phase extraction failed", exc_info=True)

    if diag.get("rc") is None:
        try:
            # best-effort: if process exists and has returncode
            acp_session = getattr(session, "_acp_session", None) if session is not None else None
            proc = getattr(acp_session, "_proc", None) if acp_session is not None else None
            rc = getattr(proc, "returncode", None)
            if rc is not None:
                diag["rc"] = int(rc)
        except Exception:
            logger.debug("build_startup_diagnostics: rc from process extraction failed", exc_info=True)

    # stdout/stderr snippets: prefer explicit snippet fields
    try:
        out = safe_str(getattr(error, "stdout_snippet", "") or "")
        err = safe_str(getattr(error, "stderr_snippet", "") or "")
        if out:
            diag["stdout_snippet"] = truncate_text(out, snippet_limit_eff)
        if err:
            diag["stderr_snippet"] = truncate_text(err, snippet_limit_eff)
    except Exception:
        logger.debug("build_startup_diagnostics: stdout/stderr snippet extraction failed", exc_info=True)

    # fallback to stdout/stderr raw
    if not diag.get("stdout_snippet"):
        try:
            out = safe_str(getattr(error, "stdout", "") or "")
            if out:
                diag["stdout_snippet"] = truncate_text(out, snippet_limit_eff)
        except Exception:
            logger.debug("build_startup_diagnostics: stdout fallback extraction failed", exc_info=True)
    if not diag.get("stderr_snippet"):
        try:
            err = safe_str(getattr(error, "stderr", "") or "")
            if err:
                diag["stderr_snippet"] = truncate_text(err, snippet_limit_eff)
        except Exception:
            logger.debug("build_startup_diagnostics: stderr fallback extraction failed", exc_info=True)

    # 规范化兜底（SSOT 在 diagnostics.normalize_startup_diagnostics）：
    # 此处仅做最小采集，不在这里重复实现“非空兜底/脱敏/截断”。

    # cmd/args 兜底：若 session/error 都未提供命令信息，best-effort 通过 resolve_agent_spec 推断。
    try:
        if not safe_str(diag.get("cmd") or "").strip() and not list(diag.get("args") or []):
            try:
                cmd2, args2 = resolve_agent_spec(agent_type, model_name=model_name)
                diag["cmd"] = safe_str(cmd2 or "")
                diag["args"] = [str(x) for x in (args2 or [])]
            except Exception:
                logger.debug("build_startup_diagnostics: resolve_agent_spec fallback failed", exc_info=True)
    except Exception:
        logger.debug("build_startup_diagnostics: cmd/args fallback check failed", exc_info=True)

    # Best-effort fail_phase inference when upstream does not provide one.
    # Run it after snippets are filled so classification has more context.
    if not diag.get("fail_phase"):
        try:
            err_blob = "\n".join(
                [
                    safe_str(diag.get("error") or ""),
                    safe_str(diag.get("stdout_snippet") or ""),
                    safe_str(diag.get("stderr_snippet") or ""),
                ]
            )
        except Exception:
            logger.debug("build_startup_diagnostics: err_blob construction failed", exc_info=True)
            err_blob = ""
        try:
            phase_guess = classify_startup_fail_phase(error=error, error_blob=err_blob)
            diag["fail_phase"] = truncate_text(safe_str(phase_guess or ""), 80)
        except Exception:
            logger.debug("build_startup_diagnostics: fail_phase inference failed", exc_info=True)
            # ultimate fallback
            diag["fail_phase"] = "start_failed"

    # fail_reason 采集：上游若提供则保留；最终兜底由 normalize 统一处理
    try:
        diag["fail_reason"] = truncate_text(safe_str(getattr(error, "fail_reason", "") or ""), 80)
    except Exception:
        logger.debug("build_startup_diagnostics: fail_reason extraction failed", exc_info=True)
        diag["fail_reason"] = ""

    # human-readable spec (best-effort)
    try:
        if session is not None and hasattr(session, "describe_agent"):
            diag["spec"] = truncate_text(safe_str(session.describe_agent()), 400)
    except Exception:
        logger.debug("build_startup_diagnostics: spec extraction failed", exc_info=True)

    # Keep alias for compatibility
    if diag.get("spec") and not diag.get("agent_spec"):
        diag["agent_spec"] = diag.get("spec")

    # If cmd/args missing but spec looks like `cmd=... args=...`, parse it best-effort.
    if (not diag.get("cmd")) and (not (diag.get("args") or [])) and diag.get("spec"):
        try:
            s = str(diag.get("spec") or "")
            # very simple parser: cmd=<...> args=<...> cwd=<...>
            if "cmd=" in s:
                cmd_part = s.split("cmd=", 1)[1]
                cmd = cmd_part.split(" ", 1)[0].strip()
                if cmd:
                    diag["cmd"] = cmd
            if "args=" in s:
                args_part = s.split("args=", 1)[1]
                args_txt = args_part.split(" cwd=", 1)[0].strip()
                if args_txt:
                    diag["args"] = [x for x in args_txt.split() if x]
        except Exception:
            logger.debug("build_startup_diagnostics: spec parsing failed", exc_info=True)

    # Ensure required fields exist and are serializable.
    if diag.get("args") is None:
        diag["args"] = []
    if not isinstance(diag.get("args"), list):
        try:
            diag["args"] = list(diag.get("args") or [])
        except Exception:
            logger.debug("build_startup_diagnostics: args list conversion failed", exc_info=True)
            diag["args"] = []
    diag["args"] = [str(x) for x in (diag.get("args") or [])]
    diag["cmd"] = safe_str(diag.get("cmd") or "")
    diag["stdout_snippet"] = safe_str(diag.get("stdout_snippet") or "")
    diag["stderr_snippet"] = safe_str(diag.get("stderr_snippet") or "")
    diag["agent_spec"] = safe_str(diag.get("agent_spec") or "")
    # Re-assert timeout_s contract: None | float
    diag["timeout_s"] = safe_float_or_none(diag.get("timeout_s"))

    # Normalize error_text (some exceptions have empty __str__).
    # Compose from: str(error) -> stderr/stdout snippets -> cause/context -> repr(error) -> type fallback.
    try:
        msg = ""
        try:
            msg = safe_str(error) if error is not None else ""
        except Exception:
            logger.debug("build_startup_diagnostics: error str conversion failed", exc_info=True)
            msg = ""

        # If message is too generic/empty, prefer stderr/stdout snippets.
        stderr_snip = safe_str(diag.get("stderr_snippet") or "")
        stdout_snip = safe_str(diag.get("stdout_snippet") or "")
        if not (msg or "").strip() or (msg or "").strip() in ("(empty)", "None"):
            msg = (stderr_snip or "").strip() or (stdout_snip or "").strip()

        # Add 1-level cause/context if still empty or extremely short.
        if (not (msg or "").strip()) or len((msg or "").strip()) < 8:
            try:
                cause = getattr(error, "__cause__", None) or getattr(error, "__context__", None)
            except Exception:
                logger.debug("build_startup_diagnostics: cause extraction failed", exc_info=True)
                cause = None
            if cause is not None and cause is not error:
                try:
                    c_msg = safe_str(cause)
                except Exception:
                    logger.debug("build_startup_diagnostics: cause str conversion failed", exc_info=True)
                    c_msg = ""
                if not (c_msg or "").strip():
                    try:
                        c_msg = repr(cause)
                    except Exception:
                        logger.debug("build_startup_diagnostics: cause repr failed", exc_info=True)
                        c_msg = ""
                c_msg = (c_msg or "").strip()
                if c_msg:
                    msg = (msg or "").strip()
                    msg = (msg + "\n" if msg else "") + f"cause={type(cause).__name__}: {c_msg}"

        # Final fallback: repr(error) or <Type>.
        if not (msg or "").strip():
            msg = (safe_str(diag.get("error_repr") or "") or "").strip()
        if not (msg or "").strip():
            et = safe_str(diag.get("error_type") or "Exception") or "Exception"
            msg = f"<{et}> (empty output)"

        # Include key structured hints if available.
        hints: list[str] = []
        try:
            rc = diag.get("rc")
            if rc is not None:
                hints.append(f"rc={int(rc)}")
        except Exception:
            logger.debug("build_startup_diagnostics: rc hint failed", exc_info=True)
        try:
            ph = safe_str(diag.get("fail_phase") or "").strip()
            if ph:
                hints.append(f"phase={truncate_text(ph, 40)}")
        except Exception:
            logger.debug("build_startup_diagnostics: phase hint failed", exc_info=True)
        if hints:
            msg = (msg or "").strip()
            msg = (msg + "\n" if msg else "") + " ".join(hints)

        # Bound size before normalize to avoid giant intermediate strings.
        msg = truncate_text(msg, 400) or "(empty)"
        diag["error_text"] = msg
        diag["error"] = msg
    except Exception:
        logger.debug("build_startup_diagnostics: error_text normalization failed", exc_info=True)
        diag["error_text"] = "<Exception> (empty output)"
        diag["error"] = diag["error_text"]

    # Final SSOT normalize: non-empty fallbacks + redaction + truncation
    return normalize_startup_diagnostics(diag, get_settings_fn=get_settings)


def format_startup_diagnostics(diag: object, *, total_limit: int = DEFAULT_DIAGNOSTICS_TOTAL_LIMIT) -> str:
    """将 diagnostics 格式化为稳定的单行字符串（JSON），避免日志为空/难 grep。"""
    base = {
        "cmd": "",
        "args": [],
        "rc": None,
        "stdout_snippet": "",
        "stderr_snippet": "",
    }
    try:
        if isinstance(diag, dict):
            base.update(diag)
        else:
            base["error"] = truncate_text(safe_str(diag), 400)
    except Exception:
        logger.debug("format_startup_diagnostics: base construction failed", exc_info=True)
        base["error"] = "format_error"

    # Redaction + truncation ordering: redact first (on JSON string), then truncate.
    try:
        # NOTE: pass get_settings() explicitly to allow tests to monkeypatch
        # src.acp.sync_adapter.get_settings without touching global config.
        cfg = get_diagnostics_config(get_settings_fn=get_settings)
        enabled = bool(cfg.redact_enabled)
        patterns = list(cfg.redact_patterns or [])
        repl = str(cfg.redact_replacement or "***REDACTED***")
        if int(cfg.total_limit or 0) > 0:
            total_limit = int(cfg.total_limit)
    except Exception:
        logger.debug("format_startup_diagnostics: config loading failed", exc_info=True)
        enabled, patterns, repl = True, [], "***REDACTED***"

    try:
        s = json.dumps(base, ensure_ascii=False, sort_keys=True)
    except Exception:
        logger.debug("format_startup_diagnostics: json.dumps failed", exc_info=True)
        try:
            s = safe_str(base)
        except Exception:
            logger.debug("format_startup_diagnostics: safe_str fallback failed", exc_info=True)
            s = '{"error":"diagnostics_unavailable"}'

    if enabled:
        try:
            s = redact_text(s, patterns, repl)
        except Exception:
            logger.debug("format_startup_diagnostics: redaction failed", exc_info=True)
    return truncate_text(s, int(total_limit or DEFAULT_DIAGNOSTICS_TOTAL_LIMIT))


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


def _build_error_text(err: Exception) -> str:
    """从异常对象中尽量拼出可用于分类的文本。"""
    parts: list[str] = []
    try:
        parts.append(str(err) or "")
    except Exception:
        logger.debug("_build_error_text: str(err) failed", exc_info=True)
        parts.append("")
    for k in ("stderr_snippet", "stdout_snippet", "stderr", "stdout"):
        try:
            v = str(getattr(err, k, "") or "")
            if v:
                parts.append(v)
        except Exception:
            logger.debug("_build_error_text: getattr %s failed", k, exc_info=True)
            continue
    return "\n".join([p for p in parts if p])


@lru_cache(maxsize=64)
def _probe_acp_serve_help(command: str) -> tuple[bool, Optional[int], str, str]:
    """探测 `<command> acp serve --help` 是否可用，并返回 (ok, rc, stdout_snip, stderr_snip)。

    - ok=True 仅表示该命令支持 ACP server 启动（可用 `acp serve`）。
    - 该探测用于 TTADK tool adapter，避免对不支持 ACP 的 tool 进入 handshake 超时。
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


def _resolve_ttadk_passthrough_args(tool_name: str) -> list[str]:
    """Resolve `ttadk code -a <args>` for a specific tool.

    Returns a list of strings for ttadk's `-a/--args` passthrough.
    """
    tool = (tool_name or "").strip().lower()
    if not tool:
        raise AgentSpecResolveError("TTADK tool 为空，无法解析启动参数")

    # `ttadk code -a` 仅透传参数给下游 tool。GhostAP 的 ACP backend 依赖下游 tool
    # 以 `acp serve` 形式输出 JSON-RPC over stdio。
    # 因此这里做一次轻量探测：不支持则立刻抛 AgentSpecResolveError，让上层走确定性降级，避免 handshake 超时。

    if tool == "coco":
        return ["acp", "serve"]

    # 兼容旧行为：对 claude 先返回 passthrough，由上层 quickcheck 决定是否降级。
    # 原因：claude/codex 等工具在不同环境下能力差异较大，resolve_agent_spec 应尽量保持纯函数。
    if tool == "claude":
        return ["acp", "serve"]

    ok, rc, out_snip, err_snip = _probe_acp_serve_help(tool)
    if ok:
        return ["acp", "serve"]

    raise AgentSpecResolveError(
        f"TTADK tool={tool} 不支持 `acp serve`（将触发降级）",
        agent_cmd=tool,
        agent_args=["acp", "serve"],
        returncode=rc,
        stdout_snippet=out_snip,
        stderr_snippet=err_snip,
    )


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


@lru_cache(maxsize=1)
def _resolve_tui2acp_adapters_dir() -> Optional[str]:
    """Locate tui2acp's bundled adapter YAML directory.

    The built-in declarative adapter registry inside tui2acp ships incomplete
    configs (missing `states`) that crash at construction time for adapters
    like pi-coding-agent / aichat / sgpt / open-interpreter. The package's
    `adapters/` directory contains complete YAML configs; we point tui2acp at
    them via `--adapters-dir` to override the broken built-ins.
    """
    import os
    import shutil

    from ..utils.env import build_clean_env

    _augmented_path = build_clean_env().get("PATH", "")

    # 1. Resolve via the tui2acp executable's npm install location.
    bin_path = shutil.which("tui2acp", path=_augmented_path)
    if bin_path:
        try:
            real = os.path.realpath(bin_path)
            # Typical layouts:
            #   <prefix>/lib/node_modules/tui2acp/dist/cli.js (real)
            #   <prefix>/lib/node_modules/tui2acp/adapters/  (target)
            pkg_root = os.path.dirname(os.path.dirname(real))
            candidate = os.path.join(pkg_root, "adapters")
            if os.path.isdir(candidate):
                return candidate
        except Exception:
            logger.debug("tui2acp adapters-dir resolve via realpath failed", exc_info=True)

    # 2. Fallback: probe common npm-global install locations.
    home = os.path.expanduser("~")
    for prefix in (
        os.path.join(home, ".npm-global"),
        "/opt/homebrew",
        "/usr/local",
    ):
        candidate = os.path.join(prefix, "lib", "node_modules", "tui2acp", "adapters")
        if os.path.isdir(candidate):
            return candidate
    return None


def resolve_agent_spec(
    agent_type: str, model_name: Optional[str] = None, *, ttadk_use_pty: bool = False
) -> tuple[str, list[str]]:
    """Resolve (command, args) for spawning an ACP agent process over stdio."""
    agent_type = (agent_type or "").lower()

    settings = get_settings()
    override_cmd, override_args = settings.get_acp_command(agent_type)
    if override_cmd:
        return override_cmd, override_args

    if agent_type.startswith("ttadk_"):
        tool_name = agent_type[len("ttadk_") :]

        # Use wrapper module to filter out TTADK banner (which breaks JSON-RPC).
        # IMPORTANT: use `-m` to avoid script/relative-import drift.
        wrapper_module = "src.ttadk.wrapper"

        # SSOT: TTADK 侧要求透传真实 model_id；这里做最后一道 best-effort 归一化，
        # 防止上游误把 display/alias 直接透传到 -m 导致 invalid model。
        input_model = (model_name or "").strip()
        resolved_model: Optional[str] = None
        resolution_source = ""
        resolution_reason = ""
        if input_model:
            try:
                from ..ttadk import get_ttadk_manager
                from ..ttadk.models import is_model_token as _is_model_token
                from ..ttadk.models import resolve_model_id as _resolve_model_id

                # 关键约束：resolve_agent_spec 必须是“纯函数/无外部副作用”。
                # 因此这里只读取内存缓存（不触发 fetch/probe/磁盘 I/O），防止单测/启动路径被阻塞。
                mgr = get_ttadk_manager()
                try:
                    with getattr(mgr, "_lock", None) or _NULL_LOCK:  # type: ignore[name-defined]
                        descriptors = list(getattr(mgr, "_tool_models_cache", {}).get(tool_name, []) or [])
                except (AttributeError, TypeError, KeyError):
                    descriptors = []

                if descriptors:
                    r, diag = _resolve_model_id(
                        tool_name=tool_name,
                        input_name=input_model,
                        descriptors=descriptors,
                        allow_unknown_passthrough=True,
                    )
                    cand = str(getattr(r, "real_name", "") or "").strip()
                    src = str(getattr(r, "source", "") or "")
                    resolution_source = src
                    if isinstance(diag, dict):
                        resolution_reason = str(diag.get("resolution_reason") or "")
                    if cand and src != "unknown":
                        resolved_model = cand
                    elif _is_model_token(input_model):
                        resolved_model = input_model
                        if not resolution_source:
                            resolution_source = "token_passthrough"
                    else:
                        resolved_model = None
                        if not resolution_source:
                            resolution_source = "drop_m"
                        if not resolution_reason:
                            resolution_reason = "unresolved_display_or_alias"
                else:
                    # 无缓存：只做 token 透传，否则不透传 -m
                    if _is_model_token(input_model):
                        resolved_model = input_model
                        resolution_source = "token_passthrough"
                        resolution_reason = "no_cache"
                    else:
                        resolved_model = None
                        resolution_source = "drop_m"
                        resolution_reason = "no_cache"
            except (ImportError, AttributeError, TypeError, ValueError):
                logger.debug("resolve_agent_spec: TTADK model resolution failed, using input_model as fallback", exc_info=True)
                resolved_model = input_model

        passthrough = _resolve_ttadk_passthrough_args(tool_name)
        args = ["-m", wrapper_module]
        if ttadk_use_pty:
            args.append("--pty")
        args.extend(["ttadk", "code", "-t", tool_name])
        if resolved_model:
            args.extend(["-m", str(resolved_model)])

        # NOTE: `-a/--args` is passthrough to downstream tool CLI.
        for arg in passthrough:
            args.extend(["-a", arg])

        logger.info(
            "[ACP:TTADK] resolve_agent_spec: tool=%s model=%s input_model=%s resolution_source=%s resolution_reason=%s passthrough=%s pty=%s",
            tool_name,
            (resolved_model or "(auto)"),
            (input_model or ""),
            (resolution_source or ""),
            (resolution_reason or ""),
            passthrough,
            bool(ttadk_use_pty),
        )
        return "python3", args

    if agent_type.startswith("tui2acp_"):
        import shutil

        from ..utils.env import build_clean_env

        _augmented_path = build_clean_env().get("PATH", "")
        if not shutil.which("tui2acp", path=_augmented_path):
            raise AgentSpecResolveError(
                "tui2acp 未安装或不在 PATH 中，请运行: npm install -g tui2acp",
                agent_cmd="tui2acp",
                agent_args=[],
            )
        adapter_name = agent_type[len("tui2acp_"):]

        # Custom command mode: "custom:<full command>"
        if adapter_name.startswith("custom:"):
            custom_cmd = adapter_name[len("custom:"):]
            # First word is the agent name for tui2acp
            parts = custom_cmd.split()
            agent_name = parts[0] if parts else "custom"
            args = ["--agent", agent_name, "--unsafe", "--minimal"]
            adapters_dir = _resolve_tui2acp_adapters_dir()
            if adapters_dir:
                args.extend(["--adapters-dir", adapters_dir])
            return "tui2acp", args

        args = ["--adapter", adapter_name, "--unsafe", "--minimal"]
        # tui2acp's built-in declarative adapter registry has incomplete
        # configs (missing `states`) that crash on construction. Force loading
        # the bundled adapter YAML files so adapter configs are complete.
        adapters_dir = _resolve_tui2acp_adapters_dir()
        if adapters_dir:
            args.extend(["--adapters-dir", adapters_dir])
        return "tui2acp", args

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
    ttadk_use_pty: bool = False,
    log_failures: bool = True,
    env: Optional[dict[str, str]] = None,
    retries: Optional[int] = None,
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

    for attempt in range(1, retries + 1):
        try:
            if env is not None:
                session = session_cls(
                    agent_type=agent_type,
                    cwd=cwd,
                    model_name=model_name,
                    ttadk_use_pty=bool(ttadk_use_pty),
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
                    session = session_cls(
                        agent_type=agent_type, cwd=cwd, model_name=model_name, ttadk_use_pty=bool(ttadk_use_pty)
                    )
                except TypeError:
                    # 兼容旧签名 / 测试桩：不支持 ttadk_use_pty 时仍应保留 model_name 透传
                    logger.debug("session_cls does not accept ttadk_use_pty, retrying without", exc_info=True)
                    try:
                        session = session_cls(agent_type=agent_type, cwd=cwd, model_name=model_name)
                    except TypeError:
                        logger.debug("session_cls does not accept model_name, using minimal signature", exc_info=True)
                        session = session_cls(agent_type=agent_type, cwd=cwd)
            else:
                session = session_cls(agent_type=agent_type, cwd=cwd, ttadk_use_pty=bool(ttadk_use_pty))
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
            # 解析 agent spec 失败（例如 TTADK tool 不支持 `acp serve`）时重试无意义，直接交给上层降级。
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
            cmd, args = resolve_agent_spec(agent_type, model_name=model_name, ttadk_use_pty=bool(ttadk_use_pty))
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


def start_agent_session_with_diagnostics(
    *,
    agent_type: str,
    cwd: str,
    startup_timeout: float = 60,
    model_name: Optional[str] = None,
    session_cls: Optional[type["SyncACPSession"]] = None,
    ttadk_use_pty: bool = False,
    log_failures: bool = True,
) -> tuple["SyncACPSession", str, dict]:
    """通用 ACP 启动器封装：成功返回 (session, session_id, diagnostics)。

    设计目的：
    - 作为上层（ACPSessionManager / engines）的“可注入启动器”候选实现
    - 统一失败诊断载体：在异常上附加 `.diagnostics`（dict，含非空 error_text）

    约定：
    - 成功：`session_id` 必须为非空字符串
    - 失败：抛出异常（优先 ACPStartupError），且 `getattr(exc, 'diagnostics', None)` 可读
    """
    try:
        s = start_session_with_retry(
            agent_type=agent_type,
            cwd=cwd,
            startup_timeout=startup_timeout,
            model_name=model_name,
            session_cls=session_cls,
            ttadk_use_pty=bool(ttadk_use_pty),
            log_failures=bool(log_failures),
        )
        sid = str(getattr(s, "session_id", "") or "").strip()
        if not sid:
            # 极端兜底：不允许“成功但无 session_id”，避免上层误判。
            raise ACPStartupError(
                "ACP session started but session_id is empty",
                agent_cmd=safe_str(getattr(s, "_agent_cmd", "") or ""),
                agent_args=[str(x) for x in (getattr(s, "_agent_args", None) or [])],
                cwd=cwd,
                returncode=None,
                stdout_snippet="",
                stderr_snippet="",
                fail_phase="missing_session_id",
                cause=None,
            )
        return (s, sid, {"attempts": [{"phase": "start", "ok": True}]})
    except Exception as e:
        # 统一构造 diagnostics（SSOT=build_startup_diagnostics/normalize_startup_diagnostics）
        try:
            d = build_startup_diagnostics(
                agent_type=agent_type,
                cwd=cwd,
                model_name=model_name,
                session=None,
                error=e,
                attempt=1,
                retries=max(1, int(getattr(get_settings(), "acp_startup_retries", 1) or 1)),
                timeout_s=float(startup_timeout or 0),
            )
            d = normalize_startup_diagnostics(d, get_settings_fn=get_settings)
        except Exception:
            logger.debug("start_agent_session_with_diagnostics: diagnostics construction failed", exc_info=True)
            d = {"error_text": get_error_detail(e), "fail_reason": "start_failed"}
        try:
            e.diagnostics = d
        except Exception:
            logger.debug("start_agent_session_with_diagnostics: attaching diagnostics failed", exc_info=True)
        raise


def _call_start_session_with_retry_compat(
    *,
    agent_type: str,
    cwd: str,
    startup_timeout: float,
    model_name: Optional[str],
    session_cls: Optional[type["SyncACPSession"]],
    ttadk_use_pty: bool,
    log_failures: bool,
) -> SyncACPSession:
    """兼容调用 `start_session_with_retry`（关键字调用 + 明确降参顺序）。

    背景：单测桩/历史版本的 `start_session_with_retry` 可能缺少部分参数，常见差异：
    - 不支持 `log_failures`
    - 不支持 `ttadk_use_pty`
    - 不支持 `session_cls`

    设计：
    - 永远使用关键字调用，避免 positional fallback 破坏 kw-only 的测试桩
    - 捕获 TypeError 后按固定顺序逐步“降参”并重试（顺序是约定的一部分）：
      1) 去掉 `log_failures`（最晚加入、最可能缺失）
      2) 去掉 `ttadk_use_pty`（TTADK 专用开关，旧签名常缺）
      3) 去掉 `session_cls`（测试桩常不接收）
    - 最终仍失败时，抛出“最后一次 TypeError”，保持现有语义

    适用范围：仅用于兼容 TTADK 启动链路/测试桩，不建议在非 TTADK 场景扩散使用。
    """

    base = {
        "agent_type": agent_type,
        "cwd": cwd,
        "startup_timeout": startup_timeout,
        "model_name": model_name,
        "session_cls": session_cls,
        "ttadk_use_pty": bool(ttadk_use_pty),
        "log_failures": bool(log_failures),
    }

    # 注意：这里的顺序是约定的一部分，请修改时同步更新单测。
    candidates: list[dict] = []
    candidates.append(dict(base))
    d = dict(base)
    d.pop("log_failures", None)
    candidates.append(d)
    d2 = dict(d)
    d2.pop("ttadk_use_pty", None)
    candidates.append(d2)
    d3 = dict(d2)
    d3.pop("session_cls", None)
    candidates.append(d3)

    last_exc: Exception | None = None
    for kw in candidates:
        try:
            # 移除 None 值，减少对签名的压力
            kw = {k: v for k, v in kw.items() if v is not None or k in ("model_name",)}
            return start_session_with_retry(**kw)
        except TypeError as e:
            last_exc = e
            continue

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("unexpected_start_failure")


def start_ttadk_session_with_pty_retry(
    *,
    agent_type: str,
    cwd: str,
    startup_timeout: float = 60,
    model_name: Optional[str] = None,
    session_cls: Optional[type["SyncACPSession"]] = None,
    log_failures: bool = True,
) -> SyncACPSession:
    """TTADK 专用：先普通启动，若命中 stdin-not-tty 则自动用 PTY 重试一次。"""
    settings = get_settings()
    pty_enabled = bool(getattr(settings, "ttadk_pty_enabled", True))
    retry_once = bool(getattr(settings, "ttadk_pty_retry_once", True))
    cooldown_s = float(getattr(settings, "ttadk_pty_retry_cooldown_s", 60.0) or 60.0)
    cooldown_s = max(0.0, cooldown_s)

    # per-tool cooldown to avoid restart thrashing
    tool = (agent_type or "").strip().lower()
    now = time.time()
    try:
        _last = getattr(start_ttadk_session_with_pty_retry, "_last_retry_ts", None)
        if not isinstance(_last, dict):
            _last = {}
            start_ttadk_session_with_pty_retry._last_retry_ts = _last
    except Exception:
        _last = {}
        logger.debug("Failed to read PTY retry state", exc_info=True)
    if not pty_enabled:
        return _call_start_session_with_retry_compat(
            agent_type=agent_type,
            cwd=cwd,
            startup_timeout=startup_timeout,
            model_name=model_name,
            session_cls=session_cls,
            ttadk_use_pty=False,
            log_failures=bool(log_failures),
        )

    def _start(*, use_pty: bool) -> SyncACPSession:
        """Best-effort call start_session_with_retry with backward compatibility."""
        return _call_start_session_with_retry_compat(
            agent_type=agent_type,
            cwd=cwd,
            startup_timeout=startup_timeout,
            model_name=model_name,
            session_cls=session_cls,
            ttadk_use_pty=bool(use_pty),
            log_failures=bool(log_failures),
        )

    try:
        return _start(use_pty=False)
    except Exception as e:
        if not retry_once:
            raise
        # 分类 stdin-not-tty 必须是 best-effort，不能因为“分类逻辑自身异常”把原始错误信息吞掉。
        try:
            from ..ttadk.models import is_stdin_not_tty_error

            blob = _build_error_text(e)
            if not is_stdin_not_tty_error(blob):
                raise
        except Exception:
            # 分类失败：按原异常抛出（保持可诊断字段）
            raise

        logger.warning(
            "[ACP:%s] Detected stdin-not-tty, retry with PTY once: tool=%s cwd=%s model=%s reason=stdin_not_tty err_type=%s",
            (agent_type or "").upper(),
            tool,
            cwd,
            model_name or "(auto)",
            type(e).__name__,
        )

        # cooldown gate
        try:
            last_ts = float(_last.get(tool, 0.0) or 0.0) if isinstance(_last, dict) else 0.0
        except Exception:
            last_ts = 0.0
            logger.debug("Failed to parse cooldown timestamp", exc_info=True)
        if cooldown_s and last_ts and (now - last_ts) < cooldown_s:
            logger.warning(
                "[ACP:%s] PTY retry suppressed by cooldown: tool=%s cooldown_s=%.1f elapsed=%.1f",
                (agent_type or "").upper(),
                tool,
                cooldown_s,
                max(0.0, now - last_ts),
            )
            # 冷却抑制：抛出原始错误（保留可诊断字段），避免出现空错误
            raise
        try:
            if isinstance(_last, dict):
                _last[tool] = now
        except Exception:
            logger.debug("Failed to update cooldown timestamp for tool=%s", tool, exc_info=True)
        try:
            return _start(use_pty=True)
        except Exception as e2:
            # 若 PTY 重试也失败，用可诊断异常包裹，确保上层日志/diagnostics 不会为空。
            blob = ""
            try:
                blob = _build_error_text(e2)
            except Exception:
                blob = ""
                logger.debug("Failed to build error text for PTY retry failure", exc_info=True)
            raise ACPStartupError(
                "TTADK PTY 重试后仍启动失败",
                agent_cmd=safe_str(getattr(e2, "agent_cmd", "") or "python3"),
                agent_args=[str(x) for x in (getattr(e2, "agent_args", None) or [])] or ["(unknown)"],
                cwd=cwd,
                returncode=getattr(e2, "returncode", None),
                stdout_snippet=truncate_text(
                    safe_str(getattr(e2, "stdout_snippet", "") or getattr(e2, "stdout", "") or ""), DEFAULT_DIAGNOSTICS_SNIPPET_LIMIT
                ),
                stderr_snippet=truncate_text(
                    safe_str(getattr(e2, "stderr_snippet", "") or getattr(e2, "stderr", "") or blob), DEFAULT_DIAGNOSTICS_SNIPPET_LIMIT
                ),
                fail_phase="pty_retry",
                cause=e2,
            ) from e2


class SyncACPSession:
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
        ttadk_use_pty: bool = False,
        env: Optional[dict[str, str]] = None,
    ):
        self._agent_type = agent_type
        self._cwd = cwd
        if agent_cmd is not None:
            self._agent_cmd = agent_cmd
            self._agent_args = agent_args or []
        else:
            cmd, args = resolve_agent_spec(agent_type, model_name=model_name, ttadk_use_pty=bool(ttadk_use_pty))
            self._agent_cmd = cmd
            self._agent_args = agent_args or args
        self._model_name = (model_name or "").strip() or None
        self._explicit_env = dict(env) if env is not None else None
        self._log_failures = True
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._acp_session: Optional[ACPSession] = None
        self._started = threading.Event()
        self._prompt_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

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

    async def _start_session(self) -> str:
        env_override = (
            dict(self._explicit_env)
            if getattr(self, "_explicit_env", None) is not None
            else None
        )
        try:
            if env_override is None and (self._agent_type or "").lower().startswith("ttadk_"):
                tool = (self._agent_type or "").lower().replace("ttadk_", "", 1)
                env_override, _ = build_ttadk_subprocess_env(
                    cwd=self._cwd or ".", agent_type=self._agent_type, tool_name=tool
                )
        except (OSError, ValueError, KeyError) as e:
            logger.debug("[ACP:%s] build_ttadk_subprocess_env failed, using default env: %s", self._agent_type, str(e))
            env_override = None

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
            agent_cmd=self._agent_cmd, agent_args=self._agent_args, cwd=self._cwd, env=env_override
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
        return session_id

    def load_session(self, session_id: str) -> None:
        """Load an existing session (for resume)."""
        if not self._acp_session:
            raise RuntimeError("Session not started")
        self._run_async(self._acp_session.load_session(session_id))
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
    ) -> PromptResult:
        """Send one prompt, rejecting concurrent callers on this wrapper."""
        prompt_lock = getattr(self, "_prompt_lock", None)
        if prompt_lock is None:
            prompt_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
            self._prompt_lock = prompt_lock
        if not prompt_lock.acquire(blocking=False):
            raise RuntimeError("ACP prompt is already running for this session")
        try:
            return self._send_prompt_once(
                text,
                on_event=on_event,
                timeout=timeout,
                idle_timeout=idle_timeout,
            )
        finally:
            prompt_lock.release()

    def _send_prompt_once(
        self,
        text: str,
        on_event: Optional[Callable[[ACPEvent], None]] = None,
        timeout: Optional[int] = None,
        idle_timeout: Optional[float] = None,
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
            last_activity_ts[0] = time.time()
            if on_event:
                on_event(ev)

        use_adaptive = effective_idle_timeout > 0
        event_cb = _activity_on_event if use_adaptive else on_event

        future = asyncio.run_coroutine_threadsafe(
            self._acp_session.prompt(text, on_event=event_cb),
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
            self.cancel(wait=True, timeout=2.0)
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

    def cancel(self, wait: bool = False, timeout: float = 2.0) -> None:
        """Cancel current prompt.

        When wait=True, block (up to `timeout` s) until the agent has acknowledged
        the cancel. This prevents the race where a follow-up `send_prompt` lands
        at the agent before cancel is processed, causing a `-32602 Invalid params`
        rejection because the session is still mid-cancel.
        """
        if not (self._acp_session and self._loop):
            return
        fut = asyncio.run_coroutine_threadsafe(
            self._acp_session.cancel(),
            self._loop,
        )
        if not wait:
            return
        try:
            fut.result(timeout=timeout)
        except (TimeoutError, OSError, RuntimeError) as e:
            logger.debug("[ACP] cancel wait skipped: %s", get_error_detail(e))

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
