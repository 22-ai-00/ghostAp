"""ACP diagnostics helpers (redaction + truncation).

Must NOT import ``src.acp.sync_adapter`` to avoid circular imports.
"""

from __future__ import annotations

import json
import logging
import numbers
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Optional

from ..config import get_settings as _default_get_settings

logger = logging.getLogger(__name__)

DEFAULT_DIAGNOSTICS_ARGS_LIMIT = 600
DEFAULT_DIAGNOSTICS_SNIPPET_LIMIT = 240
DEFAULT_DIAGNOSTICS_TOTAL_LIMIT = 2000

__all__ = [
    "DiagnosticsConfig",
    "DEFAULT_DIAGNOSTICS_ARGS_LIMIT",
    "DEFAULT_DIAGNOSTICS_SNIPPET_LIMIT",
    "DEFAULT_DIAGNOSTICS_TOTAL_LIMIT",
    "get_diagnostics_config",
    "safe_str",
    "truncate_text",
    "truncate_args",
    "redact_text",
    "normalize_startup_diagnostics",
    "format_startup_diagnostics_summary",
    "format_startup_failure_log_line",
]

_DEFAULT_REDACT_PATTERNS: list[str] = [
    r"(?i)authorization\s*:\s*[^\s]+",
    r"(?i)bearer\s+[^\s]+",
    r"sk-[A-Za-z0-9]{10,}",
    r"AKIA[0-9A-Z]{16}",
    r"(?i)api[_-]?key\s*[:=]\s*[^\s]+",
    r"(?i)secret\s*[:=]\s*[^\s]+",
    r"(?i)token\s*[:=]\s*[^\s]+",
]

_BASE_FIELDS = (
    "cmd",
    "args",
    "rc",
    "stdout_snippet",
    "stderr_snippet",
    "fail_reason",
    "error_text",
    "spec",
    "error",
)


@dataclass(frozen=True)
class DiagnosticsConfig:
    """Diagnostics redaction + truncation config (best-effort)."""

    redact_enabled: bool = True
    redact_patterns: list[str] = field(default_factory=lambda: list(_DEFAULT_REDACT_PATTERNS))
    redact_replacement: str = "***REDACTED***"
    args_limit: int = DEFAULT_DIAGNOSTICS_ARGS_LIMIT
    snippet_limit: int = DEFAULT_DIAGNOSTICS_SNIPPET_LIMIT
    total_limit: int = DEFAULT_DIAGNOSTICS_TOTAL_LIMIT


def safe_str(x: object) -> str:
    """Best-effort string conversion (never raises)."""
    try:
        if x is None:
            return ""
        return x if isinstance(x, str) else str(x)
    except Exception:
        logger.warning("Error while converting to str in safe_str", exc_info=True)
        try:
            return repr(x)
        except Exception:
            return ""


def _safe_nonneg_int(value: object, default: int) -> int:
    try:
        if value is None or isinstance(value, bool):
            return default
        if isinstance(value, (int, numbers.Integral, float)):
            v = int(value)
        elif isinstance(value, str):
            v = int(value.strip() or "0")
        else:
            return default
        return v if v >= 0 else default
    except Exception:
        logger.warning("Error while converting to int in diagnostics", exc_info=True)
        return default


def _safe_str_list(value: object, default: list[str]) -> list[str]:
    if isinstance(value, (list, tuple)):
        out = [safe_str(x).strip() for x in value]
        return [s for s in out if s] or list(default)
    if isinstance(value, str):
        out = [x.strip() for x in value.replace(",", "\n").splitlines() if x.strip()]
        return out or list(default)
    return list(default)


def get_diagnostics_config(*, get_settings_fn: Optional[Callable[[], object]] = None) -> DiagnosticsConfig:
    """Read diagnostics config from settings (best-effort, never raises)."""
    enabled = True
    patterns = list(_DEFAULT_REDACT_PATTERNS)
    repl = "***REDACTED***"
    args_limit = DEFAULT_DIAGNOSTICS_ARGS_LIMIT
    snippet_limit = DEFAULT_DIAGNOSTICS_SNIPPET_LIMIT
    total_limit = DEFAULT_DIAGNOSTICS_TOTAL_LIMIT

    try:
        s = (get_settings_fn or _default_get_settings)()
        raw_enabled = getattr(s, "acp_diagnostics_redact_enabled", True)
        if isinstance(raw_enabled, bool):
            enabled = raw_enabled
        patterns = _safe_str_list(getattr(s, "acp_diagnostics_redact_patterns", None), _DEFAULT_REDACT_PATTERNS)
        repl = safe_str(getattr(s, "acp_diagnostics_redact_replacement", repl) or repl) or repl
        args_limit = _safe_nonneg_int(getattr(s, "acp_diagnostics_args_limit", args_limit), args_limit)
        snippet_limit = _safe_nonneg_int(getattr(s, "acp_diagnostics_snippet_limit", snippet_limit), snippet_limit)
        total_limit = _safe_nonneg_int(getattr(s, "acp_diagnostics_total_limit", total_limit), total_limit)
    except Exception:
        logger.debug("diagnostics config resolution failed, using defaults", exc_info=True)

    return DiagnosticsConfig(
        redact_enabled=bool(enabled),
        redact_patterns=list(patterns or []),
        redact_replacement=repl or "***REDACTED***",
        args_limit=args_limit,
        snippet_limit=snippet_limit,
        total_limit=total_limit,
    )


@lru_cache(maxsize=128)
def _compile_redaction_patterns(patterns: tuple[str, ...]) -> tuple[re.Pattern, ...]:
    """Compile redaction regex patterns with caching (best-effort)."""
    out: list[re.Pattern] = []
    for p in patterns or ():
        if not p:
            continue
        try:
            out.append(re.compile(p))
        except Exception:
            logger.warning("Error while compiling redaction pattern", exc_info=True)
    return tuple(out)


def truncate_text(text: str, limit: int) -> str:
    """Bounded text truncation (never raises)."""
    try:
        limit = int(limit or 0)
    except Exception:
        limit = 0
    if limit <= 0:
        return ""
    s = safe_str(text)
    if not s or len(s) <= limit:
        return s
    return s[:limit] + "…(truncated)"


def truncate_args(args: list[str], limit: int) -> list[str]:
    """Bounded CLI args truncation (never raises)."""
    try:
        limit = int(limit or 0)
    except Exception:
        limit = 0
    if limit <= 0:
        return list(args or [])
    out: list[str] = []
    used = 0
    for a in args or []:
        s = safe_str(a)
        if not s:
            continue
        extra = (1 if out else 0) + len(s)
        if used + extra <= limit:
            out.append(s)
            used += extra
            continue
        out.append("…(truncated)" if out else truncate_text(s, limit))
        break
    return out


def redact_text(text: str, patterns: list[str], replacement: str) -> str:
    """Regex-based redaction with compiled-pattern cache (best-effort)."""
    s = safe_str(text)
    if not s:
        return ""
    rep = safe_str(replacement) or "***REDACTED***"
    pats = tuple(p for p in (safe_str(item) for item in (patterns or [])) if p)
    if not pats:
        return s
    try:
        compiled = _compile_redaction_patterns(pats)
    except Exception:
        compiled = ()
    for cre in compiled:
        try:
            s = cre.sub(rep, s)
        except Exception:
            logger.warning("Error while applying redaction pattern", exc_info=True)
    return s


def _resolve_diag_config(
    get_settings_fn: Optional[Callable[[], object]],
) -> tuple[bool, list[str], str, int, int, int]:
    try:
        cfg = get_diagnostics_config(get_settings_fn=get_settings_fn)
        return (
            bool(cfg.redact_enabled),
            list(cfg.redact_patterns or []),
            safe_str(cfg.redact_replacement or "***REDACTED***") or "***REDACTED***",
            cfg.args_limit or DEFAULT_DIAGNOSTICS_ARGS_LIMIT,
            cfg.snippet_limit or DEFAULT_DIAGNOSTICS_SNIPPET_LIMIT,
            cfg.total_limit or DEFAULT_DIAGNOSTICS_TOTAL_LIMIT,
        )
    except Exception:
        return (
            True,
            list(_DEFAULT_REDACT_PATTERNS),
            "***REDACTED***",
            DEFAULT_DIAGNOSTICS_ARGS_LIMIT,
            DEFAULT_DIAGNOSTICS_SNIPPET_LIMIT,
            DEFAULT_DIAGNOSTICS_TOTAL_LIMIT,
        )


def normalize_startup_diagnostics(
    diag: object,
    *,
    get_settings_fn: Optional[Callable[[], object]] = None,
) -> dict:
    """Normalize startup diagnostics into a stable, redacted, bounded dict."""
    enabled, patterns, repl, args_limit, snippet_limit, total_limit = _resolve_diag_config(get_settings_fn)
    out: dict = {k: ([] if k == "args" else (None if k == "rc" else "")) for k in _BASE_FIELDS}
    try:
        if isinstance(diag, dict):
            out.update(diag)
        elif diag is None:
            out["error"] = "(empty)"
        else:
            out["error"] = truncate_text(safe_str(diag), 400)
    except Exception:
        out["error"] = "format_error"

    try:
        xs = out.get("args")
        out["args"] = [str(x) for x in ([] if xs is None else (xs if isinstance(xs, list) else list(xs or [])))]
    except Exception:
        out["args"] = []

    try:
        rc = out.get("rc")
        out["rc"] = None if (rc is None or rc == "") else int(rc)
    except Exception:
        out["rc"] = None

    str_fields = ("cmd", "stdout_snippet", "stderr_snippet", "spec", "error", "error_text", "fail_reason")
    for field_name in str_fields:
        out[field_name] = safe_str(out.get(field_name) or "")
    if out.get("agent_spec") is not None:
        out["agent_spec"] = safe_str(out.get("agent_spec") or "")

    out["fail_reason"] = (out["fail_reason"] or safe_str(out.get("fail_phase") or "")).strip() or "start_failed"
    if not out["error_text"].strip():
        etype = safe_str(out.get("error_type") or "Exception") or "Exception"
        out["error_text"] = (
            out["stderr_snippet"].strip()
            or out["error"].strip()
            or f"<{etype}> (empty output)"
        )
    if not out["error"].strip():
        out["error"] = out["error_text"]

    if enabled:
        try:
            for key in (*str_fields, *(() if "agent_spec" not in out else ("agent_spec",))):
                out[key] = redact_text(out.get(key) or "", patterns, repl)
            out["args"] = [redact_text(str(x), patterns, repl) for x in (out.get("args") or [])]
        except Exception:
            logger.debug("redaction pass failed", exc_info=True)

    err_limit = min(400, snippet_limit) if snippet_limit > 0 else DEFAULT_DIAGNOSTICS_SNIPPET_LIMIT
    spec_limit = min(400, total_limit) if total_limit > 0 else 400
    try:
        out["args"] = truncate_args(out["args"], args_limit)
    except Exception:
        pass
    for key, lim, fallback in (
        ("cmd", snippet_limit, ""),
        ("stdout_snippet", snippet_limit, ""),
        ("stderr_snippet", snippet_limit, ""),
        ("error_text", err_limit, "(empty)"),
        ("error", err_limit, out["error_text"] or "(empty)"),
        ("fail_reason", 80, "start_failed"),
        ("spec", spec_limit, ""),
    ):
        out[key] = truncate_text(safe_str(out.get(key) or ""), lim) or fallback
    if "agent_spec" in out:
        out["agent_spec"] = truncate_text(safe_str(out.get("agent_spec") or ""), spec_limit)

    if not out["error_text"].strip():
        out["error_text"] = "<Exception> (empty output)"
    if not out["fail_reason"].strip():
        out["fail_reason"] = "start_failed"
    return out


def format_startup_diagnostics_summary(
    diag: object,
    *,
    get_settings_fn: Optional[Callable[[], object]] = None,
    total_limit: Optional[int] = None,
) -> str:
    """Format startup diagnostics as a single-line bounded redacted JSON string."""
    base: dict = {k: ([] if k == "args" else (None if k == "rc" else "")) for k in _BASE_FIELDS}
    try:
        nd = normalize_startup_diagnostics(diag, get_settings_fn=get_settings_fn)
        base.update(nd) if isinstance(nd, dict) else base.__setitem__("error", truncate_text(safe_str(nd), 400))
    except Exception:
        if isinstance(diag, dict):
            base.update(diag)
        else:
            base["error"] = truncate_text(safe_str(diag), 400) or "format_error"

    enabled, patterns, repl, _, _, cfg_total = _resolve_diag_config(get_settings_fn)
    eff_total = cfg_total if cfg_total > 0 else DEFAULT_DIAGNOSTICS_TOTAL_LIMIT
    if total_limit is not None:
        tl = _safe_nonneg_int(total_limit, 0)
        if tl > 0:
            eff_total = min(eff_total, tl)

    try:
        s = json.dumps(base, ensure_ascii=False, sort_keys=True)
    except Exception:
        s = safe_str(base)
    if enabled:
        s = redact_text(s, patterns, repl)
    return truncate_text(s, eff_total)


def format_startup_failure_log_line(
    *,
    agent_type: str,
    event: str,
    attempt: Optional[int],
    retries: Optional[int],
    error: object,
    diag: Optional[dict] = None,
    attempts: object = None,
    get_settings_fn: Optional[Callable[[], object]] = None,
) -> str:
    """Format a single-line redacted startup failure log message."""
    del attempts  # legacy arg kept for caller compatibility
    at = (agent_type or "").upper()
    ev = safe_str(event).strip() or "startup_failed"
    try:
        err_type = type(error).__name__
    except Exception:
        err_type = "Exception"
    try:
        err_repr = repr(error).strip() or f"<{err_type}>"
    except Exception:
        err_repr = f"<{err_type}>"

    enabled, patterns, repl, _, lim, _ = _resolve_diag_config(get_settings_fn)
    if enabled:
        err_repr = redact_text(err_repr, patterns, repl)
    if lim > 0:
        err_repr = truncate_text(err_repr, lim) or f"<{err_type}>"

    try:
        d = normalize_startup_diagnostics(
            diag if diag is not None else {"error_type": err_type, "error": safe_str(error), "fail_reason": ""},
            get_settings_fn=get_settings_fn,
        )
    except Exception:
        d = {"fail_reason": "start_failed", "error_text": f"<{err_type}> (empty output)"}

    has_diag = any([d.get("cmd"), d.get("args"), d.get("rc") is not None, d.get("stdout_snippet"), d.get("stderr_snippet"), d.get("spec")])
    diagnostics_summary = format_startup_diagnostics_summary(d, get_settings_fn=get_settings_fn) if has_diag else ""

    msg = f"[ACP:{at}] {ev}"
    if attempt is not None and retries is not None:
        try:
            msg += f" (attempt={int(attempt)}/{int(retries)})"
        except Exception:
            pass
    fr = safe_str(d.get("fail_reason", "")).strip() or "start_failed"
    et = safe_str(d.get("error_text", "")).strip() or f"<{err_type}> (empty output)"
    msg += f": fail_reason={fr} error_text={et} err_type={err_type} err_repr={err_repr}"
    if diagnostics_summary:
        msg += f" diagnostics_summary={diagnostics_summary}"
    return msg
