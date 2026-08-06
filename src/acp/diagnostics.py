"""ACP diagnostics helpers (redaction + truncation).

This module is intentionally dependency-light and must NOT import
`src.acp.sync_adapter` to avoid circular imports.

Public API in this module is intentionally stable and does not use leading
underscores (e.g. `get_diagnostics_config`, `redact_text`, `truncate_text`).

Compatibility note:
- Older call sites may still import private helpers (``_safe_str`` etc.).
  Those helpers are kept as thin wrappers to reduce refactor risk.

Callers may inject a `get_settings_fn` for testing.
"""

from __future__ import annotations

import logging
import numbers
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Optional

from ..config import get_settings as _default_get_settings
from ..utils.errors import get_error_detail

logger = logging.getLogger(__name__)

DEFAULT_DIAGNOSTICS_ARGS_LIMIT = 600
DEFAULT_DIAGNOSTICS_SNIPPET_LIMIT = 240
DEFAULT_DIAGNOSTICS_TOTAL_LIMIT = 2000

__all__ = [
    # Public API (stable)
    "DiagnosticsConfig",
    "DEFAULT_DIAGNOSTICS_ARGS_LIMIT",
    "DEFAULT_DIAGNOSTICS_SNIPPET_LIMIT",
    "DEFAULT_DIAGNOSTICS_TOTAL_LIMIT",
    "get_diagnostics_config",
    "safe_str",
    "truncate_text",
    "truncate_args",
    "safe_extract",
    "redact_text",
    "format_attempts_summary",
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


@dataclass(frozen=True)
class DiagnosticsConfig:
    """Diagnostics redaction + truncation config (best-effort).

    Contract:
    - Must never raise during construction from settings
    - Limits are non-negative ints
    """

    redact_enabled: bool = True
    redact_patterns: list[str] = field(default_factory=lambda: list(_DEFAULT_REDACT_PATTERNS))
    redact_replacement: str = "***REDACTED***"
    args_limit: int = DEFAULT_DIAGNOSTICS_ARGS_LIMIT
    snippet_limit: int = DEFAULT_DIAGNOSTICS_SNIPPET_LIMIT
    total_limit: int = DEFAULT_DIAGNOSTICS_TOTAL_LIMIT


def get_diagnostics_config(*, get_settings_fn: Optional[Callable[[], object]] = None) -> DiagnosticsConfig:
    """Read diagnostics config from settings (best-effort).

    - `get_settings_fn` is injectable for tests (avoid touching global config).
    - Never raises.
    """
    enabled = True
    patterns = list(_DEFAULT_REDACT_PATTERNS)
    repl = "***REDACTED***"
    args_limit = DEFAULT_DIAGNOSTICS_ARGS_LIMIT
    snippet_limit = DEFAULT_DIAGNOSTICS_SNIPPET_LIMIT
    total_limit = DEFAULT_DIAGNOSTICS_TOTAL_LIMIT

    try:
        getter = get_settings_fn or _default_get_settings
        s = getter()
        enabled = _safe_bool(getattr(s, "acp_diagnostics_redact_enabled", True), True)
        patterns = _safe_str_list(getattr(s, "acp_diagnostics_redact_patterns", None), _DEFAULT_REDACT_PATTERNS)
        repl = _safe_str(getattr(s, "acp_diagnostics_redact_replacement", repl) or repl) or repl
        args_limit = _safe_int(getattr(s, "acp_diagnostics_args_limit", args_limit), args_limit)
        snippet_limit = _safe_int(getattr(s, "acp_diagnostics_snippet_limit", snippet_limit), snippet_limit)
        total_limit = _safe_int(getattr(s, "acp_diagnostics_total_limit", total_limit), total_limit)
    except Exception:
        logger.debug("diagnostics config resolution failed, using defaults", exc_info=True)

    try:
        args_limit = int(args_limit)
    except Exception:
        logger.warning("Error while converting args limit to int", exc_info=True)
        args_limit = DEFAULT_DIAGNOSTICS_ARGS_LIMIT
    try:
        snippet_limit = int(snippet_limit)
    except Exception:
        logger.warning("Error while converting snippet limit to int", exc_info=True)
        snippet_limit = DEFAULT_DIAGNOSTICS_SNIPPET_LIMIT
    try:
        total_limit = int(total_limit)
    except Exception:
        logger.warning("Error while converting total limit to int", exc_info=True)
        total_limit = DEFAULT_DIAGNOSTICS_TOTAL_LIMIT

    if args_limit < 0:
        args_limit = DEFAULT_DIAGNOSTICS_ARGS_LIMIT
    if snippet_limit < 0:
        snippet_limit = DEFAULT_DIAGNOSTICS_SNIPPET_LIMIT
    if total_limit < 0:
        total_limit = DEFAULT_DIAGNOSTICS_TOTAL_LIMIT

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
    """Compile redaction regex patterns with caching (best-effort).

    Keyed by patterns only (replacement does not affect compilation).
    """
    out: list[re.Pattern] = []
    for p in patterns or ():
        try:
            if not p:
                continue
            out.append(re.compile(p))
        except Exception:
            logger.warning("Error while compiling redaction pattern", exc_info=True)
            continue
    return tuple(out)


def _safe_str(x: object) -> str:
    """兼容层：请改用公共 API `safe_str()`。

    移除条件：仓内不再存在对本模块 `_safe_str` 的跨模块导入，且连续回归稳定后可删除。
    """
    if isinstance(x, Exception):
        return get_error_detail(x)
    try:
        return str(x) if x is not None else ""
    except Exception:
        logger.warning("Error while converting to str in safe_str", exc_info=True)
        try:
            return repr(x)
        except Exception:
            logger.warning("Error while converting to repr in safe_str", exc_info=True)
            return ""


def _safe_bool(value: object, default: bool) -> bool:
    try:
        return value if isinstance(value, bool) else default
    except Exception:
        logger.warning("Error while converting to bool in safe_bool", exc_info=True)
        return default


def _safe_int(value: object, default: int) -> int:
    """Best-effort int parsing with MagicMock-safe semantics.

    说明：测试里经常用 `MagicMock()` 作为 settings。
    `int(MagicMock()) == 1` 会导致诊断截断上限意外变成 1，从而破坏日志/单测语义。
    这里仅接受明确的数值/字符串输入；其他类型一律回退到 default。
    """
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return default
        if isinstance(value, int):
            return value if value >= 0 else default
        if isinstance(value, numbers.Integral):
            v = int(value)
            return v if v >= 0 else default
        if isinstance(value, float):
            v = int(value)
            return v if v >= 0 else default
        if isinstance(value, str):
            v = int(value.strip() or "0")
            return v if v >= 0 else default
        return default
    except Exception:
        logger.warning("Error while converting to int in safe_int", exc_info=True)
        return default


def _safe_str_list(value: object, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for x in value:
            s = _safe_str(x).strip()
            if s:
                out.append(s)
        return out or list(default)
    if isinstance(value, str):
        raw = value.replace(",", "\n")
        out = [x.strip() for x in raw.splitlines() if x.strip()]
        return out or list(default)
    return list(default)


def _truncate_args(args: list[str], limit: int) -> list[str]:
    """兼容层：请改用公共 API `truncate_args()`。

    移除条件：仓内不再存在对本模块 `_truncate_args` 的跨模块导入，且连续回归稳定后可删除。
    """
    try:
        limit = int(limit or 0)
    except Exception:
        logger.warning("Error while converting limit to int in truncate_args", exc_info=True)
        limit = 0
    if limit <= 0:
        return list(args or [])
    out: list[str] = []
    used = 0
    for a in args or []:
        a = _safe_str(a)
        if not a:
            continue
        extra = (1 if out else 0) + len(a)
        if used + extra <= limit:
            out.append(a)
            used += extra
            continue
        if out:
            out.append("…(truncated)")
        else:
            out.append(truncate_text(a, limit))
        break
    return out


# ---------------------------------------------------------------------------
# normalize_startup_diagnostics pipeline helpers
# ---------------------------------------------------------------------------


def _resolve_diag_config(
    get_settings_fn: Optional[Callable[[], object]],
) -> tuple[bool, list[str], str, int, int, int]:
    try:
        cfg = get_diagnostics_config(get_settings_fn=get_settings_fn)
        enabled = bool(cfg.redact_enabled)
        patterns = list(cfg.redact_patterns or [])
        repl = _safe_str(cfg.redact_replacement or "***REDACTED***") or "***REDACTED***"
        args_limit = int(cfg.args_limit or 0) if int(cfg.args_limit or 0) > 0 else DEFAULT_DIAGNOSTICS_ARGS_LIMIT
        snippet_limit = int(cfg.snippet_limit or 0) if int(cfg.snippet_limit or 0) > 0 else DEFAULT_DIAGNOSTICS_SNIPPET_LIMIT
        total_limit = int(cfg.total_limit or 0) if int(cfg.total_limit or 0) > 0 else DEFAULT_DIAGNOSTICS_TOTAL_LIMIT
    except Exception:
        logger.warning("Error while resolving diagnostics config", exc_info=True)
        enabled, patterns, repl, args_limit, snippet_limit, total_limit = (
            True,
            list(_DEFAULT_REDACT_PATTERNS),
            "***REDACTED***",
            DEFAULT_DIAGNOSTICS_ARGS_LIMIT,
            DEFAULT_DIAGNOSTICS_SNIPPET_LIMIT,
            DEFAULT_DIAGNOSTICS_TOTAL_LIMIT,
        )
    return enabled, patterns, repl, args_limit, snippet_limit, total_limit


def _init_diag_container(diag: object) -> dict:
    out: dict = {
        "cmd": "",
        "args": [],
        "rc": None,
        "stdout_snippet": "",
        "stderr_snippet": "",
        "fail_reason": "",
        "error_text": "",
        "spec": "",
        "error": "",
    }
    try:
        if isinstance(diag, dict):
            out.update(diag)
        elif diag is None:
            out["error"] = "(empty)"
        else:
            out["error"] = truncate_text(_safe_str(diag), 400)
    except Exception:
        logger.warning("Error while initializing diagnostics container", exc_info=True)
        out["error"] = "format_error"
    return out


def _normalize_fields(out: dict) -> None:
    try:
        xs = out.get("args")
        if xs is None:
            xs = []
        if not isinstance(xs, list):
            xs = list(xs or [])
        out["args"] = [str(x) for x in (xs or [])]
    except Exception:
        logger.warning("Error while normalizing args field", exc_info=True)
        out["args"] = []

    out["cmd"] = _safe_str(out.get("cmd") or "")

    try:
        rc = out.get("rc")
        if rc is None or rc == "":
            out["rc"] = None
        else:
            out["rc"] = int(rc)
    except Exception:
        logger.warning("Error while normalizing rc field", exc_info=True)
        out["rc"] = None

    out["stdout_snippet"] = _safe_str(out.get("stdout_snippet") or "")
    out["stderr_snippet"] = _safe_str(out.get("stderr_snippet") or "")
    out["spec"] = _safe_str(out.get("spec") or "")
    if out.get("agent_spec") is not None:
        try:
            out["agent_spec"] = _safe_str(out.get("agent_spec") or "")
        except Exception:
            logger.warning("Error while normalizing agent_spec field", exc_info=True)
            out["agent_spec"] = ""
    out["error"] = _safe_str(out.get("error") or "")
    out["error_text"] = _safe_str(out.get("error_text") or "")
    out["fail_reason"] = _safe_str(out.get("fail_reason") or "")


def _apply_fallbacks(out: dict) -> None:
    fr = (out.get("fail_reason") or "").strip()
    if not fr:
        try:
            fr = _safe_str(out.get("fail_phase") or "").strip()
        except Exception:
            logger.warning("Error while applying fail_reason fallback", exc_info=True)
            fr = ""
    out["fail_reason"] = fr or "start_failed"

    et = (out.get("error_text") or "").strip()
    if not et:
        et = (out.get("stderr_snippet") or "").strip()
    if not et:
        et = (out.get("error") or "").strip()
    if not et:
        etype = _safe_str(out.get("error_type") or "Exception") or "Exception"
        et = f"<{etype}> (empty output)"
    out["error_text"] = et
    if not (out.get("error") or "").strip():
        out["error"] = out["error_text"]


def _apply_redaction(out: dict, enabled: bool, patterns: list[str], repl: str) -> None:
    if not enabled:
        return
    try:
        out["cmd"] = redact_text(out.get("cmd") or "", patterns, repl)
        out["stdout_snippet"] = redact_text(out.get("stdout_snippet") or "", patterns, repl)
        out["stderr_snippet"] = redact_text(out.get("stderr_snippet") or "", patterns, repl)
        out["spec"] = redact_text(out.get("spec") or "", patterns, repl)
        out["error"] = redact_text(out.get("error") or "", patterns, repl)
        out["error_text"] = redact_text(out.get("error_text") or "", patterns, repl)
        out["fail_reason"] = redact_text(out.get("fail_reason") or "", patterns, repl)
        if "agent_spec" in out:
            out["agent_spec"] = redact_text(out.get("agent_spec") or "", patterns, repl)
        out["args"] = [redact_text(str(x), patterns, repl) for x in (out.get("args") or [])]
    except Exception:
        logger.debug("redaction pass failed", exc_info=True)


def _apply_truncation(out: dict, args_limit: int, snippet_limit: int, total_limit: int) -> None:
    try:
        out["args"] = truncate_args([str(x) for x in (out.get("args") or [])], args_limit)
    except Exception:
        logger.warning("Error while truncating args", exc_info=True)
        out["args"] = [str(x) for x in (out.get("args") or [])]

    try:
        out["cmd"] = truncate_text(_safe_str(out.get("cmd") or ""), snippet_limit)
    except Exception:
        logger.warning("Error while truncating cmd field", exc_info=True)
        out["cmd"] = _safe_str(out.get("cmd") or "")
    try:
        out["stdout_snippet"] = truncate_text(_safe_str(out.get("stdout_snippet") or ""), snippet_limit)
        out["stderr_snippet"] = truncate_text(_safe_str(out.get("stderr_snippet") or ""), snippet_limit)
    except Exception:
        logger.debug("snippet truncation failed", exc_info=True)
    try:
        out["error_text"] = (
            truncate_text(
                _safe_str(out.get("error_text") or ""), min(400, snippet_limit) if snippet_limit > 0 else DEFAULT_DIAGNOSTICS_SNIPPET_LIMIT
            )
            or "(empty)"
        )
    except Exception:
        logger.warning("Error while truncating error_text field", exc_info=True)
        out["error_text"] = _safe_str(out.get("error_text") or "") or "(empty)"
    try:
        out["error"] = (
            truncate_text(_safe_str(out.get("error") or ""), min(400, snippet_limit) if snippet_limit > 0 else DEFAULT_DIAGNOSTICS_SNIPPET_LIMIT)
            or out["error_text"]
        )
    except Exception:
        logger.warning("Error while truncating error field", exc_info=True)
        out["error"] = _safe_str(out.get("error") or "") or out.get("error_text") or "(empty)"
    try:
        out["fail_reason"] = truncate_text(_safe_str(out.get("fail_reason") or ""), 80) or "start_failed"
    except Exception:
        logger.warning("Error while truncating fail_reason field", exc_info=True)
        out["fail_reason"] = _safe_str(out.get("fail_reason") or "") or "start_failed"
    try:
        out["spec"] = truncate_text(
            _safe_str(out.get("spec") or ""), min(400, total_limit) if total_limit > 0 else 400
        )
    except Exception:
        logger.warning("Error while truncating spec field", exc_info=True)
        out["spec"] = _safe_str(out.get("spec") or "")
    try:
        if "agent_spec" in out:
            out["agent_spec"] = truncate_text(
                _safe_str(out.get("agent_spec") or ""), min(400, total_limit) if total_limit > 0 else 400
            )
    except Exception:
        logger.debug("agent_spec truncation failed", exc_info=True)


def _final_guard(out: dict) -> None:
    for k in ("cmd", "args", "rc", "stdout_snippet", "stderr_snippet", "fail_reason", "error_text", "spec", "error"):
        if k not in out:
            out[k] = "" if k != "args" else []
    if not (_safe_str(out.get("error_text") or "").strip()):
        out["error_text"] = "<Exception> (empty output)"
    if not (_safe_str(out.get("fail_reason") or "").strip()):
        out["fail_reason"] = "start_failed"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_startup_diagnostics(
    diag: object,
    *,
    get_settings_fn: Optional[Callable[[], object]] = None,
) -> dict:
    """Normalize startup diagnostics into a stable, serializable dict (SSOT).

    目标：把"启动失败诊断字段契约"集中在本模块：
    - 稳定字段：cmd/args/rc/stdout_snippet/stderr_snippet/fail_reason/error_text/spec/error
    - 兼容字段：agent_spec（若存在则保留）
    - 兜底：error_text 必须非空；fail_reason 必须非空
    - 安全：脱敏 + 截断严格遵循 diagnostics 配置（best-effort，永不抛异常）

    注意：本函数不应引入对 `src.acp.sync_adapter` 的依赖，避免循环依赖。
    """
    enabled, patterns, repl, args_limit, snippet_limit, total_limit = _resolve_diag_config(get_settings_fn)
    out = _init_diag_container(diag)
    _normalize_fields(out)
    _apply_fallbacks(out)
    _apply_redaction(out, enabled, patterns, repl)
    _apply_truncation(out, args_limit, snippet_limit, total_limit)
    _final_guard(out)
    return out


def safe_str(x: object) -> str:
    """Public wrapper for safe string conversion (never raises)."""
    return _safe_str(x)


def truncate_text(text: str, limit: int) -> str:
    """Public API: bounded text truncation (never raises)."""
    try:
        limit = int(limit or 0)
    except Exception:
        logger.warning("Error while converting limit to int in truncate_text", exc_info=True)
        limit = 0
    if limit <= 0:
        return ""
    s = _safe_str(text)
    if not s:
        return ""
    if len(s) <= limit:
        return s
    return s[:limit] + "…(truncated)"


def truncate_args(args: list[str], limit: int) -> list[str]:
    """Public wrapper for args truncation (never raises)."""
    return _truncate_args(args, limit)


def safe_extract(fn: Callable[[], object], *, default: object = None, log_msg: str = "diagnostics extraction failed") -> object:
    """Run a diagnostics extractor without letting optional data collection fail startup.

    This is intentionally small and dependency-light: diagnostics collection is
    best-effort, so callers get a stable default plus a debug breadcrumb.
    """
    try:
        return fn()
    except Exception:
        logger.debug(log_msg, exc_info=True)
        return default


def redact_text(text: str, patterns: list[str], replacement: str) -> str:
    """Regex-based redaction with compiled-pattern cache (best-effort).

    Contract:
    - Never raises
    - Preserves pattern order semantics
    - Ignores invalid patterns (compile/sub errors)
    """
    s = _safe_str(text)
    if not s:
        return ""
    rep = _safe_str(replacement) or "***REDACTED***"
    pats = tuple([_safe_str(p) for p in (patterns or []) if _safe_str(p)])
    if not pats:
        return s

    try:
        compiled = _compile_redaction_patterns(pats)
    except Exception:
        logger.warning("Error while compiling redaction patterns in redact_text", exc_info=True)
        compiled = ()

    for cre in compiled:
        try:
            s = cre.sub(rep, s)
        except Exception:
            logger.warning("Error while applying redaction pattern", exc_info=True)
            continue
    return s


def format_attempts_summary(
    attempts: object,
    *,
    per_item_limit: Optional[int] = None,
    total_limit: Optional[int] = None,
    get_settings_fn: Optional[Callable[[], object]] = None,
) -> str:
    """将 startup attempts 摘要化为稳定、可 grep 的单行字符串（JSON）。

    目标：为上层日志提供 SSOT 的 attempts 摘要输出，避免在 manager 中重复实现脱敏/截断。

    Contract:
    - best-effort：永不抛异常；任何输入都返回 str
    - 稳定字段：fail_phase/decision/error_type/error_blob（缺失则兜底补齐）
    - 安全：脱敏/截断严格遵循 diagnostics 配置（可通过参数进一步收紧，但不得放宽配置上限）
    """

    try:
        cfg = get_diagnostics_config(get_settings_fn=get_settings_fn)
        enabled = bool(cfg.redact_enabled)
        patterns = list(cfg.redact_patterns or [])
        repl = _safe_str(cfg.redact_replacement or "***REDACTED***") or "***REDACTED***"
        cfg_snip = int(cfg.snippet_limit or 0)
        cfg_total = int(cfg.total_limit or 0)
    except Exception:
        logger.warning("Error while getting diagnostics config in format_attempts_summary", exc_info=True)
        enabled, patterns, repl, cfg_snip, cfg_total = (
            True,
            list(_DEFAULT_REDACT_PATTERNS),
            "***REDACTED***",
            DEFAULT_DIAGNOSTICS_SNIPPET_LIMIT,
            DEFAULT_DIAGNOSTICS_TOTAL_LIMIT,
        )

    try:
        per_lim = int(per_item_limit) if per_item_limit is not None else int(cfg_snip or 0)
    except Exception:
        logger.warning("Error while converting per_item_limit to int", exc_info=True)
        per_lim = int(cfg_snip or 0)
    try:
        total_lim = int(total_limit) if total_limit is not None else int(cfg_total or 0)
    except Exception:
        logger.warning("Error while converting total_limit to int", exc_info=True)
        total_lim = int(cfg_total or 0)

    if cfg_snip and per_lim:
        per_lim = min(int(cfg_snip), int(per_lim))
    elif cfg_snip:
        per_lim = int(cfg_snip)
    if per_lim <= 0:
        per_lim = DEFAULT_DIAGNOSTICS_SNIPPET_LIMIT

    if cfg_total and total_lim:
        total_lim = min(int(cfg_total), int(total_lim))
    elif cfg_total:
        total_lim = int(cfg_total)
    if total_lim <= 0:
        total_lim = DEFAULT_DIAGNOSTICS_TOTAL_LIMIT

    try:
        items = list(attempts or []) if isinstance(attempts, (list, tuple)) else []
    except Exception:
        logger.warning("Error while parsing attempts list", exc_info=True)
        items = []
    if not items:
        return ""

    out: list[dict] = []
    for a in items[:8]:
        d = a if isinstance(a, dict) else {}

        phase = _safe_str(d.get("phase", ""))
        ok = d.get("ok") if "ok" in d else None
        fail_phase = _safe_str(d.get("fail_phase", ""))
        decision = _safe_str(d.get("decision", ""))
        error_type = _safe_str(d.get("error_type", ""))

        blob = _safe_str(d.get("error_blob", ""))
        if not blob:
            blob = _safe_str(d.get("error", ""))
        if not blob:
            blob = _safe_str(d.get("stderr_snippet", ""))
        if not blob:
            blob = _safe_str(d.get("stdout_snippet", ""))
        if not blob:
            blob = _safe_str(d.get("stderr", ""))
        if not blob:
            blob = _safe_str(d.get("stdout", ""))
        blob = blob or "(empty)"
        blob = truncate_text(blob, per_lim) or "(empty)"

        item = {
            "phase": phase,
            "ok": ok,
            "fail_phase": fail_phase,
            "decision": decision,
            "error_type": error_type,
            "error_blob": blob,
        }
        out.append(item)

    try:
        import json

        s = json.dumps(out, ensure_ascii=False, sort_keys=True)
    except Exception:
        logger.warning("Error while serializing attempts to JSON", exc_info=True)
        s = _safe_str(out)

    if enabled:
        try:
            s = redact_text(s, patterns, repl)
        except Exception:
            logger.debug("redaction pass failed in format_attempts_summary", exc_info=True)
    return truncate_text(s, total_lim)


def format_startup_diagnostics_summary(
    diag: object,
    *,
    get_settings_fn: Optional[Callable[[], object]] = None,
    total_limit: Optional[int] = None,
) -> str:
    """将 startup diagnostics 格式化为稳定单行摘要（JSON）。

    约束：
    - best-effort：永不抛异常
    - 安全：脱敏 + 截断遵循 diagnostics 配置（参数仅允许收紧 total_limit）
    """
    base = {
        "cmd": "",
        "args": [],
        "rc": None,
        "stdout_snippet": "",
        "stderr_snippet": "",
        "spec": "",
        "fail_reason": "",
        "error_text": "",
        "error": "",
    }

    try:
        nd = normalize_startup_diagnostics(diag, get_settings_fn=get_settings_fn)
        if isinstance(nd, dict):
            base.update(nd)
        else:
            base["error"] = truncate_text(_safe_str(nd), 400)
    except Exception:
        logger.warning("Error while normalizing startup diagnostics in summary", exc_info=True)
        try:
            if isinstance(diag, dict):
                base.update(diag)
            else:
                base["error"] = truncate_text(_safe_str(diag), 400)
        except Exception:
            logger.warning("Error while applying diagnostic fallback in summary", exc_info=True)
            base["error"] = "format_error"

    try:
        cfg = get_diagnostics_config(get_settings_fn=get_settings_fn)
        enabled = bool(cfg.redact_enabled)
        patterns = list(cfg.redact_patterns or [])
        repl = str(cfg.redact_replacement or "***REDACTED***")
        cfg_total = int(cfg.total_limit or 0)
    except Exception:
        logger.warning("Error while getting diagnostics config in summary", exc_info=True)
        enabled, patterns, repl, cfg_total = True, list(_DEFAULT_REDACT_PATTERNS), "***REDACTED***", DEFAULT_DIAGNOSTICS_TOTAL_LIMIT

    eff_total = cfg_total if cfg_total > 0 else DEFAULT_DIAGNOSTICS_TOTAL_LIMIT
    if total_limit is not None:
        try:
            tl = int(total_limit)
            if tl > 0:
                eff_total = min(eff_total, tl)
        except Exception:
            logger.debug("total_limit parsing failed", exc_info=True)

    try:
        import json

        s = json.dumps(base, ensure_ascii=False, sort_keys=True)
    except Exception:
        logger.warning("Error while serializing diagnostics summary to JSON", exc_info=True)
        s = _safe_str(base)

    if enabled:
        try:
            s = redact_text(s, patterns, repl)
        except Exception:
            logger.debug("redaction pass failed in normalize_startup_diagnostics", exc_info=True)
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
    """统一启动失败日志单行格式化（SSOT）。

    输出字段（稳定）：
    - err_type, err_repr（永不为空）
    - fail_reason, error_text（永不为空；由 normalize_startup_diagnostics 兜底）
    - diagnostics_summary（可选：仅当 diag 中存在有效字段）
    - attempts_summary（可选：attempts 非空或 diag.attempts 非空）
    """

    at = (agent_type or "").upper()
    ev = _safe_str(event).strip() or "startup_failed"

    a = attempt
    r = retries
    try:
        a = int(a) if a is not None else None
    except Exception:
        logger.warning("Error while converting attempt to int", exc_info=True)
        a = None
    try:
        r = int(r) if r is not None else None
    except Exception:
        logger.warning("Error while converting retries to int", exc_info=True)
        r = None

    try:
        err_type = type(error).__name__
    except Exception:
        logger.warning("Error while getting error type", exc_info=True)
        err_type = "Exception"
    try:
        err_repr = repr(error)
    except Exception:
        logger.warning("Error while getting error repr", exc_info=True)
        err_repr = ""
    if not (err_repr or "").strip():
        err_repr = f"<{err_type}>"

    try:
        cfg = get_diagnostics_config(get_settings_fn=get_settings_fn)
        enabled = bool(cfg.redact_enabled)
        patterns = list(cfg.redact_patterns or [])
        repl = str(cfg.redact_replacement or "***REDACTED***")
        lim = int(cfg.snippet_limit or 0)
    except Exception:
        logger.warning("Error while getting diagnostics config in log line", exc_info=True)
        enabled, patterns, repl, lim = True, list(_DEFAULT_REDACT_PATTERNS), "***REDACTED***", DEFAULT_DIAGNOSTICS_SNIPPET_LIMIT

    if enabled:
        try:
            err_repr = redact_text(err_repr, patterns, repl)
        except Exception:
            logger.debug("redaction pass failed in format_startup_failure_log_line", exc_info=True)
    if lim > 0:
        err_repr = truncate_text(err_repr, lim) or f"<{err_type}>"

    diagnostics_summary = ""
    d: dict = {}
    try:
        if diag is not None:
            d = normalize_startup_diagnostics(diag, get_settings_fn=get_settings_fn)
        else:
            d = normalize_startup_diagnostics(
                {
                    "error_type": err_type,
                    "error": _safe_str(error),
                    "fail_reason": "",
                },
                get_settings_fn=get_settings_fn,
            )
    except Exception:
        logger.warning("Error while normalizing diagnostics in log line", exc_info=True)
        d = {
            "fail_reason": "start_failed",
            "error_text": f"<{err_type}> (empty output)",
            "stderr_snippet": "",
            "stdout_snippet": "",
            "cmd": "",
            "args": [],
            "rc": None,
            "spec": "",
        }

    try:
        cmd = _safe_str(d.get("cmd", ""))
        args = list(d.get("args") or [])
        rc = d.get("rc")
        out_snip = _safe_str(d.get("stdout_snippet", ""))
        err_snip = _safe_str(d.get("stderr_snippet", ""))
        spec = _safe_str(d.get("spec", ""))
        has_diag = bool(cmd or args or (rc is not None) or out_snip or err_snip or spec)
        if has_diag:
            diagnostics_summary = format_startup_diagnostics_summary(d, get_settings_fn=get_settings_fn)
    except Exception:
        logger.warning("Error while formatting diagnostics summary in log line", exc_info=True)
        diagnostics_summary = ""

    include_attempts_summary = False
    try:
        if attempts is None and isinstance(diag, dict):
            attempts = diag.get("attempts")
        include_attempts_summary = (
            attempts is not None
            or (isinstance(diag, dict) and ("attempts" in diag))
        )
    except Exception:
        logger.warning("Error while determining include_attempts_summary", exc_info=True)
        include_attempts_summary = False

    attempts_summary = ""
    if include_attempts_summary:
        try:
            if attempts:
                attempts_summary = format_attempts_summary(attempts, get_settings_fn=get_settings_fn)
        except Exception:
            logger.warning("Error while formatting attempts summary", exc_info=True)
            attempts_summary = ""
        if not attempts_summary:
            attempts_summary = "(empty)"

    msg = f"[ACP:{at}] {ev}"
    if a is not None and r is not None:
        msg += f" (attempt={a}/{r})"
    fr = _safe_str(d.get("fail_reason", "")).strip() or "start_failed"
    et = _safe_str(d.get("error_text", "")).strip() or f"<{err_type}> (empty output)"

    msg += f": fail_reason={fr} error_text={et} err_type={err_type} err_repr={err_repr}"
    if diagnostics_summary:
        msg += f" diagnostics_summary={diagnostics_summary}"
    if include_attempts_summary:
        msg += f" attempts_summary={attempts_summary}"
    return msg
