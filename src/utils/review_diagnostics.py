"""Reusable review exception diagnostics — shared by SpecEngine.

Extracted from ``src/spec_engine/review.py`` for structured diagnostics
without importing spec_engine internals.
"""

from __future__ import annotations

import logging
import traceback

from .errors import _has_timeout_in_chain  # unified implementation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stable / compat key tuples (re-exported for consumers)
# ---------------------------------------------------------------------------

REVIEW_DIAG_STABLE_KEYS = (
    "phase",
    "role",
    "cycle",
    "decision",
    "fail_reason",
    "err_type",
    "err_repr",
    "error_text",
    "traceback_snippet",
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_str(x: object) -> str:
    """Safe str() that never raises."""
    try:
        return str(x)
    except Exception:
        return ""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_review_exception_diagnostics(
    e: Exception,
    *,
    cycle: int,
    project_name: str = "",
    chat_id: str = "",
    root_path: str = "",
    agent_type: str = "",
    session_id: str = "",
    get_settings_fn=None,
) -> dict:
    """Build a structured diagnostics dict from a review exception.

    Parameters
    ----------
    e : Exception
        The caught exception.
    cycle : int
        Current iteration / cycle number.
    project_name, chat_id, root_path, agent_type, session_id :
        Optional context fields attached to the diagnostics dict.
    get_settings_fn : callable, optional
        A zero-arg callable returning the settings object (used for
        diagnostics config).  Falls back to ``config.get_settings``.
    """
    try:
        from ..config import get_settings
    except Exception:
        get_settings = None  # type: ignore[assignment]

    try:
        from ..acp.diagnostics import get_diagnostics_config, redact_text

        cfg = get_diagnostics_config(get_settings_fn=get_settings_fn or get_settings)
        redact_enabled = cfg.redact_enabled
        redact_patterns = cfg.redact_patterns or []
        redact_repl = cfg.redact_replacement or "***REDACTED***"
        cfg_snip = cfg.snippet_limit
        cfg_total = cfg.total_limit
    except Exception:
        redact_text = None  # type: ignore[assignment]
        redact_enabled, redact_patterns, redact_repl = True, [], "***REDACTED***"
        cfg_snip, cfg_total = 240, 2000

    # -- truncation helpers (closure over cfg) --

    def _truncate_strict(s: str, lim: int) -> str:
        try:
            lim = int(lim or 0)
        except Exception:
            lim = 0
        if lim <= 0:
            return ""
        ss = _safe_str(s)
        if not ss:
            return ""
        if len(ss) <= lim:
            return ss
        suffix = "…(truncated)"
        if lim <= len(suffix):
            return ss[:lim]
        return ss[: max(0, lim - len(suffix))] + suffix

    def _redact_and_truncate(text: str, *, hard_limit: int, cfg_limit: int) -> str:
        lim = hard_limit
        try:
            lim = int(hard_limit or 0)
        except Exception:
            lim = 0
        lim = max(1, lim)
        try:
            cfg_lim = int(cfg_limit or 0)
        except Exception:
            cfg_lim = 0
        if cfg_lim > 0:
            lim = min(lim, cfg_lim)

        s = _safe_str(text)
        if redact_enabled and callable(redact_text):
            try:
                s = redact_text(s, redact_patterns, redact_repl)  # type: ignore[misc]
            except Exception:
                logger.debug("Failed to apply redact_text in review diagnostics", exc_info=True)
        return _truncate_strict(s, lim)

    # -- error extraction helpers --

    def _extract_error_text(err: Exception) -> str:
        base = (_safe_str(err) or "").strip()
        if base:
            return base
        for k in ("stderr_snippet", "stdout_snippet", "stderr", "stdout", "message", "detail"):
            try:
                v = (_safe_str(getattr(err, k, "")) or "").strip()
                if v:
                    return v
            except Exception:
                continue
        return ""

    def _infer_fail_reason(err: Exception) -> str:
        et = "Exception"
        try:
            et = type(err).__name__
        except Exception:
            et = "Exception"
        if isinstance(err, TimeoutError):
            return "timeout"
        if et in ("TimeoutExpired", "ReadTimeout", "ConnectTimeout"):
            return "timeout"
        # Traverse exception chain (__cause__ / __context__) for wrapped TimeoutError
        if _has_timeout_in_chain(err):
            return "timeout"
        if et in ("JSONDecodeError",):
            return "parse_json"
        if et in ("ValueError", "TypeError"):
            return "parse_error"
        return "exception"

    def _extract_err_repr(err: Exception) -> str:
        err_type = "Exception"
        try:
            err_type = type(err).__name__
        except Exception:
            err_type = "Exception"
        try:
            s = repr(err)
        except Exception:
            s = ""
        s = (_safe_str(s) or "").strip()
        if not s:
            s = f"<{err_type}>"
        return s

    # -- assemble diagnostics --

    err_repr = _extract_err_repr(e)
    err_type = "Exception"
    try:
        err_type = type(e).__name__
    except Exception:
        err_type = "Exception"

    error_text = _extract_error_text(e)
    fail_reason = _infer_fail_reason(e)

    if not (error_text or "").strip():
        if fail_reason == "timeout":
            error_text = "审查超时，将在下一轮重试"
        else:
            error_text = "审查执行异常，请检查服务状态"

    tb = ""
    try:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    except Exception:
        tb = ""
    tb = (tb or "").strip()

    err_repr_rt = _redact_and_truncate(err_repr, hard_limit=600, cfg_limit=cfg_snip)
    if not (err_repr_rt or "").strip():
        err_repr_rt = f"<{err_type}>"
    error_text_rt = _redact_and_truncate(error_text, hard_limit=600, cfg_limit=cfg_snip)
    if not (error_text_rt or "").strip():
        error_text_rt = err_repr_rt

    diag = {
        "phase": "review",
        "role": "multi_perspective",
        "cycle": int(cycle or 0),
        "decision": "review_failed_continue",
        "fail_reason": str(fail_reason or "exception"),
        "err_type": err_type,
        "err_repr": err_repr_rt,
        "error_text": error_text_rt,
        "cycle_number": int(cycle or 0),
        "exception_type": err_type,
        "review_role": "multi_perspective",
        "traceback_snippet": _redact_and_truncate(tb, hard_limit=1600, cfg_limit=cfg_total),
        "project": (project_name or "").strip(),
        "chat_id": (chat_id or ""),
        "root_path": (root_path or ""),
        "agent_type": (agent_type or ""),
    }
    diag["session_id"] = str(session_id or "")

    return diag


def normalize_review_diagnostics(diag: object, *, error_text_limit: int = 500) -> dict:
    """Normalize a diagnostics dict to stable-key form.

    Parameters
    ----------
    error_text_limit : int
        Maximum length for the ``error_text`` field (default 500).
        Longer values are truncated with a ``…(truncated)`` marker.
    """
    d = diag if isinstance(diag, dict) else {}

    def _s(x: object) -> str:
        try:
            return str(x) if x is not None else ""
        except Exception:
            try:
                return repr(x)
            except Exception:
                return ""

    phase = (_s(d.get("phase")) or "review").strip() or "review"
    role = (_s(d.get("role")) or _s(d.get("review_role")) or "multi_perspective").strip() or "multi_perspective"

    cycle_val: int = 0
    try:
        if "cycle" in d and d.get("cycle") is not None:
            cycle_val = int(d.get("cycle") or 0)
        else:
            cycle_val = int(d.get("cycle_number") or 0)
    except Exception:
        cycle_val = 0

    decision = (_s(d.get("decision")) or "review_failed_continue").strip() or "review_failed_continue"

    fail_reason = (_s(d.get("fail_reason")) or "").strip()
    if not fail_reason:
        fail_reason = "exception" if decision.startswith("review_failed") else ""

    err_type = (_s(d.get("err_type")) or _s(d.get("exception_type")) or "Exception").strip() or "Exception"

    err_repr = (_s(d.get("err_repr")) or "").strip()
    if not err_repr:
        err_repr = f"<{err_type}>"

    error_text = (_s(d.get("error_text")) or "").strip()
    if not error_text:
        error_text = err_repr

    # Truncate error_text to prevent oversized diagnostic cards
    _et_limit = max(1, int(error_text_limit or 500))
    if len(error_text) > _et_limit:
        _suffix = "…(truncated)"
        error_text = error_text[: max(1, _et_limit - len(_suffix))] + _suffix

    tb = (_s(d.get("traceback_snippet")) or "").strip()

    out = {
        "phase": phase,
        "role": role,
        "cycle": int(cycle_val),
        "decision": decision,
        "fail_reason": fail_reason,
        "err_type": err_type,
        "err_repr": err_repr,
        "error_text": error_text,
        "traceback_snippet": tb,
    }

    try:
        return {k: out.get(k) for k in REVIEW_DIAG_STABLE_KEYS}
    except Exception:
        return out
