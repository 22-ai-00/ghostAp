from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ErrorCategory(Enum):
    """Categories of workflow errors exposed to users."""

    # Unified error surface (four categories)
    SESSION_EXPIRED = "session_expired"
    INVALID_STATE = "invalid_state"
    INVALID_ARGUMENT = "invalid_argument"
    FORBIDDEN = "forbidden"

    # Legacy detailed categories (mapped to unified categories)
    AGENT_LIMIT = "agent_limit"
    TOOL_NOT_ALLOWED = "tool_not_allowed"
    SCRIPT_VALIDATION = "script_validation"
    RUNTIME_TIMEOUT = "runtime_timeout"
    REVIEW_FAILED = "review_failed"
    INTERNAL_ERROR = "internal_error"
    CANCELLED = "cancelled"

    # Deprecated: kept for backwards compatibility
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass
class WorkflowUserError:
    """A sanitized error safe to surface to end users."""

    category: ErrorCategory
    user_message: str
    internal_detail: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_USER_MESSAGES: dict[ErrorCategory, str] = {
    # Unified error surface (four categories)
    ErrorCategory.SESSION_EXPIRED: (
        "会话已过期，请重新发起 `/wf`。"
    ),
    ErrorCategory.INVALID_STATE: (
        "当前操作与 Workflow 状态不一致。\n\n💡 恢复方式：发送 `/wf_status` 查看最新状态后重试"
    ),
    ErrorCategory.INVALID_ARGUMENT: (
        "请求参数校验失败：{detail}\n\n💡 请检查输入后重试"
    ),
    ErrorCategory.FORBIDDEN: (
        "只有 Workflow 发起者或管理员可以执行此操作。\n\n💡 如需操作请联系发起者或管理员"
    ),
    # Legacy detailed categories
    ErrorCategory.AGENT_LIMIT: (
        "已达到最大 Agent 步数限制，建议拆分为更小的子任务。"
    ),
    ErrorCategory.TOOL_NOT_ALLOWED: (
        "请求的工具在当前工作流中不被允许，请检查工具配置。"
    ),
    ErrorCategory.SCRIPT_VALIDATION: (
        "生成的脚本未通过验证检查，请检查任务描述和约束条件。"
    ),
    ErrorCategory.RUNTIME_TIMEOUT: (
        "工作流等待响应超时，请重试或增加超时限制。"
    ),
    ErrorCategory.REVIEW_FAILED: (
        "一个或多个独立评审 Agent 未正常完成，Workflow 未发布不完整结果。"
    ),
    ErrorCategory.INTERNAL_ERROR: (
        "发生内部错误，请重试。如问题持续，请联系管理员。"
    ),
    ErrorCategory.CANCELLED: (
        "工作流已被用户取消。"
    ),
    # Deprecated: kept for backwards compatibility
    ErrorCategory.BUDGET_EXHAUSTED: (
        "Token 预算已耗尽，此功能已废弃，请联系管理员。"
    ),
}


# Mapping from legacy detailed categories to unified surface categories
_LEGACY_TO_UNIFIED: dict[ErrorCategory, ErrorCategory] = {
    ErrorCategory.AGENT_LIMIT: ErrorCategory.INVALID_STATE,
    ErrorCategory.TOOL_NOT_ALLOWED: ErrorCategory.FORBIDDEN,
    ErrorCategory.SCRIPT_VALIDATION: ErrorCategory.INVALID_ARGUMENT,
    ErrorCategory.RUNTIME_TIMEOUT: ErrorCategory.RUNTIME_TIMEOUT,
    ErrorCategory.INTERNAL_ERROR: ErrorCategory.INVALID_STATE,
    ErrorCategory.CANCELLED: ErrorCategory.INVALID_STATE,
    ErrorCategory.BUDGET_EXHAUSTED: ErrorCategory.INVALID_STATE,  # deprecated
}


def to_unified_category(category: ErrorCategory) -> ErrorCategory:
    """Map a legacy detailed category to the unified four-category surface."""
    return _LEGACY_TO_UNIFIED.get(category, category)

# Patterns considered internal-only and stripped from user-facing messages.
# These are intentionally conservative — we only strip things that clearly
# look like file paths / tracebacks, NOT user-facing slash commands (e.g.
# /wf, /wf_help) which share the leading "/" but contain underscores and
# short identifiers that aren't path-like.
_INTERNAL_PATTERNS: list[re.Pattern[str]] = [
    # Python traceback header / frame lines - match first since they contain
    # '"File /path/to/file.py, line N"' which our path regex would also hit.
    re.compile(r"Traceback \(most recent call last\):.*", re.DOTALL),
    re.compile(r'^\s*File ".+", line \d+.*$', re.MULTILINE),
    # Absolute file paths that look like source files - require at least two
    # path segments AND a trailing ".py:LINE" suffix (e.g. /src/foo.py:12).
    re.compile(r"/(?:[\w.\-]+/)+[\w.\-]+\.py(?::\d+)?"),
    # Generic absolute paths with at least 3 segments AND an extension like
    # .js/.json/.txt/.log/.md (e.g. /tmp/foo.log or /etc/hosts).
    re.compile(r"/(?:[\w.\-]+/){2,}[\w.\-]+(?:\.[a-z]{2,5})?"),
    # Internal module dotted names (e.g. src.workflow_engine.executor)
    re.compile(r"\bsrc\.\w+(?:\.\w+)*\b"),
]


def _strip_internal_details(raw: str) -> str:
    """Remove file paths, tracebacks, and internal module names from *raw*."""
    cleaned = raw
    for pattern in _INTERNAL_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    # Collapse excessive whitespace left behind after stripping.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _sanitize_error(raw: str, category: ErrorCategory) -> WorkflowUserError:
    """Sanitize a raw error string into a safe user-facing WorkflowUserError.

    The user-facing message is rendered from the category's template in
    :data:`_USER_MESSAGES`; if the template contains a ``{detail}``
    placeholder, it is filled with a sanitized version of *raw* so that
    user-supplied content (e.g. unknown-command hints) can appear in the
    reply without leaking file paths, tracebacks, or dotted module names.
    """
    template = _USER_MESSAGES.get(
        category,
        _USER_MESSAGES[ErrorCategory.INTERNAL_ERROR],
    )
    if "{detail}" in template and raw:
        safe_detail = _strip_internal_details(raw)
        user_message = template.replace("{detail}", safe_detail)
    else:
        user_message = template
    return WorkflowUserError(
        category=category,
        user_message=user_message,
        internal_detail=raw if raw else None,
    )


def sanitize_for_reply(raw: str, category: ErrorCategory) -> str:
    """Convenience wrapper that returns only the safe user-facing message.

    Parameters
    ----------
    raw:
        The original error string (stored internally, never returned).
    category:
        The error category determining the user message.

    Returns
    -------
    A sanitized string safe to include in a user-facing reply.
    """
    return _sanitize_error(raw, category).user_message


# ---------------------------------------------------------------------------
# Error categorization
# ---------------------------------------------------------------------------

# Keywords that indicate a transient error (retryable)
_TRANSIENT_ERROR_KEYWORDS: tuple[str, ...] = (
    "timeout",
    "timed out",
    "connection",
    "network",
    "temporary",
    "transient",
    "rate limit",
    "ratelimit",
    "rate_limit",
    "503",
    "504",
    "service unavailable",
    "gateway timeout",
    "busy",
    "overloaded",
    "try again",
    "retry later",
)

# A full ACP prompt timeout means the backend/model call already consumed the
# per-call timeout budget. Retrying the exact same prompt keeps Workflow cards
# stuck for several timeout windows; let the workflow script choose fallback
# routing instead.
_NON_RETRYABLE_TIMEOUT_MARKERS: tuple[str, ...] = (
    "acp prompt",
    "prompt execution timeout",
    "prompt 执行超时",
)

# Keywords that indicate a permanent error (not retryable)
_PERMANENT_ERROR_KEYWORDS: tuple[str, ...] = (
    "invalid schema",
    "schema validation",
    "not allowed",
    "not in allowed",
    "permission denied",
    "forbidden",
    "403",
    "404",
    "not found",
    "invalid",
    "malformed",
    "bad request",
    "400",
    "unauthorized",
    "401",
    "circuit breaker",
    "acpstartuperror",
)

# Category detection rules: (category, keyword_list, match_all)
# Rules are checked in order; first match wins.
_CATEGORY_RULES: list[tuple[ErrorCategory, tuple[str, ...], bool]] = [
    # A nested reviewer may itself end as cancelled.  Classify the top-level
    # review contract failure before the generic cancellation keywords.
    (ErrorCategory.REVIEW_FAILED, ("independent review failed",), False),
    (ErrorCategory.AGENT_LIMIT, ("limit exceeded",), False),
    (ErrorCategory.AGENT_LIMIT, ("max agents",), False),
    (ErrorCategory.AGENT_LIMIT, ("agent limit",), False),
    (ErrorCategory.TOOL_NOT_ALLOWED, ("not in allowed",), False),
    (ErrorCategory.TOOL_NOT_ALLOWED, ("tool not allowed",), False),
    (ErrorCategory.TOOL_NOT_ALLOWED, ("not permitted",), False),
    (ErrorCategory.SCRIPT_VALIDATION, ("script validation",), False),
    (ErrorCategory.SCRIPT_VALIDATION, ("validation failed",), False),
    (ErrorCategory.RUNTIME_TIMEOUT, ("timeout",), False),
    (ErrorCategory.RUNTIME_TIMEOUT, ("timed out",), False),
    (ErrorCategory.CANCELLED, ("cancelled",), False),
    (ErrorCategory.CANCELLED, ("canceled",), False),
]


def categorize_error(error_message: str) -> ErrorCategory:
    """Categorize an error message into an ErrorCategory.

    Uses keyword matching to classify errors. Checks are case-insensitive.
    Falls back to INTERNAL_ERROR if no specific category matches.

    Parameters
    ----------
    error_message:
        The raw error message string to categorize.

    Returns
    -------
    The matching ErrorCategory, or INTERNAL_ERROR if no specific match.

    Examples
    --------
    >>> categorize_error("agent limit exceeded")
    ErrorCategory.AGENT_LIMIT
    >>> categorize_error("tool shell not in allowed list")
    ErrorCategory.TOOL_NOT_ALLOWED
    >>> categorize_error("call cancelled by user")
    ErrorCategory.CANCELLED
    """
    if not error_message:
        return ErrorCategory.INTERNAL_ERROR

    msg_lower = error_message.lower()

    for category, keywords, match_all in _CATEGORY_RULES:
        if match_all:
            if all(kw in msg_lower for kw in keywords):
                return category
        else:
            if any(kw in msg_lower for kw in keywords):
                return category

    return ErrorCategory.INTERNAL_ERROR


def is_transient_error(error_message: str) -> bool:
    """Determine if an error is transient and worth retrying.

    Checks against known transient error keywords. Permanent errors
    (schema validation, permission denied, etc.) are not retried.

    Parameters
    ----------
    error_message:
        The raw error message string to check.

    Returns
    -------
    True if the error appears transient and retryable, False otherwise.
    """
    if not error_message:
        return False

    msg_lower = error_message.lower()

    # Permanent errors are never retryable
    if any(kw in msg_lower for kw in _PERMANENT_ERROR_KEYWORDS):
        return False

    if any(marker in msg_lower for marker in _NON_RETRYABLE_TIMEOUT_MARKERS):
        return False

    # Transient errors are retryable
    if any(kw in msg_lower for kw in _TRANSIENT_ERROR_KEYWORDS):
        return True

    # Default: not retryable (safe default to avoid infinite loops)
    return False
