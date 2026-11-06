import asyncio
import logging
import re
from typing import Union

logger = logging.getLogger(__name__)

__all__ = [
    "GhostAPError",
    "sanitize_futures_msg",
    "classify_timeout",
    "fmt_error",
    "get_error_detail",
    "fmt_exception",
    "log_exception",
    "safe_error_message",
    "redact_sensitive",
]

"""Unified error formatting and base exception for user-facing messages.

All user-facing error messages should use these helpers to ensure consistent
formatting (emoji prefix, Chinese language, action-based structure).

Convention:
    ❌  Hard failure (action failed, cannot proceed)
    ⚠️  Warning / partial failure (action completed with issues)
    ⏱️  Timeout (action took too long)
"""

# ---------------------------------------------------------------------------
# Feishu API Error Codes
# ---------------------------------------------------------------------------
LARK_CODE_MESSAGE_NOT_FOUND = 230001
LARK_CODE_MESSAGE_RECALLED = 230020


class GhostAPError(Exception):
    """Base exception for GhostAP domain errors.

    Carries a user-facing message (Chinese) and an optional machine-readable
    ``action`` tag for logging/metrics.

    Can also carry structured data for QuickAction buttons.
    """

    def __init__(self, message: str, *, action: str = "", quick_actions: list[str] = None, context: dict = None):
        super().__init__(message)
        self.action = action
        self.quick_actions = quick_actions or []
        self.context = context or {}


# ---------------------------------------------------------------------------
# User-facing message formatters
# ---------------------------------------------------------------------------

_CHAIN_MAX_DEPTH = 10

# stdlib concurrent.futures 在 TimeoutError 中注入的内部诊断格式，不应暴露给用户。
# 示例: "1 (of 5) futures unfinished", "3 (of 5) futures unfinished"
_FUTURES_UNFINISHED_RE = re.compile(r"\d+\s*\(of\s*\d+\)\s*futures?\s*unfinished")

# ---------------------------------------------------------------------------
# Safe message mapping — exception type name -> user-friendly message
# ---------------------------------------------------------------------------

_SAFE_MESSAGES: dict[str, str] = {
    "TimeoutError": "执行超时",
    "asyncio.TimeoutError": "执行超时",
    "ConnectionError": "连接失败",
    "ConnectionRefusedError": "连接失败",
    "ConnectionResetError": "连接失败",
    "OSError": "系统资源暂时不可用",
    "MemoryError": "系统资源不足，请稍后重试",
    "PermissionError": "权限不足",
    "FileNotFoundError": "所需资源未找到",
    "RuntimeError": "执行过程中出现异常，请重试",
    "ValueError": "内部错误，请联系管理员",
    "KeyError": "执行过程中出现异常，请重试",
    "TypeError": "执行过程中出现异常，请重试",
    "QueueFullError": "当前任务队列已满，请稍后重试",
    "TaskQueueFullError": "当前任务队列已满，请稍后重试",
    "ExecutorQueueFullError": "系统繁忙，请稍后重试",
}

_DEFAULT_SAFE_MESSAGE = "执行过程中出现异常，请稍后重试"

# Patterns that indicate sensitive content
_SENSITIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"/[\w./]+\.py", re.IGNORECASE),  # File paths
    re.compile(r"line \d+", re.IGNORECASE),  # Line numbers
    re.compile(r"Traceback \(most recent", re.IGNORECASE),  # Stack traces
    re.compile(r"File \"[^\"]+\"", re.IGNORECASE),  # File references
    re.compile(r"\w+Error\(", re.IGNORECASE),  # Exception constructors
    re.compile(r"\w+Exception\(", re.IGNORECASE),  # Exception constructors
    re.compile(r"0x[0-9a-fA-F]+"),  # Memory addresses
    re.compile(r"(password|secret|token|key|credential)s?\s*[=:]\s*[^\s,;]+", re.IGNORECASE),  # Credential values
    re.compile(r"(password|secret|token|key|credential)s?\s*[=:]", re.IGNORECASE),  # Credentials
)


def sanitize_futures_msg(msg: str) -> str:
    """清洗 stdlib 的 \"N (of M) futures unfinished\" 内部诊断信息。

    当 concurrent.futures.as_completed(timeout) 超时时，会抛出包含该格式信息的 TimeoutError，
    这对用户没有帮助，应该被替换成干净的中文提示或直接去掉。
    """
    cleaned = _FUTURES_UNFINISHED_RE.sub("", str(msg)).strip()
    return cleaned if cleaned else "操作超时，请稍后重试"


def classify_timeout(
    exc: BaseException,
    *,
    elapsed_s: float | None = None,
    timeout_s: float | None = None,
) -> bool:
    """Determine whether *exc* should be classified as a timeout error.

    Returns ``True`` if *exc* is itself a ``TimeoutError`` /
    ``asyncio.TimeoutError``, **or** if any exception in its
    ``__cause__`` / ``__context__`` chain is a timeout type (detected by
    :func:`_has_timeout_in_chain`).

    Heuristic detection (when *elapsed_s* and *timeout_s* are both provided):
    If the elapsed time is >= 80% of the timeout threshold **and** the
    exception message (or any message in the ``__cause__`` / ``__context__``
    chain) contains "Internal error" (case-insensitive), the exception is
    also classified as a timeout. This catches backends that surface
    JSON-RPC -32603 "Internal error" instead of raising a proper
    TimeoutError when a session start times out.

    This is the single source of truth for timeout classification — callers
    should prefer this over hand-rolling ``isinstance`` + chain-walk.
    """
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return True
    if _has_timeout_in_chain(exc):
        return True

    # Heuristic: "Internal error" + elapsed time close to timeout threshold
    if elapsed_s is not None and timeout_s is not None and timeout_s > 0:
        if elapsed_s >= timeout_s * 0.8:
            if _has_internal_error_in_chain(exc):
                return True

    return False


def _has_internal_error_in_chain(err: BaseException, *, _depth: int = 0) -> bool:
    """Walk __cause__ / __context__ looking for an "Internal error" message.

    The match is case-insensitive. Returns ``True`` as soon as one is found
    (max 10 levels deep).
    """
    if _depth >= _CHAIN_MAX_DEPTH:
        return False
    msg = str(err) if err is not None else ""
    if "internal error" in msg.lower():
        return True
    for attr in ("__cause__", "__context__"):
        chained = getattr(err, attr, None)
        if chained is None:
            continue
        if _has_internal_error_in_chain(chained, _depth=_depth + 1):
            return True
    return False


def _has_timeout_in_chain(err: BaseException, *, _depth: int = 0) -> bool:
    """Walk __cause__ / __context__ for a wrapped TimeoutError (max 10 levels).

    Detects:
    - ``TimeoutError`` / ``asyncio.TimeoutError`` via isinstance
    - Third-party timeout types (``TimeoutExpired``, ``ReadTimeout``,
      ``ConnectTimeout``) via class-name matching
    """
    if _depth >= _CHAIN_MAX_DEPTH:
        return False
    for attr in ("__cause__", "__context__"):
        chained = getattr(err, attr, None)
        if chained is None:
            continue
        if isinstance(chained, (TimeoutError, asyncio.TimeoutError)):
            return True
        try:
            tn = type(chained).__name__
        except Exception:
            tn = ""
        if tn in ("TimeoutExpired", "ReadTimeout", "ConnectTimeout"):
            return True
        if _has_timeout_in_chain(chained, _depth=_depth + 1):
            return True
    return False


def fmt_error(action: str, detail: Union[str, Exception] = "") -> str:
    """Format a hard-failure message."""
    if isinstance(detail, Exception):
        if isinstance(detail, (TimeoutError, asyncio.TimeoutError)):
            msg = _FUTURES_UNFINISHED_RE.sub("", str(detail)).strip()
            if not msg:
                detail = "操作超时，请稍后重试"
            else:
                detail = f"操作超时 ({msg})"
        elif _has_timeout_in_chain(detail):
            # Exception wraps a TimeoutError in its chain
            inner_msg = _FUTURES_UNFINISHED_RE.sub("", str(detail)).strip()
            if not inner_msg:
                detail = "操作超时，请稍后重试"
            else:
                detail = f"操作超时 ({inner_msg})"
        else:
            detail = str(detail)

    if detail:
        return f"❌ {action}失败: {detail}"
    return f"❌ {action}失败"


def get_error_detail(exc: Exception, *, default: str = "未知错误") -> str:
    """统一把异常转成可展示的 detail（不含 "❌ 失败" 前缀）。

    所有返回值都会经过敏感数据脱敏处理。
    """
    try:
        formatted = fmt_error("", exc)
    except Exception:
        formatted = ""
    s = str(formatted or "")

    # 典型输出："❌ 失败: xxx" / "❌ 失败"
    if s.startswith("❌ "):
        if "失败: " in s:
            tail = s.split("失败: ", 1)[1]
            tail = str(tail or "").strip()
            if tail:
                return redact_sensitive(tail)
        if s.endswith("失败"):
            msg = str(exc or "").strip()
            return redact_sensitive(msg or default)

    # 非标准格式：保持原样
    return redact_sensitive(s)


def fmt_exception(action: str, exc: BaseException) -> str:
    """Format an unexpected exception for the user."""
    if classify_timeout(exc):
        return f"❌ {action}超时: 操作耗时过长，请重试"
    return f"❌ {action}异常: {str(exc) or repr(exc)}"


def log_exception(logger: logging.Logger, msg: str, exc: Exception, level: int = logging.ERROR) -> None:
    """Log an exception with appropriate level.

    Downgrades known business logic exceptions (GhostAPError or any exception
    carrying ``is_ghostap_error = True``) to WARNING.
    """
    if isinstance(exc, GhostAPError) or getattr(exc, "is_ghostap_error", False):
        logger.warning(f"{msg}: {get_error_detail(exc)}")
    else:
        logger.log(level, msg, exc_info=exc)


def safe_error_message(exc: Union[BaseException, str, None]) -> str:
    """Convert an exception or error string to a safe user-facing message.

    Ensures that raw exception details (stack traces, file paths, class names)
    are never exposed to users. Maps exceptions to predefined safe messages.

    Args:
        exc: An exception instance, error string, or None.

    Returns:
        A safe, user-friendly error message string.
    """
    if exc is None:
        return _DEFAULT_SAFE_MESSAGE

    if isinstance(exc, str):
        # String errors: apply redact_sensitive first (consistent with exception handling)
        redacted = redact_sensitive(exc)
        # Check if redaction removed all meaningful content
        if not redacted.strip() or all(part.strip() in ("", "[redacted]") for part in redacted.split()):
            return _DEFAULT_SAFE_MESSAGE
        # Cap length for user display
        return redacted[:80] if len(redacted) <= 80 else _DEFAULT_SAFE_MESSAGE

    # Map by exception type name
    exc_type_name = type(exc).__name__
    exc_module = type(exc).__module__ or ""

    # Try full qualified name first, then just class name
    full_name = f"{exc_module}.{exc_type_name}" if exc_module != "builtins" else exc_type_name
    safe_msg = _SAFE_MESSAGES.get(full_name) or _SAFE_MESSAGES.get(exc_type_name)

    if safe_msg:
        # Log mapped exceptions with full stack trace for observability
        logger.warning(
            "Mapped exception %s -> user message: %s",
            full_name,
            safe_msg,
            exc_info=exc,
        )
        return safe_msg

    # Log unmapped exceptions with full stack trace for debugging
    logger.warning(
        "Unmapped exception type %s: %s",
        full_name,
        redact_sensitive(str(exc)[:200]),
        exc_info=exc,
    )
    return _DEFAULT_SAFE_MESSAGE


def redact_sensitive(text: str) -> str:
    """Remove sensitive information from a text string.

    Replaces file paths, line numbers, stack traces, memory addresses,
    and credential patterns with '[redacted]'.

    Args:
        text: Raw text that may contain sensitive information.

    Returns:
        Text with sensitive patterns replaced by '[redacted]'.
    """
    if not text:
        return text

    result = text
    for pattern in _SENSITIVE_PATTERNS:
        result = pattern.sub("[redacted]", result)

    return result
