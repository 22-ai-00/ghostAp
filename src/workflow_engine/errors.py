from __future__ import annotations

import collections
import json
import re
from bisect import bisect_right
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


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

_REDACTED = "<redacted>"
_BS_64 = "\\" * 64
_BS_8 = "\\" * 8

# Patterns and structural redactors considered internal-only and stripped from
# user-facing messages.  All stages are strictly linear-time O(N), idempotent,
# and preserve GhostAP LLM/metadata telemetry, English prose, URLs, slash
# commands (e.g. /wf, /wf_status), empty values, and JSON structure.

_NON_SECRET_EXACT: frozenset[str] = frozenset(
    {
        "token_usage",
        "token_count",
        "token_counts",
        "token_limit",
        "token_limits",
        "token_budget",
        "token_budgets",
        "token_type",
        "token_types",
        "token_name",
        "token_names",
        "tokens_used",
        "max_tokens",
        "max_token",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "thinking_tokens",
        "thought_tokens",
        "reasoning_tokens",
        "cached_tokens",
        "cache_read_tokens",
        "cache_creation_tokens",
        "used_tokens",
        "remaining_tokens",
        "num_tokens",
        "count_tokens",
        "estimated_tokens",
        "min_tokens",
        "target_tokens",
        "budget_tokens",
        "context_tokens",
        "delta_context_tokens",
        "tokenized",
        "tokenizer",
        "primary_key",
        "foreign_key",
        "sort_key",
        "partition_key",
        "cache_key",
        "public_key",
        "file_key",
        "image_key",
        "breaker_key",
        "binding_key",
        "session_key",
        "engine_session_key",
        "tool_model_key",
        "key_name",
        "key_names",
        "key_type",
        "key_types",
        "key_id",
        "key_ids",
        "key_count",
        "key_file",
        "key_files",
        "key_findings",
        "keyboard_layout",
        "secret_sauce",
        "secret_name",
        "is_secret",
        "has_secret",
        "no_secret",
        "use_secret",
        "non_secret",
    }
)

_SECRET_EXACT: frozenset[str] = frozenset(
    {
        "authorization",
        "proxy_authorization",
        "password",
        "passwd",
        "passphrase",
        "secret",
        "token",
        "apikey",
        "secretkey",
        "accesskey",
        "privatekey",
        "authkey",
        "signingkey",
        "encryptionkey",
        "masterkey",
        "webhookkey",
        "clientkey",
        "accesstoken",
        "refreshtoken",
        "authtoken",
        "bearertoken",
        "sessiontoken",
        "idtoken",
        "clientsecret",
        "appsecret",
        "credential",
        "credentials",
        "api_key",
        "secret_key",
        "access_key",
        "private_key",
        "auth_key",
        "signing_key",
        "encryption_key",
        "master_key",
        "webhook_key",
        "client_key",
    }
)

_SENSITIVE_KEY_STEM_PREFIXES: frozenset[str] = frozenset(
    {
        "api",
        "access",
        "secret",
        "private",
        "auth",
        "signing",
        "encryption",
        "master",
        "webhook",
        "client",
        "app",
        "bearer",
        "ssh",
        "gpg",
        "pgp",
        "hmac",
        "jwt",
        "oauth",
        "token",
        "admin",
        "root",
        "service",
        "aws",
        "openai",
        "anthropic",
        "gemini",
        "deepseek",
        "feishu",
        "lark",
        "github",
    }
)

_NON_SECRET_TOKEN_PREFIXES: frozenset[str] = frozenset(
    {
        "max",
        "prompt",
        "completion",
        "total",
        "input",
        "output",
        "thinking",
        "thought",
        "reasoning",
        "cached",
        "cache_read",
        "cache_creation",
        "used",
        "remaining",
        "num",
        "count",
        "estimated",
        "min",
        "target",
        "budget",
        "context",
        "delta",
        "delta_context",
    }
)

_NON_SECRET_SECRET_PREFIXES: frozenset[str] = frozenset(
    {"is", "has", "no", "use", "non"}
)

_SENSITIVE_PLURAL_TOKENS: frozenset[str] = frozenset(
    {
        "access_tokens",
        "auth_tokens",
        "bearer_tokens",
        "refresh_tokens",
        "session_tokens",
        "secret_tokens",
        "api_tokens",
    }
)

_CAMEL_BOUNDARY_RE_1 = re.compile(r"([a-z0-9])([A-Z])")
_CAMEL_BOUNDARY_RE_2 = re.compile(r"([A-Z]+)([A-Z][a-z])")


def _normalize_key_name(key: str) -> str:
    s = _CAMEL_BOUNDARY_RE_1.sub(r"\1_\2", key.strip())
    s = _CAMEL_BOUNDARY_RE_2.sub(r"\1_\2", s)
    return s.lower().replace("-", "_")


def _is_sensitive_key(key: str) -> bool:
    """Return True if *key* names a credential/secret field rather than telemetry/metadata."""
    if not key or len(key) > 80:
        return False
    norm = _normalize_key_name(key)
    if not norm or norm in _NON_SECRET_EXACT:
        return False
    if norm in _SECRET_EXACT or norm in _SENSITIVE_PLURAL_TOKENS:
        return True
    parts = [p for p in norm.split("_") if p]
    if not parts:
        return False
    last = parts[-1]
    if last in {"password", "passwd", "passphrase", "credential", "credentials"}:
        return True
    if last == "secret":
        return parts[0] not in _NON_SECRET_SECRET_PREFIXES
    if last == "token":
        if parts[0] in _NON_SECRET_TOKEN_PREFIXES:
            return False
        if len(parts) >= 2 and parts[-2] in _NON_SECRET_TOKEN_PREFIXES:
            return False
        return True
    if last == "key" and len(parts) >= 2:
        return parts[-2] in _SENSITIVE_KEY_STEM_PREFIXES
    if last in {"apikey", "secretkey", "accesskey", "privatekey", "password", "passwd"}:
        return True
    return False


# ---------------------------------------------------------------------------
# Stage 1a: Embedded & recursive JSON redactor (bounded depth, O(N) scan)
# ---------------------------------------------------------------------------

_JSON_DECODER = json.JSONDecoder()
_JSON_SCAN_RE = re.compile(r"\\[^\r\n]|([\"{}\[\]\r\n])")
_MAX_JSON_STACK = 64
_MAX_JSON_DEPTH = 8


def _find_json_spans(text: str) -> list[tuple[int, int]]:
    """Find candidate JSON object/array spans in O(N) without regex backtracking."""
    spans: list[tuple[int, int]] = []
    stack: collections.deque[tuple[str, int]] = collections.deque(
        maxlen=_MAX_JSON_STACK
    )
    in_string = False

    for m in _JSON_SCAN_RE.finditer(text):
        tok = m.group(1)
        if tok is None:
            continue
        if tok in "\r\n":
            in_string = False
            continue
        if tok == '"':
            if stack:
                in_string = not in_string
            continue
        if in_string:
            continue
        if tok in "{[":
            stack.append(("}" if tok == "{" else "]", m.start()))
        elif tok in "}]":
            if stack and stack[-1][0] == tok:
                _, start = stack.pop()
                spans.append((start, m.end()))

    if len(spans) > 1:
        spans.sort(key=lambda pair: (pair[0], -pair[1]))
    return spans


def _redact_json_str(
    val: str,
    *,
    depth: int,
    max_depth: int,
) -> tuple[str, bool]:
    """Redact nested JSON or secret patterns inside a JSON string leaf."""
    if depth >= max_depth:
        return val, False
    s = val.strip()
    if (s.startswith("{") and s.endswith("}")) or (
        s.startswith("[") and s.endswith("]")
    ):
        try:
            inner = json.loads(s)
        except (json.JSONDecodeError, RecursionError, ValueError):
            inner = None
        if isinstance(inner, (dict, list)):
            new_inner, inner_changed = _redact_json_value(
                inner, depth=depth + 1, max_depth=max_depth
            )
            if inner_changed:
                try:
                    return json.dumps(new_inner, ensure_ascii=False), True
                except RecursionError:
                    pass
            else:
                return val, False
    stripped = _strip_non_top_json(val, depth=depth + 1, max_depth=max_depth)
    if stripped != val:
        return stripped, True
    return val, False


def _redact_json_value(
    val: Any,
    *,
    depth: int = 0,
    max_depth: int = _MAX_JSON_DEPTH,
) -> tuple[Any, bool]:
    """Iteratively redact sensitive keys and bounded-depth nested JSON strings."""
    if isinstance(val, str):
        return _redact_json_str(val, depth=depth, max_depth=max_depth)
    if not isinstance(val, (dict, list)):
        return val, False

    root_holder: list[Any] = [None]
    changed = False
    work_stack: list[tuple[Any, Any, Any]] = [(val, root_holder, 0)]

    while work_stack:
        node, parent, slot = work_stack.pop()
        if isinstance(node, dict):
            new_dict: dict[Any, Any] = {}
            parent[slot] = new_dict
            for k, v in node.items():
                if isinstance(k, str) and _is_sensitive_key(k):
                    if v is None or v == "" or v == _REDACTED:
                        new_dict[k] = v
                    elif isinstance(v, (dict, list)) and len(v) == 0:
                        new_dict[k] = v
                    else:
                        new_dict[k] = _REDACTED
                        changed = True
                elif isinstance(v, (dict, list)):
                    new_dict[k] = None
                    work_stack.append((v, new_dict, k))
                elif isinstance(v, str):
                    new_v, str_changed = _redact_json_str(
                        v, depth=depth, max_depth=max_depth
                    )
                    new_dict[k] = new_v
                    if str_changed:
                        changed = True
                else:
                    new_dict[k] = v
        elif isinstance(node, list):
            new_list: list[Any] = [None] * len(node)
            parent[slot] = new_list
            for idx, item in enumerate(node):
                if isinstance(item, (dict, list)):
                    work_stack.append((item, new_list, idx))
                elif isinstance(item, str):
                    new_item, str_changed = _redact_json_str(
                        item, depth=depth, max_depth=max_depth
                    )
                    new_list[idx] = new_item
                    if str_changed:
                        changed = True
                else:
                    new_list[idx] = item

    return (root_holder[0] if changed else val), changed


def _redact_embedded_json(
    text: str,
    *,
    depth: int = 0,
    max_depth: int = _MAX_JSON_DEPTH,
) -> str:
    """Redact top-level or embedded JSON payloads in *text*."""
    if depth >= max_depth:
        return text
    has_obj = "{" in text and "}" in text
    has_arr = "[" in text and "]" in text
    if not (has_obj or has_arr):
        return text

    s = text.strip()
    if (s.startswith("{") and s.endswith("}")) or (
        s.startswith("[") and s.endswith("]")
    ):
        try:
            parsed = json.loads(s)
        except (json.JSONDecodeError, RecursionError, ValueError):
            parsed = None
        if isinstance(parsed, (dict, list)):
            redacted, changed = _redact_json_value(
                parsed, depth=depth, max_depth=max_depth
            )
            if not changed:
                return text
            try:
                dumped = json.dumps(redacted, ensure_ascii=False)
            except RecursionError:
                dumped = None
            if dumped is not None:
                lead_len = len(text) - len(text.lstrip())
                trail_len = len(text) - len(text.rstrip())
                lead = text[:lead_len] if lead_len else ""
                trail = text[len(text) - trail_len :] if trail_len else ""
                return f"{lead}{dumped}{trail}"

    spans = _find_json_spans(text)
    if not spans:
        return text

    out_parts: list[str] = []
    cursor = 0
    fail_budget = min(16384, max(4096, len(text)))
    changed_any = False

    for start, span_end in spans:
        if start < cursor:
            continue
        if fail_budget <= 0:
            break
        candidate = text[start:span_end]
        if len(candidate) > 1 and candidate[1] == "\\":
            fail_budget -= 1
            continue
        try:
            obj, rel_end = _JSON_DECODER.raw_decode(candidate, 0)
        except json.JSONDecodeError as exc:
            fail_budget -= max(1, exc.pos)
            continue
        except (RecursionError, ValueError):
            fail_budget -= 16384
            break
        end = start + rel_end

        if isinstance(obj, (dict, list)):
            redacted, changed = _redact_json_value(
                obj, depth=depth, max_depth=max_depth
            )
            if changed:
                try:
                    dumped = json.dumps(redacted, ensure_ascii=False)
                except RecursionError:
                    dumped = None
                if dumped is not None:
                    out_parts.append(text[cursor:start])
                    out_parts.append(dumped)
                    cursor = end
                    changed_any = True
                    continue
        out_parts.append(text[cursor:end])
        cursor = end

    if not changed_any:
        return text
    out_parts.append(text[cursor:])
    return "".join(out_parts)


# ---------------------------------------------------------------------------
# Stage 1b: Escaped quoted-literal unwrapper (O(N) scan)
# ---------------------------------------------------------------------------


def _scan_quoted_pass(
    text: str,
    *,
    target_bs: int,
    depth: int,
    max_depth: int,
) -> str:
    """Scan matching quote pairs with backslash parity/count in strict O(N) time."""
    out_parts: list[str] = []
    cursor = 0
    lit_start: int | None = None
    changed_any = False
    pos = 0
    n = len(text)

    while pos < n:
        idx = text.find('"', pos)
        if idx == -1:
            break
        bs = 0
        j = idx - 1
        while j >= 0 and text[j] == "\\":
            bs += 1
            j -= 1
        pos = idx + 1

        if target_bs == 0:
            if bs % 2 != 0:
                continue
            delim_start = idx
        else:
            if bs != target_bs:
                continue
            delim_start = idx - target_bs

        if lit_start is None:
            lit_start = delim_start
            continue

        lit_end = idx + 1
        span_start = lit_start
        lit_start = None
        inner_raw = text[span_start + target_bs + 1 : idx - target_bs]

        if '\\"' in inner_raw and ("{" in inner_raw or "[" in inner_raw):
            json_lit = (
                text[span_start:lit_end]
                if target_bs == 0
                else f'"{inner_raw}"'
            )
            try:
                inner = json.loads(json_lit)
            except (json.JSONDecodeError, RecursionError, ValueError):
                inner = None
            if isinstance(inner, str):
                redacted_inner = _strip_pipeline(
                    inner, depth=depth + 1, max_depth=max_depth
                )
                if redacted_inner != inner:
                    dumped = json.dumps(redacted_inner, ensure_ascii=False)
                    replacement = (
                        dumped
                        if target_bs == 0
                        else f'\\"{dumped[1:-1]}\\"'
                    )
                    out_parts.append(text[cursor:span_start])
                    out_parts.append(replacement)
                    cursor = lit_end
                    changed_any = True

    if not changed_any:
        return text
    out_parts.append(text[cursor:])
    return "".join(out_parts)


def _redact_escaped_quoted_strings(
    text: str,
    *,
    depth: int = 0,
    max_depth: int = _MAX_JSON_DEPTH,
) -> str:
    """Unwrap escaped JSON inside double-quoted literals in non-JSON containers."""
    if depth >= max_depth or '\\"' not in text:
        return text
    cleaned = _scan_quoted_pass(
        text, target_bs=0, depth=depth, max_depth=max_depth
    )
    if '\\\\"' in cleaned:
        cleaned = _scan_quoted_pass(
            cleaned, target_bs=1, depth=depth, max_depth=max_depth
        )
    return cleaned


# ---------------------------------------------------------------------------
# Stage 2: Classifier-driven linear KV / auth-scheme / token-prefix patterns
# ---------------------------------------------------------------------------

_AUTH_TWO_TOKEN_SCHEMES = (
    r"(?:Bearer|Basic|Token|Api[_-]?Key|Bot|SSWS|DPoP|Negotiate|NTLM|Mutual|"
    r"HMAC(?:-SHA(?:1|256|384|512))?|Signature|Key|SharedKey|AppKey|ClientKey|PAT|JWT)"
)
_AUTH_LINE_REST_SCHEMES = r"(?:Digest|OAuth2?|AWS4-HMAC-SHA256|HOBA|SCRAM-SHA-256)"

_DQ_CLOSED = r"\"(?:\\.|[^\"\\\r\n])*\""
_SQ_CLOSED = r"'(?:\\.|[^'\\\r\n])*'"

_DQ_UNCLOSED = r"\"[^\r\n]*"
_SQ_UNCLOSED = r"'[^\r\n]*"

_AUTH_2_VAL = (
    rf"{_AUTH_TWO_TOKEN_SCHEMES}[ \t]+"
    r"(?:(?:token|key)[ \t]*[=:][ \t]*)?"
    rf"(?:{_DQ_CLOSED}|{_SQ_CLOSED}"
    rf"|{_DQ_UNCLOSED}|{_SQ_UNCLOSED}"
    r"|(?!\\+['\"])[^\s,;'\"\}\]]+)"
)
_AUTH_2_BS_QUOTE_RE = re.compile(
    rf"(?i){_AUTH_TWO_TOKEN_SCHEMES}[ \t]+"
    r"(?:(?:token|key)[ \t]*[=:][ \t]*)?"
    r"(?P<bs>\\+)(?P<q>['\"])"
)
_AUTH_REST_VAL = rf"{_AUTH_LINE_REST_SCHEMES}[ \t]+[^\r\n\}}]+"
_INSPECT_COLLECTION_RE = re.compile(r"(?:Map|Set)\(\d+\)[ \t]*\{")

_KV_HEAD_PATTERN = (
    r"(?<![A-Za-z0-9_-])"
    r"(?P<key>[A-Za-z0-9_-]{3,80})(?![A-Za-z0-9_-])"
    r"(?P<sep>(?:\\*['\"])?[ \t]*[:=][ \t]*)"
)
_KV_VAL_PATTERN = (
    rf"(?P<sq_dq>{_DQ_CLOSED})"
    rf"|(?P<sq_sq>{_SQ_CLOSED})"
    rf"|(?P<sq_unclosed_dq>{_DQ_UNCLOSED})"
    rf"|(?P<sq_unclosed_sq>{_SQ_UNCLOSED})"
    rf"|(?P<sq_auth_rest>{_AUTH_REST_VAL})"
    rf"|(?P<sq_auth_2>{_AUTH_2_VAL})"
    r"|(?P<sq_unq>(?!\\+['\"])[^\s,;'\"\{\}\[\]][^\s,;'\"\}\]]*)"
    r"|(?P<nl_ws>\r?\n[ \t]*)"
    r"(?:"
    rf"(?P<ml_dq>{_DQ_CLOSED})"
    rf"|(?P<ml_sq>{_SQ_CLOSED})"
    rf"|(?P<ml_unclosed_dq>{_DQ_UNCLOSED})"
    rf"|(?P<ml_unclosed_sq>{_SQ_UNCLOSED})"
    rf"|(?P<ml_auth_rest>{_AUTH_REST_VAL})"
    rf"|(?P<ml_auth_2>{_AUTH_2_VAL})"
    r")"
)

_KV_HEAD_RE = re.compile(rf"(?i){_KV_HEAD_PATTERN}")
_KV_VAL_RE = re.compile(rf"(?i)(?:{_KV_VAL_PATTERN})")
_NL_WS_RE = re.compile(r"\r?\n[ \t]*")

_STANDALONE_BEARER_RE = re.compile(
    rf"(?i)\bBearer[ \t]+(?:(?:token|key)[ \t]*[=:][ \t]*)?"
    r"(?:"
    rf"(?P<dq>{_DQ_CLOSED})"
    rf"|(?P<sq>{_SQ_CLOSED})"
    r"|(?P<bsq>\\+['\"])"
    rf"|(?P<udq>{_DQ_UNCLOSED})"
    rf"|(?P<usq>{_SQ_UNCLOSED})"
    r"|(?P<unq>(?!\\+['\"])[^\s,;'\"\}\]]+)"
    r")"
)

_TOKEN_PREFIX_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_\-]{6,}|gh[pousr][_-][A-Za-z0-9_\-]{6,}|"
    r"glpat-[A-Za-z0-9_\-]{6,}|xox[baprs]-[A-Za-z0-9_\-]{6,}|AKIA[0-9A-Z]{16})\b"
)


def _redact_standalone_bearer(text: str) -> str:
    """Redact standalone Bearer tokens in O(N) while preserving empty quotes and idempotency."""
    if "bearer" not in text.lower():
        return text
    n = len(text)
    out_parts: list[str] = []
    cursor = 0
    pos = 0
    changed = False

    while pos < n:
        m = _STANDALONE_BEARER_RE.search(text, pos)
        if m is None:
            break
        bsq = m.group("bsq")
        if bsq is not None:
            bs_len = len(bsq) - 1
            q = bsq[-1]
            end_idx, already_safe, _ = _scan_escaped_quote_literal(
                text, m.start("bsq"), bs_len, q
            )
            if not already_safe:
                out_parts.append(text[cursor : m.start()])
                out_parts.append(f"Bearer {_REDACTED}")
                cursor = end_idx
                changed = True
            pos = end_idx
            continue

        dq = m.group("dq")
        sq = m.group("sq")
        if dq is not None or sq is not None:
            val = dq if dq is not None else sq
            assert val is not None
            inner = val[1:-1]
            if inner != "" and inner != _REDACTED:
                out_parts.append(text[cursor : m.start()])
                out_parts.append(f"Bearer {_REDACTED}")
                cursor = m.end()
                changed = True
            pos = m.end()
            continue

        for grp in ("udq", "usq", "unq"):
            val = m.group(grp)
            if val is not None:
                if val != _REDACTED:
                    out_parts.append(text[cursor : m.start()])
                    out_parts.append(f"Bearer {_REDACTED}")
                    cursor = m.end()
                    changed = True
                break
        pos = m.end()

    if not changed:
        return text
    out_parts.append(text[cursor:])
    return "".join(out_parts)


def _is_json_continuation_line(text: str, pos: int) -> bool:
    """Return True if *text[pos:]* begins with a plausible JSON/container continuation line."""
    n = len(text)
    while pos < n and text[pos] in " \t":
        pos += 1
    if pos >= n or text[pos] in "\r\n":
        return False
    ch = text[pos]
    if ch in "\"'{}[],\\":
        return True
    if ch == "-" or ch.isdigit():
        return True
    for kw in ("true", "false", "null"):
        if text.startswith(kw, pos):
            return True
    # Accept unquoted JS / Node util.inspect property continuation lines: [A-Za-z0-9_$-]+[ \t]*:
    j = pos
    while j < n and (text[j].isalnum() or text[j] in "_$-"):
        j += 1
    if j > pos:
        while j < n and text[j] in " \t":
            j += 1
        if j < n and text[j] == ":":
            return True
    return False


def _scan_escaped_quote_literal(
    text: str,
    start: int,
    open_bs: int,
    quote_char: str,
) -> tuple[int, bool, str]:
    """Scan an escaped-quote string literal (e.g. ``\\"...\\"`` or ``\\\\"...``) in O(N).

    Supports both odd ``open_bs`` (``1, 3, 5, ...``) and even ``open_bs``
    (``2, 4, ...``).  Returns ``(end_idx, already_empty_or_redacted, close_delim)``.
    """
    n = len(text)
    i = start + open_bs + 1
    content_start = i
    is_even = open_bs % 2 == 0
    mod = 2 if is_even else (2 * (open_bs + 1) if (open_bs & (open_bs + 1)) == 0 else 4)
    target_rem = 0 if is_even else (open_bs % mod)
    open_delim = text[start : start + open_bs + 1]

    while i < n:
        ch = text[i]
        if ch in "\r\n":
            return i, False, open_delim
        if is_even and ch == quote_char:
            inner = text[content_start:i]
            return i + 1, (inner == "" or inner == _REDACTED), quote_char
        if ch == "\\":
            bs = 1
            if i + 8 < n and text[i + 1 : i + 9] == _BS_8:
                bs += 8
                while i + bs + 64 <= n and text[i + bs : i + bs + 64] == _BS_64:
                    bs += 64
                while i + bs + 8 <= n and text[i + bs : i + bs + 8] == _BS_8:
                    bs += 8
            while i + bs < n and text[i + bs] == "\\":
                bs += 1
            if i + bs < n and text[i + bs] == quote_char:
                if (is_even or bs >= open_bs) and bs % mod == target_rem:
                    inner = text[content_start:i]
                    close_delim = text[i : i + bs + 1] if is_even else open_delim
                    return i + bs + 1, (inner == "" or inner == _REDACTED), close_delim
                i += bs + (1 if bs % 2 == 1 else 0)
                continue
            if i + bs < n and text[i + bs] in "\r\n":
                i += bs
                continue
            if i + bs < n:
                i += bs + (1 if bs % 2 == 1 else 0)
                continue
            return n, False, open_delim
        i += 1
    return n, False, open_delim


class _QuoteIndex:
    """Lazy O(N) pre-index of single/double quotes and preceding backslash counts."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.has_cr = "\r" in text
        self.last_eol = max(
            text.rfind("\n"), text.rfind("\r") if self.has_cr else -1
        )
        self._eol_queries = 0
        self._nl_pos: list[int] | None = None
        self._cr_pos: list[int] | None = None
        self.even_pos: dict[str, list[int]] = {'"': [], "'": []}
        self.mersenne_pos: dict[tuple[str, int], list[int]] = {}
        self.res_pos: dict[tuple[str, int], list[int]] = {}
        self.res_bs: dict[tuple[str, int], list[int]] = {}
        self._trees: dict[tuple[str, int], tuple[int, int, list[int]]] = {}

        for q in ('"', "'"):
            even_list = self.even_pos[q]
            pos = 0
            while True:
                idx = text.find(q, pos)
                if idx == -1:
                    break
                bs = 0
                j = idx - 1
                if j >= 0 and text[j] == "\\":
                    bs = 1
                    j -= 1
                    if j >= 7 and text[j - 7 : j + 1] == _BS_8:
                        bs += 8
                        j -= 8
                        while j >= 63 and text[j - 63 : j + 1] == _BS_64:
                            bs += 64
                            j -= 64
                        while j >= 7 and text[j - 7 : j + 1] == _BS_8:
                            bs += 8
                            j -= 8
                    while j >= 0 and text[j] == "\\":
                        bs += 1
                        j -= 1
                pos = idx + 1
                if bs % 2 == 0:
                    even_list.append(idx)
                else:
                    k = (bs ^ (bs + 1)).bit_length() - 1
                    m_key = (q, (1 << k) - 1)
                    m_list = self.mersenne_pos.get(m_key)
                    if m_list is None:
                        self.mersenne_pos[m_key] = [idx]
                    else:
                        m_list.append(idx)

                    r_key = (q, bs & 3)
                    rp = self.res_pos.get(r_key)
                    if rp is None:
                        self.res_pos[r_key] = [idx]
                        self.res_bs[r_key] = [bs]
                    else:
                        rp.append(idx)
                        self.res_bs[r_key].append(bs)

    def find_next_eol(self, after_pos: int) -> int:
        if after_pos >= self.last_eol:
            return -1
        if self._eol_queries < 4:
            self._eol_queries += 1
            nl = self.text.find("\n", after_pos + 1)
            cr = self.text.find("\r", after_pos + 1) if self.has_cr else -1
            if nl == -1:
                return cr
            if cr == -1:
                return nl
            return nl if nl < cr else cr
        if self._nl_pos is None:
            nl_pos: list[int] = []
            p = 0
            while True:
                idx = self.text.find("\n", p)
                if idx == -1:
                    break
                nl_pos.append(idx)
                p = idx + 1
            self._nl_pos = nl_pos
            if self.has_cr:
                cr_pos: list[int] = []
                p = 0
                while True:
                    idx = self.text.find("\r", p)
                    if idx == -1:
                        break
                    cr_pos.append(idx)
                    p = idx + 1
                self._cr_pos = cr_pos
            else:
                self._cr_pos = []
        res = -1
        i_nl = bisect_right(self._nl_pos, after_pos)
        if i_nl < len(self._nl_pos):
            res = self._nl_pos[i_nl]
        if self._cr_pos:
            i_cr = bisect_right(self._cr_pos, after_pos)
            if i_cr < len(self._cr_pos):
                cr_v = self._cr_pos[i_cr]
                if res == -1 or cr_v < res:
                    res = cr_v
        return res

    def find_close(self, q: str, open_bs: int, after_pos: int) -> int | None:
        if open_bs == 0:
            plist = self.even_pos[q]
            lo = bisect_right(plist, after_pos)
            return plist[lo] if lo < len(plist) else None

        if (open_bs & (open_bs + 1)) == 0:
            plist = self.mersenne_pos.get((q, open_bs))
            if not plist:
                return None
            lo = bisect_right(plist, after_pos)
            return plist[lo] if lo < len(plist) else None

        r_key = (q, open_bs & 3)
        plist = self.res_pos.get(r_key)
        if not plist:
            return None
        lo = bisect_right(plist, after_pos)
        length = len(plist)
        if lo >= length:
            return None

        tree_entry = self._trees.get(r_key)
        if tree_entry is None:
            bs_list = self.res_bs[r_key]
            size = 1 << (length - 1).bit_length() if length > 1 else 1
            tree = [0] * (2 * size)
            tree[size : size + length] = bs_list
            for node in range(size - 1, 0, -1):
                lv = tree[node << 1]
                rv = tree[(node << 1) | 1]
                tree[node] = lv if lv >= rv else rv
            tree_entry = (size, length, tree)
            self._trees[r_key] = tree_entry

        size, length, tree = tree_entry
        v = size + lo
        while True:
            while (v & 1) == 0:
                v >>= 1
            if tree[v] >= open_bs:
                break
            if (v & (v + 1)) == 0:
                return None
            v += 1
        while v < size:
            v <<= 1
            if tree[v] < open_bs:
                v += 1
        leaf = v - size
        return plist[leaf] if leaf < length else None


class _ContainerScanState:
    """Shared O(N log N) memoization across _redact_kv_and_auth()."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.n = len(text)
        self.quote_index: _QuoteIndex | None = None
        self.unclosed_str_eof: dict[tuple[str, int], int] = {}
        self.closed_memo: dict[int, tuple[int, bool, int | None]] = {}
        self.failed_memo: dict[int, tuple[int, bool]] = {}
        self.p2_memo: dict[
            int, tuple[str, int, str | None, bool, int | None]
        ] = {}
        self.total_p2_steps = 0
        self.p2_memo_threshold = max(4096, self.n)

    def get_quote_index(self) -> _QuoteIndex:
        if self.quote_index is None:
            self.quote_index = _QuoteIndex(self.text)
        return self.quote_index


def _scan_container_value(
    text: str,
    start: int,
    *,
    escaped_quote: bool = False,
    escaped_bs: int | None = None,
    multiline_budget: list[int] | None = None,
    unclosed_str_eof: dict[str, int] | None = None,
    phase2_high_water: list[int] | None = None,
    scan_state: _ContainerScanState | None = None,
) -> tuple[int, bool]:
    """Scan a balanced or truncated ``{...}`` / ``[...]`` container in O(N log N) time.

    Returns ``(end_idx, is_empty)``.
    """
    if escaped_bs is None:
        escaped_bs = 1 if escaped_quote else 0
    if scan_state is None:
        scan_state = _ContainerScanState(text)
    cached_closed = scan_state.closed_memo.get(start)
    if cached_closed is not None:
        return cached_closed[0], cached_closed[1]
    cached_failed = scan_state.failed_memo.get(start)
    if cached_failed is not None:
        return cached_failed

    n = len(text)
    # Stack frames: [expected_close, open_pos, has_content, ml_start_idx, p2_pos, p2_mls]
    stack: list[list[Any]] = [
        ["}" if text[start] == "{" else "]", start, False, 0, [], []]
    ]
    ml_newlines: list[int] = []
    i = start + 1
    in_quote: str | None = None
    in_open_bs = 0
    str_open_pos: int | None = None
    str_p2_idx: int | None = None
    multiline_str_fallback: int | None = None
    seen_post_multiline = False
    recording_p2 = False

    def _record_unclosed(q: str, open_bs: int, open_pos: int | None) -> None:
        if open_pos is None:
            return
        eof_key = (q, open_bs)
        prev_eof = scan_state.unclosed_str_eof.get(eof_key)
        if prev_eof is None or open_pos < prev_eof:
            scan_state.unclosed_str_eof[eof_key] = open_pos

    def _flush_p2_path(
        p2_pos: list[int],
        p2_mls: list[tuple[int, int]],
        kind: str,
        target_pos: int,
        expected_close: str | None,
        init_has_content: bool = False,
        init_first_ml: int | None = None,
    ) -> None:
        cur_has_content = init_has_content
        cur_first_ml = init_first_ml
        ml_ptr = len(p2_mls) - 1
        p2_memo = scan_state.p2_memo
        for idx in range(len(p2_pos) - 1, -1, -1):
            packed = p2_pos[idx]
            pos = packed >> 1
            if packed & 1:
                cur_has_content = True
            while ml_ptr >= 0 and p2_mls[ml_ptr][0] >= idx:
                ml_val = p2_mls[ml_ptr][1]
                if cur_first_ml is None or ml_val < cur_first_ml:
                    cur_first_ml = ml_val
                ml_ptr -= 1
            if pos not in p2_memo:
                p2_memo[pos] = (
                    kind,
                    target_pos,
                    expected_close,
                    cur_has_content,
                    cur_first_ml,
                )

    def _finalize_fail(
        fail_pos: int, *, fail_closer: str | None = None
    ) -> tuple[int, bool]:
        ml_idx = 0
        ml_len = len(ml_newlines)
        num_frames = len(stack)
        frame_first_mls: list[int | None] = [None] * num_frames
        for k, (_, frame_open, frame_has_content, _, _, _) in enumerate(stack):
            while ml_idx < ml_len and ml_newlines[ml_idx] <= frame_open:
                ml_idx += 1
            if ml_idx < ml_len:
                fml = ml_newlines[ml_idx]
                frame_first_mls[k] = fml
                scan_state.failed_memo[frame_open] = (fml, False)
            else:
                scan_state.failed_memo[frame_open] = (
                    fail_pos,
                    not frame_has_content,
                )
        if recording_p2:
            top_idx = num_frames - 1
            p2_memo = scan_state.p2_memo
            for k in range(top_idx, -1, -1):
                _, _, _, _, f_p2_pos, f_p2_mls = stack[k]
                if k == top_idx:
                    if f_p2_pos:
                        kind = "closer" if fail_closer is not None else "fail"
                        _flush_p2_path(
                            f_p2_pos,
                            f_p2_mls,
                            kind,
                            fail_pos,
                            fail_closer,
                            False,
                            None,
                        )
                else:
                    child_open = stack[k + 1][1]
                    child_fml = frame_first_mls[k + 1]
                    if child_open not in p2_memo:
                        p2_memo[child_open] = (
                            "fail",
                            fail_pos,
                            None,
                            True,
                            child_fml,
                        )
                    if f_p2_pos:
                        _flush_p2_path(
                            f_p2_pos,
                            f_p2_mls,
                            "fail",
                            fail_pos,
                            None,
                            True,
                            child_fml,
                        )
        if multiline_str_fallback is not None:
            return multiline_str_fallback, False
        return fail_pos, not stack[0][2]

    while i < n:
        if seen_post_multiline:
            scan_state.total_p2_steps += 1
            if (
                not recording_p2
                and scan_state.total_p2_steps > scan_state.p2_memo_threshold
            ):
                recording_p2 = True

        if in_quote is not None:
            stack[-1][2] = True
            qi = scan_state.quote_index
            if qi is not None:
                next_eol = qi.find_next_eol(i - 1)
                close_p = qi.find_close(in_quote, in_open_bs, i - 1)
                if next_eol != -1 and (close_p is None or next_eol < close_p):
                    ml_newlines.append(next_eol)
                    if multiline_str_fallback is None:
                        multiline_str_fallback = next_eol
                    if recording_p2 and str_p2_idx is not None:
                        stack[-1][5].append((str_p2_idx, next_eol))
                    prev_eof = scan_state.unclosed_str_eof.get(
                        (in_quote, in_open_bs)
                    )
                    if (
                        prev_eof is not None
                        and str_open_pos is not None
                        and str_open_pos >= prev_eof
                    ):
                        return _finalize_fail(n)
                    if close_p is None:
                        _record_unclosed(in_quote, in_open_bs, str_open_pos)
                        return _finalize_fail(n)
                    in_quote = None
                    in_open_bs = 0
                    str_open_pos = None
                    str_p2_idx = None
                    seen_post_multiline = True
                    if scan_state.total_p2_steps > scan_state.p2_memo_threshold:
                        recording_p2 = True
                    i = close_p + 1
                    continue
                if close_p is None:
                    _record_unclosed(in_quote, in_open_bs, str_open_pos)
                    return _finalize_fail(n)
                in_quote = None
                in_open_bs = 0
                str_open_pos = None
                str_p2_idx = None
                i = close_p + 1
                continue

            ch = text[i]
            if ch in "\r\n":
                ml_newlines.append(i)
                if multiline_str_fallback is None:
                    multiline_str_fallback = i
                if recording_p2 and str_p2_idx is not None:
                    stack[-1][5].append((str_p2_idx, i))
                prev_eof = scan_state.unclosed_str_eof.get(
                    (in_quote, in_open_bs)
                )
                if (
                    prev_eof is not None
                    and str_open_pos is not None
                    and str_open_pos >= prev_eof
                ):
                    return _finalize_fail(n)
                close_p = scan_state.get_quote_index().find_close(
                    in_quote, in_open_bs, i
                )
                if close_p is None:
                    _record_unclosed(in_quote, in_open_bs, str_open_pos)
                    return _finalize_fail(n)
                in_quote = None
                in_open_bs = 0
                str_open_pos = None
                str_p2_idx = None
                seen_post_multiline = True
                if scan_state.total_p2_steps > scan_state.p2_memo_threshold:
                    recording_p2 = True
                i = close_p + 1
                continue
            elif in_open_bs == 0:
                if ch == "\\":
                    if i + 1 < n and text[i + 1] in "\r\n":
                        i += 1
                        continue
                    i += 2
                    continue
                if ch == in_quote:
                    in_quote = None
                    str_open_pos = None
                    str_p2_idx = None
                i += 1
                continue
            else:
                if ch == "\\":
                    bs = 1
                    if i + 8 < n and text[i + 1 : i + 9] == _BS_8:
                        bs += 8
                        while (
                            i + bs + 64 <= n
                            and text[i + bs : i + bs + 64] == _BS_64
                        ):
                            bs += 64
                        while (
                            i + bs + 8 <= n
                            and text[i + bs : i + bs + 8] == _BS_8
                        ):
                            bs += 8
                    while i + bs < n and text[i + bs] == "\\":
                        bs += 1
                    mod = (
                        2 * (in_open_bs + 1)
                        if (in_open_bs & (in_open_bs + 1)) == 0
                        else 4
                    )
                    if i + bs < n and text[i + bs] == in_quote:
                        if bs >= in_open_bs and bs % mod == in_open_bs % mod:
                            in_quote = None
                            in_open_bs = 0
                            str_open_pos = None
                            str_p2_idx = None
                            i += bs + 1
                            continue
                        i += bs + (1 if bs % 2 == 1 else 0)
                        continue
                    if i + bs < n and text[i + bs] in "\r\n":
                        i += bs
                        continue
                    if i + bs < n:
                        i += bs + (1 if bs % 2 == 1 else 0)
                        continue
                    _record_unclosed(in_quote, in_open_bs, str_open_pos)
                    return _finalize_fail(n)
                i += 1
                continue

        ch = text[i]
        if ch in "}]":
            if stack and stack[-1][0] == ch:
                (
                    _,
                    frame_open,
                    frame_has_content,
                    frame_ml_idx,
                    f_p2_pos,
                    f_p2_mls,
                ) = stack.pop()
                first_ml_nl = (
                    ml_newlines[frame_ml_idx]
                    if frame_ml_idx < len(ml_newlines)
                    else None
                )
                frame_end = i + 1
                scan_state.closed_memo[frame_open] = (
                    frame_end,
                    not frame_has_content,
                    first_ml_nl,
                )
                if f_p2_pos:
                    _flush_p2_path(
                        f_p2_pos, f_p2_mls, "closer", i, ch, False, None
                    )
                i = frame_end
                if not stack:
                    return frame_end, not frame_has_content
                if recording_p2:
                    parent_p2_pos = stack[-1][4]
                    if first_ml_nl is not None:
                        stack[-1][5].append((len(parent_p2_pos), first_ml_nl))
                    parent_p2_pos.append((frame_open << 1) | 1)
                continue
            return _finalize_fail(i, fail_closer=ch)

        if scan_state.p2_memo:
            memo_hit = scan_state.p2_memo.get(i)
            if memo_hit is not None:
                kind, target_pos, _exp_close, m_has_content, m_first_ml = (
                    memo_hit
                )
                if m_has_content:
                    stack[-1][2] = True
                if m_first_ml is not None:
                    ml_newlines.append(m_first_ml)
                    if multiline_str_fallback is None:
                        multiline_str_fallback = m_first_ml
                    seen_post_multiline = True
                    if scan_state.total_p2_steps > scan_state.p2_memo_threshold:
                        recording_p2 = True
                if recording_p2:
                    cur_p2_pos = stack[-1][4]
                    if m_first_ml is not None:
                        stack[-1][5].append((len(cur_p2_pos), m_first_ml))
                    cur_p2_pos.append((i << 1) | (1 if m_has_content else 0))
                if kind == "fail":
                    return _finalize_fail(target_pos)
                i = target_pos
                continue

        if ch in " \t\r":
            if recording_p2:
                stack[-1][4].append(i << 1)
            i += 1
            continue
        if ch == "\n":
            j = i + 1
            line_start = j
            while j < n and text[j] in " \t\r\n":
                if text[j] == "\n":
                    line_start = j + 1
                j += 1
            if recording_p2:
                stack[-1][4].append(i << 1)
            if j >= n or not _is_json_continuation_line(text, line_start):
                return _finalize_fail(i)
            i = line_start
            continue

        if ch == "\\":
            bs = 1
            if i + 8 < n and text[i + 1 : i + 9] == _BS_8:
                bs += 8
                while i + bs + 64 <= n and text[i + bs : i + bs + 64] == _BS_64:
                    bs += 64
                while i + bs + 8 <= n and text[i + bs : i + bs + 8] == _BS_8:
                    bs += 8
            while i + bs < n and text[i + bs] == "\\":
                bs += 1
            stack[-1][2] = True
            if bs % 2 == 1 and i + bs < n and text[i + bs] in "\"'":
                q = text[i + bs]
                prev_eof = scan_state.unclosed_str_eof.get((q, bs))
                if prev_eof is not None and i >= prev_eof:
                    first_nl = scan_state.get_quote_index().find_next_eol(
                        i + bs
                    )
                    if recording_p2:
                        if first_nl != -1:
                            stack[-1][5].append(
                                (len(stack[-1][4]), first_nl)
                            )
                        stack[-1][4].append((i << 1) | 1)
                    if first_nl != -1:
                        ml_newlines.append(first_nl)
                        if multiline_str_fallback is None:
                            multiline_str_fallback = first_nl
                    return _finalize_fail(n)
                if recording_p2:
                    str_p2_idx = len(stack[-1][4])
                    stack[-1][4].append((i << 1) | 1)
                else:
                    str_p2_idx = None
                in_quote = q
                in_open_bs = bs
                str_open_pos = i
                i += bs + 1
                continue
            if recording_p2:
                stack[-1][4].append((i << 1) | 1)
            i += bs
            continue
        if ch in "\"'":
            stack[-1][2] = True
            prev_eof = scan_state.unclosed_str_eof.get((ch, 0))
            if prev_eof is not None and i >= prev_eof:
                first_nl = scan_state.get_quote_index().find_next_eol(i)
                if recording_p2:
                    if first_nl != -1:
                        stack[-1][5].append((len(stack[-1][4]), first_nl))
                    stack[-1][4].append((i << 1) | 1)
                if first_nl != -1:
                    ml_newlines.append(first_nl)
                    if multiline_str_fallback is None:
                        multiline_str_fallback = first_nl
                return _finalize_fail(n)
            if recording_p2:
                str_p2_idx = len(stack[-1][4])
                stack[-1][4].append((i << 1) | 1)
            else:
                str_p2_idx = None
            in_quote = ch
            in_open_bs = 0
            str_open_pos = i
            i += 1
            continue
        if ch in "{[":
            stack[-1][2] = True
            cached_sub = scan_state.closed_memo.get(i)
            if cached_sub is not None:
                sub_end, _sub_empty, sub_first_ml_nl = cached_sub
                if recording_p2:
                    if sub_first_ml_nl is not None:
                        stack[-1][5].append(
                            (len(stack[-1][4]), sub_first_ml_nl)
                        )
                    stack[-1][4].append((i << 1) | 1)
                if sub_first_ml_nl is not None:
                    ml_newlines.append(sub_first_ml_nl)
                    if multiline_str_fallback is None:
                        multiline_str_fallback = sub_first_ml_nl
                    seen_post_multiline = True
                    if scan_state.total_p2_steps > scan_state.p2_memo_threshold:
                        recording_p2 = True
                i = sub_end
                continue
            stack.append(
                [
                    "}" if ch == "{" else "]",
                    i,
                    False,
                    len(ml_newlines),
                    [],
                    [],
                ]
            )
            i += 1
            continue
        stack[-1][2] = True
        if recording_p2:
            stack[-1][4].append((i << 1) | 1)
        i += 1

    if in_quote is not None:
        _record_unclosed(in_quote, in_open_bs, str_open_pos)
    return _finalize_fail(n)


_SEP_QUOTE_PREFIX_RE = re.compile(r"^(\\+['\"]|['\"])")

_MULTILINE_AUTH_PROSE_STOPWORDS: frozenset[str] = frozenset(
    {
        "auth",
        "authentication",
        "authorization",
        "check",
        "checks",
        "config",
        "configuration",
        "denied",
        "detail",
        "details",
        "error",
        "errors",
        "example",
        "examples",
        "expired",
        "failed",
        "failure",
        "for",
        "format",
        "guide",
        "header",
        "headers",
        "help",
        "info",
        "invalid",
        "is",
        "key",
        "level",
        "message",
        "missing",
        "mode",
        "not",
        "note",
        "ok",
        "option",
        "options",
        "password",
        "required",
        "retry",
        "rule",
        "rules",
        "scheme",
        "secret",
        "setting",
        "settings",
        "setup",
        "status",
        "step",
        "steps",
        "test",
        "testing",
        "timeout",
        "token",
        "type",
        "usage",
        "validate",
        "validation",
        "value",
        "values",
        "warning",
        "was",
        "with",
    }
)


def _container_redacted_token(sep: str) -> str:
    qm = _SEP_QUOTE_PREFIX_RE.match(sep)
    if qm is not None:
        tok = qm.group(1)
        return f"{tok}{_REDACTED}{tok}"
    return _REDACTED


def _is_multiline_auth_redactable(key: str, vm: re.Match[str]) -> bool:
    if _normalize_key_name(key) in {"authorization", "proxy_authorization"}:
        return True
    if vm.group("ml_auth_rest") is not None:
        return True
    ml_2 = vm.group("ml_auth_2")
    if ml_2 is None:
        return True
    parts = ml_2.split(None, 1)
    if len(parts) < 2:
        return False
    scheme = parts[0].lower().replace("-", "").replace("_", "")
    cred = parts[1]
    if cred[0] in "\"'\\" or (
        cred.lower().startswith(("token", "key")) and ("=" in cred or ":" in cred)
    ):
        return True
    if cred.lower() in _MULTILINE_AUTH_PROSE_STOPWORDS:
        return False
    if any(c.isdigit() or c in "+/=._-" for c in cred) or any(
        c.isupper() for c in cred[1:]
    ):
        return True
    if scheme in {
        "bearer",
        "apikey",
        "jwt",
        "ssws",
        "dpop",
        "hmac",
        "hmacsha1",
        "hmacsha256",
        "hmacsha384",
        "hmacsha512",
    }:
        return True
    if len(cred) >= 6 and all(c.isalnum() for c in cred):
        return True
    if not all(c.isalnum() or c in "+/=._-" for c in cred):
        return ":" in cred
    return False


def _redact_kv_and_auth(text: str) -> str:
    """Scan key/value and auth-scheme secrets in strict O(N) without non-sensitive swallowing."""
    if ":" not in text and "=" not in text:
        return text
    n = len(text)
    out_parts: list[str] = []
    out_cursor = 0
    scan_pos = 0
    changed_any = False
    scan_state: _ContainerScanState | None = None

    while scan_pos < n:
        hm = _KV_HEAD_RE.search(text, scan_pos)
        if hm is None:
            break
        key = hm.group("key")
        sep = hm.group("sep")
        val_start = hm.end()

        if not _is_sensitive_key(key):
            scan_pos = val_start
            continue

        sep_bs = 0
        while sep_bs < len(sep) and sep[sep_bs] == "\\":
            sep_bs += 1
        if not (sep_bs < len(sep) and sep[sep_bs] in "\"'"):
            sep_bs = 0

        lit_start = val_start
        if lit_start < n and text[lit_start] in "\r\n":
            nlm = _NL_WS_RE.match(text, lit_start)
            if nlm is not None and nlm.end() < n:
                lit_start = nlm.end()

        # Check container values ({...}, [...], or util.inspect Map/Set(N) {...}),
        # optionally after one newline+indent.
        container_start: int | None = None
        container_replace_start: int = lit_start
        if lit_start < n and text[lit_start] in "{[":
            container_start = lit_start
        elif lit_start < n and text[lit_start] in "MS":
            im = _INSPECT_COLLECTION_RE.match(text, lit_start)
            if im is not None:
                container_start = im.end() - 1

        if container_start is not None:
            if scan_state is None:
                scan_state = _ContainerScanState(text)
            end_idx, is_empty = _scan_container_value(
                text,
                container_start,
                escaped_quote=(sep_bs > 0),
                escaped_bs=sep_bs,
                scan_state=scan_state,
            )
            if not is_empty:
                out_parts.append(text[out_cursor:container_replace_start])
                out_parts.append(_container_redacted_token(sep))
                out_cursor = end_idx
                changed_any = True
            scan_pos = end_idx
            continue

        # Handle arbitrary-depth backslash escaped-quote string literals in O(N)
        # (both odd 1x/3x/5x and even 2x/4x backslashes, closed or truncated).
        lit_bs = 0
        while lit_start + lit_bs < n and text[lit_start + lit_bs] == "\\":
            lit_bs += 1
        if (
            lit_bs > 0
            and lit_start + lit_bs < n
            and text[lit_start + lit_bs] in "\"'"
        ):
            q = text[lit_start + lit_bs]
            end_idx, already_safe, close_delim = _scan_escaped_quote_literal(
                text, lit_start, lit_bs, q
            )
            if not already_safe:
                open_delim = text[lit_start : lit_start + lit_bs + 1]
                out_parts.append(text[out_cursor:lit_start])
                out_parts.append(f"{open_delim}{_REDACTED}{close_delim}")
                out_cursor = end_idx
                changed_any = True
            scan_pos = end_idx
            continue

        # Handle auth-scheme + backslash-escaped quotes (e.g. Authorization: Bearer \"...\" or \\"...")
        if lit_start < n and text[lit_start].isalpha():
            am = _AUTH_2_BS_QUOTE_RE.match(text, lit_start)
            if am is not None:
                bs_len = len(am.group("bs"))
                q = am.group("q")
                end_idx, _already_safe, _ = _scan_escaped_quote_literal(
                    text, am.start("bs"), bs_len, q
                )
                out_parts.append(text[out_cursor:lit_start])
                out_parts.append(_REDACTED)
                out_cursor = end_idx
                changed_any = True
                scan_pos = end_idx
                continue

        vm = _KV_VAL_RE.match(text, val_start)
        if vm is None:
            scan_pos = val_start
            continue

        nl_ws = vm.group("nl_ws") or ""
        if nl_ws and (vm.group("ml_auth_rest") or vm.group("ml_auth_2")):
            if not _is_multiline_auth_redactable(key, vm):
                scan_pos = val_start
                continue

        replacement: str | None = None
        for grp, wrap_open, wrap_close in (
            ("sq_dq", '"', '"'),
            ("ml_dq", '"', '"'),
            ("sq_sq", "'", "'"),
            ("ml_sq", "'", "'"),
        ):
            val = vm.group(grp)
            if val is not None:
                inner = val[len(wrap_open) : len(val) - len(wrap_close)]
                if inner == "" or inner == _REDACTED:
                    replacement = None
                else:
                    replacement = f"{wrap_open}{_REDACTED}{wrap_close}"
                break
        else:
            for grp, wrap_open, wrap_close in (
                ("sq_unclosed_dq", '"', '"'),
                ("ml_unclosed_dq", '"', '"'),
                ("sq_unclosed_sq", "'", "'"),
                ("ml_unclosed_sq", "'", "'"),
            ):
                if vm.group(grp) is not None:
                    replacement = f"{wrap_open}{_REDACTED}{wrap_close}"
                    break
            else:
                for grp in (
                    "sq_auth_rest",
                    "sq_auth_2",
                    "sq_unq",
                    "ml_auth_rest",
                    "ml_auth_2",
                ):
                    val = vm.group(grp)
                    if val is not None:
                        if val != _REDACTED:
                            replacement = _REDACTED
                        break

        if replacement is not None:
            val_replace_start = val_start + len(nl_ws)
            out_parts.append(text[out_cursor:val_replace_start])
            out_parts.append(replacement)
            out_cursor = vm.end()
            changed_any = True

        scan_pos = vm.end()

    if not changed_any:
        return text
    out_parts.append(text[out_cursor:])
    return "".join(out_parts)


# ---------------------------------------------------------------------------
# Stage 3: Line-based traceback/frame stripping, PEM, paths & anchored cleanup
# ---------------------------------------------------------------------------

_PY_FILE_FRAME_RE = re.compile(r'^File "[^"\r\n]+", line \d+(?:, in [^\r\n]+)?')
_NODE_END_LOC_RE = re.compile(r":\d{1,6}:\d{1,6}\)?$")

_PATH_EXT_RE = re.compile(
    r"(?<![:/\w.\-])/(?:[\w.\-]+/)+[\w.\-]+\."
    r"(?:py|js|mjs|cjs|ts|tsx|jsx|json|log|env|yaml|yml|toml|sh|txt|md)"
    r"(?::\d+(?::\d+)?)?"
)
_PATH_DEEP_RE = re.compile(r"(?<![:/\w.\-])/(?:[\w.\-]+/){2,}[\w.\-]+(?::\d+(?::\d+)?)?")
_WIN_PATH_RE = re.compile(r"[A-Za-z]:\\(?:[\w.\-]+\\)+[\w.\-]+(?::\d+)?")
_SRC_MOD_RE = re.compile(r"\bsrc\.\w+(?:\.\w+)*\b")

_EMPTY_QUOTES_RE = re.compile(r"(?<![\s:=])[ \t]+(['\"])\1(?=[ \t\r\n]|$)")
_TRAILING_WS_RE = re.compile(r"(?<![ \t])[ \t]+(\r?\n)")
_MULTI_NL_RE = re.compile(r"\n{3,}")


def _is_node_at_frame(stripped: str) -> bool:
    if not stripped.startswith("at "):
        return False
    if "(node:" in stripped or stripped.startswith("at node:") or stripped.startswith("at /"):
        return True
    trimmed = stripped.rstrip(" \t")
    if _NODE_END_LOC_RE.search(trimmed):
        return True
    if " (" in trimmed and trimmed.endswith(")") and (
        "/" in trimmed or "\\" in trimmed or "node:" in trimmed
    ):
        return True
    return False


def _strip_tracebacks_and_frames(text: str) -> str:
    if (
        "Traceback" not in text
        and 'File "' not in text
        and "at " not in text
    ):
        return text

    out_lines: list[str] = []
    skip_frame_indent: int | None = None

    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        stripped = body.lstrip(" \t")
        indent = len(body) - len(stripped)

        if skip_frame_indent is not None:
            if (
                stripped
                and indent > skip_frame_indent
                and not stripped.startswith('File "')
                and not _is_node_at_frame(stripped)
                and not stripped.startswith("Traceback (most recent call last):")
            ):
                continue
            skip_frame_indent = None

        if stripped.startswith("Traceback (most recent call last):") and len(stripped) < 80:
            continue
        if _PY_FILE_FRAME_RE.match(stripped):
            skip_frame_indent = indent
            continue
        if _is_node_at_frame(stripped):
            continue
        out_lines.append(line)

    return "".join(out_lines)


def _empty_quotes_repl(m: re.Match[str]) -> str:
    s = m.string
    end = m.start()
    i = end
    while i > 0 and (s[i - 1].isalnum() or s[i - 1] in "_-"):
        i -= 1
    if end - i == 6 and s[i:end].lower() == "bearer":
        return m.group(0)
    return ""


def _find_next_valid_pem_end(text: str, start_pos: int) -> tuple[int, int]:
    """Return ``(end_marker_start, close_end)`` of the next complete same-line ``-----END ...-----``."""
    n = len(text)
    pos = start_pos
    while True:
        end_marker = text.find("-----END ", pos)
        if end_marker == -1:
            return -1, -1
        tail = text.find("-----", end_marker + 9)
        if tail == -1:
            return -1, -1
        between = text[end_marker + 9 : tail]
        if "\n" in between or "\r" in between:
            # Incomplete ``-----END ...`` lacking closing ``-----`` before EOL;
            # never consume across lines.
            pos = tail
            continue
        dash_end = tail + 5
        while dash_end < n and text[dash_end] == "-":
            dash_end += 1
        if text.startswith("BEGIN ", dash_end) and (dash_end - tail) < 10:
            # 5..9 dashes belong to an adjacent same-line ``-----BEGIN `` header;
            # never consume its leading dashes.
            pos = dash_end - 5
            continue
        return end_marker, tail + 5


def _strip_pem_blocks(text: str) -> str:
    if "-----BEGIN " not in text:
        return text
    n = len(text)
    out: list[str] = []
    cursor = 0
    cached_end: tuple[int, int] | None = None
    cached_next_begin = -1
    no_more_end = False
    checked_no_private = False

    while True:
        start = text.find("-----BEGIN ", cursor)
        if start == -1:
            out.append(text[cursor:])
            break
        header_end = text.find("-----", start + 11)
        if header_end == -1:
            nl_pos = text.find("\n", start + 11)
            cr_pos = text.find("\r", start + 11)
            line_end = n
            if nl_pos != -1:
                line_end = min(line_end, nl_pos)
            if cr_pos != -1:
                line_end = min(line_end, cr_pos)
            label = text[start + 11 : line_end].strip().upper()
            if "PRIVATE KEY" in label or "SECRET KEY" in label:
                out.append(text[cursor:start])
                out.append(_REDACTED)
                break
            out.append(text[cursor:])
            break

        dash_end = header_end + 5
        while dash_end < n and text[dash_end] == "-":
            dash_end += 1
        adjacent_begin = (dash_end - header_end) < 10 and text.startswith(
            "BEGIN ", dash_end
        )

        if (
            text.find("\n", start + 11, header_end) != -1
            or text.find("\r", start + 11, header_end) != -1
            or adjacent_begin
        ):
            header_span = text[start + 11 : header_end]
            nl_pos = header_span.find("\n")
            cr_pos = header_span.find("\r")
            rel_end = len(header_span)
            if nl_pos != -1:
                rel_end = min(rel_end, nl_pos)
            if cr_pos != -1:
                rel_end = min(rel_end, cr_pos)
            label = header_span[:rel_end].strip().upper()
            if "PRIVATE KEY" in label or "SECRET KEY" in label:
                search_from = start + 11 + rel_end
                if not no_more_end and (cached_end is None or cached_end[0] < search_from):
                    cached_end = _find_next_valid_pem_end(text, search_from)
                    if cached_end[0] == -1:
                        no_more_end = True
                end_marker, close_end = (-1, -1) if no_more_end else cached_end  # type: ignore[misc]
                out.append(text[cursor:start])
                out.append(_REDACTED)
                if end_marker == -1:
                    break
                cursor = close_end
                continue
            next_cursor = (dash_end - 5) if adjacent_begin else header_end
            out.append(text[cursor:next_cursor])
            cursor = next_cursor
            continue

        label = text[start + 11 : header_end].strip().upper()
        is_private_key = "PRIVATE KEY" in label or "SECRET KEY" in label

        if not no_more_end and (cached_end is None or cached_end[0] < header_end + 5):
            cached_end = _find_next_valid_pem_end(text, header_end + 5)
            if cached_end[0] == -1:
                no_more_end = True

        end_marker, close_end = (-1, -1) if no_more_end else cached_end  # type: ignore[misc]
        if end_marker == -1:
            if is_private_key:
                out.append(text[cursor:start])
                out.append(_REDACTED)
                break
            out.append(text[cursor : header_end + 5])
            cursor = header_end + 5
            if not checked_no_private:
                checked_no_private = True
                rem_lower = text[cursor:].lower()
                if "private" not in rem_lower and "secret" not in rem_lower:
                    out.append(text[cursor:])
                    break
            continue

        if not is_private_key:
            if cached_next_begin != -2 and cached_next_begin < header_end + 5:
                nb = text.find("-----BEGIN ", header_end + 5)
                cached_next_begin = -2 if nb == -1 else nb
            if cached_next_begin != -2 and cached_next_begin < end_marker:
                out.append(text[cursor : header_end + 5])
                cursor = header_end + 5
                continue

        out.append(text[cursor:start])
        out.append(_REDACTED)
        cursor = close_end
    return "".join(out)


def _strip_non_top_json(
    text: str,
    *,
    depth: int = 0,
    max_depth: int = _MAX_JSON_DEPTH,
) -> str:
    cleaned = _redact_escaped_quoted_strings(text, depth=depth, max_depth=max_depth)
    cleaned = _strip_pem_blocks(cleaned)
    cleaned = _redact_kv_and_auth(cleaned)
    cleaned = _redact_standalone_bearer(cleaned)
    if "-" in cleaned or "AKIA" in cleaned:
        cleaned = _TOKEN_PREFIX_RE.sub(_REDACTED, cleaned)
    cleaned = _strip_tracebacks_and_frames(cleaned)
    if "/" in cleaned:
        cleaned = _PATH_EXT_RE.sub("", cleaned)
        cleaned = _PATH_DEEP_RE.sub("", cleaned)
    if "\\" in cleaned:
        cleaned = _WIN_PATH_RE.sub("", cleaned)
    if "src." in cleaned:
        cleaned = _SRC_MOD_RE.sub("", cleaned)
    if "''" in cleaned or '""' in cleaned:
        cleaned = _EMPTY_QUOTES_RE.sub(_empty_quotes_repl, cleaned)
    if "\n" in cleaned:
        cleaned = _TRAILING_WS_RE.sub(r"\1", cleaned)
        cleaned = _MULTI_NL_RE.sub("\n\n", cleaned)
    return cleaned


def _strip_pipeline(
    text: str,
    *,
    depth: int = 0,
    max_depth: int = _MAX_JSON_DEPTH,
) -> str:
    cleaned = _redact_embedded_json(text, depth=depth, max_depth=max_depth)
    return _strip_non_top_json(cleaned, depth=depth, max_depth=max_depth)


def _strip_internal_details(raw: str | None) -> str:
    """Remove secrets, file paths, tracebacks, and internal module names from *raw*."""
    if not raw:
        return ""
    return _strip_pipeline(str(raw), depth=0, max_depth=_MAX_JSON_DEPTH).strip()


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
    (
        ErrorCategory.INVALID_ARGUMENT,
        (
            "outside confirmed agent_pool",
            "agent_id is required",
            "invalid binding",
            "invalid effort",
            "invalid profile",
            "unknown teamwork preset",
        ),
        False,
    ),
    (
        ErrorCategory.RUNTIME_TIMEOUT,
        ("timeout", "timed out", "deadline exceeded", "deadline exhausted", "exceeded total timeout", "超时"),
        False,
    ),
    (ErrorCategory.CANCELLED, ("cancelled", "canceled", "已取消"), False),
]


def format_structured_error(
    raw: str,
    category: ErrorCategory | None = None,
) -> WorkflowUserError:
    """Categorize (if needed) and sanitize a raw workflow error."""
    resolved = category if category is not None else categorize_error(raw)
    return _sanitize_error(raw, resolved)


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
