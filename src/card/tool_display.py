"""Helpers for turning raw tool payloads into card-safe display text."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from src.utils.redact import redact_sensitive
from src.utils.text import sanitize_single_line_label

_MAX_LABEL_CHARS = 80
_JSON_EDGE_LINES = {"{", "}", "[", "]"}
_AGENT_TOOL_TITLES = {"agent", "subagent", "task"}
_OPAQUE_CALL_ID_RE = re.compile(r"(?<!\w)call_[A-Za-z0-9_-]+", re.IGNORECASE)
_UUID_LIKE_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_LITERAL_ESCAPE_RE = re.compile(r"""\\(?:["'nrtbfv0])""")
_CONTROL_SEPARATOR_RE = re.compile(r"(?:\\[nrtbfv0]|[\x00-\x1f\x7f])")
_INLINE_STRUCTURED_RE = re.compile(
    r"""[{\[]\s*["'][^"']+["']\s*:""",
    re.IGNORECASE,
)
_LEADING_STRUCTURED_MEMBER_RE = re.compile(
    r"""^\s*["'][^"']+["']\s*:""",
    re.IGNORECASE,
)
_MALFORMED_ARRAY_RE = re.compile(
    r"""^\[\s*(?:\{|\[|["'])"""
)
_CODE_FRAGMENT_RE = re.compile(
    r"(?:"
    r"^\s*(?:assert|class|def|elif|else|except|for|from|if|import|lambda|raise|return|try|while|with)\b"
    r"|\b(?:is\s+not|not\s+in)\b"
    r"|```"
    r"|==|!=|<=|>=|:=|=>|&&|\|\|"
    r")",
    re.IGNORECASE,
)
_ANSI_ESCAPE_RE = re.compile(
    r"(?:"
    r"(?:\x1b\]|\x9d)[\s\S]*?(?:\x07|\x1b\\|\x9c|$)"
    r"|(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]"
    r"|(?:\x1b[P^_]|\x90|\x98|\x9e|\x9f)"
    r"[\s\S]*?(?:\x1b\\|\x9c|$)"
    r"|\x1b[@-_]"
    r")",
)
_UNSAFE_FORMAT_CONTROL_RE = re.compile(
    r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]"
)
_UNSAFE_PAYLOAD_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SUBAGENT_INTERNAL_PATH_RE = re.compile(
    r"(?<![\w])(?:"
    r"[A-Za-z]:[\\/](?:[\w.@+~-]+[\\/])+[\w.@+~-]+(?::\d+)?"
    r"|(?:/[\w.@+~-]+){2,}(?::\d+)?"
    r")"
)
_FAILURE_KEYS = (
    "error",
    "error_message",
    "message",
    "reason",
    "detail",
    "stderr",
)
_FAILURE_IGNORED_KEYS = frozenset(
    {
        "call_id",
        "command",
        "cmd",
        "id",
        "input",
        "output",
        "prompt",
        "raw_output",
        "stdout",
    }
)
_OPAQUE_VALUE_KEYS = frozenset(
    {
        "call_id",
        "id",
        "request_id",
        "tool_call_id",
        "tool_use_id",
    }
)
_SENSITIVE_STRUCTURED_KEY_RE = re.compile(
    r"(?:TOKEN|SECRET|PASSWORD|PASSWD|KEY|CREDENTIAL|AUTHORIZATION|COOKIE)",
    re.IGNORECASE,
)
_INTERNAL_AGENT_ID_KEYS = frozenset(
    {
        "agentthreadid",
        "receiverthreadid",
        "receiverthreadids",
        "senderthreadid",
        "sourceid",
        "subagentsourceid",
        "threadid",
        "toolcallid",
        "tooluseid",
    }
)
_INTERNAL_AGENT_PATH_KEYS = frozenset(
    {
        "agentpath",
        "subagentpath",
    }
)
_AGENT_STATE_CONTAINER_KEYS = frozenset(
    {
        "agentsstates",
        "subagentstates",
    }
)
_QUOTED_SENSITIVE_VALUE_RE = re.compile(
    r'''(?i)(["'][^"']*(?:TOKEN|SECRET|PASSWORD|PASSWD|KEY|CREDENTIAL|AUTHORIZATION|COOKIE)[^"']*["']\s*:\s*)'''
    r'''(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^,\]}\s]+)'''
)
_COMMAND_VALUE_KEYS = frozenset({"cmd", "command"})
_MARKDOWN_META_TRANSLATION = str.maketrans(
    {
        "<": "＜",
        "\\": "＼",
        "`": "ˋ",
        "*": "＊",
        "_": "＿",
        "[": "［",
        "]": "］",
        "(": "（",
        ")": "）",
        "!": "！",
        "#": "＃",
        ">": "＞",
        "~": "～",
    }
)

_READ_TYPES = {"read", "read_file", "cat", "head", "tail", "list", "ls", "tree"}
_SEARCH_TYPES = {"grep", "search", "find", "glob", "search_codebase"}
_EDIT_TYPES = {
    "write",
    "write_file",
    "edit",
    "edit_file",
    "multi_edit",
    "patch",
    "apply_patch",
    "apply_diff",
    "delete",
    "delete_file",
}
_RUN_TYPES = {"run", "exec", "execute", "shell", "bash", "command", "read"}


def summarize_tool_call_content(content: str, *, fallback: str = "", max_chars: int = _MAX_LABEL_CHARS) -> str:
    """Return concise readable text for a tool payload.

    Structured tool JSON often contains noisy metadata plus stdout/stderr. Cards
    need a human action summary instead of the raw payload.
    """
    text = str(content or "").strip()
    fallback = str(fallback or "").strip()
    if not text:
        return _truncate(fallback, max_chars)

    parsed = _parse_json(text.strip())
    if parsed is not None:
        summary = _describe_json_payload(parsed) or fallback
        return _truncate(_first_display_line(summary) or fallback, max_chars)

    first_line = _first_display_line(text)
    return _truncate(first_line or fallback, max_chars)


def sanitize_tool_event_content(content: str, *, fallback: str = "") -> str:
    """Clean tool input/output before it enters renderable card state."""
    text = str(content or "").strip()
    parsed = _parse_json(text)
    if parsed is None:
        return text
    return summarize_tool_call_content(text, fallback=fallback, max_chars=160)


def sanitize_full_tool_event_content(
    content: object,
    *,
    opaque_ids: Iterable[object] = (),
) -> str:
    """Preserve a tool payload for paged display while removing unsafe data.

    Unlike :func:`sanitize_tool_event_content`, this helper deliberately does
    not summarize structured JSON. Workflow execution pages paginate the full
    safe projection, so collapsing it here would make later pagination unable
    to recover the omitted input or output.
    """
    if isinstance(content, str):
        text = content
    elif content is None:
        text = ""
    else:
        try:
            text = json.dumps(content, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(content)
    # Providers may prefix structured payloads with terminal color escapes,
    # BOMs, bidi marks, or other non-content controls. Normalize those before
    # JSON detection so they cannot bypass recursive secret-key redaction.
    text = _ANSI_ESCAPE_RE.sub("", text)
    text = _UNSAFE_FORMAT_CONTROL_RE.sub("", text)
    text = _UNSAFE_PAYLOAD_CONTROL_RE.sub("", text)
    parsed = _parse_json(text.strip())
    structured_opaque_values = _collect_structured_internal_values(parsed)
    if parsed is not None:
        text = json.dumps(
            _redact_structured_secrets(parsed),
            ensure_ascii=False,
            default=str,
        )
    text = _remove_opaque_values(
        text,
        _normalize_opaque_values((*opaque_ids, *structured_opaque_values)),
    )
    text = _OPAQUE_CALL_ID_RE.sub("[redacted:call_id]", text)
    text = _QUOTED_SENSITIVE_VALUE_RE.sub(r'\1"<redacted>"', text)
    text = redact_sensitive(text)
    return text


def _normalized_structured_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _redacted_agent_ids(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            (
                "<redacted:agent_id>"
                if index == 0
                else f"<redacted:agent_id:{index + 1}>"
            ): _redact_structured_secrets(item, inside_agent_state=True)
            for index, item in enumerate(value.values())
        }
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return ["<redacted:agent_id>" for _ in value]
    if value is None:
        return None
    return "<redacted:agent_id>"


def _redact_agent_state_map(value: Any) -> Any:
    if not isinstance(value, Mapping):
        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            return [
                _redact_structured_secrets(item, inside_agent_state=True)
                for item in value
            ]
        return "<redacted:agent_state>" if value is not None else None
    return {
        (
            "<redacted:agent_id>"
            if index == 0
            else f"<redacted:agent_id:{index + 1}>"
        ): _redact_structured_secrets(item, inside_agent_state=True)
        for index, item in enumerate(value.values())
    }


def _redact_structured_secrets(
    value: Any,
    *,
    inside_agent_state: bool = False,
) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            text_key = str(key)
            normalized_key = _normalized_structured_key(text_key)
            if _SENSITIVE_STRUCTURED_KEY_RE.search(text_key):
                safe_item = "<redacted>"
            elif normalized_key in _INTERNAL_AGENT_ID_KEYS:
                safe_item = _redacted_agent_ids(item)
            elif normalized_key in _INTERNAL_AGENT_PATH_KEYS or (
                inside_agent_state and normalized_key == "path"
            ):
                safe_item = (
                    None if item is None else "<redacted:agent_path>"
                )
            elif normalized_key == "agentsstates":
                safe_item = _redact_agent_state_map(item)
            else:
                safe_item = _redact_structured_secrets(
                    item,
                    inside_agent_state=(
                        inside_agent_state
                        or normalized_key in _AGENT_STATE_CONTAINER_KEYS
                    ),
                )
            redacted[text_key] = safe_item
        return redacted
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            _redact_structured_secrets(
                item,
                inside_agent_state=inside_agent_state,
            )
            for item in value
        ]
    if isinstance(value, str):
        nested = _parse_json(value.strip())
        if nested is not None:
            return json.dumps(
                _redact_structured_secrets(
                    nested,
                    inside_agent_state=inside_agent_state,
                ),
                ensure_ascii=False,
                default=str,
            )
        return redact_sensitive(value)
    return value


def sanitize_tool_failure_detail(
    content: object,
    *,
    fallback: str = "子任务执行失败",
    max_chars: int = 160,
    opaque_ids: Iterable[object] = (),
    allow_unstructured: bool = True,
) -> str:
    """Extract one bounded, markdown-neutral failure reason.

    Structured tool payloads may contain full stdout, opaque call IDs, command
    metadata, and terminal control sequences. Only explicit error fields are
    eligible when ``allow_unstructured`` is false; everything else is reduced
    to the caller-provided fallback.
    """

    if isinstance(content, Mapping) or (
        isinstance(content, Sequence)
        and not isinstance(content, (str, bytes, bytearray))
    ):
        text = ""
        parsed = content
    else:
        text = str(content or "").strip()
        parsed = _parse_json(text)
    candidate = (
        _find_failure_text(parsed)
        if parsed is not None
        else text if allow_unstructured else ""
    )
    if not candidate:
        candidate = str(fallback or "")
    candidate = _ANSI_ESCAPE_RE.sub("", candidate)
    candidate = _remove_opaque_values(
        candidate,
        (
            *_collect_opaque_values(parsed),
            *_normalize_opaque_values(opaque_ids),
        ),
    )
    candidate = _OPAQUE_CALL_ID_RE.sub("", candidate)
    candidate = redact_sensitive(candidate)
    candidate = _CONTROL_SEPARATOR_RE.sub(" ", candidate)
    candidate = _UNSAFE_FORMAT_CONTROL_RE.sub("", candidate)
    candidate = sanitize_single_line_label(
        candidate,
        fallback=fallback,
        max_chars=max_chars + 1,
    )
    candidate = candidate.translate(_MARKDOWN_META_TRANSLATION)
    return _truncate(candidate, max_chars)


def collect_subagent_opaque_ids(tool_call: Any) -> tuple[str, ...]:
    """Collect provider-only child/call identifiers from one normalized frame."""
    if tool_call is None:
        return ()
    values: list[object] = [
        getattr(tool_call, "id", None),
        getattr(tool_call, "subagent_source_id", None),
    ]
    receivers = getattr(tool_call, "collaboration_receivers", ())
    if isinstance(receivers, Sequence) and not isinstance(
        receivers,
        (str, bytes, bytearray),
    ):
        values.extend(receivers)
    states = getattr(tool_call, "subagent_states", ())
    if isinstance(states, Sequence) and not isinstance(
        states,
        (str, bytes, bytearray),
    ):
        values.extend(
            item.get("source_id")
            for item in states
            if isinstance(item, Mapping)
        )
    return tuple(dict.fromkeys(_normalize_opaque_values(values)))


def collect_subagent_opaque_values(tool_call: Any) -> tuple[str, ...]:
    """Collect child IDs and paths for Workflow's full tool projection."""
    values: list[object] = [
        *collect_subagent_opaque_ids(tool_call),
        getattr(tool_call, "subagent_path", None),
    ]
    values.extend(
        _collect_structured_internal_values(
            getattr(tool_call, "subagent_states", ())
        )
    )
    values.extend(
        _collect_structured_internal_values(
            getattr(tool_call, "full_content", None)
        )
    )
    return tuple(dict.fromkeys(_normalize_opaque_values(values)))


def sanitize_subagent_display_text(
    content: object,
    *,
    fallback: str = "",
    max_chars: int = 180,
    opaque_ids: Iterable[object] = (),
) -> str:
    """Return bounded child progress safe for cards and shared summaries.

    SubAgent messages are provider-controlled status hints, not trusted output.
    Besides the general credential/control/Markdown neutralization, hide local
    absolute paths and every known peer thread/call identifier in the frame.
    """
    text = _SUBAGENT_INTERNAL_PATH_RE.sub("<path>", str(content or ""))
    return sanitize_tool_failure_detail(
        text,
        fallback=fallback,
        max_chars=max_chars,
        opaque_ids=opaque_ids,
    )


def extract_tool_call_label(
    tool_call: Any,
    *,
    generic_labels: Iterable[str] = (),
    fallback: str = "子任务",
    max_chars: int = 60,
) -> str:
    """Extract a task/subagent label without leaking structured JSON."""
    title = str(getattr(tool_call, "title", "") or "").strip()
    content = str(getattr(tool_call, "content", "") or "").strip()
    generic = {str(item or "").strip().lower() for item in generic_labels}

    label = summarize_tool_call_content(content, fallback="", max_chars=max_chars)
    if label and not is_unhelpful_display_label(label):
        return label

    if title and title.lower() not in generic and not is_unhelpful_display_label(title):
        return _truncate(title, max_chars)
    safe_fallback = str(fallback or "").strip()
    if is_unhelpful_display_label(safe_fallback):
        safe_fallback = "子任务"
    return _truncate(safe_fallback, max_chars)


def extract_agent_tool_name(
    tool_call: Any,
    *,
    fallback: str = "子代理",
    max_chars: int = 24,
) -> str:
    """Extract a concise agent identity without exposing source fragments."""
    content = str(getattr(tool_call, "content", "") or "").strip()
    marker = "子代理："
    for line in content.splitlines():
        stripped_line = line.strip()
        if not stripped_line.startswith(marker):
            continue
        candidate = stripped_line[len(marker):].strip()
        candidate = _CONTROL_SEPARATOR_RE.split(candidate, maxsplit=1)[0].strip()
        if (
            candidate
            and re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,31}", candidate)
            and not _INLINE_STRUCTURED_RE.search(candidate)
            and not is_unhelpful_display_label(candidate)
        ):
            return _truncate(candidate, max_chars)

    title = str(getattr(tool_call, "title", "") or "").strip()
    if (
        title
        and not _INLINE_STRUCTURED_RE.search(title)
        and not is_unhelpful_display_label(title)
    ):
        normalized_title = title.lower()
        return _truncate(normalized_title if normalized_title in _AGENT_TOOL_TITLES else title, max_chars)

    safe_fallback = str(fallback or "").strip()
    if is_unhelpful_display_label(safe_fallback):
        safe_fallback = "子代理"
    return _truncate(safe_fallback, max_chars)


def is_unhelpful_display_label(value: str) -> bool:
    """Whether a display label is just JSON syntax or empty noise."""
    text = str(value or "").strip()
    if not text:
        return True
    if text in _JSON_EDGE_LINES:
        return True
    if _OPAQUE_CALL_ID_RE.search(text):
        return True
    if _UUID_LIKE_RE.fullmatch(text):
        return True
    if _LITERAL_ESCAPE_RE.search(text):
        return True
    if text.startswith(("{", "[")) and _parse_json(text) is not None:
        return True
    if text.startswith("{") or _MALFORMED_ARRAY_RE.search(text):
        return True
    if _LEADING_STRUCTURED_MEMBER_RE.search(text):
        return True
    if _INLINE_STRUCTURED_RE.search(text):
        return True
    if _CODE_FRAGMENT_RE.search(text):
        return True
    return False


def _parse_json(text: str) -> Any | None:
    if not text or text[0] not in "{[":
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _find_failure_text(value: Any) -> str:
    if isinstance(value, Mapping):
        normalized = {
            str(key).strip().lower(): item
            for key, item in value.items()
        }
        for key in _FAILURE_KEYS:
            if key not in normalized:
                continue
            item = normalized[key]
            if isinstance(item, str) and item.strip():
                return item.strip()
            nested = _find_failure_text(item)
            if nested:
                return nested
        for key, item in normalized.items():
            if key in _FAILURE_IGNORED_KEYS:
                continue
            if not isinstance(item, Mapping) and not (
                isinstance(item, Sequence)
                and not isinstance(item, (str, bytes, bytearray))
            ):
                continue
            nested = _find_failure_text(item)
            if nested:
                return nested
        return ""
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for item in value:
            nested = _find_failure_text(item)
            if nested:
                return nested
    return ""


def _collect_opaque_values(value: Any) -> tuple[str, ...]:
    values: list[str] = []

    def add(item: object) -> None:
        text = str(item if item is not None else "").strip()
        if text and text not in values:
            values.append(text)

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for raw_key, nested in item.items():
                key = str(raw_key).strip().lower()
                is_nested_collection = isinstance(nested, Mapping) or (
                    isinstance(nested, Sequence)
                    and not isinstance(nested, (str, bytes, bytearray))
                )
                if key in _OPAQUE_VALUE_KEYS and not is_nested_collection:
                    add(nested)
                elif key in _COMMAND_VALUE_KEYS:
                    for command_value in _command_opaque_values(nested):
                        add(command_value)
                if is_nested_collection:
                    visit(nested)
        elif isinstance(item, Sequence) and not isinstance(
            item,
            (str, bytes, bytearray),
        ):
            for nested in item:
                visit(nested)

    if value is not None:
        visit(value)
    return tuple(values)


def _collect_structured_internal_values(value: Any) -> tuple[str, ...]:
    """Collect provider-only child IDs/paths from known structured fields."""
    values: list[str] = []

    def add(item: object) -> None:
        text = str(item if item is not None else "").strip()
        if text and text not in values:
            values.append(text)

    def add_scalar_values(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                add(key)
                add_scalar_values(nested)
            return
        if isinstance(item, Sequence) and not isinstance(
            item,
            (str, bytes, bytearray),
        ):
            for nested in item:
                add_scalar_values(nested)
            return
        add(item)

    def visit(item: Any, *, inside_agent_state: bool = False) -> None:
        if isinstance(item, Mapping):
            for raw_key, nested in item.items():
                key = _normalized_structured_key(raw_key)
                if key in _INTERNAL_AGENT_ID_KEYS:
                    add_scalar_values(nested)
                    continue
                if key in _INTERNAL_AGENT_PATH_KEYS or (
                    inside_agent_state and key == "path"
                ):
                    add_scalar_values(nested)
                    continue
                if key == "agentsstates" and isinstance(nested, Mapping):
                    for source_id, state in nested.items():
                        add(source_id)
                        visit(state, inside_agent_state=True)
                    continue
                visit(
                    nested,
                    inside_agent_state=(
                        inside_agent_state
                        or key in _AGENT_STATE_CONTAINER_KEYS
                    ),
                )
            return
        if isinstance(item, Sequence) and not isinstance(
            item,
            (str, bytes, bytearray),
        ):
            for nested in item:
                visit(nested, inside_agent_state=inside_agent_state)
            return
        if isinstance(item, str):
            nested = _parse_json(item.strip())
            if nested is not None:
                visit(nested, inside_agent_state=inside_agent_state)

    if value is not None:
        visit(value)
    return tuple(values)


def _command_opaque_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return ()

    parts = [str(part).strip() for part in value if str(part).strip()]
    values = [" ".join(parts)] if parts else []
    if "-lc" in parts:
        index = parts.index("-lc")
        if index + 1 < len(parts):
            values.append(parts[index + 1])
    return tuple(values)


def _normalize_opaque_values(values: Iterable[object]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        values = (values,)
    normalized: list[str] = []
    for value in values:
        text = str(value if value is not None else "").strip()
        if text and text not in normalized:
            normalized.append(text)
    return tuple(normalized)


def _remove_opaque_values(text: str, values: Iterable[str]) -> str:
    cleaned = str(text or "")
    unique = sorted(
        {
            str(value if value is not None else "").strip()
            for value in values
            if str(value if value is not None else "").strip()
        },
        key=len,
        reverse=True,
    )
    for value in unique:
        cleaned = re.sub(
            rf"(?<![\w]){re.escape(value)}(?![\w])",
            "",
            cleaned,
        )
    return cleaned


def _describe_json_payload(data: Any) -> str:
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        for item in data:
            desc = _describe_json_payload(item)
            if desc:
                return desc
        return ""

    if not isinstance(data, Mapping):
        return ""

    parsed_cmd = data.get("parsed_cmd")
    if isinstance(parsed_cmd, Sequence) and not isinstance(parsed_cmd, (str, bytes, bytearray)):
        for item in parsed_cmd:
            if isinstance(item, Mapping):
                desc = _describe_parsed_cmd(item)
                if desc:
                    return desc

    for key in ("description", "summary", "query", "content", "name"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    path = _first_string(data, ("path", "file_path", "file", "directory", "dir"))
    if path:
        return f"读取 {path}"

    command = _command_text(data.get("command") or data.get("cmd"))
    if command:
        return f"运行 {command}"
    return ""


def _describe_parsed_cmd(item: Mapping[str, Any]) -> str:
    cmd_type = str(item.get("type") or "").strip().lower()
    path = _first_string(item, ("path", "file_path", "file", "directory", "dir", "name"))
    query = _first_string(item, ("query", "pattern", "keyword"))
    command = _command_text(item.get("cmd") or item.get("command"))

    if cmd_type in _SEARCH_TYPES:
        target = " · ".join(part for part in (query, path) if part)
        return f"搜索 {target}" if target else (f"运行 {command}" if command else "")
    if cmd_type in _EDIT_TYPES:
        return f"编辑 {path}" if path else (f"运行 {command}" if command else "")
    if cmd_type in _READ_TYPES:
        return f"读取 {path}" if path else (f"运行 {command}" if command else "")
    if cmd_type in _RUN_TYPES or command:
        return f"运行 {command}" if command else ""
    return path or command


def _first_string(data: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _command_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts = [str(part).strip() for part in value if str(part).strip()]
        if "-lc" in parts:
            idx = parts.index("-lc")
            if idx + 1 < len(parts):
                return parts[idx + 1]
        return " ".join(parts)
    return ""


def _first_display_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and stripped not in _JSON_EDGE_LINES:
            return stripped
    return ""


def _truncate(value: str, max_chars: int) -> str:
    value = str(value or "").strip()
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[: max_chars - 1] + "…"
