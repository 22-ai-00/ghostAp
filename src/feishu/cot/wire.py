"""COT wire-protocol constants, bindings, encoding, and sanitization helpers."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.card.events.payloads import OPAQUE_TOOL_CALL_ID_KEY
from src.card.tool_display import sanitize_full_tool_event_content
from src.utils.text import sanitize_single_line_label

if TYPE_CHECKING:
    from .api_client import _PendingCall

_COLLECTION_URI = "/open-apis/im/v1/message_cot"
_COMPLETE_URI = f"{_COLLECTION_URI}/complete/:cot_id"
_JSON_HEADERS = {"Content-Type": "application/json; charset=utf-8"}
_MAX_IDENTIFIER_CHARS = 512
_MAX_BATCH_EVENTS = 100
_MAX_UPDATE_EVENT_BYTES = 14 * 1024
# Generated Lark SDKs describe MessageCot.content as a JSON string of at most
# 4096 characters, while the live AppendCOTEvents service also rejects a
# multibyte value when its UTF-8 representation exceeds 4096 bytes. Both limits
# apply to the complete encoded content object, not only an inner delta/tool
# payload.
_MAX_EVENT_CONTENT_CHARS = 4_096
_MAX_EVENT_CONTENT_BYTES = 4_096
_MAX_JSON_DEPTH = 32
_MAX_JSON_NODES = 10_000
_MAX_STREAM_CHUNK_BYTES = 4_000
_MAX_TOOL_CONTENT_BYTES = 4_000

_COT_EVENT_TYPES = frozenset(
    {
        "RUN_STARTED",
        "RUN_FINISHED",
        "RUN_ERROR",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "REASONING_START",
        "REASONING_MESSAGE_START",
        "REASONING_MESSAGE_CONTENT",
        "REASONING_MESSAGE_END",
        "REASONING_END",
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
        "TOOL_CALL_RESULT",
    }
)

TrustRevision = tuple[int, int] | None


class COTTransportError(RuntimeError):
    """Raised when a COT request or lifecycle fails closed."""

    def __init__(
        self,
        message: str,
        *,
        request_may_be_in_flight: bool = False,
        pending_call: _PendingCall | None = None,
    ) -> None:
        super().__init__(message)
        self.request_may_be_in_flight = request_may_be_in_flight
        self._pending_call = pending_call


@dataclass(frozen=True, slots=True)
class COTBinding:
    """Immutable remote identity and provenance for one COT task."""

    chat_id: str
    cot_id: str
    message_id: str
    origin_message_id: str | None
    reply_in_thread: bool
    audit_targets: tuple[str, ...]
    trust_revision: TrustRevision

    def __post_init__(self) -> None:
        for label, value in (
            ("chat_id", self.chat_id),
            ("cot_id", self.cot_id),
            ("message_id", self.message_id),
        ):
            if not _valid_identifier(value):
                raise ValueError(f"invalid COT {label}")
        if self.origin_message_id is not None and not _valid_identifier(
            self.origin_message_id
        ):
            raise ValueError("invalid COT origin_message_id")
        if not isinstance(self.reply_in_thread, bool):
            raise ValueError("invalid COT reply_in_thread")
        if (
            not isinstance(self.audit_targets, tuple)
            or not self.audit_targets
            or any(not _valid_identifier(item) for item in self.audit_targets)
        ):
            raise ValueError("invalid COT audit provenance")
        _validate_trust_revision(self.trust_revision)


def _valid_identifier(value: object) -> bool:
    if not isinstance(value, str) or not value or len(value) > _MAX_IDENTIFIER_CHARS:
        return False
    try:
        value.encode("ascii")
    except UnicodeEncodeError:
        return False
    return all(character.isalnum() or character in {"_", "-"} for character in value)


def _validate_trust_revision(revision: TrustRevision) -> None:
    if revision is None:
        return
    if (
        not isinstance(revision, tuple)
        or len(revision) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in revision
        )
    ):
        raise ValueError("invalid COT trust revision")


def _encode_events(
    events: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if (
        not isinstance(events, Sequence)
        or isinstance(events, (str, bytes, bytearray))
        or not events
        or len(events) > _MAX_BATCH_EVENTS
    ):
        raise COTTransportError("COT event batch is invalid")
    encoded: list[dict[str, object]] = []
    for event in events:
        if not isinstance(event, Mapping) or set(event) != {
            "event_type",
            "content",
            "timestamp",
        }:
            raise COTTransportError("COT event schema is invalid")
        event_type = event.get("event_type")
        content = event.get("content")
        timestamp = event.get("timestamp")
        if event_type not in _COT_EVENT_TYPES or not isinstance(content, Mapping):
            raise COTTransportError("COT event schema is invalid")
        if isinstance(timestamp, bool):
            raise COTTransportError("COT event timestamp is invalid")
        if isinstance(timestamp, int):
            if timestamp < 0:
                raise COTTransportError("COT event timestamp is invalid")
            timestamp_text = str(timestamp)
        elif (
            isinstance(timestamp, str)
            and timestamp
            and timestamp.isascii()
            and timestamp.isdecimal()
        ):
            timestamp_text = timestamp
        else:
            raise COTTransportError("COT event timestamp is invalid")
        try:
            content_json = _encode_event_content(content)
            content_bytes = content_json.encode("utf-8")
        except (TypeError, ValueError, UnicodeEncodeError, RecursionError):
            raise COTTransportError("COT event content is invalid") from None
        if (
            len(content_json) > _MAX_EVENT_CONTENT_CHARS
            or len(content_bytes) > _MAX_EVENT_CONTENT_BYTES
        ):
            raise COTTransportError(
                "COT event content exceeds 4096 characters or UTF-8 bytes"
            )
        encoded.append(
            {
                "event_type": event_type,
                "content": content_json,
                "timestamp": timestamp_text,
            }
        )
    return encoded


def _encode_event_content(content: Mapping[str, object]) -> str:
    safe_content = _sanitize_json_value(content, budget=[0])
    return json.dumps(
        safe_content,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _sanitize_json_value(
    value: object,
    *,
    budget: list[int],
    depth: int = 0,
) -> object:
    budget[0] += 1
    if budget[0] > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
        raise ValueError("COT JSON content exceeds bounds")
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("COT JSON number is not finite")
        return value
    if isinstance(value, str):
        return _sanitize_unicode(value)
    if isinstance(value, Mapping):
        sanitized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("COT JSON object key is invalid")
            sanitized[_sanitize_unicode(key)] = _sanitize_json_value(
                item,
                budget=budget,
                depth=depth + 1,
            )
        return sanitized
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            _sanitize_json_value(item, budget=budget, depth=depth + 1)
            for item in value
        ]
    raise ValueError("COT JSON value is invalid")


def _sanitize_unicode(value: str) -> str:
    try:
        value.encode("utf-8")
        return value
    except UnicodeEncodeError:
        safe: list[str] = []
        index = 0
        while index < len(value):
            codepoint = ord(value[index])
            if 0xD800 <= codepoint <= 0xDBFF:
                if index + 1 < len(value):
                    low = ord(value[index + 1])
                    if 0xDC00 <= low <= 0xDFFF:
                        safe.append(
                            chr(
                                0x10000
                                + ((codepoint - 0xD800) << 10)
                                + (low - 0xDC00)
                            )
                        )
                        index += 2
                        continue
                safe.append("\ufffd")
            elif 0xDC00 <= codepoint <= 0xDFFF:
                safe.append("\ufffd")
            else:
                safe.append(value[index])
            index += 1
        return "".join(safe)


def _tool_opaque_ids(
    payload: Mapping[str, object],
    logical_block_id: str,
) -> tuple[str, ...]:
    """Return identifiers that must never be sent in rendered COT content."""
    raw_tool_id = payload.get(OPAQUE_TOOL_CALL_ID_KEY)
    if (
        isinstance(raw_tool_id, str)
        and raw_tool_id
        and raw_tool_id != logical_block_id
    ):
        return logical_block_id, raw_tool_id
    return (logical_block_id,)


def _safe_tool_content(value: object, *, opaque_ids: Sequence[str]) -> str:
    safe = sanitize_full_tool_event_content(value, opaque_ids=opaque_ids)
    safe = _sanitize_unicode(safe)
    return _truncate_utf8(safe, _MAX_TOOL_CONTENT_BYTES)


def _brief_tool_result(
    state: Mapping[str, object],
    *,
    failed: bool,
) -> str:
    """Render a compact tool terminal without carrying raw args or output."""
    raw_name = state.get("tool_name", "tool")
    name = raw_name if isinstance(raw_name, str) and raw_name else "tool"
    raw_summary = state.get("tool_summary", "")
    summary = raw_summary if isinstance(raw_summary, str) else ""
    parts = ["❌" if failed else "✅", name]
    if summary and summary.casefold() != name.casefold():
        parts.append(summary)
    parts.append("失败" if failed else "已完成")
    return sanitize_single_line_label(
        " · ".join(parts),
        fallback="工具调用失败" if failed else "工具调用已完成",
        max_chars=320,
    )


def _safe_tool_arguments(
    value: object,
    *,
    opaque_ids: Sequence[str],
    tool_call_id: str,
) -> str:
    """Return one valid JSON argument delta within the final content limit."""
    safe = sanitize_full_tool_event_content(value, opaque_ids=opaque_ids)
    safe = _sanitize_unicode(safe).strip()
    if not safe:
        return "{}"
    try:
        parsed = json.loads(safe)
    except (json.JSONDecodeError, RecursionError):
        parsed = {"input": safe}
    if not isinstance(parsed, Mapping):
        parsed = {"input": parsed}
    try:
        compact = json.dumps(
            parsed,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, RecursionError):
        compact = json.dumps(
            {"input": safe},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    if (
        len(compact.encode("utf-8")) <= _MAX_TOOL_CONTENT_BYTES
        and _event_content_fits(
            {"toolCallId": tool_call_id, "delta": compact}
        )
    ):
        return compact

    def _summary_delta(candidate: str) -> str:
        return json.dumps(
            {"truncated": True, "summary": candidate},
            ensure_ascii=False,
            separators=(",", ":"),
        )

    summary = _fit_string_to_event_content(
        safe,
        lambda candidate: {
            "toolCallId": tool_call_id,
            "delta": _summary_delta(candidate),
        },
        maximum_bytes=_MAX_TOOL_CONTENT_BYTES,
    )
    return _summary_delta(summary)


def _split_stream_delta(message_id: str, value: str) -> tuple[str, ...]:
    """Split a delta losslessly across byte and encoded-content boundaries."""
    chunks: list[str] = []
    for byte_chunk in _split_utf8_chunks(value, _MAX_STREAM_CHUNK_BYTES):
        remaining = byte_chunk
        while remaining:
            prefix_length = _largest_fitting_prefix(
                remaining,
                lambda candidate: {"messageId": message_id, "delta": candidate},
            )
            if prefix_length <= 0:
                raise COTTransportError("COT stream content envelope is invalid")
            chunks.append(remaining[:prefix_length])
            remaining = remaining[prefix_length:]
    return tuple(chunks)


def _fit_string_to_event_content(
    value: str,
    content_builder: Callable[[str], Mapping[str, object]],
    *,
    maximum_bytes: int | None = None,
) -> str:
    """Fit one display string inside the final MessageCot.content JSON."""
    candidate = _sanitize_unicode(value)
    if maximum_bytes is not None:
        candidate = _truncate_utf8(candidate, maximum_bytes)
    if _event_content_fits(content_builder(candidate)):
        return candidate
    ellipsis = "…"
    prefix_length = _largest_fitting_prefix(
        candidate,
        lambda prefix: content_builder(prefix + ellipsis),
    )
    if prefix_length <= 0:
        if _event_content_fits(content_builder(ellipsis)):
            return ellipsis
        raise COTTransportError(
            "COT event content envelope exceeds 4096 characters or UTF-8 bytes"
        )
    return candidate[:prefix_length] + ellipsis


def _largest_fitting_prefix(
    value: str,
    content_builder: Callable[[str], Mapping[str, object]],
) -> int:
    low = 0
    high = len(value)
    while low < high:
        middle = (low + high + 1) // 2
        if _event_content_fits(content_builder(value[:middle])):
            low = middle
        else:
            high = middle - 1
    return low


def _event_content_fits(content: Mapping[str, object]) -> bool:
    try:
        encoded = _encode_event_content(content)
        return (
            len(encoded) <= _MAX_EVENT_CONTENT_CHARS
            and len(encoded.encode("utf-8")) <= _MAX_EVENT_CONTENT_BYTES
        )
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError):
        return False


def _split_utf8_chunks(value: str, maximum_bytes: int) -> tuple[str, ...]:
    """Split text without corrupting UTF-8 or losing process content."""
    encoded = value.encode("utf-8")
    if len(encoded) <= maximum_bytes:
        return (value,)
    chunks: list[str] = []
    start = 0
    while start < len(encoded):
        end = min(start + maximum_bytes, len(encoded))
        while end < len(encoded) and end > start and encoded[end] & 0xC0 == 0x80:
            end -= 1
        if end == start:
            end = min(start + maximum_bytes, len(encoded))
            while end < len(encoded) and encoded[end] & 0xC0 == 0x80:
                end += 1
        chunks.append(encoded[start:end].decode("utf-8"))
        start = end
    return tuple(chunks)


def _truncate_utf8(value: str, maximum_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= maximum_bytes:
        return value
    ellipsis = "…"
    prefix_limit = maximum_bytes - len(ellipsis.encode("utf-8"))
    prefix = _split_utf8_chunks(value, prefix_limit)[0]
    return prefix + ellipsis


def _partition_update_events(
    events: Sequence[Mapping[str, object]],
) -> tuple[tuple[Mapping[str, object], ...], ...]:
    """Bound each Update body while preserving strict FIFO event order."""
    batches: list[tuple[Mapping[str, object], ...]] = []
    current: list[Mapping[str, object]] = []
    current_bytes = 2
    for event in events:
        encoded = _encode_events((event,))[0]
        event_bytes = len(
            json.dumps(
                encoded,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ) + (1 if current else 0)
        if current and current_bytes + event_bytes > _MAX_UPDATE_EVENT_BYTES:
            batches.append(tuple(current))
            current = []
            current_bytes = 2
            event_bytes -= 1
        current.append(event)
        current_bytes += event_bytes
    if current:
        batches.append(tuple(current))
    return tuple(batches)


def _safe_transport_failure(error: COTTransportError) -> str:
    return sanitize_single_line_label(
        str(error),
        fallback=type(error).__name__,
        max_chars=240,
    )


def _safe_response_message(value: object) -> str:
    if not isinstance(value, str) or not value:
        return ""
    return sanitize_single_line_label(
        sanitize_full_tool_event_content(value),
        fallback="",
        max_chars=160,
    )


def _safe_error_message(value: object) -> str:
    safe = sanitize_full_tool_event_content(value)
    safe = _sanitize_unicode(safe)
    if not safe:
        return "Task failed"
    return safe[:1_000]


def _hashed_identifier(kind: str, value: str) -> str:
    digest = hashlib.sha256(_sanitize_unicode(value).encode("utf-8")).hexdigest()
    return f"{kind}_{digest[:32]}"


def _is_timeout_reason(reason: str) -> bool:
    normalized = reason.strip().lower().replace("-", "_")
    return "timeout" in normalized or normalized in {"ttl_expired", "deadline"}
