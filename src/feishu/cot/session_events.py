"""CardEvent-to-AG-UI event mapping mixin and constants for FeishuCOTSession."""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

from src.card.events import CardEvent, CardEventType
from src.card.tool_display import (
    sanitize_full_tool_event_content,
    summarize_tool_call_content,
)
from src.utils.text import sanitize_single_line_label

from .wire import (
    _MAX_STREAM_CHUNK_BYTES,
    _MAX_TOOL_CONTENT_BYTES,
    COTTransportError,
    _brief_tool_result,
    _fit_string_to_event_content,
    _is_timeout_reason,
    _safe_error_message,
    _safe_tool_arguments,
    _safe_tool_content,
    _sanitize_unicode,
    _split_stream_delta,
    _tool_opaque_ids,
)

if TYPE_CHECKING:
    import threading

logger = logging.getLogger(__name__)

_SEGMENT_END = object()

_PROCESS_EVENT_TYPES = frozenset(
    {
        CardEventType.TEXT_STARTED,
        CardEventType.TEXT_DELTA,
        CardEventType.TEXT_DONE,
        CardEventType.REASONING_STARTED,
        CardEventType.REASONING_DELTA,
        CardEventType.REASONING_DONE,
        CardEventType.TOOL_STARTED,
        CardEventType.TOOL_DELTA,
        CardEventType.TOOL_DONE,
        CardEventType.TOOL_FAILED,
    }
)

_STATUS_TEXT = {
    "running": {"zh_cn": "过程记录中", "en_us": "Recording process"},
    "thinking": {"zh_cn": "正在记录过程", "en_us": "Recording process"},
    "done": {"zh_cn": "本段过程记录完成", "en_us": "Process segment recorded"},
    "error": {
        "zh_cn": "过程记录异常结束，请查看主卡",
        "en_us": "Process ended unexpectedly; see the main card",
    },
    "paused": {"zh_cn": "过程展示已切换到主卡", "en_us": "Process moved to the main card"},
    "interrupted": {"zh_cn": "过程记录已中断", "en_us": "Process interrupted"},
}


class _COTSessionEventsMixin:
    """Mixin mapping CardEvent lifecycle and process events to AG-UI COT payloads."""

    if TYPE_CHECKING:
        _input_text: str
        _detail: Literal["brief", "detailed"]
        _lock: threading.RLock
        _last_timestamp_ms: int
        _id_seed: bytes
        _thread_id: str
        _run_id: str
        _open_text: set[str]
        _closed_text: set[str]
        _open_reasoning: set[str]
        _closed_reasoning: set[str]
        _tools: dict[str, dict[str, object]]
        _completed_tools: set[str]

        def _record_failure_locked(
            self,
            error: COTTransportError,
            *,
            append_failed: bool,
        ) -> None: ...

    def _run_started_event(self) -> dict[str, object]:
        status_text = {key: dict(value) for key, value in _STATUS_TEXT.items()}
        query = _fit_string_to_event_content(
            self._input_text,
            lambda candidate: {
                "threadId": self._thread_id,
                "runId": self._run_id,
                "input": {
                    "query": candidate,
                    "statusText": status_text,
                },
            },
            maximum_bytes=_MAX_STREAM_CHUNK_BYTES,
        )
        return self._cot_event(
            "RUN_STARTED",
            {
                "threadId": self._thread_id,
                "runId": self._run_id,
                "input": {
                    "query": query,
                    "statusText": status_text,
                },
            },
        )

    def _terminal_event(
        self,
        event: object | None,
        *,
        aborting: bool,
        prior_failure: COTTransportError | None,
    ) -> tuple[dict[str, object], Literal["done", "error"], bool]:
        if aborting:
            return (
                self._degraded_terminal_event(),
                "done",
                True,
            )
        if event is _SEGMENT_END and prior_failure is None:
            return (
                self._cot_event(
                    "RUN_FINISHED",
                    {
                        "threadId": self._thread_id,
                        "runId": self._run_id,
                        "status": "done",
                    },
                ),
                "done",
                True,
            )
        if isinstance(event, CardEvent):
            # The parent card is the task-status authority.  If its real
            # terminal is already known, never replace it with a sidecar
            # transport failure that happened earlier in the same close race.
            if event.type is CardEventType.COMPLETED:
                return (
                    self._cot_event(
                        "RUN_FINISHED",
                        {
                            "threadId": self._thread_id,
                            "runId": self._run_id,
                            "status": "done",
                        },
                    ),
                    "done",
                    True,
                )
            if event.type is CardEventType.CANCELLED:
                reason = event.payload.get("reason", "")
                text_reason = reason if isinstance(reason, str) else ""
                if _is_timeout_reason(text_reason):
                    return (
                        self._run_error_event("Task timed out", "TIMEOUT"),
                        "error",
                        True,
                    )
                return (
                    self._cot_event(
                        "RUN_FINISHED",
                        {
                            "threadId": self._thread_id,
                            "runId": self._run_id,
                            "status": "interrupted",
                        },
                    ),
                    "error",
                    True,
                )
            if event.type is CardEventType.FAILED:
                raw_error = event.payload.get("error", "")
                message = (
                    _safe_error_message(raw_error)
                    if isinstance(raw_error, str) and raw_error
                    else "Task failed"
                )
                return (
                    self._run_error_event(message[:1_000], "TASK_FAILED"),
                    "error",
                    True,
                )

        if prior_failure is not None:
            logger.warning(
                "COT process transport degraded before parent terminal; "
                "failure_type=%s",
                type(prior_failure).__name__,
            )
            return self._degraded_terminal_event(), "done", True

        if not isinstance(event, CardEvent):
            with self._lock:
                self._record_failure_locked(
                    COTTransportError("COT terminal event is invalid"),
                    append_failed=False,
                )
            return (
                self._degraded_terminal_event(),
                "done",
                False,
            )
        with self._lock:
            self._record_failure_locked(
                COTTransportError("COT terminal event is invalid"),
                append_failed=False,
            )
        return (
            self._degraded_terminal_event(),
            "done",
            False,
        )

    def _degraded_terminal_event(self) -> dict[str, object]:
        """Close only the process display without asserting a task terminal."""
        return self._cot_event(
            "RUN_FINISHED",
            {
                "threadId": self._thread_id,
                "runId": self._run_id,
                "status": "paused",
            },
        )

    def _map_process_event_locked(
        self,
        event: CardEvent,
    ) -> tuple[dict[str, object], ...] | None:
        payload = event.payload
        block_id = payload.get("block_id")
        if not isinstance(block_id, str) or not block_id:
            return None

        if event.type is CardEventType.TEXT_STARTED:
            if block_id in self._open_text or block_id in self._closed_text:
                return ()
            self._open_text.add(block_id)
            return (
                self._cot_event(
                    "TEXT_MESSAGE_START",
                    {"messageId": self._hash_id("text", block_id), "role": "assistant"},
                ),
            )
        if event.type is CardEventType.TEXT_DELTA:
            if block_id not in self._open_text:
                return None
            text = payload.get("text", "")
            if not isinstance(text, str):
                return None
            text = _sanitize_unicode(text)
            if not text:
                return ()
            message_id = self._hash_id("text", block_id)
            return tuple(
                self._cot_event(
                    "TEXT_MESSAGE_CONTENT",
                    {
                        "messageId": message_id,
                        "delta": chunk,
                    },
                )
                for chunk in _split_stream_delta(message_id, text)
            )
        if event.type is CardEventType.TEXT_DONE:
            if block_id not in self._open_text:
                return None
            self._open_text.remove(block_id)
            self._closed_text.add(block_id)
            return (
                self._cot_event(
                    "TEXT_MESSAGE_END",
                    {"messageId": self._hash_id("text", block_id)},
                ),
            )
        if event.type is CardEventType.REASONING_STARTED:
            if block_id in self._open_reasoning or block_id in self._closed_reasoning:
                return ()
            self._open_reasoning.add(block_id)
            message_id = self._hash_id("reasoning", block_id)
            return (
                self._cot_event("REASONING_START", {"messageId": message_id}),
                self._cot_event(
                    "REASONING_MESSAGE_START",
                    {"messageId": message_id, "role": "reasoning"},
                ),
            )
        if event.type is CardEventType.REASONING_DELTA:
            if block_id not in self._open_reasoning:
                return None
            text = payload.get("text", "")
            if not isinstance(text, str):
                return None
            text = _sanitize_unicode(text)
            if not text:
                return ()
            message_id = self._hash_id("reasoning", block_id)
            return tuple(
                self._cot_event(
                    "REASONING_MESSAGE_CONTENT",
                    {
                        "messageId": message_id,
                        "delta": chunk,
                    },
                )
                for chunk in _split_stream_delta(message_id, text)
            )
        if event.type is CardEventType.REASONING_DONE:
            if block_id not in self._open_reasoning:
                return None
            self._open_reasoning.remove(block_id)
            self._closed_reasoning.add(block_id)
            message_id = self._hash_id("reasoning", block_id)
            return (
                self._cot_event(
                    "REASONING_MESSAGE_END",
                    {"messageId": message_id},
                ),
                self._cot_event("REASONING_END", {"messageId": message_id}),
            )
        if event.type is CardEventType.TOOL_STARTED:
            if block_id in self._tools or block_id in self._completed_tools:
                return ()
            tool_call_id = self._hash_id("tool", block_id)
            opaque_ids = _tool_opaque_ids(payload, block_id)
            raw_name = payload.get("tool_name", "")
            name_text = raw_name if isinstance(raw_name, str) else ""
            safe_name = sanitize_single_line_label(
                sanitize_full_tool_event_content(
                    name_text,
                    opaque_ids=opaque_ids,
                ),
                fallback="tool",
                max_chars=120,
            )
            raw_input = payload.get("tool_input", "")
            safe_input = _safe_tool_arguments(
                raw_input,
                opaque_ids=opaque_ids,
                tool_call_id=tool_call_id,
            )
            tool_summary = summarize_tool_call_content(
                safe_input,
                fallback=safe_name,
                max_chars=160,
            )
            tool_call_name = safe_name
            if (
                self._detail == "brief"
                and tool_summary
                and tool_summary.casefold() != safe_name.casefold()
            ):
                tool_call_name = sanitize_single_line_label(
                    f"{safe_name} · {tool_summary}",
                    fallback=safe_name,
                    max_chars=240,
                )
            self._tools[block_id] = {
                "tool_call_id": tool_call_id,
                "tool_name": safe_name,
                "tool_summary": tool_summary,
                "latest": "",
                "result_pending": False,
            }
            events = [
                self._cot_event(
                    "TOOL_CALL_START",
                    {
                        "toolCallId": tool_call_id,
                        "toolCallName": tool_call_name,
                    },
                )
            ]
            if self._detail == "detailed":
                events.append(
                    self._cot_event(
                        "TOOL_CALL_ARGS",
                        {"toolCallId": tool_call_id, "delta": safe_input},
                    )
                )
            events.append(
                self._cot_event("TOOL_CALL_END", {"toolCallId": tool_call_id}),
            )
            return tuple(events)
        if event.type is CardEventType.TOOL_DELTA:
            state = self._tools.get(block_id)
            if state is None or block_id in self._completed_tools:
                return None
            if self._detail == "brief":
                return ()
            state["latest"] = _safe_tool_content(
                payload.get("content", ""),
                opaque_ids=_tool_opaque_ids(payload, block_id),
            )
            return ()
        if event.type in {CardEventType.TOOL_DONE, CardEventType.TOOL_FAILED}:
            state = self._tools.get(block_id)
            if block_id in self._completed_tools:
                return ()
            if state is None:
                return None
            if self._detail == "brief":
                content = _brief_tool_result(
                    state,
                    failed=event.type is CardEventType.TOOL_FAILED,
                )
            else:
                if event.type is CardEventType.TOOL_FAILED:
                    candidate = payload.get("error", "")
                    fallback = "Tool failed"
                else:
                    candidate = payload.get("tool_output", "")
                    fallback = "Tool completed"
                content = _safe_tool_content(
                    candidate,
                    opaque_ids=_tool_opaque_ids(payload, block_id),
                )
                if not content:
                    latest = state.get("latest", "")
                    content = latest if isinstance(latest, str) else ""
                if not content:
                    content = fallback
            state["result_pending"] = True
            tool_call_id = state.get("tool_call_id")
            if not isinstance(tool_call_id, str):
                return None
            message_id = self._hash_id("tool-result", block_id)
            content = _fit_string_to_event_content(
                content,
                lambda candidate: {
                    "messageId": message_id,
                    "toolCallId": tool_call_id,
                    "content": candidate,
                    "role": "tool",
                },
                maximum_bytes=_MAX_TOOL_CONTENT_BYTES,
            )
            return (
                self._cot_event(
                    "TOOL_CALL_RESULT",
                    {
                        "messageId": message_id,
                        "toolCallId": tool_call_id,
                        "content": content,
                        "role": "tool",
                    },
                ),
            )
        return None

    def _commit_queued_event_locked(self, event: CardEvent) -> None:
        if event.type not in {CardEventType.TOOL_DONE, CardEventType.TOOL_FAILED}:
            return
        block_id = event.payload.get("block_id")
        if not isinstance(block_id, str):
            return
        state = self._tools.get(block_id)
        if state is None or state.get("result_pending") is not True:
            return
        self._completed_tools.add(block_id)
        self._tools.pop(block_id, None)

    def _cot_event(
        self,
        event_type: str,
        content: Mapping[str, object],
    ) -> dict[str, object]:
        return {
            "event_type": event_type,
            "content": dict(content),
            "timestamp": self._next_timestamp_ms(),
        }

    def _run_error_event(self, message: str, code: str) -> dict[str, object]:
        safe_message = _fit_string_to_event_content(
            message,
            lambda candidate: {
                "threadId": self._thread_id,
                "runId": self._run_id,
                "message": candidate,
                "code": code,
            },
            maximum_bytes=_MAX_TOOL_CONTENT_BYTES,
        )
        return self._cot_event(
            "RUN_ERROR",
            {
                "threadId": self._thread_id,
                "runId": self._run_id,
                "message": safe_message,
                "code": code,
            },
        )

    def _next_timestamp_ms(self) -> str:
        with self._lock:
            now_ms = int(time.time() * 1_000)
            self._last_timestamp_ms = max(now_ms, self._last_timestamp_ms + 1)
            return str(self._last_timestamp_ms)

    def _hash_id(self, kind: str, opaque_id: str) -> str:
        digest = hashlib.sha256(
            self._id_seed
            + b"\0"
            + kind.encode("utf-8")
            + b"\0"
            + _sanitize_unicode(opaque_id).encode("utf-8")
        ).hexdigest()
        return f"{kind.replace('-', '_')}_{digest[:32]}"
