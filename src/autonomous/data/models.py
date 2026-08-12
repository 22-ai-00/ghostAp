"""Strict immutable schemas for encrypted employee data records."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_AGENT_ID_RE = re.compile(r"agt_[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_HISTORY_ID_RE = re.compile(r"hist_[0-9a-f]{64}\Z")
_DOCUMENT_ID_RE = re.compile(r"data_[0-9a-f]{16}\Z")
_STATUS_VALUES = frozenset(
    {"completed", "failed", "canceled", "timeout", "action_required"}
)
_ERROR_CATEGORIES = frozenset(
    {
        "none",
        "timeout",
        "permission_denied",
        "validation",
        "tool_error",
        "model_error",
        "unknown",
    }
)
_CONTENT_TYPES = frozenset({"text/markdown", "application/json"})
_MAX_METADATA_LENGTH = 4096
_MAX_PAYLOAD_TEXT_LENGTH = 1_000_000
_MAX_COLLECTION_ITEMS = 10_000


def _exact_dict(value: Any, fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    missing = fields - set(value)
    extra = set(value) - fields
    if missing:
        raise ValueError(f"{name} missing fields")
    if extra:
        raise ValueError(f"{name} has unknown fields")
    return value


def _strict_str(
    value: Any,
    name: str,
    *,
    allow_empty: bool = False,
    maximum: int = _MAX_METADATA_LENGTH,
    allow_text_controls: bool = False,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if not allow_empty and not value:
        raise ValueError(f"{name} must be non-empty")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds maximum length")
    for character in value:
        if unicodedata.category(character) == "Cc" and not (
            allow_text_controls and character in "\t\n\r"
        ):
            raise ValueError(f"{name} contains control characters")
    return value


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _sha256(value: Any, name: str, *, allow_empty: bool = False) -> str:
    result = _strict_str(value, name, allow_empty=allow_empty)
    if allow_empty and not result:
        return result
    if _SHA256_RE.fullmatch(result) is None:
        raise ValueError(f"{name} must be lowercase sha256")
    return result


def _utc_timestamp(value: Any, name: str) -> str:
    text = _strict_str(value, name)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be RFC3339") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{name} must be timezone-aware UTC")
    return text


def _predecessor(sequence: Any, frame_hash: Any) -> tuple[int, str]:
    value = _strict_int(sequence, "predecessor_sequence")
    digest = _sha256(frame_hash, "predecessor_hash", allow_empty=True)
    if (value == 0) != (digest == ""):
        raise ValueError("predecessor sequence/hash mismatch")
    return value, digest


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _occurrence_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class DataKind(str, Enum):
    L1_MEMORY = "l1_memory"
    MEMORY_SUMMARY = "memory_summary"
    SKILL_PROFILE = "skill_profile"
    REASONING = "reasoning"
    KNOWLEDGE_PAGE = "knowledge_page"
    KNOWLEDGE_INDEX = "knowledge_index"
    KNOWLEDGE_REVIEW = "knowledge_review"


@dataclass(frozen=True)
class ToolUsageV1:
    name: str
    count: int
    duration_ms: int
    status: str

    _FIELDS = frozenset({"name", "count", "duration_ms", "status"})

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _strict_str(self.name, "tool name"))
        object.__setattr__(self, "count", _strict_int(self.count, "tool count"))
        object.__setattr__(
            self,
            "duration_ms",
            _strict_int(self.duration_ms, "tool duration_ms"),
        )
        object.__setattr__(self, "status", _strict_str(self.status, "tool status"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "duration_ms": self.duration_ms,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, value: Any) -> ToolUsageV1:
        data = _exact_dict(value, cls._FIELDS, "tool usage")
        return cls(**data)


@dataclass(frozen=True)
class SafeExecutionSummary:
    """Allowlisted terminal summary that cannot contain caller-provided text."""

    text: str

    @classmethod
    def build(
        cls,
        *,
        status: str,
        error_category: str = "none",
        tool_count: int = 0,
        attachment_count: int = 0,
    ) -> SafeExecutionSummary:
        status_value = _strict_str(status, "status")
        if status_value not in _STATUS_VALUES:
            raise ValueError("invalid terminal status")
        category = _strict_str(error_category, "error_category")
        if category not in _ERROR_CATEGORIES:
            raise ValueError("invalid error category")
        tools = _strict_int(tool_count, "tool_count")
        attachments = _strict_int(attachment_count, "attachment_count")
        return cls(
            f"status={status_value};error={category};"
            f"tools={tools};attachments={attachments}"
        )

    @classmethod
    def from_text(cls, value: Any) -> SafeExecutionSummary:
        text = _strict_str(value, "safe_summary", maximum=256)
        matched = re.fullmatch(
            r"status=([a-z_]+);error=([a-z_]+);tools=([0-9]+);attachments=([0-9]+)",
            text,
        )
        if matched is None:
            raise ValueError("safe_summary is not an allowlisted summary")
        expected = cls.build(
            status=matched.group(1),
            error_category=matched.group(2),
            tool_count=int(matched.group(3)),
            attachment_count=int(matched.group(4)),
        )
        if expected.text != text:
            raise ValueError("safe_summary is not canonical")
        return expected

    def __post_init__(self) -> None:
        text = _strict_str(self.text, "safe_summary", maximum=256)
        matched = re.fullmatch(
            r"status=([a-z_]+);error=([a-z_]+);tools=([0-9]+);attachments=([0-9]+)",
            text,
        )
        if matched is None or matched.group(1) not in _STATUS_VALUES or matched.group(2) not in _ERROR_CATEGORIES:
            raise ValueError("safe_summary is not an allowlisted summary")
        if str(int(matched.group(3))) != matched.group(3) or str(int(matched.group(4))) != matched.group(4):
            raise ValueError("safe_summary is not canonical")

    def __str__(self) -> str:
        return self.text

    @property
    def status(self) -> str:
        return self.text.split(";", 1)[0].removeprefix("status=")


@dataclass(frozen=True)
class ExecutionAttemptContext:
    tenant_key: str
    agent_id: str
    owner_principal_id: str
    requester_principal_id: str
    task_id: str
    run_id: str
    attempt_id: str
    message_id: str
    thread_root_id: str
    chat_id: str
    tool: str
    model: str
    effort: str
    started_at: str
    terminal_epoch: int = 1

    _FIELDS = frozenset(
        {
            "tenant_key",
            "agent_id",
            "owner_principal_id",
            "requester_principal_id",
            "task_id",
            "run_id",
            "attempt_id",
            "message_id",
            "thread_root_id",
            "chat_id",
            "tool",
            "model",
            "effort",
            "started_at",
            "terminal_epoch",
        }
    )

    def __post_init__(self) -> None:
        for name in self._FIELDS - {"terminal_epoch", "thread_root_id", "model"}:
            value = getattr(self, name)
            object.__setattr__(self, name, _strict_str(value, name))
        object.__setattr__(
            self,
            "model",
            _strict_str(self.model, "model", allow_empty=True),
        )
        object.__setattr__(
            self,
            "thread_root_id",
            _strict_str(self.thread_root_id, "thread_root_id", allow_empty=True),
        )
        if _AGENT_ID_RE.fullmatch(self.agent_id) is None:
            raise ValueError("agent_id must be canonical")
        object.__setattr__(self, "started_at", _utc_timestamp(self.started_at, "started_at"))
        object.__setattr__(
            self,
            "terminal_epoch",
            _strict_int(self.terminal_epoch, "terminal_epoch", minimum=1),
        )

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in sorted(self._FIELDS)}

    @classmethod
    def from_dict(cls, value: Any) -> ExecutionAttemptContext:
        return cls(**_exact_dict(value, cls._FIELDS, "execution attempt context"))


@dataclass(frozen=True)
class ExecutionHistoryRecordV1:
    schema_version: int
    record_id: str
    occurrence_key: str
    terminal_epoch: int
    tenant_key: str
    agent_id: str
    owner_principal_id: str
    requester_principal_id: str
    task_id: str
    run_id: str
    attempt_id: str
    message_id: str
    thread_root_id: str
    chat_id: str
    started_at: str
    ended_at: str
    duration_ms: int
    shard_day: str
    shard_timezone: str
    tool: str
    model: str
    effort: str
    status: str
    safe_summary: SafeExecutionSummary
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tool_usage: tuple[ToolUsageV1, ...]
    predecessor_sequence: int
    predecessor_hash: str

    _FIELDS = frozenset(
        {
            "schema_version", "record_id", "occurrence_key", "terminal_epoch",
            "tenant_key", "agent_id", "owner_principal_id", "requester_principal_id",
            "task_id", "run_id", "attempt_id", "message_id", "thread_root_id",
            "chat_id", "started_at", "ended_at", "duration_ms", "shard_day",
            "shard_timezone", "tool", "model", "effort", "status", "safe_summary",
            "prompt_tokens", "completion_tokens", "total_tokens", "tool_usage",
            "predecessor_sequence", "predecessor_hash",
        }
    )

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported history schema")
        for name in (
            "tenant_key", "owner_principal_id", "requester_principal_id", "task_id",
            "run_id", "attempt_id", "message_id", "chat_id", "tool", "effort",
        ):
            object.__setattr__(self, name, _strict_str(getattr(self, name), name))
        object.__setattr__(
            self,
            "model",
            _strict_str(self.model, "model", allow_empty=True),
        )
        object.__setattr__(
            self, "thread_root_id", _strict_str(self.thread_root_id, "thread_root_id", allow_empty=True)
        )
        if _AGENT_ID_RE.fullmatch(self.agent_id) is None:
            raise ValueError("agent_id must be canonical")
        terminal_epoch = _strict_int(self.terminal_epoch, "terminal_epoch", minimum=1)
        object.__setattr__(self, "terminal_epoch", terminal_epoch)
        occurrence = _occurrence_hash(
            f"{self.tenant_key}|{self.agent_id}|{self.run_id}|{self.attempt_id}|{terminal_epoch}"
        )
        if self.occurrence_key != occurrence or self.record_id != f"hist_{occurrence}":
            raise ValueError("history occurrence identity mismatch")
        if _HISTORY_ID_RE.fullmatch(self.record_id) is None:
            raise ValueError("record_id must be canonical")
        started = datetime.fromisoformat(_utc_timestamp(self.started_at, "started_at"))
        ended = datetime.fromisoformat(_utc_timestamp(self.ended_at, "ended_at"))
        if ended < started:
            raise ValueError("ended_at precedes started_at")
        expected_duration = int((ended - started).total_seconds() * 1000)
        if _strict_int(self.duration_ms, "duration_ms") != expected_duration:
            raise ValueError("duration_ms mismatch")
        try:
            timezone = ZoneInfo(_strict_str(self.shard_timezone, "shard_timezone"))
        except ZoneInfoNotFoundError as exc:
            raise ValueError("invalid shard timezone") from exc
        if self.shard_day != ended.astimezone(timezone).date().isoformat():
            raise ValueError("shard_day mismatch")
        if self.status not in _STATUS_VALUES:
            raise ValueError("invalid terminal status")
        if not isinstance(self.safe_summary, SafeExecutionSummary):
            raise ValueError("safe_summary must be trusted")
        if self.safe_summary.status != self.status:
            raise ValueError("safe_summary status does not match record status")
        prompt = _strict_int(self.prompt_tokens, "prompt_tokens")
        completion = _strict_int(self.completion_tokens, "completion_tokens")
        total = _strict_int(self.total_tokens, "total_tokens")
        if total != prompt + completion:
            raise ValueError("total_tokens mismatch")
        usage = tuple(self.tool_usage)
        if len(usage) > _MAX_COLLECTION_ITEMS or not all(
            isinstance(item, ToolUsageV1) for item in usage
        ):
            raise ValueError("invalid tool_usage")
        object.__setattr__(self, "tool_usage", usage)
        sequence, digest = _predecessor(self.predecessor_sequence, self.predecessor_hash)
        object.__setattr__(self, "predecessor_sequence", sequence)
        object.__setattr__(self, "predecessor_hash", digest)

    @classmethod
    def from_attempt(
        cls,
        context: ExecutionAttemptContext,
        *,
        ended_at: str,
        status: str,
        safe_summary: SafeExecutionSummary,
        prompt_tokens: int,
        completion_tokens: int,
        tool_usage: tuple[ToolUsageV1, ...] = (),
        predecessor_sequence: int,
        predecessor_hash: str,
        shard_timezone: str,
    ) -> ExecutionHistoryRecordV1:
        if not isinstance(context, ExecutionAttemptContext):
            raise TypeError("context must be ExecutionAttemptContext")
        occurrence = _occurrence_hash(
            f"{context.tenant_key}|{context.agent_id}|{context.run_id}|"
            f"{context.attempt_id}|{context.terminal_epoch}"
        )
        ended = datetime.fromisoformat(_utc_timestamp(ended_at, "ended_at"))
        started = datetime.fromisoformat(context.started_at)
        try:
            day = ended.astimezone(ZoneInfo(shard_timezone)).date().isoformat()
        except ZoneInfoNotFoundError as exc:
            raise ValueError("invalid shard timezone") from exc
        return cls(
            schema_version=1,
            record_id=f"hist_{occurrence}",
            occurrence_key=occurrence,
            terminal_epoch=context.terminal_epoch,
            tenant_key=context.tenant_key,
            agent_id=context.agent_id,
            owner_principal_id=context.owner_principal_id,
            requester_principal_id=context.requester_principal_id,
            task_id=context.task_id,
            run_id=context.run_id,
            attempt_id=context.attempt_id,
            message_id=context.message_id,
            thread_root_id=context.thread_root_id,
            chat_id=context.chat_id,
            started_at=context.started_at,
            ended_at=ended_at,
            duration_ms=int((ended - started).total_seconds() * 1000),
            shard_day=day,
            shard_timezone=shard_timezone,
            tool=context.tool,
            model=context.model,
            effort=context.effort,
            status=status,
            safe_summary=safe_summary,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            tool_usage=tool_usage,
            predecessor_sequence=predecessor_sequence,
            predecessor_hash=predecessor_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        result = {name: getattr(self, name) for name in self._FIELDS}
        result["safe_summary"] = self.safe_summary.text
        result["tool_usage"] = [item.to_dict() for item in self.tool_usage]
        return {name: result[name] for name in sorted(result)}

    @classmethod
    def from_dict(cls, value: Any) -> ExecutionHistoryRecordV1:
        data = _exact_dict(value, cls._FIELDS, "execution history record")
        return cls(
            **{
                **data,
                "safe_summary": SafeExecutionSummary.from_text(data["safe_summary"]),
                "tool_usage": tuple(ToolUsageV1.from_dict(item) for item in data["tool_usage"])
                if isinstance(data["tool_usage"], list)
                else (_raise("tool_usage must be a list")),
            }
        )


def _raise(message: str) -> Any:
    raise ValueError(message)


_ATTACHMENT_FIELDS = frozenset(
    {"resource_type", "resource_id", "name", "mime_type", "size", "sha256"}
)
_TOOL_CALL_FIELDS = frozenset(
    {"name", "status", "duration_ms", "input_summary", "output_summary"}
)


def _attachment(value: Any) -> Mapping[str, Any]:
    data = _exact_dict(value, _ATTACHMENT_FIELDS, "attachment")
    result = {
        "resource_type": _strict_str(data["resource_type"], "resource_type"),
        "resource_id": _strict_str(data["resource_id"], "resource_id"),
        "name": _strict_str(data["name"], "attachment name"),
        "mime_type": _strict_str(data["mime_type"], "mime_type"),
        "size": _strict_int(data["size"], "attachment size"),
        "sha256": _sha256(data["sha256"], "attachment sha256"),
    }
    return MappingProxyType(result)


def _tool_call(value: Any) -> Mapping[str, Any]:
    data = _exact_dict(value, _TOOL_CALL_FIELDS, "tool call")
    return MappingProxyType(
        {
            "name": _strict_str(data["name"], "tool call name"),
            "status": _strict_str(data["status"], "tool call status"),
            "duration_ms": _strict_int(data["duration_ms"], "tool call duration_ms"),
            "input_summary": _strict_str(
                data["input_summary"], "input_summary", allow_empty=True, maximum=8192
            ),
            "output_summary": _strict_str(
                data["output_summary"], "output_summary", allow_empty=True, maximum=8192
            ),
        }
    )


@dataclass(frozen=True)
class ExecutionHistoryPayloadV1:
    record_id: str
    occurrence_key: str
    request_text: str
    result_text: str
    error_detail: str
    attachments: tuple[Mapping[str, Any], ...] = ()
    tool_calls: tuple[Mapping[str, Any], ...] = ()
    schema_version: int = 1

    _FIELDS = frozenset(
        {"schema_version", "record_id", "occurrence_key", "request_text", "result_text", "error_detail", "attachments", "tool_calls"}
    )

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported history payload schema")
        if _HISTORY_ID_RE.fullmatch(self.record_id) is None:
            raise ValueError("record_id must be canonical")
        occurrence = _sha256(self.occurrence_key, "occurrence_key")
        if self.record_id != f"hist_{occurrence}":
            raise ValueError("payload occurrence identity mismatch")
        for name in ("request_text", "result_text", "error_detail"):
            object.__setattr__(
                self,
                name,
                _strict_str(
                    getattr(self, name),
                    name,
                    allow_empty=True,
                    maximum=_MAX_PAYLOAD_TEXT_LENGTH,
                    allow_text_controls=True,
                ),
            )
        attachments = tuple(_attachment(item) for item in self.attachments)
        tool_calls = tuple(_tool_call(item) for item in self.tool_calls)
        if len(attachments) > _MAX_COLLECTION_ITEMS or len(tool_calls) > _MAX_COLLECTION_ITEMS:
            raise ValueError("payload collection exceeds maximum")
        object.__setattr__(self, "attachments", attachments)
        object.__setattr__(self, "tool_calls", tool_calls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "occurrence_key": self.occurrence_key,
            "request_text": self.request_text,
            "result_text": self.result_text,
            "error_detail": self.error_detail,
            "attachments": [dict(item) for item in self.attachments],
            "tool_calls": [dict(item) for item in self.tool_calls],
        }

    @classmethod
    def from_dict(cls, value: Any) -> ExecutionHistoryPayloadV1:
        data = _exact_dict(value, cls._FIELDS, "execution history payload")
        if not isinstance(data["attachments"], list) or not isinstance(data["tool_calls"], list):
            raise ValueError("payload collections must be lists")
        return cls(
            **{
                **data,
                "attachments": tuple(data["attachments"]),
                "tool_calls": tuple(data["tool_calls"]),
            }
        )


@dataclass(frozen=True)
class EmployeeDataDocumentV1:
    document_id: str
    tenant_key: str
    agent_id: str
    owner_principal_id: str
    kind: DataKind
    version: int
    source_id: str
    created_at: str
    predecessor_sequence: int
    predecessor_hash: str
    content_type: str
    content_hash: str
    previous_document_id: str = ""
    legacy_source_hash: str = ""
    chat_id: str = ""
    thread_root_id: str = ""
    schema_version: int = 1

    _FIELDS = frozenset(
        {"schema_version", "document_id", "tenant_key", "agent_id", "owner_principal_id", "kind", "version", "source_id", "created_at", "predecessor_sequence", "predecessor_hash", "content_type", "content_hash", "previous_document_id", "legacy_source_hash", "chat_id", "thread_root_id"}
    )

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported document schema")
        if _DOCUMENT_ID_RE.fullmatch(self.document_id) is None:
            raise ValueError("document_id must be canonical")
        if _AGENT_ID_RE.fullmatch(self.agent_id) is None:
            raise ValueError("agent_id must be canonical")
        for name in ("tenant_key", "owner_principal_id", "source_id"):
            object.__setattr__(self, name, _strict_str(getattr(self, name), name))
        if not isinstance(self.kind, DataKind):
            raise ValueError("kind must be DataKind")
        version = _strict_int(self.version, "version", minimum=1)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "created_at", _utc_timestamp(self.created_at, "created_at"))
        sequence, digest = _predecessor(self.predecessor_sequence, self.predecessor_hash)
        object.__setattr__(self, "predecessor_sequence", sequence)
        object.__setattr__(self, "predecessor_hash", digest)
        if self.content_type not in _CONTENT_TYPES:
            raise ValueError("unsupported content_type")
        if self.kind in {
            DataKind.L1_MEMORY,
            DataKind.MEMORY_SUMMARY,
            DataKind.KNOWLEDGE_PAGE,
            DataKind.KNOWLEDGE_INDEX,
        }:
            if self.content_type != "text/markdown":
                raise ValueError("Markdown documents must use text/markdown")
        elif self.content_type != "application/json":
            raise ValueError("structured documents must use application/json")
        object.__setattr__(self, "content_hash", _sha256(self.content_hash, "content_hash"))
        previous = _strict_str(self.previous_document_id, "previous_document_id", allow_empty=True)
        if previous and _DOCUMENT_ID_RE.fullmatch(previous) is None:
            raise ValueError("previous_document_id must be canonical")
        if (version == 1) != (previous == ""):
            raise ValueError("document version/predecessor mismatch")
        object.__setattr__(
            self,
            "legacy_source_hash",
            _sha256(self.legacy_source_hash, "legacy_source_hash", allow_empty=True),
        )
        chat_id = _strict_str(self.chat_id, "chat_id", allow_empty=True)
        thread = _strict_str(self.thread_root_id, "thread_root_id", allow_empty=True)
        if self.kind is DataKind.MEMORY_SUMMARY:
            if not chat_id or self.source_id != self.memory_summary_source_id(
                chat_id=chat_id,
                thread_root_id=thread,
            ):
                raise ValueError("memory_summary source metadata mismatch")
        else:
            if chat_id or thread:
                raise ValueError("chat metadata is only valid for memory_summary")
            if self.kind in {
                DataKind.L1_MEMORY,
                DataKind.SKILL_PROFILE,
                DataKind.KNOWLEDGE_INDEX,
            } and self.source_id != self.kind.value:
                raise ValueError("document source_id does not match kind")

    @staticmethod
    def memory_summary_source_id(*, chat_id: str, thread_root_id: str) -> str:
        chat = _strict_str(chat_id, "chat_id")
        thread = _strict_str(thread_root_id, "thread_root_id", allow_empty=True)
        return _canonical_hash({"chat_id": chat, "thread_root_id": thread})

    def to_dict(self) -> dict[str, Any]:
        result = {name: getattr(self, name) for name in self._FIELDS}
        result["kind"] = self.kind.value
        return {name: result[name] for name in sorted(result)}

    @classmethod
    def from_dict(cls, value: Any) -> EmployeeDataDocumentV1:
        data = _exact_dict(value, cls._FIELDS, "employee data document")
        try:
            kind = DataKind(data["kind"])
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid data kind") from exc
        return cls(**{**data, "kind": kind})
