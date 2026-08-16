"""Strict A2A protobuf-to-domain observation normalization.

Remote protobuf objects never cross this module's boundary.  Only bounded,
canonical JSON plus SDK-free identifiers and state are returned.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from a2a.types import (
    Artifact,
    Message,
    Part,
    Role,
    StreamResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message as ProtobufMessage

MAX_OBSERVATION_BYTES = 1024 * 1024


class A2ACodecError(ValueError):
    """A non-reflective boundary error that cannot leak remote detail."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"remote A2A observation rejected ({code})")


class A2AObservationKind(StrEnum):
    TASK_STATUS = "task_status"
    ARTIFACT = "artifact"
    MESSAGE = "message"
    TASK_SNAPSHOT = "task_snapshot"


class A2ANormalizedStatus(StrEnum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input_required"
    AUTH_REQUIRED = "auth_required"
    CLAIMED_COMPLETED = "claimed_completed"
    FAILED = "failed"
    REJECTED = "rejected"
    CANCELED = "canceled"


_TASK_STATE_MAP: dict[int, A2ANormalizedStatus] = {
    TaskState.TASK_STATE_SUBMITTED: A2ANormalizedStatus.SUBMITTED,
    TaskState.TASK_STATE_WORKING: A2ANormalizedStatus.WORKING,
    TaskState.TASK_STATE_INPUT_REQUIRED: A2ANormalizedStatus.INPUT_REQUIRED,
    TaskState.TASK_STATE_AUTH_REQUIRED: A2ANormalizedStatus.AUTH_REQUIRED,
    TaskState.TASK_STATE_COMPLETED: A2ANormalizedStatus.CLAIMED_COMPLETED,
    TaskState.TASK_STATE_FAILED: A2ANormalizedStatus.FAILED,
    TaskState.TASK_STATE_REJECTED: A2ANormalizedStatus.REJECTED,
    TaskState.TASK_STATE_CANCELED: A2ANormalizedStatus.CANCELED,
}


@dataclass(frozen=True, slots=True)
class NormalizedA2AObservation:
    """Immutable, SDK-free representation of one remote observation."""

    kind: A2AObservationKind
    context_id: str
    task_id: str
    status: A2ANormalizedStatus | None
    canonical_payload: bytes = field(repr=False)
    payload_digest: str
    observation_id: str
    artifact_id: str = ""
    append: bool = False
    last_chunk: bool = False

    @property
    def payload(self) -> Any:
        """Decode a fresh copy of the canonical JSON data."""

        return json.loads(self.canonical_payload)

    def to_remote_observation(
        self,
        handle: Any,
        *,
        content_ref: Any | None = None,
        observed_state: Any | None = None,
        sequence: int | None = None,
    ) -> Any:
        """Adapt to the frozen Phase 0 API after payload anchoring."""

        from src.autonomous.remote.models import (  # noqa: PLC0415
            RemoteObservation,
            RemoteTaskState,
        )

        expected_context_id = _handle_identifier(
            handle,
            primary="context_id",
            alias="remote_context_id",
            required=True,
        )
        expected_task_id = _handle_identifier(
            handle,
            primary="task_id",
            alias="remote_task_id",
            required=False,
        )
        _bind_ids(
            context_id=self.context_id,
            task_id=self.task_id,
            expected_context_id=expected_context_id,
            expected_task_id=expected_task_id,
            allow_initial_task=self.kind is A2AObservationKind.TASK_SNAPSHOT,
            allow_empty_task=(self.kind is A2AObservationKind.MESSAGE and not expected_task_id),
        )
        if sequence is not None and (
            not isinstance(sequence, int) or isinstance(sequence, bool) or not 0 <= sequence <= 999_999_999_999
        ):
            raise A2ACodecError("invalid-sequence")
        observation_id = self.observation_id
        if self.kind is A2AObservationKind.ARTIFACT:
            if sequence is None:
                raise A2ACodecError("artifact-sequence-required")
            observation_id = f"{observation_id}_{sequence}"
        if self.status is not None:
            remote_state = RemoteTaskState(self.status.value)
        elif isinstance(observed_state, RemoteTaskState):
            remote_state = observed_state
        else:
            raise A2ACodecError("artifact-state-required")
        return RemoteObservation(
            observation_id=observation_id,
            state=remote_state,
            context_id=self.context_id,
            payload_digest=self.payload_digest,
            task_id=self.task_id,
            payload_ref=content_ref,
            artifact_id=self.artifact_id,
            append=self.append,
            last_chunk=self.last_chunk,
        )


def normalize_a2a_observation(
    value: StreamResponse | Task | Message | TaskStatusUpdateEvent | TaskArtifactUpdateEvent,
    handle: Any,
) -> NormalizedA2AObservation:
    """Validate and normalize one SDK response/event against a frozen handle."""

    _reject_unknown_proto_fields(value)
    if isinstance(value, StreamResponse):
        payload_name = value.WhichOneof("payload")
        if payload_name is None:
            raise A2ACodecError("missing-stream-payload")
        value = getattr(value, payload_name)
        _reject_unknown_proto_fields(value)

    expected_context_id = _handle_identifier(
        handle,
        primary="context_id",
        alias="remote_context_id",
        required=True,
    )
    expected_task_id = _handle_identifier(
        handle,
        primary="task_id",
        alias="remote_task_id",
        required=False,
    )

    if isinstance(value, Task):
        return _normalize_task(value, expected_context_id, expected_task_id)
    if isinstance(value, Message):
        return _normalize_direct_message(value, expected_context_id, expected_task_id)
    if isinstance(value, TaskStatusUpdateEvent):
        return _normalize_status_update(value, expected_context_id, expected_task_id)
    if isinstance(value, TaskArtifactUpdateEvent):
        return _normalize_artifact_update(value, expected_context_id, expected_task_id)
    raise A2ACodecError("unsupported-event")


def _normalize_task(
    task: Task,
    expected_context_id: str,
    expected_task_id: str,
) -> NormalizedA2AObservation:
    _bind_ids(
        context_id=task.context_id,
        task_id=task.id,
        expected_context_id=expected_context_id,
        expected_task_id=expected_task_id,
        allow_initial_task=True,
    )
    if not task.HasField("status"):
        raise A2ACodecError("missing-task-status")
    status = _normalize_status(task.status, task.context_id, task.id)
    artifacts = [_normalize_artifact(item) for item in task.artifacts]
    history = [_normalize_message(item, task.context_id, task.id, allow_user_role=True) for item in task.history]
    body = {
        "artifacts": artifacts,
        "contextId": task.context_id,
        "history": history,
        "kind": A2AObservationKind.TASK_SNAPSHOT.value,
        "metadata": _struct_to_json(task.metadata) if task.HasField("metadata") else {},
        "status": status.value,
        "taskId": task.id,
        "taskStatus": _normalize_task_status_payload(task.status, task.context_id, task.id),
    }
    return _make_observation(
        kind=A2AObservationKind.TASK_SNAPSHOT,
        context_id=task.context_id,
        task_id=task.id,
        status=status,
        body=body,
    )


def _normalize_direct_message(
    message: Message,
    expected_context_id: str,
    expected_task_id: str,
) -> NormalizedA2AObservation:
    _bind_ids(
        context_id=message.context_id,
        task_id=message.task_id,
        expected_context_id=expected_context_id,
        expected_task_id=expected_task_id,
        allow_initial_task=False,
        allow_empty_task=not expected_task_id,
    )
    normalized_message = _normalize_message(
        message,
        expected_context_id,
        expected_task_id,
        allow_user_role=False,
    )
    body = {
        "contextId": message.context_id,
        "kind": A2AObservationKind.MESSAGE.value,
        "message": normalized_message,
        "status": A2ANormalizedStatus.CLAIMED_COMPLETED.value,
        "taskId": message.task_id,
    }
    return _make_observation(
        kind=A2AObservationKind.MESSAGE,
        context_id=message.context_id,
        task_id=message.task_id,
        status=A2ANormalizedStatus.CLAIMED_COMPLETED,
        body=body,
    )


def _normalize_status_update(
    event: TaskStatusUpdateEvent,
    expected_context_id: str,
    expected_task_id: str,
) -> NormalizedA2AObservation:
    _bind_ids(
        context_id=event.context_id,
        task_id=event.task_id,
        expected_context_id=expected_context_id,
        expected_task_id=expected_task_id,
        allow_initial_task=False,
    )
    if not event.HasField("status"):
        raise A2ACodecError("missing-task-status")
    status = _normalize_status(event.status, event.context_id, event.task_id)
    body = {
        "contextId": event.context_id,
        "kind": A2AObservationKind.TASK_STATUS.value,
        "metadata": _struct_to_json(event.metadata) if event.HasField("metadata") else {},
        "status": status.value,
        "taskId": event.task_id,
        "taskStatus": _normalize_task_status_payload(
            event.status,
            event.context_id,
            event.task_id,
        ),
    }
    return _make_observation(
        kind=A2AObservationKind.TASK_STATUS,
        context_id=event.context_id,
        task_id=event.task_id,
        status=status,
        body=body,
    )


def _normalize_artifact_update(
    event: TaskArtifactUpdateEvent,
    expected_context_id: str,
    expected_task_id: str,
) -> NormalizedA2AObservation:
    _bind_ids(
        context_id=event.context_id,
        task_id=event.task_id,
        expected_context_id=expected_context_id,
        expected_task_id=expected_task_id,
        allow_initial_task=False,
    )
    if not event.HasField("artifact"):
        raise A2ACodecError("missing-artifact")
    artifact = _normalize_artifact(event.artifact)
    artifact_id = event.artifact.artifact_id
    body = {
        "append": event.append,
        "artifact": artifact,
        "artifactId": artifact_id,
        "contextId": event.context_id,
        "kind": A2AObservationKind.ARTIFACT.value,
        "lastChunk": event.last_chunk,
        "metadata": _struct_to_json(event.metadata) if event.HasField("metadata") else {},
        "taskId": event.task_id,
    }
    return _make_observation(
        kind=A2AObservationKind.ARTIFACT,
        context_id=event.context_id,
        task_id=event.task_id,
        status=None,
        body=body,
        artifact_id=artifact_id,
        append=event.append,
        last_chunk=event.last_chunk,
    )


def _normalize_status(
    status: TaskStatus,
    context_id: str,
    task_id: str,
) -> A2ANormalizedStatus:
    normalized = _TASK_STATE_MAP.get(status.state)
    if normalized is None:
        raise A2ACodecError("unsupported-task-state")
    if status.HasField("message"):
        _normalize_message(status.message, context_id, task_id, allow_user_role=False)
    if status.HasField("timestamp"):
        try:
            status.timestamp.ToJsonString()
        except (OverflowError, ValueError):
            raise A2ACodecError("invalid-timestamp") from None
    return normalized


def _normalize_task_status_payload(
    status: TaskStatus,
    context_id: str,
    task_id: str,
) -> dict[str, Any]:
    value: dict[str, Any] = {"state": _TASK_STATE_MAP[status.state].value}
    if status.HasField("message"):
        value["message"] = _normalize_message(
            status.message,
            context_id,
            task_id,
            allow_user_role=False,
        )
    if status.HasField("timestamp"):
        value["timestamp"] = status.timestamp.ToJsonString()
    return value


def _normalize_message(
    message: Message,
    context_id: str,
    task_id: str,
    *,
    allow_user_role: bool,
) -> dict[str, Any]:
    _reject_unknown_proto_fields(message)
    if not message.message_id or not message.parts:
        raise A2ACodecError("invalid-message")
    if message.context_id != context_id or message.task_id != task_id:
        raise A2ACodecError("nested-id-mismatch")
    allowed_roles = {Role.ROLE_AGENT}
    if allow_user_role:
        allowed_roles.add(Role.ROLE_USER)
    if message.role not in allowed_roles:
        raise A2ACodecError("invalid-message-role")
    return {
        "contextId": message.context_id,
        "extensions": list(message.extensions),
        "messageId": message.message_id,
        "metadata": _struct_to_json(message.metadata) if message.HasField("metadata") else {},
        "parts": [_normalize_part(part) for part in message.parts],
        "referenceTaskIds": list(message.reference_task_ids),
        "role": "agent" if message.role == Role.ROLE_AGENT else "user",
        "taskId": message.task_id,
    }


def _normalize_artifact(artifact: Artifact) -> dict[str, Any]:
    _reject_unknown_proto_fields(artifact)
    if not artifact.artifact_id or not artifact.parts:
        raise A2ACodecError("invalid-artifact")
    return {
        "artifactId": artifact.artifact_id,
        "description": artifact.description,
        "extensions": list(artifact.extensions),
        "metadata": _struct_to_json(artifact.metadata) if artifact.HasField("metadata") else {},
        "name": artifact.name,
        "parts": [_normalize_part(part) for part in artifact.parts],
    }


def _normalize_part(part: Part) -> dict[str, Any]:
    _reject_unknown_proto_fields(part)
    content_kind = part.WhichOneof("content")
    common = {
        "filename": part.filename,
        "mediaType": part.media_type,
        "metadata": _struct_to_json(part.metadata) if part.HasField("metadata") else {},
    }
    if content_kind == "text":
        return {**common, "kind": "text", "text": part.text}
    if content_kind == "data":
        data = MessageToDict(part.data)
        _validate_json_value(data)
        return {**common, "data": data, "kind": "data"}
    if content_kind in {"raw", "url"}:
        raise A2ACodecError("unsupported-part-content")
    raise A2ACodecError("missing-part-content")


def _make_observation(
    *,
    kind: A2AObservationKind,
    context_id: str,
    task_id: str,
    status: A2ANormalizedStatus | None,
    body: dict[str, Any],
    artifact_id: str = "",
    append: bool = False,
    last_chunk: bool = False,
) -> NormalizedA2AObservation:
    canonical = _canonical_json_bytes(body)
    if len(canonical) > MAX_OBSERVATION_BYTES:
        raise A2ACodecError("observation-too-large")
    digest = hashlib.sha256(canonical).hexdigest()
    return NormalizedA2AObservation(
        kind=kind,
        context_id=context_id,
        task_id=task_id,
        status=status,
        canonical_payload=canonical,
        payload_digest=digest,
        observation_id=f"obs_{digest}",
        artifact_id=artifact_id,
        append=append,
        last_chunk=last_chunk,
    )


def _bind_ids(
    *,
    context_id: str,
    task_id: str,
    expected_context_id: str,
    expected_task_id: str,
    allow_initial_task: bool,
    allow_empty_task: bool = False,
) -> None:
    if not context_id or context_id != expected_context_id:
        raise A2ACodecError("context-id-mismatch")
    if not task_id:
        if allow_empty_task:
            return
        raise A2ACodecError("task-id-mismatch")
    if expected_task_id:
        if task_id != expected_task_id:
            raise A2ACodecError("task-id-mismatch")
    elif not allow_initial_task:
        raise A2ACodecError("task-id-not-bound")


def _handle_identifier(
    handle: Any,
    *,
    primary: str,
    alias: str,
    required: bool,
) -> str:
    value = getattr(handle, primary, None)
    if value is None:
        value = getattr(handle, alias, None)
    if not isinstance(value, str) or (required and not value):
        raise A2ACodecError("invalid-frozen-handle")
    return value


def _reject_unknown_proto_fields(value: Any) -> None:
    if not isinstance(value, ProtobufMessage):
        raise A2ACodecError("unsupported-event")
    try:
        before = value.SerializeToString(deterministic=True)
        clean = type(value)()
        clean.CopyFrom(value)
        clean.DiscardUnknownFields()
        after = clean.SerializeToString(deterministic=True)
    except (TypeError, ValueError):
        raise A2ACodecError("invalid-protobuf") from None
    if before != after:
        raise A2ACodecError("unknown-protobuf-field")


def _struct_to_json(value: ProtobufMessage) -> Any:
    result = MessageToDict(value)
    _validate_json_value(result)
    return result


def _validate_json_value(value: Any) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise A2ACodecError("invalid-data-number")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise A2ACodecError("invalid-data-key")
            _validate_json_value(item)
        return
    raise A2ACodecError("invalid-data-value")


def _canonical_json_bytes(value: Any) -> bytes:
    _validate_json_value(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8", errors="strict")
    except (RecursionError, TypeError, ValueError, UnicodeEncodeError):
        raise A2ACodecError("invalid-observation-json") from None


__all__ = [
    "A2ACodecError",
    "A2ANormalizedStatus",
    "A2AObservationKind",
    "MAX_OBSERVATION_BYTES",
    "NormalizedA2AObservation",
    "normalize_a2a_observation",
]
