"""Strict, versioned IPC protocol for employee Channel workers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from src.autonomous.ingress.models import (
    EmployeeIngressAck,
    EmployeeIngressMetadata,
    EmployeeIngressPayload,
)

PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 1024 * 1024
_FRAME_KEYS = {"v", "type", "agent_id", "generation", "sequence", "payload"}
_BOOTSTRAP_KEYS = {
    "v",
    "type",
    "agent_id",
    "app_id",
    "generation",
    "app_secret",
    "tenant_key",
    "bot_principal_id",
    "ack_timeout_seconds",
    "delivery_only",
    "frozen_connection_id",
}
_READY_PAYLOAD_KEYS = {"identity", "connection_id", "connection"}
_READY_IDENTITY_KEYS = {"app_id", "open_id"}
_READY_CONNECTION_KEYS = {
    "observed",
    "sdk_connection_id",
    "service_id",
    "secure",
}
_FORBIDDEN_IPC_KEYS = {
    "app_secret",
    "credential_ref",
    "master_key",
    "vault_key",
    "access_token",
    "tenant_access_token",
    "refresh_token",
    "authorization",
    "client_secret",
    "api_key",
    "private_key",
    "password",
    "token",
}
_FORBIDDEN_IPC_COLLAPSED_KEYS = frozenset(re.sub(r"[^a-z0-9]", "", key.casefold()) for key in _FORBIDDEN_IPC_KEYS)


class ProtocolError(ValueError):
    """The worker IPC peer sent a malformed or unsafe frame."""


class FrameType(str, Enum):
    READY = "READY"
    EVENT = "EVENT"
    HEALTH = "HEALTH"
    ERROR = "ERROR"
    STOP = "STOP"
    SEND = "SEND"
    UPDATE_CARD = "UPDATE_CARD"
    INGRESS = "INGRESS"
    INGRESS_ACK = "INGRESS_ACK"


@dataclass(frozen=True, slots=True)
class ChannelFrame:
    frame_type: FrameType
    agent_id: str
    generation: int
    sequence: int
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ChannelBootstrap:
    agent_id: str
    app_id: str
    generation: int
    app_secret: str
    tenant_key: str
    bot_principal_id: str
    ack_timeout_seconds: float
    delivery_only: bool = False
    frozen_connection_id: str = ""


def encode_frame(frame: ChannelFrame) -> bytes:
    """Encode one ordinary IPC frame as canonical single-line NDJSON."""
    _validate_identity(frame.agent_id, frame.generation)
    if not isinstance(frame.sequence, int) or isinstance(frame.sequence, bool) or frame.sequence < 1:
        raise ProtocolError("invalid sequence")
    if not isinstance(frame.payload, dict):
        raise ProtocolError("payload must be an object")
    _reject_secret_fields(frame.payload)
    _validate_typed_payload(frame)
    return _encode(
        {
            "v": PROTOCOL_VERSION,
            "type": frame.frame_type.value,
            "agent_id": frame.agent_id,
            "generation": frame.generation,
            "sequence": frame.sequence,
            "payload": frame.payload,
        }
    )


def decode_frame(raw: bytes) -> ChannelFrame:
    """Decode exactly one ordinary IPC frame and reject schema drift."""
    value = _decode(raw)
    if set(value) != _FRAME_KEYS:
        raise ProtocolError("invalid frame fields")
    if value["v"] != PROTOCOL_VERSION:
        raise ProtocolError("unsupported protocol version")
    try:
        frame_type = FrameType(value["type"])
    except (TypeError, ValueError) as exc:
        raise ProtocolError("unknown frame type") from exc
    agent_id = value["agent_id"]
    generation = value["generation"]
    _validate_identity(agent_id, generation)
    sequence = value["sequence"]
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
        raise ProtocolError("invalid sequence")
    payload = value["payload"]
    if not isinstance(payload, dict):
        raise ProtocolError("payload must be an object")
    _reject_secret_fields(payload)
    frame = ChannelFrame(frame_type, agent_id, generation, sequence, payload)
    _validate_typed_payload(frame)
    return frame


def encode_bootstrap(bootstrap: ChannelBootstrap) -> bytes:
    """Encode the sole credential-bearing frame for the one-shot pipe."""
    _validate_identity(bootstrap.agent_id, bootstrap.generation)
    if not isinstance(bootstrap.app_id, str) or not bootstrap.app_id:
        raise ProtocolError("invalid app_id")
    if not isinstance(bootstrap.app_secret, str) or not bootstrap.app_secret:
        raise ProtocolError("invalid app_secret")
    _validate_bootstrap_ingress(bootstrap)
    return _encode(
        {
            "v": PROTOCOL_VERSION,
            "type": "BOOTSTRAP",
            "agent_id": bootstrap.agent_id,
            "app_id": bootstrap.app_id,
            "generation": bootstrap.generation,
            "app_secret": bootstrap.app_secret,
            "tenant_key": bootstrap.tenant_key,
            "bot_principal_id": bootstrap.bot_principal_id,
            "ack_timeout_seconds": bootstrap.ack_timeout_seconds,
            "delivery_only": bootstrap.delivery_only,
            "frozen_connection_id": bootstrap.frozen_connection_id,
        }
    )


def decode_bootstrap(raw: bytes) -> ChannelBootstrap:
    """Decode the one allowed credential frame from the bootstrap pipe."""
    value = _decode(raw)
    if set(value) != _BOOTSTRAP_KEYS or value.get("v") != PROTOCOL_VERSION:
        raise ProtocolError("invalid bootstrap frame")
    if value.get("type") != "BOOTSTRAP":
        raise ProtocolError("invalid bootstrap type")
    bootstrap = ChannelBootstrap(
        agent_id=value.get("agent_id"),
        app_id=value.get("app_id"),
        generation=value.get("generation"),
        app_secret=value.get("app_secret"),
        tenant_key=value.get("tenant_key"),
        bot_principal_id=value.get("bot_principal_id"),
        ack_timeout_seconds=value.get("ack_timeout_seconds"),
        delivery_only=value.get("delivery_only"),
        frozen_connection_id=value.get("frozen_connection_id"),
    )
    _validate_identity(bootstrap.agent_id, bootstrap.generation)
    if not isinstance(bootstrap.app_id, str) or not bootstrap.app_id:
        raise ProtocolError("invalid app_id")
    if not isinstance(bootstrap.app_secret, str) or not bootstrap.app_secret:
        raise ProtocolError("invalid app_secret")
    _validate_bootstrap_ingress(bootstrap)
    return bootstrap


def _encode(value: Mapping[str, Any]) -> bytes:
    try:
        encoded = (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    except (TypeError, ValueError) as exc:
        raise ProtocolError("frame is not JSON serializable") from exc
    if len(encoded) > MAX_FRAME_BYTES:
        raise ProtocolError("frame too large")
    return encoded


def _decode(raw: bytes) -> dict[str, Any]:
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_FRAME_BYTES:
        raise ProtocolError("invalid frame size")
    if not raw.endswith(b"\n") or b"\n" in raw[:-1] or b"\r" in raw:
        raise ProtocolError("frame must be one NDJSON line")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid JSON frame") from exc
    if not isinstance(value, dict):
        raise ProtocolError("frame must be an object")
    return value


def _validate_identity(agent_id: Any, generation: Any) -> None:
    if not isinstance(agent_id, str) or not agent_id or len(agent_id) > 256:
        raise ProtocolError("invalid agent_id")
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
        raise ProtocolError("invalid generation")


def _validate_bootstrap_ingress(bootstrap: ChannelBootstrap) -> None:
    if not isinstance(bootstrap.tenant_key, str) or not bootstrap.tenant_key:
        raise ProtocolError("invalid tenant_key")
    if not isinstance(bootstrap.bot_principal_id, str) or not bootstrap.bot_principal_id.startswith("bot_"):
        raise ProtocolError("invalid bot_principal_id")
    timeout = bootstrap.ack_timeout_seconds
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < float(timeout) < 3.0:
        raise ProtocolError("invalid ack_timeout_seconds")
    if type(bootstrap.delivery_only) is not bool:
        raise ProtocolError("invalid delivery_only")
    if bootstrap.delivery_only:
        if not _is_safe_identifier(
            bootstrap.frozen_connection_id,
            prefix="conn_",
        ):
            raise ProtocolError("invalid frozen_connection_id")
    elif bootstrap.frozen_connection_id != "":
        raise ProtocolError("invalid frozen_connection_id")


def _reject_secret_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            collapsed = re.sub(r"[^a-z0-9]", "", key.casefold()) if isinstance(key, str) else ""
            if collapsed in _FORBIDDEN_IPC_COLLAPSED_KEYS:
                raise ProtocolError("credential material is forbidden on ordinary IPC")
            _reject_secret_fields(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_secret_fields(child)


def _validate_typed_payload(frame: ChannelFrame) -> None:
    if frame.frame_type is FrameType.READY:
        validate_ready_payload(frame.payload)
        return
    if frame.frame_type is FrameType.HEALTH:
        _validate_health_payload(frame)
        return
    if frame.frame_type is FrameType.SEND:
        try:
            _validate_send_payload(frame)
        except (TypeError, ValueError, KeyError) as exc:
            raise ProtocolError("invalid employee send frame") from exc
        return
    if frame.frame_type is FrameType.UPDATE_CARD:
        try:
            _validate_update_card_payload(frame)
        except (TypeError, ValueError, KeyError) as exc:
            raise ProtocolError("invalid employee update card frame") from exc
        return
    try:
        if frame.frame_type is FrameType.INGRESS:
            _validate_ingress_payload(frame)
        elif frame.frame_type is FrameType.INGRESS_ACK:
            _validate_ingress_ack_payload(frame)
    except (TypeError, ValueError, KeyError) as exc:
        raise ProtocolError("invalid employee ingress frame") from exc


def validate_ready_payload(payload: Mapping[str, Any]) -> None:
    """Validate the complete READY authority schema without runtime state."""

    try:
        if not isinstance(payload, dict) or set(payload) != _READY_PAYLOAD_KEYS:
            raise ValueError("invalid ready payload fields")
        identity = payload["identity"]
        connection = payload["connection"]
        if not isinstance(identity, dict) or set(identity) != _READY_IDENTITY_KEYS:
            raise ValueError("invalid ready identity fields")
        if (
            not _is_safe_text(identity["app_id"])
            or not _is_safe_identifier(identity["open_id"], prefix="ou_")
            or not _is_safe_identifier(payload["connection_id"], prefix="conn_")
        ):
            raise ValueError("invalid ready authority identity")
        if (
            not isinstance(connection, dict)
            or set(connection) != _READY_CONNECTION_KEYS
            or connection["observed"] is not True
            or connection["secure"] is not True
            or not _is_safe_text(connection["sdk_connection_id"])
            or not _is_safe_text(connection["service_id"])
        ):
            raise ValueError("invalid ready connection proof")
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("invalid employee ready frame") from exc


def _is_safe_text(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and len(value) <= 256
        and value == value.strip()
        and not any(ord(character) < 32 or ord(character) == 127 for character in value)
    )


def _is_safe_identifier(value: Any, *, prefix: str) -> bool:
    return (
        _is_safe_text(value)
        and isinstance(value, str)
        and value.startswith(prefix)
        and re.fullmatch(r"[A-Za-z0-9_-]+", value) is not None
    )


def _validate_health_payload(frame: ChannelFrame) -> None:
    operation = frame.payload.get("operation")
    if operation not in {"send", "update_card"}:
        return
    request_id = frame.payload.get("request_id")
    expected_prefix = "send_" if operation == "send" else "update_"
    if (
        not isinstance(request_id, str)
        or not request_id.startswith(expected_prefix)
        or len(request_id) > 256
        or type(frame.payload.get("success")) is not bool
    ):
        raise ProtocolError("invalid employee outbound health frame")
    if frame.payload["success"] is True:
        if set(frame.payload) != {
            "operation",
            "request_id",
            "success",
            "app_id",
            "generation",
            "connection_id",
            "message_id",
        }:
            raise ProtocolError("invalid employee outbound health frame")
        if (
            not _is_safe_text(frame.payload["app_id"])
            or frame.payload["generation"] != frame.generation
            or not _is_safe_identifier(
                frame.payload["connection_id"],
                prefix="conn_",
            )
            or not _is_safe_text(frame.payload["message_id"])
        ):
            raise ProtocolError("invalid employee outbound health frame")
        return
    if set(frame.payload) != {
        "operation",
        "request_id",
        "success",
        "failure_kind",
        "error_code",
    }:
        raise ProtocolError("invalid employee outbound health frame")
    if (
        frame.payload["failure_kind"] not in {"transport", "remote", "internal"}
        or not _is_safe_text(frame.payload["error_code"])
        or re.fullmatch(r"[a-z0-9-]+", frame.payload["error_code"]) is None
    ):
        raise ProtocolError("invalid employee outbound health frame")


def _validate_send_payload(frame: ChannelFrame) -> None:
    if set(frame.payload) != {"request_id", "target", "message", "options"}:
        raise ValueError("invalid send payload fields")
    request_id = frame.payload["request_id"]
    target = frame.payload["target"]
    message = frame.payload["message"]
    options = frame.payload["options"]
    if not isinstance(request_id, str) or not request_id.startswith("send_") or len(request_id) > 256:
        raise ValueError("invalid send request_id")
    if not isinstance(target, str) or not target or len(target) > 256:
        raise ValueError("invalid send target")
    if not isinstance(message, dict) or len(message) != 1 or next(iter(message)) not in {"text", "card", "post"}:
        raise ValueError("invalid send message")
    if options is not None and not isinstance(options, dict):
        raise ValueError("invalid send options")


def _validate_update_card_payload(frame: ChannelFrame) -> None:
    if set(frame.payload) != {"request_id", "message_id", "card"}:
        raise ValueError("invalid update card payload fields")
    request_id = frame.payload["request_id"]
    message_id = frame.payload["message_id"]
    card = frame.payload["card"]
    if not isinstance(request_id, str) or not request_id.startswith("update_") or len(request_id) > 256:
        raise ValueError("invalid update card request_id")
    if not isinstance(message_id, str) or not message_id or len(message_id) > 256:
        raise ValueError("invalid update card message_id")
    if not isinstance(card, dict):
        raise ValueError("invalid update card body")


def _validate_ingress_payload(frame: ChannelFrame) -> None:
    required = {
        "request_id",
        "app_id",
        "connection_id",
        "metadata",
        "payload",
        "action_correlation",
    }
    if set(frame.payload) != required:
        raise ValueError("invalid ingress payload fields")
    request_id = frame.payload["request_id"]
    if not isinstance(request_id, str) or not request_id.startswith("req_"):
        raise ValueError("invalid ingress request_id")
    metadata = EmployeeIngressMetadata.from_dict(frame.payload["metadata"])
    payload = EmployeeIngressPayload.from_dict(frame.payload["payload"])
    if (
        metadata.agent_id != frame.agent_id
        or metadata.channel_generation != frame.generation
        or metadata.app_id != frame.payload["app_id"]
        or metadata.connection_id != frame.payload["connection_id"]
        or metadata.envelope_id != payload.envelope_id
        or metadata.payload_sha256 != payload.payload_sha256
        or metadata.payload_size_bytes != payload.canonical_size_bytes
        or metadata.attachment_count != len(payload.attachment_descriptors)
        or metadata.attachment_total_bytes != payload.attachment_total_bytes
    ):
        raise ValueError("ingress transport binding mismatch")
    correlation = frame.payload["action_correlation"]
    if correlation is not None and (not isinstance(correlation, str) or not correlation):
        raise ValueError("invalid action correlation")
    if not metadata.event_id and correlation != metadata.action_identity:
        raise ValueError("fallback action correlation mismatch")


def _validate_ingress_ack_payload(frame: ChannelFrame) -> None:
    required = {"request_id", "app_id", "connection_id", "ack"}
    if set(frame.payload) != required:
        raise ValueError("invalid ingress ACK fields")
    ack = EmployeeIngressAck.from_dict(frame.payload["ack"])
    if (
        ack.request_id != frame.payload["request_id"]
        or ack.agent_id != frame.agent_id
        or ack.app_id != frame.payload["app_id"]
        or ack.channel_generation != frame.generation
        or ack.connection_id != frame.payload["connection_id"]
    ):
        raise ValueError("ingress ACK transport binding mismatch")
