"""SDK-free contracts for durable dispatch to a remote Agent."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping
from urllib.parse import urlsplit

from ..journal.blob_store import BlobRef

A2A_SPEC_RELEASE = "1.0.1"
A2A_WIRE_PROTOCOL_VERSION = "1.0"
MAX_REMOTE_INSTRUCTION_BYTES = 64 * 1024

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _identifier(value: object, name: str, *, optional: bool = False) -> str:
    if optional and value == "":
        return ""
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"invalid {name}")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"invalid {name}")
    return value


def _trusted_https_url(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"invalid {name}")
    try:
        parsed = urlsplit(value)
        parsed.port
    except ValueError as exc:
        raise ValueError(f"invalid {name}") from exc
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"invalid {name}")
    return value


class RemoteProtocolBinding(StrEnum):
    """Protocol bindings admitted by the first outbound pilot."""

    JSONRPC = "JSONRPC"


class RemoteTaskState(StrEnum):
    """Local interpretation of an untrusted remote task observation."""

    PREPARED = "prepared"
    EXECUTING = "executing"
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input_required"
    AUTH_REQUIRED = "auth_required"
    CLAIMED_COMPLETED = "claimed_completed"
    FAILED = "failed"
    REJECTED = "rejected"
    CANCELED = "canceled"
    OUTCOME_UNCERTAIN = "outcome_uncertain"

    @property
    def is_remote_terminal(self) -> bool:
        """Whether this is a terminal remote claim, not local completion."""

        return self in {
            RemoteTaskState.CLAIMED_COMPLETED,
            RemoteTaskState.FAILED,
            RemoteTaskState.REJECTED,
            RemoteTaskState.CANCELED,
        }


class RemoteAttemptPhase(StrEnum):
    """Durable local effect phase for one remote dispatch attempt."""

    PREPARED = "prepared"
    EXECUTING = "executing"
    TRACKING = "tracking"
    CANCEL_REQUESTED = "cancel_requested"
    SEND_UNCERTAIN = "send_uncertain"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class RemoteAgentDescriptor:
    """A Card-digest-bound remote Agent admitted by local configuration."""

    tenant_key: str
    agent_id: str
    card_url: str
    endpoint_url: str
    card_digest: str
    credential_ref: str
    protocol_binding: RemoteProtocolBinding = RemoteProtocolBinding.JSONRPC
    protocol_version: str = A2A_WIRE_PROTOCOL_VERSION
    remote_tenant: str = ""

    def __post_init__(self) -> None:
        _identifier(self.tenant_key, "tenant_key")
        _identifier(self.agent_id, "agent_id")
        if not self.agent_id.startswith("agt_"):
            raise ValueError("invalid agent_id")
        _trusted_https_url(self.card_url, "card_url")
        _trusted_https_url(self.endpoint_url, "endpoint_url")
        _sha256(self.card_digest, "card_digest")
        _identifier(self.credential_ref, "credential_ref", optional=True)
        if self.protocol_binding is not RemoteProtocolBinding.JSONRPC:
            raise ValueError("unsupported remote protocol binding")
        if self.protocol_version != A2A_WIRE_PROTOCOL_VERSION:
            raise ValueError("unsupported remote protocol version")
        _identifier(self.remote_tenant, "remote_tenant", optional=True)


@dataclass(frozen=True, slots=True)
class RemoteDispatchRequest:
    """Stable coordinates and plaintext used to prepare one outbound attempt."""

    acceptance_id: str
    run_id: str
    assignment_id: str
    attempt_id: str
    message_id: str
    context_id: str
    instruction: str = field(repr=False)
    descriptor: RemoteAgentDescriptor = field(repr=False)

    def __post_init__(self) -> None:
        for value, name in (
            (self.acceptance_id, "acceptance_id"),
            (self.run_id, "run_id"),
            (self.assignment_id, "assignment_id"),
            (self.attempt_id, "attempt_id"),
            (self.message_id, "message_id"),
            (self.context_id, "context_id"),
        ):
            _identifier(value, name)
        if (
            not isinstance(self.instruction, str)
            or not self.instruction.strip()
            or "\x00" in self.instruction
            or len(self.instruction.encode("utf-8")) > MAX_REMOTE_INSTRUCTION_BYTES
        ):
            raise ValueError("invalid remote instruction")


@dataclass(frozen=True, slots=True)
class RemoteTaskHandle:
    """Durable assignment/attempt to A2A message/context/task mapping."""

    acceptance_id: str
    run_id: str
    assignment_id: str
    attempt_id: str
    message_id: str
    context_id: str
    descriptor: RemoteAgentDescriptor
    instruction_ref: BlobRef
    task_id: str = ""

    def __post_init__(self) -> None:
        for value, name in (
            (self.acceptance_id, "acceptance_id"),
            (self.run_id, "run_id"),
            (self.assignment_id, "assignment_id"),
            (self.attempt_id, "attempt_id"),
            (self.message_id, "message_id"),
            (self.context_id, "context_id"),
        ):
            _identifier(value, name)
        _identifier(self.task_id, "task_id", optional=True)
        if not isinstance(self.descriptor, RemoteAgentDescriptor):
            raise TypeError("descriptor must be RemoteAgentDescriptor")
        if not isinstance(self.instruction_ref, BlobRef):
            raise TypeError("instruction_ref must be BlobRef")

    @property
    def key(self) -> tuple[str, str, str]:
        """Return the local run/assignment/attempt authority key."""

        return self.run_id, self.assignment_id, self.attempt_id


@dataclass(frozen=True, slots=True)
class RemoteObservation:
    """One normalized remote event whose body is stored only by Blob reference."""

    observation_id: str
    state: RemoteTaskState
    context_id: str
    payload_digest: str
    task_id: str = ""
    payload_ref: BlobRef | None = None
    artifact_id: str = ""
    append: bool = False
    last_chunk: bool = False

    def __post_init__(self) -> None:
        _identifier(self.observation_id, "observation_id")
        _identifier(self.context_id, "context_id")
        _identifier(self.task_id, "task_id", optional=True)
        _identifier(self.artifact_id, "artifact_id", optional=True)
        _sha256(self.payload_digest, "payload_digest")
        if not isinstance(self.state, RemoteTaskState):
            raise TypeError("state must be RemoteTaskState")
        if self.payload_ref is not None:
            if not isinstance(self.payload_ref, BlobRef):
                raise TypeError("payload_ref must be BlobRef")
            if self.payload_ref.payload_hash != self.payload_digest:
                raise ValueError("remote observation payload digest mismatch")
        if type(self.append) is not bool or type(self.last_chunk) is not bool:
            raise TypeError("artifact chunk flags must be booleans")
        if (self.append or self.last_chunk) and not self.artifact_id:
            raise ValueError("artifact chunk flags require artifact_id")


@dataclass(frozen=True, slots=True)
class RemoteSnapshot:
    """Replayable local view of one remote attempt."""

    handle: RemoteTaskHandle
    phase: RemoteAttemptPhase
    state: RemoteTaskState
    observations: tuple[RemoteObservation, ...] = ()
    cancel_requested: bool = False
    error_code: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.handle, RemoteTaskHandle):
            raise TypeError("handle must be RemoteTaskHandle")
        if not isinstance(self.phase, RemoteAttemptPhase):
            raise TypeError("phase must be RemoteAttemptPhase")
        if not isinstance(self.state, RemoteTaskState):
            raise TypeError("state must be RemoteTaskState")
        object.__setattr__(self, "observations", tuple(self.observations))
        if any(not isinstance(item, RemoteObservation) for item in self.observations):
            raise TypeError("observations must contain RemoteObservation values")
        if type(self.cancel_requested) is not bool:
            raise TypeError("cancel_requested must be boolean")
        _identifier(self.error_code, "error_code", optional=True)

    @property
    def claimed_completed(self) -> bool:
        """Return a remote claim that still requires local verification."""

        return self.state is RemoteTaskState.CLAIMED_COMPLETED


@dataclass(frozen=True, slots=True)
class RemoteProjection:
    """Immutable projection of all known remote dispatch attempts."""

    by_key: Mapping[tuple[str, str, str], RemoteSnapshot] = field(default_factory=dict)
    by_acceptance_id: Mapping[str, tuple[str, str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_key", MappingProxyType(dict(self.by_key)))
        object.__setattr__(
            self,
            "by_acceptance_id",
            MappingProxyType(dict(self.by_acceptance_id)),
        )


__all__ = [
    "A2A_SPEC_RELEASE",
    "A2A_WIRE_PROTOCOL_VERSION",
    "MAX_REMOTE_INSTRUCTION_BYTES",
    "RemoteAgentDescriptor",
    "RemoteAttemptPhase",
    "RemoteDispatchRequest",
    "RemoteObservation",
    "RemoteProjection",
    "RemoteProtocolBinding",
    "RemoteSnapshot",
    "RemoteTaskHandle",
    "RemoteTaskState",
]
