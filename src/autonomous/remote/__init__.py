"""Backend-neutral contracts for durable remote Agent dispatch."""

from .models import (
    A2A_SPEC_RELEASE,
    A2A_WIRE_PROTOCOL_VERSION,
    RemoteAgentDescriptor,
    RemoteAttemptPhase,
    RemoteDispatchRequest,
    RemoteObservation,
    RemoteProjection,
    RemoteProtocolBinding,
    RemoteSnapshot,
    RemoteTaskHandle,
    RemoteTaskState,
)
from .ports import RemoteAgentDispatchPort
from .projection import RemoteProjectionError, rebuild_remote_projection

__all__ = [
    "A2A_SPEC_RELEASE",
    "A2A_WIRE_PROTOCOL_VERSION",
    "RemoteAgentDescriptor",
    "RemoteAgentDispatchPort",
    "RemoteAttemptPhase",
    "RemoteDispatchRequest",
    "RemoteObservation",
    "RemoteProjection",
    "RemoteProjectionError",
    "RemoteProtocolBinding",
    "RemoteSnapshot",
    "RemoteTaskHandle",
    "RemoteTaskState",
    "rebuild_remote_projection",
]
