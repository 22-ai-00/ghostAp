"""Dependency-light shared contracts for parent-owned employee Channels."""

from __future__ import annotations

from enum import Enum


class EmployeeChannelOutboundError(RuntimeError):
    """One employee Channel could not complete an outbound operation."""


class EmployeeChannelGenerationChanged(EmployeeChannelOutboundError, ValueError):
    """The selected employee Channel generation changed before delivery."""


class EmployeeChannelOutboundTimeout(EmployeeChannelOutboundError, TimeoutError):
    """One employee Channel did not acknowledge an outbound operation in time."""


class ChannelProcessState(str, Enum):
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    CRASHED = "crashed"


__all__ = [
    "ChannelProcessState",
    "EmployeeChannelGenerationChanged",
    "EmployeeChannelOutboundError",
    "EmployeeChannelOutboundTimeout",
]
