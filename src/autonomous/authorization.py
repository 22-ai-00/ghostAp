"""Shared authorization scope for employee message execution."""

from __future__ import annotations

from enum import StrEnum


class EmployeeAuthorizationScope(StrEnum):
    """Frozen authority selected before Context or execution is allowed."""

    MANAGED_GROUP = "managed_group"
    OWNER_P2P = "owner_p2p"


__all__ = ["EmployeeAuthorizationScope"]
