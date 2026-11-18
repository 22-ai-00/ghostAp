"""Immutable Employee Department domain models."""

from .employees import BotPrincipal, EmployeeDefinition
from .enums import (
    EmployeeIdOrigin,
    EmployeeState,
    WorkerType,
)
from .ids import freeze, new_id, thaw

__all__ = [
    "BotPrincipal",
    "EmployeeDefinition",
    "EmployeeIdOrigin",
    "EmployeeState",
    "WorkerType",
    "freeze",
    "new_id",
    "thaw",
]
