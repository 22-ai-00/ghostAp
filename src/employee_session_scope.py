"""Low-dependency typed outcome markers for employee session boundaries."""

from __future__ import annotations

from enum import StrEnum


class EmployeeSessionOutcome(StrEnum):
    TIMEOUT = "timeout"
    CANCELED = "canceled"


def record_employee_session_outcome(outcome: str | EmployeeSessionOutcome) -> None:
    """Validate an employee session outcome marker."""
    EmployeeSessionOutcome(outcome)


__all__ = [
    "EmployeeSessionOutcome",
    "record_employee_session_outcome",
]
