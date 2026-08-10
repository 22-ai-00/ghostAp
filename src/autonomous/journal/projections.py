"""Journal replay into the active Employee Department projection."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from ..workforce.projection import WorkforceProjectionState
from .frame import JournalEvent, TransactionFrame


class ProjectionError(RuntimeError):
    """A workforce projection reducer encountered an inconsistency."""


@dataclass
class ProjectionState(WorkforceProjectionState):
    """Canonical workforce state plus the durable Journal replay cursor."""

    cursor_sequence: int = 0
    cursor_hash: str = ""


def apply_event(state: ProjectionState, event: JournalEvent) -> None:
    """Apply a workforce event and ignore retired or foreign event domains."""
    from ..workforce.projection import apply_workforce_event

    apply_workforce_event(state, event)


def apply_frame(state: ProjectionState, frame: TransactionFrame) -> None:
    """Apply workforce events while advancing across every committed frame."""
    from ..workforce.projection import (
        normalize_workforce_aggregate_versions,
        validate_workforce_frame_events,
    )

    validate_workforce_frame_events(frame.events)
    for event in frame.events:
        apply_event(state, event)
    normalize_workforce_aggregate_versions(
        state,
        frame.aggregate_versions,
        frame.events,
    )
    state.cursor_sequence = frame.sequence
    state.cursor_hash = frame.frame_hash


class ProjectionRepository:
    """Materialize active workforce state by replaying Journal frames."""

    def __init__(self) -> None:
        self._state = ProjectionState()

    @property
    def state(self) -> ProjectionState:
        return self._state

    def rebuild(self, frames: Iterator[TransactionFrame]) -> ProjectionState:
        self._state = ProjectionState()
        for frame in frames:
            apply_frame(self._state, frame)
        return self._state

    def apply(self, frame: TransactionFrame) -> None:
        apply_frame(self._state, frame)
