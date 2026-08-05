"""RuntimeStats: snapshot of runtime context consumed by banner/sticky rendering."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeStats:
    """Snapshot of engine runtime context for banner / sticky rendering.

    Engine renderers populate fields relevant to their engine; others stay None.
    """

    elapsed_seconds: float = 0.0
    deep_phase: str | None = None
    spec_cycle: int | None = None
    spec_perspective: str | None = None
