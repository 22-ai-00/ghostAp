"""EngineSnapshot: Read-only DTO for renderer-engine decoupling.

Provides a frozen point-in-time view of engine state for renderers,
eliminating direct access to mutable engine manager internals.

Usage in renderers:
    snapshot = self.ctx.deep_engine_manager.snapshot(chat_id, root_path)
    if snapshot:
        title = snapshot.engine_name
        tool_count = snapshot.tool_calls_count
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class EngineSnapshot:
    """Frozen point-in-time snapshot of engine state for rendering.

    Contains all data a renderer needs to build cards, without exposing
    mutable engine internals. Fields are intentionally denormalized to
    avoid requiring renderers to traverse nested object graphs.
    """

    # Core identity
    engine_name: str = ""
    root_path: str = ""
    project_id: str = ""

    # Progress (Deep-specific, but available for all)
    tool_calls_count: int = 0
    completed_steps: int = 0
    total_steps: int = 0

    # Criteria (Spec)
    satisfied_count: int = 0
    total_criteria: int = 0

    # Timing
    duration_seconds: Optional[float] = None

    # Status
    status: str = ""  # Engine project status value string
    is_running: bool = False

    # Cycle info (Spec)
    iteration_count: int = 0
    cycle_count: int = 0
    cycle_count_total: int = 0

    # Extended data for detailed rendering (opaque to DTO contract)
    # Renderers that need deep access (iteration details, cycle reviews)
    # can retrieve typed data from this dict.
    ext: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Wrap ext dict in MappingProxyType to enforce true immutability."""
        if not isinstance(self.ext, MappingProxyType):
            object.__setattr__(self, "ext", MappingProxyType(self.ext))
