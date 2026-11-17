"""Deep Engine progress tracking from ACP events.

Tracks plan entries, tool calls, and modified files from ACP events
to provide structured progress information for the Deep Engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..acp.models import PlanInfo, ToolCallInfo
from ..utils.text import make_progress_bar


def _truncate_nested_data(data: Any, max_depth: int = 10, current_depth: int = 0) -> Any:
    """Truncate nested data structures to prevent recursion explosion."""
    if current_depth >= max_depth:
        return "[TRUNCATED: MAX DEPTH EXCEEDED]"

    if isinstance(data, dict):
        return {k: _truncate_nested_data(v, max_depth, current_depth + 1) for k, v in data.items()}
    elif isinstance(data, list):
        return [_truncate_nested_data(item, max_depth, current_depth + 1) for item in data]
    elif isinstance(data, tuple):
        return tuple(_truncate_nested_data(item, max_depth, current_depth + 1) for item in data)
    return data


@dataclass
class DeepProgress:
    """Tracks deep execution progress extracted from ACP events."""

    plan_entries: list[dict] = field(default_factory=list)  # [{content, status, priority}]
    tool_calls: list[ToolCallInfo] = field(default_factory=list)
    modified_files: set[str] = field(default_factory=set)
    _text_chunks: list[str] = field(default_factory=list)

    def update_plan(self, plan: PlanInfo) -> None:
        self.plan_entries = [{"content": e.content, "status": e.status, "priority": e.priority} for e in plan.entries]

    def record_tool(self, tool: ToolCallInfo) -> None:
        # Prevent stack explosion on highly nested tool results
        if tool.result:
            tool.result = _truncate_nested_data(tool.result)

        self.tool_calls.append(tool)
        for loc in tool.locations:
            self.modified_files.add(loc)

    @property
    def text_buffer(self) -> str:
        return "".join(self._text_chunks)

    def append_text(self, text: str) -> None:
        self._text_chunks.append(text)

    @property
    def completed_steps(self) -> int:
        return sum(1 for e in self.plan_entries if e["status"] == "completed")

    @property
    def total_steps(self) -> int:
        return len(self.plan_entries)

    @property
    def progress_percent(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return (self.completed_steps / self.total_steps) * 100

    @property
    def progress_bar(self) -> str:
        return make_progress_bar(self.completed_steps, self.total_steps)
