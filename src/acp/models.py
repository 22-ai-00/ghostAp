"""ACP event models — unified event abstraction over ACP session updates.

Converts raw ACP schema types (ToolCallStart, AgentMessageChunk, AgentPlanUpdate, etc.)
into simpler GhostAP-internal event objects for rendering and tracking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


@dataclass(frozen=True)
class ACPImageInfo:
    """Validated raster image emitted by an ACP agent."""

    image_id: str
    mime_type: str
    data: str
    name: str = "任务图片"
    source_uri: Optional[str] = None


@dataclass(frozen=True)
class ACPGoalInfo:
    """Trusted provider-owned goal state advertised through ACP metadata."""

    objective: str
    status: str
    control_method: str = ""
    token_budget: int | None = None
    time_used_seconds: float | None = None
    created_at: str | float | None = None

    @property
    def activity_state(self) -> bool | None:
        """Classify only the provider lifecycle states GhostAP understands."""
        if self.status == "active":
            return True
        if self.status in {"paused", "blocked", "completed"}:
            return False
        return None

    @property
    def is_active(self) -> bool:
        return self.activity_state is True


@dataclass(frozen=True)
class ACPSessionInfo:
    """Normalized control-plane state from an ACP session-info update."""

    goal_known: bool = False
    goal: ACPGoalInfo | None = None
    thread_status_observed: bool = False
    thread_status_known: bool = False
    thread_status: str = ""


class ACPEventType(Enum):
    """Event types produced from ACP session_update notifications."""

    TEXT_CHUNK = "text_chunk"
    THOUGHT_CHUNK = "thought_chunk"
    IMAGE_CHUNK = "image_chunk"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_UPDATE = "tool_call_update"
    TOOL_CALL_DONE = "tool_call_done"
    PLAN_UPDATE = "plan_update"


@dataclass
class ToolCallInfo:
    """Simplified tool call representation."""

    id: str
    title: str
    kind: str  # read/edit/delete/execute/think/search/fetch/other
    status: str  # pending/in_progress/completed/failed
    content: str = ""
    locations: list[str] = field(default_factory=list)
    # Optional structured result (best-effort, may be populated from local history)
    result: Optional[dict] = None
    # Stable child-agent identity normalized from provider namespaced metadata.
    # ``tool_call.id`` identifies one activity invocation and may change on every
    # interaction; this source id identifies the child thread across invocations.
    subagent_source_id: Optional[str] = None
    subagent_path: Optional[str] = None
    subagent_activity: Optional[str] = None
    # Parent collaboration snapshots. Entries are normalized to
    # {source_id, status, message} and never rendered without card-layer
    # sanitization.
    collaboration_tool: Optional[str] = None
    collaboration_receivers: tuple[str, ...] = ()
    collaboration_model: Optional[str] = None
    subagent_states: tuple[dict, ...] = ()
    # Sticky fail-closed marker for malformed provider child metadata.  This is
    # carried across same-id snapshots so a later clean update cannot erase
    # evidence that the lifecycle stream was structurally ambiguous.
    child_metadata_malformed: bool = False
    # Unbounded provider payload retained only for consumers that explicitly
    # request a full, sanitized projection (for example Workflow pagination).
    # Ordinary cards continue to use the bounded ``content`` field.
    full_content: object | None = None


@dataclass
class PlanEntryInfo:
    """Simplified plan entry."""

    content: str
    priority: str = "medium"  # high/medium/low
    status: str = "pending"  # pending/in_progress/completed


@dataclass
class PlanInfo:
    """Simplified plan."""

    entries: list[PlanEntryInfo] = field(default_factory=list)


@dataclass
class ACPEvent:
    """Unified ACP event, parsed from session_update notifications."""

    event_type: ACPEventType
    text: Optional[str] = None
    image: Optional[ACPImageInfo] = None
    tool_call: Optional[ToolCallInfo] = None
    plan: Optional[PlanInfo] = None
    source_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ACPSessionState:
    """ACP session state (serializable for persistence)."""

    session_id: str
    agent_type: str  # "coco" / "claude"
    cwd: str
    created_at: float = field(default_factory=time.time)
    message_count: int = 0
    is_active: bool = True
    last_active: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "agent_type": self.agent_type,
            "cwd": self.cwd,
            "created_at": self.created_at,
            "message_count": self.message_count,
            "is_active": self.is_active,
            "last_active": self.last_active,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ACPSessionState:
        return cls(
            session_id=data["session_id"],
            agent_type=data["agent_type"],
            cwd=data["cwd"],
            created_at=data.get("created_at", time.time()),
            message_count=data.get("message_count", 0),
            is_active=data.get("is_active", True),
            last_active=data.get("last_active", time.time()),
        )

@dataclass
class PromptResult:
    """Result of a prompt sent via ACP."""

    stop_reason: str  # end_turn/max_tokens/max_turn_requests/refusal/cancelled
    text: str = ""
    tool_calls: list[ToolCallInfo] = field(default_factory=list)
    # Structured tool results (best-effort). Each item is a dict with at least: kind/data/ts.
    tool_results: list[dict] = field(default_factory=list)
    plan: Optional[PlanInfo] = None
    modified_files: set[str] = field(default_factory=set)
    output_tokens: Optional[int] = None  # Output token count for discussion budget tracking
    goal: ACPGoalInfo | None = None
    cancellation_source: str | None = None

    # ---- aggregation helpers ----
    def add_text(self, chunk: str) -> None:
        if chunk:
            self.text += chunk

    def add_tool_call(self, tool_call: ToolCallInfo) -> None:
        if not tool_call:
            return
        self.tool_calls.append(tool_call)
        for p in tool_call.locations or []:
            if p:
                self.modified_files.add(p)

    def add_modified_file(self, path: str) -> None:
        if path:
            self.modified_files.add(path)

    def set_plan(self, plan: Optional[PlanInfo]) -> None:
        self.plan = plan

    def ingest_history(self, entries: list[dict]) -> None:
        """Ingest local ACP history entries.

        Expected entry format (from ACPHistoryStore):
            {"kind": "execute"|"read_file"|"write_file"|..., "data": {...}, "ts": ...}

        This method is tolerant to missing keys and malformed items.
        """
        if not entries:
            return
        for e in entries:
            if not isinstance(e, dict):
                continue
            kind = e.get("kind")
            data = e.get("data") if isinstance(e.get("data"), dict) else {}
            if kind:
                self.tool_results.append(e)
            if (
                kind == "permission"
                and str(data.get("outcome") or "").strip().casefold()
                in {"cancelled", "canceled"}
            ):
                self.cancellation_source = "permission_denied"
            # Track modified files from tool results
            if kind in ("write_file", "read_file"):
                p = data.get("path")
                if p:
                    self.modified_files.add(p)
