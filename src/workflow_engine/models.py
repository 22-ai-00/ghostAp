"""Pydantic data models for the Workflow Engine."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.spec_engine.review_agents import ReviewAgentBinding

from .constants import (
    AGENT_CALL_TIMEOUT_S,
    DEFAULT_MAX_CONCURRENT,
)
from .run_spec import WorkflowRunSpec as WorkflowRunSpec

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class WorkflowStatus(str, Enum):
    """Lifecycle states of a WorkflowProject."""

    IDLE = "idle"
    GENERATING_SCRIPT = "generating_script"
    AWAITING_AGENT_SELECT = "awaiting_agent_select"  # User selecting orchestrator agent
    AWAITING_TOOL_SELECT = "awaiting_tool_select"  # User selecting tools before script generation
    AWAITING_CONFIRM = "awaiting_confirm"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentStatus(str, Enum):
    """Status of a single agent() call."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CACHED = "cached"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Meta — describes the workflow script shape
# ---------------------------------------------------------------------------


class PhaseMeta(BaseModel):
    """A phase declaration inside the workflow meta."""

    title: str
    detail: str = ""


class WorkflowMeta(BaseModel):
    """Metadata exported from a workflow script's `export const meta = {...}`."""

    name: str
    description: str = ""
    phases: list[PhaseMeta] = Field(default_factory=list)
    max_concurrent: int = Field(default=DEFAULT_MAX_CONCURRENT, alias="maxConcurrent")
    tools: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Agent call models
# ---------------------------------------------------------------------------


class AgentCallParams(BaseModel):
    """Parameters for a single agent() invocation from the JS runtime."""

    prompt: str
    tool: str = ""
    model: Optional[str] = None
    role: Optional[str] = None
    output_schema: Optional[dict[str, Any]] = Field(default=None, alias="schema")
    label: Optional[str] = None
    phase: Optional[str] = None
    timeout: int = AGENT_CALL_TIMEOUT_S


class AgentCallResult(BaseModel):
    """Result of executing a single agent() call."""

    output: Optional[str] = None
    parsed: Optional[dict[str, Any]] = None
    # Preserve the backend's actual PromptResult terminal reason. A transport
    # returning without raising is not proof that user work completed.
    stop_reason: Optional[str] = None
    token_usage: int = 0
    duration_s: float = 0.0
    error: Optional[str] = None
    cached: bool = False
    tool: str = ""
    model: Optional[str] = None


# ---------------------------------------------------------------------------
# Journal entry
# ---------------------------------------------------------------------------


class JournalEntry(BaseModel):
    """A cached agent() call result in the journal."""

    key: str
    result: AgentCallResult
    timestamp: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Phase tracking
# ---------------------------------------------------------------------------


class PhaseProgress(BaseModel):
    """Runtime state of a phase during execution."""

    title: str
    agents: list[AgentProgress] = Field(default_factory=list)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


class AgentProgress(BaseModel):
    """Runtime state of a single agent() call."""

    label: str = ""
    tool: str = ""
    model: Optional[str] = None
    role: Optional[str] = None
    task_summary: str = ""
    status: AgentStatus = AgentStatus.PENDING
    token_usage: int = 0
    duration_s: float = 0.0
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    current_activity: str = ""  # Live activity hint (e.g. "read_file src/...", "writing code...")


class ReviewerEvidence(BaseModel):
    """Durable proof of one committed, independent reviewer invocation."""

    reviewer_index: int
    selection_key: str = ""
    display_name: str = ""
    tool: str
    model: Optional[str] = None
    status: str
    output: Optional[str] = None
    stop_reason: Optional[str] = None
    error: Optional[str] = None
    cached: bool = False
    token_usage: int = 0
    duration_s: float = 0.0
    started_at: float = Field(default_factory=time.time)
    finished_at: Optional[float] = None


# ---------------------------------------------------------------------------
# Workflow metrics
# ---------------------------------------------------------------------------


class WorkflowMetrics(BaseModel):
    """Execution metrics for observability."""

    total_agents: int = 0
    completed_agents: int = 0
    failed_agents: int = 0
    cached_agents: int = 0
    total_tokens: int = 0
    total_duration_s: float = 0.0
    phases_completed: int = 0


# ---------------------------------------------------------------------------
# PendingConfirmation — pending execution state
# ---------------------------------------------------------------------------


class PendingConfirmation(BaseModel):
    """State for a workflow awaiting user confirmation before execution."""

    created_at: float = Field(default_factory=time.time)
    script_path: Optional[str] = None
    requirement: Optional[str] = None
    meta: Optional[dict[str, Any]] = None
    is_fallback: bool = False
    initiator_user_id: Optional[str] = None
    engine_session_key: Optional[str] = None
    selected_tools: Optional[list[str]] = None
    tools_mismatch: bool = False
    orchestrator_agent: Optional[str] = None  # Selected orchestrator agent
    is_template_hint: Optional[str] = None  # Set when launched via `/wf <template>` so downstream handlers can initialize default tool selection from template meta.tools
    script_hash: Optional[str] = None  # SHA-256 of the script content at generation time — used for TOCTOU checks at confirm-time so tampered scripts are rejected before execution.
    # --- New selection flow fields ---
    orchestrator_binding: Optional[ReviewAgentBinding] = None  # ReviewAgentBinding for the main agent
    review_agents: Optional[list[ReviewAgentBinding]] = None  # ReviewAgentBinding list for review pool
    # None is reserved for restored legacy selections. New UI flows always
    # persist True (Auto) or False (explicit reviewers).
    auto_reviewer: Optional[bool] = None


# ---------------------------------------------------------------------------
# WorkflowProject — top-level state
# ---------------------------------------------------------------------------


class WorkflowProject(BaseModel):
    """Top-level state of a workflow execution."""

    workflow_id: str = ""
    name: str = ""
    description: str = ""
    status: WorkflowStatus = WorkflowStatus.IDLE
    requirement: str = ""
    script_path: Optional[str] = None
    meta: Optional[WorkflowMeta] = None
    metrics: WorkflowMetrics = Field(default_factory=WorkflowMetrics)
    phases: list[PhaseProgress] = Field(default_factory=list)
    result: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    # Pending confirmation state — set during AWAITING_CONFIRM / AWAITING_TOOL_SELECT
    pending: Optional[PendingConfirmation] = None
    # Runtime state — set when execution begins
    initiator_user_id: Optional[str] = None  # Who started this workflow (for stop auth)
    selected_tools: Optional[list[str]] = None  # Active tool whitelist during execution
    tool_model_map: dict[str, Optional[str]] = Field(default_factory=dict)  # tool_name -> confirmed model (None means the tool default)
    # Immutable WorkflowRunSpec is serialized here for state/history audit;
    # the live engine keeps the frozen object itself.
    run_spec: Optional[dict[str, Any]] = None
    reviewer_evidence: list[ReviewerEvidence] = Field(default_factory=list)
    # Selection state storage for WorktreeSelectionController
    orchestrator_selection_state: Optional[dict] = None
    review_selection_state: Optional[dict] = None

    def start_execution(self) -> None:
        """Transition from pending confirmation to execution.

        Migrates initiator_user_id, selected_tools, and tool-model bindings
        from pending to runtime fields, then clears the pending state.
        """
        if self.pending:
            if self.pending.initiator_user_id is not None:
                self.initiator_user_id = self.pending.initiator_user_id
            if self.pending.selected_tools is not None:
                self.selected_tools = self.pending.selected_tools
            # Build tool→model mapping from user selections
            mapping: dict[str, Optional[str]] = {}
            if self.pending.orchestrator_binding:
                b = self.pending.orchestrator_binding
                if b.tool_name:
                    mapping[b.tool_name] = None if b.use_default_model else b.model_name
            for agent in (self.pending.review_agents or []):
                if agent.tool_name:
                    if agent.tool_name not in mapping:
                        mapping[agent.tool_name] = None if agent.use_default_model else agent.model_name
            self.tool_model_map = mapping
            self.pending = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowProject":
        """Deserialize from persistence.

        Handles legacy format with flat pending_* fields by migrating them
        into the PendingConfirmation sub-model.
        """
        # Check for legacy flat pending_* fields and migrate
        legacy_fields = {
            "script_path": data.pop("pending_script_path", None),
            "requirement": data.pop("pending_requirement", None),
            "meta": data.pop("pending_meta", None),
            "is_fallback": data.pop("pending_is_fallback", False),
            "initiator_user_id": data.pop("pending_initiator_user_id", None),
            "engine_session_key": data.pop("pending_engine_session_key", None),
            "selected_tools": data.pop("pending_selected_tools", None),
            "tools_mismatch": data.pop("pending_tools_mismatch", False),
        }
        # Only create pending if any legacy field has a non-default value
        has_legacy = any(
            v is not None and v is not False and v != []
            for v in legacy_fields.values()
        )
        if has_legacy and "pending" not in data:
            data["pending"] = PendingConfirmation(**legacy_fields).model_dump()
        return cls.model_validate(data)
