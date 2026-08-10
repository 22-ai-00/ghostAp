"""Pydantic data models for the Workflow Engine."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.spec_engine.models import ReviewAgentBinding

from .agent_pool import WorkflowAgentBinding
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
    SELECTING_AGENTS = "selecting_agents"
    GENERATING_SCRIPT = "generating_script"
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


class SubagentStatus(str, Enum):
    """User-visible status of an ACP-internal child agent."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Meta — describes the workflow script shape
# ---------------------------------------------------------------------------


class PhaseMeta(BaseModel):
    """A phase declaration inside the workflow meta."""

    title: str
    detail: str = ""


class AgentPlanEntry(BaseModel):
    """Displayable static or runtime-selected Agent assignment."""

    node: str
    role: str = ""
    agent_id: Optional[str] = Field(default=None, alias="agentId")
    runtime: bool = False
    candidate_agent_ids: list[str] = Field(default_factory=list, alias="candidateAgentIds")

    model_config = {"populate_by_name": True}


class WorkflowMeta(BaseModel):
    """Metadata exported from a workflow script's `export const meta = {...}`."""

    name: str
    description: str = ""
    phases: list[PhaseMeta] = Field(default_factory=list)
    max_concurrent: int = Field(default=DEFAULT_MAX_CONCURRENT, alias="maxConcurrent")
    tools: list[str] = Field(default_factory=list)
    patterns: list[str] = Field(default_factory=list)
    agent_plan: list[AgentPlanEntry] = Field(default_factory=list, alias="agentPlan")

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Agent call models
# ---------------------------------------------------------------------------


class AgentCallParams(BaseModel):
    """Parameters for a single agent() invocation from the JS runtime."""

    prompt: str
    agent_id: Optional[str] = None
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
    agent_id: Optional[str] = None
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


class SubagentProgress(BaseModel):
    """Latest safe observation for one ACP-internal child thread.

    ``source_id`` is an internal merge key only. Renderers must use list order
    for display labels and must never expose this opaque provider identifier.
    ACP currently does not forward an authoritative child list, so observations
    default to non-authoritative and must never drive the outer agent terminal
    state.
    """

    source_id: str
    status: SubagentStatus = SubagentStatus.RUNNING
    progress: str = ""
    model: Optional[str] = None
    authoritative: bool = False


class AgentProgress(BaseModel):
    """Runtime state of a single agent() call."""

    label: str = ""
    agent_id: Optional[str] = None
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
    activity_updated_at: Optional[float] = None
    attempt: int = 1
    result: Optional[str] = None  # Sanitized terminal result used by Workflow cards only.
    call_index: int = 0
    subagents: list[SubagentProgress] = Field(default_factory=list)


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
# PendingWorkflow — generated execution handoff state
# ---------------------------------------------------------------------------


class PendingWorkflow(BaseModel):
    """State owned while a workflow is generated and handed to runtime."""

    created_at: float = Field(default_factory=time.time)
    script_path: Optional[str] = None
    requirement: Optional[str] = None
    meta: Optional[dict[str, Any]] = None
    initiator_user_id: Optional[str] = None
    engine_session_key: Optional[str] = None
    selected_tools: Optional[list[str]] = None
    orchestrator_agent: Optional[str] = None
    script_hash: Optional[str] = None  # SHA-256 checked again at the execution handoff.
    orchestrator_binding: Optional[ReviewAgentBinding] = None
    review_agents: list[ReviewAgentBinding] = Field(default_factory=list)
    auto_reviewer: bool = True
    agent_pool: tuple[WorkflowAgentBinding, ...] = ()
    orchestrator_agent_id: Optional[str] = None
    orchestrator_was_auto: bool = False
    selection_session_key: Optional[str] = None
    project_id: Optional[str] = None
    next_agent_sequence: int = 1
    recommended_agents: list[dict[str, Any]] = Field(default_factory=list)
    draft_tool_name: Optional[str] = None
    draft_model_name: Optional[str] = None
    draft_profile: Optional[str] = None
    draft_effort: Optional[str] = None
    selection_error: Optional[str] = None


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
    # Short-lived generated script state while GENERATING_SCRIPT owns the run.
    pending: Optional[PendingWorkflow] = None
    # Runtime state — set when execution begins
    initiator_user_id: Optional[str] = None  # Who started this workflow (for stop auth)
    selected_tools: Optional[list[str]] = None  # Active tool whitelist during execution
    tool_model_map: dict[str, Optional[str]] = Field(default_factory=dict)  # tool_name -> confirmed model (None means the tool default)
    # Immutable WorkflowRunSpec is serialized here for state/history audit;
    # the live engine keeps the frozen object itself.
    run_spec: Optional[dict[str, Any]] = None
    reviewer_evidence: list[ReviewerEvidence] = Field(default_factory=list)
    def start_execution(self) -> None:
        """Transition from generated state to execution.

        Migrates initiator_user_id, selected_tools, and tool-model bindings
        from pending to runtime fields, then clears the pending state.
        """
        if self.pending:
            if self.pending.initiator_user_id is not None:
                self.initiator_user_id = self.pending.initiator_user_id
            if self.pending.selected_tools is not None:
                self.selected_tools = self.pending.selected_tools
            # Build tool-to-model mapping from automatic bindings.
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
        """Deserialize current persisted Workflow state."""
        return cls.model_validate(data)
