"""Slock Engine data models.

Core dataclasses and enums for the multi-Agent collaboration engine.
"""

from __future__ import annotations

import re
import threading
import time
import uuid
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# Escalation resolution option constants (bilingual support)
RETRY_OPTIONS: frozenset[str] = frozenset({"Retry", "重试"})
SKIP_OPTIONS: frozenset[str] = frozenset({"Skip", "跳过"})
ABORT_OPTIONS: frozenset[str] = frozenset({"Abort", "中止"})


class AgentStatus(Enum):
    """Agent lifecycle state machine."""

    IDLE = "idle"
    WAKING = "waking"
    THINKING = "thinking"
    RUNNING = "running"
    CHECKING = "checking"
    SENDING = "sending"
    MOVING = "moving"
    DISCUSSING = "discussing"
    PENDING_DISCUSSION = "pending_discussion"


class DiscussionStatus(Enum):
    """Discussion thread lifecycle states."""

    ACTIVE = "active"  # Triggered: discussion started, rounds in progress
    CONVERGED = "converged"  # Triggered: convergence signals detected or arbiter concludes agreement
    TIMEOUT = "timeout"  # Triggered: watchdog timer exceeds slock_discussion_timeout
    MAX_ROUNDS_REACHED = "max_rounds_reached"  # Triggered: round count exceeds slock_max_discussion_rounds
    BUDGET_EXHAUSTED = "budget_exhausted"  # Triggered: token usage exceeds slock_discussion_token_budget
    MANUALLY_STOPPED = "manually_stopped"  # Triggered: user issues /discuss stop or admin intervention


class CouncilStatus(Enum):
    """Council run lifecycle stages."""

    STARTING = "starting"
    STAGE1_RUNNING = "stage1_running"
    STAGE1_DONE = "stage1_done"
    STAGE2_RUNNING = "stage2_running"
    STAGE2_DONE = "stage2_done"
    STAGE3_RUNNING = "stage3_running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStatus(Enum):
    """Task lifecycle states."""

    TODO = "todo"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    DONE = "done"


class PlanStepStatus(Enum):
    """Step-level status for collaboration plan steps (decoupled from TaskStatus)."""

    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    SKIPPED = "skipped"
    TIMED_OUT = "timed_out"


class EscalationLevel(Enum):
    """Escalation severity levels."""

    WARNING = "warning"      # Agent can continue but wants guidance
    BLOCKED = "blocked"      # Agent cannot proceed, needs admin decision
    CRITICAL = "critical"    # Fatal error, immediate admin attention needed


@dataclass
class EscalationRequest:
    """A request for admin intervention from an agent."""

    escalation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    agent_name: str = ""
    task_id: Optional[str] = None
    level: EscalationLevel = EscalationLevel.BLOCKED
    reason: str = ""
    context: str = ""  # Additional context (truncated conversation, error details)
    options: tuple[str, ...] = field(default_factory=tuple)  # Suggested resolution options (frozen)
    resolved: bool = False
    resolution: str = ""  # Admin's resolution choice
    created_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    card_message_id: Optional[str] = None  # Feishu message_id of the escalation card (for update)

    def __post_init__(self) -> None:
        # Freeze options to tuple for thread-safety — prevents mutation after creation.
        if isinstance(self.options, list):
            object.__setattr__(self, "options", tuple(self.options))

    def to_dict(self) -> dict:
        return {
            "escalation_id": self.escalation_id,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "task_id": self.task_id,
            "level": self.level.value,
            "reason": self.reason,
            "context": self.context,
            "options": list(self.options),
            "resolved": self.resolved,
            "resolution": self.resolution,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "card_message_id": self.card_message_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EscalationRequest":
        return cls(
            escalation_id=data.get("escalation_id", str(uuid.uuid4())),
            agent_id=data.get("agent_id", ""),
            agent_name=data.get("agent_name", ""),
            task_id=data.get("task_id"),
            level=EscalationLevel(data.get("level", "blocked")),
            reason=data.get("reason", ""),
            context=data.get("context", ""),
            options=data.get("options", []),
            resolved=data.get("resolved", False),
            resolution=data.get("resolution", ""),
            created_at=data.get("created_at", time.time()),
            resolved_at=data.get("resolved_at"),
            card_message_id=data.get("card_message_id"),
        )


# Agent role color mapping for card rendering
AGENT_ROLE_COLORS: dict[str, str] = {
    "coder": "blue",
    "writer": "green",
    "reviewer": "orange",
    "tester": "purple",
    "planner": "red",
    "architect": "indigo",
    "custom": "grey",
}

# Single Source of Truth: Agent status → Feishu card background color.
# All card rendering code MUST use this map (do NOT create local duplicates).
#
# DEPRECATED: This map will be moved to src/slock_engine/card_templates/common.py
# in a future release. Import from there instead of models.py.
# The STATUS_BG_STYLE_MAP in card_templates/common.py is the preferred
# three-tier visual system for modern card rendering.
_AGENT_STATUS_BG_COLOR_MAP_INTERNAL: dict[AgentStatus, str] = {
    AgentStatus.IDLE: "green",
    AgentStatus.WAKING: "turquoise",
    AgentStatus.THINKING: "yellow",
    AgentStatus.RUNNING: "blue",
    AgentStatus.CHECKING: "wathet",
    AgentStatus.SENDING: "grey",
    AgentStatus.MOVING: "orange",
    AgentStatus.DISCUSSING: "purple",
    AgentStatus.PENDING_DISCUSSION: "yellow",
}


class _DeprecatedColorMap(dict):
    """Dict wrapper that emits deprecation warning on first access."""

    _warned: bool = False

    def __getitem__(self, key):
        if not self._warned:
            warnings.warn(
                "AGENT_STATUS_BG_COLOR_MAP is deprecated and will be moved to "
                "src/slock_engine/card_templates/common.py in a future release. "
                "Use STATUS_BG_STYLE_MAP for the new three-tier visual system.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._warned = True
        return super().__getitem__(key)

    def get(self, key, default=None):
        if not self._warned:
            warnings.warn(
                "AGENT_STATUS_BG_COLOR_MAP is deprecated and will be moved to "
                "src/slock_engine/card_templates/common.py in a future release. "
                "Use STATUS_BG_STYLE_MAP for the new three-tier visual system.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._warned = True
        return super().get(key, default)


AGENT_STATUS_BG_COLOR_MAP: dict[AgentStatus, str] = _DeprecatedColorMap(
    _AGENT_STATUS_BG_COLOR_MAP_INTERNAL
)


@dataclass
class AgentIdentity:
    """Agent identity definition — persisted as identity.yaml."""

    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    emoji: str = "🤖"
    agent_type: str = "coco"  # coco/claude/codex/gemini/ttadk
    model_name: str = ""
    model_profile: str = "standard"
    reasoning_effort: str = "default"
    system_prompt: str = ""
    role: str = "custom"  # coder/writer/reviewer/tester/planner/architect/custom
    permissions: list[str] = field(default_factory=lambda: ["shell", "file_write", "git"])
    memory_path: str = ""
    notes_path: str = ""
    workspace_path: str = ""
    owner_group: str = ""  # chat_id of owning group
    member_groups: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    personality_traits: list[str] = field(default_factory=list)  # e.g. ['严谨', '注重细节']
    capabilities: list[str] = field(default_factory=list)
    security_profile: str = "legacy"
    # Wake policy: "" (inherit channel/settings default) | "on_mention" | "smart_judge".
    # Free-form to leave room for future modes; the router treats unknown values
    # as the safe default (smart_judge) and only short-circuits on the exact
    # token "on_mention".
    wake_policy: str = ""

    def __post_init__(self) -> None:
        # Sanitize agent_id to prevent path traversal; dots are allowed (e.g. model versions like v3.5)
        self.agent_id = re.sub(r'[^A-Za-z0-9_.:-]+', '_', self.agent_id)
        if '..' in self.agent_id or self.agent_id.startswith('.'):
            self.agent_id = self.agent_id.lstrip('.').replace('..', '_')
        if self.owner_group and self.owner_group not in self.member_groups:
            self.member_groups.append(self.owner_group)
        if self.security_profile not in {"legacy", "employee_v1"}:
            raise ValueError("unknown Slock agent security profile")

    @property
    def display_name(self) -> str:
        return f"{self.emoji} {self.name}" if self.name else f"{self.emoji} Agent"

    @property
    def card_color(self) -> str:
        return AGENT_ROLE_COLORS.get(self.role, "grey")

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "emoji": self.emoji,
            "agent_type": self.agent_type,
            "model_name": self.model_name,
            "model_profile": self.model_profile,
            "reasoning_effort": self.reasoning_effort,
            "system_prompt": self.system_prompt,
            "role": self.role,
            "permissions": self.permissions,
            "memory_path": self.memory_path,
            "notes_path": self.notes_path,
            "workspace_path": self.workspace_path,
            "owner_group": self.owner_group,
            "member_groups": self.member_groups,
            "created_at": self.created_at,
            "personality_traits": self.personality_traits,
            "capabilities": self.capabilities,
            "security_profile": self.security_profile,
            "wake_policy": self.wake_policy,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AgentIdentity:
        return cls(
            agent_id=data.get("agent_id", str(uuid.uuid4())),
            name=data.get("name", ""),
            emoji=data.get("emoji", "🤖"),
            agent_type=data.get("agent_type", "coco"),
            model_name=data.get("model_name", ""),
            model_profile=data.get("model_profile", "standard"),
            reasoning_effort=data.get("reasoning_effort", "default"),
            system_prompt=data.get("system_prompt", ""),
            role=data.get("role", "custom"),
            permissions=data.get("permissions", ["shell", "file_write", "git"]),
            memory_path=data.get("memory_path", ""),
            notes_path=data.get("notes_path", ""),
            workspace_path=data.get("workspace_path", ""),
            owner_group=data.get("owner_group", ""),
            member_groups=data.get("member_groups", []),
            created_at=data.get("created_at", time.time()),
            personality_traits=data.get("personality_traits", []),
            capabilities=data.get("capabilities", []),
            security_profile=data.get("security_profile", "legacy"),
            wake_policy=data.get("wake_policy", ""),
        )


@dataclass
class TaskTimelineEvent:
    """A single event in a task's lifecycle timeline."""

    event_type: str  # e.g., "claimed", "started", "completed", "rejected", "blocked"
    agent_id: str
    timestamp: float  # unix timestamp
    detail: str = ""


@dataclass
class SlockTask:
    """A task that can be claimed and executed by an Agent."""

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    status: TaskStatus = TaskStatus.TODO
    claimed_by: Optional[str] = None  # agent_id
    claimed_at: Optional[float] = None
    created_in: str = ""  # channel_id
    created_at: float = field(default_factory=time.time)
    resolved_reason: Optional[str] = None  # Non-None for abnormal completion (e.g. "超时中止")
    reasoning_snapshot: str = ""  # Snapshot of reasoning state when task was created
    chain_next_agent_id: str = ""  # Next agent in chain to hand off to
    predecessor_agent_name: str = ""  # Role-based breadcrumb: name of agent who handed off this task
    sub_tasks: list[str] = field(default_factory=list)  # child task IDs
    parent_task_id: Optional[str] = None  # parent task ID (for sub-tasks)
    collaborators: list[str] = field(default_factory=list)  # agent_ids participating
    progress_pct: int = 0  # 0-100 progress percentage
    discussion_ids: list[str] = field(default_factory=list)  # linked DiscussionThread IDs
    timeline: list[TaskTimelineEvent] = field(default_factory=list)
    # Proactive Slock: SLA tracking for patrol-based follow-up
    sla_seconds: float = 300.0  # Default 5 min SLA
    deadline_at: Optional[float] = None  # Unix timestamp; auto-set on claim if None
    sender_id: str = ""  # Original message sender (for result delivery tracking)
    execution_result: str = ""  # Agent output summary stored on completion (for context passing)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "content": self.content,
            "status": self.status.value,
            "claimed_by": self.claimed_by,
            "claimed_at": self.claimed_at,
            "created_in": self.created_in,
            "created_at": self.created_at,
            "resolved_reason": self.resolved_reason,
            "reasoning_snapshot": self.reasoning_snapshot,
            "chain_next_agent_id": self.chain_next_agent_id,
            "predecessor_agent_name": self.predecessor_agent_name,
            "sub_tasks": self.sub_tasks,
            "parent_task_id": self.parent_task_id,
            "collaborators": self.collaborators,
            "progress_pct": self.progress_pct,
            "discussion_ids": self.discussion_ids,
            "sla_seconds": self.sla_seconds,
            "deadline_at": self.deadline_at,
            "sender_id": self.sender_id,
            "execution_result": self.execution_result,
            "timeline": [
                {"event_type": e.event_type, "agent_id": e.agent_id, "timestamp": e.timestamp, "detail": e.detail}
                for e in self.timeline
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> SlockTask:
        return cls(
            task_id=data.get("task_id", str(uuid.uuid4())),
            content=data.get("content", ""),
            status=TaskStatus(data.get("status", "todo")),
            claimed_by=data.get("claimed_by"),
            claimed_at=data.get("claimed_at"),
            created_in=data.get("created_in", ""),
            created_at=data.get("created_at", time.time()),
            resolved_reason=data.get("resolved_reason"),
            reasoning_snapshot=data.get("reasoning_snapshot", ""),
            chain_next_agent_id=data.get("chain_next_agent_id", ""),
            predecessor_agent_name=data.get("predecessor_agent_name", ""),
            sub_tasks=data.get("sub_tasks", []),
            parent_task_id=data.get("parent_task_id"),
            collaborators=data.get("collaborators", []),
            progress_pct=data.get("progress_pct", 0),
            discussion_ids=data.get("discussion_ids", []),
            sla_seconds=data.get("sla_seconds", 300.0),
            deadline_at=data.get("deadline_at"),
            sender_id=data.get("sender_id", ""),
            execution_result=data.get("execution_result", ""),
            timeline=[
                TaskTimelineEvent(
                    event_type=e.get("event_type", ""),
                    agent_id=e.get("agent_id", ""),
                    timestamp=e.get("timestamp", 0.0),
                    detail=e.get("detail", ""),
                )
                for e in data.get("timeline", [])
            ],
        )


@dataclass
class SlockChannel:
    """A Feishu group with slock mode activated."""

    channel_id: str = ""
    name: str = ""
    agents: list[str] = field(default_factory=list)  # agent_id list
    shared_memory_path: str = ""
    team_name: str = ""
    owner_id: str = ""  # User who created this team (for permission checks)
    project_id: str = ""  # Durable grant identity; standalone teams use team:<name>.
    created_at: float = field(default_factory=time.time)
    bootstrap_failed: bool = False
    # Channel-level wake policy override: "" inherits settings default. Effective
    # wake policy precedence is Agent > Channel > Settings.
    wake_policy: str = ""

    def to_dict(self) -> dict:
        return {
            "channel_id": self.channel_id,
            "name": self.name,
            "agents": self.agents,
            "shared_memory_path": self.shared_memory_path,
            "team_name": self.team_name,
            "owner_id": self.owner_id,
            "project_id": self.project_id,
            "created_at": self.created_at,
            "bootstrap_failed": self.bootstrap_failed,
            "wake_policy": self.wake_policy,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SlockChannel:
        return cls(
            channel_id=data.get("channel_id", ""),
            name=data.get("name", ""),
            agents=data.get("agents", []),
            shared_memory_path=data.get("shared_memory_path", ""),
            team_name=data.get("team_name", ""),
            owner_id=data.get("owner_id", ""),
            created_at=data.get("created_at", time.time()),
            bootstrap_failed=data.get("bootstrap_failed", False),
            wake_policy=data.get("wake_policy", ""),
        )


@dataclass
class SlockMemory:
    """Three-section memory structure for an Agent."""

    role: str = ""  # Role definition
    key_knowledge: str = ""  # Long-term knowledge
    active_context: str = ""  # Active working context
    archived_context: str = ""  # Archived context from cross-group moves
    _version: int = 0  # OCC version for concurrent write detection

    def to_markdown(self) -> str:
        sections = []
        if self.role:
            sections.append(f"# Role\n{self.role}")
        if self.key_knowledge:
            sections.append(f"# Key Knowledge\n{self.key_knowledge}")
        if self.active_context:
            sections.append(f"# Active Context\n{self.active_context}")
        if self.archived_context:
            sections.append(f"# Archived Context\n{self.archived_context}")
        content = "\n\n".join(sections) if sections else ""
        # Append version comment for dual-version OCC validation
        if self._version > 0:
            content = f"{content}\n<!-- version: {self._version} -->" if content else f"<!-- version: {self._version} -->"
        return content

    @classmethod
    def from_markdown(cls, content: str) -> SlockMemory:
        import re

        role = ""
        key_knowledge = ""
        active_context = ""
        archived_context = ""
        embedded_version = 0

        if not content.strip():
            return cls()

        # Extract embedded version from HTML comment (e.g., "<!-- version: 42 -->")
        version_match = re.search(r"<!--\s*version:\s*(\d+)\s*-->", content)
        if version_match:
            embedded_version = int(version_match.group(1))

        _KNOWN_SECTIONS = {"role", "key knowledge", "active context", "archived context"}

        sections: dict[str, str] = {}
        current_section = ""
        current_lines: list[str] = []

        for line in content.split("\n"):
            # Skip version comment lines during section parsing
            if line.strip().startswith("<!--") and "version:" in line:
                continue
            if line.startswith("# "):
                candidate = line[2:].strip().lower()
                if candidate in _KNOWN_SECTIONS:
                    if current_section:
                        sections[current_section] = "\n".join(current_lines).strip()
                    current_section = candidate
                    current_lines = []
                else:
                    # Not a known section header — treat as content
                    current_lines.append(line)
            else:
                current_lines.append(line)

        if current_section:
            sections[current_section] = "\n".join(current_lines).strip()

        role = sections.get("role", "")
        key_knowledge = sections.get("key knowledge", "")
        active_context = sections.get("active context", "")
        archived_context = sections.get("archived context", "")

        return cls(
            role=role,
            key_knowledge=key_knowledge,
            active_context=active_context,
            archived_context=archived_context,
            _version=embedded_version,
        )


@dataclass
class SkillProfile:
    """Skill profile for automatic task assignment scoring."""

    tag: str = ""
    success_rate: float = 50.0  # 0-100
    total_tasks: int = 0
    last_active: float = 0.0

    def to_dict(self) -> dict:
        return {
            "tag": self.tag,
            "success_rate": self.success_rate,
            "total_tasks": self.total_tasks,
            "last_active": self.last_active,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SkillProfile:
        raw_rate = data.get("success_rate", 50.0)
        return cls(
            tag=data.get("tag", ""),
            success_rate=max(0.0, min(100.0, float(raw_rate))),
            total_tasks=data.get("total_tasks", 0),
            last_active=data.get("last_active", 0.0),
        )


# ---------------------------------------------------------------------------
# Council Protocol Models
# ---------------------------------------------------------------------------


@dataclass
class CouncilResponse:
    """Independent answer from one Slock agent."""

    label: str = ""
    agent_id: str = ""
    agent_name: str = ""
    content: str = ""
    error: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class CouncilReview:
    """Anonymous peer review/ranking from one Slock agent."""

    reviewer_agent_id: str = ""
    reviewer_name: str = ""
    content: str = ""
    parsed_ranking: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass
class CouncilAggregate:
    """Aggregated ranking for one response label."""

    label: str = ""
    agent_id: str = ""
    agent_name: str = ""
    average_rank: float = 0.0
    rankings_count: int = 0
    quality_score: float = 0.0


@dataclass
class CouncilRun:
    """A full Slock council run: independent opinions, peer reviews, synthesis."""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    channel_id: str = ""
    question: str = ""
    participant_ids: list[str] = field(default_factory=list)
    chairman_agent_id: str = ""
    status: CouncilStatus = CouncilStatus.STARTING
    responses: list[CouncilResponse] = field(default_factory=list)
    reviews: list[CouncilReview] = field(default_factory=list)
    aggregate_rankings: list[CouncilAggregate] = field(default_factory=list)
    label_to_agent: dict[str, str] = field(default_factory=dict)
    final_response: str = ""
    error: str = ""
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


# ---------------------------------------------------------------------------
# Discussion Protocol Models
# ---------------------------------------------------------------------------


@dataclass
class DiscussionConfig:
    """Configuration for an inter-agent discussion session."""

    max_rounds: int = 3
    token_budget: int = 50000
    trigger_rules: list[str] = field(default_factory=lambda: ["coder->reviewer"])
    convergence_threshold: float = 0.85  # Similarity threshold to declare convergence
    discussion_timeout: int = 300  # Total discussion timeout in seconds
    max_tokens_per_round: int = 8000  # Per-round output token hard cap

    def to_dict(self) -> dict:
        return {
            "max_rounds": self.max_rounds,
            "token_budget": self.token_budget,
            "trigger_rules": self.trigger_rules,
            "convergence_threshold": self.convergence_threshold,
            "discussion_timeout": self.discussion_timeout,
            "max_tokens_per_round": self.max_tokens_per_round,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DiscussionConfig:
        return cls(
            max_rounds=data.get("max_rounds", 3),
            token_budget=data.get("token_budget", 50000),
            trigger_rules=data.get("trigger_rules", ["coder->reviewer"]),
            convergence_threshold=data.get("convergence_threshold", 0.85),
            discussion_timeout=data.get("discussion_timeout", 300),
            max_tokens_per_round=data.get("max_tokens_per_round", 8000),
        )


@dataclass
class DiscussionMessage:
    """A single message within an inter-agent discussion thread."""

    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_agent_id: str = ""
    receiver_agent_id: str = ""
    content: str = ""
    round_num: int = 0
    timestamp: float = field(default_factory=time.time)
    token_count: int = 0  # Estimated tokens consumed by this message

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "sender_agent_id": self.sender_agent_id,
            "sender_display_name": self.sender_agent_id,
            "receiver_agent_id": self.receiver_agent_id,
            "content": self.content,
            "round_num": self.round_num,
            "timestamp": self.timestamp,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DiscussionMessage:
        return cls(
            message_id=data.get("message_id", str(uuid.uuid4())),
            sender_agent_id=data.get("sender_agent_id", ""),
            receiver_agent_id=data.get("receiver_agent_id", ""),
            content=data.get("content", ""),
            round_num=data.get("round_num", 0),
            timestamp=data.get("timestamp", time.time()),
            token_count=data.get("token_count", 0),
        )


@dataclass
class DiscussionThread:
    """A discussion thread between multiple agents."""

    thread_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    channel_id: str = ""
    participants: list[str] = field(default_factory=list)  # agent_id list
    messages: list[DiscussionMessage] = field(default_factory=list)
    _status_value: DiscussionStatus = field(default=DiscussionStatus.ACTIVE, init=False, repr=False)
    _status_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _data_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    config: DiscussionConfig = field(default_factory=DiscussionConfig)
    trigger_reason: str = ""  # Why this discussion was triggered
    topic: str = ""  # Derived from trigger_reason[:100]; used for conclusion persistence
    conclusion: str = ""  # Final conclusion after convergence
    total_tokens_used: int = 0
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    triggerer_open_id: Optional[str] = None  # Feishu open_id of user who triggered this discussion
    cancellation_event: Optional[threading.Event] = None  # Runtime-only; not serialized
    pending_hints: list[str] = field(default_factory=list)  # User hints to inject into next round

    def __init__(
        self,
        thread_id: str | None = None,
        channel_id: str = "",
        participants: list[str] | None = None,
        messages: list[DiscussionMessage] | None = None,
        status: DiscussionStatus = DiscussionStatus.ACTIVE,
        config: DiscussionConfig | None = None,
        trigger_reason: str = "",
        topic: str = "",
        conclusion: str = "",
        total_tokens_used: int = 0,
        created_at: float | None = None,
        completed_at: Optional[float] = None,
        triggerer_open_id: Optional[str] = None,
        cancellation_event: Optional[threading.Event] = None,
        pending_hints: list[str] | None = None,
    ) -> None:
        self.thread_id = thread_id if thread_id is not None else str(uuid.uuid4())
        self.channel_id = channel_id
        self.participants = participants if participants is not None else []
        self.messages = messages if messages is not None else []
        self._status_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._data_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self.status = status
        self.config = config if config is not None else DiscussionConfig()
        self.trigger_reason = trigger_reason
        self.topic = topic
        self.conclusion = conclusion
        self.total_tokens_used = total_tokens_used
        self.created_at = created_at if created_at is not None else time.time()
        self.completed_at = completed_at
        self.triggerer_open_id = triggerer_open_id
        self.cancellation_event = cancellation_event
        self.pending_hints = pending_hints if pending_hints is not None else []
        self.__post_init__()

    def __post_init__(self) -> None:
        if not self.channel_id:
            raise ValueError("DiscussionThread.channel_id must be provided (non-empty)")

    @property
    def status(self) -> DiscussionStatus:
        with self._status_lock:
            return self._status_value

    @status.setter
    def status(self, value: DiscussionStatus) -> None:
        with self._status_lock:
            self._status_value = value

    @property
    def current_round(self) -> int:
        if not self.messages:
            return 0
        return max(m.round_num for m in self.messages)

    @property
    def is_active(self) -> bool:
        return self.status == DiscussionStatus.ACTIVE

    def add_message(self, msg: DiscussionMessage) -> None:
        """Append a message to the thread in a thread-safe manner."""
        with self._data_lock:
            self.messages.append(msg)

    def add_participant(self, agent_id: str) -> None:
        """Add a participant to the thread if not already present (thread-safe)."""
        with self._data_lock:
            if agent_id not in self.participants:
                self.participants.append(agent_id)

    def add_hint(self, hint: str) -> None:
        """Add a user hint to be injected into the next discussion round (thread-safe).

        Args:
            hint: The user-provided guidance to inject into the discussion.
        """
        with self._data_lock:
            self.pending_hints.append(hint)

    def consume_hints(self) -> list[str]:
        """Consume and clear all pending hints (thread-safe).

        Returns:
            List of hints to inject; list is cleared after calling.
        """
        with self._data_lock:
            hints = list(self.pending_hints)
            self.pending_hints = []
            return hints

    def get_messages(self) -> list[DiscussionMessage]:
        """Return a shallow copy of messages under lock."""
        with self._data_lock:
            return list(self.messages)

    def get_participants(self) -> list[str]:
        """Return a copy of participants under lock."""
        with self._data_lock:
            return list(self.participants)

    def to_dict(self) -> dict:
        return {
            "thread_id": self.thread_id,
            "channel_id": self.channel_id,
            "participants": self.participants,
            "messages": [m.to_dict() for m in self.messages],
            "status": self.status.value,
            "config": self.config.to_dict(),
            "trigger_reason": self.trigger_reason,
            "topic": self.topic,
            "conclusion": self.conclusion,
            "total_tokens_used": self.total_tokens_used,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "triggerer_open_id": self.triggerer_open_id,
            "pending_hints": self.pending_hints,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DiscussionThread:
        messages = [DiscussionMessage.from_dict(m) for m in data.get("messages", [])]
        config = DiscussionConfig.from_dict(data.get("config", {}))
        status_val = data.get("status", "active")
        try:
            status = DiscussionStatus(status_val)
        except ValueError:
            status = DiscussionStatus.ACTIVE
        return cls(
            thread_id=data.get("thread_id", str(uuid.uuid4())),
            channel_id=data.get("channel_id", ""),
            participants=data.get("participants", []),
            messages=messages,
            status=status,
            config=config,
            trigger_reason=data.get("trigger_reason", ""),
            topic=data.get("topic", ""),
            conclusion=data.get("conclusion", ""),
            total_tokens_used=data.get("total_tokens_used", 0),
            created_at=data.get("created_at", time.time()),
            completed_at=data.get("completed_at"),
            triggerer_open_id=data.get("triggerer_open_id"),
            pending_hints=data.get("pending_hints", []),
        )


# ---------------------------------------------------------------------------
# Dissolve rollback snapshot
# ---------------------------------------------------------------------------


@dataclass
class TeamSnapshot:
    """Snapshot of a team's state for dissolve rollback (30s TTL).

    Captures enough information to restore a dissolved team entity
    including its channel, agents, and task bindings.
    """

    channel_id: str
    team_name: str
    owner_id: str
    channel: "SlockChannel | None" = None
    agent_ids: list[str] = field(default_factory=list)
    agent_bindings: dict[str, str] = field(default_factory=dict)  # agent_id -> role
    task_ids: list[str] = field(default_factory=list)
    task_board_data: list[dict] = field(default_factory=list)  # serialized SlockTask dicts
    created_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Collaboration Plan Models
# ---------------------------------------------------------------------------


class CollaborationPlanStatus(Enum):
    """Collaboration plan lifecycle."""

    PLANNING = "planning"  # Planner is decomposing the task
    PENDING_APPROVAL = "pending_approval"  # Waiting for user confirmation (30s timeout)
    EXECUTING = "executing"  # Auto-started or user-approved
    PAUSED = "paused"  # User paused; no new steps will start
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class PlanStep:
    """A single step in a collaboration plan."""

    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: str = ""  # Target role (coder/reviewer/tester/etc.)
    agent_id: str = ""  # Assigned agent_id (resolved at execution time)
    description: str = ""  # What this step should accomplish
    order: int = 0  # Execution order (0-based)
    status: PlanStepStatus = PlanStepStatus.TODO
    task_id: str = ""  # Created SlockTask ID once started
    depends_on: list[str] = field(default_factory=list)  # step_ids that must complete first
    predecessor_context: str = ""  # Output summary from predecessor step(s) for continuity

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "role": self.role,
            "agent_id": self.agent_id,
            "description": self.description,
            "order": self.order,
            "status": self.status.value,
            "task_id": self.task_id,
            "depends_on": self.depends_on,
            "predecessor_context": self.predecessor_context,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlanStep":
        return cls(
            step_id=data.get("step_id", str(uuid.uuid4())),
            role=data.get("role", ""),
            agent_id=data.get("agent_id", ""),
            description=data.get("description", ""),
            order=data.get("order", 0),
            status=PlanStepStatus(data.get("status", "todo")),
            task_id=data.get("task_id", ""),
            depends_on=data.get("depends_on", []),
            predecessor_context=data.get("predecessor_context", ""),
        )


@dataclass
class CollaborationPlan:
    """A multi-role collaboration plan for a task."""

    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""  # The parent task this plan is for
    task_content: str = ""  # Human-readable task description (from SlockTask.content)
    steps: list[PlanStep] = field(default_factory=list)
    status: CollaborationPlanStatus = CollaborationPlanStatus.PLANNING
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None  # Unix timestamp when plan reached terminal state
    auto_start_at: Optional[float] = None  # Unix timestamp when auto-execution begins
    chain_template: str = ""  # Which chain template was used
    planner_agent_id: str = ""  # Agent that created this plan

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "task_id": self.task_id,
            "task_content": self.task_content,
            "steps": [s.to_dict() for s in self.steps],
            "status": self.status.value,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "auto_start_at": self.auto_start_at,
            "chain_template": self.chain_template,
            "planner_agent_id": self.planner_agent_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CollaborationPlan":
        return cls(
            plan_id=data.get("plan_id", str(uuid.uuid4())),
            task_id=data.get("task_id", ""),
            task_content=data.get("task_content", ""),
            steps=[PlanStep.from_dict(s) for s in data.get("steps", [])],
            status=CollaborationPlanStatus(data.get("status", "planning")),
            created_at=data.get("created_at", time.time()),
            completed_at=data.get("completed_at"),
            auto_start_at=data.get("auto_start_at"),
            chain_template=data.get("chain_template", ""),
            planner_agent_id=data.get("planner_agent_id", ""),
        )

    @property
    def progress_pct(self) -> int:
        """Calculate overall progress based on step completion."""
        if not self.steps:
            return 0
        done = sum(1 for s in self.steps if s.status in (PlanStepStatus.DONE, PlanStepStatus.SKIPPED, PlanStepStatus.TIMED_OUT))
        return int(done / len(self.steps) * 100)

    @property
    def current_step(self) -> Optional[PlanStep]:
        """Get the first non-done step in execution order."""
        for step in sorted(self.steps, key=lambda s: s.order):
            if step.status not in (PlanStepStatus.DONE, PlanStepStatus.SKIPPED, PlanStepStatus.TIMED_OUT):
                return step
        return None
