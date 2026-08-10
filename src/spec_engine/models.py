"""Spec Engine 数据模型与枚举定义。

Spec Engine 采用 spec-kit 方法论，每个循环经历
spec → plan → task → build → review 五个阶段，
review 产生的建议驱动下一轮循环，直到所有验收标准满足且无新建议。
"""

import hashlib
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from ..engine_base import CriteriaTracker, PerspectiveReview, ReviewPerspective, ReviewResult

# Re-export shared types for convenience
__all__ = [
    # Artifacts
    "SpecArtifact",
    "PlanArtifact",
    "SpecPhase",
    "SpecProjectStatus",
    "SpecTaskStatus",
    "SpecTask",
    "SpecCycle",
    "SpecProject",
    "SpecWorkItemStatus",
    "SpecWorkItem",
    "SpecCycleMetrics",
    # Re-exported from engine_base
    "CriteriaTracker",
    "ReviewResult",
    "PerspectiveReview",
    "ReviewPerspective",
    "ReviewAgentBinding",
    "ReviewRoleSpec",
    "RoleSuggestion",
    "RoleReviewOutcome",
    "AggregatedSuggestion",
    "AggregatedReview",
    "AdaptiveReviewResult",
    "ReviewCircuitState",
    "ReviewContext",
]


@dataclass(frozen=True)
class ReviewAgentBinding:
    """Concrete reviewer backend and model binding."""

    provider: str
    tool_name: str
    display_name: str
    agent_type: str
    model_name: str | None = None
    model_display_name: str | None = None
    selection_key: str = ""
    use_default_model: bool = True

    @property
    def display_label(self) -> str:
        return f"{self.display_name or self.tool_name} / {self.model_display_name or self.model_name or '默认模型'}"

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "tool_name": self.tool_name,
            "display_name": self.display_name,
            "agent_type": self.agent_type,
            "model_name": self.model_name,
            "model_display_name": self.model_display_name,
            "selection_key": self.selection_key,
            "use_default_model": self.use_default_model,
        }

    @classmethod
    def from_dict(cls, data: object) -> "ReviewAgentBinding | None":
        if not isinstance(data, dict):
            return None
        agent = str(data.get("agent_type") or "").strip().lower()
        tool = str(data.get("tool_name") or "").strip().lower()
        if not agent and not tool:
            return None
        return cls(
            provider=str(data.get("provider") or "").strip().lower(),
            tool_name=tool,
            display_name=str(data.get("display_name") or tool or agent).strip(),
            agent_type=agent or tool or "coco",
            model_name=str(data.get("model_name") or "").strip() or None,
            model_display_name=str(data.get("model_display_name") or "").strip() or None,
            selection_key=str(data.get("selection_key") or "").strip(),
            use_default_model=bool(data.get("use_default_model", True)),
        )


@dataclass(frozen=True)
class ReviewRoleSpec:
    role_id: str
    display_name: str
    category: str
    mission: str
    review_focus: list[str]
    must_check: list[str]
    blocking: bool = True
    max_suggestions: int = 5
    base_perspective: ReviewPerspective | None = None

    def to_dict(self) -> dict:
        return {
            "role_id": self.role_id,
            "display_name": self.display_name,
            "category": self.category,
            "mission": self.mission,
            "review_focus": self.review_focus,
            "must_check": self.must_check,
            "blocking": self.blocking,
            "max_suggestions": self.max_suggestions,
            "base_perspective": self.base_perspective.value if self.base_perspective else None,
        }


@dataclass
class RoleSuggestion:
    severity: str
    confidence: str
    evidence: str
    recommendation: str
    target: str = ""
    blocking: bool = False

    def normalized_key(self) -> str:
        return " ".join(self.recommendation.strip().lower().split())


@dataclass
class RoleReviewOutcome:
    role_id: str
    role_display_name: str
    role_category: str
    passed: bool
    summary: str = ""
    suggestions: list[RoleSuggestion] = field(default_factory=list)
    raw_preview: str = ""
    error: str = ""
    blocking: bool = True
    skipped: bool = False
    base_perspective_value: str = ""
    goal_verdict: str = ""
    goal_confidence: str = ""
    goal_evidence: str = ""


@dataclass
class AggregatedSuggestion:
    suggestion_id: str
    severity: str
    confidence: str
    role_ids: list[str]
    evidence: list[str]
    recommendation: str
    target: str = ""
    blocking: bool = False

    def to_repair_text(self, role_names: dict[str, str]) -> str:
        names = ", ".join(role_names.get(role_id, role_id) for role_id in self.role_ids)
        details = ([f"evidence: {' | '.join(self.evidence)}"] if self.evidence else [])
        if self.target:
            details.append(f"target: {self.target}")
        return f"[{names}] {self.recommendation}" + (f" ({'; '.join(details)})" if details else "")


@dataclass
class AggregatedReview:
    blocking_suggestions: list[AggregatedSuggestion]
    observations: list[AggregatedSuggestion]
    role_names: dict[str, str]

    def blocking_hash(self) -> str:
        rows = [(item.recommendation, item.role_ids, item.target, item.severity) for item in self.blocking_suggestions]
        return hashlib.sha256(repr(rows).encode()).hexdigest()[:16] if rows else ""


@dataclass
class AdaptiveReviewResult(ReviewResult):
    role_outcomes: list[RoleReviewOutcome] = field(default_factory=list)
    aggregated: AggregatedReview | None = None
    role_plan_hash: str = ""
    blocking_suggestion_hash: str = ""
    blocking_review_passed: bool = False
    skipped_roles_count: int = 0
    completion_gate_met: bool = False
    completion_gate_confidence: str = ""
    completion_gate_evidence: str = ""
    completion_gate_enabled: bool = False
    completion_gate_valid: bool = True
    completion_gate_error: str = ""

    @property
    def all_passed(self) -> bool:
        return bool(self.reviews) and all(review.passed for review in self.reviews) and (
            not self.completion_gate_enabled or self.completion_gate_valid
        )


@dataclass
class ReviewCircuitState:
    last_review_failure_diag: Optional[dict] = None
    review_failure_consecutive: int = 0
    review_circuit_open_until_cycle: int = 0
    backoff_level: int = 0
    consecutive_timeouts: int = 0
    consecutive_skips: int = 0
    last_review_elapsed_ms: int = 0
    last_failure_timestamp: float = 0.0
    recent_outcomes: list[str] = field(default_factory=list)

    def _remember(self, outcome: str) -> None:
        self.recent_outcomes.append(outcome)
        del self.recent_outcomes[:-20]

    def reset_on_success(self) -> None:
        self.review_failure_consecutive = 0
        self.review_circuit_open_until_cycle = 0
        self.backoff_level = 0
        self.consecutive_timeouts = 0
        self.consecutive_skips = 0
        self.last_review_failure_diag = None
        self._remember("success")

    def on_failure(self, all_timeout: bool) -> None:
        self.review_failure_consecutive += 1
        self.consecutive_timeouts += int(all_timeout)
        self.last_failure_timestamp = time.monotonic()
        self._remember("failure")

    def to_dict(self) -> dict:
        return {
            "review_failure_consecutive": self.review_failure_consecutive,
            "review_circuit_open_until_cycle": self.review_circuit_open_until_cycle,
            "backoff_level": self.backoff_level,
            "consecutive_timeouts": self.consecutive_timeouts,
            "consecutive_skips": self.consecutive_skips,
            "last_review_elapsed_ms": self.last_review_elapsed_ms,
            "last_failure_timestamp": self.last_failure_timestamp,
            "recent_outcomes": self.recent_outcomes[-20:],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewCircuitState":
        return cls(
            review_failure_consecutive=int(data.get("review_failure_consecutive") or 0),
            review_circuit_open_until_cycle=int(data.get("review_circuit_open_until_cycle") or 0),
            backoff_level=int(data.get("backoff_level") or 0),
            consecutive_timeouts=int(data.get("consecutive_timeouts") or 0),
            consecutive_skips=int(data.get("consecutive_skips") or 0),
            last_review_elapsed_ms=int(data.get("last_review_elapsed_ms") or 0),
            last_failure_timestamp=float(data.get("last_failure_timestamp") or 0),
            recent_outcomes=[str(value) for value in data.get("recent_outcomes", [])][-20:],
        )


@dataclass
class ReviewContext:
    cycle: int
    session: Any
    settings: Any
    project: Any
    send_prompt_with_retry_fn: Callable
    build_review_exception_diagnostics_fn: Callable[..., dict]
    circuit: ReviewCircuitState
    on_review_done: Optional[Callable] = None
    artifacts: Any = None
    cancel_event: Optional[threading.Event] = None
    on_retry_status: Optional[Callable] = None
    agent_type: str = "coco"
    model_name: Optional[str] = None
    prompt_runner_factory: Optional[Callable] = None
    role_plan_override: Optional[list[ReviewRoleSpec]] = None
    review_agents: Optional[list[ReviewAgentBinding]] = None
    review_agent_rng: Any = None


# ---------------------------------------------------------------------------
# Artifacts (spec-kit inspired structured outputs)
# ---------------------------------------------------------------------------


@dataclass
class SpecArtifact:
    """结构化规格产物（JSON，可机器解析）。"""

    goals: list[str] = field(default_factory=list)
    functional_spec: list[str] = field(default_factory=list)
    non_functional_requirements: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    out_of_scope: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    clarification_questions: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "goals": self.goals,
            "functional_spec": self.functional_spec,
            "non_functional_requirements": self.non_functional_requirements,
            "acceptance_criteria": self.acceptance_criteria,
            "out_of_scope": self.out_of_scope,
            "risks": self.risks,
            "clarification_questions": self.clarification_questions,
            "decisions": self.decisions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpecArtifact":
        if not isinstance(data, dict):
            return cls()
        return cls(
            goals=[str(x) for x in data.get("goals", []) if x],
            functional_spec=[str(x) for x in data.get("functional_spec", []) if x],
            non_functional_requirements=[str(x) for x in data.get("non_functional_requirements", []) if x],
            acceptance_criteria=[str(x) for x in data.get("acceptance_criteria", []) if x],
            out_of_scope=[str(x) for x in data.get("out_of_scope", []) if x],
            risks=[str(x) for x in data.get("risks", []) if x],
            clarification_questions=[str(x) for x in data.get("clarification_questions", []) if x],
            decisions=[str(x) for x in data.get("decisions", []) if x],
        )


@dataclass
class PlanArtifact:
    """结构化规划产物（JSON，可机器解析）。"""

    architecture: str = ""
    tech_stack: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    file_changes: list[str] = field(default_factory=list)
    test_plan: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "architecture": self.architecture,
            "tech_stack": self.tech_stack,
            "steps": self.steps,
            "file_changes": self.file_changes,
            "test_plan": self.test_plan,
            "risks": self.risks,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlanArtifact":
        if not isinstance(data, dict):
            return cls()
        return cls(
            architecture=str(data.get("architecture", "") or ""),
            tech_stack=[str(x) for x in data.get("tech_stack", []) if x],
            steps=[str(x) for x in data.get("steps", []) if x],
            file_changes=[str(x) for x in data.get("file_changes", []) if x],
            test_plan=[str(x) for x in data.get("test_plan", []) if x],
            risks=[str(x) for x in data.get("risks", []) if x],
        )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SpecPhase(Enum):
    """Spec 循环内的阶段。"""

    SPEC = "spec"
    PLAN = "plan"
    TASK = "task"
    BUILD = "build"
    REVIEW = "review"

    @property
    def display_name(self) -> str:
        return {
            SpecPhase.SPEC: "规格定义",
            SpecPhase.PLAN: "方案规划",
            SpecPhase.TASK: "任务分解",
            SpecPhase.BUILD: "执行构建",
            SpecPhase.REVIEW: "多角色审查",
        }[self]

    @property
    def emoji(self) -> str:
        return {
            SpecPhase.SPEC: "📋",
            SpecPhase.PLAN: "🏗️",
            SpecPhase.TASK: "📝",
            SpecPhase.BUILD: "🔨",
            SpecPhase.REVIEW: "🔍",
        }[self]


class SpecProjectStatus(Enum):
    """Spec 项目状态机。"""

    IDLE = "idle"
    ANALYZING = "analyzing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SpecTaskStatus(Enum):
    """任务执行状态。"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class SpecWorkItemStatus(Enum):
    """长程任务中“待优化单元”（由问题发现机制产生）的状态。"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SpecTask:
    """单个可执行任务。"""

    task_id: int
    description: str
    dependencies: list[int] = field(default_factory=list)
    status: SpecTaskStatus = SpecTaskStatus.PENDING
    output: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "output": self.output,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpecTask":
        return cls(
            task_id=data["task_id"],
            description=data["description"],
            dependencies=data.get("dependencies", []),
            status=SpecTaskStatus(data.get("status", "pending")),
            output=data.get("output", ""),
        )


@dataclass
class SpecCycle:
    """一个完整的 spec→plan→task→build→review 循环。"""

    cycle_number: int
    phase: SpecPhase = SpecPhase.SPEC
    spec_content: str = ""
    spec_path: Optional[str] = None
    spec_artifact: Optional["SpecArtifact"] = None
    spec_artifact_errors: list[str] = field(default_factory=list)
    plan_content: str = ""
    plan_path: Optional[str] = None
    plan_artifact: Optional["PlanArtifact"] = None
    plan_artifact_errors: list[str] = field(default_factory=list)
    tasks: list[SpecTask] = field(default_factory=list)
    tasks_total: int = 0
    tasks_path: Optional[str] = None
    build_output: str = ""
    build_path: Optional[str] = None
    review_result: Optional[ReviewResult] = None
    review_path: Optional[str] = None
    # 审查异常可观测性（best-effort，不影响既有成功路径）
    review_decision: str = ""
    review_diagnostics: Optional[dict] = None
    discovery_path: Optional[str] = None
    metrics_path: Optional[str] = None
    status: str = "running"  # running/completed/failed
    error_message: Optional[str] = None  # 异常失败时的错误描述
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    duration: Optional[float] = None
    # 操作统计（由 PhaseTracker 累积）
    tool_call_count: int = 0
    modified_files: list[str] = field(default_factory=list)
    phase_tool_stats: dict[str, int] = field(default_factory=dict)

    def complete(self):
        self.status = "completed"
        self.completed_at = time.time()
        self.duration = self.completed_at - self.started_at

    def fail(self):
        self.status = "failed"
        self.completed_at = time.time()
        self.duration = self.completed_at - self.started_at

    def to_dict(self) -> dict:
        return {
            "cycle_number": self.cycle_number,
            "phase": self.phase.value,
            "spec_content": self.spec_content,
            "spec_path": self.spec_path,
            "spec_artifact": self.spec_artifact.to_dict() if self.spec_artifact else None,
            "spec_artifact_errors": self.spec_artifact_errors,
            "plan_content": self.plan_content,
            "plan_path": self.plan_path,
            "plan_artifact": self.plan_artifact.to_dict() if self.plan_artifact else None,
            "plan_artifact_errors": self.plan_artifact_errors,
            "tasks": [t.to_dict() for t in self.tasks],
            "tasks_total": self.tasks_total,
            "tasks_path": self.tasks_path,
            "build_output": self.build_output,
            "build_path": self.build_path,
            "review_result": self.review_result.to_dict() if self.review_result else None,
            "review_path": self.review_path,
            "review_decision": self.review_decision,
            "review_diagnostics": self.review_diagnostics,
            "discovery_path": self.discovery_path,
            "metrics_path": self.metrics_path,
            "status": self.status,
            "error_message": self.error_message,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration": self.duration,
            "tool_call_count": self.tool_call_count,
            "modified_files": self.modified_files,
            "phase_tool_stats": self.phase_tool_stats,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpecCycle":
        review_data = data.get("review_result")
        spec_artifact = data.get("spec_artifact")
        plan_artifact = data.get("plan_artifact")
        return cls(
            cycle_number=data["cycle_number"],
            phase=SpecPhase(data.get("phase", "spec")),
            spec_content=data.get("spec_content", ""),
            spec_path=data.get("spec_path"),
            spec_artifact=SpecArtifact.from_dict(spec_artifact) if spec_artifact else None,
            spec_artifact_errors=[str(x) for x in data.get("spec_artifact_errors", []) if x],
            plan_content=data.get("plan_content", ""),
            plan_path=data.get("plan_path"),
            plan_artifact=PlanArtifact.from_dict(plan_artifact) if plan_artifact else None,
            plan_artifact_errors=[str(x) for x in data.get("plan_artifact_errors", []) if x],
            tasks=[SpecTask.from_dict(t) for t in data.get("tasks", [])],
            tasks_total=int(data.get("tasks_total") or 0),
            tasks_path=data.get("tasks_path"),
            build_output=data.get("build_output", ""),
            build_path=data.get("build_path"),
            review_result=ReviewResult.from_dict(review_data) if review_data else None,
            review_path=data.get("review_path"),
            review_decision=str(data.get("review_decision") or ""),
            review_diagnostics=(
                data.get("review_diagnostics") if isinstance(data.get("review_diagnostics"), dict) else None
            ),
            discovery_path=data.get("discovery_path"),
            metrics_path=data.get("metrics_path"),
            status=data.get("status", "running"),
            error_message=data.get("error_message"),
            started_at=data.get("started_at", time.time()),
            completed_at=data.get("completed_at"),
            duration=data.get("duration"),
            tool_call_count=int(data.get("tool_call_count") or 0),
            modified_files=list(data.get("modified_files") or []),
            phase_tool_stats=dict(data.get("phase_tool_stats") or {}),
        )


@dataclass
class SpecWorkItem:
    """一个“优化性问题”对应的 spec-kit 任务单元（会落盘为 spec 文件）。"""

    item_id: str
    question: str
    created_in_cycle: int
    spec_path: str
    spec_deleted: bool = False
    status: SpecWorkItemStatus = SpecWorkItemStatus.PENDING
    used_in_cycle: Optional[int] = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "question": self.question,
            "created_in_cycle": self.created_in_cycle,
            "spec_path": self.spec_path,
            "spec_deleted": self.spec_deleted,
            "status": self.status.value,
            "used_in_cycle": self.used_in_cycle,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpecWorkItem":
        return cls(
            item_id=str(data.get("item_id", "")),
            question=str(data.get("question", "")),
            created_in_cycle=int(data.get("created_in_cycle") or 0),
            spec_path=str(data.get("spec_path", "")),
            spec_deleted=bool(data.get("spec_deleted", False)),
            status=SpecWorkItemStatus(data.get("status", "pending")),
            used_in_cycle=data.get("used_in_cycle"),
            created_at=float(data.get("created_at") or time.time()),
        )


@dataclass
class SpecCycleMetrics:
    """每轮循环的可查询指标快照。"""

    cycle_number: int
    satisfied_count: int
    total_criteria: int
    new_satisfied: int
    review_suggestions: int
    backlog_pending: int
    goal_attainment: float
    improvement_space: float
    termination_hint: str = ""
    # 审查异常观测：可用于监控与后续问题发现（best-effort）
    review_failed: bool = False
    review_decision: str = ""
    review_exception_type: str = ""
    review_error_text: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "cycle_number": self.cycle_number,
            "satisfied_count": self.satisfied_count,
            "total_criteria": self.total_criteria,
            "new_satisfied": self.new_satisfied,
            "review_suggestions": self.review_suggestions,
            "backlog_pending": self.backlog_pending,
            "goal_attainment": self.goal_attainment,
            "improvement_space": self.improvement_space,
            "termination_hint": self.termination_hint,
            "review_failed": self.review_failed,
            "review_decision": self.review_decision,
            "review_exception_type": self.review_exception_type,
            "review_error_text": self.review_error_text,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpecCycleMetrics":
        return cls(
            cycle_number=int(data.get("cycle_number") or 0),
            satisfied_count=int(data.get("satisfied_count") or 0),
            total_criteria=int(data.get("total_criteria") or 0),
            new_satisfied=int(data.get("new_satisfied") or 0),
            review_suggestions=int(data.get("review_suggestions") or 0),
            backlog_pending=int(data.get("backlog_pending") or 0),
            goal_attainment=float(data.get("goal_attainment") or 0.0),
            improvement_space=float(data.get("improvement_space") or 0.0),
            termination_hint=str(data.get("termination_hint") or ""),
            review_failed=bool(data.get("review_failed", False)),
            review_decision=str(data.get("review_decision") or ""),
            review_exception_type=str(data.get("review_exception_type") or ""),
            review_error_text=str(data.get("review_error_text") or ""),
            created_at=float(data.get("created_at") or time.time()),
        )


@dataclass
class SpecProject:
    """Spec 项目顶层容器。"""

    project_id: str
    name: str
    root_path: str
    requirement: str = ""
    cycles: list[SpecCycle] = field(default_factory=list)
    # Total cycle count (state file may only keep tail of cycles)
    cycle_count_total: int = 0
    work_items: list[SpecWorkItem] = field(default_factory=list)
    work_items_total: int = 0
    metrics_history: list[SpecCycleMetrics] = field(default_factory=list)
    # Persistence metadata (may come from compact state)
    artifacts_root: Optional[str] = None
    history_log_path: Optional[str] = None
    compact_meta: Optional[dict] = None
    acceptance_criteria: list[str] = field(default_factory=list)
    criteria_tracker: CriteriaTracker = field(default_factory=CriteriaTracker)
    status: SpecProjectStatus = SpecProjectStatus.IDLE
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    task_id: Optional[str] = None  # Human-readable task ID
    verify_command: str = ""
    review_pass_streak: int = 0
    last_review_role_plan_hash: str = ""
    last_review_blocking_suggestion_hash: str = ""
    auto_recovery_attempts: int = 0

    @classmethod
    def create(cls, name: str = "", root_path: str = "") -> "SpecProject":
        if not name:
            name = os.path.basename(root_path) or "spec_project"
        return cls(
            project_id=str(uuid.uuid4())[:8],
            name=name,
            root_path=root_path,
        )

    def start(self):
        self.status = SpecProjectStatus.RUNNING
        self.started_at = time.time()

    def complete(self):
        self.status = SpecProjectStatus.COMPLETED
        self.completed_at = time.time()

    def fail(self, reason: str):
        self.status = SpecProjectStatus.FAILED
        self.completed_at = time.time()
        self.error = reason

    def cancel(self, reason: str = "用户停止"):
        self.status = SpecProjectStatus.CANCELLED
        self.completed_at = time.time()
        self.error = reason

    @property
    def current_cycle(self) -> Optional[SpecCycle]:
        return self.cycles[-1] if self.cycles else None

    @property
    def current_cycle_number(self) -> int:
        # Prefer persisted total if available (resume from compact state)
        return self.cycle_count_total if self.cycle_count_total > len(self.cycles) else len(self.cycles)

    @property
    def satisfied_count(self) -> int:
        return self.criteria_tracker.satisfied_count

    @property
    def total_criteria(self) -> int:
        return self.criteria_tracker.total_count

    @property
    def is_all_satisfied(self) -> bool:
        return self.criteria_tracker.is_all_satisfied

    def duration(self) -> Optional[float]:
        if self.started_at:
            end = self.completed_at or time.time()
            return end - self.started_at
        return None

    def to_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "name": self.name,
            "root_path": self.root_path,
            "requirement": self.requirement,
            "cycles": [c.to_dict() for c in self.cycles],
            "cycle_count_total": self.cycle_count_total,
            "work_items": [w.to_dict() for w in self.work_items],
            "work_items_total": self.work_items_total,
            "metrics_history": [m.to_dict() for m in self.metrics_history],
            "artifacts_root": self.artifacts_root,
            "history_log_path": self.history_log_path,
            "_compact": self.compact_meta,
            "acceptance_criteria": self.acceptance_criteria,
            "criteria_tracker": self.criteria_tracker.to_dict(),
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "task_id": self.task_id,
            "verify_command": self.verify_command,
            "review_pass_streak": self.review_pass_streak,
            "last_review_role_plan_hash": self.last_review_role_plan_hash,
            "last_review_blocking_suggestion_hash": self.last_review_blocking_suggestion_hash,
            "auto_recovery_attempts": self.auto_recovery_attempts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpecProject":
        raw_status = {
            "aborted": "failed",
            "paused": "running",
            "clarifying": "running",
        }.get(str(data.get("status", "idle") or "idle"), str(data.get("status", "idle") or "idle"))
        project = cls(
            project_id=data["project_id"],
            name=data["name"],
            root_path=data["root_path"],
            requirement=data.get("requirement", ""),
            acceptance_criteria=data.get("acceptance_criteria", []),
            status=SpecProjectStatus(raw_status),
            created_at=data.get("created_at", time.time()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            error=data.get("error"),
            cycle_count_total=int(data.get("cycle_count_total") or 0),
            work_items_total=int(data.get("work_items_total") or 0),
            artifacts_root=data.get("artifacts_root"),
            history_log_path=data.get("history_log_path"),
            compact_meta=data.get("_compact") if isinstance(data.get("_compact"), dict) else None,
            task_id=data.get("task_id"),
            verify_command=str(data.get("verify_command") or ""),
            review_pass_streak=int(data.get("review_pass_streak") or 0),
            last_review_role_plan_hash=str(data.get("last_review_role_plan_hash") or ""),
            last_review_blocking_suggestion_hash=str(data.get("last_review_blocking_suggestion_hash") or ""),
            auto_recovery_attempts=max(0, int(data.get("auto_recovery_attempts") or 0)),
        )
        if data.get("cycles"):
            project.cycles = [SpecCycle.from_dict(c) for c in data["cycles"]]
        if data.get("work_items"):
            project.work_items = [SpecWorkItem.from_dict(w) for w in data.get("work_items", [])]
        if data.get("metrics_history"):
            project.metrics_history = [SpecCycleMetrics.from_dict(m) for m in data.get("metrics_history", [])]
        if data.get("criteria_tracker"):
            project.criteria_tracker = CriteriaTracker.from_dict(data["criteria_tracker"])

        # Backfill totals when missing
        if project.cycle_count_total <= 0:
            project.cycle_count_total = len(project.cycles)
        if project.work_items_total <= 0:
            project.work_items_total = len(project.work_items)
        return project
