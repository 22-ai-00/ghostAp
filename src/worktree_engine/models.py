from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


def _clean_optional_str(value: object) -> Optional[str]:
    try:
        text = str(value or "").strip()
    except Exception:
        return None
    return text or None


def _clean_str(value: object, default: str = "") -> str:
    try:
        text = str(value or "").strip()
    except Exception:
        return default
    return text or default


def _migrate_merge_notes(raw: object) -> list[dict]:
    """Deserialize merge_notes with backward-compatible format migration.

    Handles:
    - Current format: list[dict] with 'summary' key
    - Legacy format: list[dict] with 'description' key (migrated to 'summary')
    - Old format: list[str] (converted to dict with 'summary' key)
    """
    if not isinstance(raw, list):
        return []
    result: list[dict] = []
    for item in raw:
        if isinstance(item, dict):
            # Migrate legacy 'description' key → 'summary'
            if "description" in item and "summary" not in item:
                migrated = {k: v for k, v in item.items() if k != "description"}
                migrated["summary"] = item["description"]
                result.append(migrated)
            else:
                result.append(item)
        elif isinstance(item, str) and item.strip():
            # Old format: bare string → minimal dict
            result.append({"summary": item.strip(), "branch": "", "status": "unknown"})
    return result


def _parse_unit_status(value: object) -> "WorktreeUnitStatus":
    """Parse a raw status value into a WorktreeUnitStatus enum, defaulting to PENDING."""
    raw = _clean_str(value, default=WorktreeUnitStatus.PENDING.value)
    try:
        return WorktreeUnitStatus(raw)
    except ValueError:
        return WorktreeUnitStatus.PENDING


def _clean_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass
class WorktreeSelectionItem:
    provider: str
    tool_name: str
    display_name: str
    agent_name: str = ""
    model_name: Optional[str] = None
    model_display_name: Optional[str] = None
    supports_model: bool = True
    model_optional: bool = False
    skip_model_selection: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def selection_key(self) -> str:
        model = _clean_str(self.model_name, default="default")
        return f"{self.provider}:{self.tool_name}:{model}"

    @property
    def agent_display_name(self) -> str:
        agent = _clean_str(self.agent_name)
        if agent == "ttadk":
            return "TTADK"
        return agent.upper() if agent else ""

    @property
    def effective_model_name(self) -> str:
        return _clean_str(self.model_name, default="default")

    @property
    def effective_model_display_name(self) -> str:
        return _clean_str(self.model_display_name or self.model_name, default="默认模型")

    @property
    def display_label(self) -> str:
        base = _clean_str(self.display_name or self.tool_name, default="(unknown)")
        if self.agent_display_name:
            base = f"{self.agent_display_name} · {base}"
        return f"{base} / {self.effective_model_display_name}"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["agent_display_name"] = self.agent_display_name
        data["effective_model_name"] = self.effective_model_name
        data["effective_model_display_name"] = self.effective_model_display_name
        data["selection_key"] = self.selection_key
        data["display_label"] = self.display_label
        data["tool"] = _clean_str(self.display_name or self.tool_name)
        data["model"] = self.effective_model_display_name
        return data

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> Optional["WorktreeSelectionItem"]:
        if not isinstance(data, dict):
            return None
        provider = _clean_str(data.get("provider"))
        tool_name = _clean_str(data.get("tool_name"))
        display_name = _clean_str(data.get("display_name") or tool_name)
        if not provider or not tool_name:
            return None
        agent_name = _clean_str(data.get("agent_name"))
        if not agent_name and provider == "ttadk" and tool_name != "ttadk":
            agent_name = "ttadk"
        return cls(
            provider=provider,
            tool_name=tool_name,
            display_name=display_name,
            agent_name=agent_name,
            model_name=_clean_optional_str(data.get("model_name")),
            model_display_name=_clean_optional_str(data.get("model_display_name")),
            supports_model=bool(data.get("supports_model", True)),
            model_optional=bool(data.get("model_optional", False)),
            skip_model_selection=bool(data.get("skip_model_selection", False)),
            metadata=dict(data.get("metadata") or {}),
        )


class WorktreeSelectionStage(str, Enum):
    """选择流程阶段枚举，替代裸字符串确保类型安全。"""

    IDLE = "idle"
    TOOL_SELECT = "tool_select"
    MODEL_SELECT = "model_select"
    REVIEW = "review"
    READY = "ready"


class WorktreeUnitStatus(str, Enum):
    """Worktree 工作单元状态枚举，替代裸字符串确保类型安全。"""

    PENDING = "pending"
    READY = "ready"
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorktreeJourneyStatus(str, Enum):
    """高层 Worktree 旅程状态机枚举。

    该状态机聚焦于「单一旅程」视角，而非每个 unit 的细粒度状态：

    - ``IDLE``: 尚未进入 /wt 流程，或上一次执行已被完全重置；
    - ``PENDING``: 目标已解析/记入，但尚未发起自动执行（选择工具/模型阶段）；
    - ``AWAITING_GOAL``: 工具/模型已确认并创建 worktree，等待同话题第一条普通消息作为目标；
    - ``AUTO_EXECUTING``: 已进入自动执行关键路径，控制器负责串联"确认 / 创建 / 执行"逻辑；
    - ``RUNNING``: 调度器已开始实际执行 worktree 单元，进度通过回调推送；
    - ``COMPLETED``: 本次旅程成功完成且无致命错误；
    - ``FAILED``: 旅程在某个阶段出现不可忽略的错误（可由上层触发重试）。

    注意：该枚举仅描述「旅程」的生命周期，不替代 unit 级别的 ``status`` 字段，
    两者可以并存，用于不同粒度的 UI 与控制逻辑。
    """

    IDLE = "idle"
    PENDING = "pending"
    AWAITING_GOAL = "awaiting_goal"
    AUTO_EXECUTING = "auto_executing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorktreeJourneyState:
    """描述单次 /wt 旅程的高层状态。

    该结构体被设计为**纯数据模型**：
    - 不依赖外部 I/O（如 Feishu API、线程锁、调度器等）；
    - 可通过 ``to_dict``/``from_dict`` 无损序列化；
    - 可在单元测试中独立验证状态迁移逻辑。
    """

    status: WorktreeJourneyStatus = WorktreeJourneyStatus.IDLE
    goal: str = ""
    last_error: str = ""
    silent_mode: bool = False

    # 视图/控制层可选元数据（不参与决策，仅用于上层映射）：
    origin_message_id: str = ""  # 触发 /wt 的原始消息
    progress_message_id: str = ""  # 当前进度卡片消息（如果已创建）

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "goal": self.goal,
            "last_error": self.last_error,
            "silent_mode": bool(self.silent_mode),
            "origin_message_id": self.origin_message_id,
            "progress_message_id": self.progress_message_id,
        }

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "WorktreeJourneyState":
        if not isinstance(data, dict):
            return cls()

        raw_status = _clean_str(data.get("status"), default=WorktreeJourneyStatus.IDLE.value)
        try:
            status = WorktreeJourneyStatus(raw_status)
        except ValueError:
            status = WorktreeJourneyStatus.IDLE

        return cls(
            status=status,
            goal=_clean_str(data.get("goal")),
            last_error=_clean_str(data.get("last_error")),
            silent_mode=bool(data.get("silent_mode", False)),
            origin_message_id=_clean_str(data.get("origin_message_id")),
            progress_message_id=_clean_str(data.get("progress_message_id")),
        )


@dataclass
class WorktreeSelectionState:
    active: bool = False
    stage: WorktreeSelectionStage = WorktreeSelectionStage.IDLE
    pending_item: Optional[WorktreeSelectionItem] = None
    selected_items: list[WorktreeSelectionItem] = field(default_factory=list)
    pending_goal: str = ""
    last_message: str = ""
    last_error: str = ""

    def add_item(self, item: WorktreeSelectionItem) -> tuple[bool, WorktreeSelectionItem]:
        key = item.selection_key
        for existing in self.selected_items:
            if existing.selection_key == key:
                return False, existing
        self.selected_items.append(item)
        return True, item

    def remove_item(self, selection_key: str) -> Optional[WorktreeSelectionItem]:
        """Remove a selected item by selection_key. Returns the removed item or None."""
        key = (selection_key or "").strip()
        if not key:
            return None
        for idx, existing in enumerate(self.selected_items):
            if existing.selection_key == key:
                return self.selected_items.pop(idx)
        return None

    def clear_items(self) -> int:
        """Clear all selected items. Returns the number removed."""
        n = len(self.selected_items)
        self.selected_items.clear()
        return n

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": bool(self.active),
            "stage": self.stage.value if isinstance(self.stage, WorktreeSelectionStage) else _clean_str(self.stage, default="idle"),
            "pending_item": self.pending_item.to_dict() if self.pending_item else None,
            "selected_items": [item.to_dict() for item in self.selected_items],
            "pending_goal": self.pending_goal,
            "last_message": self.last_message,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "WorktreeSelectionState":
        if not isinstance(data, dict):
            return cls()
        pending_item = WorktreeSelectionItem.from_dict(data.get("pending_item"))
        selected_items = []
        for raw in list(data.get("selected_items") or []):
            item = WorktreeSelectionItem.from_dict(raw)
            if item:
                selected_items.append(item)

        raw_stage = _clean_str(data.get("stage"), default="idle")
        try:
            stage = WorktreeSelectionStage(raw_stage)
        except ValueError:
            stage = WorktreeSelectionStage.IDLE

        return cls(
            active=bool(data.get("active", False)),
            stage=stage,
            pending_item=pending_item,
            selected_items=selected_items,
            pending_goal=_clean_str(data.get("pending_goal")),
            last_message=_clean_str(data.get("last_message")),
            last_error=_clean_str(data.get("last_error")),
        )


@dataclass
class WorktreeUnit:
    unit_id: str
    selection_key: str = ""
    provider: str = ""
    agent_name: str = ""
    tool_name: str = ""
    display_name: str = ""
    model_name: Optional[str] = None
    branch_name: str = ""
    worktree_path: str = ""
    task_title: str = ""
    task_prompt: str = ""
    status: WorktreeUnitStatus = WorktreeUnitStatus.PENDING
    has_changes: bool = False
    summary: str = ""
    error: str = ""
    generation: int = 0
    stop_reason: str = ""
    cancellation_requested: bool = False
    cancellation_acknowledged: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    # Per-unit cancellation signal: set by pool-timeout to notify the worker thread.
    # Provides a memory-barrier-backed check (threading.Event is thread-safe) instead
    # of relying on bare status field reads which depend on GIL atomicity.
    _cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "selection_key": self.selection_key,
            "provider": self.provider,
            "agent_name": self.agent_name,
            "tool_name": self.tool_name,
            "display_name": self.display_name,
            "model_name": self.model_name,
            "branch_name": self.branch_name,
            "worktree_path": self.worktree_path,
            "task_title": self.task_title,
            "task_prompt": self.task_prompt,
            "status": self.status.value if isinstance(self.status, WorktreeUnitStatus) else str(self.status),
            "has_changes": self.has_changes,
            "summary": self.summary,
            "error": self.error,
            "generation": int(self.generation),
            "stop_reason": self.stop_reason,
            "cancellation_requested": bool(self.cancellation_requested),
            "cancellation_acknowledged": bool(self.cancellation_acknowledged),
            "metadata": dict(self.metadata),
            "cancelled": (self.status == WorktreeUnitStatus.CANCELLED),
        }

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> Optional["WorktreeUnit"]:
        if not isinstance(data, dict):
            return None
        unit_id = _clean_str(data.get("unit_id"))
        if not unit_id:
            return None
        provider = _clean_str(data.get("provider"))
        tool_name = _clean_str(data.get("tool_name"))
        agent_name = _clean_str(data.get("agent_name"))
        if not agent_name and provider == "ttadk" and tool_name != "ttadk":
            agent_name = "ttadk"
        unit = cls(
            unit_id=unit_id,
            selection_key=_clean_str(data.get("selection_key")),
            provider=provider,
            agent_name=agent_name,
            tool_name=tool_name,
            display_name=_clean_str(data.get("display_name") or data.get("tool_name")),
            model_name=_clean_optional_str(data.get("model_name")),
            branch_name=_clean_str(data.get("branch_name")),
            worktree_path=_clean_str(data.get("worktree_path")),
            task_title=_clean_str(data.get("task_title")),
            task_prompt=_clean_str(data.get("task_prompt")),
            status=_parse_unit_status(data.get("status")),
            has_changes=bool(data.get("has_changes", False)),
            summary=_clean_str(data.get("summary")),
            error=_clean_str(data.get("error")),
            generation=_clean_int(data.get("generation")),
            stop_reason=_clean_str(data.get("stop_reason")),
            cancellation_requested=bool(
                data.get("cancellation_requested", False)
            ),
            cancellation_acknowledged=bool(
                data.get("cancellation_acknowledged", False)
            ),
            metadata=dict(data.get("metadata") or {}),
        )
        return unit


@dataclass
class DeleteWarning:
    """Warning returned when a worktree has uncommitted changes or unmerged branches."""

    has_uncommitted: bool = False
    uncommitted_files: list[str] = field(default_factory=list)
    has_unmerged: bool = False
    unmerged_branch: str = ""

    @property
    def is_safe(self) -> bool:
        return not self.has_uncommitted and not self.has_unmerged

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_uncommitted": self.has_uncommitted,
            "uncommitted_files": list(self.uncommitted_files),
            "has_unmerged": self.has_unmerged,
            "unmerged_branch": self.unmerged_branch,
            "is_safe": self.is_safe,
        }


@dataclass
class WorktreeInfo:
    """Structured info for a single worktree entry from ``git worktree list``."""

    path: str = ""
    branch: str = ""
    commit: str = ""
    is_active: bool = False
    last_updated: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "branch": self.branch,
            "commit": self.commit,
            "is_active": self.is_active,
            "last_updated": self.last_updated,
        }


@dataclass
class WorktreeRuntimeState:
    enabled: bool = False
    git_initialized_locally: bool = False
    git_root: str = ""
    base_branch: str = ""
    merge_entry_ready: bool = False
    selection: WorktreeSelectionState = field(default_factory=WorktreeSelectionState)
    units: list[WorktreeUnit] = field(default_factory=list)
    summary_lines: list[str] = field(default_factory=list)
    merge_notes: list[dict] = field(default_factory=list)
    last_error: str = ""
    iteration_count: int = 0
    review_plan: dict[str, Any] = field(default_factory=dict)
    review_outcome: dict[str, Any] = field(default_factory=dict)
    # 高层旅程状态，用于统一 /wt 自动执行路径的状态流转
    journey: WorktreeJourneyState = field(default_factory=WorktreeJourneyState)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "git_initialized_locally": bool(self.git_initialized_locally),
            "git_root": self.git_root,
            "base_branch": self.base_branch,
            "merge_entry_ready": bool(self.merge_entry_ready),
            "selection": self.selection.to_dict(),
            "units": [unit.to_dict() for unit in self.units],
            "summary_lines": list(self.summary_lines),
            "merge_notes": list(self.merge_notes),
            "last_error": self.last_error,
            "iteration_count": int(self.iteration_count),
            "review_plan": dict(self.review_plan or {}),
            "review_outcome": dict(self.review_outcome or {}),
            "journey": self.journey.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "WorktreeRuntimeState":
        if not isinstance(data, dict):
            return cls()
        units = []
        for raw in list(data.get("units") or []):
            unit = WorktreeUnit.from_dict(raw)
            if unit:
                units.append(unit)
        journey = WorktreeJourneyState.from_dict(data.get("journey"))
        # 向前兼容：历史数据中 last_user_goal 迁移到 journey.goal
        legacy_goal = _clean_str(data.get("last_user_goal"))
        if legacy_goal and not journey.goal:
            journey.goal = legacy_goal
        return cls(
            enabled=bool(data.get("enabled", False)),
            git_initialized_locally=bool(data.get("git_initialized_locally", False)),
            git_root=_clean_str(data.get("git_root")),
            base_branch=_clean_str(data.get("base_branch")),
            merge_entry_ready=bool(data.get("merge_entry_ready", False)),
            selection=WorktreeSelectionState.from_dict(data.get("selection")),
            units=units,
            summary_lines=[_clean_str(x) for x in list(data.get("summary_lines") or []) if _clean_str(x)],
            merge_notes=_migrate_merge_notes(data.get("merge_notes")),
            last_error=_clean_str(data.get("last_error")),
            iteration_count=_clean_int(data.get("iteration_count")),
            review_plan=dict(data.get("review_plan") or {}),
            review_outcome=dict(data.get("review_outcome") or {}),
            journey=journey,
        )


def ensure_worktree_state(project: object) -> "WorktreeRuntimeState":
    """获取或懒初始化 project 上的 WorktreeRuntimeState，作为唯一入口消除重复实现。"""
    state = getattr(project, "worktree_state", None)
    if not isinstance(state, WorktreeRuntimeState):
        state = WorktreeRuntimeState()
        project.worktree_state = state  # type: ignore[attr-defined]
    return state


def truncate_goal(goal: object, max_len: int = 80) -> str:
    """按 Unicode 码点安全截断 goal 文本，超长时添加 '...' 后缀。

    ``max_len`` 默认值与 ``GOAL_DISPLAY_MAX_LEN`` 保持一致（80）。
    """
    text = _clean_str(goal)
    if not text or len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def transition_journey_state(
    state: WorktreeJourneyState,
    *,
    event: str,
    goal: Optional[str] = None,
    error: Optional[str] = None,
    silent_mode: Optional[bool] = None,
) -> WorktreeJourneyState:
    """纯函数式的 Worktree 旅程状态迁移表。

    设计原则：
    - **输入**: 当前 ``state`` + 语义化 ``event`` 与可选上下文字段；
    - **输出**: 新的 ``WorktreeJourneyState`` 实例（不就地修改入参）；
    - **非法迁移**: 保持原状态不变，仅在 ``last_error`` 中记录提示由上层决定是否使用。
    """

    event_key = (event or "").strip().lower()
    new = WorktreeJourneyState.from_dict(state.to_dict())

    if event_key == "reset":
        return WorktreeJourneyState()

    if event_key == "goal_created":
        new.goal = _clean_str(goal)
        new.status = WorktreeJourneyStatus.PENDING if new.goal else WorktreeJourneyStatus.IDLE
        new.last_error = ""
        return new

    if event_key == "awaiting_goal":
        if new.status not in {WorktreeJourneyStatus.IDLE, WorktreeJourneyStatus.PENDING}:
            new.last_error = new.last_error or "非法状态迁移: 仅在 IDLE/PENDING 时可等待目标"
            return new
        new.status = WorktreeJourneyStatus.AWAITING_GOAL
        new.last_error = ""
        return new

    if event_key == "auto_execute_started":
        if new.status not in {WorktreeJourneyStatus.PENDING, WorktreeJourneyStatus.AWAITING_GOAL, WorktreeJourneyStatus.IDLE}:
            # 不允许在运行中反复进入 auto-executing，交由上层决定是否重试
            new.last_error = new.last_error or "非法状态迁移: 仅在 PENDING/IDLE 时可进入 AUTO_EXECUTING"
            return new
        new.status = WorktreeJourneyStatus.AUTO_EXECUTING
        if goal is not None:
            new.goal = _clean_str(goal)
        if silent_mode is not None:
            new.silent_mode = bool(silent_mode)
        new.last_error = ""
        return new

    if event_key == "execution_started":
        if new.status not in {WorktreeJourneyStatus.PENDING, WorktreeJourneyStatus.AWAITING_GOAL, WorktreeJourneyStatus.AUTO_EXECUTING}:
            new.last_error = new.last_error or "非法状态迁移: 仅在 PENDING/AUTO_EXECUTING 时可进入 RUNNING"
            return new
        new.status = WorktreeJourneyStatus.RUNNING
        if goal is not None:
            new.goal = _clean_str(goal)
        if silent_mode is not None:
            new.silent_mode = bool(silent_mode)
        new.last_error = ""
        return new

    if event_key == "execution_succeeded":
        if new.status not in {WorktreeJourneyStatus.RUNNING, WorktreeJourneyStatus.AUTO_EXECUTING}:
            new.last_error = new.last_error or "非法状态迁移: 仅在 RUNNING/AUTO_EXECUTING 时可进入 COMPLETED"
            return new
        new.status = WorktreeJourneyStatus.COMPLETED
        new.last_error = ""
        return new

    if event_key == "execution_failed":
        if new.status not in {WorktreeJourneyStatus.RUNNING, WorktreeJourneyStatus.AUTO_EXECUTING}:
            new.last_error = new.last_error or "非法状态迁移: 仅在 RUNNING/AUTO_EXECUTING 时可进入 FAILED"
            return new
        new.status = WorktreeJourneyStatus.FAILED
        new.last_error = _clean_str(error)
        return new

    # 未识别事件: 保持原样，附带错误信息便于调试
    new.last_error = new.last_error or f"未知旅程事件: {event_key or '<?>'}"
    return new
