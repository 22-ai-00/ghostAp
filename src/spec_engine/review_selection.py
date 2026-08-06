from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Iterable

from src.acp.tool_discovery import AgentToolOption

if TYPE_CHECKING:
    from src.project.context import ProjectContext


def _clean_optional_str(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _clean_str(value: object, default: str = "") -> str:
    text = str(value or "").strip()
    return text or default


@dataclass
class SpecReviewSelectionItem:
    provider: str
    tool_name: str
    display_name: str
    agent_name: str = ""
    model_name: str | None = None
    model_display_name: str | None = None
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
        return agent.upper() if agent else ""

    @property
    def effective_model_name(self) -> str:
        return _clean_str(self.model_name, default="default")

    @property
    def effective_model_display_name(self) -> str:
        return _clean_str(
            self.model_display_name or self.model_name,
            default="默认模型",
        )

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
    def from_dict(
        cls,
        data: dict[str, Any] | None,
    ) -> SpecReviewSelectionItem | None:
        if not isinstance(data, dict):
            return None
        provider = _clean_str(data.get("provider"))
        tool_name = _clean_str(data.get("tool_name"))
        if not provider or not tool_name:
            return None
        agent_name = _clean_str(data.get("agent_name"))
        return cls(
            provider=provider,
            tool_name=tool_name,
            display_name=_clean_str(data.get("display_name") or tool_name),
            agent_name=agent_name,
            model_name=_clean_optional_str(data.get("model_name")),
            model_display_name=_clean_optional_str(data.get("model_display_name")),
            supports_model=bool(data.get("supports_model", True)),
            model_optional=bool(data.get("model_optional", False)),
            skip_model_selection=bool(data.get("skip_model_selection", False)),
            metadata=dict(data.get("metadata") or {}),
        )


class SpecReviewSelectionStage(str, Enum):
    IDLE = "idle"
    TOOL_SELECT = "tool_select"
    MODEL_SELECT = "model_select"
    REVIEW = "review"
    READY = "ready"


@dataclass
class SpecReviewSelectionState:
    active: bool = False
    stage: SpecReviewSelectionStage = SpecReviewSelectionStage.IDLE
    pending_item: SpecReviewSelectionItem | None = None
    selected_items: list[SpecReviewSelectionItem] = field(default_factory=list)
    pending_goal: str = ""
    last_message: str = ""
    last_error: str = ""

    def add_item(
        self,
        item: SpecReviewSelectionItem,
    ) -> tuple[bool, SpecReviewSelectionItem]:
        for existing in self.selected_items:
            if existing.selection_key == item.selection_key:
                return False, existing
        self.selected_items.append(item)
        return True, item

    def remove_item(self, selection_key: str) -> SpecReviewSelectionItem | None:
        key = str(selection_key or "").strip()
        if not key:
            return None
        for index, existing in enumerate(self.selected_items):
            if existing.selection_key == key:
                return self.selected_items.pop(index)
        return None

    def clear_items(self) -> int:
        count = len(self.selected_items)
        self.selected_items.clear()
        return count


def _provider_display_name(provider: str) -> str:
    normalized = str(provider or "").strip().lower()
    if normalized == "acp":
        return "ACP"
    return normalized.upper() or "UNKNOWN"


def _build_selection_item(option: AgentToolOption) -> SpecReviewSelectionItem:
    provider = str(option.provider or "").strip().lower()
    tool_name = str(option.tool_name or "").strip().lower()
    agent_name = str(option.agent_name or "").strip().lower()
    return SpecReviewSelectionItem(
        provider=provider,
        tool_name=tool_name,
        display_name=str(option.display_name or option.tool_name or "").strip(),
        agent_name=agent_name,
        supports_model=bool(option.supports_model),
        model_optional=bool(option.model_optional),
        skip_model_selection=bool(option.skip_model_selection),
        metadata={
            "provider_display_name": _provider_display_name(option.provider),
            "description": str(option.description or "").strip(),
        },
    )


def _apply_model_to_item(
    pending_item: SpecReviewSelectionItem,
    *,
    model_name: str | None,
    model_display_name: str | None = None,
) -> SpecReviewSelectionItem:
    return replace(
        pending_item,
        model_name=(str(model_name or "").strip() or None),
        model_display_name=(
            str(model_display_name or model_name or "").strip() or None
        ),
    )


def format_selection_lines(items: Iterable[SpecReviewSelectionItem]) -> list[str]:
    lines: list[str] = []
    for index, item in enumerate(items, start=1):
        provider_name = item.metadata.get(
            "provider_display_name"
        ) or _provider_display_name(item.provider)
        lines.append(f"{index}. `{provider_name}` · {item.display_label}")
    return lines


class SpecReviewSelectionController:
    """Manage Spec review-agent tool and model selection."""

    def __init__(
        self,
        state_getter: Callable[[ProjectContext], SpecReviewSelectionState]
        | None = None,
        state_resetter: Callable[[ProjectContext], SpecReviewSelectionState]
        | None = None,
    ) -> None:
        self._state_getter = state_getter or self._default_state_getter
        self._state_resetter = state_resetter

    @staticmethod
    def _default_state_getter(project: ProjectContext) -> SpecReviewSelectionState:
        state = getattr(project, "spec_review_selection_state", None)
        if not isinstance(state, SpecReviewSelectionState):
            state = SpecReviewSelectionState()
            project.spec_review_selection_state = state
        return state

    def _get_state(self, project: ProjectContext) -> SpecReviewSelectionState:
        return self._state_getter(project)

    def start_selection(
        self,
        project: ProjectContext,
        goal: str = "",
    ) -> SpecReviewSelectionState:
        state = self._get_state(project)
        state.active = True
        state.stage = SpecReviewSelectionStage.TOOL_SELECT
        state.pending_item = None
        state.pending_goal = str(goal or "").strip()
        state.last_error = ""
        state.last_message = "请选择一个工具开始评审组合"
        return state

    def reset_selection(self, project: ProjectContext) -> SpecReviewSelectionState:
        if self._state_resetter is not None:
            state = self._state_resetter(project)
        else:
            state = SpecReviewSelectionState()
            project.spec_review_selection_state = state
        state.active = True
        state.stage = SpecReviewSelectionStage.TOOL_SELECT
        state.last_message = "已清空已有选择，请重新开始"
        return state

    def select_tool(
        self,
        project: ProjectContext,
        option: AgentToolOption,
    ) -> SpecReviewSelectionState:
        state = self._get_state(project)
        state.active = True
        state.pending_item = _build_selection_item(option)
        state.stage = (
            SpecReviewSelectionStage.MODEL_SELECT
            if option.supports_model
            else SpecReviewSelectionStage.REVIEW
        )
        state.last_error = ""
        state.last_message = f"已选择 {option.display_name}"
        return state

    def add_pending_item(
        self,
        project: ProjectContext,
        *,
        model_name: str | None = None,
        model_display_name: str | None = None,
    ) -> tuple[SpecReviewSelectionState, bool, str]:
        state = self._get_state(project)
        pending_item = state.pending_item
        if pending_item is None:
            state.last_error = "当前没有待确认的工具选择"
            return state, False, state.last_error

        final_item = _apply_model_to_item(
            pending_item,
            model_name=model_name,
            model_display_name=model_display_name,
        )
        added, existing = state.add_item(final_item)
        state.pending_item = None
        state.stage = SpecReviewSelectionStage.REVIEW
        state.last_error = ""
        if added:
            state.last_message = f"已添加 {final_item.display_label}"
            return state, True, state.last_message
        state.last_message = f"已忽略重复选择：{existing.display_label}"
        return state, False, state.last_message

    def remove_selected_item(
        self,
        project: ProjectContext,
        selection_key: str,
    ) -> tuple[SpecReviewSelectionState, bool, str]:
        state = self._get_state(project)
        removed = state.remove_item(selection_key)
        state.active = True
        state.pending_item = None
        state.stage = SpecReviewSelectionStage.TOOL_SELECT
        state.last_error = ""
        if removed is None:
            state.last_message = "未找到对应已选项"
            return state, False, state.last_message
        state.last_message = f"已移除 {removed.display_label}"
        return state, True, state.last_message

    def clear_selected_items(
        self,
        project: ProjectContext,
    ) -> tuple[SpecReviewSelectionState, int, str]:
        state = self._get_state(project)
        count = state.clear_items()
        state.active = True
        state.pending_item = None
        state.stage = SpecReviewSelectionStage.TOOL_SELECT
        state.last_error = ""
        state.last_message = f"已清空已选 {count} 项" if count else "当前没有已选项"
        return state, count, state.last_message

    def back_to_tool_selection(
        self,
        project: ProjectContext,
    ) -> SpecReviewSelectionState:
        state = self._get_state(project)
        state.pending_item = None
        state.active = True
        state.stage = SpecReviewSelectionStage.TOOL_SELECT
        state.last_error = ""
        state.last_message = "请继续选择工具"
        return state

    def finalize_selection(
        self,
        project: ProjectContext,
    ) -> SpecReviewSelectionState:
        state = self._get_state(project)
        state.active = False
        state.pending_item = None
        state.stage = (
            SpecReviewSelectionStage.READY
            if state.selected_items
            else SpecReviewSelectionStage.TOOL_SELECT
        )
        state.last_error = ""
        state.last_message = (
            "评审 Agent 选择已完成" if state.selected_items else "请至少选择一个工具"
        )
        state.pending_goal = ""
        return state

    def set_pending_goal(self, project: ProjectContext, goal: str) -> None:
        self._get_state(project).pending_goal = str(goal or "").strip()
