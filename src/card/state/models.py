"""Immutable card state dataclasses.

ContentBlock is a tagged-union: each block kind is a distinct frozen dataclass
sharing common fields (kind, block_id, content, element_id, status).
The Union type alias `ContentBlock` enables isinstance-based dispatch.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal, TypeAlias, Union, get_args

from .runtime_stats import RuntimeStats

if TYPE_CHECKING:
    from src.card.events.payloads import TaskSnapshotPayload


TerminalStatus = Literal[
    "running", "completed", "failed", "cancelled", "paused", "archived", "blocked"
]

TerminalReason = Literal[
    "completed", "cancelled", "ttl_expired", "failed", "archived", "blocked"
]

BlockStatus = Literal["active", "completed", "failed"]


@dataclass(frozen=True)
class CardMetadata:
    """Project/tool/model metadata for the card."""
    project_name: str | None = None
    question_title: str | None = None
    mode_name: str = "Coco"
    mode_emoji: str = "🤖"
    unit_id: str | None = None
    unit_kind: str | None = None
    unit_label: str | None = None
    iteration_index: int | None = None
    iteration_total: int | None = None
    tool_name: str | None = None
    model_name: str | None = None
    engine_type: str | None = None  # "deep" / "spec" / None
    compact: bool = False
    expanded: bool = False
    expand_ac: bool = False
    # Continuation sequence is the canonical SSOT for split/rotation cards.
    # ``card_sequence`` is kept as a display override for dotted subagent IDs
    # (for example ``5.a``). When callers still pass only continuation_seq,
    # __post_init__ derives the visible card sequence from it.
    continuation_seq: int = 0  # >0 means this is a continuation card (续 #N)
    idle_timeout_seconds: int | None = None  # Session idle timeout for footer display
    card_sequence: int | str = 1
    session_started_at: float | None = None
    working_dir: str | None = None
    is_subagent: bool = False
    parent_card_seq: str | None = None
    final_state_for_freeze: "CardState | None" = None
    frozen: bool = False
    frozen_total_elapsed: float | None = None
    bridge_phrase: str | None = None
    live_ticker_frame: str | None = None
    subagents: tuple[dict, ...] = ()
    programming_text_sections: bool = False

    def __post_init__(self) -> None:
        if self.continuation_seq > 0 and self.card_sequence == 1:
            object.__setattr__(self, "card_sequence", self.continuation_seq + 1)


@dataclass(frozen=True)
class HeaderState:
    """Card header state."""
    title: str = ""
    subtitle: str | None = None
    template: str = "blue"
    header_source: Literal["lifecycle", "engine"] = "lifecycle"


@dataclass(frozen=True)
class FooterState:
    """Card footer state.

    Contains progress and warning sub-fields for clear responsibility separation.
    Progress rendering priority (mutually exclusive — highest wins):
        1. ``progress_pct`` — raw percentage (0-100), rendered as visual bar by footer
        2. ``progress`` — plain text progress description

    Only the highest-priority non-None field is rendered. Callers should set
    exactly one of these fields per dispatch to avoid ambiguity.
    """
    status: Literal["thinking", "tool_running", "idle"] | None = None
    status_text: str | None = None
    progress: str | None = None
    progress_pct: int | None = None  # Raw percentage (0-100), rendered by footer
    warning_banner: str | None = None  # Warning text shown above footer
    warning_type: Literal["success", "warning", "error", "info"] | None = None  # Semantic banner type
    persistent_warning: bool = False  # If True, warning_banner is not cleared by content reducers
    progress_started_at: float | None = None  # Monotonic time when progress tracking began
    duration_seconds: float | None = None  # Total elapsed time (set on terminal events)
    last_updated_at: str | None = None  # HH:MM timestamp of last successful delivery


@dataclass(frozen=True)
class ButtonSpec:
    """Button specification."""
    text: str = ""
    action_id: str = ""
    type: Literal["primary", "default", "danger"] = "default"
    confirm: str | None = None
    url: str | None = None  # If set, renders as URL-open button instead of callback
    disabled: bool = False
    disabled_text: str | None = None  # Tooltip shown when button is disabled
    value: dict | None = None  # Optional callback payload; action_id is used as default action


# ---------------------------------------------------------------------------
# ContentBlock tagged-union: per-kind dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextBlock:
    """Plain text content block (supports streaming via element_id)."""
    _atom_kind: ClassVar[str] = "text"
    kind: Literal["text"] = "text"
    block_id: str = ""
    content: str = ""
    element_id: str | None = None
    status: BlockStatus = "active"
    source_kind: Literal["main", "subagent"] = "main"
    source_sequence: str | None = None
    source_label: str | None = None
    source_ref: str = "main"


@dataclass(frozen=True)
class ToolBlock:
    """Tool call content block."""
    _atom_kind: ClassVar[str] = "tool_panel"
    kind: Literal["tool_call"] = "tool_call"
    block_id: str = ""
    content: str = ""
    element_id: str | None = None
    status: BlockStatus = "active"
    tool_name: str | None = None
    tool_summary: str | None = None
    tool_input: str | None = None
    tool_output: str | None = None
    is_latest_active: bool = False


@dataclass(frozen=True)
class ReasoningBlock:
    """Reasoning/thinking content block."""
    _atom_kind: ClassVar[str] = "reasoning"
    kind: Literal["reasoning"] = "reasoning"
    block_id: str = ""
    content: str = ""
    element_id: str | None = None
    status: BlockStatus = "active"
    char_count: int = 0


@dataclass(frozen=True)
class ImageBlock:
    """Image artifact uploaded to Feishu or its non-fatal fallback."""
    _atom_kind: ClassVar[str] = "image"
    kind: Literal["image"] = "image"
    block_id: str = ""
    content: str = ""
    element_id: str | None = None
    status: BlockStatus = "completed"
    image_key: str | None = None
    alt: str = "任务图片"


@dataclass(frozen=True)
class PlanBlock:
    """Plan content block."""
    _atom_kind: ClassVar[str] = "plan"
    kind: Literal["plan"] = "plan"
    block_id: str = ""
    content: str = ""
    element_id: str | None = None
    status: BlockStatus = "active"


@dataclass(frozen=True)
class PhaseBlock:
    """Engine phase content block (Spec)."""
    _atom_kind: ClassVar[str] = "phase_panel"
    kind: Literal["phase"] = "phase"
    block_id: str = ""
    content: str = ""
    element_id: str | None = None
    status: BlockStatus = "active"
    phase_name: str | None = None
    cycle_num: int | None = None


@dataclass(frozen=True)
class CriteriaBlock:
    """Acceptance criteria content block."""
    _atom_kind: ClassVar[str] = "criteria_panel"
    kind: Literal["criteria"] = "criteria"
    block_id: str = ""
    content: str = ""
    element_id: str | None = None
    status: BlockStatus = "active"


@dataclass(frozen=True)
class TaskListBlock:
    """Task list block — shows all tasks and highlights current task."""
    _atom_kind: ClassVar[str] = "task_list"
    kind: Literal["task_list"] = "task_list"
    block_id: str = "_task_list"
    tasks: tuple[TaskSnapshotPayload, ...] = ()  # Each item: task_id, name, status
    current_task_id: str = ""
    element_id: str | None = None
    status: BlockStatus = "active"


@dataclass(frozen=True)
class SeparatorBlock:
    """Section separator block — visual divider for overflow tasks merged into a card."""
    _atom_kind: ClassVar[str] = "separator"
    kind: Literal["separator"] = "separator"
    block_id: str = ""
    task_name: str = ""
    is_first_overflow: bool = False
    status_emoji: str = "⏳"
    element_id: str | None = None
    status: BlockStatus = "completed"


@dataclass(frozen=True)
class ReviewRoleBlock:
    """One Spec review role result rendered as its own collapsible panel."""
    _atom_kind: ClassVar[str] = "review_role"
    kind: Literal["review_role"] = "review_role"
    block_id: str = ""
    data: dict | None = None
    element_id: str | None = None
    status: BlockStatus = "completed"

    def __post_init__(self) -> None:
        if self.data is not None:
            object.__setattr__(self, "data", copy.deepcopy(self.data))


@dataclass(frozen=True)
class SpecPlanBlock:
    """Spec PLAN artifact rendered as a structured plan panel."""
    _atom_kind: ClassVar[str] = "spec_plan"
    kind: Literal["spec_plan"] = "spec_plan"
    block_id: str = ""
    data: dict | None = None
    element_id: str | None = None
    status: BlockStatus = "completed"

    def __post_init__(self) -> None:
        if self.data is not None:
            object.__setattr__(self, "data", copy.deepcopy(self.data))


@dataclass(frozen=True)
class SpecTaskBlock:
    """One Spec TASK artifact item rendered as a complete task panel."""
    _atom_kind: ClassVar[str] = "spec_task"
    kind: Literal["spec_task"] = "spec_task"
    block_id: str = ""
    data: dict | None = None
    element_id: str | None = None
    status: BlockStatus = "completed"

    def __post_init__(self) -> None:
        if self.data is not None:
            object.__setattr__(self, "data", copy.deepcopy(self.data))


# Tagged-union type alias
AnyContentBlock: TypeAlias = Union[
    TextBlock, ToolBlock, ReasoningBlock, ImageBlock, PlanBlock, PhaseBlock, CriteriaBlock,
    TaskListBlock, SeparatorBlock,
    ReviewRoleBlock, SpecPlanBlock, SpecTaskBlock,
]
"""Union of all content block types. Use isinstance() for type-safe dispatch."""

# Map kind → concrete class for the factory function
_BLOCK_KIND_MAP: dict[str, type] = {
    "text": TextBlock,
    "tool_call": ToolBlock,
    "reasoning": ReasoningBlock,
    "image": ImageBlock,
    "plan": PlanBlock,
    "phase": PhaseBlock,
    "criteria": CriteriaBlock,
    "task_list": TaskListBlock,
    "separator": SeparatorBlock,
    "review_role": ReviewRoleBlock,
    "spec_plan": SpecPlanBlock,
    "spec_task": SpecTaskBlock,
}

# Import-time completeness check: every AnyContentBlock subtype must be registered
_registered_kinds = set(_BLOCK_KIND_MAP.keys())
for _block_cls in get_args(AnyContentBlock):
    _fields = {f.name: f for f in dataclasses.fields(_block_cls)}
    _kind_field = _fields.get("kind")
    if _kind_field is None or _kind_field.default is dataclasses.MISSING:
        raise RuntimeError(
            f"Block class {_block_cls.__name__} missing 'kind' field with default value"
        )
    if _kind_field.default not in _registered_kinds:
        raise RuntimeError(
            f"Block class {_block_cls.__name__} has kind={_kind_field.default!r} "
            f"not registered in _BLOCK_KIND_MAP. Add it to the map."
        )
del _registered_kinds, _block_cls, _fields, _kind_field


def ContentBlock(kind: str = "text", **kwargs) -> AnyContentBlock:  # noqa: N802
    """Factory function: creates the appropriate block subtype based on kind.

    Backwards-compatible with old ``ContentBlock(kind="text", block_id=..., ...)`` usage.
    Only passes kwargs that the target class accepts, silently ignoring extras.

    Extension pattern guidance:
        - Use ``block.data`` for UI-rendering-driven structured content that is
          specific to a single block's visual representation.
        - Use ``CardState.engine_ext`` (EngineExtState) for cross-block aggregate
          metadata shared across the entire card (e.g. cycle count, phase info,
          criteria satisfaction counts).
    """
    cls = _BLOCK_KIND_MAP.get(kind, TextBlock)
    # Filter kwargs to only fields the target class knows about
    valid_fields = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in kwargs.items() if k in valid_fields}
    return cls(**filtered)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Engine extension & top-level state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EngineExtState:
    """Engine-specific extended state (Spec/Deep shared fields).

    Isolated from core CardState to keep the model engine-agnostic.

    Use this for cross-block aggregate metadata that applies to the entire card
    (e.g. iteration cycle count, current phase label, criteria satisfaction).
    For per-block structured UI data, use ``ContentBlock.data`` (TypedDict) instead.
    """
    criteria_section: str | None = None  # Rendered criteria markdown
    criteria_satisfied: int = 0
    criteria_total: int = 0
    phase_info: str | None = None  # Current phase label
    cycle_num: int = 0
    max_cycles: int = 0
    blocked_reason: str | None = None  # Reason for BLOCKED terminal state


@dataclass(frozen=True)
class CardState:
    """Top-level immutable card state produced by reducer."""
    blocks: tuple[AnyContentBlock, ...] = ()
    terminal: TerminalStatus = "running"
    terminal_reason: TerminalReason | None = None
    header: HeaderState = field(default_factory=HeaderState)
    footer: FooterState = field(default_factory=FooterState)
    buttons: tuple[ButtonSpec, ...] = ()
    metadata: CardMetadata = field(default_factory=CardMetadata)
    version: int = 0
    # Incremented only on structural changes (block add/remove, terminal, header/buttons)
    structural_version: int = 0
    # Engine-specific extended state (None for non-engine cards)
    engine_ext: EngineExtState | None = None
    # Runtime context consumed by sticky/banner rendering.
    runtime_stats: RuntimeStats = field(default_factory=RuntimeStats)
    # Lazy O(1) block_id → index cache (built on first access, not on every state change)
    _block_index_cache: dict[str, int] | None = field(
        default=None, repr=False, compare=False, init=False
    )

    @property
    def block_index(self) -> dict[str, int]:
        """O(1) block_id → index lookup. Lazy-built on first access."""
        cache = self._block_index_cache
        if cache is None:
            cache = {b.block_id: i for i, b in enumerate(self.blocks) if b.block_id}
            object.__setattr__(self, "_block_index_cache", cache)
        return cache
