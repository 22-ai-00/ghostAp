"""Main card state reducer — dispatches events to sub-reducers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace

from ..events import CardEvent, CardEventType
from .models import CardMetadata, CardState, EngineExtState
from .reducers._shared import build_header
from .reducers.criteria import reduce_criteria
from .reducers.cycle import reduce_cycle
from .reducers.image import reduce_image
from .reducers.lifecycle import reduce_lifecycle
from .reducers.phase import reduce_phase
from .reducers.plan import reduce_plan
from .reducers.reasoning import reduce_reasoning
from .reducers.review import reduce_review_result
from .reducers.separator import reduce_separator
from .reducers.spec_artifacts import reduce_spec_artifacts
from .reducers.task_list import reduce_task_list
from .reducers.text import reduce_text
from .reducers.tool import reduce_tool
from .runtime_stats import RuntimeStats

logger = logging.getLogger(__name__)

# Maximum number of completed tool blocks to retain (sliding window)
MAX_COMPLETED_TOOL_BLOCKS = 50

# Safety cap: maximum total content blocks allowed in a single card state.
# Prevents unbounded growth from any accumulation path.
MAX_TOTAL_BLOCKS = 100


def _trim_blocks_to_safety_cap(blocks: tuple) -> tuple:
    """Bound state while retaining the active phase lifecycle anchor."""
    if len(blocks) <= MAX_TOTAL_BLOCKS:
        return blocks

    protected = {
        index
        for index, block in enumerate(blocks)
        if block.kind == "phase" and block.status == "active"
    }
    if len(protected) >= MAX_TOTAL_BLOCKS:
        keep = sorted(protected)[-MAX_TOTAL_BLOCKS:]
        return tuple(blocks[index] for index in keep)

    keep = set(protected)
    for index in range(len(blocks) - 1, -1, -1):
        if len(keep) >= MAX_TOTAL_BLOCKS:
            break
        keep.add(index)
    return tuple(blocks[index] for index in sorted(keep))


# --- Inline handlers for events that don't warrant their own module ---

def _reduce_tool_model_changed(state: CardState, event: CardEvent) -> CardState:
    changes = {
        "tool_name": event.payload.get("tool_name") or state.metadata.tool_name,
        "model_name": event.payload.get("model_name") or state.metadata.model_name,
    }
    if "unit_label" in event.payload:
        changes["unit_label"] = event.payload.get("unit_label") or state.metadata.unit_label
    if "live_ticker_frame" in event.payload:
        changes["live_ticker_frame"] = event.payload.get("live_ticker_frame")
    if "subagents" in event.payload:
        changes["subagents"] = tuple(event.payload.get("subagents") or ())
    new_meta = replace(state.metadata, **changes)
    header = build_header(new_meta, state.terminal)
    return replace(state, metadata=new_meta, header=header)


def _reduce_progress_updated(state: CardState, event: CardEvent) -> CardState:
    current = event.payload.get("current", 0)
    total = event.payload.get("total", 0)
    label = event.payload.get("label", "")
    timestamp: float | None = event.payload.get("timestamp")
    footer = state.footer

    # Track when progress started (first non-zero update)
    started_at = footer.progress_started_at
    if started_at is None and current > 0 and timestamp is not None:
        started_at = timestamp

    # Spec engine uses text-only progress (criteria satisfaction semantics)
    # Other engines use visual ▰▱ progress bar (tool execution semantics)
    engine_type = state.metadata.engine_type
    use_visual_bar = engine_type not in ("spec",)

    if total > 0:
        pct = int(current / total * 100) if use_visual_bar else None
        if use_visual_bar:
            progress_text = f"步骤 {current}/{total}"
        else:
            progress_text = f"{current}/{total} 通过"
        if label:
            progress_text += f" · {label}"
        # ETA calculation: only show if we have enough data and using visual bar
        if use_visual_bar and started_at is not None and current > 1 and pct < 100 and timestamp is not None:
            elapsed = timestamp - started_at
            rate = elapsed / current  # seconds per step
            remaining_steps = total - current
            eta_secs = int(rate * remaining_steps)
            if eta_secs >= 60:
                progress_text += f" · 预计还需 {eta_secs // 60}min"
            elif eta_secs > 10:
                progress_text += f" · 预计还需 {eta_secs}s"
    else:
        pct = None
        progress_text = str(label).strip() or None
    return replace(state, footer=replace(footer, progress=progress_text, progress_pct=pct,
                                         progress_started_at=started_at))


# --- Dispatch table: CardEventType → handler(state, event) → new_state ---

# Events that require engine_ext to be initialized
_ENGINE_EXT_EVENTS = frozenset({
    CardEventType.CYCLE_STARTED,
    CardEventType.CYCLE_DONE,
    CardEventType.PHASE_STARTED,
    CardEventType.PHASE_DONE,
    CardEventType.SPEC_PLAN_UPDATED,
    CardEventType.SPEC_TASKS_UPDATED,
    CardEventType.REVIEW_RESULT_UPDATED,
    CardEventType.CRITERIA_UPDATED,
    CardEventType.WARNING_UPDATED,
    CardEventType.REVIEW_RETRY,
})

_REDUCER_DISPATCH: dict[CardEventType, Callable[[CardState, CardEvent], CardState]] = {
    # Text events
    CardEventType.TEXT_STARTED: reduce_text,
    CardEventType.TEXT_DELTA: reduce_text,
    CardEventType.TEXT_DONE: reduce_text,
    # Tool events
    CardEventType.TOOL_STARTED: reduce_tool,
    CardEventType.TOOL_DELTA: reduce_tool,
    CardEventType.TOOL_DONE: reduce_tool,
    CardEventType.TOOL_FAILED: reduce_tool,
    # Image artifact events
    CardEventType.IMAGE_ADDED: reduce_image,
    CardEventType.IMAGE_FAILED: reduce_image,
    # Reasoning events
    CardEventType.REASONING_STARTED: reduce_reasoning,
    CardEventType.REASONING_DELTA: reduce_reasoning,
    CardEventType.REASONING_DONE: reduce_reasoning,
    # Plan
    CardEventType.PLAN_UPDATED: reduce_plan,
    # Lifecycle events
    CardEventType.STARTED: reduce_lifecycle,
    CardEventType.STOPPING: reduce_lifecycle,
    CardEventType.COMPLETED: reduce_lifecycle,
    CardEventType.FAILED: reduce_lifecycle,
    CardEventType.CANCELLED: reduce_lifecycle,
    CardEventType.PAUSED: reduce_lifecycle,
    CardEventType.RESUMED: reduce_lifecycle,
    CardEventType.ARCHIVED: reduce_lifecycle,
    CardEventType.BLOCKED: reduce_lifecycle,
    # Cycle events
    CardEventType.CYCLE_STARTED: reduce_cycle,
    CardEventType.CYCLE_DONE: reduce_cycle,
    # Phase events
    CardEventType.PHASE_STARTED: reduce_phase,
    CardEventType.PHASE_DONE: reduce_phase,
    CardEventType.SPEC_PLAN_UPDATED: reduce_spec_artifacts,
    CardEventType.SPEC_TASKS_UPDATED: reduce_spec_artifacts,
    CardEventType.REVIEW_RESULT_UPDATED: reduce_review_result,
    # Criteria events
    CardEventType.CRITERIA_UPDATED: reduce_criteria,
    CardEventType.WARNING_UPDATED: reduce_criteria,
    CardEventType.REVIEW_RETRY: reduce_criteria,
    # Meta events (inline handlers)
    CardEventType.TOOL_MODEL_CHANGED: _reduce_tool_model_changed,
    CardEventType.PROGRESS_UPDATED: _reduce_progress_updated,
    # UI control events
    CardEventType.MODE_TOGGLED: reduce_lifecycle,
    CardEventType.STOP_ESCALATED: reduce_lifecycle,
    # Task-level card management
    CardEventType.TASK_LIST_UPDATED: reduce_task_list,
    # Section separator (overflow task divider)
    CardEventType.SECTION_SEPARATOR: reduce_separator,
}

# Events that cause structural changes (block add/remove, terminal, header/buttons change)
# Used to decide when to increment structural_version
_STRUCTURAL_EVENTS = frozenset({
    # Block lifecycle (new block created or status changed)
    CardEventType.TEXT_STARTED,
    CardEventType.TEXT_DONE,
    CardEventType.TOOL_STARTED,
    CardEventType.TOOL_DONE,
    CardEventType.TOOL_FAILED,
    CardEventType.IMAGE_ADDED,
    CardEventType.IMAGE_FAILED,
    CardEventType.REASONING_STARTED,
    CardEventType.REASONING_DONE,
    CardEventType.PLAN_UPDATED,
    # Lifecycle (terminal, header changes)
    CardEventType.STARTED,
    CardEventType.COMPLETED,
    CardEventType.FAILED,
    CardEventType.CANCELLED,
    CardEventType.PAUSED,
    CardEventType.RESUMED,
    CardEventType.BLOCKED,
    CardEventType.ARCHIVED,
    # NOTE: TOOL_MODEL_CHANGED is conditionally structural — see _is_structural_event().
    # Tool/model/subagents change → structural (header/buttons change);
    # live_ticker_frame-only updates → non-structural (only footer markdown update).
    # Cycle/phase (header/subtitle change)
    CardEventType.CYCLE_STARTED,
    CardEventType.CYCLE_DONE,
    CardEventType.PHASE_STARTED,
    CardEventType.PHASE_DONE,
    CardEventType.SPEC_PLAN_UPDATED,
    CardEventType.SPEC_TASKS_UPDATED,
    CardEventType.REVIEW_RESULT_UPDATED,
    # UI control (buttons change)
    CardEventType.MODE_TOGGLED,
    CardEventType.STOP_ESCALATED,
    # Task-level card management (block structure change)
    CardEventType.TASK_LIST_UPDATED,
    # Section separator (adds new block)
    CardEventType.SECTION_SEPARATOR,
})


def _is_structural_event(event: CardEvent) -> bool:
    """Decide whether an event should bump structural_version (triggers full update_page).

    Most types are looked up in _STRUCTURAL_EVENTS. TOOL_MODEL_CHANGED is special:
    - LiveTicker frame-only updates (only `live_ticker_frame` payload key) → non-structural,
      so the green-dot animation does NOT cause N×update_page on every active card per 1.2s.
    - Real tool/model/subagents updates → structural (header/buttons change).
    """
    if event.type in _STRUCTURAL_EVENTS:
        return True
    if event.type is CardEventType.TOOL_MODEL_CHANGED:
        payload = event.payload or {}
        # Structural only when header/footer metadata changes.
        if any(k in payload for k in ("tool_name", "model_name", "unit_label", "subagents")):
            return True
        # Pure ticker frame → element_content only.
        return False
    return False


def _as_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _refresh_runtime_stats(state: CardState, event: CardEvent) -> CardState:
    """Derive banner runtime state from immutable card state plus event time."""
    runtime = state.runtime_stats or RuntimeStats()
    elapsed_seconds = runtime.elapsed_seconds

    now = _as_float((event.payload or {}).get("_now"))
    started_at = _as_float(state.metadata.session_started_at)
    authoritative_duration = _as_float(
        (event.payload or {}).get("duration_seconds")
    )
    if authoritative_duration is not None:
        elapsed_seconds = max(0.0, authoritative_duration)
    elif now is not None and started_at is not None:
        elapsed_seconds = max(0.0, now - started_at)

    spec_cycle = runtime.spec_cycle
    spec_perspective = runtime.spec_perspective
    deep_phase = runtime.deep_phase

    engine_type = state.metadata.engine_type
    if engine_type == "spec" and state.engine_ext is not None:
        spec_cycle = state.engine_ext.cycle_num or state.metadata.iteration_index or spec_cycle
        if event.type is CardEventType.CYCLE_STARTED:
            spec_perspective = None
        elif state.engine_ext.phase_info:
            spec_perspective = state.engine_ext.phase_info
    elif engine_type == "deep" and state.engine_ext is not None:
        deep_phase = state.engine_ext.phase_info or deep_phase
    refreshed = RuntimeStats(
        elapsed_seconds=elapsed_seconds,
        deep_phase=deep_phase,
        spec_cycle=spec_cycle,
        spec_perspective=spec_perspective,
    )
    if refreshed == runtime:
        return state
    return replace(state, runtime_stats=refreshed)


def _reduce_card_state_untrimmed(state: CardState, event: CardEvent) -> CardState:
    """Project one event without applying either card-capacity window."""
    # Auto-initialize engine_ext for engine-specific events if not yet set
    if state.engine_ext is None and event.type in _ENGINE_EXT_EVENTS:
        state = replace(state, engine_ext=EngineExtState())

    # Route to sub-reducer via dispatch table (O(1) lookup)
    handler = _REDUCER_DISPATCH.get(event.type)
    if handler is not None:
        new_state = handler(state, event)
    else:
        logger.warning("reduce_card_state: unregistered event type '%s', returning state unchanged", event.type)
        new_state = state

    # Bump version if state changed
    if new_state is not state:
        new_state = _refresh_runtime_stats(new_state, event)
        new_version = state.version + 1
        # Bump structural_version only for structural events
        if _is_structural_event(event):
            new_structural = state.structural_version + 1
        else:
            new_structural = state.structural_version
        new_state = replace(new_state, version=new_version, structural_version=new_structural)

    return new_state


def card_state_requires_continuation(
    state: CardState,
    event: CardEvent,
    *,
    total_block_limit: int = MAX_TOTAL_BLOCKS,
    completed_tool_limit: int = MAX_COMPLETED_TOOL_BLOCKS,
) -> bool:
    """Return whether reducing ``event`` would cross a destructive trim limit.

    The projection deliberately uses the same reducers and version semantics as
    :func:`reduce_card_state`, but observes the result before either retention
    window is applied.  Both inputs are immutable, so callers can safely use
    this as a capacity preflight before deciding whether to start a new card.
    """
    projected = _reduce_card_state_untrimmed(state, event)

    if len(projected.blocks) > total_block_limit:
        return True

    if event.type in (CardEventType.TOOL_DONE, CardEventType.TOOL_FAILED):
        completed_tools = sum(
            1
            for block in projected.blocks
            if block.kind == "tool_call" and block.status == "completed"
        )
        if completed_tools > completed_tool_limit:
            return True

    return False


def reduce_card_state(
    state: CardState | None,
    event: CardEvent,
    metadata: CardMetadata | None = None,
    *,
    retain_all_blocks: bool = False,
) -> CardState:
    """
    Pure function: old state + event → new state. No side effects.

    If state is None, creates initial state (expects STARTED event or provides defaults).
    metadata is only used for initial state creation.

    ``retain_all_blocks`` is reserved for durable projections whose delivery
    layer paginates the complete history (for example one Workflow direct
    call). Ordinary live cards keep the bounded retention windows below.
    """
    if state is None:
        # Initialize with metadata
        meta = metadata or CardMetadata()
        state = CardState(metadata=meta)

    previous_version = state.version
    new_state = _reduce_card_state_untrimmed(state, event)

    # The existing retention windows only run when the reducer changed state.
    # Version advancement is the reducer's canonical signal for that condition.
    if new_state.version == previous_version:
        return new_state

    if retain_all_blocks:
        return new_state

    # Sliding window: trim completed tool blocks to MAX_COMPLETED_TOOL_BLOCKS.
    # Only run on events that can produce new completed tool blocks (perf optimization).
    if event.type in (CardEventType.TOOL_DONE, CardEventType.TOOL_FAILED):
        completed_tool_blocks = [
            block
            for block in new_state.blocks
            if block.kind == "tool_call" and block.status == "completed"
        ]
        if len(completed_tool_blocks) > MAX_COMPLETED_TOOL_BLOCKS:
            excess = len(completed_tool_blocks) - MAX_COMPLETED_TOOL_BLOCKS
            # Identify block_ids to remove (oldest completed tools)
            to_remove = {
                block.block_id
                for block in completed_tool_blocks[:excess]
            }
            trimmed = tuple(
                block
                for block in new_state.blocks
                if block.block_id not in to_remove
            )
            new_state = replace(new_state, blocks=trimmed)

    # Safety cap: prevent unbounded block accumulation from any path.
    # Keep the active phase anchor plus the most recent content so a long
    # run can still close its phase lifecycle without a false warning.
    if len(new_state.blocks) > MAX_TOTAL_BLOCKS:
        new_state = replace(
            new_state,
            blocks=_trim_blocks_to_safety_cap(new_state.blocks),
        )

    return new_state
