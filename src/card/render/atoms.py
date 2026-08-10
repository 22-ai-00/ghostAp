"""RenderAtom: smallest renderable unit for card pagination."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Literal

from src.card.render.budget import RenderBudget
from src.card.render.programming_sections import is_programming_thought_block
from src.card.state.models import ContentBlock
from src.card.ui_text import UI_TEXT
from src.utils.text import sanitize_single_line_label, utf8_replace_bytes

logger = logging.getLogger(__name__)

# Single source of truth for all atom kinds.
# renderer.py validates that _ATOM_RENDERERS keys match this type at load time.
AtomKind = Literal[
    "text", "tool_panel", "reasoning", "plan",
    "criteria_panel", "phase_panel", "warning_banner", "progress_bar",
    "task_list", "phase_banner",
    "subagent_dispatch", "activity_digest", "review_role", "spec_plan", "spec_task",
    "execution_current", "execution_history",
    "image",
]

@dataclass
class RenderAtom:
    """Smallest renderable unit. Maps to one or more Feishu schema elements."""

    kind: AtomKind
    elements: list[dict] = field(default_factory=list)
    byte_size: int = 0  # Estimated JSON byte size
    node_count: int = 0  # Element node count
    splittable: bool = False  # Whether this atom can be split across pages
    block_id: str = ""
    content: str = ""  # Raw content (for split operations)
    structural_overhead: int = 0  # Estimated wrapper bytes not present in content


def estimate_atom_size(atom: RenderAtom) -> int:
    """Estimate the JSON byte size of a RenderAtom.

    If elements exist, use actual JSON serialization size.
    Otherwise, estimate from content length * 3 + overhead.
    """
    if atom.elements:
        return len(utf8_replace_bytes(json.dumps(atom.elements)))
    overhead = 100 + max(0, atom.structural_overhead)
    if atom.structural_overhead <= 0:
        # Preserve the established conservative estimate for engine/tool atoms
        # whose eventual panel structure is not represented here.
        return len(utf8_replace_bytes(atom.content)) * 3 + overhead
    # Segmented programming text has an explicit worst-case wrapper reserve, so
    # its content can match the production ensure_ascii=False serializer.
    encoded_content = json.dumps(
        atom.content,
        ensure_ascii=False,
    )
    return len(utf8_replace_bytes(encoded_content)) + overhead


def flatten_to_atoms(
    blocks: tuple[ContentBlock, ...],
    budget: RenderBudget,
    *,
    unified_execution: bool = False,
    terminal: bool = False,
    segmented_text: bool = False,
) -> list[RenderAtom]:
    """Convert ContentBlocks into a flat list of RenderAtoms.

    Tool calls are grouped into compact activity_digest atoms (one-line summary)
    that interleave with reasoning/text atoms in the body section.
    Active (running) tools emit a compact tool_panel atom inline.

    Uses a registry dispatch pattern for simple block→atom mappings.
    """
    from src.card.render.tools import render_active_tool_line, render_activity_digest_panel

    atoms: list[RenderAtom] = []
    pending_tools: list[ContentBlock] = []
    handlers = _get_block_kind_handlers()
    current_task_name = _extract_current_task_name(blocks)
    process_blocks = [
        block for block in blocks if block.kind in {"reasoning", "tool_call"}
    ] if unified_execution else []
    process_inserted = False

    def _append_execution_flow() -> None:
        """Append one current-action row and one chronological history panel."""
        from src.card.render.execution_flow import render_execution_flow

        flow = render_execution_flow(process_blocks, terminal=terminal)
        if flow.current_element is not None:
            current_atom = RenderAtom(
                kind="execution_current",
                block_id="_execution_current",
                content=flow.current_text,
                elements=[flow.current_element],
                splittable=False,
                node_count=3,
            )
            current_atom.byte_size = estimate_atom_size(current_atom)
            atoms.append(current_atom)
        if flow.history_element is not None:
            history_atom = RenderAtom(
                kind="execution_history",
                block_id="_execution_history",
                content=flow.history_text,
                elements=[flow.history_element],
                splittable=False,
                node_count=4,
            )
            history_atom.byte_size = estimate_atom_size(history_atom)
            atoms.append(history_atom)

    def _flush_pending() -> None:
        """Flush accumulated completed/failed tools as a single activity_digest atom."""
        if not pending_tools:
            return
        digest_panel = render_activity_digest_panel(pending_tools)
        if digest_panel:
            atom = RenderAtom(
                kind="activity_digest",
                block_id=pending_tools[0].block_id,
                content=str(digest_panel["header"]["title"].get("content", "")),
                elements=[digest_panel],
                splittable=False,
                node_count=4,
            )
            atom.byte_size = estimate_atom_size(atom)
            atoms.append(atom)
        pending_tools.clear()

    i = 0
    n = len(blocks)

    while i < n:
        block = blocks[i]

        if unified_execution and block.kind in {"reasoning", "tool_call"}:
            if not process_inserted:
                _append_execution_flow()
                process_inserted = True
            i += 1
        elif block.kind == "tool_call":
            if block.status == "active":
                # Flush any pending completed tools first
                _flush_pending()
                # Active tool: emit compact one-line indicator in body
                display_block = _with_current_task_summary(block, current_task_name)
                active_text = render_active_tool_line(display_block)
                atom = RenderAtom(
                    kind="tool_panel",
                    block_id=block.block_id,
                    content=active_text,
                    splittable=False,
                    node_count=3,
                )
                atom.byte_size = estimate_atom_size(atom)
                atoms.append(atom)
            else:
                # Completed or failed: accumulate for digest
                pending_tools.append(block)
            i += 1
        elif block.kind == "spec_task":
            _flush_pending()
            task_blocks = [block]
            j = i + 1
            while j < n and blocks[j].kind == "spec_task":
                task_blocks.append(blocks[j])
                j += 1

            from src.card.render.spec_artifacts import render_spec_task_list_panel

            task_panel = render_spec_task_list_panel(task_blocks)
            if task_panel:
                atom = RenderAtom(
                    kind="spec_task",
                    block_id=task_blocks[0].block_id,
                    content=_spec_task_list_content(task_blocks),
                    elements=[task_panel],
                    splittable=False,
                    node_count=6,
                )
                atom.byte_size = estimate_atom_size(atom)
                atoms.append(atom)
            i = j
        else:
            # Non-tool block: flush pending tools, then dispatch normally
            _flush_pending()
            handler = handlers.get(block.kind)
            if handler is not None:
                if (
                    segmented_text
                    and block.kind == "text"
                    and not str(block.content or "").strip()
                ):
                    i += 1
                    continue
                atom = handler(block)
                if (
                    segmented_text
                    and block.kind == "text"
                    and is_programming_thought_block(block)
                ):
                    # collapsible_panel + header markdown + body markdown.
                    # The byte overhead covers the static Card 2.0 wrapper so
                    # pagination remains conservative before actual rendering.
                    atom.node_count = 3
                    # Worst-case static wrapper includes a sanitized 60-char
                    # Chinese subagent title. Keep this conservative while
                    # avoiding the old UTF-8 double multiplication.
                    atom.structural_overhead = 720
                    atom.byte_size = estimate_atom_size(atom)
                atoms.append(atom)
            else:
                logger.warning("flatten_to_atoms: unknown block kind %r (block_id=%s), skipping", block.kind, block.block_id)
                # Produce a visible placeholder so users know content failed to render
                error_text = UI_TEXT["card_content_load_error"]
                placeholder = RenderAtom(
                    kind="warning_banner",
                    block_id=block.block_id,
                    content=error_text,
                    node_count=1,
                )
                placeholder.byte_size = estimate_atom_size(placeholder)
                atoms.append(placeholder)
            i += 1

    # Flush any remaining pending tools at end of blocks
    _flush_pending()

    return atoms


# --- Block-to-atom handler functions (registered via _get_block_kind_handlers) ---

def _spec_task_list_content(blocks: list[ContentBlock]) -> str:
    content_parts: list[str] = []
    for block in blocks:
        data = getattr(block, "data", None) or {}
        if not isinstance(data, dict):
            continue
        dependencies = data.get("dependencies") or []
        content_parts.append(str(data.get("task_id") or ""))
        content_parts.append(str(data.get("description") or ""))
        if isinstance(dependencies, list):
            content_parts.append(",".join(str(item) for item in dependencies))
        else:
            content_parts.append(str(dependencies))
    return "\n".join(part for part in content_parts if part)


def _block_to_text_atom(block: ContentBlock) -> RenderAtom:
    atom = RenderAtom(
        kind="text", block_id=block.block_id, content=block.content,
        splittable=True, node_count=1,
    )
    atom.byte_size = estimate_atom_size(atom)
    return atom


def _block_to_image_atom(block: ContentBlock) -> RenderAtom:
    image_key = str(getattr(block, "image_key", "") or "")
    alt = sanitize_single_line_label(
        getattr(block, "alt", ""),
        fallback="任务图片",
        max_chars=120,
    )
    if image_key and block.status != "failed":
        element = {
            "tag": "img",
            "img_key": image_key,
            "alt": {"tag": "plain_text", "content": alt},
        }
    else:
        element = {
            "tag": "markdown",
            "content": "🖼️ 图片产物暂时无法展示；文本结果与本地文件不受影响。",
            "text_size": "notation",
        }
    atom = RenderAtom(
        kind="image",
        elements=[element],
        block_id=block.block_id,
        splittable=False,
        node_count=2 if image_key else 1,
    )
    atom.byte_size = estimate_atom_size(atom)
    return atom


def _extract_current_task_name(blocks: tuple[ContentBlock, ...]) -> str:
    """Return the active task name from the task list block, if available."""
    for block in blocks:
        if getattr(block, "kind", "") != "task_list":
            continue
        tasks = tuple(getattr(block, "tasks", ()) or ())
        current_task_id = str(getattr(block, "current_task_id", "") or "").strip()
        if current_task_id:
            for task in tasks:
                if _task_field(task, "task_id") == current_task_id:
                    name = _task_field(task, "name")
                    if name:
                        return name
        for task in tasks:
            if _task_field(task, "status") == "in_progress":
                name = _task_field(task, "name")
                if name:
                    return name
    return ""


def _with_current_task_summary(block: ContentBlock, current_task_name: str) -> ContentBlock:
    if not current_task_name:
        return block
    if not _is_task_tool_name(getattr(block, "tool_name", "")):
        return block
    if _is_helpful_task_text(getattr(block, "tool_summary", "")):
        return block
    return replace(block, tool_summary=current_task_name)


def _task_field(task: object, key: str) -> str:
    if isinstance(task, dict):
        return str(task.get(key) or "").strip()
    return str(getattr(task, key, "") or "").strip()


def _is_task_tool_name(tool_name: str | None) -> bool:
    return str(tool_name or "").strip().lower() == "task"


def _is_helpful_task_text(value: str | None) -> bool:
    text = str(value or "").strip()
    return bool(text) and text.lower() not in {"task", "任务", "{", "}", "[", "]"}


def _block_to_reasoning_atom(block: ContentBlock) -> RenderAtom:
    atom = RenderAtom(
        kind="reasoning", block_id=block.block_id, content=block.content or "",
        splittable=True, node_count=4,
    )
    atom.byte_size = estimate_atom_size(atom)
    return atom


def _block_to_plan_atom(block: ContentBlock) -> RenderAtom:
    atom = RenderAtom(
        kind="plan", block_id=block.block_id, content=block.content,
        splittable=False, node_count=1,
    )
    atom.byte_size = estimate_atom_size(atom)
    return atom


def _block_to_criteria_atom(block: ContentBlock) -> RenderAtom:
    atom = RenderAtom(
        kind="criteria_panel", block_id=block.block_id, content=block.content,
        splittable=False, node_count=1,
    )
    atom.byte_size = estimate_atom_size(atom)
    return atom


def _block_to_phase_atom(block: ContentBlock) -> RenderAtom:
    atom = RenderAtom(
        kind="phase_panel", block_id=block.block_id, content=block.content,
        splittable=False, node_count=1,
    )
    atom.byte_size = estimate_atom_size(atom)
    return atom


def _block_to_task_list_atom(block: ContentBlock) -> RenderAtom:
    # Build content from tasks for size estimation
    tasks = getattr(block, "tasks", ())
    content_lines = [t.get("name", "") for t in tasks] if tasks else []
    content = "\n".join(content_lines)
    atom = RenderAtom(
        kind="task_list", block_id=block.block_id, content=content,
        splittable=False, node_count=1,
    )
    atom.byte_size = estimate_atom_size(atom)
    return atom


def _block_to_review_role_atom(block: ContentBlock) -> RenderAtom:
    data = getattr(block, "data", None) or {}
    content_parts = [
        str(data.get("title") or ""),
        str(data.get("summary") or ""),
        "\n".join(str(item) for item in data.get("suggestions") or []),
    ]
    atom = RenderAtom(
        kind="review_role",
        block_id=block.block_id,
        content="\n".join(part for part in content_parts if part),
        splittable=False,
        node_count=1,
    )
    atom.byte_size = estimate_atom_size(atom)
    return atom


def _block_to_spec_plan_atom(block: ContentBlock) -> RenderAtom:
    data = getattr(block, "data", None) or {}
    content_parts = [
        str(data.get("architecture") or ""),
        "\n".join(str(item) for item in data.get("steps") or []),
        "\n".join(str(item) for item in data.get("file_changes") or []),
        "\n".join(str(item) for item in data.get("test_plan") or []),
    ]
    content = "\n".join(part for part in content_parts if part)
    atom = RenderAtom(
        kind="spec_plan",
        block_id=block.block_id,
        content=content,
        splittable=False,
        node_count=6,
    )
    atom.byte_size = estimate_atom_size(atom) + 900
    return atom


def _block_to_spec_task_atom(block: ContentBlock) -> RenderAtom:
    data = getattr(block, "data", None) or {}
    dependencies = data.get("dependencies") or []
    content_parts = [
        str(data.get("task_id") or ""),
        str(data.get("description") or ""),
        ",".join(str(item) for item in dependencies) if isinstance(dependencies, list) else str(dependencies),
    ]
    atom = RenderAtom(
        kind="spec_task",
        block_id=block.block_id,
        content="\n".join(part for part in content_parts if part),
        splittable=False,
        node_count=6,
    )
    atom.byte_size = estimate_atom_size(atom) + 800
    return atom


def _block_to_separator_atom(block: ContentBlock) -> RenderAtom:
    task_name = getattr(block, "task_name", "")
    is_first = getattr(block, "is_first_overflow", False)
    status_emoji = getattr(block, "status_emoji", "⏳")
    key = "orch_overflow_separator_first" if is_first else "orch_overflow_separator"
    content = UI_TEXT[key].format(task_name=task_name, status_emoji=status_emoji)
    atom = RenderAtom(
        kind="text", block_id=block.block_id, content=content,
        splittable=False, node_count=1,
    )
    atom.byte_size = estimate_atom_size(atom)
    return atom


# Registry: maps block.kind → handler function.
# tool_call is handled separately due to lookahead grouping logic.
# Lazy-initialized on first use to avoid import-time coupling with block_registry.

_ATOM_HANDLER_DISPATCH: dict[str, Callable[[ContentBlock], RenderAtom]] = {
    "text": _block_to_text_atom,
    "image": _block_to_image_atom,
    "reasoning": _block_to_reasoning_atom,
    "plan": _block_to_plan_atom,
    "criteria": _block_to_criteria_atom,
    "phase": _block_to_phase_atom,
    "task_list": _block_to_task_list_atom,
    "review_role": _block_to_review_role_atom,
    "spec_plan": _block_to_spec_plan_atom,
    "spec_task": _block_to_spec_task_atom,
    "separator": _block_to_separator_atom,
}

# Module-level lazy cache for block kind handlers (avoids @functools.cache semantics)
_block_kind_handlers: dict[str, Callable[[ContentBlock], RenderAtom]] | None = None
_handlers_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock


def _get_block_kind_handlers() -> dict[str, Callable[[ContentBlock], RenderAtom]]:
    """Build and cache the block-kind-to-handler mapping (lazy init).

    Deferred to first call to avoid models.py → block_registry.py → atoms.py
    import-time coupling that could lead to circular imports.
    """
    global _block_kind_handlers
    with _handlers_lock:
        if _block_kind_handlers is None:
            from src.card.state.block_registry import BLOCK_KIND_TO_ATOM

            handlers = _ATOM_HANDLER_DISPATCH.copy()

            # Startup assertion: all registered block kinds must have a handler
            # (tool_call is excluded — it has dedicated lookahead grouping logic)
            missing = set(BLOCK_KIND_TO_ATOM.keys()) - set(handlers.keys()) - {"tool_call"}
            if missing:
                raise RuntimeError(
                    f"BLOCK_KIND_TO_ATOM contains kinds with no handler registered: {missing}. "
                    f"Add handlers in _ATOM_HANDLER_DISPATCH."
                )

            _block_kind_handlers = handlers
    return _block_kind_handlers
