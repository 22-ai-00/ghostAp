"""Main card render entry point: state → RenderedCard list."""

from __future__ import annotations

import functools
import hashlib
import logging
import re
from collections.abc import Callable
from typing import get_args as _get_args

from src.card.engine_meta import engine_type_to_cmd
from src.card.render.atoms import AtomKind, RenderAtom, estimate_atom_size, flatten_to_atoms
from src.card.render.budget import RenderBudget
from src.card.render.buttons import render_buttons
from src.card.render.error import render_collapsed_error_panel
from src.card.render.footer import render_footer
from src.card.render.header import render_header
from src.card.render.layout import SectionLayout, paginate_layout
from src.card.render.plan import render_plan_panel
from src.card.render.programming_sections import (
    is_programming_thought_block,
    render_programming_text_section,
)
from src.card.render.reasoning import render_reasoning_panel, truncate_reasoning_for_compact
from src.card.render.review import render_review_role_panel
from src.card.render.spec_artifacts import render_spec_plan_panel, render_spec_task_panel
from src.card.render.sticky_head import build_sticky_head
from src.card.render.tools import build_subagent_dispatch_atom, render_tool_panel
from src.card.state.models import CardState, ContentBlock
from src.card.text_stream import soft_join_text_fragments
from src.card.themes import PANEL_STYLES
from src.card.types import ActiveElement, RenderedCard
from src.card.ui_text import UI_TEXT
from src.utils.text import utf8_replace_bytes

logger = logging.getLogger(__name__)

_STATUS_ATOM_KINDS = frozenset({"warning_banner", "progress_bar", "phase_panel", "criteria_panel", "task_list"})
_BODY_ATOM_KINDS = frozenset({"text", "image", "reasoning", "plan", "subagent_dispatch", "activity_digest", "tool_panel", "review_role", "spec_plan", "spec_task", "execution_current", "execution_history"})
_MIN_STREAMING_TEXT_CHARS = 2
_FENCE_LINE_RE = re.compile(r"^(?P<indent>\s{0,3})(?P<escaped>\\?)(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
_FENCE_LANGUAGE_ALIASES = {
    "bash": "bash",
    "sh": "bash",
    "shell": "bash",
    "zsh": "bash",
    "python": "python",
    "py": "python",
    "json": "json",
    "yaml": "yaml",
    "yml": "yaml",
    "toml": "toml",
    "markdown": "markdown",
    "md": "markdown",
    "diff": "diff",
    "sql": "sql",
    "html": "html",
    "css": "css",
    "javascript": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "ts": "typescript",
    "go": "go",
    "rust": "rust",
    "rs": "rust",
    "java": "java",
    "kotlin": "kotlin",
    "swift": "swift",
    "c": "c",
    "cpp": "cpp",
    "c++": "cpp",
    "text": "text",
    "txt": "text",
}


# Banner background color and icon by warning_type
_BANNER_STYLES: dict[str, tuple[str, str]] = {
    "error": ("red", "❌"),
    "warning": ("yellow", "⚠️"),
    "info": ("wathet", "ℹ️"),
    "success": ("green", "✅"),
}


def _build_column_banner(*, content: str, background_style: str) -> dict:
    """Build a compact banner using column_set.

    Feishu Schema 2.0 does not allow `padding`/`background_style` on `div`.
    Use `column_set` which supports `background_style`.
    """
    return {
        "tag": "column_set",
        "flex_mode": "stretch",
        "background_style": background_style,
        "columns": [
            {
                "tag": "column",
                "width": "weighted",
                "weight": 1,
                "vertical_align": "center",
                "elements": [{"tag": "markdown", "content": content, "text_align": "left"}],
            }
        ],
    }


def _engine_type_to_cmd(engine_type: str | None) -> str:
    """Map engine_type to user-facing command string."""
    return engine_type_to_cmd(engine_type, fallback="命令")


def render_card(
    state: CardState, budget: RenderBudget
) -> list[RenderedCard]:
    """Main entry: CardState → list[RenderedCard].

    Pipeline: flatten_to_atoms → paginate → assemble pages.
    Each page is a complete Feishu Schema 2.0 card JSON.
    """
    if budget.engine_cmd == "命令" or budget.engine_cmd == "对应命令":
        # Inject engine_cmd from state if caller didn't set it explicitly
        engine_cmd = _engine_type_to_cmd(state.metadata.engine_type if state.metadata else None)
        if engine_cmd != "命令" and engine_cmd != "对应命令":
            from dataclasses import replace
            budget = replace(budget, engine_cmd=engine_cmd)

    # Use pre-built block_index from CardState (dict[str, int] → O(1) lookup)
    # to construct block references on demand, avoiding O(n) dict rebuild per render call.
    block_index: dict[str, ContentBlock] = {
        bid: state.blocks[idx] for bid, idx in state.block_index.items()
    }

    # 1. Flatten blocks to atoms and build SectionLayout skeleton.
    is_programming_card = state.metadata.engine_type is None
    segmented_text = bool(
        state.metadata.programming_text_sections
        and is_programming_card
    )
    atoms = flatten_to_atoms(
        state.blocks,
        budget,
        unified_execution=(
            is_programming_card
            and not state.metadata.full_execution_blocks
        ),
        terminal=state.terminal != "running",
        segmented_text=segmented_text,
        full_execution_blocks=state.metadata.full_execution_blocks,
    )
    if state.metadata and state.metadata.compact:
        atoms = _compact_reasoning_atoms(atoms)
    subagent_atom = build_subagent_dispatch_atom(list(state.metadata.subagents)) if state.metadata.subagents else None
    if subagent_atom is not None:
        atoms.insert(0, subagent_atom)
    atoms = _coalesce_adjacent_text_fragments(atoms, block_index)
    layout = _build_section_layout(state, atoms)

    # 2. Paginate via SectionLayout (sticky_head/status/body/appendix SSOT)
    pages = paginate_layout(layout, budget)
    total_pages = len(pages)

    # 3. Compute global structure signature (for cache key / fast structural-change detection)
    global_sig = compute_structure_signature(state)
    content_hash = compute_content_hash(state)

    # 4. Assemble each page
    results: list[RenderedCard] = []
    for page_idx, page_atoms in enumerate(pages):
        # Render body elements from atoms
        body_elements = _render_atoms_to_elements(page_atoms, state, budget, block_index)

        # Promote warning banner to top of FIRST PAGE ONLY
        # (reduces JSON size on multi-page cards; users rarely view non-first pages)
        if page_idx == 0 and state.footer.warning_banner and state.footer.warning_type:
            bg_style, icon = _BANNER_STYLES.get(state.footer.warning_type, ("grey", "ℹ️"))
            banner_text = f"{icon} **{state.footer.warning_banner}**"
            top_banner = _build_column_banner(content=banner_text, background_style=bg_style)
            body_elements.insert(0, top_banner)

        # Non-first pages: styled warning banner (consistent sizing) so users on any page can see it
        if page_idx > 0 and state.footer.warning_banner:
            bg_style, icon = _BANNER_STYLES.get(state.footer.warning_type or "warning", ("grey", "ℹ️"))
            warning_note = _build_column_banner(
                content=f"{icon} **{state.footer.warning_banner}**",
                background_style=bg_style,
            )
            body_elements.insert(0, warning_note)

        # Append footer and buttons only on the last page
        live_elapsed_footer_element_ids: set[int] = set()
        if page_idx == total_pages - 1:
            footer_elements = render_footer(state, budget=budget)
            live_elapsed_footer_element_ids = {
                id(element)
                for element in footer_elements
                if (
                    element.get("tag") == "markdown"
                    and "⏱ 用时 " in str(element.get("content") or "")
                )
            }
            body_elements.extend(footer_elements)
            body_elements.extend(render_buttons(state, budget=budget))

        # Detect active element for streaming
        active_element = _find_active_element(page_atoms, block_index)

        # Determine streaming mode. Feishu CardKit's official element content
        # API streams full text into a markdown/plain_text element identified by
        # element_id, regardless of the rest of the card structure.
        is_running = state.terminal == "running"
        streaming = active_element is not None and is_running
        if not streaming:
            active_element = None
            _strip_streaming_element_ids(body_elements)

        # Build the full card JSON
        card_json = _assemble_card_json(
            state=state,
            body_elements=body_elements,
            streaming=streaming,
            active_element=active_element,
            page_index=page_idx,
            total_pages=total_pages,
        )

        # Post-render node count check (early warning for regressions)
        from src.card.shared.truncation import count_tagged_nodes
        node_count = count_tagged_nodes(card_json)
        if node_count > 200:
            logger.warning(
                "Rendered card page %d has %d nodes (exceeds Feishu 200-element limit), "
                "payload_truncator will attempt truncation at delivery layer",
                page_idx, node_count,
            )

        # Compute per-page structure signature from body content
        # This ensures only pages with actual changes trigger API updates
        page_sig_parts = [global_sig, f"page:{page_idx}"]
        streaming_element_id = (
            active_element.element_id
            if active_element is not None
            else None
        )
        for elem in body_elements:
            _append_element_structure_signature(
                page_sig_parts,
                elem,
                streaming_element_id=streaming_element_id,
                ignore_root_content=(
                    active_element is not None
                    and id(elem) in live_elapsed_footer_element_ids
                ),
            )
        page_signature = hashlib.md5(
            utf8_replace_bytes("|".join(page_sig_parts))
        ).hexdigest()

        results.append(
            RenderedCard(
                _card_json=card_json,
                structure_signature=page_signature,
                content_hash=content_hash,
                active_element=active_element,
                page_index=page_idx,
                total_pages=total_pages,
            )
        )

    return results


# Thread-safe structure signature cache using lru_cache (immutable args → safe under GIL)
# Avoids recomputing MD5 for pure text_delta events


@functools.lru_cache(maxsize=64)
def _compute_sig_cached(sv: int, parts_key: str) -> str:
    """Cached MD5 computation keyed on (structural_version, parts_key)."""
    return hashlib.md5(utf8_replace_bytes(parts_key)).hexdigest()


def compute_structure_signature(state: CardState) -> str:
    """Compute MD5 of structural parts of the card.

    Structural = block kinds + statuses + tool names + image render fields
    + terminal state + header + buttons.
    Excludes: active text content (streamed via element_content), progress_pct,
    criteria counts, and warning_banner (tracked separately in content_hash).
    This allows the delivery layer to skip full card updates when only text changed.

    Uses structural_version as cache key — returns cached value when version unchanged.
    """
    parts: list[str] = []
    for block in state.blocks:
        parts.append(f"{block.kind}:{block.block_id}:{block.status}")
        if block.kind == "tool_call":
            parts.append(f"tn:{block.tool_name}")
        elif block.kind == "text":
            parts.append(
                "text-source:"
                f"{getattr(block, 'source_kind', 'main')}:"
                f"{getattr(block, 'source_sequence', '') or ''}:"
                f"{getattr(block, 'source_ref', 'main')}"
            )
        elif block.kind == "image":
            parts.append(f"image_key:{block.image_key or ''}")
            parts.append(f"image_alt:{block.alt}")
    parts.append(f"terminal:{state.terminal}")
    parts.append(f"header:{state.header.title}:{state.header.template}")
    if state.header.subtitle is not None:
        parts.append(f"sub:{state.header.subtitle}")
    if state.metadata.tool_name:
        parts.append(f"tool:{state.metadata.tool_name}")
    if state.metadata.model_name:
        parts.append(f"model:{state.metadata.model_name}")
    if state.metadata.programming_text_sections:
        parts.append("programming-text-sections:v1")
    if state.metadata.full_execution_blocks:
        parts.append("full-execution-blocks:v1")
    if state.metadata.iteration_index:
        parts.append(f"iter:{state.metadata.iteration_index}/{state.metadata.iteration_total or ''}")
    if state.metadata.unit_label:
        parts.append(f"unit:{state.metadata.unit_kind or ''}:{state.metadata.unit_id or ''}:{state.metadata.unit_label}")
    if state.metadata.card_sequence != 1:
        parts.append(f"seq:{state.metadata.card_sequence}")
    if state.metadata.live_ticker_frame:
        parts.append(f"ticker:{state.metadata.live_ticker_frame}")
    if state.metadata.subagents:
        for item in state.metadata.subagents:
            parts.append(
                "subagent:"
                f"{item.get('sequence') or item.get('card_sequence') or ''}:"
                f"{item.get('status') or ''}:"
                f"{item.get('label') or item.get('name') or ''}"
            )

    if state.metadata.bridge_phrase:
        parts.append(f"bridge:{state.metadata.bridge_phrase}")
    if state.footer.status is not None:
        parts.append(f"footer:{state.footer.status}")
    # Add buttons to structure signature since button changes should trigger full updates
    for button in state.buttons:
        parts.append(f"button:{button.action_id}:{button.type}:{'disabled' if button.disabled else 'enabled'}")
    if state.engine_ext and state.engine_ext.phase_info:
        parts.append(f"phase:{state.engine_ext.phase_info}")

    parts_key = "|".join(parts)
    sv = state.structural_version

    return _compute_sig_cached(sv, parts_key)


def compute_content_hash(state: CardState) -> str:
    """Compute MD5 of frequently-changing content fields.

    These fields (progress_pct, criteria counts, warning_banner) change often
    but don't alter card structure. Used to decide if a content-only patch is needed.
    """
    parts: list[str] = []
    if state.footer.progress_pct is not None:
        parts.append(f"pct:{state.footer.progress_pct}")
    if state.engine_ext and state.engine_ext.criteria_section:
        parts.append(f"criteria:{state.engine_ext.criteria_satisfied}/{state.engine_ext.criteria_total}")
    if state.footer.warning_banner:
        parts.append(f"warn:{state.footer.warning_banner}")
    if state.metadata.bridge_phrase:
        parts.append(f"bridge:{state.metadata.bridge_phrase}")
    if not parts:
        return ""
    raw = "|".join(parts)
    return hashlib.md5(utf8_replace_bytes(raw)).hexdigest()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# --- Individual atom renderer functions (signature: atom, state, budget, block_index → dict|None) ---

def _render_atom_text(atom: RenderAtom, state: CardState, budget: RenderBudget, block_index: dict) -> dict:
    if (
        atom.block_id == "_error"
        and state.terminal == "failed"
        and state.metadata.is_subagent
    ):
        block = block_index.get(atom.block_id)
        return render_collapsed_error_panel(
            detail_content=_normalize_text_markdown(atom.content),
            full_content=str(getattr(block, "content", "") or atom.content),
        )
    body = _render_text_element(atom, block_index)
    block = block_index.get(atom.block_id)
    if (
        state.metadata.programming_text_sections
        and state.metadata.engine_type is None
        and block is not None
        and block.kind == "text"
        and is_programming_thought_block(block)
    ):
        return render_programming_text_section(
            body_markdown=body,
            block=block,
            state=state,
        )
    return body


def _render_atom_tool_panel(atom: RenderAtom, state: CardState, budget: RenderBudget, block_index: dict) -> dict | None:
    block = block_index.get(atom.block_id)
    if state.metadata.full_execution_blocks and block is not None:
        return render_tool_panel(block, content_override=atom.content)
    if atom.elements:
        return atom.elements[0]
    # In the new activity_digest flow, tool_panel atoms are only emitted for
    # active (running) tools with pre-rendered compact content.
    # Keep compact lines at normal size; mobile Feishu can render notation too small.
    if atom.content:
        return _build_column_banner(content=atom.content, background_style="wathet")
    if block is not None:
        return render_tool_panel(block)
    return None


def _render_atom_subagent_dispatch(atom: RenderAtom, state: CardState, budget: RenderBudget, block_index: dict) -> dict | None:
    if atom.elements:
        return atom.elements[0]
    return None


def _render_atom_reasoning(atom: RenderAtom, state: CardState, budget: RenderBudget, block_index: dict) -> dict:
    block = block_index.get(atom.block_id)
    if block is not None:
        # Use atom.content (per-block correct content) instead of block.content.
        # block_index maps shared block_ids (e.g. "_active_reasoning") to the LAST
        # block, so without override all atoms would render the last block's content.
        compact = bool(state.metadata and state.metadata.compact)
        return render_reasoning_panel(block, budget, content_override=atom.content, compact=compact)
    return {"tag": "markdown", "content": atom.content}


def _render_atom_plan(atom: RenderAtom, state: CardState, budget: RenderBudget, block_index: dict) -> dict:
    block = block_index.get(atom.block_id)
    if block is not None:
        phase = state.footer.status if state.footer.status else "running"
        return render_plan_panel(block, budget=budget, phase=phase, content_override=atom.content)
    return {"tag": "markdown", "content": atom.content}


def _render_atom_criteria(atom: RenderAtom, state: CardState, budget: RenderBudget, block_index: dict) -> dict:
    return _render_criteria_panel(atom, state)


def _render_atom_phase(atom: RenderAtom, state: CardState, budget: RenderBudget, block_index: dict) -> dict:
    return _render_phase_panel(atom)


def _render_atom_warning_banner(atom: RenderAtom, state: CardState, budget: RenderBudget, block_index: dict) -> dict:
    return {"tag": "markdown", "content": atom.content}


def _render_atom_progress_bar(atom: RenderAtom, state: CardState, budget: RenderBudget, block_index: dict) -> dict:
    return {"tag": "markdown", "content": atom.content}


def _render_atom_phase_banner(atom: RenderAtom, state: CardState, budget: RenderBudget, block_index: dict) -> dict:
    """Render phase_banner as a top sticky markdown line."""
    return {"tag": "markdown", "content": atom.content}


def _render_atom_task_list(atom: RenderAtom, state: CardState, budget: RenderBudget, block_index: dict) -> dict | None:
    """Look up the ContentBlock for this atom and delegate to render_task_list_panel."""
    from src.card.render.task_list import render_task_list_panel
    if atom.elements:
        return atom.elements[0]
    block = block_index.get(atom.block_id)
    if block is None:
        return {"tag": "markdown", "content": atom.content}
    return render_task_list_panel(
        block,
        compact=bool(state.metadata.is_subagent),
    )


def _render_atom_review_role(atom: RenderAtom, state: CardState, budget: RenderBudget, block_index: dict) -> dict | None:
    block = block_index.get(atom.block_id)
    if block is None:
        return {"tag": "markdown", "content": atom.content}
    return render_review_role_panel(block)


def _render_atom_spec_plan(atom: RenderAtom, state: CardState, budget: RenderBudget, block_index: dict) -> dict | None:
    block = block_index.get(atom.block_id)
    if block is None:
        return {"tag": "markdown", "content": atom.content}
    return render_spec_plan_panel(block)


def _render_atom_spec_task(atom: RenderAtom, state: CardState, budget: RenderBudget, block_index: dict) -> dict | None:
    if atom.elements:
        return atom.elements[0]
    block = block_index.get(atom.block_id)
    if block is None:
        return {"tag": "markdown", "content": atom.content}
    return render_spec_task_panel(block)


def _render_atom_activity_digest(atom: RenderAtom, state: CardState, budget: RenderBudget, block_index: dict) -> dict:
    """Render activity digest as a compact, mobile-readable markdown line."""
    if atom.elements:
        return atom.elements[0]
    return {"tag": "markdown", "content": atom.content, "text_size": "normal"}


def _render_atom_execution_flow(atom: RenderAtom, state: CardState, budget: RenderBudget, block_index: dict) -> dict:
    """Render a pre-built current-action row or execution-history panel."""
    if atom.elements:
        return atom.elements[0]
    return {"tag": "markdown", "content": atom.content, "text_size": "normal"}


def _render_atom_image(atom: RenderAtom, state: CardState, budget: RenderBudget, block_index: dict) -> dict:
    if atom.elements:
        return atom.elements[0]
    return {
        "tag": "markdown",
        "content": "🖼️ 图片产物暂时无法展示；文本结果与本地文件不受影响。",
        "text_size": "notation",
    }


# Atom renderer registry: maps atom.kind → renderer function.
# To add a new atom kind, define a function with the standard signature and register it here.
_ATOM_RENDERERS: dict[str, Callable[[RenderAtom, CardState, RenderBudget, dict], dict | None]] = {
    "text": _render_atom_text,
    "tool_panel": _render_atom_tool_panel,
    "subagent_dispatch": _render_atom_subagent_dispatch,
    "reasoning": _render_atom_reasoning,
    "plan": _render_atom_plan,
    "criteria_panel": _render_atom_criteria,
    "phase_panel": _render_atom_phase,
    "warning_banner": _render_atom_warning_banner,
    "progress_bar": _render_atom_progress_bar,
    "phase_banner": _render_atom_phase_banner,
    "task_list": _render_atom_task_list,
    "activity_digest": _render_atom_activity_digest,
    "review_role": _render_atom_review_role,
    "spec_plan": _render_atom_spec_plan,
    "spec_task": _render_atom_spec_task,
    "execution_current": _render_atom_execution_flow,
    "execution_history": _render_atom_execution_flow,
    "image": _render_atom_image,
}

_atom_kind_literals = set(_get_args(AtomKind))
_atom_renderer_keys = set(_ATOM_RENDERERS.keys())
if _atom_kind_literals != _atom_renderer_keys:
    raise RuntimeError(
        f"AtomKind and _ATOM_RENDERERS mismatch: "
        f"AtomKind={_atom_kind_literals}, registry={_atom_renderer_keys}"
    )
del _atom_kind_literals, _atom_renderer_keys


def _render_atoms_to_elements(
    atoms: list[RenderAtom], state: CardState, budget: RenderBudget,
    block_index: dict[str, ContentBlock],
) -> list[dict]:
    """Convert RenderAtoms to Feishu Schema 2.0 elements using the atom renderer registry."""
    elements: list[dict] = []
    bridge_phrase = (state.metadata.bridge_phrase or "").strip() if state.metadata else ""
    bridge_injected = False

    for atom in atoms:
        renderer = _ATOM_RENDERERS.get(atom.kind)
        if renderer is not None:
            el = renderer(atom, state, budget, block_index)
            if el is not None:
                if bridge_phrase and not bridge_injected and atom.kind in {
                    "text",
                    "reasoning",
                    "execution_current",
                }:
                    if _prepend_bridge_phrase(el, bridge_phrase):
                        bridge_injected = True
                elements.append(el)
        else:
            logger.warning("Unknown atom kind '%s', rendering as placeholder", atom.kind)
            # Use state-aware placeholder: running vs terminal
            if state.terminal == "running":
                fallback_text = UI_TEXT["card_content_load_error_running"]
            else:
                engine_cmd = _engine_type_to_cmd(state.metadata.engine_type if state.metadata else None)
                fallback_text = UI_TEXT["card_content_load_error"].format(engine_cmd=engine_cmd)
            elements.append({"tag": "markdown", "content": fallback_text})

    return elements


def _compact_reasoning_atoms(atoms: list[RenderAtom]) -> list[RenderAtom]:
    """Apply compact-mode reasoning truncation before pagination."""
    compacted: list[RenderAtom] = []
    for atom in atoms:
        if atom.kind != "reasoning":
            compacted.append(atom)
            continue
        compact_content = truncate_reasoning_for_compact(atom.content)
        compact_atom = RenderAtom(
            kind=atom.kind,
            elements=atom.elements,
            block_id=atom.block_id,
            content=compact_content,
            splittable=False,
            node_count=atom.node_count,
            structural_overhead=atom.structural_overhead,
        )
        compact_atom.byte_size = estimate_atom_size(compact_atom)
        compacted.append(compact_atom)
    return compacted


def _coalesce_adjacent_text_fragments(
    atoms: list[RenderAtom],
    block_index: dict[str, ContentBlock],
) -> list[RenderAtom]:
    """Merge pathological one-character text fragments into the next text atom.

    ACP streams can briefly split a Chinese word across text block boundaries
    (for example ``数`` + ``字很大``). Rendering the first block as a standalone
    markdown element makes Feishu show a single character on its own line. Keep
    normal paragraph blocks separate, but stitch a leading one-character
    fragment into the following adjacent text atom.
    """
    if len(atoms) < 2:
        return atoms

    merged: list[RenderAtom] = []
    for atom in atoms:
        if (
            atom.kind == "text"
            and merged
            and merged[-1].kind == "text"
            and _visible_text_len(merged[-1].content) == 1
            and atom.content
            and not atom.content[0].isspace()
            and _is_programming_thought_atom(merged[-1], block_index)
            and _is_programming_thought_atom(atom, block_index)
            and _text_source_ref(merged[-1], block_index)
            == _text_source_ref(atom, block_index)
        ):
            previous = merged.pop()
            content = soft_join_text_fragments(previous.content, atom.content)
            if content is None:
                merged.append(previous)
                merged.append(atom)
                continue
            stitched = RenderAtom(
                kind="text",
                block_id=atom.block_id or previous.block_id,
                content=content,
                splittable=previous.splittable or atom.splittable,
                node_count=max(previous.node_count, atom.node_count),
                structural_overhead=max(
                    previous.structural_overhead,
                    atom.structural_overhead,
                ),
            )
            stitched.byte_size = estimate_atom_size(stitched)
            merged.append(stitched)
            continue
        merged.append(atom)
    return merged


def _is_programming_thought_atom(
    atom: RenderAtom,
    block_index: dict[str, ContentBlock],
) -> bool:
    block = block_index.get(atom.block_id)
    return bool(
        block is not None
        and block.kind == "text"
        and is_programming_thought_block(block)
    )


def _text_source_ref(
    atom: RenderAtom,
    block_index: dict[str, ContentBlock],
) -> str:
    block = block_index.get(atom.block_id)
    if block is None or block.kind != "text":
        return f"block:{atom.block_id}"
    return str(getattr(block, "source_ref", "main") or "main")


def _visible_text_len(text: str) -> int:
    return len("".join(str(text or "").split()))


def _prepend_bridge_phrase(element: dict, phrase: str) -> bool:
    """Prepend bridge phrase to the first markdown content inside an element."""
    if element.get("tag") == "markdown":
        content = str(element.get("content", ""))
        element["content"] = f"{phrase}\n\n{content}" if content else phrase
        return True
    # Recurse into elements (div, collapsible_panel) and columns (column_set)
    for child in element.get("elements", []) or []:
        if isinstance(child, dict) and _prepend_bridge_phrase(child, phrase):
            return True
    for col in element.get("columns", []) or []:
        if isinstance(col, dict) and _prepend_bridge_phrase(col, phrase):
            return True
    return False


def _sanitize_schema_v2_node(node) -> None:
    """Remove fields known to be rejected on specific Feishu Schema 2.0 nodes."""
    if isinstance(node, dict):
        if node.get("tag") == "collapsible_panel":
            node.pop("background_style", None)
        for value in node.values():
            _sanitize_schema_v2_node(value)
    elif isinstance(node, list):
        for item in node:
            _sanitize_schema_v2_node(item)


def _order_atoms_by_section(atoms: list[RenderAtom]) -> list[RenderAtom]:
    """Render sections in stable order: status → body.

    Preserve relative order inside each section so streaming updates remain stable.
    Unknown atoms stay in body section by default to avoid dropping content.
    Within status section, task_list always comes first.
    """
    status_atoms: list[RenderAtom] = []
    body_atoms: list[RenderAtom] = []

    for atom in atoms:
        if atom.kind in _STATUS_ATOM_KINDS:
            status_atoms.append(atom)
        elif atom.kind in _BODY_ATOM_KINDS:
            body_atoms.append(atom)
        else:
            body_atoms.append(atom)

    # Ensure task_list atoms always appear first in status section
    task_list_atoms = [a for a in status_atoms if a.kind == "task_list"]
    other_status = [a for a in status_atoms if a.kind != "task_list"]
    status_atoms = [*task_list_atoms, *other_status]

    return [*status_atoms, *body_atoms]


def _build_section_layout(state: CardState, atoms: list[RenderAtom]) -> SectionLayout:
    """Convert flattened atoms into the SectionLayout pagination contract."""
    ordered = _order_atoms_by_section(atoms)
    sticky_head = build_sticky_head(state, state.metadata)
    sticky_block_ids = {atom.block_id for atom in sticky_head if atom.block_id}
    sticky_kinds = {atom.kind for atom in sticky_head}

    status_atoms: list[RenderAtom] = []
    body_atoms: list[RenderAtom] = []

    for atom in ordered:
        if atom.block_id in sticky_block_ids:
            continue
        if atom.kind == "task_list" and atom.kind in sticky_kinds:
            continue
        if atom.kind in _STATUS_ATOM_KINDS:
            status_atoms.append(atom)
        else:
            body_atoms.append(atom)

    if (
        state.metadata.engine_type is None
        and not state.metadata.full_execution_blocks
    ):
        execution_atoms = [
            atom
            for atom in body_atoms
            if atom.kind in {"execution_current", "execution_history"}
        ]
        other_body_atoms = [
            atom
            for atom in body_atoms
            if atom.kind not in {"execution_current", "execution_history"}
        ]
        if state.terminal == "running":
            body_atoms = [*execution_atoms, *other_body_atoms]
        else:
            answer_atoms = [
                atom for atom in other_body_atoms if atom.kind in {"text", "image"}
            ]
            remaining_atoms = [
                atom for atom in other_body_atoms if atom.kind not in {"text", "image"}
            ]
            body_atoms = [*answer_atoms, *remaining_atoms, *execution_atoms]

    return SectionLayout(
        sticky_head=sticky_head,
        status=tuple(status_atoms),
        body=tuple(body_atoms),
        appendix=(),
    )


def _render_text_element(atom: RenderAtom, block_index: dict[str, ContentBlock]) -> dict:
    """Render a text atom. If it's the active block, assign element_id for streaming.

    When content is empty, element_id is intentionally omitted — Feishu CardKit
    enters streaming mode on an empty element and renders the first character on
    a new line when content is later pushed via update_element().
    """
    block = block_index.get(atom.block_id)
    element_id = None
    if block is not None and block.element_id and block.status == "active":
        element_id = block.element_id

    content = _normalize_text_markdown(atom.content)
    if element_id:
        content = _stabilize_active_markdown(content)
    el: dict = {"tag": "markdown", "content": content}
    # Only assign element_id when content is non-empty to prevent Feishu CardKit
    # from entering streaming state with an empty element (causes first-char newline).
    if element_id and _is_streaming_text_ready(content):
        el["element_id"] = element_id
    return el


def _find_active_element(
    atoms: list[RenderAtom], block_index: dict[str, ContentBlock],
) -> ActiveElement | None:
    """Find the active streaming text element on this page.

    Returns None when the active text block is empty — this prevents Feishu
    CardKit from entering streaming mode with an empty element, which causes
    the first character to appear on a new line when content is later pushed
    via update_element().
    """
    for atom in atoms:
        if atom.kind != "text":
            continue
        block = block_index.get(atom.block_id)
        if block is not None and block.status == "active" and block.element_id:
            content = _stabilize_active_markdown(_normalize_text_markdown(atom.content))
            # Skip empty content: don't activate streaming until real text exists
            if not _is_streaming_text_ready(content):
                continue
            return ActiveElement(element_id=block.element_id, text=content)
    return None


def _normalize_text_markdown(content: str) -> str:
    """Repair common streamed fence shapes before Feishu markdown parses them."""
    if not content:
        return content
    lines = str(content).splitlines(keepends=True)
    return "".join(_normalize_fence_line(line) for line in lines)


def _stabilize_active_markdown(content: str) -> str:
    """Close in-flight markdown delimiters so partial streaming frames render locally.

    The engine may stream a markdown token before its closing delimiter. Feishu
    reparses every element_content update, so an unclosed code fence or inline
    code span makes all following text look like code until a later frame
    happens to close it. Only active streaming text gets this presentation guard;
    completed blocks keep the exact model output.
    """
    if not content:
        return content

    fence = _open_markdown_fence(content)
    if fence:
        suffix = "" if content.endswith("\n") else "\n"
        return f"{content}{suffix}{fence}"

    inline_tick = _last_unclosed_inline_code_tick(content)
    if inline_tick:
        return f"{content}{inline_tick}"

    return content


def _open_markdown_fence(content: str) -> str:
    open_fence = ""
    in_fence = False
    for raw_line in str(content).splitlines():
        line = raw_line.lstrip()
        match = _FENCE_LINE_RE.match(line)
        if not match:
            continue
        fence = match.group("fence")
        if in_fence and open_fence and fence.startswith(open_fence[0] * min(len(open_fence), len(fence))):
            in_fence = False
            open_fence = ""
            continue
        if not in_fence:
            in_fence = True
            open_fence = fence[:3]
    return open_fence if in_fence else ""


def _last_unclosed_inline_code_tick(content: str) -> str:
    last_unclosed = ""
    in_fence = False
    for raw_line in str(content).splitlines():
        line = raw_line.lstrip()
        match = _FENCE_LINE_RE.match(line)
        if match:
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for tick in _iter_unescaped_inline_backtick_runs(raw_line):
            last_unclosed = "" if last_unclosed == tick else tick
    return last_unclosed


def _iter_unescaped_inline_backtick_runs(text: str):
    i = 0
    while i < len(text):
        if text[i] != "`":
            i += 1
            continue
        escaped = i > 0 and text[i - 1] == "\\"
        j = i
        while j < len(text) and text[j] == "`":
            j += 1
        run = text[i:j]
        if not escaped and len(run) < 3:
            yield run
        i = j


def _normalize_fence_line(line: str) -> str:
    line_body = line[:-1] if line.endswith("\n") else line
    newline = "\n" if line.endswith("\n") else ""
    match = _FENCE_LINE_RE.match(line_body)
    if not match:
        return line

    info = (match.group("info") or "").strip()
    fence = match.group("fence")
    escaped = bool(match.group("escaped"))
    dirty_info = "`" in info or "~" in info
    if not escaped and not dirty_info:
        return line

    language = _extract_fence_language(info) if info else ""
    if dirty_info and not language:
        language = "text"
    return f"{match.group('indent')}{fence}{language}{newline}"


def _extract_fence_language(info: str) -> str:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_+#.+-]*", info)
    for token in reversed(tokens):
        normalized = _FENCE_LANGUAGE_ALIASES.get(token.lower())
        if normalized:
            return normalized
    return ""


def _is_streaming_text_ready(content: str) -> bool:
    return _visible_text_len(content) >= _MIN_STREAMING_TEXT_CHARS


def _strip_streaming_element_ids(nodes: list[dict]) -> None:
    """Remove element_id markers when a card will be updated via full PATCH."""
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node.pop("element_id", None)
        child_elements = node.get("elements")
        if isinstance(child_elements, list):
            _strip_streaming_element_ids(child_elements)
        columns = node.get("columns")
        if isinstance(columns, list):
            _strip_streaming_element_ids(columns)
        actions = node.get("actions")
        if isinstance(actions, list):
            _strip_streaming_element_ids(actions)


def _content_signature(content: object) -> str:
    return hashlib.md5(utf8_replace_bytes(content)).hexdigest()


def _append_element_structure_signature(
    parts: list[str],
    node: object,
    *,
    streaming_element_id: str | None,
    ignore_root_content: bool = False,
) -> None:
    """Recursively fingerprint Card 2.0 structure and static text.

    Only the selected streaming element excludes its content; CardKit can
    deliver that element through the element-content API. Other active text
    elements must retain a content fingerprint so concurrent source updates
    trigger a full card patch instead of being silently skipped.

    While an element is streaming, the live elapsed footer is also excluded
    from the signature. An element-content call cannot update that separate
    footer node without a second API call, so the next existing full-card
    update refreshes the clock instead of degrading every text chunk to PATCH.
    """
    if isinstance(node, list):
        for item in node:
            _append_element_structure_signature(
                parts,
                item,
                streaming_element_id=streaming_element_id,
            )
        return
    if not isinstance(node, dict):
        return

    tag = str(node.get("tag") or "")
    if tag:
        parts.append(tag)
    if tag in {"markdown", "plain_text"}:
        element_id = str(node.get("element_id") or "")
        if element_id:
            parts.append(f"element_id:{element_id}")
            if (
                not ignore_root_content
                and element_id != streaming_element_id
            ):
                parts.append(
                    _content_signature(node.get("content", ""))
                )
        elif not ignore_root_content:
            parts.append(_content_signature(node.get("content", "")))

    for key, value in node.items():
        if key in {"tag", "content", "element_id"}:
            continue
        if isinstance(value, (dict, list)):
            _append_element_structure_signature(
                parts,
                value,
                streaming_element_id=streaming_element_id,
            )


def _assemble_card_json(
    state: CardState,
    body_elements: list[dict],
    streaming: bool,
    active_element: ActiveElement | None,
    page_index: int = 0,
    total_pages: int = 1,
) -> dict:
    """Assemble a complete Feishu Schema 2.0 card JSON."""
    _sanitize_schema_v2_node(body_elements)
    card: dict = {
        "schema": "2.0",
        "config": {
            "wide_screen_mode": True,
            "update_multi": True,
        },
        "header": render_header(state, page_index=page_index, total_pages=total_pages),
        "body": {"elements": body_elements},
    }

    if streaming:
        card["config"]["streaming_mode"] = True

    return card


def _render_criteria_panel(atom: RenderAtom, state: CardState) -> dict:
    """Render criteria section as a collapsible markdown panel with inline count."""
    # Inline count from engine_ext if available
    if state.engine_ext and state.engine_ext.criteria_total > 0:
        header_text = UI_TEXT["criteria_panel_header_with_count"].format(
            satisfied=state.engine_ext.criteria_satisfied,
            total=state.engine_ext.criteria_total,
        )
        # Expand when criteria exist but not all satisfied; collapse when all pass
        expanded = state.engine_ext.criteria_satisfied < state.engine_ext.criteria_total
    else:
        header_text = UI_TEXT["criteria_panel_header"]
        expanded = state.metadata.expand_ac
    return {
        "tag": "collapsible_panel",
        "expanded": expanded,
        "header": {
            "title": {"tag": "markdown", "content": header_text},
            "icon": {
                "tag": "standard_icon",
                "token": "down-small-ccm_outlined",
                "size": "16px 16px",
            },
            "icon_position": "follow_text",
            "icon_expanded_angle": -180,
        },
        "border": {"color": "wathet", "corner_radius": PANEL_STYLES["corner_radius"]},
        "vertical_spacing": "8px",
        "padding": PANEL_STYLES["padding_standard"],
        "elements": [{"tag": "markdown", "content": atom.content}],
    }


def _render_phase_panel(atom: RenderAtom) -> dict:
    """Render phase info as a visually distinct block with wathet background."""
    # Avoid unsupported div styling fields (padding/background_style).
    return _build_column_banner(content=f"⚙️ {atom.content}", background_style="wathet")
