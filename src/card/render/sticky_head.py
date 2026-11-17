"""Sticky-head builder: phase banner plus state-appropriate task progress."""

from __future__ import annotations

from src.card.render.atoms import RenderAtom, estimate_atom_size
from src.card.render.banner_computer import compute_banner
from src.card.render.task_list import render_task_list_panel
from src.card.state.models import CardMetadata, CardState
from src.card.state.runtime_stats import RuntimeStats

STICKY_HEAD_MAX_NODES = 25


def build_sticky_head(
    state: CardState,
    metadata: CardMetadata,
    *,
    _force_total_nodes: int | None = None,
) -> tuple[RenderAtom, ...]:
    """Build the sandwich anchor reused on every page.

    Returns phase_banner, then optional task_list. When the sticky head exceeds
    the hard node cap, right-most optional atoms are dropped.
    """
    atoms: list[RenderAtom] = []

    runtime = getattr(state, "runtime_stats", None) or RuntimeStats(elapsed_seconds=0.0)
    banner_text = compute_banner(metadata, runtime)
    if banner_text:
        banner_atom = RenderAtom(
            kind="phase_banner",
            content=banner_text,
            node_count=1,
            block_id="_phase_banner",
        )
        banner_atom.byte_size = estimate_atom_size(banner_atom)
        atoms.append(banner_atom)

    task_list = getattr(state, "task_list", None)
    if task_list is None:
        task_list = next((b for b in getattr(state, "blocks", ()) if getattr(b, "kind", "") == "task_list"), None)
    if task_list is not None and getattr(task_list, "tasks", None):
        panel = render_task_list_panel(task_list)
        if panel is not None:
            tl_atom = RenderAtom(
                kind="task_list",
                elements=[panel],
                node_count=8,
                block_id="_sticky_task_list",
                content="",
            )
            tl_atom.byte_size = estimate_atom_size(tl_atom)
            atoms.append(tl_atom)

    total = _force_total_nodes if _force_total_nodes is not None else sum(a.node_count for a in atoms)
    while total > STICKY_HEAD_MAX_NODES and len(atoms) > 1:
        atoms.pop()
        total = sum(a.node_count for a in atoms)

    return tuple(atoms)
