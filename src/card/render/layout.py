"""SectionLayout: SSOT for card section ordering and pagination contract."""

from __future__ import annotations

from dataclasses import dataclass

from src.card.render.atoms import RenderAtom, estimate_atom_size
from src.card.render.budget import RenderBudget
from src.card.render.pagination import BASE_OVERHEAD, FIXED_NODE_OVERHEAD, split_atom


@dataclass(frozen=True)
class SectionLayout:
    """Single source of truth for card section ordering and pagination.

    sticky_head: repeated on every page, never moved by pagination.
    status:      first page only; secondary status panels (progress, criteria).
    body:        primary content; subject to greedy pagination.
    appendix:    last page only; reserved for future use.
    """

    sticky_head: tuple[RenderAtom, ...]
    status: tuple[RenderAtom, ...]
    body: tuple[RenderAtom, ...]
    appendix: tuple[RenderAtom, ...]

    def assemble_for_page(
        self,
        page_idx: int,
        total_pages: int,
        body_slice: tuple[RenderAtom, ...],
    ) -> tuple[RenderAtom, ...]:
        """Build full atom sequence for one page."""
        result: list[RenderAtom] = list(self.sticky_head)
        if page_idx == 0:
            result.extend(self.status)
        result.extend(body_slice)
        if page_idx == total_pages - 1:
            result.extend(self.appendix)
        return tuple(result)


def paginate_layout(layout: SectionLayout, budget: RenderBudget) -> list[tuple[RenderAtom, ...]]:
    """Paginate body atoms with sticky_head reserved on every page."""
    sticky_size = sum(_atom_size(a) for a in layout.sticky_head)
    sticky_nodes = sum(a.node_count for a in layout.sticky_head)
    status_size = sum(_atom_size(a) for a in layout.status)
    status_nodes = sum(a.node_count for a in layout.status)

    base_byte = budget.byte_budget - BASE_OVERHEAD - sticky_size
    base_node = budget.node_budget - FIXED_NODE_OVERHEAD - sticky_nodes

    body_pages: list[list[RenderAtom]] = [[]]
    cur_bytes = 0
    cur_nodes = 0
    is_first_page = True

    def remaining_byte() -> int:
        extra = status_size if is_first_page else 0
        return base_byte - extra - cur_bytes

    def remaining_node() -> int:
        extra = status_nodes if is_first_page else 0
        return base_node - extra - cur_nodes

    def start_new_page() -> None:
        nonlocal is_first_page, cur_bytes, cur_nodes
        body_pages.append([])
        is_first_page = False
        cur_bytes = 0
        cur_nodes = 0

    def append_atom(atom: RenderAtom) -> None:
        nonlocal cur_bytes, cur_nodes
        body_pages[-1].append(atom)
        cur_bytes += _atom_size(atom)
        cur_nodes += atom.node_count

    def fits(atom: RenderAtom) -> bool:
        return (
            _atom_size(atom) <= remaining_byte()
            and atom.node_count <= remaining_node()
        )

    for original_atom in layout.body:
        atom = original_atom
        while True:
            if fits(atom):
                append_atom(atom)
                break

            split_result = None
            if atom.node_count <= remaining_node():
                split_result = split_atom(
                    atom,
                    max(remaining_byte(), 0),
                )
            if (
                split_result is not None
                and len(split_result) == 2
                and fits(split_result[0])
            ):
                first_part, atom = split_result
                append_atom(first_part)
                start_new_page()
                continue

            page_has_reserved_content = (
                is_first_page
                and (status_size > 0 or status_nodes > 0)
            )
            if body_pages[-1] or page_has_reserved_content:
                start_new_page()
                continue

            # An unsplittable atom larger than an otherwise empty page is left
            # intact for the delivery-layer truncation guard.
            append_atom(atom)
            break

    if not body_pages or (len(body_pages) == 1 and not body_pages[0]):
        body_pages = [[]]

    total = len(body_pages)
    return [layout.assemble_for_page(idx, total, tuple(slice_)) for idx, slice_ in enumerate(body_pages)]


def _atom_size(atom: RenderAtom) -> int:
    return atom.byte_size if atom.byte_size > 0 else estimate_atom_size(atom)
