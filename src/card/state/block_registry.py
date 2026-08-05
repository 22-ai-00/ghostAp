"""Centralized BlockKind registry.

Single source of truth for block kind → type and block kind → atom_kind mappings.
Reuses _BLOCK_KIND_MAP from models.py and derives atom_kind from each class's _atom_kind ClassVar.

Design assumption: The registry is IMMUTABLE after import time.
It does not support runtime dynamic registration or hot-reload of block types.

Adding a new block kind only requires:
1. Define the frozen dataclass with `_atom_kind: ClassVar[str]` and `kind: Literal[...]`
2. Add it to the AnyContentBlock Union in models.py
3. Add it to _BLOCK_KIND_MAP in models.py

The registry auto-collects atom_kind mappings at import time from _BLOCK_KIND_MAP.
"""

from __future__ import annotations

from src.card.state.models import _BLOCK_KIND_MAP


def _build_atom_registry() -> dict[str, str]:
    """Build kind→atom_kind mapping from _BLOCK_KIND_MAP entries' _atom_kind ClassVar."""
    kind_to_atom: dict[str, str] = {}

    for kind_value, block_cls in _BLOCK_KIND_MAP.items():
        atom_kind = getattr(block_cls, "_atom_kind", None)
        if atom_kind is None:
            raise RuntimeError(
                f"Block class {block_cls.__name__} missing '_atom_kind' ClassVar"
            )
        kind_to_atom[kind_value] = atom_kind

    return kind_to_atom


# Re-export BLOCK_KIND_MAP from models (single source of truth)
BLOCK_KIND_MAP = _BLOCK_KIND_MAP
"""kind string → block class (e.g. "text" → TextBlock)"""

# Derived at import time from BLOCK_KIND_MAP
BLOCK_KIND_TO_ATOM = _build_atom_registry()
"""kind string → atom kind (e.g. "text" → "text", "spec_plan" → "spec_plan")"""
