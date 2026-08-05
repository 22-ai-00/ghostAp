"""Centralized engine metadata — single source of truth for engine type mappings.

All modules that need engine_type → command/name/label mappings MUST import from here.
This eliminates the previous 6+ duplicate definitions across session.py, renderer.py,
hooks.py, payload_truncator.py, buttons.py, and lifecycle.py.
"""

from __future__ import annotations

# Engine type → user-facing slash command
ENGINE_CMD_MAP: dict[str, str] = {
    "deep": "/deep",
    "spec": "/spec",
}

# Engine type → user-facing display name
ENGINE_NAME_MAP: dict[str, str] = {
    "deep": "Deep",
    "spec": "Spec",
}

# Engine type → restart button label (with emoji prefix)
ENGINE_LABELS: dict[str, str] = {
    "deep": "🔄 重新开始 /deep",
    "spec": "🔄 重新开始 /spec",
}

# Default fallback label when engine_type is unknown
ENGINE_LABEL_DEFAULT: str = "🔄 重新开始"


def engine_type_to_cmd(engine_type: str | None, fallback: str = "命令") -> str:
    """Map engine_type to user-facing command string.

    Args:
        engine_type: The engine type key (e.g. "deep", "spec").
        fallback: Value to return when engine_type is not recognized.

    Returns:
        The slash command string (e.g. "/deep") or the fallback.
    """
    return ENGINE_CMD_MAP.get(engine_type or "", fallback)


def engine_type_to_name(engine_type: str | None, fallback: str = "") -> str:
    """Map engine_type to user-facing display name.

    Args:
        engine_type: The engine type key (e.g. "deep", "spec").
        fallback: Value to return when engine_type is not recognized.

    Returns:
        The display name (e.g. "Deep") or the fallback.
    """
    return ENGINE_NAME_MAP.get(engine_type or "", fallback)
