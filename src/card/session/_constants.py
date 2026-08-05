"""Shared constants for CardSession subsystem.

Single source of truth for engine-command → UI text key mappings
and action ID constants used by the session core.
"""

from __future__ import annotations

from src.card.engine_meta import ENGINE_CMD_MAP, ENGINE_NAME_MAP

# Maps engine slash-command to the engine-specific TTL expired text key in UI_TEXT.
# Fallback (command not found): "card_session_ttl_expired" (generic).
TTL_ENGINE_KEY_MAP: dict[str, str] = {
    "/spec": "card_session_ttl_expired_spec",
    "/deep": "card_session_ttl_expired_deep",
}

# Action ID for TTL keep-alive button (mirrors src.card.actions.dispatch.TTL_KEEP_ALIVE)
TTL_KEEP_ALIVE = "ttl_keep_alive"

__all__ = ["TTL_ENGINE_KEY_MAP", "TTL_KEEP_ALIVE", "ENGINE_CMD_MAP", "ENGINE_NAME_MAP"]
