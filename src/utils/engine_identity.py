"""Unified engine identity resolution.

This module is the SSOT for mapping interaction mode to:
- UI engine display name
- backend agent_type
- optional model_name
- transport kind (acp/cli)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..mode import InteractionMode


@dataclass(frozen=True)
class EngineIdentity:
    engine_name: str
    agent_type: str
    model_name: Optional[str]
    transport: str


def resolve_engine_identity(
    *,
    mode: InteractionMode,
    acp_tool_name: Optional[str] = None,
    acp_model_name: Optional[str] = None,
) -> EngineIdentity:
    """Resolve engine identity from interaction mode and optional project hints."""
    if mode == InteractionMode.CLAUDE:
        return EngineIdentity(engine_name="Claude", agent_type="claude", model_name=None, transport="cli")

    if mode == InteractionMode.AIDEN:
        model = acp_model_name if (acp_tool_name or "").strip().lower() == "aiden" else None
        return EngineIdentity(engine_name="Aiden", agent_type="aiden", model_name=model, transport="acp")

    if mode == InteractionMode.CODEX:
        model = acp_model_name if (acp_tool_name or "").strip().lower() == "codex" else None
        return EngineIdentity(engine_name="Codex", agent_type="codex", model_name=model, transport="acp")

    if mode == InteractionMode.GEMINI:
        model = acp_model_name if (acp_tool_name or "").strip().lower() == "gemini" else None
        return EngineIdentity(engine_name="Gemini", agent_type="gemini", model_name=model, transport="acp")

    if mode == InteractionMode.TRAEX:
        model = acp_model_name if (acp_tool_name or "").strip().lower() == "traex" else None
        return EngineIdentity(engine_name="Traex", agent_type="traex", model_name=model, transport="acp")

    if mode == InteractionMode.GROK:
        model = acp_model_name if (acp_tool_name or "").strip().lower() == "grok" else None
        return EngineIdentity(engine_name="Grok", agent_type="grok", model_name=model, transport="acp")

    if mode == InteractionMode.DSH:
        model = acp_model_name if (acp_tool_name or "").strip().lower() == "dsh" else None
        return EngineIdentity(engine_name="DSH", agent_type="dsh", model_name=model, transport="acp")

    # SMART / COCO / SHELL default to Coco engine for orchestrators
    model = acp_model_name if (acp_tool_name or "").strip().lower() == "coco" else None
    return EngineIdentity(engine_name="Coco", agent_type="coco", model_name=model, transport="acp")
