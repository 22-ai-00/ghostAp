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

from ..mode.backends import BACKEND_SPEC_BY_KEY, BACKEND_SPEC_BY_MODE, InteractionMode


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
    # SMART / COCO / SHELL default to Coco engine for orchestrators
    spec = BACKEND_SPEC_BY_MODE.get(mode) or BACKEND_SPEC_BY_KEY["coco"]
    if spec.mode_key == "claude":
        return EngineIdentity(
            engine_name=spec.display_name,
            agent_type=spec.mode_key,
            model_name=None,
            transport=spec.transport,
        )

    normalized_tool = (acp_tool_name or "").strip().lower()
    accepted_tools = {spec.mode_key, spec.cli_name.lower()}
    model = acp_model_name if normalized_tool in accepted_tools else None
    return EngineIdentity(
        engine_name=spec.display_name,
        agent_type=spec.mode_key,
        model_name=model,
        transport=spec.transport,
    )
