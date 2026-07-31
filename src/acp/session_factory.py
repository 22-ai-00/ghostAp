"""ACP Session Factory — abstracts the creation of different session types.

Decouples the manager from concrete session implementations (ACP, CLI, TTADK).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Protocol

if TYPE_CHECKING:
    from ..agent_session import SyncSession
    from ..config import Settings


logger = logging.getLogger(__name__)


class ACPSessionFactory(Protocol):
    """Protocol for session creation."""

    def create_session(
        self,
        agent_type: str,
        cwd: str,
        model_name: Optional[str] = None,
    ) -> SyncSession:
        """Create a new session instance."""
        ...


class DefaultACPSessionFactory:
    """Default implementation of ACPSessionFactory."""

    def __init__(self, settings: Settings):
        self._settings = settings

    def create_session(
        self,
        agent_type: str,
        cwd: str,
        model_name: Optional[str] = None,
    ) -> SyncSession:
        """Create a session by agent_type.

        - claude: SyncClaudeCLISession (CLI)
        - ttadk_*: SyncTTADKCLISession (CLI bridge)
        - others: SyncACPSession (ACP)
        """
        from ..agent_session import (
            SyncClaudeCLISession,
            SyncTTADKCLISession,
        )
        from ..coco_model import get_coco_model_manager
        from ..utils.path import normalize_ttadk_cwd
        from .sync_adapter import SyncACPSession

        agent_type = (agent_type or "").lower()
        raw_cwd = cwd
        norm_cwd = normalize_ttadk_cwd(raw_cwd)
        cwd = norm_cwd or raw_cwd

        if agent_type == "claude":
            return SyncClaudeCLISession(cwd=cwd, model_name=model_name)

        if agent_type.startswith("ttadk_"):
            try:
                from ..ttadk.startup_common import precheck_ttadk_startup_model

                info = precheck_ttadk_startup_model(
                    agent_type=agent_type, cwd=cwd, model_intent=model_name
                )
                model_name = info.get("model")
                logger.info(
                    "[SessionFactory] ttadk precheck: tool=%s model=%s validated=%s",
                    info.get("tool") or "",
                    (model_name or "(auto)"),
                    bool(info.get("validated")),
                )
            except Exception as e:
                from ..utils.errors import get_error_detail
                logger.debug("TTADK precheck failed: %s", get_error_detail(e))
                model_name = None
            return SyncTTADKCLISession(agent_type=agent_type, cwd=cwd, model_name=model_name)

        # Default to ACP session
        effective_model = model_name
        if not effective_model and agent_type in ("coco", ""):
            try:
                effective_model = get_coco_model_manager().get_current_model()
            except Exception:
                effective_model = None

        # ACP sessions may support model_name in constructor
        try:
            return SyncACPSession(
                agent_type=agent_type or "coco",
                cwd=cwd,
                model_name=effective_model,
            )
        except TypeError:
            # Fallback for older SyncACPSession signature
            return SyncACPSession(agent_type=agent_type or "coco", cwd=cwd)
