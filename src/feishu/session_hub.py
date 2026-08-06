import logging
from typing import TYPE_CHECKING

from ..acp.manager import ACPSessionManager

if TYPE_CHECKING:
    from ..acp.telemetry import IdleHealthConfig
    from ..config import Settings

logger = logging.getLogger(__name__)

class SessionManagerHub:
    """Hub for all ACP-based session managers."""

    def __init__(self, settings: "Settings", idle_health_cfg: "IdleHealthConfig"):
        self.coco = ACPSessionManager(
            "coco",
            session_timeout=settings.coco_session_timeout,
            keepalive_interval=settings.acp_keepalive_interval,
            idle_healthcheck_s=settings.acp_session_idle_healthcheck_s,
            idle_health_config=idle_health_cfg,
        )
        self.claude = ACPSessionManager(
            "claude",
            session_timeout=settings.claude_session_timeout,
            keepalive_interval=settings.acp_keepalive_interval,
            idle_healthcheck_s=settings.acp_session_idle_healthcheck_s,
            idle_health_config=idle_health_cfg,
        )
        self.aiden = ACPSessionManager(
            "aiden",
            session_timeout=settings.coco_session_timeout,
            keepalive_interval=settings.acp_keepalive_interval,
            idle_healthcheck_s=settings.acp_session_idle_healthcheck_s,
            idle_health_config=idle_health_cfg,
        )
        self.codex = ACPSessionManager(
            "codex",
            session_timeout=settings.coco_session_timeout,
            keepalive_interval=settings.acp_keepalive_interval,
            idle_healthcheck_s=settings.acp_session_idle_healthcheck_s,
            idle_health_config=idle_health_cfg,
        )
        self.gemini = ACPSessionManager(
            "gemini",
            session_timeout=settings.coco_session_timeout,
            keepalive_interval=settings.acp_keepalive_interval,
            idle_healthcheck_s=settings.acp_session_idle_healthcheck_s,
            idle_health_config=idle_health_cfg,
        )
        self.traex = ACPSessionManager(
            "traex",
            session_timeout=settings.coco_session_timeout,
            keepalive_interval=settings.acp_keepalive_interval,
            idle_healthcheck_s=settings.acp_session_idle_healthcheck_s,
            idle_health_config=idle_health_cfg,
        )

    def cleanup_all(self):
        """Cleanup all session managers."""
        for name in ("coco", "claude", "aiden", "codex", "gemini", "traex"):
            mgr = getattr(self, name)
            try:
                mgr.cleanup_all()
            except Exception as e:
                from ..utils.errors import get_error_detail
                logger.debug("Cleanup %s session manager failed: %s", name, get_error_detail(e))
