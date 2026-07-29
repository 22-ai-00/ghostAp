"""Resource cleanup helpers for Feishu WebSocket client."""

from __future__ import annotations

import logging
import time
from typing import Any

from ..utils.errors import get_error_detail

logger = logging.getLogger(__name__)


class EngineResourceGroup:
    """Manage stop/wait/cleanup lifecycle for one engine manager."""

    def __init__(self, name: str, manager: Any) -> None:
        self.name = name
        self.manager = manager

    def stop_running_engines(self) -> list[Any]:
        try:
            engines = list(self.manager.list_engines())
        except Exception:
            logger.debug("failed to list %s engines", self.name, exc_info=True)
            return []

        for engine in engines:
            try:
                if engine and getattr(engine, "is_running", False):
                    engine.stop()
            except Exception:
                logger.debug("failed to stop %s engine instance", self.name, exc_info=True)
        return engines

    @staticmethod
    def wait_stopped(
        engines: list[Any],
        timeout_s: float = 5.0,
        interval_s: float = 0.05,
    ) -> bool:
        deadline = time.time() + max(0.1, timeout_s)
        while time.time() < deadline:
            any_running = False
            for engine in engines:
                try:
                    if engine and getattr(engine, "is_running", False):
                        any_running = True
                        break
                except Exception:
                    continue
            if not any_running:
                return True
            time.sleep(interval_s)
        for engine in engines:
            try:
                if engine and getattr(engine, "is_running", False):
                    return False
            except Exception:
                logger.debug(
                    "failed to observe engine shutdown state",
                    exc_info=True,
                )
                return False
        return True

    def cleanup_all(self) -> None:
        try:
            self.manager.cleanup_all()
        except Exception as exc:
            logger.debug("清理%s_manager失败: %s", self.name, get_error_detail(exc))
