"""Process-level lifecycle registry for card delivery engines."""

from __future__ import annotations

import atexit
import logging
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.card.delivery.engine import CardDelivery

logger = logging.getLogger(__name__)


class DeliveryRegistry:
    """Track living delivery engines and coordinate graceful shutdown."""

    def __init__(self) -> None:
        self._instances: set[CardDelivery] = set()
        self._shutdown_done: bool = False
        self._atexit_installed: bool = False
        self._lock: threading.Lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

    def install_atexit(self) -> None:
        """Register atexit handler for graceful shutdown (idempotent).

        Call this once during application bootstrap (e.g. ws_client startup).
        Multiple calls are safe — only the first one registers the handler.
        """
        with self._lock:
            if self._atexit_installed:
                return
            self._atexit_installed = True

        def _atexit_shutdown():
            """Best-effort drain + shutdown on interpreter exit."""
            self.drain_in_flight(timeout=5)
            self.shutdown_all()

        atexit.register(_atexit_shutdown)

    def register(self, instance: CardDelivery) -> None:
        """Register a new CardDelivery instance.

        Automatically installs atexit handler on first registration to ensure
        graceful shutdown regardless of application entry point.
        """
        self.install_atexit()
        with self._lock:
            self._instances.add(instance)

    def unregister(self, instance: CardDelivery) -> None:
        """Unregister a CardDelivery instance (e.g. on shutdown)."""
        with self._lock:
            self._instances.discard(instance)

    def shutdown_all(self) -> None:
        """Shut down all living CardDelivery instances. Called during graceful shutdown."""
        with self._lock:
            if self._shutdown_done:
                return
            instances = list(self._instances)
            self._shutdown_done = True
        for instance in instances:
            try:
                instance._shutdown()
            except Exception:
                pass

    def drain_in_flight(self, timeout: float = 5.0) -> bool:
        """Wait for in-flight deliveries to finish across all living instances.

        Uses per-instance atomic fence+drain to avoid cross-instance deadlock:
        each instance is fenced and drained before moving to the next.

        Returns:
            True if all in-flight deliveries were drained successfully,
            False if timeout was reached.
        """
        deadline = time.monotonic() + timeout
        with self._lock:
            instances = list(self._instances)
        for instance in instances:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.debug("drain_in_flight: timeout reached")
                return False
            if not instance._drain(timeout=remaining):
                return False
        return True

# Module-level singleton
delivery_registry = DeliveryRegistry()
