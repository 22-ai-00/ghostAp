"""Process-level lifecycle registry for card delivery engines."""

from __future__ import annotations

import atexit
import logging
import threading
import time
import weakref
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.card.delivery.engine import CardDelivery

logger = logging.getLogger(__name__)


class DeliveryRegistry:
    """Track living delivery engines and coordinate graceful shutdown."""

    def __init__(self) -> None:
        # The registry coordinates living engines without becoming an
        # additional ownership root. Explicit CardDelivery.shutdown() remains
        # responsible for promptly stopping the eviction worker.
        self._instances: weakref.WeakSet[CardDelivery] = weakref.WeakSet()
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
            # A registry remains reusable after an earlier shutdown wave.  A
            # later instance must participate in the next shutdown_all().
            self._shutdown_done = False

    def unregister(self, instance: CardDelivery) -> None:
        """Unregister a CardDelivery instance (e.g. on shutdown)."""
        with self._lock:
            self._instances.discard(instance)

    def shutdown_all(self, timeout: float = 5.0) -> bool:
        """Drain and shut down every living delivery within one deadline."""
        with self._lock:
            if self._shutdown_done:
                return True
            instances = list(self._instances)
        deadline = time.monotonic() + max(0.0, timeout)
        all_shutdown = True
        for instance in instances:
            try:
                remaining = max(0.0, deadline - time.monotonic())
                if not instance.shutdown(timeout=remaining):
                    all_shutdown = False
            except Exception:
                logger.debug("CardDelivery shutdown failed", exc_info=True)
                all_shutdown = False
        with self._lock:
            # A delivery may be registered while the snapshot is shutting
            # down.  Do not mark the registry complete until no live instance
            # remains; the next shutdown wave will pick it up.
            self._shutdown_done = all_shutdown and not self._instances
            return self._shutdown_done

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
