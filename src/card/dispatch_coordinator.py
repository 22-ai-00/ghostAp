"""DispatchDeliveryCoordinator: delivery orchestration extracted from CardSession.

Handles the deliver-and-track cycle:
- Successful delivery tracking
- Failure handling with retry scheduling
- Rejected-delivery user notification
- Terminal event finalization (hooks, callbacks)

Thread-safety: Methods are called outside the session lock (delivery I/O)
except where explicitly noted. The session coordinates lock acquisition.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from src.card.delivery.tracker import DeliveryTracker
from src.card.ui_text import UI_TEXT

if TYPE_CHECKING:
    from src.card.delivery.engine import CardDelivery
    from src.card.hooks import HookFirer
    from src.card.session.ttl import TTLHandler
    from src.card.state.models import CardState

logger = logging.getLogger(__name__)


class DispatchDeliveryCoordinator:
    """Orchestrates delivery, failure tracking, retry, and terminal finalization.

    Constructed by CardSession; shares collaborators by reference.
    """

    __slots__ = (
        "_session_id",
        "_chat_id",
        "_delivery",
        "_tracker",
        "_hook_firer",
        "_ttl_handler",
        "_notify_callback",
        "_cancel_callback",
        "_reply_text_fn",
        "_reply_to",
        "_last_rejected_notify",
    )

    def __init__(
        self,
        *,
        session_id: str,
        chat_id: str,
        delivery: CardDelivery,
        tracker: DeliveryTracker,
        hook_firer: HookFirer,
        ttl_handler: TTLHandler,
        notify_callback: Callable | None,
        cancel_callback: Callable | None,
        reply_text_fn: Callable | None,
        reply_to: str | None,
    ) -> None:
        self._session_id = session_id
        self._chat_id = chat_id
        self._delivery = delivery
        self._tracker = tracker
        self._hook_firer = hook_firer
        self._ttl_handler = ttl_handler
        self._notify_callback = notify_callback
        self._cancel_callback = cancel_callback
        self._reply_text_fn = reply_text_fn
        self._reply_to = reply_to
        self._last_rejected_notify: float | None = None

    # ------------------------------------------------------------------
    # Main entry: deliver + track outcomes
    # ------------------------------------------------------------------

    def consume_pending_actions(self) -> list:
        """Delegate: consume pending banner actions from the tracker."""
        return self._tracker.consume_pending_actions()

    @property
    def last_failure_timestamp(self) -> str | None:
        """Delegate: last failure timestamp from the tracker."""
        return self._tracker.last_failure_timestamp

    def deliver(self, rendered: list, *, reply_to: str | None = None, is_terminal: bool = False) -> list:
        """Call CardDelivery.deliver() and return outcomes. Raises on transport error.

        ``is_terminal`` propagates to the delivery engine so terminal renders
        flush every page (no history-page freeze) — the first card message must
        show the final completed/failed state, not a stale mid-run snapshot.
        """
        return self._delivery.deliver(
            session_id=self._session_id,
            chat_id=self._chat_id,
            rendered=rendered,
            reply_to=reply_to if reply_to is not None else self._reply_to,
            is_terminal=is_terminal,
        )

    def on_success(self, is_terminal: bool) -> None:
        """Record successful delivery in tracker. MUST be called under session lock."""
        self._tracker.on_success(is_terminal)

    def on_failure(self, exc: Exception, rendered: list, is_terminal: bool, engine_cmd: str = "") -> None:
        """Handle delivery failure: track, notify, and schedule retry if terminal."""
        self._tracker.on_failure()
        failures = self._tracker.delivery_failures
        should_notify = self._tracker.should_notify_max_failures
        logger.warning(
            "CardSession %s: delivery failed (%d): %s",
            self._session_id, failures, exc,
        )
        if should_notify and self._notify_callback:
            from datetime import datetime
            try:
                timestamp = datetime.now().strftime("%H:%M")
                notify_text = UI_TEXT["card_session_max_failures_banner"].format(
                    timestamp=timestamp, engine_cmd=engine_cmd or UI_TEXT["card_session_fallback_cmd"],
                )
            except Exception:
                notify_text = UI_TEXT["card_session_max_failures_banner_no_ts"]
            try:
                self._notify_callback(self._chat_id, notify_text)
            except Exception as notify_exc:
                logger.warning("CardSession %s: notify_callback failed: %s", self._session_id, repr(notify_exc))
        if is_terminal:
            self.schedule_terminal_retry(rendered)

    # ------------------------------------------------------------------
    # Terminal finalization
    # ------------------------------------------------------------------

    def finalize_terminal(self, state: CardState | None, terminal_reason: str | None) -> None:
        """Close delivery binding and fire terminal lifecycle hooks."""
        try:
            self._delivery.close(self._session_id)
        except Exception as exc:
            logger.debug("CardSession %s: delivery.close() failed: %s", self._session_id, repr(exc))
        reason = terminal_reason or "completed"
        self._hook_firer.fire_terminal(state, reason)
        if reason == "cancelled" and self._cancel_callback:
            try:
                self._cancel_callback()
            except Exception as exc:
                logger.debug("CardSession %s: cancel_callback failed: %s", self._session_id, repr(exc))

    # ------------------------------------------------------------------
    # Retry scheduling
    # ------------------------------------------------------------------

    def schedule_terminal_retry(self, rendered: list) -> None:
        """Schedule a single delayed retry for terminal event delivery failure."""
        self._ttl_handler.schedule_terminal_retry(rendered)

    # ------------------------------------------------------------------
    # Rejected delivery notification
    # ------------------------------------------------------------------

    def notify_rejected(self, engine_cmd: str = "", reason: str = "") -> dict | None:
        """Notify user when delivery is rejected (capacity exhaustion or shutdown).

        Deduplicates: only fires once per 60s window per session.
        Returns a toast dict when throttled, so callers can relay feedback.

        Args:
            engine_cmd: The engine command hint for the user.
            reason: Optional reason string; "shutting down" triggers maintenance text.
        """
        now = time.monotonic()
        if self._last_rejected_notify is not None and now - self._last_rejected_notify < 60.0:
            return {"toast": {"type": "info", "content": UI_TEXT.get("card_session_rejected_throttled", "请稍后重试")}}
        self._last_rejected_notify = now
        if "shutting down" in reason:
            notice = UI_TEXT["card_session_rejected_shutdown"]
        else:
            notice = UI_TEXT["card_session_rejected_notice"].format(engine_cmd=engine_cmd or "/deep")
        if self._notify_callback:
            try:
                self._notify_callback(self._chat_id, notice)
            except Exception as exc:
                logger.debug("CardSession %s: rejected notify failed: %s", self._session_id, repr(exc))
        elif self._reply_text_fn and self._reply_to:
            try:
                self._reply_text_fn(self._reply_to, notice)
            except Exception as exc:
                logger.debug("CardSession %s: rejected reply_text fallback failed: %s", self._session_id, repr(exc))
        else:
            logger.warning("CardSession %s: delivery rejected but no callback available", self._session_id)
        return None
