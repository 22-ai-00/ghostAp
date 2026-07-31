"""TTL callback handlers extracted from CardSession for size reduction.

TTLHandler manages idle-timeout expiration and prewarning logic.
It interacts with the owning session exclusively through the
TTLDecider + TTLActuator protocols — no direct access to private attributes.
All methods are designed to be called as timer callbacks (from daemon threads).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.card.events import CardEvent
from src.card.session._constants import TTL_ENGINE_KEY_MAP
from src.card.session.ttl_activity import has_active_card_work
from src.card.ui_text import UI_TEXT

if TYPE_CHECKING:
    from src.card.protocols import TTLActuator, TTLDecider

logger = logging.getLogger(__name__)


class TTLHandler:
    """Handles TTL expiration and prewarning callbacks for a CardSession."""

    __slots__ = ("_d", "_a", "_reduce_failure_count")

    _MAX_REDUCE_FAILURES: int = 3
    _MAX_TERMINAL_STATE_RETRIES: int = 3

    def __init__(self, decider: TTLDecider, actuator: TTLActuator) -> None:
        self._d = decider
        self._a = actuator
        self._reduce_failure_count: int = 0

    def on_ttl_expired(self) -> None:
        """Timer callback: proactively close session on idle timeout."""
        d, a = self._d, self._a
        state = d.get_ttl_state()

        # Lock contention — schedule retry or force-close
        if state is None:
            if not a.schedule_ttl_retry(self.on_ttl_expired):
                self._force_close()
            return

        if state.closed or state.ttl_warned:
            return
        if state.idle_seconds <= state.ttl_seconds:
            return
        if has_active_card_work(state.state_snapshot):
            logger.info(
                "CardSession %s: TTL deferred because card still has active work",
                state.session_id,
            )
            a.defer_idle_timeout(self.on_ttl_expired, self.on_ttl_prewarning)
            return

        # Mark as expired before attempting reduce/render
        a.mark_ttl_expired()
        logger.info("CardSession %s: TTL expired (%.0fs idle)", state.session_id, state.ttl_seconds)

        # Select engine-specific expired text key
        ttl_key = TTL_ENGINE_KEY_MAP.get(d.engine_cmd, "card_session_ttl_expired")
        # Generic fallback uses {expired_commands} placeholder (full command list);
        # engine-specific keys use {engine_cmd} and {engine_name}.
        if ttl_key == "card_session_ttl_expired":
            ttl_text = UI_TEXT[ttl_key].format(
                expired_commands=UI_TEXT["card_session_ttl_expired_commands"],
            )
        else:
            ttl_text = UI_TEXT[ttl_key].format(engine_cmd=d.engine_cmd, engine_name=d.engine_name)
        events = [
            CardEvent.warning_updated(ttl_text),
            CardEvent.cancelled(reason="ttl_expired"),
        ]
        try:
            rendered = a.reduce_and_render(events)
        except Exception as exc:
            logger.error("CardSession %s: TTL reduce/render failed: %s", state.session_id, exc, exc_info=True)
            a.rollback_ttl_warned()
            self._reduce_failure_count += 1
            if self._reduce_failure_count >= self._MAX_REDUCE_FAILURES:
                logger.warning("CardSession %s: TTL reduce failed %d times, force-closing", state.session_id, self._reduce_failure_count)
                self._force_close()
                return
            a.schedule_retry(self.on_ttl_expired)
            return

        self._reduce_failure_count = 0
        a.deliver_terminal(rendered)

    def _force_close(self) -> None:
        """Force-close when lock cannot be acquired after retries.

        Delegates the entire force-close operation to the session's
        force_terminate() method which handles stale-state, delivery,
        notification, and hook firing internally.
        """
        self._a.force_terminate("ttl_expired")

    # Prewarning fires when this fraction of TTL has elapsed in idle
    _PREWARNING_THRESHOLD = 0.75

    def on_ttl_prewarning(self) -> None:
        """Timer callback: show prewarning banner when ~25% TTL remains.

        The threshold (0.75) means prewarning fires when 75% of idle TTL
        has elapsed, giving users ~25% remaining time to resume activity
        before the hard expiry fires.
        """
        d, a = self._d, self._a
        state = d.get_ttl_state()

        # Lock contention — schedule retry
        if state is None:
            if not a.schedule_ttl_retry(self.on_ttl_prewarning):
                logger.debug("CardSession: TTL prewarning retry exhausted, force expiring")
                a.mark_ttl_expired()
                a.notify_user(UI_TEXT["card_session_ttl_lock_contention"].format(engine_cmd=d.engine_cmd))
            return

        if state.closed or state.ttl_warned:
            return
        if state.idle_seconds < state.ttl_seconds * self._PREWARNING_THRESHOLD:
            return
        if has_active_card_work(state.state_snapshot):
            logger.debug(
                "CardSession %s: TTL prewarning deferred because card still has active work",
                state.session_id,
            )
            a.defer_idle_timeout(self.on_ttl_expired, self.on_ttl_prewarning)
            return

        remaining_min = max(1, int((state.ttl_seconds - state.idle_seconds) / 60))
        warning_text = UI_TEXT["card_session_ttl_prewarning"].format(minutes=remaining_min, engine_name=d.engine_name)

        events = [CardEvent.warning_updated(warning_text)]
        try:
            rendered = a.reduce_and_render(events)
        except Exception as exc:
            logger.debug("CardSession %s: TTL prewarning render failed: %s", state.session_id, repr(exc))
            # Fallback: notify user via text even if card rendering failed
            notify_text = UI_TEXT["card_session_ttl_prewarning"].format(minutes=remaining_min, engine_name=d.engine_name)
            a.notify_user(notify_text)
            return

        # Deliver as non-terminal (prewarning only — no separate chat notification
        # to avoid dual-notification; card banner + keep-alive button is sufficient)
        a.deliver_update(rendered)

    def schedule_terminal_retry(self, rendered: list) -> None:
        """Schedule a single delayed retry for terminal event delivery failure."""
        d, a = self._d, self._a
        a.flag_retry_pending()
        state_retry_count = 0

        def _retry() -> None:
            nonlocal state_retry_count
            # Check if already closed (state=None means lock contention, skip)
            state = d.get_ttl_state()
            if state is None:
                if state_retry_count < self._MAX_TERMINAL_STATE_RETRIES:
                    state_retry_count += 1
                    a.schedule_retry(_retry)
                    return
                logger.error(
                    "CardSession terminal retry state remained contended after %d retries",
                    state_retry_count,
                )
                try:
                    a.mark_closed()
                    notice_text = UI_TEXT["card_session_terminal_fallback_notice"].format(
                        engine_cmd=d.engine_cmd
                    )
                    a.notify_user(notice_text)
                    a.close_delivery()
                finally:
                    d.release_terminal_resources()
                return
            if state.closed:
                return
            try:
                try:
                    reason = getattr(state.state_snapshot, "terminal_reason", None) or "completed"
                    a.force_deliver(rendered)
                    a.mark_closed()
                    a.fire_terminal_hook(reason)
                except Exception as exc:
                    logger.error("CardSession %s: terminal retry failed: %s", state.session_id, repr(exc))
                    a.mark_closed()
                    notice_text = UI_TEXT["card_session_terminal_fallback_notice"].format(engine_cmd=d.engine_cmd)
                    a.notify_user(notice_text)
                a.close_delivery()
            finally:
                d.release_terminal_resources()

        a.schedule_retry(_retry)
