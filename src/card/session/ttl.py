"""Idle timeout and terminal-delivery recovery for ``CardSession``."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.card.events import CardEvent
from src.card.protocols import TTLState
from src.card.render.renderer import render_card
from src.card.session._constants import TTL_ENGINE_KEY_MAP
from src.card.session.ttl_activity import has_active_card_work
from src.card.state.reducer import reduce_card_state
from src.card.ui_text import UI_TEXT

if TYPE_CHECKING:
    from src.card.session.core import CardSession

logger = logging.getLogger(__name__)


def _with_now(event: CardEvent, now: float) -> CardEvent:
    if "_now" in event.payload:
        return event
    return CardEvent(type=event.type, payload={**event.payload, "_now": now})


class TTLHandler:
    """Own the timeout lifecycle for one card session."""

    __slots__ = ("_session", "_reduce_failure_count")

    _LOCK_ACQUIRE_TIMEOUT = 1.0
    _MAX_REDUCE_FAILURES = 3
    _MAX_TERMINAL_STATE_RETRIES = 3
    _PREWARNING_THRESHOLD = 0.75

    def __init__(self, session: CardSession) -> None:
        self._session = session
        self._reduce_failure_count = 0

    def _snapshot(self) -> TTLState | None:
        s = self._session
        if not s._lock.acquire(timeout=self._LOCK_ACQUIRE_TIMEOUT):
            return None
        try:
            return TTLState(
                closed=s._closed.is_set(),
                ttl_warned=s._ttl_warned,
                idle_seconds=s._clock() - s._last_dispatch_time,
                ttl_seconds=s._ttl_seconds,
                session_id=s._session_id,
                state_snapshot=s._state,
            )
        finally:
            s._lock.release()

    def _defer(self) -> None:
        s = self._session
        with s._lock:
            if s._closed.is_set():
                return
            s._last_dispatch_time = s._clock()
            s._ttl_warned = False
            s._timers.reset_ttl_timer(
                on_expired=self.on_ttl_expired,
                on_prewarning=self.on_ttl_prewarning,
            )

    def _reduce_and_render(self, events: list[CardEvent]) -> list:
        s = self._session
        with s._lock:
            snapshot = s._state
            now = s._clock()
            try:
                for event in events:
                    s._state = reduce_card_state(
                        s._state, _with_now(event, now), s._metadata
                    )
                assert s._state is not None
                return render_card(s._state, s._budget)
            except Exception:
                s._state = snapshot
                raise

    def _notify_user(self, text: str) -> None:
        s = self._session
        if s._notify_callback:
            try:
                s._notify_callback(s._chat_id, text)
            except Exception as exc:
                logger.debug(
                    "CardSession %s: notify callback failed: %s",
                    s._session_id,
                    repr(exc),
                )
        elif s._reply_text_fn and s._reply_to:
            try:
                s._reply_text_fn(s._reply_to, text)
            except Exception as exc:
                logger.debug(
                    "CardSession %s: reply fallback failed: %s",
                    s._session_id,
                    repr(exc),
                )
        else:
            logger.warning(
                "CardSession %s: no timeout notification channel", s._session_id
            )

    def _close_delivery(self) -> None:
        s = self._session
        try:
            s._delivery.close(s._session_id)
        except Exception as exc:
            logger.debug(
                "CardSession %s: delivery close failed: %s",
                s._session_id,
                repr(exc),
            )

    def _force_deliver(self, rendered: list) -> None:
        s = self._session
        outcomes = s._delivery.deliver(
            session_id=s._session_id,
            chat_id=s._chat_id,
            rendered=rendered,
            reply_to=s._reply_to,
            is_terminal=True,
        )
        failed = [outcome for outcome in outcomes if outcome.kind in {"rejected", "reconcile"}]
        if failed:
            raise RuntimeError(
                f"terminal retry delivery failed: {failed[0].kind}:{failed[0].message}"
            )

    def _force_close(self) -> None:
        s = self._session
        reason = "ttl_expired"
        logger.warning("CardSession %s: force-close (reason=%s)", s._session_id, reason)
        s._closed.set()
        s._timers.close()
        delivered = False
        terminal_state = s._state
        acquired = s._lock.acquire(timeout=0)
        try:
            ttl_key = TTL_ENGINE_KEY_MAP.get(s.engine_cmd, "card_session_ttl_expired")
            effective_cmd = s.engine_cmd
            if (
                ttl_key == "card_session_ttl_expired"
                and effective_cmd == UI_TEXT.get("card_session_fallback_cmd", "")
            ):
                effective_cmd = UI_TEXT["card_session_ttl_expired_commands"]
            text = UI_TEXT[ttl_key].format(
                engine_cmd=effective_cmd, engine_name=s.engine_name
            )
            snapshot = reduce_card_state(
                s._state,
                _with_now(CardEvent.warning_updated(text), s._clock()),
                s._metadata,
            )
            snapshot = reduce_card_state(
                snapshot,
                _with_now(CardEvent.cancelled(reason=reason), s._clock()),
                s._metadata,
            )
            terminal_state = snapshot
            if acquired:
                s._state = snapshot
                s._terminal_reason = reason
            rendered = render_card(snapshot, s._budget)
            self._force_deliver(rendered)
            delivered = True
        except Exception as exc:
            logger.debug(
                "CardSession %s: force-close card update failed: %s",
                s._session_id,
                repr(exc),
            )
        finally:
            if acquired:
                s._lock.release()
        self._close_delivery()
        if not delivered:
            self._notify_user(
                UI_TEXT["card_session_ttl_force_close_notice"].format(
                    engine_cmd=s.engine_cmd, engine_name=s.engine_name
                )
            )
        s._hook_firer.fire_terminal(terminal_state, reason)

    def on_ttl_expired(self) -> None:
        """Close an abandoned session, but never interrupt active card work."""
        s = self._session
        state = self._snapshot()
        if state is None:
            if not s._timers.schedule_ttl_retry(self.on_ttl_expired):
                self._force_close()
            return
        if state.closed or state.ttl_warned or state.idle_seconds <= state.ttl_seconds:
            return
        if has_active_card_work(state.state_snapshot):
            logger.info(
                "CardSession %s: TTL deferred because card still has active work",
                state.session_id,
            )
            self._defer()
            return

        with s._lock:
            s._ttl_warned = True
            s._terminal_reason = "ttl_expired"
        ttl_key = TTL_ENGINE_KEY_MAP.get(s.engine_cmd, "card_session_ttl_expired")
        if ttl_key == "card_session_ttl_expired":
            text = UI_TEXT[ttl_key].format(
                expired_commands=UI_TEXT["card_session_ttl_expired_commands"]
            )
        else:
            text = UI_TEXT[ttl_key].format(
                engine_cmd=s.engine_cmd, engine_name=s.engine_name
            )
        try:
            rendered = self._reduce_and_render(
                [CardEvent.warning_updated(text), CardEvent.cancelled(reason="ttl_expired")]
            )
        except Exception as exc:
            logger.error(
                "CardSession %s: TTL reduce/render failed: %s",
                state.session_id,
                exc,
                exc_info=True,
            )
            with s._lock:
                s._ttl_warned = False
            self._reduce_failure_count += 1
            if self._reduce_failure_count >= self._MAX_REDUCE_FAILURES:
                self._force_close()
            else:
                s._timers.schedule_retry(self.on_ttl_expired)
            return
        self._reduce_failure_count = 0
        s._deliver_and_track(rendered, is_terminal=True)

    def on_ttl_prewarning(self) -> None:
        """Show one warning when an inactive session approaches expiry."""
        s = self._session
        state = self._snapshot()
        if state is None:
            if not s._timers.schedule_ttl_retry(self.on_ttl_prewarning):
                with s._lock:
                    s._ttl_warned = True
                self._notify_user(
                    UI_TEXT["card_session_ttl_lock_contention"].format(
                        engine_cmd=s.engine_cmd
                    )
                )
            return
        if (
            state.closed
            or state.ttl_warned
            or state.idle_seconds < state.ttl_seconds * self._PREWARNING_THRESHOLD
        ):
            return
        if has_active_card_work(state.state_snapshot):
            self._defer()
            return
        remaining = max(1, int((state.ttl_seconds - state.idle_seconds) / 60))
        text = UI_TEXT["card_session_ttl_prewarning"].format(
            minutes=remaining, engine_name=s.engine_name
        )
        try:
            rendered = self._reduce_and_render([CardEvent.warning_updated(text)])
        except Exception as exc:
            logger.debug(
                "CardSession %s: TTL prewarning render failed: %s",
                state.session_id,
                repr(exc),
            )
            self._notify_user(text)
            return
        s._deliver_and_track(rendered, is_terminal=False)

    def schedule_terminal_retry(self, rendered: list) -> None:
        """Retry a terminal render once delivery becomes available."""
        s = self._session
        with s._lock:
            s._tracker.flag_retry_pending()
        state_retries = 0

        def retry() -> None:
            nonlocal state_retries
            state = self._snapshot()
            if state is None:
                if state_retries < self._MAX_TERMINAL_STATE_RETRIES:
                    state_retries += 1
                    s._timers.schedule_retry(retry)
                    return
                s._closed.set()
                try:
                    self._notify_user(
                        UI_TEXT["card_session_terminal_fallback_notice"].format(
                            engine_cmd=s.engine_cmd
                        )
                    )
                    self._close_delivery()
                finally:
                    s.release_terminal_resources()
                return
            if state.closed:
                return
            try:
                try:
                    reason = (
                        getattr(state.state_snapshot, "terminal_reason", None)
                        or "completed"
                    )
                    self._force_deliver(rendered)
                    s._closed.set()
                    s._hook_firer.fire_terminal(s._state, reason)
                except Exception as exc:
                    logger.error(
                        "CardSession %s: terminal retry failed: %s",
                        state.session_id,
                        repr(exc),
                    )
                    s._closed.set()
                    self._notify_user(
                        UI_TEXT["card_session_terminal_fallback_notice"].format(
                            engine_cmd=s.engine_cmd
                        )
                    )
                self._close_delivery()
            finally:
                s.release_terminal_resources()

        s._timers.schedule_retry(retry)
