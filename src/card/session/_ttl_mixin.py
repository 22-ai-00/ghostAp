"""TTLActuator: extracted TTL actuator protocol implementation (composition).

This module provides the TTLActuator class which implements the TTLActuator
protocol via composition. It receives a TTLContext dataclass containing
references to all shared mutable state it needs from the owning CardSession.

All methods access state through self._ctx rather than directly on self,
enabling independent instantiation and testing.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.card.events import CardEvent
from src.card.protocols import TTLState
from src.card.session._constants import TTL_ENGINE_KEY_MAP
from src.card.ui_text import UI_TEXT

if TYPE_CHECKING:
    import threading

    from src.card.delivery.engine import CardDelivery
    from src.card.delivery.tracker import DeliveryTracker
    from src.card.hooks import HookFirer
    from src.card.render.budget import RenderBudget
    from src.card.state.models import CardMetadata, CardState
    from src.card.timers.manager import SessionTimerManager

logger = logging.getLogger(__name__)


def _with_monotonic_now(event: CardEvent, now: float) -> CardEvent:
    """Return an event carrying the session clock used by normal dispatch."""
    if "_now" in event.payload:
        return event
    return CardEvent(
        type=event.type,
        payload={**event.payload, "_now": now},
    )


@dataclass(frozen=False)
class TTLContext:
    """Shared mutable state container injected into TTLActuator.

    Holds references to the owning CardSession's internal objects.
    Not frozen because some fields (like _state, _ttl_warned, _terminal_reason,
    _last_dispatch_time) need to be mutated by the actuator under lock.

    The owning CardSession constructs this with references to its own attributes.
    """

    lock: threading.Lock
    clock: Callable[[], float]
    closed: threading.Event
    session_id: str
    chat_id: str
    metadata: CardMetadata
    budget: RenderBudget
    timers: SessionTimerManager
    delivery: CardDelivery
    reply_to: str | None
    notify_callback: Callable[[str, str], None] | None
    reply_text_fn: Callable[[str, str], None] | None
    hook_firer: HookFirer
    tracker: DeliveryTracker
    deliver_and_track: Callable[..., None]
    engine_cmd_fn: Callable[[], str]
    engine_name_fn: Callable[[], str]

    # Mutable state fields — mutated under lock by actuator
    # These are accessed via property-like patterns on the owning session,
    # but we store a mutable container to allow updates.
    class _MutableState:
        """Inner mutable state bucket to keep TTLContext fields simple."""

        __slots__ = ("state", "ttl_warned", "terminal_reason", "last_dispatch_time", "ttl_seconds")

        def __init__(
            self,
            state: CardState | None,
            ttl_warned: bool,
            terminal_reason: str | None,
            last_dispatch_time: float,
            ttl_seconds: float,
        ) -> None:
            self.state = state
            self.ttl_warned = ttl_warned
            self.terminal_reason = terminal_reason
            self.last_dispatch_time = last_dispatch_time
            self.ttl_seconds = ttl_seconds

    mutable: _MutableState = None  # type: ignore[assignment]  # Set by CardSession.__init__


class TTLActuator:
    """Composition-based TTL actuator implementing the TTLActuator protocol.

    Receives a TTLContext at construction and operates on it.
    CardSession holds an instance as self._ttl_actuator.
    """

    __slots__ = ("_ctx",)

    _LOCK_ACQUIRE_TIMEOUT: float = 1.0

    def __init__(self, ctx: TTLContext) -> None:
        self._ctx = ctx

    def get_ttl_state(self) -> TTLState | None:
        """Return a consistent snapshot of TTL-relevant session state.

        Returns None if the lock cannot be acquired within _LOCK_ACQUIRE_TIMEOUT (contention).
        """
        ctx = self._ctx
        acquired = ctx.lock.acquire(timeout=self._LOCK_ACQUIRE_TIMEOUT)
        if not acquired:
            return None
        try:
            idle = ctx.clock() - ctx.mutable.last_dispatch_time
            return TTLState(
                closed=ctx.closed.is_set(),
                ttl_warned=ctx.mutable.ttl_warned,
                idle_seconds=idle,
                ttl_seconds=ctx.mutable.ttl_seconds,
                session_id=ctx.session_id,
                state_snapshot=ctx.mutable.state,
            )
        finally:
            ctx.lock.release()

    def reduce_and_render(self, events: list[CardEvent]) -> list:
        """Apply events to card state and render; returns rendered payload.

        Holds internal lock during reduce+render. Raises on failure.
        Uses the reduce_card_state binding from src.card.session.core so that
        monkeypatching in tests works correctly.
        """
        # Late import to use same binding as core.py (supports monkeypatch)
        import src.card.session.core as _core

        ctx = self._ctx
        with ctx.lock:
            snapshot = ctx.mutable.state
            now = ctx.clock()
            try:
                for ev in events:
                    ctx.mutable.state = _core.reduce_card_state(
                        ctx.mutable.state,
                        _with_monotonic_now(ev, now),
                        ctx.metadata,
                    )
                assert ctx.mutable.state is not None
                rendered = _core.render_card(ctx.mutable.state, ctx.budget)
            except Exception:
                ctx.mutable.state = snapshot
                raise
            return rendered

    def mark_ttl_expired(self) -> None:
        """Atomically set ttl_warned=True and terminal_reason='ttl_expired' (under lock)."""
        ctx = self._ctx
        with ctx.lock:
            ctx.mutable.ttl_warned = True
            ctx.mutable.terminal_reason = "ttl_expired"

    def rollback_ttl_warned(self) -> None:
        """Reset ttl_warned=False (for error recovery, under lock)."""
        ctx = self._ctx
        with ctx.lock:
            ctx.mutable.ttl_warned = False

    def defer_idle_timeout(
        self,
        on_expired: Callable[[], None],
        on_prewarning: Callable[[], None],
    ) -> None:
        """Treat currently active engine work as fresh activity and reschedule TTL."""
        ctx = self._ctx
        with ctx.lock:
            if ctx.closed.is_set():
                return
            ctx.mutable.last_dispatch_time = ctx.clock()
            ctx.mutable.ttl_warned = False
            ctx.timers.reset_ttl_timer(
                on_expired=on_expired,
                on_prewarning=on_prewarning,
            )

    def mark_closed(self) -> None:
        """Mark the session closed without waiting on a contended state lock."""
        self._ctx.closed.set()

    def force_terminate(self, reason: str) -> None:
        """Force-close the session when lock cannot be acquired normally.

        Internally handles try-acquire, stale-state rendering, delivery,
        notification, and hook firing.
        """
        ctx = self._ctx
        logger.warning("CardSession %s: force-close (reason=%s)", ctx.session_id, reason)
        ctx.closed.set()
        ctx.timers.close()

        # Late import to use same binding as core.py (supports monkeypatch)
        import src.card.session.core as _core

        lightweight_delivered = False
        engine_cmd_str = ctx.engine_cmd_fn()
        engine_name_str = ctx.engine_name_fn()
        terminal_state = ctx.mutable.state

        # Attempt lock for state access — non-blocking to avoid deadlock
        state_lock_acquired = ctx.lock.acquire(timeout=0)
        try:
            # Select engine-specific TTL text
            ttl_key = TTL_ENGINE_KEY_MAP.get(engine_cmd_str, "card_session_ttl_expired")
            # When generic fallback is used and engine_cmd is a placeholder, degrade to full command list
            effective_cmd = engine_cmd_str
            if ttl_key == "card_session_ttl_expired" and effective_cmd == UI_TEXT.get("card_session_fallback_cmd", ""):
                effective_cmd = UI_TEXT["card_session_ttl_expired_commands"]
            ttl_text = UI_TEXT[ttl_key].format(engine_cmd=effective_cmd, engine_name=engine_name_str)
            snap = ctx.mutable.state
            if not state_lock_acquired:
                logger.warning(
                    "CardSession %s: force-close proceeding without lock — state snapshot may be stale",
                    ctx.session_id,
                )
            now = ctx.clock()
            snap = _core.reduce_card_state(
                snap,
                _with_monotonic_now(CardEvent.warning_updated(ttl_text), now),
                ctx.metadata,
            )
            snap = _core.reduce_card_state(
                snap,
                _with_monotonic_now(CardEvent.cancelled(reason=reason), now),
                ctx.metadata,
            )
            terminal_state = snap
            if state_lock_acquired:
                ctx.mutable.state = snap
                ctx.mutable.terminal_reason = reason
            rendered = _core.render_card(snap, ctx.budget)
            ctx.delivery.deliver(
                session_id=ctx.session_id, chat_id=ctx.chat_id,
                rendered=rendered, reply_to=ctx.reply_to, is_terminal=True,
            )
            lightweight_delivered = True
        except Exception as exc:
            logger.debug("CardSession %s: force-close card update failed: %s", ctx.session_id, repr(exc))
        finally:
            if state_lock_acquired:
                ctx.lock.release()

        try:
            ctx.delivery.close(ctx.session_id)
        except Exception as exc:
            logger.exception("CardSession %s: delivery.close() failed in force-close: %s", ctx.session_id, repr(exc))

        if not lightweight_delivered:
            notice_text = UI_TEXT["card_session_ttl_force_close_notice"].format(
                engine_cmd=engine_cmd_str, engine_name=engine_name_str,
            )
            self.notify_user(notice_text)

        ctx.hook_firer.fire_terminal(terminal_state, reason)

    def deliver_terminal(self, rendered: list) -> None:
        """Deliver rendered payload as a terminal event (with tracking)."""
        self._ctx.deliver_and_track(rendered, is_terminal=True)

    def deliver_update(self, rendered: list) -> None:
        """Deliver rendered payload as a non-terminal update (with tracking)."""
        self._ctx.deliver_and_track(rendered, is_terminal=False)

    def force_deliver(self, rendered: list) -> None:
        """Deliver rendered payload directly without tracking (for force-close path)."""
        ctx = self._ctx
        outcomes = ctx.delivery.deliver(
            session_id=ctx.session_id,
            chat_id=ctx.chat_id,
            rendered=rendered,
            reply_to=ctx.reply_to,
            is_terminal=True,
        )
        failed = [o for o in outcomes if o.kind in {"rejected", "reconcile"}]
        if failed:
            raise RuntimeError(f"terminal retry delivery failed: {failed[0].kind}:{failed[0].message}")

    def close_delivery(self) -> None:
        """Close the delivery channel for this session."""
        ctx = self._ctx
        try:
            ctx.delivery.close(ctx.session_id)
        except Exception as exc:
            logger.debug("CardSession %s: close_delivery() failed: %s", ctx.session_id, repr(exc))

    def notify_user(self, text: str) -> None:
        """Send a notification to the user (notify_callback → reply_text fallback)."""
        ctx = self._ctx
        if ctx.notify_callback:
            try:
                ctx.notify_callback(ctx.chat_id, text)
            except Exception as exc:
                logger.debug("CardSession %s: notify_user callback failed: %s", ctx.session_id, repr(exc))
        elif ctx.reply_text_fn and ctx.reply_to:
            try:
                ctx.reply_text_fn(ctx.reply_to, text)
            except Exception as exc:
                logger.debug("CardSession %s: notify_user reply_text fallback failed: %s", ctx.session_id, repr(exc))
        else:
            logger.warning("CardSession %s: notify_user — no callback available", ctx.session_id)

    def fire_terminal_hook(self, reason: str) -> None:
        """Fire on_terminal lifecycle hooks."""
        self._ctx.hook_firer.fire_terminal(self._ctx.mutable.state, reason)

    def schedule_ttl_retry(self, callback: Callable[[], None]) -> bool:
        """Handle a TTL retry; False only means an open session exhausted it."""
        return self._ctx.timers.schedule_ttl_retry(callback)

    def cancel_timers(self) -> None:
        """Cancel all active timers for this session."""
        self._ctx.timers.cancel_all()

    def schedule_retry(self, callback: Callable[[], None]) -> None:
        """Schedule a terminal delivery retry timer."""
        self._ctx.timers.schedule_retry(callback)

    def flag_retry_pending(self) -> None:
        """Flag that a terminal retry is pending in the delivery tracker."""
        ctx = self._ctx
        with ctx.lock:
            ctx.tracker.flag_retry_pending()
