"""CardSession: Handler's single interaction point for card management.

Orchestrates the full pipeline: dispatch → reduce → render → deliver.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
import uuid
import warnings
import weakref
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

from src.card.actions.router import ActionRouter
from src.card.delivery.tracker import DeliveryTracker, PendingAction
from src.card.dispatch_coordinator import DispatchDeliveryCoordinator
from src.card.events import VALIDATE_PAYLOAD, CardEvent, CardEventType
from src.card.hooks import HookFirer
from src.card.render.fallback import render_fallback_card
from src.card.render.renderer import render_card
from src.card.session._constants import ENGINE_CMD_MAP, ENGINE_NAME_MAP
from src.card.session.config import SessionCallbacks
from src.card.session.ttl import TTLHandler
from src.card.session.ttl_activity import has_active_card_work
from src.card.state.reducer import reduce_card_state
from src.card.timers.manager import SessionTimerManager
from src.card.ui_text import UI_TEXT

if TYPE_CHECKING:
    from src.card.delivery.engine import CardDelivery
    from src.card.hooks import SessionHook
    from src.card.session.config import SessionConfig
    from src.card.state.models import CardState
    from src.card.timers.scheduler import TimerHandle

logger = logging.getLogger(__name__)

# Terminal events that trigger immediate flush
_TERMINAL_EVENTS = frozenset({
    CardEventType.COMPLETED,
    CardEventType.FAILED,
    CardEventType.CANCELLED,
    CardEventType.ARCHIVED,
    CardEventType.BLOCKED,
})

# Map PendingAction → CardEvent for banner state management
_PENDING_ACTION_WARNING_MAP: dict[PendingAction, str] = {
    PendingAction.SHOW_RECOVERY: "card_session_recovery_banner",
    PendingAction.CLEAR_BANNER: "",
    PendingAction.SHOW_MAX_FAILURES_WARNING: "card_session_max_failures_banner",
    PendingAction.SHOW_RETRY_PENDING: "card_session_retry_pending_banner",
}

_ENGINE_CMD_MAP = ENGINE_CMD_MAP
_ENGINE_NAME_MAP = ENGINE_NAME_MAP

# Action ID for TTL keep-alive (avoids importing src.card.actions.dispatch)
_TTL_KEEP_ALIVE = "ttl_keep_alive"


def _pending_action_to_event(action: PendingAction, tracker, engine_cmd: str = "") -> CardEvent:
    """Convert a PendingAction enum to the corresponding CardEvent.warning_updated().

    Args:
        tracker: Any object with a ``last_failure_timestamp`` property (DeliveryTracker or coordinator).
    """
    warning_key = _PENDING_ACTION_WARNING_MAP[action]
    # CLEAR_BANNER uses empty string directly; others use UI_TEXT lookup
    warning_text = UI_TEXT[warning_key] if warning_key else ""
    # Format placeholders using actual failure time and engine command
    if "{timestamp}" in warning_text or "{engine_cmd}" in warning_text:
        timestamp = tracker.last_failure_timestamp
        if timestamp:
            warning_text = warning_text.format(timestamp=timestamp, engine_cmd=engine_cmd or UI_TEXT["card_session_fallback_cmd"])
        elif "{timestamp}" in warning_text:
            # No timestamp available — show simplified text without timestamp
            warning_text = UI_TEXT["card_session_paused_notice"]
        else:
            warning_text = warning_text.format(engine_cmd=engine_cmd or UI_TEXT["card_session_fallback_cmd"])
    return CardEvent.warning_updated(warning_text)


# ---------------------------------------------------------------------------
# weakref.finalize callback (module-level, no reference to self)
# ---------------------------------------------------------------------------

def _release_lock(delivery_ref: weakref.ref, session_id: str) -> None:
    """Release delivery session lock on GC — weakref.finalize callback.

    This function must NOT reference the CardSession instance (prevents GC).
    It operates solely on the delivery weakref and session_id.
    """
    delivery = delivery_ref()
    if delivery is None:
        return
    try:
        warnings.warn(
            f"CardSession {session_id!r} was garbage-collected without close(). "
            "This may leak delivery locks. Always call close() explicitly.",
            ResourceWarning,
            stacklevel=1,
        )
        delivery.release_session_lock(session_id)
    except (TypeError, AttributeError):
        # Interpreter shutting down — attributes already gone
        pass
    except Exception:
        try:
            logger.debug("CardSession %s: _release_lock failed during finalization", session_id)
        except Exception:
            # Logger may be unavailable during shutdown
            print(f"CardSession {session_id}: _release_lock failed during finalization", file=sys.stderr)


class CardSession:
    """Card session: dispatch(event) → reduce → render → deliver pipeline.

    Recommended: Use ``CardSessionFactory.create()`` rather than constructing directly.
    """

    def __init__(
        self,
        chat_id: str,
        config: SessionConfig,
        delivery: CardDelivery,
        *,
        session_id: str | None = None,
        callbacks: SessionCallbacks | None = None,
        # Convenience kwargs — merged into callbacks if provided directly
        hooks: tuple[SessionHook, ...] | None = None,
        action_registry: dict[str, Callable[[dict], CardEvent]] | None = None,
    ) -> None:
        # --- Parameter normalization ---
        cbs = self._merge_callbacks(callbacks, hooks, action_registry)
        self._warn_no_notify(chat_id, cbs, session_id)

        # --- Core identity & state ---
        self._session_id = session_id or str(uuid.uuid4())
        self._chat_id = chat_id
        self._metadata = config.metadata
        self._delivery = delivery
        open_delivery_session = getattr(self._delivery, "open_session", None)
        if callable(open_delivery_session):
            open_delivery_session(self._session_id, chat_id)
        self._budget = config.budget
        self._reply_to = config.reply_to
        self._closed = threading.Event()
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._clock = config.clock
        # v2 elapsed/cumulative timing uses the session clock, which is
        # monotonic by SessionConfig contract. Persist the start instant in
        # metadata once so renderers never mix wall-clock and monotonic time.
        if self._metadata.session_started_at is None:
            self._metadata = replace(self._metadata, session_started_at=self._clock())
        self._sync_delivery = bool(config.sync_delivery)
        self._notify_callback = cbs.notify_callback
        self._cancel_callback = cbs.cancel_callback
        self._reply_text_fn = cbs.reply_text_fn
        self._hooks: tuple[SessionHook, ...] = cbs.hooks
        self._action_registry: dict[str, Callable[[dict], CardEvent]] = cbs.action_registry or {}
        self._stop_escalation_handle: "TimerHandle | None" = None
        self._stop_escalation_delay: float = 30.0
        self._pending_card_split: tuple[str, str, str | None] | None = None
        self.on_card_split_completed: Callable[..., None] | None = None
        self._delivery_gate = threading.Condition(threading.Lock())  # leaf lock: never held while acquiring a LockLevel lock
        self._delivery_in_flight = False
        self._delivery_in_flight_terminal = False
        self._delivery_pending: tuple[list, bool, CardEvent, int | None] | None = None
        self._delivery_highest_revision = -1
        self._first_deliver_fired: bool = False
        self._state: CardState | None = None
        self._ttl_warned = False
        self._terminal_reason: str | None = None
        self._last_dispatch_time = self._clock()
        self._ttl_seconds = config.ttl_seconds if config.ttl_seconds is not None else 1800.0

        self._action_router = ActionRouter(
            session_id=self._session_id,
            engine_type=self._metadata.engine_type or "",
            action_registry=self._action_registry,
        )
        self._tracker = DeliveryTracker()
        self._hook_firer = HookFirer(self._hooks, self._session_id)
        self._timers = SessionTimerManager(
            session_id=self._session_id,
            ttl_seconds=self._ttl_seconds,
            clock=self._clock,
            retry_delay=config.retry_delay,
            warn_before_seconds=(
                config.warn_before_seconds
                if config.warn_before_seconds is not None
                else 420.0
            ),
        )
        self._ttl_handler = TTLHandler(self)
        self._coordinator = DispatchDeliveryCoordinator(
            session_id=self._session_id,
            chat_id=self._chat_id,
            delivery=self._delivery,
            tracker=self._tracker,
            hook_firer=self._hook_firer,
            ttl_handler=self._ttl_handler,
            notify_callback=self._notify_callback,
            cancel_callback=self._cancel_callback,
            reply_text_fn=self._reply_text_fn,
            reply_to=self._reply_to,
        )
        self._finalizer = weakref.finalize(
            self, _release_lock, weakref.ref(self._delivery), self._session_id
        )
        self._reset_ttl_timer()

    # ------------------------------------------------------------------
    # Private helpers split from __init__ for readability
    # ------------------------------------------------------------------

    def add_hook(self, hook: "SessionHook") -> None:
        """Append a lifecycle hook after construction.

        Thread-safe; delegates to HookFirer.append_hook().
        Should be called before the session is exposed to concurrent dispatch.
        """
        self._hook_firer.append_hook(hook)

    @staticmethod
    def _merge_callbacks(
        callbacks: SessionCallbacks | None,
        hooks: tuple[SessionHook, ...] | None,
        action_registry: dict[str, Callable[[dict], CardEvent]] | None,
    ) -> SessionCallbacks:
        """Merge convenience kwargs into a single SessionCallbacks instance."""
        if hooks is not None or action_registry is not None:
            base = callbacks or SessionCallbacks()
            return SessionCallbacks(
                notify_callback=base.notify_callback,
                cancel_callback=base.cancel_callback,
                reply_text_fn=base.reply_text_fn,
                action_registry=action_registry if action_registry is not None else base.action_registry,
                hooks=hooks if hooks is not None else base.hooks,
            )
        return callbacks or SessionCallbacks()

    @staticmethod
    def _warn_no_notify(chat_id: str, cbs: SessionCallbacks, session_id: str | None) -> None:
        if chat_id and cbs.notify_callback is None and cbs.reply_text_fn is None:
            logger.warning(
                "CardSession %s: no notification channel (notify_callback/reply_text_fn). "
                "Capacity-exhaustion rejections will be silently lost.",
                session_id or "(pending)",
            )

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def sequence(self) -> int | str:
        return self._metadata.card_sequence

    @property
    def session_started_at(self) -> float:
        return self._metadata.session_started_at or self._clock()

    @property
    def is_subagent(self) -> bool:
        return self._metadata.is_subagent

    @property
    def parent_card_seq(self) -> str | None:
        return self._metadata.parent_card_seq

    @property
    def final_state_for_freeze(self) -> CardState | None:
        return self._metadata.final_state_for_freeze

    @property
    def state(self) -> CardState | None:
        """Read-only access to current card state."""
        with self._lock:
            return self._state

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    @property
    def engine_cmd(self) -> str:
        """User-facing engine command (e.g. '/deep')."""
        return _ENGINE_CMD_MAP.get(self._metadata.engine_type or "", UI_TEXT["card_session_fallback_cmd"])

    @property
    def engine_name(self) -> str:
        """User-facing engine display name (e.g. 'Deep')."""
        return _ENGINE_NAME_MAP.get(self._metadata.engine_type or "", UI_TEXT["card_session_fallback_engine_name"])

    @property
    def delivered_message_id(self) -> str:
        """Return the Feishu message_id of the delivered card (page 0), or empty string."""
        get_binding = getattr(self._delivery, "get_binding", None)
        if get_binding is None:
            return ""
        binding = get_binding(self._session_id)
        if binding and 0 in binding.pages:
            return binding.pages[0].message_id
        return ""

    def _reset_ttl_timer(self) -> None:
        """Cancel existing TTL timer and schedule a new one (idle-timeout)."""
        if self._ttl_seconds <= 0:
            return
        self._timers.reset_ttl_timer(
            on_expired=self._ttl_handler.on_ttl_expired,
            on_prewarning=self._ttl_handler.on_ttl_prewarning,
        )

    def _schedule_stop_escalation(self) -> None:
        """Schedule a 30s timer to escalate STOPPING → force-stop button."""
        from src.card.timers.scheduler import get_timer_scheduler
        scheduler = get_timer_scheduler()
        # Cancel any existing escalation timer first
        if self._stop_escalation_handle is not None:
            scheduler.cancel(self._stop_escalation_handle)
        self._stop_escalation_handle = scheduler.schedule(
            self._stop_escalation_delay,
            self._on_stop_escalation_timeout,
            session_id=self._session_id,
        )

    def _cancel_stop_escalation(self) -> None:
        """Cancel the stop escalation timer (called on terminal events)."""
        if self._stop_escalation_handle is not None:
            from src.card.timers.scheduler import get_timer_scheduler
            get_timer_scheduler().cancel(self._stop_escalation_handle)
            self._stop_escalation_handle = None

    def _on_stop_escalation_timeout(self) -> None:
        """Timer callback: dispatch STOP_ESCALATED to show force-stop button."""
        self._stop_escalation_handle = None
        closed_event = getattr(self, "_closed", None)
        if closed_event is not None and closed_event.is_set():
            return
        try:
            self.dispatch(CardEvent.stop_escalated())
        except Exception:
            logger.debug("CardSession %s: stop escalation dispatch failed (session may be closed)", self._session_id)

    def dispatch(self, event: CardEvent) -> None:
        """Dispatch event through reduce → render → deliver pipeline (thread-safe)."""
        if self._closed.is_set():
            logger.debug("CardSession %s: dispatch after close, ignoring %s", self._session_id, event.type)
            return

        if VALIDATE_PAYLOAD:
            assert isinstance(event.payload, Mapping), f"payload must be Mapping, got {type(event.payload)}"

        if event.type == CardEventType.CARD_SPLIT:
            self._handle_card_split(event)
            return

        # Phase 1: Lock-protected state mutation (reduce only)
        ttl_expired = False
        state_snapshot = None
        is_terminal = False
        with self._lock:
            if self._should_ignore_dispatch_locked(event):
                return
            ttl_expired = self._check_ttl_inline()
            if not ttl_expired:
                event = self._enrich_event(event)
                state_snapshot, is_terminal = self._reduce_safe(event)
                if state_snapshot is None:
                    return
                # Schedule stop escalation timer on STOPPING
                if event.type == CardEventType.STOPPING:
                    self._schedule_stop_escalation()
                # Cancel stop escalation timer on terminal events
                elif event.type in _TERMINAL_EVENTS:
                    self._cancel_stop_escalation()

        # Phase 1b: Render outside lock (CPU-bound, operates on frozen snapshot)
        if ttl_expired:
            self._ttl_handler.on_ttl_expired()
            return
        rendered = self._render_safe(state_snapshot, event)
        if rendered is None:
            rendered = render_fallback_card(state_snapshot, self._metadata.engine_type if self._metadata else None)
            if rendered is None:
                return

        # Phase 2: Deliver outside lock (I/O-bound) — submit to thread pool
        self._hook_firer.fire_dispatched(event, self._state)
        self._submit_delivery(rendered, is_terminal, event, revision=state_snapshot.version)

    def _should_ignore_dispatch_locked(self, event: CardEvent) -> bool:
        """Fence dispatches after close or a logical terminal transition."""
        if self._closed.is_set():
            return True
        # Feishu delivery closes asynchronously; terminal state is authoritative
        # during that gap so late ACP/tool/text revisions cannot overtake it.
        if self._state is None or self._state.terminal_reason is None:
            return False
        logger.debug(
            "CardSession %s: dispatch after terminal state, ignoring %s",
            self._session_id,
            event.type,
        )
        return True

    def _handle_card_split(self, event: CardEvent) -> None:
        """Close this session at a semantic split boundary and notify upstream."""
        reason = str(event.payload.get("reason", ""))
        hint = str(event.payload.get("hint", ""))
        bridge_phrase = event.payload.get("bridge_phrase")
        if not reason:
            logger.warning("CardSession %s: card_split ignored without reason", self._session_id)
            return
        self._pending_card_split = (reason, hint, str(bridge_phrase) if bridge_phrase else None)
        self.dispatch(CardEvent.completed())

    # -- dispatch sub-methods (called under self._lock) ----------------------

    def _check_ttl_inline(self) -> bool:
        """Check idle TTL inline. Returns True if TTL expired, False otherwise.

        Called under ``self._lock``. Does NOT mutate state — just detects expiry.
        """
        if self._ttl_warned:
            return False
        if (self._clock() - self._last_dispatch_time) <= self._ttl_seconds:
            return False
        if has_active_card_work(self._state):
            self._last_dispatch_time = self._clock()
            self._reset_ttl_timer()
            return False
        return True

    def _enrich_event(self, event: CardEvent) -> CardEvent:
        """Refresh idle timer, consume pending banners, inject timestamps. Called under lock."""
        self._last_dispatch_time = self._clock()
        self._reset_ttl_timer()

        # Process pending banner actions from DeliveryTracker (via coordinator)
        for action in self._coordinator.consume_pending_actions():
            engine_cmd = _ENGINE_CMD_MAP.get(self._metadata.engine_type or "", UI_TEXT["card_session_fallback_cmd"])
            banner_event = _pending_action_to_event(action, self._coordinator, engine_cmd)
            self._state = reduce_card_state(self._state, banner_event, self._metadata)

        # Inject one monotonic timestamp for reducer-owned runtime/ETA fields.
        now = self._clock()
        payload = dict(event.payload or {})
        if "_now" not in payload:
            payload["_now"] = now
        if event.type == CardEventType.PROGRESS_UPDATED and "timestamp" not in payload:
            payload["timestamp"] = now
        if payload != event.payload:
            event = CardEvent(type=event.type, payload=payload)

        return event

    def _reduce_safe(self, event: CardEvent) -> tuple["CardState | None", bool]:
        """Reduce state under lock. Returns (new_state_snapshot, is_terminal) or (None, False) on failure."""
        prev_state = self._state
        try:
            self._state = reduce_card_state(self._state, event, self._metadata)
        except Exception:
            logger.exception(
                "CardSession %s: reduce failed for event %s, state preserved",
                self._session_id, event.type,
            )
            self._state = prev_state
            # Attempt fallback warning state
            try:
                warning_event = CardEvent.warning_updated(UI_TEXT["card_session_warning_render_fail"].format(engine_cmd=self.engine_cmd))
                self._state = reduce_card_state(self._state, warning_event, self._metadata)
            except Exception:
                was_terminal = event.type in _TERMINAL_EVENTS
                if was_terminal:
                    self._closed.set()
                    self._hook_firer.fire_terminal(self._state, event.type.value if hasattr(event.type, 'value') else str(event.type))
                return None, False
        return self._state, event.type in _TERMINAL_EVENTS

    def _render_safe(self, state_snapshot: "CardState", event: CardEvent) -> list | None:
        """Render a frozen state snapshot outside the lock. Returns rendered list or None."""
        try:
            return render_card(state_snapshot, self._budget)
        except Exception:
            logger.exception(
                "CardSession %s: render failed for event %s",
                self._session_id, event.type,
            )
            return None

    def _submit_delivery(
        self,
        rendered: list,
        is_terminal: bool,
        event: CardEvent,
        *,
        revision: int | None = None,
    ) -> None:
        """Submit delivery to the global thread pool (non-blocking).

        When ``_sync_delivery`` is True (configured via SessionConfig.sync_delivery),
        delivery runs synchronously on the calling thread for deterministic test behavior.

        ``revision`` is the monotonic CardState version that produced ``rendered``.
        Rendering happens outside the state lock, so a lower revision may finish after
        a newer one. The delivery gate fences those stale renders before they can make
        the visible card move backwards.
        """
        closed_event = getattr(self, "_closed", None)
        if closed_event is not None and closed_event.is_set():
            return
        if self._sync_delivery:
            with self._delivery_gate:
                if not self._accept_delivery_revision(revision):
                    return
                self._deliver_and_track(rendered, is_terminal, event=event)
            return
        with self._delivery_gate:
            if not self._accept_delivery_revision(revision):
                return
            if self._delivery_in_flight:
                if self._delivery_in_flight_terminal:
                    return
                pending_is_terminal = bool(self._delivery_pending and self._delivery_pending[1])
                if is_terminal or not pending_is_terminal:
                    self._delivery_pending = (rendered, is_terminal, event, revision)
                return
            self._delivery_in_flight = True
            self._delivery_in_flight_terminal = is_terminal
        CardSession._submit_delivery_job(self, rendered, is_terminal, event)

    def _accept_delivery_revision(self, revision: int | None) -> bool:
        """Accept one render revision while ``_delivery_gate`` is held."""
        if revision is None:
            return True
        if revision < self._delivery_highest_revision:
            logger.debug(
                "CardSession %s: dropping stale render revision %d < %d",
                self._session_id,
                revision,
                self._delivery_highest_revision,
            )
            return False
        self._delivery_highest_revision = revision
        return True

    def _submit_delivery_job(self, rendered: list, is_terminal: bool, event: CardEvent) -> None:
        from src.card.delivery.pool import get_delivery_pool
        try:
            get_delivery_pool().submit(CardSession._run_delivery_job, self, rendered, is_terminal, event)
        except RuntimeError:
            # Pool shut down: fall back to synchronous delivery.
            logger.warning("CardSession %s: delivery pool unavailable, delivering synchronously", self._session_id)
            CardSession._run_delivery_job(self, rendered, is_terminal, event)
        except Exception:
            logger.exception("CardSession %s: delivery pool submit failed, delivering synchronously", self._session_id)
            CardSession._run_delivery_job(self, rendered, is_terminal, event)

    def _run_delivery_job(self, rendered: list, is_terminal: bool, event: CardEvent) -> None:
        try:
            self._deliver_and_track(rendered, is_terminal, event=event)
        finally:
            CardSession._submit_pending_delivery_if_any(self)

    def wait_delivery_idle(self, timeout: float = 2.0) -> bool:
        """Block until no in-flight delivery remains for this session.

        Returns True if idle was reached, False on timeout.
        Used by SessionRotator to drain old session before archiving.
        """
        if self._sync_delivery:
            return True
        deadline = time.monotonic() + timeout
        with self._delivery_gate:
            while self._delivery_in_flight or self._delivery_pending is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._delivery_gate.wait(timeout=remaining)
        return True

    def _submit_pending_delivery_if_any(self) -> None:
        with self._delivery_gate:
            closed_event = getattr(self, "_closed", None)
            if closed_event is not None and closed_event.is_set():
                self._delivery_pending = None
                self._delivery_in_flight = False
                self._delivery_in_flight_terminal = False
                self._delivery_gate.notify_all()
                return
            pending = self._delivery_pending
            if pending is None:
                self._delivery_in_flight = False
                self._delivery_in_flight_terminal = False
                self._delivery_gate.notify_all()
                return
            self._delivery_pending = None
            self._delivery_in_flight_terminal = pending[1]
        rendered, is_terminal, event, _revision = pending
        CardSession._submit_delivery_job(self, rendered, is_terminal, event)

    def _deliver_and_track(self, rendered: list, is_terminal: bool, *, event: CardEvent | None = None) -> None:
        """Deliver rendered cards via coordinator and handle outcomes."""
        _fire_first_deliver_msg_id: str | None = None
        try:
            outcomes = self._coordinator.deliver(rendered, is_terminal=is_terminal)
            # Check for rejected outcomes (capacity exhausted or shutdown)
            rejected = [o for o in outcomes if o.kind == "rejected"]
            if rejected:
                reason = rejected[0].message
                self._coordinator.notify_rejected(self.engine_cmd, reason=reason)
                if is_terminal:
                    self._coordinator.schedule_terminal_retry(rendered)
                return
            reconcile = [o for o in outcomes if o.kind == "reconcile"]
            if reconcile:
                reason = reconcile[0].message
                self._coordinator.on_failure(
                    RuntimeError(f"delivery reconcile required: {reason}"),
                    rendered,
                    is_terminal,
                    engine_cmd=self.engine_cmd,
                )
                return
            # Delivery succeeded — update tracker and close if terminal
            with self._lock:
                self._coordinator.on_success(is_terminal)
                # Snapshot first-deliver state inside lock, execute hooks outside.
                if not self._first_deliver_fired:
                    _candidate_msg_id = self.delivered_message_id
                    if self._hook_firer.has_hooks and _candidate_msg_id:
                        self._first_deliver_fired = True
                        _fire_first_deliver_msg_id = _candidate_msg_id
                # Update footer timestamp on successful delivery (non-terminal)
                if not is_terminal and self._state and self._state.footer:
                    _now_hhmm = time.strftime("%m-%d %H:%M")
                    self._state = replace(
                        self._state,
                        footer=replace(self._state.footer, last_updated_at=_now_hhmm),
                    )
                if is_terminal:
                    if self._state and not self._terminal_reason:
                        self._terminal_reason = self._state.terminal_reason
                    self._closed.set()
        except Exception as exc:
            self._coordinator.on_failure(exc, rendered, is_terminal, engine_cmd=self.engine_cmd)
            return

        # Fire first-delivered hooks OUTSIDE the lock to avoid lock nesting.
        if _fire_first_deliver_msg_id:
            self._hook_firer.fire_first_delivered(_fire_first_deliver_msg_id)

        # Finalize only on successful terminal delivery
        if is_terminal:
            try:
                self._coordinator.finalize_terminal(self._state, self._terminal_reason)
                self._fire_card_split_completed_if_pending()
            finally:
                self.release_terminal_resources()

    def _fire_card_split_completed_if_pending(self) -> None:
        """Fire card_split completion callback after terminal delivery/hooks."""
        split = self._pending_card_split
        if split is None:
            return
        self._pending_card_split = None
        cb = self.on_card_split_completed
        if cb is None:
            return
        try:
            reason, hint, bridge_phrase = split
            if bridge_phrase is None:
                cb(reason, hint)
            else:
                cb(reason, hint, bridge_phrase)
        except Exception:
            logger.debug("CardSession %s: on_card_split_completed callback failed", self._session_id)

    def close(self) -> None:
        """Explicitly close the session. Idempotent."""
        if self._closed.is_set():
            return
        with self._lock:
            if self._closed.is_set():
                return
            # If state is still running, dispatch cancelled to trigger terminal hooks
            if self._state is not None and self._state.terminal == "running":
                try:
                    self._state = reduce_card_state(
                        self._state, CardEvent.cancelled(reason="explicit_close"), self._metadata
                    )
                    self._terminal_reason = "cancelled"
                except Exception as exc:
                    logger.debug("CardSession %s: close() cancel reduce failed: %s", self._session_id, repr(exc))
            self._closed.set()
        try:
            self._delivery.close(self._session_id)
        except Exception as exc:
            logger.exception("CardSession %s: delivery.close() failed in close(): %s", self._session_id, repr(exc))
        self.release_terminal_resources()
        # Fire terminal hooks if we have state
        if self._state is not None:
            reason = self._terminal_reason or "cancelled"
            self._hook_firer.fire_terminal(self._state, reason)

    def release_terminal_resources(self) -> None:
        """Release timer/finalizer references after a session becomes terminal."""
        self._timers.close()
        self._cancel_stop_escalation()
        release_lock = getattr(self._delivery, "release_session_lock", None)
        lock_released = release_lock is None
        if release_lock is not None:
            try:
                release_lock(self._session_id)
                lock_released = True
            except Exception as exc:
                logger.warning(
                    "CardSession %s: delivery session-lock release failed: %s",
                    self._session_id,
                    repr(exc),
                )
        if lock_released:
            self._finalizer.detach()

    def snapshot(self) -> tuple[str, str] | None:
        """Render current state without delivering (for status commands).

        Returns (msg_type, card_json_str) or None if no state exists.
        Thread-safe: acquires internal lock.
        """
        with self._lock:
            if self._state is None:
                return None
            rendered = render_card(self._state, self._budget)
            if rendered:
                return ("interactive", json.dumps(rendered[0].to_feishu_json(), ensure_ascii=False))
            return None

    def inbound_action(self, action_id: str, payload: dict | None = None) -> dict | None:
        """Handle inbound user action (button click) → dispatch or toast.

        Delegates to ActionRouter for action resolution and toast generation.
        """
        if self._closed.is_set():
            logger.debug("CardSession %s: inbound_action after close, ignoring '%s'", self._session_id, action_id)
            return self._action_router.route_closed(action_id, self._terminal_reason)

        # Special handling: TTL keep-alive resets idle timer without dispatching
        if action_id == _TTL_KEEP_ALIVE:
            with self._lock:
                self._last_dispatch_time = self._clock()
                self._reset_ttl_timer()
            return {"toast": {"type": "success", "content": UI_TEXT["ttl_keep_alive_toast"]}}

        result = self._action_router.resolve(action_id, payload or {})
        if isinstance(result, dict):
            # Toast response (unknown action or factory error)
            return result

        # result is a CardEvent — dispatch it
        try:
            self.dispatch(result)
        except Exception as exc:
            logger.warning("CardSession %s: dispatch error for '%s': %s", self._session_id, action_id, repr(exc))
            return {"toast": {"type": "error", "content": UI_TEXT["card_session_toast_dispatch_error"]}}
        return {"toast": {"type": "info", "content": UI_TEXT.get("card_session_toast_action_ack", "操作已收到")}}
