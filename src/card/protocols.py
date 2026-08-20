"""Session protocol: unified lifecycle interface for all session types.

Provides a minimal protocol that CardSession, StaticCardSession, and
SessionRotator all implement, enabling session registry / GC / monitoring
to operate on a single type constraint.

Also provides TTL management protocols: TTLDecider (read-only state query)
and TTLActuator (combined mutation/delivery/timer/idle interface) for
TTLHandler to interact with CardSession through semantic operations,
fully decoupling TTL management from CardSession's internal attributes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.acp.models import ACPEvent
    from src.card.events import CardEvent
    from src.card.state.models import CardState
    from src.card.types import RenderedCard


# ---------------------------------------------------------------------------
# Session lifecycle protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Session(Protocol):
    """Minimal lifecycle protocol for all session types.

    Guarantees:
    - session_id: Unique identifier for the session.
    - closed: Whether the session has been finalized.
    - close(): Idempotent finalization.
    """

    @property
    def session_id(self) -> str: ...

    @property
    def closed(self) -> bool: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# TTL management interface
# ---------------------------------------------------------------------------


class TTLState(NamedTuple):
    """Read-only snapshot of session state needed for TTL decision-making.

    Constructed under lock by the owning session and returned to TTLHandler
    so it can make decisions without holding any lock.
    """

    closed: bool
    ttl_warned: bool
    idle_seconds: float
    ttl_seconds: float
    session_id: str
    state_snapshot: CardState | None


class TTLDecider(Protocol):
    """Session interface for TTL decisions and terminal resource cleanup.

    TTLHandler uses this to inspect session state without modifying it.
    """

    def get_ttl_state(self) -> TTLState | None:
        """Return a consistent snapshot of TTL-relevant session state.

        Returns None if the internal lock cannot be acquired within a
        reasonable timeout (e.g. 1s), signaling contention.
        """
        ...

    @property
    def engine_cmd(self) -> str:
        """User-facing engine command (e.g. '/deep')."""
        ...

    @property
    def engine_name(self) -> str:
        """User-facing engine display name (e.g. 'Deep')."""
        ...

    def release_terminal_resources(self) -> None:
        """Cancel timers and detach the completed session's GC finalizer."""
        ...


class TTLActuator(Protocol):
    """Combined TTL actuator protocol for TTL lifecycle management.

    Aggregates state mutation, delivery, timer/notification, and idle
    extension methods into a single protocol. CardSession owns a
    composition-based implementation of this protocol.
    """

    # --- State mutation ---

    def mark_ttl_expired(self) -> None:
        """Atomically set ttl_warned=True and terminal_reason='ttl_expired' (under lock)."""
        ...

    def rollback_ttl_warned(self) -> None:
        """Reset ttl_warned=False (for error recovery, under lock)."""
        ...

    def mark_closed(self) -> None:
        """Mark the session closed without waiting on the state lock."""
        ...

    def force_terminate(self, reason: str) -> None:
        """Force-close the session (lock-timeout fallback path).

        Handles try-acquire, stale-state rendering, delivery, notification,
        and hook firing internally.
        """
        ...

    def flag_retry_pending(self) -> None:
        """Flag that a terminal retry is pending in the delivery tracker."""
        ...

    # --- Delivery ---

    def reduce_and_render(self, events: "list[CardEvent]") -> "list[RenderedCard]":
        """Apply events to state and render; returns rendered payload.

        Holds internal lock during reduce+render. Raises on failure
        (caller should handle rollback semantics).

        Raises:
            RuntimeError: If the internal state lock cannot be acquired.
            ValueError: If an event has an invalid type or payload.
        """
        ...

    def deliver_terminal(self, rendered: "list[RenderedCard]") -> None:
        """Deliver rendered payload as a terminal event (with tracking)."""
        ...

    def deliver_update(self, rendered: "list[RenderedCard]") -> None:
        """Deliver rendered payload as a non-terminal update (with tracking)."""
        ...

    def force_deliver(self, rendered: "list[RenderedCard]") -> None:
        """Deliver rendered payload directly (no tracking, for force-close path)."""
        ...

    def close_delivery(self) -> None:
        """Close the delivery channel for this session."""
        ...

    # --- Timer and notification ---

    def notify_user(self, text: str) -> None:
        """Send a notification to the user (notify_callback or reply_text fallback)."""
        ...

    def fire_terminal_hook(self, reason: str) -> None:
        """Fire on_terminal lifecycle hooks."""
        ...

    def schedule_ttl_retry(self, callback: Callable[[], None]) -> bool:
        """Handle a TTL retry; False only means an open session exhausted it."""
        ...

    def cancel_timers(self) -> None:
        """Cancel all active timers for this session."""
        ...

    def schedule_retry(self, callback: Callable[[], None]) -> None:
        """Schedule a terminal delivery retry timer."""
        ...

    # --- Idle extension ---

    def defer_idle_timeout(
        self,
        on_expired: Callable[[], None],
        on_prewarning: Callable[[], None],
    ) -> None:
        """Refresh idle activity and reschedule TTL timers."""
        ...


# ---------------------------------------------------------------------------
# Renderer protocol for BaseEngineHandler
# ---------------------------------------------------------------------------


@runtime_checkable
class RendererProtocol(Protocol):
    """Protocol for engine renderer instances used by BaseEngineHandler.

    Defines the minimal interface that DeepRenderer and SpecRenderer implement
    to be used
    from handler toggle/error-handling methods.
    """

    def update_ui_state(self, project_id: str, **kwargs: Any) -> None:
        """Update UI state fields for a project."""
        ...

    def build_error_card(
        self,
        *,
        project: Any = None,
        engine_name: str = "",
        error_msg: str = "",
        **kwargs: Any,
    ) -> tuple[str, dict]:
        """Build an error card for engine failure display.

        Returns:
            Tuple of (message_type, card_json).
        """
        ...


# ---------------------------------------------------------------------------
# Dispatch protocols (canonical definitions — protocols.py defines these)
# ---------------------------------------------------------------------------


@runtime_checkable
class Dispatchable(Protocol):
    """Protocol for objects that can dispatch CardEvents (CardSession or SessionRotator)."""

    @property
    def closed(self) -> bool: ...

    def dispatch(self, event: "CardEvent") -> None: ...


@runtime_checkable
class ManagedDispatchable(Dispatchable, Protocol):
    """Extended protocol that also supports lifecycle management (close)."""

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Delivery layer protocols
# ---------------------------------------------------------------------------


class CardAPIClient(Protocol):
    """Protocol for Feishu card API operations.

    Implementations MUST enforce a socket/request timeout (recommended: 30s)
    on all network calls. The delivery engine does not add its own timeout
    layer — it relies on the client implementation for timeout protection.
    """

    def create_card(
        self,
        chat_id: str,
        card_json: dict,
        *,
        reply_to: str | None = None,
        idempotency_key: str | None = None,
    ) -> tuple[str, str]:
        """Create a card message. Returns (message_id, card_id)."""
        ...

    def update_card(self, card_id: str, card_json: dict, *, sequence: int = 0) -> None:
        """Update (PATCH) a card by card_id."""
        ...

    def update_element(self, card_id: str, element_id: str, content: str, *, sequence: int = 0) -> None:
        """Update a single element's content (element_content API)."""
        ...

    def create_streaming_card(self, card_json: dict) -> str:
        """Create a CardKit card entity with streaming mode. Returns card_id."""
        ...

    def send_card_reference(
        self,
        chat_id: str,
        card_id: str,
        *,
        reply_to: str | None = None,
        idempotency_key: str | None = None,
    ) -> str:
        """Send an IM message referencing a CardKit card. Returns message_id."""
        ...


# ---------------------------------------------------------------------------
# Session collaborator protocols (for CardSession decoupling)
# ---------------------------------------------------------------------------


class TTLManager(Protocol):
    """Protocol for TTL expiration handler — manages idle timeout logic."""

    def on_ttl_expired(self) -> None:
        """Timer callback: check and handle idle timeout expiration."""
        ...


class ActionDispatcher(Protocol):
    """Protocol for action routing — maps inbound button clicks to CardEvents or toasts."""

    def route_closed(self, action_id: str, terminal_reason: str) -> dict:
        """Generate toast response for actions on a closed/expired session."""
        ...

    def resolve(self, action_id: str, payload: dict) -> "CardEvent | dict":
        """Resolve an action_id + payload into a CardEvent or toast dict."""
        ...


# ---------------------------------------------------------------------------
# StreamBridge protocol (ACP event → CardEvent bridge)
# ---------------------------------------------------------------------------


@runtime_checkable
class StreamBridge(Protocol):
    """Protocol for ACP stream bridges that normalize ACP events into CardEvent sequences.

    Implementations convert raw ACP streaming events (text chunks, thought chunks,
    tool calls) into structured CardEvent sequences dispatched to a Dispatchable target.
    """

    def on_event(self, acp_event: "ACPEvent") -> None:
        """Process an ACP event and dispatch corresponding CardEvents."""
        ...

    def close_open_blocks(self) -> None:
        """Close any currently open text/reasoning blocks."""
        ...

    def bind(self, dispatchable: "Dispatchable") -> None:
        """Rebind the bridge to a new dispatchable target (closes open blocks first)."""
        ...


class ProcessEventSink(Protocol):
    """Optional ordered transport for programming process events.

    The result card remains the authoritative task/conclusion projection.  A
    sink may mirror text, reasoning, and tool events into a native process UI
    such as Feishu COT without making the card layer depend on that transport.
    """

    @property
    def started(self) -> bool: ...

    @property
    def message_id(self) -> str | None: ...

    @property
    def healthy(self) -> bool:
        """Whether queued process events can still reach the remote UI."""
        ...

    def start(self) -> None:
        """Create the remote process message and open its run lifecycle."""
        ...

    def emit(self, event: "CardEvent") -> bool:
        """Queue one supported process event; return whether it was handled."""
        ...

    def rollover(self) -> "ProcessSegmentRollover":
        """Seal the current process segment and start a continuation segment."""
        ...

    def complete(self, event: "CardEvent") -> bool:
        """Close the run and its remote lifecycle for one terminal event."""
        ...

    def abort(self) -> None:
        """Best-effort close after local setup or delivery aborts."""
        ...


class ProcessSegmentRollover(NamedTuple):
    """Outcome of replacing one bounded native-process segment.

    ``sealed`` means every event accepted by the old segment is durably visible
    and no longer needs the card replay buffer. ``started`` means a new segment
    is ready to accept the next event.
    """

    sealed: bool
    started: bool
    replay_events: tuple[object, ...] = ()


__all__ = [
    "Session",
    "TTLState",
    "TTLDecider",
    "TTLActuator",
    "TTLManager",
    "ActionDispatcher",
    "RendererProtocol",
    "Dispatchable",
    "ManagedDispatchable",
    "CardAPIClient",
    "StreamBridge",
    "ProcessEventSink",
    "ProcessSegmentRollover",
]
