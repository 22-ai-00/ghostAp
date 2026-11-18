"""Session protocol: unified lifecycle interface for all session types.

Provides a minimal protocol that CardSession, StaticCardSession, and
SessionRotator all implement, enabling session registry / GC / monitoring
to operate on a single type constraint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.acp.models import ACPEvent
    from src.card.events import CardEvent
    from src.card.state.models import CardState


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
    "RendererProtocol",
    "Dispatchable",
    "CardAPIClient",
    "StreamBridge",
    "ProcessEventSink",
    "ProcessSegmentRollover",
]
