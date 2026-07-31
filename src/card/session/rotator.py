"""SessionRotator: Atomic session rotation for iteration-based engines.

Ensures that session rotation (close old + create new) is atomic,
preventing dispatch-to-closed-session during the transition window.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Callable

from src.card.events import CardEvent
from src.card.events.types import CardEventType
from src.card.nav_link import format_navigation_link
from src.card.protocols import Session  # noqa: F401 — structural compliance
from src.card.ui_text import UI_TEXT

if TYPE_CHECKING:
    from src.card.session.core import CardSession

from src.config import get_settings

logger = logging.getLogger(__name__)

# Content/streaming events should NOT be retried on rotation — they are non-idempotent
# and retrying would cause duplicate text in both old and new sessions.
_CONTENT_EVENT_TYPES: frozenset[CardEventType] = frozenset({
    CardEventType.TEXT_STARTED,
    CardEventType.TEXT_DELTA,
    CardEventType.TEXT_DONE,
    CardEventType.REASONING_STARTED,
    CardEventType.REASONING_DELTA,
    CardEventType.REASONING_DONE,
    CardEventType.TOOL_DELTA,
    CardEventType.PLAN_UPDATED,
})


class SessionRotator:
    """Manages atomic session rotation for engines with iteration boundaries.

    Usage:
        rotator = SessionRotator(initial_session)
        # ... dispatch events via rotator.dispatch(event)
        # At iteration boundary:
        rotator.rotate(session_factory)
    """

    def __init__(self, session: "CardSession") -> None:
        self._session = session
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._closed = False
        self._rotation_count = 0  # Tracks how many times we've rotated (for archived card numbering)

    @property
    def session_id(self) -> str:
        """Delegate to current session's ID."""
        return self._session.session_id

    @property
    def closed(self) -> bool:
        """Whether this rotator has been closed."""
        return self._closed

    @property
    def rotation_count(self) -> int:
        """Current rotation count (number of completed rotations)."""
        return self._rotation_count

    @property
    def current(self) -> "CardSession":
        """Get the current active session (thread-safe read)."""
        with self._lock:
            return self._session

    def dispatch(self, event) -> None:
        """Dispatch event to current session (thread-safe).

        If the session reference becomes stale (i.e., a rotation happened between
        reading the reference and completing the dispatch), retries once with the
        new current session to prevent event loss at rotation boundaries.

        No-op if rotator has been closed.
        """
        with self._lock:
            if self._closed:
                logger.debug("SessionRotator: dispatch ignored after close")
                return
            session = self._session
        # Dispatch outside lock to avoid holding it during delivery
        session.dispatch(event)
        # Check if session was rotated out during the dispatch.
        # If so, retry once with the new session to avoid event loss.
        # Content events (text_delta, etc.) are NOT retried — they are non-idempotent.
        if hasattr(event, "type") and event.type in _CONTENT_EVENT_TYPES:
            return
        with self._lock:
            if self._closed:
                return
            current = self._session
        if current is not session:
            logger.debug("SessionRotator: session rotated during dispatch, retrying once")
            try:
                current.dispatch(event)
            except Exception:
                logger.warning("SessionRotator: retry dispatch failed (session may be closed), event dropped", exc_info=True)

    def rotate(
        self,
        factory: Callable[[], "CardSession"],
        *,
        enforce_max_rotations: bool = True,
        archive_with_hint: bool = True,
    ) -> "CardSession | None":
        """Atomically rotate to a new session.

        Strategy: pre-create outside lock → swap reference inside lock → archive old.
        Factory is called OUTSIDE the lock to avoid blocking concurrent dispatch() calls
        during potentially slow I/O (e.g., streaming card creation).

        Orphan handling: if a concurrent rotate() wins the swap race, the pre-created
        session is immediately closed to prevent resource leaks.

        Returns None if rotator has been closed or factory fails.

        Args:
            factory: Callable that creates a new CardSession.
            enforce_max_rotations: Apply the configured semantic rotation cap.
                Capacity-driven continuation cards disable this limit because
                retaining user-visible history is a correctness requirement.
            archive_with_hint: Allow ARCHIVED reduction to append navigation
                content blocks. Capacity rotations disable this when the old
                card is already full.

        Returns:
            The new active session, or None if closed/failed.
        """
        # Phase 1: Pre-check (fast exit for closed/truncated state)
        with self._lock:
            if self._closed:
                logger.debug("SessionRotator: rotate ignored after close")
                return None
            max_rotations = get_settings().card.session_max_rotations
            if enforce_max_rotations and self._rotation_count >= max_rotations:
                logger.warning(
                    "SessionRotator: max rotations reached (%d/%d), entering truncation mode — "
                    "no new session will be created",
                    self._rotation_count, max_rotations,
                )
                # Dispatch user-visible truncation notice
                try:
                    _engine_cmd = getattr(self._session, "engine_cmd", "")
                    cmd_hint = f"\n发送 {_engine_cmd}_status 可查看完整记录" if _engine_cmd else ""
                    self._session.dispatch(CardEvent.warning_updated(
                        UI_TEXT["card_session_rotation_truncated"].format(cmd_hint=cmd_hint)
                    ))
                except Exception:
                    logger.debug("SessionRotator: truncation notice dispatch failed", exc_info=True)
                return self._session
            pre_rotation_count = self._rotation_count

        # Phase 2: Create new session OUTSIDE lock (may involve I/O)
        try:
            t0 = time.monotonic()
            new_session = factory()
            elapsed_ms = (time.monotonic() - t0) * 1000
            if elapsed_ms > 500:
                logger.warning(
                    "SessionRotator: factory() took %.0fms (>500ms threshold)",
                    elapsed_ms,
                )
        except Exception:
            logger.error("SessionRotator: factory() failed during rotate, keeping old session", exc_info=True)
            return None
        if new_session is None:
            logger.error("SessionRotator: factory() returned None during rotate, keeping old session")
            return None

        # Phase 3: Atomic swap inside lock (CAS — check rotation_count hasn't changed)
        with self._lock:
            if self._closed:
                # Rotator closed during factory — orphan cleanup
                new_session.close()
                return None
            if self._rotation_count != pre_rotation_count:
                # Another rotate() won the race — close our orphan session
                logger.debug("SessionRotator: concurrent rotate detected, closing orphan session")
                new_session.close()
                return self._session
            old_session = self._session
            self._session = new_session
            self._rotation_count += 1
            rotation_seq = self._rotation_count

        # Phase 4: Drain old session's delivery pipeline before archiving.
        # This ensures the old card's last content update completes before the
        # ARCHIVED event is dispatched, preventing "old card still updating" UX.
        _drain = getattr(old_session, "wait_delivery_idle", None)
        if _drain is not None:
            _drain(timeout=2.0)

        # Phase 5: Mark old card as archived (outside lock)
        new_msg_id = getattr(new_session, "delivered_message_id", "") or ""

        nav_summary, fallback_notice = format_navigation_link(
            new_msg_id=new_msg_id or None,
            rotation_seq=rotation_seq,
        )
        if fallback_notice:
            try:
                new_session.notify_user(fallback_notice)
            except Exception:
                logger.debug("SessionRotator: fallback notify failed for session %s", new_session.session_id)
        try:
            old_session.dispatch(CardEvent.archived(
                summary=nav_summary,
                sequence=rotation_seq,
                new_message_id=new_msg_id,
                bridge_phrase=f"续接 #{rotation_seq + 1} ↓",
                append_hint=archive_with_hint,
            ))
        except Exception:
            logger.warning(
                "SessionRotator: archived dispatch failed for session %s",
                old_session.session_id,
                exc_info=True,
            )
        return new_session

    def close(self) -> None:
        """Close the current session. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            session = self._session
        session.close()
