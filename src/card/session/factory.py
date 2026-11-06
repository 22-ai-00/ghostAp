"""CardSessionFactory: Creates CardSession instances with shared delivery."""

from __future__ import annotations

import logging
import time
import uuid
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING

from src.card.delivery.engine import CardDelivery
from src.card.render.budget import RenderBudget
from src.card.session.config import SessionCallbacks, SessionConfig
from src.card.state.models import CardMetadata

if TYPE_CHECKING:
    from src.card.session.core import CardSession

logger = logging.getLogger(__name__)

# Retry constants for capacity exhaustion
_MAX_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY = 1.0  # seconds
_RETRY_BACKOFF_FACTOR = 2.0


class CardSessionFactory:
    """Factory for creating CardSession instances.

    Injects shared delivery instance so handlers don't manage it.
    Tracks all created sessions via WeakValueDictionary for monitoring/audit.
    """

    def __init__(self, delivery: CardDelivery, budget: RenderBudget | None = None) -> None:
        self._delivery = delivery
        self._budget = budget
        self._sessions: weakref.WeakValueDictionary[str, CardSession] = weakref.WeakValueDictionary()

    @property
    def active_sessions(self) -> dict[str, CardSession]:
        """Snapshot of currently alive (non-GC'd) sessions, keyed by session_id."""
        return dict(self._sessions)

    def create(
        self,
        chat_id: str,
        metadata: CardMetadata,
        *,
        session_id: str | None = None,
        reply_to: str | None = None,
        budget: RenderBudget | None = None,
        callbacks: SessionCallbacks | None = None,
        ttl_seconds: float | None = None,
        clock: Callable[[], float] | None = None,
        retry_delay: float = 3.0,
    ) -> CardSession:
        """Create a new CardSession.

        Pass callbacks, action registry, notification handlers, and hooks via
        ``SessionCallbacks``.
        """
        from src.card.session.core import CardSession as _CS

        if not metadata.engine_type:
            logger.warning("CardSessionFactory.create(): metadata.engine_type is not set for chat_id=%s", chat_id)

        cbs = callbacks or SessionCallbacks()

        # Fail-fast: production sessions must have at least one notification channel
        if chat_id and cbs.notify_callback is None and cbs.reply_text_fn is None:
            raise ValueError(
                "CardSessionFactory.create() requires at least one notification channel "
                "(notify_callback or reply_text_fn in SessionCallbacks). Without either, "
                "capacity-exhaustion rejections would be silently lost."
            )

        from dataclasses import asdict

        from src.config import get_settings

        effective_budget = budget or self._budget or RenderBudget()

        # Clamp visible_chars to card_max_chars (was previously in SessionConfig.__post_init__)
        _settings = get_settings()
        card_max_chars = _settings.card.max_chars
        if effective_budget.visible_chars > card_max_chars:
            logger.warning(
                "RenderBudget.visible_chars (%d) exceeds card_max_chars (%d), clamping to %d",
                effective_budget.visible_chars, card_max_chars, card_max_chars,
            )
            fields = asdict(effective_budget)
            fields["visible_chars"] = card_max_chars
            effective_budget = RenderBudget(**fields)

        # Resolve TTL defaults from config (so core never needs to import settings)
        effective_ttl = ttl_seconds if ttl_seconds is not None else float(_settings.card.session_idle_timeout)
        effective_warn = float(_settings.card.session_idle_warn_at_remaining)

        config = SessionConfig(
            metadata=metadata,
            budget=effective_budget,
            reply_to=reply_to,
            ttl_seconds=effective_ttl,
            warn_before_seconds=effective_warn,
            clock=clock if clock is not None else time.monotonic,
            retry_delay=retry_delay,
        )
        # Retry on capacity exhaustion (exponential backoff: 1s, 2s, 4s)
        last_exc: RuntimeError | None = None
        for attempt in range(1, _MAX_RETRY_ATTEMPTS + 1):
            try:
                session = _CS(
                    chat_id=chat_id,
                    config=config,
                    delivery=self._delivery,
                    session_id=session_id,
                    callbacks=cbs,
                )
                self._sessions[session.session_id] = session
                return session
            except RuntimeError as exc:
                if "capacity exhausted" not in str(exc) or attempt == _MAX_RETRY_ATTEMPTS:
                    raise
                last_exc = exc
                delay = _RETRY_BASE_DELAY * (_RETRY_BACKOFF_FACTOR ** (attempt - 1))
                logger.warning(
                    "CardSession capacity exhausted, retrying in %.1fs (attempt %d/%d)",
                    delay, attempt, _MAX_RETRY_ATTEMPTS,
                )
                time.sleep(delay)
        # Should not reach here, but safety fallback
        raise last_exc or RuntimeError("CardSession creation failed after retries")

    def create_snapshot(
        self,
        metadata: CardMetadata,
        *,
        budget: RenderBudget | None = None,
    ) -> CardSession:
        """Create a snapshot-only session for rendering without delivery."""
        from src.card.session.core import CardSession as _CS

        effective_budget = budget or self._budget or RenderBudget()
        config = SessionConfig(
            metadata=metadata,
            budget=effective_budget,
            ttl_seconds=None,
        )
        session = _CS(
            chat_id="",
            config=config,
            delivery=self._delivery,
            session_id=f"snapshot-{uuid.uuid4().hex[:8]}",
            callbacks=SessionCallbacks(),
        )
        self._sessions[session.session_id] = session
        return session
