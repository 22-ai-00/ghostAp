"""StaticCardSession: lightweight card delivery for pre-built JSON.

Wraps CardDelivery directly without reduce/render pipeline.
Suitable for static/pre-built cards (diagnostics, project info, etc.).

Thread-safety: Uses threading.Event for closed state, consistent with CardSession.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from typing import Optional

from src.card.delivery.engine import CardDelivery
from src.card.protocols import Session  # noqa: F401 — structural compliance
from src.card.types import RenderedCard

logger = logging.getLogger(__name__)


def _compute_json_signature(card_json: dict) -> str:
    """Compute MD5 signature of the full card JSON for change detection."""
    raw = json.dumps(card_json, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


class StaticCardSession:
    """Lightweight card session for pre-built card JSON.

    Bypasses reduce/render pipeline — delivers pre-built card JSON through
    CardDelivery for unified binding management and sequence tracking.

    Suitable for:
    - Static info cards (diagnostics, project info)
    - Progress cards with manual update patterns
    - Any card where the caller builds the full JSON externally
    """

    def __init__(
        self,
        delivery: CardDelivery,
        chat_id: str,
        *,
        session_id: str | None = None,
        reply_to: str | None = None,
        reply_in_thread: bool | None = None,
    ) -> None:
        self._delivery = delivery
        self._chat_id = chat_id
        self._session_id = session_id or str(uuid.uuid4())
        open_delivery_session = getattr(self._delivery, "open_session", None)
        if callable(open_delivery_session):
            open_delivery_session(self._session_id, chat_id)
        self._reply_to = reply_to
        self._reply_in_thread = reply_in_thread
        self._closed = threading.Event()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def message_id(self) -> str | None:
        """Get the message_id of the first page (if created)."""
        binding = self._delivery.get_binding(self._session_id)
        if binding and binding.pages:
            page = binding.pages.get(0)
            if page:
                return page.message_id
        return None

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    def send(self, card_json: dict | str) -> Optional[str]:
        """Send or update a card.

        First call creates the card; subsequent calls patch it.
        Returns message_id on success, None on failure.

        Args:
            card_json: Complete Feishu Schema 2.0 card JSON (dict or JSON string).
        """
        if self._closed.is_set():
            logger.debug("StaticCardSession %s: send after close, ignoring", self._session_id)
            return None

        if isinstance(card_json, str):
            card_json = json.loads(card_json)

        signature = _compute_json_signature(card_json)
        content_hash = hashlib.md5(json.dumps(card_json, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        rendered = [
            RenderedCard(
                _card_json=card_json,
                structure_signature=signature,
                content_hash=content_hash,
                active_element=None,
                page_index=0,
                total_pages=1,
            )
        ]

        try:
            self._delivery.deliver(
                session_id=self._session_id,
                chat_id=self._chat_id,
                rendered=rendered,
                reply_to=self._reply_to,
                reply_in_thread=self._reply_in_thread,
            )
        except Exception as exc:
            logger.warning("StaticCardSession %s: deliver failed: %s", self._session_id, repr(exc))
            return None
        return self.message_id

    def close(self) -> None:
        """Finalize session: release bindings and sequences. Idempotent."""
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            self._delivery.close(self._session_id)
        except Exception as exc:
            logger.debug("StaticCardSession %s: close failed: %s", self._session_id, repr(exc))
