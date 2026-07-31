"""Factory for creating CardDelivery instances with Settings injection."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import CardAPIClient, CardDelivery

__all__ = ["create_card_delivery"]


def create_card_delivery(
    client: "CardAPIClient",
    *,
    payload_transform: Callable[[str, dict], dict] | None = None,
) -> "CardDelivery":
    """Create a CardDelivery with max_session_locks/session_lock_ttl from Settings.

    This ensures all CardDelivery instances honour the same configuration
    regardless of where they are instantiated.
    """
    from src.config import get_settings

    from .engine import CardDelivery

    settings = get_settings()
    return CardDelivery(
        client,
        max_session_locks=settings.card.session_lock_max,
        session_lock_ttl=settings.card.session_lock_ttl,
        payload_transform=payload_transform,
    )
