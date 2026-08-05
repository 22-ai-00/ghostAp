"""Tests for src/card/delivery/factory.py — verify Settings injection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_create_card_delivery_injects_settings():
    """Factory must pass max_session_locks and session_lock_ttl from Settings."""
    mock_client = MagicMock()

    mock_settings = MagicMock()
    mock_settings.card.session_lock_max = 5000
    mock_settings.card.session_lock_ttl = 300.0

    with patch("src.config.get_settings", return_value=mock_settings):
        from src.card.delivery.factory import create_card_delivery
        delivery = create_card_delivery(mock_client)

    assert delivery._max_session_locks == 5000
    assert delivery._session_lock_ttl == 300.0


@pytest.mark.parametrize("kwargs, match", [
    ({"max_session_locks": 0},   "max_session_locks must be > 0"),
    ({"session_lock_ttl": -1},   "session_lock_ttl must be > 0"),
    ({"eviction_interval": 0},   "eviction_interval must be > 0"),
    ({"eviction_interval": -1},  "eviction_interval must be > 0"),
])
def test_invalid_constructor_args(kwargs, match):
    """CardDelivery must reject invalid constructor args with ValueError."""
    from src.card.delivery.engine import CardDelivery

    mock_client = MagicMock()
    with pytest.raises(ValueError, match=match):
        CardDelivery(client=mock_client, **kwargs)
