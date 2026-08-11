"""Strict Card JSON 2.0 boundary validation."""

from __future__ import annotations

from typing import Any


def require_card_json_2(card: Any) -> dict[str, Any]:
    """Return *card* only when it is a complete Card JSON 2.0 payload.

    Renderer fragments are deliberately not accepted here. They must be
    assembled into a full card by their owning handler before delivery.
    """

    if not isinstance(card, dict):
        raise ValueError("Card JSON 2.0 payload must be an object")
    if card.get("schema") != "2.0":
        raise ValueError("Card JSON 2.0 payload must declare schema '2.0'")
    if "elements" in card:
        raise ValueError(
            "Card JSON 2.0 payload must use body.elements, not root elements"
        )
    body = card.get("body")
    if not isinstance(body, dict) or not isinstance(body.get("elements"), list):
        raise ValueError("Card JSON 2.0 payload must contain body.elements")
    return card


__all__ = ["require_card_json_2"]
