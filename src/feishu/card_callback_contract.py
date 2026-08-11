"""Latest Feishu card callback contract and response builders."""

from __future__ import annotations

from typing import Any

from lark_channel.event.callback.model.p2_card_action_trigger import (
    P2CardActionTriggerResponse,
)

from src.card.schema import require_card_json_2

CARD_CALLBACK_SCHEMA = "2.0"
CARD_CALLBACK_EVENT_TYPE = "card.action.trigger"
_TOAST_TYPES = frozenset({"info", "success", "warning", "error"})


def _required_text(owner: Any, field: str) -> str:
    value = getattr(owner, field, None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"latest Feishu card callback requires {field}")
    return value


def validate_card_action_trigger(data: Any) -> None:
    """Reject any callback outside the current Card 2.0 event contract."""

    if getattr(data, "schema", None) != CARD_CALLBACK_SCHEMA:
        raise ValueError(
            "latest Feishu card callback requires schema '2.0'"
        )
    header = getattr(data, "header", None)
    if getattr(header, "event_type", None) != CARD_CALLBACK_EVENT_TYPE:
        raise ValueError(
            "latest Feishu card callback requires card.action.trigger"
        )
    _required_text(header, "event_id")

    event = getattr(data, "event", None)
    if event is None:
        raise ValueError("latest Feishu card callback requires event")
    if getattr(event, "operator", None) is None:
        raise ValueError("latest Feishu card callback requires operator")
    if getattr(event, "action", None) is None:
        raise ValueError("latest Feishu card callback requires action")
    context = getattr(event, "context", None)
    if context is None:
        raise ValueError("latest Feishu card callback requires context")
    _required_text(context, "open_message_id")
    _required_text(context, "open_chat_id")


def build_card_action_response(
    *,
    toast_type: str | None = None,
    toast_content: str | None = None,
    card: dict[str, Any] | None = None,
) -> P2CardActionTriggerResponse:
    """Build the official latest callback response envelope."""

    payload: dict[str, Any] = {}
    if toast_type is not None or toast_content is not None:
        if toast_type not in _TOAST_TYPES:
            raise ValueError("unsupported Feishu callback toast type")
        if not isinstance(toast_content, str) or not toast_content.strip():
            raise ValueError("Feishu callback toast content must be non-empty")
        payload["toast"] = {
            "type": toast_type,
            "content": toast_content,
        }
    if card is not None:
        payload["card"] = {
            "type": "raw",
            "data": require_card_json_2(card),
        }
    return P2CardActionTriggerResponse(payload)


def empty_card_action_response() -> P2CardActionTriggerResponse:
    """Acknowledge a callback without requesting a toast or card refresh."""

    return build_card_action_response()


__all__ = [
    "CARD_CALLBACK_EVENT_TYPE",
    "CARD_CALLBACK_SCHEMA",
    "build_card_action_response",
    "empty_card_action_response",
    "validate_card_action_trigger",
]
