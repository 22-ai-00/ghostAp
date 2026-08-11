from __future__ import annotations

import json

import pytest
from lark_channel.core.json import JSON
from lark_channel.event.callback.model.p2_card_action_trigger import (
    P2CardActionTrigger,
    P2CardActionTriggerResponse,
)

from src.feishu.card_callback_contract import (
    build_card_action_response,
    validate_card_action_trigger,
)


def _callback_event(*, schema: str = "2.0", event_type: str = "card.action.trigger"):
    return P2CardActionTrigger(
        {
            "schema": schema,
            "header": {
                "event_id": "evt_latest",
                "event_type": event_type,
                "tenant_key": "tenant_latest",
            },
            "event": {
                "operator": {"open_id": "ou_latest"},
                "action": {"tag": "button", "value": {"action": "show_status"}},
                "context": {
                    "open_message_id": "om_latest",
                    "open_chat_id": "oc_latest",
                },
            },
        }
    )


def test_latest_card_callback_is_accepted() -> None:
    validate_card_action_trigger(_callback_event())


@pytest.mark.parametrize(
    ("schema", "event_type"),
    [
        ("1.0", "card.action.trigger"),
        ("", "card.action.trigger"),
        ("2.0", "card.action.trigger_v1"),
    ],
)
def test_old_or_schema_less_card_callback_is_rejected(
    schema: str,
    event_type: str,
) -> None:
    with pytest.raises(ValueError, match="latest Feishu card callback"):
        validate_card_action_trigger(
            _callback_event(schema=schema, event_type=event_type)
        )


def test_callback_requires_latest_context_coordinates() -> None:
    event = _callback_event()
    event.event.context.open_chat_id = ""

    with pytest.raises(ValueError, match="open_chat_id"):
        validate_card_action_trigger(event)


def test_toast_response_uses_typed_latest_callback_response() -> None:
    response = build_card_action_response(
        toast_type="info",
        toast_content="操作已受理",
    )

    assert isinstance(response, P2CardActionTriggerResponse)
    assert response.toast.type == "info"
    assert response.toast.content == "操作已受理"
    assert json.loads(JSON.marshal(response))["toast"] == {
        "type": "info",
        "content": "操作已受理",
    }


def test_raw_card_response_uses_official_v2_envelope() -> None:
    card = {
        "schema": "2.0",
        "body": {"elements": [{"tag": "markdown", "content": "完成"}]},
    }

    response = build_card_action_response(card=card)

    assert isinstance(response, P2CardActionTriggerResponse)
    assert response.card.type == "raw"
    assert response.card.data == card
    assert json.loads(JSON.marshal(response))["card"] == {
        "type": "raw",
        "data": card,
    }


@pytest.mark.parametrize(
    "card",
    [
        {"body": {"elements": []}},
        {"schema": "1.0", "body": {"elements": []}},
        {"schema": "2.0", "elements": []},
        {"schema": "2.0", "body": {}},
    ],
)
def test_raw_card_response_rejects_non_v2_card(card: dict) -> None:
    with pytest.raises(ValueError, match="Card JSON 2.0"):
        build_card_action_response(card=card)


def test_latest_validator_rejects_missing_action() -> None:
    event = _callback_event()
    event.event.action = None

    with pytest.raises(ValueError, match="action"):
        validate_card_action_trigger(event)
