from __future__ import annotations

import inspect
import json

from src.autonomous.outbox.cards import build_employee_status_card
from src.autonomous.outbox.models import EmployeeCardState
from src.autonomous.provisioning.lark_app import current_registration_manifest
from src.card.builders.core import CoreBuilder
from src.feishu import ws_client
from src.feishu.handlers.base import BaseHandler


def test_main_bot_registers_only_latest_card_callback() -> None:
    source = inspect.getsource(ws_client.FeishuWSClient._build_event_handler)

    assert ".register_p2_card_action_trigger(" in source
    assert "card.action.trigger_v1" not in source
    assert "register_p1" not in source


def test_employee_registration_manifest_replaces_old_callback_subscription() -> None:
    assert current_registration_manifest().callbacks == ("card.action.trigger",)


def test_representative_cards_are_card_json_2_only() -> None:
    cards = [
        CoreBuilder._wrap_card("标题", "blue", []),
        build_employee_status_card(
            title="任务",
            state=EmployeeCardState.RUNNING,
            summary="处理中",
            progress_percent=50,
            attempt_id="attempt-1",
        ),
    ]

    for card in cards:
        assert card["schema"] == "2.0"
        assert isinstance(card["body"]["elements"], list)
        assert "elements" not in card


def test_interactive_delivery_rejects_old_card_shapes() -> None:
    old_cards = [
        {"elements": []},
        {"schema": "1.0", "elements": []},
        {"schema": "2.0", "elements": []},
    ]

    for old_card in old_cards:
        try:
            BaseHandler._normalize_interactive_card_content(
                json.dumps(old_card, ensure_ascii=False)
            )
        except ValueError as exc:
            assert "Card JSON 2.0" in str(exc)
        else:
            raise AssertionError(f"old card was accepted: {old_card!r}")
