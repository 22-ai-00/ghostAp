import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.feishu.handlers.base import BaseHandler, DurableMainBotReplyError
from src.feishu.im_client import FeishuIMClient


class _Response:
    def __init__(self, message_id: str = "") -> None:
        self.data = SimpleNamespace(message_id=message_id) if message_id else None

    def success(self) -> bool:
        return bool(self.data)


class _FailedResponse:
    def __init__(self, code: int, msg: str = "remote detail") -> None:
        self.code = code
        self.msg = msg
        self.data = None

    def success(self) -> bool:
        return False


def test_durable_reply_uses_explicit_provenance_and_fixed_payload() -> None:
    handler = object.__new__(BaseHandler)
    handler.im_client = MagicMock()
    handler.im_client.reply_message.return_value = _Response("om_warning_reply")

    receipt = handler.reply_durable_text(
        message_id="om_origin",
        tenant_key="tenant_a",
        chat_id="oc_group",
        text="状态未知，请勿重试",
        idempotency_key="employee-warning-stable",
    )

    assert receipt == "om_warning_reply"
    handler.im_client.reply_message.assert_called_once_with(
        "om_origin",
        json.dumps({"text": "状态未知，请勿重试"}, ensure_ascii=False),
        msg_type="text",
        reply_in_thread=False,
        idempotency_key="employee-warning-stable",
        audit_aliases=("oc_group",),
        audit_tenant_key="tenant_a",
    )


def test_durable_reply_empty_receipt_is_retryable_failure() -> None:
    handler = object.__new__(BaseHandler)
    handler.im_client = MagicMock()
    handler.im_client.reply_message.return_value = None

    with pytest.raises(DurableMainBotReplyError, match="receipt"):
        handler.reply_durable_text(
            message_id="om_origin",
            tenant_key="tenant_a",
            chat_id="oc_group",
            text="状态未知，请勿重试",
            idempotency_key="employee-warning-stable",
        )


@pytest.mark.parametrize(
    ("feishu_code", "safe_error_code"),
    (
        (230001, "feishu_message_not_found"),
        (230020, "feishu_message_recalled"),
    ),
)
def test_durable_reply_exposes_only_typed_safe_permanent_error(
    feishu_code: int,
    safe_error_code: str,
) -> None:
    from src.feishu.handlers.base import DurableMainBotReplyPermanentError

    handler = object.__new__(BaseHandler)
    handler.im_client = MagicMock()
    handler.im_client.reply_message.return_value = _FailedResponse(
        feishu_code,
        msg="secret remote diagnostic",
    )

    with pytest.raises(DurableMainBotReplyPermanentError) as raised:
        handler.reply_durable_text(
            message_id="om_origin",
            tenant_key="tenant_a",
            chat_id="oc_group",
            text="状态未知，请勿重试",
            idempotency_key="employee-warning-stable",
        )

    assert raised.value.error_code == safe_error_code
    assert str(raised.value) == safe_error_code
    assert "secret remote diagnostic" not in str(raised.value)


def test_im_reply_can_audit_recovered_tenant_without_context() -> None:
    events: list[tuple[str, str, str]] = []
    message_api = MagicMock()
    message_api.reply.return_value = _Response("om_warning_reply")
    client_obj = MagicMock()
    client_obj.im.v1.message = message_api
    client = FeishuIMClient(
        lambda: client_obj,
        MagicMock(im_api_max_retries=1),
        outbound_audit=lambda tenant, operation, target: events.append(
            (tenant, operation, target)
        ),
        tenant_key_resolver=lambda: "",
    )

    client.reply_message(
        "om_origin",
        '{"text":"warning"}',
        audit_aliases=("oc_group",),
        audit_tenant_key="tenant_recovered",
    )

    assert events == [
        ("tenant_recovered", "reply", "om_origin"),
        ("tenant_recovered", "reply", "oc_group"),
    ]
