from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from lark_oapi.core.model.base_response import BaseResponse
from lark_oapi.core.model.raw_response import RawResponse

from src.autonomous.provisioning.slash_commands import (
    SlashCommand,
    SlashCommandAPIError,
    SlashCommandReconciler,
    SlashPermissionRequired,
)
from src.autonomous.provisioning.slash_lark import LarkSlashCommandAPI
from src.feishu.ws_client import FeishuWSClient

_APP_ID = "cli_a9224f1c7ca25bd2"
_READ_SCOPE = "application:app_slash_command:read"
_GRANT_URL = (
    f"https://open.feishu.cn/app/{_APP_ID}/auth"
    f"?q={_READ_SCOPE}&op_from=openapi&token_type=tenant"
)


def _permission_response(*, grant_url: str = _GRANT_URL) -> BaseResponse:
    response = BaseResponse()
    response.code = 99991672
    response.msg = "Access denied"
    response.raw = RawResponse()
    response.raw.status_code = 400
    response.raw.content = json.dumps(
        {
            "code": 99991672,
            "msg": f"Access denied; 点击链接申请并开通权限：{grant_url}",
            "error": {
                "permission_violations": [
                    {
                        "subject": _READ_SCOPE,
                        "type": "action_scope_required",
                    }
                ]
            },
        }
    ).encode("utf-8")
    return response


@pytest.mark.asyncio
async def test_slash_adapter_preserves_valid_one_click_permission_link() -> None:
    client = SimpleNamespace(arequest=AsyncMock(return_value=_permission_response()))
    api = LarkSlashCommandAPI(client, expected_app_id=_APP_ID)

    with pytest.raises(SlashPermissionRequired) as exc_info:
        await api.list_commands()

    assert exc_info.value.scopes == (_READ_SCOPE,)
    assert exc_info.value.authorization_url == _GRANT_URL


@pytest.mark.asyncio
async def test_slash_adapter_rejects_permission_link_for_another_app() -> None:
    foreign_url = _GRANT_URL.replace(_APP_ID, "cli_foreign_app")
    client = SimpleNamespace(
        arequest=AsyncMock(
            return_value=_permission_response(grant_url=foreign_url)
        )
    )
    api = LarkSlashCommandAPI(client, expected_app_id=_APP_ID)

    with pytest.raises(SlashCommandAPIError) as exc_info:
        await api.list_commands()

    assert not isinstance(exc_info.value, SlashPermissionRequired)
    assert foreign_url not in str(exc_info.value)


@pytest.mark.asyncio
async def test_reconciler_does_not_erase_permission_recovery_details() -> None:
    required = SlashPermissionRequired(
        operation="GET",
        scopes=(_READ_SCOPE,),
        authorization_url=_GRANT_URL,
    )
    api = SimpleNamespace(list_commands=AsyncMock(side_effect=required))

    with pytest.raises(SlashPermissionRequired) as exc_info:
        await SlashCommandReconciler(
            api,
            desired=(SlashCommand("/help", "Show help"),),
        ).reconcile()

    assert exc_info.value is required


def _client_shell(*, admins: frozenset[str] = frozenset()) -> FeishuWSClient:
    client = FeishuWSClient.__new__(FeishuWSClient)
    client.settings = SimpleNamespace(
        app_id=_APP_ID,
        admin_user_ids=admins,
    )
    client._closed = False
    client._slash_command_sync_wakeup = MagicMock()
    client._slash_permission_notice_lock = threading.Lock()
    client._slash_permission_notice_recipients = set()
    client._slash_permission_notifications = set()
    client._slash_permission_unroutable_logs = set()
    client._pending_slash_permission = None
    client._sync_main_bot_identity = MagicMock(return_value="ou_bot")
    client._get_api_client = MagicMock(return_value=object())
    send_response = SimpleNamespace(
        success=lambda: True,
        data=SimpleNamespace(message_id="om_notice"),
    )
    im_client = SimpleNamespace(
        send_message=MagicMock(return_value=send_response)
    )
    client._handler_ctx = SimpleNamespace(
        handlers={"coco": SimpleNamespace(im_client=im_client)}
    )
    return client


def test_main_slash_sync_notifies_admin_and_recovers_without_restart() -> None:
    client = _client_shell(admins=frozenset({"ou_admin"}))
    required = SlashPermissionRequired(
        operation="GET",
        scopes=(_READ_SCOPE,),
        authorization_url=_GRANT_URL,
    )
    verified = SimpleNamespace(
        observed=(),
        created=(),
        updated=(),
        deleted=(),
    )
    client._slash_command_sync_wakeup.wait.side_effect = [False]

    with patch(
        "src.feishu.ws_client.reconcile_main_agent_slash_commands",
        new=AsyncMock(side_effect=[required, verified]),
    ) as reconcile:
        client._sync_main_slash_commands()

    assert reconcile.await_count == 2
    client._slash_command_sync_wakeup.wait.assert_called_once()
    send = client._handler_ctx.handlers["coco"].im_client.send_message
    send.assert_called_once()
    assert send.call_args.args[0:2] == ("open_id", "ou_admin")
    notice = json.loads(send.call_args.args[2])["text"]
    assert _GRANT_URL in notice
    assert "消息、Shell 和编程工具仍可正常使用" in notice
    assert client._closed is False


def test_main_slash_permission_notice_is_deduplicated() -> None:
    client = _client_shell(admins=frozenset({"ou_admin"}))
    required = SlashPermissionRequired(
        operation="GET",
        scopes=(_READ_SCOPE,),
        authorization_url=_GRANT_URL,
    )

    client._notify_main_slash_permission_required(required)
    client._notify_main_slash_permission_required(required)

    send = client._handler_ctx.handlers["coco"].im_client.send_message
    send.assert_called_once()


def test_first_admitted_private_user_is_notified_when_admin_is_unset() -> None:
    client = _client_shell()
    required = SlashPermissionRequired(
        operation="GET",
        scopes=(_READ_SCOPE,),
        authorization_url=_GRANT_URL,
    )
    client._pending_slash_permission = required

    client._register_slash_permission_notice_recipient(
        sender_id="ou_private_user",
        chat_type="p2p",
    )
    client._notify_main_slash_permission_required(required)

    client._slash_command_sync_wakeup.set.assert_called_once_with()
    send = client._handler_ctx.handlers["coco"].im_client.send_message
    send.assert_called_once()
    assert send.call_args.args[0:2] == ("open_id", "ou_private_user")
