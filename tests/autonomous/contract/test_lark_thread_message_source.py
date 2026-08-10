from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import src.autonomous.context.lark_source as lark_source
from src.autonomous.context import (
    ContextUnavailableError,
    ContextUnavailableReason,
    EmployeeMessageScope,
)
from src.autonomous.domain.employees import BotPrincipal


def _scope(**overrides: str) -> EmployeeMessageScope:
    values = {
        "tenant_key": "tenant_1",
        "agent_id": "agt_1",
        "bot_principal_id": "bot_1",
        "app_id": "cli_1",
        "chat_id": "oc_1",
        "thread_root_message_id": "om_root",
        "current_message_id": "om_current",
    }
    values.update(overrides)
    return EmployeeMessageScope(**values)


def _principal(**overrides: object) -> BotPrincipal:
    values: dict[str, object] = {
        "bot_principal_id": "bot_1",
        "tenant_key": "tenant_1",
        "agent_id": "agt_1",
        "app_id": "cli_1",
        "credential_ref": "cred_1",
    }
    values.update(overrides)
    return BotPrincipal(**values)


def _message(
    message_id: str = "om_current",
    *,
    chat_id: str = "oc_1",
    root_id: str | None = "om_root",
    thread_id: str | None = "omt_1",
    position: int = 1,
    message_position: int = 11,
    content: object = None,
    msg_type: str = "text",
) -> SimpleNamespace:
    if content is None:
        content = {"text": message_id}
    return SimpleNamespace(
        message_id=message_id,
        root_id=root_id,
        parent_id="" if message_id == "om_root" else root_id,
        thread_id=thread_id,
        msg_type=msg_type,
        create_time="1700000000000",
        update_time="1700000000000",
        deleted=False,
        updated=False,
        chat_id=chat_id,
        sender=SimpleNamespace(
            id="ou_1",
            id_type="open_id",
            sender_type="user",
            tenant_key="tenant_1",
        ),
        body=SimpleNamespace(content=json.dumps(content)),
        message_position=message_position,
        thread_message_position=position,
    )


class _Response:
    def __init__(
        self,
        *,
        items=(),
        code: int = 0,
        has_more: bool = False,
        page_token: str = "",
    ) -> None:
        self.code = code
        self.data = SimpleNamespace(
            items=list(items),
            has_more=has_more,
            page_token=page_token,
        )

    def success(self) -> bool:
        return self.code == 0


class _MessageAPI:
    def __init__(self, *, get_responses, list_responses=()) -> None:
        self.get_responses = list(get_responses)
        self.list_responses = list(list_responses)
        self.get_requests = []
        self.list_requests = []

    def get(self, request):
        self.get_requests.append(request)
        return self.get_responses.pop(0)

    def list(self, request):
        self.list_requests.append(request)
        return self.list_responses.pop(0)


class _Vault:
    def __init__(self, secret: str = "employee-secret") -> None:
        self.secret = secret
        self.calls = []

    def resolve(self, credential_ref: str, agent_id: str, app_id: str) -> str:
        self.calls.append((credential_ref, agent_id, app_id))
        return self.secret


def _factory(*, credential_resolver, client_builder, timeout: float = 7.5):
    with patch.object(lark_source, "_default_client_builder", client_builder):
        return lark_source.LarkEmployeeMessageSourceFactory(
            credential_resolver=credential_resolver,
            request_timeout_seconds=timeout,
        )


def _open_source(
    *,
    scope=None,
    principal=None,
    get_responses=None,
    list_responses=(),
):
    api = _MessageAPI(
        get_responses=get_responses or [_Response(items=[_message()])],
        list_responses=list_responses,
    )
    vault = _Vault()
    builds = []

    def build_client(*, app_id: str, app_secret: str, timeout: float):
        builds.append((app_id, app_secret, timeout))
        return SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(message=api)))

    factory = _factory(
        credential_resolver=vault,
        client_builder=build_client,
    )
    source = factory.open(
        scope=scope or _scope(),
        principal=principal or _principal(),
    )
    return source, api, vault, builds


def test_resolve_thread_binds_current_chat_root_and_thread() -> None:
    source, api, vault, builds = _open_source()

    with source:
        resolved = source.resolve_thread()

    assert resolved.current_message_id == "om_current"
    assert resolved.thread_root_message_id == "om_root"
    assert resolved.feishu_thread_id == "omt_1"
    assert vault.calls == [("cred_1", "agt_1", "cli_1")]
    assert builds == [("cli_1", "employee-secret", 7.5)]
    request = api.get_requests[0]
    assert request.paths == {"message_id": "om_current"}
    assert request.queries == [
        ("user_id_type", "open_id"),
        ("card_msg_content_type", "user_card_content"),
    ]


@pytest.mark.parametrize(
    "items",
    [
        [],
        [_message(), _message("om_other")],
        [_message("om_other")],
        [_message(chat_id="oc_other")],
        [_message(root_id="om_wrong")],
        [_message(thread_id="")],
    ],
)
def test_resolve_thread_fails_closed_on_any_binding_mismatch(items) -> None:
    source, _, _, _ = _open_source(get_responses=[_Response(items=items)])

    with source, pytest.raises(ContextUnavailableError) as raised:
        source.resolve_thread()

    assert raised.value.reason is ContextUnavailableReason.ROOT_THREAD_BINDING


def test_plain_group_root_is_a_stable_current_message() -> None:
    root = _message(
        "om_root",
        root_id=None,
        thread_id=None,
        position=0,
        message_position=10,
    )
    source, _, _, _ = _open_source(
        scope=_scope(current_message_id="om_root"),
        get_responses=[_Response(items=[root]), _Response(items=[root])],
    )

    with source:
        resolved = source.resolve_thread()
        page = source.list_thread_messages()

    assert resolved.feishu_thread_id == ""
    assert [message.message_id for message in page.messages] == ["om_root"]
    assert page.messages[0].thread_id == ""


def test_thread_and_chat_requests_keep_their_own_scope_and_order() -> None:
    thread_page = _Response(items=[_message()], has_more=True, page_token="next")
    chat_page = _Response(
        items=[
            _message(
                "om_group",
                root_id="",
                thread_id="",
                position=20,
                message_position=20,
            )
        ]
    )
    source, api, _, _ = _open_source(list_responses=[thread_page, chat_page])

    with source:
        source.resolve_thread()
        thread = source.list_thread_messages(page_size=50)
        chat = source.list_chat_messages(page_size=20)

    assert thread.page_token == "next" and thread.has_more is True
    assert chat.has_more is False
    assert api.list_requests[0].queries == [
        ("container_id_type", "thread"),
        ("container_id", "omt_1"),
        ("sort_type", "ByCreateTimeAsc"),
        ("page_size", "50"),
        ("card_msg_content_type", "user_card_content"),
    ]
    assert api.list_requests[1].queries == [
        ("container_id_type", "chat"),
        ("container_id", "oc_1"),
        ("sort_type", "ByCreateTimeDesc"),
        ("page_size", "20"),
        ("card_msg_content_type", "user_card_content"),
    ]


def test_cursor_must_be_nonempty_and_cannot_restart() -> None:
    source, _, _, _ = _open_source(
        list_responses=[
            _Response(items=[_message()], has_more=True, page_token="next")
        ]
    )

    with source:
        source.resolve_thread()
        assert source.list_thread_messages().page_token == "next"
        with pytest.raises(ContextUnavailableError) as raised:
            source.list_thread_messages()

    assert raised.value.reason is ContextUnavailableReason.PAGINATION


@pytest.mark.parametrize("next_token", ["", "same"])
def test_cursor_must_advance_without_cycles(next_token: str) -> None:
    responses = [
        _Response(items=[_message()], has_more=True, page_token="same")
    ]
    if next_token == "same":
        responses.append(
            _Response(
                items=[_message("om_second", position=2, message_position=12)],
                has_more=True,
                page_token="same",
            )
        )
    else:
        responses[0] = _Response(
            items=[_message()],
            has_more=True,
            page_token="",
        )
    source, _, _, _ = _open_source(list_responses=responses)

    with source:
        source.resolve_thread()
        if next_token == "same":
            source.list_thread_messages()

            def call():
                return source.list_thread_messages(page_token="same")
        else:
            call = source.list_thread_messages
        with pytest.raises(ContextUnavailableError) as raised:
            call()

    assert raised.value.reason is ContextUnavailableReason.PAGINATION


@pytest.mark.parametrize(
    "messages",
    [
        [
            _message("om_second", position=2, message_position=12),
            _message("om_first", position=1, message_position=11),
        ],
        [
            _message("om_same", position=1, message_position=11),
            _message("om_same", position=2, message_position=12),
        ],
        [
            _message("om_first", position=1, message_position=11),
            _message("om_second", position=1, message_position=12),
        ],
    ],
)
def test_thread_order_and_deduplication_fail_closed(messages) -> None:
    source, _, _, _ = _open_source(
        list_responses=[_Response(items=messages)]
    )

    with source:
        source.resolve_thread()
        with pytest.raises(ContextUnavailableError) as raised:
            source.list_thread_messages()

    assert raised.value.reason is ContextUnavailableReason.ORDERING


def test_list_rejects_messages_from_another_chat() -> None:
    source, _, _, _ = _open_source(
        list_responses=[_Response(items=[_message(chat_id="oc_other")])]
    )

    with source:
        source.resolve_thread()
        with pytest.raises(ContextUnavailableError) as raised:
            source.list_thread_messages()

    assert raised.value.reason is ContextUnavailableReason.SCOPE


@pytest.mark.parametrize(
    ("msg_type", "content", "expected"),
    [
        ("text", {"text": "hello"}, "hello"),
        ("image", {"image_key": "img_sensitive_key"}, "[image]"),
        (
            "post",
            {
                "title": "Status",
                "content": [[
                    {"tag": "text", "text": "hello"},
                    {"tag": "media", "file_key": "file_sensitive_key"},
                ]],
            },
            "# Status\n\nhello[media:]",
        ),
    ],
)
def test_official_normalizer_flattens_text_and_redacts_attachment_keys(
    msg_type: str,
    content: object,
    expected: str,
) -> None:
    message = _message(msg_type=msg_type, content=content)
    source, _, _, _ = _open_source(
        list_responses=[_Response(items=[message])]
    )

    with source:
        source.resolve_thread()
        normalized = source.list_thread_messages().messages[0]

    assert normalized.text == expected
    assert "sensitive_key" not in repr(normalized)


@pytest.mark.parametrize(
    ("msg_type", "content"),
    [
        ("unknown_future_type", {"value": "opaque"}),
        ("text", {}),
        ("image", {}),
        ("interactive", {}),
    ],
)
def test_unknown_or_missing_official_content_fields_fail_closed(
    msg_type: str,
    content: object,
) -> None:
    source, _, _, _ = _open_source(
        list_responses=[
            _Response(items=[_message(msg_type=msg_type, content=content)])
        ]
    )

    with source:
        source.resolve_thread()
        with pytest.raises(ContextUnavailableError) as raised:
            source.list_thread_messages()

    assert raised.value.reason is ContextUnavailableReason.CONTENT
