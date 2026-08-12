from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest

from src.autonomous.ingress.models import EmployeeIngressMetadata
from src.autonomous.ingress.targeted_task import (
    MAX_TARGETED_TASK_DESCRIPTION_CHARS,
    TARGETED_TASK_DIGEST_VERSION,
    TargetedTaskParseResult,
    TargetedTaskState,
    is_group_slash_observation,
    parse_targeted_group_task,
    targeted_group_task_digest,
)

_TENANT_KEY = "tenant_1"
_BOT_OPEN_ID = "ou_current_employee_bot"
_MENTION_KEY = "@_user_1"
_RAW_CHAT_ID = "oc_team"
_RAW_MESSAGE_ID = "om_message"
_RAW_ROOT_ID = "om_root"


def _indexed(prefix: str, value: str) -> str:
    return prefix + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _metadata() -> EmployeeIngressMetadata:
    return EmployeeIngressMetadata(
        schema_version=1,
        envelope_id="ing_targeted_task",
        tenant_key=_TENANT_KEY,
        agent_id="agt_employee",
        bot_principal_id="bot_employee",
        app_id="cli_employee",
        channel_generation=7,
        connection_id="conn_employee",
        event_id="evt_targeted_task",
        message_id=_indexed("om_", _RAW_MESSAGE_ID),
        event_type="im.message.receive_v1",
        action_identity="",
        chat_id=_indexed("oc_", _RAW_CHAT_ID),
        thread_root_message_id=_indexed("om_", _RAW_ROOT_ID),
        sender_principal_id="ou_employee_app_owner",
        received_at="2026-08-12T00:00:00Z",
        semantic_digest="a" * 64,
        payload_sha256="a" * 64,
        payload_size_bytes=1,
        attachment_count=0,
        attachment_total_bytes=0,
    )


def _mention(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "key": _MENTION_KEY,
        "open_id": _BOT_OPEN_ID,
        "tenant_key": _TENANT_KEY,
    }
    value.update(overrides)
    return value


def _part(
    text: object = f"{_MENTION_KEY} /task ship the release",
    *,
    mentions: object = None,
    **overrides: object,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "type": "message",
        "message_type": "text",
        "chat_type": "group",
        "content": {"text": text},
        "sender_id": "ou_employee_app_owner",
        "sender_union_id": "on_owner",
        "sender_id_type": "open_id",
        "sender_type": "user",
        "sender_tenant_key": _TENANT_KEY,
        "feishu_thread_id": "",
        "remote_chat_id": _RAW_CHAT_ID,
        "remote_message_id": _RAW_MESSAGE_ID,
        "remote_root_id": _RAW_ROOT_ID,
        "mentions": (_mention(),) if mentions is None else mentions,
    }
    value.update(overrides)
    return value


def _parse(
    part: object,
    *,
    metadata: EmployeeIngressMetadata | None = None,
    expected_bot_open_id: str = _BOT_OPEN_ID,
    expected_tenant_key: str = _TENANT_KEY,
):
    return parse_targeted_group_task(
        metadata=_metadata() if metadata is None else metadata,
        part=part,
        expected_bot_open_id=expected_bot_open_id,
        expected_tenant_key=expected_tenant_key,
    )


@pytest.mark.parametrize(
    ("text", "description"),
    (
        (f"{_MENTION_KEY} /task ship the release", "ship the release"),
        (f"{_MENTION_KEY} /task keep exact  spacing inside", "keep exact  spacing inside"),
        (f"\t{_MENTION_KEY}\n/task\tfirst line\nsecond line  ", "first line\nsecond line"),
    ),
)
def test_unique_current_bot_task_returns_frozen_description_and_versioned_digest(
    text: str,
    description: str,
) -> None:
    result = _parse(_part(text))

    assert result.state is TargetedTaskState.TARGETED_VALID
    assert result.description == description
    expected = targeted_group_task_digest(description)
    assert expected == hashlib.sha256(f"{TARGETED_TASK_DIGEST_VERSION}\0{description}".encode("utf-8")).hexdigest()
    assert result.input_digest == expected
    assert description not in repr(result)
    with pytest.raises(FrozenInstanceError):
        result.description = "mutated"  # type: ignore[misc]


def test_exact_description_limit_is_accepted() -> None:
    description = "工" * MAX_TARGETED_TASK_DESCRIPTION_CHARS

    result = _parse(_part(f"{_MENTION_KEY} /task {description}"))

    assert result.state is TargetedTaskState.TARGETED_VALID
    assert result.description == description


def test_empty_remote_root_is_bound_to_empty_metadata_root() -> None:
    metadata = replace(_metadata(), thread_root_message_id="")

    result = _parse(_part(remote_root_id=""), metadata=metadata)

    assert result.state is TargetedTaskState.TARGETED_VALID


def test_result_and_digest_reject_inconsistent_plaintext() -> None:
    with pytest.raises(ValueError, match="trimmed"):
        targeted_group_task_digest(" task ")
    with pytest.raises(ValueError, match="digest"):
        TargetedTaskParseResult(
            TargetedTaskState.TARGETED_VALID,
            description="task",
            input_digest="0" * 64,
        )


@pytest.mark.parametrize(
    "text",
    (
        f"{_MENTION_KEY} /task",
        f"{_MENTION_KEY} /task \t\n",
        f"{_MENTION_KEY} /task {'x' * (MAX_TARGETED_TASK_DESCRIPTION_CHARS + 1)}",
    ),
)
def test_exact_task_with_invalid_description_is_safe_usage_candidate(text: str) -> None:
    result = _parse(_part(text))

    assert result.state is TargetedTaskState.TARGETED_INVALID
    assert result.description == ""
    assert result.input_digest == ""


@pytest.mark.parametrize("control", ("\x00", "\x1b", "\x7f", "\r"))
def test_targeted_task_rejects_description_control_characters(control: str) -> None:
    result = _parse(_part(f"{_MENTION_KEY} /task safe{control}unsafe"))

    assert result.state is TargetedTaskState.TARGETED_INVALID


@pytest.mark.parametrize(
    "text",
    (
        f"{_MENTION_KEY} hello",
        f"{_MENTION_KEY} /tasks show mine",
        f"{_MENTION_KEY} /taskfoo nope",
        f"{_MENTION_KEY} /help",
        f"{_MENTION_KEY} /role",
        f"{_MENTION_KEY} /Task wrong-case",
        f"hello {_MENTION_KEY} /task misplaced",
        f"/task mention-after-command {_MENTION_KEY}",
    ),
)
def test_non_task_text_and_other_slashes_are_not_targeted(text: str) -> None:
    assert _parse(_part(text)).state is TargetedTaskState.NOT_TARGETED


def test_missing_foreign_duplicate_and_all_mentions_fail_closed() -> None:
    cases = (
        _part(mentions=()),
        _part(mentions=(_mention(open_id="ou_other_employee_bot"),)),
        _part(mentions=(_mention(tenant_key="tenant_2"),)),
        _part(mentions=(_mention(key="@all"),)),
        _part(mentions=(_mention(key="@_user_x"),)),
        _part(mentions=(_mention(), _mention(key="@_user_2"))),
        _part(f"{_MENTION_KEY} {_MENTION_KEY} /task duplicate placeholder"),
        _part(f"{_MENTION_KEY} /task includes unbound @_user_2 placeholder"),
        _part(f"{_MENTION_KEY} /task includes unbound @all placeholder"),
        _part("/task placeholder missing from body"),
    )

    for index, part in enumerate(cases):
        assert _parse(part).state is TargetedTaskState.NOT_TARGETED, index


def test_mention_requires_the_exact_authoritative_field_set() -> None:
    missing = _mention()
    del missing["tenant_key"]
    extra = {**_mention(), "name": "Employee"}

    for mention in (missing, extra, "not-a-mapping"):
        result = _parse(_part(mentions=(mention,)))
        assert result.state is TargetedTaskState.NOT_TARGETED


def test_message_sender_tenant_and_remote_coordinates_are_bound_to_metadata() -> None:
    base = _metadata()
    cases = (
        (_part(), replace(base, event_type="im.message.recalled_v1"), _TENANT_KEY),
        (_part(), replace(base, action_identity="card-action"), _TENANT_KEY),
        (_part(type="card_action"), base, _TENANT_KEY),
        (_part(message_type="post"), base, _TENANT_KEY),
        (_part(chat_type="p2p"), base, _TENANT_KEY),
        (_part(sender_type="bot"), base, _TENANT_KEY),
        (_part(sender_id_type="union_id"), base, _TENANT_KEY),
        (_part(sender_id="ou_other_sender"), base, _TENANT_KEY),
        (_part(sender_tenant_key="tenant_2"), base, _TENANT_KEY),
        (_part(remote_chat_id="oc_other"), base, _TENANT_KEY),
        (_part(remote_message_id="om_other"), base, _TENANT_KEY),
        (_part(remote_root_id="om_other"), base, _TENANT_KEY),
        (_part(remote_chat_id=None), base, _TENANT_KEY),
        (_part(remote_message_id=None), base, _TENANT_KEY),
        (_part(remote_root_id=None), base, _TENANT_KEY),
        (_part(), base, "tenant_2"),
    )

    for index, (part, metadata, expected_tenant_key) in enumerate(cases):
        result = _parse(
            part,
            metadata=metadata,
            expected_tenant_key=expected_tenant_key,
        )
        assert result.state is TargetedTaskState.NOT_TARGETED, index


@pytest.mark.parametrize(
    "part",
    (
        None,
        "not-a-mapping",
        {},
        {**_part(), "content": "not-a-mapping"},
        {**_part(), "content": {}},
        {**_part(), "content": {"text": 123}},
    ),
)
def test_malformed_normalized_part_fails_closed(part: object) -> None:
    assert _parse(part).state is TargetedTaskState.NOT_TARGETED


@pytest.mark.parametrize("expected_bot_open_id", ("", "bot_internal", "ou_other"))
def test_only_expected_current_ready_bot_can_be_targeted(expected_bot_open_id: str) -> None:
    result = _parse(_part(), expected_bot_open_id=expected_bot_open_id)

    assert result.state is TargetedTaskState.NOT_TARGETED


@pytest.mark.parametrize(
    "text",
    (
        "/task direct",
        f"{_MENTION_KEY} /task addressed",
        f"{_MENTION_KEY} /tasks",
        f"{_MENTION_KEY} @_user_2 /help",
        "@all /role",
    ),
)
def test_group_slash_observation_detects_direct_and_mention_prefixed_commands(
    text: str,
) -> None:
    assert is_group_slash_observation(_part(text)) is True


@pytest.mark.parametrize(
    "text",
    ("ordinary task", f"{_MENTION_KEY} hello", f"hello {_MENTION_KEY} /task later"),
)
def test_group_slash_observation_leaves_ordinary_group_text_alone(text: str) -> None:
    assert is_group_slash_observation(_part(text)) is False
