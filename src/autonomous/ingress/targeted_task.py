"""Pure classification for explicitly targeted employee group tasks."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from .models import EmployeeIngressMetadata

MAX_TARGETED_TASK_DESCRIPTION_CHARS = 14_000
TARGETED_TASK_DIGEST_VERSION = "ghostap.targeted-group-task.v1"
TARGETED_TASK_INPUT_KIND = "targeted_group_task_v1"
_MESSAGE_RECEIVE_EVENT_TYPE = "im.message.receive_v1"
_MENTION_FIELDS = frozenset({"key", "open_id", "tenant_key"})
_MENTION_KEY_RE = re.compile(r"@_user_[0-9]+\Z")
_BODY_MENTION_RE = re.compile(r"@_user_[0-9]+|@all")
_LEADING_MENTION_RE = re.compile(r"(?:@_user_[0-9]+|@all)(?=\s|$)")


def _has_forbidden_control(value: str) -> bool:
    return any(
        unicodedata.category(character) == "Cc"
        and character not in {"\n", "\t"}
        for character in value
    )


class TargetedTaskState(StrEnum):
    """Fail-closed classification of one normalized employee ingress part."""

    INDETERMINATE = "indeterminate"
    NOT_TARGETED = "not_targeted"
    TARGETED_INVALID = "targeted_invalid"
    TARGETED_VALID = "targeted_valid"


@dataclass(frozen=True, slots=True)
class TargetedTaskParseResult:
    """Safe classifier output; plaintext descriptions never appear in reprs."""

    state: TargetedTaskState
    description: str = field(default="", repr=False)
    input_digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.state, TargetedTaskState):
            raise TypeError("state must be a TargetedTaskState")
        if self.state is TargetedTaskState.TARGETED_VALID:
            if (
                not isinstance(self.description, str)
                or not self.description
                or self.description != self.description.strip()
                or len(self.description) > MAX_TARGETED_TASK_DESCRIPTION_CHARS
                or _has_forbidden_control(self.description)
            ):
                raise ValueError("valid targeted task requires a bounded description")
            if self.input_digest != targeted_group_task_digest(self.description):
                raise ValueError("valid targeted task digest does not match description")
            return
        if self.description or self.input_digest:
            raise ValueError("non-valid targeted task cannot expose task input")


_NOT_TARGETED = TargetedTaskParseResult(TargetedTaskState.NOT_TARGETED)


def targeted_group_task_digest(description: str) -> str:
    """Return the versioned digest for a validated task description."""

    if (
        not isinstance(description, str)
        or not description
        or description != description.strip()
        or len(description) > MAX_TARGETED_TASK_DESCRIPTION_CHARS
        or _has_forbidden_control(description)
    ):
        raise ValueError("task description must be non-empty, trimmed, and bounded")
    material = f"{TARGETED_TASK_DIGEST_VERSION}\0{description}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _bound_coordinate(indexed: str, raw: object, prefix: str) -> bool:
    if not isinstance(raw, str):
        return False
    if not raw:
        return not indexed
    return indexed == prefix + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _has_bound_message_authority(
    metadata: EmployeeIngressMetadata,
    part: Mapping[str, object],
    *,
    expected_tenant_key: str,
) -> bool:
    if (
        metadata.event_type != _MESSAGE_RECEIVE_EVENT_TYPE
        or metadata.action_identity
        or not expected_tenant_key
        or metadata.tenant_key != expected_tenant_key
        or part.get("type") != "message"
        or part.get("message_type") != "text"
        or part.get("chat_type") != "group"
        or part.get("sender_type") != "user"
        or part.get("sender_id_type") != "open_id"
        or part.get("sender_id") != metadata.sender_principal_id
        or part.get("sender_tenant_key") != metadata.tenant_key
    ):
        return False
    return (
        _bound_coordinate(metadata.chat_id, part.get("remote_chat_id"), "oc_")
        and _bound_coordinate(
            metadata.message_id,
            part.get("remote_message_id"),
            "om_",
        )
        and _bound_coordinate(
            metadata.thread_root_message_id,
            part.get("remote_root_id"),
            "om_",
        )
    )


def _target_mention(
    part: Mapping[str, object],
    *,
    expected_bot_open_id: str,
    expected_tenant_key: str,
) -> str | None:
    mentions = part.get("mentions")
    if not isinstance(mentions, tuple) or len(mentions) != 1:
        return None
    mention = mentions[0]
    if not isinstance(mention, Mapping) or set(mention) != _MENTION_FIELDS:
        return None
    key = mention.get("key")
    if (
        not isinstance(expected_bot_open_id, str)
        or not expected_bot_open_id.startswith("ou_")
        or not isinstance(key, str)
        or _MENTION_KEY_RE.fullmatch(key) is None
        or mention.get("open_id") != expected_bot_open_id
        or mention.get("tenant_key") != expected_tenant_key
    ):
        return None
    return key


def parse_targeted_group_task(
    *,
    metadata: EmployeeIngressMetadata,
    part: Mapping[str, object] | object,
    expected_bot_open_id: str,
    expected_tenant_key: str,
) -> TargetedTaskParseResult:
    """Classify one normalized part against a caller-proven READY Bot identity.

    The caller proves that ``expected_bot_open_id`` belongs to the current
    READY generation.  This parser independently binds the receive event,
    sender, tenant, and encrypted remote coordinates back to ``metadata``.
    """

    if (
        not isinstance(metadata, EmployeeIngressMetadata)
        or not isinstance(part, Mapping)
        or not isinstance(expected_tenant_key, str)
        or not _has_bound_message_authority(
            metadata,
            part,
            expected_tenant_key=expected_tenant_key,
        )
    ):
        return _NOT_TARGETED
    mention_key = _target_mention(
        part,
        expected_bot_open_id=expected_bot_open_id,
        expected_tenant_key=expected_tenant_key,
    )
    if mention_key is None:
        return _NOT_TARGETED
    content = part.get("content")
    text = content.get("text") if isinstance(content, Mapping) else None
    if not isinstance(text, str) or text.count(mention_key) != 1 or _BODY_MENTION_RE.findall(text) != [mention_key]:
        return _NOT_TARGETED
    command = text.lstrip()
    if not command.startswith(mention_key):
        return _NOT_TARGETED
    command = command[len(mention_key) :].lstrip()
    if command == "/task":
        return TargetedTaskParseResult(TargetedTaskState.TARGETED_INVALID)
    if not command.startswith("/task") or not command[5:6].isspace():
        return _NOT_TARGETED
    description = command[5:].strip()
    if (
        not description
        or len(description) > MAX_TARGETED_TASK_DESCRIPTION_CHARS
        or _has_forbidden_control(description)
    ):
        return TargetedTaskParseResult(TargetedTaskState.TARGETED_INVALID)
    return TargetedTaskParseResult(
        TargetedTaskState.TARGETED_VALID,
        description=description,
        input_digest=targeted_group_task_digest(description),
    )


def is_group_slash_observation(part: Mapping[str, object] | object) -> bool:
    """Identify a slash command after zero or more leading mention tokens.

    This helper is deny-only: it never grants authority and intentionally
    accepts malformed/unbound mention metadata so suspicious slash-shaped
    observations are suppressed instead of entering an employee mailbox.
    """

    if (
        not isinstance(part, Mapping)
        or part.get("type") != "message"
        or part.get("chat_type") != "group"
    ):
        return False
    content = part.get("content")
    text = content.get("text") if isinstance(content, Mapping) else None
    if not isinstance(text, str):
        return False
    remainder = text.lstrip()
    while True:
        mention = _LEADING_MENTION_RE.match(remainder)
        if mention is None:
            break
        remainder = remainder[mention.end() :].lstrip()
    return remainder.startswith("/")


__all__ = [
    "MAX_TARGETED_TASK_DESCRIPTION_CHARS",
    "TARGETED_TASK_DIGEST_VERSION",
    "TARGETED_TASK_INPUT_KIND",
    "TargetedTaskParseResult",
    "TargetedTaskState",
    "is_group_slash_observation",
    "parse_targeted_group_task",
    "targeted_group_task_digest",
]
