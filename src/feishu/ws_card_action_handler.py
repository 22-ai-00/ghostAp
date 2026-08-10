"""Helpers for Feishu card action callback inspection."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any


def bind_managed_trust_revisions(
    card: dict[str, Any],
    *,
    group_revision: int,
    grant_revision: int,
) -> dict[str, Any]:
    """Copy a card and stamp every callback value with issued revisions."""

    if type(group_revision) is not int or group_revision < 1:
        raise ValueError("group_revision must be positive")
    if type(grant_revision) is not int or grant_revision < 1:
        raise ValueError("grant_revision must be positive")
    bound = deepcopy(card)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in tuple(value.items()):
                if (
                    key == "value"
                    and isinstance(child, dict)
                    and ("action" in child or "action_id" in child)
                ):
                    child["group_revision"] = group_revision
                    child["grant_revision"] = grant_revision
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(bound)
    return bound


def _extract_behavior_value(behavior: Any) -> Any:
    if isinstance(behavior, Mapping):
        return behavior.get("value")
    return getattr(behavior, "value", None)


def _raw_action_value(action: Any) -> Any:
    value = getattr(action, "value", None)
    behaviors = getattr(action, "behaviors", None)
    if isinstance(behaviors, list) and behaviors:
        behavior_value = _extract_behavior_value(behaviors[0])
        if behavior_value is not None:
            return behavior_value
    return value


class CardActionFailureAction(str, Enum):
    ACK_AND_IGNORE = "ack_and_ignore"
    REPLY_FAILURE_CARD = "reply_failure_card"
    RAISE = "raise"


@dataclass(frozen=True)
class CardActionErrorClassification:
    action: CardActionFailureAction
    phase: str
    user_reachable: bool


def classify_card_action_error(error: Exception, *, phase: str) -> CardActionErrorClassification:
    if phase in {"payload_parse", "dedup"}:
        return CardActionErrorClassification(CardActionFailureAction.ACK_AND_IGNORE, phase, False)
    if phase == "dispatch":
        return CardActionErrorClassification(CardActionFailureAction.REPLY_FAILURE_CARD, phase, True)
    return CardActionErrorClassification(CardActionFailureAction.RAISE, phase, False)


class CardActionInspector:
    """Pure helpers for extracting stable card action callback fields."""

    @classmethod
    def value_dict(cls, action: Any) -> dict[str, Any]:
        value_raw = _raw_action_value(action)
        if isinstance(value_raw, dict):
            return value_raw
        if isinstance(value_raw, str):
            try:
                parsed = json.loads(value_raw)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @classmethod
    def action_type(cls, action: Any) -> str:
        return str(cls.value_dict(action).get("action", "") or "")

    @classmethod
    def project_id(cls, action: Any) -> str | None:
        project_id = cls.value_dict(action).get("project_id")
        return project_id if isinstance(project_id, str) and project_id else None

    @classmethod
    def trust_revisions(cls, action: Any) -> tuple[int | None, int | None]:
        """Extract an optional immutable managed-group revision snapshot."""

        value = cls.value_dict(action)

        def revision(name: str) -> int | None:
            item = value.get(name)
            return item if type(item) is int and item > 0 else None

        return revision("group_revision"), revision("grant_revision")

    @classmethod
    def is_system_action(cls, action: Any) -> bool:
        from .action_registry import is_registered_action

        return is_registered_action(cls.action_type(action))

    @classmethod
    def dedup_fingerprint(cls, action: Any) -> str:
        payload: dict[str, Any] = {"value": cls._normalize_value(_raw_action_value(action))}
        for attr in ("option", "options", "form_value", "input_value"):
            extra = getattr(action, attr, None)
            if isinstance(extra, (str, int, float, bool, list, tuple, dict)):
                payload[attr] = cls._normalize_value(extra)

        try:
            canonical = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        except (TypeError, ValueError):
            canonical = str(payload)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value
