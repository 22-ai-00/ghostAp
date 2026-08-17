"""Pure deny-by-default authorization for inbound Feishu messages."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

from .config import (
    IngressAccessMode,
    SecurityFinding,
    SecuritySeverity,
)

if TYPE_CHECKING:
    from .feishu.slash_command_parser import CommandMatch
    from .trust.models import EffectiveTrust

AdminBootstrapScope = Literal["any_chat", "p2p_only"]

_MESSAGE_ID_RE = re.compile(r"^om_[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_CHAT_ID_RE = re.compile(r"^oc_[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_SENDER_ID_RE = re.compile(r"^ou_[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_SUPPORTED_CHAT_TYPES = frozenset({"p2p", "group"})


class AccessOperation(str, Enum):
    """The only ingress operations that can cross the authorization boundary."""

    NORMAL_MESSAGE = "normal_message"
    BOOTSTRAP_HELP = "bootstrap_help"
    BOOTSTRAP_ADMIN = "bootstrap_admin"
    ENROL_CURRENT_CHAT = "enrol_current_chat"


@dataclass(frozen=True, slots=True)
class IngressAccessRequest:
    """Request-scoped facts used by the pure access policy."""

    message_id: str
    sender_id: str
    chat_id: str
    chat_type: str
    command_match: CommandMatch | None


@dataclass(frozen=True, slots=True)
class AccessDecision:
    """Immutable authorization result for one inbound message."""

    allowed: bool
    operation: AccessOperation
    reason_code: str
    prospective_allowed: bool
    effective_trust: EffectiveTrust | None = None


@dataclass(frozen=True, slots=True)
class IngressAccessPolicy:
    """An immutable, complete snapshot of inbound access configuration."""

    admin_ids: frozenset[str]
    allowed_user_ids: frozenset[str]
    allowed_chat_ids: frozenset[str]
    mode: IngressAccessMode
    admin_bootstrap_scope: AdminBootstrapScope

    def __post_init__(self) -> None:
        if not isinstance(self.admin_ids, frozenset):
            raise TypeError("admin_ids must be a frozenset")
        if not isinstance(self.allowed_user_ids, frozenset):
            raise TypeError("allowed_user_ids must be a frozenset")
        if not isinstance(self.allowed_chat_ids, frozenset):
            raise TypeError("allowed_chat_ids must be a frozenset")
        if not isinstance(self.mode, IngressAccessMode):
            raise TypeError("mode must be an IngressAccessMode")
        if self.admin_bootstrap_scope not in {"any_chat", "p2p_only"}:
            raise ValueError("invalid admin_bootstrap_scope")

    def decide(self, request: IngressAccessRequest) -> AccessDecision:
        """Return a decision without reading files, logging, or mutating state."""

        if not are_canonical_ingress_facts(
            message_id=request.message_id,
            sender_id=request.sender_id,
            chat_id=request.chat_id,
            chat_type=request.chat_type,
        ):
            return AccessDecision(
                allowed=False,
                operation=AccessOperation.NORMAL_MESSAGE,
                reason_code="invalid_ingress_facts",
                prospective_allowed=False,
            )

        match = request.command_match
        command = match.command if match is not None else ""
        arguments = match.args.strip().casefold() if match is not None else ""

        bootstrap_scope_allowed = (
            self.admin_bootstrap_scope == "any_chat"
            or request.chat_type == "p2p"
        )
        if (
            self.mode is IngressAccessMode.ENFORCED
            and not self.admin_ids
            and command == "/help"
            and not arguments
            and request.chat_type == "p2p"
        ):
            return AccessDecision(
                allowed=True,
                operation=AccessOperation.BOOTSTRAP_HELP,
                reason_code="bootstrap_help",
                prospective_allowed=False,
            )
        if (
            not self.admin_ids
            and command == "/setadmin"
            and bootstrap_scope_allowed
        ):
            return AccessDecision(
                allowed=True,
                operation=AccessOperation.BOOTSTRAP_ADMIN,
                reason_code="bootstrap_admin",
                prospective_allowed=True,
            )

        if (
            request.sender_id in self.admin_ids
            and request.chat_type == "group"
            and bool(request.chat_id)
            and command == "/access"
            and arguments == "allow-chat"
        ):
            return AccessDecision(
                allowed=True,
                operation=AccessOperation.ENROL_CURRENT_CHAT,
                reason_code="admin_chat_enrolment",
                prospective_allowed=True,
            )

        user_allowed = (
            request.sender_id in self.admin_ids
            or request.sender_id in self.allowed_user_ids
        )
        chat_allowed = request.chat_id in self.allowed_chat_ids
        prospective_allowed = user_allowed and chat_allowed

        if self.mode is IngressAccessMode.LEGACY_ALLOW_ALL:
            return AccessDecision(
                allowed=True,
                operation=AccessOperation.NORMAL_MESSAGE,
                reason_code="legacy_allow_all",
                prospective_allowed=prospective_allowed,
            )
        if self.mode is IngressAccessMode.SHADOW:
            return AccessDecision(
                allowed=True,
                operation=AccessOperation.NORMAL_MESSAGE,
                reason_code=(
                    "shadow_allowed"
                    if prospective_allowed
                    else "shadow_would_deny"
                ),
                prospective_allowed=prospective_allowed,
            )
        return AccessDecision(
            allowed=prospective_allowed,
            operation=AccessOperation.NORMAL_MESSAGE,
            reason_code=(
                "allowed" if prospective_allowed else "access_not_enrolled"
            ),
            prospective_allowed=prospective_allowed,
        )


def is_canonical_message_id(value: object) -> bool:
    """Return whether *value* is a canonical Feishu message identifier."""

    return isinstance(value, str) and _MESSAGE_ID_RE.fullmatch(value) is not None


def is_canonical_chat_id(value: object) -> bool:
    """Return whether *value* is a canonical Feishu chat identifier."""

    return isinstance(value, str) and _CHAT_ID_RE.fullmatch(value) is not None


def is_canonical_sender_id(value: object) -> bool:
    """Return whether *value* is a canonical Feishu open identifier."""

    return isinstance(value, str) and _SENDER_ID_RE.fullmatch(value) is not None


def is_supported_chat_type(value: object) -> bool:
    """Only actual Feishu P2P and group message contexts are supported."""

    return isinstance(value, str) and value in _SUPPORTED_CHAT_TYPES


def are_canonical_ingress_facts(
    *,
    message_id: object,
    sender_id: object,
    chat_id: object,
    chat_type: object,
) -> bool:
    """Validate the complete request trust root before parsing message content."""

    return (
        is_canonical_message_id(message_id)
        and is_canonical_sender_id(sender_id)
        and is_canonical_chat_id(chat_id)
        and is_supported_chat_type(chat_type)
    )


class IngressAccessPolicyProvider:
    """Publish complete immutable snapshots with one atomic reference swap."""

    def __init__(self, initial: IngressAccessPolicy) -> None:
        if not isinstance(initial, IngressAccessPolicy):
            raise TypeError("initial must be an IngressAccessPolicy")
        self._current = initial
        self._blocking_findings: tuple[SecurityFinding, ...] = ()
        self._swap_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

    @property
    def current(self) -> IngressAccessPolicy:
        return self._current

    def get(self) -> IngressAccessPolicy:
        """Compatibility-friendly explicit snapshot accessor."""

        return self._current

    @property
    def blocking_findings(self) -> tuple[SecurityFinding, ...]:
        return self._blocking_findings

    def swap(self, replacement: IngressAccessPolicy) -> None:
        if not isinstance(replacement, IngressAccessPolicy):
            raise TypeError("replacement must be an IngressAccessPolicy")
        with self._swap_lock:
            self._current = replacement

    def record_blocking_finding(self, code: str, message: str) -> None:
        finding = SecurityFinding(
            code=code,
            severity=SecuritySeverity.BLOCKING,
            message=message,
        )
        with self._swap_lock:
            retained = tuple(
                item
                for item in self._blocking_findings
                if item.code != finding.code
            )
            self._blocking_findings = (*retained, finding)

    def clear_blocking_finding(self, code: str) -> None:
        with self._swap_lock:
            self._blocking_findings = tuple(
                finding
                for finding in self._blocking_findings
                if finding.code != code
            )


def normalize_access_ids(value: object) -> frozenset[str]:
    """Normalize Settings values and lightweight test doubles consistently."""

    if isinstance(value, str):
        return frozenset(
            item
            for part in value.split(",")
            if (item := part.strip())
        )
    try:
        return frozenset(
            item
            for part in (value or ())
            if (item := str(part).strip())
        )
    except TypeError:
        return frozenset()


def build_ingress_access_policy(settings: object) -> IngressAccessPolicy:
    """Build a complete immutable policy snapshot from Settings-like data."""

    raw_mode = getattr(
        settings,
        "ingress_access_mode",
        IngressAccessMode.ENFORCED.value,
    )
    if isinstance(raw_mode, IngressAccessMode):
        mode = raw_mode
    elif isinstance(raw_mode, str):
        mode = IngressAccessMode(raw_mode)
    else:
        # Lightweight test doubles and partial compatibility settings must
        # fall back secure; arbitrary object stringification is never config.
        mode = IngressAccessMode.ENFORCED
    raw_scope = getattr(settings, "admin_bootstrap_scope", "p2p_only")
    scope: AdminBootstrapScope
    if raw_scope == "any_chat":
        scope = "any_chat"
    elif raw_scope == "p2p_only":
        scope = "p2p_only"
    elif not isinstance(raw_scope, str):
        scope = "p2p_only"
    else:
        raise ValueError("invalid admin_bootstrap_scope")
    return IngressAccessPolicy(
        admin_ids=normalize_access_ids(
            getattr(settings, "admin_user_ids", frozenset())
        ),
        allowed_user_ids=normalize_access_ids(
            getattr(settings, "allowed_user_ids", frozenset())
        ),
        allowed_chat_ids=normalize_access_ids(
            getattr(settings, "allowed_chat_ids", frozenset())
        ),
        mode=mode,
        admin_bootstrap_scope=scope,
    )
