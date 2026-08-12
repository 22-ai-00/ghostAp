"""Authority-bound durable Router for encrypted employee Inbox records.

This module deliberately stops at a durable dispatch grant.  The Phase 6
Gateway owns attempt anchoring and ACP execution; raw Channel payload values
can only make this Router reject an acceptance, never grant authority.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any, Callable, Mapping, Protocol

from ...trust.models import ActorKind, TrustZone
from ...trust.registry import ManagedGroupRegistry
from ...trust.resolver import TrustZoneResolver
from ..authorization import EmployeeAuthorizationScope
from ..context.models import AuthorizedContextRequest
from ..domain import EmployeeState, WorkerType
from ..journal.blob_store import BlobRef
from ..journal.frame import (
    GENESIS_HASH,
    JournalEvent,
    JournalIntegrityError,
    TransactionFrame,
)
from ..journal.writer import CommitState, JournalDeadlineExceededError, JournalWriter
from ..supervisor.channel_models import ChannelProcessState
from ..workforce.projection import is_workforce_event
from ..workforce.registry import (
    ProjectedAgentRegistry,
    ProjectedContextBinding,
)
from .models import EmployeeIngressMetadata, EmployeeIngressPayload
from .service import (
    EmployeeIngressService,
    IngressBlobError,
    IngressBlobRetryableError,
)
from .targeted_task import (
    TARGETED_TASK_INPUT_KIND,
    TargetedTaskParseResult,
    TargetedTaskState,
    is_group_slash_observation,
    parse_targeted_group_task,
)

_ROUTER_PREFIX = "employee.ingress.router_"
_ROUTER_EVENTS = frozenset(
    {
        _ROUTER_PREFIX + "authorized",
        _ROUTER_PREFIX + "staging",
        _ROUTER_PREFIX + "queued",
        _ROUTER_PREFIX + "inbox_retry",
        _ROUTER_PREFIX + "context_retry",
        _ROUTER_PREFIX + "dispatching",
        _ROUTER_PREFIX + "terminal",
    }
)

_INBOX_MAX_FAILURES = 3
_HANDOFF_ABANDON_MAX_RETRIES = 4
_AUTHORITY_DEPENDENCY_UNAVAILABLE = "authority_dependency_unavailable"



def _canonical_utc(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("Router UTC timestamp requires an aware datetime")
    text = value.astimezone(UTC).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )
    return text.replace(".000000Z", "Z")


def _parse_canonical_utc(value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("Router timestamp must be canonical UTC")
    parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    if _canonical_utc(parsed) != value:
        raise ValueError("Router timestamp must be canonical UTC")
    return parsed


def _is_eligible(record: "RouterLifecycleRecord", now: datetime) -> bool:
    if not record.next_eligible_at:
        return True
    return _parse_canonical_utc(record.next_eligible_at) <= now.astimezone(UTC)


def _is_inbox_eligible(record: "RouterLifecycleRecord", now: datetime) -> bool:
    if not record.inbox_next_eligible_at:
        return True
    return _parse_canonical_utc(record.inbox_next_eligible_at) <= now.astimezone(UTC)


def _remote_index(value: str, prefix: str) -> str:
    return prefix + hashlib.sha256(value.encode()).hexdigest()


def _bound_remote_coordinate(
    indexed_value: str,
    raw_value: object,
    prefix: str,
) -> str | None:
    """Return a raw encrypted coordinate only when it matches its public index."""

    if raw_value is None:
        # Synthetic/internal ingress predating dual-coordinate payloads keeps
        # raw coordinates in metadata. Real SDK ingress always supplies raw.
        return indexed_value
    if not isinstance(raw_value, str):
        return None
    if not raw_value:
        return "" if not indexed_value else None
    return raw_value if _remote_index(raw_value, prefix) == indexed_value else None


def _bound_encrypted_remote_coordinate(
    indexed_value: str,
    raw_value: object,
    prefix: str,
) -> str | None:
    """Bind a public message coordinate without treating its hash as raw data."""

    if not isinstance(raw_value, str):
        return None
    if not raw_value:
        return "" if not indexed_value else None
    return raw_value if _remote_index(raw_value, prefix) == indexed_value else None


def _same_app_requester_principal(
    *,
    tenant_key: str,
    agent_id: str,
    owner_principal_id: str,
    sender_principal_id: str,
    sender_union_id: str,
    **_transport: object,
) -> str | None:
    """Preserve the historical same-app requester identity by default."""

    del tenant_key, agent_id, owner_principal_id, sender_union_id
    return sender_principal_id or None


_TERMINAL_REASONS = frozenset(
    {
        "authority_denied",
        "authority_stale",
        "requester_denied",
        "sender_invalid",
        "bot_loop",
        "membership_degraded",
        "card_action_unsupported",
        "unsupported_event",
        "attachment_staging_unavailable",
        "attachment_staging_failed",
        "queue_full",
        "queue_rebalanced",
        "context_coordinates_invalid",
        "inbox_not_dispatchable",
        "handoff_unconfirmed",
        "completed",
        "failed",
        "canceled",
        "timeout",
        "action_required",
        "team_unavailable",
        "context_unavailable",
        "canonical_context_unavailable",
        "team_step_inactive",
        "team_step_expired",
        "team_step_canceled",
        "team_assignment_invalid",
    }
)
_DISPATCH_TERMINAL_REASONS = frozenset(
    {"completed", "failed", "canceled", "timeout", "action_required"}
)
_PREQUEUE_ONLY_TERMINAL_REASONS = frozenset({"handoff_unconfirmed"})
_DISPATCH_REJECTION_REASONS = frozenset(
    {
        "team_unavailable",
        "context_unavailable",
        "canonical_context_unavailable",
        "team_step_inactive",
        "team_step_expired",
        "team_step_canceled",
        "team_assignment_invalid",
    }
)
_LEGACY_REPLAY_TERMINAL_REASONS = frozenset({"control_consumed"})


class RouterProjectionError(RuntimeError):
    """The anchored Router history violates its exact lifecycle contract."""


class RouterWriteDisabledError(RuntimeError):
    """A Router transition did not reach the monotonic Journal anchor."""


class RequesterAclPort(Protocol):
    def is_authorized(self, request: AuthorizedContextRequest) -> bool: ...


class ChannelStatusPort(Protocol):
    def status(self, agent_id: str) -> Any: ...


class MembershipHealthPort(Protocol):
    """Required deny-only live membership health signal."""

    def is_degraded(self, agent_id: str, team_id: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class RouterQueueLimits:
    per_employee: int
    per_team: int
    global_limit: int

    def __post_init__(self) -> None:
        values = (self.per_employee, self.per_team, self.global_limit)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError("Router queue limits must be integers")
        if not (1 <= self.per_employee <= self.per_team <= self.global_limit):
            raise ValueError("Router queue limits require per_employee <= per_team <= global")


@dataclass(frozen=True, slots=True)
class RouterAuthoritySnapshot:
    """Frozen projected/live authority used for every later revalidation."""

    tenant_key: str
    agent_id: str
    bot_principal_id: str
    app_id: str
    channel_generation: int
    connection_id: str
    authorization_scope: EmployeeAuthorizationScope
    team_id: str
    requester_principal_id: str
    projection_sequence: int
    projection_hash: str
    employee_version: int
    tool: str
    model: str
    effort: str
    constraints_digest: str = ""
    system_prompt_token_reserve: int = 0
    effective_input_kind: str = ""
    effective_input_digest: str = ""
    target_bot_open_id_digest: str = ""

    _FIELDS = frozenset(
        {
            "tenant_key",
            "agent_id",
            "bot_principal_id",
            "app_id",
            "channel_generation",
            "connection_id",
            "authorization_scope",
            "team_id",
            "requester_principal_id",
            "projection_sequence",
            "projection_hash",
            "employee_version",
            "tool",
            "model",
            "effort",
            "constraints_digest",
            "system_prompt_token_reserve",
            "effective_input_kind",
            "effective_input_digest",
            "target_bot_open_id_digest",
        }
    )

    def __post_init__(self) -> None:
        required = (
            self.tenant_key,
            self.agent_id,
            self.bot_principal_id,
            self.app_id,
            self.connection_id,
            self.team_id,
            self.requester_principal_id,
            self.tool,
            self.effort,
        )
        if not all(isinstance(value, str) and value for value in required):
            raise ValueError("Router authority snapshot contains blank identity")
        if (
            not isinstance(self.model, str)
            or self.model != self.model.strip()
            or len(self.model) > 4096
            or any(ord(character) < 32 for character in self.model)
        ):
            raise ValueError("Router authority snapshot contains invalid model")
        if not isinstance(
            self.authorization_scope,
            EmployeeAuthorizationScope,
        ):
            raise TypeError("invalid Router authorization scope")
        integers = (
            self.channel_generation,
            self.projection_sequence,
            self.employee_version,
            self.system_prompt_token_reserve,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in integers):
            raise ValueError("Router authority snapshot contains invalid integer")
        if self.channel_generation < 1:
            raise ValueError("Router authority channel generation must be positive")
        if self.projection_hash and (
            len(self.projection_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.projection_hash)
        ):
            raise ValueError("Router projection hash must be lowercase SHA-256")
        if self.constraints_digest and (
            len(self.constraints_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.constraints_digest)
        ):
            raise ValueError("Router constraints digest must be lowercase SHA-256")
        if self.system_prompt_token_reserve and not self.constraints_digest:
            raise ValueError("Router reserve requires constraints digest")
        if not all(
            isinstance(value, str)
            for value in (
                self.effective_input_kind,
                self.effective_input_digest,
                self.target_bot_open_id_digest,
            )
        ):
            raise ValueError("Router effective input binding must be text")
        if (
            self.effective_input_kind != self.effective_input_kind.strip()
            or self.effective_input_digest != self.effective_input_digest.strip()
            or self.target_bot_open_id_digest
            != self.target_bot_open_id_digest.strip()
        ):
            raise ValueError("Router effective input binding must be trimmed")
        if not (
            bool(self.effective_input_kind)
            == bool(self.effective_input_digest)
            == bool(self.target_bot_open_id_digest)
        ):
            raise ValueError("Router effective input binding is incomplete")
        if self.effective_input_kind and (
            self.effective_input_kind != TARGETED_TASK_INPUT_KIND
            or self.authorization_scope
            is not EmployeeAuthorizationScope.MANAGED_GROUP
            or any(
                len(digest) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in digest
                )
                for digest in (
                    self.effective_input_digest,
                    self.target_bot_open_id_digest,
                )
            )
        ):
            raise ValueError("Router effective input binding is invalid")

    def to_dict(self) -> dict[str, object]:
        result = {name: getattr(self, name) for name in sorted(self._FIELDS)}
        result["authorization_scope"] = self.authorization_scope.value
        return result

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        allow_legacy_replay: bool = False,
    ) -> RouterAuthoritySnapshot:
        if not isinstance(value, dict):
            raise ValueError("Router authority snapshot must use exact schema")
        normalized = dict(value)
        previous_fields = cls._FIELDS - {
            "effective_input_kind",
            "effective_input_digest",
            "target_bot_open_id_digest",
        }
        oldest_fields = previous_fields - {"authorization_scope"}
        if allow_legacy_replay and frozenset(normalized) in {
            frozenset(previous_fields),
            frozenset(oldest_fields),
        }:
            normalized.setdefault(
                "authorization_scope",
                EmployeeAuthorizationScope.MANAGED_GROUP.value,
            )
            normalized["effective_input_kind"] = ""
            normalized["effective_input_digest"] = ""
            normalized["target_bot_open_id_digest"] = ""
        if set(normalized) != cls._FIELDS:
            raise ValueError("Router authority snapshot must use exact schema")
        try:
            normalized["authorization_scope"] = EmployeeAuthorizationScope(
                normalized["authorization_scope"]
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid Router authorization scope") from exc
        return cls(**normalized)


@dataclass(frozen=True, slots=True)
class _AuthorityResolution:
    """Ephemeral validated authority plus its non-serializable credential ref."""

    snapshot: RouterAuthoritySnapshot
    credential_ref: str = field(repr=False)
    targeted_task: TargetedTaskParseResult | None = field(
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.credential_ref, str) or not self.credential_ref:
            raise ValueError("authority credential ref is required")


@dataclass(frozen=True, slots=True)
class RouterResponseObligation:
    """Minimal reply target frozen atomically with Employee queue ownership."""

    schema_version: int
    chat_id: str
    reply_to_message_id: str
    reply_coordinate_kind: str
    authority_binding_sha256: str

    _FIELDS = frozenset(
        {
            "schema_version",
            "chat_id",
            "reply_to_message_id",
            "reply_coordinate_kind",
            "authority_binding_sha256",
        }
    )

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported Router response obligation schema")
        if (
            not isinstance(self.chat_id, str)
            or not self.chat_id.startswith("oc_")
            or len(self.chat_id) > 256
        ):
            raise ValueError("invalid Router response chat")
        if (
            not isinstance(self.reply_to_message_id, str)
            or not self.reply_to_message_id.startswith("om_")
            or len(self.reply_to_message_id) > 256
        ):
            raise ValueError("invalid Router response message")
        if self.reply_coordinate_kind not in {"root", "message"}:
            raise ValueError("invalid Router response coordinate kind")
        if (
            not isinstance(self.authority_binding_sha256, str)
            or len(self.authority_binding_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.authority_binding_sha256
            )
        ):
            raise ValueError("invalid Router response authority binding")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in sorted(self._FIELDS)}

    def is_bound_to(
        self,
        record: RouterLifecycleRecord,
        authority: RouterAuthoritySnapshot,
    ) -> bool:
        """Verify this target against the accepted indexes and frozen authority."""

        return _response_obligation_matches_record(record, authority, self)

    @classmethod
    def from_dict(cls, value: object) -> RouterResponseObligation:
        if not isinstance(value, dict) or set(value) != cls._FIELDS:
            raise ValueError("Router response obligation must use exact schema")
        return cls(**value)

    @classmethod
    def create(
        cls,
        *,
        authority: RouterAuthoritySnapshot,
        acceptance_id: str,
        ingress_aggregate_id: str,
        envelope_id: str,
        event_type: str,
        chat_id: str,
        message_id: str,
        thread_root_message_id: str,
    ) -> RouterResponseObligation:
        reply_kind = "root" if thread_root_message_id else "message"
        reply_to = thread_root_message_id or message_id
        return cls(
            schema_version=1,
            chat_id=chat_id,
            reply_to_message_id=reply_to,
            reply_coordinate_kind=reply_kind,
            authority_binding_sha256=_response_authority_binding_sha256(
                acceptance_id=acceptance_id,
                ingress_aggregate_id=ingress_aggregate_id,
                envelope_id=envelope_id,
                event_type=event_type,
                authority=authority,
            ),
        )


def _response_authority_binding_sha256(
    *,
    acceptance_id: str,
    ingress_aggregate_id: str,
    envelope_id: str,
    event_type: str,
    authority: RouterAuthoritySnapshot,
) -> str:
    binding = {
        "acceptance_id": acceptance_id,
        "authority": authority.to_dict(),
        "envelope_id": envelope_id,
        "event_type": event_type,
        "ingress_aggregate_id": ingress_aggregate_id,
    }
    encoded = json.dumps(
        binding,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(
        b"ghostap.employee-router-response-authority.v1\0" + encoded
    ).hexdigest()


def _response_coordinate_matches_index(
    indexed_value: str,
    raw_value: str,
    prefix: str,
) -> bool:
    return _remote_index(raw_value, prefix) == indexed_value


def _response_obligation_matches_record(
    record: RouterLifecycleRecord,
    authority: RouterAuthoritySnapshot,
    obligation: RouterResponseObligation,
) -> bool:
    expected_binding = _response_authority_binding_sha256(
        acceptance_id=record.acceptance_id,
        ingress_aggregate_id=record.aggregate_id,
        envelope_id=record.envelope_id,
        event_type=record.event_type,
        authority=authority,
    )
    if (
        obligation.authority_binding_sha256 != expected_binding
        or obligation.chat_id != authority.team_id
        or not _response_coordinate_matches_index(
            record.indexed_chat_id,
            obligation.chat_id,
            "oc_",
        )
    ):
        return False
    if obligation.reply_coordinate_kind == "root":
        return bool(record.indexed_thread_root_message_id) and (
            _response_coordinate_matches_index(
                record.indexed_thread_root_message_id,
                obligation.reply_to_message_id,
                "om_",
            )
        )
    return not record.indexed_thread_root_message_id and (
        _response_coordinate_matches_index(
            record.message_id,
            obligation.reply_to_message_id,
            "om_",
        )
    )


@dataclass(frozen=True, slots=True)
class RouterLifecycleRecord:
    aggregate_id: str
    acceptance_id: str
    envelope_id: str
    tenant_key: str
    agent_id: str
    bot_principal_id: str
    app_id: str
    channel_generation: int
    connection_id: str
    team_id: str
    message_id: str
    event_type: str
    requester_principal_id: str
    state: str
    accepted_sequence: int
    authority: RouterAuthoritySnapshot | None = None
    indexed_chat_id: str = ""
    indexed_thread_root_message_id: str = ""
    response_obligation: RouterResponseObligation | None = field(
        default=None,
        repr=False,
    )
    queue_position: int = 0
    queued_sequence: int = 0
    inbox_failures: int = 0
    inbox_next_eligible_at: str = ""
    context_failures: int = 0
    next_eligible_at: str = ""
    reason_code: str = ""


@dataclass(slots=True)
class RouterProjectionState:
    by_acceptance_id: dict[str, RouterLifecycleRecord] = field(default_factory=dict)
    ignored_legacy_acceptances: dict[str, RouterLifecycleRecord] = field(
        default_factory=dict
    )
    cursor_sequence: int = 0
    cursor_hash: str = ""

    def clone(self) -> RouterProjectionState:
        return copy.deepcopy(self)


@dataclass(frozen=True, slots=True)
class RouterDispatchGrant:
    record: RouterLifecycleRecord
    request: AuthorizedContextRequest
    payload: EmployeeIngressPayload = field(repr=False)
    targeted_task: TargetedTaskParseResult | None = field(
        default=None,
        repr=False,
    )


def _accepted_record(
    event: JournalEvent,
    sequence: int,
) -> RouterLifecycleRecord:
    payload = event.payload
    current_fields = {
        "metadata",
        "acceptance_id",
        "accepted_at",
        "blob_ref",
        "transport_message_proof",
    }
    legacy_fields = current_fields - {"transport_message_proof"}
    if (
        frozenset(payload)
        not in {
            frozenset(current_fields),
            frozenset(legacy_fields),
        }
        or type(payload.get("transport_message_proof", False)) is not bool
    ):
        raise RouterProjectionError("invalid employee ingress acceptance")
    try:
        metadata = EmployeeIngressMetadata.from_dict(payload["metadata"])
        BlobRef.from_dict(payload["blob_ref"])
    except (TypeError, ValueError) as exc:
        raise RouterProjectionError("invalid employee ingress metadata") from exc
    acceptance_id = payload.get("acceptance_id")
    if not isinstance(acceptance_id, str) or not acceptance_id.startswith("acc_"):
        raise RouterProjectionError("invalid employee ingress acceptance identity")
    if event.aggregate_id != metadata.dedup_key:
        raise RouterProjectionError("Router acceptance aggregate mismatch")
    if not isinstance(payload["accepted_at"], str) or not payload["accepted_at"]:
        raise RouterProjectionError("invalid employee ingress acceptance timestamp")
    return RouterLifecycleRecord(
        aggregate_id=event.aggregate_id,
        acceptance_id=acceptance_id,
        envelope_id=metadata.envelope_id,
        tenant_key=metadata.tenant_key,
        agent_id=metadata.agent_id,
        bot_principal_id=metadata.bot_principal_id,
        app_id=metadata.app_id,
        channel_generation=metadata.channel_generation,
        connection_id=metadata.connection_id,
        team_id=metadata.chat_id,
        message_id=metadata.message_id,
        event_type=metadata.event_type,
        requester_principal_id=metadata.sender_principal_id,
        state="accepted",
        accepted_sequence=sequence,
        indexed_chat_id=metadata.chat_id,
        indexed_thread_root_message_id=metadata.thread_root_message_id,
    )


def _reduce_router_event(
    state: RouterProjectionState,
    event: JournalEvent,
    *,
    sequence: int,
    allow_legacy_replay: bool = False,
) -> None:
    payload = event.payload
    acceptance_id = payload.get("acceptance_id") if isinstance(payload, dict) else None
    record = state.by_acceptance_id.get(acceptance_id or "")
    if record is None or event.aggregate_id != record.aggregate_id:
        raise RouterProjectionError("Router transition references unknown acceptance")
    if event.event_type == _ROUTER_PREFIX + "authorized":
        current_payload = {
            "acceptance_id",
            "authority",
            "source_requester_principal_id",
        }
        legacy_payload = {"acceptance_id", "authority"}
        is_legacy_replay = allow_legacy_replay and set(payload) == legacy_payload
        if (
            set(payload) != current_payload
            and not is_legacy_replay
        ) or record.state != "accepted":
            raise RouterProjectionError("invalid Router authorized transition")
        try:
            authority = RouterAuthoritySnapshot.from_dict(
                payload["authority"],
                allow_legacy_replay=allow_legacy_replay,
            )
        except (TypeError, ValueError) as exc:
            raise RouterProjectionError("invalid Router authority snapshot") from exc
        accepted_coordinates = (
            record.tenant_key,
            record.agent_id,
            record.bot_principal_id,
            record.app_id,
            record.channel_generation,
            record.connection_id,
        )
        authority_coordinates = (
            authority.tenant_key,
            authority.agent_id,
            authority.bot_principal_id,
            authority.app_id,
            authority.channel_generation,
            authority.connection_id,
        )
        team_matches = record.team_id == authority.team_id or (
            record.team_id == _remote_index(authority.team_id, "oc_")
        )
        source_requester_principal_id = (
            record.requester_principal_id
            if is_legacy_replay
            else payload["source_requester_principal_id"]
        )
        requester_matches = (
            source_requester_principal_id == record.requester_principal_id
        )
        if (
            authority_coordinates != accepted_coordinates
            or not team_matches
            or not requester_matches
        ):
            raise RouterProjectionError("Router authority acceptance mismatch")
        updated = replace(
            record,
            state="authorized",
            team_id=authority.team_id,
            requester_principal_id=authority.requester_principal_id,
            authority=authority,
        )
    elif event.event_type == _ROUTER_PREFIX + "staging":
        if set(payload) != {"acceptance_id"} or record.state != "authorized":
            raise RouterProjectionError("invalid Router staging transition")
        updated = replace(record, state="staging")
    elif event.event_type == _ROUTER_PREFIX + "queued":
        current_fields = {
            "acceptance_id",
            "authority",
            "queue_position",
            "response_obligation",
        }
        legacy_fields = current_fields - {"response_obligation"}
        is_legacy_replay = allow_legacy_replay and set(payload) == legacy_fields
        if (
            set(payload) != current_fields
            and not is_legacy_replay
        ) or record.state != "staging":
            raise RouterProjectionError("invalid Router queued transition")
        try:
            authority = RouterAuthoritySnapshot.from_dict(
                payload["authority"],
                allow_legacy_replay=allow_legacy_replay,
            )
        except (TypeError, ValueError) as exc:
            raise RouterProjectionError("invalid queued Router authority") from exc
        position = payload["queue_position"]
        if authority != record.authority or isinstance(position, bool) or not isinstance(position, int) or position < 1:
            raise RouterProjectionError("invalid Router queue disposition")
        response_obligation = None
        if not is_legacy_replay:
            raw_obligation = payload["response_obligation"]
            if record.event_type == "im.message.receive_v1":
                try:
                    response_obligation = RouterResponseObligation.from_dict(
                        raw_obligation
                    )
                except (TypeError, ValueError) as exc:
                    raise RouterProjectionError(
                        "invalid Router response obligation"
                    ) from exc
                if not _response_obligation_matches_record(
                    record,
                    authority,
                    response_obligation,
                ):
                    raise RouterProjectionError(
                        "Router response obligation does not match acceptance"
                    )
            elif raw_obligation is not None:
                raise RouterProjectionError(
                    "non-message Router work cannot carry a response obligation"
                )
        updated = replace(
            record,
            state="queued",
            queue_position=position,
            queued_sequence=sequence,
            response_obligation=response_obligation,
        )
    elif event.event_type == _ROUTER_PREFIX + "dispatching":
        if set(payload) != {"acceptance_id"} or record.state != "queued":
            raise RouterProjectionError("invalid Router dispatch transition")
        updated = replace(record, state="dispatching")
    elif event.event_type == _ROUTER_PREFIX + "inbox_retry":
        if set(payload) != {
            "acceptance_id",
            "failure_count",
            "next_eligible_at",
        } or record.state not in {"accepted", "authorized", "staging", "queued"}:
            raise RouterProjectionError("invalid Router inbox retry transition")
        failure_count = payload["failure_count"]
        if (
            isinstance(failure_count, bool)
            or not isinstance(failure_count, int)
            or failure_count != record.inbox_failures + 1
        ):
            raise RouterProjectionError("invalid Router inbox retry count")
        try:
            canonical_eligibility = _canonical_utc(
                _parse_canonical_utc(payload["next_eligible_at"])
            )
        except (TypeError, ValueError) as exc:
            raise RouterProjectionError(
                "invalid Router inbox retry eligibility"
            ) from exc
        updated = replace(
            record,
            inbox_failures=failure_count,
            inbox_next_eligible_at=canonical_eligibility,
        )
    elif event.event_type == _ROUTER_PREFIX + "context_retry":
        if set(payload) != {
            "acceptance_id",
            "failure_count",
            "next_eligible_at",
        } or record.state != "queued":
            raise RouterProjectionError("invalid Router context retry transition")
        failure_count = payload["failure_count"]
        if (
            isinstance(failure_count, bool)
            or not isinstance(failure_count, int)
            or failure_count != record.context_failures + 1
        ):
            raise RouterProjectionError("invalid Router context retry count")
        try:
            canonical_eligibility = _canonical_utc(
                _parse_canonical_utc(payload["next_eligible_at"])
            )
        except (TypeError, ValueError) as exc:
            raise RouterProjectionError("invalid Router context retry eligibility") from exc
        updated = replace(
            record,
            context_failures=failure_count,
            next_eligible_at=canonical_eligibility,
        )
    elif event.event_type == _ROUTER_PREFIX + "terminal":
        if set(payload) != {"acceptance_id", "reason_code"} or record.state not in {
            "accepted",
            "authorized",
            "staging",
            "queued",
            "dispatching",
        }:
            raise RouterProjectionError("invalid Router terminal transition")
        reason = payload["reason_code"]
        if reason not in _TERMINAL_REASONS and not (
            allow_legacy_replay
            and reason in _LEGACY_REPLAY_TERMINAL_REASONS
        ):
            raise RouterProjectionError("invalid Router terminal reason")
        if (
            reason in _PREQUEUE_ONLY_TERMINAL_REASONS
            and record.state not in {"accepted", "authorized", "staging"}
        ):
            raise RouterProjectionError(
                "Router terminal reason does not match lifecycle"
            )
        if (record.state == "dispatching") != (
            reason in _DISPATCH_TERMINAL_REASONS
        ):
            raise RouterProjectionError("Router terminal reason does not match lifecycle")
        updated = replace(record, state="terminal", reason_code=reason)
    else:
        raise RouterProjectionError("unknown Router event")
    state.by_acceptance_id[acceptance_id] = updated


def _apply_router_frame_events(
    state: RouterProjectionState,
    frame: TransactionFrame,
    *,
    allow_legacy_replay: bool = False,
) -> None:
    """Project one frame while tombstoning unproven legacy public history."""

    for event in frame.events:
        if event.event_type == "employee.ingress.accepted":
            record = _accepted_record(event, frame.sequence)
            if (
                record.acceptance_id in state.by_acceptance_id
                or record.acceptance_id in state.ignored_legacy_acceptances
            ):
                raise RouterProjectionError("duplicate Router acceptance")
            if (
                record.event_type == "im.message.receive_v1"
                and event.payload.get("transport_message_proof", False) is not True
            ):
                state.ignored_legacy_acceptances[record.acceptance_id] = record
            else:
                state.by_acceptance_id[record.acceptance_id] = record
        elif event.event_type in _ROUTER_EVENTS:
            acceptance_id = (
                event.payload.get("acceptance_id")
                if isinstance(event.payload, dict)
                else None
            )
            ignored_record = state.ignored_legacy_acceptances.get(
                acceptance_id or ""
            )
            if ignored_record is not None:
                if event.aggregate_id != ignored_record.aggregate_id:
                    raise RouterProjectionError(
                        "Router transition references unknown acceptance"
                    )
                ignored_state = RouterProjectionState(
                    by_acceptance_id={ignored_record.acceptance_id: ignored_record}
                )
                _reduce_router_event(
                    ignored_state,
                    event,
                    sequence=frame.sequence,
                    allow_legacy_replay=allow_legacy_replay,
                )
                state.ignored_legacy_acceptances[ignored_record.acceptance_id] = (
                    ignored_state.by_acceptance_id[ignored_record.acceptance_id]
                )
                continue
            _reduce_router_event(
                state,
                event,
                sequence=frame.sequence,
                allow_legacy_replay=allow_legacy_replay,
            )
        elif event.event_type.startswith(_ROUTER_PREFIX):
            raise RouterProjectionError("unknown Router event")


class DurableEmployeeIngressRouter:
    """Consume only projected Inbox records plus authenticated decrypted blobs."""

    def __init__(
        self,
        *,
        writer: JournalWriter,
        ingress_service: EmployeeIngressService,
        registry_provider: Callable[[], ProjectedAgentRegistry],
        channel_status_provider: ChannelStatusPort,
        requester_acl: RequesterAclPort,
        queue_limits: RouterQueueLimits,
        membership_health: MembershipHealthPort,
        attachment_staging: Any | None = None,
        constraints_digest: str = "",
        system_prompt_token_reserve: int = 0,
        context_retry_base_seconds: float = 1.0,
        context_retry_max_seconds: float = 30.0,
        clock: Callable[[], datetime] | None = None,
        requester_principal_resolver: Callable[..., str | None] | None = None,
        managed_group_registry_provider: Callable[[], ManagedGroupRegistry] | None = None,
        managed_group_owner_id: str = "",
        employee_bot_ids_provider: Callable[[], frozenset[str]] | None = None,
    ) -> None:
        if not isinstance(writer, JournalWriter):
            raise TypeError("writer must be JournalWriter")
        if not isinstance(ingress_service, EmployeeIngressService):
            raise TypeError("ingress_service must be EmployeeIngressService")
        if not isinstance(queue_limits, RouterQueueLimits):
            raise TypeError("queue_limits must be RouterQueueLimits")
        if not callable(registry_provider) or not hasattr(channel_status_provider, "status"):
            raise TypeError("Router authority providers are invalid")
        if not hasattr(requester_acl, "is_authorized"):
            raise TypeError("requester_acl is invalid")
        if requester_principal_resolver is not None and not callable(
            requester_principal_resolver
        ):
            raise TypeError("requester_principal_resolver is invalid")
        if managed_group_registry_provider is not None and not callable(
            managed_group_registry_provider
        ):
            raise TypeError("managed_group_registry_provider is invalid")
        if employee_bot_ids_provider is not None and not callable(
            employee_bot_ids_provider
        ):
            raise TypeError("employee_bot_ids_provider is invalid")
        if not callable(getattr(membership_health, "is_degraded", None)):
            raise TypeError("membership_health is invalid")
        # Reuse the request contract for strict trusted reserve validation.
        if system_prompt_token_reserve and not constraints_digest:
            raise ValueError("non-zero reserve requires constraints_digest")
        if constraints_digest and (
            len(constraints_digest) != 64
            or any(character not in "0123456789abcdef" for character in constraints_digest)
        ):
            raise ValueError("constraints_digest must be lowercase SHA-256")
        for name, value in (
            ("context_retry_base_seconds", context_retry_base_seconds),
            ("context_retry_max_seconds", context_retry_max_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not (0 < float(value) < float("inf")):
                raise ValueError(f"{name} must be finite and positive")
        if context_retry_base_seconds > context_retry_max_seconds:
            raise ValueError("context retry base cannot exceed maximum")
        self._writer = writer
        self._ingress = ingress_service
        self._registry_provider = registry_provider
        self._channels = channel_status_provider
        self._requester_acl = requester_acl
        self._requester_principal_resolver = requester_principal_resolver
        self._managed_group_registry_provider = managed_group_registry_provider
        self._managed_group_owner_id = managed_group_owner_id
        self._employee_bot_ids_provider = employee_bot_ids_provider
        self._limits = queue_limits
        self._attachment_staging = attachment_staging
        self._membership_health = membership_health
        self._constraints_digest = constraints_digest
        self._reserve = system_prompt_token_reserve
        self._context_retry_base_seconds = float(context_retry_base_seconds)
        self._context_retry_max_seconds = float(context_retry_max_seconds)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._mutex = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        self._state = RouterProjectionState()
        self.rebuild_projection()

    @property
    def state(self) -> RouterProjectionState:
        return self._state

    def record_snapshot(
        self,
        acceptance_id: str,
        *,
        deadline: float | None = None,
        allow_immediate: bool = False,
    ) -> RouterLifecycleRecord | None:
        """Return one immutable in-memory record without replaying the Journal."""

        if not isinstance(acceptance_id, str) or not acceptance_id:
            return None
        hard_deadline = self._validated_handoff_deadline(deadline)
        if not self._acquire_handoff_mutex(
            hard_deadline,
            allow_immediate=allow_immediate,
        ):
            raise RouterWriteDisabledError(
                "Router snapshot deadline expired before Router lock"
            )
        try:
            return self._state.by_acceptance_id.get(acceptance_id)
        finally:
            self._mutex.release()

    @contextmanager
    def _ingress_dispatch_guard(self) -> Iterator[None]:
        """Join the parent Ingress tier; callers use the Ingress service guard."""

        with self._mutex:
            yield

    def synchronize_projection_unlocked(self) -> RouterProjectionState:
        """Refresh while the caller owns the combined Ingress tier."""

        return self.rebuild_projection()

    def preflight_dispatch_event_unlocked(
        self,
        *,
        acceptance_id: str,
    ) -> JournalEvent:
        record = self._record(acceptance_id)
        if record.state != "queued":
            raise RouterProjectionError("only queued Router work can dispatch")
        if any(
            other.state == "dispatching" and other.agent_id == record.agent_id
            for other in self._state.by_acceptance_id.values()
        ):
            raise RouterProjectionError("employee already has dispatching work")
        event = JournalEvent(
            event_type=_ROUTER_PREFIX + "dispatching",
            aggregate_id=record.aggregate_id,
            payload={"acceptance_id": record.acceptance_id},
        )
        probe = self._state.clone()
        _reduce_router_event(
            probe,
            event,
            sequence=self._state.cursor_sequence + 1,
        )
        return event

    def preflight_terminal_event_unlocked(
        self,
        *,
        acceptance_id: str,
        reason_code: str,
    ) -> JournalEvent:
        if reason_code not in _DISPATCH_TERMINAL_REASONS:
            raise ValueError("invalid dispatch terminal reason")
        record = self._record(acceptance_id)
        if record.state == "terminal":
            if record.reason_code != reason_code:
                raise RouterProjectionError("terminal Router result conflicts with replay")
        elif record.state != "dispatching":
            raise RouterProjectionError("only dispatching Router work can finish")
        event = JournalEvent(
            event_type=_ROUTER_PREFIX + "terminal",
            aggregate_id=record.aggregate_id,
            payload={
                "acceptance_id": record.acceptance_id,
                "reason_code": reason_code,
            },
        )
        if record.state != "terminal":
            probe = self._state.clone()
            _reduce_router_event(
                probe,
                event,
                sequence=self._state.cursor_sequence + 1,
            )
        return event

    def preflight_rejection_event_unlocked(
        self,
        *,
        acceptance_id: str,
        reason_code: str,
    ) -> JournalEvent:
        """Build a queued rejection for a caller holding the Ingress tier."""

        if reason_code not in _DISPATCH_REJECTION_REASONS:
            raise ValueError("invalid dispatch rejection reason")
        record = self._record(acceptance_id)
        if record.state != "queued":
            raise RouterProjectionError("only queued Router work can be rejected")
        event = JournalEvent(
            event_type=_ROUTER_PREFIX + "terminal",
            aggregate_id=record.aggregate_id,
            payload={
                "acceptance_id": record.acceptance_id,
                "reason_code": reason_code,
            },
        )
        probe = self._state.clone()
        _reduce_router_event(
            probe,
            event,
            sequence=self._state.cursor_sequence + 1,
        )
        return event

    def preflight_frame_unlocked(self, frame: TransactionFrame) -> None:
        if not frame.committed:
            raise RouterProjectionError("Router frame must be committed")
        if frame.sequence != self._state.cursor_sequence + 1:
            raise RouterProjectionError("Router frame sequence is not continuous")
        expected_previous = self._state.cursor_hash or GENESIS_HASH
        if frame.previous_hash != expected_previous:
            raise RouterProjectionError("Router frame previous hash mismatch")
        probe = self._state.clone()
        _apply_router_frame_events(probe, frame)

    def apply_committed_frame_unlocked(self, frame: TransactionFrame) -> None:
        self.preflight_frame_unlocked(frame)
        _apply_router_frame_events(self._state, frame)
        self._state.cursor_sequence = frame.sequence
        self._state.cursor_hash = frame.frame_hash

    def rebuild_projection(
        self,
        *,
        deadline: float | None = None,
    ) -> RouterProjectionState:
        hard_deadline = self._validated_handoff_deadline(deadline)
        if not self._acquire_handoff_mutex(hard_deadline):
            raise RouterWriteDisabledError(
                "Router projection deadline expired before Router lock"
            )
        try:
            self._ensure_handoff_deadline(hard_deadline)
            fresh = RouterProjectionState()
            anchor = self._writer.anchor.read()
            self._ensure_handoff_deadline(hard_deadline)
            cursor_hash = "" if anchor.sequence == 0 else anchor.frame_hash
            if getattr(self, "_projection_verified", False) and (
                self._state.cursor_sequence,
                self._state.cursor_hash,
            ) == (anchor.sequence, cursor_hash):
                return self._state
            anchored_hash = GENESIS_HASH
            replay = (
                self._writer.replay()
                if hard_deadline is None
                else self._writer.replay(deadline=hard_deadline)
            )
            for frame in replay:
                self._ensure_handoff_deadline(hard_deadline)
                if frame.sequence > anchor.sequence:
                    break
                _apply_router_frame_events(
                    fresh,
                    frame,
                    allow_legacy_replay=True,
                )
                fresh.cursor_sequence = frame.sequence
                fresh.cursor_hash = frame.frame_hash
                anchored_hash = frame.frame_hash
            if anchored_hash != anchor.frame_hash:
                raise RouterWriteDisabledError("Router projection cannot verify Journal anchor")
            self._ensure_handoff_deadline(hard_deadline)
            self._state = fresh
            self._projection_verified = True
            return self._state
        except JournalDeadlineExceededError as exc:
            raise RouterWriteDisabledError(
                "Router projection deadline expired"
            ) from exc
        finally:
            self._mutex.release()

    def is_inbox_candidate_eligible(self, acceptance_id: str) -> bool:
        """Return whether pre-dispatch payload work may read its Blob now."""

        with self._mutex:
            self.rebuild_projection()
            record = self._record(acceptance_id)
            if record.state not in {"accepted", "authorized", "staging"}:
                return False
            now = self._clock()
            if now.tzinfo is None or now.utcoffset() is None:
                raise ValueError("Router clock must return an aware datetime")
            return _is_inbox_eligible(record, now)

    def defer_inbox_candidate(
        self,
        acceptance_id: str,
        *,
        max_failures: int = _INBOX_MAX_FAILURES,
    ) -> RouterLifecycleRecord:
        """Persist one payload availability failure and bound retries durably."""

        if isinstance(max_failures, bool) or not isinstance(max_failures, int):
            raise TypeError("max_failures must be an integer")
        if max_failures < 1:
            raise ValueError("max_failures must be positive")
        with self._mutex, self._writer.transaction_guard():
            self.rebuild_projection()
            record = self._record(acceptance_id)
            if record.state == "terminal":
                return record
            if record.state not in {"accepted", "authorized", "staging", "queued"}:
                raise RouterProjectionError(
                    "only pre-dispatch Router work can defer an Inbox read"
                )
            failure_count = record.inbox_failures + 1
            if failure_count >= max_failures:
                return self._terminal_unlocked(record, "inbox_not_dispatchable")
            delay = min(
                self._context_retry_max_seconds,
                self._context_retry_base_seconds * (2 ** (failure_count - 1)),
            )
            eligible_at = _canonical_utc(self._clock() + timedelta(seconds=delay))
            return self._transition_unlocked(
                record,
                "inbox_retry",
                {
                    "failure_count": failure_count,
                    "next_eligible_at": eligible_at,
                },
            )

    def route(self, acceptance_id: str) -> RouterLifecycleRecord:
        try:
            return self._route(acceptance_id)
        finally:
            self._sweep_terminal_attachments()

    def _route(self, acceptance_id: str) -> RouterLifecycleRecord:
        """Authorize, stage, and atomically admit one accepted Inbox record."""

        with self._mutex:
            self.rebuild_projection()
            current = self._record(acceptance_id)
            if current.state in {"queued", "dispatching", "terminal"}:
                return current
            now = self._clock()
            if now.tzinfo is None or now.utcoffset() is None:
                raise ValueError("Router clock must return an aware datetime")
            if not _is_inbox_eligible(current, now):
                return current
        try:
            record, ingress_record, payload = self._dispatch_snapshot(acceptance_id)
        except IngressBlobRetryableError:
            return self.defer_inbox_candidate(acceptance_id)
        except IngressBlobError:
            return self._terminal_inbox_failure(acceptance_id)
        if record.state in {"queued", "dispatching", "terminal"}:
            return record
        accepted_identity = self._ingress_identity(ingress_record, payload)
        if ingress_record.metadata.event_type == "card.action.trigger":
            return self._terminal_for_snapshot(
                acceptance_id,
                accepted_identity,
                "card_action_unsupported",
            )
        if ingress_record.metadata.event_type not in {
            "im.message.receive_v1",
            "ghostap.team.assignment.v1",
        }:
            return self._terminal_for_snapshot(
                acceptance_id,
                accepted_identity,
                "unsupported_event",
            )

        # Registry, Channel, membership, and ACL ports may block.  Sample them
        # before entering the short Journal commit section below.
        resolution, reason = self._resolve_authority(
            ingress_record.metadata,
            payload,
        )
        if resolution is None and reason == _AUTHORITY_DEPENDENCY_UNAVAILABLE:
            return self.defer_inbox_candidate(acceptance_id)
        if (
            resolution is not None
            and resolution.targeted_task is not None
            and resolution.targeted_task.state
            is TargetedTaskState.TARGETED_INVALID
        ):
            return self._terminal_for_snapshot(
                acceptance_id,
                accepted_identity,
                "sender_invalid",
            )
        try:
            with self._ingress.dispatch_snapshot_guard(acceptance_id) as (
                current_ingress,
                current_payload,
            ), self._mutex, self._writer.transaction_guard():
                self.rebuild_projection()
                record = self._record(acceptance_id)
                if record.state in {"queued", "dispatching", "terminal"}:
                    return record
                if self._ingress_identity(current_ingress, current_payload) != accepted_identity:
                    return self._terminal_unlocked(record, "inbox_not_dispatchable")
                if resolution is None:
                    return self._terminal_unlocked(record, reason)
                authority = resolution.snapshot
                if self._projection_missed_workforce_change(
                    authority.projection_sequence,
                    authority.projection_hash,
                ):
                    return self._terminal_unlocked(record, "authority_stale")
                if record.state == "accepted":
                    record = self._transition_unlocked(
                        record,
                        "authorized",
                        {
                            "authority": authority.to_dict(),
                            "source_requester_principal_id": (
                                record.requester_principal_id
                            ),
                        },
                    )
                elif not self._authority_matches(record.authority, authority):
                    return self._terminal_unlocked(record, "authority_stale")
        except IngressBlobRetryableError:
            return self.defer_inbox_candidate(acceptance_id)
        except IngressBlobError:
            return self._terminal_inbox_failure(acceptance_id)

        if record.state == "authorized" and payload.attachment_descriptors:
            if self._attachment_staging is None:
                return self._terminal_after_external(
                    acceptance_id,
                    "attachment_staging_unavailable",
                )
            staging_resolution, staging_reason = self._resolve_authority(
                ingress_record.metadata,
                payload,
            )
            if staging_resolution is None:
                if staging_reason == _AUTHORITY_DEPENDENCY_UNAVAILABLE:
                    return self.defer_inbox_candidate(acceptance_id)
                return self._terminal_after_external(
                    acceptance_id,
                    staging_reason,
                )
            if (
                not self._authority_matches(
                    record.authority,
                    staging_resolution.snapshot,
                )
                or staging_resolution.credential_ref != resolution.credential_ref
            ):
                return self._terminal_after_external(
                    acceptance_id,
                    "authority_stale",
                )
            try:
                self._stage_attachments(
                    acceptance_id,
                    ingress_record.metadata,
                    payload,
                    staging_resolution,
                )
            except Exception:
                return self._terminal_after_external(
                    acceptance_id,
                    "attachment_staging_failed",
                )

        try:
            record, current_ingress, current_payload = self._dispatch_snapshot(
                acceptance_id
            )
        except IngressBlobRetryableError:
            return self.defer_inbox_candidate(acceptance_id)
        except IngressBlobError:
            return self._terminal_inbox_failure(acceptance_id)
        if record.state in {"queued", "dispatching", "terminal"}:
            return record
        current_identity = self._ingress_identity(current_ingress, current_payload)
        if current_identity != accepted_identity:
            return self._terminal_for_snapshot(
                acceptance_id,
                current_identity,
                "inbox_not_dispatchable",
            )
        current_resolution, current_reason = self._resolve_authority(
            current_ingress.metadata,
            current_payload,
        )
        if (
            current_resolution is None
            and current_reason == _AUTHORITY_DEPENDENCY_UNAVAILABLE
        ):
            return self.defer_inbox_candidate(acceptance_id)
        current = (
            current_resolution.snapshot
            if current_resolution is not None
            else None
        )
        credential_changed = (
            bool(payload.attachment_descriptors)
            and current_resolution is not None
            and current_resolution.credential_ref != resolution.credential_ref
        )
        try:
            with self._ingress.dispatch_snapshot_guard(acceptance_id) as (
                final_ingress,
                final_payload,
            ), self._mutex, self._writer.transaction_guard():
                self.rebuild_projection()
                record = self._record(acceptance_id)
                if record.state in {"queued", "dispatching", "terminal"}:
                    return record
                if self._ingress_identity(final_ingress, final_payload) != accepted_identity:
                    return self._terminal_unlocked(record, "inbox_not_dispatchable")
                if (
                    current is None
                    or credential_changed
                    or not self._authority_matches(record.authority, current)
                ):
                    return self._terminal_unlocked(
                        record,
                        "authority_stale" if current is not None else current_reason,
                    )
                if self._projection_missed_workforce_change(
                    current.projection_sequence,
                    current.projection_hash,
                ):
                    return self._terminal_unlocked(record, "authority_stale")
                if record.state == "authorized":
                    record = self._transition_unlocked(record, "staging", {})
                if self._queue_full_unlocked(record.authority):
                    victim = self._rebalance_victim_unlocked(record.authority)
                    if victim is None:
                        return self._terminal_unlocked(record, "queue_full")
                    return self._rebalance_and_queue_unlocked(
                        record,
                        victim,
                        response_obligation=self._response_obligation(
                            record,
                            final_ingress.metadata,
                            final_payload,
                        ),
                    )
                position = 1 + sum(
                    candidate.state == "queued"
                    and candidate.authority is not None
                    and candidate.authority.agent_id == record.authority.agent_id
                    for candidate in self._state.by_acceptance_id.values()
                )
                return self._transition_unlocked(
                    record,
                    "queued",
                    {
                        "authority": record.authority.to_dict(),
                        "queue_position": position,
                        "response_obligation": self._response_obligation(
                            record,
                            final_ingress.metadata,
                            final_payload,
                        ),
                    },
                )
        except IngressBlobRetryableError:
            return self.defer_inbox_candidate(acceptance_id)
        except IngressBlobError:
            return self._terminal_inbox_failure(acceptance_id)

    def reject_dispatch_candidate(
        self,
        acceptance_id: str,
        *,
        reason_code: str,
    ) -> RouterLifecycleRecord:
        """Durably reject queued work that cannot enter the unique coordinator."""

        if reason_code not in _DISPATCH_REJECTION_REASONS:
            raise ValueError("invalid dispatch rejection reason")
        with self._mutex, self._writer.transaction_guard():
            self.rebuild_projection()
            record = self._record(acceptance_id)
            if record.state == "terminal":
                if record.reason_code != reason_code:
                    raise RouterProjectionError("Router rejection conflicts with terminal")
                return record
            if record.state != "queued":
                raise RouterProjectionError("only queued Router work can be rejected")
            return self._terminal_unlocked(record, reason_code)

    def abandon_message_handoff(
        self,
        acceptance_id: str,
        *,
        channel_generation: int,
        connection_id: str,
        deadline: float | None = None,
    ) -> RouterLifecycleRecord:
        """Durably stop work that has not committed to the Employee queue.

        A concurrent route or dispatch that already crossed into ``queued``
        wins; callers must then treat the Employee lane as authoritative.
        Stale transport callers can observe the current record but cannot
        mutate it.
        """

        if type(channel_generation) is not int or channel_generation <= 0:
            raise ValueError("channel_generation must be a positive integer")
        if not isinstance(connection_id, str) or not connection_id.startswith(
            "conn_"
        ):
            raise ValueError("connection_id is invalid")
        hard_deadline = self._validated_handoff_deadline(deadline)
        last_conflict: JournalIntegrityError | None = None
        alias_witness: tuple[str, int, str] | None = None

        for _attempt in range(_HANDOFF_ABANDON_MAX_RETRIES):
            self._ensure_handoff_deadline(
                hard_deadline,
                cause=last_conflict,
            )

            # Journal replay may scan the entire history.  Keep it outside the
            # cross-domain writer guard so one handoff cannot stall unrelated
            # ingress, Outbox, or lifecycle writers.
            if not self._acquire_handoff_mutex(hard_deadline):
                raise RouterWriteDisabledError(
                    "Router handoff abandon deadline expired before Router lock"
                ) from last_conflict
            try:
                self.rebuild_projection(deadline=hard_deadline)
                record = self._record(acceptance_id)
                canonical_transport = (
                    record.channel_generation,
                    record.connection_id,
                )
                if (
                    record.state not in {"accepted", "authorized", "staging"}
                ):
                    return record
                router_identity = (
                    record.tenant_key,
                    record.agent_id,
                    record.bot_principal_id,
                    record.app_id,
                    record.event_type,
                )
            finally:
                self._mutex.release()

            if canonical_transport != (channel_generation, connection_id):
                # A reconnect/redelivery keeps the canonical Router record but
                # adds an append-only exact transport witness in Ingress.  Do
                # this cross-domain lookup without the Router mutex so the
                # global lock order remains Ingress -> Router.
                canonical_ingress = self._ingress.record_snapshot(
                    acceptance_id,
                    **(
                        {}
                        if hard_deadline is None
                        else {"deadline": hard_deadline}
                    ),
                )
                metadata = getattr(canonical_ingress, "metadata", None)
                if metadata is None or (
                    metadata.tenant_key,
                    metadata.agent_id,
                    metadata.bot_principal_id,
                    metadata.app_id,
                    metadata.event_type,
                ) != router_identity:
                    raise RouterProjectionError(
                        "Router handoff canonical Ingress identity mismatch"
                    )
                outcome = self._ingress.observe_anchored_message_acceptance(
                    tenant_key=metadata.tenant_key,
                    agent_id=metadata.agent_id,
                    bot_principal_id=metadata.bot_principal_id,
                    app_id=metadata.app_id,
                    event_type=metadata.event_type,
                    chat_id=metadata.chat_id,
                    message_id=metadata.message_id,
                    channel_generation=channel_generation,
                    connection_id=connection_id,
                    **(
                        {}
                        if hard_deadline is None
                        else {"deadline": hard_deadline}
                    ),
                )
                witnessed_acceptance = getattr(outcome, "acceptance", None)
                if (
                    getattr(outcome, "status", None) != "accepted"
                    or getattr(witnessed_acceptance, "acceptance_id", None)
                    != acceptance_id
                    or getattr(outcome, "channel_generation", None)
                    != channel_generation
                    or getattr(outcome, "connection_id", None) != connection_id
                ):
                    if not self._acquire_handoff_mutex(hard_deadline):
                        raise RouterWriteDisabledError(
                            "Router handoff alias recheck deadline expired"
                        )
                    try:
                        return self._record(acceptance_id)
                    finally:
                        self._mutex.release()
                alias_witness = (
                    acceptance_id,
                    channel_generation,
                    connection_id,
                )

            if not self._acquire_handoff_mutex(hard_deadline):
                raise RouterWriteDisabledError(
                    "Router handoff abandon deadline expired before Router lock"
                ) from last_conflict
            try:
                # The witness is append-only, but the Router lane may have
                # changed while the cross-domain lookup ran.  Rebuild on the
                # next loop if its cursor moved.
                record = self._record(acceptance_id)
                if record.state not in {"accepted", "authorized", "staging"}:
                    return record
                if (
                    (record.channel_generation, record.connection_id)
                    != (channel_generation, connection_id)
                    and alias_witness
                    != (acceptance_id, channel_generation, connection_id)
                ):
                    return record
                prepared_cursor = (
                    self._state.cursor_sequence,
                    self._state.cursor_hash,
                )
                event = JournalEvent(
                    event_type=_ROUTER_PREFIX + "terminal",
                    aggregate_id=record.aggregate_id,
                    payload={
                        "acceptance_id": record.acceptance_id,
                        "reason_code": "handoff_unconfirmed",
                    },
                )
                # Clone/preflight can scale with the whole Router projection;
                # keep that work out of the global writer transaction guard.
                preflight = self._state.clone()
                _reduce_router_event(
                    preflight,
                    event,
                    sequence=self._state.cursor_sequence + 1,
                )
            finally:
                self._mutex.release()

            self._ensure_handoff_deadline(
                hard_deadline,
                cause=last_conflict,
            )

            if not self._acquire_handoff_mutex(hard_deadline):
                raise RouterWriteDisabledError(
                    "Router handoff abandon deadline expired before Router lock"
                ) from last_conflict
            try:
                # Another operation on this Router may have refreshed or
                # advanced the projection while the mutex was released.
                if prepared_cursor != (
                    self._state.cursor_sequence,
                    self._state.cursor_hash,
                ):
                    continue
                record = self._record(acceptance_id)
                if (
                    record.state not in {"accepted", "authorized", "staging"}
                    or (
                        (
                            record.channel_generation,
                            record.connection_id,
                        )
                        != (channel_generation, connection_id)
                        and alias_witness
                        != (acceptance_id, channel_generation, connection_id)
                    )
                ):
                    return record
                self._ensure_handoff_deadline(
                    hard_deadline,
                    cause=last_conflict,
                )
                try:
                    # This is intentionally the only writer-guarded section:
                    # preflight, aggregate-version read, CAS commit, and the
                    # in-memory reducer are all bounded by one small frame.
                    guard = (
                        self._writer.transaction_guard()
                        if hard_deadline is None
                        else self._writer.transaction_guard(
                            deadline=hard_deadline,
                        )
                    )
                    with guard:
                        self._ensure_handoff_deadline(hard_deadline)
                        versions = self._writer.get_aggregate_versions(
                            (event.aggregate_id,),
                            **(
                                {}
                                if hard_deadline is None
                                else {"deadline": hard_deadline}
                            ),
                        )
                        result = self._writer.commit(
                            (event,),
                            versions,
                            expected_head_sequence=self._state.cursor_sequence,
                            expected_head_hash=self._state.cursor_hash or None,
                            **(
                                {}
                                if hard_deadline is None
                                else {"deadline": hard_deadline}
                            ),
                        )
                        if result.state is not CommitState.ANCHORED:
                            raise RouterWriteDisabledError(
                                "Router handoff abandon was not anchored"
                            )
                        _reduce_router_event(
                            self._state,
                            event,
                            sequence=result.frame.sequence,
                        )
                        self._state.cursor_sequence = result.frame.sequence
                        self._state.cursor_hash = result.frame.frame_hash
                        return self._record(acceptance_id)
                except JournalIntegrityError as exc:
                    # A different Journal aggregate may have advanced the head
                    # after the lock-free replay.  Refresh outside the writer
                    # guard and re-evaluate whether queue ownership won.
                    last_conflict = exc
                except JournalDeadlineExceededError as exc:
                    raise RouterWriteDisabledError(
                        "Router handoff abandon deadline expired"
                    ) from exc
            finally:
                self._mutex.release()

        raise RouterWriteDisabledError(
            "Router handoff abandon could not stabilize the Journal head"
        ) from last_conflict

    @staticmethod
    def _validated_handoff_deadline(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        if (
            isinstance(deadline, bool)
            or not isinstance(deadline, (int, float))
            or not math.isfinite(float(deadline))
        ):
            raise ValueError("deadline must be a finite monotonic timestamp")
        return float(deadline)

    @staticmethod
    def _ensure_handoff_deadline(
        deadline: float | None,
        *,
        cause: BaseException | None = None,
    ) -> None:
        if deadline is not None and time.monotonic() >= deadline:
            raise RouterWriteDisabledError(
                "Router handoff abandon deadline expired"
            ) from cause

    def _acquire_handoff_mutex(
        self,
        deadline: float | None,
        *,
        allow_immediate: bool = False,
    ) -> bool:
        if deadline is None:
            self._mutex.acquire()
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return bool(allow_immediate and self._mutex.acquire(blocking=False))
        return self._mutex.acquire(
            timeout=min(remaining, threading.TIMEOUT_MAX),
        )

    def defer_dispatch_candidate(
        self,
        acceptance_id: str,
        *,
        max_failures: int = 3,
        terminal_reason: str = "context_unavailable",
    ) -> RouterLifecycleRecord:
        """Persist one transient Context failure and bound retries durably."""

        if isinstance(max_failures, bool) or not isinstance(max_failures, int):
            raise TypeError("max_failures must be an integer")
        if max_failures < 1:
            raise ValueError("max_failures must be positive")
        if terminal_reason not in {
            "context_unavailable",
            "canonical_context_unavailable",
        }:
            raise ValueError("invalid Context terminal reason")
        with self._mutex, self._writer.transaction_guard():
            self.rebuild_projection()
            record = self._record(acceptance_id)
            if record.state == "terminal":
                if record.reason_code != terminal_reason:
                    raise RouterProjectionError(
                        "Router context retry conflicts with terminal"
                    )
                return record
            if record.state != "queued":
                raise RouterProjectionError(
                    "only queued Router work can defer Context assembly"
                )
            failure_count = record.context_failures + 1
            if failure_count >= max_failures:
                return self._terminal_unlocked(record, terminal_reason)
            delay = min(
                self._context_retry_max_seconds,
                self._context_retry_base_seconds * (2 ** (failure_count - 1)),
            )
            eligible_at = _canonical_utc(self._clock() + timedelta(seconds=delay))
            return self._transition_unlocked(
                record,
                "context_retry",
                {
                    "failure_count": failure_count,
                    "next_eligible_at": eligible_at,
                },
            )

    def peek_dispatch_candidate(self) -> RouterDispatchGrant | None:
        """Return one fully revalidated grant without changing Router state."""

        while True:
            with self._mutex, self._writer.transaction_guard():
                self.rebuild_projection()
                active = {
                    record.agent_id
                    for record in self._state.by_acceptance_id.values()
                    if record.state == "dispatching"
                }
                now = self._clock()
                if now.tzinfo is None or now.utcoffset() is None:
                    raise ValueError("Router clock must return an aware datetime")
                candidates = sorted(
                    (
                        record
                        for record in self._state.by_acceptance_id.values()
                        if record.state == "queued"
                        and record.agent_id not in active
                        and _is_eligible(record, now)
                        and _is_inbox_eligible(record, now)
                    ),
                    key=lambda record: (record.queued_sequence, record.acceptance_id),
                )
                if not candidates:
                    return None
                candidate = candidates[0]
            try:
                current, ingress_record, payload = self._dispatch_snapshot(
                    candidate.acceptance_id
                )
            except IngressBlobRetryableError:
                self.defer_inbox_candidate(candidate.acceptance_id)
                continue
            except IngressBlobError:
                self._terminal_inbox_failure(candidate.acceptance_id)
                continue
            if current.state != "queued":
                continue
            snapshot_identity = self._ingress_identity(ingress_record, payload)
            resolution, reason = self._resolve_authority(
                ingress_record.metadata,
                payload,
            )
            if resolution is None and reason == _AUTHORITY_DEPENDENCY_UNAVAILABLE:
                self.defer_inbox_candidate(candidate.acceptance_id)
                continue
            authority = resolution.snapshot if resolution is not None else None
            if authority is None or not self._authority_matches(current.authority, authority):
                self._terminal_for_snapshot(
                    candidate.acceptance_id,
                    snapshot_identity,
                    "authority_stale",
                    allow_queued=True,
                )
                continue
            try:
                request = self._context_request(current, ingress_record.metadata, payload)
            except (TypeError, ValueError):
                self._terminal_for_snapshot(
                    candidate.acceptance_id,
                    snapshot_identity,
                    "context_coordinates_invalid",
                    allow_queued=True,
                )
                continue
            try:
                with self._ingress.dispatch_snapshot_guard(
                    candidate.acceptance_id
                ) as (final_ingress, final_payload), self._mutex:
                    self.rebuild_projection()
                    current = self._record(candidate.acceptance_id)
                    if current.state != "queued":
                        continue
                    if any(
                        other.state == "dispatching"
                        and other.agent_id == current.agent_id
                        for other in self._state.by_acceptance_id.values()
                    ):
                        continue
                    if self._ingress_identity(final_ingress, final_payload) != snapshot_identity:
                        self._terminal_unlocked(current, "inbox_not_dispatchable")
                        continue
                    if not self._authority_matches(current.authority, authority) or (
                        self._projection_missed_workforce_change(
                            authority.projection_sequence,
                            authority.projection_hash,
                        )
                    ):
                        self._terminal_unlocked(current, "authority_stale")
                        continue
                    targeted_task = resolution.targeted_task
                    if (
                        authority.effective_input_kind
                        == TARGETED_TASK_INPUT_KIND
                    ) != (
                        targeted_task is not None
                        and targeted_task.state
                        is TargetedTaskState.TARGETED_VALID
                    ):
                        self._terminal_unlocked(current, "authority_stale")
                        continue
                    return RouterDispatchGrant(
                        current,
                        request,
                        payload,
                        targeted_task,
                    )
            except IngressBlobRetryableError:
                self.defer_inbox_candidate(candidate.acceptance_id)
                continue
            except IngressBlobError:
                self._terminal_inbox_failure(candidate.acceptance_id)
                continue

    def classify_targeted_group_task(
        self,
        metadata: EmployeeIngressMetadata,
        payload: EmployeeIngressPayload,
    ) -> TargetedTaskParseResult | None:
        """Return an authorized target view, re-derived from authenticated ingress."""

        resolution, reason = self._resolve_authority(metadata, payload)
        if resolution is None and reason == _AUTHORITY_DEPENDENCY_UNAVAILABLE:
            return TargetedTaskParseResult(TargetedTaskState.INDETERMINATE)
        return None if resolution is None else resolution.targeted_task

    def finish(self, acceptance_id: str, *, reason_code: str) -> RouterLifecycleRecord:
        try:
            return self._finish(acceptance_id, reason_code=reason_code)
        finally:
            self._sweep_terminal_attachments()

    def _finish(
        self,
        acceptance_id: str,
        *,
        reason_code: str,
    ) -> RouterLifecycleRecord:
        """Anchor a Phase 6 terminal result without performing execution here."""

        if reason_code not in _DISPATCH_TERMINAL_REASONS:
            raise ValueError("invalid dispatch terminal reason")
        with self._mutex, self._writer.transaction_guard():
            self.rebuild_projection()
            record = self._record(acceptance_id)
            if record.state == "terminal":
                if record.reason_code != reason_code:
                    raise RouterProjectionError("terminal Router result conflicts with replay")
                return record
            if record.state != "dispatching":
                raise RouterProjectionError("only dispatching Router work can finish")
            return self._terminal_unlocked(record, reason_code)

    def recover_terminal_attachments(self) -> int:
        """Resume interrupted staging, then converge terminal attachments."""

        if self._attachment_staging is None:
            return 0
        recovered = self._attachment_staging.recover()
        if type(recovered) is not int or recovered < 0:
            raise RouterProjectionError("attachment recovery returned invalid result")
        self._sweep_terminal_attachments()
        return recovered


    def _record(self, acceptance_id: str) -> RouterLifecycleRecord:
        record = self._state.by_acceptance_id.get(acceptance_id)
        if record is None:
            raise KeyError(acceptance_id)
        return record

    def _terminal_attachment_acceptance_ids(self) -> tuple[str, ...]:
        with self._mutex, self._writer.transaction_guard():
            self.rebuild_projection()
            return tuple(
                sorted(
                    record.acceptance_id
                    for record in self._state.by_acceptance_id.values()
                    if record.state == "terminal"
                )
            )

    def _sweep_terminal_attachments(self) -> None:
        if self._attachment_staging is None:
            return
        acceptance_ids = self._terminal_attachment_acceptance_ids()
        # From here onward no Router mutex or Journal transaction guard is held.
        for acceptance_id in acceptance_ids:
            try:
                staged = self._attachment_staging.completed_for_acceptance(
                    acceptance_id
                )
            except Exception:
                continue
            if staged is None:
                continue
            staging_id = getattr(staged, "staging_id", None)
            if (
                type(staging_id) is not str
                or not staging_id
                or type(getattr(staged, "status", None)) is not str
                or getattr(staged, "status", None) != "completed"
                or type(getattr(staged, "cleanup_state", None)) is not str
                or getattr(staged, "cleanup_state", None) != "none"
            ):
                continue
            try:
                self._attachment_staging.cleanup(staging_id)
            except Exception:
                continue

    def _transition_unlocked(
        self,
        record: RouterLifecycleRecord,
        state: str,
        extra: Mapping[str, object],
    ) -> RouterLifecycleRecord:
        event = JournalEvent(
            event_type=_ROUTER_PREFIX + state,
            aggregate_id=record.aggregate_id,
            payload={"acceptance_id": record.acceptance_id, **dict(extra)},
        )
        return self._commit_events_unlocked((event,), record.acceptance_id)

    def _commit_events_unlocked(
        self,
        events: tuple[JournalEvent, ...],
        result_acceptance_id: str,
    ) -> RouterLifecycleRecord:
        """Preflight and anchor one complete Router transaction frame."""

        # Reject programmer errors before they can enter the append-only log.
        preflight = self._state.clone()
        next_sequence = self._state.cursor_sequence + 1
        for event in events:
            _reduce_router_event(preflight, event, sequence=next_sequence)
        aggregate_ids = {event.aggregate_id for event in events}
        versions = self._writer.get_aggregate_versions(aggregate_ids)
        result = self._writer.commit(
            events,
            versions,
            expected_head_sequence=self._state.cursor_sequence,
            expected_head_hash=self._state.cursor_hash or None,
        )
        if result.state is not CommitState.ANCHORED:
            raise RouterWriteDisabledError("Router transition was not anchored")
        for event in events:
            _reduce_router_event(self._state, event, sequence=result.frame.sequence)
        self._state.cursor_sequence = result.frame.sequence
        self._state.cursor_hash = result.frame.frame_hash
        return self._record(result_acceptance_id)

    def _terminal_unlocked(
        self,
        record: RouterLifecycleRecord,
        reason_code: str,
    ) -> RouterLifecycleRecord:
        return self._transition_unlocked(record, "terminal", {"reason_code": reason_code})

    def _terminal_inbox_failure(
        self,
        acceptance_id: str,
    ) -> RouterLifecycleRecord:
        return self._terminal_after_external(
            acceptance_id,
            "inbox_not_dispatchable",
        )

    def _terminal_after_external(
        self,
        acceptance_id: str,
        reason_code: str,
    ) -> RouterLifecycleRecord:
        with self._mutex, self._writer.transaction_guard():
            self.rebuild_projection()
            record = self._record(acceptance_id)
            if record.state == "terminal":
                return record
            if record.state == "dispatching":
                # Only Task 6/finish owns terminal attempt outcomes.
                return record
            if reason_code.startswith("attachment_") and record.state == "queued":
                # Another Router instance already proved staging/admission.
                # A losing external callback must not revoke that durable work.
                return record
            return self._terminal_unlocked(record, reason_code)

    def _dispatch_snapshot(self, acceptance_id: str):
        """Read immutable Inbox/Router state without invoking external ports."""

        with self._ingress.dispatch_snapshot_guard(acceptance_id) as (
            ingress_record,
            payload,
        ), self._mutex:
            self.rebuild_projection()
            return self._record(acceptance_id), ingress_record, payload

    @staticmethod
    def _ingress_identity(ingress_record: Any, payload: EmployeeIngressPayload) -> tuple:
        return (
            ingress_record.acceptance,
            ingress_record.metadata,
            payload.payload_sha256,
        )

    def _terminal_for_snapshot(
        self,
        acceptance_id: str,
        expected_identity: tuple,
        reason_code: str,
        *,
        allow_queued: bool = False,
    ) -> RouterLifecycleRecord:
        try:
            with self._ingress.dispatch_snapshot_guard(acceptance_id) as (
                ingress_record,
                payload,
            ), self._mutex, self._writer.transaction_guard():
                self.rebuild_projection()
                record = self._record(acceptance_id)
                if record.state in {"dispatching", "terminal"} or (
                    record.state == "queued" and not allow_queued
                ):
                    return record
                if self._ingress_identity(ingress_record, payload) != expected_identity:
                    reason_code = "inbox_not_dispatchable"
                return self._terminal_unlocked(record, reason_code)
        except IngressBlobRetryableError:
            return self.defer_inbox_candidate(acceptance_id)
        except IngressBlobError:
            return self._terminal_inbox_failure(acceptance_id)

    def _resolve_authority(
        self,
        metadata: EmployeeIngressMetadata,
        payload: EmployeeIngressPayload,
    ) -> tuple[_AuthorityResolution | None, str]:
        part, reason = self._message_provenance(metadata, payload)
        if part is None:
            return None, reason
        coordinate_binder = (
            _bound_encrypted_remote_coordinate
            if metadata.event_type == "im.message.receive_v1"
            else _bound_remote_coordinate
        )
        remote_chat_id = coordinate_binder(
            metadata.chat_id,
            part.get("remote_chat_id"),
            "oc_",
        )
        if not remote_chat_id:
            return None, "sender_invalid"
        chat_type = part.get("chat_type")
        if metadata.event_type == "ghostap.team.assignment.v1":
            authorization_scope = EmployeeAuthorizationScope.MANAGED_GROUP
        elif chat_type == "group":
            authorization_scope = EmployeeAuthorizationScope.MANAGED_GROUP
        elif chat_type == "p2p":
            authorization_scope = EmployeeAuthorizationScope.OWNER_P2P
        else:
            return None, "sender_invalid"
        try:
            registry = self._registry_provider()
            if type(registry) is not ProjectedAgentRegistry:
                return None, "authority_denied"
            employee = registry.get(metadata.tenant_key, metadata.agent_id)
            if (
                employee is None
                or employee.state is not EmployeeState.ACTIVE
                or employee.worker_type is not WorkerType.VISIBLE
                or employee.bot_principal_id != metadata.bot_principal_id
            ):
                return None, "authority_denied"
            status = self._channels.status(metadata.agent_id)
            identity = getattr(status, "identity", None)
            ready_metadata = getattr(status, "ready_metadata", None)
            status_agent_id = getattr(status, "agent_id", None)
            status_app_id = getattr(status, "app_id", None)
            status_tenant_key = getattr(status, "tenant_key", None)
            status_bot_principal_id = getattr(status, "bot_principal_id", None)
            status_generation = getattr(status, "generation", None)
            status_state = getattr(status, "state", None)
            identity_app_id = (
                identity.get("app_id") if isinstance(identity, Mapping) else None
            )
            identity_open_id = (
                identity.get("open_id") if isinstance(identity, Mapping) else None
            )
            connection_id = (
                ready_metadata.get("connection_id")
                if isinstance(ready_metadata, Mapping)
                else None
            )
            required_strings = (
                status_agent_id,
                status_app_id,
                status_tenant_key,
                status_bot_principal_id,
                identity_app_id,
                connection_id,
            )
            if (
                not isinstance(identity, Mapping)
                or not isinstance(ready_metadata, Mapping)
                or any(
                    not isinstance(value, str) or not value
                    for value in required_strings
                )
                or type(status_generation) is not int
                or status_generation < 1
                or status_state is not ChannelProcessState.READY
                or status_agent_id != metadata.agent_id
                or status_tenant_key != metadata.tenant_key
                or status_bot_principal_id != metadata.bot_principal_id
                or status_app_id != metadata.app_id
                or status_generation != metadata.channel_generation
                or identity_app_id != metadata.app_id
                or connection_id != metadata.connection_id
            ):
                return None, "authority_denied"

            targeted_task: TargetedTaskParseResult | None = None
            if (
                authorization_scope is EmployeeAuthorizationScope.MANAGED_GROUP
                and metadata.event_type == "im.message.receive_v1"
            ):
                slash_observation = is_group_slash_observation(part)
                if not isinstance(identity_open_id, str) or not identity_open_id:
                    if slash_observation:
                        return None, "authority_denied"
                else:
                    parsed_task = parse_targeted_group_task(
                        metadata=metadata,
                        part=part,
                        expected_bot_open_id=identity_open_id,
                        expected_tenant_key=metadata.tenant_key,
                    )
                    if parsed_task.state is not TargetedTaskState.NOT_TARGETED:
                        targeted_task = parsed_task
                    elif slash_observation:
                        return None, "authority_denied"
            sender_union_id = part.get("sender_union_id", "")
            if not isinstance(sender_union_id, str):
                return None, "sender_invalid"
            if (
                authorization_scope is EmployeeAuthorizationScope.OWNER_P2P
                and not sender_union_id
            ):
                return None, "sender_invalid"
            if metadata.event_type == "ghostap.team.assignment.v1":
                requester_principal_id = metadata.sender_principal_id
            elif (
                authorization_scope is EmployeeAuthorizationScope.MANAGED_GROUP
                and targeted_task is None
            ):
                # Ambient group text retains the same-App identity contract.
                # Cross-App union normalization is reserved for a uniquely
                # targeted employee command so one event cannot fan out.
                requester_principal_id = metadata.sender_principal_id
            else:
                requester_principal_resolver = self._requester_principal_resolver
                if requester_principal_resolver is None:
                    if (
                        authorization_scope is EmployeeAuthorizationScope.OWNER_P2P
                        or targeted_task is not None
                    ):
                        return None, "requester_denied"
                    requester_principal_resolver = _same_app_requester_principal
                try:
                    requester_principal_id = requester_principal_resolver(
                        tenant_key=metadata.tenant_key,
                        agent_id=metadata.agent_id,
                        owner_principal_id=employee.owner_principal_id,
                        sender_principal_id=metadata.sender_principal_id,
                        sender_union_id=sender_union_id,
                        app_id=metadata.app_id,
                        bot_principal_id=metadata.bot_principal_id,
                        channel_generation=metadata.channel_generation,
                        connection_id=metadata.connection_id,
                        channel_identity_app_id=identity_app_id,
                    )
                except Exception:
                    return None, _AUTHORITY_DEPENDENCY_UNAVAILABLE
            if (
                not isinstance(requester_principal_id, str)
                or not requester_principal_id
            ):
                return None, "requester_denied"
            if targeted_task is not None and (
                requester_principal_id != employee.owner_principal_id
                or requester_principal_id != self._managed_group_owner_id
            ):
                # A uniquely targeted group command is an owner control lane.
                # Do not delegate this invariant to a pluggable resolver or
                # permissive requester ACL.
                return None, "requester_denied"
            if authorization_scope is EmployeeAuthorizationScope.MANAGED_GROUP:
                if (
                    targeted_task is not None
                    and self._managed_group_registry_provider is None
                ):
                    return None, "authority_denied"
                if self._managed_group_registry_provider is not None:
                    try:
                        managed_registry = self._managed_group_registry_provider()
                        if type(managed_registry) is not ManagedGroupRegistry:
                            return None, "authority_denied"
                        employee_bot_ids = (
                            self._employee_bot_ids_provider()
                            if self._employee_bot_ids_provider is not None
                            else frozenset()
                        )
                        if type(employee_bot_ids) is not frozenset:
                            return None, "authority_denied"
                        managed_group, project_grant = (
                            managed_registry.trust_snapshot(remote_chat_id)
                        )
                        trust = TrustZoneResolver(
                            owner_id=self._managed_group_owner_id,
                            managed_groups=(
                                () if managed_group is None else (managed_group,)
                            ),
                            project_grants=(
                                () if project_grant is None else (project_grant,)
                            ),
                            employee_bot_ids=employee_bot_ids,
                        ).resolve(
                            sender_id=(
                                requester_principal_id
                                if targeted_task is not None
                                else metadata.sender_principal_id
                            ),
                            chat_id=remote_chat_id,
                            chat_type=str(chat_type),
                        )
                        if trust.zone is not TrustZone.MANAGED_AGENT_GROUP or (
                            targeted_task is not None
                            and (
                                trust.actor is not ActorKind.OWNER
                                or trust.managed_group is None
                                or trust.managed_group.owner_id
                                != requester_principal_id
                            )
                        ):
                            return None, "authority_denied"
                    except Exception:
                        return None, _AUTHORITY_DEPENDENCY_UNAVAILABLE
            binding = registry.context_binding(
                tenant_key=metadata.tenant_key,
                agent_id=metadata.agent_id,
                bot_principal_id=metadata.bot_principal_id,
                app_id=metadata.app_id,
                chat_id=remote_chat_id,
                requester_principal_id=requester_principal_id,
                authorization_scope=authorization_scope,
            )
            if type(binding) is not ProjectedContextBinding:
                return None, "authority_denied"
            employee = binding.employee
            principal = binding.principal
            if (
                employee.state is not EmployeeState.ACTIVE
                or employee.worker_type is not WorkerType.VISIBLE
                or employee.tenant_key != metadata.tenant_key
                or employee.agent_id != metadata.agent_id
                or employee.bot_principal_id != metadata.bot_principal_id
                or (
                    authorization_scope
                    is EmployeeAuthorizationScope.MANAGED_GROUP
                    and remote_chat_id not in employee.member_groups
                )
                or (
                    authorization_scope
                    is EmployeeAuthorizationScope.OWNER_P2P
                    and requester_principal_id
                    != employee.owner_principal_id
                )
                or principal.bot_principal_id != metadata.bot_principal_id
                or principal.tenant_key != metadata.tenant_key
                or principal.agent_id != metadata.agent_id
                or principal.app_id != metadata.app_id
            ):
                return None, "authority_denied"
            if self._projection_missed_workforce_change(
                binding.projection_sequence,
                binding.projection_hash,
            ):
                return None, "authority_denied"
            if authorization_scope is EmployeeAuthorizationScope.MANAGED_GROUP:
                try:
                    membership_degraded = self._membership_health.is_degraded(
                        metadata.agent_id,
                        remote_chat_id,
                    )
                except Exception:
                    return None, _AUTHORITY_DEPENDENCY_UNAVAILABLE
                if (
                    type(membership_degraded) is not bool
                    or membership_degraded is not False
                ):
                    return None, "membership_degraded"
            snapshot = RouterAuthoritySnapshot(
                tenant_key=metadata.tenant_key,
                agent_id=employee.agent_id,
                bot_principal_id=principal.bot_principal_id,
                app_id=principal.app_id,
                channel_generation=metadata.channel_generation,
                connection_id=metadata.connection_id,
                authorization_scope=authorization_scope,
                team_id=remote_chat_id,
                requester_principal_id=requester_principal_id,
                projection_sequence=binding.projection_sequence,
                projection_hash=(
                    binding.projection_hash
                    if binding.projection_sequence
                    else GENESIS_HASH
                ),
                employee_version=employee.aggregate_version,
                tool=employee.tool,
                model=employee.model,
                effort=employee.effort,
                constraints_digest=self._constraints_digest,
                system_prompt_token_reserve=self._reserve,
                effective_input_kind=(
                    TARGETED_TASK_INPUT_KIND
                    if targeted_task is not None
                    and targeted_task.state is TargetedTaskState.TARGETED_VALID
                    else ""
                ),
                effective_input_digest=(
                    targeted_task.input_digest
                    if targeted_task is not None
                    and targeted_task.state is TargetedTaskState.TARGETED_VALID
                    else ""
                ),
                target_bot_open_id_digest=(
                    hashlib.sha256(identity_open_id.encode("utf-8")).hexdigest()
                    if targeted_task is not None
                    and targeted_task.state is TargetedTaskState.TARGETED_VALID
                    and isinstance(identity_open_id, str)
                    else ""
                ),
            )
            request = self._request_from(metadata, part, snapshot)
            try:
                requester_authorized = self._requester_acl.is_authorized(request)
            except Exception:
                return None, _AUTHORITY_DEPENDENCY_UNAVAILABLE
            if type(requester_authorized) is not bool or requester_authorized is not True:
                return None, "requester_denied"
            credential_ref = principal.credential_ref
            if not isinstance(credential_ref, str) or not credential_ref:
                return None, "authority_denied"
            return _AuthorityResolution(
                snapshot,
                credential_ref,
                targeted_task,
            ), ""
        except Exception:
            return None, _AUTHORITY_DEPENDENCY_UNAVAILABLE

    def _projection_missed_workforce_change(
        self,
        sequence: int,
        frame_hash: str,
    ) -> bool:
        anchored_sequence = self._writer.anchor.read().sequence
        if sequence < 0 or sequence > anchored_sequence:
            return True
        if sequence == 0:
            if frame_hash not in {"", GENESIS_HASH}:
                return True
        else:
            coordinate = next(
                (
                    frame
                    for frame in self._writer.replay(from_sequence=sequence)
                    if frame.sequence == sequence
                ),
                None,
            )
            if coordinate is None or coordinate.frame_hash != frame_hash:
                return True
        if sequence == anchored_sequence:
            return False
        for frame in self._writer.replay(from_sequence=max(1, sequence + 1)):
            if frame.sequence > anchored_sequence:
                break
            if any(is_workforce_event(event.event_type) for event in frame.events):
                return True
        return False

    @staticmethod
    def _message_provenance(
        metadata: EmployeeIngressMetadata,
        payload: EmployeeIngressPayload,
    ) -> tuple[Mapping[str, object] | None, str]:
        if len(payload.normalized_parts) != 1:
            return None, "sender_invalid"
        part = payload.normalized_parts[0]
        expected_type = (
            "team_assignment"
            if metadata.event_type == "ghostap.team.assignment.v1"
            else "message"
        )
        if part.get("type") != expected_type:
            return None, "sender_invalid"
        if expected_type == "team_assignment" and (
            not isinstance(part.get("team_instruction"), str)
            or not part.get("team_instruction")
            or len(part.get("team_instruction")) > 14_000
        ):
            return None, "sender_invalid"
        chat_type = part.get("chat_type")
        if (
            expected_type == "team_assignment"
            and chat_type != "group"
        ) or (
            expected_type == "message"
            and chat_type not in {"group", "p2p"}
        ):
            return None, "sender_invalid"
        sender_id = part.get("sender_id")
        sender_type = part.get("sender_type")
        if sender_type in {"bot", "app"}:
            return None, "bot_loop"
        if (
            sender_id != metadata.sender_principal_id
            or part.get("sender_id_type") != "open_id"
            or sender_type != "user"
            or part.get("sender_tenant_key") != metadata.tenant_key
        ):
            return None, "sender_invalid"
        remote_fields = (
            "remote_chat_id",
            "remote_message_id",
            "remote_root_id",
        )
        if any(field in part for field in remote_fields):
            if not all(field in part for field in remote_fields):
                return None, "sender_invalid"
            bound_chat = _bound_encrypted_remote_coordinate(
                metadata.chat_id,
                part.get("remote_chat_id"),
                "oc_",
            )
            bound_message = _bound_encrypted_remote_coordinate(
                metadata.message_id,
                part.get("remote_message_id"),
                "om_",
            )
            bound_root = _bound_encrypted_remote_coordinate(
                metadata.thread_root_message_id,
                part.get("remote_root_id"),
                "om_",
            )
            if bound_chat is None or bound_message is None or bound_root is None:
                return None, "sender_invalid"
        return part, ""

    def _context_request(
        self,
        record: RouterLifecycleRecord,
        metadata: EmployeeIngressMetadata,
        payload: EmployeeIngressPayload,
    ) -> AuthorizedContextRequest:
        if record.authority is None or len(payload.normalized_parts) != 1:
            raise ValueError("Router record lacks authority")
        return self._request_from(metadata, payload.normalized_parts[0], record.authority)

    @staticmethod
    def _request_from(
        metadata: EmployeeIngressMetadata,
        part: Mapping[str, object],
        authority: RouterAuthoritySnapshot,
    ) -> AuthorizedContextRequest:
        thread_id = part.get("feishu_thread_id", "")
        if not isinstance(thread_id, str):
            raise ValueError("invalid Feishu thread identity")
        coordinate_binder = (
            _bound_encrypted_remote_coordinate
            if metadata.event_type == "im.message.receive_v1"
            else _bound_remote_coordinate
        )
        remote_message_id = coordinate_binder(
            metadata.message_id,
            part.get("remote_message_id"),
            "om_",
        )
        remote_root_id = coordinate_binder(
            metadata.thread_root_message_id,
            part.get("remote_root_id"),
            "om_",
        )
        if not remote_message_id or remote_root_id is None:
            raise ValueError("invalid remote message coordinates")
        return AuthorizedContextRequest(
            tenant_key=authority.tenant_key,
            agent_id=authority.agent_id,
            bot_principal_id=authority.bot_principal_id,
            app_id=authority.app_id,
            channel_generation=authority.channel_generation,
            chat_id=authority.team_id,
            thread_root_message_id=remote_root_id or remote_message_id,
            feishu_thread_id=thread_id,
            current_message_id=remote_message_id,
            requester_principal_id=authority.requester_principal_id,
            source_requester_principal_id=metadata.sender_principal_id,
            authorization_scope=authority.authorization_scope,
            system_prompt_token_reserve=authority.system_prompt_token_reserve,
            constraints_digest=authority.constraints_digest,
        )

    def _queue_full_unlocked(
        self,
        authority: RouterAuthoritySnapshot,
    ) -> bool:
        queued = [
            record
            for record in self._state.by_acceptance_id.values()
            if record.state == "queued"
        ]
        employee_count = sum(record.agent_id == authority.agent_id for record in queued)
        team_count = sum(
            record.authority is not None
            and record.authority.team_id == authority.team_id
            for record in queued
        )
        return (
            employee_count >= self._limits.per_employee
            or team_count >= self._limits.per_team
            or len(queued) >= self._limits.global_limit
        )

    def _rebalance_victim_unlocked(
        self,
        authority: RouterAuthoritySnapshot,
    ) -> RouterLifecycleRecord | None:
        """Choose the latest queued peer only when the newcomer owns no slot."""

        queued = tuple(
            record
            for record in self._state.by_acceptance_id.values()
            if record.state == "queued" and record.authority is not None
        )
        employee_count = sum(
            record.authority.agent_id == authority.agent_id for record in queued
        )
        if employee_count >= self._limits.per_employee:
            return None
        team_count = sum(
            record.authority.team_id == authority.team_id for record in queued
        )
        team_full = team_count >= self._limits.per_team
        global_full = len(queued) >= self._limits.global_limit
        if not team_full and not global_full:
            return None
        scope = tuple(
            record
            for record in queued
            if not team_full or record.authority.team_id == authority.team_id
        )
        scope_counts: dict[str, int] = {}
        for candidate in scope:
            scope_counts[candidate.authority.agent_id] = (
                scope_counts.get(candidate.authority.agent_id, 0) + 1
            )
        if scope_counts.get(authority.agent_id, 0) != 0:
            return None
        most_slots = max(scope_counts.values(), default=0)
        if most_slots <= 1:
            return None
        candidates = tuple(
            candidate
            for candidate in scope
            if scope_counts[candidate.authority.agent_id] == most_slots
        )
        return max(
            candidates,
            key=lambda record: (record.queued_sequence, record.acceptance_id),
        )

    def _rebalance_and_queue_unlocked(
        self,
        record: RouterLifecycleRecord,
        victim: RouterLifecycleRecord,
        *,
        response_obligation: dict[str, object] | None,
    ) -> RouterLifecycleRecord:
        if record.authority is None:
            raise RouterProjectionError("staging Router record lacks authority")
        position = 1 + sum(
            candidate.state == "queued"
            and candidate.acceptance_id != victim.acceptance_id
            and candidate.authority is not None
            and candidate.authority.agent_id == record.authority.agent_id
            for candidate in self._state.by_acceptance_id.values()
        )
        events = (
            JournalEvent(
                event_type=_ROUTER_PREFIX + "terminal",
                aggregate_id=victim.aggregate_id,
                payload={
                    "acceptance_id": victim.acceptance_id,
                    "reason_code": "queue_rebalanced",
                },
            ),
            JournalEvent(
                event_type=_ROUTER_PREFIX + "queued",
                aggregate_id=record.aggregate_id,
                payload={
                    "acceptance_id": record.acceptance_id,
                    "authority": record.authority.to_dict(),
                    "queue_position": position,
                    "response_obligation": response_obligation,
                },
            ),
        )
        return self._commit_events_unlocked(events, record.acceptance_id)

    @staticmethod
    def _response_obligation(
        record: RouterLifecycleRecord,
        metadata: EmployeeIngressMetadata,
        payload: EmployeeIngressPayload,
    ) -> dict[str, object] | None:
        if record.event_type != "im.message.receive_v1":
            return None
        if record.authority is None or len(payload.normalized_parts) != 1:
            raise RouterProjectionError("message queue lacks response authority")
        part = payload.normalized_parts[0]
        remote_chat_id = _bound_encrypted_remote_coordinate(
            metadata.chat_id,
            part.get("remote_chat_id"),
            "oc_",
        )
        remote_message_id = _bound_encrypted_remote_coordinate(
            metadata.message_id,
            part.get("remote_message_id"),
            "om_",
        )
        remote_root_id = _bound_encrypted_remote_coordinate(
            metadata.thread_root_message_id,
            part.get("remote_root_id"),
            "om_",
        )
        if (
            not remote_chat_id
            or not remote_message_id
            or remote_root_id is None
            or remote_chat_id != record.authority.team_id
        ):
            raise RouterProjectionError("message queue response target is invalid")
        return RouterResponseObligation.create(
            authority=record.authority,
            acceptance_id=record.acceptance_id,
            ingress_aggregate_id=record.aggregate_id,
            envelope_id=record.envelope_id,
            event_type=record.event_type,
            chat_id=remote_chat_id,
            message_id=remote_message_id,
            thread_root_message_id=remote_root_id,
        ).to_dict()

    @staticmethod
    def _authority_matches(
        frozen: RouterAuthoritySnapshot | None,
        current: RouterAuthoritySnapshot,
    ) -> bool:
        """Allow an authenticated projection head to advance, never roll back."""

        if frozen is None or current.projection_sequence < frozen.projection_sequence:
            return False
        if (
            current.projection_sequence == frozen.projection_sequence
            and current.projection_hash != frozen.projection_hash
        ):
            return False
        ignored = {"projection_sequence", "projection_hash"}
        return all(
            getattr(current, field_name) == getattr(frozen, field_name)
            for field_name in RouterAuthoritySnapshot._FIELDS - ignored
        )

    def _stage_attachments(
        self,
        acceptance_id: str,
        metadata: EmployeeIngressMetadata,
        payload: EmployeeIngressPayload,
        resolution: _AuthorityResolution,
    ) -> None:
        from .attachments import (
            AttachmentStateError,
            AuthorizedAttachmentStagingRequest,
            EmployeeAttachmentDescriptor,
        )

        authority = resolution.snapshot
        if self._completed_attachment_stage(acceptance_id) is not None:
            return
        part = payload.normalized_parts[0]
        remote_message_id = _bound_encrypted_remote_coordinate(
            metadata.message_id,
            part.get("remote_message_id"),
            "om_",
        )
        if not remote_message_id:
            raise AttachmentStateError("invalid remote attachment message coordinate")
        descriptors = tuple(
            EmployeeAttachmentDescriptor(
                schema_version=1,
                message_id=remote_message_id,
                resource_type=str(item["resource_type"]),
                resource_id=str(item["resource_id"]),
                declared_mime_type=str(item["mime_type"]),
                declared_size_bytes=int(item["size_bytes"]),
                declared_sha256=str(item["sha256"]),
                user_filename=str(item["resource_id"]),
            )
            for item in payload.attachment_descriptors
        )
        request = AuthorizedAttachmentStagingRequest(
            schema_version=1,
            acceptance_id=acceptance_id,
            envelope_id=metadata.envelope_id,
            tenant_key=authority.tenant_key,
            agent_id=authority.agent_id,
            app_id=authority.app_id,
            credential_ref=resolution.credential_ref,
            descriptors=descriptors,
        )
        try:
            self._attachment_staging.stage(request)
        except AttachmentStateError:
            if self._completed_attachment_stage(acceptance_id) is None:
                raise

    def _completed_attachment_stage(self, acceptance_id: str) -> object | None:
        method = getattr(
            self._attachment_staging,
            "completed_for_acceptance",
            None,
        )
        if not callable(method):
            raise RouterProjectionError("attachment completion port unavailable")
        return method(acceptance_id)


__all__ = [
    "DurableEmployeeIngressRouter",
    "RouterAuthoritySnapshot",
    "RouterDispatchGrant",
    "RouterLifecycleRecord",
    "RouterProjectionError",
    "RouterProjectionState",
    "RouterQueueLimits",
    "RouterWriteDisabledError",
]
