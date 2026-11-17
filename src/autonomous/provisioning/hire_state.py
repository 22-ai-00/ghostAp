"""Frozen, Journal-rebuilt state for the production employee hire Saga."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping

from ..journal.frame import JournalEvent, TransactionFrame


class HireProjectionError(RuntimeError):
    """A durable hire event stream violates the Saga contract."""


class HirePhase(str, Enum):
    PROVISIONING_APP = "provisioning_app"
    STORING_CREDENTIAL = "storing_credential"
    CONFIGURING = "configuring"
    VALIDATING = "validating"
    READY_PENDING_VERIFICATION = "ready_pending_verification"
    ACTIVE = "active"
    RETIRING = "retiring"
    ACTION_REQUIRED = "action_required"
    ARCHIVED = "archived"


class HireEffectState(str, Enum):
    PLANNED = "planned"
    PREPARED = "prepared"
    EXECUTING = "executing"
    COMMITTED = "committed"
    ACTION_REQUIRED = "action_required"


@dataclass(frozen=True)
class DurableHireState:
    intent_id: str = ""
    tenant_key: str = ""
    message_id: str = ""
    chat_id: str = ""
    requester_principal_id: str = ""
    requester_union_id: str = ""
    employee_name: str = ""
    tool: str = ""
    model: str = ""
    effort: str = ""
    profile: str = "standard"
    role: str = ""
    persona: str = ""
    personality_traits: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    permissions: tuple[str, ...] = ()
    existing_app_id: str = ""
    agent_id: str = ""
    bot_principal_id: str = ""
    attempt_id: str = ""
    app_id: str = ""
    credential_ref: str = ""
    slash_spec_hash: str = ""
    slash_observed_hash: str = ""
    slash_verified_at: float = 0.0
    channel_generation: int = 0
    automatic_reactivation_generation: int = 0
    channel_identity_app_id: str = ""
    channel_connection_id: str = ""
    channel_verified_at: float = 0.0
    verification_consumed: bool = False
    activation_source: str = ""
    activation_verified_at: float = 0.0
    phase: HirePhase = HirePhase.PROVISIONING_APP
    effects: tuple[tuple[str, HireEffectState], ...] = ()
    effect_types: tuple[tuple[str, str], ...] = ()
    effect_metadata: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = ()
    last_sequence: int = 0

    def effect_state(self, effect_id: str) -> HireEffectState | None:
        return dict(self.effects).get(effect_id)

    def metadata_for(self, effect_id: str) -> Mapping[str, str]:
        return MappingProxyType(dict(dict(self.effect_metadata).get(effect_id, ())))


_EFFECT_EVENTS = {
    "hire.effect.prepared": HireEffectState.PREPARED,
    "hire.effect.executing": HireEffectState.EXECUTING,
    "hire.effect.committed": HireEffectState.COMMITTED,
    "hire.effect.action_required": HireEffectState.ACTION_REQUIRED,
}
_ALLOWED_EFFECT_PREDECESSORS = {
    HireEffectState.PREPARED: {None, HireEffectState.PLANNED},
    HireEffectState.EXECUTING: {HireEffectState.PREPARED},
    HireEffectState.COMMITTED: {HireEffectState.EXECUTING},
    HireEffectState.ACTION_REQUIRED: {
        HireEffectState.PREPARED,
        HireEffectState.EXECUTING,
    },
}
_ALLOWED_PHASE_SUCCESSORS = {
    HirePhase.PROVISIONING_APP: {
        HirePhase.STORING_CREDENTIAL,
        HirePhase.CONFIGURING,
        HirePhase.RETIRING,
        HirePhase.ACTION_REQUIRED,
        HirePhase.ARCHIVED,
    },
    HirePhase.STORING_CREDENTIAL: {
        HirePhase.CONFIGURING,
        HirePhase.RETIRING,
        HirePhase.ACTION_REQUIRED,
        HirePhase.ARCHIVED,
    },
    HirePhase.CONFIGURING: {
        HirePhase.VALIDATING,
        HirePhase.RETIRING,
        HirePhase.ACTION_REQUIRED,
        HirePhase.ARCHIVED,
    },
    HirePhase.VALIDATING: {
        HirePhase.READY_PENDING_VERIFICATION,
        HirePhase.RETIRING,
        HirePhase.ACTION_REQUIRED,
        HirePhase.ARCHIVED,
    },
    HirePhase.READY_PENDING_VERIFICATION: {
        HirePhase.VALIDATING,
        HirePhase.ACTIVE,
        HirePhase.RETIRING,
        HirePhase.ACTION_REQUIRED,
        HirePhase.ARCHIVED,
    },
    HirePhase.ACTIVE: {
        HirePhase.VALIDATING,
        HirePhase.RETIRING,
        HirePhase.ACTION_REQUIRED,
        HirePhase.ARCHIVED,
    },
    HirePhase.RETIRING: {
        HirePhase.ACTION_REQUIRED,
        HirePhase.ARCHIVED,
    },
    HirePhase.ACTION_REQUIRED: {HirePhase.RETIRING, HirePhase.ARCHIVED},
    HirePhase.ARCHIVED: set(),
}
_EFFECT_PAYLOAD_KEYS = frozenset({"effect_id", "effect_type", "metadata"})
_EFFECT_METADATA_KEYS = frozenset(
    {
        "app_id",
        "credential_ref",
        "slash_spec_hash",
        "slash_observed_hash",
        "slash_verified_at",
        "generation",
        "identity_app_id",
        "connection_id",
        "channel_verified_at",
        "send_request_id",
        "ingress_event_id",
        "reply_app_id",
        "reply_message_id",
        "main_bot_send_count",
        "error_code",
    }
)


def _required_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise HireProjectionError(f"{key} is required")
    return value


def _model_string(payload: Mapping[str, object]) -> str:
    value = payload.get("model")
    if not isinstance(value, str):
        raise HireProjectionError("model must be a string")
    return value


def _created_state(event: JournalEvent, sequence: int) -> DurableHireState | None:
    payload = event.payload
    if event.event_type != "employee.created" or "hire_intent_id" not in payload:
        return None
    try:
        phase = HirePhase(_required_string(payload, "state"))
    except ValueError as exc:
        raise HireProjectionError("invalid initial hire phase") from exc
    if phase is not HirePhase.PROVISIONING_APP:
        raise HireProjectionError("invalid initial hire phase")
    agent_id = _required_string(payload, "agent_id")
    if agent_id != event.aggregate_id:
        raise HireProjectionError("hire employee aggregate mismatch")
    requester_union_id = payload.get("requester_union_id", "")
    if not isinstance(requester_union_id, str):
        raise HireProjectionError("requester_union_id must be a string")
    existing_app_id = payload.get("existing_app_id", "")
    if (
        not isinstance(existing_app_id, str)
        or (
            existing_app_id
            and re.fullmatch(r"cli_[A-Za-z0-9_-]{3,128}", existing_app_id) is None
        )
    ):
        raise HireProjectionError("existing_app_id is invalid")
    return DurableHireState(
        intent_id=_required_string(payload, "hire_intent_id"),
        tenant_key=_required_string(payload, "tenant_key"),
        message_id=_required_string(payload, "hire_message_id"),
        chat_id=_required_string(payload, "hire_chat_id"),
        requester_principal_id=_required_string(payload, "owner_principal_id"),
        requester_union_id=requester_union_id,
        employee_name=_required_string(payload, "name"),
        tool=_required_string(payload, "tool"),
        model=_model_string(payload),
        effort=_required_string(payload, "effort"),
        profile=_required_string(payload, "profile"),
        role=str(payload.get("role", "")),
        persona=str(payload.get("persona", "")),
        personality_traits=tuple(payload.get("personality_traits", ())),
        capabilities=tuple(payload.get("capabilities", ())),
        permissions=tuple(payload.get("permissions", ())),
        existing_app_id=existing_app_id,
        agent_id=agent_id,
        bot_principal_id=_required_string(payload, "planned_bot_principal_id"),
        attempt_id=_required_string(payload, "provisioning_attempt_id"),
        phase=phase,
        last_sequence=sequence,
    )


def _effect_fact(
    event: JournalEvent,
) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    payload = event.payload
    if not {"effect_id", "effect_type"}.issubset(payload) or not set(payload).issubset(
        _EFFECT_PAYLOAD_KEYS
    ):
        raise HireProjectionError("invalid hire effect payload")
    effect_id = _required_string(payload, "effect_id")
    effect_type = _required_string(payload, "effect_type")
    raw_metadata = payload.get("metadata", {})
    if not isinstance(raw_metadata, Mapping):
        raise HireProjectionError("invalid hire effect metadata")
    if not set(raw_metadata).issubset(_EFFECT_METADATA_KEYS):
        raise HireProjectionError("invalid hire effect metadata")
    metadata: dict[str, str] = {}
    for key, value in raw_metadata.items():
        if not isinstance(value, str) or not value:
            raise HireProjectionError("invalid hire effect metadata")
        metadata[key] = value
    if "credential_ref" in metadata and "app_id" not in metadata:
        raise HireProjectionError("credential metadata requires app_id")
    return effect_id, effect_type, tuple(sorted(metadata.items()))


@dataclass(frozen=True)
class HireProjection:
    states: Mapping[str, DurableHireState]

    @classmethod
    def empty(cls) -> HireProjection:
        return cls(MappingProxyType({}))

    def get(self, intent_id: str) -> DurableHireState | None:
        return self.states.get(intent_id)

    @classmethod
    def rebuild(cls, frames: Iterable[TransactionFrame]) -> HireProjection:
        states: dict[str, DurableHireState] = {}
        agent_intents: dict[str, str] = {}
        bot_intents: dict[str, str] = {}
        existing_app_intents: dict[str, str] = {}
        for frame in frames:
            for event in frame.events:
                created = _created_state(event, frame.sequence)
                if created is not None:
                    if created.intent_id in states or created.agent_id in agent_intents:
                        raise HireProjectionError("duplicate durable hire admission")
                    existing_app_owner = existing_app_intents.get(
                        created.existing_app_id
                    )
                    if (
                        created.existing_app_id
                        and existing_app_owner is not None
                        and states[existing_app_owner].phase is not HirePhase.ARCHIVED
                    ):
                        raise HireProjectionError("duplicate existing_app_id admission")
                    states[created.intent_id] = created
                    agent_intents[created.agent_id] = created.intent_id
                    bot_intents[created.bot_principal_id] = created.intent_id
                    if created.existing_app_id:
                        existing_app_intents[created.existing_app_id] = (
                            created.intent_id
                        )
                    continue
                intent_id = (
                    event.aggregate_id
                    if event.aggregate_id in states
                    else agent_intents.get(event.aggregate_id)
                    or bot_intents.get(event.aggregate_id)
                )
                if intent_id is None:
                    continue
                current = states[intent_id]
                if event.event_type == "hire.channel.crashed":
                    generation = event.payload.get("generation")
                    if (
                        set(event.payload) != {"generation"}
                        or event.aggregate_id != current.intent_id
                        or current.phase
                        not in {
                            HirePhase.CONFIGURING,
                            HirePhase.VALIDATING,
                            HirePhase.ACTIVE,
                            HirePhase.READY_PENDING_VERIFICATION,
                        }
                        or isinstance(generation, bool)
                        or not isinstance(generation, int)
                        or generation <= 0
                        or generation != current.channel_generation
                    ):
                        raise HireProjectionError(
                            "invalid Channel revalidation binding"
                        )
                    states[intent_id] = replace(
                        current,
                        automatic_reactivation_generation=generation,
                        last_sequence=frame.sequence,
                    )
                    continue
                if event.event_type == "hire.requester_identity_bound":
                    if set(event.payload) != {"requester_union_id"}:
                        raise HireProjectionError("invalid requester identity payload")
                    requester_union_id = _required_string(
                        event.payload,
                        "requester_union_id",
                    )
                    if (
                        event.aggregate_id != current.intent_id
                        or (
                            current.requester_union_id
                            and current.requester_union_id != requester_union_id
                        )
                    ):
                        raise HireProjectionError("requester identity binding mismatch")
                    states[intent_id] = replace(
                        current,
                        requester_union_id=requester_union_id,
                        last_sequence=frame.sequence,
                    )
                    continue
                if event.event_type == "employee.state_changed":
                    if set(event.payload) != {"state"}:
                        raise HireProjectionError("invalid hire phase payload")
                    try:
                        phase = HirePhase(_required_string(event.payload, "state"))
                    except ValueError as exc:
                        raise HireProjectionError("invalid hire phase") from exc
                    if phase not in _ALLOWED_PHASE_SUCCESSORS[current.phase]:
                        raise HireProjectionError("invalid hire phase transition")
                    states[intent_id] = replace(
                        current,
                        phase=phase,
                        last_sequence=frame.sequence,
                    )
                    if (
                        phase is HirePhase.ARCHIVED
                        and current.existing_app_id
                        and existing_app_intents.get(current.existing_app_id)
                        == intent_id
                    ):
                        existing_app_intents.pop(current.existing_app_id)
                    continue
                if event.event_type == "bot_principal.bound":
                    if event.aggregate_id != current.bot_principal_id:
                        raise HireProjectionError("hire bot principal mismatch")
                    if event.payload.get("agent_id") != current.agent_id:
                        raise HireProjectionError("hire bot principal agent mismatch")
                    app_id = _required_string(event.payload, "app_id")
                    credential_ref = _required_string(event.payload, "credential_ref")
                    if current.app_id and current.app_id != app_id:
                        raise HireProjectionError("hire app_id evidence mismatch")
                    if current.credential_ref and current.credential_ref != credential_ref:
                        raise HireProjectionError("hire credential evidence mismatch")
                    states[intent_id] = replace(
                        current,
                        app_id=app_id,
                        credential_ref=credential_ref,
                        last_sequence=frame.sequence,
                    )
                    continue
                if event.event_type == "hire.activation.automatic":
                    expected_keys = {
                        "tenant_key",
                        "app_id",
                        "agent_id",
                        "generation",
                        "slash_spec_hash",
                        "channel_connection_id",
                        "requester_principal_id",
                        "requester_union_id",
                        "source",
                        "activated_at",
                    }
                    activated_at = event.payload.get("activated_at")
                    automatic_reactivation = (
                        current.verification_consumed
                        and current.automatic_reactivation_generation > 0
                        and current.channel_generation
                        > current.automatic_reactivation_generation
                    )
                    if (
                        set(event.payload) != expected_keys
                        or event.aggregate_id != current.intent_id
                        or current.phase is not HirePhase.READY_PENDING_VERIFICATION
                        or (
                            current.verification_consumed
                            and not automatic_reactivation
                        )
                        or event.payload["tenant_key"] != current.tenant_key
                        or event.payload["app_id"] != current.app_id
                        or event.payload["agent_id"] != current.agent_id
                        or event.payload["generation"] != current.channel_generation
                        or event.payload["slash_spec_hash"] != current.slash_spec_hash
                        or event.payload["channel_connection_id"]
                        != current.channel_connection_id
                        or event.payload["requester_principal_id"]
                        != current.requester_principal_id
                        or event.payload["requester_union_id"]
                        != current.requester_union_id
                        or event.payload["source"] != "channel_ready"
                        or isinstance(activated_at, bool)
                        or not isinstance(activated_at, (int, float))
                        or not math.isfinite(float(activated_at))
                        or float(activated_at)
                        < max(current.slash_verified_at, current.channel_verified_at)
                    ):
                        raise HireProjectionError("invalid automatic activation binding")
                    states[intent_id] = replace(
                        current,
                        automatic_reactivation_generation=0,
                        verification_consumed=True,
                        activation_source="channel_ready",
                        activation_verified_at=float(activated_at),
                        last_sequence=frame.sequence,
                    )
                    continue
                next_effect = _EFFECT_EVENTS.get(event.event_type)
                if next_effect is None:
                    continue
                effect_id, effect_type, metadata_items = _effect_fact(event)
                effects = dict(current.effects)
                effect_types = dict(current.effect_types)
                effect_metadata = dict(current.effect_metadata)
                previous = effects.get(effect_id)
                previous_type = effect_types.get(effect_id)
                previous_metadata_items = effect_metadata.get(effect_id, ())
                if previous is next_effect:
                    if (
                        previous_type != effect_type
                        or previous_metadata_items != metadata_items
                    ):
                        raise HireProjectionError(
                            "conflicting duplicate hire effect transition"
                        )
                    states[intent_id] = replace(
                        current,
                        last_sequence=frame.sequence,
                    )
                    continue
                if previous not in _ALLOWED_EFFECT_PREDECESSORS[next_effect]:
                    raise HireProjectionError("invalid hire effect transition")
                if previous_type not in (None, effect_type):
                    raise HireProjectionError("hire effect type changed")
                previous_metadata = dict(previous_metadata_items)
                metadata = dict(metadata_items)
                for key, value in previous_metadata.items():
                    if key in metadata and metadata[key] != value:
                        raise HireProjectionError("hire effect metadata changed")
                    metadata.setdefault(key, value)
                app_id = metadata.get("app_id", current.app_id)
                credential_ref = metadata.get(
                    "credential_ref", current.credential_ref
                )
                slash_spec_hash = metadata.get(
                    "slash_spec_hash", current.slash_spec_hash
                )
                slash_observed_hash = metadata.get(
                    "slash_observed_hash", current.slash_observed_hash
                )
                slash_verified_at = current.slash_verified_at
                if "slash_verified_at" in metadata:
                    try:
                        slash_verified_at = float(metadata["slash_verified_at"])
                    except ValueError as exc:
                        raise HireProjectionError("invalid Slash evidence time") from exc
                    if slash_verified_at <= 0:
                        raise HireProjectionError("invalid Slash evidence time")
                generation_text = metadata.get("generation")
                channel_generation = current.channel_generation
                if generation_text is not None:
                    try:
                        channel_generation = int(generation_text)
                    except ValueError as exc:
                        raise HireProjectionError("invalid Channel generation") from exc
                    if channel_generation <= 0 or str(channel_generation) != generation_text:
                        raise HireProjectionError("invalid Channel generation")
                channel_identity_app_id = metadata.get(
                    "identity_app_id", current.channel_identity_app_id
                )
                channel_connection_id = metadata.get(
                    "connection_id", current.channel_connection_id
                )
                channel_verified_at = current.channel_verified_at
                if "channel_verified_at" in metadata:
                    try:
                        channel_verified_at = float(metadata["channel_verified_at"])
                    except ValueError as exc:
                        raise HireProjectionError("invalid Channel evidence time") from exc
                    if channel_verified_at <= 0:
                        raise HireProjectionError("invalid Channel evidence time")
                if current.app_id and app_id != current.app_id:
                    raise HireProjectionError("hire app_id evidence mismatch")
                if current.credential_ref and credential_ref != current.credential_ref:
                    raise HireProjectionError("hire credential evidence mismatch")
                effects[effect_id] = next_effect
                effect_types[effect_id] = effect_type
                effect_metadata[effect_id] = tuple(sorted(metadata.items()))
                states[intent_id] = replace(
                    current,
                    effects=tuple(sorted(effects.items())),
                    effect_types=tuple(sorted(effect_types.items())),
                    effect_metadata=tuple(sorted(effect_metadata.items())),
                    app_id=app_id,
                    credential_ref=credential_ref,
                    slash_spec_hash=slash_spec_hash,
                    slash_observed_hash=slash_observed_hash,
                    slash_verified_at=slash_verified_at,
                    channel_generation=channel_generation,
                    channel_identity_app_id=channel_identity_app_id,
                    channel_connection_id=channel_connection_id,
                    channel_verified_at=channel_verified_at,
                    last_sequence=frame.sequence,
                )
        return cls(MappingProxyType(states))


__all__ = [
    "DurableHireState",
    "HireEffectState",
    "HirePhase",
    "HireProjection",
    "HireProjectionError",
]
