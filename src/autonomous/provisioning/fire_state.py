"""Immutable replay state for the production employee retirement saga."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

from ..journal.frame import TransactionFrame


class FireProjectionError(RuntimeError):
    """A durable fire event stream violates the retirement contract."""


class FireEffectState(StrEnum):
    PREPARED = "prepared"
    EXECUTING = "executing"
    COMMITTED = "committed"
    ACTION_REQUIRED = "action_required"


class FirePhase(StrEnum):
    RETIRING = "retiring"
    ACTION_REQUIRED = "action_required"
    ARCHIVED = "archived"
    SUPERSEDED = "superseded"


class FireCleanupMode(StrEnum):
    """Authority available for cleaning resources at retirement admission."""

    BOUND = "bound"
    SAFE_ABORT = "safe_abort"
    RECOVERABLE = "recoverable"
    EXTERNAL_UNKNOWN = "external_unknown"


FIRE_EFFECT_ORDER: tuple[str, ...] = (
    "execution_quiesce",
    "slash_cleanup",
    "channel_stop",
    "membership_cleanup",
    "credential_destroy",
    "archive_move",
)


@dataclass(frozen=True, slots=True)
class DurableFireState:
    intent_id: str
    tenant_key: str
    message_id: str
    chat_id: str
    requester_principal_id: str
    agent_id: str
    employee_name: str
    bot_principal_id: str
    app_id: str
    credential_ref: str
    drain: bool
    cleanup_mode: FireCleanupMode = FireCleanupMode.BOUND
    phase: FirePhase = FirePhase.RETIRING
    effects: tuple[tuple[str, FireEffectState], ...] = ()
    error_code: str = ""
    external_disposition_confirmed: bool = False
    external_disposition_ref: str = ""
    external_disposed_by: str = ""
    external_disposed_at: str = ""
    requested_sequence: int = 0
    last_sequence: int = 0

    def effect_state(self, effect_type: str) -> FireEffectState | None:
        return dict(self.effects).get(effect_type)

    @property
    def pre_binding(self) -> bool:
        return self.cleanup_mode is not FireCleanupMode.BOUND


def rebuild_fire_projection(
    frames: tuple[TransactionFrame, ...],
) -> Mapping[str, DurableFireState]:
    states: dict[str, DurableFireState] = {}
    for frame in frames:
        for event in frame.events:
            current = states.get(event.aggregate_id)
            if event.event_type == "fire.requested":
                if current is not None:
                    raise FireProjectionError("duplicate fire request")
                payload = event.payload
                required = {
                    "intent_id", "tenant_key", "message_id", "chat_id",
                    "requester_principal_id", "agent_id", "employee_name", "bot_principal_id",
                    "app_id", "credential_ref", "drain",
                }
                allowed_shapes = {
                    frozenset(required),
                    frozenset(required | {"pre_binding"}),
                    frozenset(required | {"cleanup_mode"}),
                }
                if set(payload) not in allowed_shapes or payload.get("intent_id") != event.aggregate_id:
                    raise FireProjectionError("invalid fire request")
                try:
                    if "cleanup_mode" in payload:
                        cleanup_mode = FireCleanupMode(payload["cleanup_mode"])
                    elif payload.get("pre_binding") is True:
                        # Legacy pre-binding requests did not prove that no external
                        # call had started. Replay them fail-closed.
                        cleanup_mode = FireCleanupMode.EXTERNAL_UNKNOWN
                    elif payload.get("pre_binding") in {None, False}:
                        cleanup_mode = FireCleanupMode.BOUND
                    else:
                        raise ValueError
                except (TypeError, ValueError) as exc:
                    raise FireProjectionError("invalid fire request") from exc
                strings = {key: payload[key] for key in required - {"drain"}}
                identity_strings = {
                    key: strings[key]
                    for key in {
                        "intent_id",
                        "tenant_key",
                        "message_id",
                        "chat_id",
                        "requester_principal_id",
                        "agent_id",
                        "employee_name",
                    }
                }
                if any(
                    not isinstance(value, str) or not value
                    for value in identity_strings.values()
                ):
                    raise FireProjectionError("invalid fire request")
                resource_strings = {
                    key: strings[key]
                    for key in {"bot_principal_id", "app_id", "credential_ref"}
                }
                if any(
                    not isinstance(value, str)
                    for value in resource_strings.values()
                ) or (
                    cleanup_mode
                    in {
                        FireCleanupMode.SAFE_ABORT,
                        FireCleanupMode.EXTERNAL_UNKNOWN,
                    }
                    and (
                        resource_strings["bot_principal_id"]
                        or resource_strings["credential_ref"]
                    )
                ) or (
                    cleanup_mode
                    in {FireCleanupMode.BOUND, FireCleanupMode.RECOVERABLE}
                    and any(not value for value in resource_strings.values())
                ):
                    raise FireProjectionError("invalid fire request")
                if type(payload["drain"]) is not bool:
                    raise FireProjectionError("invalid fire request")
                states[event.aggregate_id] = DurableFireState(
                    **{
                        **{
                            key: value
                            for key, value in payload.items()
                            if key not in {"pre_binding", "cleanup_mode"}
                        },
                        "cleanup_mode": cleanup_mode,
                        "requested_sequence": frame.sequence,
                    },
                    last_sequence=frame.sequence,
                )
                continue
            if not event.event_type.startswith("fire."):
                continue
            if current is None:
                raise FireProjectionError("fire event references unknown request")
            if current.phase in {FirePhase.ARCHIVED, FirePhase.SUPERSEDED}:
                raise FireProjectionError(
                    "terminal fire request rejects later events"
                )
            if event.event_type == "fire.external_disposition_confirmed":
                required = {
                    "disposition_ref",
                    "disposed_by",
                    "disposed_at",
                    "confirmation_message_id",
                }
                if set(event.payload) != required or any(
                    not isinstance(event.payload.get(key), str)
                    or not event.payload[key]
                    for key in required
                ):
                    raise FireProjectionError(
                        "invalid external disposition confirmation"
                    )
                if (
                    current.cleanup_mode is not FireCleanupMode.EXTERNAL_UNKNOWN
                    or current.phase is not FirePhase.ACTION_REQUIRED
                    or current.error_code
                    != "external_cleanup_authority_unavailable"
                    or current.effect_state("credential_destroy")
                    is not FireEffectState.ACTION_REQUIRED
                ):
                    raise FireProjectionError(
                        "external disposition confirmation is not admissible"
                    )
                expected_ref = current.app_id or "NO_APP_FOUND"
                if event.payload["disposition_ref"] != expected_ref:
                    raise FireProjectionError(
                        "external disposition reference mismatch"
                    )
                try:
                    datetime.fromisoformat(event.payload["disposed_at"])
                except ValueError as exc:
                    raise FireProjectionError(
                        "invalid external disposition timestamp"
                    ) from exc
                effects = dict(current.effects)
                effects["credential_destroy"] = FireEffectState.COMMITTED
                current = replace(
                    current,
                    effects=tuple(
                        (name, effects[name])
                        for name in FIRE_EFFECT_ORDER
                        if name in effects
                    ),
                    phase=FirePhase.RETIRING,
                    error_code="",
                    external_disposition_confirmed=True,
                    external_disposition_ref=event.payload["disposition_ref"],
                    external_disposed_by=event.payload["disposed_by"],
                    external_disposed_at=event.payload["disposed_at"],
                    last_sequence=frame.sequence,
                )
            elif event.event_type == "fire.effect.reconciled":
                if set(event.payload) != {
                    "effect_type",
                    "resolution_code",
                } or event.payload.get("resolution_code") != "observed_committed":
                    raise FireProjectionError("invalid reconciled fire effect")
                effect_type = event.payload.get("effect_type")
                if (
                    effect_type not in FIRE_EFFECT_ORDER
                    or current.phase is not FirePhase.ACTION_REQUIRED
                    or current.effect_state(effect_type)
                    is not FireEffectState.ACTION_REQUIRED
                ):
                    raise FireProjectionError("invalid reconciled fire effect")
                effects = dict(current.effects)
                effects[effect_type] = FireEffectState.COMMITTED
                current = replace(
                    current,
                    effects=tuple(
                        (name, effects[name])
                        for name in FIRE_EFFECT_ORDER
                        if name in effects
                    ),
                    phase=FirePhase.RETIRING,
                    error_code="",
                    last_sequence=frame.sequence,
                )
            elif event.event_type.startswith("fire.effect."):
                effect_type = event.payload.get("effect_type")
                if effect_type not in FIRE_EFFECT_ORDER or set(event.payload) - {
                    "effect_type", "error_code"
                }:
                    raise FireProjectionError("invalid fire effect")
                next_state = FireEffectState(event.event_type.removeprefix("fire.effect."))
                previous = current.effect_state(effect_type)
                allowed = {
                    FireEffectState.PREPARED: {None},
                    FireEffectState.EXECUTING: {FireEffectState.PREPARED},
                    FireEffectState.COMMITTED: {FireEffectState.EXECUTING},
                    FireEffectState.ACTION_REQUIRED: {FireEffectState.EXECUTING},
                }
                membership_revalidation = (
                    next_state is FireEffectState.EXECUTING
                    and previous is FireEffectState.COMMITTED
                    and effect_type == "membership_cleanup"
                    and current.phase is FirePhase.RETIRING
                )
                if previous not in allowed[next_state] and not membership_revalidation:
                    raise FireProjectionError("invalid fire effect transition")
                effects = dict(current.effects)
                effects[effect_type] = next_state
                error = str(event.payload.get("error_code") or "")
                if next_state is FireEffectState.ACTION_REQUIRED and not error:
                    raise FireProjectionError("action required needs an error code")
                current = replace(
                    current,
                    effects=tuple((name, effects[name]) for name in FIRE_EFFECT_ORDER if name in effects),
                    phase=(FirePhase.ACTION_REQUIRED if next_state is FireEffectState.ACTION_REQUIRED else current.phase),
                    error_code=error,
                    last_sequence=frame.sequence,
                )
            elif event.event_type == "fire.superseded":
                if (
                    set(event.payload) != {"canonical_intent_id"}
                    or not isinstance(
                        event.payload.get("canonical_intent_id"), str
                    )
                    or not event.payload["canonical_intent_id"]
                    or event.payload["canonical_intent_id"] == current.intent_id
                    or current.phase
                    not in {FirePhase.RETIRING, FirePhase.ACTION_REQUIRED}
                ):
                    raise FireProjectionError("invalid fire supersession")
                current = replace(
                    current,
                    phase=FirePhase.SUPERSEDED,
                    last_sequence=frame.sequence,
                )
            elif event.event_type == "fire.completed":
                disposition = event.payload.get("external_app_disposition")
                if set(event.payload) != {"external_app_disposition"} or disposition not in {
                    "manual_deletion_required",
                    "manual_disposition_confirmed",
                }:
                    raise FireProjectionError("invalid fire completion")
                if (
                    current.cleanup_mode is FireCleanupMode.EXTERNAL_UNKNOWN
                    and (
                        not current.external_disposition_confirmed
                        or disposition != "manual_disposition_confirmed"
                    )
                ) or (
                    current.cleanup_mode is not FireCleanupMode.EXTERNAL_UNKNOWN
                    and disposition == "manual_disposition_confirmed"
                ):
                    raise FireProjectionError(
                        "fire completion lacks external disposition evidence"
                    )
                if any(current.effect_state(name) is not FireEffectState.COMMITTED for name in FIRE_EFFECT_ORDER):
                    raise FireProjectionError("fire completed before all effects")
                current = replace(current, phase=FirePhase.ARCHIVED, last_sequence=frame.sequence)
            else:
                raise FireProjectionError("unknown fire event")
            states[event.aggregate_id] = current
    return MappingProxyType(states)


__all__ = [
    "DurableFireState",
    "FIRE_EFFECT_ORDER",
    "FireCleanupMode",
    "FireEffectState",
    "FirePhase",
    "FireProjectionError",
    "rebuild_fire_projection",
]
