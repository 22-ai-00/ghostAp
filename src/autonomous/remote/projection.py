"""Pure replay projection for outbound remote dispatch Journal events."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Iterable, Mapping

from ..journal.blob_store import BlobRef
from ..journal.frame import JournalEvent
from .models import (
    RemoteAgentDescriptor,
    RemoteAttemptPhase,
    RemoteDispatchRequest,
    RemoteObservation,
    RemoteProjection,
    RemoteProtocolBinding,
    RemoteSnapshot,
    RemoteTaskHandle,
    RemoteTaskState,
)

EVENT_PREPARED = "remote.v1.dispatch.prepared"
EVENT_EXECUTING = "remote.v1.dispatch.executing"
EVENT_TASK_BOUND = "remote.v1.task_bound"
EVENT_OBSERVATION = "remote.v1.observation_recorded"
EVENT_SEND_UNCERTAIN = "remote.v1.send_uncertain"
EVENT_CANCEL_REQUESTED = "remote.v1.cancel_requested"


class RemoteProjectionError(RuntimeError):
    """A remote Journal event violated its frozen authority or lifecycle."""


def _descriptor_payload(descriptor: RemoteAgentDescriptor) -> dict[str, str]:
    return {
        "tenant_key": descriptor.tenant_key,
        "agent_id": descriptor.agent_id,
        "card_url": descriptor.card_url,
        "endpoint_url": descriptor.endpoint_url,
        "card_digest": descriptor.card_digest,
        "credential_ref": descriptor.credential_ref,
        "protocol_binding": descriptor.protocol_binding.value,
        "protocol_version": descriptor.protocol_version,
        "remote_tenant": descriptor.remote_tenant,
    }


def _descriptor_from_payload(payload: Mapping[str, object]) -> RemoteAgentDescriptor:
    return RemoteAgentDescriptor(
        tenant_key=str(payload["tenant_key"]),
        agent_id=str(payload["agent_id"]),
        card_url=str(payload["card_url"]),
        endpoint_url=str(payload["endpoint_url"]),
        card_digest=str(payload["card_digest"]),
        credential_ref=str(payload.get("credential_ref", "")),
        protocol_binding=RemoteProtocolBinding(str(payload["protocol_binding"])),
        protocol_version=str(payload["protocol_version"]),
        remote_tenant=str(payload.get("remote_tenant", "")),
    )


def prepared_event(
    request: RemoteDispatchRequest,
    instruction_ref: BlobRef,
) -> JournalEvent:
    """Create the durable intent that must be anchored before any send."""

    if not isinstance(instruction_ref, BlobRef):
        raise TypeError("instruction_ref must be BlobRef")
    return JournalEvent(
        event_type=EVENT_PREPARED,
        aggregate_id=request.acceptance_id,
        payload={
            "schema_version": 1,
            "run_id": request.run_id,
            "assignment_id": request.assignment_id,
            "attempt_id": request.attempt_id,
            "acceptance_id": request.acceptance_id,
            "message_id": request.message_id,
            "context_id": request.context_id,
            "blob_ref": instruction_ref.to_dict(),
            **_descriptor_payload(request.descriptor),
        },
    )


def executing_event(handle: RemoteTaskHandle) -> JournalEvent:
    """Create the second anchor immediately preceding the network effect."""

    return JournalEvent(
        event_type=EVENT_EXECUTING,
        aggregate_id=handle.acceptance_id,
        payload={"schema_version": 1, "run_id": handle.run_id},
    )


def task_bound_event(handle: RemoteTaskHandle, task_id: str) -> JournalEvent:
    """Bind the first server-issued task ID to the frozen local attempt."""

    return JournalEvent(
        event_type=EVENT_TASK_BOUND,
        aggregate_id=handle.acceptance_id,
        payload={
            "schema_version": 1,
            "run_id": handle.run_id,
            "context_id": handle.context_id,
            "task_id": task_id,
            "agent_id": handle.descriptor.agent_id,
            "card_digest": handle.descriptor.card_digest,
        },
    )


def observation_event(
    handle: RemoteTaskHandle,
    observation: RemoteObservation,
) -> JournalEvent:
    """Create an observation event after its untrusted body is published."""

    payload: dict[str, object] = {
        "schema_version": 1,
        "run_id": handle.run_id,
        "observation_id": observation.observation_id,
        "state": observation.state.value,
        "context_id": observation.context_id,
        "task_id": observation.task_id,
        "payload_digest": observation.payload_digest,
        "artifact_id": observation.artifact_id,
        "append": observation.append,
        "last_chunk": observation.last_chunk,
        "agent_id": handle.descriptor.agent_id,
        "card_digest": handle.descriptor.card_digest,
    }
    if observation.payload_ref is not None:
        payload["blob_ref"] = observation.payload_ref.to_dict()
    return JournalEvent(
        event_type=EVENT_OBSERVATION,
        aggregate_id=handle.acceptance_id,
        payload=payload,
    )


def send_uncertain_event(handle: RemoteTaskHandle) -> JournalEvent:
    """Record an unknown send outcome without reflecting exception detail."""

    return JournalEvent(
        event_type=EVENT_SEND_UNCERTAIN,
        aggregate_id=handle.acceptance_id,
        payload={
            "schema_version": 1,
            "run_id": handle.run_id,
            "error_code": "remote_send_outcome_unknown",
        },
    )


def cancel_requested_event(handle: RemoteTaskHandle) -> JournalEvent:
    """Anchor local cancel intent before calling the remote service."""

    return JournalEvent(
        event_type=EVENT_CANCEL_REQUESTED,
        aggregate_id=handle.acceptance_id,
        payload={
            "schema_version": 1,
            "run_id": handle.run_id,
            "task_id": handle.task_id,
            "context_id": handle.context_id,
        },
    )


def rebuild_remote_projection(frames: Iterable[object]) -> RemoteProjection:
    """Rebuild the authoritative remote-attempt mapping from Journal frames."""

    by_key: dict[tuple[str, str, str], RemoteSnapshot] = {}
    by_acceptance_id: dict[str, tuple[str, str, str]] = {}
    for frame in frames:
        for event in frame.events:
            if event.event_type.startswith("remote.v1."):
                _apply_event(by_key, by_acceptance_id, event)
    return RemoteProjection(by_key, by_acceptance_id)


def project_event(
    projection: RemoteProjection,
    event: JournalEvent,
) -> RemoteProjection:
    """Apply one event to an existing immutable projection."""

    by_key = dict(projection.by_key)
    by_acceptance_id = dict(projection.by_acceptance_id)
    _apply_event(by_key, by_acceptance_id, event)
    return RemoteProjection(by_key, by_acceptance_id)


def _apply_event(
    by_key: dict[tuple[str, str, str], RemoteSnapshot],
    by_acceptance_id: dict[str, tuple[str, str, str]],
    event: JournalEvent,
) -> None:
    payload = event.payload
    if payload.get("schema_version") != 1:
        raise RemoteProjectionError("unsupported remote event schema")
    acceptance_id = event.aggregate_id
    if event.event_type == EVENT_PREPARED:
        if acceptance_id in by_acceptance_id:
            raise RemoteProjectionError("duplicate remote dispatch preparation")
        try:
            descriptor = _descriptor_from_payload(payload)
            handle = RemoteTaskHandle(
                acceptance_id=str(payload["acceptance_id"]),
                run_id=str(payload["run_id"]),
                assignment_id=str(payload["assignment_id"]),
                attempt_id=str(payload["attempt_id"]),
                message_id=str(payload["message_id"]),
                context_id=str(payload["context_id"]),
                descriptor=descriptor,
                instruction_ref=BlobRef.from_dict(payload["blob_ref"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RemoteProjectionError("invalid remote dispatch preparation") from exc
        if handle.acceptance_id != acceptance_id or handle.key in by_key:
            raise RemoteProjectionError("remote dispatch authority collision")
        by_key[handle.key] = RemoteSnapshot(
            handle=handle,
            phase=RemoteAttemptPhase.PREPARED,
            state=RemoteTaskState.PREPARED,
        )
        by_acceptance_id[acceptance_id] = handle.key
        return

    key = by_acceptance_id.get(acceptance_id)
    if key is None:
        raise RemoteProjectionError("remote event precedes preparation")
    snapshot = by_key[key]
    handle = snapshot.handle
    if str(payload.get("run_id", "")) != handle.run_id:
        raise RemoteProjectionError("remote event run authority mismatch")

    if event.event_type == EVENT_EXECUTING:
        if snapshot.phase is not RemoteAttemptPhase.PREPARED:
            raise RemoteProjectionError("remote dispatch cannot execute twice")
        by_key[key] = replace(
            snapshot,
            phase=RemoteAttemptPhase.EXECUTING,
            state=RemoteTaskState.EXECUTING,
        )
        return
    if event.event_type == EVENT_TASK_BOUND:
        _assert_frozen_authority(payload, handle)
        task_id = str(payload.get("task_id", ""))
        context_id = str(payload.get("context_id", ""))
        if context_id != handle.context_id or not task_id:
            raise RemoteProjectionError("remote task binding mismatch")
        if handle.task_id and handle.task_id != task_id:
            raise RemoteProjectionError("remote task ID drift")
        if snapshot.phase not in {
            RemoteAttemptPhase.EXECUTING,
            RemoteAttemptPhase.TRACKING,
            RemoteAttemptPhase.CANCEL_REQUESTED,
        }:
            raise RemoteProjectionError("remote task binding is out of order")
        by_key[key] = replace(
            snapshot,
            handle=replace(handle, task_id=task_id),
            phase=(
                snapshot.phase
                if snapshot.phase is RemoteAttemptPhase.CANCEL_REQUESTED
                else RemoteAttemptPhase.TRACKING
            ),
        )
        return
    if event.event_type == EVENT_OBSERVATION:
        _assert_frozen_authority(payload, handle)
        try:
            payload_ref = (
                BlobRef.from_dict(payload["blob_ref"])
                if "blob_ref" in payload
                else None
            )
            observation = RemoteObservation(
                observation_id=str(payload["observation_id"]),
                state=RemoteTaskState(str(payload["state"])),
                context_id=str(payload["context_id"]),
                task_id=str(payload.get("task_id", "")),
                payload_digest=str(payload["payload_digest"]),
                payload_ref=payload_ref,
                artifact_id=str(payload.get("artifact_id", "")),
                append=payload.get("append", False),
                last_chunk=payload.get("last_chunk", False),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RemoteProjectionError("invalid remote observation") from exc
        if observation.context_id != handle.context_id:
            raise RemoteProjectionError("remote observation context mismatch")
        if handle.task_id:
            if observation.task_id != handle.task_id:
                raise RemoteProjectionError("remote observation task mismatch")
        elif observation.task_id:
            raise RemoteProjectionError("remote observation precedes task binding")
        duplicate = next(
            (
                item
                for item in snapshot.observations
                if item.observation_id == observation.observation_id
            ),
            None,
        )
        if duplicate is not None:
            if duplicate != observation:
                raise RemoteProjectionError("remote observation ID collision")
            return
        if observation.artifact_id and any(
            item.artifact_id == observation.artifact_id and item.last_chunk
            for item in snapshot.observations
        ):
            raise RemoteProjectionError("remote artifact continued after final chunk")
        _assert_state_transition(snapshot.state, observation.state)
        phase = (
            RemoteAttemptPhase.TERMINAL
            if observation.state.is_remote_terminal
            else snapshot.phase
        )
        if phase not in {
            RemoteAttemptPhase.TRACKING,
            RemoteAttemptPhase.CANCEL_REQUESTED,
            RemoteAttemptPhase.TERMINAL,
            RemoteAttemptPhase.EXECUTING,
        }:
            raise RemoteProjectionError("remote observation is out of order")
        by_key[key] = replace(
            snapshot,
            phase=phase,
            state=observation.state,
            observations=snapshot.observations + (observation,),
        )
        return
    if event.event_type == EVENT_SEND_UNCERTAIN:
        if snapshot.phase is not RemoteAttemptPhase.EXECUTING or handle.task_id:
            raise RemoteProjectionError("remote send uncertainty is out of order")
        by_key[key] = replace(
            snapshot,
            phase=RemoteAttemptPhase.SEND_UNCERTAIN,
            state=RemoteTaskState.OUTCOME_UNCERTAIN,
            error_code="remote_send_outcome_unknown",
        )
        return
    if event.event_type == EVENT_CANCEL_REQUESTED:
        if (
            not handle.task_id
            or str(payload.get("task_id", "")) != handle.task_id
            or str(payload.get("context_id", "")) != handle.context_id
        ):
            raise RemoteProjectionError("remote cancel authority mismatch")
        if snapshot.phase is RemoteAttemptPhase.CANCEL_REQUESTED:
            return
        if snapshot.phase is not RemoteAttemptPhase.TRACKING:
            raise RemoteProjectionError("remote cancel is out of order")
        by_key[key] = replace(
            snapshot,
            phase=RemoteAttemptPhase.CANCEL_REQUESTED,
            cancel_requested=True,
        )
        return
    raise RemoteProjectionError("unknown remote Journal event")


def _assert_frozen_authority(
    payload: Mapping[str, object],
    handle: RemoteTaskHandle,
) -> None:
    if (
        str(payload.get("agent_id", "")) != handle.descriptor.agent_id
        or str(payload.get("card_digest", "")) != handle.descriptor.card_digest
    ):
        raise RemoteProjectionError("remote Agent binding drift")


def _assert_state_transition(
    previous: RemoteTaskState,
    observed: RemoteTaskState,
) -> None:
    if previous.is_remote_terminal:
        raise RemoteProjectionError("remote terminal observation cannot be replaced")
    allowed = {
        RemoteTaskState.EXECUTING: {
            RemoteTaskState.EXECUTING,
            RemoteTaskState.SUBMITTED,
            RemoteTaskState.WORKING,
            RemoteTaskState.INPUT_REQUIRED,
            RemoteTaskState.AUTH_REQUIRED,
            RemoteTaskState.CLAIMED_COMPLETED,
            RemoteTaskState.FAILED,
            RemoteTaskState.REJECTED,
            RemoteTaskState.CANCELED,
        },
        RemoteTaskState.SUBMITTED: {
            RemoteTaskState.SUBMITTED,
            RemoteTaskState.WORKING,
            RemoteTaskState.INPUT_REQUIRED,
            RemoteTaskState.AUTH_REQUIRED,
            RemoteTaskState.CLAIMED_COMPLETED,
            RemoteTaskState.FAILED,
            RemoteTaskState.REJECTED,
            RemoteTaskState.CANCELED,
        },
        RemoteTaskState.WORKING: {
            RemoteTaskState.WORKING,
            RemoteTaskState.INPUT_REQUIRED,
            RemoteTaskState.AUTH_REQUIRED,
            RemoteTaskState.CLAIMED_COMPLETED,
            RemoteTaskState.FAILED,
            RemoteTaskState.REJECTED,
            RemoteTaskState.CANCELED,
        },
        RemoteTaskState.INPUT_REQUIRED: {
            RemoteTaskState.INPUT_REQUIRED,
            RemoteTaskState.WORKING,
            RemoteTaskState.CLAIMED_COMPLETED,
            RemoteTaskState.FAILED,
            RemoteTaskState.REJECTED,
            RemoteTaskState.CANCELED,
        },
        RemoteTaskState.AUTH_REQUIRED: {
            RemoteTaskState.AUTH_REQUIRED,
            RemoteTaskState.WORKING,
            RemoteTaskState.CLAIMED_COMPLETED,
            RemoteTaskState.FAILED,
            RemoteTaskState.REJECTED,
            RemoteTaskState.CANCELED,
        },
    }
    if observed not in allowed.get(previous, set()):
        raise RemoteProjectionError("invalid remote task state transition")


def synthetic_frame(*events: JournalEvent) -> object:
    """Return a tiny frame adapter useful for projection validation/tests."""

    return SimpleNamespace(events=tuple(events))


__all__ = [
    "EVENT_CANCEL_REQUESTED",
    "EVENT_EXECUTING",
    "EVENT_OBSERVATION",
    "EVENT_PREPARED",
    "EVENT_SEND_UNCERTAIN",
    "EVENT_TASK_BOUND",
    "RemoteProjectionError",
    "cancel_requested_event",
    "executing_event",
    "observation_event",
    "prepared_event",
    "project_event",
    "rebuild_remote_projection",
    "send_uncertain_event",
    "synthetic_frame",
    "task_bound_event",
]
