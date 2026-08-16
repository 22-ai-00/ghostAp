from __future__ import annotations

import hashlib
import json

import pytest

from src.autonomous.journal.blob_store import BlobRef
from src.autonomous.remote.models import (
    RemoteAgentDescriptor,
    RemoteAttemptPhase,
    RemoteDispatchRequest,
    RemoteObservation,
    RemoteTaskState,
)
from src.autonomous.remote.projection import (
    RemoteProjectionError,
    cancel_requested_event,
    executing_event,
    observation_event,
    prepared_event,
    rebuild_remote_projection,
    send_uncertain_event,
    synthetic_frame,
    task_bound_event,
)


def _ref(payload: bytes, *, marker: str) -> BlobRef:
    labels = {"kind": "a2a-test", "marker": marker}
    payload_hash = hashlib.sha256(payload).hexdigest()
    return BlobRef(
        blob_id=hashlib.sha256(marker.encode()).hexdigest(),
        ciphertext_hash=hashlib.sha256(marker.encode()).hexdigest(),
        payload_hash=payload_hash,
        content_hash=payload_hash,
        labels_hash=hashlib.sha256(
            json.dumps(labels, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        size=len(payload),
        labels=labels,
        key_ref="k1",
    )


def _request() -> RemoteDispatchRequest:
    return RemoteDispatchRequest(
        acceptance_id="a2a-acceptance-1",
        run_id="run-1",
        assignment_id="assignment-1",
        attempt_id="attempt-1",
        message_id="message-1",
        context_id="context-1",
        instruction="perform an independent review",
        descriptor=RemoteAgentDescriptor(
            tenant_key="tenant-1",
            agent_id="agt_remote-reviewer",
            card_url="https://agent.example/.well-known/agent-card.json",
            endpoint_url="https://agent.example/a2a",
            card_digest="b" * 64,
            credential_ref="cred-1",
        ),
    )


def _prepared():
    request = _request()
    instruction_ref = _ref(request.instruction.encode(), marker="instruction")
    first = prepared_event(request, instruction_ref)
    projection = rebuild_remote_projection((synthetic_frame(first),))
    return request, first, projection.by_key[("run-1", "assignment-1", "attempt-1")]


def test_mapping_is_anchored_before_execution_and_remote_task_binding() -> None:
    request, prepared, snapshot = _prepared()
    assert request.instruction not in json.dumps(prepared.payload)
    assert snapshot.phase is RemoteAttemptPhase.PREPARED
    assert snapshot.handle.message_id == "message-1"
    assert snapshot.handle.context_id == "context-1"
    assert snapshot.handle.task_id == ""
    assert snapshot.handle.descriptor.card_digest == "b" * 64

    executing = executing_event(snapshot.handle)
    task_bound = task_bound_event(snapshot.handle, "task-1")
    projection = rebuild_remote_projection(
        (synthetic_frame(prepared), synthetic_frame(executing), synthetic_frame(task_bound))
    )
    observed = projection.by_key[snapshot.handle.key]
    assert observed.phase is RemoteAttemptPhase.TRACKING
    assert observed.handle.task_id == "task-1"


def test_claimed_completion_is_persisted_but_never_locally_completed() -> None:
    _request_value, prepared, snapshot = _prepared()
    events = [prepared, executing_event(snapshot.handle), task_bound_event(snapshot.handle, "task-1")]
    projection = rebuild_remote_projection((synthetic_frame(*events),))
    handle = projection.by_key[snapshot.handle.key].handle
    payload_ref = _ref(b'{"review":"pass"}', marker="result")
    completed = RemoteObservation(
        observation_id="obs-completed-1",
        state=RemoteTaskState.CLAIMED_COMPLETED,
        context_id=handle.context_id,
        task_id=handle.task_id,
        payload_digest=payload_ref.payload_hash,
        payload_ref=payload_ref,
        artifact_id="artifact-review",
        last_chunk=True,
    )
    projection = rebuild_remote_projection(
        (synthetic_frame(*events, observation_event(handle, completed)),)
    )
    result = projection.by_key[handle.key]
    assert result.phase is RemoteAttemptPhase.TERMINAL
    assert result.claimed_completed is True
    assert result.state.value != "completed"


def test_duplicate_observation_is_idempotent_but_collision_fails_closed() -> None:
    _request_value, prepared, snapshot = _prepared()
    events = [prepared, executing_event(snapshot.handle), task_bound_event(snapshot.handle, "task-1")]
    projection = rebuild_remote_projection((synthetic_frame(*events),))
    handle = projection.by_key[snapshot.handle.key].handle
    digest = hashlib.sha256(b"").hexdigest()
    working = RemoteObservation(
        observation_id="obs-working-1",
        state=RemoteTaskState.WORKING,
        context_id=handle.context_id,
        task_id=handle.task_id,
        payload_digest=digest,
    )
    event = observation_event(handle, working)
    projection = rebuild_remote_projection((synthetic_frame(*events, event, event),))
    assert projection.by_key[handle.key].observations == (working,)

    collision = RemoteObservation(
        observation_id=working.observation_id,
        state=RemoteTaskState.FAILED,
        context_id=handle.context_id,
        task_id=handle.task_id,
        payload_digest=digest,
    )
    with pytest.raises(RemoteProjectionError, match="collision"):
        rebuild_remote_projection(
            (
                synthetic_frame(
                    *events,
                    event,
                    observation_event(handle, collision),
                ),
            )
        )


def test_binding_drift_and_late_terminal_replacement_fail_closed() -> None:
    _request_value, prepared, snapshot = _prepared()
    executing = executing_event(snapshot.handle)
    wrong_context = task_bound_event(snapshot.handle, "task-1")
    wrong_context.payload["context_id"] = "context-other"
    with pytest.raises(RemoteProjectionError, match="binding mismatch"):
        rebuild_remote_projection((synthetic_frame(prepared, executing, wrong_context),))

    bound = task_bound_event(snapshot.handle, "task-1")
    projection = rebuild_remote_projection((synthetic_frame(prepared, executing, bound),))
    handle = projection.by_key[snapshot.handle.key].handle
    digest = hashlib.sha256(b"").hexdigest()
    failed = RemoteObservation(
        observation_id="obs-failed",
        state=RemoteTaskState.FAILED,
        context_id=handle.context_id,
        task_id=handle.task_id,
        payload_digest=digest,
    )
    late = RemoteObservation(
        observation_id="obs-late-working",
        state=RemoteTaskState.WORKING,
        context_id=handle.context_id,
        task_id=handle.task_id,
        payload_digest=digest,
    )
    with pytest.raises(RemoteProjectionError, match="terminal"):
        rebuild_remote_projection(
            (
                synthetic_frame(
                    prepared,
                    executing,
                    bound,
                    observation_event(handle, failed),
                    observation_event(handle, late),
                ),
            )
        )


def test_unknown_send_and_cancel_are_anchored_in_distinct_safe_states() -> None:
    _request_value, prepared, snapshot = _prepared()
    executing = executing_event(snapshot.handle)
    uncertain = rebuild_remote_projection(
        (synthetic_frame(prepared, executing, send_uncertain_event(snapshot.handle)),)
    ).by_key[snapshot.handle.key]
    assert uncertain.phase is RemoteAttemptPhase.SEND_UNCERTAIN
    assert uncertain.state is RemoteTaskState.OUTCOME_UNCERTAIN
    assert uncertain.handle.message_id == snapshot.handle.message_id

    bound_projection = rebuild_remote_projection(
        (
            synthetic_frame(
                prepared,
                executing,
                task_bound_event(snapshot.handle, "task-1"),
            ),
        )
    )
    handle = bound_projection.by_key[snapshot.handle.key].handle
    cancel = cancel_requested_event(handle)
    canceled = rebuild_remote_projection(
        (
            synthetic_frame(
                prepared,
                executing,
                task_bound_event(snapshot.handle, "task-1"),
                cancel,
                cancel,
            ),
        )
    ).by_key[handle.key]
    assert canceled.phase is RemoteAttemptPhase.CANCEL_REQUESTED
    assert canceled.cancel_requested is True


def test_external_effect_cannot_start_without_preparation() -> None:
    _request_value, _prepared_event, snapshot = _prepared()
    with pytest.raises(RemoteProjectionError, match="precedes preparation"):
        rebuild_remote_projection((synthetic_frame(executing_event(snapshot.handle)),))
