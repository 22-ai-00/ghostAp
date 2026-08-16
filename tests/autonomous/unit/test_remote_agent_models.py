from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.autonomous.journal.blob_store import BlobRef
from src.autonomous.remote.models import (
    A2A_SPEC_RELEASE,
    A2A_WIRE_PROTOCOL_VERSION,
    RemoteAgentDescriptor,
    RemoteDispatchRequest,
    RemoteObservation,
    RemoteTaskState,
)


def _ref(payload: bytes = b"payload") -> BlobRef:
    labels = {"kind": "a2a-test"}
    payload_hash = hashlib.sha256(payload).hexdigest()
    return BlobRef(
        blob_id="a" * 64,
        ciphertext_hash="a" * 64,
        payload_hash=payload_hash,
        content_hash=payload_hash,
        labels_hash=hashlib.sha256(
            json.dumps(labels, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        size=len(payload),
        labels=labels,
        key_ref="k1",
    )


def _descriptor(**overrides: object) -> RemoteAgentDescriptor:
    values: dict[str, object] = {
        "tenant_key": "tenant-1",
        "agent_id": "agt_remote-reviewer",
        "card_url": "https://agent.example/.well-known/agent-card.json",
        "endpoint_url": "https://agent.example/a2a",
        "card_digest": "b" * 64,
        "credential_ref": "cred_remote-reviewer",
    }
    values.update(overrides)
    return RemoteAgentDescriptor(**values)


def test_phase_zero_contracts_are_frozen_and_sdk_free() -> None:
    assert A2A_SPEC_RELEASE == "1.0.1"
    assert A2A_WIRE_PROTOCOL_VERSION == "1.0"
    descriptor = _descriptor()
    with pytest.raises(FrozenInstanceError):
        descriptor.agent_id = "agt_other"  # type: ignore[misc]

    remote_root = Path("src/autonomous/remote")
    imported_roots: set[str] = set()
    for path in remote_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
    assert "a2a" not in imported_roots


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("card_url", "http://agent.example/card"),
        ("card_url", "https://user:secret@agent.example/card"),
        ("endpoint_url", "https://agent.example/a2a?token=secret"),
        ("endpoint_url", "https://agent.example/a2a#fragment"),
        ("card_digest", "not-a-digest"),
        ("protocol_version", "1.1.2"),
    ],
)
def test_descriptor_rejects_unfrozen_or_ambiguous_binding(
    field: str,
    value: str,
) -> None:
    with pytest.raises(ValueError):
        _descriptor(**{field: value})


def test_remote_completed_is_only_a_claimed_completion() -> None:
    assert "COMPLETED" not in RemoteTaskState.__members__
    assert RemoteTaskState.CLAIMED_COMPLETED.is_remote_terminal is True

    ref = _ref()
    observation = RemoteObservation(
        observation_id="obs-1",
        state=RemoteTaskState.CLAIMED_COMPLETED,
        context_id="ctx-1",
        task_id="task-1",
        payload_digest=ref.payload_hash,
        payload_ref=ref,
    )
    assert observation.state.value == "claimed_completed"


def test_dispatch_request_hides_and_bounds_plaintext_instruction() -> None:
    request = RemoteDispatchRequest(
        acceptance_id="a2a-acceptance-1",
        run_id="run-1",
        assignment_id="assignment-1",
        attempt_id="attempt-1",
        message_id="message-1",
        context_id="context-1",
        instruction="review the patch",
        descriptor=_descriptor(),
    )
    assert "review the patch" not in repr(request)

    with pytest.raises(ValueError, match="instruction"):
        RemoteDispatchRequest(
            acceptance_id="a2a-acceptance-1",
            run_id="run-1",
            assignment_id="assignment-1",
            attempt_id="attempt-1",
            message_id="message-1",
            context_id="context-1",
            instruction="\x00",
            descriptor=_descriptor(),
        )


def test_observation_body_must_match_encrypted_blob_digest() -> None:
    with pytest.raises(ValueError, match="digest mismatch"):
        RemoteObservation(
            observation_id="obs-1",
            state=RemoteTaskState.WORKING,
            context_id="ctx-1",
            payload_digest="f" * 64,
            payload_ref=_ref(),
        )
