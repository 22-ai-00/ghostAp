"""Team blobs remain live while sharing the Employee Ingress BlobStore."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from src.autonomous.ingress.models import EmployeeIngressMetadata, EmployeeIngressPayload
from src.autonomous.ingress.projection import IngressProjectionState
from src.autonomous.ingress.service import EmployeeIngressService, IngressBlobError
from src.autonomous.journal.anchor import FileAnchor
from src.autonomous.journal.blob_store import (
    AesGcmEncryptionProvider,
    BlobPublishError,
    BlobRef,
    BlobStore,
)
from src.autonomous.journal.writer import JournalWriter
from src.autonomous.team.coordinator import TeamCoordinatorActor, TeamCoordinatorError
from src.autonomous.team.models import TeamRunPhase
from tests.autonomous.team_helpers import ImmediateTeamBackend

_HMAC_KEY = b"team-shared-blob-retention-journal-key"
_DATA_KEY = b"t" * 32


def _open_writer(root: Path, anchor: object, *, epoch: int) -> JournalWriter:
    return JournalWriter.open(
        root / "journal",
        anchor=anchor,
        hmac_key=_HMAC_KEY,
        writer_epoch=epoch,
    )


def _open_store(root: Path) -> BlobStore:
    return BlobStore(
        root / "ingress-blobs",
        AesGcmEncryptionProvider(lambda _key_ref: _DATA_KEY),
    )


def _open_ingress(writer: JournalWriter, store: BlobStore) -> EmployeeIngressService:
    return EmployeeIngressService(
        writer=writer,
        blob_store=store,
        ingress_state=IngressProjectionState(),
        active_key_id="team-key",
    )


def _open_team(
    writer: JournalWriter,
    ingress: EmployeeIngressService,
    backend: ImmediateTeamBackend,
) -> TeamCoordinatorActor:
    return TeamCoordinatorActor(
        writer=writer,
        blob_store=ingress.blob_store,
        active_key_id="team-key",
        backend=backend,
        poll_seconds=0.001,
        blob_retainer=ingress.retain_shared_blob,
        blob_releaser=ingress.release_shared_blob,
    )


def _team_blob_refs(actor: TeamCoordinatorActor) -> tuple[BlobRef, ...]:
    projection = actor.projection()
    refs: dict[str, BlobRef] = {}
    for run in projection.runs.values():
        refs[run.task_ref.blob_id] = run.task_ref
        if run.final_result_ref is not None:
            refs[run.final_result_ref.blob_id] = run.final_result_ref
    for assignment in projection.assignments.values():
        refs[assignment.instruction_ref.blob_id] = assignment.instruction_ref
        if assignment.contribution_ref is not None:
            refs[assignment.contribution_ref.blob_id] = assignment.contribution_ref
    return tuple(refs.values())


def test_live_team_blobs_survive_ingress_hygiene_and_restart_order(
    tmp_path: Path,
) -> None:
    anchor = FileAnchor(tmp_path / "anchor.json")
    writer = _open_writer(tmp_path, anchor, epoch=1)
    store = _open_store(tmp_path)
    ingress = _open_ingress(writer, store)
    backend = ImmediateTeamBackend()
    first = _open_team(writer, ingress, backend)
    original_phase = first._phase  # noqa: SLF001

    def stop_after_first_contribution(run, phase, **kwargs):
        if phase is TeamRunPhase.REVIEWING:
            raise SystemExit("simulated process stop after contribution commit")
        return original_phase(run, phase, **kwargs)

    first._phase = stop_after_first_contribution  # type: ignore[method-assign] # noqa: SLF001
    run = first.start_task(
        tenant_key="tenant_1",
        message_id="om_shared_team",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task="Build, review, and deliver the shared-blob fix",
    )
    first.drain()
    assert first.projection().runs[run.run_id].phase is TeamRunPhase.DISPATCHING
    before_restart_refs = _team_blob_refs(first)
    assert {
        str((ref.labels or {}).get("kind")) for ref in before_restart_refs
    } >= {"team_task", "team_instruction", "team_contribution"}

    ingress.rebuild_projection()
    assert ingress.quarantine_unreferenced_blobs() == 0
    for ref in before_restart_refs:
        assert store.read(ref)
    expected_blob_ids = {ref.blob_id for ref in before_restart_refs}

    first.close()
    ingress.close()
    writer.close()

    recovered_writer = _open_writer(tmp_path, anchor, epoch=2)
    recovered_store = _open_store(tmp_path)
    # Production constructs Ingress first.  Its startup hygiene cannot yet see
    # Team references, so it temporarily quarantines all Team-owned blobs.
    recovered_ingress = _open_ingress(recovered_writer, recovered_store)
    assert expected_blob_ids.isdisjoint(recovered_store.iter_blob_ids())

    # Team construction must replay and restore before runtime recovery invokes
    # Ingress rebuild/hygiene again.
    recovered = _open_team(recovered_writer, recovered_ingress, backend)
    assert expected_blob_ids <= set(recovered_store.iter_blob_ids())
    recovered_ingress.rebuild_projection()
    assert recovered_ingress.quarantine_unreferenced_blobs() == 0
    for ref in _team_blob_refs(recovered):
        assert recovered_store.read(ref)

    assert recovered.recover() == 1
    recovered.drain()
    assert recovered.projection().runs[run.run_id].phase is TeamRunPhase.COMPLETED
    for ref in _team_blob_refs(recovered):
        assert recovered_store.read(ref)

    recovered.close()
    recovered_ingress.close()
    recovered_writer.close()


def test_post_replace_anchor_error_keeps_team_blob_until_verified_replay(
    tmp_path: Path,
) -> None:
    durable_anchor = FileAnchor(tmp_path / "anchor.json")

    class RaiseAfterAnchor:
        production_safe = True

        @staticmethod
        def read():
            return durable_anchor.read()

        @staticmethod
        def compare_and_swap(*args):
            assert durable_anchor.compare_and_swap(*args) is True
            raise OSError("anchor directory fsync outcome unknown")

    writer = _open_writer(tmp_path, RaiseAfterAnchor(), epoch=1)
    store = _open_store(tmp_path)
    ingress = _open_ingress(writer, store)
    actor = _open_team(writer, ingress, ImmediateTeamBackend())
    try:
        with pytest.raises(TeamCoordinatorError, match="was not anchored"):
            actor.admit_task(
                tenant_key="tenant_1",
                message_id="om_ambiguous_team",
                chat_id="oc_team",
                requester_principal_id="ou_user",
                task="preserve the ambiguously anchored Team task",
            )

        assert durable_anchor.read().sequence == 1
        assert len(store.iter_blob_ids()) == 1
        assert ingress.quarantine_unreferenced_blobs() == 0
        projection = actor.projection()
        ref = next(iter(projection.runs.values())).task_ref
        assert store.read(ref)
    finally:
        actor.close()
        ingress.close()
        writer.close()

    recovered_writer = _open_writer(tmp_path, durable_anchor, epoch=2)
    recovered_store = _open_store(tmp_path)
    recovered_ingress = _open_ingress(recovered_writer, recovered_store)
    assert recovered_store.iter_blob_ids() == ()
    recovered = _open_team(
        recovered_writer,
        recovered_ingress,
        ImmediateTeamBackend(),
    )
    try:
        projection = recovered.projection()
        ref = next(iter(projection.runs.values())).task_ref
        assert recovered_store.read(ref)
        assert recovered_ingress.quarantine_unreferenced_blobs() == 0
    finally:
        recovered.close()
        recovered_ingress.close()
        recovered_writer.close()


def test_rejected_team_anchor_releases_only_after_verified_rebuild(
    tmp_path: Path,
) -> None:
    durable_anchor = FileAnchor(tmp_path / "anchor.json")

    class RejectingAnchor:
        production_safe = True

        @staticmethod
        def read():
            return durable_anchor.read()

        @staticmethod
        def compare_and_swap(*_args):
            return False

    writer = _open_writer(tmp_path, RejectingAnchor(), epoch=1)
    store = _open_store(tmp_path)
    ingress = _open_ingress(writer, store)
    actor = _open_team(writer, ingress, ImmediateTeamBackend())
    try:
        with pytest.raises(TeamCoordinatorError, match="was not anchored"):
            actor.admit_task(
                tenant_key="tenant_1",
                message_id="om_rejected_team",
                chat_id="oc_team",
                requester_principal_id="ou_user",
                task="reject this Team anchor",
            )

        assert durable_anchor.read().sequence == 0
        assert ingress.quarantine_unreferenced_blobs() == 0
        assert len(store.iter_blob_ids()) == 1

        assert not actor.projection().runs
        assert ingress.quarantine_unreferenced_blobs() == 1
        assert store.iter_blob_ids() == ()
    finally:
        actor.close()
        ingress.close()
        writer.close()


def test_stale_team_projection_cannot_release_concurrently_published_blob(
    tmp_path: Path,
) -> None:
    anchor = FileAnchor(tmp_path / "anchor.json")
    writer = _open_writer(tmp_path, anchor, epoch=1)
    store = _open_store(tmp_path)
    ingress = _open_ingress(writer, store)
    actor = _open_team(writer, ingress, ImmediateTeamBackend())
    actor._schedule = lambda _run_id: None  # type: ignore[method-assign] # noqa: SLF001
    original_replay = writer.replay
    stale_projection_started = threading.Event()
    release_stale_projection = threading.Event()

    def paused_replay(*args, **kwargs):
        if not stale_projection_started.is_set():
            stale_projection_started.set()
            assert release_stale_projection.wait(timeout=2)
        yield from original_replay(*args, **kwargs)

    writer.replay = paused_replay  # type: ignore[method-assign]
    projection_thread = threading.Thread(target=actor.projection)
    projection_thread.start()
    assert stale_projection_started.wait(timeout=1)

    result: dict[str, object] = {}

    def publish_task() -> None:
        result["run"] = actor.admit_task(
            tenant_key="tenant_1",
            message_id="om_concurrent_team",
            chat_id="oc_team",
            requester_principal_id="ou_user",
            task="retain a concurrently published Team task",
        )[0]

    publish_thread = threading.Thread(target=publish_task)
    publish_thread.start()
    try:
        publish_thread.join(timeout=0.1)
        assert publish_thread.is_alive()
    finally:
        release_stale_projection.set()
        projection_thread.join(timeout=1)
        publish_thread.join(timeout=1)
    assert not projection_thread.is_alive()
    assert not publish_thread.is_alive()

    run = result["run"]
    assert ingress.quarantine_unreferenced_blobs() == 0
    assert store.read(run.task_ref)

    actor.close()
    ingress.close()
    writer.close()


def test_ingress_publish_failure_does_not_quarantine_concurrent_retained_team_blob(
    tmp_path: Path,
) -> None:
    writer = _open_writer(tmp_path, FileAnchor(tmp_path / "anchor.json"), epoch=1)
    store = _open_store(tmp_path)
    ingress = _open_ingress(writer, store)
    payload = EmployeeIngressPayload(
        schema_version=1,
        envelope_id="ing_" + "1" * 64,
        normalized_parts=({"type": "text", "text": "failing ingress"},),
        attachment_descriptors=(),
    )
    metadata = EmployeeIngressMetadata(
        schema_version=1,
        envelope_id=payload.envelope_id,
        tenant_key="tenant_1",
        agent_id="agt_alpha",
        bot_principal_id="bot_alpha",
        app_id="cli_alpha",
        channel_generation=1,
        connection_id="conn_1",
        event_id="evt_ingress_failure",
        message_id="om_ingress_failure",
        event_type="ghostap.team.assignment.v1",
        action_identity="",
        chat_id="oc_team",
        thread_root_message_id="",
        sender_principal_id="ou_user",
        received_at="2026-08-12T00:00:00Z",
        semantic_digest=payload.payload_sha256,
        payload_sha256=payload.payload_sha256,
        payload_size_bytes=payload.canonical_size_bytes,
        attachment_count=0,
        attachment_total_bytes=0,
    )
    ingress_publish_started = threading.Event()
    allow_ingress_failure = threading.Event()
    team_ready_to_publish = threading.Event()
    allow_team_publish = threading.Event()
    team_blob_retained = threading.Event()
    team_event_anchored = threading.Event()
    allow_team_projection = threading.Event()
    retained_ref: list[BlobRef] = []
    ingress_errors: list[BaseException] = []
    team_errors: list[BaseException] = []
    original_stage = store.stage_and_publish
    original_retain = ingress.retain_shared_blob

    def staged_publish(value, labels, key_ref):
        if labels.get("schema") == "employee-ingress-v1":
            ingress_publish_started.set()
            assert allow_ingress_failure.wait(timeout=2)
            raise BlobPublishError("injected ingress publication failure")
        ref = original_stage(value, labels, key_ref)
        if labels.get("kind") == "team_task":
            retained_ref.append(ref)
        return ref

    def observed_retain(blob_id: str) -> None:
        original_retain(blob_id)
        team_blob_retained.set()

    actor: TeamCoordinatorActor | None = None
    ingress_thread: threading.Thread | None = None
    team_thread: threading.Thread | None = None
    try:
        with (
            patch.object(store, "stage_and_publish", side_effect=staged_publish),
            patch.object(ingress, "retain_shared_blob", side_effect=observed_retain),
        ):
            actor = _open_team(writer, ingress, ImmediateTeamBackend())
            actor._schedule = lambda _run_id: None  # type: ignore[method-assign] # noqa: SLF001
            original_projection = actor.projection
            projection_calls = 0

            def pause_after_team_anchor():
                nonlocal projection_calls
                projection_calls += 1
                if projection_calls == 1:
                    projection = original_projection()
                    team_ready_to_publish.set()
                    assert allow_team_publish.wait(timeout=2)
                    return projection
                if projection_calls == 2:
                    team_event_anchored.set()
                    assert allow_team_projection.wait(timeout=2)
                return original_projection()

            actor.projection = pause_after_team_anchor  # type: ignore[method-assign]

            def accept_ingress() -> None:
                try:
                    ingress.accept(metadata, payload, request_id="req_ingress_failure")
                except BaseException as exc:
                    ingress_errors.append(exc)

            def publish_team_task() -> None:
                try:
                    actor.admit_task(
                        tenant_key="tenant_1",
                        message_id="om_team_retained",
                        chat_id="oc_team",
                        requester_principal_id="ou_user",
                        task="retain this Team task through concurrent Ingress cleanup",
                    )
                except BaseException as exc:
                    team_errors.append(exc)

            ingress_thread = threading.Thread(target=accept_ingress)
            team_thread = threading.Thread(target=publish_team_task)
            team_thread.start()
            assert team_ready_to_publish.wait(timeout=2)
            ingress_thread.start()
            assert ingress_publish_started.wait(timeout=2)
            allow_team_publish.set()
            assert team_blob_retained.wait(timeout=2)
            allow_ingress_failure.set()
            ingress_thread.join(timeout=2)
            assert not ingress_thread.is_alive()
            assert len(ingress_errors) == 1
            assert isinstance(ingress_errors[0], IngressBlobError)
            assert team_event_anchored.wait(timeout=2)
            assert len(retained_ref) == 1
            assert any(
                event.event_type == "team.v2.run.created"
                for frame in writer.replay()
                for event in frame.events
            )
            assert retained_ref[0].blob_id in set(store.iter_blob_ids())
            assert store.read(retained_ref[0])
    finally:
        allow_ingress_failure.set()
        allow_team_publish.set()
        allow_team_projection.set()
        if ingress_thread is not None:
            ingress_thread.join(timeout=2)
        if team_thread is not None:
            team_thread.join(timeout=2)
        if actor is not None:
            actor.close()
        ingress.close()
        writer.close()

    assert team_errors == []


@pytest.mark.parametrize(
    ("blob_kind", "precommit_event", "anchoring_event"),
    (
        (
            "team_instruction",
            "team.v2.assignment.created",
            "team.v2.assignment.created",
        ),
        (
            "team_contribution",
            "team.v2.assignment.completed",
            "team.v2.assignment.completed",
        ),
    ),
)
def test_background_drive_keeps_published_blob_retained_until_reference_anchor(
    tmp_path: Path,
    blob_kind: str,
    precommit_event: str,
    anchoring_event: str,
) -> None:
    """A stale projection cannot release a background publish before its commit."""

    writer = _open_writer(tmp_path, FileAnchor(tmp_path / "anchor.json"), epoch=1)
    store = _open_store(tmp_path)
    ingress = _open_ingress(writer, store)
    actor = _open_team(writer, ingress, ImmediateTeamBackend())
    target_refs: list[BlobRef] = []
    target_released = threading.Event()
    before_reference_commit = threading.Event()
    allow_reference_commit = threading.Event()
    reference_anchored = threading.Event()
    allow_drive_to_continue = threading.Event()
    projection_attempted = threading.Event()
    projection_finished = threading.Event()
    thread_errors: list[BaseException] = []
    original_publish_text = actor._publish_text  # noqa: SLF001
    original_record = actor._record  # noqa: SLF001
    original_release = ingress.release_shared_blob

    def observed_publish_text(value: str, **labels: str) -> BlobRef:
        ref = original_publish_text(value, **labels)
        if labels.get("kind") == blob_kind and not target_refs:
            target_refs.append(ref)
        return ref

    precommit_paused = False
    anchor_paused = False

    def paused_record(event_type: str, aggregate_id: str, **payload: object) -> None:
        nonlocal precommit_paused, anchor_paused
        is_target_publish = bool(target_refs)
        if (
            is_target_publish
            and event_type == precommit_event
            and not precommit_paused
        ):
            precommit_paused = True
            before_reference_commit.set()
            assert allow_reference_commit.wait(timeout=5)
        original_record(event_type, aggregate_id, **payload)
        if (
            is_target_publish
            and event_type == anchoring_event
            and not anchor_paused
        ):
            anchor_paused = True
            reference_anchored.set()
            assert allow_drive_to_continue.wait(timeout=5)

    def observed_release(blob_id: str) -> None:
        original_release(blob_id)
        if target_refs and blob_id == target_refs[0].blob_id:
            target_released.set()

    def project_concurrently() -> None:
        projection_attempted.set()
        try:
            actor.projection()
        except BaseException as exc:
            thread_errors.append(exc)
        finally:
            projection_finished.set()

    projection_thread: threading.Thread | None = None
    try:
        actor._publish_text = observed_publish_text  # type: ignore[method-assign] # noqa: SLF001
        actor._record = paused_record  # type: ignore[method-assign] # noqa: SLF001
        actor._blob_releaser = observed_release  # noqa: SLF001
        actor.start_task(
            tenant_key="tenant_1",
            message_id=f"om_background_{blob_kind}",
            chat_id="oc_team",
            requester_principal_id="ou_user",
            task=f"exercise the real background drive for {blob_kind}",
        )
        assert before_reference_commit.wait(timeout=5)
        assert len(target_refs) == 1

        # The fixed path owns the actor lock across retain -> reference commit.
        # On the vulnerable path the lock is free, so wait for the stale
        # projection's observed release before invoking hygiene.
        actor_lock_was_free = actor._lock.acquire(blocking=False)  # noqa: SLF001
        if actor_lock_was_free:
            actor._lock.release()  # noqa: SLF001
        projection_thread = threading.Thread(target=project_concurrently)
        projection_thread.start()
        assert projection_attempted.wait(timeout=1)
        if actor_lock_was_free:
            assert target_released.wait(timeout=5)
            assert projection_finished.wait(timeout=5)
        else:
            assert not target_released.is_set()

        # A pre-commit reservation remains live while its owner still holds the
        # serialization boundary; hygiene must never quarantine it.
        assert ingress.quarantine_unreferenced_blobs() == 0
        allow_reference_commit.set()
        assert reference_anchored.wait(timeout=5)
        assert store.read(target_refs[0])
    finally:
        allow_reference_commit.set()
        allow_drive_to_continue.set()
        if projection_thread is not None:
            projection_thread.join(timeout=5)
        actor.drain()
        actor.close()
        ingress.close()
        writer.close()

    assert thread_errors == []
    assert projection_thread is not None and not projection_thread.is_alive()
    assert not target_released.is_set()
