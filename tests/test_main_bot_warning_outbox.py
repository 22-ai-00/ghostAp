from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from src.autonomous.journal.anchor import FileAnchor
from src.autonomous.journal.blob_store import AesGcmEncryptionProvider, BlobStore
from src.autonomous.journal.frame import JournalEvent
from src.autonomous.journal.writer import JournalWriter


def _runtime(
    tmp_path: Path,
    *,
    epoch: int = 1,
    main_app_id: str = "cli_main_bot",
):
    from src.autonomous.acceptance.main_bot_warning_outbox import MainBotWarningOutbox

    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=FileAnchor(tmp_path / "journal.anchor"),
        hmac_key=b"j" * 32,
        writer_epoch=epoch,
    )
    blob_store = BlobStore(
        tmp_path / "warning-blobs",
        AesGcmEncryptionProvider(lambda _key_ref: b"b" * 32),
    )
    try:
        outbox = MainBotWarningOutbox(
            writer=writer,
            blob_store=blob_store,
            active_key_id="data-key-1",
            main_app_id=main_app_id,
        )
    except BaseException:
        blob_store.close()
        writer.close()
        raise
    return outbox, writer, blob_store


def test_enqueue_is_durable_encrypted_and_idempotent(tmp_path: Path) -> None:
    from src.autonomous.acceptance.main_bot_warning_outbox import (
        MainBotWarningConflictError,
        MainBotWarningState,
        main_bot_warning_id,
        main_bot_warning_idempotency_key,
    )

    stable_key = main_bot_warning_idempotency_key(
        "tenant-secret",
        "oc_secret_chat",
        "om_secret_message",
    )

    outbox, writer, blob_store = _runtime(tmp_path)
    try:
        first = outbox.enqueue(
            message_id="om_secret_message",
            tenant_key="tenant-secret",
            chat_id="oc_secret_chat",
            text="sensitive warning text",
            idempotency_key=stable_key,
        )
        duplicate = outbox.prepare(
            message_id="om_secret_message",
            tenant_key="tenant-secret",
            chat_id="oc_secret_chat",
            text="sensitive warning text",
            idempotency_key=stable_key,
        )

        assert duplicate == first
        assert first.warning_id == main_bot_warning_id(
            "tenant-secret",
            "oc_secret_chat",
            "om_secret_message",
        )
        assert first.idempotency_key == stable_key
        assert len(stable_key) <= 50
        assert first.state is MainBotWarningState.PREPARED
        assert first.attempt == 0
        assert len(tuple(writer.replay())) == 1

        durable_bytes = writer.journal_path.read_bytes() + b"".join(
            (blob_store.root / f"{blob_id}.blob").read_bytes() for blob_id in blob_store.iter_blob_ids()
        )
        for secret in (
            b"om_secret_message",
            b"tenant-secret",
            b"oc_secret_chat",
            b"sensitive warning text",
            stable_key.encode(),
        ):
            assert secret not in durable_bytes

        with pytest.raises(MainBotWarningConflictError):
            outbox.enqueue(
                message_id="om_secret_message",
                tenant_key="tenant-secret",
                chat_id="oc_secret_chat",
                text="a contradictory warning",
                idempotency_key=stable_key,
            )
        with pytest.raises(ValueError, match="idempotency"):
            outbox.enqueue(
                message_id="om_secret_message",
                tenant_key="tenant-secret",
                chat_id="oc_secret_chat",
                text="sensitive warning text",
                idempotency_key="caller-selected-key",
            )
    finally:
        outbox.close()
        writer.close()


def test_delivery_rejects_cross_app_transport_before_send(tmp_path: Path) -> None:
    from src.autonomous.acceptance.main_bot_warning_outbox import (
        MainBotWarningConflictError,
        MainBotWarningState,
        main_bot_warning_idempotency_key,
    )

    outbox, writer, _blob_store = _runtime(tmp_path)
    warning = outbox.enqueue(
        message_id="om_origin",
        tenant_key="tenant-a",
        chat_id="oc_chat",
        text="warning",
        idempotency_key=main_bot_warning_idempotency_key(
            "tenant-a",
            "oc_chat",
            "om_origin",
        ),
    )
    frames_before_rejection = tuple(writer.replay())
    journal_before_rejection = writer.journal_path.read_bytes()
    record_before_rejection = outbox.pending_records()
    calls = 0

    class WrongAppTransport:
        main_app_id = "cli_other_bot"

        def send_warning(self, **_kwargs: object) -> str:
            nonlocal calls
            calls += 1
            return "om_reply"

    try:
        with pytest.raises(MainBotWarningConflictError, match="app authority"):
            outbox.attempt_delivery(warning.warning_id, WrongAppTransport())
        assert calls == 0
        assert outbox.pending_records() == record_before_rejection
        assert outbox.pending_records()[0].state is MainBotWarningState.PREPARED
        assert outbox.pending_records()[0].attempt == 0
        assert tuple(writer.replay()) == frames_before_rejection
        assert writer.journal_path.read_bytes() == journal_before_rejection
    finally:
        outbox.close()
        writer.close()


def test_recovery_rejects_warning_prepared_by_another_main_app(
    tmp_path: Path,
) -> None:
    from src.autonomous.acceptance.main_bot_warning_outbox import (
        MainBotWarningCorruptionError,
        MainBotWarningState,
        main_bot_warning_idempotency_key,
    )

    outbox, writer, _blob_store = _runtime(tmp_path)
    outbox.enqueue(
        message_id="om_origin",
        tenant_key="tenant-a",
        chat_id="oc_chat",
        text="warning",
        idempotency_key=main_bot_warning_idempotency_key(
            "tenant-a",
            "oc_chat",
            "om_origin",
        ),
    )
    durable_frames = tuple(writer.replay())
    durable_journal = writer.journal_path.read_bytes()
    outbox.close()
    writer.close()

    with pytest.raises(MainBotWarningCorruptionError, match="main app authority"):
        _runtime(tmp_path, epoch=2, main_app_id="cli_other_bot")

    recovered, writer2, _blob_store2 = _runtime(
        tmp_path,
        epoch=3,
        main_app_id="cli_main_bot",
    )
    try:
        assert tuple(writer2.replay()) == durable_frames
        assert writer2.journal_path.read_bytes() == durable_journal
        assert len(recovered.pending_records()) == 1
        assert recovered.pending_records()[0].state is MainBotWarningState.PREPARED
        assert recovered.pending_records()[0].attempt == 0
    finally:
        recovered.close()
        writer2.close()


def test_stable_uuid_survives_terminal_transition_and_restart(tmp_path: Path) -> None:
    from src.autonomous.acceptance.main_bot_warning_outbox import (
        MainBotWarningState,
        main_bot_warning_idempotency_key,
    )

    stable_key = main_bot_warning_idempotency_key(
        "tenant-a",
        "oc_chat",
        "om_origin",
    )
    outbox, writer, _blob_store = _runtime(tmp_path)
    prepared = outbox.enqueue(
        message_id="om_origin",
        tenant_key="tenant-a",
        chat_id="oc_chat",
        text="warning",
        idempotency_key=stable_key,
    )
    calls: list[str] = []

    class Transport:
        main_app_id = "cli_main_bot"

        def send_warning(self, **kwargs: str) -> str:
            calls.append(kwargs["idempotency_key"])
            return "om_warning_reply"

    committed = outbox.attempt_delivery(prepared.warning_id, Transport())
    duplicate = outbox.enqueue(
        message_id="om_origin",
        tenant_key="tenant-a",
        chat_id="oc_chat",
        text="warning",
        idempotency_key=stable_key,
    )
    assert prepared.idempotency_key == stable_key
    assert committed.state is MainBotWarningState.COMMITTED
    assert committed.idempotency_key == stable_key
    assert duplicate == committed
    assert outbox.attempt_delivery(prepared.warning_id, Transport()) == committed
    assert calls == [stable_key]
    outbox.close()
    writer.close()

    recovered, writer2, _blob_store2 = _runtime(tmp_path, epoch=2)
    try:
        terminal = next(
            record
            for record in recovered.rebuild_projection()
            if record.warning_id == prepared.warning_id
        )
        assert terminal.state is MainBotWarningState.COMMITTED
        assert terminal.idempotency_key == stable_key
        assert recovered.attempt_delivery(prepared.warning_id, Transport()) == terminal
        assert calls == [stable_key]
    finally:
        recovered.close()
        writer2.close()


def test_timeout_recovery_reuses_frozen_payload_and_stable_key(tmp_path: Path) -> None:
    from src.autonomous.acceptance.main_bot_warning_outbox import (
        MainBotWarningState,
        main_bot_warning_idempotency_key,
    )

    stable_key = main_bot_warning_idempotency_key(
        "tenant-a",
        "oc_chat",
        "om_origin",
    )
    outbox, writer, _blob_store = _runtime(tmp_path)
    warning = outbox.enqueue(
        message_id="om_origin",
        tenant_key="tenant-a",
        chat_id="oc_chat",
        text="warning",
        idempotency_key=stable_key,
    )

    class TimeoutTransport:
        main_app_id = "cli_main_bot"

        def send_warning(self, **_kwargs: object) -> str:
            raise TimeoutError("ambiguous remote outcome")

    with pytest.raises(TimeoutError, match="ambiguous"):
        outbox.attempt_delivery(warning.warning_id, TimeoutTransport())
    assert outbox.pending_records()[0].state is MainBotWarningState.EXECUTING
    outbox.close()
    writer.close()

    recovered, writer2, _blob_store2 = _runtime(tmp_path, epoch=2)
    calls: list[dict[str, object]] = []

    class RecoveredTransport:
        main_app_id = "cli_main_bot"

        def send_warning(self, **kwargs: object) -> str:
            calls.append(kwargs)
            return "om_warning_reply"

    try:
        drained = recovered.recover_pending(RecoveredTransport())

        assert drained.attempted_warning_ids == (warning.warning_id,)
        assert drained.committed_warning_ids == (warning.warning_id,)
        assert drained.failed_warning_ids == ()
        assert calls == [
            {
                "message_id": "om_origin",
                "tenant_key": "tenant-a",
                "chat_id": "oc_chat",
                "text": "warning",
                "idempotency_key": stable_key,
            }
        ]
        assert recovered.recover_pending(RecoveredTransport()).attempted_warning_ids == ()
        assert [event.event_type for frame in writer2.replay() for event in frame.events] == [
            "main_bot.warning.prepared",
            "main_bot.warning.executing",
            "main_bot.warning.committed",
        ]
    finally:
        recovered.close()
        writer2.close()


def test_concurrent_delivery_is_single_flight(tmp_path: Path) -> None:
    from src.autonomous.acceptance.main_bot_warning_outbox import (
        main_bot_warning_idempotency_key,
    )

    outbox, writer, _blob_store = _runtime(tmp_path)
    stable_key = main_bot_warning_idempotency_key(
        "tenant-a",
        "oc_chat",
        "om_origin",
    )
    warning = outbox.enqueue(
        message_id="om_origin",
        tenant_key="tenant-a",
        chat_id="oc_chat",
        text="warning",
        idempotency_key=stable_key,
    )
    first_entered = threading.Event()
    second_entered = threading.Event()
    release = threading.Event()
    call_lock = threading.Lock()
    call_count = 0

    class BlockingTransport:
        main_app_id = "cli_main_bot"

        def send_warning(self, **_kwargs: object) -> str:
            nonlocal call_count
            with call_lock:
                call_count += 1
                if call_count == 2:
                    second_entered.set()
            first_entered.set()
            assert release.wait(1)
            return "om_warning_reply"

    transport = BlockingTransport()
    errors: list[BaseException] = []

    def deliver() -> None:
        try:
            outbox.attempt_delivery(warning.warning_id, transport)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    first = threading.Thread(target=deliver)
    second = threading.Thread(target=deliver)
    try:
        first.start()
        assert first_entered.wait(1)
        second.start()
        raced = second_entered.wait(0.1)
    finally:
        release.set()
        first.join(timeout=1)
        second.join(timeout=1)

    try:
        assert not raced
        assert errors == []
        assert call_count == 1
    finally:
        outbox.close()
        writer.close()


def test_delivery_deadline_keeps_inflight_transport_and_owned_resources(
    tmp_path: Path,
) -> None:
    from src.autonomous.acceptance.main_bot_warning_outbox import (
        MainBotWarningRetryableDeliveryError,
        MainBotWarningState,
        main_bot_warning_idempotency_key,
    )

    outbox, writer, blob_store = _runtime(tmp_path)
    warning = outbox.enqueue(
        message_id="om_origin",
        tenant_key="tenant-a",
        chat_id="oc_chat",
        text="warning",
        idempotency_key=main_bot_warning_idempotency_key(
            "tenant-a",
            "oc_chat",
            "om_origin",
        ),
    )
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    class BlockingTransport:
        main_app_id = "cli_main_bot"

        def send_warning(self, **_kwargs: object) -> str:
            nonlocal calls
            calls += 1
            entered.set()
            # The test owns the release boundary. A wall-clock timeout here
            # can expire under a loaded full-suite run and invalidate the
            # in-flight premise before close() observes it.
            release.wait()
            return "om_warning_reply"

    transport = BlockingTransport()
    started = time.monotonic()
    try:
        with pytest.raises(MainBotWarningRetryableDeliveryError, match="deadline"):
            outbox.attempt_delivery(
                warning.warning_id,
                transport,
                deadline=time.monotonic() + 0.03,
            )
        assert entered.wait(1)
        assert time.monotonic() - started < 0.5
        assert calls == 1
        assert not blob_store.closed
        assert outbox.pending_records()[0].state is MainBotWarningState.EXECUTING

        with pytest.raises(MainBotWarningRetryableDeliveryError, match="in flight"):
            outbox.close()
        assert not blob_store.closed

        release.set()
        committed = outbox.attempt_delivery(
            warning.warning_id,
            transport,
            deadline=time.monotonic() + 1.0,
        )
        assert committed.state is MainBotWarningState.COMMITTED
        assert calls == 1
    finally:
        release.set()
        if not blob_store.closed:
            try:
                outbox.attempt_delivery(
                    warning.warning_id,
                    transport,
                    deadline=time.monotonic() + 1.0,
                )
            except Exception:
                pass
            outbox.close()
        writer.close()


def test_recovery_deadline_bounds_delivery_lock_contention(tmp_path: Path) -> None:
    from src.autonomous.acceptance.main_bot_warning_outbox import (
        MainBotWarningRetryableDeliveryError,
    )

    outbox, writer, _blob_store = _runtime(tmp_path)
    held = threading.Event()
    release = threading.Event()

    def hold_delivery_lock() -> None:
        with outbox._delivery_lock:  # noqa: SLF001 - fault-injection boundary
            held.set()
            assert release.wait(2)

    holder = threading.Thread(target=hold_delivery_lock)
    holder.start()
    try:
        assert held.wait(1)
        started = time.monotonic()
        with pytest.raises(MainBotWarningRetryableDeliveryError, match="lock deadline"):
            outbox.recover_pending(
                object(),
                deadline=time.monotonic() + 0.03,
            )
        assert time.monotonic() - started < 0.5
    finally:
        release.set()
        holder.join(timeout=1)
        outbox.close()
        writer.close()


def test_shared_journal_commit_waits_for_warning_projection_transaction(
    tmp_path: Path,
) -> None:
    from src.autonomous.acceptance.main_bot_warning_outbox import (
        MainBotWarningState,
        main_bot_warning_idempotency_key,
    )

    outbox, writer, _blob_store = _runtime(tmp_path)
    original_publish = outbox.blob_store.stage_and_publish
    publish_entered = threading.Event()
    allow_publish = threading.Event()
    other_finished = threading.Event()

    def blocking_publish(*args, **kwargs):
        publish_entered.set()
        assert allow_publish.wait(1)
        return original_publish(*args, **kwargs)

    outbox.blob_store.stage_and_publish = blocking_publish  # type: ignore[method-assign]
    warning: list[object] = []
    failures: list[BaseException] = []

    def prepare_warning() -> None:
        try:
            warning.append(
                outbox.enqueue(
                    message_id="om_origin",
                    tenant_key="tenant-a",
                    chat_id="oc_chat",
                    text="warning",
                    idempotency_key=main_bot_warning_idempotency_key(
                        "tenant-a",
                        "oc_chat",
                        "om_origin",
                    ),
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    def commit_other_domain() -> None:
        try:
            with writer.transaction_guard():
                event = JournalEvent(
                    event_type="test.other_domain",
                    aggregate_id="other-aggregate",
                    payload={},
                )
                writer.commit(
                    (event,),
                    writer.get_aggregate_versions((event.aggregate_id,)),
                )
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)
        finally:
            other_finished.set()

    first = threading.Thread(target=prepare_warning)
    second = threading.Thread(target=commit_other_domain)
    try:
        first.start()
        assert publish_entered.wait(1)
        second.start()
        assert not other_finished.wait(0.1)
        allow_publish.set()
        first.join(timeout=1)
        second.join(timeout=1)

        assert failures == []
        assert len(warning) == 1
        assert warning[0].state is MainBotWarningState.PREPARED
        assert other_finished.is_set()
        assert [event.event_type for frame in writer.replay() for event in frame.events] == [
            "main_bot.warning.prepared",
            "test.other_domain",
        ]
    finally:
        allow_publish.set()
        first.join(timeout=1)
        second.join(timeout=1)
        outbox.close()
        writer.close()


def test_explicit_permanent_failure_becomes_action_required(tmp_path: Path) -> None:
    from src.autonomous.acceptance.main_bot_warning_outbox import (
        MainBotWarningPermanentDeliveryError,
        MainBotWarningState,
        main_bot_warning_idempotency_key,
    )

    outbox, writer, _blob_store = _runtime(tmp_path)
    warning = outbox.enqueue(
        message_id="om_origin",
        tenant_key="tenant-a",
        chat_id="oc_chat",
        text="warning",
        idempotency_key=main_bot_warning_idempotency_key(
            "tenant-a",
            "oc_chat",
            "om_origin",
        ),
    )

    class PermanentFailureTransport:
        main_app_id = "cli_main_bot"

        def send_warning(self, **_kwargs: object) -> str:
            raise MainBotWarningPermanentDeliveryError("invalid_target")

    try:
        result = outbox.attempt_delivery(warning.warning_id, PermanentFailureTransport())

        assert result.state is MainBotWarningState.ACTION_REQUIRED
        assert result.error_code == "invalid_target"
        assert outbox.pending_records() == ()
    finally:
        outbox.close()
        writer.close()


def test_projection_catches_up_from_verified_journal_tail_without_full_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unrelated Journal traffic must not make an empty warning poll replay history."""

    from src.autonomous.journal.frame import JournalEvent

    outbox, writer, _blob_store = _runtime(tmp_path)
    for index in range(64):
        event = JournalEvent(
            event_type="test.other_domain",
            aggregate_id=f"other-{index}",
            payload={},
        )
        writer.commit(
            (event,),
            writer.get_aggregate_versions((event.aggregate_id,)),
        )

    monkeypatch.setattr(
        writer,
        "replay",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("projection catch-up re-read the full Journal")
        ),
    )
    try:
        assert outbox.pending_records(deadline=time.monotonic() + 0.05) == ()
        assert outbox._cursor_sequence == writer.get_last_frame().sequence  # noqa: SLF001
    finally:
        outbox.close()
        writer.close()


@pytest.mark.parametrize(
    ("feishu_code", "safe_error_code"),
    (
        (230001, "feishu_message_not_found"),
        (230020, "feishu_message_recalled"),
    ),
)
def test_feishu_permanent_reply_failure_reaches_action_required(
    tmp_path: Path,
    feishu_code: int,
    safe_error_code: str,
) -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.autonomous.acceptance.main_bot_warning_outbox import (
        MainBotWarningState,
        main_bot_warning_idempotency_key,
    )
    from src.feishu.handlers.base import BaseHandler
    from src.feishu.ws_client import _MainBotWarningReplyTransport

    failed_response = SimpleNamespace(
        code=feishu_code,
        msg="secret remote diagnostic",
        data=None,
        success=lambda: False,
    )
    handler = object.__new__(BaseHandler)
    handler.im_client = MagicMock()
    handler.im_client.reply_message.return_value = failed_response
    transport = _MainBotWarningReplyTransport(
        handler=handler,
        main_app_id="cli_main_bot",
    )
    outbox, writer, blob_store = _runtime(tmp_path)
    warning = outbox.enqueue(
        message_id="om_origin",
        tenant_key="tenant-a",
        chat_id="oc_chat",
        text="warning",
        idempotency_key=main_bot_warning_idempotency_key(
            "tenant-a",
            "oc_chat",
            "om_origin",
        ),
    )

    try:
        result = outbox.attempt_delivery(warning.warning_id, transport)

        assert result.state is MainBotWarningState.ACTION_REQUIRED
        assert result.error_code == safe_error_code
        durable_bytes = writer.journal_path.read_bytes() + b"".join(
            (blob_store.root / f"{blob_id}.blob").read_bytes() for blob_id in blob_store.iter_blob_ids()
        )
        assert b"secret remote diagnostic" not in durable_bytes
    finally:
        outbox.close()
        writer.close()


def test_empty_reply_receipt_remains_executing_for_retry(tmp_path: Path) -> None:
    from unittest.mock import MagicMock

    from src.autonomous.acceptance.main_bot_warning_outbox import (
        MainBotWarningRetryableDeliveryError,
        MainBotWarningState,
        main_bot_warning_idempotency_key,
    )
    from src.feishu.handlers.base import BaseHandler
    from src.feishu.ws_client import _MainBotWarningReplyTransport

    handler = object.__new__(BaseHandler)
    handler.im_client = MagicMock()
    handler.im_client.reply_message.return_value = None
    transport = _MainBotWarningReplyTransport(
        handler=handler,
        main_app_id="cli_main_bot",
    )
    outbox, writer, _blob_store = _runtime(tmp_path)
    warning = outbox.enqueue(
        message_id="om_origin",
        tenant_key="tenant-a",
        chat_id="oc_chat",
        text="warning",
        idempotency_key=main_bot_warning_idempotency_key(
            "tenant-a",
            "oc_chat",
            "om_origin",
        ),
    )

    try:
        with pytest.raises(MainBotWarningRetryableDeliveryError, match="receipt"):
            outbox.attempt_delivery(warning.warning_id, transport)

        assert outbox.pending_records()[0].state is MainBotWarningState.EXECUTING
    finally:
        outbox.close()
        writer.close()


def test_failed_prepare_commit_quarantines_blob_on_next_verified_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.autonomous.acceptance.main_bot_warning_outbox import (
        MainBotWarningWriteDisabledError,
        main_bot_warning_idempotency_key,
    )

    outbox, writer, blob_store = _runtime(tmp_path)
    monkeypatch.setattr(
        writer,
        "commit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk failed")),
    )
    try:
        with pytest.raises(MainBotWarningWriteDisabledError) as raised:
            outbox.enqueue(
                message_id="om_origin",
                tenant_key="tenant-a",
                chat_id="oc_chat",
                text="warning",
                idempotency_key=main_bot_warning_idempotency_key(
                    "tenant-a",
                    "oc_chat",
                    "om_origin",
                ),
            )
        assert isinstance(raised.value.__cause__, OSError)

        # Commit/anchor outcome is not guessed in-process.  Preserving this
        # blob also covers an anchor write that succeeded immediately before
        # its directory fsync raised.
        assert len(blob_store.iter_blob_ids()) == 1
        assert tuple((blob_store.root / "quarantine").glob("*.blob")) == ()
    finally:
        outbox.close()
        writer.close()

    recovered, writer2, recovered_blobs = _runtime(tmp_path, epoch=2)
    try:
        assert recovered.pending_records() == ()
        assert recovered_blobs.iter_blob_ids() == ()
        assert len(tuple((recovered_blobs.root / "quarantine").glob("*.blob"))) == 1
    finally:
        recovered.close()
        writer2.close()


def test_ambiguous_anchor_exception_preserves_referenced_blob_for_reopen(
    tmp_path: Path,
) -> None:
    from src.autonomous.acceptance.main_bot_warning_outbox import (
        MainBotWarningOutbox,
        MainBotWarningWriteDisabledError,
        main_bot_warning_idempotency_key,
    )

    durable_anchor = FileAnchor(tmp_path / "journal.anchor")

    class RaiseAfterAnchor:
        production_safe = True

        @staticmethod
        def read():
            return durable_anchor.read()

        @staticmethod
        def compare_and_swap(*args):
            assert durable_anchor.compare_and_swap(*args) is True
            raise OSError("directory fsync outcome unknown")

    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=RaiseAfterAnchor(),
        hmac_key=b"j" * 32,
        writer_epoch=1,
    )
    blobs = BlobStore(
        tmp_path / "warning-blobs",
        AesGcmEncryptionProvider(lambda _key_ref: b"b" * 32),
    )
    outbox = MainBotWarningOutbox(
        writer=writer,
        blob_store=blobs,
        active_key_id="data-key-1",
        main_app_id="cli_main_bot",
    )
    try:
        with pytest.raises(MainBotWarningWriteDisabledError):
            outbox.enqueue(
                message_id="om_origin",
                tenant_key="tenant-a",
                chat_id="oc_chat",
                text="warning",
                idempotency_key=main_bot_warning_idempotency_key(
                    "tenant-a",
                    "oc_chat",
                    "om_origin",
                ),
            )
        assert len(blobs.iter_blob_ids()) == 1
    finally:
        outbox.close()
        writer.close()

    recovered, writer2, _recovered_blobs = _runtime(tmp_path, epoch=2)
    try:
        pending = recovered.pending_records()
        assert len(pending) == 1
        assert pending[0].message_id == "om_origin"
        assert pending[0].text == "warning"
    finally:
        recovered.close()
        writer2.close()


def test_keyring_factory_builds_composition_owned_encrypted_store(tmp_path: Path) -> None:
    from src.autonomous.acceptance.main_bot_warning_outbox import (
        MainBotWarningOutbox,
        main_bot_warning_idempotency_key,
    )

    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=FileAnchor(tmp_path / "journal.anchor"),
        hmac_key=b"j" * 32,
        writer_epoch=1,
    )

    class Keyring:
        active_key_id = "data-key-1"

        @staticmethod
        def resolve(_key_ref: str) -> bytes:
            return b"b" * 32

    outbox = MainBotWarningOutbox.from_keyring(
        writer=writer,
        keyring=Keyring(),
        blob_root=tmp_path / "warning-blobs",
        main_app_id="cli_main_bot",
    )
    try:
        record = outbox.enqueue(
            message_id="om_origin",
            tenant_key="tenant-a",
            chat_id="oc_chat",
            text="warning",
            idempotency_key=main_bot_warning_idempotency_key(
                "tenant-a",
                "oc_chat",
                "om_origin",
            ),
        )
        assert record.message_id == "om_origin"
    finally:
        outbox.close()
        writer.close()


def test_delivery_is_anchored_before_transport_and_committed_after_receipt(
    tmp_path: Path,
) -> None:
    from src.autonomous.acceptance.main_bot_warning_outbox import (
        MainBotWarningState,
        main_bot_warning_idempotency_key,
    )

    stable_key = main_bot_warning_idempotency_key(
        "tenant-a",
        "oc_chat",
        "om_origin",
    )

    outbox, writer, _blob_store = _runtime(tmp_path)
    warning = outbox.enqueue(
        message_id="om_origin",
        tenant_key="tenant-a",
        chat_id="oc_chat",
        text="warning",
        idempotency_key=stable_key,
    )

    class Transport:
        main_app_id = "cli_main_bot"

        def __init__(self) -> None:
            self.calls: list[dict[str, str]] = []

        def send_warning(self, **kwargs: str) -> str:
            event_types = [event.event_type for frame in writer.replay() for event in frame.events]
            assert event_types == [
                "main_bot.warning.prepared",
                "main_bot.warning.executing",
            ]
            self.calls.append(kwargs)
            return "om_warning_reply"

    transport = Transport()
    try:
        committed = outbox.attempt_delivery(warning.warning_id, transport)

        assert committed.state is MainBotWarningState.COMMITTED
        assert committed.attempt == 1
        assert transport.calls == [
            {
                "message_id": "om_origin",
                "tenant_key": "tenant-a",
                "chat_id": "oc_chat",
                "text": "warning",
                "idempotency_key": stable_key,
            }
        ]
        assert [event.event_type for frame in writer.replay() for event in frame.events] == [
            "main_bot.warning.prepared",
            "main_bot.warning.executing",
            "main_bot.warning.committed",
        ]
    finally:
        outbox.close()
        writer.close()


@pytest.mark.parametrize("retry_before_close", [False, True])
def test_ambiguous_commit_that_is_authoritatively_terminal_releases_delivery_future(
    tmp_path: Path,
    retry_before_close: bool,
) -> None:
    from src.autonomous.acceptance.main_bot_warning_outbox import (
        MainBotWarningOutbox,
        MainBotWarningState,
        MainBotWarningWriteDisabledError,
        main_bot_warning_idempotency_key,
    )

    durable_anchor = FileAnchor(tmp_path / "journal.anchor")

    class RaiseAfterTerminalAnchor:
        production_safe = True

        @staticmethod
        def read():
            return durable_anchor.read()

        @staticmethod
        def compare_and_swap(*args):
            anchored = durable_anchor.compare_and_swap(*args)
            if args[2] == 3:
                assert anchored is True
                raise OSError("terminal anchor directory fsync outcome unknown")
            return anchored

    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=RaiseAfterTerminalAnchor(),
        hmac_key=b"j" * 32,
        writer_epoch=1,
    )
    blobs = BlobStore(
        tmp_path / "warning-blobs",
        AesGcmEncryptionProvider(lambda _key_ref: b"b" * 32),
    )
    outbox = MainBotWarningOutbox(
        writer=writer,
        blob_store=blobs,
        active_key_id="data-key-1",
        main_app_id="cli_main_bot",
    )
    warning = outbox.enqueue(
        message_id="om_origin",
        tenant_key="tenant-a",
        chat_id="oc_chat",
        text="warning",
        idempotency_key=main_bot_warning_idempotency_key(
            "tenant-a",
            "oc_chat",
            "om_origin",
        ),
    )
    calls = 0

    class Transport:
        main_app_id = "cli_main_bot"

        def send_warning(self, **_kwargs: object) -> str:
            nonlocal calls
            calls += 1
            return "om_reply"

    transport = Transport()
    try:
        with pytest.raises(MainBotWarningWriteDisabledError, match="COMMITTED"):
            outbox.attempt_delivery(warning.warning_id, transport)

        assert outbox.pending_records() == ()
        if retry_before_close:
            replayed = outbox.attempt_delivery(warning.warning_id, transport)
            assert replayed.state is MainBotWarningState.COMMITTED
        assert calls == 1

        outbox.close()
    finally:
        if not outbox._closed:  # noqa: SLF001 - fault-injection cleanup
            outbox._delivery_futures.clear()  # noqa: SLF001 - fault-injection cleanup
            outbox.close()
        writer.close()


def test_ambiguous_commit_without_terminal_anchor_keeps_delivery_future_fenced(
    tmp_path: Path,
) -> None:
    from src.autonomous.acceptance.main_bot_warning_outbox import (
        MainBotWarningOutbox,
        MainBotWarningRetryableDeliveryError,
        MainBotWarningState,
        MainBotWarningWriteDisabledError,
        main_bot_warning_idempotency_key,
    )

    durable_anchor = FileAnchor(tmp_path / "journal.anchor")

    class RejectTerminalAnchor:
        production_safe = True

        @staticmethod
        def read():
            return durable_anchor.read()

        @staticmethod
        def compare_and_swap(*args):
            if args[2] == 3:
                return False
            return durable_anchor.compare_and_swap(*args)

    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=RejectTerminalAnchor(),
        hmac_key=b"j" * 32,
        writer_epoch=1,
    )
    blobs = BlobStore(
        tmp_path / "warning-blobs",
        AesGcmEncryptionProvider(lambda _key_ref: b"b" * 32),
    )
    outbox = MainBotWarningOutbox(
        writer=writer,
        blob_store=blobs,
        active_key_id="data-key-1",
        main_app_id="cli_main_bot",
    )
    warning = outbox.enqueue(
        message_id="om_origin",
        tenant_key="tenant-a",
        chat_id="oc_chat",
        text="warning",
        idempotency_key=main_bot_warning_idempotency_key(
            "tenant-a",
            "oc_chat",
            "om_origin",
        ),
    )
    calls = 0

    class Transport:
        main_app_id = "cli_main_bot"

        def send_warning(self, **_kwargs: object) -> str:
            nonlocal calls
            calls += 1
            return "om_reply"

    try:
        with pytest.raises(MainBotWarningWriteDisabledError, match="COMMITTED"):
            outbox.attempt_delivery(warning.warning_id, Transport())

        pending = outbox.pending_records()
        assert len(pending) == 1
        assert pending[0].state is MainBotWarningState.EXECUTING
        assert calls == 1
        with pytest.raises(MainBotWarningRetryableDeliveryError, match="in flight"):
            outbox.close()
    finally:
        outbox._delivery_futures.clear()  # noqa: SLF001 - fault-injection cleanup
        outbox.close()
        writer.close()


def test_recovery_rotates_past_retryable_fifo_head_with_strict_batch_bound(
    tmp_path: Path,
) -> None:
    from src.autonomous.acceptance.main_bot_warning_outbox import (
        MainBotWarningState,
        main_bot_warning_idempotency_key,
    )

    outbox, writer, _blob_store = _runtime(tmp_path)
    warnings = []
    for index in range(17):
        message_id = f"om_{index:02d}"
        warnings.append(
            outbox.enqueue(
                message_id=message_id,
                tenant_key="tenant-a",
                chat_id="oc_chat",
                text=f"warning-{index}",
                idempotency_key=main_bot_warning_idempotency_key(
                    "tenant-a",
                    "oc_chat",
                    message_id,
                ),
            )
        )

    calls: list[str] = []

    class PoisonPrefixTransport:
        main_app_id = "cli_main_bot"

        def send_warning(self, **kwargs: str) -> str:
            message_id = kwargs["message_id"]
            calls.append(message_id)
            if message_id != "om_16":
                raise TimeoutError("retryable poison")
            return "om_healthy_receipt"

    try:
        first = outbox.recover_pending(PoisonPrefixTransport(), max_items=16)
        second = outbox.recover_pending(PoisonPrefixTransport(), max_items=16)

        assert len(first.attempted_warning_ids) == 16
        assert first.failed_warning_ids == first.attempted_warning_ids
        assert first.committed_warning_ids == ()
        assert second.attempted_warning_ids[0] == warnings[16].warning_id
        assert second.committed_warning_ids == (warnings[16].warning_id,)
        assert len(second.attempted_warning_ids) == 16
        assert calls[:17] == [f"om_{index:02d}" for index in range(17)]
        assert (
            next(record for record in outbox.rebuild_projection() if record.warning_id == warnings[16].warning_id).state
            is MainBotWarningState.COMMITTED
        )
    finally:
        outbox.close()
        writer.close()


def test_recovery_cursor_drops_terminal_items_and_appends_new_items_after_survivors(
    tmp_path: Path,
) -> None:
    from src.autonomous.acceptance.main_bot_warning_outbox import (
        MainBotWarningPermanentDeliveryError,
        main_bot_warning_idempotency_key,
    )

    outbox, writer, _blob_store = _runtime(tmp_path)

    def enqueue(message_id: str):
        return outbox.enqueue(
            message_id=message_id,
            tenant_key="tenant-a",
            chat_id="oc_chat",
            text=message_id,
            idempotency_key=main_bot_warning_idempotency_key(
                "tenant-a",
                "oc_chat",
                message_id,
            ),
        )

    oldest = enqueue("om_oldest")
    independently_committed = enqueue("om_committed")
    survivor = enqueue("om_survivor")

    class RetryAll:
        main_app_id = "cli_main_bot"

        def send_warning(self, **_kwargs: str) -> str:
            raise TimeoutError("retry")

    class Commit:
        main_app_id = "cli_main_bot"

        def send_warning(self, **_kwargs: str) -> str:
            return "om_receipt"

    class PermanentOldest:
        main_app_id = "cli_main_bot"

        def send_warning(self, **kwargs: str) -> str:
            if kwargs["message_id"] == "om_oldest":
                raise MainBotWarningPermanentDeliveryError("invalid_target")
            raise TimeoutError("retry")

    try:
        first = outbox.recover_pending(RetryAll(), max_items=1)
        assert first.attempted_warning_ids == (oldest.warning_id,)

        outbox.attempt_delivery(independently_committed.warning_id, Commit())
        newcomer = enqueue("om_newcomer")

        second = outbox.recover_pending(PermanentOldest(), max_items=2)
        third = outbox.recover_pending(PermanentOldest(), max_items=1)

        assert second.attempted_warning_ids == (
            survivor.warning_id,
            oldest.warning_id,
        )
        assert second.action_required_warning_ids == (oldest.warning_id,)
        assert third.attempted_warning_ids == (newcomer.warning_id,)
    finally:
        outbox.close()
        writer.close()
