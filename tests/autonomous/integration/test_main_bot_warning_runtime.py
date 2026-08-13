import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock

from src.autonomous.acceptance.main_bot_warning_outbox import (
    MainBotWarningConflictError,
    MainBotWarningDrainResult,
    MainBotWarningOutbox,
    MainBotWarningRetryableDeliveryError,
    MainBotWarningState,
    main_bot_warning_idempotency_key,
)
from src.autonomous.journal.anchor import FileAnchor
from src.autonomous.journal.blob_store import AesGcmEncryptionProvider, BlobStore
from src.autonomous.journal.writer import JournalWriter
from src.autonomous.provisioning.composition import (
    EmployeeDepartmentRuntime,
    MainBotWarningPreparationError,
)


def _runtime() -> EmployeeDepartmentRuntime:
    runtime = object.__new__(EmployeeDepartmentRuntime)
    runtime._main_bot_warning_outbox = MagicMock()
    runtime._main_bot_warning_transport = MagicMock()
    runtime._reporting_wakeup = MagicMock()
    runtime._reporting_thread = None
    runtime._closing = False
    return runtime


def test_queue_main_bot_warning_only_prepares_and_wakes_recovery() -> None:
    runtime = _runtime()
    warning_id = "mbw_warning"
    runtime._main_bot_warning_outbox.enqueue.return_value = SimpleNamespace(
        warning_id=warning_id,
    )
    anchored = runtime.queue_main_bot_warning(
        tenant_key="tenant-a",
        chat_id="oc_group",
        message_id="om_origin",
        text="状态未知，请勿重试",
    )

    stable_key = main_bot_warning_idempotency_key(
        "tenant-a",
        "oc_group",
        "om_origin",
    )
    runtime._main_bot_warning_outbox.enqueue.assert_called_once_with(
        tenant_key="tenant-a",
        chat_id="oc_group",
        message_id="om_origin",
        text="状态未知，请勿重试",
        idempotency_key=stable_key,
    )
    runtime._main_bot_warning_outbox.attempt_delivery.assert_not_called()
    runtime._reporting_wakeup.set.assert_called_once_with()
    assert anchored is True


def test_queue_main_bot_warning_treats_existing_origin_conflict_as_fenced() -> None:
    runtime = _runtime()
    runtime._main_bot_warning_outbox.enqueue.side_effect = MainBotWarningConflictError(
        "first warning already won"
    )

    assert runtime.queue_main_bot_warning(
        tenant_key="tenant-a",
        chat_id="oc_group",
        message_id="om_origin",
        text="状态未知，请勿重试",
    ) is True
    runtime._reporting_wakeup.set.assert_called_once_with()


def test_queue_main_bot_warning_can_prepare_before_transport_binding() -> None:
    runtime = _runtime()
    runtime._main_bot_warning_transport = None
    runtime._main_bot_warning_outbox.enqueue.return_value = SimpleNamespace(
        warning_id="mbw_warning",
    )

    assert (
        runtime.queue_main_bot_warning(
            tenant_key="tenant-a",
            chat_id="oc_group",
            message_id="om_origin",
            text="状态未知，请勿重试",
        )
        is True
    )
    runtime._main_bot_warning_outbox.attempt_delivery.assert_not_called()
    runtime._reporting_wakeup.set.assert_called_once_with()


def test_queue_main_bot_warning_normalizes_prepare_failure() -> None:
    runtime = _runtime()
    runtime._main_bot_warning_outbox.enqueue.side_effect = OSError("anchor unavailable")

    try:
        runtime.queue_main_bot_warning(
            tenant_key="tenant-a",
            chat_id="oc_group",
            message_id="om_origin",
            text="状态未知，请勿重试",
        )
    except MainBotWarningPreparationError as exc:
        assert isinstance(exc.__cause__, OSError)
    else:  # pragma: no cover - contract assertion
        raise AssertionError("prepare failure was reported as anchored")
    runtime._reporting_wakeup.set.assert_not_called()


def test_queue_main_bot_warning_rejects_new_admission_during_close() -> None:
    runtime = _runtime()
    runtime._closing = True

    try:
        runtime.queue_main_bot_warning(
            tenant_key="tenant-a",
            chat_id="oc_group",
            message_id="om_origin",
            text="状态未知，请勿重试",
        )
    except MainBotWarningPreparationError as exc:
        assert "closing" in str(exc)
    else:  # pragma: no cover - contract assertion
        raise AssertionError("closing runtime admitted a new warning")
    runtime._main_bot_warning_outbox.enqueue.assert_not_called()


def test_reporting_lane_drains_main_bot_warning_outbox() -> None:
    runtime = _runtime()
    runtime._main_bot_warning_outbox.recover_pending.return_value = (
        MainBotWarningDrainResult(
            pending_count=1,
            attempted_warning_ids=("mbw_warning",),
            committed_warning_ids=("mbw_warning",),
            failed_warning_ids=(),
            action_required_warning_ids=(),
        )
    )

    assert runtime._drain_main_bot_warning_outbox_once() is True
    runtime._main_bot_warning_outbox.recover_pending.assert_called_once_with(
        runtime._main_bot_warning_transport,
        max_items=16,
        deadline=ANY,
    )


def test_empty_warning_outbox_skips_busy_unrelated_journal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "src.autonomous.provisioning.composition._MAIN_BOT_WARNING_REPORTING_DRAIN_SECONDS",
        0.02,
    )
    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=FileAnchor(tmp_path / "journal.anchor"),
        hmac_key=b"j" * 32,
        writer_epoch=1,
    )
    outbox = MainBotWarningOutbox(
        writer=writer,
        blob_store=BlobStore(
            tmp_path / "warning-blobs",
            AesGcmEncryptionProvider(lambda _key_ref: b"b" * 32),
        ),
        active_key_id="data-key-1",
        main_app_id="cli_main_bot",
    )
    runtime = EmployeeDepartmentRuntime()
    runtime._main_bot_warning_outbox = outbox  # type: ignore[assignment]  # noqa: SLF001
    runtime._main_bot_warning_transport = MagicMock()  # noqa: SLF001
    lock_held = threading.Event()
    release = threading.Event()

    def hold_unrelated_journal_write_lock() -> None:
        with writer._mutex:  # noqa: SLF001 - deterministic contention regression
            lock_held.set()
            release.wait(1.0)

    holder = threading.Thread(target=hold_unrelated_journal_write_lock)
    holder.start()
    try:
        assert lock_held.wait(1.0)
        started = time.monotonic()
        assert runtime._drain_main_bot_warning_outbox_once() is False  # noqa: SLF001
        assert time.monotonic() - started < 0.02
    finally:
        release.set()
        holder.join(1.0)
        outbox.close()
        writer.close()


def test_reporting_lane_surfaces_all_retryable_warning_batch_for_backoff() -> None:
    runtime = _runtime()
    runtime._main_bot_warning_outbox.recover_pending.return_value = (
        MainBotWarningDrainResult(
            pending_count=1,
            attempted_warning_ids=("mbw_warning",),
            committed_warning_ids=(),
            failed_warning_ids=("mbw_warning",),
            action_required_warning_ids=(),
        )
    )

    try:
        runtime._drain_main_bot_warning_outbox_once()
    except MainBotWarningRetryableDeliveryError as exc:
        assert "deferred" in str(exc)
    else:  # pragma: no cover - contract assertion
        raise AssertionError("retryable batch bypassed reporting backoff")


def test_binding_main_bot_warning_transport_rejects_invalid_adapter() -> None:
    runtime = _runtime()

    try:
        runtime.bind_main_bot_warning_transport(object())
    except TypeError as exc:
        assert "send_warning" in str(exc)
    else:  # pragma: no cover - contract assertion
        raise AssertionError("invalid warning transport was accepted")


def test_recover_without_warning_transport_starts_reporting_on_late_bind() -> None:
    runtime = EmployeeDepartmentRuntime()
    outbox = MagicMock()
    drained = threading.Event()
    first_transport = MagicMock()
    second_transport = MagicMock()
    observed_transports: list[object] = []

    def recover_pending(transport: object, **_kwargs: object) -> MainBotWarningDrainResult:
        observed_transports.append(transport)
        drained.set()
        return MainBotWarningDrainResult(
            pending_count=0,
            attempted_warning_ids=(),
            committed_warning_ids=(),
            failed_warning_ids=(),
            action_required_warning_ids=(),
        )

    outbox.recover_pending.side_effect = recover_pending
    outbox.pending_records.return_value = ()
    runtime._main_bot_warning_outbox = outbox  # type: ignore[assignment]  # noqa: SLF001

    try:
        runtime.recover()
        assert runtime._reporting_thread is None  # noqa: SLF001

        runtime.bind_main_bot_warning_transport(first_transport)
        assert drained.wait(1)
        reporting_thread = runtime._reporting_thread  # noqa: SLF001
        assert reporting_thread is not None and reporting_thread.is_alive()
        assert first_transport in observed_transports

        drained.clear()
        runtime.bind_main_bot_warning_transport(second_transport)
        runtime.bind_main_bot_warning_transport(second_transport)
        assert drained.wait(1)
        assert runtime._reporting_thread is reporting_thread  # noqa: SLF001
        assert second_transport in observed_transports
    finally:
        runtime.close()


def test_binding_warning_transport_during_close_is_fenced() -> None:
    runtime = EmployeeDepartmentRuntime()
    existing_transport = MagicMock()
    replacement_transport = MagicMock()
    runtime._main_bot_warning_transport = existing_transport  # noqa: SLF001
    runtime._closing = True  # noqa: SLF001

    try:
        runtime.bind_main_bot_warning_transport(replacement_transport)
    except RuntimeError as exc:
        assert "closing" in str(exc)
    else:  # pragma: no cover - contract assertion
        raise AssertionError("closing runtime accepted a warning transport")

    assert runtime._main_bot_warning_transport is existing_transport  # noqa: SLF001
    assert runtime._reporting_thread is None  # noqa: SLF001


def test_execution_storage_binds_warning_outbox_to_settings_app_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.autonomous.provisioning import composition

    runtime = EmployeeDepartmentRuntime(runtime_enabled=True)
    runtime._writer = MagicMock()  # noqa: SLF001
    runtime._vault = MagicMock()  # noqa: SLF001
    runtime._data_keyring = SimpleNamespace(active_key_id="data-key-1")  # noqa: SLF001
    ingress = SimpleNamespace(
        blob_store=MagicMock(),
        retain_shared_blob=MagicMock(),
        release_shared_blob=MagicMock(),
        close=MagicMock(),
    )
    warning_outbox = MagicMock()
    warning_factory = MagicMock(return_value=warning_outbox)
    monkeypatch.setattr(
        composition,
        "build_employee_data_composition",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        composition.EmployeeIngressService,
        "from_keyring",
        MagicMock(return_value=ingress),
    )
    monkeypatch.setattr(
        composition,
        "GroupContextLedger",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        composition.EmployeeOutboxService,
        "from_keyring",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        composition.MainBotWarningOutbox,
        "from_keyring",
        warning_factory,
    )
    monkeypatch.setattr(
        composition,
        "AttachmentStagingService",
        MagicMock(return_value=MagicMock()),
    )
    settings = SimpleNamespace(
        autonomous_visible_employee_limit=1,
        autonomous_employee_storage_base=str(tmp_path / "employees"),
        autonomous_employee_ingress_blob_dir=str(tmp_path / "ingress"),
        autonomous_employee_outbox_blob_dir=str(tmp_path / "outbox"),
        autonomous_main_bot_warning_blob_dir=str(tmp_path / "warnings"),
        autonomous_employee_attachment_staging_dir=str(tmp_path / "attachments"),
        autonomous_context_fetch_timeout_seconds=1.0,
        admin_user_ids=("ou_admin",),
        app_id="cli_main_bot_authority",
    )

    runtime._compose_execution_storage(settings)  # noqa: SLF001

    warning_factory.assert_called_once_with(
        writer=runtime._writer,  # noqa: SLF001
        keyring=runtime._data_keyring,  # noqa: SLF001
        blob_root=str(tmp_path / "warnings"),
        main_app_id="cli_main_bot_authority",
    )
    assert runtime._main_bot_warning_outbox is warning_outbox  # noqa: SLF001
    assert runtime._execution_blockers == ()  # noqa: SLF001


def test_startup_rebuilds_warning_projection_before_starting_reporting() -> None:
    runtime = EmployeeDepartmentRuntime()
    runtime._main_bot_warning_outbox = MagicMock()  # noqa: SLF001
    runtime._main_bot_warning_transport = MagicMock()  # noqa: SLF001
    runtime._start_reporting_worker = MagicMock()  # type: ignore[method-assign]

    runtime._recover_once()  # noqa: SLF001

    runtime._main_bot_warning_outbox.rebuild_projection.assert_called_once_with()
    runtime._start_reporting_worker.assert_called_once_with()


def test_startup_starts_warning_reporting_before_employee_core_recovery() -> None:
    runtime = EmployeeDepartmentRuntime(runtime_enabled=True)
    calls: list[str] = []
    runtime._main_bot_warning_outbox = MagicMock()  # noqa: SLF001
    runtime._main_bot_warning_outbox.rebuild_projection.side_effect = (  # noqa: SLF001
        lambda: calls.append("warning_projection")
    )
    runtime._main_bot_warning_transport = MagicMock()  # noqa: SLF001
    runtime._start_reporting_worker = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda: calls.append("warning_reporting")
    )
    runtime._service = MagicMock()  # noqa: SLF001
    runtime._service.recover.side_effect = RuntimeError("employee core unavailable")

    try:
        runtime._recover_once()  # noqa: SLF001
    except RuntimeError as exc:
        assert str(exc) == "employee core unavailable"
    else:  # pragma: no cover - contract assertion
        raise AssertionError("employee core recovery failure was hidden")

    assert calls == ["warning_projection", "warning_reporting"]
    assert runtime._core_recovered is False  # noqa: SLF001


def test_close_drains_and_closes_main_bot_warning_outbox() -> None:
    runtime = EmployeeDepartmentRuntime()
    outbox = MagicMock()
    outbox.pending_records.side_effect = (
        (SimpleNamespace(warning_id="mbw_warning"),),
        (),
    )
    outbox.recover_pending.return_value = MainBotWarningDrainResult(
        pending_count=1,
        attempted_warning_ids=("mbw_warning",),
        committed_warning_ids=("mbw_warning",),
        failed_warning_ids=(),
        action_required_warning_ids=(),
    )
    runtime._main_bot_warning_outbox = outbox  # type: ignore[assignment]  # noqa: SLF001
    runtime._main_bot_warning_transport = MagicMock()  # noqa: SLF001

    runtime.close()

    outbox.recover_pending.assert_called_once_with(
        runtime._main_bot_warning_transport,
        max_items=1,
        deadline=ANY,
    )
    outbox.close.assert_called_once_with()


def test_close_employee_and_warning_drains_share_one_absolute_deadline() -> None:
    runtime = EmployeeDepartmentRuntime()
    runtime._shutdown_employee_handoff_finalization_executor = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda *, timeout: True
    )
    runtime._outbox = MagicMock()  # noqa: SLF001
    runtime._outbox_delivery = MagicMock()  # noqa: SLF001
    runtime._main_bot_warning_outbox = MagicMock()  # noqa: SLF001
    runtime._main_bot_warning_transport = MagicMock()  # noqa: SLF001
    observed: list[tuple[str, float]] = []

    def drain_employee(*, deadline: float | None = None) -> None:
        assert deadline is not None
        observed.append(("employee", deadline))
        time.sleep(0.01)

    def drain_warning(*, deadline: float | None = None) -> None:
        assert deadline is not None
        observed.append(("warning", deadline))

    runtime._drain_employee_outbox_until_idle = drain_employee  # type: ignore[method-assign]  # noqa: SLF001
    runtime._drain_main_bot_warning_outbox_until_idle = drain_warning  # type: ignore[method-assign]  # noqa: SLF001

    runtime.close()

    assert [label for label, _deadline in observed] == ["employee", "warning"]
    assert observed[0][1] == observed[1][1]


def test_close_preserves_warning_outbox_when_delivery_is_unresolved() -> None:
    runtime = EmployeeDepartmentRuntime()
    outbox = MagicMock()
    outbox.pending_records.return_value = (
        SimpleNamespace(warning_id="mbw_warning"),
    )
    outbox.recover_pending.return_value = MainBotWarningDrainResult(
        pending_count=1,
        attempted_warning_ids=("mbw_warning",),
        committed_warning_ids=(),
        failed_warning_ids=("mbw_warning",),
        action_required_warning_ids=(),
    )
    runtime._main_bot_warning_outbox = outbox  # type: ignore[assignment]  # noqa: SLF001
    runtime._main_bot_warning_transport = MagicMock()  # noqa: SLF001

    try:
        runtime.close()
    except RuntimeError as exc:
        assert "main_bot_warning_drain" in str(exc)
    else:  # pragma: no cover - contract assertion
        raise AssertionError("unresolved warning delivery was silently discarded")
    outbox.close.assert_not_called()


def test_close_drains_main_bot_warning_after_employee_outbox_failure() -> None:
    from src.autonomous.outbox.delivery import EmployeeOutboxItemDeliveryError

    runtime = EmployeeDepartmentRuntime()
    runtime._outbox = MagicMock()  # noqa: SLF001
    runtime._outbox_delivery = MagicMock()  # noqa: SLF001
    runtime._drain_employee_outbox_until_idle = MagicMock(  # type: ignore[method-assign]
        side_effect=EmployeeOutboxItemDeliveryError("employee poison")
    )
    warning_outbox = MagicMock()
    warning_outbox.pending_records.side_effect = (
        (SimpleNamespace(warning_id="mbw_warning"),),
        (),
    )
    warning_outbox.recover_pending.return_value = MainBotWarningDrainResult(
        pending_count=1,
        attempted_warning_ids=("mbw_warning",),
        committed_warning_ids=("mbw_warning",),
        failed_warning_ids=(),
        action_required_warning_ids=(),
    )
    runtime._main_bot_warning_outbox = warning_outbox  # type: ignore[assignment]  # noqa: SLF001
    runtime._main_bot_warning_transport = MagicMock()  # noqa: SLF001

    try:
        runtime.close()
    except RuntimeError as exc:
        assert "outbox_drain" in str(exc)
    else:  # pragma: no cover - contract assertion
        raise AssertionError("employee Outbox failure was hidden")

    warning_outbox.recover_pending.assert_called_once_with(
        runtime._main_bot_warning_transport,
        max_items=1,
        deadline=ANY,
    )
    # Employee failure keeps the shared Journal alive for a later close retry.
    warning_outbox.close.assert_not_called()


def test_close_completes_initial_warning_fair_round_before_failing() -> None:
    runtime = EmployeeDepartmentRuntime()
    poison = tuple(
        SimpleNamespace(warning_id=f"mbw_poison_{index}")
        for index in range(16)
    )
    healthy = SimpleNamespace(warning_id="mbw_healthy")
    outbox = MagicMock()
    outbox.pending_records.side_effect = (
        poison + (healthy,),
        poison + (healthy,),
        poison,
    )
    outbox.recover_pending.side_effect = (
        MainBotWarningDrainResult(
            pending_count=17,
            attempted_warning_ids=tuple(item.warning_id for item in poison),
            committed_warning_ids=(),
            failed_warning_ids=tuple(item.warning_id for item in poison),
            action_required_warning_ids=(),
        ),
        MainBotWarningDrainResult(
            pending_count=17,
            attempted_warning_ids=(healthy.warning_id,),
            committed_warning_ids=(healthy.warning_id,),
            failed_warning_ids=(),
            action_required_warning_ids=(),
        ),
    )
    runtime._main_bot_warning_outbox = outbox  # type: ignore[assignment]  # noqa: SLF001
    runtime._main_bot_warning_transport = MagicMock()  # noqa: SLF001

    try:
        runtime._drain_main_bot_warning_outbox_until_idle()  # noqa: SLF001
    except MainBotWarningRetryableDeliveryError as exc:
        assert "unresolved" in str(exc)
    else:  # pragma: no cover - contract assertion
        raise AssertionError("poison warnings were silently discarded")

    assert outbox.recover_pending.call_count == 2
    assert outbox.recover_pending.call_args_list[0].kwargs["max_items"] == 16
    assert outbox.recover_pending.call_args_list[1].kwargs["max_items"] == 1


def test_main_bot_warning_deferred_does_not_block_employee_reporting() -> None:
    events: list[str] = []
    runtime = EmployeeDepartmentRuntime()

    def warning() -> bool:
        events.append("main_bot_warning")
        raise MainBotWarningRetryableDeliveryError("warning poison")

    runtime._drain_main_bot_warning_outbox_once = warning  # type: ignore[method-assign]  # noqa: SLF001
    runtime._reconcile_terminal_ingress = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda: events.append("terminal") or 0
    )
    runtime._recover_retirement_delivery_channels = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda: events.append("retirement") or ()
    )
    runtime._drain_employee_outbox_once = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda: events.append("employee_outbox") or True
    )
    runtime._outbox = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        gc_superseded_snapshots=lambda: events.append("outbox_gc") or 0,
    )
    runtime._ingress = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        gc_terminal_payloads=lambda: events.append("ingress_gc") or 0,
    )

    assert runtime._drain_employee_reporting_once() is True  # noqa: SLF001

    assert events == [
        "main_bot_warning",
        "terminal",
        "retirement",
        "employee_outbox",
    ]
    assert runtime._warning_reporting_failures == 1  # noqa: SLF001
    assert runtime._warning_reporting_not_before > time.monotonic()  # noqa: SLF001


def test_reporting_tick_bounds_blocked_warning_and_reuses_inflight_send(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "src.autonomous.provisioning.composition._MAIN_BOT_WARNING_REPORTING_DRAIN_SECONDS",
        0.03,
    )
    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=FileAnchor(tmp_path / "journal.anchor"),
        hmac_key=b"j" * 32,
        writer_epoch=1,
    )
    blob_store = BlobStore(
        tmp_path / "warning-blobs",
        AesGcmEncryptionProvider(lambda _key_ref: b"b" * 32),
    )
    outbox = MainBotWarningOutbox(
        writer=writer,
        blob_store=blob_store,
        active_key_id="data-key-1",
        main_app_id="cli_main_bot",
    )
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
    entered = threading.Event()
    release = threading.Event()
    send_calls = 0

    class BlockingTransport:
        main_app_id = "cli_main_bot"

        def send_warning(self, **_kwargs: object) -> str:
            nonlocal send_calls
            send_calls += 1
            entered.set()
            assert release.wait(2.0)
            return "om_warning_reply"

    runtime = EmployeeDepartmentRuntime()
    events: list[str] = []
    runtime._main_bot_warning_outbox = outbox  # type: ignore[assignment]  # noqa: SLF001
    runtime._main_bot_warning_transport = BlockingTransport()  # type: ignore[assignment]  # noqa: SLF001
    runtime._reconcile_terminal_ingress = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda: events.append("terminal") or 0
    )
    runtime._recover_retirement_delivery_channels = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda: events.append("retirement") or ()
    )
    runtime._drain_employee_outbox_once = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda: events.append("employee_outbox") or True
    )
    runtime._outbox = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        gc_superseded_snapshots=lambda: 0,
    )
    runtime._ingress = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        gc_terminal_payloads=lambda: 0,
    )

    started = time.monotonic()
    try:
        assert runtime._drain_employee_reporting_once() is True  # noqa: SLF001

        assert entered.wait(1.0)
        assert time.monotonic() - started < 0.5
        assert events == ["terminal", "retirement", "employee_outbox"]
        assert send_calls == 1
        assert outbox.pending_records()[0].state is MainBotWarningState.EXECUTING

        release.set()
        runtime._warning_reporting_not_before = 0.0  # noqa: SLF001
        assert runtime._drain_employee_reporting_once() is True  # noqa: SLF001
        assert events == [
            "terminal",
            "retirement",
            "employee_outbox",
            "terminal",
            "retirement",
            "employee_outbox",
        ]
        assert send_calls == 1
        assert outbox.pending_records() == ()
    finally:
        release.set()
        if outbox.pending_records():
            try:
                outbox.recover_pending(
                    runtime._main_bot_warning_transport,  # noqa: SLF001
                    deadline=time.monotonic() + 1.0,
                )
            except Exception:
                pass
        outbox.close()
        writer.close()
