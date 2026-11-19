"""Ordered, bounded CardEvent-to-AG-UI lifecycle for one Feishu COT segment."""

from __future__ import annotations

import hashlib
import logging
import math
import queue
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from typing import Literal

from src.card.events import CardEvent

from .api_client import (
    FeishuCOTAPIClient,
    _COTResponseRejected,
    _LateOutcome,
)
from .api_client import (
    _SyncLarkClient as _SyncLarkClient,
)
from .session_events import (
    _PROCESS_EVENT_TYPES,
    _SEGMENT_END,
    _COTSessionEventsMixin,
)
from .wire import (
    COTBinding,
    COTTransportError,
    _hashed_identifier,
    _partition_update_events,
    _safe_transport_failure,
    _sanitize_unicode,
    _valid_identifier,
)

logger = logging.getLogger(__name__)

_MAX_BATCH_ITEMS = 32
_DEFAULT_FLUSH_INTERVAL = 0.6
_DEFAULT_QUEUE_CAPACITY = 256
_DEFAULT_SESSION_REQUEST_TIMEOUT_SECONDS = 5.0
_DEFAULT_SESSION_CLOSE_TIMEOUT_SECONDS = 4.5
_MAX_SESSION_CLOSE_TIMEOUT_SECONDS = 6.0
_LATE_CLEANUP_WORKERS = 4
_LATE_CLEANUP_QUEUE_CAPACITY = 64
_STOP = object()


class _LateCleanupDispatcher:
    """Small shared daemon pool for non-authoritative late cleanup work."""

    _queue: queue.Queue[Callable[[], None]] = queue.Queue(
        maxsize=_LATE_CLEANUP_QUEUE_CAPACITY
    )
    _start_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
    _started = False
    _admission_slots = threading.BoundedSemaphore(_LATE_CLEANUP_QUEUE_CAPACITY)

    @classmethod
    def admit(cls) -> bool:
        return cls._admission_slots.acquire(blocking=False)

    @classmethod
    def release(cls) -> None:
        cls._admission_slots.release()

    @classmethod
    def submit(cls, task: Callable[[], None]) -> bool:
        cls._ensure_started()
        try:
            cls._queue.put_nowait(task)
        except queue.Full:
            logger.critical("COT late-cleanup admission invariant was violated")
            return False
        return True

    @classmethod
    def _ensure_started(cls) -> None:
        if cls._started:
            return
        with cls._start_lock:
            if cls._started:
                return
            for index in range(_LATE_CLEANUP_WORKERS):
                threading.Thread(
                    target=cls._worker,
                    name=f"feishu-cot-late-cleanup-{index}",
                    daemon=True,
                ).start()
            cls._started = True

    @classmethod
    def _worker(cls) -> None:
        while True:
            task = cls._queue.get()
            try:
                task()
            except Exception:
                logger.warning("COT late-cleanup task failed", exc_info=True)
            finally:
                cls._queue.task_done()


class FeishuCOTSession(_COTSessionEventsMixin):
    """Ordered, bounded CardEvent-to-AG-UI lifecycle for one task."""

    def __init__(
        self,
        api: FeishuCOTAPIClient,
        *,
        chat_id: str,
        origin_message_id: str | None,
        reply_in_thread: bool = False,
        input_text: str = "",
        detail: Literal["brief", "detailed"] = "detailed",
        flush_interval: float = _DEFAULT_FLUSH_INTERVAL,
        queue_capacity: int = _DEFAULT_QUEUE_CAPACITY,
        request_timeout: float | None = None,
        close_timeout: float | None = None,
    ) -> None:
        if not isinstance(api, FeishuCOTAPIClient):
            raise TypeError("api must be FeishuCOTAPIClient")
        if not _valid_identifier(chat_id):
            raise ValueError("invalid COT chat_id")
        if origin_message_id is not None and not _valid_identifier(
            origin_message_id
        ):
            raise ValueError("invalid COT origin_message_id")
        if not isinstance(reply_in_thread, bool):
            raise ValueError("reply_in_thread must be bool")
        if not isinstance(input_text, str):
            raise TypeError("input_text must be str")
        if detail not in {"brief", "detailed"}:
            raise ValueError("detail must be 'brief' or 'detailed'")
        if (
            isinstance(flush_interval, bool)
            or not isinstance(flush_interval, (int, float))
            or not math.isfinite(float(flush_interval))
            or flush_interval <= 0
        ):
            raise ValueError("flush_interval must be a finite positive number")
        if (
            isinstance(queue_capacity, bool)
            or not isinstance(queue_capacity, int)
            or queue_capacity <= 0
        ):
            raise ValueError("queue_capacity must be a positive integer")
        if request_timeout is None:
            requested_timeout = _DEFAULT_SESSION_REQUEST_TIMEOUT_SECONDS
        elif (
            isinstance(request_timeout, bool)
            or not isinstance(request_timeout, (int, float))
            or not math.isfinite(float(request_timeout))
            or request_timeout <= 0
        ):
            raise ValueError("request_timeout must be a finite positive number")
        else:
            requested_timeout = float(request_timeout)
        resolved_request_timeout = min(api.request_timeout, requested_timeout)
        if close_timeout is None:
            resolved_close_timeout = _DEFAULT_SESSION_CLOSE_TIMEOUT_SECONDS
        elif (
            isinstance(close_timeout, bool)
            or not isinstance(close_timeout, (int, float))
            or not math.isfinite(float(close_timeout))
            or close_timeout <= 0
            or close_timeout > _MAX_SESSION_CLOSE_TIMEOUT_SECONDS
        ):
            raise ValueError(
                "close_timeout must be positive and no greater than 6.0 seconds"
            )
        else:
            resolved_close_timeout = float(close_timeout)

        self._api = api
        self._chat_id = chat_id
        self._origin_message_id = origin_message_id
        self._reply_in_thread = reply_in_thread
        self._input_text = _sanitize_unicode(input_text)
        self._detail = detail
        self._flush_interval = float(flush_interval)
        self._request_timeout = resolved_request_timeout
        self._close_timeout = resolved_close_timeout
        self._close_phase_timeout = resolved_close_timeout / 3
        self._queue: queue.Queue[object] = queue.Queue(maxsize=queue_capacity)
        self._lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        self._worker_done = threading.Event()
        self._worker: threading.Thread | None = None
        self._binding: COTBinding | None = None
        self._starting = False
        self._started = False
        self._close_requested: tuple[object | None, bool] | None = None
        self._accepting = False
        self._closing = False
        self._closed = False
        self._terminal_result: bool | None = None
        self._complete_attempted = False
        self._failure: COTTransportError | None = None
        self._append_failed = False
        self._force_worker_stop = False
        self._ambiguous_request_seen = False
        self._late_cleanup_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._cleanup_admitted = False
        self._cleanup_runner_active = False
        self._cleanup_actions: deque[Callable[[], None]] = deque(maxlen=8)
        self._close_deadline: float | None = None
        self._deferred_close: tuple[COTBinding, object | None, bool] | None = None
        self._deferred_cleanup_needed = False
        self._terminal_owner_claimed = False
        self._last_timestamp_ms = 0
        nonce = uuid.uuid4().hex
        self._id_seed = hashlib.sha256(
            f"{chat_id}\0{origin_message_id or ''}\0{nonce}".encode("utf-8")
        ).digest()
        self._thread_id = _hashed_identifier("thread", chat_id)
        self._run_id = self._hash_id("run", origin_message_id or nonce)
        self._open_text: set[str] = set()
        self._closed_text: set[str] = set()
        self._open_reasoning: set[str] = set()
        self._closed_reasoning: set[str] = set()
        self._tools: dict[str, dict[str, object]] = {}
        self._completed_tools: set[str] = set()

    @property
    def started(self) -> bool:
        """Whether Create and RUN_STARTED both succeeded."""
        with self._lock:
            return self._started

    @property
    def healthy(self) -> bool:
        """Whether this transport has avoided synchronous/asynchronous failure."""
        with self._lock:
            return self._failure is None

    @property
    def message_id(self) -> str | None:
        """Return the COT message ID after Create has bound it."""
        with self._lock:
            return self._binding.message_id if self._binding is not None else None

    def start(self) -> None:
        """Synchronously Create the COT and append RUN_STARTED once."""
        with self._lock:
            if self._started:
                return
            if self._starting:
                raise COTTransportError("COT session start is already in progress")
            if self._closed or self._failure is not None:
                raise self._failure or COTTransportError("COT session is closed")
            if not _LateCleanupDispatcher.admit():
                raise COTTransportError("COT cleanup capacity is exhausted")
            self._cleanup_admitted = True
            self._starting = True

        binding: COTBinding | None = None
        request_stage = "create"
        close_cleanup_scheduled = False
        try:
            binding = self._api.create(
                self._chat_id,
                origin_message_id=self._origin_message_id,
                reply_in_thread=self._reply_in_thread,
                timeout_seconds=self._request_timeout,
            )
            with self._lock:
                self._binding = binding
                close_requested = self._close_requested
            if close_requested is not None:
                error = COTTransportError("COT start was closed during Create")
                with self._lock:
                    self._record_failure_locked(error, append_failed=False)
                    self._starting = False
                    self._closed = True
                self._enqueue_cleanup_action(
                    lambda: self._cleanup_created_before_start(binding)
                )
                close_cleanup_scheduled = True
                raise error
            request_stage = "run_started"
            self._api.append(
                binding,
                (self._run_started_event(),),
                timeout_seconds=self._request_timeout,
            )
            with self._lock:
                close_requested = self._close_requested
            if close_requested is not None:
                error = COTTransportError(
                    "COT start was closed after RUN_STARTED"
                )
                with self._lock:
                    self._record_failure_locked(error, append_failed=False)
                    self._starting = False
                    self._closed = True
                self._enqueue_cleanup_action(
                    lambda: self._late_close_degraded(binding)
                )
                close_cleanup_scheduled = True
                raise error
        except Exception as exc:
            error = self._as_transport_error(exc, "COT start failed")
            with self._lock:
                self._record_failure_locked(error, append_failed=True)
                self._closed = True
                self._starting = False
            if error.request_may_be_in_flight:
                self._schedule_late_cleanup(
                    error,
                    stage=request_stage,
                    binding=binding,
                )
            if (
                binding is not None
                and not error.request_may_be_in_flight
                and not close_cleanup_scheduled
            ):
                self._best_effort_start_cleanup(binding)
            if not error.request_may_be_in_flight and not close_cleanup_scheduled:
                with self._lock:
                    cleanup_pending = self._ambiguous_request_seen
                if not cleanup_pending:
                    self._release_cleanup_admission()
            raise error

        try:
            worker = threading.Thread(
                target=self._worker_main,
                args=(binding,),
                name=f"feishu-cot-{self._run_id[-10:]}",
                daemon=True,
            )
            with self._lock:
                self._worker = worker
            worker.start()
        except Exception as exc:
            error = self._as_transport_error(exc, "COT worker start failed")
            with self._lock:
                self._record_failure_locked(error, append_failed=True)
                self._closed = True
                self._starting = False
            self._best_effort_started_cleanup(binding)
            with self._lock:
                cleanup_pending = self._ambiguous_request_seen
            if not cleanup_pending:
                self._release_cleanup_admission()
            raise error
        with self._lock:
            close_requested = self._close_requested
            self._started = True
            self._accepting = close_requested is None
            self._starting = False
            if close_requested is not None:
                close_event, close_aborting = close_requested
                self._closing = True
                self._close_deadline = time.monotonic() + self._close_timeout
                self._deferred_close = (binding, close_event, close_aborting)
        if close_requested is not None:
            self._enqueue_cleanup_action(
                lambda: self._finish_close(
                    binding=binding,
                    event=close_event,
                    aborting=close_aborting,
                )
            )

    def emit(self, event: CardEvent) -> bool:
        """Queue one supported process event without raising on the hot path."""
        try:
            event_type = getattr(event, "type", None)
            if event_type not in _PROCESS_EVENT_TYPES:
                return False
            with self._lock:
                if (
                    not self._started
                    or not self._accepting
                    or self._closed
                    or self._failure is not None
                ):
                    return False
                if self._queue.full():
                    self._record_failure_locked(
                        COTTransportError("COT event queue is full"),
                        append_failed=False,
                    )
                    return False
                mapped = self._map_process_event_locked(event)
                if mapped is None:
                    return False
                if not mapped:
                    return True
                try:
                    self._queue.put_nowait(mapped)
                except queue.Full:
                    self._record_failure_locked(
                        COTTransportError("COT event queue is full"),
                        append_failed=False,
                    )
                    return False
                self._commit_queued_event_locked(event)
                return True
        except Exception as exc:
            with self._lock:
                self._record_failure_locked(
                    self._as_transport_error(exc, "COT event mapping failed"),
                    append_failed=False,
                )
            return False

    def complete(self, event: CardEvent) -> bool:
        """Drain process events, append one terminal event, and close COT."""
        try:
            return self._close(event=event, aborting=False)
        except Exception as exc:
            with self._lock:
                self._record_failure_locked(
                    self._as_transport_error(exc, "COT close failed"),
                    append_failed=True,
                )
                self._closed = True
                self._closing = False
                self._terminal_result = False
            return False

    def seal_segment(self) -> bool:
        """Finish one healthy process segment without ending the parent task."""
        try:
            return self._close(event=_SEGMENT_END, aborting=False)
        except Exception as exc:
            with self._lock:
                self._record_failure_locked(
                    self._as_transport_error(exc, "COT segment seal failed"),
                    append_failed=True,
                )
                self._closed = True
                self._closing = False
                self._terminal_result = False
            return False

    def abort(self) -> None:
        """Fence immediately and close the non-authoritative process sidecar."""
        with self._lock:
            if self._starting and not self._started:
                self._close_requested = (None, True)
                self._record_failure_locked(
                    COTTransportError("COT session aborted"),
                    append_failed=False,
                )
                return
            if (
                self._closed
                or self._closing
                or not self._started
                or self._binding is None
            ):
                return
            self._closing = True
            self._accepting = False
            self._record_failure_locked(
                COTTransportError("COT session aborted"),
                append_failed=False,
            )
            binding = self._binding
            self._close_deadline = time.monotonic() + self._close_timeout
            self._deferred_close = (binding, None, True)

        submitted = self._enqueue_cleanup_action(
            lambda: self._finish_close(
                binding=binding,
                event=None,
                aborting=True,
            )
        )
        if not submitted:
            with self._lock:
                self._terminal_result = False
                self._closed = True
                self._closing = False

    def _close(self, *, event: object | None, aborting: bool) -> bool:
        with self._lock:
            if self._starting and not self._started:
                self._close_requested = (event, aborting)
                self._accepting = False
                return False
            if self._closed:
                return bool(self._terminal_result)
            if not self._started or self._binding is None:
                return False
            if self._closing:
                return False
            self._closing = True
            self._accepting = False
            binding = self._binding
            self._close_deadline = time.monotonic() + self._close_timeout
            self._deferred_close = (binding, event, aborting)

        return self._finish_close(
            binding=binding,
            event=event,
            aborting=aborting,
        )

    def _finish_close(
        self,
        *,
        binding: COTBinding,
        event: object | None,
        aborting: bool,
    ) -> bool:
        with self._lock:
            close_deadline = self._close_deadline
            if close_deadline is None:
                close_deadline = time.monotonic() + self._close_timeout
                self._close_deadline = close_deadline
        drain_deadline = min(
            close_deadline,
            time.monotonic() + self._close_phase_timeout,
        )
        drained = self._stop_and_drain_worker(drain_deadline)
        if not drained:
            with self._lock:
                self._terminal_result = False
                self._deferred_cleanup_needed = True
            self._resume_deferred_close()
            return False
        with self._lock:
            if self._terminal_owner_claimed:
                return False
            self._terminal_owner_claimed = True
            self._deferred_close = None
            self._deferred_cleanup_needed = False
            prior_failure = self._failure
            ambiguous_request_seen = self._ambiguous_request_seen

        # A timed-out SDK request continues on its bounded daemon worker.  Its
        # eventual server-side result is unknowable here, so issuing a terminal
        # append or Complete now could overtake it.  Fail closed without another
        # request; the caller can cut over to the card transport.
        if ambiguous_request_seen:
            with self._lock:
                self._terminal_result = False
                self._closed = True
                self._closing = False
            return False

        terminal_event, complete_reason, terminal_is_valid = self._terminal_event(
            event,
            aborting=aborting,
            prior_failure=prior_failure,
        )
        terminal_ok = False
        terminal_timeout = min(
            self._request_timeout,
            self._close_phase_timeout,
            close_deadline - time.monotonic(),
        )
        if terminal_timeout > 0:
            try:
                self._api.append(
                    binding,
                    (terminal_event,),
                    timeout_seconds=terminal_timeout,
                )
                terminal_ok = True
            except Exception as exc:
                terminal_error = self._as_transport_error(
                    exc,
                    "COT terminal append failed",
                )
                with self._lock:
                    self._record_failure_locked(
                        terminal_error,
                        append_failed=True,
                    )
                if terminal_error.request_may_be_in_flight:
                    self._schedule_late_cleanup(
                        terminal_error,
                        stage="terminal",
                        binding=binding,
                        complete_reason=complete_reason,
                    )
        else:
            with self._lock:
                self._record_failure_locked(
                    COTTransportError("COT terminal deadline expired"),
                    append_failed=True,
                )

        complete_ok = False
        with self._lock:
            should_complete = (
                not self._complete_attempted
                and not self._ambiguous_request_seen
            )
            if should_complete:
                self._complete_attempted = True
        if should_complete:
            complete_timeout = min(
                self._request_timeout,
                self._close_phase_timeout,
                close_deadline - time.monotonic(),
            )
            if complete_timeout > 0:
                try:
                    self._api.complete(
                        binding,
                        # Complete describes the authoritative parent outcome,
                        # never the health of this optional process transport.
                        reason=complete_reason,
                        timeout_seconds=complete_timeout,
                    )
                    complete_ok = True
                except Exception as exc:
                    complete_error = self._as_transport_error(
                        exc,
                        "COT complete failed",
                    )
                    with self._lock:
                        self._record_failure_locked(
                            complete_error,
                            append_failed=True,
                        )
                    if complete_error.request_may_be_in_flight:
                        self._schedule_late_cleanup(
                            complete_error,
                            stage="complete",
                            binding=binding,
                        )
            else:
                with self._lock:
                    self._record_failure_locked(
                        COTTransportError("COT complete deadline expired"),
                        append_failed=True,
                    )

        succeeded = (
            drained
            and prior_failure is None
            and terminal_is_valid
            and terminal_ok
            and complete_ok
            and not aborting
        )
        with self._lock:
            if aborting and self._failure is None:
                self._failure = COTTransportError("COT session aborted")
            self._terminal_result = succeeded
            self._closed = True
            self._closing = False
            cleanup_pending = self._ambiguous_request_seen
        if not cleanup_pending:
            self._release_cleanup_admission()
        return succeeded

    def _stop_and_drain_worker(self, deadline: float) -> bool:
        try:
            self._queue.put(_STOP, timeout=max(0.0, deadline - time.monotonic()))
        except queue.Full:
            with self._lock:
                self._force_worker_stop = True
                self._record_failure_locked(
                    COTTransportError("COT worker drain timed out"),
                    append_failed=True,
                )
            return False
        remaining = max(0.0, deadline - time.monotonic())
        if not self._worker_done.wait(remaining):
            with self._lock:
                self._record_failure_locked(
                    COTTransportError("COT worker drain timed out"),
                    append_failed=True,
                )
            return False
        worker = self._worker
        if worker is not None:
            worker.join(timeout=0)
        return True

    def _worker_main(self, binding: COTBinding) -> None:
        stop_after_batch = False
        try:
            while not stop_after_batch:
                item = self._queue.get()
                if item is _STOP:
                    break
                batch_items = [item]
                deadline = time.monotonic() + self._flush_interval
                while len(batch_items) < _MAX_BATCH_ITEMS:
                    timeout = deadline - time.monotonic()
                    if timeout <= 0:
                        break
                    try:
                        next_item = self._queue.get(timeout=timeout)
                    except queue.Empty:
                        break
                    if next_item is _STOP:
                        stop_after_batch = True
                        break
                    batch_items.append(next_item)

                events = tuple(
                    cot_event
                    for queued_item in batch_items
                    if isinstance(queued_item, tuple)
                    for cot_event in queued_item
                )
                with self._lock:
                    append_failed = self._append_failed
                    force_worker_stop = self._force_worker_stop
                if not append_failed and events:
                    request_timeout = self._remaining_worker_request_timeout()
                    if request_timeout is None:
                        with self._lock:
                            self._record_failure_locked(
                                COTTransportError(
                                    "COT worker request deadline expired"
                                ),
                                append_failed=True,
                            )
                        continue
                    for event_batch in _partition_update_events(events):
                        try:
                            self._api.append(
                                binding,
                                event_batch,
                                timeout_seconds=request_timeout,
                            )
                        except Exception as exc:
                            error = self._as_transport_error(
                                exc,
                                "COT asynchronous update failed",
                            )
                            logger.warning(
                                "COT request failed; stage=update failure=%s "
                                "batch_events=%d",
                                _safe_transport_failure(error),
                                len(event_batch),
                            )
                            with self._lock:
                                self._record_failure_locked(
                                    error,
                                    append_failed=True,
                                )
                            if error.request_may_be_in_flight:
                                self._schedule_late_cleanup(
                                    error,
                                    stage="update",
                                    binding=binding,
                                )
                            truncated = self._discard_queued_process_items()
                            if truncated:
                                with self._lock:
                                    self._record_failure_locked(
                                        COTTransportError(
                                            "COT worker discarded queued events "
                                            "after append failure"
                                        ),
                                        append_failed=True,
                                    )
                            stop_after_batch = True
                            break
                if force_worker_stop:
                    truncated = self._discard_queued_process_items()
                    if truncated:
                        with self._lock:
                            self._record_failure_locked(
                                COTTransportError(
                                    "COT worker drain truncated queued events"
                                ),
                                append_failed=True,
                            )
                    stop_after_batch = True
        except Exception as exc:
            with self._lock:
                self._record_failure_locked(
                    self._as_transport_error(exc, "COT worker failed"),
                    append_failed=True,
                )
        finally:
            self._worker_done.set()
            self._resume_deferred_close()

    def _discard_queued_process_items(self) -> bool:
        truncated = False
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return truncated
            if item is not _STOP:
                truncated = True

    def _remaining_worker_request_timeout(self) -> float | None:
        with self._lock:
            close_deadline = self._close_deadline
        if close_deadline is None:
            return self._request_timeout
        remaining = close_deadline - time.monotonic()
        if remaining <= 0:
            return None
        return min(self._request_timeout, remaining)

    def _resume_deferred_close(self) -> None:
        with self._lock:
            deferred = self._deferred_close
            if (
                deferred is None
                or self._terminal_owner_claimed
                or not self._deferred_cleanup_needed
                or not self._worker_done.is_set()
            ):
                return
        binding, event, aborting = deferred
        self._enqueue_cleanup_action(
            lambda: self._finish_deferred_close(
                binding=binding,
                event=event,
                aborting=aborting,
            )
        )

    def _finish_deferred_close(
        self,
        *,
        binding: COTBinding,
        event: object | None,
        aborting: bool,
    ) -> None:
        with self._lock:
            if self._terminal_owner_claimed:
                return
            self._terminal_owner_claimed = True
            self._deferred_close = None
            self._deferred_cleanup_needed = False
            ambiguous_request_seen = self._ambiguous_request_seen
            prior_failure = self._failure
            self._terminal_result = False
            self._closed = True
            self._closing = False
        if ambiguous_request_seen:
            return
        terminal_event, complete_reason, _ = self._terminal_event(
            event,
            aborting=aborting,
            prior_failure=prior_failure,
        )
        self._late_append_and_complete(
            binding,
            (terminal_event,),
            reason=complete_reason,
        )

    def _record_failure_locked(
        self,
        error: COTTransportError,
        *,
        append_failed: bool,
    ) -> None:
        if self._failure is None:
            self._failure = error
        self._ambiguous_request_seen = (
            self._ambiguous_request_seen or error.request_may_be_in_flight
        )
        if append_failed:
            self._append_failed = True
        self._accepting = False

    @staticmethod
    def _as_transport_error(exc: Exception, fallback: str) -> COTTransportError:
        if isinstance(exc, COTTransportError):
            return exc
        return COTTransportError(f"{fallback} ({type(exc).__name__})")

    def _schedule_late_cleanup(
        self,
        error: COTTransportError,
        *,
        stage: str,
        binding: COTBinding | None,
        complete_reason: Literal["done", "error"] = "done",
    ) -> None:
        pending_call = error._pending_call
        if pending_call is None:
            logger.error("COT ambiguous request has no pending-call handle")
            return
        pending_call.add_done_callback(
            lambda outcome: self._enqueue_cleanup_action(
                lambda: self._handle_late_outcome(
                    stage=stage,
                    binding=binding,
                    outcome=outcome,
                    complete_reason=complete_reason,
                )
            )
        )

    def _enqueue_cleanup_action(self, action: Callable[[], None]) -> bool:
        with self._lock:
            if not self._cleanup_admitted:
                return False
            if len(self._cleanup_actions) == self._cleanup_actions.maxlen:
                logger.critical("COT per-session cleanup queue invariant was violated")
                return False
            self._cleanup_actions.append(action)
            if self._cleanup_runner_active:
                return True
            self._cleanup_runner_active = True
        if _LateCleanupDispatcher.submit(self._run_cleanup_actions):
            return True
        with self._lock:
            self._cleanup_runner_active = False
        return False

    def _run_cleanup_actions(self) -> None:
        while True:
            with self._lock:
                if not self._cleanup_actions:
                    self._cleanup_runner_active = False
                    return
                action = self._cleanup_actions.popleft()
            try:
                action()
            except Exception:
                logger.warning("COT session cleanup action failed", exc_info=True)

    def _release_cleanup_admission(self) -> None:
        with self._lock:
            if not self._cleanup_admitted:
                return
            self._cleanup_admitted = False
        _LateCleanupDispatcher.release()

    def _finalize_late_cleanup(self) -> None:
        with self._lock:
            self._terminal_owner_claimed = True
            self._deferred_close = None
            self._deferred_cleanup_needed = False
            self._terminal_result = False
            self._closed = True
            self._closing = False
        self._release_cleanup_admission()

    def _handle_late_outcome(
        self,
        *,
        stage: str,
        binding: COTBinding | None,
        outcome: _LateOutcome,
        complete_reason: Literal["done", "error"],
    ) -> None:
        with self._late_cleanup_lock:
            logger.warning(
                "COT late request resolved; stage=%s succeeded=%s",
                stage,
                outcome.succeeded,
            )
            if stage == "create":
                if not outcome.succeeded:
                    logger.warning("COT Create failed after timeout")
                    self._finalize_late_cleanup()
                    return
                if not isinstance(outcome.value, COTBinding):
                    logger.warning("COT late Create result was invalid")
                    self._finalize_late_cleanup()
                    return
                binding = outcome.value
                with self._lock:
                    if self._binding is None:
                        self._binding = binding
                self._late_start_and_close_degraded(binding)
                return
            if binding is None:
                logger.warning("COT late %s result lacks binding", stage)
                self._finalize_late_cleanup()
                return
            if stage == "run_started":
                # A non-zero server response proves that the original start
                # did not take effect, so repair it with a paired lifecycle.
                # Connection loss and malformed responses remain ambiguous:
                # never blind-retry RUN_STARTED in those cases.
                if not outcome.succeeded and isinstance(
                    outcome.value,
                    _COTResponseRejected,
                ):
                    self._late_start_and_close_degraded(binding)
                else:
                    self._late_close_degraded(binding)
                return
            if stage == "update":
                self._late_close_degraded(binding)
                return
            if stage == "terminal":
                self._late_complete(binding, reason=complete_reason)
                return
            if stage == "complete":
                self._finalize_late_cleanup()
                return
            logger.warning("COT late result stage is invalid: %s", stage)
            self._finalize_late_cleanup()

    def _late_start_and_close_degraded(self, binding: COTBinding) -> None:
        """Repair a late Create with a valid, neutral lifecycle then close it."""
        self._late_close_degraded(binding, include_run_started=True)

    def _late_close_degraded(
        self,
        binding: COTBinding,
        *,
        include_run_started: bool = False,
    ) -> None:
        terminal_events = (
            (self._run_started_event(), self._degraded_terminal_event())
            if include_run_started
            else (self._degraded_terminal_event(),)
        )
        self._late_append_and_complete(binding, terminal_events, reason="done")

    def _late_append_and_complete(
        self,
        binding: COTBinding,
        terminal_events: tuple[dict[str, object], ...],
        *,
        reason: Literal["done", "error"],
    ) -> None:
        try:
            self._api.append(
                binding,
                terminal_events,
                timeout_seconds=self._request_timeout,
            )
        except Exception as exc:
            error = self._as_transport_error(
                exc,
                "COT late terminal append failed",
            )
            with self._lock:
                self._record_failure_locked(error, append_failed=True)
            if error.request_may_be_in_flight:
                self._schedule_late_cleanup(
                    error,
                    stage="terminal",
                    binding=binding,
                    complete_reason=reason,
                )
                return
            self._late_complete(binding, reason=reason)
            return
        self._late_complete(binding, reason=reason)

    def _late_complete(
        self,
        binding: COTBinding,
        *,
        reason: Literal["done", "error"],
    ) -> None:
        with self._lock:
            if self._complete_attempted:
                return
            self._complete_attempted = True
        try:
            self._api.complete(
                binding,
                reason=reason,
                timeout_seconds=self._request_timeout,
            )
        except Exception as exc:
            error = self._as_transport_error(exc, "COT late Complete failed")
            with self._lock:
                self._record_failure_locked(error, append_failed=True)
            if error.request_may_be_in_flight:
                self._schedule_late_cleanup(
                    error,
                    stage="complete",
                    binding=binding,
                    complete_reason=reason,
                )
                return
        self._finalize_late_cleanup()

    def _best_effort_start_cleanup(self, binding: COTBinding) -> None:
        self._late_start_and_close_degraded(binding)

    def _cleanup_created_before_start(self, binding: COTBinding) -> None:
        self._best_effort_start_cleanup(binding)
        with self._lock:
            cleanup_pending = self._ambiguous_request_seen
        if not cleanup_pending:
            self._finalize_late_cleanup()

    def _best_effort_started_cleanup(self, binding: COTBinding) -> None:
        self._late_close_degraded(binding)
