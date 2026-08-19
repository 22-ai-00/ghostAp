"""Task-scoped Feishu IM COT transport for programming process events.

Feishu's Python SDK does not expose typed ``message_cot`` resources in the
version used by GhostAP.  This module therefore uses the SDK's generic
``Client.request(BaseRequest)`` entry point while retaining the same strict
response, outbound-audit, and managed-scope fences as the existing message
and card transports.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import queue
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from lark_oapi.core.enum import AccessTokenType, HttpMethod
from lark_oapi.core.model.base_request import BaseRequest
from lark_oapi.core.model.base_response import BaseResponse

from src.card.events import CardEvent, CardEventType
from src.card.tool_display import sanitize_full_tool_event_content
from src.utils.text import sanitize_single_line_label

logger = logging.getLogger(__name__)

_COLLECTION_URI = "/open-apis/im/v1/message_cot"
_COMPLETE_URI = f"{_COLLECTION_URI}/complete/:cot_id"
_JSON_HEADERS = {"Content-Type": "application/json; charset=utf-8"}
_MAX_IDENTIFIER_CHARS = 512
_MAX_BATCH_EVENTS = 100
_MAX_BATCH_ITEMS = 32
_MAX_UPDATE_EVENT_BYTES = 14 * 1024
_MAX_JSON_DEPTH = 32
_MAX_JSON_NODES = 10_000
_MAX_STREAM_CHUNK_BYTES = 4_000
_MAX_TOOL_CONTENT_BYTES = 4_000
_DEFAULT_FLUSH_INTERVAL = 0.3
_DEFAULT_QUEUE_CAPACITY = 256
_DEFAULT_API_TIMEOUT_SECONDS = 35.0
_DEFAULT_SESSION_REQUEST_TIMEOUT_SECONDS = 5.0
_DEFAULT_SESSION_CLOSE_TIMEOUT_SECONDS = 4.5
_MAX_SESSION_CLOSE_TIMEOUT_SECONDS = 6.0
_LATE_CLEANUP_WORKERS = 4
_LATE_CLEANUP_QUEUE_CAPACITY = 64
_STOP = object()

_COT_EVENT_TYPES = frozenset(
    {
        "RUN_STARTED",
        "RUN_FINISHED",
        "RUN_ERROR",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "REASONING_START",
        "REASONING_MESSAGE_START",
        "REASONING_MESSAGE_CONTENT",
        "REASONING_MESSAGE_END",
        "REASONING_END",
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
        "TOOL_CALL_RESULT",
    }
)

_PROCESS_EVENT_TYPES = frozenset(
    {
        CardEventType.TEXT_STARTED,
        CardEventType.TEXT_DELTA,
        CardEventType.TEXT_DONE,
        CardEventType.REASONING_STARTED,
        CardEventType.REASONING_DELTA,
        CardEventType.REASONING_DONE,
        CardEventType.TOOL_STARTED,
        CardEventType.TOOL_DELTA,
        CardEventType.TOOL_DONE,
        CardEventType.TOOL_FAILED,
    }
)

_STATUS_TEXT = {
    "running": {"zh_cn": "过程记录中", "en_us": "Recording process"},
    "thinking": {"zh_cn": "正在记录过程", "en_us": "Recording process"},
    "done": {"zh_cn": "过程记录完成", "en_us": "Process recorded"},
    "error": {
        "zh_cn": "过程记录异常结束，请查看主卡",
        "en_us": "Process ended unexpectedly; see the main card",
    },
    "paused": {"zh_cn": "过程展示已切换到主卡", "en_us": "Process moved to the main card"},
    "interrupted": {"zh_cn": "过程记录已中断", "en_us": "Process interrupted"},
}


class _SyncLarkClient(Protocol):
    def request(self, request: BaseRequest) -> BaseResponse: ...


class COTTransportError(RuntimeError):
    """Raised when a COT request or lifecycle fails closed."""

    def __init__(
        self,
        message: str,
        *,
        request_may_be_in_flight: bool = False,
        pending_call: _PendingCall | None = None,
    ) -> None:
        super().__init__(message)
        self.request_may_be_in_flight = request_may_be_in_flight
        self._pending_call = pending_call


class _COTResponseRejected(COTTransportError):
    """The server returned an explicit non-success response.

    Unlike a transport exception or malformed response, this outcome proves
    that RUN_STARTED was rejected and can therefore be retried once during
    late cleanup without risking a duplicate remote start.
    """


@dataclass(frozen=True, slots=True)
class _LateOutcome:
    succeeded: bool
    value: object


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


class _PendingCall:
    """One timed request whose strictly validated outcome remains observable."""

    def __init__(self) -> None:
        self._done = threading.Event()
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._outcome: _LateOutcome | None = None
        self._callbacks: list[Callable[[_LateOutcome], None]] = []

    def resolve(self, outcome: _LateOutcome) -> None:
        with self._lock:
            if self._outcome is not None:
                return
            self._outcome = outcome
            callbacks = tuple(self._callbacks)
            self._callbacks.clear()
            self._done.set()
        for callback in callbacks:
            callback(outcome)

    def wait(self, timeout: float) -> _LateOutcome | None:
        if not self._done.wait(timeout):
            return None
        with self._lock:
            return self._outcome

    def add_done_callback(
        self,
        callback: Callable[[_LateOutcome], None],
    ) -> None:
        with self._lock:
            outcome = self._outcome
            if outcome is None:
                self._callbacks.append(callback)
                return
        callback(outcome)


class _AmbiguousRequestTimeout(TimeoutError):
    """The caller timed out while the daemon SDK request may still finish."""

    def __init__(self, message: str, pending_call: _PendingCall) -> None:
        super().__init__(message)
        self.pending_call = pending_call


class _SDKInvocationError(RuntimeError):
    """No response was received after entering the SDK request call."""

    def __init__(self, exception_name: str) -> None:
        super().__init__(exception_name)
        self.exception_name = exception_name


TrustRevision = tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class COTBinding:
    """Immutable remote identity and provenance for one COT task."""

    chat_id: str
    cot_id: str
    message_id: str
    origin_message_id: str | None
    reply_in_thread: bool
    audit_targets: tuple[str, ...]
    trust_revision: TrustRevision

    def __post_init__(self) -> None:
        for label, value in (
            ("chat_id", self.chat_id),
            ("cot_id", self.cot_id),
            ("message_id", self.message_id),
        ):
            if not _valid_identifier(value):
                raise ValueError(f"invalid COT {label}")
        if self.origin_message_id is not None and not _valid_identifier(
            self.origin_message_id
        ):
            raise ValueError("invalid COT origin_message_id")
        if not isinstance(self.reply_in_thread, bool):
            raise ValueError("invalid COT reply_in_thread")
        if (
            not isinstance(self.audit_targets, tuple)
            or not self.audit_targets
            or any(not _valid_identifier(item) for item in self.audit_targets)
        ):
            raise ValueError("invalid COT audit provenance")
        _validate_trust_revision(self.trust_revision)


class FeishuCOTAPIClient:
    """Strict generic-SDK client for Feishu's untyped IM COT endpoints."""

    _worker_slots = threading.BoundedSemaphore(64)

    def __init__(
        self,
        client: _SyncLarkClient,
        *,
        outbound_audit: Callable[[str, str, str], None] | None = None,
        outbound_audit_failure: Callable[[Exception], None] | None = None,
        tenant_key_resolver: Callable[[], str] | None = None,
        outbound_target_aliases: Callable[[str], tuple[str, ...]] | None = None,
        trust_revision_provider: Callable[[str], TrustRevision] | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        if not callable(getattr(client, "request", None)):
            raise TypeError("COT client must provide request(BaseRequest)")
        self._client = client
        self._outbound_audit = outbound_audit
        self._outbound_audit_failure = outbound_audit_failure
        self._tenant_key_resolver = tenant_key_resolver
        self._outbound_target_aliases = outbound_target_aliases
        self._trust_revision_provider = trust_revision_provider
        if timeout_seconds is not None and (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ValueError("COT API timeout must be a finite positive number")
        self._timeout_seconds = (
            float(timeout_seconds) if timeout_seconds is not None else None
        )

    @property
    def request_timeout(self) -> float:
        """Return the hard deadline applied to each individual SDK request."""
        if self._timeout_seconds is not None:
            return self._timeout_seconds
        try:
            from src.config import get_settings

            timeout = float(get_settings().card.delivery_api_timeout)
            if math.isfinite(timeout) and timeout > 0:
                return timeout
        except Exception:
            pass
        return _DEFAULT_API_TIMEOUT_SECONDS

    def create(
        self,
        chat_id: str,
        *,
        origin_message_id: str | None = None,
        reply_in_thread: bool = False,
        timeout_seconds: float | None = None,
    ) -> COTBinding:
        """Create and bind one task-scoped COT message."""
        if not _valid_identifier(chat_id):
            raise COTTransportError("COT chat provenance is invalid")
        if origin_message_id is not None and not _valid_identifier(
            origin_message_id
        ):
            raise COTTransportError("COT origin provenance is invalid")
        if not isinstance(reply_in_thread, bool):
            raise COTTransportError("COT reply_in_thread is invalid")

        revision = self._snapshot_revision(chat_id)
        if origin_message_id is not None:
            audit_targets = self._audit_create_reply(origin_message_id, chat_id)
        else:
            audit_targets = self._audit_targets("create", (chat_id,))
        if self._snapshot_revision(chat_id) != revision:
            raise COTTransportError("COT trust revision changed")

        body: dict[str, object] = {"receive_id": chat_id}
        if origin_message_id is not None:
            body["origin_message_id"] = origin_message_id
        if reply_in_thread:
            body["reply_in_thread"] = True
        def _binding_from_payload(payload: dict[str, Any]) -> COTBinding:
            data = payload["data"]
            if not {"cot_id", "message_id"}.issubset(data):
                raise COTTransportError("COT create response schema is invalid")
            cot_id = data.get("cot_id")
            message_id = data.get("message_id")
            if not _valid_identifier(cot_id) or not _valid_identifier(message_id):
                raise COTTransportError("COT create response schema is invalid")
            try:
                return COTBinding(
                    chat_id=chat_id,
                    cot_id=cot_id,
                    message_id=message_id,
                    origin_message_id=origin_message_id,
                    reply_in_thread=reply_in_thread,
                    audit_targets=audit_targets,
                    trust_revision=revision,
                )
            except ValueError as exc:
                raise COTTransportError(
                    "COT create response schema is invalid"
                ) from exc

        binding = self._request(
            HttpMethod.POST,
            _COLLECTION_URI,
            operation="create",
            queries=[("receive_id_type", "chat_id")],
            body=body,
            timeout_seconds=timeout_seconds,
            transform=_binding_from_payload,
        )
        if not isinstance(binding, COTBinding):
            raise COTTransportError("COT create response schema is invalid")
        return binding

    def append(
        self,
        binding: COTBinding,
        events: Sequence[Mapping[str, object]],
        *,
        timeout_seconds: float | None = None,
    ) -> None:
        """Append one non-empty, ordered AG-UI event batch."""
        self._validate_binding(binding)
        encoded = _encode_events(events)
        self._assert_trust_revision(binding)
        self._audit_bound_mutation(binding)
        self._assert_trust_revision(binding)

        def _empty_update(payload: dict[str, Any]) -> None:
            if payload["data"]:
                raise COTTransportError("COT update response schema is invalid")

        self._request(
            HttpMethod.PUT,
            _COLLECTION_URI,
            operation="update",
            body={
                "message_id": binding.message_id,
                "cot_id": binding.cot_id,
                "events": encoded,
            },
            timeout_seconds=timeout_seconds,
            transform=_empty_update,
        )

    def complete(
        self,
        binding: COTBinding,
        *,
        reason: Literal["done", "error"],
        timeout_seconds: float | None = None,
    ) -> None:
        """Close one bound COT message exactly once at the caller layer."""
        self._validate_binding(binding)
        if reason not in {"done", "error"}:
            raise COTTransportError("COT complete reason is invalid")
        self._assert_trust_revision(binding)
        self._audit_bound_mutation(binding)
        self._assert_trust_revision(binding)

        def _empty_complete(payload: dict[str, Any]) -> None:
            if payload["data"]:
                raise COTTransportError("COT complete response schema is invalid")

        self._request(
            HttpMethod.POST,
            _COMPLETE_URI,
            operation="complete",
            paths={"cot_id": binding.cot_id},
            queries=[("message_id", binding.message_id), ("reason", reason)],
            body=None,
            timeout_seconds=timeout_seconds,
            transform=_empty_complete,
        )

    @staticmethod
    def _validate_binding(binding: COTBinding) -> None:
        if not isinstance(binding, COTBinding):
            raise COTTransportError("COT binding is invalid")

    def _snapshot_revision(self, chat_id: str) -> TrustRevision:
        provider = self._trust_revision_provider
        if provider is None:
            return None
        try:
            revision = provider(chat_id)
            _validate_trust_revision(revision)
        except Exception as exc:
            raise COTTransportError(
                f"COT trust revision unavailable ({type(exc).__name__})"
            ) from None
        return revision

    def _assert_trust_revision(self, binding: COTBinding) -> None:
        if self._snapshot_revision(binding.chat_id) != binding.trust_revision:
            raise COTTransportError("COT trust revision changed")

    def _audit_create_reply(
        self,
        origin_message_id: str,
        chat_id: str,
    ) -> tuple[str, ...]:
        if self._outbound_audit is None:
            return (origin_message_id, chat_id)
        resolver = self._outbound_target_aliases
        aliases: tuple[str, ...] = ()
        if resolver is not None:
            try:
                resolved = resolver(origin_message_id)
            except Exception:
                resolved = ()
            if (
                isinstance(resolved, tuple)
                and resolved
                and all(_valid_identifier(alias) for alias in resolved)
            ):
                aliases = resolved
        if not aliases:
            raise COTTransportError("COT reply recipient scope is unavailable")
        if chat_id not in aliases:
            raise COTTransportError("COT reply chat scope is unavailable")
        return self._audit_targets(
            "reply",
            tuple(dict.fromkeys((origin_message_id, *aliases))),
        )

    def _audit_bound_mutation(self, binding: COTBinding) -> None:
        if not binding.audit_targets:
            raise COTTransportError("COT mutation provenance is unavailable")
        self._audit_targets(
            "patch",
            tuple(
                dict.fromkeys(
                    (binding.message_id, binding.cot_id, *binding.audit_targets)
                )
            ),
        )

    def _audit_targets(
        self,
        operation: Literal["create", "reply", "patch"],
        targets: tuple[str, ...],
    ) -> tuple[str, ...]:
        if not targets or any(not _valid_identifier(target) for target in targets):
            raise COTTransportError("COT outbound provenance is unavailable")
        audit = self._outbound_audit
        if audit is None:
            return tuple(dict.fromkeys(targets))
        tenant_key = ""
        if self._tenant_key_resolver is not None:
            try:
                resolved = self._tenant_key_resolver()
                tenant_key = resolved if isinstance(resolved, str) else ""
            except Exception:
                tenant_key = ""
        unique_targets = tuple(dict.fromkeys(targets))
        for target in unique_targets:
            try:
                audit(tenant_key, operation, target)
            except Exception as exc:
                logger.error(
                    "main Bot COT audit failed closed: %s",
                    type(exc).__name__,
                )
                callback = self._outbound_audit_failure
                if callback is not None:
                    try:
                        callback(exc)
                    except Exception:
                        logger.error(
                            "main Bot COT audit failure callback failed",
                            exc_info=True,
                        )
                raise COTTransportError("COT outbound audit failed") from None
        return unique_targets

    def _request(
        self,
        method: HttpMethod,
        uri: str,
        *,
        operation: str,
        paths: dict[str, str] | None = None,
        queries: list[tuple[str, str]] | None = None,
        body: object,
        timeout_seconds: float | None = None,
        transform: Callable[[dict[str, Any]], object] | None = None,
    ) -> object:
        request = (
            BaseRequest.builder()
            .http_method(method)
            .uri(uri)
            .token_types({AccessTokenType.TENANT})
            .paths(paths or {})
            .queries(queries or [])
            .headers(dict(_JSON_HEADERS))
            .body(body)
            .build()
        )

        def _execute() -> object:
            try:
                response = self._client.request(request)
            except Exception as exc:
                raise _SDKInvocationError(type(exc).__name__) from None
            payload = self._decode_response(response, operation)
            return transform(payload) if transform is not None else payload

        try:
            return self._call_api(
                operation,
                _execute,
                timeout_seconds=timeout_seconds,
            )
        except _AmbiguousRequestTimeout as exc:
            raise COTTransportError(
                f"COT {operation} request failed (TimeoutError)",
                request_may_be_in_flight=True,
                pending_call=exc.pending_call,
            ) from None
        except Exception as exc:
            if isinstance(exc, COTTransportError):
                raise
            raise COTTransportError(
                f"COT {operation} request failed ({type(exc).__name__})"
            ) from None

    @staticmethod
    def _decode_response(response: object, operation: str) -> dict[str, Any]:
        if not isinstance(response, BaseResponse):
            raise COTTransportError(f"COT {operation} failed (code=invalid)")
        if isinstance(response.code, bool) or not isinstance(response.code, int):
            raise COTTransportError(
                f"COT {operation} response schema is invalid"
            )
        raw = response.raw
        if (
            raw is None
            or isinstance(raw.status_code, bool)
            or not isinstance(raw.status_code, int)
            or not isinstance(raw.content, bytes)
        ):
            raise COTTransportError(
                f"COT {operation} response schema is invalid"
            )
        try:
            payload = json.loads(raw.content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise COTTransportError(
                f"COT {operation} response schema is invalid"
            ) from None
        code = payload.get("code") if isinstance(payload, dict) else None
        if isinstance(code, bool) or not isinstance(code, int):
            raise COTTransportError(
                f"COT {operation} response schema is invalid"
            )
        if (
            not response.success()
            or response.code != 0
            or not 200 <= raw.status_code < 300
            or code != 0
        ):
            raise _COTResponseRejected(
                f"COT {operation} failed (code={code})"
            )
        if not isinstance(payload.get("data"), dict):
            raise COTTransportError(
                f"COT {operation} response schema is invalid"
            )
        return payload

    def _call_api(
        self,
        operation: str,
        call: Callable[[], object],
        *,
        timeout_seconds: float | None,
    ) -> object:
        configured_timeout = self.request_timeout
        timeout = configured_timeout if timeout_seconds is None else timeout_seconds
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise TimeoutError(f"COT {operation} request deadline expired")
        timeout = min(configured_timeout, float(timeout))
        pending_call = _PendingCall()
        slots = type(self)._worker_slots
        if not slots.acquire(blocking=False):
            raise TimeoutError(f"COT {operation} request worker slots exhausted")

        def _target() -> None:
            try:
                outcome = _LateOutcome(True, call())
            except Exception as exc:
                outcome = _LateOutcome(False, exc)
            finally:
                slots.release()
            pending_call.resolve(outcome)

        worker = threading.Thread(
            target=_target,
            name=f"feishu-cot-api-{operation}",
            daemon=True,
        )
        try:
            worker.start()
        except Exception:
            slots.release()
            raise
        outcome = pending_call.wait(float(timeout))
        if outcome is None:
            raise _AmbiguousRequestTimeout(
                f"COT {operation} request timed out after {float(timeout):.1f}s",
                pending_call,
            )
        if outcome.succeeded:
            return outcome.value
        if isinstance(outcome.value, _SDKInvocationError):
            raise COTTransportError(
                f"COT {operation} request failed "
                f"({outcome.value.exception_name})",
                request_may_be_in_flight=True,
                pending_call=pending_call,
            )
        if isinstance(outcome.value, Exception):
            raise outcome.value
        raise RuntimeError(f"COT {operation} request failed")


class FeishuCOTSession:
    """Ordered, bounded CardEvent-to-AG-UI lifecycle for one task."""

    def __init__(
        self,
        api: FeishuCOTAPIClient,
        *,
        chat_id: str,
        origin_message_id: str | None,
        reply_in_thread: bool = False,
        input_text: str = "",
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
        self._close_requested: tuple[CardEvent | None, bool] | None = None
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
        self._deferred_close: tuple[COTBinding, CardEvent | None, bool] | None = None
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

    def _close(self, *, event: CardEvent | None, aborting: bool) -> bool:
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
        event: CardEvent | None,
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
        event: CardEvent | None,
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

    def _run_started_event(self) -> dict[str, object]:
        return self._cot_event(
            "RUN_STARTED",
            {
                "threadId": self._thread_id,
                "runId": self._run_id,
                "input": {
                    "query": self._input_text,
                    "statusText": {
                        key: dict(value) for key, value in _STATUS_TEXT.items()
                    },
                },
            },
        )

    def _terminal_event(
        self,
        event: CardEvent | None,
        *,
        aborting: bool,
        prior_failure: COTTransportError | None,
    ) -> tuple[dict[str, object], Literal["done", "error"], bool]:
        if aborting:
            return (
                self._degraded_terminal_event(),
                "done",
                True,
            )
        if isinstance(event, CardEvent):
            # The parent card is the task-status authority.  If its real
            # terminal is already known, never replace it with a sidecar
            # transport failure that happened earlier in the same close race.
            if event.type is CardEventType.COMPLETED:
                return (
                    self._cot_event(
                        "RUN_FINISHED",
                        {
                            "threadId": self._thread_id,
                            "runId": self._run_id,
                            "status": "done",
                        },
                    ),
                    "done",
                    True,
                )
            if event.type is CardEventType.CANCELLED:
                reason = event.payload.get("reason", "")
                text_reason = reason if isinstance(reason, str) else ""
                if _is_timeout_reason(text_reason):
                    return (
                        self._run_error_event("Task timed out", "TIMEOUT"),
                        "error",
                        True,
                    )
                return (
                    self._cot_event(
                        "RUN_FINISHED",
                        {
                            "threadId": self._thread_id,
                            "runId": self._run_id,
                            "status": "interrupted",
                        },
                    ),
                    "error",
                    True,
                )
            if event.type is CardEventType.FAILED:
                raw_error = event.payload.get("error", "")
                message = (
                    _safe_error_message(raw_error)
                    if isinstance(raw_error, str) and raw_error
                    else "Task failed"
                )
                return (
                    self._run_error_event(message[:1_000], "TASK_FAILED"),
                    "error",
                    True,
                )

        if prior_failure is not None:
            logger.warning(
                "COT process transport degraded before parent terminal; "
                "failure_type=%s",
                type(prior_failure).__name__,
            )
            return self._degraded_terminal_event(), "done", True

        if not isinstance(event, CardEvent):
            with self._lock:
                self._record_failure_locked(
                    COTTransportError("COT terminal event is invalid"),
                    append_failed=False,
                )
            return (
                self._degraded_terminal_event(),
                "done",
                False,
            )
        with self._lock:
            self._record_failure_locked(
                COTTransportError("COT terminal event is invalid"),
                append_failed=False,
            )
        return (
            self._degraded_terminal_event(),
            "done",
            False,
        )

    def _degraded_terminal_event(self) -> dict[str, object]:
        """Close only the process display without asserting a task terminal."""
        return self._cot_event(
            "RUN_FINISHED",
            {
                "threadId": self._thread_id,
                "runId": self._run_id,
                "status": "paused",
            },
        )

    def _map_process_event_locked(
        self,
        event: CardEvent,
    ) -> tuple[dict[str, object], ...] | None:
        payload = event.payload
        block_id = payload.get("block_id")
        if not isinstance(block_id, str) or not block_id:
            return None

        if event.type is CardEventType.TEXT_STARTED:
            if block_id in self._open_text or block_id in self._closed_text:
                return ()
            self._open_text.add(block_id)
            return (
                self._cot_event(
                    "TEXT_MESSAGE_START",
                    {"messageId": self._hash_id("text", block_id), "role": "assistant"},
                ),
            )
        if event.type is CardEventType.TEXT_DELTA:
            if block_id not in self._open_text:
                return None
            text = payload.get("text", "")
            if not isinstance(text, str):
                return None
            text = _sanitize_unicode(text)
            if not text:
                return ()
            return tuple(
                self._cot_event(
                    "TEXT_MESSAGE_CONTENT",
                    {
                        "messageId": self._hash_id("text", block_id),
                        "delta": chunk,
                    },
                )
                for chunk in _split_utf8_chunks(text, _MAX_STREAM_CHUNK_BYTES)
            )
        if event.type is CardEventType.TEXT_DONE:
            if block_id not in self._open_text:
                return None
            self._open_text.remove(block_id)
            self._closed_text.add(block_id)
            return (
                self._cot_event(
                    "TEXT_MESSAGE_END",
                    {"messageId": self._hash_id("text", block_id)},
                ),
            )
        if event.type is CardEventType.REASONING_STARTED:
            if block_id in self._open_reasoning or block_id in self._closed_reasoning:
                return ()
            self._open_reasoning.add(block_id)
            message_id = self._hash_id("reasoning", block_id)
            return (
                self._cot_event("REASONING_START", {"messageId": message_id}),
                self._cot_event(
                    "REASONING_MESSAGE_START",
                    {"messageId": message_id, "role": "reasoning"},
                ),
            )
        if event.type is CardEventType.REASONING_DELTA:
            if block_id not in self._open_reasoning:
                return None
            text = payload.get("text", "")
            if not isinstance(text, str):
                return None
            text = _sanitize_unicode(text)
            if not text:
                return ()
            return tuple(
                self._cot_event(
                    "REASONING_MESSAGE_CONTENT",
                    {
                        "messageId": self._hash_id("reasoning", block_id),
                        "delta": chunk,
                    },
                )
                for chunk in _split_utf8_chunks(text, _MAX_STREAM_CHUNK_BYTES)
            )
        if event.type is CardEventType.REASONING_DONE:
            if block_id not in self._open_reasoning:
                return None
            self._open_reasoning.remove(block_id)
            self._closed_reasoning.add(block_id)
            message_id = self._hash_id("reasoning", block_id)
            return (
                self._cot_event(
                    "REASONING_MESSAGE_END",
                    {"messageId": message_id},
                ),
                self._cot_event("REASONING_END", {"messageId": message_id}),
            )
        if event.type is CardEventType.TOOL_STARTED:
            if block_id in self._tools or block_id in self._completed_tools:
                return ()
            tool_call_id = self._hash_id("tool", block_id)
            raw_name = payload.get("tool_name", "")
            name_text = raw_name if isinstance(raw_name, str) else ""
            safe_name = sanitize_single_line_label(
                sanitize_full_tool_event_content(
                    name_text,
                    opaque_ids=(block_id,),
                ),
                fallback="tool",
                max_chars=120,
            )
            raw_input = payload.get("tool_input", "")
            safe_input = _safe_tool_content(raw_input, opaque_id=block_id)
            self._tools[block_id] = {
                "tool_call_id": tool_call_id,
                "latest": "",
                "result_pending": False,
            }
            return (
                self._cot_event(
                    "TOOL_CALL_START",
                    {"toolCallId": tool_call_id, "toolCallName": safe_name},
                ),
                self._cot_event(
                    "TOOL_CALL_ARGS",
                    {"toolCallId": tool_call_id, "delta": safe_input or "{}"},
                ),
                self._cot_event("TOOL_CALL_END", {"toolCallId": tool_call_id}),
            )
        if event.type is CardEventType.TOOL_DELTA:
            state = self._tools.get(block_id)
            if state is None or block_id in self._completed_tools:
                return None
            state["latest"] = _safe_tool_content(
                payload.get("content", ""),
                opaque_id=block_id,
            )
            return ()
        if event.type in {CardEventType.TOOL_DONE, CardEventType.TOOL_FAILED}:
            state = self._tools.get(block_id)
            if block_id in self._completed_tools:
                return ()
            if state is None:
                return None
            if event.type is CardEventType.TOOL_FAILED:
                candidate = payload.get("error", "")
                fallback = "Tool failed"
            else:
                candidate = payload.get("tool_output", "")
                fallback = "Tool completed"
            content = _safe_tool_content(candidate, opaque_id=block_id)
            if not content:
                latest = state.get("latest", "")
                content = latest if isinstance(latest, str) else ""
            if not content:
                content = fallback
            state["result_pending"] = True
            tool_call_id = state.get("tool_call_id")
            if not isinstance(tool_call_id, str):
                return None
            return (
                self._cot_event(
                    "TOOL_CALL_RESULT",
                    {
                        "messageId": self._hash_id("tool-result", block_id),
                        "toolCallId": tool_call_id,
                        "content": content,
                        "role": "tool",
                    },
                ),
            )
        return None

    def _commit_queued_event_locked(self, event: CardEvent) -> None:
        if event.type not in {CardEventType.TOOL_DONE, CardEventType.TOOL_FAILED}:
            return
        block_id = event.payload.get("block_id")
        if not isinstance(block_id, str):
            return
        state = self._tools.get(block_id)
        if state is None or state.get("result_pending") is not True:
            return
        self._completed_tools.add(block_id)
        self._tools.pop(block_id, None)

    def _cot_event(
        self,
        event_type: str,
        content: Mapping[str, object],
    ) -> dict[str, object]:
        return {
            "event_type": event_type,
            "content": dict(content),
            "timestamp": self._next_timestamp_ms(),
        }

    def _run_error_event(self, message: str, code: str) -> dict[str, object]:
        return self._cot_event(
            "RUN_ERROR",
            {
                "threadId": self._thread_id,
                "runId": self._run_id,
                "message": message,
                "code": code,
            },
        )

    def _next_timestamp_ms(self) -> int:
        with self._lock:
            now_ms = int(time.time() * 1_000)
            self._last_timestamp_ms = max(now_ms, self._last_timestamp_ms + 1)
            return self._last_timestamp_ms

    def _hash_id(self, kind: str, opaque_id: str) -> str:
        digest = hashlib.sha256(
            self._id_seed
            + b"\0"
            + kind.encode("utf-8")
            + b"\0"
            + _sanitize_unicode(opaque_id).encode("utf-8")
        ).hexdigest()
        return f"{kind.replace('-', '_')}_{digest[:32]}"

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


def _valid_identifier(value: object) -> bool:
    if not isinstance(value, str) or not value or len(value) > _MAX_IDENTIFIER_CHARS:
        return False
    try:
        value.encode("ascii")
    except UnicodeEncodeError:
        return False
    return all(character.isalnum() or character in {"_", "-"} for character in value)


def _validate_trust_revision(revision: TrustRevision) -> None:
    if revision is None:
        return
    if (
        not isinstance(revision, tuple)
        or len(revision) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in revision
        )
    ):
        raise ValueError("invalid COT trust revision")


def _encode_events(
    events: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if (
        not isinstance(events, Sequence)
        or isinstance(events, (str, bytes, bytearray))
        or not events
        or len(events) > _MAX_BATCH_EVENTS
    ):
        raise COTTransportError("COT event batch is invalid")
    encoded: list[dict[str, object]] = []
    for event in events:
        if not isinstance(event, Mapping) or set(event) != {
            "event_type",
            "content",
            "timestamp",
        }:
            raise COTTransportError("COT event schema is invalid")
        event_type = event.get("event_type")
        content = event.get("content")
        timestamp = event.get("timestamp")
        if event_type not in _COT_EVENT_TYPES or not isinstance(content, Mapping):
            raise COTTransportError("COT event schema is invalid")
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, int)
            or timestamp < 0
        ):
            raise COTTransportError("COT event timestamp is invalid")
        try:
            safe_content = _sanitize_json_value(content, budget=[0])
            content_json = json.dumps(
                safe_content,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            content_json.encode("utf-8")
        except (TypeError, ValueError, UnicodeEncodeError, RecursionError):
            raise COTTransportError("COT event content is invalid") from None
        encoded.append(
            {
                "event_type": event_type,
                "content": content_json,
                "timestamp": timestamp,
            }
        )
    return encoded


def _sanitize_json_value(
    value: object,
    *,
    budget: list[int],
    depth: int = 0,
) -> object:
    budget[0] += 1
    if budget[0] > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
        raise ValueError("COT JSON content exceeds bounds")
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("COT JSON number is not finite")
        return value
    if isinstance(value, str):
        return _sanitize_unicode(value)
    if isinstance(value, Mapping):
        sanitized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("COT JSON object key is invalid")
            sanitized[_sanitize_unicode(key)] = _sanitize_json_value(
                item,
                budget=budget,
                depth=depth + 1,
            )
        return sanitized
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            _sanitize_json_value(item, budget=budget, depth=depth + 1)
            for item in value
        ]
    raise ValueError("COT JSON value is invalid")


def _sanitize_unicode(value: str) -> str:
    try:
        value.encode("utf-8")
        return value
    except UnicodeEncodeError:
        safe: list[str] = []
        index = 0
        while index < len(value):
            codepoint = ord(value[index])
            if 0xD800 <= codepoint <= 0xDBFF:
                if index + 1 < len(value):
                    low = ord(value[index + 1])
                    if 0xDC00 <= low <= 0xDFFF:
                        safe.append(
                            chr(
                                0x10000
                                + ((codepoint - 0xD800) << 10)
                                + (low - 0xDC00)
                            )
                        )
                        index += 2
                        continue
                safe.append("\ufffd")
            elif 0xDC00 <= codepoint <= 0xDFFF:
                safe.append("\ufffd")
            else:
                safe.append(value[index])
            index += 1
        return "".join(safe)


def _safe_tool_content(value: object, *, opaque_id: str) -> str:
    safe = sanitize_full_tool_event_content(value, opaque_ids=(opaque_id,))
    safe = _sanitize_unicode(safe)
    return _truncate_utf8(safe, _MAX_TOOL_CONTENT_BYTES)


def _split_utf8_chunks(value: str, maximum_bytes: int) -> tuple[str, ...]:
    """Split text without corrupting UTF-8 or losing process content."""
    encoded = value.encode("utf-8")
    if len(encoded) <= maximum_bytes:
        return (value,)
    chunks: list[str] = []
    start = 0
    while start < len(encoded):
        end = min(start + maximum_bytes, len(encoded))
        while end < len(encoded) and end > start and encoded[end] & 0xC0 == 0x80:
            end -= 1
        if end == start:
            end = min(start + maximum_bytes, len(encoded))
            while end < len(encoded) and encoded[end] & 0xC0 == 0x80:
                end += 1
        chunks.append(encoded[start:end].decode("utf-8"))
        start = end
    return tuple(chunks)


def _truncate_utf8(value: str, maximum_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= maximum_bytes:
        return value
    ellipsis = "…"
    prefix_limit = maximum_bytes - len(ellipsis.encode("utf-8"))
    prefix = _split_utf8_chunks(value, prefix_limit)[0]
    return prefix + ellipsis


def _partition_update_events(
    events: Sequence[Mapping[str, object]],
) -> tuple[tuple[Mapping[str, object], ...], ...]:
    """Bound each Update body while preserving strict FIFO event order."""
    batches: list[tuple[Mapping[str, object], ...]] = []
    current: list[Mapping[str, object]] = []
    current_bytes = 2
    for event in events:
        encoded = _encode_events((event,))[0]
        event_bytes = len(
            json.dumps(
                encoded,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ) + (1 if current else 0)
        if current and current_bytes + event_bytes > _MAX_UPDATE_EVENT_BYTES:
            batches.append(tuple(current))
            current = []
            current_bytes = 2
            event_bytes -= 1
        current.append(event)
        current_bytes += event_bytes
    if current:
        batches.append(tuple(current))
    return tuple(batches)


def _safe_transport_failure(error: COTTransportError) -> str:
    return sanitize_single_line_label(
        str(error),
        fallback=type(error).__name__,
        max_chars=240,
    )


def _safe_error_message(value: object) -> str:
    safe = sanitize_full_tool_event_content(value)
    safe = _sanitize_unicode(safe)
    if not safe:
        return "Task failed"
    return safe[:1_000]


def _hashed_identifier(kind: str, value: str) -> str:
    digest = hashlib.sha256(_sanitize_unicode(value).encode("utf-8")).hexdigest()
    return f"{kind}_{digest[:32]}"


def _is_timeout_reason(reason: str) -> bool:
    normalized = reason.strip().lower().replace("-", "_")
    return "timeout" in normalized or normalized in {"ttl_expired", "deadline"}


__all__ = [
    "COTBinding",
    "COTTransportError",
    "FeishuCOTAPIClient",
    "FeishuCOTSession",
]
