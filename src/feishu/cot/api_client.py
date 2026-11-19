"""Strict generic-SDK client for Feishu's untyped IM COT endpoints."""

from __future__ import annotations

import json
import logging
import math
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import lark_oapi.core  # noqa: F401
from lark_oapi.core.enum import AccessTokenType, HttpMethod
from lark_oapi.core.model.base_request import BaseRequest
from lark_oapi.core.model.base_response import BaseResponse

from .wire import (
    _COLLECTION_URI,
    _COMPLETE_URI,
    _JSON_HEADERS,
    COTBinding,
    COTTransportError,
    TrustRevision,
    _encode_events,
    _safe_response_message,
    _valid_identifier,
    _validate_trust_revision,
)

logger = logging.getLogger(__name__)

_DEFAULT_API_TIMEOUT_SECONDS = 35.0


class _SyncLarkClient(Protocol):
    def request(self, request: BaseRequest) -> BaseResponse: ...


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
            response_message = _safe_response_message(payload.get("msg"))
            detail = f", msg={response_message}" if response_message else ""
            raise _COTResponseRejected(
                f"COT {operation} failed (code={code}{detail})"
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
