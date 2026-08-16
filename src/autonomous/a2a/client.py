"""Durable outbound-only A2A JSON-RPC 1.0 client adapter.

This module is the only production boundary that lets the A2A SDK touch the
autonomous domain.  The public Agent Card is fetched without credentials,
validated against an exact local registration, and only then may a bearer
credential be resolved.  Every remote observation is anchored by the ledger
before it is yielded to a caller.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import math
import re
import socket
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx
from a2a.client import Client, ClientCallContext, ClientConfig, ClientFactory
from a2a.client.transports.jsonrpc import (
    JSONRPC20Response,
    JsonRpcTransport,
    get_http_args,
)
from a2a.types import (
    CancelTaskRequest,
    GetTaskRequest,
    Message,
    Part,
    Role,
    SendMessageConfiguration,
    SendMessageRequest,
    StreamResponse,
    SubscribeToTaskRequest,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
)
from a2a.utils.constants import TransportProtocol
from google.protobuf import json_format

from src.autonomous.remote.models import (
    RemoteAgentDescriptor,
    RemoteAttemptPhase,
    RemoteDispatchRequest,
    RemoteObservation,
    RemoteProjection,
    RemoteSnapshot,
    RemoteTaskHandle,
    RemoteTaskState,
)

from .card import (
    MAX_AGENT_CARD_BYTES,
    AgentCardValidationError,
    PilotAgentRegistration,
    TrustedAgentCard,
    load_trusted_agent_card,
)
from .codec import (
    MAX_OBSERVATION_BYTES,
    A2ACodecError,
    NormalizedA2AObservation,
    normalize_a2a_observation,
)

_MAX_BEARER_BYTES = 8 * 1024
_MAX_CARD_TIMEOUT_SECONDS = 30.0
_MAX_REQUEST_TIMEOUT_SECONDS = 300.0
_MAX_STREAM_IDLE_SECONDS = 300.0
_BEARER_TOKEN_RE = re.compile(r"[A-Za-z0-9\-._~+/]+={0,}\Z", re.ASCII)


class RemoteA2AClientError(RuntimeError):
    """Stable, non-reflective failure at the remote transport boundary."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"remote A2A operation failed ({code})")


class BearerCredentialResolver(Protocol):
    """Resolve one credential already bound to a trusted descriptor."""

    async def resolve_bearer(self, descriptor: RemoteAgentDescriptor) -> str: ...


class _RemoteDispatchLedgerPort(Protocol):
    """Durable state boundary used by the outbound transport."""

    def projection(self) -> RemoteProjection: ...

    def snapshot(self, acceptance_id: str) -> RemoteSnapshot | None: ...

    def prepare(self, request: RemoteDispatchRequest) -> RemoteTaskHandle: ...

    def mark_executing(self, handle: RemoteTaskHandle) -> RemoteSnapshot: ...

    def bind_task(self, handle: RemoteTaskHandle, task_id: str) -> RemoteTaskHandle: ...

    def record_observation(
        self,
        handle: RemoteTaskHandle,
        observation: NormalizedA2AObservation | RemoteObservation,
        payload: bytes | None = None,
        *,
        observed_state: RemoteTaskState | None = None,
        sequence: int | None = None,
    ) -> RemoteObservation: ...

    def mark_send_uncertain(self, handle: RemoteTaskHandle) -> RemoteSnapshot: ...

    def request_cancel(self, handle: RemoteTaskHandle) -> RemoteSnapshot: ...


DNSResolver = Callable[[str, int], Awaitable[Sequence[str]]]
ResponsePeerVerifier = Callable[[httpx.Response], Awaitable[None]]


class _AsyncNetworkStream(Protocol):
    def get_extra_info(self, info: str) -> Any: ...

    async def aclose(self) -> None: ...


class _AsyncNetworkBackend(Protocol):
    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> _AsyncNetworkStream: ...

    async def sleep(self, seconds: float) -> None: ...


class _PublicOnlyNetworkBackend:
    """Reject a rebound private peer before HTTP or credential bytes are sent."""

    def __init__(self, backend: _AsyncNetworkBackend) -> None:
        self._backend = backend

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> _AsyncNetworkStream:
        stream = await self._backend.connect_tcp(
            host,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )
        try:
            _verify_connected_stream_peer(stream)
        except Exception:
            await stream.aclose()
            raise
        return stream

    async def connect_unix_socket(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RemoteA2AClientError("unix-socket-forbidden")

    async def sleep(self, seconds: float) -> None:
        await self._backend.sleep(seconds)


@dataclass(frozen=True, slots=True)
class A2AClientLimits:
    """Finite transport limits for the first outbound pilot."""

    connect_timeout_seconds: float = 5.0
    card_timeout_seconds: float = 10.0
    request_timeout_seconds: float = 30.0
    stream_idle_timeout_seconds: float = 30.0
    max_connections: int = 4
    max_keepalive_connections: int = 2

    def __post_init__(self) -> None:
        _validate_finite_timeout(
            self.connect_timeout_seconds,
            maximum=_MAX_CARD_TIMEOUT_SECONDS,
        )
        _validate_finite_timeout(
            self.card_timeout_seconds,
            maximum=_MAX_CARD_TIMEOUT_SECONDS,
        )
        _validate_finite_timeout(
            self.request_timeout_seconds,
            maximum=_MAX_REQUEST_TIMEOUT_SECONDS,
        )
        _validate_finite_timeout(
            self.stream_idle_timeout_seconds,
            maximum=_MAX_STREAM_IDLE_SECONDS,
        )
        if (
            type(self.max_connections) is not int
            or not 1 <= self.max_connections <= 32
            or type(self.max_keepalive_connections) is not int
            or not 0 <= self.max_keepalive_connections <= self.max_connections
        ):
            raise ValueError("invalid A2A connection limits")


@dataclass(frozen=True, slots=True)
class _BearerCredential:
    value: str = field(repr=False)

    def authorization_header(self) -> str:
        return f"Bearer {self.value}"


class _StrictJsonRpcTransport(JsonRpcTransport):
    """SDK JSON-RPC transport with strict duplicate/non-finite rejection."""

    async def _send_request(
        self,
        payload: dict[str, Any],
        context: ClientCallContext | None = None,
    ) -> dict[str, Any]:
        async with self.httpx_client.stream(
            "POST",
            self.url,
            json=payload,
            **(get_http_args(context) or {}),
        ) as response:
            response.raise_for_status()
            media_type = response.headers.get("content-type", "").partition(";")[0]
            if media_type.strip().lower() != "application/json":
                raise RemoteA2AClientError("jsonrpc-content-type")
            raw = await _read_bounded_response(
                response,
                max_bytes=MAX_OBSERVATION_BYTES,
            )
        value = _parse_strict_json_object(raw, max_bytes=MAX_OBSERVATION_BYTES)
        _validate_jsonrpc_envelope(value, request_id=payload.get("id"))
        return value

    async def _send_stream_request(
        self,
        rpc_request_payload: dict[str, Any],
        context: ClientCallContext | None = None,
    ) -> AsyncIterator[StreamResponse]:
        async with self.httpx_client.stream(
            "POST",
            self.url,
            json=rpc_request_payload,
            **get_http_args(context),
        ) as response:
            response.raise_for_status()
            media_type = response.headers.get("content-type", "").partition(";")[0]
            if media_type.strip().lower() != "text/event-stream":
                raise RemoteA2AClientError("jsonrpc-stream-content-type")
            events = _iter_bounded_sse_data(
                response,
                max_event_bytes=MAX_OBSERVATION_BYTES,
            )
            async for event_name, sse_data in events:
                if event_name == "error":
                    self._handle_strict_sse_error(
                        sse_data,
                        request_id=rpc_request_payload.get("id"),
                    )
                    raise AssertionError("strict SSE error handler must raise")
                value = _parse_strict_json_object(
                    sse_data.encode("utf-8", errors="strict"),
                    max_bytes=MAX_OBSERVATION_BYTES,
                )
                _validate_jsonrpc_envelope(
                    value,
                    request_id=rpc_request_payload.get("id"),
                )
                json_rpc_response = JSONRPC20Response(**value)
                if json_rpc_response.error:
                    raise self._create_jsonrpc_error(json_rpc_response.error)
                parsed_response = json_format.ParseDict(
                    json_rpc_response.result,
                    StreamResponse(),
                    ignore_unknown_fields=False,
                    max_recursion_depth=100,
                )
                yield parsed_response

    def _handle_strict_sse_error(self, sse_data: str, *, request_id: object) -> None:
        value = _parse_strict_json_object(
            sse_data.encode("utf-8", errors="strict"),
            max_bytes=MAX_OBSERVATION_BYTES,
        )
        _validate_jsonrpc_envelope(value, request_id=request_id)
        json_rpc_response = JSONRPC20Response(**value)
        if json_rpc_response.error:
            raise self._create_jsonrpc_error(json_rpc_response.error)
        raise RemoteA2AClientError("sse-error-event-invalid")


async def _close_after_failed_create(client: httpx.AsyncClient) -> None:
    """Best-effort cleanup that lets an in-flight close survive cancellation."""

    close_task = asyncio.create_task(client.aclose())
    try:
        await asyncio.shield(close_task)
    except asyncio.CancelledError:
        try:
            await close_task
        except BaseException:
            pass
        raise
    except BaseException:
        pass


class RemoteA2ADispatchAdapter:
    """A fixed-card, fixed-endpoint implementation of remote dispatch."""

    __slots__ = (
        "_accepted_output_modes",
        "_client",
        "_closed",
        "_credential",
        "_dns_resolver",
        "_ledger",
        "_limits",
        "_trusted_card",
    )

    def __init__(
        self,
        *,
        trusted_card: TrustedAgentCard,
        credential: _BearerCredential,
        ledger: _RemoteDispatchLedgerPort,
        client: Client,
        dns_resolver: DNSResolver,
        limits: A2AClientLimits,
        accepted_output_modes: tuple[str, ...],
    ) -> None:
        self._trusted_card = trusted_card
        self._credential = credential
        self._ledger = ledger
        self._client = client
        self._dns_resolver = dns_resolver
        self._limits = limits
        self._accepted_output_modes = accepted_output_modes
        self._closed = False

    def __repr__(self) -> str:
        registration = self._trusted_card.registration
        return (
            f"{type(self).__name__}(agent_id={registration.agent_id!r}, "
            f"card_digest={self._trusted_card.canonical_digest!r}, "
            f"closed={self._closed!r})"
        )

    @classmethod
    async def create(
        cls,
        registration: PilotAgentRegistration,
        ledger: _RemoteDispatchLedgerPort,
        credential_resolver: BearerCredentialResolver,
        *,
        limits: A2AClientLimits | None = None,
        dns_resolver: DNSResolver | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        response_peer_verifier: ResponsePeerVerifier | None = None,
    ) -> RemoteA2ADispatchAdapter:
        """Fetch and validate a Card before resolving any credential."""

        chosen_limits = limits or A2AClientLimits()
        chosen_resolver = dns_resolver or _resolve_host_addresses
        chosen_peer_verifier = response_peer_verifier or _verify_response_peer
        if response_peer_verifier is not None and transport is None:
            raise ValueError("a custom peer verifier requires a custom transport")
        chosen_transport = transport or _hardened_http_transport(chosen_limits)
        timeout = httpx.Timeout(
            timeout=chosen_limits.request_timeout_seconds,
            connect=chosen_limits.connect_timeout_seconds,
        )
        http_client = httpx.AsyncClient(
            follow_redirects=False,
            limits=httpx.Limits(
                max_connections=chosen_limits.max_connections,
                max_keepalive_connections=chosen_limits.max_keepalive_connections,
            ),
            timeout=timeout,
            transport=chosen_transport,
            trust_env=False,
            verify=True,
            event_hooks={"response": [chosen_peer_verifier]},
        )
        try:
            await _assert_public_resolution(
                registration.card_url,
                chosen_resolver,
                timeout_seconds=chosen_limits.connect_timeout_seconds,
            )
            raw_card = await _fetch_public_card(
                http_client,
                registration.card_url,
                timeout_seconds=chosen_limits.card_timeout_seconds,
            )
            trusted = load_trusted_agent_card(registration, raw_card)
            if trusted.selected_interface.tenant != registration.remote_tenant:
                raise RemoteA2AClientError("interface-tenant-mismatch")
            await _assert_public_resolution(
                registration.endpoint_url,
                chosen_resolver,
                timeout_seconds=chosen_limits.connect_timeout_seconds,
            )
            descriptor = trusted.to_remote_descriptor()
            accepted_output_modes = trusted.accepted_output_modes

            try:
                bearer_value = await credential_resolver.resolve_bearer(descriptor)
            except Exception:
                raise RemoteA2AClientError("credential-resolution-failed") from None
            credential = _validate_bearer_credential(bearer_value)

            config = ClientConfig(
                streaming=True,
                polling=False,
                httpx_client=http_client,
                supported_protocol_bindings=[TransportProtocol.JSONRPC],
                use_client_preference=True,
                accepted_output_modes=list(accepted_output_modes),
                push_notification_config=None,
            )
            try:
                factory = ClientFactory(config)
                factory.register(
                    TransportProtocol.JSONRPC,
                    lambda card, url, client_config: _StrictJsonRpcTransport(
                        client_config.httpx_client,
                        card,
                        url,
                    ),
                )
                client = factory.create(trusted.sdk_card)
            except Exception:
                raise RemoteA2AClientError("sdk-client-creation-failed") from None
        except BaseException as exc:
            await _close_after_failed_create(http_client)
            if isinstance(exc, AgentCardValidationError):
                raise RemoteA2AClientError(f"agent-card-{exc.code}") from None
            if isinstance(exc, RemoteA2AClientError):
                raise
            if isinstance(exc, asyncio.CancelledError):
                raise
            if isinstance(exc, Exception):
                raise RemoteA2AClientError("agent-card-fetch-failed") from None
            raise

        return cls(
            trusted_card=trusted,
            credential=credential,
            ledger=ledger,
            client=client,
            dns_resolver=chosen_resolver,
            limits=chosen_limits,
            accepted_output_modes=accepted_output_modes,
        )

    @property
    def descriptor(self) -> RemoteAgentDescriptor:
        """Return the exact immutable descriptor admitted for this client."""

        return self._trusted_card.to_remote_descriptor()

    async def close(self) -> None:
        """Close the SDK transport and its owned hardened HTTP client once."""

        if self._closed:
            return
        try:
            await self._client.close()
        except asyncio.CancelledError:
            raise
        except Exception:
            raise RemoteA2AClientError("client-close-failed") from None
        self._closed = True

    async def __aenter__(self) -> RemoteA2ADispatchAdapter:
        self._ensure_open()
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.close()

    async def dispatch(
        self,
        request: RemoteDispatchRequest,
    ) -> AsyncIterator[RemoteObservation]:
        """Send once, or reconcile an already-bound task without resending."""

        self._ensure_open()
        self._assert_descriptor(request.descriptor)
        snapshot = self._safe_snapshot(request.acceptance_id)
        if snapshot is None:
            handle = self._safe_prepare(request)
            snapshot = self._require_snapshot(request.acceptance_id)
        else:
            handle = snapshot.handle
            self._assert_request_matches_handle(request, handle)

        if snapshot.phase is RemoteAttemptPhase.TERMINAL:
            return
        if handle.task_id:
            async for observation in self._recover_known_task(handle):
                yield observation
            return
        if snapshot.phase is not RemoteAttemptPhase.PREPARED:
            if snapshot.phase is RemoteAttemptPhase.EXECUTING:
                self._mark_send_uncertain(handle)
            raise RemoteA2AClientError("send-outcome-unknown")

        self._assert_instruction_matches_handle(request, handle)
        await self._guard_endpoint()
        send_request = SendMessageRequest(
            tenant=request.descriptor.remote_tenant,
            message=Message(
                message_id=request.message_id,
                context_id=request.context_id,
                role=Role.ROLE_USER,
                parts=[Part(text=request.instruction, media_type="text/plain")],
            ),
            configuration=SendMessageConfiguration(
                accepted_output_modes=list(self._accepted_output_modes),
                return_immediately=False,
            ),
        )
        try:
            self._ledger.mark_executing(handle)
        except Exception:
            raise RemoteA2AClientError("journal-anchor-failed") from None

        observed_any = False
        try:
            stream = self._client.send_message(send_request, context=self._call_context())
            async for response in self._iterate_stream(
                stream,
                failure_code="send-stream-failed",
            ):
                observed_any = True
                handle, observation = self._record_response(handle, response)
                if observation is not None:
                    yield observation
        except RemoteA2AClientError:
            current = self._safe_snapshot(request.acceptance_id)
            if current is not None and current.handle.task_id:
                async for observation in self._recover_known_task(current.handle):
                    yield observation
                return
            self._mark_send_uncertain(handle)
            raise
        except (A2ACodecError, TypeError, ValueError):
            current = self._safe_snapshot(request.acceptance_id)
            if current is None or not current.handle.task_id:
                self._mark_send_uncertain(handle)
            raise RemoteA2AClientError("remote-response-invalid") from None
        except Exception:
            current = self._safe_snapshot(request.acceptance_id)
            if current is None or not current.handle.task_id:
                self._mark_send_uncertain(handle)
            raise RemoteA2AClientError("journal-write-failed") from None

        current = self._require_snapshot(request.acceptance_id)
        if current.phase is RemoteAttemptPhase.TERMINAL:
            return
        if current.handle.task_id:
            observation = await self._get_and_record(current.handle)
            if observation is not None:
                yield observation
            return
        if not observed_any:
            self._mark_send_uncertain(handle)
            raise RemoteA2AClientError("send-outcome-unknown")
        # A direct Message may legitimately complete without creating a Task.
        if current.state is not RemoteTaskState.CLAIMED_COMPLETED:
            self._mark_send_uncertain(handle)
            raise RemoteA2AClientError("send-outcome-unknown")

    async def get_task(self, handle: RemoteTaskHandle) -> RemoteSnapshot:
        """Fetch and anchor a known task snapshot."""

        self._ensure_open()
        self._assert_handle(handle, require_task=True)
        await self._get_and_record(handle)
        return self._require_snapshot(handle.acceptance_id)

    async def subscribe(
        self,
        handle: RemoteTaskHandle,
    ) -> AsyncIterator[RemoteObservation]:
        """Subscribe to and durably anchor updates for one known task."""

        self._ensure_open()
        self._assert_handle(handle, require_task=True)
        await self._guard_endpoint()
        request = SubscribeToTaskRequest(
            tenant=handle.descriptor.remote_tenant,
            id=handle.task_id,
        )
        stream = self._client.subscribe(request, context=self._call_context())
        async for response in self._iterate_stream(
            stream,
            failure_code="subscribe-failed",
        ):
            try:
                handle, observation = self._record_response(handle, response)
            except (A2ACodecError, TypeError, ValueError):
                raise RemoteA2AClientError("remote-response-invalid") from None
            except Exception:
                raise RemoteA2AClientError("journal-write-failed") from None
            if observation is not None:
                yield observation

    async def cancel(self, handle: RemoteTaskHandle) -> RemoteSnapshot:
        """Anchor cancellation intent before sending CancelTask."""

        self._ensure_open()
        self._assert_handle(handle, require_task=False)
        current = self._require_snapshot(handle.acceptance_id)
        if current.phase is RemoteAttemptPhase.TERMINAL:
            return current
        try:
            requested = self._ledger.request_cancel(handle)
        except Exception:
            raise RemoteA2AClientError("journal-anchor-failed") from None
        handle = requested.handle
        if not handle.task_id:
            return requested

        await self._guard_endpoint()
        request = CancelTaskRequest(
            tenant=handle.descriptor.remote_tenant,
            id=handle.task_id,
        )
        try:
            task = await self._await_request(
                self._client.cancel_task(request, context=self._call_context()),
                failure_code="cancel-failed",
            )
            self._record_response(handle, task)
        except RemoteA2AClientError:
            try:
                await self._get_and_record(handle)
            except RemoteA2AClientError:
                raise RemoteA2AClientError("cancel-outcome-unknown") from None
        except (A2ACodecError, TypeError, ValueError):
            raise RemoteA2AClientError("remote-response-invalid") from None
        except Exception:
            raise RemoteA2AClientError("journal-write-failed") from None
        return self._require_snapshot(handle.acceptance_id)

    async def _recover_known_task(
        self,
        handle: RemoteTaskHandle,
    ) -> AsyncIterator[RemoteObservation]:
        snapshot = self._require_snapshot(handle.acceptance_id)
        if snapshot.phase is RemoteAttemptPhase.TERMINAL:
            return
        try:
            async for observation in self.subscribe(handle):
                yield observation
        except RemoteA2AClientError as exc:
            if exc.code not in {"stream-idle-timeout", "subscribe-failed"}:
                raise

        snapshot = self._require_snapshot(handle.acceptance_id)
        if snapshot.phase is RemoteAttemptPhase.TERMINAL:
            return
        observation = await self._get_and_record(snapshot.handle)
        if observation is not None:
            yield observation

    async def _get_and_record(
        self,
        handle: RemoteTaskHandle,
    ) -> RemoteObservation | None:
        self._assert_handle(handle, require_task=True)
        await self._guard_endpoint()
        request = GetTaskRequest(
            tenant=handle.descriptor.remote_tenant,
            id=handle.task_id,
            history_length=20,
        )
        task = await self._await_request(
            self._client.get_task(request, context=self._call_context()),
            failure_code="get-task-failed",
        )
        try:
            _handle, observation = self._record_response(handle, task)
        except (A2ACodecError, TypeError, ValueError):
            raise RemoteA2AClientError("remote-response-invalid") from None
        except Exception:
            raise RemoteA2AClientError("journal-write-failed") from None
        return observation

    def _record_response(
        self,
        handle: RemoteTaskHandle,
        response: StreamResponse | Task,
    ) -> tuple[RemoteTaskHandle, RemoteObservation | None]:
        context_id, task_id = _response_ids(response)
        if context_id != handle.context_id:
            raise A2ACodecError("context-id-mismatch")
        if task_id:
            if handle.task_id and task_id != handle.task_id:
                raise A2ACodecError("task-id-mismatch")
            if not handle.task_id:
                handle = self._ledger.bind_task(handle, task_id)

        normalized = normalize_a2a_observation(response, handle)
        before = self._require_snapshot(handle.acceptance_id)
        known_ids = {item.observation_id for item in before.observations}
        observation = self._ledger.record_observation(
            handle,
            normalized,
            observed_state=before.state,
            sequence=None,
        )
        after = self._require_snapshot(handle.acceptance_id)
        return after.handle, None if observation.observation_id in known_ids else observation

    async def _iterate_stream(
        self,
        stream: AsyncIterator[StreamResponse],
        *,
        failure_code: str,
    ) -> AsyncIterator[StreamResponse]:
        iterator = aiter(stream)
        while True:
            try:
                async with asyncio.timeout(self._limits.stream_idle_timeout_seconds):
                    response = await anext(iterator)
            except StopAsyncIteration:
                return
            except TimeoutError:
                raise RemoteA2AClientError("stream-idle-timeout") from None
            except RemoteA2AClientError:
                raise
            except Exception:
                raise RemoteA2AClientError(failure_code) from None
            yield response

    async def _await_request(self, value: Awaitable[Task], *, failure_code: str) -> Task:
        try:
            async with asyncio.timeout(self._limits.request_timeout_seconds):
                return await value
        except TimeoutError:
            raise RemoteA2AClientError("request-timeout") from None
        except RemoteA2AClientError:
            raise
        except Exception:
            raise RemoteA2AClientError(failure_code) from None

    async def _guard_endpoint(self) -> None:
        await _assert_public_resolution(
            self._trusted_card.registration.endpoint_url,
            self._dns_resolver,
            timeout_seconds=self._limits.connect_timeout_seconds,
        )

    def _call_context(self) -> ClientCallContext:
        return ClientCallContext(
            timeout=self._limits.request_timeout_seconds,
            service_parameters={
                "Authorization": self._credential.authorization_header(),
            },
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RemoteA2AClientError("client-closed")

    def _assert_descriptor(self, descriptor: RemoteAgentDescriptor) -> None:
        if descriptor != self.descriptor:
            raise RemoteA2AClientError("descriptor-mismatch")

    def _assert_handle(self, handle: RemoteTaskHandle, *, require_task: bool) -> None:
        if not isinstance(handle, RemoteTaskHandle):
            raise RemoteA2AClientError("invalid-handle")
        self._assert_descriptor(handle.descriptor)
        snapshot = self._safe_snapshot(handle.acceptance_id)
        if snapshot is None or snapshot.handle != handle:
            raise RemoteA2AClientError("handle-mismatch")
        if require_task and not handle.task_id:
            raise RemoteA2AClientError("task-id-unbound")

    def _assert_request_matches_handle(
        self,
        request: RemoteDispatchRequest,
        handle: RemoteTaskHandle,
    ) -> None:
        if (
            request.acceptance_id != handle.acceptance_id
            or request.run_id != handle.run_id
            or request.assignment_id != handle.assignment_id
            or request.attempt_id != handle.attempt_id
            or request.message_id != handle.message_id
            or request.context_id != handle.context_id
            or request.descriptor != handle.descriptor
        ):
            raise RemoteA2AClientError("dispatch-identity-mismatch")
        self._assert_instruction_matches_handle(request, handle)

    @staticmethod
    def _assert_instruction_matches_handle(
        request: RemoteDispatchRequest,
        handle: RemoteTaskHandle,
    ) -> None:
        instruction_digest = hashlib.sha256(request.instruction.encode("utf-8")).hexdigest()
        if instruction_digest != handle.instruction_ref.payload_hash:
            raise RemoteA2AClientError("dispatch-instruction-mismatch")

    def _safe_snapshot(self, acceptance_id: str) -> RemoteSnapshot | None:
        try:
            return self._ledger.snapshot(acceptance_id)
        except Exception:
            raise RemoteA2AClientError("journal-read-failed") from None

    def _require_snapshot(self, acceptance_id: str) -> RemoteSnapshot:
        snapshot = self._safe_snapshot(acceptance_id)
        if snapshot is None:
            raise RemoteA2AClientError("journal-state-missing")
        return snapshot

    def _safe_prepare(self, request: RemoteDispatchRequest) -> RemoteTaskHandle:
        try:
            return self._ledger.prepare(request)
        except Exception:
            raise RemoteA2AClientError("journal-anchor-failed") from None

    def _mark_send_uncertain(self, handle: RemoteTaskHandle) -> None:
        try:
            self._ledger.mark_send_uncertain(handle)
        except Exception:
            raise RemoteA2AClientError("journal-anchor-failed") from None


async def _read_bounded_response(
    response: httpx.Response,
    *,
    max_bytes: int,
) -> bytes:
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body.extend(chunk)
        if len(body) > max_bytes:
            raise RemoteA2AClientError("jsonrpc-response-too-large")
    return bytes(body)


async def _iter_bounded_sse_data(
    response: httpx.Response,
    *,
    max_event_bytes: int,
) -> AsyncIterator[tuple[str, str]]:
    """Parse SSE incrementally without allowing an unbounded line or event."""

    line = bytearray()
    data_lines: list[bytes] = []
    data_size = 0
    event_name = "message"

    def consume_line(raw_line: bytes) -> tuple[str, str] | None:
        nonlocal data_lines, data_size, event_name
        if not raw_line:
            if not data_lines:
                event_name = "message"
                return None
            joined = b"\n".join(data_lines)
            try:
                data = joined.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                raise RemoteA2AClientError("sse-invalid-utf8") from None
            result = event_name, data
            data_lines = []
            data_size = 0
            event_name = "message"
            return result
        if raw_line.startswith(b":"):
            return None
        field, separator, value = raw_line.partition(b":")
        if separator and value.startswith(b" "):
            value = value[1:]
        if field == b"data":
            data_size += len(value) + (1 if data_lines else 0)
            if data_size > max_event_bytes:
                raise RemoteA2AClientError("sse-event-too-large")
            data_lines.append(value)
        elif field == b"event":
            try:
                event_name = value.decode("utf-8", errors="strict") or "message"
            except UnicodeDecodeError:
                raise RemoteA2AClientError("sse-invalid-utf8") from None
        return None

    pending_cr = False
    async for chunk in response.aiter_bytes():
        for byte in chunk:
            if pending_cr:
                pending_cr = False
                if byte == 0x0A:
                    continue
            if byte == 0x0D:
                event = consume_line(bytes(line))
                line.clear()
                pending_cr = True
                if event is not None:
                    yield event
            elif byte == 0x0A:
                event = consume_line(bytes(line))
                line.clear()
                if event is not None:
                    yield event
            else:
                line.append(byte)
                if len(line) > max_event_bytes:
                    raise RemoteA2AClientError("sse-line-too-large")

    if line:
        event = consume_line(bytes(line))
        if event is not None:
            yield event
    final_event = consume_line(b"")
    if final_event is not None:
        yield final_event


def _parse_strict_json_object(raw: bytes, *, max_bytes: int) -> dict[str, Any]:
    if len(raw) > max_bytes:
        raise RemoteA2AClientError("jsonrpc-response-too-large")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise RemoteA2AClientError("jsonrpc-invalid-json") from None

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise RemoteA2AClientError("jsonrpc-duplicate-key")
            value[key] = item
        return value

    def reject_non_finite(_value: str) -> None:
        raise RemoteA2AClientError("jsonrpc-non-finite-number")

    try:
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_non_finite,
        )
    except RemoteA2AClientError:
        raise
    except (RecursionError, TypeError, ValueError, json.JSONDecodeError):
        raise RemoteA2AClientError("jsonrpc-invalid-json") from None
    if not isinstance(value, dict):
        raise RemoteA2AClientError("jsonrpc-invalid-envelope")
    return value


def _validate_jsonrpc_envelope(value: dict[str, Any], *, request_id: object) -> None:
    allowed_keys = {"jsonrpc", "id", "result", "error"}
    if (
        set(value) - allowed_keys
        or value.get("jsonrpc") != "2.0"
        or value.get("id") != request_id
        or (("result" in value) == ("error" in value))
    ):
        raise RemoteA2AClientError("jsonrpc-invalid-envelope")


def _hardened_http_transport(limits: A2AClientLimits) -> httpx.AsyncBaseTransport:
    """Build the pinned HTTPX transport with connect-time peer admission.

    HTTPX 0.28 does not expose httpcore's network backend in its public
    constructor.  This intentionally version-pinned seam is covered by a
    dependency contract and fails closed if that internal shape changes.
    """

    transport = httpx.AsyncHTTPTransport(
        verify=True,
        trust_env=False,
        limits=httpx.Limits(
            max_connections=limits.max_connections,
            max_keepalive_connections=limits.max_keepalive_connections,
        ),
        retries=0,
    )
    pool = getattr(transport, "_pool", None)
    backend = getattr(pool, "_network_backend", None)
    if pool is None or backend is None:
        raise RemoteA2AClientError("network-backend-unavailable")
    pool._network_backend = _PublicOnlyNetworkBackend(backend)
    return transport


def _verify_connected_stream_peer(stream: _AsyncNetworkStream) -> None:
    try:
        server_address = stream.get_extra_info("server_addr")
    except Exception:
        raise RemoteA2AClientError("peer-address-unavailable") from None
    if not isinstance(server_address, (tuple, list)) or not server_address:
        raise RemoteA2AClientError("peer-address-unavailable")
    _assert_global_address(server_address[0], error_code="peer-address-rejected")


async def _verify_response_peer(response: httpx.Response) -> None:
    """Fail closed unless httpcore exposes a public connected peer address."""

    network_stream = response.extensions.get("network_stream")
    if network_stream is None or not hasattr(network_stream, "get_extra_info"):
        raise RemoteA2AClientError("peer-address-unavailable")
    _verify_connected_stream_peer(network_stream)


async def _fetch_public_card(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout_seconds: float,
) -> bytes:
    try:
        async with asyncio.timeout(timeout_seconds):
            async with client.stream(
                "GET",
                url,
                headers={"Accept": "application/json"},
            ) as response:
                if response.status_code != httpx.codes.OK:
                    raise RemoteA2AClientError("agent-card-http-status")
                media_type = response.headers.get("content-type", "").partition(";")[0]
                if media_type.strip().lower() != "application/json":
                    raise RemoteA2AClientError("agent-card-content-type")
                declared_length = response.headers.get("content-length")
                if declared_length is not None:
                    try:
                        if int(declared_length) > MAX_AGENT_CARD_BYTES:
                            raise RemoteA2AClientError("agent-card-too-large")
                    except ValueError:
                        raise RemoteA2AClientError("agent-card-invalid-length") from None
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_AGENT_CARD_BYTES:
                        raise RemoteA2AClientError("agent-card-too-large")
                return bytes(body)
    except RemoteA2AClientError:
        raise
    except TimeoutError:
        raise RemoteA2AClientError("agent-card-timeout") from None
    except Exception:
        raise RemoteA2AClientError("agent-card-fetch-failed") from None


async def _resolve_host_addresses(host: str, port: int) -> Sequence[str]:
    loop = asyncio.get_running_loop()
    records = await loop.getaddrinfo(
        host,
        port,
        family=socket.AF_UNSPEC,
        type=socket.SOCK_STREAM,
    )
    return tuple(str(record[4][0]) for record in records)


async def _assert_public_resolution(
    url: str,
    resolver: DNSResolver,
    *,
    timeout_seconds: float,
) -> None:
    parsed = urlsplit(url)
    hostname = parsed.hostname
    if hostname is None:
        raise RemoteA2AClientError("endpoint-resolution-rejected")
    try:
        async with asyncio.timeout(timeout_seconds):
            addresses = tuple(await resolver(hostname, parsed.port or 443))
    except TimeoutError:
        raise RemoteA2AClientError("endpoint-resolution-timeout") from None
    except RemoteA2AClientError:
        raise
    except Exception:
        raise RemoteA2AClientError("endpoint-resolution-failed") from None
    if not addresses:
        raise RemoteA2AClientError("endpoint-resolution-rejected")
    for value in addresses:
        _assert_global_address(value, error_code="endpoint-resolution-rejected")


def _assert_global_address(value: object, *, error_code: str) -> None:
    if not isinstance(value, str):
        raise RemoteA2AClientError(error_code)
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        raise RemoteA2AClientError(error_code) from None
    if not address.is_global or (isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None):
        raise RemoteA2AClientError(error_code)


def _response_ids(response: StreamResponse | Task) -> tuple[str, str]:
    value: object = response
    if isinstance(response, StreamResponse):
        payload_name = response.WhichOneof("payload")
        if payload_name is None:
            return "", ""
        value = getattr(response, payload_name)
    if isinstance(value, Task):
        return value.context_id, value.id
    if isinstance(value, Message):
        return value.context_id, value.task_id
    if isinstance(value, (TaskStatusUpdateEvent, TaskArtifactUpdateEvent)):
        return value.context_id, value.task_id
    return "", ""


def _validate_bearer_credential(value: object) -> _BearerCredential:
    if not isinstance(value, str) or _BEARER_TOKEN_RE.fullmatch(value) is None:
        raise RemoteA2AClientError("credential-invalid")
    try:
        encoded = value.encode("ascii", errors="strict")
    except UnicodeEncodeError:
        raise RemoteA2AClientError("credential-invalid") from None
    if len(encoded) > _MAX_BEARER_BYTES:
        raise RemoteA2AClientError("credential-invalid")
    return _BearerCredential(value)


def _validate_finite_timeout(value: object, *, maximum: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 < float(value) <= maximum
    ):
        raise ValueError("invalid A2A timeout")


__all__ = [
    "A2AClientLimits",
    "BearerCredentialResolver",
    "DNSResolver",
    "ResponsePeerVerifier",
    "RemoteA2AClientError",
    "RemoteA2ADispatchAdapter",
]
