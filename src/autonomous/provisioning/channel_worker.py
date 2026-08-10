"""Fresh-interpreter worker hosting exactly one employee FeishuChannel."""

from __future__ import annotations

import ctypes
import dataclasses
import json
import os
import queue
import resource
import select
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

_LARK_DOMAIN = "https://open.feishu.cn"

# ``python -I /absolute/worker.py`` intentionally excludes the repository from
# sys.path. Add only the immutable repository root derived from this file.
_REPOSITORY_ROOT = str(Path(__file__).resolve().parents[3])
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

from src.autonomous.ingress.models import (  # noqa: E402
    EmployeeIngressAck,
    EmployeeIngressMetadata,
    EmployeeIngressPayload,
)
from src.autonomous.provisioning.channel_protocol import (  # noqa: E402
    MAX_FRAME_BYTES,
    ChannelFrame,
    FrameType,
    decode_bootstrap,
    decode_frame,
    encode_frame,
)


class WorkerSecurityError(RuntimeError):
    """Mandatory process hardening could not be applied."""


@dataclass(slots=True)
class _PendingIngressAck:
    request_id: str
    agent_id: str
    app_id: str
    generation: int
    connection_id: str
    semantic_digest: str
    envelope_id: str
    dedup_key: str
    completed: threading.Event = field(default_factory=threading.Event)
    ack: EmployeeIngressAck | None = None
    closed: bool = False


class IngressAckMailbox:
    """Contextual owner for bounded, current-generation durable ACK waits."""

    def __init__(self) -> None:
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._pending: dict[str, _PendingIngressAck] = {}
        self._closed = False

    def register(
        self,
        *,
        request_id: str,
        agent_id: str,
        app_id: str,
        generation: int,
        connection_id: str,
        semantic_digest: str,
        envelope_id: str,
        dedup_key: str,
    ) -> _PendingIngressAck:
        pending = _PendingIngressAck(
            request_id=request_id,
            agent_id=agent_id,
            app_id=app_id,
            generation=generation,
            connection_id=connection_id,
            semantic_digest=semantic_digest,
            envelope_id=envelope_id,
            dedup_key=dedup_key,
        )
        with self._lock:
            if self._closed:
                raise EOFError("employee ingress parent unavailable")
            if request_id in self._pending:
                raise ValueError("duplicate ingress request")
            self._pending[request_id] = pending
        return pending

    def deliver(self, ack: EmployeeIngressAck) -> bool:
        if not isinstance(ack, EmployeeIngressAck):
            return False
        with self._lock:
            pending = self._pending.get(ack.request_id)
            if pending is None or pending.closed:
                return False
            if (
                ack.agent_id != pending.agent_id
                or ack.app_id != pending.app_id
                or ack.channel_generation != pending.generation
                or ack.connection_id != pending.connection_id
                or ack.semantic_digest != pending.semantic_digest
                or ack.acceptance.envelope_id != pending.envelope_id
                or ack.acceptance.dedup_key != pending.dedup_key
            ):
                return False
            pending.ack = ack
            pending.completed.set()
            return True

    def wait(
        self,
        pending: _PendingIngressAck,
        *,
        timeout: float,
    ) -> EmployeeIngressAck:
        if not pending.completed.wait(timeout):
            with self._lock:
                if self._pending.get(pending.request_id) is pending:
                    self._pending.pop(pending.request_id, None)
                    pending.closed = True
            raise TimeoutError("employee ingress durable ACK timed out")
        with self._lock:
            if self._pending.get(pending.request_id) is pending:
                self._pending.pop(pending.request_id, None)
            pending.closed = True
            ack = pending.ack
        if ack is None:
            raise EOFError("employee ingress parent unavailable")
        return ack

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            pending = tuple(self._pending.values())
            self._pending.clear()
            for item in pending:
                item.closed = True
                item.completed.set()

    def cancel_pending(self) -> None:
        """Fail current waits while keeping the mailbox open for reconnect."""
        with self._lock:
            pending = tuple(self._pending.values())
            self._pending.clear()
            for item in pending:
                item.closed = True
                item.completed.set()

    def cancel(self, pending: _PendingIngressAck) -> None:
        """Remove one request that could not be emitted to the parent."""
        with self._lock:
            if self._pending.get(pending.request_id) is pending:
                self._pending.pop(pending.request_id, None)
            pending.closed = True
            pending.completed.set()


class _ConnectionAdmission:
    """Atomically fence callback ownership to one observed WS connection."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._epoch = 0
        self._connection_id = ""
        self._ready = False

    @property
    def epoch(self) -> int:
        with self._condition:
            return self._epoch

    def begin_reconnect(self, mailbox: IngressAckMailbox) -> int:
        with self._condition:
            self._epoch += 1
            self._connection_id = ""
            self._ready = False
            mailbox.cancel_pending()
            return self._epoch

    def publish_observed(
        self,
        connection_id: str,
        *,
        expected_epoch: int,
        emit_ready: Callable[[], None],
    ) -> bool:
        with self._condition:
            if self._epoch != expected_epoch or self._ready:
                return False
            self._connection_id = connection_id
            self._ready = True
            try:
                emit_ready()
            except BaseException:
                self._connection_id = ""
                self._ready = False
                raise
            self._condition.notify_all()
            return True

    def wait_snapshot(self, *, deadline: float) -> tuple[int, str]:
        with self._condition:
            while not self._ready:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._condition.wait(timeout=remaining):
                    raise TimeoutError(
                        "employee ingress connection readiness timed out"
                    )
            return self._epoch, self._connection_id

    def register(
        self,
        mailbox: IngressAckMailbox,
        *,
        expected_epoch: int,
        request_id: str,
        agent_id: str,
        app_id: str,
        generation: int,
        connection_id: str,
        semantic_digest: str,
        envelope_id: str,
        dedup_key: str,
    ) -> _PendingIngressAck:
        with self._condition:
            if (
                not self._ready
                or self._epoch != expected_epoch
                or self._connection_id != connection_id
            ):
                raise EOFError("employee ingress connection ownership changed")
            return mailbox.register(
                request_id=request_id,
                agent_id=agent_id,
                app_id=app_id,
                generation=generation,
                connection_id=connection_id,
                semantic_digest=semantic_digest,
                envelope_id=envelope_id,
                dedup_key=dedup_key,
            )


def run_low_level_employee_channel(
    bootstrap_fd: int,
    control_fd: int,
    event_fd: int,
    *,
    sandbox_proof_fd: int | None = None,
    sandbox_proof_nonce: str = "",
) -> None:
    """Run the pinned synchronous SDK dispatcher with parent-durable ACKs."""

    apply_process_hardening()
    from src.autonomous.ingress.sdk_capability import (
        collect_sdk_distribution_identity,
        prepare_controlled_sdk_import_cache,
    )

    channel_tmp = Path(os.environ.get("GHOSTAP_CHANNEL_TMP", "/tmp"))
    prepare_controlled_sdk_import_cache(
        channel_tmp / f"ghostap-employee-channel-sdk-{os.getpid()}"
    )
    collect_sdk_distribution_identity(require_controlled_import_cache=True)
    if sandbox_proof_fd is not None:
        emit_macos_sandbox_proof(sandbox_proof_fd, sandbox_proof_nonce)

    with os.fdopen(bootstrap_fd, "rb", buffering=0) as bootstrap_stream:
        bootstrap = decode_bootstrap(bootstrap_stream.readline(MAX_FRAME_BYTES + 1))
    tenant_key = bootstrap.tenant_key
    bot_principal_id = bootstrap.bot_principal_id
    ack_timeout = bootstrap.ack_timeout_seconds

    emitter = _FrameEmitter(event_fd, bootstrap.agent_id, bootstrap.generation)
    mailbox = IngressAckMailbox()
    stop_requested = threading.Event()
    admission = _ConnectionAdmission()

    import lark_oapi as lark
    from lark_channel.core.enum import LogLevel
    from lark_channel.event.callback.model.p2_card_action_trigger import (
        P2CardActionTriggerResponse,
    )
    from lark_channel.event.dispatcher_handler import EventDispatcherHandler
    from lark_channel.ws import Client as WSClient

    from src.autonomous.provisioning.lark_outbound import LarkEmployeeOutbound

    security = _strict_sdk_security_config()
    lark_client = (
        lark.Client.builder()
        .app_id(bootstrap.app_id)
        .app_secret(bootstrap.app_secret)
        .domain(_LARK_DOMAIN)
        .timeout(10.0)
        .source("employee-channel")
        .build()
    )
    bot_open_id = _fetch_employee_bot_open_id(
        lark_client,
        expected_app_id=bootstrap.app_id,
    )
    outbound = LarkEmployeeOutbound(lark_client)
    control_reader = threading.Thread(
        target=_read_low_level_control,
        args=(
            control_fd,
            bootstrap,
            mailbox,
            emitter,
            stop_requested,
            outbound,
            admission,
        ),
        name=f"employee-channel-control-{bootstrap.agent_id}-{bootstrap.generation}",
        daemon=True,
    )
    control_reader.start()

    def wait_for_parent(event: Any, *, kind: str) -> EmployeeIngressAck:
        deadline = time.monotonic() + float(ack_timeout)
        if stop_requested.is_set():
            raise EOFError("employee ingress admission is closed")
        connection_epoch, connection_id = admission.wait_snapshot(
            deadline=deadline
        )
        if stop_requested.is_set():
            raise EOFError("employee ingress admission is closed")
        metadata, payload, correlation = _normalize_sdk_ingress(
            event,
            kind=kind,
            agent_id=bootstrap.agent_id,
            app_id=bootstrap.app_id,
            generation=bootstrap.generation,
            connection_id=connection_id,
            tenant_key=tenant_key,
            bot_principal_id=bot_principal_id,
        )
        request_id = f"req_{uuid.uuid4().hex}"
        pending = admission.register(
            mailbox,
            expected_epoch=connection_epoch,
            request_id=request_id,
            agent_id=bootstrap.agent_id,
            app_id=bootstrap.app_id,
            generation=bootstrap.generation,
            connection_id=connection_id,
            semantic_digest=metadata.semantic_digest,
            envelope_id=metadata.envelope_id,
            dedup_key=metadata.dedup_key,
        )
        try:
            emitter.emit(
                FrameType.INGRESS,
                {
                    "request_id": request_id,
                    "app_id": bootstrap.app_id,
                    "connection_id": connection_id,
                    "metadata": metadata.to_dict(),
                    "payload": payload.to_dict(),
                    "action_correlation": correlation,
                },
                deadline=deadline,
            )
        except BaseException:
            mailbox.cancel(pending)
            raise
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            mailbox.wait(pending, timeout=0.0)
        return mailbox.wait(pending, timeout=remaining)

    def on_message(event: Any) -> None:
        wait_for_parent(event, kind="message")

    def on_card_action(event: Any) -> Any:
        wait_for_parent(event, kind="card")
        return P2CardActionTriggerResponse({})

    def on_bot_added(event: Any) -> None:
        wait_for_parent(event, kind="membership_added")

    def on_bot_deleted(event: Any) -> None:
        wait_for_parent(event, kind="membership_deleted")

    dispatcher = (
        EventDispatcherHandler.builder("", "", security=security)
        .register_p2_im_message_receive_v1(on_message)
        .register_p2_card_action_trigger(on_card_action)
        .register_p2_im_chat_member_bot_added_v1(on_bot_added)
        .register_p2_im_chat_member_bot_deleted_v1(on_bot_deleted)
        .build()
    )
    client = WSClient(
        app_id=bootstrap.app_id,
        app_secret=bootstrap.app_secret,
        log_level=LogLevel.ERROR,
        event_handler=dispatcher,
        domain=_LARK_DOMAIN,
        auto_reconnect=True,
        proxy_url=None,
        trust_env_proxy=False,
        handshake_timeout=10.0,
        security=security,
    )

    def on_reconnecting() -> None:
        admission.begin_reconnect(mailbox)
        _emit_nonblocking(emitter, "reconnecting", {})

    def on_reconnected() -> None:
        connection_epoch = admission.epoch
        connection_id = f"conn_{uuid.uuid4().hex}"
        _emit_nonblocking(emitter, "reconnected", {"connection_id": connection_id})
        _publish_observed_low_level_connection(
            client,
            bootstrap,
            bot_open_id,
            connection_id,
            emitter,
            admission,
            expected_epoch=connection_epoch,
        )

    client.on_reconnecting = on_reconnecting
    client.on_reconnected = on_reconnected
    readiness = threading.Thread(
        target=_observe_low_level_connection,
        args=(
            client,
            bootstrap,
            bot_open_id,
            emitter,
            stop_requested,
            admission,
        ),
        name=f"employee-channel-ready-{bootstrap.agent_id}-{bootstrap.generation}",
        daemon=True,
    )
    readiness.start()
    try:
        client.start()
    finally:
        stop_requested.set()
        mailbox.close()
        emitter.close()


def _strict_sdk_security_config() -> Any:
    from lark_channel.channel.config import SecurityConfig

    return SecurityConfig(
        mode="strict",
        allow_insecure_ws=False,
        allow_local_insecure_ws=False,
        max_ws_fragment_parts=8,
        max_ws_fragment_bytes=256 * 1024,
        max_concurrent_ws_handlers=1,
        resource_overflow_policy="drop",
    )


def _fetch_employee_bot_open_id(
    client: Any,
    *,
    expected_app_id: str,
) -> str:
    """Resolve this employee app's bot identity through official OpenAPI."""

    from lark_oapi.core.enum import AccessTokenType, HttpMethod
    from lark_oapi.core.model.base_request import BaseRequest
    from lark_oapi.core.model.base_response import BaseResponse

    request = (
        BaseRequest.builder()
        .http_method(HttpMethod.GET)
        .uri("/open-apis/bot/v3/info")
        .token_types({AccessTokenType.TENANT})
        .paths({})
        .body(None)
        .build()
    )
    try:
        response = client.request(request)
    except Exception as exc:
        raise WorkerSecurityError(
            f"employee bot identity lookup failed ({type(exc).__name__})"
        ) from None
    raw = response.raw if isinstance(response, BaseResponse) else None
    if (
        not isinstance(response, BaseResponse)
        or not response.success()
        or raw is None
        or not isinstance(raw.status_code, int)
        or not 200 <= raw.status_code < 300
        or not isinstance(raw.content, bytes)
    ):
        raise WorkerSecurityError("employee bot identity lookup was rejected")
    try:
        payload = json.loads(raw.content)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError):
        raise WorkerSecurityError("employee bot identity response is invalid") from None
    if not isinstance(payload, dict):
        raise WorkerSecurityError("employee bot identity response is invalid")
    bot = payload.get("bot")
    open_id = bot.get("open_id") if isinstance(bot, dict) else None
    observed_app_id = bot.get("app_id") if isinstance(bot, dict) else None
    if (
        payload.get("code") != 0
        or not isinstance(bot, dict)
        or not isinstance(open_id, str)
        or not open_id.startswith("ou_")
        or len(open_id) > 256
        or (
            observed_app_id is not None
            and observed_app_id != expected_app_id
        )
    ):
        raise WorkerSecurityError("employee bot identity response is invalid")
    return open_id


def _read_low_level_control(
    control_fd: int,
    bootstrap: Any,
    mailbox: IngressAckMailbox,
    emitter: _FrameEmitter,
    stop_requested: threading.Event,
    outbound: Any,
    admission: _ConnectionAdmission,
) -> None:
    inbound_sequence = 0
    try:
        with os.fdopen(control_fd, "rb", buffering=0) as control:
            while raw := control.readline(MAX_FRAME_BYTES + 1):
                try:
                    frame = decode_frame(raw)
                except Exception:
                    continue
                if (
                    frame.agent_id != bootstrap.agent_id
                    or frame.generation != bootstrap.generation
                    or frame.sequence <= inbound_sequence
                ):
                    continue
                inbound_sequence = frame.sequence
                if frame.frame_type is FrameType.INGRESS_ACK:
                    try:
                        ack = EmployeeIngressAck.from_dict(frame.payload["ack"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    mailbox.deliver(ack)
                elif frame.frame_type is FrameType.STOP:
                    stop_requested.set()
                    mailbox.close()
                    return
                elif frame.frame_type in {FrameType.SEND, FrameType.UPDATE_CARD}:
                    _handle_low_level_outbound(
                        frame,
                        bootstrap,
                        outbound,
                        admission,
                        emitter,
                    )
    finally:
        stop_requested.set()
        mailbox.close()


def _handle_low_level_outbound(
    frame: ChannelFrame,
    bootstrap: Any,
    outbound: Any,
    admission: _ConnectionAdmission,
    emitter: _FrameEmitter,
) -> None:
    """Execute one parent-authorized send with current Channel authority."""

    operation = "send" if frame.frame_type is FrameType.SEND else "update_card"
    request_id = frame.payload.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        emitter.emit(
            FrameType.HEALTH,
            {
                "operation": operation,
                "request_id": "",
                "success": False,
                "error_code": "invalid-outbound-request",
            },
        )
        return
    try:
        _epoch, connection_id = admission.wait_snapshot(
            deadline=time.monotonic() + 10.0
        )
        if frame.frame_type is FrameType.SEND:
            result = outbound.send(
                frame.payload.get("target"),
                frame.payload.get("message"),
                frame.payload.get("options"),
            )
        else:
            result = outbound.update_card(
                frame.payload.get("message_id"),
                frame.payload.get("card"),
            )
        if getattr(result, "success", None) is not True:
            raise RuntimeError("employee outbound operation was not acknowledged")
        message_id = getattr(result, "message_id", "")
        if not isinstance(message_id, str) or not message_id:
            raise RuntimeError("employee outbound receipt is invalid")
    except Exception as exc:
        emitter.emit(
            FrameType.HEALTH,
            {
                "operation": operation,
                "request_id": request_id,
                "success": False,
                "error_code": type(exc).__name__,
            },
        )
        return
    emitter.emit(
        FrameType.HEALTH,
        {
            "operation": operation,
            "request_id": request_id,
            "success": True,
            "app_id": bootstrap.app_id,
            "generation": bootstrap.generation,
            "connection_id": connection_id,
            "message_id": message_id,
        },
    )


def _observe_low_level_connection(
    client: Any,
    bootstrap: Any,
    bot_open_id: str,
    emitter: _FrameEmitter,
    stop_requested: threading.Event,
    admission: _ConnectionAdmission,
) -> None:
    connection_epoch = admission.epoch
    connection_id = f"conn_{uuid.uuid4().hex}"
    while not stop_requested.is_set():
        if _publish_observed_low_level_connection(
            client,
            bootstrap,
            bot_open_id,
            connection_id,
            emitter,
            admission,
            expected_epoch=connection_epoch,
        ):
            return
        if admission.epoch != connection_epoch:
            return
        stop_requested.wait(0.01)


def _publish_observed_low_level_connection(
    client: Any,
    bootstrap: Any,
    bot_open_id: str,
    connection_id: str,
    emitter: _FrameEmitter,
    admission: _ConnectionAdmission,
    *,
    expected_epoch: int,
) -> bool:
    connection = getattr(client, "_conn", None)
    sdk_connection_id = getattr(client, "_conn_id", "")
    service_id = getattr(client, "_service_id", "")
    connection_url = getattr(client, "_conn_url", "")
    if (
        connection is None
        or not isinstance(sdk_connection_id, str)
        or not sdk_connection_id
        or not isinstance(service_id, str)
        or not service_id
        or not isinstance(connection_url, str)
        or not connection_url.startswith("wss://")
    ):
        return False
    ready_payload = {
        "identity": {
            "app_id": bootstrap.app_id,
            "open_id": bot_open_id,
        },
        "connection_id": connection_id,
        "connection": {
            "observed": True,
            "sdk_connection_id": sdk_connection_id,
            "service_id": service_id,
            "secure": True,
        },
    }
    return admission.publish_observed(
        connection_id,
        expected_epoch=expected_epoch,
        emit_ready=lambda: emitter.emit(
            FrameType.READY,
            ready_payload,
        ),
    )


def _emit_nonblocking(
    emitter: _FrameEmitter,
    event_name: str,
    data: dict[str, Any],
) -> None:
    emitter.try_emit(
        FrameType.EVENT,
        {"event": event_name, "data": _json_safe(data)},
    )


def _normalize_message_mentions(message: Any) -> tuple[dict[str, str], ...]:
    """Keep canonical SDK mention identity fields in the encrypted blob."""

    raw_mentions = getattr(message, "mentions", None)
    if raw_mentions is None:
        return ()
    if not isinstance(raw_mentions, (list, tuple)) or len(raw_mentions) > 50:
        raise ValueError("employee ingress mentions are invalid")
    normalized: list[dict[str, str]] = []
    for mention in raw_mentions:
        identity = getattr(mention, "id", None)
        fields = {
            "key": getattr(mention, "key", None),
            "mentioned_type": getattr(mention, "mentioned_type", None),
            "open_id": getattr(identity, "open_id", None),
            "tenant_key": getattr(mention, "tenant_key", None),
        }
        if any(
            not isinstance(value, str) or not value or len(value) > 256
            for value in fields.values()
        ):
            raise ValueError("employee ingress mention identity is invalid")
        normalized.append(fields)
    return tuple(normalized)


def _normalize_sdk_ingress(
    event: Any,
    *,
    kind: str,
    agent_id: str,
    app_id: str,
    generation: int,
    connection_id: str,
    tenant_key: str,
    bot_principal_id: str,
) -> tuple[EmployeeIngressMetadata, EmployeeIngressPayload, str | None]:
    header = getattr(event, "header", None)
    body = getattr(event, "event", None)
    raw_event_id = getattr(header, "event_id", "")
    event_type = getattr(header, "event_type", "")
    header_tenant = getattr(header, "tenant_key", "")
    if header_tenant != tenant_key or getattr(header, "app_id", "") != app_id:
        raise ValueError("employee ingress header authority mismatch")
    received_at = _sdk_timestamp(getattr(header, "create_time", ""))
    correlation: str | None = None
    if kind == "message":
        message = getattr(body, "message", None)
        sender = getattr(body, "sender", None)
        sender_identity = getattr(sender, "sender_id", None)
        sender_id = getattr(sender_identity, "open_id", "")
        sender_union_id = getattr(sender_identity, "union_id", "")
        message_id = getattr(message, "message_id", "")
        chat_id = getattr(message, "chat_id", "")
        root_id = getattr(message, "root_id", "") or getattr(message, "parent_id", "") or ""
        content_raw = getattr(message, "content", "")
        try:
            content = json.loads(content_raw)
        except (TypeError, json.JSONDecodeError):
            content = content_raw
        mentions = _normalize_message_mentions(message)
        normalized_message = {
            "type": "message",
            "message_type": getattr(message, "message_type", ""),
            "chat_type": getattr(message, "chat_type", ""),
            "content": content,
            "sender_id": sender_id,
            "sender_union_id": sender_union_id,
            "sender_id_type": "open_id",
            "sender_type": getattr(sender, "sender_type", ""),
            "sender_tenant_key": getattr(sender, "tenant_key", ""),
            "feishu_thread_id": getattr(message, "thread_id", "") or "",
            # Remote coordinates stay inside the encrypted blob.  Public
            # ingress metadata contains only their one-way indexes.
            "remote_chat_id": chat_id,
            "remote_message_id": message_id,
            "remote_root_id": root_id,
        }
        if mentions:
            normalized_message["mentions"] = mentions
        parts = (
            normalized_message,
        )
        action_identity = ""
    elif kind == "card":
        context = getattr(body, "context", None)
        operator = getattr(body, "operator", None)
        sender_id = getattr(operator, "open_id", "")
        message_id = getattr(context, "open_message_id", "")
        chat_id = getattr(context, "open_chat_id", "")
        root_id = ""
        action_identity = ""
        parts = (
            {
                "type": "card_action",
                "sender_id": sender_id,
                "sender_id_type": "open_id",
                "sender_type": getattr(operator, "sender_type", "") or "",
                "sender_tenant_key": getattr(operator, "tenant_key", "") or "",
                "remote_chat_id": chat_id,
                "remote_message_id": message_id,
                "remote_root_id": root_id,
            },
        )
    elif kind in {"membership_added", "membership_deleted"}:
        sender_id = getattr(getattr(body, "operator_id", None), "open_id", "")
        message_id = raw_event_id
        chat_id = getattr(body, "chat_id", "")
        root_id = ""
        action_identity = kind
        parts = (
            {
                "type": "membership_event",
                "operation": (
                    "added" if kind == "membership_added" else "deleted"
                ),
                # The public metadata keeps only a one-way index.  The raw
                # Feishu chat identifier is required for the official API and
                # therefore travels only inside the encrypted ingress blob.
                "remote_chat_id": chat_id,
                "operator_tenant_key": getattr(body, "operator_tenant_key", ""),
                "external": getattr(body, "external", False),
            },
        )
    else:
        raise ValueError("unsupported employee ingress event kind")
    if not raw_event_id:
        raise ValueError("employee ingress requires trusted event identity")
    event_id = _canonical_ingress_id(raw_event_id, "evt_")
    message_id = _canonical_ingress_id(message_id, "om_")
    chat_id = _canonical_ingress_id(chat_id, "oc_")
    if root_id:
        root_id = _canonical_ingress_id(root_id, "om_")
    stable = json.dumps(
        [tenant_key, agent_id, event_id, message_id, event_type, action_identity],
        separators=(",", ":"),
    ).encode()
    import hashlib

    payload = EmployeeIngressPayload(
        schema_version=1,
        envelope_id="ing_" + hashlib.sha256(stable).hexdigest(),
        normalized_parts=parts,
        attachment_descriptors=(),
    )
    metadata = EmployeeIngressMetadata(
        schema_version=1,
        envelope_id=payload.envelope_id,
        tenant_key=tenant_key,
        agent_id=agent_id,
        bot_principal_id=bot_principal_id,
        app_id=app_id,
        channel_generation=generation,
        connection_id=connection_id,
        event_id=event_id,
        message_id=message_id,
        event_type=event_type,
        action_identity=action_identity,
        chat_id=chat_id,
        thread_root_message_id=root_id,
        sender_principal_id=sender_id,
        received_at=received_at,
        semantic_digest=payload.payload_sha256,
        payload_sha256=payload.payload_sha256,
        payload_size_bytes=payload.canonical_size_bytes,
        attachment_count=0,
        attachment_total_bytes=0,
    )
    return metadata, payload, correlation


def _sdk_timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value.isdigit():
        raise ValueError("invalid employee ingress timestamp")
    return datetime.fromtimestamp(int(value) / 1000, tz=UTC).isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")


def _canonical_ingress_id(value: Any, prefix: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("missing employee ingress identifier")
    import hashlib

    return prefix + hashlib.sha256(value.encode()).hexdigest()


def apply_process_hardening() -> None:
    """Disable core dumps, dumpability, and privilege acquisition."""
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "linux":
        prctl = libc.prctl
        prctl.argtypes = [
            ctypes.c_int,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
        ]
        prctl.restype = ctypes.c_int
        if prctl(4, 0, 0, 0, 0) != 0:  # PR_SET_DUMPABLE
            raise WorkerSecurityError("dumpability hardening failed")
        if prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
            raise WorkerSecurityError("privilege hardening failed")
        return
    if sys.platform == "darwin":
        ptrace = libc.ptrace
        ptrace.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        ptrace.restype = ctypes.c_int
        result = ptrace(31, 0, None, 0)  # PT_DENY_ATTACH
        if result != 0:
            raise WorkerSecurityError("debug attachment hardening failed")
        return
    raise WorkerSecurityError("unsupported worker platform")


def emit_macos_sandbox_proof(fd: int, nonce: str) -> None:
    """Prove the deny-default Seatbelt profile before reading credentials."""
    if (
        sys.platform != "darwin"
        or len(nonce) != 32
        or any(character not in "0123456789abcdef" for character in nonce)
    ):
        raise WorkerSecurityError("invalid macOS sandbox proof request")
    repository_root = Path(__file__).resolve().parents[3]
    canary = repository_root / "AGENTS.md"

    def readable(path: Path) -> bool:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            return False
        os.close(descriptor)
        return True

    try:
        descriptor = os.open(canary, os.O_RDONLY)
    except OSError as exc:
        canary_errno = int(exc.errno or 0)
    else:
        os.close(descriptor)
        canary_errno = 0
    proof = {
        "schema_version": 1,
        "nonce": nonce,
        "pid": os.getpid(),
        "source_readable": readable(Path(__file__).resolve()),
        "runtime_readable": readable(Path(sys.executable).resolve()),
        "repository_canary_errno": canary_errno,
    }
    raw = json.dumps(proof, separators=(",", ":")).encode()
    try:
        os.write(fd, raw)
    finally:
        os.close(fd)


@dataclass(slots=True)
class _EmitRequest:
    frame_type: FrameType
    payload: dict[str, Any]
    deadline: float
    required: bool
    completed: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


class _FrameEmitter:
    """Single-owner, deadline-bounded NDJSON writer for the child event pipe."""

    _DEFAULT_TIMEOUT_SECONDS = 10.0
    _QUEUE_CAPACITY = 128

    def __init__(self, fd: int, agent_id: str, generation: int) -> None:
        self._fd = fd
        self._agent_id = agent_id
        self._generation = generation
        self._sequence = 0
        self._requests: queue.Queue[_EmitRequest | None] = queue.Queue(
            maxsize=self._QUEUE_CAPACITY
        )
        self._admission_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._state_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._failed: BaseException | None = None
        self._closed = False
        os.set_blocking(fd, False)
        self._writer = threading.Thread(
            target=self._write_loop,
            name=f"employee-channel-emitter-{agent_id}-{generation}",
            daemon=True,
        )
        self._writer.start()

    def emit(
        self,
        frame_type: FrameType,
        payload: dict[str, Any],
        *,
        deadline: float | None = None,
    ) -> None:
        if deadline is None:
            deadline = time.monotonic() + self._DEFAULT_TIMEOUT_SECONDS
        request = _EmitRequest(frame_type, payload, deadline, True)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            error = TimeoutError("employee Channel IPC emit timed out")
            self._fail(error)
            raise error
        try:
            with self._admission_lock:
                self._raise_if_unavailable()
                self._requests.put(request, timeout=remaining)
        except queue.Full as exc:
            error = TimeoutError("employee Channel IPC emitter queue timed out")
            self._fail(error)
            raise error from exc
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not request.completed.wait(remaining):
            error = TimeoutError("employee Channel IPC emit timed out")
            self._fail(error)
            raise error
        if request.error is not None:
            raise request.error

    def try_emit(self, frame_type: FrameType, payload: dict[str, Any]) -> bool:
        """Queue a best-effort notification without blocking its SDK callback."""
        request = _EmitRequest(
            frame_type,
            payload,
            time.monotonic(),
            False,
        )
        try:
            with self._admission_lock:
                with self._state_lock:
                    if self._closed or self._failed is not None:
                        return False
                self._requests.put_nowait(request)
        except queue.Full:
            self._fail(BufferError("employee Channel IPC emitter queue full"))
            return False
        return True

    def close(self) -> None:
        with self._admission_lock:
            with self._state_lock:
                already_closed = self._closed
                self._closed = True
            if not already_closed:
                self._signal_stop()
        if self._writer is not threading.current_thread():
            self._writer.join()

    def _raise_if_unavailable(self) -> None:
        with self._state_lock:
            if self._failed is not None:
                raise EOFError("employee Channel IPC emitter failed") from self._failed
            if self._closed:
                raise EOFError("employee Channel IPC emitter closed")

    def _write_loop(self) -> None:
        try:
            while True:
                try:
                    request = self._requests.get(timeout=0.1)
                except queue.Empty:
                    with self._state_lock:
                        if self._closed:
                            return
                    continue
                if request is None:
                    return
                try:
                    self._write_request(request)
                except BaseException as exc:
                    request.error = exc
                    self._fail(exc)
                    return
                finally:
                    request.completed.set()
        finally:
            descriptor = self._fd
            self._fd = -1
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass

    def _write_request(self, request: _EmitRequest) -> None:
        self._raise_if_unavailable()
        self._sequence += 1
        raw = encode_frame(
            ChannelFrame(
                frame_type=request.frame_type,
                agent_id=self._agent_id,
                generation=self._generation,
                sequence=self._sequence,
                payload=_json_safe(request.payload),
            )
        )
        view = memoryview(raw)
        bytes_written = 0
        while view:
            self._raise_if_unavailable()
            try:
                written = os.write(self._fd, view)
            except BlockingIOError:
                written = 0
            if written > 0:
                bytes_written += written
                view = view[written:]
                continue
            if not request.required and bytes_written == 0:
                raise BlockingIOError(
                    "employee Channel IPC notification could not be emitted"
                )
            remaining = request.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("employee Channel IPC emit timed out")
            _, writable, _ = select.select([], [self._fd], [], remaining)
            if not writable:
                raise TimeoutError("employee Channel IPC emit timed out")

    def _fail(self, error: BaseException) -> None:
        with self._admission_lock:
            with self._state_lock:
                if self._failed is None:
                    self._failed = error
                self._closed = True
            while True:
                try:
                    queued = self._requests.get_nowait()
                except queue.Empty:
                    break
                if queued is None:
                    continue
                queued.error = EOFError("employee Channel IPC emitter failed")
                queued.completed.set()
            if self._writer is not threading.current_thread():
                self._signal_stop()

    def _signal_stop(self) -> None:
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            pass


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if hasattr(value, "__dict__"):
        return {
            str(key): _json_safe(child)
            for key, child in vars(value).items()
            if not str(key).startswith("_")
        }
    return {"type": type(value).__name__}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) not in {3, 5}:
        return 64
    try:
        fds = [int(value) for value in args[:3]]
    except ValueError:
        return 64
    proof_fd: int | None = None
    proof_nonce = ""
    if len(args) == 5:
        try:
            proof_fd = int(args[3])
        except ValueError:
            return 64
        proof_nonce = args[4]
    try:
        run_low_level_employee_channel(
            *fds,
            sandbox_proof_fd=proof_fd,
            sandbox_proof_nonce=proof_nonce,
        )
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
