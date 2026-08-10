"""Wire-level capability contract for the pinned employee Channel SDK."""

from __future__ import annotations

import datetime as dt
import json
import multiprocessing
import os
import queue
import ssl
import threading
import time
from contextlib import AbstractContextManager
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from websockets.sync.server import Server, serve

pytestmark = pytest.mark.slow

_PROCESS_WAIT_SECONDS = 5.0
_ACK_BUDGET_SECONDS = 1.5
_NO_EARLY_RESPONSE_SECONDS = _ACK_BUDGET_SECONDS


def _strict_security_config():
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


def _run_sdk_client(
    domain: str,
    ca_file: str,
    event_kind: str,
    callback_entered: Any,
    release_callback: Any,
    callback_wait_seconds: float = _PROCESS_WAIT_SECONDS,
    request_reconnect: Any | None = None,
    callback_finished: Any | None = None,
    callback_behavior: str = "barrier",
    second_callback_entered: Any | None = None,
    auto_reconnect: bool = False,
    log_file: str | None = None,
    secret_sentinel: str | None = None,
) -> None:
    """Run the real pinned SDK in a disposable fresh interpreter."""
    if log_file is not None:
        log_fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.dup2(log_fd, 1)
            os.dup2(log_fd, 2)
        finally:
            os.close(log_fd)
    os.environ["SSL_CERT_FILE"] = ca_file
    # A poisoned environment proves the explicit direct-connect policy wins.
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:1"
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:1"
    os.environ["ALL_PROXY"] = "http://127.0.0.1:1"

    from src.autonomous.ingress.sdk_capability import (
        collect_sdk_distribution_identity,
        prepare_controlled_sdk_import_cache,
    )

    prepare_controlled_sdk_import_cache(
        Path(ca_file).parent / f"sdk-bytecode-{os.getpid()}"
    )
    collect_sdk_distribution_identity(require_controlled_import_cache=True)

    from lark_channel.core.enum import LogLevel
    from lark_channel.event.callback.model.p2_card_action_trigger import (
        P2CardActionTriggerResponse,
    )
    from lark_channel.event.dispatcher_handler import EventDispatcherHandler
    from lark_channel.ws import Client as WSClient

    security = _strict_security_config()

    callback_count = 0

    def hold_callback(_event: Any) -> Any:
        nonlocal callback_count
        callback_count += 1
        if callback_count == 1:
            callback_entered.set()
        elif second_callback_entered is not None:
            second_callback_entered.set()
        try:
            if callback_behavior == "parent_close":
                raise EOFError("employee ingress parent unavailable")
            if callback_behavior == "exception":
                raise RuntimeError("employee ingress callback failed")
            if not release_callback.wait(callback_wait_seconds):
                raise TimeoutError("employee ingress anchor deadline exceeded")
            if event_kind == "card":
                return P2CardActionTriggerResponse({})
            return None
        finally:
            if callback_finished is not None:
                callback_finished.set()

    builder = EventDispatcherHandler.builder("", "", security=security)
    if event_kind == "message":
        builder = builder.register_p2_im_message_receive_v1(hold_callback)
    else:
        builder = builder.register_p2_card_action_trigger(hold_callback)
    dispatcher = builder.build()

    client = WSClient(
        app_id="cli_capability",
        app_secret=secret_sentinel or "test-only-secret",
        log_level=LogLevel.ERROR,
        event_handler=dispatcher,
        domain=domain,
        auto_reconnect=auto_reconnect,
        proxy_url=None,
        trust_env_proxy=False,
        handshake_timeout=2.0,
        security=security,
    )
    if request_reconnect is not None:

        def reconnect_when_requested() -> None:
            if request_reconnect.wait(_PROCESS_WAIT_SECONDS):
                client.request_reconnect()

        threading.Thread(
            target=reconnect_when_requested,
            name="employee-sdk-capability-reconnect",
            daemon=True,
        ).start()
    client.start()


class _EndpointHandler(BaseHTTPRequestHandler):
    endpoint_url = ""
    bot_open_id = "ou_capability_bot"

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - stdlib HTTP hook
        if self.path == "/open-apis/auth/v3/tenant_access_token/internal":
            self._send_json(
                {
                    "code": 0,
                    "msg": "ok",
                    "tenant_access_token": "t-capability",
                    "expire": 7200,
                }
            )
            return
        if self.path != "/callback/ws/endpoint":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send_json(
            {
                "code": 0,
                "msg": "ok",
                "data": {
                    "URL": self.endpoint_url,
                    "ClientConfig": {
                        "ReconnectCount": 1,
                        "ReconnectInterval": 1,
                        "ReconnectNonce": 1,
                        "PingInterval": 1,
                    },
                },
            }
        )

    def do_GET(self) -> None:  # noqa: N802 - stdlib HTTP hook
        if self.path != "/open-apis/bot/v3/info":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send_json(
            {
                "code": 0,
                "msg": "ok",
                "bot": {
                    "app_id": "cli_capability",
                    "open_id": self.bot_open_id,
                    "app_name": "Capability Employee",
                },
            }
        )

    def log_message(self, _format: str, *_args: Any) -> None:
        return


class _WireHarness(AbstractContextManager["_WireHarness"]):
    def __init__(
        self,
        tmp_path: Path,
        payload: dict[str, Any],
        *,
        wire_type: str = "event",
        send_event_once: bool = False,
        declared_fragment_sum: int = 1,
        send_two_events: bool = False,
        close_first_connection: bool = False,
        idle_first_connection: bool = False,
        keep_open_after_response: bool = False,
        endpoint_query_sentinel: str | None = None,
        fragment_byte_overflow: bool = False,
        recover_after_fragment_drop: bool = False,
    ) -> None:
        self._tmp_path = tmp_path
        self._payload = payload
        self._wire_type = wire_type
        self._send_event_once = send_event_once
        self._declared_fragment_sum = declared_fragment_sum
        self._send_two_events = send_two_events
        self._close_first_connection = close_first_connection
        self._idle_first_connection = idle_first_connection
        self._keep_open_after_response = keep_open_after_response
        self._endpoint_query_sentinel = endpoint_query_sentinel
        self._fragment_byte_overflow = fragment_byte_overflow
        self._recover_after_fragment_drop = recover_after_fragment_drop
        self.response_received = threading.Event()
        self.second_response_received = threading.Event()
        self.read_waiting = threading.Event()
        self.second_connection = threading.Event()
        self.first_connection_ready = threading.Event()
        self.control_ping_received = threading.Event()
        self.ping_after_response = threading.Event()
        self.release_recovery_frame = threading.Event()
        self.connection_count = 0
        self.responses: queue.Queue[bytes | BaseException] = queue.Queue()
        self._wss_server: Server | None = None
        self._http_server: ThreadingHTTPServer | None = None
        self._wss_thread: threading.Thread | None = None
        self._http_thread: threading.Thread | None = None
        self._connection_lock = threading.Lock()
        self.ca_file = self._write_test_certificate()
        self.domain = ""
        self.bot_open_id = _EndpointHandler.bot_open_id

    def __enter__(self) -> _WireHarness:
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(str(self.ca_file), str(self._tmp_path / "server-key.pem"))
        self._wss_server = serve(self._handle_ws, "127.0.0.1", 0, ssl=tls)
        wss_port = self._wss_server.socket.getsockname()[1]
        self._wss_thread = threading.Thread(
            target=self._wss_server.serve_forever,
            name="employee-sdk-capability-wss",
            daemon=True,
        )
        self._wss_thread.start()

        sentinel_query = (
            f"&sentinel={self._endpoint_query_sentinel}"
            if self._endpoint_query_sentinel is not None
            else ""
        )
        endpoint_type = type(
            "BoundEndpointHandler",
            (_EndpointHandler,),
            {
                "endpoint_url": (
                    f"wss://localhost:{wss_port}/callback"
                    f"?device_id=dev-capability&service_id=1{sentinel_query}"
                )
            },
        )
        self._http_server = ThreadingHTTPServer(("127.0.0.1", 0), endpoint_type)
        http_port = self._http_server.server_address[1]
        self.domain = f"http://127.0.0.1:{http_port}"
        self._http_thread = threading.Thread(
            target=self._http_server.serve_forever,
            name="employee-sdk-capability-http",
            daemon=True,
        )
        self._http_thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server.server_close()
        if self._wss_server is not None:
            self._wss_server.shutdown()
        if self._http_thread is not None:
            self._http_thread.join(timeout=2)
        if self._wss_thread is not None:
            self._wss_thread.join(timeout=2)

    def _handle_ws(self, connection: Any) -> None:
        with self._connection_lock:
            self.connection_count += 1
            connection_number = self.connection_count
        if connection_number > 1:
            self.second_connection.set()
            if self._send_event_once:
                return
        else:
            self.first_connection_ready.set()
        if connection_number == 1 and self._close_first_connection:
            return
        if connection_number == 1 and self._idle_first_connection:
            try:
                while True:
                    connection.recv(timeout=_PROCESS_WAIT_SECONDS)
            except BaseException:
                return

        if self._fragment_byte_overflow:
            overflow_payload = json.loads(json.dumps(self._payload))
            overflow_payload["event"]["message"]["content"] = "x" * (300 * 1024)
            encoded = json.dumps(overflow_payload, separators=(",", ":")).encode()
            split = len(encoded) // 2
            connection.send(
                _event_frame(
                    encoded[:split],
                    wire_type=self._wire_type,
                    message_id="msg-capability",
                    declared_sum=2,
                    sequence=0,
                )
            )
            connection.send(
                _event_frame(
                    encoded[split:],
                    wire_type=self._wire_type,
                    message_id="msg-capability",
                    declared_sum=2,
                    sequence=1,
                )
            )
        else:
            connection.send(
                _event_frame(
                    self._payload,
                    wire_type=self._wire_type,
                    message_id="msg-capability",
                    declared_sum=self._declared_fragment_sum,
                )
            )
        self.read_waiting.set()
        if self._recover_after_fragment_drop:
            if not self.release_recovery_frame.wait(_PROCESS_WAIT_SECONDS):
                return
            connection.send(
                _event_frame(
                    self._payload,
                    wire_type=self._wire_type,
                    message_id="msg-capability-2",
                )
            )
        if self._send_two_events:
            second_payload = json.loads(json.dumps(self._payload))
            second_payload["header"]["event_id"] = "evt-second-capability"
            connection.send(
                _event_frame(
                    second_payload,
                    wire_type=self._wire_type,
                    message_id="msg-capability-2",
                )
            )
        from lark_channel.ws.enum import FrameType
        from lark_channel.ws.pb.pbbp2_pb2 import Frame

        responses: set[str] = set()
        while True:
            try:
                raw = connection.recv(timeout=_PROCESS_WAIT_SECONDS)
            except BaseException as exc:  # captured for the asserting test thread
                self.responses.put(exc)
                return
            if not isinstance(raw, bytes):
                self.responses.put(TypeError("SDK response frame must be bytes"))
                return
            frame = Frame()
            frame.ParseFromString(raw)
            if frame.method == FrameType.CONTROL.value:
                self.control_ping_received.set()
                if self.response_received.is_set():
                    self.ping_after_response.set()
                    if self._keep_open_after_response:
                        return
                continue
            if frame.method != FrameType.DATA.value:
                continue
            headers = {header.key: header.value for header in frame.headers}
            message_id = headers.get("message_id", "")
            if message_id not in {"msg-capability", "msg-capability-2"}:
                continue
            responses.add(message_id)
            self.responses.put(raw)
            if message_id == "msg-capability":
                self.response_received.set()
            else:
                self.second_response_received.set()
            if self._send_two_events and len(responses) < 2:
                continue
            if not self._keep_open_after_response:
                return

    def response_frame(self) -> Any:
        raw = self.responses.get(timeout=_PROCESS_WAIT_SECONDS)
        if isinstance(raw, BaseException):
            raise raw
        from lark_channel.ws.pb.pbbp2_pb2 import Frame

        frame = Frame()
        frame.ParseFromString(raw)
        return frame

    def _write_test_certificate(self) -> Path:
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        now = dt.datetime.now(dt.UTC)
        subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=1))
            .not_valid_after(now + dt.timedelta(hours=1))
            .add_extension(
                x509.SubjectAlternativeName([x509.DNSName("localhost")]),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )
        cert_path = self._tmp_path / "server-cert.pem"
        key_path = self._tmp_path / "server-key.pem"
        cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        key_path.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        return cert_path


def _event_frame(
    payload: dict[str, Any] | bytes,
    *,
    wire_type: str = "event",
    message_id: str = "msg-capability",
    declared_sum: int = 1,
    sequence: int = 0,
) -> bytes:
    from lark_channel.ws.enum import FrameType
    from lark_channel.ws.pb.pbbp2_pb2 import Frame

    frame = Frame()
    frame.method = FrameType.DATA.value
    frame.service = 1
    frame.SeqID = 1
    frame.LogID = 1
    for key, value in (
        ("type", wire_type),
        ("message_id", message_id),
        ("trace_id", "trace-capability"),
        ("sum", str(declared_sum)),
        ("seq", str(sequence)),
    ):
        header = frame.headers.add()
        header.key = key
        header.value = value
    frame.payload = (
        payload
        if isinstance(payload, bytes)
        else json.dumps(payload, separators=(",", ":")).encode()
    )
    return frame.SerializeToString()


def _message_payload() -> dict[str, Any]:
    now_ms = str(int(time.time() * 1000))
    return {
        "schema": "2.0",
        "header": {
            "event_id": "evt-message-capability",
            "event_type": "im.message.receive_v1",
            "create_time": now_ms,
            "token": "token-capability",
            "app_id": "cli_capability",
            "tenant_key": "tenant-capability",
        },
        "event": {
            "sender": {
                "sender_id": {"open_id": "ou_sender"},
                "sender_type": "user",
                "tenant_key": "tenant-capability",
            },
            "message": {
                "message_id": "om_capability",
                "root_id": "",
                "parent_id": "",
                "create_time": now_ms,
                "chat_id": "oc_capability",
                "chat_type": "p2p",
                "message_type": "text",
                "content": "{\"text\":\"hello\"}",
            },
        },
    }


def _card_payload() -> dict[str, Any]:
    now_ms = str(int(time.time() * 1000))
    return {
        "schema": "2.0",
        "header": {
            "event_id": "evt-card-capability",
            "event_type": "card.action.trigger",
            "create_time": now_ms,
            "token": "token-capability",
            "app_id": "cli_capability",
            "tenant_key": "tenant-capability",
        },
        "event": {
            "operator": {"tenant_key": "tenant-capability", "open_id": "ou_sender"},
            "token": "card-token-capability",
            "action": {
                "tag": "button",
                "value": {"correlation_id": "server-issued-capability"},
            },
            "host": "im_message",
            "delivery_type": "ws",
            "context": {
                "open_message_id": "om_card_capability",
                "open_chat_id": "oc_capability",
            },
        },
    }


def _stop_process(process: multiprocessing.Process, release: Any | None = None) -> None:
    if release is not None:
        release.set()
    process.terminate()
    process.join(timeout=2)
    if process.is_alive():
        process.kill()
        process.join(timeout=2)
    assert not process.is_alive()


def _start_client(
    context: Any,
    harness: _WireHarness,
    event_kind: str,
    callback_entered: Any,
    release_callback: Any,
    *extra: Any,
) -> multiprocessing.Process:
    process = context.Process(
        target=_run_sdk_client,
        args=(
            harness.domain,
            str(harness.ca_file),
            event_kind,
            callback_entered,
            release_callback,
            *extra,
        ),
    )
    process.start()
    return process


def _assert_low_level_waits_for_callback(tmp_path: Path, event_kind: str) -> None:
    payload = _message_payload() if event_kind == "message" else _card_payload()
    context = multiprocessing.get_context("spawn")
    callback_entered = context.Event()
    release_callback = context.Event()

    with _WireHarness(tmp_path, payload) as harness:
        process = _start_client(
            context,
            harness,
            event_kind,
            callback_entered,
            release_callback,
        )
        try:
            assert harness.read_waiting.wait(_PROCESS_WAIT_SECONDS)
            assert callback_entered.wait(_PROCESS_WAIT_SECONDS)
            assert not harness.response_received.wait(_NO_EARLY_RESPONSE_SECONDS)
            release_callback.set()
            assert harness.response_received.wait(_PROCESS_WAIT_SECONDS)
            response = harness.response_frame()
            assert json.loads(response.payload)["code"] == HTTPStatus.OK
            assert response.SeqID == 1
            assert response.LogID == 1
            assert response.service == 1
            assert response.method == 1
            headers = {header.key: header.value for header in response.headers}
            assert headers["message_id"] == "msg-capability"
            assert headers["trace_id"] == "trace-capability"
            assert headers["biz_rt"]
            assert harness.connection_count == 1
        finally:
            _stop_process(process, release_callback)


@pytest.mark.integration
def test_message_wire_response_waits_for_parent_anchor(tmp_path: Path) -> None:
    _assert_low_level_waits_for_callback(tmp_path, "message")


@pytest.mark.integration
def test_card_action_wire_response_waits_for_parent_anchor(tmp_path: Path) -> None:
    _assert_low_level_waits_for_callback(tmp_path, "card")


def _assert_callback_failure(
    tmp_path: Path,
    event_kind: str,
    *,
    callback_wait_seconds: float = _PROCESS_WAIT_SECONDS,
    callback_behavior: str = "barrier",
) -> float:
    context = multiprocessing.get_context("spawn")
    callback_entered = context.Event()
    release_callback = context.Event()
    payload = _message_payload() if event_kind == "message" else _card_payload()

    with _WireHarness(tmp_path, payload) as harness:
        process = _start_client(
            context,
            harness,
            event_kind,
            callback_entered,
            release_callback,
            callback_wait_seconds,
            None,
            None,
            callback_behavior,
        )
        try:
            assert callback_entered.wait(_PROCESS_WAIT_SECONDS)
            started = time.monotonic()
            assert harness.response_received.wait(_PROCESS_WAIT_SECONDS)
            elapsed = time.monotonic() - started
            assert json.loads(harness.response_frame().payload)["code"] == 500
            assert harness.connection_count == 1
            return elapsed
        finally:
            _stop_process(process)


@pytest.mark.integration
def test_message_callback_timeout_at_ack_deadline_is_wire_500(tmp_path: Path) -> None:
    elapsed = _assert_callback_failure(
        tmp_path,
        "message",
        callback_wait_seconds=_ACK_BUDGET_SECONDS,
    )
    assert elapsed >= _ACK_BUDGET_SECONDS * 0.9
    assert elapsed < _ACK_BUDGET_SECONDS + 0.5


@pytest.mark.integration
def test_card_callback_timeout_at_ack_deadline_is_wire_500(tmp_path: Path) -> None:
    elapsed = _assert_callback_failure(
        tmp_path,
        "card",
        callback_wait_seconds=_ACK_BUDGET_SECONDS,
    )
    assert elapsed >= _ACK_BUDGET_SECONDS * 0.9
    assert elapsed < _ACK_BUDGET_SECONDS + 0.5


@pytest.mark.integration
def test_parent_close_is_wire_500(tmp_path: Path) -> None:
    _assert_callback_failure(tmp_path, "message", callback_behavior="parent_close")


@pytest.mark.integration
def test_callback_exception_is_wire_500(tmp_path: Path) -> None:
    _assert_callback_failure(tmp_path, "message", callback_behavior="exception")


@pytest.mark.integration
def test_raw_card_wire_type_is_not_card_action_capability(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    callback_entered = context.Event()
    release_callback = context.Event()

    with _WireHarness(tmp_path, _card_payload(), wire_type="card") as harness:
        process = _start_client(
            context,
            harness,
            "card",
            callback_entered,
            release_callback,
        )
        try:
            assert harness.read_waiting.wait(_PROCESS_WAIT_SECONDS)
            assert not callback_entered.wait(_ACK_BUDGET_SECONDS)
            assert not harness.response_received.is_set()
            assert harness.connection_count == 1
        finally:
            _stop_process(process)


@pytest.mark.integration
def test_reconnect_requested_during_callback_is_bounded(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    callback_entered = context.Event()
    release_callback = context.Event()
    request_reconnect = context.Event()

    with _WireHarness(tmp_path, _message_payload(), send_event_once=True) as harness:
        process = _start_client(
            context,
            harness,
            "message",
            callback_entered,
            release_callback,
            _PROCESS_WAIT_SECONDS,
            request_reconnect,
        )
        try:
            assert callback_entered.wait(_PROCESS_WAIT_SECONDS)
            request_reconnect.set()
            assert not harness.second_connection.wait(_NO_EARLY_RESPONSE_SECONDS)
            release_callback.set()
            assert harness.response_received.wait(_PROCESS_WAIT_SECONDS)
            assert json.loads(harness.response_frame().payload)["code"] == 200
            assert harness.second_connection.wait(_PROCESS_WAIT_SECONDS)
            assert harness.connection_count == 2
        finally:
            _stop_process(process, release_callback)


@pytest.mark.integration
def test_idle_reconnect_is_bounded(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    callback_entered = context.Event()
    release_callback = context.Event()
    request_reconnect = context.Event()

    with _WireHarness(tmp_path, _message_payload(), idle_first_connection=True) as harness:
        process = _start_client(
            context,
            harness,
            "message",
            callback_entered,
            release_callback,
            _PROCESS_WAIT_SECONDS,
            request_reconnect,
        )
        try:
            assert harness.first_connection_ready.wait(_PROCESS_WAIT_SECONDS)
            assert not callback_entered.is_set()
            request_reconnect.set()
            assert harness.second_connection.wait(_PROCESS_WAIT_SECONDS)
            assert callback_entered.wait(_PROCESS_WAIT_SECONDS)
            release_callback.set()
            assert harness.response_received.wait(_PROCESS_WAIT_SECONDS)
            assert json.loads(harness.response_frame().payload)["code"] == 200
            assert harness.connection_count == 2
        finally:
            _stop_process(process, release_callback)


@pytest.mark.integration
def test_blocked_callback_worker_termination_is_bounded(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    callback_entered = context.Event()
    release_callback = context.Event()

    with _WireHarness(tmp_path, _message_payload()) as harness:
        process = _start_client(
            context,
            harness,
            "message",
            callback_entered,
            release_callback,
            60.0,
        )
        assert callback_entered.wait(_PROCESS_WAIT_SECONDS)
        started = time.monotonic()
        _stop_process(process)
        assert time.monotonic() - started < 2.5
        assert not harness.response_received.is_set()


@pytest.mark.integration
def test_control_ping_resumes_after_callback(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    callback_entered = context.Event()
    release_callback = context.Event()

    with _WireHarness(
        tmp_path,
        _message_payload(),
        keep_open_after_response=True,
    ) as harness:
        process = _start_client(
            context,
            harness,
            "message",
            callback_entered,
            release_callback,
        )
        try:
            assert callback_entered.wait(_PROCESS_WAIT_SECONDS)
            assert harness.control_ping_received.wait(_PROCESS_WAIT_SECONDS)
            release_callback.set()
            assert harness.response_received.wait(_PROCESS_WAIT_SECONDS)
            assert harness.ping_after_response.wait(_PROCESS_WAIT_SECONDS)
        finally:
            _stop_process(process, release_callback)


@pytest.mark.integration
def test_fragment_overflow_is_dropped_before_callback(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    callback_entered = context.Event()
    release_callback = context.Event()

    with _WireHarness(
        tmp_path,
        _message_payload(),
        declared_fragment_sum=9,
        recover_after_fragment_drop=True,
    ) as harness:
        process = _start_client(
            context,
            harness,
            "message",
            callback_entered,
            release_callback,
        )
        try:
            assert harness.read_waiting.wait(_PROCESS_WAIT_SECONDS)
            assert not callback_entered.wait(_ACK_BUDGET_SECONDS)
            assert not harness.response_received.is_set()
            harness.release_recovery_frame.set()
            assert callback_entered.wait(_PROCESS_WAIT_SECONDS)
            release_callback.set()
            assert harness.second_response_received.wait(_PROCESS_WAIT_SECONDS)
            response = harness.response_frame()
            assert json.loads(response.payload)["code"] == 200
            headers = {header.key: header.value for header in response.headers}
            assert headers["message_id"] == "msg-capability-2"
        finally:
            _stop_process(process)


@pytest.mark.integration
def test_fragment_byte_overflow_is_dropped_before_callback(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    callback_entered = context.Event()
    release_callback = context.Event()

    with _WireHarness(
        tmp_path,
        _message_payload(),
        fragment_byte_overflow=True,
        recover_after_fragment_drop=True,
    ) as harness:
        process = _start_client(
            context,
            harness,
            "message",
            callback_entered,
            release_callback,
        )
        try:
            assert harness.read_waiting.wait(_PROCESS_WAIT_SECONDS)
            assert not callback_entered.wait(_ACK_BUDGET_SECONDS)
            assert not harness.response_received.is_set()
            harness.release_recovery_frame.set()
            assert callback_entered.wait(_PROCESS_WAIT_SECONDS)
            release_callback.set()
            assert harness.second_response_received.wait(_PROCESS_WAIT_SECONDS)
            response = harness.response_frame()
            assert json.loads(response.payload)["code"] == 200
            headers = {header.key: header.value for header in response.headers}
            assert headers["message_id"] == "msg-capability-2"
        finally:
            _stop_process(process)


@pytest.mark.integration
def test_concurrency_cap_holds_second_callback(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    callback_entered = context.Event()
    second_callback_entered = context.Event()
    release_callback = context.Event()

    with _WireHarness(tmp_path, _message_payload(), send_two_events=True) as harness:
        process = _start_client(
            context,
            harness,
            "message",
            callback_entered,
            release_callback,
            _PROCESS_WAIT_SECONDS,
            None,
            None,
            "barrier",
            second_callback_entered,
        )
        try:
            assert callback_entered.wait(_PROCESS_WAIT_SECONDS)
            assert not second_callback_entered.wait(_NO_EARLY_RESPONSE_SECONDS)
            release_callback.set()
            assert second_callback_entered.wait(_PROCESS_WAIT_SECONDS)
            assert harness.response_received.wait(_PROCESS_WAIT_SECONDS)
            assert harness.second_response_received.wait(_PROCESS_WAIT_SECONDS)
        finally:
            _stop_process(process, release_callback)


@pytest.mark.integration
def test_single_frame_payload_limit_requires_parent_gate(tmp_path: Path) -> None:
    """The SDK fragment byte cap does not cover one large frame."""
    context = multiprocessing.get_context("spawn")
    callback_entered = context.Event()
    release_callback = context.Event()
    payload = _message_payload()
    payload["event"]["message"]["content"] = "x" * (256 * 1024 + 1)

    with _WireHarness(tmp_path, payload) as harness:
        process = _start_client(
            context,
            harness,
            "message",
            callback_entered,
            release_callback,
        )
        try:
            assert callback_entered.wait(_PROCESS_WAIT_SECONDS)
            release_callback.set()
            assert harness.response_received.wait(_PROCESS_WAIT_SECONDS)
            assert json.loads(harness.response_frame().payload)["code"] == 200
        finally:
            _stop_process(process, release_callback)


@pytest.mark.integration
def test_sensitive_sentinels_are_absent_from_worker_logs(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    callback_entered = context.Event()
    release_callback = context.Event()
    sentinel = "employee-capability-secret-sentinel"
    log_file = tmp_path / "worker.log"
    payload = _message_payload()
    payload["event"]["message"]["content"] = sentinel

    with _WireHarness(
        tmp_path,
        payload,
        endpoint_query_sentinel=sentinel,
    ) as harness:
        process = _start_client(
            context,
            harness,
            "message",
            callback_entered,
            release_callback,
            _PROCESS_WAIT_SECONDS,
            None,
            None,
            "exception",
            None,
            False,
            str(log_file),
            sentinel,
        )
        try:
            assert callback_entered.wait(_PROCESS_WAIT_SECONDS)
            assert harness.response_received.wait(_PROCESS_WAIT_SECONDS)
            assert json.loads(harness.response_frame().payload)["code"] == 500
        finally:
            _stop_process(process)

    assert sentinel not in log_file.read_text(encoding="utf-8")
