"""Card delivery adapter backed by the official Lark Channel SDK.

The card pipeline is synchronous by design, while ``FeishuChannel`` exposes
async outbound APIs.  This adapter owns that boundary and keeps presentation
state in CardSession/CardDelivery; it does not use the SDK's markdown stream
controller because GhostAP cards contain rich task, tool, progress and footer
elements in addition to the active markdown element.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable, Coroutine
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any

from lark_channel import OutboundCard
from lark_channel.api.cardkit.v1.model.card import Card
from lark_channel.api.cardkit.v1.model.update_card_request import UpdateCardRequest
from lark_channel.api.cardkit.v1.model.update_card_request_body import (
    UpdateCardRequestBody,
)

from src.card.delivery.types import SequenceConflictError, TransportError
from src.config import get_settings

logger = logging.getLogger(__name__)

_DEFAULT_API_TIMEOUT_SECONDS = 35.0
_UPDATE_NAMESPACE = uuid.UUID("7a988460-9101-492a-bdb6-98b8d27f998d")
_RAW_CODE_RE = re.compile(r"(?:raw_code|ErrCode|code)['\"\s:=]+(\d+)", re.IGNORECASE)


def _sanitize_text(value: str) -> str:
    try:
        value.encode("utf-8")
        return value
    except UnicodeEncodeError:
        return value.encode("utf-8", errors="surrogatepass").decode(
            "utf-8", errors="replace"
        )


def _sanitize_card(card_json: dict) -> dict:
    encoded = _sanitize_text(json.dumps(card_json, ensure_ascii=False))
    return json.loads(encoded)


def _raw_error_code(value: object) -> int:
    error = getattr(value, "error", None)
    for candidate in (
        getattr(error, "raw_code", None),
        getattr(value, "raw_code", None),
        getattr(value, "code", None),
    ):
        if isinstance(candidate, int):
            return candidate
    raw = getattr(value, "raw", None)
    if isinstance(raw, dict) and isinstance(raw.get("code"), int):
        return raw["code"]
    match = _RAW_CODE_RE.search(str(value))
    return int(match.group(1)) if match else 0


def _error_message(value: object, operation: str) -> str:
    error = getattr(value, "error", None)
    hint = getattr(error, "hint", None)
    if isinstance(hint, str) and hint:
        return f"{operation} failed: {hint}"
    message = getattr(value, "msg", None) or str(value)
    return f"{operation} failed: {message}"


def _raise_transport_error(value: object, operation: str, *, sequence: int = 0) -> None:
    code = _raw_error_code(value)
    message = _error_message(value, operation)
    if code == 300317 or (code == 230099 and "sequence" in message.lower()):
        raise SequenceConflictError(next_floor=sequence + 1)
    raise TransportError(message, code=code)


class LarkChannelCardAPIClient:
    """Implement ``CardAPIClient`` with ``lark-channel-sdk`` APIs."""

    allow_streaming_fallback = False
    target_aware_streaming_create = True

    def __init__(
        self,
        channel: Any,
        *,
        timeout_seconds: float | None = None,
        preallocate_cards: bool = False,
        default_reply_in_thread: bool | None = None,
        outbound_audit: Callable[[str, str, str], None] | None = None,
        outbound_audit_failure: Callable[[Exception], None] | None = None,
        tenant_key_resolver: Callable[[], str] | None = None,
        outbound_target_aliases: Callable[[str], tuple[str, ...]] | None = None,
    ) -> None:
        self._channel = channel
        self._timeout_seconds = timeout_seconds
        self.preallocate_cards = preallocate_cards
        self._default_reply_in_thread = default_reply_in_thread
        self._outbound_audit = outbound_audit
        self._outbound_audit_failure = outbound_audit_failure
        self._tenant_key_resolver = tenant_key_resolver
        self._outbound_target_aliases = outbound_target_aliases
        self._audit_aliases_by_target: OrderedDict[str, tuple[str, ...]] = OrderedDict()
        self._audit_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

    def _api_timeout_seconds(self) -> float:
        if self._timeout_seconds is not None:
            return float(self._timeout_seconds)
        try:
            return float(get_settings().card.delivery_api_timeout)
        except Exception:
            return _DEFAULT_API_TIMEOUT_SECONDS

    def _run(
        self,
        operation: str,
        awaitable: Coroutine[Any, Any, Any],
        *,
        sequence: int = 0,
    ) -> Any:
        timeout = self._api_timeout_seconds()

        async def _capture_result() -> tuple[bool, Any]:
            try:
                return True, await awaitable
            except Exception as exc:
                return False, exc

        captured = _capture_result()
        try:
            future = self._channel.schedule(captured)
        except Exception as exc:
            captured.close()
            awaitable.close()
            _raise_transport_error(exc, operation, sequence=sequence)
            raise AssertionError("unreachable") from exc
        try:
            succeeded, value = future.result(timeout=timeout)
        except FutureTimeoutError as exc:
            future.cancel()
            awaitable.close()
            raise TimeoutError(
                f"Channel SDK {operation} timed out after {timeout:.1f}s"
            ) from exc
        except (SequenceConflictError, TransportError):
            raise
        except Exception as exc:
            _raise_transport_error(exc, operation, sequence=sequence)
            raise AssertionError("unreachable") from exc
        if succeeded:
            return value
        if isinstance(value, (SequenceConflictError, TransportError)):
            raise value
        _raise_transport_error(value, operation, sequence=sequence)
        raise AssertionError("unreachable")

    def _audit_outbound(self, operation: str, target: str) -> tuple[str, ...]:
        audit = self._outbound_audit
        if audit is None:
            return (target,)
        aliases: tuple[str, ...] = ()
        if operation in {"reply", "patch"}:
            with self._audit_lock:
                aliases = self._audit_aliases_by_target.get(target, ())
            if not aliases and self._outbound_target_aliases is not None:
                try:
                    resolved = self._outbound_target_aliases(target)
                except Exception:
                    resolved = ()
                if isinstance(resolved, tuple) and all(
                    isinstance(alias, str) and alias for alias in resolved
                ):
                    aliases = resolved
            if not aliases:
                raise RuntimeError(
                    f"main Bot Channel card {operation} recipient scope is unavailable"
                )
        targets = tuple(dict.fromkeys((target, *aliases)))
        tenant_key = ""
        if self._tenant_key_resolver is not None:
            try:
                resolved_tenant = self._tenant_key_resolver()
                tenant_key = resolved_tenant if isinstance(resolved_tenant, str) else ""
            except Exception:
                tenant_key = ""
        for audit_target in targets:
            try:
                audit(tenant_key, operation, audit_target)
            except Exception as exc:
                logger.error(
                    "main Bot Channel card audit failed closed: %s",
                    type(exc).__name__,
                )
                if self._outbound_audit_failure is not None:
                    try:
                        self._outbound_audit_failure(exc)
                    except Exception:
                        logger.error(
                            "main Bot Channel audit failure callback failed",
                            exc_info=True,
                        )
                raise
        return targets

    _MAX_AUDIT_ALIASES = 4096

    def _remember_aliases(
        self,
        targets: tuple[str, ...],
        *identifiers: str,
    ) -> None:
        if not targets:
            return
        with self._audit_lock:
            for identifier in identifiers:
                if isinstance(identifier, str) and identifier:
                    self._audit_aliases_by_target.pop(identifier, None)
                    self._audit_aliases_by_target[identifier] = targets
                    while len(self._audit_aliases_by_target) > self._MAX_AUDIT_ALIASES:
                        self._audit_aliases_by_target.popitem(last=False)

    def _send_options(
        self,
        *,
        reply_to: str | None,
        reply_in_thread: bool | None,
        idempotency_key: str | None,
    ) -> dict:
        if (
            reply_to
            and reply_in_thread is None
            and self._default_reply_in_thread is not None
        ):
            reply_in_thread = self._default_reply_in_thread
        return {
            "receive_id_type": "chat_id",
            "reply_to": reply_to,
            "reply_in_thread": reply_in_thread,
            "reply_target_gone": "fail",
            "uuid": idempotency_key,
        }

    @staticmethod
    def _message_id(result: object, operation: str) -> str:
        if not bool(getattr(result, "success", False)):
            _raise_transport_error(result, operation)
        message_id = getattr(result, "message_id", None)
        if not isinstance(message_id, str) or not message_id:
            raise TransportError(f"{operation} response missing message_id")
        return message_id

    def create_card(
        self,
        chat_id: str,
        card_json: dict,
        *,
        reply_to: str | None = None,
        reply_in_thread: bool | None = None,
        idempotency_key: str | None = None,
    ) -> tuple[str, str]:
        operation = "reply" if reply_to else "create"
        target = reply_to or chat_id
        aliases = self._audit_outbound(operation, target)
        result = self._run(
            "channel.send.card",
            self._channel.send(
                chat_id,
                OutboundCard(card=_sanitize_card(card_json)),
                self._send_options(
                    reply_to=reply_to,
                    reply_in_thread=reply_in_thread,
                    idempotency_key=idempotency_key,
                ),
            ),
        )
        message_id = self._message_id(result, "channel.send.card")
        self._remember_aliases(aliases, message_id)
        return message_id, message_id

    def create_streaming_card(self, card_json: dict) -> str:
        if self._outbound_audit is not None:
            raise RuntimeError(
                "target-aware audit is required before creating a Channel streaming card"
            )
        return self._create_streaming_card(card_json)

    def create_streaming_card_for_target(
        self,
        card_json: dict,
        *,
        target: str,
        operation: str,
    ) -> str:
        """Audit the logical recipient before uploading a CardKit entity."""
        aliases = self._audit_outbound(operation, target)
        card_id = self._create_streaming_card(card_json)
        self._remember_aliases(aliases, card_id)
        return card_id

    def _create_streaming_card(self, card_json: dict) -> str:
        payload = _sanitize_card(card_json)
        payload.setdefault("config", {})["streaming_mode"] = True
        card_id = self._run(
            "channel.create_card_instance",
            self._channel.create_card_instance(payload),
        )
        if not isinstance(card_id, str) or not card_id:
            raise TransportError("channel.create_card_instance response missing card_id")
        return card_id

    def send_card_reference(
        self,
        chat_id: str,
        card_id: str,
        *,
        reply_to: str | None = None,
        reply_in_thread: bool | None = None,
        idempotency_key: str | None = None,
    ) -> str:
        operation = "reply" if reply_to else "create"
        target = reply_to or chat_id
        aliases = self._audit_outbound(operation, target)
        result = self._run(
            "channel.send.card_reference",
            self._channel.send(
                chat_id,
                OutboundCard(card_id=card_id),
                self._send_options(
                    reply_to=reply_to,
                    reply_in_thread=reply_in_thread,
                    idempotency_key=idempotency_key,
                ),
            ),
        )
        message_id = self._message_id(result, "channel.send.card_reference")
        self._remember_aliases(aliases, message_id, card_id)
        return message_id

    def update_card(
        self,
        card_id: str,
        card_json: dict,
        *,
        sequence: int = 0,
    ) -> None:
        self._audit_outbound("patch", card_id)
        payload = _sanitize_card(card_json)
        entity = (
            Card.builder()
            .type("card_json")
            .data(_sanitize_text(json.dumps(payload, ensure_ascii=False)))
            .build()
        )
        request = (
            UpdateCardRequest.builder()
            .card_id(card_id)
            .request_body(
                UpdateCardRequestBody.builder()
                .card(entity)
                .uuid(str(uuid.uuid5(_UPDATE_NAMESPACE, f"{card_id}:{sequence}")))
                .sequence(sequence)
                .build()
            )
            .build()
        )
        response = self._run(
            "channel.cardkit.card.update",
            self._channel.client.cardkit.v1.card.aupdate(request),
            sequence=sequence,
        )
        code = _raw_error_code(response)
        if code:
            _raise_transport_error(
                response,
                "channel.cardkit.card.update",
                sequence=sequence,
            )

    def update_element(
        self,
        card_id: str,
        element_id: str,
        content: str,
        *,
        sequence: int = 0,
    ) -> None:
        self._audit_outbound("patch", card_id)
        self._run(
            "channel.update_card_element_content",
            self._channel.update_card_element_content(
                card_id,
                element_id,
                _sanitize_text(content),
                sequence,
            ),
            sequence=sequence,
        )

    def finish_streaming_card(self, card_id: str, *, sequence: int) -> None:
        self._audit_outbound("patch", card_id)
        self._run(
            "channel.finish_streaming_card",
            self._channel.finish_streaming_card(card_id, sequence),
            sequence=sequence,
        )


__all__ = ["LarkChannelCardAPIClient"]
