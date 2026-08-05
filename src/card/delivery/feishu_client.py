"""FeishuCardAPIClient: bridges CardAPIClient protocol to actual Feishu SDK calls."""

from __future__ import annotations

import json
import logging
import queue
import threading
import uuid as _uuid
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import lark_oapi as lark

from src.card.delivery.types import SequenceConflictError, TransportError
from src.config import get_settings

logger = logging.getLogger(__name__)

_DEFAULT_API_TIMEOUT_SECONDS = 35.0
_UPDATE_NAMESPACE = _uuid.UUID("7a988460-9101-492a-bdb6-98b8d27f998d")


def _sanitize_card_content(content: str) -> str:
    """Remove surrogate code points that cannot be encoded in UTF-8.

    lark_oapi internally encodes content to UTF-8, which fails on unpaired
    surrogates (U+D800-U+DFFF). Replace them with U+FFFD.
    """
    try:
        content.encode("utf-8")
        return content
    except UnicodeEncodeError:
        return content.encode("utf-8", errors="surrogatepass").decode(
            "utf-8", errors="replace"
        )


class FeishuCardAPIClient:
    """Implements CardAPIClient protocol using lark_oapi SDK.

    Maps the abstract card operations to concrete Feishu IM APIs:
    - create_card → im.v1.message.reply / im.v1.message.create
    - update_card → IM patch for message cards, CardKit update for entities
    - update_element → CardKit element content update

    Note: In Feishu's API, card_id == message_id for interactive messages.
    The `create_card` returns (message_id, message_id) since there's no
    separate card_id at the message-level API.
    """

    _worker_slots = threading.BoundedSemaphore(64)

    def __init__(
        self,
        client: "lark.Client",
        *,
        outbound_audit: Callable[[str, str, str], None] | None = None,
        outbound_audit_failure: Callable[[Exception], None] | None = None,
        tenant_key_resolver: Callable[[], str] | None = None,
        outbound_target_aliases: Callable[[str], tuple[str, ...]] | None = None,
    ) -> None:
        self._client = client
        self._settings = get_settings()
        self._outbound_audit = outbound_audit
        self._outbound_audit_failure = outbound_audit_failure
        self._tenant_key_resolver = tenant_key_resolver
        self._outbound_target_aliases = outbound_target_aliases
        self._audit_aliases_by_message: dict[str, tuple[str, ...]] = {}
        self._cardkit_entity_ids: set[str] = set()
        self._audit_alias_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

    def _audit_outbound(self, operation: str, target: str) -> tuple[str, ...]:
        audit = self._outbound_audit
        if audit is None:
            return (target,)
        aliases: tuple[str, ...] = ()
        if operation in {"reply", "patch"}:
            with self._audit_alias_lock:
                aliases = self._audit_aliases_by_message.get(target, ())
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
                    f"main Bot card {operation} recipient scope is unavailable"
                )
        targets = tuple(dict.fromkeys((target, *aliases)))
        tenant_key = ""
        if self._tenant_key_resolver is not None:
            try:
                resolved = self._tenant_key_resolver()
                tenant_key = resolved if isinstance(resolved, str) else ""
            except Exception:
                tenant_key = ""
        for audit_target in targets:
            try:
                audit(tenant_key, operation, audit_target)
            except Exception as exc:
                logger.error("main Bot card audit failed closed: %s", type(exc).__name__)
                if self._outbound_audit_failure is not None:
                    try:
                        self._outbound_audit_failure(exc)
                    except Exception:
                        logger.error("main Bot card audit failure callback failed", exc_info=True)
                raise
        return targets

    def _remember_message_audit_aliases(
        self,
        message_id: str,
        aliases: tuple[str, ...],
    ) -> None:
        if not isinstance(message_id, str) or not message_id or not aliases:
            return
        with self._audit_alias_lock:
            self._audit_aliases_by_message[message_id] = aliases

    def _api_timeout_seconds(self) -> float:
        try:
            return float(getattr(self._settings.card, "delivery_api_timeout", _DEFAULT_API_TIMEOUT_SECONDS))
        except Exception:
            return _DEFAULT_API_TIMEOUT_SECONDS

    def _call_api(self, operation: str, fn: Callable[[], object]) -> object:
        """Run one Feishu SDK request with a hard deadline.

        The lark client is configured with its own timeout, but a stuck SDK call
        must not hold CardDelivery's per-session lock forever. A daemon worker
        lets the delivery path fail fast and release that lock even if the SDK
        call does not return.
        """
        result: "queue.Queue[tuple[bool, object]]" = queue.Queue(maxsize=1)
        slots = type(self)._worker_slots
        if not slots.acquire(blocking=False):
            raise TimeoutError(f"Feishu API {operation} worker slots exhausted")

        def _target() -> None:
            try:
                result.put((True, fn()), block=False)
            except Exception as exc:
                result.put((False, exc), block=False)
            finally:
                slots.release()

        worker = threading.Thread(
            target=_target,
            name=f"feishu-api-{operation}",
            daemon=True,
        )
        try:
            worker.start()
        except Exception:
            slots.release()
            raise
        timeout = self._api_timeout_seconds()
        try:
            ok, value = result.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError(f"Feishu API {operation} timed out after {timeout:.1f}s") from exc
        if ok:
            return value
        raise value

    def create_card(
        self,
        chat_id: str,
        card_json: dict,
        *,
        reply_to: str | None = None,
        reply_in_thread: bool | None = None,
        idempotency_key: str | None = None,
    ) -> tuple[str, str]:
        """Create a card message. Returns (message_id, card_id).

        Uses reply API if reply_to is set, otherwise creates directly.
        Respects reply_in_thread override, falling back to config default_reply_mode.
        """
        from lark_oapi.api.im.v1 import (
            CreateMessageRequest,
            CreateMessageRequestBody,
            ReplyMessageRequest,
            ReplyMessageRequestBody,
        )

        content = _sanitize_card_content(json.dumps(card_json, ensure_ascii=False))

        if reply_to:
            audit_aliases = self._audit_outbound("reply", reply_to)
            if reply_in_thread is None:
                reply_in_thread = self._settings.default_reply_mode == "thread"
            request = (
                ReplyMessageRequest.builder()
                .message_id(reply_to)
                .request_body(
                    ReplyMessageRequestBody.builder()
                    .msg_type("interactive")
                    .content(content)
                    .reply_in_thread(reply_in_thread)
                    .uuid(idempotency_key)
                    .build()
                )
                .build()
            )
            response = self._call_api(
                "im.message.reply",
                lambda: self._client.im.v1.message.reply(request),
            )
        else:
            audit_aliases = self._audit_outbound("create", chat_id)
            request = (
                CreateMessageRequest.builder()
                .receive_id_type("chat_id")
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(chat_id)
                    .msg_type("interactive")
                    .content(content)
                    .uuid(idempotency_key)
                    .build()
                )
                .build()
            )
            response = self._call_api(
                "im.message.create",
                lambda: self._client.im.v1.message.create(request),
            )

        if not response.success():
            raise TransportError(
                f"Card create failed: code={response.code}, msg={response.msg}"
            )

        message_id = response.data.message_id
        self._remember_message_audit_aliases(message_id, audit_aliases)
        # In Feishu IM API, card_id == message_id for interactive messages
        return message_id, message_id

    def update_card(self, card_id: str, card_json: dict, *, sequence: int = 0) -> None:
        """Replace a card through the API matching its transport identity.

        IM interactive cards use ``message.patch``. CardKit entities learned
        during entity creation/reference use ``card.update``.
        Sequence conflicts (code 300317) are raised as SequenceConflictError.
        """
        with self._audit_alias_lock:
            is_cardkit_entity = card_id in self._cardkit_entity_ids
        if is_cardkit_entity:
            self._update_cardkit_entity(card_id, card_json, sequence=sequence)
            return

        from lark_oapi.api.im.v1 import PatchMessageRequest, PatchMessageRequestBody

        content = _sanitize_card_content(json.dumps(card_json, ensure_ascii=False))

        request = (
            PatchMessageRequest.builder()
            .message_id(card_id)
            .request_body(
                PatchMessageRequestBody.builder()
                .content(content)
                .build()
            )
            .build()
        )

        self._audit_outbound("patch", card_id)
        response = self._call_api(
            "im.message.patch",
            lambda: self._client.im.v1.message.patch(request),
        )

        if not response.success():
            if response.code == 300317:
                raise SequenceConflictError(next_floor=sequence + 1)
            raise TransportError(
                f"Patch failed: code={response.code}, msg={response.msg}, card_id={card_id}",
                code=response.code,
            )

    def _update_cardkit_entity(
        self,
        card_id: str,
        card_json: dict,
        *,
        sequence: int,
    ) -> None:
        from lark_oapi.api.cardkit.v1 import (
            Card,
            UpdateCardRequest,
            UpdateCardRequestBody,
        )

        entity = (
            Card.builder()
            .type("card_json")
            .data(_sanitize_card_content(json.dumps(card_json, ensure_ascii=False)))
            .build()
        )
        request = (
            UpdateCardRequest.builder()
            .card_id(card_id)
            .request_body(
                UpdateCardRequestBody.builder()
                .card(entity)
                .uuid(str(_uuid.uuid5(_UPDATE_NAMESPACE, f"{card_id}:{sequence}")))
                .sequence(sequence)
                .build()
            )
            .build()
        )

        self._audit_outbound("patch", card_id)
        response = self._call_api(
            "cardkit.card.update",
            lambda: self._client.cardkit.v1.card.update(request),
        )
        if not response.success():
            if response.code == 300317:
                raise SequenceConflictError(next_floor=sequence + 1)
            raise TransportError(
                f"CardKit update failed: code={response.code}, msg={response.msg}, "
                f"card_id={card_id}",
                code=response.code,
            )

    def update_element(
        self, card_id: str, element_id: str, content: str, *, sequence: int = 0
    ) -> None:
        """Update a single element's content via CardKit v2 element_content API."""
        from lark_oapi.api.cardkit.v1 import (
            ContentCardElementRequest,
            ContentCardElementRequestBody,
        )

        request = (
            ContentCardElementRequest.builder()
            .card_id(card_id)
            .element_id(element_id)
            .request_body(
                ContentCardElementRequestBody.builder()
                .content(content)
                .sequence(sequence)
                .uuid(str(_uuid.uuid4()))
                .build()
            )
            .build()
        )

        self._audit_outbound("patch", card_id)
        response = self._call_api(
            "cardkit.card_element.content",
            lambda: self._client.cardkit.v1.card_element.content(request),
        )

        if not response.success():
            if response.code == 300317:  # Sequence conflict
                raise SequenceConflictError(next_floor=sequence + 1)
            raise TransportError(
                f"Element update failed: code={response.code}, msg={response.msg}",
                code=response.code,
            )

    def create_streaming_card(self, card_json: dict) -> str:
        """Create a card entity via CardKit API with streaming mode enabled.

        The card_json must have "config": {"streaming_mode": true} set.
        Returns the card_id from the created card entity.
        """
        from lark_oapi.api.cardkit.v1 import (
            CreateCardRequest,
            CreateCardRequestBody,
        )

        # Ensure streaming_mode is set in config
        config = card_json.setdefault("config", {})
        config["streaming_mode"] = True

        request = (
            CreateCardRequest.builder()
            .request_body(
                CreateCardRequestBody.builder()
                .type("card_json")
                .data(_sanitize_card_content(json.dumps(card_json, ensure_ascii=False)))
                .build()
            )
            .build()
        )

        response = self._call_api(
            "cardkit.card.create",
            lambda: self._client.cardkit.v1.card.create(request),
        )

        if not response.success():
            raise TransportError(
                f"Streaming card create failed: code={response.code}, msg={response.msg}"
            )

        card_id = response.data.card_id
        with self._audit_alias_lock:
            self._cardkit_entity_ids.add(card_id)
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
        """Send an IM message referencing a CardKit card entity.

        Returns the message_id of the sent message.
        """
        from lark_oapi.api.im.v1 import (
            CreateMessageRequest,
            CreateMessageRequestBody,
            ReplyMessageRequest,
            ReplyMessageRequestBody,
        )

        content = json.dumps({"type": "card", "data": {"card_id": card_id}})

        if reply_to:
            audit_aliases = self._audit_outbound("reply", reply_to)
            if reply_in_thread is None:
                reply_in_thread = self._settings.default_reply_mode == "thread"
            request = (
                ReplyMessageRequest.builder()
                .message_id(reply_to)
                .request_body(
                    ReplyMessageRequestBody.builder()
                    .msg_type("interactive")
                    .content(content)
                    .reply_in_thread(reply_in_thread)
                    .uuid(idempotency_key)
                    .build()
                )
                .build()
            )
            response = self._call_api(
                "im.message.reply",
                lambda: self._client.im.v1.message.reply(request),
            )
        else:
            audit_aliases = self._audit_outbound("create", chat_id)
            request = (
                CreateMessageRequest.builder()
                .receive_id_type("chat_id")
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(chat_id)
                    .msg_type("interactive")
                    .content(content)
                    .uuid(idempotency_key)
                    .build()
                )
                .build()
            )
            response = self._call_api(
                "im.message.create",
                lambda: self._client.im.v1.message.create(request),
            )

        if not response.success():
            raise TransportError(
                f"Card reference send failed: code={response.code}, msg={response.msg}"
            )

        message_id = response.data.message_id
        self._remember_message_audit_aliases(message_id, audit_aliases)
        # CardKit mutations address the card entity, while IM delivery returns a
        # separate message id.  Both targets represent the same recipient scope
        # and must survive provenance lookup failures after the initial send.
        self._remember_message_audit_aliases(card_id, audit_aliases)
        with self._audit_alias_lock:
            self._cardkit_entity_ids.add(card_id)
        return message_id
