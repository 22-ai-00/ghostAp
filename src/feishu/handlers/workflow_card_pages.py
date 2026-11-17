"""Shared page delivery state for Workflow renderer cards."""

from __future__ import annotations

import copy
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

_STOP_ACTION = "workflow_stop_running"
_PAGE_KEY_FIELD = "_workflow_page_key"
_PAGE_KEY_KINDS = frozenset({"status", "agent"})
_TERMINAL_STATUS_PAGE_CONTENT = (
    "📎 **运行进度已收起**\n"
    "任务已结束；本页仅用于运行期分页，最终状态请查看首张 Workflow 卡片。"
)
_PageIdentity = int | str
_PageKey = tuple[str, _PageIdentity, int]


@dataclass(frozen=True)
class WorkflowPageDeliveryResult:
    """Outcome of one ordered Workflow card-page delivery."""

    status_message_id: str | None
    status_delivered: bool
    failed_page_indexes: tuple[int, ...] = ()
    delivered_page_indexes: tuple[int, ...] = ()
    page_count: int = 0

    @property
    def fully_delivered(self) -> bool:
        """Whether every requested page was proven delivered."""
        return bool(
            self.page_count > 0
            and self.status_delivered
            and not self.failed_page_indexes
            and len(self.delivered_page_indexes) == self.page_count
        )


class WorkflowCardPageDelivery:
    """Keep stable message bindings for one Workflow card page sequence.

    Page zero is the mutable status card. Later keyed pages hold direct-call
    execution streams or result ledgers: an existing binding is patched and a
    missing binding is created without reassigning another page's message.
    Bindings are committed only after a successful create, so a failed create
    can be retried without losing any previously delivered page.
    """

    def __init__(self, page_message_ids: list[str | None]) -> None:
        if not page_message_ids:
            page_message_ids.append(None)
        self._page_message_ids = page_message_ids
        self._key_to_message_id: dict[_PageKey, str | None] = {}
        self._last_pages_by_key: dict[_PageKey, dict[str, Any]] = {}
        self._delivered_wire_pages_by_key: dict[_PageKey, dict[str, Any]] = {}
        self._known_page_keys: list[_PageKey] = []
        self._lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock

    @property
    def page_message_ids(self) -> tuple[str | None, ...]:
        with self._lock:
            return tuple(self._page_message_ids)

    def deliver(
        self,
        card_data: dict[str, Any] | Sequence[dict[str, Any]],
        *,
        replace_or_send: Callable[..., str | None],
        chat_id: str,
        origin_message_id: str | None = None,
        status_fallback_to_new: bool = True,
        terminal: bool = False,
    ) -> WorkflowPageDeliveryResult:
        """Deliver keyed Workflow renderer pages while keeping stable message bindings."""
        incoming_pages = _normalize_pages(card_data)
        with self._lock:
            page_keys = [_page_key(page) for page in incoming_pages]
            if any(key is None for key in page_keys):
                raise ValueError("Workflow renderer pages must all include _workflow_page_key")
            if len(set(page_keys)) != len(page_keys):
                raise ValueError("Workflow renderer page keys must be unique")

            return self._deliver_keyed(
                incoming_pages,
                tuple(key for key in page_keys if key is not None),
                replace_or_send=replace_or_send,
                chat_id=chat_id,
                origin_message_id=origin_message_id,
                status_fallback_to_new=status_fallback_to_new,
                terminal=terminal,
            )

    def _deliver_keyed(
        self,
        incoming_pages: list[dict[str, Any]],
        incoming_keys: tuple[_PageKey, ...],
        *,
        replace_or_send: Callable[..., str | None],
        chat_id: str,
        origin_message_id: str | None,
        status_fallback_to_new: bool,
        terminal: bool,
    ) -> WorkflowPageDeliveryResult:
        pages = self._keyed_pages_for_delivery(
            incoming_pages,
            incoming_keys,
            terminal=terminal,
        )
        failed: list[int] = []
        delivered: list[int] = []
        status_delivered = False
        creation_blocked = False
        status_key = ("status", -1, 0)

        if status_key in self._known_page_keys and status_key not in self._key_to_message_id:
            self._key_to_message_id[status_key] = self._page_message_ids[0]

        for page_index, (page_key, page) in enumerate(pages):
            current_message_id = self._key_to_message_id.get(page_key)
            if current_message_id is None and creation_blocked:
                failed.append(page_index)
                continue

            is_status = page_key == status_key
            wire_page = _without_page_key(page)
            if (
                current_message_id is not None
                and page_key in self._delivered_wire_pages_by_key
                and self._delivered_wire_pages_by_key[page_key] == wire_page
            ):
                delivered.append(page_index)
                if is_status:
                    status_delivered = True
                continue

            fallback_to_new = (
                status_fallback_to_new if is_status else current_message_id is None
            )
            call_kwargs: dict[str, Any] = {
                "card_message_id": current_message_id,
                "chat_id": chat_id,
                "card_data": wire_page,
                "fallback_to_new": fallback_to_new,
            }
            if origin_message_id is not None:
                call_kwargs["origin_message_id"] = origin_message_id

            try:
                delivered_message_id = replace_or_send(**call_kwargs)
            except Exception:
                delivered_message_id = None

            if not delivered_message_id:
                failed.append(page_index)
                if current_message_id is None:
                    creation_blocked = True
                continue

            self._key_to_message_id[page_key] = delivered_message_id
            self._delivered_wire_pages_by_key[page_key] = copy.deepcopy(wire_page)
            if delivered_message_id not in self._page_message_ids:
                self._page_message_ids.append(delivered_message_id)
            delivered.append(page_index)
            if is_status:
                status_delivered = True

        status_message_id = self._key_to_message_id.get(status_key)
        return WorkflowPageDeliveryResult(
            status_message_id=status_message_id,
            status_delivered=status_delivered,
            failed_page_indexes=tuple(failed),
            delivered_page_indexes=tuple(delivered),
            page_count=len(pages),
        )

    def _keyed_pages_for_delivery(
        self,
        incoming_pages: list[dict[str, Any]],
        incoming_keys: tuple[_PageKey, ...],
        *,
        terminal: bool,
    ) -> list[tuple[_PageKey, dict[str, Any]]]:
        for page_key, page in zip(incoming_keys, incoming_pages):
            self._last_pages_by_key[page_key] = copy.deepcopy(page)
            if page_key not in self._known_page_keys:
                self._known_page_keys.append(page_key)

        delivery_keys = list(incoming_keys)
        if terminal:
            delivery_keys.extend(
                page_key
                for page_key in self._known_page_keys
                if page_key not in incoming_keys
                and not (
                    _is_status_continuation(page_key)
                    and self._key_to_message_id.get(page_key) is None
                )
            )

        terminal_status_page = next(
            (
                page
                for page_key, page in zip(incoming_keys, incoming_pages)
                if page_key == ("status", -1, 0)
            ),
            None,
        )
        pages: list[tuple[_PageKey, dict[str, Any]]] = []
        for page_key in delivery_keys:
            if (
                terminal
                and page_key not in incoming_keys
                and _is_status_continuation(page_key)
            ):
                page = _retired_status_page(terminal_status_page)
                page[_PAGE_KEY_FIELD] = page_key
            else:
                page = copy.deepcopy(self._last_pages_by_key[page_key])
            if terminal:
                page = _without_stop_actions(page)
                self._last_pages_by_key[page_key] = copy.deepcopy(page)
            pages.append((page_key, page))
        return pages


def _normalize_pages(
    card_data: dict[str, Any] | Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(card_data, dict):
        return [copy.deepcopy(card_data)]
    if isinstance(card_data, Sequence) and not isinstance(card_data, (str, bytes)):
        pages = list(card_data)
        if not all(isinstance(page, dict) for page in pages):
            raise TypeError("Workflow renderer pages must all be dictionaries")
        return [copy.deepcopy(page) for page in pages]
    raise TypeError("Workflow renderer output must be a card dictionary or page sequence")


def _page_key(card: dict[str, Any]) -> _PageKey | None:
    raw = card.get(_PAGE_KEY_FIELD)
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError("Invalid Workflow renderer page key")
    kind, page_identity, local_page_index = raw
    if kind not in _PAGE_KEY_KINDS:
        raise ValueError("Invalid Workflow renderer page key kind")
    if isinstance(page_identity, bool) or not isinstance(page_identity, (int, str)):
        raise ValueError("Invalid Workflow renderer page identity")
    if isinstance(page_identity, str) and (
        kind != "agent" or not page_identity.strip()
    ):
        raise ValueError("Invalid Workflow renderer agent page identity")
    if (
        isinstance(local_page_index, bool)
        or not isinstance(local_page_index, int)
        or local_page_index < 0
    ):
        raise ValueError("Invalid Workflow renderer local page index")
    return str(kind), page_identity, local_page_index


def _without_page_key(card: dict[str, Any]) -> dict[str, Any]:
    cleaned = copy.deepcopy(card)
    cleaned.pop(_PAGE_KEY_FIELD, None)
    return cleaned


def _is_status_continuation(page_key: _PageKey) -> bool:
    """Whether a key names a transient status page after the primary page."""
    kind, page_identity, local_page_index = page_key
    return kind == "status" and page_identity == -1 and local_page_index > 0


def _retired_status_page(
    terminal_status_page: dict[str, Any] | None,
) -> dict[str, Any]:
    """Replace a stale running-only status continuation with a terminal stub."""
    header_source = (
        terminal_status_page.get("header")
        if isinstance(terminal_status_page, dict)
        else None
    )
    header = (
        copy.deepcopy(header_source)
        if isinstance(header_source, dict)
        else {
            "title": {"tag": "plain_text", "content": "Workflow · 已结束"},
            "template": "grey",
        }
    )
    return {
        "header": header,
        "elements": [
            {
                "tag": "markdown",
                "content": _TERMINAL_STATUS_PAGE_CONTENT,
            }
        ],
    }


def _without_stop_actions(card: dict[str, Any]) -> dict[str, Any]:
    cleaned = _clean_node(copy.deepcopy(card))
    return cleaned if isinstance(cleaned, dict) else {}


def _clean_node(node: Any) -> Any:
    if isinstance(node, list):
        cleaned_items = []
        for item in node:
            cleaned = _clean_node(item)
            if cleaned is not None:
                cleaned_items.append(cleaned)
        return cleaned_items

    if not isinstance(node, dict):
        return node

    value = node.get("value")
    if (
        node.get("tag") == "button"
        and isinstance(value, dict)
        and value.get("action") == _STOP_ACTION
    ):
        return None

    cleaned_node = {
        key: cleaned
        for key, raw_value in node.items()
        if (cleaned := _clean_node(raw_value)) is not None
    }
    if cleaned_node.get("tag") == "action" and not cleaned_node.get("actions"):
        return None
    return cleaned_node
