"""Shared page delivery state for Workflow renderer cards."""

from __future__ import annotations

import copy
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

_STOP_ACTION = "workflow_stop_running"


@dataclass(frozen=True)
class WorkflowPageDeliveryResult:
    """Outcome of one ordered Workflow card-page delivery."""

    status_message_id: str | None
    status_delivered: bool
    failed_page_indexes: tuple[int, ...] = ()


class WorkflowCardPageDelivery:
    """Keep stable message bindings for one Workflow card page sequence.

    Page zero is the mutable status card. Later pages are append-only result
    cards: an existing binding is patched and a missing binding is created.
    Bindings are committed only after a successful create, so a failed create
    can be retried without losing any previously delivered page.
    """

    def __init__(self, page_message_ids: list[str | None]) -> None:
        if not page_message_ids:
            page_message_ids.append(None)
        self._page_message_ids = page_message_ids
        self._last_pages: list[dict[str, Any]] = []
        self._lock = threading.RLock()

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
        """Deliver renderer pages in index order while preserving page bindings."""
        incoming_pages = _normalize_pages(card_data)
        with self._lock:
            pages = self._pages_for_delivery(incoming_pages, terminal=terminal)
            failed: list[int] = []
            status_delivered = False
            creation_blocked = False

            for page_index, page in enumerate(pages):
                self._ensure_page_slot(page_index)
                current_message_id = self._page_message_ids[page_index]

                # Do not create later pages after an earlier create failed: that
                # would reverse their visible message order. Existing pages are
                # still patched, including during terminal delivery.
                if current_message_id is None and creation_blocked:
                    failed.append(page_index)
                    continue

                fallback_to_new = (
                    status_fallback_to_new
                    if page_index == 0
                    else current_message_id is None
                )
                call_kwargs: dict[str, Any] = {
                    "card_message_id": current_message_id,
                    "chat_id": chat_id,
                    "card_data": page,
                }
                if origin_message_id is not None:
                    call_kwargs["origin_message_id"] = origin_message_id
                # True is the existing method's default. Omit it to retain the
                # legacy WorkflowScriptMixin call shape.
                if not fallback_to_new:
                    call_kwargs["fallback_to_new"] = False

                try:
                    delivered_message_id = replace_or_send(**call_kwargs)
                except Exception:
                    failed.append(page_index)
                    if current_message_id is None:
                        creation_blocked = True
                    continue

                if not delivered_message_id:
                    failed.append(page_index)
                    if current_message_id is None:
                        creation_blocked = True
                    continue

                self._page_message_ids[page_index] = delivered_message_id
                if page_index == 0:
                    status_delivered = True

            return WorkflowPageDeliveryResult(
                status_message_id=self._page_message_ids[0],
                status_delivered=status_delivered,
                failed_page_indexes=tuple(failed),
            )

    def _pages_for_delivery(
        self,
        incoming_pages: list[dict[str, Any]],
        *,
        terminal: bool,
    ) -> list[dict[str, Any]]:
        for page_index, page in enumerate(incoming_pages):
            if page_index < len(self._last_pages):
                self._last_pages[page_index] = copy.deepcopy(page)
            else:
                self._last_pages.append(copy.deepcopy(page))

        if not terminal:
            return [copy.deepcopy(page) for page in incoming_pages]

        # Result pages are append-only. If the terminal renderer returns fewer
        # pages, patch every existing page with its last content after removing
        # the now-invalid stop action instead of deleting it.
        page_count = max(
            len(incoming_pages),
            len(self._last_pages),
            len(self._page_message_ids),
        )
        terminal_pages: list[dict[str, Any]] = []
        for page_index in range(page_count):
            if page_index < len(incoming_pages):
                source = incoming_pages[page_index]
            elif page_index < len(self._last_pages):
                source = self._last_pages[page_index]
            else:
                continue
            cleaned = _without_stop_actions(source)
            if page_index < len(self._last_pages):
                self._last_pages[page_index] = copy.deepcopy(cleaned)
            else:
                self._last_pages.append(copy.deepcopy(cleaned))
            terminal_pages.append(cleaned)
        return terminal_pages

    def _ensure_page_slot(self, page_index: int) -> None:
        while len(self._page_message_ids) <= page_index:
            self._page_message_ids.append(None)


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
