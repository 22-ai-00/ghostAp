"""Delivery binding: tracks which message/card corresponds to each page."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class PageBinding:
    """Binding for a single card page."""

    message_id: str = ""
    card_id: str = ""
    signature: str = ""
    last_text: str = ""
    page_index: int = 0
    source_page_index: int = 0
    is_frozen: bool = False
    is_streaming: bool = False
    supports_element_update: bool = False


@dataclass
class DeliveryBinding:
    """Binding for all pages of a session."""

    session_id: str = ""
    chat_id: str = ""
    pages: dict[int, PageBinding] = field(default_factory=dict)
    segment_index: int = 0
    message_high_watermark: int = -1
    latest_source_page_index: int = -1
    stale_replacement_pages: dict[int, int] = field(default_factory=dict)


class BindingStore:
    """Thread-safe store for session → DeliveryBinding mappings."""

    def __init__(self) -> None:
        self._bindings: dict[str, DeliveryBinding] = {}
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

    def get(self, session_id: str) -> DeliveryBinding | None:
        """Get binding for a session."""
        with self._lock:
            return self._bindings.get(session_id)

    def has(self, session_id: str) -> bool:
        """Check if a binding exists for a session."""
        with self._lock:
            return session_id in self._bindings

    def create(self, session_id: str, chat_id: str) -> DeliveryBinding:
        """Create a new binding for a session."""
        with self._lock:
            binding = DeliveryBinding(session_id=session_id, chat_id=chat_id)
            self._bindings[session_id] = binding
            return binding

    def set_page(
        self,
        session_id: str,
        page_index: int,
        message_id: str,
        card_id: str,
        signature: str,
        last_text: str = "",
        source_page_index: int | None = None,
        is_streaming: bool = False,
        supports_element_update: bool = False,
    ) -> None:
        """Set or update a page binding."""
        with self._lock:
            binding = self._bindings.get(session_id)
            if binding is None:
                return
            binding.pages[page_index] = PageBinding(
                message_id=message_id,
                card_id=card_id,
                signature=signature,
                last_text=last_text,
                page_index=page_index,
                source_page_index=(
                    page_index if source_page_index is None else source_page_index
                ),
                is_streaming=is_streaming,
                supports_element_update=supports_element_update,
            )
            if page_index >= binding.message_high_watermark:
                binding.message_high_watermark = page_index
                binding.latest_source_page_index = (
                    page_index if source_page_index is None else source_page_index
                )

    def update_text(self, session_id: str, page_index: int, text: str) -> None:
        """Update the last_text for a page (after element_content push)."""
        with self._lock:
            binding = self._bindings.get(session_id)
            if binding is None:
                return
            page = binding.pages.get(page_index)
            if page is not None:
                page.last_text = text

    def update_signature(self, session_id: str, page_index: int, signature: str) -> None:
        """Update the signature for a page (after card.update)."""
        with self._lock:
            binding = self._bindings.get(session_id)
            if binding is None:
                return
            page = binding.pages.get(page_index)
            if page is not None:
                page.signature = signature

    def update_source_page_index(
        self,
        session_id: str,
        page_index: int,
        source_page_index: int,
    ) -> None:
        """Record which logical renderer page occupies a visible message page."""
        with self._lock:
            binding = self._bindings.get(session_id)
            if binding is None:
                return
            page = binding.pages.get(page_index)
            if page is None:
                return
            page.source_page_index = source_page_index
            if page_index == binding.message_high_watermark:
                binding.latest_source_page_index = source_page_index

    def mark_frozen(self, session_id: str, page_index: int, *, frozen: bool = True) -> None:
        """Mark whether a page is an immutable history snapshot."""
        with self._lock:
            binding = self._bindings.get(session_id)
            if binding is None:
                return
            page = binding.pages.get(page_index)
            if page is not None:
                page.is_frozen = frozen

    def mark_streaming(
        self,
        session_id: str,
        page_index: int,
        *,
        streaming: bool,
    ) -> None:
        """Record whether a visible page still has an open CardKit stream."""
        with self._lock:
            binding = self._bindings.get(session_id)
            if binding is None:
                return
            page = binding.pages.get(page_index)
            if page is not None:
                page.is_streaming = streaming

    def remove_page(self, session_id: str, page_index: int) -> PageBinding | None:
        """Remove a page from the binding."""
        with self._lock:
            binding = self._bindings.get(session_id)
            if binding is None:
                return None
            return binding.pages.pop(page_index, None)

    def remove(self, session_id: str) -> DeliveryBinding | None:
        """Remove and return the binding for a session."""
        with self._lock:
            return self._bindings.pop(session_id, None)

    def page_count(self, session_id: str) -> int:
        """Get the number of pages for a session."""
        with self._lock:
            binding = self._bindings.get(session_id)
            if binding is None:
                return 0
            return len(binding.pages)
