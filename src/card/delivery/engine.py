"""CardDelivery: unified delivery engine for Feishu card operations."""

from __future__ import annotations

import dataclasses
import logging
import re
import threading
from collections.abc import Callable

from src.card.delivery.binding import BindingStore, PageBinding
from src.card.delivery.lock_pool import SessionLockPool
from src.card.delivery.page_mutator import PageMutator, sanitize_card_text_for_audit
from src.card.delivery.registry import DeliveryRegistry, delivery_registry
from src.card.delivery.sequence import SequenceManager
from src.card.delivery.ttl_set import TTLSet
from src.card.delivery.types import MutationOutcome, SequenceConflictError, TransportError
from src.card.protocols import CardAPIClient
from src.card.types import RenderedCard

logger = logging.getLogger(__name__)

_RECREATE_OUTCOME_PREFIX = "recreate:"


# ---------------------------------------------------------------------------
# Outcome types (re-exported from src.card.delivery.types for backwards compat)
# ---------------------------------------------------------------------------

__all__ = ["CardDelivery", "MutationOutcome", "SequenceConflictError", "TransportError", "CardAPIClient"]


# ---------------------------------------------------------------------------
# CardDelivery engine
# ---------------------------------------------------------------------------


class CardDelivery:
    """Unified delivery engine.

    Merges the responsibilities of card creation + element update:
    - Decides operation type (create / update / element_content)
    - Manages sequence numbers for optimistic concurrency
    - Handles reconciliation on conflict

    Public API: deliver() and close() only.
    Lifecycle management (shutdown/drain) is accessed via DeliveryRegistry.

    Thread-safety: deliver() and close() are idempotent and concurrency-safe.
    After close(session_id) completes, subsequent deliver() calls for that
    session_id are no-ops.
    """

    def __init__(
        self,
        client: CardAPIClient,
        *,
        max_session_locks: int = 10_000,
        session_lock_ttl: float = 600.0,
        eviction_interval: float = 30.0,
        registry: "DeliveryRegistry | None" = None,
        payload_transform: Callable[[str, dict], dict] | None = None,
        trust_revision_provider: Callable[[str], tuple[int, int] | None] | None = None,
    ) -> None:
        self._client = client
        self._bindings = BindingStore()
        self._sequences = SequenceManager()
        self._closed_sessions = TTLSet(ttl=3600.0, max_size=50_000)
        self._mutator = PageMutator(client, self._bindings, self._sequences)
        self._payload_transform = payload_transform
        self._trust_revision_provider = trust_revision_provider
        self._trust_lock = threading.Lock()
        self._expected_trust_revisions: dict[str, tuple[int, int] | None] = {}
        self._trust_snapshot_failed: set[str] = set()

        # Delegate lock pool management
        self._lock_pool = SessionLockPool(
            max_locks=max_session_locks,
            lock_ttl=session_lock_ttl,
            eviction_interval=eviction_interval,
            has_active_binding=self._bindings.has,
            purge_callback=lambda: self._closed_sessions.purge(),
        )

        # Independent lock for _closed_sessions (decoupled from lock pool internals)
        self._closed_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

        self._max_session_locks = max_session_locks
        self._session_lock_ttl = session_lock_ttl
        self._eviction_interval = eviction_interval

        self._registry = registry or delivery_registry
        self._registry.register(self)

    def _shutdown(self) -> None:
        """Stop background eviction thread. Called via DeliveryRegistry on graceful shutdown."""
        self._lock_pool.shutdown()
        self._registry.unregister(self)

    @property
    def _accepting_work(self) -> bool:
        """Whether this delivery engine is accepting new work (read-only proxy to lock pool)."""
        return self._lock_pool.accepting_work

    def _drain(self, timeout: float = 5.0) -> bool:
        """Wait for all in-flight deliveries on this instance to finish.

        Returns True if all drained within *timeout*, False otherwise.
        """
        self._lock_pool.fence()
        return self._lock_pool.drain(timeout=timeout)

    def release_session_lock(self, session_id: str) -> None:
        """Release the delivery lock for *session_id* (idempotent).

        Public API — called by CardSession finalizer and close() to release
        the per-session delivery lock on session termination.
        """
        self._lock_pool.release(session_id)

    def __del__(self) -> None:
        """Defense-in-depth: stop eviction thread if instance leaks without explicit shutdown."""
        try:
            self._lock_pool.shutdown()
        except Exception:
            pass

    def __enter__(self) -> CardDelivery:
        return self

    def __exit__(self, *exc_info) -> None:
        self._shutdown()

    def deliver(
        self,
        session_id: str,
        chat_id: str,
        rendered: list[RenderedCard],
        *,
        reply_to: str | None = None,
        reply_in_thread: bool | None = None,
        is_terminal: bool = False,
    ) -> list[MutationOutcome]:
        """Deliver rendered cards to Feishu.

        Decision logic:
        - No binding → card.create
        - Signature changed → card.update
        - Only text changed → element_content
        - No change → skip

        Pagination follows an append-only message contract: pages are created in
        ascending order, a page is flushed once when it becomes history, and only
        the highest visible page continues to receive live or terminal updates.
        This preserves Feishu message chronology even when rendered page counts
        grow or shrink.

        Idempotent: returns empty list if session is already closed.

        Timeout: This method does NOT enforce its own timeout. The underlying
        Feishu API client is expected to raise TimeoutError (or similar) when
        a request exceeds its configured timeout (default 30s). Such exceptions
        are caught by PageMutator and surfaced as ``MutationOutcome(kind="reconcile")``.
        Callers should NOT wrap deliver() in an external timeout — doing so risks
        leaving per-session locks unreleased.
        """
        if not self._lock_pool.accepting_work:
            return [MutationOutcome(kind="rejected", message="delivery shutting down")]
        with self._closed_lock:
            if session_id in self._closed_sessions:
                return []
        self.open_session(session_id, chat_id)
        # Acquire per-session lock (creates if needed, may evict LRU)
        try:
            session_lock = self._lock_pool.acquire(session_id)
        except RuntimeError:
            logger.error(
                "Session lock capacity exhausted, rejecting new session %s",
                session_id,
            )
            return [MutationOutcome(kind="rejected", message="session lock capacity exhausted")]

        self._lock_pool.enter_delivery()
        try:
            with session_lock:
                return self._deliver_unlocked(session_id, chat_id, rendered, reply_to=reply_to, reply_in_thread=reply_in_thread, is_terminal=is_terminal)
        finally:
            self._lock_pool.exit_delivery()

    def _deliver_unlocked(
        self,
        session_id: str,
        chat_id: str,
        rendered: list[RenderedCard],
        *,
        reply_to: str | None = None,
        reply_in_thread: bool | None = None,
        is_terminal: bool = False,
    ) -> list[MutationOutcome]:
        """Internal deliver implementation (caller holds per-session lock)."""
        # Second check under session lock: eliminates TOCTOU window between
        # the fast-path _closed_lock check and per-session lock acquisition.
        if session_id in self._closed_sessions:
            return []
        if not self._current_trust_matches(session_id, chat_id):
            return [MutationOutcome(kind="rejected", message="managed trust revision changed")]
        binding = self._bindings.get(session_id)
        outcomes: list[MutationOutcome] = []
        ordered = sorted(rendered, key=lambda card: card.page_index)
        if self._payload_transform is not None:
            try:
                ordered = [
                    self._transform_rendered_payload(session_id, chat_id, card)
                    for card in ordered
                ]
            except (RuntimeError, TypeError, ValueError):
                logger.warning("managed card payload transform rejected", exc_info=True)
                return [MutationOutcome(kind="rejected", message="managed trust payload rejected")]
        if not ordered:
            return outcomes
        latest_rendered_idx = ordered[-1].page_index

        if binding is None:
            binding = self._bindings.create(session_id, chat_id)
            previous_page_index: int | None = None
            for card in ordered:
                outcome = self._create_page(
                    session_id,
                    chat_id,
                    card,
                    reply_to=reply_to,
                    reply_in_thread=reply_in_thread,
                    is_terminal=is_terminal,
                )
                outcomes.append(outcome)
                if outcome.kind != "applied":
                    break
                if previous_page_index is not None:
                    freeze_outcome = self._freeze_page(session_id, previous_page_index)
                    if freeze_outcome.kind == "reconcile":
                        outcomes.append(freeze_outcome)
                        break
                previous_page_index = card.page_index
            return outcomes

        message_high_watermark = binding.message_high_watermark

        # A failed append leaves the visible slot reserved by the monotonic high
        # watermark while no binding exists for it. Retry from the same logical
        # source page at the same next visible index/idempotency key.
        if (
            message_high_watermark >= 0
            and message_high_watermark not in binding.pages
        ):
            return self._append_from_source(
                session_id,
                chat_id,
                binding,
                ordered,
                source_page_index=binding.latest_source_page_index,
                reply_to=reply_to,
                reply_in_thread=reply_in_thread,
                is_terminal=is_terminal,
            )

        if not binding.pages:
            previous_page_index = None
            for card in ordered:
                outcome = self._create_page(
                    session_id,
                    chat_id,
                    card,
                    reply_to=reply_to,
                    reply_in_thread=reply_in_thread,
                    is_terminal=is_terminal,
                )
                outcomes.append(outcome)
                if outcome.kind != "applied":
                    break
                if previous_page_index is not None:
                    freeze_outcome = self._freeze_page(session_id, previous_page_index)
                    if freeze_outcome.kind == "reconcile":
                        outcomes.append(freeze_outcome)
                        break
                previous_page_index = card.page_index
            return outcomes

        latest_page = binding.pages[message_high_watermark]
        latest_source_idx = latest_page.source_page_index

        # Renderer compaction can reduce its page count. Never delete or reuse an
        # older Feishu message: move the renderer's newest state onto the existing
        # highest message and preserve all lower bindings as immutable history.
        if latest_rendered_idx < latest_source_idx:
            latest_card = self._remap_latest_card(
                ordered[-1],
                page_index=message_high_watermark,
                total_pages=message_high_watermark + 1,
            )
            self._bindings.mark_frozen(
                session_id,
                message_high_watermark,
                frozen=False,
            )
            outcome = self._mutate_page(
                session_id,
                latest_page,
                latest_card,
                is_terminal=is_terminal,
            )
            if self._is_stale_binding_outcome(outcome):
                outcome = self._append_visible_page(
                    session_id,
                    chat_id,
                    binding,
                    ordered[-1],
                    reply_to=reply_to,
                    reply_in_thread=reply_in_thread,
                    is_terminal=is_terminal,
                )
            elif outcome.kind == "applied":
                self._bindings.update_source_page_index(
                    session_id,
                    message_high_watermark,
                    latest_rendered_idx,
                )
            outcomes.append(outcome)
            return outcomes

        pending_freeze_index: int | None = None
        for card in ordered:
            page_idx = card.page_index
            existing_page = self._latest_page_for_source(binding, page_idx)
            mutated_visible_index = (
                existing_page.page_index if existing_page is not None else None
            )

            if existing_page is not None and existing_page.is_frozen:
                if is_terminal and existing_page.is_streaming:
                    finish_outcome = self._mutator.finish_streaming_page(
                        session_id,
                        existing_page,
                    )
                    outcomes.append(finish_outcome)
                    if finish_outcome.kind == "reconcile":
                        break
                    continue
                outcomes.append(MutationOutcome(kind="skipped", message="history_page_frozen"))
                continue
            if existing_page is None and page_idx < latest_source_idx:
                outcomes.append(MutationOutcome(kind="skipped", message="history_page_frozen"))
                continue

            if existing_page is None:
                outcome = self._append_visible_page(
                    session_id,
                    chat_id,
                    binding,
                    card,
                    reply_to=reply_to,
                    reply_in_thread=reply_in_thread,
                    is_terminal=is_terminal,
                )
            else:
                target_card = card
                if existing_page.page_index != card.page_index:
                    target_card = self._remap_latest_card(
                        card,
                        page_index=existing_page.page_index,
                        total_pages=existing_page.page_index + 1,
                    )
                outcome = self._mutate_page(
                    session_id,
                    existing_page,
                    target_card,
                    is_terminal=is_terminal,
                )
            if self._is_stale_binding_outcome(outcome):
                outcome = self._append_visible_page(
                    session_id,
                    chat_id,
                    binding,
                    card,
                    reply_to=reply_to,
                    reply_in_thread=reply_in_thread,
                    is_terminal=is_terminal,
                )
                mutated_visible_index = binding.message_high_watermark
            outcomes.append(outcome)
            if outcome.kind == "reconcile":
                break

            if pending_freeze_index is not None:
                freeze_outcome = self._freeze_page(session_id, pending_freeze_index)
                if freeze_outcome.kind == "reconcile":
                    outcomes.append(freeze_outcome)
                    break
                pending_freeze_index = None

            # When pagination grows, the previous live page is first flushed with
            # its boundary content above. Freeze it only after the following page
            # is confirmed visible, so a failed append leaves a live page that a
            # later compact terminal render can still update.
            if page_idx < latest_rendered_idx:
                pending_freeze_index = (
                    mutated_visible_index
                    if mutated_visible_index is not None
                    else binding.message_high_watermark
                )

        return outcomes

    def _transform_rendered_payload(
        self,
        session_id: str,
        chat_id: str,
        card: RenderedCard,
    ) -> RenderedCard:
        payload = self._payload_transform(chat_id, card.to_feishu_json())
        if not isinstance(payload, dict):
            raise TypeError("card payload transform must return dict")
        revisions: set[tuple[int, int]] = set()
        callback_revisions: list[tuple[int, int] | None] = []

        def collect(node) -> None:
            if isinstance(node, dict):
                group_revision = node.get("group_revision")
                grant_revision = node.get("grant_revision")
                if type(group_revision) is int and type(grant_revision) is int:
                    revisions.add((group_revision, grant_revision))
                for key, value in node.items():
                    if (
                        key == "value"
                        and isinstance(value, dict)
                        and ("action" in value or "action_id" in value)
                    ):
                        callback_group_revision = value.get("group_revision")
                        callback_grant_revision = value.get("grant_revision")
                        callback_revisions.append(
                            (
                                callback_group_revision,
                                callback_grant_revision,
                            )
                            if type(callback_group_revision) is int
                            and type(callback_grant_revision) is int
                            else None
                        )
                    collect(value)
            elif isinstance(node, list):
                for value in node:
                    collect(value)

        collect(payload)
        with self._trust_lock:
            expected = self._expected_trust_revisions.get(session_id)
        if self._trust_revision_provider is not None:
            if expected is None and revisions:
                raise ValueError("unmanaged card payload contains managed trust revisions")
            if expected is not None:
                if revisions - {expected}:
                    raise ValueError("managed card payload trust revision mismatch")
                if any(revision != expected for revision in callback_revisions):
                    raise ValueError("managed card callback trust revision mismatch")
                if not callback_revisions and revisions:
                    raise ValueError("passive managed card contains trust revisions")
        revision_suffix = ""
        if revisions:
            revision_suffix = ":trust:" + ",".join(
                f"{group_revision}/{grant_revision}"
                for group_revision, grant_revision in sorted(revisions)
            )
        return dataclasses.replace(
            card,
            _card_json=payload,
            structure_signature=card.structure_signature + revision_suffix,
            content_hash=card.content_hash + revision_suffix,
        )

    def open_session(self, session_id: str, chat_id: str) -> None:
        """Capture the immutable managed trust snapshot for one card run."""

        if self._trust_revision_provider is None:
            return
        with self._trust_lock:
            if (
                session_id in self._expected_trust_revisions
                or session_id in self._trust_snapshot_failed
            ):
                return
        try:
            revisions = self._trust_revision_provider(chat_id)
            if revisions is not None and (
                type(revisions) is not tuple
                or len(revisions) != 2
                or any(type(value) is not int or value < 1 for value in revisions)
            ):
                raise ValueError("invalid managed trust revision snapshot")
        except Exception:
            with self._trust_lock:
                self._trust_snapshot_failed.add(session_id)
            logger.warning("managed card trust snapshot failed closed", exc_info=True)
            return
        with self._trust_lock:
            if session_id not in self._trust_snapshot_failed:
                self._expected_trust_revisions.setdefault(session_id, revisions)

    def _current_trust_matches(self, session_id: str, chat_id: str) -> bool:
        if self._trust_revision_provider is None:
            return True
        with self._trust_lock:
            if session_id in self._trust_snapshot_failed:
                return False
            expected = self._expected_trust_revisions.get(session_id)
        try:
            return self._trust_revision_provider(chat_id) == expected
        except Exception:
            logger.warning("managed card current trust unavailable", exc_info=True)
            return False

    @staticmethod
    def _latest_page_for_source(binding, source_page_index: int) -> PageBinding | None:
        matches = [
            page
            for page in binding.pages.values()
            if page.source_page_index == source_page_index
        ]
        if not matches:
            return None
        return max(matches, key=lambda page: page.page_index)

    def _append_from_source(
        self,
        session_id: str,
        chat_id: str,
        binding,
        ordered: list[RenderedCard],
        *,
        source_page_index: int,
        reply_to: str | None,
        reply_in_thread: bool | None,
        is_terminal: bool,
    ) -> list[MutationOutcome]:
        """Retry a missing live page and append any newer logical pages."""
        start = next(
            (
                index
                for index, card in enumerate(ordered)
                if card.page_index == source_page_index
            ),
            len(ordered) - 1,
        )
        outcomes: list[MutationOutcome] = []
        pending_freeze_index: int | None = None
        for index, card in enumerate(ordered[start:], start=start):
            outcome = self._append_visible_page(
                session_id,
                chat_id,
                binding,
                card,
                reply_to=reply_to,
                reply_in_thread=reply_in_thread,
                is_terminal=is_terminal,
            )
            outcomes.append(outcome)
            if outcome.kind != "applied":
                break
            if pending_freeze_index is not None:
                freeze_outcome = self._freeze_page(session_id, pending_freeze_index)
                if freeze_outcome.kind == "reconcile":
                    outcomes.append(freeze_outcome)
                    break
                pending_freeze_index = None
            if index < len(ordered) - 1:
                pending_freeze_index = binding.message_high_watermark
        return outcomes

    def _append_visible_page(
        self,
        session_id: str,
        chat_id: str,
        binding,
        card: RenderedCard,
        *,
        reply_to: str | None,
        reply_in_thread: bool | None,
        is_terminal: bool,
    ) -> MutationOutcome:
        """Append one logical renderer page at the next visible message index."""
        replacement_index = binding.message_high_watermark + 1
        replacement = self._remap_latest_card(
            card,
            page_index=replacement_index,
            total_pages=replacement_index + 1,
        )
        return self._create_page(
            session_id,
            chat_id,
            replacement,
            reply_to=reply_to,
            reply_in_thread=reply_in_thread,
            source_page_index=card.page_index,
            is_terminal=is_terminal,
        )

    @staticmethod
    def _remap_latest_card(
        card: RenderedCard,
        *,
        page_index: int,
        total_pages: int,
    ) -> RenderedCard:
        """Move compacted output to the newest message and keep its page label truthful."""
        payload = card.to_feishu_json()
        header = payload.get("header")
        if isinstance(header, dict):
            for field in ("title", "subtitle"):
                value = header.get(field)
                if not isinstance(value, dict) or not isinstance(value.get("content"), str):
                    continue
                value["content"] = re.sub(
                    r"(?:\s*·\s*)?页\s+\d+/\d+",
                    "",
                    value["content"],
                ).strip()
            title = header.get("title")
            if isinstance(title, dict) and isinstance(title.get("content"), str):
                title["content"] = (
                    f"{title['content']} · 页 {page_index + 1}/{total_pages}"
                )

        return dataclasses.replace(
            card,
            _card_json=payload,
            structure_signature=(
                f"{card.structure_signature}:message-page:{page_index + 1}/{total_pages}"
            ),
            page_index=page_index,
            total_pages=total_pages,
        )

    def _mutate_page(
        self,
        session_id: str,
        page: PageBinding,
        card: RenderedCard,
        *,
        is_terminal: bool = False,
    ) -> MutationOutcome:
        """Apply the smallest safe mutation to one existing page."""
        if (
            is_terminal
            and page.is_streaming
            and callable(getattr(self._client, "finish_streaming_card", None))
        ):
            return self._update_page(
                session_id,
                page,
                card,
                finish_streaming=True,
            )
        if page.signature != card.structure_signature:
            return self._update_page(session_id, page, card)
        if (
            card.active_element is not None
            and sanitize_card_text_for_audit(card.active_element.text) != page.last_text
        ):
            return self._stream_element(session_id, page, card)
        return MutationOutcome(kind="skipped")

    def _freeze_page(
        self,
        session_id: str,
        page_index: int,
    ) -> MutationOutcome:
        """Finish a live CardKit page before making it immutable history."""
        binding = self._bindings.get(session_id)
        page = binding.pages.get(page_index) if binding is not None else None
        if page is None:
            return MutationOutcome(kind="skipped")
        outcome = self._mutator.finish_streaming_page(session_id, page)
        if outcome.kind != "reconcile":
            self._bindings.mark_frozen(session_id, page_index)
        return outcome

    def close(self, session_id: str) -> None:
        """Finalize a session: remove bindings, sequences, and session lock.

        Idempotent: safe to call multiple times for the same session_id.
        """
        with self._closed_lock:
            if session_id in self._closed_sessions:
                return
            self._closed_sessions.add(session_id)
        with self._trust_lock:
            self._expected_trust_revisions.pop(session_id, None)
            self._trust_snapshot_failed.discard(session_id)

        # Acquire existing session lock (or create temporary) for cleanup serialization
        session_lock = self._lock_pool.get_existing(session_id)
        if session_lock is None:
            session_lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock

        try:
            with session_lock:
                binding = self._bindings.get(session_id)
                if binding is not None:
                    for page in list(binding.pages.values()):
                        if page.is_streaming:
                            outcome = self._mutator.finish_streaming_page(session_id, page)
                            if outcome.kind == "reconcile":
                                logger.warning(
                                    "Card stream cleanup failed during close for %s: %s",
                                    page.card_id,
                                    outcome.message,
                                )
                        if page.card_id:
                            self._sequences.reset(page.card_id)
                self._bindings.remove(session_id)
        finally:
            # The per-session lock is independent of binding cleanup.  Never
            # retain it when finalizing a session, even if a mutator raises.
            self.release_session_lock(session_id)

    def get_binding(self, session_id: str):
        """Get the current binding for inspection/testing."""
        return self._bindings.get(session_id)

    # ----- Internal operations (delegated to PageMutator) -----

    def _create_page(
        self,
        session_id: str,
        chat_id: str,
        card: RenderedCard,
        *,
        reply_to: str | None = None,
        reply_in_thread: bool | None = None,
        source_page_index: int | None = None,
        is_terminal: bool = False,
    ) -> MutationOutcome:
        """Create a new card page via API."""
        outcome = self._mutator.create_page(
            session_id,
            chat_id,
            card,
            reply_to=reply_to,
            reply_in_thread=reply_in_thread,
            source_page_index=source_page_index,
        )
        if outcome.kind != "applied" or not is_terminal:
            return outcome
        binding = self._bindings.get(session_id)
        page = binding.pages.get(card.page_index) if binding is not None else None
        if (
            page is None
            or not page.is_streaming
            or not callable(getattr(self._client, "finish_streaming_card", None))
        ):
            return outcome
        return self._mutator.update_page(
            session_id,
            page,
            card,
            force_finish_streaming=True,
        )

    def _update_page(
        self,
        session_id: str,
        page: PageBinding,
        card: RenderedCard,
        *,
        finish_streaming: bool = False,
    ) -> MutationOutcome:
        """Update card structure via PATCH API."""
        return self._mutator.update_page(
            session_id,
            page,
            card,
            force_finish_streaming=finish_streaming,
        )

    def _stream_element(
        self, session_id: str, page: PageBinding, card: RenderedCard
    ) -> MutationOutcome:
        """Push text update via CardKit element_content API."""
        return self._mutator.stream_element(session_id, page, card)

    def _finalize_page(self, session_id: str, page: PageBinding) -> None:
        """Finalize a stale page."""
        self._mutator.finalize_page(session_id, page)

    @staticmethod
    def _is_recreate_outcome(outcome: MutationOutcome) -> bool:
        return outcome.kind == "reconcile" and outcome.message.startswith(_RECREATE_OUTCOME_PREFIX)

    @staticmethod
    def _is_stale_binding_outcome(outcome: MutationOutcome) -> bool:
        """Return whether a confirmed-missing binding is safe to replace now."""
        if not CardDelivery._is_recreate_outcome(outcome):
            return False
        try:
            code = int(outcome.message.removeprefix(_RECREATE_OUTCOME_PREFIX))
        except ValueError:
            return False
        return code in TransportError.RECREATE_CODES
