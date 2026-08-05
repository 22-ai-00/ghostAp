from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from src.card.render.throttle import StreamThrottle
from src.card.shared.truncation import check_and_truncate_payload as _check_and_truncate
from src.card.thresholds import THRESHOLDS

if TYPE_CHECKING:
    from ...card.protocols import Dispatchable
    from ...card.render.budget import RenderBudget
    from ...card.session import CardSession
    from ...card.session.rotator import SessionRotator
    from ..handlers.base import BaseHandler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stream throttle (re-exported for backward compat; canonical at src.card.render.throttle)
# ---------------------------------------------------------------------------

_StreamThrottle = StreamThrottle


class EngineRenderStrategy:
    """Shared facade for engine renderer composition policies.

    Keeps split/continuation wording and terminal dispatch helpers in one
    place so Deep and Spec renderers use the same card composition semantics
    without changing their public rendering behavior.
    """

    DEFAULT_BRIDGE_PHRASE = "续接："

    def __init__(self, renderer: "BaseRenderer") -> None:
        self.renderer = renderer

    def dispatch_card_split(
        self,
        session,
        *,
        reason: str,
        hint: str | None = None,
        bridge_phrase: str | None = None,
    ) -> None:
        self.renderer._dispatch_card_split(
            session,
            reason=reason,
            hint=hint,
            bridge_phrase=bridge_phrase or self.DEFAULT_BRIDGE_PHRASE,
        )

    def dispatch_terminal(self, session, *, completed: bool, summary: str | None = None, error: str | None = None) -> None:
        from ...card.events import CardEvent

        if completed:
            session.dispatch(CardEvent.completed(summary=summary))
        else:
            session.dispatch(CardEvent.failed(error or ""))


def _dispatch_text_block(dispatchable, block_id: str, content: str) -> None:
    """Dispatch a completed text block when content is non-empty."""
    if not content:
        return
    from ...card.events import CardEvent

    dispatchable.dispatch(CardEvent.text_started(block_id))
    dispatchable.dispatch(CardEvent.text_delta(block_id, content))
    dispatchable.dispatch(CardEvent.text_done(block_id))


class _ACPStreamBridge:
    """Normalize ACP streaming into programming-mode-like card blocks.

    Reused by Deep/Spec renderers so text, reasoning and tool panels follow
    the same event sequencing as direct programming mode.

    DEPRECATED: Use src.card.stream_bridge.ACPStreamBridge directly.
    This class is kept for backward-compatible imports only.
    """

    def __init__(self, dispatchable) -> None:
        from src.card.stream_bridge import ACPStreamBridge as _Impl
        self._impl = _Impl(dispatchable)

    def bind(self, dispatchable) -> None:
        self._impl.bind(dispatchable)

    def on_event(self, acp_event) -> None:
        self._impl.on_event(acp_event)

    def close_open_blocks(self) -> None:
        self._impl.close_open_blocks()


@dataclass(frozen=True)
class EngineProfile:
    """Declarative engine identity — define once, reuse for session creation.

    Captures the static parts of CardMetadata + RenderBudget that are identical
    across all invocations of an engine type.  Dynamic per-invocation values
    (tool_name, working_dir, project_name, model_name) are passed separately
    to ``BaseRenderer._setup_engine_session``.
    """

    engine_type: str       # "deep" | "spec"
    mode_name: str         # "Deep" | "Spec"
    mode_emoji: str        # "🧠" | "📋"
    engine_cmd: str        # "/deep" | "/spec"
    include_context_hook: bool = False
    success_emoji: str = ""
    ttl_seconds: float | None = None


class BaseRenderer:
    """
    Base class for renderers handling UI state and message sending.
    """

    def __init__(self, handler: "BaseHandler") -> None:
        self.handler = handler
        self.ctx = handler.ctx
        self.settings = handler.settings
        # project_id -> state dict
        self.ui_states: dict[str, dict[str, Any]] = {}
        self._session_factory = None
        self.render_strategy = EngineRenderStrategy(self)

    def get_active_session(self) -> "Dispatchable | None":
        """Return the currently active session for this renderer, or None.

        Subclasses should override to expose their tracked session reference.
        Used by BaseEngineHandler._on_engine_error() to route error events
        through the session pipeline for full hook lifecycle support.
        """
        return None

    def _dispatch_card_split(
        self,
        session,
        *,
        reason: str,
        hint: str | None = None,
        bridge_phrase: str | None = None,
    ) -> None:
        """Dispatch a card_split event and wire completion callback.

        Subclasses may override ``_on_card_split_completed`` to create a fresh
        continuation session after the current card has closed successfully.
        """
        from ...card.events import CardEvent

        normalized_hint = hint or ""
        session.on_card_split_completed = self._on_card_split_completed
        session.dispatch(CardEvent.card_split(reason=reason, hint=normalized_hint, bridge_phrase=bridge_phrase))

    def _on_card_split_completed(self, reason: str, hint: str | None, bridge_phrase: str | None = None) -> None:
        """Hook for subclasses to start a fresh continuation session."""
        return None

    def build_unit_metadata(
        self,
        metadata,
        *,
        unit_id: str | None = None,
        unit_kind: str | None = None,
        unit_label: str | None = None,
        iteration_index: int | None = None,
        iteration_total: int | None = None,
        continuation_seq: int | None = None,
    ):
        """Clone base metadata with iteration/cycle-scoped labeling."""
        changes: dict[str, Any] = {}
        if unit_id is not None:
            changes["unit_id"] = unit_id
        if unit_kind is not None:
            changes["unit_kind"] = unit_kind
        if unit_label is not None:
            changes["unit_label"] = unit_label
        if iteration_index is not None:
            changes["iteration_index"] = iteration_index
        if iteration_total is not None:
            changes["iteration_total"] = iteration_total
        if continuation_seq is not None:
            changes["continuation_seq"] = continuation_seq
            changes["card_sequence"] = continuation_seq + 1
        return replace(metadata, **changes) if changes else metadata

    def _build_hooks(
        self,
        message_id: str,
        *,
        include_context_hook: bool = False,
        context_update_fn=None,
        chat_id: str | None = None,
        engine_type: str | None = None,
        success_emoji: str | None = None,
    ) -> tuple:
        """Build standard lifecycle hooks for engine sessions.

        Args:
            message_id: Message to react to with emoji.
            include_context_hook: When True, context_update_fn/chat_id/engine_type
                are required and a ContextPersistenceHook will be added.
            context_update_fn: Optional callable(state) for context persistence on completion.
            chat_id: Chat ID for failure notifications (used by ContextPersistenceHook).
            engine_type: Engine type for dynamic command hints in failure messages.

        Returns:
            Tuple of SessionHook instances.
        """
        from ...card.hooks import ContextPersistenceHook, EmojiHook

        hooks = [
            EmojiHook(
                add_reaction=self.handler.add_reaction,
                message_id=message_id,
                chat_id=chat_id,
                success_emoji=success_emoji,
            )
        ]
        if include_context_hook or context_update_fn:
            hooks.append(ContextPersistenceHook(
                update_fn=context_update_fn,
                notify_callback=self.handler.send_text_to_chat,
                chat_id=chat_id,
                engine_type=engine_type,
            ))
        return tuple(hooks)

    def create_session(
        self,
        chat_id: str,
        message_id: str,
        metadata=None,
        *,
        session_id: str | None = None,
        hooks: tuple = (),
        budget: "RenderBudget | None" = None,
        action_registry: "dict | None" = None,
        notify_callback=None,
        cancel_callback=None,
        ttl_seconds: float | None = None,
    ) -> "CardSession":
        """Create a CardSession via the unified factory path.

        This is the single session creation entry point for all engine renderers.

        Args:
            action_registry: Optional dict mapping action_id → handler callable.
                If provided, passed to factory.create() for button click routing.
            notify_callback: Optional (chat_id, text) callable for OOB notifications.
            cancel_callback: Optional () callable for terminal cancellation cleanup.
        """
        from ...card.actions.dispatch import build_common_action_registry
        from ...card.session.config import SessionCallbacks
        from ...card.state.models import CardMetadata

        meta = metadata if isinstance(metadata, CardMetadata) else CardMetadata()
        # Merge common actions (mode toggle, engine stop) with any engine-specific registry
        merged_registry = build_common_action_registry()
        if action_registry:
            merged_registry.update(action_registry)
        cbs = SessionCallbacks(
            notify_callback=notify_callback,
            cancel_callback=cancel_callback,
            # Fail-close: factory requires at least one notification channel.
            # reply_text_fn is the most universally safe default for Feishu handlers.
            reply_text_fn=self.handler.reply_text,
            action_registry=merged_registry,
            hooks=hooks,
        )
        kwargs: dict[str, Any] = dict(
            chat_id=chat_id,
            metadata=meta,
            session_id=session_id,
            reply_to=message_id,
            callbacks=cbs,
            budget=budget,
        )
        if ttl_seconds is not None:
            kwargs["ttl_seconds"] = ttl_seconds

        # Delegate to factory (retry logic is handled inside factory.create())
        try:
            return self._get_session_factory().create(**kwargs)
        except Exception:
            logger.exception("Failed to create CardSession, falling back to text reply")
            self.handler.reply_text(message_id, "当前使用人数较多，请稍后重试，或重新发送命令")
            raise

    def _get_session_factory(self):
        """Return a CardSessionFactory bound to this handler's delivery.

        Uses lazy caching: creates factory on first call, reuses thereafter.
        Call _invalidate_session_factory() if the delivery instance changes.
        """
        if self._session_factory is None:
            from ...card.session.factory import CardSessionFactory
            delivery = self.handler.get_card_delivery()
            self._session_factory = CardSessionFactory(delivery)
        return self._session_factory

    def _invalidate_session_factory(self) -> None:
        """Invalidate cached factory (e.g. after delivery reconnect)."""
        self._session_factory = None

    def get_default_ui_state(self) -> dict[str, Any]:
        """
        Return the default UI state dictionary.
        Subclasses should override this to provide specific defaults.
        """
        return {
            "compact": False,
            "expanded": False,
            "expand_ac": False,
            "view_mode": "status",
            "view_context": {},
        }

    def get_ui_state(self, project_id: str) -> dict[str, Any]:
        """
        Get the UI state for a specific project.
        Initializes with defaults if not present.
        """
        if not project_id:
            return self.get_default_ui_state()

        if project_id not in self.ui_states:
            self.ui_states[project_id] = self.get_default_ui_state()

        return self.ui_states[project_id]

    def update_ui_state(self, project_id: str, **kwargs) -> None:
        """Update specific fields in the UI state."""
        state = self.get_ui_state(project_id)
        state.update(kwargs)

    def check_warning_banner(self, duration: float, is_executing: bool = True) -> str | None:
        if not is_executing:
            return None
        timeout_raw = getattr(self.settings, "engine_timeout_warning_seconds", 0)
        duration_s = duration if isinstance(duration, (int, float)) else 0
        timeout_s = timeout_raw if isinstance(timeout_raw, (int, float)) else 0
        if timeout_s > 0 and duration_s > timeout_s:
            return "执行耗时较长，若无响应可尝试停止后重试"
        return None

    def _render_collapsible_section(
        self, content: str, total_items: int, expanded: bool, completed_count: int = 0
    ) -> str:
        """
        Render a section that can be collapsed if too long.
        Generic version shared by engine renderers.

        Args:
            content: The full content string (e.g. list of ACs, or long text)
            total_items: Total number of items (or approx lines/paragraphs)
            expanded: Whether the section is currently expanded
            completed_count: Number of completed items (for AC lists), used to generate summary
        """
        if not content or total_items == 0:
            return content

        # Threshold for collapsing
        COLLAPSE_THRESHOLD = THRESHOLDS["COLLAPSE_ITEM_THRESHOLD"]

        # If few items or expanded, show all
        if total_items <= COLLAPSE_THRESHOLD or expanded:
            return content

        # Folding logic: Filter out completed items or truncate text
        # Simple text processing approach assuming list format with checkmarks
        lines = content.split("\n")
        kept_lines = []
        hidden_count = 0

        for line in lines:
            if "✅" in line:
                hidden_count += 1
            else:
                kept_lines.append(line)

        # If we couldn't identify completed items by checkmark, but it's long text (Spec mode)
        # We might want to just truncate
        if hidden_count == 0 and len(lines) > THRESHOLDS["COLLAPSE_LINE_THRESHOLD"]:  # Long text fallback
            summary = f"📄 内容较长 (共 {len(lines)} 行)，点击下方'展开'查看全部"
            return f"{summary}\n\n" + "\n".join(lines[:THRESHOLDS["COLLAPSE_DISPLAY_LINES"]]) + "\n..."

        if hidden_count == 0:
            return content

        # Add summary of hidden items
        summary = f"✅ 已通过 {hidden_count} 项 (点击下方'展开'查看全部)"

        final_lines = []
        inserted = False
        for line in kept_lines:
            final_lines.append(line)
            # Try to insert summary after a header if present
            if ("验收标准" in line or "Criteria" in line) and not inserted:
                final_lines.append(summary)
                inserted = True

        if not inserted:
            # If header not found, prepend
            final_lines.insert(0, summary)

        return "\n".join(final_lines)

    def _check_and_truncate_payload(
        self, card_content: str, max_size: int | None = None, *, engine_type: str | None = None
    ) -> str:
        """Check if card content exceeds size limit and truncate if necessary.

        Delegates to src.card.render.payload_truncator.
        """
        return _check_and_truncate(card_content, max_size, engine_type=engine_type)

    def _create_rotator(
        self,
        chat_id: str,
        message_id: str,
        metadata,
        *,
        hooks: tuple = (),
        budget=None,
        ttl_seconds: float | None = None,
    ):
        """Create a SessionRotator wrapping a new CardSession.

        Template method used by SpecRenderer to eliminate
        duplicated SessionRotator instantiation boilerplate.
        """
        from ...card.session.rotator import SessionRotator

        session = self.create_session(
            chat_id, message_id, metadata, hooks=hooks, budget=budget,
            ttl_seconds=ttl_seconds,
        )
        return SessionRotator(session)

    def _setup_engine_session(
        self,
        profile: "EngineProfile",
        *,
        chat_id: str,
        message_id: str,
        engine_name: str = "",
        model_name: str = "",
        project=None,
        context_update_fn=None,
        ttl_seconds: float | None = None,
        question_title: str | None = None,
    ) -> "SessionRotator":
        """Create a full session from an EngineProfile in one call.

        Replaces the repeated 5-step boilerplate (metadata → hooks → budget →
        rotator → assign session) in each renderer's create_*_callbacks method.

        Args:
            ttl_seconds: Override profile.ttl_seconds if provided.

        Returns:
            A SessionRotator ready to dispatch events.
        """
        from ...card.render.budget import RenderBudget
        from ...card.state.models import CardMetadata

        metadata = CardMetadata(
            engine_type=profile.engine_type,
            mode_name=profile.mode_name,
            mode_emoji=profile.mode_emoji,
            tool_name=engine_name or None,
            model_name=model_name or None,
            working_dir=project.root_path if project else None,
            question_title=question_title,
            project_name=(
                getattr(project, "project_name", None)
                or (project.project_id if project else None)
            ),
        )

        hooks = self._build_hooks(
            message_id,
            include_context_hook=profile.include_context_hook,
            context_update_fn=context_update_fn,
            chat_id=chat_id,
            engine_type=profile.engine_type,
            success_emoji=profile.success_emoji or None,
        )

        budget = RenderBudget(engine_cmd=profile.engine_cmd)

        effective_ttl = ttl_seconds if ttl_seconds is not None else profile.ttl_seconds
        rotator = self._create_rotator(
            chat_id, message_id, metadata,
            hooks=hooks, budget=budget, ttl_seconds=effective_ttl,
        )
        self._current_session = rotator
        return rotator

    def _render_empty_status(self, engine_cmd: str) -> str:
        """Render a default empty-status placeholder when no engine snapshot is available.

        Template method — subclasses may override for engine-specific wording.
        """
        return f"等待 {engine_cmd} 引擎启动…"
