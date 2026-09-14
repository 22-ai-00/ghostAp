from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from ...card.events import CardEvent
from ...card.render.budget import RenderBudget
from ...card.state.models import CardMetadata
from ...card.ui_text import UI_TEXT
from ...spec_engine import SpecEngineCallbacks
from ...spec_engine.review_display import build_review_role_payloads, format_review_overview
from ._rotating_mixin import EngineView, RotatingRendererMixin
from ._spec_stream_processor import SpecStreamProcessor
from .base import BaseRenderer, EngineProfile, _StreamThrottle

if TYPE_CHECKING:
    from ...card.protocols import Dispatchable
    from ...card.session import CardSession
    from ...card.session.rotator import SessionRotator
    from ...project import ProjectContext
    from ..handlers.spec import SpecHandler

logger = logging.getLogger(__name__)


class SpecRenderer(RotatingRendererMixin, BaseRenderer):
    """
    Handles UI rendering and state management for Spec Engine interactions.
    """

    _PROFILE = EngineProfile(
        engine_type="spec",
        mode_name="Spec",
        mode_emoji="📋",
        engine_cmd="/spec",
    )

    # Backward-compat aliases used by RotatingRendererMixin
    _engine_type = _PROFILE.engine_type
    _mode_prefix = _PROFILE.mode_name
    _mode_emoji = _PROFILE.mode_emoji
    _engine_cmd = _PROFILE.engine_cmd

    def __init__(self, handler: "SpecHandler") -> None:
        super().__init__(handler)
        self._current_session: "Dispatchable | None" = None
        self._selection_sessions: dict[tuple[str, str, str], "CardSession"] = {}
        self._last_cycle: int | None = None
        self._last_perspective: str | None = None
        self._pending_split_hint: str | None = None

    def get_or_create_session(
        self, chat_id: str, project_id: str, *, reply_to: str | None = None
    ) -> "CardSession":
        """Create a lightweight Spec card session for pre-run selection UI."""
        selection_key = (chat_id, project_id, reply_to or "")
        existing = self._selection_sessions.get(selection_key)
        if existing is not None and not getattr(existing, "closed", False):
            self._current_session = existing
            return existing

        metadata = CardMetadata(
            engine_type="spec",
            mode_name="Spec Review",
            mode_emoji="📋",
            tool_name="Review Agents",
        )
        hooks = self._build_hooks(reply_to or "", chat_id=chat_id)
        session = self.create_session(
            chat_id=chat_id,
            message_id=reply_to or "",
            metadata=metadata,
            hooks=hooks,
            budget=RenderBudget(engine_cmd="/spec"),
            notify_callback=self.handler.send_text_to_chat,
        )
        self._selection_sessions[selection_key] = session
        self._current_session = session
        return session

    def get_active_session(self) -> "Dispatchable | None":
        """Return the currently active spec engine session (rotator)."""
        return self._current_session

    def notify_cycle_change(self, *, current_cycle: int, perspective: str | None) -> None:
        """Hook into the spec engine cycle/perspective lifecycle."""
        # Spec's SessionRotator owns card boundaries.  Phase/perspective changes
        # must keep updating the current cycle card; sending CARD_SPLIT to the
        # rotator terminalizes the current CardSession without rotating, which
        # makes later phase updates disappear and can fire success reactions
        # before the Spec project is complete.
        self._last_cycle = current_cycle
        self._last_perspective = perspective

    def _on_card_split_completed(self, reason: str, hint: str | None, bridge_phrase: str | None = None) -> None:
        self._pending_split_hint = hint

    def _get_reporter(self):
        return self.ctx.spec_reporter

    def get_default_ui_state(self) -> dict:
        return super().get_default_ui_state()

    def create_spec_callbacks(
        self, message_id: str, chat_id: str, project: Optional["ProjectContext"],
        engine_name: str = "Coco", model_name: str = "",
    ) -> SpecEngineCallbacks:
        self.handler.ensure_request_id(
            message_id, chat_id=chat_id, project_id=(project.project_id if project else None)
        )
        reporter = self.ctx.spec_reporter
        profile = self._PROFILE

        spec_project_id = project.project_id if project else self.handler.get_working_dir(chat_id)

        # Build metadata from profile + per-invocation dynamic values
        metadata = CardMetadata(
            engine_type=profile.engine_type,
            mode_name=profile.mode_name,
            mode_emoji=profile.mode_emoji,
            tool_name=engine_name,
            model_name=model_name or None,
            working_dir=project.root_path if project else None,
            project_name=project.project_name if project else None,
            programming_text_sections=True,
            retain_full_history=True,
        )

        hooks = self._build_hooks(message_id)
        budget = RenderBudget(engine_cmd=profile.engine_cmd)
        # Spec engine runs can take hours (review cycles, build phases).
        _spec_ttl = float(self.settings.spec_execution_timeout)
        rotator: SessionRotator = self._create_rotator(
            chat_id, message_id, metadata,
            hooks=hooks, budget=budget, ttl_seconds=_spec_ttl,
        )
        self._current_session = rotator

        _throttle = _StreamThrottle(
            min_interval=self.settings.deep_stream_interval,
            min_chars=self.settings.deep_stream_min_chars,
        )

        processor = SpecStreamProcessor(
            rotator=rotator,
            reporter=reporter,
            metadata=metadata,
            hooks=hooks,
            budget=budget,
            spec_project_id=spec_project_id,
            message_id=message_id,
            chat_id=chat_id,
            renderer=self,
            project_root_path=project.root_path if project else "",
            throttle=_throttle,
        )

        return processor.build_callbacks()

    def render_current_view(
        self,
        message_id: str,
        chat_id: str,
        project: Optional["ProjectContext"] = None,
        origin_message_id: Optional[str] = None,
    ):
        if project is None:
            project = self.handler.project_manager.get_active_project(chat_id)

        root_path = project.root_path if project else self.handler.get_working_dir(chat_id)
        snap = self.ctx.spec_engine_manager.snapshot(chat_id, root_path)

        spec_project_id = project.project_id if project else root_path
        state = self.get_ui_state(spec_project_id)

        view_mode = state.get("view_mode", "status")
        view_context = state.get("view_context", {})

        if not snap or not snap.ext.get("project"):
            snaps = self.ctx.spec_engine_manager.snapshot_active(chat_id)
            if len(snaps) == 1 and snaps[0].ext.get("project"):
                snap = snaps[0]
            else:
                engine_name = self.handler.get_engine_name(
                    chat_id, project_id=(project.project_id if project else None)
                )
                metadata = CardMetadata(
                    engine_type="spec",
                    mode_name="Spec",
                    mode_emoji="📋",
                    tool_name=engine_name,
                    working_dir=project.root_path if project else None,
                    project_name=project.project_name if project else None,
                )
                session = self.create_session(chat_id, message_id, metadata, budget=RenderBudget(engine_cmd="/spec"))
                session.dispatch(CardEvent.started())
                session.dispatch(CardEvent.text_started("_status"))
                session.dispatch(CardEvent.text_delta("_status", UI_TEXT["spec_status_empty"]))
                session.dispatch(CardEvent.text_done("_status"))
                session.dispatch(CardEvent.completed())
                return

        # Build a lightweight view object for internal methods
        engine = EngineView(snap)

        if view_mode == "status":
            self._render_status_view(message_id, chat_id, project, engine, state)
        elif view_mode == "cycle_done":
            cycle_num = view_context.get("cycle_num")
            self._render_cycle_view(message_id, chat_id, project, engine, state, cycle_num)
        elif view_mode == "review_done":
            cycle_num = view_context.get("cycle_num")
            self._render_review_view(message_id, chat_id, project, engine, state, cycle_num)
        elif view_mode == "error":
            error_msg = view_context.get("error", "未知错误")
            self._render_error_view(message_id, chat_id, project, engine, state, error_msg)
        else:
            self._render_status_view(message_id, chat_id, project, engine, state)

    def _render_cycle_view(self, message_id: str, chat_id: str, project, engine, state, cycle_num):
        reporter = self.ctx.spec_reporter
        engine_name = engine.engine_name
        spec_project = engine.project

        cycle = next((c for c in spec_project.cycles if c.cycle_number == cycle_num), None)
        if not cycle:
            self._render_status_view(message_id, chat_id, project, engine, state)
            return

        session = self._build_view_session(chat_id, message_id, engine_name, state)
        session.dispatch(CardEvent.started())

        content = reporter.format_cycle_done(cycle_num, cycle)
        session.dispatch(CardEvent.text_delta("_cycle_done", content))

        criteria_section = reporter.format_criteria_section(spec_project)
        session.dispatch(CardEvent.criteria_updated(
            criteria_section,
            satisfied_count=spec_project.satisfied_count,
            total_count=spec_project.total_criteria,
        ))

        session.dispatch(CardEvent.completed())

    def _render_review_view(self, message_id: str, chat_id: str, project, engine, state, cycle_num):
        reporter = self.ctx.spec_reporter
        engine_name = engine.engine_name
        spec_project = engine.project

        cycle = next((c for c in spec_project.cycles if c.cycle_number == cycle_num), None)
        if not cycle or not cycle.review_result:
            self._render_status_view(message_id, chat_id, project, engine, state)
            return

        session = self._build_view_session(chat_id, message_id, engine_name, state)
        session.dispatch(CardEvent.started())

        content = format_review_overview(cycle.review_result, cycle_num)
        session.dispatch(CardEvent.text_delta("_review", content))
        session.dispatch(CardEvent.review_result_updated(
            cycle_num,
            build_review_role_payloads(cycle.review_result, cycle_num),
        ))

        criteria_section = reporter.format_criteria_section(spec_project)
        session.dispatch(CardEvent.criteria_updated(
            criteria_section,
            satisfied_count=spec_project.satisfied_count,
            total_count=spec_project.total_criteria,
        ))

        warning = self.check_warning_banner(spec_project.duration(), is_executing=True)
        if warning:
            session.dispatch(CardEvent.warning_updated(warning))

        session.dispatch(CardEvent.completed())

    def build_error_card(
        self,
        *,
        project,
        engine_name: str,
        error_msg: str,
        state: Optional[dict] = None,
        project_id: Optional[str] = None,
        engine_project_id: Optional[str] = None,
        footer_note: Optional[str] = None,
        terminal_state: Optional[str] = None,
        error_details: str = "",
        detail_action: Optional[dict] = None,
        retry_action: Optional[dict] = None,
    ) -> tuple[str, str]:
        """Build error card via CardSession and return (msg_type, card_content).

        Used by SpecHandler for inline error cards.
        """
        ui_state = state or self.get_default_ui_state()
        metadata = CardMetadata(
            engine_type="spec",
            mode_name="Spec",
            mode_emoji="📋",
            tool_name=engine_name,
            compact=ui_state.get("compact", False),
            expanded=ui_state.get("expanded", False),
            expand_ac=ui_state.get("expand_ac", False),
            working_dir=project.root_path if project else None,
            project_name=project.project_name if project else None,
        )
        # Use a snapshot-only session (no TTL timer, no delivery)
        factory = self._get_session_factory()
        session = factory.create_snapshot(metadata=metadata)
        session.dispatch(CardEvent.started())
        session.dispatch(
            CardEvent.failed(
                error_msg,
                details=error_details,
                detail_action=detail_action,
                retry_action=retry_action,
            )
        )

        # Snapshot returns the rendered card
        result = session.snapshot()
        if result:
            msg_type, card_content = result
            return (msg_type, card_content)
        return ("interactive", "{}")
