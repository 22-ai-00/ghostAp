from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from ...card.events import CardEvent
from ...card.render.budget import RenderBudget
from ...card.state.models import CardMetadata
from ...card.ui_text import UI_TEXT
from ...deep_engine import DeepEngineCallbacks
from ...deep_engine.models import DeepProjectStatus
from ...project import ContextSourceMode
from ...utils.text import summarize_question_title
from ._deep_stream_processor import DeepStreamProcessor
from .base import BaseRenderer, EngineProfile

if TYPE_CHECKING:
    from ...card.protocols import Dispatchable
    from ...project import ProjectContext
    from ..handlers.deep import DeepHandler

logger = logging.getLogger(__name__)

class DeepRenderer(BaseRenderer):
    """
    Handles UI rendering and state management for Deep Engine interactions.
    Separated from DeepHandler to improve maintainability.
    """

    _PROFILE = EngineProfile(
        engine_type="deep",
        mode_name="Deep",
        mode_emoji="🧠",
        engine_cmd="/deep",
        include_context_hook=True,
    )

    def __init__(self, handler: "DeepHandler") -> None:
        super().__init__(handler)
        self._current_session: "Dispatchable | None" = None

    def get_active_session(self) -> "Dispatchable | None":
        """Return the currently active deep engine session."""
        return self._current_session

    def create_deep_callbacks(
        self,
        message_id: str,
        chat_id: str,
        project: Optional["ProjectContext"],
        engine_name: str = "Coco",
        root_path: Optional[str] = None,
        initial_message_id: Optional[str] = None,
        requirement_text: str | None = None,
    ) -> DeepEngineCallbacks:
        self.handler.ensure_request_id(
            message_id, chat_id=chat_id, project_id=(project.project_id if project else None)
        )
        # Build lifecycle hooks for side effects (emoji, context persistence)
        context_update_fn = None
        if project:
            def context_update_fn(state):
                """Persist deep engine results to project context."""
                self.handler.context_manager.update_context(
                    project.project_id,
                    deep_result={"data": {"completed": True}},
                    chat_id=chat_id,
                )
                ctx = self.handler.context_manager.store.get(project.project_id, chat_id=chat_id)
                if ctx:
                    ctx.create_version(
                        reason="deep_engine_done",
                        source_mode=ContextSourceMode.DEEP_ENGINE,
                        summary="Deep Engine completed",
                    )

        rotator = self._setup_engine_session(
            self._PROFILE,
            chat_id=chat_id,
            message_id=message_id,
            engine_name=engine_name,
            project=project,
            context_update_fn=context_update_fn,
            question_title=summarize_question_title(requirement_text),
        )

        return DeepStreamProcessor(
            rotator=rotator,
            renderer=self,
            message_id=message_id,
            chat_id=chat_id,
            root_path=root_path,
            project=project,
        ).build_callbacks()

    def _on_card_split_completed(self, reason: str, hint: str | None, bridge_phrase: str | None = None) -> None:
        self._pending_split_hint = hint

    def _get_engine(self, chat_id: str, root_path: Optional[str], project: Optional["ProjectContext"]):
        """Helper to get the deep engine instance via snapshot interface."""
        rp = root_path or (project.root_path if project else "")
        if rp:
            snap = self.ctx.deep_engine_manager.snapshot(chat_id, rp)
            if snap:
                return snap
        try:
            snaps = self.ctx.deep_engine_manager.snapshot_active(chat_id)
            if len(snaps) == 1:
                return snaps[0]
        except Exception:
            logger.debug("failed to get running engine snapshot", exc_info=True)
        return None

    def render_deep_status(
        self,
        message_id: str,
        chat_id: str,
        project: Optional["ProjectContext"] = None,
        origin_message_id: Optional[str] = None,
    ):
        if project is None:
            project = self.handler.project_manager.get_active_project(chat_id)

        root_path = project.root_path if project else self.handler.get_working_dir(chat_id)
        snap = self.ctx.deep_engine_manager.snapshot(chat_id, root_path)
        reporter = self.ctx.progress_reporter

        if not snap or not snap.ext.get("project"):
            snaps = self.ctx.deep_engine_manager.snapshot_active(chat_id)
            if len(snaps) == 1 and snaps[0].ext.get("project"):
                snap = snaps[0]
            elif len(snaps) > 1:
                self.handler.show_deep_board(message_id, chat_id)
                return
            else:
                # No active engine — build a simple status card via CardSession snapshot
                engine_name = self.handler.get_engine_name(
                    chat_id, project_id=(project.project_id if project else None)
                )
                metadata = CardMetadata(
                    project_name=project.project_id if project else None,
                    mode_name="Deep",
                    mode_emoji="🧠",
                    engine_type="deep",
                    tool_name=engine_name,
                    working_dir=project.root_path if project else None,
                )
                session = self.create_session(chat_id, message_id, metadata, budget=RenderBudget(engine_cmd="/deep"))
                session.dispatch(CardEvent.started())
                session.dispatch(CardEvent.text_started("_status"))
                session.dispatch(CardEvent.text_delta("_status", UI_TEXT["deep_status_empty"]))
                session.dispatch(CardEvent.text_done("_status"))
                session.dispatch(CardEvent.completed())
                # Card was auto-delivered by the session
                return

        engine_name = snap.engine_name
        deep_project = snap.ext.get("project")
        snap.ext.get("progress")

        if project is None:
            try:
                project = self.handler.project_manager.find_project_by_path(snap.root_path, chat_id=chat_id)
            except Exception:
                project = None

        status_content = reporter.format_status(deep_project)
        status_title = reporter.get_status_title()
        progress_info = reporter.get_progress_info(
            deep_project,
            completed=snap.completed_steps,
            total=snap.total_steps,
        )

        deep_project_id = progress_info["project_id"]
        state = self.get_ui_state(deep_project_id) if deep_project_id else self.get_default_ui_state()

        # Build status card via CardSession
        metadata = CardMetadata(
            project_name=project.project_id if project else None,
            mode_name="Deep",
            mode_emoji="🧠",
            engine_type="deep",
            tool_name=engine_name,
            working_dir=project.root_path if project else None,
        )
        session = self.create_session(chat_id, message_id, metadata, budget=RenderBudget(engine_cmd="/deep"))
        session.dispatch(CardEvent.started())

        # Apply collapsing to status content
        status_content = self._render_collapsible_section(
            status_content, total_items=len(status_content.split("\n")), expanded=state.get("expand_ac", False)
        )

        # Dispatch content
        session.dispatch(CardEvent.text_delta("_status", f"**{status_title}**\n\n{status_content}"))

        # Progress
        if progress_info.get("progress_bar"):
            session.dispatch(CardEvent.progress_updated(
                current=progress_info.get("completed", 0),
                total=progress_info.get("total", 0),
                label=UI_TEXT["deep_progress_executing"] if progress_info["is_executing"] else UI_TEXT["deep_progress_done"],
            ))

        # Terminal state
        if deep_project.status == DeepProjectStatus.COMPLETED:
            session.dispatch(CardEvent.completed())
        elif deep_project.status == DeepProjectStatus.CANCELLED:
            session.dispatch(CardEvent.cancelled(reason="cancelled"))
        elif deep_project.status == DeepProjectStatus.FAILED:
            session.dispatch(CardEvent.failed(deep_project.error or "Deep 执行失败"))
        # If still executing, leave session open (card delivered via dispatch)
