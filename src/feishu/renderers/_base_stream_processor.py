"""BaseStreamProcessor — shared lifecycle logic for ACP-driven card stream processors.

Extracts the common patterns from DeepStreamProcessor and SpecStreamProcessor:
- ACPStreamBridge + ACPEventRenderer creation
- _started_dispatched idempotent guard
- Warning banner check + dispatch
- Terminal lifecycle: completed / failed + renderer cleanup
- Phase transition helpers

Subclasses (Deep/Spec) provide engine-specific event handling, task tracking,
and multi-cycle/multi-phase logic.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ...acp import ACPEventRenderer
from ...card.events import CardEvent
from ...card.stream_bridge import ACPStreamBridge

if TYPE_CHECKING:
    from ...card.session.rotator import SessionRotator
    from .base import BaseRenderer


class BaseStreamProcessor:
    """Shared lifecycle logic for Deep/Spec StreamProcessor classes.

    Provides common methods that both processors use identically.
    Does NOT enforce method override patterns — subclasses call these
    helpers as needed within their own callback implementations.
    """

    def __init__(
        self,
        *,
        rotator: "SessionRotator",
        renderer: "BaseRenderer",
        message_id: str,
        chat_id: str,
    ) -> None:
        self._rotator = rotator
        self._renderer = renderer
        self._message_id = message_id
        self._chat_id = chat_id
        self._acp_renderer = ACPEventRenderer()
        image_uploader = getattr(renderer.handler, "upload_acp_image", None)
        self._image_uploader = image_uploader if callable(image_uploader) else None
        self._stream_bridge = ACPStreamBridge(
            rotator,
            image_uploader=self._image_uploader,
        )
        self._started_dispatched = False
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # Shared: started guard
    # ------------------------------------------------------------------

    def _dispatch_started_once(self) -> bool:
        """Dispatch CardEvent.started() at most once. Returns True if dispatched now."""
        if self._started_dispatched:
            return False
        self._started_dispatched = True
        self._rotator.dispatch(CardEvent.started())
        return True

    # ------------------------------------------------------------------
    # Shared: phase transition helpers
    # ------------------------------------------------------------------

    def _dispatch_phase_transition(
        self,
        *,
        cycle: int,
        from_phase: str,
        to_phase: str,
        done_content: str,
        started_subtitle: str,
        started_content: str,
        done_subtitle: str | None = None,
    ) -> None:
        """Dispatch phase_done for current phase + phase_started for next phase."""
        self._rotator.dispatch(CardEvent.phase_done(
            cycle, from_phase, done_content, subtitle=done_subtitle,
        ))
        self._rotator.dispatch(CardEvent.phase_started(
            cycle, to_phase, subtitle=started_subtitle, content=started_content,
        ))

    # ------------------------------------------------------------------
    # Shared: warning banner
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Shared: terminal lifecycle
    # ------------------------------------------------------------------

    def _dispatch_failed(
        self,
        error: str,
        *,
        duration_seconds: float | None = None,
    ) -> None:
        """Close open blocks, dispatch failed event, clear renderer session."""
        self._stream_bridge.close_open_blocks()
        self._rotator.dispatch(CardEvent.failed(
            error,
            duration_seconds=duration_seconds,
        ))
        self._renderer._current_session = None
