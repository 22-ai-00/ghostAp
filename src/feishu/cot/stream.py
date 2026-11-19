"""Multi-segment Feishu COT stream wrapper with bounded segment rollover."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Literal

from src.card.events import CardEvent, CardEventType
from src.card.protocols import ProcessSegmentRollover

from .api_client import FeishuCOTAPIClient
from .session import (
    _DEFAULT_FLUSH_INTERVAL,
    _DEFAULT_QUEUE_CAPACITY,
    FeishuCOTSession,
)
from .wire import COTTransportError

logger = logging.getLogger(__name__)

_DEFAULT_MAX_STREAM_SEGMENTS = 32


class FeishuCOTStream:
    """Task-scoped process stream composed of bounded native COT segments.

    Task duration does not rotate the stream. ``ProgrammingCardSession`` asks
    for a rollover only when its current-segment replay buffer reaches the
    existing event/byte safety budget. A successful seal makes that old segment
    authoritative for its own process slice, then a fresh COT continues the same
    parent task without replaying ACP work.
    """

    def __init__(
        self,
        api: FeishuCOTAPIClient,
        *,
        chat_id: str,
        origin_message_id: str | None,
        reply_in_thread: bool = False,
        input_text: str = "",
        detail: Literal["brief", "detailed"] = "detailed",
        flush_interval: float = _DEFAULT_FLUSH_INTERVAL,
        queue_capacity: int = _DEFAULT_QUEUE_CAPACITY,
        request_timeout: float | None = None,
        close_timeout: float | None = None,
        max_segments: int = _DEFAULT_MAX_STREAM_SEGMENTS,
        on_segment_started: Callable[[str], None] | None = None,
    ) -> None:
        if (
            isinstance(max_segments, bool)
            or not isinstance(max_segments, int)
            or max_segments <= 0
        ):
            raise ValueError("max_segments must be a positive integer")
        if on_segment_started is not None and not callable(on_segment_started):
            raise TypeError("on_segment_started must be callable")

        self._api = api
        self._chat_id = chat_id
        self._origin_message_id = origin_message_id
        self._reply_in_thread = reply_in_thread
        self._input_text = input_text
        self._detail = detail
        self._flush_interval = flush_interval
        self._queue_capacity = queue_capacity
        self._request_timeout = request_timeout
        self._close_timeout = close_timeout
        self._max_segments = max_segments
        self._on_segment_started = on_segment_started
        self._lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        self._segment_index = 1
        self._message_ids: list[str] = []
        self._active_anchors: dict[tuple[str, str], CardEvent] = {}
        self._started = False
        self._closed = False
        self._failed = False
        self._current_sealed = False
        # Construct eagerly so invalid public arguments fail before the main
        # card begins delivery, matching the old single-session contract.
        self._current = self._build_segment(self._segment_index)

    @property
    def started(self) -> bool:
        """Whether the stream has an active unclosed COT segment."""
        with self._lock:
            return self._started and not self._closed

    @property
    def healthy(self) -> bool:
        """Whether the active COT segment is started, open, and healthy."""
        with self._lock:
            return (
                self._started
                and not self._closed
                and not self._failed
                and self._current.healthy
            )

    @property
    def message_id(self) -> str | None:
        """Return the current segment's bound COT message ID, if any."""
        with self._lock:
            return self._current.message_id

    @property
    def message_ids(self) -> tuple[str, ...]:
        """Return all started COT segment message IDs in order."""
        with self._lock:
            return tuple(self._message_ids)

    def start(self) -> None:
        """Start the initial COT segment and record its bound message ID."""
        with self._lock:
            if self._started and not self._closed:
                return
            if self._closed or self._failed:
                raise COTTransportError("COT stream is closed")
            segment = self._current
            segment.start()
            self._started = True
            self._remember_started_segment_locked(segment)

    def emit(self, event: CardEvent) -> bool:
        """Forward one process event to the active segment and track open anchors."""
        with self._lock:
            if not self.healthy:
                return False
            handled = self._current.emit(event)
            if handled:
                self._update_active_anchors_locked(event)
            return handled

    def rollover(self) -> ProcessSegmentRollover:
        """Seal the active segment and open a fresh continuation segment."""
        with self._lock:
            if not self.healthy:
                return ProcessSegmentRollover(sealed=False, started=False)
            current = self._current
            continuation_anchors = tuple(self._active_anchors.values())
            for closure in self._segment_closure_events_locked():
                if not current.emit(closure) or not current.healthy:
                    self._failed = True
                    self._started = False
                    return ProcessSegmentRollover(sealed=False, started=False)
            if not current.seal_segment():
                self._failed = True
                self._started = False
                return ProcessSegmentRollover(sealed=False, started=False)

            # Segment N is now durably sealed on Feishu. Every subsequent
            # segment limit or continuation failure must report sealed=True so
            # the main card replays only open continuation anchors.
            self._current_sealed = True

            if self._segment_index >= self._max_segments:
                self._failed = True
                self._started = False
                logger.warning(
                    "Feishu COT segment limit reached after sealing segment=%d "
                    "max_segments=%d",
                    self._segment_index,
                    self._max_segments,
                )
                return ProcessSegmentRollover(
                    sealed=True,
                    started=False,
                    replay_events=continuation_anchors,
                )

            next_index = self._segment_index + 1
            next_segment: FeishuCOTSession | None = None
            try:
                next_segment = self._build_segment(next_index)
                next_segment.start()
            except Exception:
                self._abort_segment_safely(next_segment)
                self._failed = True
                self._started = False
                logger.warning(
                    "Feishu COT continuation segment failed to start; segment=%d",
                    next_index,
                    exc_info=True,
                )
                return ProcessSegmentRollover(
                    sealed=True,
                    started=False,
                    replay_events=continuation_anchors,
                )

            if not next_segment.started or not next_segment.healthy:
                self._abort_segment_safely(next_segment)
                self._failed = True
                self._started = False
                logger.warning(
                    "Feishu COT continuation segment unhealthy after start; segment=%d",
                    next_index,
                )
                return ProcessSegmentRollover(
                    sealed=True,
                    started=False,
                    replay_events=continuation_anchors,
                )

            if next_segment.message_id is not None:
                self._remember_started_segment_locked(
                    next_segment,
                    segment_index=next_index,
                )

            for anchor in continuation_anchors:
                if not next_segment.emit(anchor) or not next_segment.healthy:
                    self._abort_segment_safely(next_segment)
                    self._failed = True
                    self._started = False
                    logger.warning(
                        "Feishu COT continuation anchors were rejected; segment=%d",
                        next_index,
                    )
                    return ProcessSegmentRollover(
                        sealed=True,
                        started=False,
                        replay_events=continuation_anchors,
                    )

            self._current = next_segment
            self._current_sealed = False
            self._segment_index = next_index
            return ProcessSegmentRollover(
                sealed=True,
                started=True,
                replay_events=continuation_anchors,
            )

    def complete(self, event: CardEvent) -> bool:
        """Complete the current segment with the parent task terminal event."""
        with self._lock:
            if not self._started or self._closed:
                return False
            completed = self._current.complete(event)
            self._closed = True
            self._started = False
            if not completed:
                self._failed = True
            return completed

    def abort(self) -> None:
        """Abort the active unsealed segment and close the stream."""
        with self._lock:
            if self._closed:
                return
            if not self._current_sealed:
                self._abort_segment_safely(self._current)
            self._closed = True
            self._started = False
            self._failed = True

    @staticmethod
    def _abort_segment_safely(segment: FeishuCOTSession | None) -> None:
        if segment is None:
            return
        try:
            segment.abort()
        except Exception:
            logger.warning(
                "Failed to abort Feishu COT segment",
                exc_info=True,
            )

    def _build_segment(self, index: int) -> FeishuCOTSession:
        input_text = (
            self._input_text
            if index == 1
            else f"过程续接 · 第 {index} 段"
        )
        return FeishuCOTSession(
            self._api,
            chat_id=self._chat_id,
            origin_message_id=self._origin_message_id,
            reply_in_thread=self._reply_in_thread,
            input_text=input_text,
            detail=self._detail,
            flush_interval=self._flush_interval,
            queue_capacity=self._queue_capacity,
            request_timeout=self._request_timeout,
            close_timeout=self._close_timeout,
        )

    def _remember_started_segment_locked(
        self,
        segment: FeishuCOTSession,
        *,
        segment_index: int | None = None,
    ) -> None:
        message_id = segment.message_id
        if message_id is None:
            return
        self._message_ids.append(message_id)
        callback = self._on_segment_started
        if callback is None:
            return
        try:
            callback(message_id)
        except Exception:
            logger.warning(
                "Feishu COT segment-link callback failed; segment=%d",
                self._segment_index if segment_index is None else segment_index,
                exc_info=True,
            )

    def _update_active_anchors_locked(self, event: CardEvent) -> None:
        block_id = event.payload.get("block_id")
        if not isinstance(block_id, str) or not block_id:
            return
        event_type = event.type
        if event_type is CardEventType.TEXT_STARTED:
            self._active_anchors[("text", block_id)] = event
        elif event_type is CardEventType.TEXT_DONE:
            self._active_anchors.pop(("text", block_id), None)
        elif event_type is CardEventType.REASONING_STARTED:
            self._active_anchors[("reasoning", block_id)] = event
        elif event_type is CardEventType.REASONING_DONE:
            self._active_anchors.pop(("reasoning", block_id), None)
        elif event_type is CardEventType.TOOL_STARTED:
            self._active_anchors[("tool", block_id)] = event
        elif event_type in {CardEventType.TOOL_DONE, CardEventType.TOOL_FAILED}:
            self._active_anchors.pop(("tool", block_id), None)

    def _segment_closure_events_locked(self) -> tuple[CardEvent, ...]:
        """Balance open AG-UI blocks before RUN_FINISHED for one segment."""
        closures: list[CardEvent] = []
        for (kind, block_id), _anchor in self._active_anchors.items():
            if kind == "text":
                closures.append(CardEvent.text_done(block_id))
            elif kind == "reasoning":
                closures.append(CardEvent.reasoning_done(block_id))
            elif kind == "tool":
                closures.append(
                    CardEvent(
                        type=CardEventType.TOOL_DONE,
                        payload={
                            "block_id": block_id,
                            "tool_output": "工具仍在执行；结果续接下一段。",
                        },
                    )
                )
        return tuple(closures)
