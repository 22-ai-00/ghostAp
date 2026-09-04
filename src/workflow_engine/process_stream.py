"""Workflow process projection onto one optional native process stream."""

from __future__ import annotations

import json
import logging
import threading

from src.card.events import CardEvent
from src.card.protocols import ProcessEventSink

logger = logging.getLogger(__name__)

_SEGMENT_MAX_EVENTS = 2_048
_SEGMENT_MAX_BYTES = 2 * 1024 * 1024


class WorkflowProcessStream:
    """Route every direct Agent call through one task-scoped process sink.

    The Workflow state model remains the complete local record. If this
    best-effort sidecar fails, callers can immediately render that state back
    into the existing lossless card pages without replaying any Agent work.
    """

    def __init__(self, sink: ProcessEventSink | None) -> None:
        self._sink = sink
        self._lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        self._started = False
        self._failed = sink is None
        self._terminal = False
        self._marker_sequence = 0
        self._segment_events = 0
        self._segment_bytes = 0

    @property
    def active(self) -> bool:
        """Whether process events still have an authoritative remote sink."""
        with self._lock:
            if not self._started or self._failed or self._terminal:
                return False
            sink = self._sink
            if sink is None or not bool(getattr(sink, "healthy", True)):
                self._fail_locked("sink_unhealthy")
                return False
            return True

    @property
    def started(self) -> bool:
        with self._lock:
            return self._started

    def start(self) -> bool:
        """Open the process sidecar before the first progress snapshot."""
        with self._lock:
            return self._ensure_started_locked()

    def agent_started(self, label: str, tool: str) -> bool:
        """Start the process sidecar lazily and add a compact call boundary."""
        with self._lock:
            if not self._ensure_started_locked():
                return False
            return self._emit_marker_locked(f"▶ {label} · {tool or 'Agent'}")

    def emit(self, label: str, event: CardEvent) -> bool:
        """Emit one direct-call event, namespaced to its Workflow Agent."""
        with self._lock:
            if not self._ensure_started_locked():
                return False
            block_id = event.payload.get("block_id")
            if isinstance(block_id, str) and block_id:
                event = CardEvent(
                    type=event.type,
                    payload={
                        **event.payload,
                        "block_id": f"workflow:{label}:{block_id}",
                    },
                )
            return self._emit_locked(event)

    def agent_done(self, label: str, payload: dict) -> bool:
        """Append a compact terminal boundary for one Agent call."""
        with self._lock:
            if not self._ensure_started_locked():
                return False
            state = "失败" if payload.get("error") else "完成"
            cached = " · 缓存命中" if payload.get("cached") else ""
            return self._emit_marker_locked(f"{state} · {label}{cached}")

    def complete(self, event: CardEvent) -> bool:
        """Close the process sidecar with the Workflow terminal outcome."""
        with self._lock:
            if not self._started or self._failed or self._terminal:
                return False
            sink = self._sink
            self._terminal = True
            if sink is None or not bool(getattr(sink, "healthy", True)):
                self._fail_locked("terminal_sink_unhealthy")
                return False
            try:
                completed = bool(sink.complete(event))
            except Exception:
                logger.warning(
                    "Workflow process stream completion failed",
                    exc_info=True,
                )
                completed = False
            self._started = False
            if not completed:
                self._fail_locked("terminal_delivery_failed")
            return completed

    def abort(self) -> None:
        """Best-effort close of a non-authoritative process sidecar."""
        with self._lock:
            self._fail_locked("aborted")
            self._terminal = True

    def _ensure_started_locked(self) -> bool:
        if self._failed or self._terminal:
            return False
        sink = self._sink
        if sink is None:
            self._failed = True
            return False
        if self._started:
            if bool(getattr(sink, "healthy", True)):
                return True
            self._fail_locked("sink_unhealthy")
            return False
        try:
            sink.start()
            self._started = bool(sink.started)
        except Exception:
            logger.warning(
                "Workflow process stream activation failed; using card pages",
                exc_info=True,
            )
            self._started = False
        if not self._started or not bool(getattr(sink, "healthy", True)):
            self._fail_locked("activation_failed")
            return False
        return True

    def _emit_marker_locked(self, text: str) -> bool:
        self._marker_sequence += 1
        block_id = f"workflow-marker-{self._marker_sequence}"
        return all(
            self._emit_locked(event)
            for event in (
                CardEvent.text_started(block_id),
                CardEvent.text_delta(block_id, text),
                CardEvent.text_done(block_id),
            )
        )

    def _emit_locked(self, event: CardEvent) -> bool:
        sink = self._sink
        if sink is None:
            self._failed = True
            return False
        event_bytes = len(
            json.dumps(
                {
                    "type": event.type.value,
                    "payload": dict(event.payload),
                },
                ensure_ascii=False,
                default=str,
            ).encode("utf-8", errors="surrogatepass")
        )
        if (
            self._segment_events + 1 > _SEGMENT_MAX_EVENTS
            or self._segment_bytes + event_bytes > _SEGMENT_MAX_BYTES
        ):
            try:
                rollover = sink.rollover()
            except Exception:
                logger.warning(
                    "Workflow process segment rollover failed; using card pages",
                    exc_info=True,
                )
                rollover = None
            if not (
                rollover is not None
                and bool(getattr(rollover, "sealed", False))
                and bool(getattr(rollover, "started", False))
            ):
                self._fail_locked("segment_rollover_failed")
                return False
            replay_events = tuple(getattr(rollover, "replay_events", ()) or ())
            self._segment_events = len(replay_events)
            self._segment_bytes = sum(
                len(
                    json.dumps(
                        {
                            "type": replay.type.value,
                            "payload": dict(replay.payload),
                        },
                        ensure_ascii=False,
                        default=str,
                    ).encode("utf-8", errors="surrogatepass")
                )
                for replay in replay_events
            )
        try:
            accepted = bool(sink.emit(event))
        except Exception:
            logger.warning(
                "Workflow process event failed; using card pages",
                exc_info=True,
            )
            accepted = False
        if not accepted or not bool(getattr(sink, "healthy", True)):
            self._fail_locked("event_delivery_failed")
            return False
        self._segment_events += 1
        self._segment_bytes += event_bytes
        return True

    def _fail_locked(self, reason: str) -> None:
        if self._failed:
            return
        self._failed = True
        self._started = False
        sink = self._sink
        if sink is None:
            return
        try:
            sink.abort()
        except Exception:
            logger.debug(
                "Workflow process stream abort failed: %s",
                reason,
                exc_info=True,
            )
