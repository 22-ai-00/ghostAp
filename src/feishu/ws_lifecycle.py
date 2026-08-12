"""Feishu WebSocket lifecycle helpers.

This module keeps low-level lark-channel WebSocket lifecycle observation out of
``ws_client.py`` so the main client can stay focused on orchestration.
"""

from __future__ import annotations

import base64
import http
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

from lark_channel import ws
from lark_channel.core.const import UTF_8
from lark_channel.core.json import JSON
from lark_channel.ws.const import (
    HEADER_BIZ_RT,
    HEADER_MESSAGE_ID,
    HEADER_SEQ,
    HEADER_SUM,
    HEADER_TRACE_ID,
    HEADER_TYPE,
)
from lark_channel.ws.enum import MessageType
from lark_channel.ws.model import Response

logger = logging.getLogger(__name__)


class WSLifecycleAction(str, Enum):
    RECORD_ACTIVITY_AND_CONTINUE = "record_activity_and_continue"
    PROPAGATE = "propagate"


@dataclass(frozen=True)
class WSLifecycleErrorClassification:
    action: WSLifecycleAction
    phase: str


def classify_lifecycle_error(error: Exception, *, phase: str) -> WSLifecycleErrorClassification:
    if phase == "disconnect":
        return WSLifecycleErrorClassification(WSLifecycleAction.RECORD_ACTIVITY_AND_CONTINUE, phase)
    return WSLifecycleErrorClassification(WSLifecycleAction.PROPAGATE, phase)


def frame_header_value(frame: Any, key: str) -> Optional[str]:
    for header in getattr(frame, "headers", []) or []:
        if getattr(header, "key", None) == key:
            return getattr(header, "value", None)
    return None


class ObservedLarkWSClient(ws.Client):
    """Wrap the official Channel SDK WS client with activity hooks."""

    def __init__(
        self,
        *args,
        on_activity: Callable[[str], None],
        on_response_written: Callable[[bool], None] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._on_activity = on_activity
        self._on_response_written = on_response_written

    async def _connect(self) -> None:
        await super()._connect()
        self._on_activity("connected")

    async def _disconnect(self, *, expected_conn=None):
        disconnected = await super()._disconnect(expected_conn=expected_conn)
        if disconnected:
            self._on_activity("disconnected")
        return disconnected

    async def _handle_control_frame(self, frame):
        message_type = frame_header_value(frame, HEADER_TYPE)
        if message_type == MessageType.PONG.value:
            self._on_activity("pong")
        elif message_type == MessageType.PING.value:
            self._on_activity("ping")
        else:
            self._on_activity("control")
        return await super()._handle_control_frame(frame)

    async def _handle_data_frame(self, frame):
        self._on_activity("data")
        message_type = frame_header_value(frame, HEADER_TYPE)
        if message_type == MessageType.CARD.value:
            return await self._handle_card_callback_frame(frame)
        if message_type != MessageType.EVENT.value:
            return await super()._handle_data_frame(frame)
        try:
            result = await super()._handle_data_frame(frame)
        except BaseException:
            self._notify_response_written(False)
            raise
        self._notify_response_written(self._response_is_success(frame))
        return result

    @staticmethod
    def _response_is_success(frame: Any) -> bool:
        try:
            payload = JSON.unmarshal(frame.payload.decode(UTF_8), dict)
            return payload.get("code") == http.HTTPStatus.OK
        except (AttributeError, TypeError, ValueError):
            return False

    def _notify_response_written(self, written: bool) -> None:
        callback = getattr(self, "_on_response_written", None)
        if not callable(callback):
            return
        try:
            callback(written)
        except Exception:
            # ACK bytes are already committed (or the write has failed).
            # Advisory cleanup must never rewrite transport outcome.
            logger.warning("post-response callback failed", exc_info=True)

    async def _handle_card_callback_frame(self, frame: Any) -> None:
        """Dispatch callback frames that the upstream WS client drops.

        Feishu can transport ``card.action.trigger`` over a ``card`` data
        frame.  ``lark_channel.ws.Client`` currently returns without either
        dispatching or acknowledging that frame, which makes the client show
        error 200530.  The payload is still the latest P2 callback envelope, so
        route it through the same typed event dispatcher used for ``event``
        frames and preserve the official response encoding.
        """

        headers = frame.headers
        message_id = frame_header_value(frame, HEADER_MESSAGE_ID) or ""
        trace_id = frame_header_value(frame, HEADER_TRACE_ID) or ""
        part_count = int(frame_header_value(frame, HEADER_SUM) or "1")
        sequence = int(frame_header_value(frame, HEADER_SEQ) or "0")
        payload = frame.payload
        if part_count > 1:
            payload = self._combine(message_id, part_count, sequence, payload)
            if payload is None:
                return

        response = Response(code=http.HTTPStatus.OK)
        started_ms = int(round(time.time() * 1000))
        try:
            result = self._event_handler._do_without_validation(payload)
            elapsed_ms = int(round(time.time() * 1000)) - started_ms
            header = headers.add()
            header.key = HEADER_BIZ_RT
            header.value = str(elapsed_ms)
            if result is not None:
                response.data = base64.b64encode(JSON.marshal(result).encode(UTF_8))
        except Exception as exc:
            logger.error(
                "card callback frame dispatch failed: message_id=%s trace_id=%s err=%s",
                message_id,
                trace_id,
                type(exc).__name__,
                exc_info=True,
            )
            response = Response(code=http.HTTPStatus.INTERNAL_SERVER_ERROR)

        frame.payload = JSON.marshal(response).encode(UTF_8)
        try:
            await self._write_message(frame.SerializeToString())
        except BaseException:
            self._notify_response_written(False)
            raise
        self._notify_response_written(response.code == http.HTTPStatus.OK)
