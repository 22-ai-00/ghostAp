"""Workflow-only safety projection for streamed card text."""

from __future__ import annotations

import re

from src.card.events import CardEvent, CardEventType
from src.utils.redact import redact_sensitive

_STREAM_TEXT_EVENT_TYPES = frozenset(
    {
        CardEventType.TEXT_STARTED,
        CardEventType.TEXT_DELTA,
        CardEventType.REASONING_STARTED,
        CardEventType.REASONING_DELTA,
    }
)
_STREAM_TEXT_PAYLOAD_FIELDS = frozenset(
    {
        "text",
        "source_sequence",
        "source_label",
        "source_ref",
    }
)
_ANSI_ESCAPE_RE = re.compile(
    r"(?:"
    r"(?:\x1b\]|\x9d)[\s\S]*?(?:\x07|\x1b\\|\x9c|$)"
    r"|(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]"
    r"|(?:\x1b[P^_]|\x90|\x98|\x9e|\x9f)"
    r"[\s\S]*?(?:\x1b\\|\x9c|$)"
    r"|\x1b[@-_]"
    r")",
)
_UNSAFE_STREAM_CONTROL_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f"
    r"\u061c\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]"
)


def _sanitize_stream_text(value: str) -> str:
    """Remove invisible control tricks, then redact the normalized text."""
    normalized = _ANSI_ESCAPE_RE.sub("", value)
    normalized = _UNSAFE_STREAM_CONTROL_RE.sub("", normalized)
    return redact_sensitive(normalized)


def sanitize_workflow_stream_event(event: CardEvent) -> CardEvent:
    """Return a card event safe to persist in Workflow execution history.

    Ordinary programming cards keep their existing projection behavior. This
    boundary is intentionally Workflow-specific because Workflow retains the
    complete execution stream and later republishes it across paged cards.
    """
    if event.type not in _STREAM_TEXT_EVENT_TYPES:
        return event

    payload = dict(event.payload)
    changed = False
    for key in _STREAM_TEXT_PAYLOAD_FIELDS:
        value = payload.get(key)
        if not isinstance(value, str):
            continue
        safe_value = _sanitize_stream_text(value)
        if safe_value != value:
            payload[key] = safe_value
            changed = True
    if not changed:
        return event
    return CardEvent(type=event.type, payload=payload)
