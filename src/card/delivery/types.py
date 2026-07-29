"""Shared types for the card delivery layer."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MutationOutcome:
    """Result of a card mutation attempt."""

    kind: Literal["applied", "reconcile", "skipped", "rejected"]
    message: str = ""


class SequenceConflictError(Exception):
    """Raised when Feishu returns 300317 (sequence conflict)."""

    def __init__(self, next_floor: int = 0):
        self.next_floor = next_floor
        super().__init__(f"Sequence conflict, floor={next_floor}")


class TransportError(Exception):
    """Raised on API failure from Feishu API."""

    # The message/card binding is stale. Recreating the card can recover.
    RECREATE_CODES = frozenset({
        99992354,  # message_id not exists / invalid
    })
    CONTENT_INVALID_CODES = frozenset({
        230099,  # card content invalid; Feishu often wraps schema errors in this code
    })
    AUDIT_REJECTED_CODES = frozenset({
        230028,  # message/card content failed Feishu audit, e.g. EMAIL_ADDRESS
    })
    CONTENT_INVALID_SUBCODES = frozenset({
        200621,  # card content parse/schema error
        200861,  # card content parse/schema error
    })

    def __init__(self, message: str = "", *, code: int = 0):
        self.code = code
        super().__init__(message)

    @property
    def is_permanent(self) -> bool:
        """Whether this error is permanent (retrying won't help)."""
        return self.needs_recreate or self.is_content_invalid or self.is_audit_rejected

    @property
    def needs_recreate(self) -> bool:
        """Whether removing the binding and recreating the card can recover."""
        return self.code in self.RECREATE_CODES

    @property
    def is_content_invalid(self) -> bool:
        """Whether the failure means the emitted card JSON is invalid."""
        if self.code in self.CONTENT_INVALID_CODES:
            return True
        text = str(self)
        for match in re.findall(r"(?:ErrCode|code)\s*[:=]\s*(\d+)", text, flags=re.IGNORECASE):
            try:
                if int(match) in self.CONTENT_INVALID_SUBCODES:
                    return True
            except ValueError:
                continue
        return "invalid image key" in text.lower()

    @property
    def is_audit_rejected(self) -> bool:
        """Whether Feishu rejected the payload during content audit."""
        if self.code in self.AUDIT_REJECTED_CODES:
            return True
        text = str(self).lower()
        return "do not pass the audit" in text or "contain sensitive data" in text
