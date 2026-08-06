"""Use case for retrying the original mode from degraded error cards."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from src.card.ui_text import UI_TEXT


class ModeRetryGateway(Protocol):
    def retry(self, *, message_id: str, chat_id: str, project_id: str | None, payload: dict[str, Any]) -> "RetryDecision": ...


class RetryDecisionStatus(Enum):
    ACCEPTED = "accepted"
    MANUAL_REQUIRED = "manual_required"
    CONTEXT_MISSING = "context_missing"
    NOT_RETRYABLE = "not_retryable"
    COOLDOWN = "cooldown"
    DUPLICATE = "duplicate"


@dataclass(frozen=True)
class RetryDecision:
    status: RetryDecisionStatus
    mode: str
    message: str


@dataclass(frozen=True)
class RetryOriginalPayload:
    original_mode: str
    degraded_to: str
    retry_mode: str


_ALLOWED_RETRY_MODES = {
    "coco": "Coco",
    "claude": "Claude",
    "claude cli": "Claude CLI",
    "aiden": "Aiden",
    "codex": "Codex",
    "gemini": "Gemini",
}


def _normalize_retry_mode(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return _ALLOWED_RETRY_MODES.get(raw.lower(), "")


def validate_retry_original_payload(payload: dict[str, Any]) -> RetryOriginalPayload | None:
    """Validate retry-original card payload with explicit mode semantics.

    ``original_mode`` is the mode that failed, ``degraded_to`` is the mode the
    user can continue with, and ``retry_mode`` is the requested retry target.
    Retry is allowed only when the requested target matches the original failed
    mode and both names are from the supported mode allowlist.
    """

    required_fields = ("original_mode", "retry_mode", "degraded_to")
    if any(field not in payload for field in required_fields):
        return None

    original_mode = _normalize_retry_mode(payload.get("original_mode"))
    retry_mode = _normalize_retry_mode(payload.get("retry_mode"))
    degraded_to = _normalize_retry_mode(payload.get("degraded_to"))
    if not original_mode or not retry_mode or not degraded_to:
        return None
    if retry_mode != original_mode:
        return None
    if degraded_to == original_mode:
        return None
    return RetryOriginalPayload(
        original_mode=original_mode,
        degraded_to=degraded_to,
        retry_mode=retry_mode,
    )


@dataclass(frozen=True)
class RetryOriginalModeUseCase:
    """Centralize retry-original validation and retry decision making.

    This use case deliberately returns a decision only.  Feishu replies are an
    adapter-layer side effect handled by ``action_registry``.
    """

    gateway: ModeRetryGateway | None = None

    @staticmethod
    def accepted(mode: str) -> RetryDecision:
        return RetryDecision(
            status=RetryDecisionStatus.ACCEPTED,
            mode=mode,
            message=UI_TEXT["card_lifecycle_retry_original_started"].format(mode=mode),
        )

    def __call__(self, message_id: str, chat_id: str, project_id: str | None, payload: dict[str, Any]) -> RetryDecision:
        retry_payload = validate_retry_original_payload(payload)
        if retry_payload is None:
            return RetryDecision(
                status=RetryDecisionStatus.CONTEXT_MISSING,
                mode="",
                message=UI_TEXT["card_lifecycle_retry_original_unavailable"],
            )
        mode = retry_payload.retry_mode

        if self.gateway is not None:
            return self.gateway.retry(message_id=message_id, chat_id=chat_id, project_id=project_id, payload=dict(payload))
        return RetryDecision(
            status=RetryDecisionStatus.MANUAL_REQUIRED,
            mode=mode,
            message=UI_TEXT["card_lifecycle_retry_original_manual_required"].format(mode=mode),
        )
