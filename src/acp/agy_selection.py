"""AGY ACP model and thought-level selection wire helpers."""

from __future__ import annotations

from .model_selection import (
    CODEX_REASONING_EFFORTS,
    compose_slash_model_effort,
    split_slash_model_effort,
)

AGY_MODEL_CONFIG_ID = "model"
AGY_THOUGHT_LEVEL_CONFIG_ID = "thought_level"
AGY_REASONING_EFFORTS = CODEX_REASONING_EFFORTS


def compose_agy_model_selection(
    model_id: str,
    effort: str | None = None,
) -> str:
    """Compose GhostAP's durable AGY selection from ACP-native identifiers."""
    model = str(model_id or "").strip()
    if not model:
        raise ValueError("invalid AGY model selection: empty model")
    normalized_effort = str(effort or "").strip().lower()
    if normalized_effort and normalized_effort not in AGY_REASONING_EFFORTS:
        raise ValueError(f"invalid AGY thought level: {effort}")
    return compose_slash_model_effort(
        model,
        normalized_effort or None,
        allowed_efforts=AGY_REASONING_EFFORTS,
    )


def split_agy_model_selection(
    selection: str | None,
) -> tuple[str, str | None]:
    """Split a durable AGY selection into model and optional thought-level."""
    raw = str(selection or "").strip()
    if not raw:
        raise ValueError(f"invalid AGY model selection: {selection}")
    model, effort = split_slash_model_effort(
        raw,
        allowed_efforts=AGY_REASONING_EFFORTS,
    )
    if not model or "/" in model:
        raise ValueError(f"invalid AGY model selection: {selection}")
    return model, effort
