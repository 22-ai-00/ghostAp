"""Provider-specific model selection value helpers."""

from __future__ import annotations

from typing import Optional

CODEX_REASONING_EFFORTS = frozenset(
    {
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
        "ultra",
    }
)

#: Ordered effort levels for the static codex-w CLI bridge catalog.  Mirrors the
#: ascending ladder exposed by the Codex CLI/gateway (the ACP path instead reads
#: the order straight from the backend's config options).
CODEX_REASONING_EFFORT_ORDER: tuple[str, ...] = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
    "ultra",
)

#: Pseudo-model meaning "CLI/gateway default model": the codex-w bridge omits
#: the model override entirely and lets the wrapper keep its own selection.
CODEX_DEFAULT_MODEL_TOKEN = "default"

#: Sentinel effort shown in the card dropdown for "do not pass an effort
#: override" (inherit ~/.codex/config.toml).
CODEX_DEFAULT_EFFORT_TOKEN = "default"


def split_codex_model_selection(
    value: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """Split a persisted ``model/effort`` Codex selection.

    Only a recognized final effort token is removed. Provider-qualified model
    identifiers containing ``/`` remain intact.
    """
    selection = str(value or "").strip()
    if not selection:
        return None, None
    model_id, separator, suffix = selection.rpartition("/")
    if separator and model_id and suffix.lower() in CODEX_REASONING_EFFORTS:
        return model_id, suffix.lower()
    return selection, None


def compose_codex_model_selection(
    model_id: str,
    effort: Optional[str],
) -> str:
    """Compose the stable UI/persistence value for a Codex selection."""
    model = str(model_id or "").strip()
    reasoning_effort = str(effort or "").strip().lower()
    if not model:
        return ""
    if reasoning_effort in CODEX_REASONING_EFFORTS:
        return f"{model}/{reasoning_effort}"
    return model
