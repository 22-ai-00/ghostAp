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


def split_slash_model_effort(
    value: Optional[str],
    *,
    allowed_efforts: frozenset[str] | set[str],
) -> tuple[Optional[str], Optional[str]]:
    """Split a persisted ``model/effort`` selection against *allowed_efforts*.

    Only a recognized final effort token is removed when a non-empty base model
    precedes ``/``.  Provider-qualified model identifiers containing ``/``
    remain intact when the trailing token is not in *allowed_efforts*.
    """
    selection = str(value or "").strip()
    if not selection:
        return None, None
    model_id, separator, suffix = selection.rpartition("/")
    normalized_suffix = suffix.strip().lower()
    if separator and model_id.strip() and normalized_suffix in allowed_efforts:
        return model_id.strip(), normalized_suffix
    return selection, None


def compose_slash_model_effort(
    model_id: str,
    effort: Optional[str],
    *,
    allowed_efforts: frozenset[str] | set[str],
) -> str:
    """Compose ``model/effort`` when *effort* is in *allowed_efforts*, else ``model``."""
    model = str(model_id or "").strip()
    if not model:
        return ""
    level = str(effort or "").strip().lower()
    if level in allowed_efforts:
        return f"{model}/{level}"
    return model


def split_codex_model_selection(
    value: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """Split a persisted ``model/effort`` Codex selection."""
    return split_slash_model_effort(value, allowed_efforts=CODEX_REASONING_EFFORTS)


def compose_codex_model_selection(
    model_id: str,
    effort: Optional[str],
) -> str:
    """Compose the stable UI/persistence value for a Codex selection."""
    return compose_slash_model_effort(
        model_id,
        effort,
        allowed_efforts=CODEX_REASONING_EFFORTS,
    )
