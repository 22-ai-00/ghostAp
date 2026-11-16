"""AGY ACP model and thought-level selection wire helpers."""

from __future__ import annotations

AGY_MODEL_CONFIG_ID = "model"
AGY_THOUGHT_LEVEL_CONFIG_ID = "thought_level"
AGY_REASONING_EFFORTS = frozenset(
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


def compose_agy_model_selection(
    model_id: str,
    effort: str | None = None,
) -> str:
    """Compose GhostAP's durable AGY selection from ACP-native identifiers."""
    model = str(model_id or "").strip()
    if not model:
        raise ValueError("invalid AGY model selection: empty model")
    normalized_effort = str(effort or "").strip().lower()
    if not normalized_effort:
        return model
    if normalized_effort not in AGY_REASONING_EFFORTS:
        raise ValueError(f"invalid AGY thought level: {effort}")
    return f"{model}/{normalized_effort}"


def split_agy_model_selection(
    selection: str | None,
) -> tuple[str, str | None]:
    """Split a durable AGY selection into model and optional thought-level."""
    raw = str(selection or "").strip()
    if not raw:
        raise ValueError(f"invalid AGY model selection: {selection}")
    if "/" in raw:
        model_id, _, suffix = raw.rpartition("/")
        model = model_id.strip()
        effort = suffix.strip().lower()
        if (
            not model
            or not effort
            or effort not in AGY_REASONING_EFFORTS
            or "/" in model
        ):
            raise ValueError(f"invalid AGY model selection: {selection}")
        return model, effort
    return raw, None
