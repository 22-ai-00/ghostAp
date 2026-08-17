"""DSH ACP model and reasoning selection wire helpers."""

from __future__ import annotations

import json

DSH_MODEL_CONFIG_ID = "dsh.model"
DSH_REASONING_CONFIG_ID = "dsh.reasoning_effort"
DSH_DEFAULT_REASONING = "default"


def decode_dsh_model_value(value: str) -> tuple[str, str]:
    """Decode the DSH ACP model selector value into provider and model IDs."""
    try:
        parsed = json.loads(str(value or ""))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid DSH model selection: {value}") from exc
    if (
        not isinstance(parsed, list)
        or len(parsed) != 2
        or not all(isinstance(item, str) and item.strip() for item in parsed)
    ):
        raise ValueError(f"invalid DSH model selection: {value}")
    return parsed[0], parsed[1]


def compose_dsh_model_selection(model_value: str, effort: str | None = None) -> str:
    """Compose GhostAP's durable DSH selection from ACP-native identifiers."""
    provider, model = decode_dsh_model_value(model_value)
    values = [provider, model]
    normalized_effort = str(effort or "").strip().lower()
    if normalized_effort:
        values.append(normalized_effort)
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def split_dsh_model_selection(selection: str) -> tuple[str, str | None]:
    """Split a durable DSH selection into model and reasoning ACP values."""
    try:
        parsed = json.loads(str(selection or ""))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid DSH model selection: {selection}") from exc
    if (
        not isinstance(parsed, list)
        or len(parsed) not in {2, 3}
        or not all(isinstance(item, str) and item.strip() for item in parsed)
    ):
        raise ValueError(f"invalid DSH model selection: {selection}")
    model_value = json.dumps(parsed[:2], ensure_ascii=False, separators=(",", ":"))
    if len(parsed) == 2:
        return model_value, None
    effort = parsed[2].strip().lower()
    reasoning_value = (
        json.dumps([DSH_DEFAULT_REASONING], separators=(",", ":"))
        if effort == DSH_DEFAULT_REASONING
        else json.dumps(["effort", effort], ensure_ascii=False, separators=(",", ":"))
    )
    return model_value, reasoning_value


def decode_dsh_reasoning_value(value: str) -> str | None:
    """Return a display/persistence effort ID from DSH's ACP selector value."""
    try:
        parsed = json.loads(str(value or ""))
    except (TypeError, ValueError):
        return None
    if parsed == [DSH_DEFAULT_REASONING]:
        return DSH_DEFAULT_REASONING
    if (
        isinstance(parsed, list)
        and len(parsed) == 2
        and parsed[0] == "effort"
        and isinstance(parsed[1], str)
        and parsed[1]
    ):
        return parsed[1].strip().lower()
    return None
