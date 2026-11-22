"""AGY ACP and Orchestrator CLI model and thought-level selection helpers."""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .model_selection import (
    CODEX_REASONING_EFFORTS,
    compose_slash_model_effort,
    split_slash_model_effort,
)

AGY_MODEL_CONFIG_ID = "model"
AGY_THOUGHT_LEVEL_CONFIG_ID = "thought_level"
AGY_EFFORT_CONFIG_IDS: tuple[str, ...] = (
    AGY_THOUGHT_LEVEL_CONFIG_ID,
    "reasoning_effort",
    "effort",
)
AGY_REASONING_EFFORTS = CODEX_REASONING_EFFORTS
AGY_ORCHESTRATOR_DEFAULT_PROFILE = "default"

AGY_EFFORT_DISPLAY_LABELS: dict[str, str] = {
    "none": "None",
    "minimal": "Minimal",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "xhigh": "XHigh",
    "max": "Max",
    "ultra": "Ultra",
}


@dataclass(frozen=True)
class AGYOrchestratorSpec:
    """Mapping from an AGY Orchestrator preset ID to Planner/Executor base models."""

    model_id: str
    planner_model_id: str
    planner_display_base: str
    executor_model_id: str
    executor_display_base: str


@dataclass(frozen=True)
class AGYParsedSelection:
    """Structured parse result for any durable AGY model/effort selection."""

    model_id: str
    thought_level: str | None = None
    planner_effort: str | None = None
    executor_effort: str | None = None


AGY_ORCHESTRATOR_SPECS: dict[str, AGYOrchestratorSpec] = {
    "gemini-pro-orchestrator-neon": AGYOrchestratorSpec(
        model_id="gemini-pro-orchestrator-neon",
        planner_model_id="gemini-4-neon",
        planner_display_base="Gemini 4 Neon",
        executor_model_id="gemini-4-hydrogen",
        executor_display_base="Gemini 4 Hydrogen",
    ),
    "gemini-pro-orchestrator": AGYOrchestratorSpec(
        model_id="gemini-pro-orchestrator",
        planner_model_id="gemini-4-carbon",
        planner_display_base="Gemini 4 Carbon",
        executor_model_id="gemini-4-hydrogen",
        executor_display_base="Gemini 4 Hydrogen",
    ),
    "gemini-pro-orchestrator-argon": AGYOrchestratorSpec(
        model_id="gemini-pro-orchestrator-argon",
        planner_model_id="gemini-4-argon",
        planner_display_base="Gemini 4 Argon",
        executor_model_id="gemini-4-hydrogen",
        executor_display_base="Gemini 4 Hydrogen",
    ),
}

_LABEL_LOCK = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
_LIVE_DISPLAY_LABELS: dict[tuple[str, str], str] = {}
_LIVE_BASE_DISPLAY_NAMES: dict[str, str] = {}


def is_agy_orchestrator_model(model_id: str | None) -> bool:
    """Return True if *model_id* (or selection base) is a known AGY Orchestrator preset."""
    raw = str(model_id or "").strip()
    if not raw:
        return False
    base = raw.split("/", 1)[0].strip()
    return base in AGY_ORCHESTRATOR_SPECS


def get_agy_orchestrator_spec(model_id: str | None) -> AGYOrchestratorSpec | None:
    """Return the :class:`AGYOrchestratorSpec` for an Orchestrator ID or selection."""
    raw = str(model_id or "").strip()
    if not raw:
        return None
    base = raw.split("/", 1)[0].strip()
    return AGY_ORCHESTRATOR_SPECS.get(base)


def register_agy_model_display_labels(
    labels: Mapping[tuple[str, str], str],
    *,
    base_names: Mapping[str, str] | None = None,
) -> None:
    """Cache live `(base_model_id, effort) -> exact display label` mappings from ACP."""
    with _LABEL_LOCK:
        for (base_id, effort), display_label in (labels or {}).items():
            norm_base = str(base_id or "").strip()
            norm_effort = str(effort or "").strip().lower()
            norm_label = str(display_label or "").strip()
            if norm_base and norm_effort and norm_label:
                _LIVE_DISPLAY_LABELS[(norm_base, norm_effort)] = norm_label
        for base_id, display_name in (base_names or {}).items():
            norm_base = str(base_id or "").strip()
            norm_name = str(display_name or "").strip()
            if norm_base and norm_name:
                _LIVE_BASE_DISPLAY_NAMES[norm_base] = norm_name


def format_agy_model_effort_label(
    model_id: str,
    effort: str,
    *,
    display_base: str | None = None,
) -> str:
    """Resolve the exact display label expected by `agy --settings` `orchestrator`."""
    norm_id = str(model_id or "").strip()
    norm_effort = str(effort or "").strip().lower()
    if not norm_id or norm_effort not in AGY_REASONING_EFFORTS:
        raise ValueError(f"invalid AGY model or effort for label: {model_id!r}, {effort!r}")
    with _LABEL_LOCK:
        live = _LIVE_DISPLAY_LABELS.get((norm_id, norm_effort))
        if live:
            return live
        base_label = _LIVE_BASE_DISPLAY_NAMES.get(norm_id)
    if not base_label:
        base_label = str(display_base or "").strip() or norm_id
    effort_label = AGY_EFFORT_DISPLAY_LABELS.get(norm_effort, norm_effort.title())
    return f"{base_label} ({effort_label})"


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


def compose_agy_orchestrator_selection(
    model_id: str,
    planner_effort: str | None = None,
    executor_effort: str | None = None,
) -> str:
    """Compose GhostAP's durable AGY Orchestrator selection (`orch` or `orch/p/e`)."""
    spec = get_agy_orchestrator_spec(model_id)
    if spec is None or "/" in str(model_id or ""):
        raise ValueError(f"invalid AGY orchestrator model: {model_id}")
    norm_planner = str(planner_effort or "").strip().lower()
    norm_executor = str(executor_effort or "").strip().lower()
    if (not norm_planner or norm_planner == AGY_ORCHESTRATOR_DEFAULT_PROFILE) and (
        not norm_executor or norm_executor == AGY_ORCHESTRATOR_DEFAULT_PROFILE
    ):
        return spec.model_id
    if norm_planner not in AGY_REASONING_EFFORTS:
        raise ValueError(f"invalid AGY planner effort: {planner_effort}")
    if not norm_executor or norm_executor == AGY_ORCHESTRATOR_DEFAULT_PROFILE:
        norm_executor = norm_planner
    elif norm_executor not in AGY_REASONING_EFFORTS:
        raise ValueError(f"invalid AGY executor effort: {executor_effort}")
    return f"{spec.model_id}/{norm_planner}/{norm_executor}"


def parse_agy_model_selection(
    selection: str | None,
) -> AGYParsedSelection:
    """Parse a durable AGY selection (regular or Orchestrator Planner/Executor)."""
    raw = str(selection or "").strip()
    if not raw:
        raise ValueError(f"invalid AGY model selection: {selection}")
    parts = raw.split("/")
    if len(parts) == 1:
        return AGYParsedSelection(model_id=parts[0].strip())
    if len(parts) == 2:
        base = parts[0].strip()
        eff = parts[1].strip().lower()
        if not base or eff not in AGY_REASONING_EFFORTS:
            raise ValueError(f"invalid AGY model selection: {selection}")
        if base in AGY_ORCHESTRATOR_SPECS:
            return AGYParsedSelection(
                model_id=base,
                planner_effort=eff,
                executor_effort=eff,
            )
        return AGYParsedSelection(model_id=base, thought_level=eff)
    if len(parts) == 3:
        base = parts[0].strip()
        p_eff = parts[1].strip().lower()
        e_eff = parts[2].strip().lower()
        if (
            base not in AGY_ORCHESTRATOR_SPECS
            or p_eff not in AGY_REASONING_EFFORTS
            or e_eff not in AGY_REASONING_EFFORTS
        ):
            raise ValueError(f"invalid AGY model selection: {selection}")
        return AGYParsedSelection(
            model_id=base,
            planner_effort=p_eff,
            executor_effort=e_eff,
        )
    raise ValueError(f"invalid AGY model selection: {selection}")


def split_agy_model_selection(
    selection: str | None,
) -> tuple[str, str | None]:
    """Split a durable AGY ACP-native selection into `(model, thought_level)`.

    Rejects invalid strings and Orchestrator custom-effort selections (which
    require the CLI `--settings` transport rather than `agy acp`).
    """
    raw = str(selection or "").strip()
    if not raw:
        raise ValueError(f"invalid AGY model selection: {selection}")
    model, effort = split_slash_model_effort(
        raw,
        allowed_efforts=AGY_REASONING_EFFORTS,
    )
    if not model or "/" in model:
        raise ValueError(f"invalid AGY model selection: {selection}")
    if effort is not None and model in AGY_ORCHESTRATOR_SPECS:
        raise ValueError(
            f"invalid AGY ACP selection (orchestrator effort requires CLI settings): {selection}"
        )
    return model, effort


def requires_agy_cli_orchestrator_session(selection: str | None) -> bool:
    """Return True when *selection* is an Orchestrator with custom Planner/Executor effort."""
    raw = str(selection or "").strip()
    if not raw or "/" not in raw:
        return False
    try:
        parsed = parse_agy_model_selection(raw)
    except ValueError:
        return False
    return bool(
        parsed.planner_effort is not None
        and parsed.executor_effort is not None
        and parsed.model_id in AGY_ORCHESTRATOR_SPECS
    )


def resolve_agy_orchestrator_settings(
    selection: str | None,
) -> tuple[str, dict[str, Any]]:
    """Build `(orchestrator_model_id, settings_dict)` for `agy --settings`."""
    parsed = parse_agy_model_selection(selection)
    spec = AGY_ORCHESTRATOR_SPECS.get(parsed.model_id)
    if spec is None or parsed.planner_effort is None or parsed.executor_effort is None:
        raise ValueError(f"selection is not an AGY orchestrator custom effort: {selection}")
    planner_label = format_agy_model_effort_label(
        spec.planner_model_id,
        parsed.planner_effort,
        display_base=spec.planner_display_base,
    )
    executor_label = format_agy_model_effort_label(
        spec.executor_model_id,
        parsed.executor_effort,
        display_base=spec.executor_display_base,
    )
    return spec.model_id, {
        "orchestrator": {
            "planner": planner_label,
            "executor": executor_label,
        }
    }


def normalize_agy_model_input(
    raw: str | None,
    known_model_names: Sequence[str] = (),
) -> str:
    """Normalize AGY CLI hyphenated slugs or 2-part Orchestrator shorthands."""
    text = str(raw or "").strip()
    if not text:
        return ""
    known_list = [str(name or "").strip() for name in (known_model_names or ()) if str(name or "").strip()]
    known_set = set(known_list)
    if text in known_set:
        return text

    # 1. Normalize 2-part Orchestrator shorthand: `<orch_id>/<effort>` -> `<orch_id>/<effort>/<exec>`
    if "/" in text:
        parts = text.split("/")
        if len(parts) == 2:
            base = parts[0].strip()
            eff = parts[1].strip().lower()
            if base in AGY_ORCHESTRATOR_SPECS and eff in AGY_REASONING_EFFORTS:
                same_target = f"{base}/{eff}/{eff}"
                if same_target in known_set:
                    return same_target
                prefix = f"{base}/{eff}/"
                for candidate in known_list:
                    if candidate.startswith(prefix):
                        return candidate
                if not known_list:
                    return same_target
        return text

    # 2. Normalize official AGY CLI hyphenated slug: e.g. `gemini-4-neon-xhigh` -> `gemini-4-neon/xhigh`
    for effort in sorted(AGY_REASONING_EFFORTS, key=len, reverse=True):
        suffix = f"-{effort}"
        if text.lower().endswith(suffix) and len(text) > len(suffix):
            prefix = text[: -len(suffix)]
            if prefix in AGY_ORCHESTRATOR_SPECS:
                candidate = f"{prefix}/{effort}/{effort}"
                if candidate in known_set:
                    return candidate
                orch_prefix = f"{prefix}/{effort}/"
                for known in known_list:
                    if known.startswith(orch_prefix):
                        return known
                if not known_list:
                    return candidate
            slash_candidate = f"{prefix}/{effort}"
            if slash_candidate in known_set:
                return slash_candidate
            if not known_list and not prefix.startswith("gpt-"):
                return slash_candidate
    return text
